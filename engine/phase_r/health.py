"""Health checks before an arm and the invalidation rules after it (BENCH_PLAN sections 5 and 7), thresholds fixed in
the plan:

Before every arm (a failure excludes the arm, and with it the host's block):
  - no hardware slowdown or thermal clock-event reason active on the arm's GPUs (the software power cap is logged,
    not excluded)
  - no compute process left on the arm's GPUs (fresh engine process)
  - fio O_DIRECT sequential read within 10% of the host's first probe
  - free host memory (MemAvailable) at least the cgroup budget plus 8 GiB
  - workload session and turn counts exact (and the file's sha256 when the cell gives it)
  - SGLang arms: the HiCacheFile fadvise patch present
  - vLLM arms: vLLM PR #60045 (the sliding-window admission count) present
After the arm (on the replay window, cut by the runner's replay_start and replay_end markers):
  - any memory.events oom or oom_kill increase (whole arm, boot included)
  - PSI "full" stall time over 1% of the window, for each resource in psi_resources (default memory; io is logged.
    Under io.max every disk arm stalls on I/O by design, so io PSI measures the regime, not a fault; see the report)
  - a hardware slowdown or thermal event on the arm's GPUs in the window (sampled reason bitmask, and the cumulative
    clock-event counters where the driver has them)
  - engine crash: the runner's rc, "server failed" (the sidecar too), no replay_end marker, or "SERVER DIED" (the
    runner found the server, the LMCache server or the sidecar dead, or /health failing, after the replay)
  - engine death seen by the client: more than 0.5% of the arm's turn rows (served, failed and censored) failed on a
    lost connection (replay.py's conn_error flag: refused, reset, disconnected, truncated stream). Pre-registered
    threshold, conn_fail_max_frac
  - vLLM fell back from O_DIRECT on a disk tier (its own warning in the server log), slow and medium regimes
  - disk tier over capacity: the tier directory's size (telemetry samples and the runner's du after the replay) above
    the cell's disk-tier bytes (vLLM's fs tier has no bound of its own, so this is how the 2 P capacity holds)
  - page-cache slack used: in the replay window, the disk tier's cached bytes (cachestat; not for a RAM tier, whose
    pages are the tier) or the cgroup's unmapped page cache (memory.stat file minus max(shmem, file_mapped): the
    pinned CPU tier in /dev/shm and a tmpfs tier are shmem, and mapped pages are the libraries and the mapped shmem
    tier, logged as file_mapped) above page_cache_max_bytes (256 MiB, the budget's slack); the maxima are reported
    per arm. The weights' clean cache from loading is reclaimed at replay_start (arm_hook.sh), before the window
  - engine stall: requests in flight and no first token or completion for engine_stall_s (300 s) or more in the
    replay (the engine hung while /health still answered); counted as an engine crash
Covariates logged: SM clock per GPU (median and min in the window), software power-cap time, max temperature,
max fan speed, io PSI, the tier's peak size, the page-cache maxima, the connection-failure fraction.
"""
import csv
import hashlib
import io
import json
import os
import statistics

from common import GiB, MiB

# NVML clock-event reason bits (nvmlClocksEventReason*)
REASON_BITS = {0x1: "gpu_idle", 0x2: "applications_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
               0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown",
               0x80: "hw_power_brake_slowdown", 0x100: "display_clock_setting"}
EXCLUDE_MASK = 0x8 | 0x20 | 0x40 | 0x80     # hardware slowdown and every thermal or power-brake reason
EXCLUDE_COUNTERS = ("hw_thermal_slowdown", "hw_power_brake_slowdown", "sw_thermal_slowdown")
GPU_FIELDS = ["timestamp", "index", "clocks.sm", "clocks.mem", "clocks.max.sm", "temperature.gpu", "power.draw",
              "power.limit", "utilization.gpu", "memory.used", "pcie.link.gen.current", "pcie.link.width.current",
              "clocks_event_reasons.active"]
COUNTER_FIELDS = ["clocks_event_reasons_counters." + c for c in
                  ("sw_power_cap", "sw_thermal_slowdown", "hw_thermal_slowdown", "hw_power_brake_slowdown")]
DEFAULTS = {"fio_tolerance": 0.10, "free_mem_margin_bytes": 8 * GiB, "psi_full_max_frac": 0.01,
            "psi_resources": ["memory"], "conn_fail_max_frac": 0.005, "page_cache_max_bytes": 256 * MiB,
            "engine_stall_s": 300.0}
CONN_PATTERNS = ("ClientConnectorError", "ServerDisconnectedError", "ClientOSError", "ClientPayloadError",
                 "Connection refused", "Connection reset")
ODIRECT_FALLBACK = "O_DIRECT is not supported at"


def decode(mask):
    return [n for b, n in REASON_BITS.items() if mask & b]


def mask_of(v):
    v = (v or "").strip()
    try:
        return int(v, 16) if v.lower().startswith("0x") else int(v)
    except ValueError:
        return 0


def gpu_fields(runner):
    """The query fields this driver accepts: counters only when it has them, the old throttle name on old drivers."""
    h = runner.run(["nvidia-smi", "--help-query-gpu"], kind="gpu",
                   fake='"clocks_event_reasons.active"\n' + "\n".join(f'"{c}"' for c in COUNTER_FIELDS)
                   + '\n"fan.speed"').out
    fields = list(GPU_FIELDS)
    if '"clocks_event_reasons.active"' not in h:
        fields[-1] = "clocks_throttle_reasons.active"
    fields += [c for c in COUNTER_FIELDS if f'"{c}"' in h]
    if '"fan.speed"' in h:      # last, so the earlier columns keep their places
        fields.append("fan.speed")
    return fields


def gpu_query_cmd(fields, gpus=None, loop_ms=None, out=None):
    cmd = ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"]
    if gpus:
        cmd += ["-i", gpus]
    if loop_ms:
        cmd += ["-lms", str(loop_ms)]
    if out:
        cmd += ["-f", out]
    return cmd


def parse_gpu_csv(text, fields):
    rows = []
    for rec in csv.reader(io.StringIO(text)):
        if len(rec) != len(fields):
            continue
        d = {f: v.strip() for f, v in zip(fields, rec)}
        d["mask"] = mask_of(d.get("clocks_event_reasons.active") or d.get("clocks_throttle_reasons.active"))
        rows.append(d)
    return rows


def gpu_now(runner, fields, gpus=None, fake_mask="0x0000000000000000"):
    fake = ", ".join({"index": "0", "clocks_event_reasons.active": fake_mask,
                      "clocks_throttle_reasons.active": fake_mask}.get(f, "0") for f in fields) + "\n"
    r = runner.run(gpu_query_cmd(fields, gpus), kind="gpu", fake=fake)
    return parse_gpu_csv(r.out, fields)


def gpu_procs(runner, gpus=None):
    cmd = ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"]
    if gpus:
        cmd += ["-i", gpus]
    return [ln for ln in runner.run(cmd, kind="gpu").out.splitlines() if ln.strip()]


def mem_available(runner):
    t = runner.read("/proc/meminfo", fake="MemAvailable: 200000000 kB\n")
    for ln in t.splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) * 1024
    return 0


def workload_counts(path):
    """(sessions, turns, sha256) of a replay workload (engine/workloads format)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    with open(path) as f:
        W = json.load(f)
    S = W["sessions"]
    return len(S), sum(len(s["turns"]) for s in S), h.hexdigest()


def pre_arm(*, gpu_rows, gpu_procs_list, probe_now, probe_first, mem_avail, budget, workload, expect,
            sglang_patch=None, vllm_patch=None, settings=None):
    """List of failures; each {'check', 'detail', 'host_level'}. host_level failures mean the host needs replacing."""
    s = dict(DEFAULTS, **(settings or {}))
    bad = []
    for g in gpu_rows:
        if g["mask"] & EXCLUDE_MASK:
            bad.append({"check": "gpu_throttle", "host_level": True,
                        "detail": f"GPU {g.get('index')} reasons {decode(g['mask'])}"})
    if gpu_procs_list:
        bad.append({"check": "gpu_busy", "host_level": False, "detail": "; ".join(gpu_procs_list)})
    if probe_now is not None and probe_first is not None:
        a, b = probe_first.get("seq_read_Bps"), probe_now.get("seq_read_Bps")
        if probe_now.get("error") or not a or b is None:
            bad.append({"check": "fio", "host_level": True, "detail": probe_now.get("error", "no read number")})
        elif abs(b - a) / a > s["fio_tolerance"]:
            bad.append({"check": "fio", "host_level": True,
                        "detail": f"seq read {b / 1e6:.0f} MB/s vs first {a / 1e6:.0f} MB/s "
                                  f"({(b - a) / a:+.1%}, limit {s['fio_tolerance']:.0%})"})
    need = budget + s["free_mem_margin_bytes"]
    if mem_avail < need:
        bad.append({"check": "free_memory", "host_level": False,
                    "detail": f"MemAvailable {mem_avail / GiB:.1f} GiB < budget + margin {need / GiB:.1f} GiB"})
    if workload is not None and expect:
        sess, turns, sha = workload
        if (expect.get("sessions") not in (None, sess)) or (expect.get("turns") not in (None, turns)):
            bad.append({"check": "workload_counts", "host_level": False,
                        "detail": f"{sess} sessions, {turns} turns; expected {expect.get('sessions')}, "
                                  f"{expect.get('turns')}"})
        if expect.get("sha256") and expect["sha256"] != sha:
            bad.append({"check": "workload_sha256", "host_level": False, "detail": sha})
    if sglang_patch is not None and not sglang_patch.startswith(("PATCHED", "ALREADY_PATCHED")):
        bad.append({"check": "sglang_patch", "host_level": False, "detail": sglang_patch.strip()})
    if vllm_patch is not None and not vllm_patch.startswith(("PATCHED", "ALREADY_PATCHED")):
        bad.append({"check": "vllm_patch", "host_level": False, "detail": vllm_patch.strip()})
    return bad


def read_markers(path):
    out = {}
    if os.path.exists(path):
        with open(path) as f:
            for ln in f:
                p = ln.split()
                if len(p) == 2:
                    out.setdefault(p[0], float(p[1]))
    return out


def _at(rows, t, key=lambda r: r["t"]):
    """Row nearest to time t (rows sorted by time)."""
    return min(rows, key=lambda r: abs(key(r) - t)) if rows else None


def psi_full_frac(tel_rows, cg, resource, t0, t1):
    """Fraction of the window [t0, t1] the cgroup spent fully stalled on resource (PSI 'full' total, us)."""
    rows = [r for r in tel_rows if cg in r.get("cg", {})]
    a, b = _at(rows, t0), _at(rows, t1)
    if a is None or b is None or t1 <= t0:
        return None
    k = f"{resource}.pressure"
    ta = a["cg"][cg].get(k, {}).get("full", {}).get("total")
    tb = b["cg"][cg].get(k, {}).get("full", {}).get("total")
    if ta is None or tb is None:
        return None
    return (tb - ta) / ((t1 - t0) * 1e6)


def _gpu_t(r):
    from datetime import datetime
    try:
        return datetime.strptime(r["timestamp"], "%Y/%m/%d %H:%M:%S.%f").timestamp()
    except (KeyError, ValueError):
        return None


def conn_failures(turn_rows):
    """(turn rows, rows that failed on a lost connection) of a replay output's rows."""
    rows = [r for r in turn_rows if "session" in r and "turn" in r and "ttft" in r]
    bad = sum(1 for r in rows if r.get("failed") and (r.get("conn_error")
                                                      or any(p in str(r.get("error", "")) for p in CONN_PATTERNS)))
    return len(rows), bad


def longest_stall(rows):
    """(seconds, start) of the longest stretch of the replay with requests in flight and no first token or completion,
    from a replay output's rows (turn rows' sent + ttft and sent + e2e, the concurrency_series row, wall_s). The
    in-flight count dropping to 0 ends a stretch (idle, not stalled), except in the last second (the cap's drain).
    (0.0, None) without a concurrency series."""
    INF = float("inf")
    ev = {0.0}
    for r in rows:
        if "ttft" in r and r.get("sent") is not None:
            for x in (r.get("ttft"), r.get("e2e")):
                if x is not None and x != INF:
                    ev.add(float(r["sent"]) + float(x))
    conc = next((r["concurrency_series"] for r in rows if "concurrency_series" in r), None)
    wall = next((r["wall_s"] for r in rows if "wall_s" in r), None)
    if not conc or wall is None:
        return 0.0, None
    bounds = sorted(e for e in ev if e <= wall) + [float(wall)]
    best = (0.0, None)
    for a, b in zip(bounds, bounds[1:]):
        if b - a <= best[0]:
            continue
        k = 0
        for t, n in conc:
            if t <= a:
                k = n
        if k > 0 and not any(a < t < b - 1.0 and n == 0 for t, n in conc):
            best = (b - a, a)
    return best


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path, errors="replace") as f:
            for ln in f:
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    pass
    return out


def verdict(*, markers, cg_before, cg_after, tel_rows, cg_name, gpu_rows, runner_rc, runner_log="", server_log="",
            regime=None, engine=None, settings=None, turn_rows=None, disk_bytes=None, tier_bytes_end=None,
            ram_tier=False, lmcache_log=""):
    """{'valid', 'reasons', 'window_s', 'covariates'} for one arm. gpu_rows: parse_gpu_csv rows of the arm's GPUs,
    their 'timestamp' in the host's local time (nvidia-smi's own format). turn_rows: the replay output's rows;
    disk_bytes: the cell's disk-tier capacity; tier_bytes_end: the runner's du of the tier after the replay; ram_tier:
    the tier is a tmpfs (its cachestat is the tier itself, not page-cache slack). lmcache_log: the LM arm's LMCache
    server log; its dropped stores (L1 allocations refused while write-through stores pin L1, and failed L2 store
    tasks) are covariates, never exclusions: they are how LMCache behaves at the cell's capacities."""
    s = dict(DEFAULTS, **(settings or {}))
    reasons, cov = [], {}
    t0, t1 = markers.get("replay_start"), markers.get("replay_end")
    died = "SERVER DIED" in runner_log
    if runner_rc != 0 or "server failed" in runner_log or died or t0 is None or t1 is None:
        reasons.append(f"engine_crash: rc={runner_rc}, markers={sorted(markers)}"
                       + (", server failed" if "server failed" in runner_log else "")
                       + (", server died during the replay" if died else ""))
    if lmcache_log:
        cov["lm_alloc_failed"] = lmcache_log.count("Failed to batched allocate")
        cov["lm_store_task_failed"] = lmcache_log.lower().count("store task failed")
    if turn_rows is not None:
        n, bad = conn_failures(turn_rows)
        frac = bad / n if n else 0.0
        cov["conn_failed"], cov["conn_failed_frac"] = bad, frac
        if frac > s["conn_fail_max_frac"]:
            reasons.append(f"engine death: {bad} of {n} turns ({frac:.2%}) failed on a lost connection "
                           f"> {s['conn_fail_max_frac']:.1%}")
        st, at = longest_stall(turn_rows)
        cov["stall_max_s"] = st
        if st >= s["engine_stall_s"]:
            reasons.append(f"engine stall: requests in flight and no token for {st:.0f} s from {at:.0f} s of the "
                           f"replay (>= {s['engine_stall_s']:.0f} s)")
    for k in ("oom", "oom_kill"):
        d = cg_after["memory.events"].get(k, 0) - cg_before["memory.events"].get(k, 0)
        if d > 0:
            reasons.append(f"memory.events {k} +{d}")
    if t0 is not None and t1 is not None:
        for res in ("memory", "io"):
            f = psi_full_frac(tel_rows, cg_name, res, t0, t1)
            cov[f"psi_full_{res}_frac"] = f
            if res in s["psi_resources"] and f is not None and f > s["psi_full_max_frac"]:
                reasons.append(f"psi full {res} {f:.2%} of the window > {s['psi_full_max_frac']:.0%}")
        win = [r for r in gpu_rows if (_gpu_t(r) or -1) >= t0 and (_gpu_t(r) or 1e18) <= t1]
        hits = [r for r in win if r["mask"] & EXCLUDE_MASK]
        if hits:
            reasons.append(f"gpu hardware/thermal event in window: {decode(hits[0]['mask'])} on GPU "
                           f"{hits[0].get('index')} ({len(hits)} samples)")
        by_gpu = {}
        for r in win:
            by_gpu.setdefault(r.get("index"), []).append(r)
        for g, rs in by_gpu.items():
            for c in EXCLUDE_COUNTERS:
                f = "clocks_event_reasons_counters." + c
                vals = [float(r[f]) for r in rs if r.get(f) not in (None, "", "[N/A]")]
                if len(vals) >= 2 and vals[-1] - vals[0] > 0:
                    reasons.append(f"gpu {g} {c} counter +{(vals[-1] - vals[0]) / 1e6:.3f} s in window")
            sm = [float(r["clocks.sm"]) for r in rs if r.get("clocks.sm") not in (None, "", "[N/A]")]
            pc = [float(r[f]) for r in rs for f in ["clocks_event_reasons_counters.sw_power_cap"]
                  if r.get(f) not in (None, "", "[N/A]")]
            temp = [float(r["temperature.gpu"]) for r in rs if r.get("temperature.gpu") not in (None, "", "[N/A]")]
            fan = [float(r["fan.speed"]) for r in rs if r.get("fan.speed") not in (None, "", "[N/A]")]
            cov[f"gpu{g}"] = {"sm_clock_median": statistics.median(sm) if sm else None, "sm_clock_min": min(sm, default=None),
                              "sw_power_cap_s": (pc[-1] - pc[0]) / 1e6 if len(pc) >= 2 else None,
                              "sw_power_cap_samples": sum(1 for r in rs if r["mask"] & 0x4),
                              "temp_max": max(temp, default=None), "fan_max": max(fan, default=None),
                              "samples": len(rs)}
        cov["window_s"] = t1 - t0
    if engine == "vllm" and regime in ("slow", "medium") and ODIRECT_FALLBACK in server_log:
        reasons.append("vLLM fell back to buffered I/O on the disk tier (O_DIRECT probe failed)")
    io_b = cg_before.get("io.stat", {})
    io_a = cg_after.get("io.stat", {})
    cov["io_bytes"] = {mm: {k: v - io_b.get(mm, {}).get(k, 0) for k, v in d.items() if k in ("rbytes", "wbytes")}
                       for mm, d in io_a.items()}
    cov["memory_peak"] = cg_after.get("memory.peak")
    cov["file_bytes_end"] = cg_after.get("memory.stat", {}).get("file")
    tier = [r["tier"] for r in tel_rows if r.get("tier")]
    if tier:
        cov["tier_cached_max"] = {k: max(t.get(k, {}).get("cached", 0) for t in tier) for k in tier[-1]}
    # disk-tier capacity: the largest size the tier directory reached (telemetry samples, the runner's final du)
    sizes = [t.get(cg_name, {}).get("bytes") for t in tier] + [tier_bytes_end]
    sizes = [x for x in sizes if x is not None]
    cov["tier_bytes_max"] = max(sizes) if sizes else None
    if disk_bytes and sizes and max(sizes) > disk_bytes:
        reasons.append(f"disk tier reached {max(sizes) / GiB:.2f} GiB > the cell's {disk_bytes / GiB:.2f} GiB")
    # page-cache slack, in the replay window: the tier's cached pages and the cgroup's non-shmem file pages
    if t0 is not None and t1 is not None:
        win = [r for r in tel_rows if t0 <= r.get("t", -1) <= t1]
        cached = [r["tier"][cg_name]["cached"] for r in win if cg_name in (r.get("tier") or {})]
        filec = []
        for r in win:
            st = (r.get("cg", {}).get(cg_name) or {}).get("memory.stat") or {}
            if st.get("file") is not None:
                filec.append(st["file"] - max(st.get("shmem") or 0, st.get("file_mapped") or 0))
        cov["page_cache"] = {"tier_cached_max": max(cached, default=None), "cg_file_nonshmem_max": max(filec, default=None),
                             "limit": s["page_cache_max_bytes"], "ram_tier": ram_tier}
        lim = s["page_cache_max_bytes"]
        if cached and not ram_tier and max(cached) > lim:
            reasons.append(f"page cache: tier cached {max(cached) / MiB:.0f} MiB > {lim / MiB:.0f} MiB in the window")
        # BENCH_PLAN 5: tier cachestat, or (no tier to measure) the cgroup's unmapped non-shmem file pages. With a tier,
        # the cgroup figure is a covariate: pilot 2 traced its growth to filesystem metadata (about 6 KB of block-device
        # cache per tier file: Gemma's 2.6 MB chunks, 56,000 files in one arm) and lazily loaded library code, with
        # the tier's own pages at 0 to 4 KiB.
        if filec and not cached and max(filec) > lim:
            reasons.append(f"page cache: cgroup file (unmapped, non-shmem) {max(filec) / MiB:.0f} MiB > "
                           f"{lim / MiB:.0f} MiB in the window")
    return {"valid": not reasons, "reasons": reasons, "covariates": cov}
