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
After the arm (on the replay window, cut by the runner's replay_start and replay_end markers):
  - any memory.events oom or oom_kill increase (whole arm, boot included)
  - PSI "full" stall time over 1% of the window, for each resource in psi_resources (default memory; io is logged.
    Under io.max every disk arm stalls on I/O by design, so io PSI measures the regime, not a fault; see the report)
  - a hardware slowdown or thermal event on the arm's GPUs in the window (sampled reason bitmask, and the cumulative
    clock-event counters where the driver has them)
  - engine crash: the runner's rc, "server failed", no replay_end marker
  - vLLM fell back from O_DIRECT on a disk tier (its own warning in the server log), slow and medium regimes
Covariates logged: SM clock per GPU (median and min in the window), software power-cap time, max temperature, io PSI.
"""
import csv
import hashlib
import io
import json
import os
import statistics

from common import GiB

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
            "psi_resources": ["memory"]}
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
                   fake='"clocks_event_reasons.active"\n' + "\n".join(f'"{c}"' for c in COUNTER_FIELDS)).out
    fields = list(GPU_FIELDS)
    if '"clocks_event_reasons.active"' not in h:
        fields[-1] = "clocks_throttle_reasons.active"
    fields += [c for c in COUNTER_FIELDS if f'"{c}"' in h]
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
            sglang_patch=None, settings=None):
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


def verdict(*, markers, cg_before, cg_after, tel_rows, cg_name, gpu_rows, runner_rc, runner_log="", server_log="",
            regime=None, engine=None, settings=None):
    """{'valid', 'reasons', 'window_s', 'covariates'} for one arm. gpu_rows: parse_gpu_csv rows of the arm's GPUs,
    their 'timestamp' in the host's local time (nvidia-smi's own format)."""
    s = dict(DEFAULTS, **(settings or {}))
    reasons, cov = [], {}
    t0, t1 = markers.get("replay_start"), markers.get("replay_end")
    if runner_rc != 0 or "server failed" in runner_log or t0 is None or t1 is None:
        reasons.append(f"engine_crash: rc={runner_rc}, markers={sorted(markers)}"
                       + (", server failed" if "server failed" in runner_log else ""))
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
            cov[f"gpu{g}"] = {"sm_clock_median": statistics.median(sm) if sm else None, "sm_clock_min": min(sm, default=None),
                              "sw_power_cap_s": (pc[-1] - pc[0]) / 1e6 if len(pc) >= 2 else None,
                              "sw_power_cap_samples": sum(1 for r in rs if r["mask"] & 0x4),
                              "temp_max": max(temp, default=None), "samples": len(rs)}
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
    return {"valid": not reasons, "reasons": reasons, "covariates": cov}
