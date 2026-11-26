#!/usr/bin/env python3
"""Phase R run orchestrator: one block (every arm of one cell on one host in one repeat), as BENCH_PLAN section 7 runs
it. Runs on the GPU host as root; the engines run through the existing runners (engine/run_arms.sh for vLLM,
engine/run_arms_sglang.sh for SGLang), on the host or inside the long-lived `park` container (engine/node_prep.sh).

  orchestrate.py preflight --cell CELL.json --host ID [--out DIR] [--dry-run]
  orchestrate.py plan      --cell CELL.json --repeat N [--attempt A] [--aa LABEL]
  orchestrate.py regime    --pilot PILOT.json --load heavy --read-bps B --write-bps B
  orchestrate.py run       --cell CELL.json --host ID --repeat N --out DIR [--attempt A] [--aa LABEL] [--dry-run]
  orchestrate.py hicache-pilot --cell CELL.json --host ID --out DIR [--dry-run]     (SGLang cells)

A block, per attempt:
  1. Full fio probe of every slot's disk (O_DIRECT 4 MiB write, read, random read); the host's first full probe is
     kept in OUT/hosts/HOST/ and every later quick probe is held to it (10%).
  2. Regime check before any arm: r and w from the pilot JSON and the probe measured inside a cgroup carrying the
     cell's io.max (or the RAM tier, buffered); the read rate is min(sequential read, random read at the tier's chunk
     size kv_chunk_bytes), the read path the engine issues. A cell whose measured regime differs from the declared
     one stops here (exit 4), before an arm runs.
  3. Arms in the seeded random order (order.py), in lockstep rounds when the node runs two engines; a round with
     fewer arms than slots fills the empty slot with an unscored filler (arm C0's runner on the same workload, its
     replay output deleted, indexed with status "filler"), so every measured arm has a neighbour; the time each arm's
     replay window overlapped its neighbour's is a covariate (verdict covariates "neighbour"). Before each
     round: the previous engines' cgroups killed and removed, page cache dropped, quick probe, disk tier wiped (rm and
     fstrim, mkfs, or a fresh tmpfs), page cache dropped again, a fresh cgroup per engine (memory.max, swap 0, cpuset,
     io.max) read back, a throttled probe inside it when io.max is set, then the health checks (health.py).
  4. During the arm: telemetry.py (cgroup memory, PSI, io.stat, diskstats, tier page-cache residency), nvidia-smi
     -lms 1000 (clocks, clock-event reasons and counters) and iostat, each to a file named by cell, arm and host.
  5. After the arm: verdict (health.py: OOM, PSI full > 1% of the replay window, hardware or thermal GPU event,
     engine crash or death (the runner's check, or > 0.5% of turns failed on a lost connection), vLLM O_DIRECT
     fallback, disk tier above capacities.disk_bytes, page cache above 256 MiB). Any exclusion ends the attempt;
     the whole block reruns (pairing rule),
     on this host (exit code path "rerun") or, for a host-level failure, on a replacement host (exit 3).
  6. Full probe again after the block.

Exit codes: 0 block done and valid; 2 escalate (attempts used up; the owner decides); 3 host fault (rerun the block
on a replacement host with --attempt A+1); 4 configuration or regime mismatch; 5 preflight failure.

Outputs (OUT/CELL/HOST/rREPEAT/aATTEMPT/POS.LABEL/): the runner's per-turn jsonl, server, client and runner logs
renamed to CELL.LABEL.HOST.*, plus CELL.LABEL.HOST.{harness_env.json, health.json, probe.json, cgroup.json,
telemetry.jsonl, gpu.csv, iostat.log, markers, verdict.json}; block.json per attempt (order, rounds, outcome);
commands.log (every command, its kind: root, gpu or plain); OUT/runs.jsonl indexes every arm of every attempt.

--dry-run prints every command and kernel-file write with its kind and executes none (no root, no GPU, no disk),
with canned probe and GPU values (the cell's "dry_run" section can set them), so a cell file can be walked end to end
on any machine.

The cell file format is documented in engine/phase_r/examples/cell_main_vllm_slow_heavy.json. hard_cap_s (optional)
becomes the runner's HARD_CAP (replay --hard-cap: 3x C0's pilot duration in the plan). Arm codes are checked against the
runner's (ARM_CODES); CT needs CT_ETA in its env, T2 takes T2_MIN_HITS, T2_DECAY_S and T2_COUNT. The pilot JSON must
be for the cell's engine. capacities.disk_bytes (required) is the disk tier's capacity (the runner's DISK_BYTES).
Every CPU tier gets capacities.cpu_bytes exactly (LMCache's L1 included; no env may override cpu_bytes_to_use).

Live scoring: an arm with code K needs the cell's "live_scoring" {"side_file", "ckpt", "calib", optional "device"}
(the replay's side file and the Layby-Dwell release, paths where the engine runs); without it, or with a file
missing, the cell is refused (exit 4). The runner starts the sidecar in the arm's cgroup, so the pilot's (or
cgroup's) sidecar_rss_bytes is added to K's RSS in memory.max. The sidecar listens on the slot's sidecar_port
(default park_port + 1).

hicache-pilot (SGLang cells): HC-wt's L3 write throughput with the fdatasync patch, the sync_file_range variant and
stock SGLang, each inside a cgroup carrying the cell's io.max (engine/phase_r/sglang/hicache_write_pilot.py), written
to OUT/CELL/HOST/hicache_pilot.json. The pilot decides the variant; the cell's "sglang_patch_variant"
(fdatasync, the default, or sync_file_range) is what every SGLang arm then runs with.
"""
import argparse
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cgroup  # noqa: E402
import health  # noqa: E402
import order  # noqa: E402
import probe  # noqa: E402
import regime  # noqa: E402
from common import GiB, Runner, append_jsonl, dump, load_json, log, parse_size, read_text  # noqa: E402

EXIT_OK, EXIT_ESCALATE, EXIT_HOST_FAULT, EXIT_CONFIG, EXIT_PREFLIGHT = 0, 2, 3, 4, 5
HOST_CG_EXEC = os.path.join(HERE, "cg_exec.sh")   # probes run on the host; the engine's copy is under layby_root
RUNNERS = {"vllm": "engine/run_arms.sh", "sglang": "engine/run_arms_sglang.sh"}
WIPES = ("fstrim", "mkfs", "ram")
# the runners' arm codes (a repeat suffix ".N" allowed); the docstrings of run_arms.sh and run_arms_sglang.sh say
# what each one runs
ARM_CODES = {"vllm": {"C0", "T", "T2", "LM", "LMR", "LMS", "CT", "PT", "K", "K_km", "K_ewma", "K0", "KW", "KD", "SM"},
             "sglang": {"C0", "T", "K", "K_km", "K_ewma"}}
LIVE_CODES = {"K"}                       # arms that score live through the sidecar (K_km and K_ewma post controls)
PATCH_VARIANTS = ("fdatasync", "sync_file_range")
FILLER_CODE = "C0"


def lmcache_gb(n):
    """The --l1-size-gb / max_capacity_gb string run_arms.sh passes for n bytes, and the bytes LMCache makes of it
    (int(float(gb) * 2^30))."""
    gb = repr(int(n) / 2 ** 30)
    return gb, int(float(gb) * (1 << 30))


# ---------------------------------------------------------------- cell spec

def load_cell(path, aa=None):
    c = load_json(path)
    base = os.path.dirname(os.path.abspath(path))
    for k in ("cell", "seed", "engine", "model", "regime", "load", "workload", "capacities", "runner", "arms"):
        if k not in c:
            raise ValueError(f"cell spec lacks {k!r}")
    if c["engine"] not in RUNNERS:
        raise ValueError(f"engine {c['engine']!r} not in {sorted(RUNNERS)}")
    if c["regime"] not in ("slow", "medium", "fast"):
        raise ValueError(f"regime {c['regime']!r}")
    if "slots" not in c:
        c["slots"] = [{k: c[k] for k in ("cuda", "port", "park_port", "cpus", "mems", "disk") if k in c}]
        c["slots"][0].setdefault("name", "A")
    for s in c["slots"]:
        for k in ("name", "cuda", "port", "park_port", "cpus", "mems", "disk"):
            if k not in s:
                raise ValueError(f"slot {s.get('name')} lacks {k!r}")
        d = s["disk"]
        d.setdefault("tier_subdir", "kvtier")
        d.setdefault("wipe", "fstrim")
        if d["wipe"] not in WIPES:
            raise ValueError(f"slot {s['name']}: wipe {d['wipe']!r} not in {WIPES}")
        if d["wipe"] == "mkfs" and not d.get("device"):
            raise ValueError(f"slot {s['name']}: mkfs needs disk.device (a dedicated device)")
        if d["wipe"] == "ram" and not c.get("ram_tier_bytes"):
            raise ValueError("a RAM tier needs ram_tier_bytes")
        d["tier_dir"] = os.path.join(d["mount"], d["tier_subdir"])
        d.setdefault("probe_dir", None if d["wipe"] == "ram" else d["mount"])
    if c["regime"] == "slow" and any(s["disk"]["wipe"] != "ram" for s in c["slots"]) and not c.get("io_max") \
            and not c.get("io_max_none_reason"):
        raise ValueError("slow regime on a disk without io_max: give io_max, or io_max_none_reason for a real slow "
                         "disk (the anchor cell)")
    if c["regime"] == "fast" and any(s["disk"]["wipe"] != "ram" for s in c["slots"]):
        log("note: fast regime on a real disk (not the RAM tier)")
    if c.get("ram_tier_bytes") and any(s["disk"]["wipe"] != "ram" for s in c["slots"]):
        raise ValueError("ram_tier_bytes set but a slot's disk is not wipe=ram")
    cpus = [cgroup._cpus(s["cpus"]) for s in c["slots"]]
    for i in range(len(cpus)):
        for j in range(i + 1, len(cpus)):
            if cpus[i] & cpus[j]:
                raise ValueError("slots share CPUs")
    labels = [a["label"] for a in c["arms"]]
    if len(set(labels)) != len(labels) or any("." in x or "/" in x or x.startswith("_") for x in labels):
        raise ValueError(f"arm labels must be unique, without dots or slashes, not starting with _: {labels}")
    caps = c["capacities"]
    for k in ("gpu_kv_bytes", "cpu_bytes", "disk_bytes"):
        if k not in caps:
            raise ValueError(f"capacities lacks {k!r}")
    if c.get("ram_tier_bytes") and parse_size(c["ram_tier_bytes"]) < parse_size(caps["disk_bytes"]):
        raise ValueError("ram_tier_bytes is smaller than capacities.disk_bytes: the RAM tier would fill first")
    if not c.get("kv_chunk_bytes"):
        raise ValueError("kv_chunk_bytes missing: the regime's read rate is min(sequential read, random read at the "
                         "tier's chunk size)")
    if c.get("sglang_patch_variant", "fdatasync") not in PATCH_VARIANTS:
        raise ValueError(f"sglang_patch_variant {c['sglang_patch_variant']!r} not in {PATCH_VARIANTS}")
    for a in c["arms"]:
        a.setdefault("code", a["label"])
        a.setdefault("env", {})
        check_arm(c, a)
    cap = c.get("hard_cap_s")
    if cap is not None:
        rt = float((c["runner"].get("env") or {}).get("REPLAY_TIMEOUT", 5400))
        if rt < float(cap) + 300:
            raise ValueError(f"runner env REPLAY_TIMEOUT {rt:g} must exceed hard_cap_s {cap} by 300 s")
    if aa:
        src = [a for a in c["arms"] if a["label"] == aa]
        if not src:
            raise ValueError(f"--aa {aa}: no such arm")
        c["arms"] = [dict(a, aa_of=src[0]["label"]) for a in order.aa_arms(src[0])]
        c["cell"] = f"{c['cell']}-AA"
    pil = c.get("pilot")
    c["_pilot"] = load_json(pil if os.path.isabs(pil) else os.path.join(base, pil)) if pil else None
    if c["_pilot"] is not None and c["_pilot"].get("engine") != c["engine"]:
        raise ValueError(f"pilot {pil} is for engine {c['_pilot'].get('engine')!r}, the cell runs {c['engine']!r}")
    for s in c["slots"]:
        s.setdefault("sidecar_port", int(s["park_port"]) + 1)
    if any(is_live(a) for a in c["arms"]):
        ls = c.get("live_scoring") or {}
        miss = [k for k in ("side_file", "ckpt", "calib") if not ls.get(k)]
        if miss:
            raise ValueError(f"arm K scores live: the cell needs live_scoring {miss} (the replay's side file and the "
                             f"Layby-Dwell release)")
    check_cpu_tiers(c)
    c.setdefault("max_block_attempts", 2)
    c.setdefault("fio", {})
    c.setdefault("health", {})
    c.setdefault("telemetry", {})
    c.setdefault("cgroup", {})
    c.setdefault("dry_run", {})
    for a in c["arms"]:   # every arm has its own RSS before anything runs
        arm_rss(c, a)
    if len(c["slots"]) > 1 and len(c["arms"]) % len(c["slots"]):
        arm_rss(c, filler_arm(c))   # a filler runs C0's configuration
    return c


def is_live(arm):
    return arm["code"].split(".")[0] in LIVE_CODES


def filler_arm(c):
    """The unscored neighbour of an arm that would run alone in a two-engine round: C0's runner on the same workload,
    C0's RSS (the cell's arm with code C0, else the label C0)."""
    src = next((a.get("rss_of") or a["label"] for a in c["arms"] if a["code"].split(".")[0] == FILLER_CODE),
               FILLER_CODE)
    return {"label": order.FILLER, "code": FILLER_CODE, "env": {}, "filler": True, "rss_of": src}


CPU_TIER_ALIGN = 64 << 20   # LMCache lazy_memory_allocator.PIN_CHUNK_SIZE


def check_cpu_tiers(c):
    """Every arm's CPU tier is capacities.cpu_bytes exactly: no env overrides cpu_bytes_to_use, and LMCache's L1
    (a float of GiB) gives back the same bytes. Returns {label: bytes}."""
    cpu = parse_size(c["capacities"]["cpu_bytes"])
    if cpu % CPU_TIER_ALIGN:
        # LMCache 0.5.5's lazy L1 allocator rounds its address space up to 64 MiB but registers the configured size
        # with NIXL: a store placed in the tail is rejected (NIXL_ERR_INVALID_PARAM, pilot 2: 16 store tasks, about 4%
        # of keys dropped) and arm LM gets more L1 than the cell; below a page it fails at startup (pilot 1)
        raise ValueError(f"capacities.cpu_bytes {cpu} is not a multiple of {CPU_TIER_ALIGN} B (LMCache's L1 pin chunk)")
    envs = [c["runner"].get("env") or {}] + [s.get("env") or {} for s in c["slots"]] + [a["env"] for a in c["arms"]]
    for e in envs:
        if "cpu_bytes_to_use" in str(e.get("OFF_EXTRA", "")):
            raise ValueError("OFF_EXTRA sets cpu_bytes_to_use: every arm gets capacities.cpu_bytes")
        if str(e.get("LIVE_SCORING", "on")) == "off":
            raise ValueError("LIVE_SCORING=off is for pre-Phase R runs: arm K scores live in Phase R")
    tiers = {}
    for a in c["arms"]:
        code = a["code"].split(".")[0]
        tiers[a["label"]] = lmcache_gb(cpu)[1] if code in ("LM", "LMR", "LMS") else cpu
    if len(set(tiers.values())) > 1:
        raise ValueError(f"CPU-tier bytes differ across arms: {tiers}")
    return tiers


def check_arm(c, a):
    """The arm's code is one the cell's runner knows, and the settings its configuration needs are there."""
    code = a["code"].split(".")[0]
    if code not in ARM_CODES[c["engine"]]:
        raise ValueError(f"arm {a['label']}: code {a['code']!r} unknown to the {c['engine']} runner "
                         f"({sorted(ARM_CODES[c['engine']])})")
    env = dict(c["runner"].get("env") or {}, **a["env"])
    if code == "CT":
        if "CT_ETA" not in env:
            raise ValueError(f"arm {a['label']}: CT needs env CT_ETA (layby.continuum.memoryfulness of the workload)")
        if not 0.0 <= float(env["CT_ETA"]) <= 1.0:
            raise ValueError(f"arm {a['label']}: CT_ETA {env['CT_ETA']} outside [0, 1]")
    if code == "T2":
        if env.get("T2_COUNT", "request") not in ("request", "store"):
            raise ValueError(f"arm {a['label']}: T2_COUNT must be request or store")
        if int(env.get("T2_MIN_HITS", 2)) < 1 or float(env.get("T2_DECAY_S", 600)) < 0:
            raise ValueError(f"arm {a['label']}: T2_MIN_HITS >= 1 and T2_DECAY_S >= 0")


def arm_by_label(c):
    return {a["label"]: a for a in c["arms"]}


def arm_rss(c, arm):
    """(bytes, {process: bytes}) of one arm's measured RSS beyond its pinned tier, from the cell's
    cgroup.arm_rss_bytes or the pilot's arm_rss_bytes, keyed by arm label (an A/A arm uses its base label). A value is
    bytes, or a map of the arm's processes to bytes (summed): an LMCache arm counts its LMCache server, which runs in
    the same cgroup (run_arms.sh starts it through SERVER_WRAP too). There is no per-engine default: every arm needs
    its own number."""
    table = c["cgroup"].get("arm_rss_bytes")
    if table is None and c["_pilot"]:
        table = c["_pilot"].get("arm_rss_bytes")
    if not isinstance(table, dict):
        raise ValueError("no per-arm RSS: give the pilot's arm_rss_bytes (or cgroup.arm_rss_bytes), keyed by label")
    label = arm["label"]
    base = arm.get("rss_of") or arm.get("aa_of")
    v = table.get(label, table.get(base) if base else None)
    if v is None:
        raise ValueError(f"arm_rss_bytes has no entry for arm {label!r}")
    parts = {k: parse_size(x) for k, x in v.items()} if isinstance(v, dict) else {"engine": parse_size(v)}
    if is_live(arm) and "sidecar" not in parts:   # the live-scoring sidecar runs in the arm's cgroup
        sc = c["cgroup"].get("sidecar_rss_bytes")
        if sc is None and c["_pilot"]:
            sc = c["_pilot"].get("sidecar_rss_bytes")
        if sc is None:
            raise ValueError(f"arm {label!r} scores live: give the pilot's sidecar_rss_bytes (or cgroup's), the "
                             f"sidecar's RSS, counted in the arm's memory.max")
        parts["sidecar"] = parse_size(sc)
    return sum(parts.values()), parts


def budget_of(c, arm):
    cg = c["cgroup"]
    pinned = parse_size(cg.get("pinned_bytes", c["capacities"]["cpu_bytes"]))
    rss, _ = arm_rss(c, arm)
    ram = parse_size(c["ram_tier_bytes"]) if c.get("ram_tier_bytes") else 0
    return cgroup.budget(pinned, rss, parse_size(cg.get("slack_bytes", cgroup.SLACK)), ram)


def io_limits(c, dev):
    m = c.get("io_max")
    if not m or dev is None or dev.get("majmin") is None:
        return []
    return [(dev["majmin"], parse_size(m["rbps"]) if m.get("rbps") else None,
             parse_size(m["wbps"]) if m.get("wbps") else None)]


def cg_name(c, slot, label, repeat, attempt):
    return f"{c['cell']}.r{repeat}a{attempt}.{slot['name']}.{label}"


def layby_path(c, rel):
    return os.path.join(c["runner"]["layby_root"], rel)


def arm_env(c, slot, arm, cgpath):
    r = c["runner"]
    env = {"VENV": r["venv"], "ENGINE_DIR": layby_path(c, "engine"), "REPO": r["layby_root"], "MODEL": c["model"],
           "PORT": str(slot["port"]), "PARK_PORT": str(slot["park_port"]), "CUDA_VISIBLE_DEVICES": str(slot["cuda"]),
           "PYTHONHASHSEED": "0", "SERVER_WRAP": f"{layby_path(c, 'engine/phase_r/cg_exec.sh')} {cgpath}",
           "ARM_HOOK": layby_path(c, "engine/phase_r/arm_hook.sh")}
    if c.get("hard_cap_s") is not None:
        env["HARD_CAP"] = str(c["hard_cap_s"])
    env["DISK_BYTES"] = str(parse_size(c["capacities"]["disk_bytes"]))
    if is_live(arm):
        ls = c["live_scoring"]
        env.update(SIDE_FILE=ls["side_file"], SIDECAR_CKPT=ls["ckpt"], SIDECAR_CALIB=ls["calib"],
                   SIDECAR_PORT=str(slot["sidecar_port"]))
        if ls.get("device"):
            env["SIDECAR_DEVICE"] = str(ls["device"])
    env.update({k: str(v) for k, v in (r.get("env") or {}).items()})
    env.update({k: str(v) for k, v in (slot.get("env") or {}).items()})
    env.update({k: str(v) for k, v in arm["env"].items()})
    return env


def exec_prefix(c, env):
    r = c["runner"]
    if r.get("mode", "host") == "docker":
        cmd = ["docker", "exec"]
        for k, v in env.items():
            cmd += ["-e", f"{k}={v}"]
        return cmd + [r["container"]]
    return ["env"] + [f"{k}={v}" for k, v in env.items()]


def arm_command(c, slot, arm, armdir, cgpath):
    env = arm_env(c, slot, arm, cgpath)
    caps = c["capacities"]
    return exec_prefix(c, env) + ["bash", layby_path(c, RUNNERS[c["engine"]]), c["workload"]["path"], armdir,
                                  str(parse_size(caps["gpu_kv_bytes"])), str(parse_size(caps["cpu_bytes"])),
                                  slot["disk"]["tier_dir"], arm["code"]]


def in_runner(c, cmd):
    """A command run where the engine runs (inside the container in docker mode)."""
    r = c["runner"]
    if r.get("mode", "host") == "docker":
        return ["docker", "exec", r["container"]] + cmd
    return cmd


# ---------------------------------------------------------------- host actions

def drop_caches(rn):
    rn.run(["sync"], kind="root")
    rn.write("/proc/sys/vm/drop_caches", 3)


def wipe(rn, c, slot, dev):
    """Empty the disk tier: rm and fstrim, mkfs, or a fresh tmpfs. Returns notes (fstrim unsupported is a note)."""
    d, notes = slot["disk"], []
    if d["wipe"] == "fstrim":
        rn.run(["rm", "-rf", d["tier_dir"]], kind="root")
        rn.run(["mkdir", "-p", d["tier_dir"]], kind="root")
        rn.run(["sync"], kind="root")
        r = rn.run(["fstrim", "-v", d["mount"]], kind="root", fake=f"{d['mount']}: 0 B trimmed\n")
        if not r.ok:
            notes.append(f"fstrim {d['mount']} rc={r.rc}: {r.err.strip()[:200]} (discard unsupported; rm only)")
    elif d["wipe"] == "mkfs":
        fs = d.get("mkfs_type", "ext4")
        rn.run(["umount", d["mount"]], kind="root")
        rn.run(["wipefs", "-a", d["device"]], kind="root", check=not rn.dry)
        mk = ["mkfs.ext4", "-F", "-q", "-E", "lazy_itable_init=0,lazy_journal_init=0"] if fs == "ext4" \
            else [f"mkfs.{fs}", "-f", "-q"]
        rn.run(mk + [d["device"]], kind="root", check=not rn.dry)
        rn.run(["mount", d["device"], d["mount"]], kind="root", check=not rn.dry)
        rn.run(["mkdir", "-p", d["tier_dir"]], kind="root")
    else:
        size = parse_size(c["ram_tier_bytes"])
        if not rn.dry and os.path.ismount(d["mount"]):
            rn.run(["umount", d["mount"]], kind="root", check=True)
        rn.run(["mkdir", "-p", d["mount"]], kind="root")
        rn.run(["mount", "-t", "tmpfs", "-o", f"size={size},mode=0755", "layby_ram_tier", d["mount"]], kind="root",
               check=not rn.dry)
        rn.run(["mkdir", "-p", d["tier_dir"]], kind="root")
    return notes


def host_env(rn, c, host):
    def out(cmd, fake=""):
        r = rn.run(cmd, fake=fake)
        return r.out.strip() if r.ok else None
    env = {"host": host, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "uname": out(["uname", "-a"]), "cpu": out(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo"]),
           "nproc": out(["nproc"]), "meminfo_total": out(["sh", "-c", "grep MemTotal /proc/meminfo"]),
           "gpus": out(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total,pcie.link.gen.max,"
                        "pcie.link.width.max,vbios_version", "--format=csv,noheader"]),
           "lsblk": out(["lsblk", "-o", "NAME,MAJ:MIN,SIZE,ROTA,TYPE,MOUNTPOINTS,MODEL"]),
           "mounts": [out(["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", "--target", s["disk"]["mount"]])
                      for s in c["slots"]],
           "layby_commit": out(["git", "-C", c["runner"]["layby_root"], "rev-parse", "HEAD"]),
           "cgroup_controllers": rn.read("/sys/fs/cgroup/cgroup.controllers", fake="cpuset cpu io memory").strip()}
    if c["runner"].get("mode") == "docker":
        env["container_image"] = out(["docker", "inspect", "-f", "{{.Image}} {{.Config.Image}}",
                                      c["runner"]["container"]])
    return env


def sglang_patch(rn, c, mode):
    py = os.path.join(c["runner"]["venv"], "bin", "python")
    variant = c.get("sglang_patch_variant", "fdatasync")
    r = rn.run(in_runner(c, ["bash", layby_path(c, "engine/phase_r/sglang/apply_patch.sh"), py, mode, variant]),
               fake=f"PATCHED 0.5.21 {variant} (dry-run)\n")
    return r.out.strip() if r.ok else (r.out.strip() or r.err.strip() or f"rc={r.rc}")


def vllm_patch(rn, c, mode):
    """vLLM PR #60045, the sliding-window admission count (engine/phase_r/vllm/pr60045_swa_admission.patch), on the
    engine's vLLM: every vLLM arm runs with it, stock baselines and Layby alike."""
    py = os.path.join(c["runner"]["venv"], "bin", "python")
    r = rn.run(in_runner(c, [py, layby_path(c, "engine/phase_r/vllm/apply_patch.py"), mode]),
               fake="PATCHED 0.31.0 pr60045 (dry-run)\n")
    return r.out.strip() if r.ok else (r.out.strip() or r.err.strip() or f"rc={r.rc}")


# ---------------------------------------------------------------- preflight

def live_problems(rn, c):
    """Arm K's live-scoring inputs, checked where the engine runs (inside the container in docker mode): a missing
    one is a configuration error (exit 4), like the runner's own check."""
    if not any(is_live(a) for a in c["arms"]):
        return []
    out = []
    for k in ("side_file", "ckpt", "calib"):
        p = c["live_scoring"][k]
        if not rn.run(in_runner(c, ["test", "-f", p])).ok:
            out.append(f"live_scoring {k} {p} missing")
    return out


def preflight(rn, c, out_root=None):
    problems = []
    if not rn.dry and os.geteuid() != 0:
        problems.append("not root (cgroup writes, drop_caches, fstrim, mkfs and mount need root)")
    problems += cgroup.check_v2(rn)
    tools = ["fio", "nvidia-smi", "findmnt"]
    tools += {"fstrim": ["fstrim"], "mkfs": ["wipefs", "mkfs." + c["slots"][0]["disk"].get("mkfs_type", "ext4")],
              "ram": []}[c["slots"][0]["disk"]["wipe"]]
    if c["runner"].get("mode") == "docker":
        tools.append("docker")
    for t in tools:
        if not rn.dry and shutil.which(t) is None:
            problems.append(f"{t} not on PATH")
    if not rn.dry and shutil.which("iostat") is None:
        log("iostat not installed (sysstat): diskstats in telemetry.jsonl still carry the same counters")
    for rel in (RUNNERS[c["engine"]], "engine/phase_r/cg_exec.sh", "engine/phase_r/arm_hook.sh"):
        p = layby_path(c, rel)
        if not rn.dry and not os.path.exists(p):
            problems.append(f"missing {p} (runner.layby_root must be the same path on the host and in the container)")
    w = c["workload"]["path"]
    if not rn.dry and not os.path.exists(w):
        problems.append(f"workload {w} missing")
    if c["_pilot"] is None:
        problems.append("no pilot JSON (cell 'pilot'): the regime check and the RSS budget need it")
    else:
        try:
            regime.load_inputs(c["_pilot"], c["load"])
        except ValueError as e:
            problems.append(str(e))
    for s in c["slots"]:
        d = s["disk"]
        if not rn.dry and not os.path.isdir(d["mount"]) and d["wipe"] != "ram":
            problems.append(f"slot {s['name']}: {d['mount']} missing")
        if d["wipe"] != "ram" and c.get("io_max"):
            dev = cgroup.block_device(rn, d["mount"], d.get("device"))
            if dev["majmin"] is None:
                problems.append(f"slot {s['name']}: no block device under {d['mount']} ({dev['fstype']}): io.max "
                                f"cannot apply")
    if c["runner"].get("mode") == "docker":
        ctr = c["runner"]["container"]
        r = rn.run(["docker", "inspect", "-f", "{{.State.Running}} {{.HostConfig.Privileged}} "
                    "{{.HostConfig.CgroupnsMode}}", ctr], fake="true true host\n")
        st = r.out.split()
        if st[:1] != ["true"]:
            problems.append(f"container {ctr} not running")
        elif st[1:3] != ["true", "host"]:
            problems.append(f"container {ctr} needs --privileged --cgroupns=host (node_prep.sh DOCKER_EXTRA); has "
                            f"privileged={st[1:2]} cgroupns={st[2:3]}")
        cgroup.ensure_parent(rn)
        r = rn.run(["docker", "exec", ctr, "test", "-w", "/sys/fs/cgroup/layby/cgroup.procs"])
        if not r.ok:
            problems.append(f"/sys/fs/cgroup/layby not writable inside {ctr}")
        for s in c["slots"]:
            mark = os.path.join(s["disk"]["mount"], ".layby_visible")
            rn.run(["touch", mark], kind="root")
            if not rn.run(["docker", "exec", ctr, "test", "-f", mark]).ok:
                problems.append(f"{s['disk']['mount']} not visible inside {ctr} at the same path (bind it with "
                                f"bind-propagation=rshared so the tier's mounts reach the container)")
            rn.run(["rm", "-f", mark], kind="root")
        if out_root and not rn.run(["docker", "exec", ctr, "test", "-d", out_root]).ok:
            problems.append(f"output dir {out_root} not visible inside {ctr} at the same path")
    if c["engine"] == "sglang":
        st = sglang_patch(rn, c, "apply")
        if not st.startswith(("PATCHED", "ALREADY_PATCHED")):
            problems.append(f"SGLang fadvise patch: {st}")
    if c["engine"] == "vllm":
        st = vllm_patch(rn, c, "apply")
        if not st.startswith(("PATCHED", "ALREADY_PATCHED")):
            problems.append(f"vLLM PR #60045 patch: {st}")
    return problems


# ---------------------------------------------------------------- one block

def restore_read_Bps(p):
    """The read rate r is computed from: the slower of sequential reads and random reads of one tier chunk
    (kv_chunk_bytes, the size the engine reads), since a restore issues chunk reads, not one long stream."""
    rr = p.get("rand_read_Bps")
    return min(p["seq_read_Bps"], rr) if rr else p["seq_read_Bps"]


class Block:
    def __init__(self, rn, c, host, repeat, out, attempt):
        self.rn, self.c, self.host, self.repeat, self.out, self.attempt = rn, c, host, repeat, out, attempt
        self.repdir = os.path.join(out, c["cell"], host, f"r{repeat}")
        self.fields = None
        self.devs = {}
        self.dry = c["dry_run"]

    def first_probe_path(self, slot):
        return os.path.join(self.out, "hosts", self.host, f"first_probe.{slot['name']}.json")

    def probe_raw(self, slot, kind):
        d = slot["disk"]
        if not d.get("probe_dir"):
            return None
        return probe.run(self.rn, d["probe_dir"], kind, self.c["fio"], direct=True,
                         rand_bs=self.c.get("kv_chunk_bytes"), fake=self.dry.get("probe"))

    def regime_check(self, attdir, before):
        """Measured regime per slot before any arm: probe inside a cgroup with the cell's io.max (or the RAM tier);
        a slot with neither uses the block's raw probe (before)."""
        c, rn, res = self.c, self.rn, {}
        ok = True
        for s in c["slots"]:
            d = s["disk"]
            if d["wipe"] == "ram":
                wipe(rn, c, s, None)
                p = probe.run(rn, d["tier_dir"], "full", dict(c["fio"], size=c["fio"].get("ram_probe_size", "2G")),
                              direct=False, rand_bs=c.get("kv_chunk_bytes"), fake=self.dry.get("ram_probe"))
            elif c.get("io_max"):
                name = f"{c['cell']}.probe.{s['name']}"
                p_cg = cgroup.create(rn, name, 4 * GiB, s["cpus"], s["mems"], io_limits(c, self.devs[s["name"]]))
                p = probe.run(rn, d["mount"], "full", c["fio"], direct=True,
                              wrap=[HOST_CG_EXEC, p_cg],
                              rand_bs=c.get("kv_chunk_bytes"), fake=self.dry.get("throttled_probe"))
                cgroup.destroy(rn, p_cg)
            else:
                p = before[s["name"]]
            if p is None or p.get("error"):
                ok = False
                res[s["name"]] = {"error": (p or {}).get("error", "no probe")}
                continue
            read = restore_read_Bps(p)
            good, detail = regime.check(c["regime"], c["_pilot"], c["load"], read, p["seq_write_Bps"])
            detail["read_path"] = {"seq_read_Bps": p.get("seq_read_Bps"), "rand_read_Bps": p.get("rand_read_Bps"),
                                   "rand_bs": c.get("kv_chunk_bytes"), "used": read}
            detail["bounds"] = regime.bandwidth_bounds(c["_pilot"], c["load"])
            detail["probe"] = {k: v for k, v in p.items() if k != "raw"}
            res[s["name"]] = detail
            ok &= good
        dump(os.path.join(attdir, "regime.json"), res)
        return ok, res

    def run(self):
        c, rn = self.c, self.rn
        labels = [a["label"] for a in c["arms"]]
        arms = arm_by_label(c)
        if len(c["slots"]) > 1:
            arms[order.FILLER] = filler_arm(c)
        attempts = []
        while True:
            att = self.attempt
            attdir = os.path.join(self.repdir, f"a{att}")
            rn.set_log(os.path.join(self.repdir, "commands.log"))
            log(f"block {c['cell']} host {self.host} repeat {self.repeat} attempt {att}")
            if self.fields is None:
                self.fields = health.gpu_fields(rn)
                for s in c["slots"]:
                    self.devs[s["name"]] = cgroup.block_device(rn, s["disk"]["mount"], s["disk"].get("device")) \
                        if s["disk"]["wipe"] != "ram" else None
            env = host_env(rn, c, self.host)
            dump(os.path.join(attdir, "host_env.json"), env)
            first, before = {}, {}
            for s in c["slots"]:
                p = self.probe_raw(s, "full")
                before[s["name"]] = p
                fp = self.first_probe_path(s)
                if p is not None and not p.get("error") and not os.path.exists(fp):
                    dump(fp, dict(p, host=self.host, saved_at=time.time()))
                first[s["name"]] = load_json(fp) if os.path.exists(fp) else None
            dump(os.path.join(attdir, "probe_before.json"), before)
            ok, reg = self.regime_check(attdir, before)
            if not ok:
                log(f"regime check failed: {json.dumps({k: (v.get('regime'), v.get('r'), v.get('w')) for k, v in reg.items()})}")
                return EXIT_CONFIG
            rounds = order.plan_rounds(labels, c["seed"], c["cell"], self.repeat, att, len(c["slots"]))
            plan = {"cell": c["cell"], "plan": c.get("plan"), "host": self.host, "repeat": self.repeat, "attempt": att,
                    "seed": c["seed"], "rounds": [[(c["slots"][s]["name"], a) for s, a in r] for r in rounds],
                    "regime": reg, "cpu_tier_bytes": check_cpu_tiers(c),
                    "disk_bytes": parse_size(c["capacities"]["disk_bytes"]),
                    "sglang_patch_variant": c.get("sglang_patch_variant", "fdatasync") if c["engine"] == "sglang" else None}
            dump(os.path.join(attdir, "block.json"), plan)
            verdicts, failures, pos = {}, [], 0
            self.fillers = []
            for rnd in rounds:
                stop = self.run_round(rnd, attdir, pos, arms, first, verdicts, failures)
                pos += len(rnd)
                if stop:
                    break
            after = {s["name"]: self.probe_raw(s, "full") for s in c["slots"]}
            dump(os.path.join(attdir, "probe_after.json"), after)
            outcome = order.block_outcome(labels, verdicts, failures)
            step = order.next_step(outcome, att, c["max_block_attempts"])
            plan.update(outcome=outcome, verdicts=verdicts, failures=failures, next=step, fillers=self.fillers)
            dump(os.path.join(attdir, "block.json"), plan)
            attempts.append({"attempt": att, "outcome": outcome, "verdicts": verdicts})
            log(f"attempt {att}: {step} {outcome['reasons'][:3]}")
            if step == "done":
                return EXIT_OK
            if step == "rerun":
                self.attempt += 1
                continue
            return EXIT_HOST_FAULT if step == "host_fault" else EXIT_ESCALATE

    def run_round(self, rnd, attdir, pos, arms, first, verdicts, failures):
        """Run one round (one arm per slot); returns True if the attempt must stop (an exclusion)."""
        c, rn = self.c, self.rn
        slots = [(c["slots"][si], arms[lab]) for si, lab in rnd]
        drop_caches(rn)
        ctx = []
        for k, (s, arm) in enumerate(slots):
            armdir = os.path.join(attdir, f"{pos + k:02d}.{arm['label']}")
            os.makedirs(armdir, exist_ok=True)
            prefix = f"{c['cell']}.{arm['label']}.{self.host}"
            quick = self.probe_raw(s, "quick")
            quick_tries = [_strip(quick)] if quick is not None else []
            # NRD reads swing about 7% between arms and dip up to 20% right after a heavy write (pilot 2: 1,638 to
            # 2,111 MB/s over 13 probes; network_ssd 449 to 461): a read off the first probe is probed again after a
            # pause, up to fio_retries times, before the health check judges the last one (PILOT2_PLAN: "repeat
            # the probe once")
            first_bps = (first.get(s["name"]) or {}).get("seq_read_Bps")
            tol = c["health"].get("fio_tolerance", 0.10)
            for _ in range(int(c["health"].get("fio_retries", 2))):
                got = (quick or {}).get("seq_read_Bps")
                if quick is None or not first_bps or (got is not None and abs(got - first_bps) / first_bps <= tol):
                    break
                log(f"{s['name']}: quick read {got} B/s vs first {first_bps}: probing again in "
                    f"{c['health'].get('fio_retry_wait_s', 90)} s")
                if not rn.dry:
                    time.sleep(float(c["health"].get("fio_retry_wait_s", 90)))
                quick = self.probe_raw(s, "quick")
                quick_tries.append(_strip(quick))
            notes = wipe(rn, c, s, self.devs.get(s["name"]))
            ctx.append({"slot": s, "arm": arm, "dir": armdir, "prefix": prefix, "quick": quick, "notes": notes,
                        "quick_tries": quick_tries})
        drop_caches(rn)
        round_budget = 0
        for x in ctx:
            s, arm = x["slot"], x["arm"]
            b = budget_of(c, arm)
            round_budget += b
            io = io_limits(c, self.devs.get(s["name"]))
            x["cg"] = cgroup.create(rn, cg_name(c, s, arm["label"], self.repeat, self.attempt), b, s["cpus"], s["mems"], io)
            x["budget"], x["io"] = b, io
            x["cg_bad"] = cgroup.verify(rn, x["cg"], b, s["cpus"], io)
            x["throttled"] = None
            if io:   # io.max holds: a short read inside the arm's own cgroup, then the tier is trimmed again
                x["throttled"] = probe.run(rn, s["disk"]["tier_dir"], "quick",
                                           dict(c["fio"], size=c["fio"].get("throttled_size", "1G")), direct=True,
                                           wrap=[HOST_CG_EXEC, x["cg"]],
                                           fake=self.dry.get("throttled_probe"))
                if s["disk"]["wipe"] == "fstrim":
                    rn.run(["fstrim", s["disk"]["mount"]], kind="root")
        wl = self.workload_counts()
        mem = health.mem_available(rn)
        stop = False
        for x in ctx:
            s, arm = x["slot"], x["arm"]
            g = health.gpu_now(rn, self.fields, str(s["cuda"]), fake_mask=self.dry.get("gpu_mask", "0x0"))
            procs = health.gpu_procs(rn, str(s["cuda"]))
            patch = sglang_patch(rn, c, "check") if c["engine"] == "sglang" else None
            vpatch = vllm_patch(rn, c, "check") if c["engine"] == "vllm" else None
            bad = health.pre_arm(gpu_rows=g, gpu_procs_list=procs, probe_now=x["quick"], probe_first=first[s["name"]],
                                 mem_avail=mem, budget=round_budget, workload=wl, expect=c["workload"],
                                 sglang_patch=patch, vllm_patch=vpatch, settings=c["health"])
            for msg in x["cg_bad"]:
                bad.append({"check": "cgroup", "detail": msg, "host_level": True})
            t = x["throttled"]
            if t is not None:
                want = x["io"][0][1]
                got = t.get("seq_read_Bps")
                if want and (got is None or abs(got - want) / want > c["health"].get("fio_tolerance", 0.10)):
                    bad.append({"check": "io_max", "host_level": True,
                                "detail": f"throttled read {got} B/s vs io.max rbps {want}"})
            x["health"] = {"gpu": g, "gpu_procs": procs, "quick_probe": _strip(x["quick"]), "quick_tries": x["quick_tries"],
                           "throttled": _strip(t),
                           "mem_available": mem, "round_budget": round_budget, "workload": wl, "sglang_patch": patch,
                           "vllm_patch": vpatch,
                           "wipe_notes": x["notes"], "failures": bad}
            dump(os.path.join(x["dir"], f"{x['prefix']}.health.json"), x["health"])
            for b in bad:
                failures.append(dict(b, label=arm["label"]))
            stop |= bool(bad)
        if stop:
            for x in ctx:
                cgroup.destroy(rn, x["cg"])
                self.index(x, None, "health_failed")
            return True
        # launch
        for x in ctx:
            s, arm = x["slot"], x["arm"]
            pre = os.path.join(x["dir"], x["prefix"])
            x["cg_before"] = cgroup.snapshot(rn, x["cg"])
            dev = self.devs.get(s["name"])
            tcmd = [sys.executable, os.path.join(HERE, "telemetry.py"), "--out", pre + ".telemetry.jsonl",
                    "--interval", str(c["telemetry"].get("interval", 1.0)), "--cgroup", f"{arm['label']}={x['cg']}",
                    "--tier", f"{arm['label']}={s['disk']['tier_dir']}",
                    "--cachestat-every", str(c["telemetry"].get("cachestat_every", 10))]
            if dev and dev.get("name"):
                tcmd += ["--dev", dev["name"]]
            hk = c["telemetry"].get("housekeeping_cpus")
            if hk:
                tcmd = ["taskset", "-c", hk] + tcmd
            x["procs"] = [rn.popen(tcmd, pre + ".telemetry.log"),
                          rn.popen(health.gpu_query_cmd(self.fields, str(s["cuda"]), 1000, pre + ".gpu.csv"),
                                   pre + ".gpu.log", kind="gpu")]
            if dev and dev.get("name") and (rn.dry or shutil.which("iostat")):
                x["procs"].append(rn.popen(["iostat", "-dxm", "-t", "-y", "5", dev["name"]], pre + ".iostat.log"))
            dump(pre + ".harness_env.json", {
                "cell": c["cell"], "plan": c.get("plan"), "label": arm["label"], "code": arm["code"],
                "arm_env": arm["env"], "host": self.host, "repeat": self.repeat, "attempt": self.attempt,
                "position": int(os.path.basename(x["dir"]).split(".")[0]), "slot": s["name"], "seed": c["seed"],
                "engine": c["engine"], "model": c["model"], "regime": c["regime"], "load": c["load"],
                "cgroup": x["cg"], "memory_max": x["budget"], "rss": arm_rss(c, arm)[1], "cpus": s["cpus"], "mems": s["mems"],
                "io_max": [cgroup.io_max_line(*i) for i in x["io"]], "disk": s["disk"], "device": dev,
                "capacities": c["capacities"], "workload": c["workload"]})
            cmd = arm_command(c, s, arm, x["dir"], x["cg"])
            x["t_launch"] = time.time()
            x["runner"] = rn.popen(cmd, os.path.join(x["dir"], "runner.log"), kind="gpu")
        timeout = float(c["runner"].get("timeout_s", 4 * 3600))
        for x in ctx:
            x["rc"] = self.wait(x["runner"], x["t_launch"] + timeout)
        for x in ctx:
            for p in x["procs"]:
                rn.stop(p)
            x["cg_after"] = cgroup.snapshot(rn, x["cg"])
            cgroup.destroy(rn, x["cg"])
            self.collect(x)
        for x in ctx:
            x["neighbour"] = self.overlap(x, ctx)
        for x in ctx:
            v = self.judge(x)
            if x["arm"].get("filler"):   # unscored: its replay output is discarded, its verdict kept for the record
                self.discard(x)
                self.fillers.append({"slot": x["slot"]["name"], "dir": x["dir"], "valid": v["valid"],
                                     "reasons": v["reasons"], "neighbour": x["neighbour"]})
                self.index(x, v, "filler")
                continue
            verdicts[x["arm"]["label"]] = v
            stop |= not v["valid"]
            self.index(x, v, "ran")
        return stop

    def overlap(self, x, ctx):
        """Seconds this arm's replay window [replay_start, replay_end] overlapped its neighbours' (two-engine
        rounds), and the fraction of its own window; None on a one-engine node or without markers (dry run)."""
        others = [y for y in ctx if y is not x]
        if not others:
            return None
        def win(y):
            m = health.read_markers(os.path.join(y["dir"], y["prefix"] + ".markers"))
            return (m["replay_start"], m["replay_end"]) if "replay_start" in m and "replay_end" in m else None
        mine = win(x)
        out = {"labels": [y["arm"]["label"] for y in others], "filler": [bool(y["arm"].get("filler")) for y in others],
               "overlap_s": None, "overlap_frac": None}
        theirs = [w for w in map(win, others) if w]
        if mine and theirs:
            ov = sum(max(0.0, min(mine[1], w[1]) - max(mine[0], w[0])) for w in theirs)
            out.update(overlap_s=ov, overlap_frac=ov / (mine[1] - mine[0]) if mine[1] > mine[0] else None)
        return out

    def discard(self, x):
        """A filler's replay rows are not data: delete them so no analysis can pick them up (logs stay)."""
        p = os.path.join(x["dir"], x["prefix"] + ".jsonl")
        if not self.rn.dry and os.path.exists(p):
            os.remove(p)

    def wait(self, proc, deadline):
        import subprocess
        try:
            return proc.wait(timeout=max(1.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            log("runner timed out; stopping it")
            self.rn.stop(proc, grace=30)
            return 124

    def workload_counts(self):
        w = self.c["workload"]
        if self.rn.dry and not os.path.exists(w["path"]):
            return (w.get("sessions"), w.get("turns"), w.get("sha256") or "dry-run")
        return health.workload_counts(w["path"])

    def collect(self, x):
        """Rename the runner's CODE.* outputs to CELL.LABEL.HOST.*."""
        code, d = x["arm"]["code"], x["dir"]
        if self.rn.dry:
            return
        for f in os.listdir(d):
            if f.startswith(code + "."):
                os.replace(os.path.join(d, f), os.path.join(d, x["prefix"] + f[len(code):]))
        if os.path.exists(os.path.join(d, "runner.log")):
            os.replace(os.path.join(d, "runner.log"), os.path.join(d, x["prefix"] + ".runner.log"))

    def judge(self, x):
        c, pre = self.c, os.path.join(x["dir"], x["prefix"])
        if self.rn.dry:
            v = {"valid": True, "reasons": [], "covariates": {"dry_run": True}}
        else:
            def text(suffix, errors="strict"):
                return read_text(pre + suffix, errors) if os.path.exists(pre + suffix) else ""
            tel = [json.loads(ln) for ln in text(".telemetry.jsonl").splitlines() if ln.strip()]
            gpu = health.parse_gpu_csv(text(".gpu.csv"), self.fields)
            rl = text(".runner.log", "replace")
            sl = text(".server.log", "replace")
            rc = x["rc"]
            arm_rc = [ln for ln in rl.splitlines() if ln.startswith(f"arm {x['arm']['code']} rc=")]
            if rc == 0 and arm_rc and not arm_rc[-1].split("rc=")[1].startswith("0 "):
                rc = int(arm_rc[-1].split("rc=")[1].split()[0])
            tb = text(".tier_bytes").strip()
            v = health.verdict(markers=health.read_markers(pre + ".markers"), cg_before=x["cg_before"],
                               cg_after=x["cg_after"], tel_rows=tel, cg_name=x["arm"]["label"], gpu_rows=gpu,
                               runner_rc=rc, runner_log=rl, server_log=sl, lmcache_log=text(".lmcache.log", "replace"),
                               regime=c["regime"], engine=c["engine"],
                               settings=c["health"], turn_rows=health.read_jsonl(pre + ".jsonl"),
                               disk_bytes=parse_size(c["capacities"]["disk_bytes"]),
                               tier_bytes_end=int(tb) if tb.isdigit() else None,
                               ram_tier=x["slot"]["disk"]["wipe"] == "ram")
        v["rc"] = x.get("rc")
        v.setdefault("covariates", {})["neighbour"] = x.get("neighbour")
        dump(pre + ".cgroup.json", {"before": x["cg_before"], "after": x["cg_after"]})
        dump(pre + ".verdict.json", v)
        return v

    def index(self, x, v, status):
        append_jsonl(os.path.join(self.out, "runs.jsonl"), {
            "cell": self.c["cell"], "label": x["arm"]["label"], "code": x["arm"]["code"], "host": self.host,
            "repeat": self.repeat, "attempt": self.attempt, "slot": x["slot"]["name"], "dir": x["dir"],
            "prefix": x["prefix"], "status": status, "valid": None if v is None else v["valid"],
            "reasons": None if v is None else v["reasons"], "t": time.time(), "dry_run": self.rn.dry,
            "neighbour": x.get("neighbour")})


def hicache_pilot(rn, c, host, out):
    """HC-wt's L3 write throughput per patch variant and stock SGLang, inside a cgroup with the cell's io.max on each
    slot's disk tier (or the RAM tier). Records every variant; the pilot decides sglang_patch_variant."""
    hp = dict({"bytes": "4G", "threads": 1}, **(c.get("hicache_pilot") or {}))
    res = {"cell": c["cell"], "host": host, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "io_max": c.get("io_max"), "page_bytes": int(c["kv_chunk_bytes"]), "bytes": parse_size(hp["bytes"]),
           "threads": int(hp["threads"]), "slots": {}}
    tarm = next((a for a in c["arms"] if a["code"].split(".")[0] == "T"), None)
    mem = budget_of(c, tarm) if tarm else 4 * GiB          # HC-wt's own memory.max: its page-cache room
    res["memory_max"] = mem
    py = os.path.join(c["runner"]["venv"], "bin", "python")
    for s in c["slots"]:
        d = s["disk"]
        dev = cgroup.block_device(rn, d["mount"], d.get("device")) if d["wipe"] != "ram" else None
        wipe(rn, c, s, dev)
        cg = cgroup.create(rn, f"{c['cell']}.hicache_pilot.{s['name']}", mem, s["cpus"], s["mems"], io_limits(c, dev))
        fake = "\n".join(json.dumps({"variant": v, "set_Bps": 1.0e8, "durable_Bps": 1.0e8, "dry_run": True})
                         for v in ("stock", "fdatasync", "sync_file_range")) + "\n"
        cmd = in_runner(c, [layby_path(c, "engine/phase_r/cg_exec.sh"), cg, py,
                            layby_path(c, "engine/phase_r/sglang/hicache_write_pilot.py"), "--dir", d["tier_dir"],
                            "--bytes", str(res["bytes"]), "--page-bytes", str(res["page_bytes"]),
                            "--threads", str(res["threads"])])
        r = rn.run(cmd, kind="root", fake=fake, timeout=7200)
        cgroup.destroy(rn, cg)
        rows = [json.loads(ln) for ln in r.out.splitlines() if ln.strip().startswith("{")]
        res["slots"][s["name"]] = {"io_max": [cgroup.io_max_line(*i) for i in io_limits(c, dev)], "rc": r.rc,
                                   "error": None if r.ok else r.err[-1000:], "variants": {x["variant"]: x for x in rows}}
    path = os.path.join(out, c["cell"], host, "hicache_pilot.json")
    dump(path, res)
    for name, sl in res["slots"].items():
        for v, x in sl["variants"].items():
            print(f"{name} {v}: set {x['set_Bps'] / 1e6:.0f} MB/s, durable {x['durable_Bps'] / 1e6:.0f} MB/s")
    print("written", path)
    return EXIT_OK if all(sl["rc"] == 0 and len(sl["variants"]) == 3 for sl in res["slots"].values()) else EXIT_PREFLIGHT


def _strip(p):
    return None if p is None else {k: v for k, v in p.items() if k != "raw"}


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("preflight", "plan", "run", "hicache-pilot"):
        p = sub.add_parser(name)
        p.add_argument("--cell", required=True)
        p.add_argument("--aa", help="A/A mode: run this arm twice as LABEL_aa1 and LABEL_aa2 (cell id CELL-AA)")
        p.add_argument("--dry-run", action="store_true")
        p.add_argument("--host", default="dryhost")
        p.add_argument("--repeat", type=int, default=1)
        p.add_argument("--attempt", type=int, default=1)
        p.add_argument("--out", default=os.path.expanduser("~/.cache/layby-phase-r"))
    p = sub.add_parser("regime")
    p.add_argument("--pilot", required=True)
    p.add_argument("--load", required=True)
    p.add_argument("--read-bps", type=float, required=True)
    p.add_argument("--write-bps", type=float, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "regime":
        pil = load_json(a.pilot)
        print(json.dumps({**regime.cell_regime(pil, a.load, a.read_bps, a.write_bps),
                          "bounds": regime.bandwidth_bounds(pil, a.load)}, indent=1))
        return EXIT_OK
    try:
        c = load_cell(a.cell, a.aa)
    except (ValueError, KeyError, FileNotFoundError) as e:
        log(f"cell spec: {e}")
        return EXIT_CONFIG
    if a.cmd == "plan":
        rounds = order.plan_rounds([x["label"] for x in c["arms"]], c["seed"], c["cell"], a.repeat, a.attempt,
                                   len(c["slots"]))
        for i, r in enumerate(rounds):
            print(f"round {i}: " + "  ".join(f"{c['slots'][s]['name']}={lab}" for s, lab in r))
        return EXIT_OK
    rn = Runner(dry_run=a.dry_run)
    out = os.path.abspath(a.out)
    live = live_problems(rn, c)
    if live:
        for p in live:
            print("CONFIG FAIL:", p)
        return EXIT_CONFIG
    if a.cmd == "hicache-pilot":
        if c["engine"] != "sglang":
            log("hicache-pilot is for SGLang cells")
            return EXIT_CONFIG
        rn.set_log(os.path.join(out, c["cell"], a.host, "hicache_pilot.commands.log"))
        return hicache_pilot(rn, c, a.host, out)
    if a.cmd == "preflight":
        probs = preflight(rn, c, out)
        for p in probs:
            print("PREFLIGHT FAIL:", p)
        print("PREFLIGHT OK" if not probs else f"PREFLIGHT: {len(probs)} problem(s)")
        return EXIT_OK if not probs else EXIT_PREFLIGHT
    rn.set_log(os.path.join(out, c["cell"], a.host, f"r{a.repeat}", "commands.log"))
    probs = preflight(rn, c, out)
    if probs:
        for p in probs:
            print("PREFLIGHT FAIL:", p)
        return EXIT_PREFLIGHT
    return Block(rn, c, a.host, a.repeat, out, a.attempt).run()


if __name__ == "__main__":
    sys.exit(main())
