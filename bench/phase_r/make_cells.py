#!/usr/bin/env python3
"""Phase R cell specs (engine/phase_r/orchestrate.py) for Qwen3.8-27B and Gemma 4 31B on vLLM 0.31.0, from the
workloads' capacities.json and manifests and the pilot's measurements (BENCH_PLAN draft 6).

Disks, one Nebius 1x RTX PRO 6000 VM (uk-south2), three settings:
  slow    the 2 TB network_ssd root disk (/data) throttled with cgroup io.max, read and write SLOW_IO_MAX bytes/s, set so
          the measured r is about 0.3 on both models (r = io.max / tier bytes per token / prefill tok/s at 8,192
          tokens, batch 8: Qwen 0.29, Gemma 0.26; at the 32K prefill rate 0.34 and 0.44, still below 0.5)
  medium  the same network_ssd without io.max (pilot: r 0.59 Qwen, 0.52 Gemma; the regime check at block start
          decides: a measured r below 0.5 stops the block with exit 4)
  fast    the 2.5 TB network_ssd_non_replicated disk (/data2), no io.max (pilot: r 2.74 Qwen, 2.38 Gemma)
Matrix per model (BENCH_PLAN draft 6.6): slow pressure (sweep_1.5x: 1.5x P, H1 and H4 primary, owner 2026-10-13),
slow heavy (measured, H1 secondary), slow light (light), fast heavy (measured); the unthrottled network_ssd (medium)
was in pilot 2 only, for its r and w. Arms C0 T T2 LM CT K K_ewma in the slow cells,
K_km added in Qwen slow pressure; C0 T T2 LM K K_ewma in fast (CT runs in the slow cells only, plan 5.2).

Start-block cells (s-*, written to cells/start one arm per block): C0 dry runs on the slow disk for the measured,
pressure and light workloads (each sets its cells' hard cap), and CT and LM on the pilot workload (GPU checks of the
fixes since pilot 2). The pilot-2 cells (p2-*) in cells/ and cells/p2 are the pilot's record and are not rewritten.

Per cell: capacities from capacities.json (GPU and CPU per model, disk per workload; the kdisk cells from their
manifest), MAX_NUM_SEQS on Qwen at most its GPU KV blocks (vLLM's Mamba CUDA graph rule; 64 otherwise), MAXLEN the workloads' context cap (capacities.json max_model_len), CT_ETA of the model's measured workload, LMCache chunk 784 on Qwen, kv_chunk_bytes the vLLM CPU tier's
chunk (Qwen 51,380,224 B, Gemma 2,621,440 B), hard_cap_s 1.5 x the workload's projected C0 makespan (manifest
c0_projection, fit_service.py; replaced by the C0 dry run's makespan once it has run), REPLAY_TIMEOUT the cap + 600 s,
the runner timeout 1,800 s more. Pilot JSON per model: tier bytes per token and prefill tok/s measured in pilot 1
(micro, batch 8 at 8,192 tokens; Gemma at its chunk of 1,024 tokens, pilot 2 micro_c1024: derive_chunk.py), arm RSS from pilot 1's telemetry (peak memory beyond the CPU tier and the page
cache, rounded up with 1.5 GiB or more of headroom), and write-through's offered write rate per load: T's bytes
written per unique token in pilot 1 (Qwen 1.403, Gemma 1.455 x tier bytes) x the workload's unique tokens / its
projected C0 makespan (pilot 2's T arms on the fast disk measure it).

The start block runs each arm as its own one-attempt block (cells/start/CELL.NN.ARM.json), as the pilots did.

Tuned settings (BENCH_PLAN section 8; bench/phase_r/make_tuning.py writes them to cells/tune/tuned.json, read with
--tuned, by default that file when it exists): C0's eviction and chunked-prefill budget carry to every cell of the
model (runner env EVICTION, and SERVE_ARGS with --max-num-batched-tokens on Qwen; Gemma's 1,024 is fixed in its serve
args); every arm the runner gives vLLM's own CPU policy (C0 T T2 CT) takes the eviction, every arm the budget. K's
ParkCachePolicy orders its unhinted chunks LRU whatever EVICTION says (layby/vllm/policy.py), and LMCache has its own
L1 eviction. T, T2 and LM take their tuned knobs per disk (the cell's regime): T BLOCKS_PER_CHUNK and FS_THREADS, T2
its threshold, count mode and decay on T's disk knobs, LM LM_CHUNK and its prefetch policy (code LMR for retain,
label LM). An arm with no tuned entry keeps the defaults above (T2_ENV, LM_CHUNK 784 on Qwen and the runner's own).

Usage: make_cells.py WORKLOAD_DIR [--out engine/phase_r/cells] [--tuned cells/tune/tuned.json]
"""
import argparse
import hashlib
import json
import math
import os

GiB = 1 << 30
SEED = 20261010
PLAN = "BENCH_PLAN draft 6.5 (2026-10-13; freeze commit to fill)"
SLOW_IO_MAX = 210 * (1 << 20)          # 220,200,960 B/s
# pilot 1 in-block probes (pilot.json regime): min(seq 4 MiB, random chunk) read; 4 MiB write
NETWORK_SSD = dict(seq_read_Bps=449_519_063, seq_write_Bps=462_519_988, rand_read_Bps=500_137_467)
NRD = dict(seq_read_Bps=2_041_400_456, seq_write_Bps=1_412_483_080, rand_read_Bps=2_141_000_000)
WT_PER_TOKEN = {"qwen38_27b": 1.403, "gemma4_31b": 1.455}

HOST = {}     # --host profile (bench/phase_r/hosts/HOST.json); empty for the Nebius VMs the cells were made for
DRYRUN = {}   # workload -> its C0 dry run's makespan, s (c0_dryrun.json); the hard cap's basis once measured
MAXLEN = {}   # --max-model-len per model, capacities.json max_model_len (the workloads' context cap)
MODELS = {
    "qwen38_27b": dict(
        tag="qwen", hf="Qwen/Qwen3.8-27B-FP8", serve="--revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        lm_chunk=784, chunk_bytes=51_380_224, tier_bpt=72_472, prefill=10_477.5, kkm_cell="slow-pressure", mamba=True,
        rss=dict(vllm=7 * GiB, k=7 * GiB, sidecar=2 * GiB, lm_vllm=6 * GiB, lm_server=2 * GiB)),
    "gemma4_31b": dict(
        tag="gemma", hf="google/gemma-4-31B-it",
        serve='--revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --quantization fp8 '
              '--limit-mm-per-prompt {"image":0,"audio":0,"video":0} --max-num-batched-tokens 1024',
        lm_chunk=None, chunk_bytes=2_621_440, tier_bpt=131_714, prefill=6_201.4, kkm_cell=None, mamba=False,
        rss=dict(vllm=9 * GiB, k=9 * GiB, sidecar=2 * GiB, lm_vllm=8 * GiB, lm_server=2 * GiB)),
}
SLOW_ARMS = ["C0", "T", "T2", "LM", "CT", "K", "K_ewma"]
OTHER_ARMS = ["C0", "T", "T2", "LM", "K", "K_ewma"]
T2_ENV = {"T2_MIN_HITS": "2", "T2_DECAY_S": "600", "T2_COUNT": "request"}   # the default until T2 is tuned
TUNED_FAMILIES = ("T", "T2", "LM")   # tuned per disk; C0 once per model (slow disk), carried to every cell
# readable arm labels (owner 2026-11-08: one-letter names are unreadable); the runner codes stay inside run_arms.sh.
# Used for the --host cells; the Nebius cells keep the labels they were run with.
NAMES = {"C0": "no-disk", "T": "write-through", "T2": "write-hot", "LM": "lmcache", "LMR": "lmcache-retain",
         "CT": "continuum", "K": "layby", "K_km": "layby-avg", "K_ewma": "layby-ewma"}
CODE_FAMILY = {"T": "T", "T2": "T2", "LM": "LM", "LMR": "LM"}   # the tuned family of a runner code
# the --host campaign's matrix (BENCH_PLAN draft 6.8, owner 2026-11-08: a research question, not a qualification):
# slow pressure with every arm (3 repeats), fast heavy with the four arms that answer "does Layby hurt on a fast disk"
HOST_FAST_ARMS = ["C0", "T", "LM", "K"]


def c0_env(key, knobs):
    """Runner env of C0's tuned knobs {eviction, budget}: EVICTION, and SERVE_ARGS with --max-num-batched-tokens when
    the model's budget is tuned (Qwen); a budget beside one already in the serve args (Gemma's 1,024) is refused."""
    env = {"EVICTION": str(knobs["eviction"])}
    if knobs.get("budget") is not None:
        serve = MODELS[key]["serve"]
        if "--max-num-batched-tokens" in serve:
            raise ValueError(f"{key}: the serve args already fix --max-num-batched-tokens")
        env["SERVE_ARGS"] = f"{serve} --max-num-batched-tokens {int(knobs['budget'])}"
    return env


def family_arm(family, knobs):
    """(runner code, arm env) of one T, T2 or LM grid point (run_arms.sh docstring): T {blocks_per_chunk,
    fs_threads}; T2 {min_hits, count, decay_s} plus T's two; LM {chunk, prefetch: default or retain (code LMR), eager: the eager prefetch}."""
    if family in ("T", "T2"):
        env = {"BLOCKS_PER_CHUNK": str(int(knobs["blocks_per_chunk"])), "FS_THREADS": str(int(knobs["fs_threads"]))}
        if family == "T2":
            env.update(T2_MIN_HITS=str(int(knobs["min_hits"])), T2_COUNT=str(knobs["count"]),
                       T2_DECAY_S=str(int(knobs["decay_s"])))
        return family, env
    if family == "LM":
        if knobs["prefetch"] not in ("default", "retain"):
            raise ValueError(f"LM prefetch {knobs['prefetch']!r}")
        env = {"LM_CHUNK": str(int(knobs["chunk"]))}
        if knobs.get("eager"):
            env["LM_EAGER"] = "1"
        return ("LMR" if knobs["prefetch"] == "retain" else "LM"), env
    raise ValueError(f"no tuned knobs for family {family!r}")


def load_tuned(path):
    """The tuned settings JSON (make_tuning.py next) and its sha256, or (None, None) without a file."""
    if not path or not os.path.exists(path):
        return None, None
    raw = open(path, "rb").read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def apply_tuned(c, key, tuned, sha=None, path=None):
    """Carry the tuned settings into cell c (in place): C0's knobs into the runner env, each tuned family's knobs
    for the cell's disk (regime) into the arm labelled with that family. Records what it applied in c["tuned"]."""
    t = (tuned or {}).get(key) or {}
    applied = {}
    if t.get("C0"):
        c["runner"]["env"].update(c0_env(key, t["C0"]["knobs"]))
        applied["C0"] = t["C0"]["knobs"]
    for a in c["arms"]:
        fam = CODE_FAMILY.get(a["code"]) if HOST else a["label"]
        entry = (t.get(fam) or {}).get(c["regime"]) if fam in TUNED_FAMILIES else None
        if entry:
            code, env = family_arm(fam, entry["knobs"])
            a["code"] = code
            a["env"] = dict({k: v for k, v in (a.get("env") or {}).items() if not k.startswith("T2_")}, **env)
            applied[fam] = entry["knobs"]
    if applied:
        c["tuned"] = {"file": os.path.basename(path) if path else None, "sha256": sha, "applied": applied}
    return applied


def offered_write(key, man):
    """write-through's average write rate on this workload at its projected C0 makespan, bytes/s"""
    L, P = man["load"], man["c0_projection"]
    return round(WT_PER_TOKEN[key] * L["unique_kv_tokens"] * MODELS[key]["tier_bpt"] / P["makespan_s"])


def pilot_json(key, m, heavy, light, pressure):
    """The pilot JSON regime.py reads; with --host, prefill_tok_s is that host's micro and the offered write rates
    scale with it (a faster or slower prefill offers write-through's bytes at that rate)."""
    r = m["rss"]
    pf = HOST["prefill_tok_s"][key] if HOST else m["prefill"]
    if HOST:
        k = pf / m["prefill"]
        heavy, light, pressure = round(heavy * k), round(light * k), round(pressure * k)
    return {
        "_comment": (
            "Pilot 1 measurements (BENCH_PLAN draft 6, bench/phase_r/make_cells.py docstring). kv_bytes_per_token: "
            "the disk tier's measured bytes per token; prefill_tok_s: micro batch 8 at 8,192 tokens; "
            "wt_offered_write_Bps: T's measured bytes per unique token x the load's workload unique tokens / its "
            "projected C0 makespan; RSS: pilot 1 peak beyond the CPU tier and the page cache (C0, T, T2, CT, PT about "
            "5.2 to 6.9 GiB, K 7.2 to 7.4 GiB with the sidecar, LM 6.3 GiB beyond its L1) plus headroom; pilot 2 "
            "re-measures each."),
        "model": m["hf"], "engine": "vllm", "kv_bytes_per_token": m["tier_bpt"],
        "arm_rss_bytes": {**{x: r["vllm"] for x in ("C0", "T", "T2", "PT", "CT", "K_ewma", "K_km")}, "K": r["k"],
                          "LM": {"vllm": r["lm_vllm"], "lmcache_server": r["lm_server"]}},
        "sidecar_rss_bytes": r["sidecar"],
        "loads": {"heavy": {"prefill_tok_s": pf, "wt_offered_write_Bps": heavy},
                  "light": {"prefill_tok_s": pf, "wt_offered_write_Bps": light},
                  "pressure": {"prefill_tok_s": pf, "wt_offered_write_Bps": pressure}},
    }


def host_disk(regime, c, key):
    """The --host profile's slots and io.max (BENCH_PLAN draft 6.7): both regimes emulated so r matches Nebius,
    io.max = the Nebius disk's rate x this host's prefill / Nebius's prefill for the model (slow: SLOW_IO_MAX read and
    write; fast: the NRD's measured read and write)."""
    if regime not in ("slow", "fast"):
        raise ValueError(f"--host {HOST['host']}: regime {regime!r} has no emulation")
    ratio = HOST["prefill_tok_s"][key] / MODELS[key]["prefill"]
    if regime == "slow":
        rb = wb = round(SLOW_IO_MAX * ratio)
        src = "210 MiB/s"
    else:
        rb, wb = round(NRD["seq_read_Bps"] * ratio), round(NRD["seq_write_Bps"] * ratio)
        src = "the NRD's measured read and write"
    c["slots"] = [json.loads(json.dumps(s)) for s in HOST["slots"]]
    c["io_max"] = {"rbps": str(rb), "wbps": str(wb)}
    c["io_max_note"] = (f"{regime} regime emulated on {HOST['host']} (BENCH_PLAN draft 6.7): {src} x this host's "
                        f"prefill {HOST['prefill_tok_s'][key]:g} / Nebius {MODELS[key]['prefill']:g} tok/s = "
                        f"x{ratio:.4f}, so r is Nebius's")
    c["dry_run"] = {"probe": dict(seq_read_Bps=12e9, seq_write_Bps=11e9, rand_read_Bps=10e9),
                    "throttled_probe": dict(seq_read_Bps=rb, seq_write_Bps=wb, rand_read_Bps=rb)}
    c["telemetry"]["housekeeping_cpus"] = HOST["housekeeping_cpus"]
    c["host_profile"] = {"host": HOST["host"], "gpus": HOST["gpus"], "prefill_source": HOST["prefill_source"]}


def disk(regime, c, key=None):
    if HOST:
        return host_disk(regime, c, key)
    slot = {"cuda": "0", "port": 8000, "park_port": 28765, "cpus": "2-23", "mems": "0"}
    if regime in ("slow", "medium"):
        # one slot name for the network_ssd: the host's first unthrottled probe holds both settings to it
        c["slots"] = [dict(slot, name="netssd", disk={"mount": "/data", "wipe": "fstrim"})]
        c["dry_run"] = {"probe": dict(NETWORK_SSD)}
        if regime == "slow":
            c["io_max"] = {"rbps": str(SLOW_IO_MAX), "wbps": str(SLOW_IO_MAX)}
            c["io_max_note"] = ("network_ssd throttled to 210 MiB/s read and write (BENCH_PLAN draft 6): r about 0.3 "
                                "on both models at the pilot's prefill rates")
            c["dry_run"]["throttled_probe"] = dict(seq_read_Bps=SLOW_IO_MAX, seq_write_Bps=SLOW_IO_MAX,
                                                   rand_read_Bps=SLOW_IO_MAX)
        else:
            c["io_max_none_note"] = "network_ssd without io.max: medium if its measured r is 0.5 or more"
    else:
        c["slots"] = [dict(slot, name="nrd", disk={"mount": "/data2", "wipe": "fstrim"})]
        c["dry_run"] = {"probe": dict(NRD)}


def cell(name, key, man, caps, regime, load, wl, arms, ct_eta, cap_src, extra=None, tuned=None):
    """One cell spec (engine/phase_r/examples/cell_main_vllm_slow_heavy.json documents the format). tuned: (settings,
    sha256, path) from load_tuned, carried into the cell by apply_tuned."""
    m = MODELS[key]
    proj = man.get("c0_projection") or {}
    projected = proj.get("makespan_s") or man["load"]["est_makespan_s"]
    dry = None if name.startswith("s-") else DRYRUN.get(wl)   # the start block's cells are the dry runs themselves
    mk = dry or projected
    cap = int(math.ceil(1.5 * mk))
    env = {"REPLAY_TIMEOUT": str(cap + 600), "BOOT_WAIT": "360", "MAXLEN": str(MAXLEN[key]), "SERVE_ARGS": m["serve"]}
    if m["lm_chunk"]:
        env["LM_CHUNK"] = str(m["lm_chunk"])
    if m["mamba"]:
        # vLLM 0.31.0 needs one Mamba block per decode sequence for CUDA graph capture: max_num_seqs at most the GPU
        # KV blocks, rounded down to the graph batch step of 8 (Qwen at 2.99 GB: 58 blocks, 56)
        blocks = int(caps["gpu_kv_bytes"]) // m["chunk_bytes"]
        env["MAX_NUM_SEQS"] = str(min(64, blocks // 8 * 8))
    A = []
    for x in arms:
        if isinstance(x, dict):   # a built arm (make_tuning.py: grid points with their own labels and env)
            A.append(x)
            continue
        a = {"label": NAMES.get(x, x) if HOST else x, "code": x}
        if x == "T2":
            a["env"] = dict(T2_ENV)
        if x == "CT":
            a["env"] = {"CT_ETA": str(ct_eta)}
        A.append(a)
    c = {
        "_comment": (
            f"Generated by bench/phase_r/make_cells.py (its docstring has every rule). Workload {wl}: sha256, sessions "
            f"and turns from its manifest; capacities from {cap_src}; hard_cap_s = 1.5 x "
            + (f"the C0 dry run's makespan {mk:g} s (c0_dryrun.json)." if dry else
               f"the projected C0 makespan {mk:g} s (fit_service.py), to be replaced by 1.5 x the C0 dry run's makespan.")),
        "cell": name, "plan": PLAN, "seed": SEED, "engine": "vllm", "model": m["hf"], "regime": regime, "load": load,
        "pilot": f"pilot_{key}_vllm.json",
        "workload": {"path": f"/root/phase_r/{wl}.json", "sessions": man["load"]["sessions"],
                     "turns": man["load"]["turns"], "sha256": man["sha256"]},
        "capacities": caps, "hard_cap_s": cap, "projected_c0_makespan_s": projected, "cgroup": {"slack_bytes": "256M"},
        "kv_chunk_bytes": m["chunk_bytes"],
        "live_scoring": {"side_file": f"/root/phase_r/{wl}.side.jsonl", "ckpt": "/root/dwell/model.safetensors",
                         "calib": "/root/dwell/config.json"},
        "runner": {"mode": "docker", "container": "park", "layby_root": "/root/layby", "venv": "/root/venv",
                   "env": env, "timeout_s": cap + 600 + 1800},
        "telemetry": {"interval": 1.0, "cachestat_every": 10, "housekeeping_cpus": "0-1"},
        "health": {"fio_tolerance": 0.10, "free_mem_margin_bytes": 8 * GiB, "psi_full_max_frac": 0.01,
                   "psi_resources": ["memory"]},
        "fio": {"ioengine": "libaio", "size": "4G", "runtime_full": 30, "runtime_quick": 8},
        "max_block_attempts": 2, "arms": A,
    }
    if dry:
        c["c0_dryrun_makespan_s"] = dry
    disk(regime, c, key)
    if extra:
        c.update(extra)
    if tuned and tuned[0]:
        apply_tuned(c, key, *tuned)
    if HOST:   # the pilot's RSS table is keyed by runner code; readable labels point at it
        for a in c["arms"]:
            a.setdefault("rss_of", "LM" if a["code"] in ("LM", "LMR") else a["code"])
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workloads")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine",
                                                  "phase_r", "cells"))
    ap.add_argument("--tuned", help="tuned settings JSON (default OUT/tune/tuned.json when it exists)")
    ap.add_argument("--host", help="host profile bench/phase_r/hosts/HOST.json (default: the Nebius VMs); cells go to "
                                   "OUT/HOST")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    hosts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hosts")
    if a.host:
        HOST.update(json.load(open(os.path.join(hosts_dir, f"{a.host}.json"))))
        miss = [k for k, v in HOST["prefill_tok_s"].items() if not v]
        if miss:
            ap.error(f"--host {a.host}: prefill_tok_s not measured for {miss} (run the host's micro first)")
        for key in MODELS:
            MODELS[key]["serve"] = f"{MODELS[key]['serve']} {HOST['serve_extra']}".strip()
        global PLAN
        PLAN = f"BENCH_PLAN draft 6.7 (host {a.host}; freeze commit to fill)"
        tuned_default = os.path.join(out, "tune", "tuned.json")
        out = os.path.join(out, a.host)
    else:
        tuned_default = os.path.join(out, "tune", "tuned.json")
    os.makedirs(out, exist_ok=True)
    tuned_path = a.tuned or tuned_default
    tuned, tuned_sha = load_tuned(tuned_path)
    if a.tuned and tuned is None:
        ap.error(f"--tuned {a.tuned}: no such file")
    tune = (tuned, tuned_sha, tuned_path)
    C = json.load(open(os.path.join(a.workloads, "capacities.json")))
    dr = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f"c0_dryrun_{a.host}.json" if a.host else "c0_dryrun.json")
    if os.path.exists(dr):
        DRYRUN.update({k: float(v) for k, v in json.load(open(dr)).items() if not k.startswith("_")})
    MAXLEN.update({k: int(v["max_model_len"]) for k, v in C["models"].items()})
    man = lambda n: json.load(open(os.path.join(a.workloads, f"{n}.manifest.json")))   # noqa: E731
    written = []

    def put(name, obj):
        p = os.path.join(out, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
        written.append(p)
        if name.startswith("s-"):
            # the start block runs each arm as its own one-attempt block, as the pilots did, so one exclusion stops
            # nothing else
            for i, arm in enumerate(obj["arms"], 1):
                put(os.path.join("start", f"{obj['cell']}.{i:02d}.{arm['label']}.json"),
                    dict(obj, arms=[arm], max_block_attempts=1, pilot="../" + obj["pilot"]))

    for key, m in MODELS.items():
        mc = C["models"][key]
        eta = C["workloads"][f"{key}.measured"]["ct_eta"]
        put(f"pilot_{key}_vllm.json", pilot_json(key, m, offered_write(key, man(f"{key}.measured")),
                                                 offered_write(key, man(f"{key}.light")),
                                                 offered_write(key, man(f"{key}.sweep_1.5x"))))
        caps = lambda wl: {"gpu_kv_bytes": str(mc["gpu_kv_bytes"]), "cpu_bytes": str(mc["cpu_bytes"]),   # noqa: E731
                           "disk_bytes": str(C["workloads"][wl]["disk_bytes"])}
        src = "capacities.json (GPU and CPU per model, disk per workload)"
        matrix = ((("slow", "pressure", "sweep_1.5x"), ("fast", "heavy", "measured")) if HOST else
                  (("slow", "heavy", "measured"), ("slow", "pressure", "sweep_1.5x"),
                   ("slow", "light", "light"), ("fast", "heavy", "measured")))
        for regime, load, wlk in matrix:
            wl = f"{key}.{wlk}"
            other = HOST_FAST_ARMS if HOST else OTHER_ARMS
            arms = (SLOW_ARMS if regime == "slow" else other) + (
                ["K_km"] if m["kkm_cell"] == f"{regime}-{load}" else [])
            name = f"{m['tag']}-{regime}-{load}"
            put(f"{name}.json", cell(name, key, man(wl), caps(wl), regime, load, wl, arms, eta, src, tuned=tune))
        # start block (BENCH_PLAN draft 6.5): C0 dry runs on the slow disk for every workload a cell runs (each sets
        # its cells' hard cap, 1.5x its makespan), then CT and LM once on the pilot workload: the GPU checks of the CT
        # pin release (90d52d5, 9adb696) and of the whole-64 MiB CPU tier (6bea6ae), neither yet run on a GPU host
        dry_loads = (("heavy", "measured"), ("pressure", "sweep_1.5x")) if HOST else (
            ("heavy", "measured"), ("pressure", "sweep_1.5x"), ("light", "light"))
        for load, wlk in dry_loads:
            wl = f"{key}.{wlk}"
            put(f"s-{m['tag']}-c0dry-{load}.json", cell(f"s-{m['tag']}-c0dry-{load}", key, man(wl), caps(wl), "slow",
                                                        load, wl, ["C0"], eta, src, tuned=tune))
        if HOST:   # the CT and LM GPU checks passed on Nebius (start block, 2026-10-13); not repeated
            continue
        wl = f"{key}.pilot"
        put(f"s-{m['tag']}-check.json", cell(f"s-{m['tag']}-check", key, man(wl), caps(wl), "slow", "heavy", wl,
                                             ["CT", "LM"], eta, src, tuned=tune))
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
