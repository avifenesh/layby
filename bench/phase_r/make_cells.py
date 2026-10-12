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
Matrix per model (coordinator, 2026-10-12): slow heavy (measured), slow light (light), fast heavy (measured); the
unthrottled network_ssd (medium) is in pilot 2 only, for its r and w. Arms C0 T T2 LM CT K K_ewma in the slow cells,
K_km added in Qwen slow heavy; C0 T T2 LM K K_ewma in fast (CT runs in the slow cells only, plan 5.2).

Pilot 2 cells (p2-*): C0 dry runs on the measured workload on the slow disk (they set every cell's hard cap), T on the
unthrottled network_ssd (medium: r and w), K and T on the fast disk (T's write rate at the new load is w's numerator), the Gemma arms not yet run (T2 LM CT K K_ewma)
on the slow disk, the Gemma K disk-path test with stock vLLM's tier (T) as the reference (check_k_disk.py
--reference-rows), and a C0 run for the engine-stall check.

Per cell: capacities from capacities.json (GPU and CPU per model, disk per workload; the kdisk cells from their
manifest), MAXLEN the workloads' context cap (capacities.json max_model_len), CT_ETA of the model's measured workload, LMCache chunk 784 on Qwen, kv_chunk_bytes the vLLM CPU tier's
chunk (Qwen 51,380,224 B, Gemma 2,621,440 B), hard_cap_s 1.5 x the workload's projected C0 makespan (manifest
c0_projection, fit_service.py; replaced by the C0 dry run's makespan once it has run), REPLAY_TIMEOUT the cap + 600 s,
the runner timeout 1,800 s more. Pilot JSON per model: tier bytes per token and prefill tok/s measured in pilot 1
(micro, batch 8 at 8,192 tokens), arm RSS from pilot 1's telemetry (peak memory beyond the CPU tier and the page
cache, rounded up with 1.5 GiB or more of headroom), and write-through's offered write rate per load: T's bytes
written per unique token in pilot 1 (Qwen 1.403, Gemma 1.455 x tier bytes) x the workload's unique tokens / its
projected C0 makespan (pilot 2's T arms on the fast disk measure it).

Pilot 2 runs each arm as its own one-attempt block (cells/p2/CELL.NN.ARM.json, written beside each p2 cell), as pilot 1
did.

Usage: make_cells.py WORKLOAD_DIR [--out engine/phase_r/cells]
"""
import argparse
import json
import math
import os

GiB = 1 << 30
SEED = 20261010
PLAN = "BENCH_PLAN draft 6 (2026-10-12; freeze commit to fill)"
SLOW_IO_MAX = 210 * (1 << 20)          # 220,200,960 B/s
# pilot 1 in-block probes (pilot.json regime): min(seq 4 MiB, random chunk) read; 4 MiB write
NETWORK_SSD = dict(seq_read_Bps=449_519_063, seq_write_Bps=462_519_988, rand_read_Bps=500_137_467)
NRD = dict(seq_read_Bps=2_041_400_456, seq_write_Bps=1_412_483_080, rand_read_Bps=2_141_000_000)
WT_PER_TOKEN = {"qwen38_27b": 1.403, "gemma4_31b": 1.455}

MAXLEN = {}   # --max-model-len per model, capacities.json max_model_len (the workloads' context cap)
MODELS = {
    "qwen38_27b": dict(
        tag="qwen", hf="Qwen/Qwen3.8-27B-FP8", serve="--revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        lm_chunk=784, chunk_bytes=51_380_224, tier_bpt=72_472, prefill=10_477.5, kkm_cell="slow-heavy",
        rss=dict(vllm=7 * GiB, k=7 * GiB, sidecar=2 * GiB, lm_vllm=6 * GiB, lm_server=2 * GiB)),
    "gemma4_31b": dict(
        tag="gemma", hf="google/gemma-4-31B-it",
        serve='--revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --quantization fp8 '
              '--limit-mm-per-prompt {"image":0,"audio":0,"video":0}',
        lm_chunk=None, chunk_bytes=2_621_440, tier_bpt=131_714, prefill=6_526.2, kkm_cell=None,
        rss=dict(vllm=9 * GiB, k=9 * GiB, sidecar=2 * GiB, lm_vllm=8 * GiB, lm_server=2 * GiB)),
}
SLOW_ARMS = ["C0", "T", "T2", "LM", "CT", "K", "K_ewma"]
OTHER_ARMS = ["C0", "T", "T2", "LM", "K", "K_ewma"]
T2_ENV = {"T2_MIN_HITS": "2", "T2_DECAY_S": "600", "T2_COUNT": "request"}   # tuned on the tuning workload later


def offered_write(key, man):
    """write-through's average write rate on this workload at its projected C0 makespan, bytes/s"""
    L, P = man["load"], man["c0_projection"]
    return round(WT_PER_TOKEN[key] * L["unique_kv_tokens"] * MODELS[key]["tier_bpt"] / P["makespan_s"])


def pilot_json(key, m, heavy, light):
    r = m["rss"]
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
        "loads": {"heavy": {"prefill_tok_s": m["prefill"], "wt_offered_write_Bps": heavy},
                  "light": {"prefill_tok_s": m["prefill"], "wt_offered_write_Bps": light}},
    }


def disk(regime, c):
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


def cell(name, key, man, caps, regime, load, wl, arms, ct_eta, cap_src, extra=None):
    m = MODELS[key]
    proj = man.get("c0_projection") or {}
    mk = proj.get("makespan_s") or man["load"]["est_makespan_s"]
    cap = int(math.ceil(1.5 * mk))
    env = {"REPLAY_TIMEOUT": str(cap + 600), "BOOT_WAIT": "360", "MAXLEN": str(MAXLEN[key]), "SERVE_ARGS": m["serve"]}
    if m["lm_chunk"]:
        env["LM_CHUNK"] = str(m["lm_chunk"])
    A = []
    for x in arms:
        a = {"label": x, "code": x}
        if x == "T2":
            a["env"] = dict(T2_ENV)
        if x == "CT":
            a["env"] = {"CT_ETA": str(ct_eta)}
        A.append(a)
    c = {
        "_comment": (
            f"Generated by bench/phase_r/make_cells.py (its docstring has every rule). Workload {wl}: sha256, sessions "
            f"and turns from its manifest; capacities from {cap_src}; hard_cap_s = 1.5 x the projected C0 makespan "
            f"{mk:g} s (fit_service.py), to be replaced by 1.5 x the C0 dry run's makespan."),
        "cell": name, "plan": PLAN, "seed": SEED, "engine": "vllm", "model": m["hf"], "regime": regime, "load": load,
        "pilot": f"pilot_{key}_vllm.json",
        "workload": {"path": f"/root/phase_r/{wl}.json", "sessions": man["load"]["sessions"],
                     "turns": man["load"]["turns"], "sha256": man["sha256"]},
        "capacities": caps, "hard_cap_s": cap, "projected_c0_makespan_s": mk, "cgroup": {"slack_bytes": "256M"},
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
    disk(regime, c)
    if extra:
        c.update(extra)
    return c


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("workloads")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine",
                                                  "phase_r", "cells"))
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    C = json.load(open(os.path.join(a.workloads, "capacities.json")))
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
        if name.startswith("p2-"):
            # pilot 2 runs each arm as its own one-attempt block, as pilot 1 did, so one exclusion stops nothing else
            for i, arm in enumerate(obj["arms"], 1):
                put(os.path.join("p2", f"{obj['cell']}.{i:02d}.{arm['label']}.json"),
                    dict(obj, arms=[arm], max_block_attempts=1, pilot="../" + obj["pilot"]))

    for key, m in MODELS.items():
        mc = C["models"][key]
        eta = C["workloads"][f"{key}.measured"]["ct_eta"]
        put(f"pilot_{key}_vllm.json", pilot_json(key, m, offered_write(key, man(f"{key}.measured")),
                                                 offered_write(key, man(f"{key}.light"))))
        caps = lambda wl: {"gpu_kv_bytes": str(mc["gpu_kv_bytes"]), "cpu_bytes": str(mc["cpu_bytes"]),   # noqa: E731
                           "disk_bytes": str(C["workloads"][wl]["disk_bytes"])}
        src = "capacities.json (GPU and CPU per model, disk per workload)"
        for regime, load, wlk in (("slow", "heavy", "measured"), ("slow", "light", "light"),
                                  ("fast", "heavy", "measured")):
            wl = f"{key}.{wlk}"
            arms = (SLOW_ARMS if regime == "slow" else OTHER_ARMS) + (
                ["K_km"] if m["kkm_cell"] == f"{regime}-{load}" else [])
            name = f"{m['tag']}-{regime}-{load}"
            put(f"{name}.json", cell(name, key, man(wl), caps(wl), regime, load, wl, arms, eta, src))
        # pilot 2
        wl = f"{key}.measured"
        put(f"p2-{m['tag']}-c0dry.json", cell(f"p2-{m['tag']}-c0dry", key, man(wl), caps(wl), "slow", "heavy", wl,
                                              ["C0"], eta, src))
        wl = f"{key}.pilot"
        put(f"p2-{m['tag']}-fast.json", cell(f"p2-{m['tag']}-fast", key, man(wl), caps(wl), "fast", "heavy", wl,
                                             ["T", "K"], eta, src))
        put(f"p2-{m['tag']}-medium.json", cell(f"p2-{m['tag']}-medium", key, man(wl), caps(wl), "medium", "heavy", wl,
                                               ["T"], eta, src))
        if key == "gemma4_31b":
            put(f"p2-{m['tag']}-slow.json", cell(f"p2-{m['tag']}-slow", key, man(wl), caps(wl), "slow", "heavy", wl,
                                                 ["T2", "LM", "CT", "K", "K_ewma"], eta, src))
        wl = f"{key}.kdisk"
        km = man(wl)
        kc = km["capacities"]
        kcaps = {"gpu_kv_bytes": str(kc["gpu_kv_bytes"]), "cpu_bytes": str(kc["cpu_bytes"]), "disk_bytes": str(kc["disk_bytes"])}
        km = dict(km, c0_projection={"makespan_s": km["load"]["est_makespan_s"]})
        if key == "gemma4_31b":
            kd = cell(f"p2-{m['tag']}-kdisk", key, km, kcaps, "slow", "heavy", wl, ["T", "K"], eta,
                      "the kdisk manifest", extra={
                          "pass": {
                              "criterion": (
                                  "K: the rule chose ins or park, wrote chunks, restored from disk on verified turns "
                                  "(replay --verify-disk), and its share of verified turns whose greedy output differs "
                                  "from a cold recompute is not above stock vLLM's (arm T, its own disk tier, same "
                                  "workload and probe) at one-sided Fisher exact p < 0.05 (BENCH_PLAN draft 6)"),
                              "command": "python3 engine/phase_r/check_k_disk.py --rows CELL.K.HOST.jsonl --park-live "
                                         "CELL.K.HOST.park_live.json --min-verified 10 --reference-rows CELL.T.HOST.jsonl",
                              "capacity_rule": (f"CPU tier {kc['cpu_bytes']} B = 1.5 x the largest session's tier "
                                                f"bytes, {kc['cpu_over_peak_tier_bytes']} of the peak live tier bytes")}})
            for x in kd["arms"]:
                x["env"] = dict(x.get("env", {}), REPLAY_EXTRA="--verify-disk 20")
            put(f"p2-{m['tag']}-kdisk.json", kd)
        else:
            # engine-stall check (50780f4): C0 on the short kdisk workload, its engine stopped for 330 s mid-replay
            put(f"p2-{m['tag']}-stall.json", cell(f"p2-{m['tag']}-stall", key, km, kcaps, "slow", "heavy", wl, ["C0"],
                                                  eta, "the kdisk manifest", extra={
                "stall_check": ("PILOT2_PLAN.md job 3: SIGSTOP the engine for 330 s after replay_start + 120 s; the "
                                "verdict must exclude the arm as an engine stall (health.py engine_stall_s 300)")}))
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
