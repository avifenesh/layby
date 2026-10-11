#!/usr/bin/env python3
"""Phase R cell specs (engine/phase_r/orchestrate.py) for Qwen3.8-27B and Gemma 4 31B on vLLM 0.31.0, from the
workloads' capacities.json and manifests and the compat check (compat.json, micro.json).

Matrix (coordinator, 2026-10-11): per model slow heavy (measured), slow light (light), medium heavy (measured);
arms C0 T T2 LM CT K K_ewma, plus K_km on qwen slow heavy only. SGLang is dropped (no K arm on hybrid models), so is
T-LRU. Pilot cells per model: slow (the pilot workload, every arm label the matrix uses plus PT, so the pilot measures
each arm's RSS), medium (C0 T K: the NRD disk's probe, io.max and offered write), and kdisk (the K disk-path test
under CPU pressure, bench/phase_r/kdisk_workload.py, pass criterion engine/phase_r/check_k_disk.py).

Hosts: Nebius 1x RTX PRO 6000 (1gpu-24vcpu-218gb), uk-south2. Slow: the 2 TB network_ssd root disk with /data on it,
the provider's cap (compat fio O_DIRECT 457 MiB/s read, 460 MiB/s write), no io.max. Medium: a 2.5 TB
network_ssd_non_replicated data disk at /data2, with an io.max read cap so the regime is medium (at its nominal 2 GiB/s
the compat inputs give r >= 2 and w < 0.5, i.e. fast, for both models).

Fixed per cell: max-model-len 131,072 (both models boot there: compat boot_maxlen; Gemma's KV need for one request
is 25.01 GB with or without a connector, capacities.json gpu_request_check), capacities from capacities.json (GPU and
CPU per model, disk per workload; the kdisk cells take theirs from the kdisk manifest), CT_ETA of the model's
measured workload, LMCache chunk 784 tokens on Qwen (the hybrid block; 256 is refused), kv_chunk_bytes the vLLM
CPU tier's chunk (compat server logs: Qwen 16 GiB / 334 chunks, one 784-token block of 16 layers, 51,380,224 B;
Gemma 32 GiB / 13,107 chunks, one 16-token block of a 10-layer group, 2,621,440 B), hard_cap_s 3x the workload's
estimated C0 makespan (the builder's service-time estimate, est_makespan_s: no queueing), REPLAY_TIMEOUT the cap
+ 600 s, the runner timeout 1,800 s more.

PLACEHOLDER in the pilot JSONs (engine/phase_r/cells/pilot_MODEL_vllm.json), replaced by the pilot: arm RSS, sidecar
RSS, prefill tok/s at each load (compat c8 rate at 8,192 tokens) and write-through's offered write rate (that prefill
rate x the tier's measured bytes per token: write-through at full prefill, the compat check's convention), and the
medium cells' io.max rate.

Usage: make_cells.py WORKLOAD_DIR [--out engine/phase_r/cells]
"""
import argparse
import json
import os

GiB = 1 << 30
SEED = 20261010
PLAN = "BENCH_PLAN draft 4.3 + coordinator's 2026-10-11 model change (freeze commit to fill)"
NETWORK_SSD = dict(seq_read_Bps=478_779_802, seq_write_Bps=482_764_390, rand_read_Bps=475_844_198)  # compat fio, MiB/s x 2^20
NRD_NOMINAL = dict(seq_read_Bps=2 * GiB, seq_write_Bps=2 * GiB, rand_read_Bps=2 * GiB)
MEDIUM_RBPS = 1_000_000_000

MODELS = {
    "qwen38_27b": dict(
        tag="qwen", hf="Qwen/Qwen3.8-27B-FP8", serve="--revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a",
        lm_chunk=784, chunk_bytes=51_380_224, tier_bpt=72_472, prefill=10_428.8, kkm_cell="slow-heavy"),
    "gemma4_31b": dict(
        tag="gemma", hf="google/gemma-4-31B-it",
        serve='--revision 842da3794eaa0b77d5f08bae87a17459d91ff475 --quantization fp8 '
              '--limit-mm-per-prompt {"image":0,"audio":0,"video":0}',
        lm_chunk=None, chunk_bytes=2_621_440, tier_bpt=131_714, prefill=6_515.2, kkm_cell=None),
}
ARMS = ["C0", "T", "T2", "LM", "CT", "K", "K_ewma"]
T2_ENV = {"T2_MIN_HITS": "2", "T2_DECAY_S": "600", "T2_COUNT": "request"}   # tuned on the tuning workload later


def pilot_json(name, m):
    off = round(m["prefill"] * m["tier_bpt"])
    rss = 16 * GiB
    return {
        "_comment": (
            "PLACEHOLDER values for the dry runs until the pilot measures them (engine/phase_r/examples/"
            "pilot_qwen3_8b_vllm.json documents each field). kv_bytes_per_token: the tier's measured bytes per token "
            "(compat micro.json tier_measured, arm T); prefill_tok_s: compat c8 prefill at 8,192 tokens; "
            "wt_offered_write_Bps: that prefill rate x kv_bytes_per_token (write-through at full prefill); RSS: 16 GiB "
            "per vLLM arm, LMCache 15 + 4 GiB, sidecar 3 GiB, guesses."),
        "model": m["hf"], "engine": "vllm", "kv_bytes_per_token": m["tier_bpt"],
        "arm_rss_bytes": {**{x: rss for x in ("C0", "T", "T2", "PT", "CT", "K", "K_ewma", "K_km")},
                          "LM": {"vllm": 15 * GiB, "lmcache_server": 4 * GiB}},
        "sidecar_rss_bytes": 3 * GiB,
        "loads": {"heavy": {"prefill_tok_s": m["prefill"], "wt_offered_write_Bps": off},
                  "light": {"prefill_tok_s": m["prefill"], "wt_offered_write_Bps": off}},
    }


def cell(name, m, man, caps, regime, load, wl, arms, ct_eta, extra=None, pilot_cell=False):
    est = man["load"]["est_makespan_s"]
    cap = int(round(3 * est))
    env = {"REPLAY_TIMEOUT": str(cap + 600), "BOOT_WAIT": "360", "MAXLEN": "131072", "SERVE_ARGS": m["serve"]}
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
            f"Generated by bench/phase_r/make_cells.py (its docstring has every rule). Workload {wl}: sha256, "
            f"sessions and turns from its manifest; capacities from capacities.json (GPU and CPU per model, disk per "
            f"workload) or the kdisk manifest; hard_cap_s = 3 x est_makespan_s {est:g} s (builder estimate, no "
            f"queueing). PLACEHOLDER: the pilot JSON's RSS, prefill and offered write; the medium io.max rate."),
        "cell": name, "plan": PLAN, "seed": SEED, "engine": "vllm", "model": m["hf"], "regime": regime, "load": load,
        "pilot": f"pilot_{m['key']}_vllm.json",
        "workload": {"path": f"/root/phase_r/{wl}.json", "sessions": man["load"]["sessions"],
                     "turns": man["load"]["turns"], "sha256": man["sha256"]},
        "capacities": caps, "hard_cap_s": cap, "cgroup": {"slack_bytes": "256M"},
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
    # one slot, named by its disk: the host's first fio probe is kept per host and slot name, and the slow and
    # medium cells share a VM, so each disk is held to its own first probe
    slot = {"cuda": "0", "port": 8000, "park_port": 28765, "cpus": "2-23", "mems": "0"}
    if regime == "slow":
        c["slots"] = [dict(slot, name="netssd", disk={"mount": "/data", "wipe": "fstrim"})]
        c["io_max_none_reason"] = ("real Nebius network_ssd: the 2 TB root disk with /data on it, provider cap about "
                                   "450 MiB/s (compat fio O_DIRECT 457 MiB/s read, 460 MiB/s write)")
        c["dry_run"] = {"probe": dict(NETWORK_SSD)}
    else:
        c["slots"] = [dict(slot, name="nrd", disk={"mount": "/data2", "wipe": "fstrim"})]
        c["io_max"] = {"rbps": str(MEDIUM_RBPS)}
        c["io_max_note"] = ("PLACEHOLDER read cap: the 2.5 TB network_ssd_non_replicated disk at its nominal 2 GiB/s "
                            "gives r >= 2 and w < 0.5 (fast) on the compat inputs; 1 GB/s gives r 1.32 (Qwen) and 1.17 "
                            "(Gemma), w 0.35 and 0.40: medium. The pilot sets it from orchestrate.py regime bounds.")
        c["dry_run"] = {"probe": dict(NRD_NOMINAL),
                        "throttled_probe": dict(NRD_NOMINAL, seq_read_Bps=MEDIUM_RBPS, rand_read_Bps=MEDIUM_RBPS)}
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
    man = lambda n: json.load(open(os.path.join(a.workloads, f"{n}.manifest.json")))   # noqa: E731
    written = []

    def put(name, obj):
        p = os.path.join(out, name)
        with open(p, "w") as f:
            json.dump(obj, f, indent=1)
            f.write("\n")
        written.append(p)

    for key, m in MODELS.items():
        m["key"] = key
        mc = C["models"][key]
        eta = C["workloads"][f"{key}.measured"]["ct_eta"]
        put(f"pilot_{key}_vllm.json", pilot_json(key, m))
        caps = lambda wl: {"gpu_kv_bytes": str(mc["gpu_kv_bytes"]), "cpu_bytes": str(mc["cpu_bytes"]),   # noqa: E731
                           "disk_bytes": str(C["workloads"][wl]["disk_bytes"])}
        for regime, load, wlk in (("slow", "heavy", "measured"), ("slow", "light", "light"), ("medium", "heavy", "measured")):
            wl = f"{key}.{wlk}"
            arms = ARMS + (["K_km"] if m["kkm_cell"] == f"{regime}-{load}" else [])
            name = f"{m['tag']}-{regime}-{load}"
            put(f"{name}.json", cell(name, m, man(wl), caps(wl), regime, load, wl, arms, eta))
        wl = f"{key}.pilot"
        parms = ARMS + ["PT"] + (["K_km"] if m["kkm_cell"] else [])
        put(f"pilot-{m['tag']}-slow.json", cell(f"pilot-{m['tag']}-slow", m, man(wl), caps(wl), "slow", "heavy", wl,
                                               parms, eta))
        put(f"pilot-{m['tag']}-medium.json", cell(f"pilot-{m['tag']}-medium", m, man(wl), caps(wl), "medium", "heavy",
                                                 wl, ["C0", "T", "K"], eta))
        wl = f"{key}.kdisk"
        km = man(wl)
        kc = km["capacities"]
        kcaps = {"gpu_kv_bytes": str(kc["gpu_kv_bytes"]), "cpu_bytes": str(kc["cpu_bytes"]), "disk_bytes": str(kc["disk_bytes"])}
        kd = cell(f"pilot-{m['tag']}-kdisk", m, km, kcaps, "slow", "heavy", wl, ["C0", "K"], eta, extra={
            "pass": {
                "criterion": ("K: the rule chose ins or park (park_live counts), wrote chunks (disk.chunks_on_disk or "
                              "written_jobs > 0), restored from disk on at least one turn that ran alone (replay "
                              "--verify-disk: the disk tier's read bytes rose during the turn) and every such turn's "
                              "greedy output equals a cold recompute of the same prompt (fresh cache_salt)"),
                "command": "python3 engine/phase_r/check_k_disk.py --rows CELL.K.HOST.jsonl --park-live "
                           "CELL.K.HOST.park_live.json --min-verified 1",
                "capacity_rule": (f"CPU tier {kc['cpu_bytes']} B = 1.5 x the largest session's tier bytes, "
                                  f"{kc['cpu_over_peak_tier_bytes']} of the peak live tier bytes (the measured cells hold "
                                  f"0.4 P); GPU KV the model's cell value")}})
        for x in kd["arms"]:
            if x["label"] == "K":
                x["env"] = {"REPLAY_EXTRA": "--verify-disk 20"}
        put(f"pilot-{m['tag']}-kdisk.json", kd)
    for p in written:
        print(p)


if __name__ == "__main__":
    main()
