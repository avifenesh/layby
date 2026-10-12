#!/usr/bin/env python3
"""The K disk-path pilot workload: a short cut of a model's pilot workload (pilot partition) with capacities under
which the CPU tier cannot hold the idle sessions, so the rule must choose ins or park for some of them.

The compat check's K smoke never exercised the disk path: its CPU tier held every idle session, and the rule chose
none each time. Here every session of SRC (NAME.pilot.json, curves attached: surv_v6, surv_km, surv_ewma) is kept
with the turns that end, on the builder's service-time estimate (build_workloads.py est: C0 + TPOT x out + B x
prompt / 1000, plus the capped gaps), within --minutes of the run start; a session left with fewer than two turns is
dropped, and the last kept turn ends the session (gap_after None). Turns, contexts, keep cuts, starts and curves are
the source's: nothing is re-sampled.

Capacities (--tier-bytes-per-token: the CPU and disk tiers' measured bytes per token, compat micro.json
tier_measured): a restore from disk is staged through the CPU tier, so the tier must hold the largest session
(--cpu-sessions x the largest context x tier bytes per token, rounded up to a whole tier chunk); every other session's
idle KV has to go to disk or be recomputed. GPU KV is the model's cell value (capacities.json gpu_kv_bytes). The
script refuses a cut whose CPU tier holds more than --max-cpu-share of the peak live tier bytes (default 0.25; the
measured cells hold 0.4 P). Disk tier: 1.2 x unique KV at turn ends, as finalize_workloads.py.

With --service-fit FIT.json (fit_service.py, BENCH_PLAN draft 6) the cut and the timeline come from the fitted C0
projection (sim_sizing.project at the model's cell capacities from capacities.json, end cap --minutes) instead of the per-turn estimate.

Writes OUT/NAME.json, OUT/NAME.manifest.json (source, cut rule, counts, capacities, sha256) and, from the source's
side file, OUT/NAME.side.jsonl (mode 600; the live sidecar's input).

Usage: kdisk_workload.py SRC.json --models models_phase_r.json --capacities capacities.json --name NAME \
       --tier-bytes-per-token B --chunk-bytes C [--minutes 20] [--cpu-sessions 1.5] [--out DIR]
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def kv_bytes(k, c):
    """build_workloads.py kv_bytes: grow x c + recurrent state + min(c, window) x window bytes."""
    return k["grow_bytes_per_token"] * c + k["state_bytes_per_session"] + k["window_bytes_per_token"] * min(c, k["window_tokens"])


def est(M, prompt, out_len):
    return M["c0"] + M["tpot"] * out_len + M["b"] * prompt / 1000.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src")
    ap.add_argument("--models", required=True)
    ap.add_argument("--capacities", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tier-bytes-per-token", type=float, required=True)
    ap.add_argument("--chunk-bytes", type=int, required=True)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--cpu-sessions", type=float, default=1.5)
    ap.add_argument("--max-cpu-share", type=float, default=0.25)
    ap.add_argument("--service-fit", help="FIT.json: cut and time on the fitted C0 projection")
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.dirname(os.path.abspath(a.src))
    W = json.load(open(a.src))
    M = next(m for m in json.load(open(a.models)) if m["name"] == W["model"])
    caps = json.load(open(a.capacities))["models"][W["model"]]
    k = M["kv"]
    end = a.minutes * 60.0

    if a.service_fit:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from sim_sizing import load_fit, project
        cc = dict(gpu_bytes=caps["gpu_kv_bytes"], cpu_bytes=caps["cpu_bytes"])     # the model's cell capacities
        sessions, rows, _, _ = project([dict(s) for s in W["sessions"]], load_fit(a.service_fit), end, min_turns=2, caps=cc)
        E = {(r["session"], r["turn"]): r["t_done"] - r["t_send"] for r in rows}
        SEND = {(r["session"], r["turn"]): r["t_send"] for r in rows}
    else:
        sessions, E, SEND = None, None, None

    def dur(s, i, t):
        return E[(s["id"], i)] if E is not None else est(M, t["ctx"] - t["out_len"], t["out_len"])

    est_path = sessions is None
    sessions = [] if est_path else sessions
    for s in (W["sessions"] if est_path else []):
        tm, keep = s["start"], []
        for t in s["turns"]:
            e = est(M, t["ctx"] - t["out_len"], t["out_len"])
            if tm + e > end:
                break
            keep.append(dict(t))
            tm += e
            if t["gap_after"] is None:
                break
            tm += t["gap_after"]
        if len(keep) < 2:
            continue
        keep[-1]["gap_after"] = None
        sessions.append(dict(s, turns=keep))

    ev, busy, ends, uniq = [], 0.0, [], 0
    for s in sessions:
        tm, curb = s["start"], 0.0
        for i, t in enumerate(s["turns"]):
            e = dur(s, i, t)
            tm = SEND[(s["id"], i)] if SEND is not None else tm
            b = kv_bytes(k, t["ctx"])
            ev.append((tm, b - curb)); curb = b
            tm += e + (t["gap_after"] or 0.0)
            busy += e
            uniq += t["new_tokens"] + t["out_len"]
        ev.append((tm, -curb)); ends.append(tm)
    ev.sort()
    P = float(np.max(np.cumsum([d for _, d in ev])))
    peak_tokens = 0.0
    evt = []
    for s in sessions:
        tm, cur = s["start"], 0
        for i, t in enumerate(s["turns"]):
            tm = SEND[(s["id"], i)] if SEND is not None else tm
            evt.append((tm, t["ctx"] - cur)); cur = t["ctx"]
            tm += dur(s, i, t) + (t["gap_after"] or 0.0)
        evt.append((tm, -cur))
    evt.sort()
    peak_tokens = float(np.max(np.cumsum([d for _, d in evt])))
    mx = np.array([max(t["ctx"] for t in s["turns"]) for s in sessions])
    tier_peak = peak_tokens * a.tier_bytes_per_token
    cpu = math.ceil(a.cpu_sessions * mx.max() * a.tier_bytes_per_token / a.chunk_bytes) * a.chunk_bytes
    share = cpu / tier_peak
    T = [t for s in sessions for t in s["turns"]]
    resumed = [(s["turns"][i - 1]["gap_after"], s["turns"][i]["kind"]) for s in sessions for i in range(1, len(s["turns"]))]
    win = int(sum(min(t["new_tokens"] + t["out_len"], k["window_tokens"]) for t in T)) if k["window_tokens"] else 0
    ub = k["grow_bytes_per_token"] * uniq + k["state_bytes_per_session"] * len(T) + k["window_bytes_per_token"] * win
    stats = dict(sessions=len(sessions), turns=len(T), resumed_turns=len(resumed),
                 resumed_after_gap_ge_60s=int(sum(g >= 60 for g, _ in resumed)),
                 resumed_after_gap_ge_300s=int(sum(g >= 300 for g, _ in resumed)),
                 max_ctx_p50_p90_max=[float(np.median(mx)), float(np.percentile(mx, 90)), int(mx.max())],
                 peak_live_kv_bytes=int(P), peak_live_tokens=int(peak_tokens), peak_live_tier_bytes=int(tier_peak),
                 unique_kv_tokens=int(uniq), unique_kv_bytes=int(ub), est_makespan_s=round(max(ends), 0),
                 est_mean_inflight=round(busy / max(ends), 3))
    capacities = dict(gpu_kv_bytes=caps["gpu_kv_bytes"], cpu_bytes=int(cpu), disk_bytes=int(1.2 * ub),
                      cpu_over_peak_tier_bytes=round(share, 4), cpu_over_largest_session=a.cpu_sessions,
                      tier_bytes_per_token=a.tier_bytes_per_token, chunk_bytes=a.chunk_bytes)
    print(json.dumps(dict(stats, capacities=capacities)), flush=True)
    if share > a.max_cpu_share:
        sys.exit(f"CPU tier {cpu} B holds {share:.2f} of the peak live tier bytes (> {a.max_cpu_share}): no CPU pressure")

    W2 = {kk: v for kk, v in W.items() if kk != "sessions"}
    W2.update(src=f"Layby Phase R {a.name}: {os.path.basename(a.src)} cut at {a.minutes:g} minutes (K disk-path pilot)",
              sessions=sessions)
    p = os.path.join(out, f"{a.name}.json")
    with open(p + ".tmp", "w") as f:
        json.dump(W2, f)
    os.replace(p + ".tmp", p)
    keys = {(s["id"], i) for s in sessions for i in range(len(s["turns"]))}
    side_src = a.src[: -len(".json")] + ".side.jsonl"
    sj = os.path.join(out, f"{a.name}.side.jsonl")
    n_side = 0
    with open(sj + ".tmp", "w") as f:
        for ln in open(side_src):
            r = json.loads(ln)
            if (r["session"], int(r["turn"])) in keys:
                f.write(ln); n_side += 1
    os.chmod(sj + ".tmp", 0o600)
    os.replace(sj + ".tmp", sj)
    man = dict(name=a.name, model=W["model"], source=dict(path=os.path.basename(a.src), sha256=sha256(a.src)),
               rule=dict(minutes=a.minutes, min_turns=2, cpu_sessions=a.cpu_sessions, max_cpu_share=a.max_cpu_share,
                         script="bench/phase_r/kdisk_workload.py",
                         timeline=("fitted C0 projection " + os.path.basename(a.service_fit)) if a.service_fit
                         else "builder estimate"),
               load=stats, capacities=capacities, curve=W.get("curve"),
               side_jsonl=dict(rows=n_side, sha256=sha256(sj)), bytes=os.path.getsize(p), sha256=sha256(p),
               pass_criterion="engine/phase_r/check_k_disk.py: K chose ins or park, wrote chunks, restored from disk "
                              "(replay --verify-disk) with output identical to a cold recompute")
    with open(os.path.join(out, f"{a.name}.manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("wrote", p, sj, flush=True)


if __name__ == "__main__":
    main()
