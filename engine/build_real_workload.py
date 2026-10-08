#!/usr/bin/env python3
"""Engine replay workload from real-length pools (build_replay_pop.py --no-scale), with curves attached.

Draws N sessions over the pools like eval_pop.workload (an even share per pool, starts uniform over --start-window
seconds), except that a pool with fewer sessions than its share gives the rest to the others, so N is reached when
the pools hold N sessions in all. Gaps are capped at --gap-cap (the replay cap; a longer wait is a 300 s return).
Every turn carries surv_v6 from PROBS (ref, p0..p14) joined on the pool's refs (sid, turn, feat_ref); a turn with no
curve is reported and left without one.

Prints the load: per-session max context, token-turns, the peak over time of the KV tokens of started, unfinished
sessions (each at its current context, ignoring service time), and the replay length (latest start plus gap sum).

Usage: build_real_workload.py OUT.json N --pools P1.json:P1.refs.parquet ... --probs A.parquet [B.parquet ...]
       [--seed 0] [--gap-cap 300] [--start-window 600]
"""
import argparse, json
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("n", type=int)
ap.add_argument("--pools", nargs="+", required=True)
ap.add_argument("--probs", nargs="+", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gap-cap", type=float, default=300.0)
ap.add_argument("--start-window", type=float, default=600.0)
a = ap.parse_args()

P = pd.concat([pd.read_parquet(p) for p in a.probs]).drop_duplicates("ref").set_index("ref")
pcols = [f"p{j}" for j in range(15)]
rng = np.random.default_rng(a.seed)
pools = []
for spec in a.pools:
    js, refs = spec.split(":")
    S = json.load(open(js))["sessions"]
    R = pd.read_parquet(refs).set_index(["sid", "turn"]).feat_ref
    pools.append((js, S, R))

# even shares, the remainder of small pools handed to the others
want = [a.n // len(pools) + (i < a.n % len(pools)) for i in range(len(pools))]
take = [min(w, len(S)) for w, (_, S, _) in zip(want, pools)]
left = a.n - sum(take)
while left > 0:
    room = [i for i, (_, S, _) in enumerate(pools) if take[i] < len(S)]
    if not room:
        break
    for i in room:
        if left == 0:
            break
        take[i] += 1; left -= 1

sessions, missing, total = [], 0, 0
for k, (js, S, R) in zip(take, pools):
    tag = js.rsplit("/", 1)[-1][:3]
    for j in rng.choice(len(S), k, replace=False):
        s = S[j]
        turns = []
        for i, t in enumerate(s["turns"]):
            t = {x: t[x] for x in ("new_tokens", "out_len", "kind", "ctx", "gap_after")}
            if t["gap_after"] is not None:
                t["gap_after"] = float(min(t["gap_after"], a.gap_cap))
            ref = R.get((s["id"], i))
            total += 1
            if ref is not None and ref in P.index:
                t["surv_v6"] = [float(x) for x in P.loc[ref, pcols]]
            else:
                missing += 1
            turns.append(t)
        sessions.append(dict(id=f"{tag}{s['id']}", pool=s["pool"], seed=s["seed"],
                             start=float(rng.uniform(0, a.start_window)), turns=turns))

json.dump(dict(gap_cap=a.gap_cap, ctx_cap=None, src="real-length public pools", sessions=sessions), open(a.out, "w"))

# load
mx = np.array([max(t["ctx"] for t in s["turns"]) for s in sessions])
tt = sum(t["ctx"] for s in sessions for t in s["turns"])
ev = []                                   # (time, +/- tokens): a session holds its current context from turn to turn
end = []
for s in sessions:
    tm, cur = s["start"], 0
    for t in s["turns"]:
        ev.append((tm, t["ctx"] - cur)); cur = t["ctx"]
        tm += t["gap_after"] or 0.0
    ev.append((tm, -cur)); end.append(tm)
ev.sort()
peak = np.max(np.cumsum([d for _, d in ev]))
print(f"{a.out}: sessions {len(sessions)} {dict(zip([p[0].rsplit('/',1)[-1] for p in pools], take))} turns {total}"
      f" (no curve {missing}) | max ctx p50 {np.median(mx):.0f} p90 {np.percentile(mx, 90):.0f} max {mx.max()}"
      f" | token-turns {tt / 1e6:.0f}M | peak live KV {peak / 1e6:.2f}M tokens | replay length {max(end) / 3600:.2f} h"
      f" (session p50 {np.median(np.array(end) - [s['start'] for s in sessions]) / 60:.0f} min)")
