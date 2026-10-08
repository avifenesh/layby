#!/usr/bin/env python3
"""Attach survival curves to a replay: turn j of session sid gets surv_NAME = [P(T > E2[k])] from
PROBS (ref, p0..p14), joined through REFS (sid, turn, KEY). Turns without a score get null.

Usage: attach_curves.py replay.json REFS.parquet PROBS.parquet NAME OUT.json [--key ref]
"""
import argparse, json
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("replay"); ap.add_argument("refs"); ap.add_argument("probs"); ap.add_argument("name"); ap.add_argument("out")
ap.add_argument("--key", default="ref")
a = ap.parse_args()
W = json.load(open(a.replay))
R = pd.read_parquet(a.refs, columns=["sid", "turn", a.key]).rename(columns={a.key: "ref"})
P = pd.read_parquet(a.probs).drop_duplicates("ref")
cols = [f"p{j}" for j in range(15)]
M = R.merge(P[["ref"] + cols], on="ref", how="left")
M = M.dropna(subset=["p0"])
cur = {(s, int(j)): [float(x) for x in v] for s, j, v in zip(M.sid, M.turn, M[cols].values)}
miss = 0
for s in W["sessions"]:
    for j, t in enumerate(s["turns"]):
        t[f"surv_{a.name}"] = cur.get((s["id"], j))
        miss += t[f"surv_{a.name}"] is None
json.dump(W, open(a.out, "w"))
print(f"turns {sum(len(s['turns']) for s in W['sessions'])} without a curve {miss}")
