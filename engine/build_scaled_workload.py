#!/usr/bin/env python3
"""Engine workload from the 32k-scaled ReturnBench pools, as mix36_w0.json: N sessions drawn with sim.eval_pop.workload
(an even share per pool, starts uniform over 600 s, numpy seed W), gaps capped at 300 s, user/tool/prog fields dropped.

Usage: build_scaled_workload.py OUT.json N W POOL.json [POOL.json ...]
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from sim.eval_pop import workload  # noqa: E402

out, n, w, pools = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
W = workload("+".join(pools), n, w)
for s in W["sessions"]:
    s.pop("user", None)
    for t in s["turns"]:
        if t["gap_after"] is not None:
            t["gap_after"] = float(min(t["gap_after"], 300.0))
        t.pop("tool", None)
        t.pop("prog", None)
W["src"] = f"ReturnBench mix ({'+'.join(os.path.basename(p).split('.')[0] for p in pools)}), n={n}, workload {w}, gap cap 300 s"
json.dump(W, open(out, "w"))
print(out, len(W["sessions"]), "sessions", sum(len(s["turns"]) for s in W["sessions"]), "turns")
