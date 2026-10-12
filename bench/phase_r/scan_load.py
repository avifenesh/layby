#!/usr/bin/env python3
"""Load scan for the Phase R re-size (BENCH_PLAN draft 6): the first N candidate sessions of a partition, in the
builder's order (build_workloads.py --dump-candidates), cut where their projected C0 end passes --end-cap, run
through the fitted C0 service model (fit_service.py, sim_sizing.py). Per N: projected peak live KV, mean requests in
flight, resumed-turn TTFT p50 / p95 and its queueing part, makespan. The builder then sizes by KV pressure; this
scan shows what load that KV pressure implies and where C0's queue stops draining.

Usage: scan_load.py CANDIDATES.json FIT.json --n 4,8,12,... [--end-cap 3000] [--gpu-kv-bytes B] [--out OUT.json]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_sizing import load_fit, project, stats  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidates")
    ap.add_argument("fit")
    ap.add_argument("--n", required=True)
    ap.add_argument("--end-cap", type=float, default=3000.0)
    ap.add_argument("--gpu-kv-bytes", type=float, help="report GPU KV / peak live KV")
    ap.add_argument("--out")
    a = ap.parse_args()
    C = json.load(open(a.candidates))
    fit = load_fit(a.fit)
    rows = []
    for n in [int(x) for x in a.n.split(",")]:
        S, R, info, rounds = project([dict(x) for x in C["sessions"][:n]], fit, a.end_cap)
        st = stats(S, R, C["kv"])
        st = dict(n_candidates=n, rounds=rounds, **st)
        if a.gpu_kv_bytes:
            st["gpu_over_P"] = round(a.gpu_kv_bytes / st["peak_live_kv_bytes"], 4)
        rows.append(st)
        print(json.dumps(st), flush=True)
    if a.out:
        json.dump(dict(candidates=os.path.basename(a.candidates), fit=os.path.basename(a.fit), end_cap=a.end_cap,
                       rows=rows), open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
