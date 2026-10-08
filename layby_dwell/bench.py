"""Scorer speed and precision check.

Usage: python -m layby_dwell.bench STATES.parquet [--model DIR] [--device cpu|cuda]
           [--precision fp32 bf16 int8w fp8w] [--n 512] [--ref REF.parquet]

STATES.parquet: columns state (JSON) and kind; workflow kinds and src fields are mapped to the server
view. The first precision is the reference; every other one reports the largest absolute difference
of P(T > h) against it. --ref: survival rows from an earlier scorer (columns ref, p0..p14) to check
the reference against. Prints single-state latency (median of 50) and throughput at the batch size.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from layby_dwell.model import Dwell


def server(state):
    d = json.loads(state)
    d.pop("src", None)
    if d.get("kind") == "workflow":
        d["kind"] = "human"
    if d.get("recent"):
        d["recent"] = [["human" if x and x[0] == "workflow" else x[0]] + list(x[1:]) for x in d["recent"]]
    m = d.get("meta") or {}
    if m.get("prev_kind") == "workflow":
        m["prev_kind"] = "human"
    return json.dumps(d, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states"); ap.add_argument("--model"); ap.add_argument("--device", default="cpu")
    ap.add_argument("--precision", nargs="+", default=["fp32"]); ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batch", type=int, default=32); ap.add_argument("--ref"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = pd.read_parquet(a.states)
    df = df.sample(min(a.n, len(df)), random_state=a.seed).reset_index(drop=True)
    states = [server(s) for s in df.state]
    kinds = np.where(df.kind.values == "workflow", "human", df.kind.values)
    base = None
    for prec in a.precision:
        m = Dwell(a.model, device=a.device, precision=prec, batch_size=a.batch)
        items = m.encode(states)
        lens = np.array([len(x[0]) for x in items if x is not None])
        if base is None:
            print(f"{len(states)} states, {int((~np.array([x is None for x in items])).sum())} fit; tokens p50 {np.median(lens):.0f} "
                  f"p90 {np.quantile(lens, 0.9):.0f} max {lens.max()}")
        m.survival(states[:a.batch], kinds[:a.batch])                     # warm-up
        one = []
        for i in range(50):
            t = time.perf_counter(); m.survival([states[i]], [kinds[i]])
            if a.device == "cuda":
                torch.cuda.synchronize()
            one.append(time.perf_counter() - t)
        t = time.perf_counter(); S = m.survival(states, kinds)
        if a.device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t
        msg = f"{a.device} {prec}: one state {np.median(one) * 1e3:.1f} ms, batch {a.batch}: {len(states) / dt:.1f} states/s"
        if base is None:
            base = S
            if a.ref:
                r = pd.read_parquet(a.ref).set_index("ref").loc[df.ref.values][[f"p{j}" for j in range(S.shape[1])]].values
                msg += f", max |dP| vs ref {np.nanmax(np.abs(S - r)):.4f}"
        else:
            d = np.abs(S - base)
            msg += f", max |dP| vs {a.precision[0]} {np.nanmax(d):.4f} (p99 {np.nanquantile(d, 0.99):.4f})"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
