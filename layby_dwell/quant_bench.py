"""8-bit study: curves, speed and memory per precision on one GPU.

Usage: python -m layby_dwell.quant_bench STATES.parquet OUT.npz [--model DIR] [--precision bf16 fp32 int8w fp8w]
       [--repeats 3]

STATES.parquet: columns state (JSON) and kind (as layby_dwell.bench). For every precision: the survival rows of
all states (batch 32), single-state latency (median of 50 calls; median over --repeats runs), throughput at batch
1, 8, 32 and 128 over the first 1,024 states (median over --repeats runs), and peak GPU memory. Writes the rows and
a JSON summary into OUT.npz; compare them with layby_dwell.quant_report.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from layby_dwell.bench import server
from layby_dwell.model import Dwell


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states"); ap.add_argument("out"); ap.add_argument("--model")
    ap.add_argument("--precision", nargs="+", default=["bf16", "fp32", "int8w", "fp8w"])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    df = pd.read_parquet(a.states)
    states = [server(s) for s in df.state]
    kinds = np.where(df.kind.values == "workflow", "human", df.kind.values)
    rows, summary = {}, {}
    for prec in a.precision:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            m = Dwell(a.model, device="cuda", precision=prec, batch_size=32)
            m.survival(states[:64], kinds[:64])                         # warm-up (and kernel autotune)
        except Exception as e:  # noqa: BLE001  (a precision this GPU or torchao build cannot run)
            summary[prec] = dict(error=repr(e)[:300]); print(prec, "failed:", repr(e)[:300]); continue
        S = m.survival(states, kinds)
        rows[prec] = S
        lat, thr = [], {b: [] for b in (1, 8, 32, 128)}
        sub, sk = states[:1024], kinds[:1024]
        for _ in range(a.repeats):
            one = [timed(lambda i=i: m.survival([states[i]], [kinds[i]])) for i in range(50)]
            lat.append(float(np.median(one)))
            for b in thr:
                m.batch_size = b
                try:
                    thr[b].append(len(sub) / timed(lambda: m.survival(sub, sk)))
                except torch.OutOfMemoryError:
                    thr[b].append(float("nan")); torch.cuda.empty_cache()   # this batch does not fit the card
            m.batch_size = 32
        summary[prec] = dict(latency_ms=round(1e3 * float(np.median(lat)), 2),
                             states_per_s={b: (round(float(np.median(v)), 1) if np.isfinite(v).all() else "oom")
                                           for b, v in thr.items()},
                             peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                             nan_rows=int(np.isnan(S).all(1).sum()))
        print(prec, json.dumps(summary[prec]), flush=True)
        del m
    np.savez(a.out, summary=json.dumps(dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, n=len(states),
                                            precisions=summary)), **rows)


if __name__ == "__main__":
    main()
