"""Where scoring time goes: tokenizing states (CPU) vs the forward pass (GPU), per precision.

Usage: python -m layby_dwell.quant_split STATES.parquet [--model DIR] [--precision bf16 int8w fp8w] [--repeats 3]

Encodes the first 1,024 states once (timed), then times only the forward pass (Dwell.logits on the encoded items)
at batch 8, 32 and 128. Medians over --repeats runs.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch

from layby_dwell.bench import server
from layby_dwell.model import Dwell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("states"); ap.add_argument("--model")
    ap.add_argument("--precision", nargs="+", default=["bf16", "int8w", "fp8w"])
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    df = pd.read_parquet(a.states)
    states = [server(s) for s in df.state][:1024]
    out = {}
    for prec in a.precision:
        try:
            m = Dwell(a.model, device="cuda", precision=prec)
        except Exception as e:  # noqa: BLE001
            out[prec] = dict(error=repr(e)[:200]); continue
        enc = []
        for _ in range(a.repeats):
            t = time.perf_counter(); items = m.encode(states); enc.append(time.perf_counter() - t)
        m.logits(items[:64])
        fw = {}
        for b in (8, 32, 128):
            m.batch_size = b
            ts = []
            for _ in range(a.repeats):
                torch.cuda.synchronize(); t = time.perf_counter(); m.logits(items); torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            fw[b] = round(len(items) / float(np.median(ts)), 1)
        out[prec] = dict(encode_states_per_s=round(len(states) / float(np.median(enc)), 1), forward_states_per_s=fw)
        print(prec, json.dumps(out[prec]), flush=True)
        del m; torch.cuda.empty_cache()
    print(json.dumps(dict(gpu=torch.cuda.get_device_name(), split=out)))


if __name__ == "__main__":
    main()
