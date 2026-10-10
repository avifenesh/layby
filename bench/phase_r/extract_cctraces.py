#!/usr/bin/env python3
"""SemiAnalysis cc-traces (semianalysisai/cc-traces-weka-no-subagents-051226, Apache-2.0) -> one row per request.

The source is content-free: request bodies were redacted at capture (privacy_mode anon), so a trace gives
per request the offset t, model, prompt tokens `in` (proxy tokenizer), output tokens `out`, api_time,
think_time (wall gap from the previous request's end), ttft and per-request 64-token block hashes that are
comparable only within a trace. No text, no tool names, no turn kind.

Output columns: trace (the source trace id), idx, t, model, type, in_tok, out_tok, api_time, think_time,
ttft, n_blocks, prefix_blocks (leading blocks shared with the previous request), prefix_same_model (leading
blocks shared with the previous request of the same model: what that model's prefix cache can reuse when
utility calls of another model are interleaved), gap_after (the next request's think_time; null on the last
request, where the trace ends).

Usage: extract_cctraces.py traces.jsonl OUT.parquet
"""
import json
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]


def common(a, b):
    n = 0
    for x, y in zip(a, b or ()):
        if x != y:
            break
        n += 1
    return n


rows = []
with open(src) as f:
    for line in f:
        tr = json.loads(line)
        prev, last = None, {}
        reqs = tr["requests"]
        for i, r in enumerate(reqs):
            h = r.get("hash_ids") or []
            pb, pm = common(h, prev), common(h, last.get(r.get("model")))
            last[r.get("model")] = h
            nxt = reqs[i + 1].get("think_time") if i + 1 < len(reqs) else None
            rows.append((str(tr["id"]), i, r.get("t"), r.get("model"), r.get("type"), r.get("in"), r.get("out"),
                         r.get("api_time"), r.get("think_time"), r.get("ttft"), len(h), pb, pm, nxt))
            prev = h
d = pd.DataFrame(rows, columns=["trace", "idx", "t", "model", "type", "in_tok", "out_tok", "api_time", "think_time",
                                "ttft", "n_blocks", "prefix_blocks", "prefix_same_model", "gap_after"])
d.to_parquet(out, index=False)
print(f"{out}: traces {d.trace.nunique()} requests {len(d)}")
