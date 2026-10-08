#!/usr/bin/env python3
"""Summarize vLLM closed-loop arms (run_arms.sh output).

Per arm: wall time, resumed-turn TTFT (turns after an idle wait) p50/p95/p99 overall, after tool
waits and after human waits, and by the hint that governed the wait; plus token provenance from
the server's /metrics delta: GPU prefix-cache hits, CPU-tier loads, recomputed prompt tokens.
A session-bootstrap CI (200 resamples) on the p95 resumed-turn TTFT ratio vs arm B.

Usage: analyze_arms.py OUTDIR [BASE_ARM]   (default base: lru_offload or B)
"""
import glob, json, os, re, sys
import numpy as np

d = sys.argv[1]
KIB = 144


def metric(lines, name, **labels):
    tot = 0.0
    for l in lines:
        if not l.startswith(name + "{") and not l.startswith(name + " "):
            continue
        if all(f'{k}="{v}"' in l for k, v in labels.items()):
            tot += float(l.rsplit(" ", 1)[1])
    return tot


arms = {}
for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
    name = os.path.basename(f)[:-6]
    rows, mets, wall = [], {}, None  # error rows are skipped below
    for l in open(f):
        r = json.loads(l)
        if "metrics" in r:
            mets[r["metrics"]] = r["lines"]
        elif "warm_at" in r:
            mets.setdefault("_warms", []).append(r)
        elif "error" in r:
            continue
        elif "truncated_at_window" in r:          # replay.py --window: the turn not sent
            mets.setdefault("_truncated", []).append(r)
        elif "wall_s" in r:
            wall = r["wall_s"]
        else:
            rows.append(r)
    arms[name] = (rows, mets, wall)


def pct(v, q):
    return float(np.quantile(v, q)) * 1000 if len(v) else float("nan")


def resumed(rows):
    return [r for r in rows if r["turn"] > 0 and r["ttft"] is not None]


base = arms.get(sys.argv[2]) if len(sys.argv) > 2 else (arms.get("lru_offload") or arms.get("B"))
out = {}
for name, (rows, mets, wall) in arms.items():
    res = resumed(rows)
    def stats(sel):
        v = np.array([r["ttft"] for r in sel])
        return dict(n=len(v), p50_ms=pct(v, .5), p95_ms=pct(v, .95), p99_ms=pct(v, .99), mean_ms=float(v.mean() * 1000) if len(v) else None)
    prev_kind = {}
    for r in rows:
        prev_kind[(r["session"], r["turn"] + 1)] = r["kind"]
    tool = [r for r in res if prev_kind.get((r["session"], r["turn"])) == "tool"]
    human = [r for r in res if prev_kind.get((r["session"], r["turn"])) == "human"]
    o = dict(wall_s=wall, turns=len(rows), resumed=stats(res), after_tool=stats(tool), after_human=stats(human),
             by_prev_hint={h: stats([r for r in res if r["prev_hint"] == h]) for h in ("move", "stay", "none")})
    if "before" in mets and "after" in mets:
        b, a = mets["before"], mets["after"]
        delta = lambda n, **kw: metric(a, n, **kw) - metric(b, n, **kw)
        q = delta("vllm:prefix_cache_queries_total")
        h = delta("vllm:prefix_cache_hits_total")
        ext_h = delta("vllm:external_prefix_cache_hits_total")
        load = ext_h  # tokens served from the CPU tier
        o["tokens"] = dict(prompt_tokens_queried=q, gpu_hit_tokens=h, cpu_loaded_tokens=load,
                           external_hits=ext_h, recomputed=q - h - load,
                           gpu_hit_pct=100 * h / q if q else None, cpu_hit_pct=100 * load / q if q else None,
                           preemptions=delta("vllm:num_preemptions_total"))
    # per-request provenance: share of the resumed turn's prompt served from cache (GPU or CPU)
    def cached_frac(r):
        u = r.get("usage") or {}
        c = ((u.get("prompt_tokens_details") or {}).get("cached_tokens"))
        if c is None:
            c = u.get("cached_tokens")              # SGLang meta_info
        return None if c is None or not u.get("prompt_tokens") else c / u["prompt_tokens"]
    cf = [(cached_frac(r), r["ttft"]) for r in res if cached_frac(r) is not None]
    if cf:
        full = [x for f, x in cf if f >= 0.9]
        part = [x for f, x in cf if f < 0.9]
        o["provenance"] = dict(resumed_mostly_cached_pct=100 * len(full) / len(cf),
                               ttft_p95_mostly_cached_ms=pct(np.array(full), .95),
                               ttft_p95_recomputed_ms=pct(np.array(part), .95),
                               n_recomputed=len(part))
    o["warms"] = len(mets.get("_warms", []))
    out[name] = o

# session-bootstrap of p95 resumed TTFT ratio vs B
if base:
    rng = np.random.default_rng(0)
    bres = resumed(base[0])
    for name, (rows, _, _) in arms.items():
        res = resumed(rows)
        sess = sorted({r["session"] for r in res} & {r["session"] for r in bres})
        by = {s: [r["ttft"] for r in res if r["session"] == s] for s in sess}
        byb = {s: [r["ttft"] for r in bres if r["session"] == s] for s in sess}
        ratios = []
        for _ in range(200):
            pick = rng.choice(sess, len(sess))
            v = np.concatenate([by[s] for s in pick]); vb = np.concatenate([byb[s] for s in pick])
            ratios.append(np.quantile(v, .95) / np.quantile(vb, .95))
        out[name]["p95_vs_lru_offload"] = dict(ratio=float(np.median(ratios)),
                                               ci95=[float(np.quantile(ratios, .025)), float(np.quantile(ratios, .975))])
print(json.dumps(out, indent=1))
