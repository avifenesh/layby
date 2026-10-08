#!/usr/bin/env python3
"""Pool repeated vLLM arms (run_arms3.sh output: C0, C0.2, ...) and compare them head to head.

Per arm group: resumed-turn TTFT p50/p95/p99 per repeat and pooled, pooled after tool / after human waits,
warms, recomputed prompt tokens and CPU<->GPU offload GB per repeat. Ratios (p50 and p95) of every group against the base
group and of each --pair, with a session bootstrap (200 resamples) where sessions are keyed per repeat:
key (i, s) pairs repeat i of one group with repeat i of the other on session s.

Usage: pool_arms.py OUTDIR [--base C0] [--pair V6:Sp ...] [--exclude s002,s008,s014] [--out pooled.json]
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("d"); ap.add_argument("--base", default="C0")
ap.add_argument("--pair", action="append", default=[])
ap.add_argument("--exclude", default="")
ap.add_argument("--out", help="also write the JSON to this file")
a = ap.parse_args()
EX = set(x for x in a.exclude.split(",") if x)


def metric(lines, name):
    return sum(float(l.rsplit(" ", 1)[1]) for l in lines if l.startswith(name + "{") or l.startswith(name + " "))


groups = {}
for f in sorted(glob.glob(os.path.join(a.d, "*.jsonl"))):
    name = os.path.basename(f)[:-6]
    rows, mets, warms, xfer = [], {}, 0, None
    for l in open(f):
        r = json.loads(l)
        if "metrics" in r:
            mets[r["metrics"]] = r["lines"]
        elif "warm_at" in r:
            warms += 1
        elif "error" in r or "wall_s" in r:
            continue
        else:
            rows.append(r)
    prev = {(r["session"], r["turn"] + 1): r["kind"] for r in rows}
    res = [dict(s=r["session"], t=r["ttft"], after=prev.get((r["session"], r["turn"])))
           for r in rows if r["turn"] > 0 and r["ttft"] is not None and r["session"] not in EX]
    rec = None
    if "before" in mets and "after" in mets:
        dl = lambda n: metric(mets["after"], n) - metric(mets["before"], n)
        rec = dl("vllm:prefix_cache_queries_total") - dl("vllm:prefix_cache_hits_total") - dl("vllm:external_prefix_cache_hits_total")
        xfer = {t: round((metric([l for l in mets["after"] if f'transfer_type="{t}"' in l], "vllm:kv_offload_total_bytes_total")
                          - metric([l for l in mets["before"] if f'transfer_type="{t}"' in l], "vllm:kv_offload_total_bytes_total")) / 1e9, 1)
                for t in ("CPU_to_GPU", "GPU_to_CPU")}
    groups.setdefault(name.split(".")[0], []).append(dict(rep=name, res=res, warms=warms, recomputed=rec, xfer_GB=xfer, turns=len(rows)))
for g in groups.values():
    g.sort(key=lambda x: x["rep"])

q = lambda v, p: float(np.quantile(v, p)) if len(v) else float("nan")


def vals(g, sel=None):
    return np.array([r["t"] for rep in g for r in rep["res"] if sel is None or r["after"] == sel])


def boot(gx, gy):
    n = min(len(gx), len(gy))
    keys = sorted({(i, r["s"]) for i in range(n) for r in gx[i]["res"]} & {(i, r["s"]) for i in range(n) for r in gy[i]["res"]})
    bx = {k: [r["t"] for r in gx[k[0]]["res"] if r["s"] == k[1]] for k in keys}
    by = {k: [r["t"] for r in gy[k[0]]["res"] if r["s"] == k[1]] for k in keys}
    vx = np.concatenate([bx[k] for k in keys]); vy = np.concatenate([by[k] for k in keys])
    rng = np.random.default_rng(0); r50, r95 = [], []
    for _ in range(200):
        pick = rng.integers(0, len(keys), len(keys))
        x = np.concatenate([bx[keys[i]] for i in pick]); y = np.concatenate([by[keys[i]] for i in pick])
        r50.append(q(x, .5) / q(y, .5)); r95.append(q(x, .95) / q(y, .95))
    ci = lambda r: [round(q(r, .025), 3), round(q(r, .975), 3)]
    return dict(reps=n, p95=round(q(vx, .95) / q(vy, .95), 3), p95_ci=ci(r95),
                p50=round(q(vx, .5) / q(vy, .5), 3), p50_ci=ci(r50))


out = dict(excluded=sorted(EX), arms={}, vs_base={}, pairs={})
for name, g in groups.items():
    v = vals(g); t = vals(g, "tool"); h = vals(g, "human")
    out["arms"][name] = dict(
        reps=[dict(rep=r["rep"], turns=r["turns"], n=len(r["res"]), p50=round(q([x["t"] for x in r["res"]], .5), 2),
                   p95=round(q([x["t"] for x in r["res"]], .95), 2), warms=r["warms"],
                   recomputed_M=None if r["recomputed"] is None else round(r["recomputed"] / 1e6, 2), xfer_GB=r["xfer_GB"]) for r in g],
        pooled=dict(n=len(v), p50=round(q(v, .5), 2), p95=round(q(v, .95), 2), p99=round(q(v, .99), 2), mean=round(float(v.mean()), 2)),
        after_tool=dict(n=len(t), p50=round(q(t, .5), 2), p95=round(q(t, .95), 2)),
        after_human=dict(n=len(h), p50=round(q(h, .5), 2), p95=round(q(h, .95), 2)))
if a.base in groups:
    for name, g in groups.items():
        if name != a.base:
            out["vs_base"][name] = boot(g, groups[a.base])
for p in a.pair:
    x, y = p.split(":")
    if x in groups and y in groups:
        out["pairs"][p] = boot(groups[x], groups[y])
print(json.dumps(out, indent=1))
if a.out:
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1); f.write("\n")
