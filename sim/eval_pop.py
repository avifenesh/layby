"""Run placement policies on workloads sampled from ReturnBench pools, in the engine simulator.

A cell is (pool, n, disk). A workload draws n sessions from a pool (several pools joined with +, drawn
evenly) with fresh starts uniform over 600 s. Each workload runs under `seeds` engine seeds at each disk
speed. Per cell and rule: the median over workloads of the per-workload median-over-seeds p50 and p95
TTFT of returning turns, and the geometric mean over workloads of the ratio against C0.
"""
import importlib, json, os, sys
from multiprocessing import Pool
import numpy as np

from .engine_sim import simulate, summarize, Params
from .cost_rule import StkRule, CostRule, Oracle
from .baselines import ContinuumTTL, ChoiJoshi, KeepAlive, WriteThroughDisk

POOLS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "pools")
# engine calibration of the reference runs (Qwen3-8B on vLLM 0.30, see README)
CAL = dict(f0=1 / 6000, att=0.0, t0=0.025, bw=600e9, jitter=0.05)
# stk_p constants (tuned in simulation in the Layby work); load_gate is not applied
STK = dict(alpha=0.1, c_tool=0.9, c_human=1.01, hmax=8.0, c_park=0.5, park_h=30.0, eta_q=0.9)
# the reference table: 11 (pool, n) pairs x disks 0.5, 1.5, 3 GB/s = 33 cells
REFERENCE = [("swechat", 6), ("swechat", 12), ("swechat", 18), ("tracelab_claude", 12), ("tracelab_claude", 18),
             ("tracelab_claude", 25), ("wildchat", 60), ("wildchat", 120), ("wildchat", 211),
             ("swechat+tracelab_claude+wildchat", 18), ("swechat+tracelab_claude+wildchat", 36)]
DISKS = (0.5, 1.5, 3.0)


def _oracle(options):
    o = Oracle(); o.options = options
    return o


# name -> factory(W, curve). C0 (None) is vLLM's default: LRU GPU cache, LRU CPU tier, no disk tier.
RULES = dict(
    C0=lambda W, c: None,
    wt=lambda W, c: WriteThroughDisk(W),
    stk=lambda W, c: StkRule(c, STK),
    cost=lambda W, c: CostRule(c),
    orc=lambda W, c: Oracle(),
    cont=lambda W, c: ContinuumTTL(W),
    cj=lambda W, c: ChoiJoshi(W),
    ka=lambda W, c: KeepAlive(W),
    c_nopark=lambda W, c: CostRule(c, ("none", "ins", "drop")),
    c_disk=lambda W, c: CostRule(c, ("ins", "park")),
    o_disk=lambda W, c: _oracle(("ins", "park")),
)


def register(spec):
    """NAME=module:Class adds a policy; the class is built as Class(W) (the workload) or Class()."""
    name, target = spec.split("=", 1)
    mod, cls = target.split(":")
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())          # the console script does not put the working directory on the path
    C = getattr(importlib.import_module(mod), cls)

    def make(W, c):
        try:
            return C(W)
        except TypeError:
            return C()
    RULES[name] = make
    return name


def pool_path(name):
    return name if os.path.sep in name or name.endswith(".json") else os.path.join(POOLS, name + ".json")


def workload(pool, n, w):
    rng = np.random.default_rng(w)
    parts = pool.split("+")
    ss = []
    for i, p in enumerate(parts):
        path = pool_path(p)
        S = json.load(open(path))["sessions"]
        k = n // len(parts) + (i < n % len(parts))
        ss += [dict(S[j], id=f"{os.path.basename(path)[:3]}{S[j]['id']}") for j in rng.choice(len(S), min(k, len(S)), replace=False)]
    for s in ss:
        s["start"] = float(rng.uniform(0, 600))
    return dict(gap_cap=300.0, ctx_cap=32768, sessions=ss)


def run(job):
    pool, n, w, seed, disk, name, curve, window, extra = job
    for spec in extra:                       # policies added with register() must exist in the worker too
        register(spec)
    W = workload(pool, n, w)
    rule = RULES[name](W, curve)
    rows, info = simulate(W, None if rule is None else "x", Params(seed=seed, disk_gbps=disk, window=window, **CAL), rule)
    s = summarize(rows)
    return dict(pool=pool, n=n, w=w, seed=seed, disk=disk, rule=name, p50=s["p50"], p95=s["p95"], p99=s["p99"], warms=info["warms"])


def evaluate(cells, rules, workloads=8, seeds=3, disks=DISKS, curve="v6", window=None, procs=8, extra=()):
    jobs = [(p, n, w, s, d, r, curve, window, tuple(extra)) for p, n in cells for d in disks
            for w in range(workloads) for s in range(seeds) for r in rules]
    with Pool(procs) as P:
        return P.map(run, jobs, chunksize=1)


def cell_ratios(rows):
    """{(pool, n, disk): {rule: (p50 median, p95 median, geomean p50 ratio, geomean p95 ratio)}}."""
    by = {}
    for x in rows:
        by.setdefault((x["pool"], x["n"], x["disk"], x["rule"], x["w"]), []).append(x)
    med = {k: (float(np.median([x["p50"] for x in v])), float(np.median([x["p95"] for x in v]))) for k, v in by.items()}
    cells = sorted({k[:3] for k in med})
    out = {}
    for c in cells:
        ws = sorted({k[4] for k in med if k[:3] == c and k[3] == "C0"})
        out[c] = {}
        for r in sorted({k[3] for k in med if k[:3] == c}):
            if not all(c + (r, w) in med for w in ws):
                continue
            v = np.array([med[c + (r, w)] for w in ws]); b = np.array([med[c + ("C0", w)] for w in ws])
            g50, g95 = (float(np.exp(np.mean(np.log(v[:, i] / b[:, i])))) for i in (0, 1))
            out[c][r] = (float(np.median(v[:, 0])), float(np.median(v[:, 1])), g50, g95)
    return out


def summary(rows, ref="wt"):
    """Geomean over cells of the per-cell p95 ratio vs C0, overall and by disk speed, and vs `ref`."""
    cr = cell_ratios(rows)
    rules = sorted({r for c in cr.values() for r in c}, key=lambda r: list(RULES).index(r) if r in RULES else 99)
    disks = sorted({c[2] for c in cr})
    gm = lambda xs: float(np.exp(np.mean(np.log(xs)))) if xs else float("nan")
    res = {}
    for r in rules:
        cells = [c for c in cr if r in cr[c]]
        e = dict(cells=len(cells), vs_C0=gm([cr[c][r][3] for c in cells]),
                 worst_vs_C0=max(cr[c][r][3] for c in cells),
                 by_disk={d: gm([cr[c][r][3] for c in cells if c[2] == d]) for d in disks})
        if ref in rules:
            both = [c for c in cells if ref in cr[c]]
            e["vs_" + ref] = gm([cr[c][r][3] / cr[c][ref][3] for c in both])
            e["by_disk_vs_" + ref] = {d: gm([cr[c][r][3] / cr[c][ref][3] for c in both if c[2] == d]) for d in disks}
        res[r] = e
    return cr, res
