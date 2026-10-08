"""Compare quant_bench precisions against bf16: curve differences and cost-rule decisions.

Usage: python -m layby_dwell.quant_report OUT.npz

Curves: max, p99 and mean |P(T > h) - P_bf16(T > h)| over all states and horizons. Decisions: the cost rule
(sim.rule.decide) on every state's curve, at session sizes of 16k and 32k tokens (bf16 picks none, park and
drop there; at 4k every curve gives drop), against one fixed engine
state built here (a loaded engine with a busy CPU tier, a contended disk and both residency curves populated),
and the share of (state, size) pairs where a precision picks a different option than bf16.
"""
import json
import sys

import numpy as np

from sim.live import AG, EngineParams, Live
from sim.rule import TG, decide, interp_surv

SIZES = (16384, 32768)     # at 4k tokens every curve gives drop in this engine state, so it is left out


def engine():
    """A fixed, loaded engine: 120 s of decayed history with CPU and disk traffic, admissions, waits and
    evictions on both tiers at a range of idle ages."""
    L = Live(120.0)
    L.tick(0.0)
    rng = np.random.default_rng(7)
    t = 0.0
    for _ in range(600):
        t += 0.2
        L.add(t, n_cpu=1.0, wait_cpu=0.05, n_disk=0.3, wait_disk=0.8, n_adm=1.0, qwait=0.6, pf=0.4, nw=0.6,
              busy_cpu=0.02, busy_disk=0.06)
        a = float(rng.exponential(40.0))
        L.end_idle(t, "g", a, 16, bool(rng.random() < 0.6))
        L.end_idle(t, "c", a * 3, 16, bool(rng.random() < 0.4))
        if rng.random() < 0.2:
            L.miss(t, "c", float(rng.exponential(60.0)), float(rng.exponential(0.5)))
    p = EngineParams(bytes_per_token=147456, gpu_tokens=116480, cpu_tokens=203904, cpu_gbps=20.0, disk_gbps=1.5,
                     f0=1 / 6000, t0=0.025)
    return L, p, t


def main():
    z = np.load(sys.argv[1])
    summ = json.loads(str(z["summary"]))
    ref = z["bf16"]
    L, p, now = engine()
    idle = (np.zeros(0), np.zeros(0))

    def opts(S):
        out = []
        for row in S:
            if np.isnan(row).any():
                out.extend([None] * len(SIZES)); continue
            s = interp_surv(row, TG)
            for n in SIZES:
                out.append(decide(s, n, 0, idle, idle, L, p, now)["opt"])
        return out

    base = opts(ref)
    import collections
    mix = collections.Counter(base)
    print(f"GPU {summ['gpu']}, torch {summ['torch']}, {summ['n']} public states, reference bf16; bf16 decisions {dict(mix)}")
    print("precision | max |dP| | p99 |dP| | mean |dP| | decisions same | latency ms | states/s b1 b8 b32 b128 | peak GB")
    res = {}
    for prec, info in summ["precisions"].items():
        if "error" in info:
            print(f"{prec} | failed: {info['error'][:120]}"); res[prec] = info; continue
        S = z[prec]
        ok = ~(np.isnan(S).any(1) | np.isnan(ref).any(1))
        d = np.abs(S[ok] - ref[ok])
        o = opts(S)
        same = float(np.mean([a == b for a, b in zip(o, base) if a is not None and b is not None]))
        tp = info["states_per_s"]
        res[prec] = dict(info, max=float(d.max()), p99=float(np.quantile(d, 0.99)), mean=float(d.mean()), same=same)
        print(f"{prec} | {d.max():.4f} | {np.quantile(d, 0.99):.4f} | {d.mean():.5f} | {100 * same:.2f}% | "
              f"{info['latency_ms']} | {tp['1']} {tp['8']} {tp['32']} {tp['128']} | {info['peak_gb']}")
    print(json.dumps(dict(gpu=summ["gpu"], torch=summ["torch"], n=summ["n"], results=res)))


if __name__ == "__main__":
    main()
