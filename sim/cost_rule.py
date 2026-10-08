#!/usr/bin/env python3
"""Decision rules for engine_sim.simulate(rule=...), driven by a per-turn survival curve.

A replay turn carries surv_<name>: P(T > E2[j]) for the idle period after it (T = seconds to the
session's next request), from a model such as Laya v6. The simulator calls rule.decide(view) when a
turn finishes; view holds the turn, its KV size in tokens, the time, the engine parameters and the
live engine state (engine_sim.Live). decide returns eta (park_eta for the CPU tier; -1 = first out of the CPU tier), disk (write the
session to disk), pre (seconds after the turn ends to fire a 1-token warm, or None) and warm_eta
(park_eta the warm carries).

StkRule  a threshold rule on the curve (stk_p) with fitted constants alpha, c_tool, c_human, hmax, c_park,
         park_h, eta_q, and a fixed lead of KV bytes / 20 GB/s + 0.3 s.
CostRule no fitted constants. Every term is a probability from the curve times a cost the engine
         measures while it runs (engine_sim.Live):
           g(a), c(a)     P(idle KV is still on the GPU / in the CPU tier a seconds after its last use),
                          a token-weighted Kaplan-Meier curve over recent idle periods;
           k_c, k_d, k_r  cost of restoring the session's KV from the CPU tier, from disk (via CPU), or
                          by recompute: its own time (with the link's mean queue wait) plus the time it
                          delays others: a transfer of x delays the about lam * (wait + x) transfers
                          that arrive behind it by x, and its N allocated GPU tokens slow admission by
                          N / C for x, delaying each waiting request as much; a recompute of x delays
                          the requests in prefill and those admitted while it runs by x;
           price          holding n more tokens on a tier that evicts e tokens/s moves every eviction
                          n/e seconds earlier: e * lam(n/e) seconds per second, lam(x) the measured
                          loss per evicted token from returns within x seconds of the eviction. A copy
                          in an LRU tier is gone once about C tokens newer than it came in, so it pays
                          the price for at most C / e seconds.
         Placement: the cheapest in expectation of none (CPU copy only), ins (also write to disk, keep
         the CPU hint), park (write to disk, first out of the CPU tier) and drop (first out of the CPU
         tier, no disk write). A first-out copy stays until the tier's next eviction. Requests wait Q
         (the mean admission wait) before the engine looks up their KV, so the return at T costs
         (1 - g(T + Q)) times the restore from wherever the KV is then: the CPU tier, disk (ins, park)
         or the disk copy the session already has plus recompute (none, drop).
         eta = the median return (park and drop: -1). No warms: on held-out populations in engine_sim
         they lost or tied even with this cost model, and helped only light chat load under perfect
         prediction.
Oracle   CostRule fed a step curve at the true gap (perfect prediction, same costs).
"""
import math
import numpy as np

from .rule import E2, TG, decide, interp_surv

KIB = 144


def curve_of(t, name):
    """The turn's survival curve at E2; a turn the model did not score gets 0.5 everywhere (as the replay builder)."""
    c = t.get(f"surv_{name}")
    return np.full(len(E2), 0.5) if c is None else np.nan_to_num(np.array(c, float), nan=0.5)


class StkRule:
    def __init__(self, name, r, reload_gbps=20.0, gap_cap=300.0):
        self.name, self.r, self.bw, self.gap_cap = name, r, reload_gbps, gap_cap

    def decide(self, v):
        t, r = v["turn"], self.r
        P = curve_of(t, self.name)
        S = interp_surv(P, TG)
        q = lambda level: TG[np.argmax(S <= level)] if (S <= level).any() else math.inf
        lead = t["ctx"] * KIB * 1024 / (self.bw * 1e9) + 0.3
        qa, q50, qe = q(1 - r["alpha"]), q(0.5), q(1 - r["eta_q"])
        park = interp_surv(P, r["park_h"])[0] >= r["c_park"]
        eta = -1.0 if park else min(qe, self.gap_cap)
        c = r["c_human"] if t["kind"] == "human" else r["c_tool"]
        fire = interp_surv(P, lead + 1.0)[0] >= c and math.isfinite(qa) and qa > lead + 1.0 and q50 - qa <= r["hmax"]
        pre = qa - lead if fire else None
        we = None if pre is None else (-1.0 if eta < 0 else max(eta - pre, 0.5))
        return dict(eta=eta, disk=None, pre=pre, warm_eta=we)


class CostRule:
    """sim.rule.decide driven from the simulator view (see sim/rule.py for the rule)."""
    def __init__(self, name, options=None):
        self.name = name
        self.options = options
        self.log = []

    def curve(self, t):
        return interp_surv(curve_of(t, self.name), TG)

    def decide(self, v):
        t, p = v["turn"], v["p"]
        p.bytes_per_token = KIB * 1024
        d = decide(self.curve(t), v["tokens"], v["on_disk"], v["idle_g"], v["idle_c"], v["live"], p, v["now"],
                   **({"options": self.options} if self.options else {}))
        self.log.append(dict(opt=d["opt"], Q=d["Q"], pc=d["price_c"], wd=d["wait_disk"], npf=d["npf"], N=v["tokens"],
                             kind=t["kind"], gap=t["gap_after"]))
        return dict(eta=d["eta"], disk=d["disk"], pre=None, warm_eta=None)


class Oracle(CostRule):
    """CostRule with perfect prediction: a step curve at the replayed gap."""
    def __init__(self):
        super().__init__("oracle")

    def curve(self, t):
        g = t["gap_after"]
        return np.where(TG < g, 1.0, 0.0) if g is not None else np.ones(len(TG))
