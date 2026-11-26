"""Return-time curves with no learned model, for the controls K_km and K_ewma.

Both feed the unchanged cost rule (layby.rule.decide) through the path every curve takes: a 15-value curve
P(T > E2[j]) per turn, stored in the workload as surv_NAME and sent by the client when the turn ends (engine/replay.py
--park-hints NAME posts it to the adapter's hint port, the same way as --park-hints v6), or read by the simulator from
the same field. Only the curve's source differs from K; the adapter and the rule do not change.

K_km   the per-kind Kaplan-Meier curve: one curve per wait kind (the kind of the turn that ends: "tool" when the
       response ends in a tool call, "human" otherwise), fit on sessions outside the evaluated ones (fit_km, saved as
       JSON {"tool": [15], "human": [15], ..., "all": [15]}). Every turn of a kind gets the same curve; a kind the
       file lacks gets "all".
K_ewma a per-session curve from the session's own past waits of the same kind: at the end of turn i, the waits after
       its earlier turns j < i whose kind equals turn i's kind (all known by then: the wait after turn j ended when
       turn j + 1 arrived), g_1..g_m in order. With m >= 2, the curve is exponential with the exponentially weighted
       mean of those waits, m_i = EWMA_alpha(g_1..g_m) (m_1 = g_1, then m <- alpha g + (1 - alpha) m), so
       S(t) = exp(-t / m_i). With m < 2, the turn gets the K_km curve of its kind.

A curve is P(T > t) at the horizons E2 (layby.rule.E2), T the seconds from the turn's end to the session's next
request. Gaps as stored in the workload (the engine workloads cap them at the replay cap, a longer wait being a
return at the cap; pass --cap to fit on the same capped scale). A fitting turn with "returned": false is a wait that
never ended in a return (the session's last boundary): KM censors it at its gap_after.

CLI:
  python -m layby.curves fit-km SRC.json [SRC2.json ...] --out km.json [--cap 300]
  python -m layby.curves annotate W.json --km km.json --out W2.json [--alpha 0.5]
SRC and W are pools or replay workloads ({"sessions": [{"turns": [{"kind", "gap_after", ...}]}]}). annotate adds
surv_km and surv_ewma to every turn that has a gap (the turns a client posts a curve for) and leaves the rest of the
file as it was.
"""
import argparse
import json
import math

import numpy as np

from layby.rule import E2

ALPHA = 0.5


def km(gaps, events=None, H=E2):
    """Kaplan-Meier P(T > H[j]) from waits (events: 1 a return at that time, 0 censored there; default all returns).
    With no censoring it is the empirical fraction of waits longer than H[j]."""
    g = np.asarray(gaps, float)
    if len(g) == 0:
        raise ValueError("km of no waits")
    e = np.ones(len(g), bool) if events is None else np.asarray(events, bool)
    order = np.argsort(g, kind="stable")
    g, e = g[order], e[order]
    ut, first = np.unique(g, return_index=True)       # distinct times, the first sorted index of each
    at_risk = len(g) - first                           # waits >= t
    d = np.add.reduceat(e.astype(float), first)        # returns at t
    S = np.cumprod(1.0 - d / at_risk)
    k = np.searchsorted(ut, np.asarray(H, float), side="right") - 1   # last distinct time <= h
    out = np.where(k >= 0, S[np.maximum(k, 0)], 1.0)
    return out


def _turns(sessions, cap=None):
    for s in sessions:
        for t in s["turns"]:
            if t.get("gap_after") is None:
                continue
            g = float(t["gap_after"])
            yield t.get("kind") or "?", (min(g, cap) if cap is not None else g), t.get("returned", True) is not False


def fit_km(sessions, cap=None):
    """{"<kind>": curve, ..., "all": curve} from every turn with a wait (gap_after not None); a turn with
    "returned": false is censored at its wait. _n counts waits, _censored the censored ones."""
    by, ev = {}, {}
    allg, alle = [], []
    for kind, g, e in _turns(sessions, cap):
        by.setdefault(kind, []).append(g); ev.setdefault(kind, []).append(e)
        allg.append(g); alle.append(e)
    out = {k: km(v, ev[k]).tolist() for k, v in sorted(by.items())}
    out["all"] = km(allg, alle).tolist()
    out["_n"] = {k: len(v) for k, v in sorted(by.items())} | {"all": len(allg)}
    if not all(alle):
        out["_censored"] = {k: len(v) - sum(v) for k, v in sorted(ev.items())} | {"all": len(alle) - sum(alle)}
    return out


def km_of(KM, kind):
    return list(KM.get(kind) or KM["all"])


def ewma(gaps, alpha=ALPHA):
    m = float(gaps[0])
    for g in gaps[1:]:
        m = alpha * float(g) + (1.0 - alpha) * m
    return m


def exp_curve(mean, H=E2):
    m = max(float(mean), 1e-3)
    return np.exp(-np.asarray(H, float) / m)


def session_curves(turns, KM, alpha=ALPHA, min_gaps=2):
    """K_ewma and K_km curves for each turn of one session, in order (None for a turn with no wait after it)."""
    seen = {}            # kind -> waits after earlier turns of that kind
    out = []
    for t in turns:
        kind = t.get("kind") or "?"
        prior = seen.get(kind, [])
        c_km = km_of(KM, kind)
        if t.get("gap_after") is None:
            out.append((None, None))
        elif len(prior) >= min_gaps:
            out.append((exp_curve(ewma(prior, alpha)).tolist(), c_km))
        else:
            out.append((c_km, c_km))
        if t.get("gap_after") is not None:
            seen.setdefault(kind, []).append(float(t["gap_after"]))
    return out


def annotate(W, KM, alpha=ALPHA):
    """Add surv_ewma and surv_km to every turn of W with a wait. Returns counts (turns, ewma from own waits)."""
    n = own = 0
    for s in W["sessions"]:
        for t, (ce, ck) in zip(s["turns"], session_curves(s["turns"], KM, alpha)):
            if ce is None:
                continue
            t["surv_ewma"], t["surv_km"] = [float(x) for x in ce], [float(x) for x in ck]
            n += 1
            own += ce is not ck
    return dict(turns=n, ewma_own=own, alpha=alpha)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="layby.curves", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit-km", help="fit the per-kind KM curves")
    f.add_argument("src", nargs="+"); f.add_argument("--out", required=True)
    f.add_argument("--cap", type=float, help="cap waits at this many seconds (a longer wait is a return at the cap)")
    a = sub.add_parser("annotate", help="add surv_km and surv_ewma to a workload or pool")
    a.add_argument("workload"); a.add_argument("--km", required=True); a.add_argument("--out", required=True)
    a.add_argument("--alpha", type=float, default=ALPHA)
    x = ap.parse_args(argv)
    if x.cmd == "fit-km":
        S = [s for p in x.src for s in json.load(open(p))["sessions"]]
        K = fit_km(S, x.cap)
        K["_src"] = x.src
        K["_cap"] = x.cap
        json.dump(K, open(x.out, "w"), indent=1)
        print(f"{x.out}: {K['_n']}")
    else:
        if not 0.0 < x.alpha <= 1.0 or math.isnan(x.alpha):
            ap.error("--alpha must be in (0, 1]")
        W = json.load(open(x.workload))
        r = annotate(W, json.load(open(x.km)), x.alpha)
        json.dump(W, open(x.out, "w"))
        print(f"{x.out}: {r['turns']} turns with curves, {r['ewma_own']} from the session's own waits (alpha {x.alpha})")


if __name__ == "__main__":
    main()
