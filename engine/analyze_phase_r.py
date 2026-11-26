#!/usr/bin/env python3
"""Phase R analysis (paper/BENCH_PLAN.md sections 1 and 9): per-run percentiles on the common served set,
within-host ratios, the geometric mean over hosts with a host-level t interval, a crossed bootstrap as a check,
and the pre-registered decision rules. It has no option to exclude sessions or turns: the plan forbids
post-hoc exclusion (excluded runs are simply not in the directory, and the exclusion is reported elsewhere).

Inputs: a directory of replay.py outputs named CELL__ARM__HOST.jsonl (double underscores; the parts must not
contain them). Other files are ignored and listed.

Per run (one cell, arm and host): rows of resumed turns (turn > 0) on the common served set: the (session, turn)
pairs every arm of that cell has a row for on that host. With run-to-completion every arm writes a row for every
turn (served, failed or censored), so the set is the whole workload unless a log lost rows; pairs missing from
any arm's log are counted per run. Failed and censored turns enter as +inf (worse than any served turn) and are
reported as counts. Metrics: TTFT p50, p95, p99; e2e p95; TPOT p50 and p95 ((e2e - ttft) / (out_tokens - 1) on
turns with at least two output tokens, on the pairs every arm of the host served, since a turn not served has no
decode); TTFT p50/p95 by gap bin (gap before the turn: < 10 s, 10 to 60 s, 60 to 300 s, > 300 s) and by wait kind
(the previous turn's kind); recomputed prompt tokens and bytes per tier from the /metrics delta, when present;
concurrency (peak and time-weighted mean); live-scoring checks (max |live - precomputed| over turns).

Pairs X:Y in a cell (Y may be "best", resolved per cell from --best CELL=ARM or the spec): per host, the ratio of
X's statistic to Y's; estimand = geometric mean over hosts; decision interval = host-level t interval on the log
ratios (df = n - 1), reported as one-sided bounds at --alpha (default 0.05: the upper bound is the "upper 95%
bound" of the plan; lower and upper together are the two-sided 90% interval TOST uses). A crossed bootstrap
(--boot B resamples, default 10,000: one resample of session ids applied to every host, then a resample of hosts)
gives the same bounds as a check for TTFT p95. Every pooled number is shown with its per-host values.

Decision rules (spec file, --spec): tests run in order:
  {"name": "H1", "rule": "iut", "cells": [...] | "all", "pairs": ["K:C0", "K:best"], "threshold": 1.0,
   "metric": "p95"}       holds if every upper bound (each cell x pair) is below the threshold
  {"name": "H5", "rule": "tost", "cells": "all", "pairs": ["K:C0"], "metric": "tpot_p50", "margin": 0.05}
                          holds if in every cell both bounds lie inside (1 - margin, 1 + margin)
  {"name": "H4a", "rule": "pooled", "cells": [...], "pair": "K:K_ewma", "threshold": 1.0, "metric": "p95"}
                          the geometric mean over the cells of the per-cell estimands (equal cell weights),
                          Welch-Satterthwaite df; holds if its upper bound is below the threshold
  {"name": "H5b", "rule": "throughput_tost", "cells": "all", "pair": "K:C0", "margin": 0.10}
                          open-loop sweeps (below): holds if in every cell the bounds of the peak sustainable rate
                          ratio lie inside (1 - margin, 1 + margin); the bounds are the wider of the host bootstrap's
                          and the host-level t interval's (lower = the smaller lower, upper = the larger upper)
  {"name": "H6", "rule": "kendall", "sim": {cell: {arm: predicted p95 ratio}} or a JSON path, "reference": "C0",
   "threshold": 0.6}      per cell, Kendall tau-b between the simulator's ratios and the engine's p95 ratios to the
                          reference, engine arms whose host-level bounds overlap counted as ties (the reference is
                          1 with no width); holds if the mean over cells reaches the threshold (an undefined tau,
                          one side all ties, counts as 0)
and "holm": [["H4a", "H4b"], ...] adjusts one-sided p-values (H0: ratio >= threshold; an IUT's p-value is the
largest of its parts) within each group by Holm at alpha. Spec keys "best" ({cell: arm}), "alpha" and "live_arms"
are optional.

Live scoring validity: a run whose hinted turns fell back to the precomputed curve (no side-file row, a failed or
late scoring call) on more than 1% of turns is invalid (plan section 1). A run of an arm listed in the spec's
"live_arms" (the arms that must score live, K) with no live-check rows at all is invalid too: its replay ran without
--side-file, so nothing shows the live curves were used. Invalid runs are left out of the common set and every pair,
and listed under invalid_runs with the reason.

Censored percentiles: a run whose statistic is +inf (enough failed or censored turns to reach that quantile) has no
finite ratio. When the comparator (Y) is the censored one the ratio is nan, not 0, and the pair is reported as
"comparator censored at p95" with its hosts (and "arm censored at p95" when X is), instead of "no interval".

Open-loop sweeps (H5 throughput half): runs named CELL__ARM__HOST__rate<R>.jsonl. Per (cell, arm, host): every
request's TTFT; a rate is sustainable when the served p95 is under the cell's SLO (spec "slo": {cell: seconds})
and at most 1% of the offered arrivals failed, were censored or were dropped (an arrival that found no session ready
to send, the summary's arrivals_dropped: offered load the server did not take); the peak sustainable rate is found by linear interpolation
between the last sustainable rate and the next one. Per pair: per-host ratios, geometric mean, a bootstrap over
hosts (the repeats) and the t bounds.

Run order (--order {host: ["cell/arm", ...]}): per pair, OLS of the per-host log p95 ratio on the position
difference pos(X) - pos(Y) across hosts, the slope with one-sided bounds at alpha (df = n - 2), as a sensitivity
analysis beside the primary estimate.

Usage: analyze_phase_r.py DIR [--pair K:C0 ...] [--best CELL=ARM ...] [--spec spec.json] [--alpha 0.05]
       [--boot 10000] [--seed 0] [--order order.json]
       [--out result.json]
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
from scipy import stats

INF = float("inf")
GAP_BINS = (("<10s", 0.0, 10.0), ("10-60s", 10.0, 60.0), ("60-300s", 60.0, 300.0), (">300s", 300.0, INF))
RATIO_METRICS = ("p50", "p95", "p99", "e2e_p95", "tpot_p50", "tpot_p95")


# --- quantiles with +inf ------------------------------------------------------------------------
def quantile(v, p):
    """np.quantile's linear method on values that may hold +inf: a point between a finite value and +inf is +inf
    (numpy would give nan). Empty input gives nan."""
    v = np.sort(np.asarray(v, float))
    if not len(v):
        return float("nan")
    pos = p * (len(v) - 1)
    k = int(math.floor(pos)); f = pos - k
    if f == 0 or k + 1 >= len(v):
        return float(v[k])
    lo, hi = v[k], v[k + 1]
    return INF if hi == INF else float(lo + (hi - lo) * f)


def weighted_quantiles(vals, idx, counts, p):
    """Quantile p of the multiset that repeats turn j counts[b, idx[j]] times, for every row b of counts, with the
    same linear rule as quantile(). vals must be sorted ascending (idx aligned). Returns shape (B,)."""
    w = counts[:, idx]                                    # (B, T) integer weights
    cw = np.cumsum(w, axis=1)
    N = cw[:, -1]
    out = np.full(len(counts), np.nan)
    ok = N > 0
    if not ok.any():
        return out
    cw, N = cw[ok], N[ok]
    pos = p * (N - 1)
    k = np.floor(pos).astype(np.int64); f = pos - k
    B, T = cw.shape
    big = int(N.max()) + 1
    flat = (cw + (np.arange(B, dtype=np.int64) * big)[:, None]).ravel()

    def elem(j):                                          # the j-th element (0-based) of each row's expansion
        q = np.searchsorted(flat, j + np.arange(B, dtype=np.int64) * big, side="right") - np.arange(B) * T
        return vals[np.minimum(q, T - 1)]

    lo, hi = elem(k), elem(np.minimum(k + 1, N - 1))
    with np.errstate(invalid="ignore"):
        res = np.where(f == 0, lo, np.where(hi == INF, INF, lo + (hi - lo) * f))
    out[ok] = res
    return out


# --- loading --------------------------------------------------------------------------------------
def metric_sum(lines, name):
    return sum(float(l.rsplit(" ", 1)[1]) for l in lines if l.startswith(name + "{") or l.startswith(name + " "))


def counters(lines):
    """{series (name with labels): value} for counter lines."""
    out = {}
    for l in lines:
        try:
            k, v = l.rsplit(" ", 1)
            out[k] = out.get(k, 0.0) + float(v)
        except ValueError:
            pass
    return out


def load_run(path):
    """One replay.py output: turn rows keyed by (session, turn), and the run's side records."""
    turns, mets, curves, series, summary = {}, {}, [], None, None
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        if "metrics" in r:
            mets[r["metrics"]] = r.get("lines") or []
        elif "curve" in r:
            curves.append(r)
        elif "concurrency_series" in r:
            series = r["concurrency_series"]
        elif "wall_s" in r:
            summary = r
        elif "session" in r and "turn" in r and "ttft" in r:
            key = (r["session"], int(r["turn"]))
            if key in turns:
                raise ValueError(f"{path}: two rows for {key}")
            status = "censored" if r.get("censored") else "failed" if r.get("failed") else "served"
            ttft = r["ttft"]
            if status == "served" and ttft is None:      # a pre-Phase R row with no token: failed
                status = "failed"
            ok = status == "served"
            turns[key] = dict(status=status, ttft=float(ttft) if ok else INF,
                              e2e=float(r["e2e"]) if ok and r.get("e2e") is not None else INF,
                              out=int(r.get("out_tokens") or 0), prev_gap=r.get("prev_gap"),
                              prev_kind=r.get("prev_kind"), conc=r.get("concurrency"))
    return dict(turns=turns, mets=mets, curves=curves, series=series, summary=summary)


def load_dir(d):
    """{cell: {host: {arm: run}}} from CELL__ARM__HOST.jsonl files, plus the names that did not parse."""
    runs, skipped = {}, []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        parts = os.path.basename(f)[:-6].split("__")
        if len(parts) == 4 and rate_of(parts[3]) is not None:
            continue                                      # an open-loop sweep run: load_sweeps
        if len(parts) != 3 or not all(parts):
            skipped.append(os.path.basename(f)); continue
        cell, arm, host = parts
        runs.setdefault(cell, {}).setdefault(host, {})[arm] = load_run(f)
    return runs, skipped


def rate_of(tag):
    if not tag.startswith("rate"):
        return None
    try:
        return float(tag[4:])
    except ValueError:
        return None


def load_sweeps(d):
    """{cell: {host: {arm: {rate: run}}}} from open-loop runs named CELL__ARM__HOST__rate<R>.jsonl."""
    sweeps = {}
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        parts = os.path.basename(f)[:-6].split("__")
        if len(parts) == 4 and all(parts) and rate_of(parts[3]) is not None:
            cell, arm, host, tag = parts
            sweeps.setdefault(cell, {}).setdefault(host, {}).setdefault(arm, {})[rate_of(tag)] = load_run(f)
    return sweeps


# --- per run --------------------------------------------------------------------------------------
def common_sets(host_runs):
    """(resumed pairs every arm has a row for, resumed pairs every arm served) on one host."""
    keys = None
    served = None
    for run in host_runs.values():
        k = {x for x in run["turns"] if x[1] > 0}
        s = {x for x in k if run["turns"][x]["status"] == "served"}
        keys = k if keys is None else keys & k
        served = s if served is None else served & s
    return sorted(keys or ()), sorted(served or ())


def gap_bin(g):
    if g is None:
        return None
    for name, lo, hi in GAP_BINS:
        if lo <= g < hi or (hi == 300.0 and g == 300.0):
            return name
    return None


def run_metrics(run, common, served_all):
    T = run["turns"]
    ttft = np.array([T[k]["ttft"] for k in common])
    e2e = np.array([T[k]["e2e"] for k in common])
    tp = [(T[k]["e2e"] - T[k]["ttft"]) / (T[k]["out"] - 1) for k in served_all if T[k]["out"] > 1]
    st = [T[k]["status"] for k in common]
    m = dict(n=len(common), served=st.count("served"), failed=st.count("failed"), censored=st.count("censored"),
             not_in_common=sum(1 for k in T if k[1] > 0) - len(common),
             p50=quantile(ttft, .5), p95=quantile(ttft, .95), p99=quantile(ttft, .99), e2e_p95=quantile(e2e, .95),
             tpot_n=len(tp), tpot_p50=quantile(tp, .5), tpot_p95=quantile(tp, .95))
    by_gap, by_kind = {}, {}
    for name, _, _ in GAP_BINS:
        v = [T[k]["ttft"] for k in common if gap_bin(T[k]["prev_gap"]) == name]
        by_gap[name] = dict(n=len(v), p50=quantile(v, .5), p95=quantile(v, .95))
    for kind in sorted({T[k]["prev_kind"] for k in common if T[k]["prev_kind"]}):
        v = [T[k]["ttft"] for k in common if T[k]["prev_kind"] == kind]
        by_kind[kind] = dict(n=len(v), p50=quantile(v, .5), p95=quantile(v, .95))
    m["by_gap"], m["by_kind"] = by_gap, by_kind
    m.update(tiers(run["mets"]))
    if run["series"]:
        s = np.array(run["series"], float)
        dt = np.diff(s[:, 0])
        m["concurrency"] = dict(peak=int(s[:, 1].max()),
                                mean=float((s[:-1, 1] * dt).sum() / dt.sum()) if dt.sum() > 0 else None)
    if run["curves"]:
        m["live"] = live_check(run["curves"])
    return m


LIVE_FLAG_MAX = 0.01      # plan section 1: more than 1% flagged turns (fallback curve used) invalidates a K run


def live_check(curves):
    """Live scoring: max |live - precomputed|, and the share of hinted turns that fell back to the precomputed
    curve (no side-file row, or a failed or late scoring call). Over LIVE_FLAG_MAX the run is invalid."""
    d = [c["max_abs_diff"] for c in curves if c.get("max_abs_diff") is not None]
    missing = sum(1 for c in curves if c.get("live_missing"))
    errors = sum(1 for c in curves if c.get("live_error"))
    frac = (missing + errors) / len(curves)
    return dict(hinted=len(curves), compared=len(d), max_abs_diff=max(d) if d else None, missing=missing,
                errors=errors, flagged_frac=frac, valid=frac <= LIVE_FLAG_MAX)


def tiers(mets):
    """Recomputed prompt tokens and bytes per tier from the /metrics delta, when both scrapes are present."""
    if "before" not in mets or "after" not in mets or not mets["after"]:
        return {}
    b, a = mets["before"], mets["after"]
    dl = lambda n: metric_sum(a, n) - metric_sum(b, n)
    out = {}
    if any(l.startswith("vllm:prefix_cache_queries") for l in a):
        out["recomputed_tokens"] = dl("vllm:prefix_cache_queries_total") - dl("vllm:prefix_cache_hits_total") \
            - dl("vllm:external_prefix_cache_hits_total")
    elif any(l.startswith("sglang:cached_tokens") for l in a):
        out["recomputed_tokens"] = dl("sglang:prompt_tokens_total") - dl("sglang:cached_tokens_total")
    cb, ca = counters(b), counters(a)
    by = {k: ca[k] - cb.get(k, 0.0) for k in ca if "bytes" in k.split("{")[0] and ca[k] - cb.get(k, 0.0) > 0}
    if by:
        out["bytes"] = by
    return out


# --- intervals ------------------------------------------------------------------------------------
def t_bounds(logs, alpha):
    """One-sided (1 - alpha) bounds on the mean log ratio, as ratios; None when n < 2 or a log is not finite."""
    logs = np.asarray(logs, float)
    n = len(logs)
    if n < 2 or not np.all(np.isfinite(logs)):
        return None
    mu, se = logs.mean(), logs.std(ddof=1) / math.sqrt(n)
    q = stats.t.ppf(1 - alpha, n - 1)
    return dict(n=n, gm=math.exp(mu), lower=math.exp(mu - q * se), upper=math.exp(mu + q * se),
                mean_log=mu, se_log=se, df=n - 1)


def p_below(b, threshold):
    """One-sided p-value of H0: ratio >= threshold given t_bounds output (small p: ratio below threshold)."""
    if b is None:
        return 1.0
    if b["se_log"] == 0:
        return 0.0 if b["mean_log"] < math.log(threshold) else 1.0
    return float(stats.t.cdf((b["mean_log"] - math.log(threshold)) / b["se_log"], b["df"]))


def pooled_bounds(cell_logs, alpha):
    """Geometric mean over cells of per-cell estimands (equal weights), Welch-Satterthwaite df."""
    cs = [np.asarray(l, float) for l in cell_logs]
    if not cs or any(len(c) < 2 or not np.all(np.isfinite(c)) for c in cs):
        return None
    k = len(cs)
    v = [c.var(ddof=1) / len(c) / k ** 2 for c in cs]
    mu = sum(c.mean() for c in cs) / k
    se = math.sqrt(sum(v))
    df = sum(v) ** 2 / sum(vi ** 2 / (len(c) - 1) for vi, c in zip(v, cs)) if se > 0 else sum(len(c) - 1 for c in cs)
    q = stats.t.ppf(1 - alpha, df)
    return dict(n=sum(len(c) for c in cs), cells=k, gm=math.exp(mu), lower=math.exp(mu - q * se),
                upper=math.exp(mu + q * se), mean_log=mu, se_log=se, df=df)


def crossed_bootstrap(hosts, alpha, B, seed, p=0.95, chunk=200):
    """hosts: list of (x_vals, x_sess, y_vals, y_sess) per host, on that host's common set, session ids as ints
    0..S-1. Each resample draws session ids with replacement once and applies the draw to every host, then draws
    hosts with replacement; the statistic is the geometric mean of the drawn hosts' quantile-p ratios."""
    rng = np.random.default_rng(seed)
    S = 1 + max(int(max(h[1].max(initial=-1), h[3].max(initial=-1))) for h in hosts)
    prepared = []
    for xv, xs, yv, ys in hosts:
        ox, oy = np.argsort(xv, kind="stable"), np.argsort(yv, kind="stable")
        prepared.append((xv[ox], xs[ox], yv[oy], ys[oy]))
    n = len(hosts)
    gms = []
    done = 0
    while done < B:
        b = min(chunk, B - done)
        counts = rng.multinomial(S, np.full(S, 1.0 / S), size=b)
        logr = np.empty((b, n))
        for j, (xv, xs, yv, ys) in enumerate(prepared):
            with np.errstate(divide="ignore", invalid="ignore"):
                logr[:, j] = np.log(weighted_quantiles(xv, xs, counts, p) / weighted_quantiles(yv, ys, counts, p))
        pick = rng.integers(0, n, size=(b, n))
        g = np.take_along_axis(logr, pick, axis=1).mean(axis=1)
        gms.append(g)
        done += b
    g = np.concatenate(gms)
    fin = g[np.isfinite(g)]
    if not len(fin):
        return dict(B=B, finite=0)
    return dict(B=B, finite=int(len(fin)), gm=float(math.exp(np.median(fin))),
                lower=float(math.exp(np.quantile(fin, alpha))), upper=float(math.exp(np.quantile(fin, 1 - alpha))))


# --- pairs and tests ------------------------------------------------------------------------------
def resolve(arm, cell, best):
    if arm == "best":
        if cell not in best:
            raise KeyError(f"no best arm given for cell {cell}")
        return best[cell]
    return arm


def drop_invalid(runs, live_arms=()):
    """Runs whose live scoring fell back on more than 1% of turns are invalid (plan section 1), and so are runs of a
    live arm (live_arms) with no live-check rows: they leave the analysis (common set and pairs) and are listed with
    the reason; the protocol reruns the host's block."""
    kept, invalid = {}, {}
    for cell, hosts in runs.items():
        for host, arms in hosts.items():
            for arm, run in arms.items():
                lc = live_check(run["curves"]) if run["curves"] else None
                if lc is not None and not lc["valid"]:
                    invalid[f"{cell}__{arm}__{host}"] = dict(lc, reason=f"live scoring fell back on "
                                                             f"{lc['flagged_frac']:.1%} of hinted turns (over "
                                                             f"{LIVE_FLAG_MAX:.0%})")
                elif lc is None and arm in live_arms:
                    invalid[f"{cell}__{arm}__{host}"] = dict(hinted=0, valid=False, reason="live arm with no live-check "
                                                             "rows (replay ran without --side-file)")
                else:
                    kept.setdefault(cell, {}).setdefault(host, {})[arm] = run
    return kept, invalid


def analyze(runs, pairs, best, alpha, boot, seed, live_arms=()):
    runs, invalid = drop_invalid(runs, live_arms)
    out = dict(runs={}, pairs={}, invalid_runs=invalid)
    per = {}                       # (cell, host, arm) -> metrics
    common = {}
    for cell, hosts in runs.items():
        for host, arms in hosts.items():
            keys, served = common_sets(arms)
            common[(cell, host)] = keys
            for arm, run in arms.items():
                m = run_metrics(run, keys, served)
                per[(cell, host, arm)] = m
                out["runs"][f"{cell}__{arm}__{host}"] = m
    for cell, hosts in runs.items():
        for spec in pairs:
            x, y = spec.split(":")
            try:
                X, Y = resolve(x, cell, best), resolve(y, cell, best)
            except KeyError:
                continue
            hs = sorted(h for h in hosts if X in hosts[h] and Y in hosts[h])
            if not hs:
                continue
            res = dict(x=X, y=Y, hosts=hs)
            for met in RATIO_METRICS:
                r, cens = [], dict(arm=[], comparator=[])
                for h in hs:
                    xv, yv = per[(cell, h, X)][met], per[(cell, h, Y)][met]
                    if yv == INF:                         # comparator censored: no finite ratio (x / inf is not 0)
                        cens["comparator"].append(h)
                        if xv == INF:
                            cens["arm"].append(h)
                        r.append(float("nan"))
                        continue
                    if xv == INF:
                        cens["arm"].append(h)
                    r.append(xv / yv if yv else float("nan"))
                with np.errstate(divide="ignore", invalid="ignore"):
                    logs = np.log(np.array(r, float))
                res[met] = dict(per_host=dict(zip(hs, r)), t=t_bounds(logs, alpha),
                                two_sided_contains_1=None, censored=cens)
                tb = res[met]["t"]
                if tb:
                    tb["p_below_1"] = p_below(tb, 1.0)
                    res[met]["two_sided_contains_1"] = tb["lower"] <= 1.0 <= tb["upper"]
            bins = {}
            for name in [g[0] for g in GAP_BINS] + sorted({k for h in hs for k in per[(cell, h, X)]["by_kind"]}):
                src = "by_gap" if name in dict((g[0], 1) for g in GAP_BINS) else "by_kind"
                r = []
                for h in hs:
                    a, b = per[(cell, h, X)][src].get(name), per[(cell, h, Y)][src].get(name)
                    if a and b and a["n"] and b["p95"] and math.isfinite(a["p95"]) and math.isfinite(b["p95"]):
                        r.append(a["p95"] / b["p95"])
                bins[name] = dict(p95_gm=float(math.exp(np.mean(np.log(r)))) if r else None, hosts=len(r))
            res["p95_by_bin"] = bins
            if boot:
                sid = {s: i for i, s in enumerate(sorted({k[0] for h in hs for k in common[(cell, h)]}))}
                hdat = []
                for h in hs:
                    ks = common[(cell, h)]
                    sx = np.array([sid[k[0]] for k in ks], np.int64)
                    hdat.append((np.array([runs[cell][h][X]["turns"][k]["ttft"] for k in ks]), sx,
                                 np.array([runs[cell][h][Y]["turns"][k]["ttft"] for k in ks]), sx))
                if all(len(d[0]) for d in hdat) and len(hs) >= 2:
                    res["p95"]["crossed_bootstrap"] = crossed_bootstrap(hdat, alpha, boot, seed)
            out["pairs"].setdefault(cell, {})[f"{x}:{y}"] = res
    return out


def iut(uppers, threshold):
    """Intersection-union: holds iff every upper bound exists and is below the threshold."""
    return bool(uppers) and all(u is not None and u < threshold for u in uppers)


def tost(lower, upper, margin):
    """Two one-sided tests on a ratio: equivalent iff (lower, upper) lies inside (1 - margin, 1 + margin)."""
    return lower is not None and upper is not None and 1 - margin < lower and upper < 1 + margin


def holm(pvals, alpha):
    """Holm step-down: {name: p} -> {name: (adjusted p, rejected)}."""
    names = sorted(pvals, key=lambda k: pvals[k])
    m, out, run = len(names), {}, 0.0
    stop = False
    for i, k in enumerate(names):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        stop = stop or pvals[k] > alpha / (m - i)
        out[k] = (run, not stop)
    return out


def load_sim(sim):
    return json.load(open(sim)) if isinstance(sim, str) else sim


def run_tests(spec, res, alpha):
    cells_all = sorted(res["pairs"])
    tests = {}
    for t in spec.get("tests", []):
        cells = cells_all if t.get("cells", "all") == "all" else t["cells"]
        met = t.get("metric", "p95")
        if t["rule"] in ("throughput_tost", "kendall"):
            continue                                      # handled in main (they need the sweeps / the simulator)
        if t["rule"] == "pooled":
            logs, used = [], []
            for c in cells:
                r = res["pairs"].get(c, {}).get(t["pair"])
                if r:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        logs.append(np.log(np.array(list(r[met]["per_host"].values()), float)))
                    used.append(c)
            b = pooled_bounds(logs, alpha) if len(used) == len(cells) else None
            tests[t["name"]] = dict(rule="pooled", metric=met, cells=cells, bounds=b,
                                    holds=b is not None and b["upper"] < t["threshold"],
                                    p=p_below(b, t["threshold"]))
            continue
        parts = []
        for c in cells:
            for p in t["pairs"]:
                r = res["pairs"].get(c, {}).get(p)
                b = r[met]["t"] if r else None
                parts.append(dict(cell=c, pair=p, lower=b and b["lower"], upper=b and b["upper"],
                                  gm=b and b["gm"], n=b and b["n"],
                                  p=p_below(b, t["threshold"]) if t["rule"] == "iut" else None))
        if t["rule"] == "iut":
            tests[t["name"]] = dict(rule="iut", metric=met, threshold=t["threshold"], parts=parts,
                                    holds=iut([x["upper"] for x in parts], t["threshold"]),
                                    p=max((x["p"] for x in parts), default=1.0))
        elif t["rule"] == "tost":
            tests[t["name"]] = dict(rule="tost", metric=met, margin=t["margin"], parts=parts,
                                    holds=bool(parts) and all(tost(x["lower"], x["upper"], t["margin"]) for x in parts))
        else:
            raise ValueError(f"unknown rule {t['rule']}")
    for t in spec.get("tests", []):
        if t["rule"] == "kendall":
            sim = load_sim(t["sim"])
            cells = sorted(sim) if t.get("cells", "all") == "all" else t["cells"]
            tests[t["name"]] = h6(res, sim, t.get("reference", "C0"), cells, t["threshold"])
        elif t["rule"] == "throughput_tost":
            tp = res.get("throughput", {}).get("pairs", {})
            cells = sorted(tp) if t.get("cells", "all") == "all" else t["cells"]
            parts = []
            for c in cells:
                pr = tp.get(c, {}).get(t["pair"]) or {}
                b, tb = pr.get("bootstrap"), pr.get("t")
                have = [x for x in (b, tb) if x]
                # the wider of the two: the host bootstrap is narrow with few hosts, the t interval is the decision
                # interval of every other rule; TOST must hold under both
                parts.append(dict(cell=c, lower=min((x["lower"] for x in have), default=None),
                                  upper=max((x["upper"] for x in have), default=None), gm=pr.get("gm"),
                                  boot=b and [b["lower"], b["upper"]], t=tb and [tb["lower"], tb["upper"]]))
            tests[t["name"]] = dict(rule="throughput_tost", pair=t["pair"], margin=t["margin"], parts=parts,
                                    holds=bool(parts) and all(tost(x["lower"], x["upper"], t["margin"]) for x in parts))
    for group in spec.get("holm", []):
        adj = holm({g: tests[g]["p"] for g in group}, alpha)
        for g, (pa, rej) in adj.items():
            tests[g]["holm"] = dict(group=group, p_adj=pa, holds=rej)
    return tests


# --- H5 throughput half: open-loop rate sweeps -------------------------------------------------------
BAD_MAX = 0.01            # a rate is sustainable only with at most 1% failed or censored requests


def sweep_point(run):
    """One open-loop run at a fixed rate: p95 TTFT over served requests (every row: open-loop requests do not
    resume a session) and the share of offered arrivals that were not served: failed or censored requests plus the
    arrivals the replay dropped (no session ready), over requests sent plus dropped."""
    T = list(run["turns"].values())
    served = [t["ttft"] for t in T if t["status"] == "served"]
    dropped = int((run.get("summary") or {}).get("arrivals_dropped") or 0)
    offered = len(T) + dropped
    return dict(n=len(T), dropped=dropped, p95=quantile(served, .95),
                bad_frac=(offered - len(served)) / offered if offered else 1.0)


def peak_rate(points, slo):
    """points {rate: sweep_point}. Excess g(rate) = max(p95 / slo, bad_frac / BAD_MAX); sustainable when g <= 1.
    Going up from the lowest rate, the peak is where g crosses 1 between the last sustainable rate and the next
    one, by linear interpolation in g (a later rate that passes again does not count). Every rate sustainable:
    the top rate, flagged above_grid (a lower bound). The lowest rate not sustainable: None, below_grid."""
    rates = sorted(points)
    g = []
    for r in rates:
        p = points[r]
        x = max(p["p95"] / slo, p["bad_frac"] / BAD_MAX)
        g.append(INF if math.isnan(x) else x)
    out = dict(rates=rates, excess=g)
    if g[0] > 1:
        return dict(out, peak=None, below_grid=True)
    i = 0
    while i + 1 < len(rates) and g[i + 1] <= 1:
        i += 1
    if i + 1 == len(rates):
        return dict(out, peak=rates[-1], above_grid=True)
    r0, r1, g0, g1 = rates[i], rates[i + 1], g[i], g[i + 1]
    peak = r0 if g1 == INF else r0 + (1 - g0) / (g1 - g0) * (r1 - r0)
    return dict(out, peak=peak)


def host_bootstrap(logs, alpha, B, seed):
    """Resample hosts (the repeats) with replacement; percentile bounds of the geometric mean."""
    logs = np.asarray(logs, float)
    if len(logs) < 2 or not np.all(np.isfinite(logs)):
        return None
    rng = np.random.default_rng(seed)
    g = logs[rng.integers(0, len(logs), size=(B, len(logs)))].mean(axis=1)
    return dict(B=B, n=len(logs), gm=math.exp(logs.mean()), lower=math.exp(np.quantile(g, alpha)),
                upper=math.exp(np.quantile(g, 1 - alpha)))


def throughput(sweeps, pairs, slo, alpha, B, seed):
    """Peak sustainable rate per (cell, arm, host) and, per pair X:Y, the per-host ratios, their geometric mean, a
    bootstrap over hosts and the host-level t bounds."""
    out = dict(peaks={}, pairs={})
    for cell, hosts in sweeps.items():
        if cell not in slo:
            continue
        for host, arms in hosts.items():
            for arm, pts in arms.items():
                pk = peak_rate({r: sweep_point(run) for r, run in pts.items()}, slo[cell])
                out["peaks"][f"{cell}__{arm}__{host}"] = pk
        for spec in pairs:
            x, y = spec.split(":")
            hs = sorted(h for h in hosts if x in hosts[h] and y in hosts[h])
            if not hs:
                continue
            pk = {h: (out["peaks"][f"{cell}__{x}__{h}"]["peak"], out["peaks"][f"{cell}__{y}__{h}"]["peak"]) for h in hs}
            r = {h: (a / b if a is not None and b else None) for h, (a, b) in pk.items()}
            ok = all(v is not None and v > 0 for v in r.values())
            logs = [math.log(v) for v in r.values()] if ok else []
            out["pairs"].setdefault(cell, {})[spec] = dict(
                per_host=r, slo=slo[cell], gm=math.exp(np.mean(logs)) if ok else None,
                bootstrap=host_bootstrap(logs, alpha, B, seed) if ok else None,
                t=t_bounds(logs, alpha) if ok else None,
                above_grid=[h for h in hs for a in (x, y) if out["peaks"][f"{cell}__{a}__{h}"].get("above_grid")])
    return out


# --- H6: simulator fidelity ---------------------------------------------------------------------
def kendall_tau_b(sim, est, lo, hi):
    """Kendall tau-b between the simulator's values and the engine's, where two engine arms whose intervals
    [lo, hi] overlap are tied (and two equal simulator values are tied). None when either side is all ties."""
    arms = sorted(sim)
    nc = nd = t1 = t2 = n0 = 0
    for i in range(len(arms)):
        for j in range(i + 1, len(arms)):
            a, b = arms[i], arms[j]
            n0 += 1
            ss = np.sign(sim[a] - sim[b])
            es = 0 if (lo[a] <= hi[b] and lo[b] <= hi[a]) else np.sign(est[a] - est[b])
            t1 += ss == 0; t2 += es == 0
            nc += ss * es > 0; nd += ss * es < 0
    den = math.sqrt((n0 - t1) * (n0 - t2))
    return (nc - nd) / den if den else None


def h6(res, sim, ref, cells, threshold):
    """Per cell: the engine's p95 ratio of every simulator arm to the reference arm, with its host-level bounds
    (the reference arm is 1 with no width), against the simulator's predicted ratios. An undefined tau counts as 0
    toward the mean (no agreement shown)."""
    per = {}
    for c in cells:
        est, lo, hi, missing = {}, {}, {}, []
        for arm in sim.get(c, {}):
            if arm == ref:
                est[arm] = lo[arm] = hi[arm] = 1.0; continue
            r = res["pairs"].get(c, {}).get(f"{arm}:{ref}")
            t = r and r["p95"]["t"]
            if not t:
                missing.append(arm); continue
            est[arm], lo[arm], hi[arm] = t["gm"], t["lower"], t["upper"]
        sv = {a: sim[c][a] for a in est}
        tau = kendall_tau_b(sv, est, lo, hi) if len(est) >= 2 else None
        per[c] = dict(tau=tau, arms=sorted(est), missing=missing, engine=est, engine_lower=lo, engine_upper=hi)
    taus = [0.0 if v["tau"] is None else v["tau"] for v in per.values()]
    mean = float(np.mean(taus)) if taus else None
    return dict(rule="kendall", per_cell=per, mean=mean, threshold=threshold,
                undefined=[c for c, v in per.items() if v["tau"] is None],
                holds=mean is not None and mean >= threshold)


# --- run-order covariate ----------------------------------------------------------------------------
def order_sensitivity(res, order, alpha):
    """order {host: ["cell/arm", ...]} in run order. Per pair X:Y: regress the per-host log p95 ratio on the
    position difference pos(X) - pos(Y) across hosts (OLS with intercept, df = n - 2). Reports the slope (log ratio
    per position), its one-sided bounds at alpha, and the intercept as a ratio (the estimate at no difference)."""
    pos = {h: {e: i for i, e in enumerate(lst)} for h, lst in order.items()}
    out = {}
    for cell, ps in res["pairs"].items():
        for name, r in ps.items():
            X, Y = r["x"], r["y"]
            hs = [h for h in r["p95"]["per_host"] if h in pos and f"{cell}/{X}" in pos[h] and f"{cell}/{Y}" in pos[h]]
            d = np.array([pos[h][f"{cell}/{X}"] - pos[h][f"{cell}/{Y}"] for h in hs], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log(np.array([r["p95"]["per_host"][h] for h in hs], float))
            rec = dict(hosts=hs, position_diff=d.tolist(), n=len(hs))
            sxx = ((d - d.mean()) ** 2).sum() if len(d) else 0.0
            if len(hs) >= 3 and sxx > 0 and np.all(np.isfinite(y)):
                b = ((d - d.mean()) * (y - y.mean())).sum() / sxx
                a0 = y.mean() - b * d.mean()
                s2 = ((y - a0 - b * d) ** 2).sum() / (len(hs) - 2)
                se = math.sqrt(s2 / sxx)
                q = stats.t.ppf(1 - alpha, len(hs) - 2)
                rec.update(slope=b, lower=b - q * se, upper=b + q * se, se=se, df=len(hs) - 2,
                           per_position=math.exp(b), intercept_ratio=math.exp(a0))
            else:
                rec["slope"] = None
            out.setdefault(cell, {})[name] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("d")
    ap.add_argument("--pair", action="append", default=[], help="X:Y, Y may be 'best'")
    ap.add_argument("--best", action="append", default=[], help="CELL=ARM: the tuned best baseline of a cell")
    ap.add_argument("--spec", help="JSON with tests, holm groups, best and alpha")
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", help="JSON {host: [\"cell/arm\", ...]} in run order: adds the position sensitivity")
    ap.add_argument("--live-arm", action="append", default=[], help="an arm that must score live (adds to the spec's "
                    "live_arms): a run of it with no live-check rows is invalid")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    spec = json.load(open(a.spec)) if a.spec else {}
    alpha = a.alpha if a.alpha is not None else spec.get("alpha", 0.05)
    best = dict(spec.get("best", {}))
    best.update(dict(x.split("=", 1) for x in a.best))
    tests = spec.get("tests", [])
    pairs = [p for t in tests if t["rule"] not in ("throughput_tost", "kendall") for p in t.get("pairs", [t.get("pair")]) if p]
    for t in tests:                                       # H6 needs every simulator arm against the reference
        if t["rule"] == "kendall":
            sim = load_sim(t["sim"])
            pairs += [f"{arm}:{t.get('reference', 'C0')}" for c in sim for arm in sim[c] if arm != t.get("reference", "C0")]
    pairs = list(dict.fromkeys(a.pair + pairs))
    tp_pairs = list(dict.fromkeys(t["pair"] for t in tests if t["rule"] == "throughput_tost"))
    live_arms = list(dict.fromkeys(spec.get("live_arms", []) + a.live_arm))
    runs, skipped = load_dir(a.d)
    if skipped:
        print("ignored (not CELL__ARM__HOST.jsonl):", ", ".join(skipped), file=sys.stderr)
    res = analyze(runs, pairs, best, alpha, a.boot, a.seed, live_arms)
    res.update(alpha=alpha, best=best, ignored=skipped, live_arms=live_arms)
    sweeps = load_sweeps(a.d)
    if sweeps:
        res["throughput"] = throughput(sweeps, tp_pairs, spec.get("slo", {}), alpha, a.boot, a.seed)
    if a.order:
        res["order_sensitivity"] = order_sensitivity(res, json.load(open(a.order)), alpha)
    for name, why in res["invalid_runs"].items():
        print(f"INVALID ({why['reason']}): {name}", file=sys.stderr)
    if spec.get("tests"):
        res["tests"] = run_tests(spec, res, alpha)
    for cell, ps in sorted(res["pairs"].items()):
        for name, r in ps.items():
            t = r["p95"]["t"]
            hosts = " ".join(f"{h}={v:.3f}" for h, v in r["p95"]["per_host"].items())
            cens = r["p95"]["censored"]
            why = [f"{who} censored at p95 on {','.join(cens[k])}" for who, k in (("comparator", "comparator"),
                                                                                    ("arm", "arm")) if cens[k]]
            iv = f"gm {t['gm']:.3f} [{t['lower']:.3f}, {t['upper']:.3f}]" if t else "; ".join(why) or "no interval"
            bs = r["p95"].get("crossed_bootstrap")
            bss = f" | boot [{bs['lower']:.3f}, {bs['upper']:.3f}]" if bs and bs.get("finite") else ""
            print(f"{cell} {name} ({r['x']}:{r['y']}) p95 {iv}{bss} | {hosts}")
    for name, t in res.get("tests", {}).items():
        h = t.get("holm")
        print(f"{name}: {'holds' if t['holds'] else 'does not hold'}" + (f" (Holm: {'holds' if h['holds'] else 'does not hold'}, p_adj {h['p_adj']:.4f})" if h else ""))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1, default=float); f.write("\n")
    return res


if __name__ == "__main__":
    main()
