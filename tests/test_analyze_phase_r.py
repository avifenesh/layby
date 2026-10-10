"""CPU tests of engine/analyze_phase_r.py: quantiles with +inf, censored and failed turns, the common served set,
the decision rules (IUT, TOST, Holm), and the estimator and its intervals on synthetic runs with a known effect.

  python -m pytest -q tests/test_analyze_phase_r.py      (needs numpy and scipy; a few seconds)
"""
import importlib.util
import json
import math
import os

import numpy as np
import pytest

pytest.importorskip("scipy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("analyze_phase_r", os.path.join(ROOT, "engine", "analyze_phase_r.py"))
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)
INF = float("inf")


def row(s, t, ttft=None, status="served", out=4, prev_gap=5.0, prev_kind="tool"):
    r = dict(session=s, turn=t, kind="tool", prev_gap=prev_gap if t else None, prev_kind=prev_kind if t else None)
    if status == "served":
        r.update(ttft=ttft, e2e=ttft + 0.03 * (out - 1), out_tokens=out, concurrency=1)
    elif status == "failed":
        r.update(ttft=INF, e2e=INF, failed=True, error="HTTP 500")
    else:
        r.update(ttft=INF, e2e=INF, censored=True)
    return r


def write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_of(ttfts):
    """In-memory run as load_run returns it: {(session, turn): ttft} (inf: censored)."""
    return dict(turns={k: dict(status="served" if math.isfinite(v) else "censored", ttft=v,
                               e2e=v + 0.1 if math.isfinite(v) else INF, out=4, prev_gap=5.0, prev_kind="tool",
                               conc=None) for k, v in ttfts.items()},
                mets={}, curves=[], series=None, summary=None)


# --- quantiles ------------------------------------------------------------------------------------
def test_quantile_with_inf():
    assert A.quantile([1, 2, 3, INF], .5) == 2.5
    assert A.quantile([1, 2, INF], .5) == 2.0                  # lands on a finite point
    assert A.quantile([1, INF], .5) == INF                     # between a finite value and inf: inf, not nan
    assert A.quantile([1, 2, 3, 4], .95) == pytest.approx(np.quantile([1, 2, 3, 4], .95))
    assert math.isnan(A.quantile([], .5))


def test_weighted_quantiles_match_expanded_sample():
    rng = np.random.default_rng(1)
    vals = np.sort(np.concatenate([rng.lognormal(size=60), [INF] * 3]))
    idx = rng.integers(0, 10, len(vals))
    counts = rng.multinomial(10, np.full(10, .1), size=50)
    for p in (.5, .95, .99):
        got = A.weighted_quantiles(vals, idx, counts, p)
        want = [A.quantile(np.repeat(vals, c[idx]), p) for c in counts]
        np.testing.assert_allclose(got, want)


# --- censoring and the common served set ----------------------------------------------------------
def test_censored_failed_as_inf_and_common_set(tmp_path):
    rng = np.random.default_rng(0)
    rows_a, rows_b = [], []
    for s in range(20):
        for t in range(6):
            rows_a.append(row(f"s{s}", t, float(rng.uniform(.1, 1))))
            rows_b.append(row(f"s{s}", t, float(rng.uniform(.1, 1))))
    # A: two censored turns and one failed one; B: lost its row for (s3, 2)
    rows_a[7] = row("s1", 1, status="censored"); rows_a[8] = row("s1", 2, status="censored")
    rows_a[20] = row("s3", 2, status="failed")
    rows_b = [r for r in rows_b if (r["session"], r["turn"]) != ("s3", 2)]
    rows_a.append({"concurrency_series": [[0, 0], [1, 2], [3, 1], [4, 0]]})
    rows_a.append({"wall_s": 5.0, "served": 117, "failed": 1, "censored": 2})
    write(tmp_path / "c__A__h1.jsonl", rows_a); write(tmp_path / "c__B__h1.jsonl", rows_b)
    write(tmp_path / "notes.jsonl", [])
    runs, skipped = A.load_dir(str(tmp_path))
    assert skipped == ["notes.jsonl"]
    keys, served = A.common_sets(runs["c"]["h1"])
    assert len(keys) == 20 * 5 - 1 and ("s3", 2) not in keys          # missing from B: dropped for both
    assert ("s1", 1) in keys and ("s1", 1) not in served                # censored in A: kept, not in TPOT set
    m = A.run_metrics(runs["c"]["h1"]["A"], keys, served)
    assert (m["n"], m["censored"], m["failed"], m["not_in_common"]) == (99, 2, 0, 1)   # the failed turn was the dropped pair
    assert m["p99"] == INF and math.isfinite(m["p95"])                  # 2 of 99 infinite: only the top tail
    finite = sorted(r["ttft"] for r in rows_a if "ttft" in r and r["turn"] > 0
                    and (r["session"], r["turn"]) != ("s3", 2) and math.isfinite(r["ttft"]))
    assert m["p50"] == pytest.approx(A.quantile(finite + [INF, INF], .5))
    assert m["tpot_n"] == 99 - 2 and m["tpot_p50"] == pytest.approx(0.03)
    assert m["concurrency"]["peak"] == 2 and m["concurrency"]["mean"] == pytest.approx((2 * 2 + 1 * 1) / 4)
    assert m["by_gap"]["<10s"]["n"] == 99 and m["by_kind"]["tool"]["n"] == 99


def test_enough_censoring_makes_p95_infinite_and_the_ratio_uncomputable():
    base = {(f"s{s}", t): 1.0 + s * 0.01 for s in range(20) for t in range(1, 6)}
    cens = dict(base)
    for s in range(6):                                                  # 30 of 100 censored
        for t in range(1, 6):
            cens[(f"s{s}", t)] = INF
    runs = {"c": {h: {"K": run_of(cens), "C0": run_of(base)} for h in ("h1", "h2", "h3")}}
    res = A.analyze(runs, ["K:C0"], {}, 0.05, 0, 0)
    r = res["pairs"]["c"]["K:C0"]["p95"]
    assert all(v == INF for v in r["per_host"].values()) and r["t"] is None
    spec = {"tests": [{"name": "H3", "rule": "iut", "cells": "all", "pairs": ["K:C0"], "threshold": 1.1}]}
    assert A.run_tests(spec, res, 0.05)["H3"]["holds"] is False         # not computable: the claim fails


# --- decision rules -------------------------------------------------------------------------------
def test_iut():
    assert A.iut([0.8, 0.95], 1.0)
    assert not A.iut([0.8, 1.01], 1.0)
    assert not A.iut([0.8, None], 1.0)
    assert not A.iut([], 1.0)


def test_tost():
    assert A.tost(0.97, 1.03, 0.05)
    assert not A.tost(0.94, 1.03, 0.05)
    assert not A.tost(0.97, 1.06, 0.05)
    assert not A.tost(None, 1.0, 0.05)


def test_holm():
    out = A.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert out["a"] == (pytest.approx(0.03), True)                     # 0.01 <= 0.05 / 3
    assert out["c"] == (pytest.approx(0.06), False)                    # 0.03 > 0.05 / 2: stop here
    assert out["b"] == (pytest.approx(0.06), False)
    out = A.holm({"a": 0.01, "b": 0.02}, 0.05)
    assert out["a"][1] and out["b"][1] and out["b"][0] == pytest.approx(0.02)


def test_t_bounds_and_p_value():
    logs = np.log([0.8, 0.85, 0.9])
    b = A.t_bounds(logs, 0.05)
    assert b["gm"] == pytest.approx(math.exp(logs.mean()))
    from scipy import stats
    q = stats.t.ppf(0.95, 2)
    assert b["upper"] == pytest.approx(math.exp(logs.mean() + q * logs.std(ddof=1) / math.sqrt(3)))
    assert A.p_below(b, b["upper"]) == pytest.approx(0.05)             # the bound is where p crosses alpha
    assert A.t_bounds(logs[:1], 0.05) is None


# --- the estimator on synthetic runs --------------------------------------------------------------
def synth(effect, n_hosts, rng, n_sess=60, n_turns=8, host_sd=0.08, censor=0.0):
    """Two arms per host. TTFT = host scale x session effect x lognormal noise; arm K's turns are drawn from the
    same family scaled by effect x a host-specific factor exp(N(0, host_sd)), so the population p95 ratio on host h
    is effect x that factor and the estimand (geometric mean over the host population) is `effect`."""
    runs = {}
    sess = rng.normal(0, 0.4, n_sess)
    for h in range(n_hosts):
        a = math.exp(rng.normal(0, 0.3)); rho = effect * math.exp(rng.normal(0, host_sd))
        c0, k = {}, {}
        for s in range(n_sess):
            for t in range(n_turns):
                c0[(f"s{s}", t)] = a * math.exp(sess[s] + rng.normal())
                k[(f"s{s}", t)] = INF if rng.random() < censor else rho * a * math.exp(sess[s] + rng.normal())
        runs[f"h{h}"] = {"C0": run_of(c0), "K": run_of(k)}
    return {"cell": runs}


def test_estimator_recovers_effect_and_t_interval_covers():
    rng = np.random.default_rng(7)
    effect, reps, upper_miss, lower_miss, gms = 0.7, 200, 0, 0, []
    for _ in range(reps):
        res = A.analyze(synth(effect, 5, rng), ["K:C0"], {}, 0.05, 0, 0)
        t = res["pairs"]["cell"]["K:C0"]["p95"]["t"]
        gms.append(math.log(t["gm"]))
        upper_miss += t["upper"] < effect
        lower_miss += t["lower"] > effect
    assert math.exp(np.mean(gms)) == pytest.approx(effect, rel=0.03)    # unbiased on the log scale, near enough
    # one-sided 95% bounds: each misses about 5% of the time (binomial sd about 1.5% at 200 reps)
    assert 0.01 <= upper_miss / reps <= 0.10 and 0.01 <= lower_miss / reps <= 0.10


def test_crossed_bootstrap_brackets_the_estimate_and_ignores_no_session():
    rng = np.random.default_rng(3)
    runs = synth(0.7, 5, rng, censor=0.01)
    res = A.analyze(runs, ["K:C0"], {}, 0.05, 2000, 0)
    p95 = res["pairs"]["cell"]["K:C0"]["p95"]
    bs, t = p95["crossed_bootstrap"], p95["t"]
    assert bs["finite"] == 2000
    assert bs["lower"] < t["gm"] < bs["upper"]
    assert bs["lower"] < 0.7 < bs["upper"]
    # the bootstrap and the t interval agree on scale (both host-dominated here)
    assert 0.5 < (math.log(bs["upper"]) - math.log(bs["lower"])) / (math.log(t["upper"]) - math.log(t["lower"])) < 2


def test_spec_end_to_end(tmp_path):
    rng = np.random.default_rng(11)
    runs = synth(0.4, 3, rng)
    for h, arms in runs["cell"].items():
        for arm, run in arms.items():
            write(tmp_path / f"slow__{arm}__{h}.jsonl",
                  [row(s, t, v["ttft"]) for (s, t), v in run["turns"].items()])
    spec = dict(best={"slow": "C0"}, tests=[
        dict(name="H1", rule="iut", cells=["slow"], pairs=["K:C0", "K:best"], threshold=1.0),
        dict(name="H5", rule="tost", cells="all", pairs=["K:C0"], metric="tpot_p50", margin=0.05),
        dict(name="H4a", rule="pooled", cells=["slow"], pair="K:C0", threshold=1.0)],
        holm=[["H1", "H4a"]])
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    res = A.main([str(tmp_path), "--spec", str(tmp_path / "spec.json"), "--boot", "200", "--out",
                  str(tmp_path / "res.json")])
    assert res["tests"]["H1"]["holds"] and res["tests"]["H5"]["holds"]
    assert res["tests"]["H4a"]["holds"] and res["tests"]["H1"]["holm"]["holds"]
    assert set(res["pairs"]["slow"]) == {"K:C0", "K:best"}
    assert res["pairs"]["slow"]["K:best"]["y"] == "C0"
    assert json.load(open(tmp_path / "res.json"))["alpha"] == 0.05


# --- live-scoring validity ----------------------------------------------------------------------------
def with_curves(run, n, flagged):
    run["curves"] = [dict(curve=f"s-{i}", pre=[.5] * 15, live=[.5] * 15, max_abs_diff=0.0) for i in range(n - flagged)] \
        + [dict(curve=f"f-{i}", pre=[.5] * 15, live_missing=True) for i in range(flagged)]
    return run


def test_live_fallback_over_one_percent_invalidates_the_run():
    rng = np.random.default_rng(5)
    runs = synth(0.7, 3, rng)
    with_curves(runs["cell"]["h0"]["K"], 100, 2)                 # 2%: invalid
    with_curves(runs["cell"]["h1"]["K"], 100, 1)                 # 1%: valid (more than 1% invalidates)
    res = A.analyze(runs, ["K:C0"], {}, 0.05, 0, 0)
    assert list(res["invalid_runs"]) == ["cell__K__h0"] and res["invalid_runs"]["cell__K__h0"]["flagged_frac"] == 0.02
    assert res["pairs"]["cell"]["K:C0"]["hosts"] == ["h1", "h2"]
    assert res["runs"]["cell__K__h1"]["live"]["valid"] and "cell__K__h0" not in res["runs"]


# --- H5 throughput half ---------------------------------------------------------------------------------
def pts(p95s, bad=None):
    return {r: dict(n=100, p95=v, bad_frac=(bad or {}).get(r, 0.0)) for r, v in p95s.items()}


def test_peak_rate_interpolates_and_flags_the_grid_edges():
    pk = A.peak_rate(pts({1: .5, 2: .8, 3: 1.4, 4: .9}), 1.0)
    assert pk["peak"] == pytest.approx(2 + 0.2 / 0.6)              # rate 4 passing again does not count
    pk = A.peak_rate(pts({1: .5, 2: .5}, bad={2: .03}), 1.0)        # failures, not latency, end it: g = 3 at rate 2
    assert pk["peak"] == pytest.approx(1 + 0.5 / 2.5)
    assert A.peak_rate(pts({1: .5, 2: .6}), 1.0) == dict(rates=[1, 2], excess=[.5, .6], peak=2, above_grid=True)
    assert A.peak_rate(pts({1: 1.5, 2: 2.0}), 1.0)["peak"] is None


def test_dropped_arrivals_count_as_not_served(tmp_path):
    # 99 served of 99 sent, but 2 arrivals found no session ready: 2 of 101 offered not served, over the 1% bound
    p = tmp_path / "c__K__h1__rate5.jsonl"
    write(p, [row(f"q{i}", 0, 0.1) for i in range(99)] +
          [dict(wall_s=20.0, served=99, failed=0, censored=0, open_loop_rate=5.0, requests=99, arrivals_dropped=2)])
    pt = A.sweep_point(A.load_run(str(p)))
    assert (pt["n"], pt["dropped"]) == (99, 2) and pt["bad_frac"] == pytest.approx(2 / 101)
    assert A.peak_rate({1.0: dict(pt, bad_frac=0.0), 5.0: pt}, 1.0)["peak"] < 5.0


def test_throughput_sweep_end_to_end(tmp_path):
    f = {"h1": 1.02, "h2": 0.98, "h3": 1.0}
    for h, fh in f.items():
        for arm, scale in (("C0", 1.0), ("K", fh)):
            for r in (1, 2, 3, 4, 5):
                write(tmp_path / f"tp__{arm}__{h}__rate{r}.jsonl", [row(f"q{i}", 0, 0.2 * r * scale) for i in range(100)])
    spec = dict(slo={"tp": 0.7}, tests=[dict(name="H5b", rule="throughput_tost", pair="K:C0", margin=0.10),
                                        dict(name="H5b_tight", rule="throughput_tost", pair="K:C0", margin=0.005)])
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    res = A.main([str(tmp_path), "--spec", str(tmp_path / "spec.json"), "--boot", "2000"])
    assert res["ignored"] == [] and res["pairs"] == {}
    assert res["throughput"]["peaks"]["tp__C0__h1"]["peak"] == pytest.approx(3.5)
    per = res["throughput"]["pairs"]["tp"]["K:C0"]["per_host"]
    for h, fh in f.items():                                         # peak where 0.2 r fh = 0.7, inside (3, 4)
        assert per[h] == pytest.approx((3.5 / fh) / 3.5, rel=1e-9)
    assert res["tests"]["H5b"]["holds"] and not res["tests"]["H5b_tight"]["holds"]


# --- H6 ------------------------------------------------------------------------------------------------------
def test_kendall_tau_b_matches_scipy_and_counts_overlaps_as_ties():
    from scipy import stats
    rng = np.random.default_rng(2)
    for _ in range(20):
        sim = {f"a{i}": float(v) for i, v in enumerate(rng.integers(0, 4, 6))}      # with ties
        est = {a: float(rng.normal()) for a in sim}
        want = stats.kendalltau([sim[a] for a in sorted(sim)], [est[a] for a in sorted(sim)]).statistic
        assert A.kendall_tau_b(sim, est, est, est) == pytest.approx(want)
    sim = {"a": 1, "b": 2, "c": 3}
    est = {"a": 1.0, "b": 1.1, "c": 3.0}
    lo, hi = {"a": .9, "b": 1.0, "c": 2.9}, {"a": 1.05, "b": 1.2, "c": 3.1}                # a and b overlap
    assert A.kendall_tau_b(sim, est, lo, hi) == pytest.approx(2 / math.sqrt(3 * 2))
    assert A.kendall_tau_b(sim, est, {a: 0 for a in sim}, {a: 9 for a in sim}) is None   # all tied


def test_h6_end_to_end():
    rng = np.random.default_rng(4)
    base = {(f"s{s}", t): float(v) for s in range(30) for t in range(1, 6) for v in [rng.lognormal()]}
    runs = {"cell": {}}
    for h in range(4):
        j = 1 + 0.01 * rng.normal(size=2)
        runs["cell"][f"h{h}"] = {"C0": run_of(base), "K": run_of({k: v * 0.6 * j[0] for k, v in base.items()}),
                                 "T": run_of({k: v * 0.9 * j[1] for k, v in base.items()})}
    sim_good = {"cell": {"C0": 1.0, "K": 0.5, "T": 0.8}}
    sim_bad = {"cell": {"C0": 1.0, "K": 0.8, "T": 0.5}}
    res = A.analyze(runs, ["K:C0", "T:C0"], {}, 0.05, 0, 0)
    good = A.h6(res, sim_good, "C0", ["cell"], 0.6)
    bad = A.h6(res, sim_bad, "C0", ["cell"], 0.6)
    assert good["mean"] == pytest.approx(1.0) and good["holds"]
    assert bad["mean"] == pytest.approx(1 / 3) and not bad["holds"]


# --- run-order covariate ------------------------------------------------------------------------------------
def test_order_sensitivity_recovers_a_position_slope():
    rng = np.random.default_rng(9)
    d = {"h1": -2, "h2": -1, "h3": 1, "h4": 2, "h5": 3}
    order, per = {}, {}
    for h, dh in d.items():                       # K at position 3 + dh, C0 at 3
        lst = [f"x/{i}" for i in range(8)]
        lst[3] = "c/C0"; lst[3 + dh] = "c/K"
        order[h] = lst
        per[h] = math.exp(math.log(0.8) + 0.05 * dh + 0.01 * rng.normal())
    order["h9"] = ["c/K", "c/C0"]                 # not in the pair: ignored
    res = {"pairs": {"c": {"K:C0": dict(x="K", y="C0", p95=dict(per_host=per))}}}
    o = A.order_sensitivity(res, order, 0.05)["c"]["K:C0"]
    assert o["n"] == 5 and o["position_diff"] == [-2, -1, 1, 2, 3]
    assert o["lower"] < 0.05 < o["upper"] and o["upper"] - o["lower"] < 0.05
    assert o["intercept_ratio"] == pytest.approx(0.8, rel=0.03)
    res["pairs"]["c"]["K:C0"]["p95"]["per_host"] = {h: per[h] for h in ("h1", "h2")}
    assert A.order_sensitivity(res, order, 0.05)["c"]["K:C0"]["slope"] is None          # n < 3
