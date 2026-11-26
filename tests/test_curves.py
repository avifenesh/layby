"""K_km and K_ewma curves (layby.curves) and their path into the unchanged cost rule, CPU only.

1. km: the Kaplan-Meier curve at the 15 horizons; with no censoring it is the fraction of waits longer than each
   horizon; with censoring it matches a hand computation.
2. fit_km: one curve per wait kind plus "all".
3. session_curves: a turn gets the exponential curve with the EWMA mean of its session's earlier waits of the same
   kind once two exist, else the per-kind KM curve; a turn's own wait never enters its curve; a turn with no wait
   after it gets none. annotate writes surv_km and surv_ewma and touches nothing else.
4. A curve so made reaches layby.rule.decide exactly as a Layby-Dwell curve does: posted as a hint for a finished
   request, drained by ParkConnector, decided with the same inputs as a direct call to decide.
Run: PYTHONPATH=. python tests/test_curves.py
"""
import copy
import json
import os
import tempfile
import time

import numpy as np

from layby.curves import annotate, ewma, exp_curve, fit_km, km, main, session_curves
from layby.rule import E2, TG, decide, interp_surv

# --- 1. km ----------------------------------------------------------------------------------------------------
g = np.array([0.4, 1.0, 2.5, 7.0, 7.0, 40.0, 100.0, 2000.0])
S = km(g)
assert np.allclose(S, [(g > h).mean() for h in E2]), (S, [(g > h).mean() for h in E2])
# censored: 1 (event), 2 (censored), 3 (event), 4 (event): S(1) = 3/4, S(3) = 3/4 * (1 - 1/2), S(4) = 0
Sc = km([1, 2, 3, 4], [1, 0, 1, 1], H=np.array([0.5, 1, 2, 3, 5]))
assert np.allclose(Sc, [1.0, 0.75, 0.75, 0.375, 0.0]), Sc

# --- 2. fit_km ------------------------------------------------------------------------------------------------
sessions = [dict(turns=[dict(kind="tool", gap_after=2.0), dict(kind="tool", gap_after=4.0),
                        dict(kind="human", gap_after=100.0), dict(kind="tool", gap_after=8.0),
                        dict(kind="tool", gap_after=None)]),
            dict(turns=[dict(kind="human", gap_after=600.0), dict(kind="tool", gap_after=1.0),
                        dict(kind="tool", gap_after=None)])]
K = fit_km(sessions)
assert np.allclose(K["tool"], km([2, 4, 8, 1])) and np.allclose(K["human"], km([100, 600]))
assert np.allclose(K["all"], km([2, 4, 100, 8, 600, 1])) and K["_n"] == {"human": 2, "tool": 4, "all": 6}
Kc = fit_km(sessions, cap=300)
assert np.allclose(Kc["human"], km([100, 300]))
assert "_censored" not in K
# a wait that never returned ("returned": false) is censored at its gap; unmarked turns stay returns
cens = copy.deepcopy(sessions); cens[1]["turns"][0]["returned"] = False
Kx = fit_km(cens)
assert np.allclose(Kx["human"], km([100, 600], [1, 0])) and np.allclose(Kx["tool"], K["tool"])
assert np.allclose(Kx["all"], km([2, 4, 100, 8, 600, 1], [1, 1, 1, 1, 0, 1]))
assert Kx["_censored"] == {"human": 1, "tool": 0, "all": 1} and Kx["_n"] == K["_n"]

# --- 3. session curves ------------------------------------------------------------------------------------------
C = session_curves(sessions[0]["turns"], K, alpha=0.5)
assert C[0] == (K["tool"], K["tool"])                     # no earlier tool wait
assert C[1] == (K["tool"], K["tool"])                     # one earlier tool wait (2 s): still KM
assert C[2] == (K["human"], K["human"])                   # no earlier human wait
assert np.allclose(C[3][0], np.exp(-E2 / 3.0)), C[3]      # earlier tool waits 2, 4: EWMA 0.5 * 4 + 0.5 * 2 = 3
assert C[3][1] == K["tool"]
assert C[4] == (None, None)
assert ewma([2, 4, 8], 0.5) == 5.5 and ewma([2, 4, 8], 1.0) == 8.0
# a turn's own wait does not enter its curve: changing it changes only later turns
t2 = copy.deepcopy(sessions[0]["turns"]); t2[3]["gap_after"] = 1e4
assert np.allclose(session_curves(t2, K)[3][0], C[3][0])
W = {"gap_cap": 300, "sessions": copy.deepcopy(sessions)}
r = annotate(W, K, alpha=0.5)
assert r == dict(turns=6, ewma_own=1, alpha=0.5), r
t = W["sessions"][0]["turns"]
assert t[3]["surv_ewma"] == exp_curve(3.0).tolist() and t[3]["surv_km"] == K["tool"] and "surv_km" not in t[4]
assert {k for k in t[0]} == {"kind", "gap_after", "surv_km", "surv_ewma"}
d = tempfile.mkdtemp(prefix="pp_curves_", dir=os.path.expanduser("~/.cache"))
try:
    json.dump(W | {"sessions": sessions}, open(f"{d}/w.json", "w"))
    main(["fit-km", f"{d}/w.json", "--out", f"{d}/km.json"])
    main(["annotate", f"{d}/w.json", "--km", f"{d}/km.json", "--out", f"{d}/w2.json"])
    W2 = json.load(open(f"{d}/w2.json"))
    assert W2["gap_cap"] == 300 and W2["sessions"][0]["turns"][3]["surv_ewma"] == t[3]["surv_ewma"]
finally:
    for f in os.listdir(d):
        os.unlink(os.path.join(d, f))
    os.rmdir(d)
print("curves ok: KM per kind, EWMA of the session's own earlier waits of the same kind, annotate")

# --- 5. the simulator's controls get the engine's curves ------------------------------------------------------
import sim.eval_pop as E

assert E.rule_known("cost_ewma") and E.rule_known("cost_ewma_a0.3") and not E.rule_known("cost_ewma_a2")
os.environ.pop("RETURNBENCH_KM", None)
try:
    E.make_rule("cost_ewma", {"sessions": copy.deepcopy(sessions)}, "v6")
    raise AssertionError("cost_ewma without KM curves must fail")
except ValueError:
    pass
fd, kmp = tempfile.mkstemp(suffix=".json", dir=os.path.expanduser("~/.cache"))
try:
    with os.fdopen(fd, "w") as f:
        json.dump(K, f)
    os.environ["RETURNBENCH_KM"] = kmp
    for alpha, name in ((0.3, "cost_ewma_a0.3"), (0.5, "cost_ewma"), (None, "cost_km")):
        Ws, Wa = {"sessions": copy.deepcopy(sessions)}, {"sessions": copy.deepcopy(sessions)}
        rule = E.make_rule(name, Ws, "v6")
        annotate(Wa, K, alpha=alpha or 0.5)              # what the engine workload carries (layby.curves annotate)
        field = "surv_km" if alpha is None else "surv_ewma"
        n = 0
        for ss, sa in zip(Ws["sessions"], Wa["sessions"]):
            for ts, ta in zip(ss["turns"], sa["turns"]):
                if field in ta:
                    assert ts[f"surv_{rule.name}"] == ta[field], (name, ts, ta)
                    assert np.allclose(rule.curve(ts), interp_surv(np.asarray(ta[field]), TG))
                    n += 1
                else:
                    assert f"surv_{rule.name}" not in ts
        assert n == 6, (name, n)
finally:
    os.environ.pop("RETURNBENCH_KM", None)
    os.unlink(kmp)
print("sim ok: cost_km and cost_ewma[_aALPHA] read the curves layby.curves annotate gives the engine workload")

# a replay workload with a context cut (keep) and the GLM profile: the cut turn reuses only its kept prefix
from sim.engine_sim import Params, simulate
wf = {"gap_cap": 300, "ctx_cap": 32768, "sessions": [{"id": "s0", "start": 0.0, "turns": [
    dict(kind="tool", new_tokens=1000, out_len=10, ctx=1010, gap_after=1.0, keep=None),
    dict(kind="tool", new_tokens=500, out_len=10, ctx=1520, gap_after=1.0, keep=None),
    dict(kind="tool", new_tokens=300, out_len=10, ctx=410, gap_after=None, keep=100)]}]}
fd, wp = tempfile.mkstemp(suffix=".json", dir=os.path.expanduser("~/.cache"))
try:
    with os.fdopen(fd, "w") as f:
        json.dump(wf, f)
    Wr = E.workload(wp, 0, 3)
    assert [s["start"] for s in Wr["sessions"]] == [0.0] and len(Wr["sessions"][0]["turns"]) == 3
    rows, _ = simulate(Wr, None, Params(**dict(E.CAL, **E.PROFILES["glm53_flash"])))
    P = {r["turn"]: (r["P"], r["hit"]) for r in rows}
    assert P[1] == (1510, 1010) and P[2] == (400, 100), P     # 100 kept + 300 new; only the 100 match
finally:
    os.unlink(wp)
print("sim ok: replay workloads run as they are, a keep cut matches only the kept prefix")

# --- 4. into the unchanged rule -------------------------------------------------------------------------------
from types import SimpleNamespace as NS
import layby.vllm.connector as PCm
from layby.vllm.state import STATE

PC = PCm.ParkConnector
STATE.__init__()
STATE.policy, STATE.fs_tier = None, None
STATE.gpu_tokens, STATE.cpu_tokens, STATE.bytes_per_token = 1 << 20, 4 << 20, 144 * 1024
now = time.monotonic()
STATE.live.add(now - 30, n_adm=5.0, qwait=2.0, pf=3.0, n_cpu=4.0, wait_cpu=0.2, w_c=5000.0, ne_c=2.0)
for name, surv in (("ewma", t[3]["surv_ewma"]), ("km", t[3]["surv_km"])):
    params = {"park_key": "s1-3"}
    me = NS(_finished={"s1-3": dict(keys=[], tokens=20000, params=params, t=now)}, _early={})
    me._decide = lambda *a: PC._decide(me, *a)
    STATE.hints.put(("s1-3", surv))                     # what replay.py --park-hints NAME posts at the turn's end
    PC._drain_hints(me, now)
    got = STATE.dlog[-1]
    want = decide(interp_surv(np.nan_to_num(np.asarray(surv, float), nan=0.5), TG), 20000, 0, STATE.idle_g(now),
                  (np.zeros(0), np.zeros(0)), STATE.live, STATE.params(), now, write=STATE.write_mode)
    assert got["opt"] == want["opt"] and params["park_eta"] == want["eta"], (name, got, want["opt"], want["eta"])
    assert got["tot"] == {k: (round(x, 4), round(y, 4)) for k, (x, y) in want["tot"].items()}, (name, got["tot"])
    assert STATE.counts["late"] >= 1
print("rule ok: K_km and K_ewma curves take the same path into layby.rule.decide as Layby-Dwell's")
