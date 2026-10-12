"""C0 projections for workload sizing (BENCH_PLAN draft 6): the fitted service model of fit_service.py run on a
candidate workload. Used by build_workloads.py --service-fit to cut sessions where their projected C0 end passes the
run's end cap and to size by peak live KV on the projected timeline, and by scan_load.py to choose the load.

A session is a dict with id, start and turns (the replay format); turns carry new_tokens, out_len, ctx, gap_after,
keep. The projection is closed loop: a turn is sent when the previous one ends plus its recorded gap."""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sim.engine_sim import Params, simulate  # noqa: E402


def load_fit(path):
    d = json.load(open(path))
    sp = dict(d["sim_params"])
    sp.pop("until", None)
    return dict(path=path, kp=d["fit"]["kp"], kd=d["fit"]["kd"], params=sp, model=d["model"])


def run_c0(sessions, fit, until=None):
    """Simulated C0 replay; rows per served turn (session, turn, ttft, t_send, t_first, t_done, P, hit, ext, queue)."""
    W = {"sessions": [dict(id=s["id"], start=s["start"], turns=s["turns"]) for s in sessions]}
    p = Params(**dict(fit["params"], until=until))
    rows, info = simulate(W, None, p)
    return rows, info


def _cut_one(s, n):
    T = [dict(t) for t in s["turns"][:n]]
    T[-1]["gap_after"] = None
    return dict(s, turns=T)


def project(sessions, fit, end_cap, min_turns=3, max_iter=8):
    """Cut each session at its last turn whose projected C0 end is within end_cap seconds of the run start; drop a
    session left with fewer than min_turns turns. Re-simulate the cut workload until every kept turn ends by the cap
    (a cut lowers the load, so this converges in a few rounds). Returns (sessions, rows, info, rounds)."""
    cur = [s for s in sessions]
    for it in range(1, max_iter + 1):
        rows, info = run_c0(cur, fit, until=end_cap * 4 + 3600)
        done = {}
        for r in rows:
            done.setdefault(r["session"], {})[r["turn"]] = r["t_done"]
        nxt, changed = [], False
        for s in cur:
            d = done.get(s["id"], {})
            n = 0
            while n < len(s["turns"]) and d.get(n, math.inf) <= end_cap:
                n += 1
            if n < len(s["turns"]):
                changed = True
            if n >= min_turns:
                nxt.append(s if n == len(s["turns"]) else _cut_one(s, n))
            else:
                changed = True
        cur = nxt
        if not changed:
            return cur, rows, info, it
    rows, info = run_c0(cur, fit, until=end_cap * 4 + 3600)
    return cur, rows, info, max_iter + 1


def kv_bytes(kv, c):
    if not kv:
        return 0.0
    return kv["grow_bytes_per_token"] * c + kv["state_bytes_per_session"] + kv["window_bytes_per_token"] * min(c, kv["window_tokens"])


def stats(sessions, rows, kv=None):
    """Peak live KV on the projected timeline (a session is live from its start to its last turn's end, holding its
    context from each turn's send), mean requests in flight, resumed-turn TTFT, makespan and the per-turn service."""
    by = {(r["session"], r["turn"]): r for r in rows}
    ev, evb = [], []
    ends = {}
    for s in sessions:
        cur, curb, last = 0, 0.0, s["start"]
        for j, t in enumerate(s["turns"]):
            r = by.get((s["id"], j))
            if r is None:
                break
            b = kv_bytes(kv, t["ctx"])
            ev.append((r["t_send"], t["ctx"] - cur)); evb.append((r["t_send"], b - curb))
            cur, curb, last = t["ctx"], b, r["t_done"]
        ev.append((last, -cur)); evb.append((last, -curb))
        ends[s["id"]] = last
    ev.sort(); evb.sort()
    mk = max(r["t_done"] for r in rows) if rows else 0.0
    res = np.array([r["ttft"] for r in rows if r["turn"] > 0])
    q = np.array([r["queue"] for r in rows if r["turn"] > 0])
    busy = sum(r["t_done"] - r["t_send"] for r in rows)
    T = [t for s in sessions for t in s["turns"]]
    return dict(sessions=len(sessions), turns=len(T), served=len(rows),
                peak_live_kv_tokens=int(np.max(np.cumsum([d for _, d in ev]))) if ev else 0,
                peak_live_kv_bytes=int(np.max(np.cumsum([d for _, d in evb]))) if evb else 0,
                makespan_s=round(mk, 1), mean_inflight=round(busy / mk, 3) if mk else 0.0,
                resumed_ttft_p50_s=round(float(np.median(res)), 3) if len(res) else None,
                resumed_ttft_p90_s=round(float(np.quantile(res, .9)), 3) if len(res) else None,
                resumed_ttft_p95_s=round(float(np.quantile(res, .95)), 3) if len(res) else None,
                resumed_queue_p50_s=round(float(np.median(q)), 3) if len(q) else None,
                recompute_share=round(1 - sum(r["hit"] + r["ext"] for r in rows) / max(1, sum(r["P"] for r in rows)), 4),
                session_end_max_s=round(max(ends.values()), 1) if ends else 0.0)


def service_rows(rows):
    """Per served turn: seconds from send to end (TTFT + decode) at the projected load."""
    return {(r["session"], r["turn"]): r["t_done"] - r["t_send"] for r in rows}
