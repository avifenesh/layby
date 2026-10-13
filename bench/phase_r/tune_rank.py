#!/usr/bin/env python3
"""Rank one baseline-tuning block (BENCH_PLAN section 8): every grid point of a tuning cell
(bench/phase_r/make_tuning.py) by p95 TTFT on resumed turns, and the points that advance to the next stage.

Input: the tuning cell spec (its "tuning" section: model, family, disk, stage, the full grid in grid order) and the
orchestrator's output root (engine/phase_r/orchestrate.py run --out OUT). One block is one attempt directory,
OUT/CELL/HOST/rREPEAT/aATTEMPT/, holding one directory per arm, POS.LABEL/, with the runner's per-turn rows renamed to
CELL.LABEL.HOST.jsonl and the harness verdict CELL.LABEL.HOST.verdict.json. The block ranked is the highest attempt
(a block that reruns, on this host or a replacement, takes the next attempt number); --host, --repeat and --attempt
pick another, and two blocks with the same highest attempt (the cell run on two hosts) must be told apart with --host.

Rows and statistics, the analysis rules (engine/analyze_phase_r.py; its row rule and quantile are copied here,
since that module needs scipy): a resumed turn is a turn row with turn > 0; a row is censored (the hard cap ended
it), failed, or served, and a censored or failed turn counts as +inf TTFT, worse than any served turn; p50 and p95 by
numpy's linear method, where a point between a finite value and +inf is +inf. Only the TTFT and status columns are
read. Every ranked arm is scored on the same turns: the resumed (session, turn) pairs every ranked arm has a row for
(with run-to-completion that is the whole workload; rows missing from an arm's log are counted). An arm is not
ranked, and is reported with the reason, when its verdict.json is missing (the attempt stopped before it ran) or
invalid (health.py: the exclusion rule of section 7), or its row file is missing or unreadable. A ranking with an
unranked grid point is partial; make_tuning.py next refuses a partial ranking unless told otherwise, since the plan
reruns the whole block instead.

Order: p95 ascending, ties broken by p50, then by grid order (the cell's tuning.grid; a statistic that is not a
number, no resumed turn at all, counts as +inf).

Advancing (successive halving over the stage's grid points, n = the arms in the cell, ranked or not):
  stage 1: the top ceil(n / 3) go to stage 2 (the 40-minute workload) when that is more than 2 points; otherwise
           the top 2 go straight to stage 3 (the full tuning workload)
  stage 2: the top 2 go to stage 3
  stage 3: the first is the winner (its knobs carry to the main cells: make_tuning.py next)
Fewer ranked points than that advance as many as there are.

Output (--out, default rank/CELL.json beside the cell spec): the block, every arm's counts (resumed turns, served,
failed, censored, missing rows), p50 and p95 ("inf" for +inf), rank and verdict reasons, the unranked arms with
reasons, and "advance", "next_stage" and "winner". It prints the ranking as a table.

Usage: tune_rank.py --cell CELL.json --runs OUT [--host HOST] [--repeat N] [--attempt A] [--out RANKING.json]
"""
import argparse
import glob
import json
import math
import os
import re
import sys

import numpy as np

INF = float("inf")


def quantile(v, p):
    """engine/analyze_phase_r.py's quantile (copied: that module needs scipy): numpy's linear method on values that
    may hold +inf, where a point between a finite value and +inf is +inf (numpy would give nan); empty gives nan."""
    v = np.sort(np.asarray(v, float))
    if not len(v):
        return float("nan")
    pos = p * (len(v) - 1)
    k = int(math.floor(pos)); f = pos - k
    if f == 0 or k + 1 >= len(v):
        return float(v[k])
    lo, hi = v[k], v[k + 1]
    return INF if hi == INF else float(lo + (hi - lo) * f)


def load_turns(path):
    """{(session, turn): {status, ttft}} of one replay.py output, by engine/analyze_phase_r.py load_run's rule: a
    turn row has session, turn and ttft; censored, else failed, else served (a served row without a TTFT is failed);
    a turn not served has TTFT +inf. Two rows for one turn are an error."""
    turns = {}
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if not ("session" in r and "turn" in r and "ttft" in r) or "metrics" in r or "curve" in r:
                continue
            key = (r["session"], int(r["turn"]))
            if key in turns:
                raise ValueError(f"{path}: two rows for {key}")
            status = "censored" if r.get("censored") else "failed" if r.get("failed") else "served"
            if status == "served" and r["ttft"] is None:
                status = "failed"
            turns[key] = dict(status=status, ttft=float(r["ttft"]) if status == "served" else INF)
    return turns


def advance_rule(stage, n):
    """(next stage or None, how many advance) for a stage of n grid points."""
    if stage == 1:
        k = math.ceil(n / 3)
        return (2, k) if k > 2 else (3, 2)
    if stage == 2:
        return 3, 2
    if stage == 3:
        return None, 1
    raise ValueError(f"stage {stage!r}")


def _num(x):
    """A statistic for ranking: nan (no turns) is +inf."""
    return INF if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def rank_key(arm, grid):
    return (_num(arm["p95"]), _num(arm["p50"]), grid.index(arm["label"]))


def score(runs):
    """{label: stats} of loaded runs ({label: load_turns result}) on their common resumed turns."""
    common = None
    for T in runs.values():
        k = {x for x in T if x[1] > 0}
        common = k if common is None else common & k
    common = sorted(common or ())
    out = {}
    for label, T in runs.items():
        st = [T[k]["status"] for k in common]
        v = [T[k]["ttft"] for k in common]
        out[label] = dict(n=len(common), served=st.count("served"), failed=st.count("failed"),
                          censored=st.count("censored"), missing=sum(1 for k in T if k[1] > 0) - len(common),
                          p50=quantile(v, .5), p95=quantile(v, .95))
    return out


def rank(stats, grid):
    """Arms (dicts with label, p50, p95) sorted by the ranking rule, each given its 1-based rank."""
    ranked = sorted((dict(s, label=lab) for lab, s in stats.items()), key=lambda a: rank_key(a, grid))
    for i, a in enumerate(ranked, 1):
        a["rank"] = i
    return ranked


def decide(stage, n_points, ranked):
    """(next_stage, advancing labels, winner label or None)."""
    nxt, k = advance_rule(stage, n_points)
    top = [a["label"] for a in ranked[:k]]
    if nxt is None:
        return None, [], (top[0] if top else None)
    return nxt, top, None


def find_block(runs_root, cell, host=None, repeat=None, attempt=None):
    """(host, repeat, attempt, directory) of the block to rank (module docstring)."""
    pat = re.compile(r"^r(\d+)$"), re.compile(r"^a(\d+)$")
    found = []
    for d in glob.glob(os.path.join(runs_root, cell, "*", "r*", "a*")):
        a_dir = os.path.basename(d)
        r_dir = os.path.basename(os.path.dirname(d))
        h = os.path.basename(os.path.dirname(os.path.dirname(d)))
        mr, ma = pat[0].match(r_dir), pat[1].match(a_dir)
        if not (mr and ma and os.path.isdir(d)):
            continue
        b = (h, int(mr.group(1)), int(ma.group(1)), d)
        if (host is None or h == host) and (repeat is None or b[1] == repeat) and (attempt is None or b[2] == attempt):
            found.append(b)
    if not found:
        raise FileNotFoundError(f"no block of cell {cell} under {runs_root} (host {host}, repeat {repeat}, "
                                f"attempt {attempt})")
    top = max(b[2] for b in found)
    last = [b for b in found if b[2] == top]
    if len(last) > 1:
        raise ValueError(f"cell {cell}: attempt {top} exists for {sorted((b[0], b[1]) for b in last)}; pick one "
                         f"with --host / --repeat")
    return last[0]


def arm_dirs(block_dir):
    """{label: arm directory} of one attempt (POS.LABEL; the unscored filler is not an arm)."""
    out = {}
    for d in sorted(glob.glob(os.path.join(block_dir, "*.*"))):
        if not os.path.isdir(d):
            continue
        label = os.path.basename(d).split(".", 1)[1]
        if not label.startswith("_"):
            out[label] = d
    return out


def collect(spec, runs_root, host=None, repeat=None, attempt=None):
    """The ranking of one tuning block (module docstring) as a JSON-ready dict."""
    tun = spec.get("tuning")
    if not tun:
        raise ValueError(f"cell {spec.get('cell')} has no tuning section (not a make_tuning.py cell)")
    cell = spec["cell"]
    labels = [a["label"] for a in spec["arms"]]
    grid = list(tun["grid"])
    if any(lab not in grid for lab in labels):
        raise ValueError(f"arms {labels} not all in the grid {grid}")
    h, rep, att, bdir = find_block(runs_root, cell, host, repeat, attempt)
    dirs = arm_dirs(bdir)
    runs, unranked, verdicts = {}, [], {}
    for lab in labels:
        d = dirs.get(lab)
        pre = os.path.join(d, f"{cell}.{lab}.{h}") if d else None
        if d is None or not os.path.exists(pre + ".verdict.json"):
            unranked.append({"label": lab, "reason": "no verdict: the attempt stopped before this arm ran"})
            continue
        v = json.load(open(pre + ".verdict.json"))
        verdicts[lab] = v
        if not v.get("valid"):
            unranked.append({"label": lab, "reason": "invalid verdict", "reasons": v.get("reasons")})
            continue
        if not os.path.exists(pre + ".jsonl"):
            unranked.append({"label": lab, "reason": "no row file"})
            continue
        try:
            runs[lab] = load_turns(pre + ".jsonl")
        except (ValueError, json.JSONDecodeError) as e:
            unranked.append({"label": lab, "reason": f"unreadable rows: {e}"})
    ranked = rank(score(runs), grid) if runs else []
    nxt, adv, win = decide(int(tun["stage"]), len(labels), ranked)
    arm_spec = {a["label"]: a for a in spec["arms"]}
    for a in ranked:
        a.update(code=arm_spec[a["label"]]["code"], knobs=arm_spec[a["label"]].get("knobs"),
                 grid_index=grid.index(a["label"]), p50=num_out(a["p50"]), p95=num_out(a["p95"]))
    return {
        "cell": cell, "tuning": tun, "block": {"host": h, "repeat": rep, "attempt": att, "dir": bdir},
        "rule": "p95 TTFT on resumed turns (failed and censored +inf), ties by p50 then grid order",
        "arms": ranked, "unranked": unranked, "complete": not unranked,
        "next_stage": nxt, "advance": adv, "winner": win,
    }


def num_out(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return "inf" if x == INF else round(float(x), 6)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cell", required=True)
    ap.add_argument("--runs", required=True, help="orchestrate.py run --out root")
    ap.add_argument("--host")
    ap.add_argument("--repeat", type=int)
    ap.add_argument("--attempt", type=int)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    spec = json.load(open(a.cell))
    res = collect(spec, os.path.abspath(a.runs), a.host, a.repeat, a.attempt)
    res["cell_spec"] = os.path.abspath(a.cell)
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.cell)), "rank", f"{spec['cell']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(res, f, indent=1)
        f.write("\n")
    b = res["block"]
    print(f"{res['cell']} stage {res['tuning']['stage']} ({b['host']} r{b['repeat']} a{b['attempt']}): "
          f"{len(res['arms'])} ranked, {len(res['unranked'])} not ranked")
    print(f"{'rank':>4} {'label':<22} {'p95':>10} {'p50':>10} {'n':>5} {'failed':>6} {'cens':>5} {'miss':>5}")
    for x in res["arms"]:
        f = lambda v: "inf" if v == "inf" else ("nan" if v is None else f"{v:.3f}")   # noqa: E731
        print(f"{x['rank']:>4} {x['label']:<22} {f(x['p95']):>10} {f(x['p50']):>10} {x['n']:>5} {x['failed']:>6} "
              f"{x['censored']:>5} {x['missing']:>5}")
    for u in res["unranked"]:
        print(f"   - {u['label']:<22} not ranked: {u['reason']} {u.get('reasons') or ''}")
    if res["winner"]:
        print(f"winner: {res['winner']}")
    elif res["next_stage"] and res["advance"]:
        print(f"advance to stage {res['next_stage']}: {', '.join(res['advance'])}")
    else:
        print("nothing ranked: nothing advances")
    print("written", out)
    return 0 if res["complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
