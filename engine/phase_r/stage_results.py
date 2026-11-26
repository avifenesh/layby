#!/usr/bin/env python3
"""Stage a Phase R results tree for analyze_phase_r.py.

Reads OUT/runs.jsonl (one row per arm run, written by orchestrate.py) and, for every (cell, host, repeat), picks the
attempt in which every non-filler arm ran and was judged valid (the pairing rule: an attempt counts whole or not at
all). Each picked arm's turn log OUT/.../PREFIX.jsonl is linked as STAGE/CELL__LABEL__HOST-rREPEAT.jsonl, so repeats
take the place of hosts (BENCH_PLAN section 7, draft 6.7). Nothing is excluded here beyond the orchestrator's own
verdicts; void attempts are listed in STAGE/staging.json with their reasons.

Usage: stage_results.py OUT STAGE [--cell CELL ...] [--path-map OLD=NEW]
  --path-map rewrites the run dirs recorded on the measuring host to where the tree was copied.
"""
import argparse
import collections
import json
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("stage")
    ap.add_argument("--cell", action="append", default=[], help="keep only these cells (default: all main cells)")
    ap.add_argument("--path-map", default=None, help="OLD=NEW prefix rewrite for run dirs")
    a = ap.parse_args(argv)
    old, new = (a.path_map.split("=", 1) if a.path_map else (None, None))
    rows = [json.loads(ln) for ln in open(os.path.join(a.out, "runs.jsonl")) if ln.strip()]
    att = collections.defaultdict(list)
    for r in rows:
        if r.get("dry_run") or r["cell"].startswith(("s-", "t-", "p2-")):
            continue
        if a.cell and r["cell"] not in a.cell:
            continue
        att[(r["cell"], r["host"], r["repeat"], r["attempt"])].append(r)
    # A cell's full arm set is the union of labels over its attempts; an attempt still running (or cut short) lacks some.
    labels = collections.defaultdict(set)
    for (cell, _h, _r, _a), rs in att.items():
        labels[cell].update(r["label"] for r in rs if not r["label"].startswith("_"))
    picked, void, partial = {}, [], []
    for (cell, host, rep, at), rs in sorted(att.items()):
        arms = [r for r in rs if not r["label"].startswith("_")]
        if {r["label"] for r in arms} != labels[cell]:
            if all(r["valid"] is True for r in arms):
                partial.append({"cell": cell, "host": host, "repeat": rep, "attempt": at,
                                "missing_arms": sorted(labels[cell] - {r["label"] for r in arms})})
                continue
        ok = arms and {r["label"] for r in arms} == labels[cell] and all(
            r["status"] == "ran" and r["valid"] is True for r in arms)
        if ok:
            picked[(cell, host, rep)] = (at, arms)  # later valid attempts would replace earlier ones; none expected
        else:
            void.append({"cell": cell, "host": host, "repeat": rep, "attempt": at,
                         "reasons": {r["label"]: r["reasons"] for r in arms if r["valid"] is not True}})
    os.makedirs(a.stage, exist_ok=True)
    staged, missing = [], []
    for (cell, host, rep), (at, arms) in sorted(picked.items()):
        for r in arms:
            d = r["dir"] if old is None else r["dir"].replace(old, new, 1)
            src = os.path.join(d, r["prefix"] + ".jsonl")
            if not os.path.exists(src):
                missing.append(src)
                continue
            dst = os.path.join(a.stage, f"{cell}__{r['label']}__{host}-r{rep}.jsonl")
            if os.path.lexists(dst):
                os.remove(dst)
            os.symlink(os.path.abspath(src), dst)
            staged.append({"cell": cell, "label": r["label"], "host": f"{host}-r{rep}", "attempt": at, "src": src})
    by_cell = collections.Counter(s["cell"] for s in staged)
    reps = collections.defaultdict(set)
    for s in staged:
        reps[s["cell"]].add(s["host"])
    summary = {"staged": staged, "void_attempts": void, "partial_attempts": partial, "missing": missing,
               "per_cell": {c: {"runs": by_cell[c], "repeats": sorted(reps[c])} for c in sorted(by_cell)}}
    with open(os.path.join(a.stage, "staging.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for c, v in summary["per_cell"].items():
        print(f"{c}: {v['runs']} runs, repeats {', '.join(v['repeats'])}")
    print(f"void attempts: {len(void)}; partial (running or cut short): {len(partial)}; missing turn logs: {len(missing)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
