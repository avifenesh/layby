#!/usr/bin/env python3
"""Port Nebius tuning cells (bench/phase_r/make_tuning.py output) to a --host profile (BENCH_PLAN draft 6.8: the
close-call stage-3 reruns on wrx). Same arms and knobs; the host's slots and io.max (make_cells.host_disk), vLLM bound
to 127.0.0.1, the projection and hard cap scaled by the host's prefill (prefill-bound workloads), and readable labels
(LM_c512_retain -> lmcache-c512-retain). Writes engine/phase_r/cells/HOST/tune/CELL.json.

Usage: host_tune_cells.py HOST CELL.json [CELL.json ...]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_cells as mc   # noqa: E402

KEY = {"Qwen/Qwen3.8-27B-FP8": "qwen38_27b", "google/gemma-4-31B-it": "gemma4_31b"}
PREFIX = {"LM": "lmcache", "T2": "write-hot", "T": "write-through", "C0": "no-disk", "CT": "continuum"}


def readable(label):
    head, _, rest = label.partition("_")
    name = PREFIX.get(head, head.lower())
    return f"{name}-{rest.replace('_', '-')}" if rest else name


def port(host, path):
    c = json.load(open(path))
    key = KEY[c["model"]]
    k = mc.MODELS[key]["prefill"] / mc.HOST["prefill_tok_s"][key]
    c["projected_c0_makespan_s"] = round(c["projected_c0_makespan_s"] * k, 1)
    cap = int(math.ceil(1.5 * c["projected_c0_makespan_s"]))
    c["hard_cap_s"] = cap
    env = c["runner"]["env"]
    env["REPLAY_TIMEOUT"] = str(cap + 600)
    env["SERVE_ARGS"] = f"{env['SERVE_ARGS']} {mc.HOST['serve_extra']}".strip()
    c["runner"]["timeout_s"] = cap + 600 + 1800
    mc.host_disk(c["regime"], c, key)
    for a in c["arms"]:
        a.setdefault("rss_of", "LM" if a["code"] in ("LM", "LMR") else a["code"])
        a["label"] = readable(a["label"])
    if "tuning" in c:   # tune_rank.py checks the arm labels against the grid
        c["tuning"]["grid"] = [readable(g) for g in c["tuning"]["grid"]]
    c["plan"] = f"BENCH_PLAN draft 6.8 (host {host}; close-call rerun of {c['cell']})"
    c["_comment"] = f"{c.get('_comment', '')} Ported to host {host} by bench/phase_r/host_tune_cells.py."
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "engine", "phase_r", "cells", host,
                       "tune", os.path.basename(path))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    c["pilot"] = os.path.join("..", os.path.basename(c["pilot"]))
    with open(out, "w") as f:
        json.dump(c, f, indent=1)
        f.write("\n")
    print(os.path.abspath(out))


def main():
    host, cells = sys.argv[1], sys.argv[2:]
    mc.HOST.update(json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hosts", f"{host}.json"))))
    for p in cells:
        port(host, p)


if __name__ == "__main__":
    main()
