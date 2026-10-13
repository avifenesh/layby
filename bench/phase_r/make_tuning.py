#!/usr/bin/env python3
"""Phase R baseline tuning cells (BENCH_PLAN section 8) for Qwen3.8-27B and Gemma 4 31B on vLLM 0.31.0, written to
engine/phase_r/cells/tune/, and the tuned settings the main cells carry (cells/tune/tuned.json, read by make_cells.py).

Procedure, the same for every tuned arm: successive halving per model and disk at heavy load. Stage 1 runs every grid
point on the 20-minute tuning workload ({model}.tuning_20min), stage 2 the top third on the 40-minute one
({model}.tuning_40min) when that is more than 2 points, stage 3 the top 2 on the full tuning workload
({model}.tuning); each stage runs its workload to completion. Ranking (tune_rank.py): p95 TTFT on resumed turns,
failed and censored turns +inf, ties by p50 then grid order. Disks: slow = the network_ssd under io.max, fast = NRD
(make_cells.disk). One cell is one block: one family's points of one stage on one disk, every grid point an arm
labelled FAMILY_KNOBS (no dots: orchestrate.py) with its runner env, the family's pilot RSS (rss_of) and its knobs.

Grids (section 8 and the decisions since), in grid order (first knob slowest):
  C0  slow only, once per model: eviction {lru, arc} x chunked-prefill budget {4096, 8192} on Qwen (SERVE_ARGS
      --max-num-batched-tokens); eviction {lru, arc} only on Gemma (its 1,024 budget is derived and fixed in its
      serve args). The winner's eviction and budget carry to every other tuning cell and every main cell of the
      model (make_cells.apply_tuned), so C0 runs first: no other family's cell is written before C0 has a winner.
  T   per disk: blocks_per_chunk {1, 16} (BLOCKS_PER_CHUNK, vLLM's kv_connector_extra_config blocks_per_chunk) x fs
      threads {16, 32} (FS_THREADS). blocks_per_chunk is connector-wide: the CPU tier's chunk is 16 blocks too. On
      Qwen that is 16 x 51,380,224 B, so the 4,831,838,208 B CPU tier holds 5 chunks (80 blocks, 4.11 GB) where the
      plan's restore-beside-write minimum is 2 x 47 blocks in whole chunks (6 chunks), and an offload hit rounds down
      to 12,544 tokens (the Mamba alignment); on Gemma 206 chunks of 16 x 2,621,440 B, the minimum exactly. The cell's
      kv_chunk_bytes (the regime check's random-read size) stays the 1-block chunk, the smaller and stricter read.
  T2  per disk, after T on that disk: threshold {2, 3} (T2_MIN_HITS) x count mode {request, store} (T2_COUNT), decay
      600 s (T2_DECAY_S), on T's tuned blocks_per_chunk and fs threads.
  LM  per disk: chunk (Qwen {784, 1568}; Gemma {256, 512}; LM_CHUNK) x prefetch {default, retain} (code LM or LMR),
      direct I/O on (the runner's L2 adapter). The plan's eviction {LRU, ARC} cannot run: LMCache 0.5.5's server
      takes --eviction-policy LRU, IsolatedLRU or noop only (lmcache/v1/distributed/config.py), so eviction stays LRU.
      LMR also raises L1's eviction watermark to 0.95 (ratio 0.1), as run_arms.sh has run it since 5c24f29.
  CT  slow only (CT runs in slow cells only): no knob, one full-length run (stage 3 directly) with the model's
      measured-workload CT_ETA, on C0's carried settings, so it can be ranked for "best".
T, LM and CT wait for C0's winner; T2 on a disk waits for T's winner on that disk.

Cells: cells/tune/t-{qwen,gemma}-{family}-{slow,fast}-s{1,2,3}.json, cell id the file stem. Capacities from
capacities.json (GPU and CPU per model, disk per tuning workload); hard_cap_s 1.5 x the tuning workload's projected
C0 makespan (manifest c0_projection); everything else as make_cells.cell. Each cell's "tuning" section records the
model, family, disk, stage, workload, the full grid (labels in grid order), the knobs carried in (C0's, and T's for
T2) and any grid points the engine cannot run.

Modes:
  make_tuning.py stage1 WORKLOAD_DIR [--out engine/phase_r/cells]
      writes the first-stage cell of every (model, family, disk) whose prerequisites are in cells/tune/tuned.json and
      that is not tuned yet, and prints what waits on what
  make_tuning.py next WORKLOAD_DIR --ranking RANK.json [--out ...] [--allow-partial]
      from a tune_rank.py ranking: writes the next stage's cell (the advancing points), or, after stage 3, records the
      winner's knobs in cells/tune/tuned.json (CT has none to record). A partial ranking (a grid point not ranked) is
      refused unless --allow-partial: the plan reruns the whole block instead.
"""
import argparse
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_cells as M  # noqa: E402

STAGE_WORKLOAD = {1: "tuning_20min", 2: "tuning_40min", 3: "tuning"}
FAMILIES = ("C0", "T", "LM", "CT", "T2")          # the order prerequisites allow
DISKS = {"C0": ("slow",), "CT": ("slow",), "T": ("slow", "fast"), "LM": ("slow", "fast"), "T2": ("slow", "fast")}
GRID = {
    "qwen38_27b": {"c0_budget": (4096, 8192), "lm_chunk": (784, 1568)},
    "gemma4_31b": {"c0_budget": (None,), "lm_chunk": (256, 512)},
}
T2_DECAY_S = 600
DROPPED = {"LM": ["eviction ARC: LMCache 0.5.5's --eviction-policy takes LRU, IsolatedLRU or noop only "
                  "(lmcache/v1/distributed/config.py); eviction stays LRU"]}


def grid(key, family, t_knobs=None):
    """[(label, knobs)] of one family's grid in grid order. t_knobs: T's tuned knobs on the disk (T2 only)."""
    g = GRID[key]
    if family == "C0":
        return [(f"C0_{ev}" + (f"_b{b}" if b else ""), {"eviction": ev, "budget": b})
                for ev, b in itertools.product(("lru", "arc"), g["c0_budget"])]
    if family == "T":
        return [(f"T_b{b}_t{t}", {"blocks_per_chunk": b, "fs_threads": t})
                for b, t in itertools.product((1, 16), (16, 32))]
    if family == "T2":
        if not t_knobs:
            raise ValueError("T2's grid sits on T's tuned disk knobs")
        base = {"blocks_per_chunk": t_knobs["blocks_per_chunk"], "fs_threads": t_knobs["fs_threads"]}
        return [(f"T2_n{n}_{cm}", dict(base, min_hits=n, count=cm, decay_s=T2_DECAY_S))
                for n, cm in itertools.product((2, 3), ("request", "store"))]
    if family == "LM":
        return [(f"LM_c{c}_{p}", {"chunk": c, "prefetch": p})
                for c, p in itertools.product(g["lm_chunk"], ("default", "retain"))]
    if family == "CT":
        return [("CT", {})]
    raise ValueError(f"family {family!r}")


def first_stage(family):
    return 3 if family == "CT" else 1


def tuned_entry(tuned, key, family, disk):
    t = (tuned or {}).get(key) or {}
    return t.get("C0") if family == "C0" else (t.get(family) or {}).get(disk)


def needs(family, disk):
    """The tuned entries a family's cells carry: [(family, disk)]."""
    return {"C0": [], "T2": [("C0", "slow"), ("T", disk)]}.get(family, [("C0", "slow")])


def carried_of(tuned, key, family, disk):
    """{family: knobs} carried into this family's cells, or None when a prerequisite is not tuned yet."""
    out = {}
    for f, d in needs(family, disk):
        e = tuned_entry(tuned, key, f, d)
        if not e:
            return None
        out[f] = e["knobs"]
    return out


class Workloads:
    def __init__(self, wdir):
        self.dir = wdir
        self.C = json.load(open(os.path.join(wdir, "capacities.json")))
        M.MAXLEN.update({k: int(v["max_model_len"]) for k, v in self.C["models"].items()})

    def man(self, wl):
        return json.load(open(os.path.join(self.dir, f"{wl}.manifest.json")))

    def caps(self, key, wl):
        mc = self.C["models"][key]
        return {"gpu_kv_bytes": str(mc["gpu_kv_bytes"]), "cpu_bytes": str(mc["cpu_bytes"]),
                "disk_bytes": str(self.C["workloads"][wl]["disk_bytes"])}

    def ct_eta(self, key):
        return self.C["workloads"][f"{key}.measured"]["ct_eta"]


def cell_name(key, family, disk, stage):
    return f"t-{M.MODELS[key]['tag']}-{family.lower()}-{disk}-s{stage}"


def build(W, key, family, disk, stage, carried, labels=None):
    """One tuning cell (module docstring); labels: the advancing points (stages 2 and 3), else the whole grid."""
    if disk not in DISKS[family]:
        raise ValueError(f"{family} is not tuned on the {disk} disk")
    pts = grid(key, family, carried.get("T"))
    full = [lab for lab, _ in pts]
    if labels is not None:
        miss = set(labels) - set(full)
        if miss:
            raise ValueError(f"{sorted(miss)} not in the {family} grid {full}")
        pts = [(lab, k) for lab, k in pts if lab in labels]
    eta = W.ct_eta(key)
    arms = []
    for lab, kn in pts:
        if family == "C0":
            code, env = "C0", M.c0_env(key, kn)
        elif family == "CT":
            code, env = "CT", {"CT_ETA": str(eta)}
        else:
            code, env = M.family_arm(family, kn)
        arms.append({"label": lab, "code": code, "env": env, "rss_of": family, "knobs": kn})
    wl = f"{key}.{STAGE_WORKLOAD[stage]}"
    man = W.man(wl)
    name = cell_name(key, family, disk, stage)
    c = M.cell(name, key, man, W.caps(key, wl), disk, "heavy", wl, arms, eta,
               "capacities.json (GPU and CPU per model, disk per tuning workload)")
    c["pilot"] = "../" + c["pilot"]
    if "C0" in carried:
        c["runner"]["env"].update(M.c0_env(key, carried["C0"]))
    c["_comment"] = (
        f"Generated by bench/phase_r/make_tuning.py (its docstring has every rule): {family} tuning stage {stage} on "
        f"the {disk} disk, workload {wl}; hard_cap_s = 1.5 x its projected C0 makespan "
        f"{c['projected_c0_makespan_s']:g} s. Carried in: {json.dumps(carried, sort_keys=True)}.")
    c["tuning"] = {"model": key, "family": family, "disk": disk, "stage": stage, "workload": wl, "grid": full,
                   "carried": carried, "dropped": DROPPED.get(family, [])}
    return c


def write(out, c):
    p = os.path.join(out, "tune", f"{c['cell']}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(c, f, indent=1)
        f.write("\n")
    return p


def tuned_path(out):
    return os.path.join(out, "tune", "tuned.json")


def load_tuned(out):
    p = tuned_path(out)
    return json.load(open(p)) if os.path.exists(p) else {}


def stage1(W, out):
    tuned = load_tuned(out)
    written, waits = [], []
    for key in M.MODELS:
        for fam in FAMILIES:
            for disk in DISKS[fam]:
                if tuned_entry(tuned, key, fam, disk):
                    print(f"tuned: {key} {fam} {disk} ({tuned_entry(tuned, key, fam, disk)['label']})")
                    continue
                carried = carried_of(tuned, key, fam, disk)
                if carried is None:
                    miss = [f"{f} {d}" for f, d in needs(fam, disk) if not tuned_entry(tuned, key, f, d)]
                    waits.append(f"{key} {fam} {disk} waits for {', '.join(miss)}")
                    continue
                written.append(write(out, build(W, key, fam, disk, first_stage(fam), carried)))
    for p in written:
        print(p)
    for w in waits:
        print("wait:", w)
    return 0


def next_stage(W, out, ranking, allow_partial=False):
    rk = json.load(open(ranking))
    tun = rk["tuning"]
    key, fam, disk, stage = tun["model"], tun["family"], tun["disk"], int(tun["stage"])
    if not rk["complete"] and not allow_partial:
        print(f"{rk['cell']}: partial ranking, not ranked: {[u['label'] for u in rk['unranked']]}; the plan reruns "
              f"the block (or pass --allow-partial)")
        return 2
    tuned = load_tuned(out)
    now = carried_of(tuned, key, fam, disk)
    if now != tun["carried"]:
        print(f"{rk['cell']}: carried settings changed since this stage was written: {tun['carried']} -> {now}")
        return 4
    if rk["next_stage"]:
        if not rk["advance"]:
            print(f"{rk['cell']}: nothing ranked, nothing advances")
            return 2
        print(write(out, build(W, key, fam, disk, int(rk["next_stage"]), tun["carried"], rk["advance"])))
        return 0
    win = rk["winner"]
    if stage != 3 or not win:
        print(f"{rk['cell']}: stage {stage} has no winner to record")
        return 2
    arm = next(a for a in rk["arms"] if a["label"] == win)
    if fam == "CT":
        print(f"{key} CT: p95 {arm['p95']}, p50 {arm['p50']} (no knob to carry)")
        return 0
    if arm["p95"] == "inf":
        print(f"warning: {key} {fam} {disk} winner {win} has p95 +inf (censored or failed turns)")
    entry = {"label": win, "knobs": arm["knobs"], "p50": arm["p50"], "p95": arm["p95"],
             "ranking": os.path.relpath(os.path.abspath(ranking), os.path.join(out, "tune")), "cell": rk["cell"],
             "block": {k: v for k, v in rk["block"].items() if k != "dir"}, "carried": tun["carried"]}
    t = tuned.setdefault(key, {})
    if fam == "C0":
        t["C0"] = entry
    else:
        t.setdefault(fam, {})[disk] = entry
    tuned["_comment"] = ("Tuned settings (bench/phase_r/make_tuning.py next, from the stage-3 rankings of "
                         "tune_rank.py): per model, C0's knobs (carried to every cell) and T, T2, LM per disk; "
                         "make_cells.py --tuned reads it.")
    p = tuned_path(out)
    with open(p, "w") as f:
        json.dump(tuned, f, indent=1, sort_keys=True)
        f.write("\n")
    print(f"{key} {fam}{'' if fam == 'C0' else ' ' + disk}: {win} {arm['knobs']} -> {p}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=("stage1", "next"))
    ap.add_argument("workloads")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "engine", "phase_r", "cells"))
    ap.add_argument("--ranking", help="tune_rank.py output (mode next)")
    ap.add_argument("--allow-partial", action="store_true")
    a = ap.parse_args(argv)
    W = Workloads(a.workloads)
    out = os.path.abspath(a.out)
    if a.mode == "stage1":
        return stage1(W, out)
    if not a.ranking:
        ap.error("next needs --ranking")
    return next_stage(W, out, a.ranking, a.allow_partial)


if __name__ == "__main__":
    sys.exit(main())
