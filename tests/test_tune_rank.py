"""CPU tests of the baseline-tuning tools (BENCH_PLAN section 8): bench/phase_r/tune_rank.py ranks a tuning block by
p95 TTFT on resumed turns with failed and censored turns as +inf, breaks ties by p50 then grid order, advances the
top third (when more than 2) or the top 2, and leaves arms with an invalid or missing verdict unranked;
bench/phase_r/make_tuning.py writes C0's cells first, the next stage from a ranking and the winner into tuned.json,
which make_cells.py carries into the main cells. Synthetic orchestrator outputs and workload manifests only.

  python -m pytest -q tests/test_tune_rank.py      (needs numpy; about a second)
"""
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench", "phase_r"))
import make_cells as M  # noqa: E402
import make_tuning as MT  # noqa: E402
import tune_rank as R  # noqa: E402

INF = float("inf")
HOST = "h1"


def row(s, t, ttft=0.1, status="served"):
    r = dict(session=s, turn=t, kind="tool", prev_gap=5.0 if t else None, prev_kind="tool" if t else None)
    if status == "served":
        r.update(ttft=ttft, e2e=ttft + 0.1, out_tokens=4)
    elif status == "failed":
        r.update(ttft=None, e2e=None, failed=True, error="HTTP 500")
    else:
        r.update(ttft=None, e2e=None, censored=True)
    return r


def rows(ttfts, statuses=None, first_turn_ttft=0.05):
    """One session: turn 0 (not resumed) then one resumed turn per value."""
    out = [row("s0", 0, first_turn_ttft)]
    for i, x in enumerate(ttfts, 1):
        out.append(row("s0", i, x, (statuses or {}).get(i, "served")))
    return out


def spec(labels, grid=None, stage=1, cell="t-qwen-t-slow-s1", family="T"):
    return {"cell": cell, "arms": [{"label": lab, "code": family, "env": {}, "knobs": {"i": i}}
                                   for i, lab in enumerate(labels)],
            "tuning": {"model": "qwen38_27b", "family": family, "disk": "slow", "stage": stage,
                       "workload": "w", "grid": list(grid or labels), "carried": {}, "dropped": []}}


def block(root, cell, arms, attempt=1, host=HOST):
    """Orchestrator outputs: arms {label: (rows or None, valid or None)}; None rows: no row file, None valid: no
    verdict (the attempt stopped before the arm)."""
    for pos, (lab, (rws, valid)) in enumerate(arms.items()):
        d = os.path.join(root, cell, host, "r1", f"a{attempt}", f"{pos:02d}.{lab}")
        os.makedirs(d, exist_ok=True)
        pre = os.path.join(d, f"{cell}.{lab}.{host}")
        if rws is not None:
            with open(pre + ".jsonl", "w") as f:
                for r in rws:
                    f.write(json.dumps(r) + "\n")
                f.write(json.dumps({"wall_s": 1.0, "served": len(rws)}) + "\n")
        if valid is not None:
            with open(pre + ".verdict.json", "w") as f:
                json.dump({"valid": valid, "reasons": [] if valid else ["engine crash: rc 86"]}, f)


def labels_of(res):
    return [a["label"] for a in res["arms"]]


# --- ranking ---------------------------------------------------------------------------------------------------------
def test_failed_and_censored_turns_rank_as_inf(tmp_path):
    # 19 fast turns and 1 censored: p95 sits between 0.1 and +inf, so it is +inf and loses to a slow arm that served
    # every turn; a failed turn counts the same; turn 0 (not resumed) is ignored even when it is the slowest
    fast = [0.1] * 20
    s = spec(["T_a", "T_b", "T_c"])
    block(str(tmp_path), s["cell"], {
        "T_a": (rows(fast, {20: "censored"}), True),
        "T_b": (rows([5.0] * 20, first_turn_ttft=900.0), True),
        "T_c": (rows(fast, {3: "failed"}), True)})
    res = R.collect(s, str(tmp_path))
    assert labels_of(res) == ["T_b", "T_a", "T_c"]
    by = {a["label"]: a for a in res["arms"]}
    assert by["T_a"]["p95"] == "inf" and by["T_c"]["p95"] == "inf" and by["T_b"]["p95"] == 5.0
    assert (by["T_a"]["censored"], by["T_c"]["failed"], by["T_b"]["n"]) == (1, 1, 20)
    assert by["T_a"]["p50"] == 0.1


def test_ties_break_by_p50_then_grid_order(tmp_path):
    # both p95 +inf: the lower p50 wins; identical rows: the earlier grid point wins, whatever the cell's arm order
    # or the labels' sort order
    grid = ["T_z", "T_y", "T_x", "T_w"]
    s = spec(["T_w", "T_x", "T_y", "T_z"], grid=grid)
    cens = {19: "censored", 20: "censored"}
    block(str(tmp_path), s["cell"], {
        "T_w": (rows([1.0] * 20), True),
        "T_x": (rows([1.0] * 20), True),
        "T_y": (rows([0.5] * 20, cens), True),
        "T_z": (rows([0.9] * 20, cens), True)})
    res = R.collect(s, str(tmp_path))
    # finite p95 1.0 (T_x, T_w: grid order puts T_x first), then the +inf pair by p50 (T_y 0.5 before T_z 0.9)
    assert labels_of(res) == ["T_x", "T_w", "T_y", "T_z"]
    assert [a["rank"] for a in res["arms"]] == [1, 2, 3, 4]


@pytest.mark.parametrize("stage,n,want", [
    (1, 2, (3, 2)), (1, 4, (3, 2)), (1, 6, (3, 2)), (1, 7, (2, 3)), (1, 8, (2, 3)), (1, 12, (2, 4)),
    (2, 3, (3, 2)), (3, 2, (None, 1))])
def test_top_third_rule(stage, n, want):
    assert R.advance_rule(stage, n) == want


def test_stage_one_advances_top_third_of_eight(tmp_path):
    labs = [f"LM_{i}" for i in range(8)]
    s = spec(labs, family="LM")
    block(str(tmp_path), s["cell"], {lab: (rows([1.0 + i] * 20), True) for i, lab in enumerate(reversed(labs))})
    res = R.collect(s, str(tmp_path))
    assert (res["next_stage"], res["advance"], res["winner"]) == (2, ["LM_7", "LM_6", "LM_5"], None)
    s4 = spec(labs[:4])
    block(str(tmp_path / "b"), s4["cell"], {lab: (rows([1.0 + i] * 20), True) for i, lab in enumerate(labs[:4])})
    res = R.collect(s4, str(tmp_path / "b"))
    assert (res["next_stage"], res["advance"]) == (3, ["LM_0", "LM_1"])
    s3 = spec(labs[:2], stage=3)
    block(str(tmp_path / "c"), s3["cell"], {lab: (rows([2.0 - i] * 20), True) for i, lab in enumerate(labs[:2])})
    res = R.collect(s3, str(tmp_path / "c"))
    assert (res["next_stage"], res["advance"], res["winner"]) == (None, [], "LM_1")


def test_invalid_or_missing_verdict_is_reported_not_ranked(tmp_path):
    s = spec(["T_a", "T_b", "T_c", "T_d"])
    block(str(tmp_path), s["cell"], {
        "T_a": (rows([9.0] * 20), True),
        "T_b": (rows([0.1] * 20), False),            # fastest, but its verdict is invalid
        "T_c": (rows([0.2] * 20), None)})            # never judged; T_d never ran
    res = R.collect(s, str(tmp_path))
    assert labels_of(res) == ["T_a"]
    un = {u["label"]: u for u in res["unranked"]}
    assert set(un) == {"T_b", "T_c", "T_d"} and un["T_b"]["reasons"] == ["engine crash: rc 86"]
    assert res["complete"] is False


def test_common_turns_and_latest_attempt(tmp_path):
    # attempt 1 (excluded block) is ignored; in attempt 2 an arm that lost a row shrinks the common set for every arm
    s = spec(["T_a", "T_b"])
    block(str(tmp_path), s["cell"], {"T_a": (rows([0.1] * 20), True), "T_b": (rows([9.0] * 20), False)})
    lost = rows([0.3] * 20)
    del lost[5]
    block(str(tmp_path), s["cell"], {"T_a": (rows([0.1] * 19 + [50.0]), True), "T_b": (lost, True)}, attempt=2)
    res = R.collect(s, str(tmp_path))
    assert res["block"]["attempt"] == 2 and res["complete"]
    by = {a["label"]: a for a in res["arms"]}
    assert by["T_a"]["n"] == by["T_b"]["n"] == 19
    assert (by["T_a"]["missing"], by["T_b"]["missing"]) == (1, 0)
    assert labels_of(res) == ["T_b", "T_a"]           # T_a's 50 s turn is in the common set


def test_two_hosts_at_the_same_attempt_need_host(tmp_path):
    s = spec(["T_a"])
    block(str(tmp_path), s["cell"], {"T_a": (rows([0.1] * 5), True)}, host="h1")
    block(str(tmp_path), s["cell"], {"T_a": (rows([0.2] * 5), True)}, host="h2")
    with pytest.raises(ValueError):
        R.collect(s, str(tmp_path))
    assert R.collect(s, str(tmp_path), host="h2")["arms"][0]["p50"] == 0.2


# --- the tuning flow -------------------------------------------------------------------------------------------------
@pytest.fixture
def workloads(tmp_path):
    """Minimal capacities.json and manifests for both models' tuning and measured workloads."""
    d = tmp_path / "wl"
    d.mkdir()
    C = {"models": {}, "workloads": {}}
    for key in M.MODELS:
        C["models"][key] = {"max_model_len": 32768, "gpu_kv_bytes": 2 << 30, "cpu_bytes": 4 << 30}
        for w in ("measured", "tuning_20min", "tuning_40min", "tuning"):
            C["workloads"][f"{key}.{w}"] = {"disk_bytes": 100 << 30, "ct_eta": 0.034}
            (d / f"{key}.{w}.manifest.json").write_text(json.dumps(
                {"sha256": "0" * 64, "load": {"sessions": 9, "turns": 200}, "c0_projection": {"makespan_s": 1000.0}}))
    (d / "capacities.json").write_text(json.dumps(C))
    return str(d)


def run_stage(tmp_path, cell_path, order):
    """Rank a written cell on synthetic rows: the arms in `order` get p95 1, 2, 3, ..."""
    c = json.load(open(cell_path))
    runs = tmp_path / "runs"
    block(str(runs), c["cell"], {lab: (rows([1.0 + order.index(lab)] * 20), True) for lab in order})
    res = R.collect(c, str(runs))
    p = tmp_path / f"{c['cell']}.rank.json"
    p.write_text(json.dumps(res))
    return str(p)


def test_c0_first_then_carried(tmp_path, workloads):
    out = str(tmp_path / "cells")
    W = MT.Workloads(workloads)
    MT.stage1(W, out)
    tune = os.path.join(out, "tune")
    assert sorted(os.listdir(tune)) == ["t-gemma-c0-slow-s1.json", "t-qwen-c0-slow-s1.json"]
    q = json.load(open(os.path.join(tune, "t-qwen-c0-slow-s1.json")))
    assert q["tuning"]["grid"] == ["C0_lru_b4096", "C0_lru_b8192", "C0_arc_b4096", "C0_arc_b8192"]
    a = {x["label"]: x for x in q["arms"]}
    assert a["C0_arc_b4096"]["env"]["EVICTION"] == "arc"
    assert a["C0_arc_b4096"]["env"]["SERVE_ARGS"].endswith("--max-num-batched-tokens 4096")
    assert all(x["rss_of"] == "C0" and "." not in x["label"] for x in q["arms"])
    g = json.load(open(os.path.join(tune, "t-gemma-c0-slow-s1.json")))
    assert [x["label"] for x in g["arms"]] == ["C0_lru", "C0_arc"] and "SERVE_ARGS" not in g["arms"][0]["env"]
    # stage 1 (4 points) -> stage 3 with the top 2 -> the winner in tuned.json
    r1 = run_stage(tmp_path, os.path.join(tune, "t-qwen-c0-slow-s1.json"),
                   ["C0_arc_b4096", "C0_lru_b8192", "C0_lru_b4096", "C0_arc_b8192"])
    assert MT.next_stage(W, out, r1) == 0
    s3 = json.load(open(os.path.join(tune, "t-qwen-c0-slow-s3.json")))
    assert [x["label"] for x in s3["arms"]] == ["C0_lru_b8192", "C0_arc_b4096"]      # grid order
    assert s3["workload"]["path"].endswith("qwen38_27b.tuning.json") and s3["tuning"]["stage"] == 3
    (tmp_path / "s3").mkdir()
    r3 = run_stage(tmp_path / "s3", os.path.join(tune, "t-qwen-c0-slow-s3.json"), ["C0_arc_b4096", "C0_lru_b8192"])
    assert MT.next_stage(W, out, r3) == 0
    tuned = json.load(open(os.path.join(tune, "tuned.json")))
    assert tuned["qwen38_27b"]["C0"]["knobs"] == {"eviction": "arc", "budget": 4096}
    # now T, LM and CT of Qwen are written, carrying C0's eviction and budget; T2 still waits for T
    MT.stage1(W, out)
    names = sorted(os.listdir(tune))
    for n in ("t-qwen-t-slow-s1.json", "t-qwen-t-fast-s1.json", "t-qwen-lm-slow-s1.json", "t-qwen-lm-fast-s1.json",
              "t-qwen-ct-slow-s3.json"):
        assert n in names
    assert not any(n.startswith("t-qwen-t2") or n.startswith("t-gemma-t-") for n in names)
    t = json.load(open(os.path.join(tune, "t-qwen-t-fast-s1.json")))
    assert t["runner"]["env"]["EVICTION"] == "arc"
    assert t["runner"]["env"]["SERVE_ARGS"].endswith("--max-num-batched-tokens 4096")
    assert {x["label"]: x["env"]["BLOCKS_PER_CHUNK"] for x in t["arms"]}["T_b16_t32"] == "16"
    lm = json.load(open(os.path.join(tune, "t-qwen-lm-slow-s1.json")))
    assert {x["label"]: x["code"] for x in lm["arms"]} == {
        "LM_c784_default": "LM", "LM_c784_retain": "LMR", "LM_c1568_default": "LM", "LM_c1568_retain": "LMR"}
    # a main cell picks up C0's knobs everywhere and a tuned family's knobs on its own disk only
    tuned["qwen38_27b"]["T"] = {"fast": {"label": "T_b16_t32", "knobs": {"blocks_per_chunk": 16, "fs_threads": 32}}}
    tuned["qwen38_27b"]["LM"] = {"slow": {"label": "LM_c1568_retain", "knobs": {"chunk": 1568, "prefetch": "retain"}}}
    man = json.load(open(os.path.join(workloads, "qwen38_27b.measured.manifest.json")))
    caps = {"gpu_kv_bytes": "1", "cpu_bytes": "1", "disk_bytes": "1"}
    for regime in ("slow", "fast"):
        c = M.cell("x", "qwen38_27b", man, caps, regime, "heavy", "qwen38_27b.measured", ["C0", "T", "T2", "LM", "K"],
                   0.03, "test", tuned=(tuned, None, None))
        a = {x["label"]: x for x in c["arms"]}
        assert c["runner"]["env"]["EVICTION"] == "arc"
        assert c["runner"]["env"]["SERVE_ARGS"].endswith("--max-num-batched-tokens 4096")
        assert a["T2"]["env"] == M.T2_ENV                       # T2 not tuned: today's defaults
        if regime == "fast":
            assert a["T"]["env"] == {"BLOCKS_PER_CHUNK": "16", "FS_THREADS": "32"}
            assert a["LM"]["code"] == "LM"
        else:
            assert "BLOCKS_PER_CHUNK" not in (a["T"].get("env") or {})
            assert (a["LM"]["code"], a["LM"]["env"]["LM_CHUNK"]) == ("LMR", "1568")


def test_partial_ranking_does_not_advance(tmp_path, workloads):
    out = str(tmp_path / "cells")
    W = MT.Workloads(workloads)
    MT.stage1(W, out)
    c = json.load(open(os.path.join(out, "tune", "t-gemma-c0-slow-s1.json")))
    block(str(tmp_path / "runs"), c["cell"], {"C0_lru": (rows([1.0] * 20), True), "C0_arc": (rows([0.5] * 20), False)})
    p = tmp_path / "rank.json"
    p.write_text(json.dumps(R.collect(c, str(tmp_path / "runs"))))
    assert MT.next_stage(W, out, str(p)) == 2
    assert not os.path.exists(os.path.join(out, "tune", "t-gemma-c0-slow-s3.json"))
