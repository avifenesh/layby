"""Unit tests for zero-laya Layby-Dwell v2 inference, feature parity, and sidecar improvements.

Run: PYTHONPATH=. python tests/test_dwell_v2_sidecar.py
"""
import json
import os
import sys

import numpy as np

# Explicitly block laya to guarantee Layby-Dwell v2 has ZERO dependency on the legacy laya package
sys.modules["laya"] = None  # type: ignore[assignment]

from layby_dwell import Dwell, DwellV2
from layby_dwell.v2 import EDGES, NUM_COLS, is_v2_config, logits_to_survival, state_to_v2_features
from layby.sidecar.proxy import OnlineCalib
from layby.sidecar.scorer import Scorer
from layby.sidecar.session import Tracker


def test_state_to_v2_features():
    st = {
        "kind": "tool",
        "model": "claude-opus-4-6",
        "tool": "Bash",
        "program": "pytest",
        "arg_chars": 80,
        "ctx_tokens": 12000,
        "turn": 4,
        "session_age_s": 120.0,
        "hour": 12,
        "dow": 2,
        "tool_history": {"n": 10, "mean_s": 2.5, "p90_s": 10.0, "share_under_1s": 0.4},
        "meta": {
            "cmd_chars": 32,
            "timeout_s": 60,
            "n_tools": 1,
            "background": False,
            "since_prev_s": 5.0,
            "prev_gap_s": 1.2,
            "prev_kind": "tool",
            "prev_same_kind_s": 1.2,
            "ewma_same_kind_s": 1.8,
            "n_same_kind": 3,
            "program_history": {"n": 5, "mean_s": 4.0, "p90_s": 15.0, "share_under_1s": 0.2},
            "user_kind_history": {"n": 20, "mean_s": 3.0, "p90_s": 12.0, "share_under_1s": 0.3},
        },
        "tool_call": 'Bash {"command": "pytest -q"}',
        "assistant": "Running unit tests now.",
        "user": "Please run the tests.",
    }
    row, txt, kind = state_to_v2_features(json.dumps(st))
    assert row.shape == (len(NUM_COLS),)
    assert kind == "tool"
    assert abs(row[NUM_COLS.index("ctx_log")] - np.log1p(12000.0)) < 1e-5
    assert row[NUM_COLS.index("ctx_missing")] == 0.0
    assert row[NUM_COLS.index("is_tool_kind")] == 1.0
    assert row[NUM_COLS.index("prev_is_tool")] == 1.0
    assert "kind=tool tool=Bash prog=pytest model=claude-opus-4-6" in txt
    assert "TOOL_CALL: Bash" in txt and "ASSISTANT: Running" in txt and "USER: Please run" in txt

    # Missing ctx <= 0 normalized cleanly to NaN + ctx_missing=1
    row2, _, kind2 = state_to_v2_features({"kind": "workflow", "ctx_tokens": 0})
    assert kind2 == "human"
    assert np.isnan(row2[NUM_COLS.index("ctx_log")])
    assert row2[NUM_COLS.index("ctx_missing")] == 1.0


def test_tracker_bounded_lru_and_return_tracking():
    t = [100.0]
    tr = Tracker(clock=lambda: t[0], max_tips=4, max_hist=10)
    tips = []
    for i in range(6):
        body = {"model": "m", "messages": [{"role": "user", "content": f"hi {i}"}]}
        sess = tr.on_request(body, f"u{i}")
        tr.on_response(sess, body, {"content": f"hello {i}", "tool_calls": []}, {"prompt_tokens": 10, "completion_tokens": 5}, f"u{i}", key=f"k{i}")
        tips.append((body, f"hello {i}", f"u{i}"))
        t[0] += 1.0
    assert len(tr.by_tip) == 4 and tr.tips_evicted == 2, (len(tr.by_tip), tr.tips_evicted)
    # Latest session (i=5) links and records exact gap in last_returned
    t[0] += 12.5
    b5, r5, u5 = tips[-1]
    body2 = {"model": "m", "messages": b5["messages"] + [{"role": "assistant", "content": r5}, {"role": "user", "content": "next"}]}
    s5 = tr.on_request(body2, u5)
    assert s5.last_returned is not None
    rk, rgap, rkind = s5.last_returned
    assert rk == "k5" and abs(rgap - 13.5) < 1e-6 and rkind == "human", s5.last_returned


def test_online_calib():
    oc = OnlineCalib({"_all": 1.0, "tool": 1.0, "human": 1.0}, enabled=True, window=200)
    rng = np.random.default_rng(0)
    # Overconfident logits (need higher temperature > 1.0)
    for i in range(60):
        lg = rng.normal(0.0, 4.0, size=16).astype(np.float32)
        S = logits_to_survival(lg[None, :], ["tool"], oc.temps)[0]
        oc.record_prediction(f"k{i}", lg, S, "tool")
        oc.observe_return(f"k{i}", float(rng.choice([0.3, 2.5, 15.0, 90.0])), "tool")
    sm = oc.summary()
    assert sm["by_kind"]["tool"]["n"] == 60
    assert sm["temps"]["tool"] > 1.0, f"expected online calib to raise T on overconfident logits: {sm}"
    # Exact bin-edge parity with training (side="left"): gap == 0.5 belongs to bin 0
    oc2 = OnlineCalib({"_all": 1.0}, enabled=False)
    z0 = np.zeros(16, dtype=np.float32)
    oc2.record_prediction("edge", z0, np.ones(15, dtype=np.float32), "tool")
    oc2.observe_return("edge", 0.5, "tool")
    assert oc2._obs["tool"][0][2] == 0, f"expected bin 0 for gap=0.5 (side='left'), got {oc2._obs['tool'][0][2]}"


def test_dwell_v2_real_checkpoint_zero_laya():
    ckpt_dir = None
    for cand in (
        os.path.expanduser("~/layby-data/runs/rung4_32m_all_corpora"),
        "/mnt/Data/layby-data/runs/rung4_32m_all_corpora",
    ):
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "best.safetensors")):
            ckpt_dir = cand
            break
    if ckpt_dir is None:
        print("skipping real checkpoint test (rung4_32m_all_corpora not on this host)")
        return
    cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    assert is_v2_config(cfg)
    dw = Dwell(ckpt_dir, device="cpu", precision="fp32")
    assert dw.model_version == 2
    states = [
        json.dumps({"kind": "tool", "tool": "Bash", "program": "git", "ctx_tokens": 16000, "turn": 3,
                    "tool_call": 'Bash {"command": "git status"}', "assistant": "Checking repo."}),
        json.dumps({"kind": "human", "ctx_tokens": 24000, "turn": 5,
                    "assistant": "I finished refactoring the module. Ready for your review!", "user": "Refactor it."}),
    ]
    S = dw.survival(states, ["tool", "human"])
    assert S.shape == (2, len(EDGES))
    assert np.all((S >= 0.0) & (S <= 1.0))
    # Monotonic non-increasing survival across horizons
    assert np.all(np.diff(S, axis=1) <= 1e-6), f"non-monotonic survival: {S}"
    # Tool git status should return much sooner than human review handoff!
    med = dw.median_return(S)
    assert med[0] < med[1], f"expected tool median ({med[0]}s) < human handoff ({med[1]}s)"
    # Also test sidecar Scorer auto-detection (zero laya!)
    sc = Scorer(os.path.join(ckpt_dir, "best.safetensors"), os.path.join(ckpt_dir, "config.json"), device="cpu")
    assert sc.model_version == 2
    S2 = sc.score(states, ["tool", "human"])
    assert np.allclose(S, S2, atol=1e-5)


def test_dwell_v2_config_only_hf_bundle():
    """Verify a self-contained HF bundle (encoder/config.json only, sanitized config.json) loads offline with exact parity."""
    import shutil
    import tempfile

    ckpt_dir = None
    for cand in (
        os.path.expanduser("~/layby-data/runs/rung4_32m_all_corpora"),
        "/mnt/Data/layby-data/runs/rung4_32m_all_corpora",
    ):
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "best.safetensors")):
            ckpt_dir = cand
            break
    if ckpt_dir is None:
        print("skipping HF bundle test (rung4_32m_all_corpora not on this host)")
        return

    src_cfg = json.load(open(os.path.join(ckpt_dir, "config.json")))
    enc_src = src_cfg["args"]["encoder"]
    if not os.path.isdir(enc_src):
        enc_src = "/mnt/Data/layby-data/models/ettin-encoder-32m"

    with tempfile.TemporaryDirectory(prefix="layby_hf_bundle_") as td:
        os.makedirs(os.path.join(td, "encoder"))
        os.makedirs(os.path.join(td, "tokenizer"))
        os.symlink(os.path.join(ckpt_dir, "best.safetensors"), os.path.join(td, "model.safetensors"))
        shutil.copy2(os.path.join(enc_src, "config.json"), os.path.join(td, "encoder", "config.json"))
        for tf in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
            p = os.path.join(enc_src, tf)
            if os.path.exists(p):
                shutil.copy2(p, os.path.join(td, "tokenizer", tf))
        clean_cfg = {
            "model_version": 2,
            "arch": src_cfg["args"]["arch"],
            "mode": src_cfg["args"]["mode"],
            "max_len": src_cfg["args"]["max_len"],
            "aux_prefix": False,
            "base_encoder": "jhu-clsp/ettin-encoder-32m",
            "encoder": "encoder",
            "edges": src_cfg["edges"],
            "num_cols": src_cfg["num_cols"],
            "num_mean": src_cfg["num_mean"],
            "num_std": src_cfg["num_std"],
            "temperature": src_cfg["temperature"],
            "layer_mix_weights": src_cfg["layer_mix_weights"],
        }
        with open(os.path.join(td, "config.json"), "w") as f:
            json.dump(clean_cfg, f, indent=2)

        old_offline = os.environ.get("HF_HUB_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            dw_bundle = Dwell(td, device="cpu", precision="fp32")
            dw_ref = Dwell(ckpt_dir, device="cpu", precision="fp32")
        finally:
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline

        states = [
            json.dumps({"kind": "tool", "tool": "Bash", "program": "pytest", "ctx_tokens": 12000, "turn": 3,
                        "tool_call": 'Bash {"command": "pytest -q"}', "assistant": "Running tests."}),
            json.dumps({"kind": "human", "ctx_tokens": 24000, "turn": 5,
                        "assistant": "Finished the PR, ready for review.", "user": "Open the PR."}),
        ]
        S_bundle = dw_bundle.survival(states)
        S_ref = dw_ref.survival(states, ["tool", "human"])
        assert np.allclose(S_bundle, S_ref, atol=1e-6), f"bundle mismatch: max diff {np.max(np.abs(S_bundle - S_ref))}"


if __name__ == "__main__":
    test_state_to_v2_features()
    test_tracker_bounded_lru_and_return_tracking()
    test_online_calib()
    test_dwell_v2_real_checkpoint_zero_laya()
    test_dwell_v2_config_only_hf_bundle()
    print("dwell v2 & sidecar tests ok (zero laya dependency verified)")
