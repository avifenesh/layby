"""The simulator's prefill-phase reserve (vLLM 0.31.0 sliding windows). At an 18.65 GB Gemma 4 GPU KV pool, a 30K
prompt that arrives while another long request decodes must wait for it to finish: its prefill needs the window
plus two 8,192-token chunks in every sliding group (14.27 GB) on top of its global KV (pilot 2's Gemma C0 run:
'Running: 1 reqs'). With the decode reserve alone it is admitted at once."""
from sim.engine_sim import Params, simulate


def run(prefill):
    W = {"sessions": [
        dict(id=0, start=0.0, turns=[dict(kind="tool", new_tokens=30000, out_len=400, gap_after=None)]),
        dict(id=1, start=9.0, turns=[dict(kind="tool", new_tokens=30000, out_len=16, gap_after=None)])]}
    p = Params(gpu_tokens=18_653_904_896 // 81920, cpu_tokens=0, budget=8192, f0=1 / 4000, att=0.0, t0=0.03,
               bw=1e15, req_tokens=10560, req_tokens_prefill=prefill)
    rows, _ = simulate(W, None, p)
    a = next(r for r in rows if r["session"] == 0)
    b = next(r for r in rows if r["session"] == 1)
    return a, b


def test_prefill_reserve_waits_for_the_decoding_request():
    a, b = run(174240)
    assert b["t_first"] >= a["t_done"]    # first token only after A has finished


def test_decode_reserve_alone_admits_at_once():
    a, b = run(None)
    assert b["t_first"] < a["t_done"]
