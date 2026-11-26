"""The simulator's prefill-phase reserve (vLLM 0.31.0 sliding windows: per request, its in-flight prompt tokens). At an 18.65 GB Gemma 4 GPU KV pool, a 30K
prompt that arrives while another long request decodes must wait for it to finish: its prefill needs the window
plus two 8,192-token chunks in every sliding group (14.27 GB) on top of its global KV (pilot 2's Gemma C0 run:
'Running: 1 reqs'). With the decode reserve alone it is admitted at once."""
from sim.engine_sim import Params, simulate


GEMMA = dict(win_groups=5, win_lead=1023, win_block=16, win_page_tokens=32)   # 2,621,440 B pages / 81,920 B


def run(window, second=30000):
    W = {"sessions": [
        dict(id=0, start=0.0, turns=[dict(kind="tool", new_tokens=30000, out_len=400, gap_after=None)]),
        dict(id=1, start=9.0, turns=[dict(kind="tool", new_tokens=second, out_len=16, gap_after=None)])]}
    p = Params(gpu_tokens=18_653_904_896 // 81920, cpu_tokens=0, budget=8192, f0=1 / 4000, att=0.0, t0=0.03,
               bw=1e15, req_tokens=10560, **(GEMMA if window else {}))
    rows, _ = simulate(W, None, p)
    a = next(r for r in rows if r["session"] == 0)
    b = next(r for r in rows if r["session"] == 1)
    return a, b


def test_prefill_reserve_waits_for_the_decoding_request():
    a, b = run(True)
    assert b["t_first"] >= a["t_done"]    # first token only after A has finished


def test_decode_reserve_alone_admits_at_once():
    a, b = run(False)
    assert b["t_first"] < a["t_done"]


def test_a_short_prompt_holds_only_its_own_tokens_of_window():
    # 500 new tokens: 5 x (cdiv(1023 + 500, 16) + 1) pages, 0.5 GB, not two 8,192-token chunks (14.27 GB)
    a, b = run(True, second=500)
    assert b["t_first"] < a["t_done"]
