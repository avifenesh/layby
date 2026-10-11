"""CPU tests of engine/replay.py's Phase R behaviour against a mock vLLM server (aiohttp, no engine, no GPU):
run to completion with a hard cap (unserved turns censored), failed turns that do not end their session,
per-turn concurrency, tool names carried to the rows, the workload count health check, and live scoring
through a sidecar (both curves logged). Also the sidecar's POST /park/score with a stub scorer.

  python -m pytest -q tests/test_replay_phase_r.py      (needs aiohttp and numpy)
"""
import asyncio
import json
import os
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

aiohttp = pytest.importorskip("aiohttp")
from aiohttp import web  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY = os.path.join(ROOT, "engine", "replay.py")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(apps):
    """Run aiohttp apps {port: app} on a daemon thread's own loop."""
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        for port, app in apps.items():
            r = web.AppRunner(app)
            loop.run_until_complete(r.setup())
            loop.run_until_complete(web.TCPSite(r, "127.0.0.1", port).start())
        loop.run_forever()
    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.5)


def mock_vllm(port, hint_port, seen, behave=None, sidecar_port=None, live=None):
    """OpenAI completions with token ids streamed. behave[park_key]: "fail" (HTTP 500) or "hang" (no answer)."""
    behave = behave or {}

    async def completions(req):
        b = await req.json()
        key = (b.get("kv_transfer_params") or {}).get("park_key")
        seen.append(("vllm", time.time(), b))
        if behave.get(key) == "fail":
            return web.Response(status=500, text="boom")
        if behave.get(key) == "hang":
            await asyncio.sleep(60)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        if b.get("return_token_ids"):   # vLLM's first chunk echoes the prompt ids: one SSE line as long as the prompt
            await resp.write(f"data: {json.dumps({'choices': [{'text': '', 'prompt_token_ids': b['prompt']}]})}\n\n".encode())
        for i in range(b["max_tokens"]):
            await asyncio.sleep(0.05)
            await resp.write(f"data: {json.dumps({'choices': [{'token_ids': [7 + i]}]})}\n\n".encode())
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': len(b['prompt'])}})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def metrics(req):
        return web.Response(text="vllm:prompt_tokens_total 1\n")

    async def hint(req):
        seen.append(("hint", time.time(), await req.json()))
        return web.json_response({"queued": 1})

    async def score(req):
        d = await req.json()
        seen.append(("score", time.time(), d))
        return web.json_response({"key": d["key"], "surv": live})

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    happ = web.Application()
    happ.router.add_post("/hint", hint)
    apps = {port: app, hint_port: happ}
    if sidecar_port:
        sapp = web.Application()
        sapp.router.add_post("/park/score", score)
        apps[sidecar_port] = sapp
    serve(apps)


def T(gap, tool=None, kind="tool", surv=0.5):
    return dict(new_tokens=8, out_len=2, kind=kind, ctx=10, gap_after=gap, tool=tool, surv_v6=[surv] * 15)


def write_wl(tmp_path, sessions):
    w = tmp_path / "w.json"
    w.write_text(json.dumps(dict(sessions=sessions)))
    return w


def run(w, out, port, hport, extra=(), check=True):
    cmd = [sys.executable, REPLAY, str(w), str(out), "--url", f"http://127.0.0.1:{port}", "--park-hints", "v6",
           "--park-port", str(hport), *extra]
    p = subprocess.run(cmd, check=check, timeout=60, capture_output=True, text=True)
    rows = [json.loads(l) for l in open(out)] if os.path.exists(out) else []
    return p, rows


def turn_rows(rows):
    return {(r["session"], r["turn"]): r for r in rows if "ttft" in r}


def test_failed_turn_continues_cap_censors_concurrency_and_tools(tmp_path):
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen, behave={"s0-1": "fail", "s1-1": "hang"})
    w = write_wl(tmp_path, [
        dict(id="s0", pool="p", seed=1, start=0.0, turns=[T(0.1, "bash"), T(0.1, "edit"), T(None)]),
        dict(id="s1", pool="p", seed=2, start=0.0, turns=[T(0.1, "read"), T(0.1), T(None)])])
    p, rows = run(w, tmp_path / "o.jsonl", port, hport, ["--hard-cap", "3", "--expect-sessions", "2",
                                                          "--expect-turns", "6"])
    assert "sessions 2 turns 6" in p.stdout
    R = turn_rows(rows)
    assert set(R) == {(s, i) for s in ("s0", "s1") for i in range(3)}          # every turn has a row
    # the failed turn is infinite and its session goes on
    assert R[("s0", 1)]["failed"] and R[("s0", 1)]["ttft"] == float("inf") and "500" in R[("s0", 1)]["error"]
    assert not R[("s0", 2)].get("failed") and R[("s0", 2)]["ttft"] < 5
    # history after a failure: turn 0 prompt (8) + its output (2) + turn 1's new tokens (8) + turn 2's (8)
    bodies = {b["kv_transfer_params"]["park_key"]: b for k, _, b in seen if k == "vllm"}
    assert len(bodies["s0-2"]["prompt"]) == 26
    assert bodies["s0-2"]["prompt"][:18] == bodies["s0-1"]["prompt"]
    # the hung request is in flight at the cap: censored, and so is the turn after it
    assert R[("s1", 1)]["censored"] and R[("s1", 2)]["censored"] and R[("s1", 2)]["ttft"] == float("inf")
    # no hint for the failed turn nor for last turns
    assert sorted(b["key"] for k, _, b in seen if k == "hint") == ["s0-0", "s1-0"]
    # both turn-0 requests were in flight together
    assert max(R[("s0", 0)]["concurrency"], R[("s1", 0)]["concurrency"]) == 2
    series = next(r["concurrency_series"] for r in rows if "concurrency_series" in r)
    assert max(n for _, n in series) == 2 and series[-1][1] == 0
    # tool names and the previous turn's kind ride on every row, censored ones included
    assert R[("s0", 0)]["tool"] == "bash" and R[("s0", 1)]["prev_tool"] == "bash" and R[("s1", 1)]["prev_tool"] == "read"
    assert R[("s1", 2)]["prev_kind"] == "tool" and R[("s0", 1)]["prev_gap"] == 0.1
    summ = next(r for r in rows if "wall_s" in r)
    assert (summ["served"], summ["failed"], summ["censored"], summ["cap_hit"]) == (3, 1, 2, True)
    assert summ["wall_s"] < 6                                     # the cap cut the 60 s hang


def test_long_prompt_is_served_although_its_echoed_ids_exceed_the_default_line_limit(tmp_path):
    """A 131K-token prompt echoed as ids is one SSE line of about 1 MB, above aiohttp's default 512 KiB line limit
    (2 x read_bufsize): the turn must be served, not failed with LineTooLong (2026-10-11 pilot: every turn above about
    85K tokens failed)."""
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen)
    big = dict(new_tokens=131000, out_len=2, kind="tool", ctx=131000, gap_after=None, tool=None, surv_v6=[0.5] * 15)
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=[big])])
    p, rows = run(w, tmp_path / "o.jsonl", port, hport, ["--vocab", "150000"])
    R = turn_rows(rows)
    assert not R[("s0", 0)].get("failed"), R[("s0", 0)].get("error")
    assert R[("s0", 0)]["out_tokens"] == 2 and R[("s0", 0)]["ttft"] < 30


def test_health_check_mismatch_sends_nothing(tmp_path):
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen)
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=[T(0.1), T(None)])])
    p, _ = run(w, tmp_path / "o.jsonl", port, hport, ["--expect-sessions", "2"], check=False)
    assert p.returncode == 2 and "HEALTH CHECK FAILED" in p.stderr
    assert not seen


def test_live_scoring_logs_both_curves(tmp_path):
    port, hport, sport, seen = free_port(), free_port(), free_port(), []
    mock_vllm(port, hport, seen, sidecar_port=sport, live=[0.25] * 15)
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0,
                                 turns=[T(0.1, kind="human"), T(0.1), T(None)])])
    side = tmp_path / "side.jsonl"
    side.write_text(json.dumps(dict(session="s0", turn=0, state={"user": "real text"})) + "\n")
    _, rows = run(w, tmp_path / "o.jsonl", port, hport, ["--side-file", str(side), "--sidecar",
                                                          f"http://127.0.0.1:{sport}"])
    curves = {r["curve"]: r for r in rows if "curve" in r}
    assert curves["s0-0"]["live"] == [0.25] * 15 and curves["s0-0"]["pre"] == [0.5] * 15
    assert curves["s0-0"]["max_abs_diff"] == pytest.approx(0.25)
    assert curves["s0-1"]["live_missing"]                       # no side-file row: precomputed curve posted
    scored = [d for k, _, d in seen if k == "score"]
    assert len(scored) == 1 and scored[0]["key"] == "s0-0" and scored[0]["kind"] == "human"
    assert json.loads(scored[0]["state"]) == {"user": "real text"}
    assert [b["key"] for k, _, b in seen if k == "hint"] == ["s0-1"]   # the sidecar sends s0-0's hint itself


def test_sidecar_score_endpoint():
    from layby.sidecar.proxy import Sidecar

    class Stub:
        def score(self, states, kinds):
            return np.array([[0.1 * len(s)] * 15 for s in states])

    hport, sport, got = free_port(), free_port(), []

    async def main():
        async def hint(req):
            got.append(await req.json())
            return web.json_response({"queued": 1})
        happ = web.Application(); happ.router.add_post("/hint", hint)
        hr = web.AppRunner(happ); await hr.setup(); await web.TCPSite(hr, "127.0.0.1", hport).start()
        sc = Sidecar(SimpleNamespace(engine="vllm", hint_url=f"http://127.0.0.1:{hport}/hint", batch=8))
        sc.scorer = Stub()
        sc.http = aiohttp.ClientSession()
        task = asyncio.create_task(sc.score_loop())
        app = web.Application(); app.router.add_post("/park/score", sc.score)
        r = web.AppRunner(app); await r.setup(); await web.TCPSite(r, "127.0.0.1", sport).start()
        async with aiohttp.ClientSession() as http:
            async with http.post(f"http://127.0.0.1:{sport}/park/score",
                                 json={"key": "k1", "state": "abcd", "kind": "tool"}) as resp:
                d = await resp.json()
        task.cancel(); await sc.http.close(); await r.cleanup(); await hr.cleanup()
        return d

    d = asyncio.run(main())
    assert d["key"] == "k1" and d["surv"] == pytest.approx([0.4] * 15)
    assert got == [[{"key": "k1", "surv": pytest.approx([0.4] * 15)}]]


def test_open_loop_poisson_arrivals_session_order_and_name(tmp_path):
    import random
    import importlib.util
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen)
    w = write_wl(tmp_path, [dict(id=f"s{k}", pool="p", seed=k, start=500.0, turns=[T(300.0), T(None)])
                            for k in range(6)])
    out = tmp_path / "c__K__h1__rate10.jsonl"
    p, rows = run(w, out, port, hport, ["--open-loop-rate", "10", "--requests", "8", "--seed", "3",
                                        "--hard-cap", "30"])
    R = turn_rows(rows)
    assert len(R) == 8 and not any(r.get("failed") or r.get("censored") for r in R.values())
    # arrival times: the seed's Poisson process at 10/s, the same in every arm (no arrival dropped here)
    rng, t, want = random.Random(3), 0.0, []
    for _ in range(8):
        t += rng.expovariate(10.0); want.append(round(t, 4))
    assert sorted(r["arrival"] for r in R.values()) == want
    assert all(abs(r["sent"] - r["arrival"]) < 0.5 for r in R.values())      # sent on arrival, not after gaps
    summ = next(r for r in rows if "wall_s" in r)
    assert (summ["requests"], summ["arrivals_dropped"], summ["open_loop_rate"]) == (8, 0, 10.0)
    assert summ["wall_s"] < 10                                                   # the 300 s gaps and starts unused
    # a session's turn 1 goes after its turn 0 finished, on its history
    bodies = {b["kv_transfer_params"]["park_key"]: b for k, _, b in seen if k == "vllm"}
    for (sid, i), r in R.items():
        if i == 1:
            r0 = R[(sid, 0)]
            assert r["sent"] >= r0["sent"] + r0["e2e"] - 1e-3
            assert bodies[f"{sid}-1"]["prompt"][:10] == bodies[f"{sid}-0"]["prompt"] + [7, 8]
    # the analysis reads it as a sweep point
    spec = importlib.util.spec_from_file_location("apr", os.path.join(ROOT, "engine", "analyze_phase_r.py"))
    A = importlib.util.module_from_spec(spec); spec.loader.exec_module(A)
    sw = A.load_sweeps(str(tmp_path))
    assert A.sweep_point(sw["c"]["h1"]["K"][10.0])["n"] == 8


def test_open_loop_drops_arrivals_when_no_session_is_ready_and_checks_the_name(tmp_path):
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen)
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=[T(1.0), T(1.0), T(None)])])
    _, rows = run(w, tmp_path / "c__K__h1__rate50.jsonl", port, hport,
                  ["--open-loop-rate", "50", "--requests", "3", "--hard-cap", "30"])
    R = turn_rows(rows)
    assert sorted(R) == [("s0", 0), ("s0", 1), ("s0", 2)]                       # one session: strictly in order
    assert next(r for r in rows if "wall_s" in r)["arrivals_dropped"] > 0        # 50/s while a turn takes ~0.1 s
    p, _ = run(w, tmp_path / "c__K__h1.jsonl", port, hport, ["--open-loop-rate", "50", "--requests", "3"],
               check=False)
    assert p.returncode == 2 and "rate50.jsonl" in p.stderr


def test_continuum_params_ride_on_every_request(tmp_path):
    # arm CT: layby.vllm.continuum reads park_key, park_kind, park_tool and park_last; no curve is posted
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen)
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0,
                                 turns=[T(0.1, "bash"), T(0.1, kind="human"), T(None, "edit")])])
    cmd = [sys.executable, REPLAY, str(w), str(tmp_path / "o.jsonl"), "--url", f"http://127.0.0.1:{port}",
           "--continuum"]
    subprocess.run(cmd, check=True, timeout=60, capture_output=True, text=True)
    got = [b["kv_transfer_params"] for k, _, b in seen if k == "vllm"]
    assert got == [dict(park_key="s0-0", park_kind="tool", park_tool="bash", park_last=False),
                   dict(park_key="s0-1", park_kind="human", park_tool=None, park_last=False),
                   dict(park_key="s0-2", park_kind="tool", park_tool="edit", park_last=True)]
    assert not [x for x in seen if x[0] == "hint"]
    p = subprocess.run(cmd + ["--park-hints", "v6"], timeout=60, capture_output=True, text=True)
    assert p.returncode == 2 and "--continuum" in p.stderr


def test_control_curves_are_posted_and_a_missing_one_stops_the_run(tmp_path):
    # --park-hints km|ewma posts the turn's surv_km / surv_ewma (layby.curves annotate); a turn with a wait and no
    # such curve stops the run before anything is sent (it would be posted as 0.5)
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen)
    turns = [dict(T(0.1), surv_ewma=[0.3] * 15), dict(T(0.1), surv_ewma=[0.2] * 15), T(None)]
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=turns)])
    p, _ = run(w, tmp_path / "o.jsonl", port, hport, ["--park-hints", "ewma"])
    assert "surv_ewma: 2 of 2" in p.stdout
    assert [(b["key"], b["surv"][0]) for k, _, b in seen if k == "hint"] == [("s0-0", 0.3), ("s0-1", 0.2)]
    seen.clear()
    del turns[1]["surv_ewma"]
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=turns)])
    p, _ = run(w, tmp_path / "o2.jsonl", port, hport, ["--park-hints", "ewma"], check=False)
    assert p.returncode == 2 and "1 turns with a wait lack surv_ewma" in p.stderr and not seen


def test_open_loop_requests_in_flight_at_the_cap_are_written_censored(tmp_path):
    # a request hanging at the hard cap is cancelled; it must still get a censored row (before the fix the cancel
    # path dropped it from the in-flight set, so the request vanished from the output)
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen, behave={"s0-0": "hang", "s1-0": "hang"})
    w = write_wl(tmp_path, [dict(id=f"s{k}", pool="p", seed=k, start=0.0, turns=[T(0.1), T(None)]) for k in range(4)])
    _, rows = run(w, tmp_path / "c__K__h1__rate20.jsonl", port, hport,
                  ["--open-loop-rate", "20", "--requests", "4", "--seed", "1", "--hard-cap", "3"])
    R = turn_rows(rows)
    summ = next(r for r in rows if "wall_s" in r)
    assert summ["requests"] == len(R)                                  # every request sent has a row
    sent = {b["kv_transfer_params"]["park_key"] for k, _, b in seen if k == "vllm"}
    hung = sent & {"s0-0", "s1-0"}
    assert hung                                                         # the seed sends at least one hanging turn
    for key in hung:
        sid = key.split("-")[0]
        assert R[(sid, 0)]["censored"] and R[(sid, 0)]["ttft"] == float("inf")
    assert summ["censored"] == len(hung) and summ["cap_hit"] and summ["wall_s"] < 10


def test_keep_cuts_the_history_before_the_new_tokens(tmp_path):
    # context compaction: a turn with keep restarts its prompt from the first keep tokens of the history
    port, hport, seen = free_port(), free_port(), []
    mock_vllm(port, hport, seen)
    turns = [T(0.1), dict(T(0.1), keep=4), dict(T(None), keep=0)]
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=turns)])
    run(w, tmp_path / "o.jsonl", port, hport)
    b = {x["kv_transfer_params"]["park_key"]: x["prompt"] for k, _, x in seen if k == "vllm"}
    assert len(b["s0-0"]) == 8
    assert len(b["s0-1"]) == 4 + 8 and b["s0-1"][:4] == b["s0-0"][:4]       # 4 kept, then turn 1's 8 new tokens
    assert len(b["s0-2"]) == 8                                               # keep 0: only its new tokens


def test_lost_connection_is_flagged_conn_error(tmp_path):
    # no server on the port: every turn fails on the connection, and the row says so (the harness counts these)
    port, hport = free_port(), free_port()
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=[T(0.1), T(None)])])
    _, rows = run(w, tmp_path / "o.jsonl", port, hport)
    R = turn_rows(rows)
    assert all(r["failed"] and r.get("conn_error") for r in R.values()) and len(R) == 2


def mock_vllm_disk(port, seen, disk_turns, corrupt=()):
    """vLLM with a disk tier: a request whose park_key is in disk_turns reads the disk (the /metrics counter
    vllm:kv_offload_tiering_read_bytes rises while it runs); a cold request (cache_salt) answers with the same ids,
    or other ids when its prompt belongs to a turn in corrupt."""
    state = dict(read=0.0)
    prompts = {}

    async def completions(req):
        b = await req.json()
        seen.append(b)
        n = b["max_tokens"]
        if b.get("cache_salt"):
            key = prompts.get(tuple(b["prompt"]))
            ids = [7 + i for i in range(n)]
            if key in corrupt:
                ids[1] = 999
            return web.json_response({"choices": [{"token_ids": ids}]})
        key = (b.get("kv_transfer_params") or {}).get("park_key")
        prompts[tuple(b["prompt"])] = key
        if key in disk_turns:
            state["read"] += 1e6
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        for i in range(n):
            await asyncio.sleep(0.02)
            await resp.write(f"data: {json.dumps({'choices': [{'token_ids': [7 + i]}]})}\n\n".encode())
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': len(b['prompt'])}})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def metrics(req):
        return web.Response(text=f'vllm:kv_offload_tiering_read_bytes_total{{tier="1"}} {state["read"]}\n'
                                 f"vllm:external_prefix_cache_hits_total 0\n")

    async def hint(req):
        return web.json_response({"queued": 1})

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    app.router.add_get("/metrics", metrics)
    happ = web.Application()
    happ.router.add_post("/hint", hint)
    return app, happ


@pytest.mark.parametrize("corrupt,verdict", [((), "PASS"), (("s0-2",), "FAIL")])
def test_verify_disk_compares_disk_restored_turns_with_a_cold_recompute(tmp_path, corrupt, verdict):
    # K disk-path pilot: only turns that read the disk tier are re-sent cold (with a cache_salt); the row records
    # whether the output matched, and check_k_disk.py passes only when every verified turn is identical
    port, hport, seen = free_port(), free_port(), []
    app, happ = mock_vllm_disk(port, seen, disk_turns={"s0-1", "s0-2"}, corrupt=corrupt)
    serve({port: app, hport: happ})
    turns = [T(0.05), T(0.05), T(0.05), T(None)]
    w = write_wl(tmp_path, [dict(id="s0", pool="p", seed=1, start=0.0, turns=turns)])
    _, rows = run(w, tmp_path / "o.jsonl", port, hport, extra=("--verify-disk", "5"))
    R = turn_rows(rows)
    assert "verify" not in R[("s0", 0)] and "verify" not in R[("s0", 3)]
    v1, v2 = R[("s0", 1)]["verify"], R[("s0", 2)]["verify"]
    assert v1["identical"] and v1["disk_read_bytes"] == 1e6 and v1["first_diff"] is None
    assert v2["identical"] == (not corrupt) and v2["first_diff"] == (1 if corrupt else None)
    salted = [b for b in seen if b.get("cache_salt")]
    assert len(salted) == 2 and len({b["cache_salt"] for b in salted}) == 2
    sys.path.insert(0, os.path.join(ROOT, "engine", "phase_r"))
    import check_k_disk
    live = {"counts": {"ins": 1, "park": 2}, "disk": {"written_jobs": 3, "chunks_on_disk": 3}}
    assert check_k_disk.check(rows, live)["verdict"] == verdict
    assert check_k_disk.check(rows, dict(live, counts={"ins": 0, "park": 0}))["verdict"] == "FAIL"
