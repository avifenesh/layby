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
