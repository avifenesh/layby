"""CPU tests of the SGLang park adapter (run with the SGLang 0.5.21 venv, no GPU):

  CUDA_VISIBLE_DEVICES= .venv-sglang/bin/python -m pytest -q tests/test_park_sglang.py

Eviction order and placement on mock nodes; the plugin's registration and hook targets against the
installed SGLang; a real ParkRadixCache on the Python TreeCore with a CPU allocator (no host tier:
HiCache needs CUDA) for split, eviction events and the tick path; the hint port; the sidecar tag;
replay.py against a mock SGLang server, with and without --window.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from types import SimpleNamespace

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sglang.srt.mem_cache.unified_cache.components import ComponentType  # noqa: E402

from layby.sglang import policy  # noqa: E402
from layby.sglang.state import GRACE, NEVER, STATE, SglState  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Each test on a clean STATE (the module-level object is shared by every layby.sglang module)."""
    s = SglState()
    for k, v in vars(s).items():
        monkeypatch.setattr(STATE, k, v, raising=False)
    yield


class Node:
    """The fields of UnifiedTreeNode the policy reads."""
    _ids = iter(range(10**6))

    def __init__(self, last, evicted=True, lock=0, key_len=64, parent=None):
        self.id = next(Node._ids)
        self.last_access_time = float(last)
        self.evicted = evicted
        self.key = [0] * key_len
        self.parent = parent
        self.component_data = {ComponentType.FULL: SimpleNamespace(lock_ref=lock, host_lock_ref=0)}


def ticks(*pairs):
    """Tick log: (counter, clock) pairs; the clock ends at the last one."""
    for c, t in pairs:
        STATE.tick_c.append(float(c))
        STATE.tick_t.append(float(t))
    STATE.clock = float(pairs[-1][1])


def order(nodes):
    """Host eviction order of SGLang's heap: smallest priority first."""
    return sorted(nodes, key=lambda n: (policy.priority(n), n.last_access_time))


def test_lru_order_without_hints():
    ticks((0, 100.0), (10, 110.0), (20, 120.0), (30, 130.0))
    a, b, c = Node(5), Node(15), Node(25)            # last used at 100, 110, 120
    assert order([c, a, b]) == [a, b, c]


def test_gpu_stays_lru_on_the_logical_counter():
    ticks((0, 100.0), (30, 130.0))
    a, b = Node(5, evicted=False), Node(25, evicted=False)
    policy.STATE.next_use[a.id] = (NEVER, 1e9)       # a hint never reorders the GPU
    assert policy.priority(a) == 5 and policy.priority(b) == 25


def test_hinted_against_lru_estimate(monkeypatch):
    ticks((0, 100.0), (10, 110.0), (20, 120.0))       # clock 120
    monkeypatch.setattr(policy, "counter", lambda: 21.0)
    lru = Node(5)                                      # last use 100: estimate 120 + 20 = 140
    soon, late = Node(15), Node(16)
    policy.place([soon], 125.0)
    policy.place([late], 200.0)
    assert order([soon, lru, late]) == [late, lru, soon]


def test_first_out_goes_first(monkeypatch):
    ticks((0, 100.0), (20, 120.0))
    monkeypatch.setattr(policy, "counter", lambda: 21.0)
    old, parked = Node(1), Node(19)
    policy.place([parked], NEVER)
    assert order([old, parked])[0] is parked


def test_hint_ends_on_touch_and_past_due(monkeypatch):
    ticks((0, 100.0), (20, 120.0))
    monkeypatch.setattr(policy, "counter", lambda: 21.0)
    n = Node(10)
    policy.place([n], 500.0)
    assert policy.valid_next_use(n) == 500.0
    n.last_access_time = 21.0 - 1e-5 * 3             # matched by a returning request (deep node)
    assert policy.valid_next_use(n) is None
    m = Node(10)
    policy.place([m], 121.0)
    STATE.clock = 121.0 + GRACE + 0.1                # GRACE past due: LRU order again
    assert policy.valid_next_use(m) is None
    f = Node(10)
    policy.place([f], NEVER)                          # first out never expires
    assert policy.valid_next_use(f) == NEVER


def test_place_skips_in_use_and_keeps_earlier_use(monkeypatch):
    ticks((0, 100.0), (20, 120.0))
    monkeypatch.setattr(policy, "counter", lambda: 21.0)
    shared, busy = Node(10), Node(10, lock=1)
    policy.place([shared], 150.0)                     # another session placed it first
    assert policy.place([shared, busy], NEVER) == 1
    assert STATE.next_use[shared.id][0] == 150.0      # earliest predicted use wins
    assert busy.id not in STATE.next_use


def test_split_restamps_suffix_and_splits_idle(monkeypatch):
    ticks((0, 100.0), (20, 120.0))
    monkeypatch.setattr(policy, "counter", lambda: 21.0)
    child = Node(10, key_len=128)
    policy.place([child], 300.0)
    STATE.g_idle[child.id] = (50.0, 128.0)
    was = policy.valid_before_split(child)
    new = Node(10, key_len=64)                        # the prefix the split creates
    child.key = [0] * 64
    child.last_access_time = 22.0                     # _split_node bumps the suffix
    monkeypatch.setattr(policy, "counter", lambda: 23.0)
    policy.on_split(new, child, was)
    assert policy.valid_next_use(child) == 300.0
    assert new.id not in STATE.next_use
    assert STATE.g_idle[new.id] == (50.0, 64.0) and STATE.g_idle[child.id] == (50.0, 64.0)


def test_strategy_never_raises():
    s = policy.ParkStrategy(port=None)
    bad = SimpleNamespace(evicted=True, id=1, last_access_time=3.0)   # no component data, no ticks
    STATE.next_use[1] = "broken"
    assert s.get_priority(bad) == 3.0


# --- plugin and hook targets against the installed SGLang ------------------------------------
def test_plugin_registers_and_hooks_resolve():
    import pkgutil
    from importlib.metadata import entry_points
    from sglang.srt.plugins import load_plugins
    from sglang.srt.plugins.hook_registry import HookRegistry
    assert ("park", "layby.sglang.plugin:register") in [(e.name, e.value) for e in entry_points(group="sglang.srt.plugins")]
    load_plugins()
    from layby.sglang.hooks import HOOKS
    for target, _, _ in HOOKS:
        obj, attr = target.rsplit(".", 1)
        assert target in HookRegistry._patched, target
        assert hasattr(getattr(pkgutil.resolve_name(obj), attr), "__wrapped__"), target
    from sglang.srt.arg_groups.choices import RADIX_EVICTION_POLICY_CHOICES
    from sglang.srt.mem_cache.registry import get_radix_cache_factory
    from sglang.srt.mem_cache.utils import get_eviction_strategy
    assert "park" in RADIX_EVICTION_POLICY_CHOICES and get_radix_cache_factory("park") is not None
    assert type(get_eviction_strategy("park", {"port": 8765})).__name__ == "ParkStrategy"
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    for m in ("insert", "insert_req", "on_release", "inc_lock_ref", "loading_check", "init_hicache",
              "_log_write_ack_metrics", "write_backup_storage"):
        assert callable(getattr(UnifiedRadixCache, m)), m


# --- a real tree on CPU ------------------------------------------------------------------------
def make_cache(size=256):
    from sglang.srt.plugins import load_plugins
    load_plugins()
    from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from layby.sglang.cache import ParkRadixCache
    alloc = TokenToKVPoolAllocator(size, torch.float16, "cpu", None, False)
    params = CacheInitParams(disable=False, req_to_token_pool=None, token_to_kv_pool_allocator=alloc,
                             page_size=1, eviction_policy="park", eviction_policy_config={"port": None},
                             tree_components=(ComponentType.FULL,), tree_core_backend="python")
    return ParkRadixCache(params), alloc


def insert(cache, alloc, toks):
    from sglang.srt.mem_cache.base_prefix_cache import InsertParams
    from sglang.srt.mem_cache.radix_cache import RadixKey
    return cache.insert(InsertParams(key=RadixKey(toks), value=alloc.alloc(len(toks))))


def tick_roundtrip(hooks):
    pulled = hooks.pull_after([], SimpleNamespace())
    assert isinstance(pulled[-1], hooks.ParkTick)
    return hooks.recv_after(pulled, SimpleNamespace())


def test_cache_events_split_and_tick_placement():
    from sglang.srt.mem_cache.base_prefix_cache import EvictParams
    from layby.sglang import hooks
    cache, alloc = make_cache()
    assert type(cache.tree_core.eviction_strategy).__name__ == "ParkStrategy"
    a = list(range(100, 132))
    r1 = insert(cache, alloc, a)
    leaf = r1.last_device_node
    STATE.g_idle[leaf] = (time.monotonic() - 5.0, 32.0)
    insert(cache, alloc, a[:16] + list(range(900, 916)))     # splits a's node at 16 tokens
    path = cache.park_path(leaf)
    assert len(path) == 2 and path[0].id == leaf                  # suffix keeps its id
    assert STATE.g_idle[path[1].id][1] == 16.0                    # the prefix inherited half the period
    # a curve that says "back soon": the rule runs on the leader, the placement lands on the tick
    STATE.port, STATE.server = 1, False              # late hints on, no listening socket in a test
    STATE.entries["k1"] = dict(leaf=leaf, tokens=32, t=0.0)
    STATE.hints.put(("k1", [0.9] * 15))
    rest = tick_roundtrip(hooks)
    assert rest == [] and "k1" not in STATE.entries
    assert STATE.counts["late"] == 1 and sum(STATE.counts[o] for o in ("none", "ins", "park", "drop")) == 1
    assert leaf in STATE.next_use
    # an unknown key waits as an early hint
    STATE.hints.put(("k2", [0.1] * 15))
    tick_roundtrip(hooks)
    assert "k2" in STATE.early
    # GPU eviction: residency event and page eviction times; the deleted node's records go
    n0 = STATE.live.acc["ne_g"]
    res = cache.evict(EvictParams(num_tokens=16))
    assert res.num_tokens_evicted >= 16
    assert STATE.live.acc["ne_g"] > n0 and len(STATE.g_evicted) >= 16
    cache.sanity_check()


def test_finish_without_curve_or_port_registers_nothing():
    cache, alloc = make_cache()
    r = insert(cache, alloc, list(range(10, 42)))
    req = SimpleNamespace(rid="r1", finished=lambda: True,
                          sampling_params=SimpleNamespace(custom_params={"park_key": "s-0"}))
    STATE.port = None
    cache._park_finish(req, 32, [r])
    assert "s-0" not in STATE.entries and STATE.leaf_of["r1"] == r.last_device_node
    req.sampling_params.custom_params["park_surv"] = [0.5] * 15     # the curve rides on the request
    cache._park_finish(req, 32, [r])
    assert STATE.entries["s-0"]["decided"] and STATE.outbox[-1][0] == "s-0"


# --- hint port and sidecar ---------------------------------------------------------------------
def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_hint_port_serves_this_state():
    from layby.vllm import server
    port = free_port()
    srv = server.start(port, state=STATE)
    try:
        body = json.dumps([{"key": "a", "surv": [0.5] * 15}, {"key": "b", "surv": [0.2] * 15}]).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/hint", body, {"Content-Type": "application/json"})
        assert json.load(urllib.request.urlopen(req)) == {"queued": 2}
        assert STATE.hints.get_nowait()[0] == "a"
        live = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/live"))
        assert live["engine"] == "sglang" and "rates" in live
    finally:
        srv.shutdown()


def test_sidecar_tags_custom_params():
    from layby.sidecar.proxy import ADAPTERS
    body = {"custom_params": {"x": 1}}
    ADAPTERS["sglang"](None).tag(body, "key1")
    assert body["custom_params"] == {"x": 1, "park_key": "key1"}


# --- replay.py against a mock SGLang server ----------------------------------------------------
def mock_sglang(port, hint_port, seen):
    from aiohttp import web
    import asyncio

    async def generate(req):
        b = await req.json()
        seen.append(("gen", time.time(), b))
        n = b["sampling_params"]["max_new_tokens"]
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        ids = []
        for i in range(n):
            ids.append(7 + i)
            meta = {"prompt_tokens": len(b["input_ids"]), "completion_tokens": len(ids), "cached_tokens": 0}
            await resp.write(f"data: {json.dumps({'output_ids': list(ids), 'meta_info': meta})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def metrics(req):
        return web.Response(text="sglang:prompt_tokens_total 1\nvllm:prompt_tokens_total 1\n")

    async def completions(req):                       # vLLM's OpenAI completions, token ids streamed
        b = await req.json()
        seen.append(("vllm", time.time(), b))
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(req)
        for i in range(b["max_tokens"]):
            await resp.write(f"data: {json.dumps({'choices': [{'token_ids': [7 + i]}]})}\n\n".encode())
        await resp.write(f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': len(b['prompt'])}})}\n\n".encode())
        await resp.write(b"data: [DONE]\n\n")
        return resp

    async def hint(req):
        seen.append(("hint", time.time(), await req.json()))
        return web.json_response({"queued": 1})

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app = web.Application()
        app.router.add_post("/generate", generate)
        app.router.add_get("/metrics", metrics)
        app.router.add_post("/v1/completions", completions)
        happ = web.Application()
        happ.router.add_post("/hint", hint)
        r1, r2 = web.AppRunner(app), web.AppRunner(happ)
        loop.run_until_complete(r1.setup()); loop.run_until_complete(r2.setup())
        loop.run_until_complete(web.TCPSite(r1, "127.0.0.1", port).start())
        loop.run_until_complete(web.TCPSite(r2, "127.0.0.1", hint_port).start())
        loop.run_forever()

    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.5)


def run_replay(tmp_path, port, hint_port, extra=()):
    turns = [dict(new_tokens=8, out_len=3, kind="tool", gap_after=0.2, surv_v6=[0.5] * 15),
             dict(new_tokens=8, out_len=3, kind="human", gap_after=5.0, surv_v6=[0.4] * 15),
             dict(new_tokens=8, out_len=3, kind="tool", gap_after=None)]
    wl = dict(sessions=[dict(id="s0", pool="p", seed=1, start=0.0, turns=turns),
                        dict(id="s1", pool="p", seed=2, start=30.0, turns=turns)])
    w, out = tmp_path / "w.json", tmp_path / "out.jsonl"
    w.write_text(json.dumps(wl))
    cmd = [sys.executable, os.path.join(ROOT, "engine", "replay.py"), str(w), str(out), "--engine", "sglang",
           "--url", f"http://127.0.0.1:{port}", "--park-hints", "v6", "--park-port", str(hint_port), *extra]
    t0 = time.time()
    subprocess.run(cmd, check=True, timeout=60)
    return [json.loads(l) for l in open(out)], time.time() - t0


def test_replay_sglang_and_window(tmp_path):
    port, hport, seen = free_port(), free_port(), []
    mock_sglang(port, hport, seen)
    rows, wall = run_replay(tmp_path, port, hport, ["--window", "2"])
    turns = [r for r in rows if "ttft" in r]
    trunc = [r for r in rows if r.get("truncated_at_window")]
    # s0: turn 0 and 1 sent; turn 1's 5 s gap would pass the 2 s window: stop at once at turn 2.
    # s1 starts at 30 s, past the window: stopped before turn 0 without waiting for it.
    assert [(r["session"], r["turn"]) for r in turns] == [("s0", 0), ("s0", 1)]
    assert sorted((r["session"], r["turn"]) for r in trunc) == [("s0", 2), ("s1", 0)]
    assert wall < 10
    gens = [b for k, _, b in seen if k == "gen"]
    assert gens[0]["sampling_params"]["custom_params"] == {"park_key": "s0-0"}
    assert gens[0]["sampling_params"]["ignore_eos"] and gens[0]["sampling_params"]["max_new_tokens"] == 3
    # turn 1's prompt is turn 0's prompt, its output ids, and 8 new tokens
    assert gens[1]["input_ids"][:11] == gens[0]["input_ids"] + [7, 8, 9] and len(gens[1]["input_ids"]) == 19
    hints = sorted(b["key"] for k, _, b in seen if k == "hint")
    assert hints == ["s0-0", "s0-1"]                  # the truncated session's last hint still goes out
    assert turns[0]["usage"]["prompt_tokens"] == 8 and turns[0]["out_tokens"] == 3


def test_replay_sglang_without_window_runs_all_turns(tmp_path):
    port, hport, seen = free_port(), free_port(), []
    mock_sglang(port, hport, seen)
    turns = [dict(new_tokens=8, out_len=2, kind="tool", gap_after=0.1, surv_v6=[0.5] * 15),
             dict(new_tokens=8, out_len=2, kind="tool", gap_after=None)]
    wl = dict(sessions=[dict(id="s0", pool="p", seed=1, start=0.0, turns=turns)])
    w, out = tmp_path / "w.json", tmp_path / "out.jsonl"
    w.write_text(json.dumps(wl))
    subprocess.run([sys.executable, os.path.join(ROOT, "engine", "replay.py"), str(w), str(out), "--engine",
                    "sglang", "--url", f"http://127.0.0.1:{port}"], check=True, timeout=60)
    rows = [json.loads(l) for l in open(out)]
    assert [r["turn"] for r in rows if "ttft" in r] == [0, 1]
    assert not any(r.get("truncated_at_window") for r in rows)


def test_replay_vllm_path_unchanged(tmp_path):
    port, hport, seen = free_port(), free_port(), []
    mock_sglang(port, hport, seen)
    turns = [dict(new_tokens=8, out_len=2, kind="tool", gap_after=0.1, surv_v6=[0.5] * 15),
             dict(new_tokens=8, out_len=2, kind="tool", gap_after=None)]
    wl = dict(sessions=[dict(id="s0", pool="p", seed=1, start=0.0, turns=turns)])
    w, out = tmp_path / "w.json", tmp_path / "out.jsonl"
    w.write_text(json.dumps(wl))
    subprocess.run([sys.executable, os.path.join(ROOT, "engine", "replay.py"), str(w), str(out), "--url",
                    f"http://127.0.0.1:{port}", "--park-hints", "v6", "--park-port", str(hport)],
                   check=True, timeout=60)
    rows = [json.loads(l) for l in open(out)]
    assert [r["turn"] for r in rows if "ttft" in r] == [0, 1]
    bodies = [b for k, _, b in seen if k == "vllm"]
    assert bodies[0]["kv_transfer_params"] == {"park_key": "s0-0"} and bodies[0]["return_token_ids"]
    assert bodies[1]["prompt"][:10] == bodies[0]["prompt"] + [7, 8]
    assert [b["key"] for k, _, b in seen if k == "hint"] == ["s0-0"]
    assert any(r.get("metrics") == "after" and r["lines"] == ["vllm:prompt_tokens_total 1"] for r in rows)


def test_link_telemetry_from_storage_jobs_and_load_acks():
    import threading as th
    from layby.sglang import hooks
    cache, _ = make_cache()
    STATE.page_size, STATE.bytes_per_token = 64, 1000.0
    cache._park_disk, cache._park_disk_lock = dict(n=0, mark=None, bytes=0), th.Lock()

    def backup(op):
        time.sleep(0.02)
        op.completed_tokens += 128

    class Prefetch(SimpleNamespace):
        def is_terminated(self):
            return False

    cache._park_disk_job(backup, SimpleNamespace(id=1, completed_tokens=0), time.monotonic() - 0.5)
    cache._park_disk_job(lambda op: time.sleep(0.02), Prefetch(request_id="r", hash_value=["h"] * 2), None)
    hooks._drain_threads()
    assert abs(STATE.speeds.acc["b_disk"] - 256e3) < 256e3 * 0.01 and STATE.speeds.acc["s_disk"] > 0.03
    assert abs(STATE.live.acc["n_disk"] - 2.0) < 0.01 and STATE.live.acc["wait_disk"] > 0.4
    ev = SimpleNamespace(elapsed_time=lambda other: 10.0)          # 10 ms on the CUDA events
    ack = SimpleNamespace(timing_enabled=True, num_bytes=2e8, start_event=ev, finish_event=None)
    cache._park_load_sub = {id(ack): time.monotonic() - 1.0}
    cache._park_load_ack(ack)
    assert abs(STATE.speeds.gbps("cpu") - 20.0) < 1e-9
    assert abs(STATE.live.acc["n_cpu"] - 1.0) < 0.01 and 0.9 < STATE.live.acc["wait_cpu"] < 1.0
