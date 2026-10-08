"""A chunk hinted first out again on a later turn is evicted once (vLLM's CPU primary tier, tiering manager and FS
tier, CPU only).

ParkCachePolicy keeps hinted chunks in a lazy heap and treated an entry as current when its time equals the chunk's
hint. First out is the constant NEVER, so when a session's next turn reused a prefix (the touch moves it back to LRU,
leaving the old entry in the heap) and the rule again said park or drop, the old entry came back to life: the chunk
had two current entries. An eviction that reached both returned the same chunk twice. The manager freed its slot
twice (two later keys share one CPU slot) and lowered its evictable count by two for one chunk, so the count ran
below the chunks with ref_cnt 0 and a later pin went past zero: AssertionError in CPUOffloadingManager._prepare_load,
from ParkFsTier.serve_external_requests (the crash after ~70 minutes of heavy multi-session replay on vLLM 0.31).

Before the fix: B's store evicts A's first chunk twice. After it: four distinct victims, distinct slots, the count
matches the ref counts and a late write of every resident chunk runs.

Run: PYTHONPATH=. python tests/test_park_vllm_double_hint.py
"""
import mmap
import os
import shutil
import tempfile
from types import SimpleNamespace as NS

from vllm.v1.kv_offload.base import ReqContext, ScheduleEndContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager

import layby.vllm.fs_tier as F
from layby.vllm.state import STATE

B = 4096                                    # bytes per chunk
F.probe_disk = lambda *a, **k: None         # no disk probe in a unit test


def build(n_chunks, root):
    STATE.__init__()
    STATE.chunk_tokens, STATE.bytes_per_token = 16, B / 16
    prim = object.__new__(CPUPrimaryTierOffloadingManager)
    CPUOffloadingManager.__init__(prim, num_chunks=n_chunks, cache_policy="ParkCachePolicy",
                                  cache_policy_module_path="layby.vllm.policy")
    if "prepare_read" not in CPUPrimaryTierOffloadingManager.__dict__:      # vLLM 0.30: an alias set in __init__
        prim.prepare_read = prim.prepare_load
    prim.complete_read, prim.prepare_write, prim.complete_write = prim.complete_load, prim.prepare_store, prim.complete_store
    buf = mmap.mmap(-1, n_chunks * B)
    prim._kv_memoryview = memoryview(buf).cast("B", (n_chunks, B))
    par = NS(tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, is_parallelism_agnostic=True)
    spec = NS(blocks_per_chunk=1, kv_events_config=NS(enable_kv_cache_events=False),
              config=NS(groups=[NS(tokens_per_block=16, layer_names=["l0"])], model=NS(name="test/m", dtype="bf16"),
                        cache=NS(tokens_per_hash=16), parallel=par, canonical_layout=False, replicated_layout=False))
    fs = F.ParkFsTier(spec, prim.get_kv_memoryview(), "ParkFsTier", root_dir=root, n_read_threads=2, n_write_threads=2)
    return TieringOffloadingManager(prim, [fs]), prim, fs, prim._policy


def key(s, i):
    return make_offload_key(f"{s}{i:03d}".encode().ljust(16, b"."), 0)


def settle(mgr, fs):
    fs.drain_jobs()
    mgr._process_finished_jobs()
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))


def store(mgr, req, keys, eta=None):
    """A request stores keys GPU -> CPU and finishes. With eta, the rule decides it at its finish, as ParkConnector
    does: _decide calls apply() and sets park_eta, then the manager's finish applies park_eta to the idle chunks."""
    ctx = ReqContext(req_id=f"r-{req}", kv_transfer_params={"park_key": req})
    mgr.on_new_request(ctx)
    out = mgr.prepare_store(keys, ctx)
    assert out is not None, f"{req}: store refused"
    mgr.complete_store(out.keys_to_store, ctx)
    if eta is not None:
        STATE.policy.apply(keys, eta)
        ctx.kv_transfer_params["park_eta"] = eta
    mgr.on_request_finished(ctx)
    return out


def check(prim, pol):
    """The manager's evictable count is the number of resident chunks with ref_cnt 0; no two keys share a slot."""
    ids = [c.chunk_id for c in pol.chunks.values()]
    assert len(ids) == len(set(ids)), "two keys share one CPU slot"
    assert len(prim._free_list) == len(set(prim._free_list)), f"a slot is free twice: {prim._free_list}"
    assert not set(ids) & set(prim._free_list), "a resident chunk's slot is on the free list"
    free_refs = sum(1 for c in pol.chunks.values() if c.ref_cnt == 0)
    assert prim._num_evictable_cache_chunks == free_refs, \
        f"manager counts {prim._num_evictable_cache_chunks} evictable chunks, {free_refs} have ref_cnt 0"
    assert pol.evictable <= {k for k, c in pol.chunks.items() if c.ref_cnt == 0}, "an evictable chunk is pinned"


root = tempfile.mkdtemp(prefix="pp_dhint_", dir=os.path.expanduser("~/.cache"))
try:
    mgr, prim, fs, pol = build(8, root)
    A = [key("a", i) for i in range(2)]
    A2 = A + [key("a", 2)]
    C = [key("c", i) for i in range(5)]
    store(mgr, "a-1", A, eta=-1.0)         # turn 1: park or drop, first out
    settle(mgr, fs)
    store(mgr, "a-2", A2, eta=-1.0)        # turn 2 reuses A (a touch: back to LRU), then first out again
    settle(mgr, fs)
    store(mgr, "c-1", C)                   # unhinted; the tier is full
    settle(mgr, fs)
    check(prim, pol)
    # B needs 4 slots: the 3 first-out chunks of session a, then the LRU head of C
    out = store(mgr, "b-1", [key("b", i) for i in range(4)])
    ev = list(out.evicted_keys)
    assert len(ev) == len(set(ev)), f"one chunk evicted twice: {ev}"
    assert set(A2) <= set(ev) and len(set(ev) & set(C)) == 1, ev
    settle(mgr, fs)
    check(prim, pol)
    # the crash path: a late write of everything resident pins every chunk through create_store_job
    resident = list(pol.chunks)
    fs.queue_write(resident)
    settle(mgr, fs)
    settle(mgr, fs)
    check(prim, pol)
    assert all(k in fs._on_disk for k in resident), "the late write did not complete"
    print("double hint ok: each chunk evicted once, slots distinct, evictable count matches ref counts")
finally:
    shutil.rmtree(root, ignore_errors=True)
