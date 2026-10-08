"""Write-on-evict (layby.vllm.demote) on vLLM's own CPU primary tier, tiering manager and file-system tier (CPU only).

Scenario: an 8-chunk CPU tier. Session A stores 6 chunks; the rule marks its first two for spill ("none": keep in
RAM, write to disk on eviction) and makes them first out. Session B then stores 4 chunks, so 2 must be evicted.

  demote (default)  A's two spill victims are written to disk before their slots are reused; B's store succeeds
                    from A's next two chunks; only the two victims are pinned, only while written; once written
                    they go first, and a later lookup restores them from disk byte for byte. A fallback case
                    (no other evictable chunk) drops instead of refusing the store.
  stock             no demoter (the code before this change): the victims are dropped and lost. Fails.
  ahead             the old write-ahead: the next victims are pinned by their writes before the store. B's store
                    is refused. Fails.

Run: PYTHONPATH=. python tests/test_park_vllm_demote.py [demote|stock|ahead]
"""
import mmap
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

from vllm.v1.kv_offload.base import LookupResult, ReqContext, ScheduleEndContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager

import layby.vllm.fs_tier as F
from layby.vllm import demote
from layby.vllm.state import STATE

MODE = sys.argv[1] if len(sys.argv) > 1 else "demote"
B = 4096                                    # bytes per chunk (page aligned, so O_DIRECT works where supported)
F.probe_disk = lambda *a, **k: None         # no 512 MB disk probe in a unit test


def build(n_chunks, root):
    STATE.__init__()
    STATE.write_mode = "demand"
    STATE.chunk_tokens, STATE.bytes_per_token = 16, B / 16
    prim = object.__new__(CPUPrimaryTierOffloadingManager)
    CPUOffloadingManager.__init__(prim, num_chunks=n_chunks, cache_policy="ParkCachePolicy",
                                  cache_policy_module_path="layby.vllm.policy")
    if "prepare_read" not in CPUPrimaryTierOffloadingManager.__dict__:      # vLLM 0.30: an alias set in __init__
        prim.prepare_read = prim.prepare_load
    prim.complete_read, prim.prepare_write, prim.complete_write = prim.complete_load, prim.prepare_store, prim.complete_store
    buf = mmap.mmap(-1, n_chunks * B)
    prim._kv_memoryview = memoryview(buf).cast("B", (n_chunks, B))
    prim._test_flat = memoryview(buf)
    par = NS(tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, is_parallelism_agnostic=True)
    spec = NS(blocks_per_chunk=1, kv_events_config=NS(enable_kv_cache_events=False),
              config=NS(groups=[NS(tokens_per_block=16, layer_names=["l0"])], model=NS(name="test/m", dtype="bf16"),
                        cache=NS(tokens_per_hash=16), parallel=par, canonical_layout=False, replicated_layout=False))
    fs = F.ParkFsTier(spec, prim.get_kv_memoryview(), "ParkFsTier", root_dir=root, n_read_threads=2, n_write_threads=2)
    mgr = TieringOffloadingManager(prim, [fs])
    return mgr, prim, fs, prim._policy


def key(s, i):
    return make_offload_key(f"{s}{i:03d}".encode().ljust(16, b"."), 0)


def settle(mgr, fs):
    fs.drain_jobs()
    mgr._process_finished_jobs()
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))


def store(mgr, prim, sess, keys):
    """A request stores keys GPU -> CPU (the slot gets a byte pattern), finishes, and its cascades settle."""
    ctx = ReqContext(req_id=f"r-{sess}", kv_transfer_params={"park_key": f"{sess}-1"})
    mgr.on_new_request(ctx)
    out = mgr.prepare_store(keys, ctx)
    if out is not None:
        for k, cid in zip(out.keys_to_store, out.store_spec.chunk_ids):
            prim._test_flat[int(cid) * B:(int(cid) + 1) * B] = bytes([k[0]]) * B
        mgr.complete_store(out.keys_to_store, ctx)
    mgr.on_request_finished(ctx)
    return out


def ref(prim, k):
    c = prim._policy.get(k)
    return None if c is None else c.ref_cnt


root = tempfile.mkdtemp(prefix="pp_demote_", dir=os.path.expanduser("~/.cache"))
try:
    mgr, prim, fs, pol = build(8, root)
    if MODE == "demote":
        assert demote.install(mgr) is not None
    A = [key("a", i) for i in range(6)]
    store(mgr, prim, "a", A)
    settle(mgr, fs)
    spill = A[:2]
    for k in spill:
        STATE.spill[k] = "a"                    # rule: "none", spill on eviction
    pol.apply(spill, -1.0)                      # first out
    if MODE == "ahead":
        # write-ahead as built for KD on vLLM 0.30: the look-ahead (largest recent burst: the whole session) pins
        # the next victims with their disk writes before they are evicted
        job = mgr.create_store_job(A, ReqContext(req_id="ahead", kv_transfer_params={"park_disk": True}), 0)
        fs.submit_store(job)
    Bk = [key("b", i) for i in range(4)]
    out = store(mgr, prim, "b", Bk)
    assert out is not None, "B's store was refused"
    for k in spill:
        assert mgr.lookup(k, ReqContext(req_id="peek")) is LookupResult.HIT, "victim left the CPU tier before written"
    # B's own chunks are pinned by their cascade, as in stock vLLM; of A's chunks only the victims may be pinned
    pinned = [k for k in A if (ref(prim, k) or 0) > 0]
    assert sorted(pinned) == sorted(spill), f"pinned {pinned}, want only the victims"
    assert len(out.evicted_keys) == 2 and set(out.evicted_keys) <= set(A[2:]), out.evicted_keys
    assert prim._num_evictable_cache_chunks == len(pol.evictable), "manager and policy disagree on evictable chunks"
    settle(mgr, fs)
    for k in spill:
        path = fs.file_mapper.get_file_name(k)
        assert k in fs._on_disk and os.path.exists(path), "victim not on disk"
        with open(path, "rb") as f:
            assert f.read() == bytes([k[0]]) * B, "disk copy differs"
        assert ref(prim, k) == 0 and k in pol.evictable and k not in pol.demoting
    # once written the victims go first: the next store evicts them (plain drop, they have a disk copy)
    out = store(mgr, prim, "c", [key("c", 0), key("c", 1)])
    assert out is not None and sorted(out.evicted_keys) == sorted(spill), out.evicted_keys
    settle(mgr, fs)
    # a returning request restores a victim from disk into the CPU tier, same bytes
    ctx = ReqContext(req_id="r-a2", kv_transfer_params={"park_key": "a-2"})
    mgr.on_new_request(ctx)
    assert mgr.lookup(spill[0], ctx) is LookupResult.HIT_PENDING      # promotion started
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))
    settle(mgr, fs)
    assert mgr.lookup(spill[0], ctx) is LookupResult.HIT
    cid = int(mgr.prepare_load([spill[0]], ctx).chunk_ids[0])
    assert bytes(prim._test_flat[cid * B:(cid + 1) * B]) == bytes([spill[0][0]]) * B, "restored bytes differ"
    mgr.complete_load([spill[0]], ctx)
    mgr.on_request_finished(ctx)
    print("demote ok: victims written on eviction, store served by other chunks, restored from disk")

    # fallback 1: the store protects all but the two spill victims, so no other slot can be freed: drop them
    shutil.rmtree(root); os.makedirs(root)
    mgr, prim, fs, pol = build(8, root)
    if MODE == "demote":
        dem = demote.install(mgr)
    A = [key("d", i) for i in range(6)]
    store(mgr, prim, "d", A)
    settle(mgr, fs)
    for k in A:
        STATE.spill[k] = "d"
    out = store(mgr, prim, "d", A[2:] + [key("e", i) for i in range(4)])
    assert out is not None and sorted(out.evicted_keys) == sorted(A[:2]), "fallback refused the store"
    assert not pol.demoting and not any((ref(prim, k) or 0) > 0 for k in A), "fallback pinned chunks"
    assert MODE != "demote" or dem.dropped == 2
    # fallback 2: evicting half the tier leaves no reserve to demote into: plain drop
    shutil.rmtree(root); os.makedirs(root)
    mgr, prim, fs, pol = build(4, root)
    if MODE == "demote":
        demote.install(mgr)
    A = [key("f", i) for i in range(4)]
    store(mgr, prim, "f", A)
    settle(mgr, fs)
    for k in A:
        STATE.spill[k] = "f"
    out = store(mgr, prim, "g", [key("g", 0), key("g", 1)])
    assert out is not None and len(out.evicted_keys) == 2, "fallback refused the store"
    assert not pol.demoting and not any((ref(prim, k) or 0) > 0 for k in A), "fallback pinned chunks"
    print("fallback ok: no other slot or no reserve, victims dropped, store served")
finally:
    if "fs" in globals():
        fs.drain_jobs()                         # no write may land after the directory is removed
    shutil.rmtree(root, ignore_errors=True)
