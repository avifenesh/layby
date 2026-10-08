"""A request the offload tier defers while it promotes its prefix from disk gives up once the wait exceeds the
recompute it would save (vLLM 0.31 offloading connector scheduler, tiering manager, CPU primary tier with
ParkCachePolicy and ParkFsTier; CPU only).

Measured on GLM 5.3 Flash under heavy load with a saturated disk: get_num_new_matched_tokens returned None step
after step while the disk queue held the promotion, and 33 resumed turns waited 30 to 42 minutes for their first
token. The base offloading connector has no bound on that wait: here a promotion that never completes keeps the
base deferring forever (documented below with the base scheduler called directly).

ParkConnector's rule: a deferral runs until the time since the first deferral exceeds the cost of recomputing
the tokens it waits on, (prompt - local hit - tokens ready now) * f0 + t0, priced as layby.rule prices a recompute
(x (1 + requests in prefill + admissions per second * x)). Then the base's own lookup runs with every chunk that
is not ready read as a miss, so the scheduler admits the request with the prefix the CPU tier can load now (or
nothing) and recomputes the rest. The promotion stays in flight and lands in the CPU tier when the disk delivers
it; the CPU manager's evictable count and ref counts must stay consistent through that, with or without the
request still running.

Before the change: the request is still deferred 1000 s past its recompute cost. After it: give-up with 0 external
tokens for a prompt fully pending on disk, give-up with a partial hit (the ready chunks) when the head of the
prefix is in the CPU tier, counts and rlog rows, and a consistent CPU tier once the promotions complete.

Run: PYTHONPATH=. python tests/test_defer_cap.py
"""
import mmap
import os
import shutil
import sys
import tempfile
from types import SimpleNamespace as NS

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import OffloadingWorkerMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.base import ReqContext, ScheduleEndContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager
from vllm.v1.outputs import KVConnectorOutput

import layby.vllm.connector as C
import layby.vllm.fs_tier as F
from layby.vllm.state import STATE

B = 4096                                    # bytes per chunk
T = 16                                      # tokens per block, per chunk and per hash
F.probe_disk = lambda *a, **k: None         # no disk probe in a unit test

clock = [__import__("time").monotonic()]    # the connector's clock, moved by hand; every other module keeps time.monotonic
C.time = NS(monotonic=lambda: clock[0])


def build(n_chunks, root):
    STATE.__init__()
    STATE.chunk_tokens = STATE.block_tokens = STATE.hash_tokens = T
    STATE.bytes_per_token = B / T
    prim = object.__new__(CPUPrimaryTierOffloadingManager)
    CPUOffloadingManager.__init__(prim, num_chunks=n_chunks, cache_policy="ParkCachePolicy",
                                  cache_policy_module_path="layby.vllm.policy")
    prim.complete_read, prim.prepare_write, prim.complete_write = prim.complete_load, prim.prepare_store, prim.complete_store
    buf = mmap.mmap(-1, n_chunks * B)
    prim._kv_memoryview = memoryview(buf).cast("B", (n_chunks, B))
    par = NS(tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, is_parallelism_agnostic=True)
    spec = NS(blocks_per_chunk=1, tokens_per_hash=T, offload_prompt_only=False, tokens_per_block=[T],
              kv_events_config=NS(enable_kv_cache_events=False, self_describing_kv_events=False),
              config=NS(groups=[NS(group_id=0, tokens_per_block=T, layer_names=["l0"])], model=NS(name="test/m", dtype="bf16"),
                        cache=NS(tokens_per_hash=T), parallel=par, canonical_layout=False, replicated_layout=False))
    fs = F.ParkFsTier(spec, prim.get_kv_memoryview(), "ParkFsTier", root_dir=root, n_read_threads=2, n_write_threads=2)
    mgr = TieringOffloadingManager(prim, [fs])
    spec.get_manager = lambda: mgr
    kvc = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(
        layer_names=["l0"], kv_cache_spec=FullAttentionSpec(block_size=T, num_kv_heads=1, head_size=8, dtype=torch.bfloat16))])
    vc = NS(cache_config=NS(prefix_cache_retention_interval=None, enable_prefix_caching=True), speculative_config=None,
            parallel_config=NS(world_size=1, decode_context_parallel_size=1))
    cs = OffloadingConnectorScheduler(spec, vc, kvc)
    # ParkConnector's scheduler side without a vLLM config: the request-tracking state its __init__ sets
    conn = object.__new__(C.ParkConnector)
    conn.connector_scheduler, conn.connector_worker = cs, None
    conn.__dict__["_bounding_group_ids"] = ()          # every group is offloaded: no bound on the load
    conn._waiting, conn._deferred, conn._local, conn._giveups = {}, {}, {}, {}
    conn._loads, conn._finished, conn._early, conn._hints_on = {}, {}, {}, False
    return conn, cs, mgr, prim, fs, prim._policy


def hashes(s, n):
    return [f"{s}{i:03d}".encode().ljust(16, b".") for i in range(n)]


def key(h):
    return make_offload_key(h, 0)


def settle(mgr, fs):
    fs.drain_jobs()
    mgr._process_finished_jobs()
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))


def store(mgr, req, keys):
    ctx = ReqContext(req_id=f"r-{req}", kv_transfer_params={"park_key": req})
    mgr.on_new_request(ctx)
    out = mgr.prepare_store(keys, ctx)
    assert out is not None, f"{req}: store refused"
    mgr.complete_store(out.keys_to_store, ctx)
    mgr.on_request_finished(ctx)
    return out


def check(prim, pol):
    """The manager's evictable count is the number of resident chunks with ref_cnt 0; no two keys share a slot
    (tests/test_park_vllm_double_hint.py)."""
    ids = [c.chunk_id for c in pol.chunks.values()]
    assert len(ids) == len(set(ids)), "two keys share one CPU slot"
    assert len(prim._free_list) == len(set(prim._free_list)), f"a slot is free twice: {prim._free_list}"
    assert not set(ids) & set(prim._free_list), "a resident chunk's slot is on the free list"
    free_refs = sum(1 for c in pol.chunks.values() if c.ref_cnt == 0)
    assert prim._num_evictable_cache_chunks == free_refs, \
        f"manager counts {prim._num_evictable_cache_chunks} evictable chunks, {free_refs} have ref_cnt 0"
    assert pol.evictable <= {k for k, c in pol.chunks.items() if c.ref_cnt == 0}, "an evictable chunk is pinned"
    pend = sum(1 for c in pol.chunks.values() if c.ref_cnt == -1)
    assert prim._num_write_pending_chunks == pend, f"manager counts {prim._num_write_pending_chunks} pending writes, {pend} chunks have ref_cnt -1"


def request(rid, hs, prompt):
    return NS(request_id=rid, num_computed_tokens=0, num_prompt_tokens=prompt, num_tokens=prompt, block_hashes=hs,
              kv_transfer_params={"park_key": rid}, kv_hints=None, skip_reading_prefix_cache=False,
              is_finished=lambda: False)


def lookup(conn, cs, mgr, req, local=0):
    """One scheduler step for a waiting request: the connector's lookup, then the end of the schedule (the tiering
    manager submits the promotions the lookup queued)."""
    out = conn.get_num_new_matched_tokens(req, local)
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))
    return out


def blocks_for(n_tokens, first_id=10):
    n = -(-n_tokens // T)
    return NS(blocks=([NS(block_id=first_id + i, is_null=False, block_hash=None) for i in range(n)],))


def recompute_cost(n):
    p = STATE.params()
    return n * p.f0 + p.t0


root = tempfile.mkdtemp(prefix="pp_dcap_", dir=os.path.expanduser("~/.cache"))
try:
    conn, cs, mgr, prim, fs, pol = build(8, root)
    held = []                                              # promotions the disk never serves until the test says so
    real_submit_load = fs.submit_load
    fs.submit_load = lambda job: held.append(job)

    # session a: 4 chunks in the CPU tier, written to disk, then evicted from the CPU tier by session c
    HA = hashes("a", 4)
    A = [key(h) for h in HA]
    store(mgr, "a-1", A)
    fs.queue_write(A)
    settle(mgr, fs); settle(mgr, fs)
    assert all(k in fs._on_disk for k in A), "session a is not on disk"
    HC = hashes("c", 8)
    store(mgr, "c-1", [key(h) for h in HC])
    settle(mgr, fs)
    assert not any(k in pol.chunks for k in A), "session a still in the CPU tier"
    check(prim, pol)

    # --- 1. the base alone: deferred forever ------------------------------------------------------------
    HP = HA + hashes("p", 1)                               # a's prefix plus a new chunk: 80 prompt tokens
    r0 = request("r0", HP, 5 * T)
    conn.on_new_request(r0)
    out = lookup(conn, cs, mgr, r0)
    assert out[0] is None, f"first lookup should defer (promotion from disk), got {out}"
    assert len(held) == 1 and sorted(held[0].keys) == sorted(A), "promotion of a's chunks not submitted"
    base_cap = recompute_cost(5 * T)
    clock[0] += base_cap + 1000.0
    base = OffloadingConnectorScheduler.get_num_new_matched_tokens(cs, r0, 0)
    assert base[0] is None, f"the base scheduler no longer defers: {base}"

    # --- 2. the rule: admitted with nothing external once the cap passes ---------------------------------
    out = lookup(conn, cs, mgr, r0)
    assert out[0] is not None, "deferred 1000 s past its recompute cost: the cap did not fire"
    assert out == (0, False), f"nothing of r0's prefix is ready: expected (0, False), got {out}"
    assert len(held) == 1, "the give-up lookup started a second promotion"
    assert STATE.counts["defer_giveup"] == 1
    # no GPU blocks for it this step: the scheduler asks again next step, the give-up is counted once
    out = lookup(conn, cs, mgr, r0)
    assert out == (0, False) and STATE.counts["defer_giveup"] == 1, (out, STATE.counts)
    conn.update_state_after_alloc(r0, blocks_for(0), 0)   # the scheduler admits it: nothing to load
    assert "r0" not in conn._deferred and "r0" not in conn._waiting
    assert STATE.counts["defer_giveup"] == 1
    row = STATE.rlog[-1]
    assert row.get("giveup") is True and row["P"] == 5 * T and row["local"] == 0 and row["ext"] == 0, row
    assert row["rec"] == 5 * T and row["wait"] > base_cap, row
    check(prim, pol)

    # --- 3. the rule under the cap: still waiting --------------------------------------------------------
    r1 = request("r1", HP, 5 * T)
    conn.on_new_request(r1)
    out = lookup(conn, cs, mgr, r1)
    assert out[0] is None, out
    clock[0] += recompute_cost(5 * T) * 0.5
    out = lookup(conn, cs, mgr, r1)
    assert out[0] is None, f"gave up under the cap: {out}"
    assert STATE.counts["defer_giveup"] == 1
    # the wait counts from the first deferral: 0.5 + 0.6 of the cost passes it
    clock[0] += recompute_cost(5 * T) * 0.6
    out = lookup(conn, cs, mgr, r1)
    assert out == (0, False), out
    conn.update_state_after_alloc(r1, blocks_for(0), 0)
    assert STATE.counts["defer_giveup"] == 2
    assert len(held) == 1, "r1 should share r0's in-flight promotion, not start one"

    # --- 4. partial hit: the head of the prefix is ready in the CPU tier, the tail is on disk -------------
    # r0 finishes and the promotion completes meanwhile: a's chunks land in the CPU tier for a request that stopped
    # waiting for them; then they are evicted but the first two are stored back
    conn.request_finished(r0, [10, 11, 12, 13, 14])
    mgr.on_request_finished(cs._req_status["r0"].req_context)
    real_submit_load(held.pop())
    settle(mgr, fs)
    assert all(k in pol.chunks and pol.chunks[k].is_ready for k in A), "the promotion did not land"
    check(prim, pol)
    store(mgr, "c-2", [key(h) for h in hashes("d", 8)])   # evicts everything idle
    settle(mgr, fs)
    assert not any(k in pol.chunks for k in A), "a still resident"
    store(mgr, "a-2", A[:2])
    settle(mgr, fs)
    check(prim, pol)
    r2 = request("r2", HP, 5 * T)
    conn.on_new_request(r2)
    out = lookup(conn, cs, mgr, r2)
    assert out[0] is None, out
    assert len(held) == 1 and sorted(held[0].keys) == sorted(A[2:]), "promotion of a's tail not submitted"
    # the cap is the recompute of the 3 chunks not ready, less than the recompute of all 5
    clock[0] += (recompute_cost(3 * T) + recompute_cost(5 * T)) / 2
    out = lookup(conn, cs, mgr, r2)
    assert out == (2 * T, True), f"expected the 2 ready chunks, got {out}"
    conn.update_state_after_alloc(r2, blocks_for(2 * T), 2 * T)
    (job_id, job), = cs._current_batch_load_jobs.items()
    assert list(job.src_spec.chunk_ids) == [pol.chunks[k].chunk_id for k in A[:2]], "load job does not read the ready chunks"
    assert all(pol.chunks[k].ref_cnt == 1 for k in A[:2]), "the load did not pin the ready chunks"
    check(prim, pol)
    row = STATE.rlog[-1]
    assert row.get("giveup") is True and row["ext"] == 2 * T and row["rec"] == 3 * T, row
    assert STATE.counts["defer_giveup"] == 3
    # the GPU load completes
    cs._current_batch_load_jobs.clear()
    wm = OffloadingWorkerMetadata(completed_jobs={job_id: 1})
    conn.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=wm))
    assert all(pol.chunks[k].ref_cnt == 0 for k in A[:2])
    check(prim, pol)
    # the promotion of the tail completes while r2 runs: no assert, consistent counts, the tail is ready
    real_submit_load(held.pop())
    settle(mgr, fs)
    assert all(k in pol.chunks and pol.chunks[k].is_ready for k in A), "the tail promotion did not land"
    check(prim, pol)
    # r2 stores its recomputed chunks (the ready ones are skipped as present), finishes
    ctx2 = cs._req_status["r2"].req_context
    out = mgr.prepare_store([key(h) for h in HP], ctx2)
    assert out is not None and sorted(out.keys_to_store) == [key(HP[4])], out.keys_to_store
    mgr.complete_store(out.keys_to_store, ctx2)
    conn.request_finished(r2, [10, 11, 12, 13, 14])
    mgr.on_request_finished(ctx2)
    settle(mgr, fs)
    check(prim, pol)

    # --- 5. the seconds saved: measured when the promotion the request gave up on resolves ---------------
    saved = STATE.live.acc["giveup_saved"]
    conn._settle_giveups(clock[0])
    assert STATE.live.acc["n_giveup"] == 3.0, STATE.live.acc
    assert STATE.live.acc["giveup_saved"] > saved, "no saving recorded for a promotion that completed after the give-up"
    assert not conn._giveups, conn._giveups
    print("defer cap ok: base defers forever; rule admits at the cap with 0 or the ready prefix; promotions land safely")
finally:
    shutil.rmtree(root, ignore_errors=True)
