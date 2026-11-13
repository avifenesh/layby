"""ParkConnector's scheduler side on vLLM 0.31's offloading connector scheduler, tiering manager, CPU primary tier
(ParkCachePolicy) and ParkFsTier, without a vLLM config or a GPU: the pieces tests/test_defer_cap.py builds, for
tests that need the real promotion path."""
import mmap
from types import SimpleNamespace as NS

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import OffloadingConnectorScheduler
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.kv_offload.base import ReqContext, ScheduleEndContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager

import layby.vllm.connector as C
import layby.vllm.fs_tier as F
from layby.vllm.state import STATE

B = 4096                                    # bytes per chunk
T = 16                                      # tokens per block, per chunk and per hash
F.probe_disk = lambda *a, **k: None         # no disk probe in a unit test


def build(n_chunks, root, n_threads=2):
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
    fs = F.ParkFsTier(spec, prim.get_kv_memoryview(), "ParkFsTier", root_dir=root, n_read_threads=n_threads,
                      n_write_threads=n_threads)
    mgr = TieringOffloadingManager(prim, [fs])
    spec.get_manager = lambda: mgr
    kvc = KVCacheConfig(num_blocks=64, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(
        layer_names=["l0"], kv_cache_spec=FullAttentionSpec(block_size=T, num_kv_heads=1, head_size=8, dtype=torch.bfloat16))])
    vc = NS(cache_config=NS(prefix_cache_retention_interval=None, enable_prefix_caching=True), speculative_config=None,
            parallel_config=NS(world_size=1, decode_context_parallel_size=1))
    cs = OffloadingConnectorScheduler(spec, vc, kvc)
    conn = object.__new__(C.ParkConnector)
    conn.connector_scheduler, conn.connector_worker = cs, None
    conn.__dict__["_bounding_group_ids"] = ()
    conn._waiting, conn._deferred, conn._local, conn._giveups, conn._ready, conn._prefetched = {}, {}, {}, {}, {}, {}
    conn._loads, conn._finished, conn._early, conn._hints_on = {}, {}, {}, False
    return conn, cs, mgr, prim, fs, prim._policy


def hashes(s, n):
    return [f"{s}{i:03d}".encode().ljust(16, b".") for i in range(n)]


def key(h):
    return make_offload_key(h, 0)


def end_schedule(mgr):
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))


def settle(mgr, fs):
    fs.drain_jobs()
    mgr._process_finished_jobs()
    end_schedule(mgr)


def store(mgr, req, keys):
    ctx = ReqContext(req_id=f"r-{req}", kv_transfer_params={"park_key": req})
    mgr.on_new_request(ctx)
    out = mgr.prepare_store(keys, ctx)
    assert out is not None, f"{req}: store refused"
    mgr.complete_store(out.keys_to_store, ctx)
    mgr.on_request_finished(ctx)
    return out


def check(prim, pol):
    """The manager's evictable count matches the resident chunks with ref_cnt 0; no two keys share a slot."""
    ids = [c.chunk_id for c in pol.chunks.values()]
    assert len(ids) == len(set(ids)), "two keys share one CPU slot"
    assert len(prim._free_list) == len(set(prim._free_list)), f"a slot is free twice: {prim._free_list}"
    assert not set(ids) & set(prim._free_list), "a resident chunk's slot is on the free list"
    free_refs = sum(1 for c in pol.chunks.values() if c.ref_cnt == 0)
    assert prim._num_evictable_cache_chunks == free_refs, \
        f"manager counts {prim._num_evictable_cache_chunks} evictable chunks, {free_refs} have ref_cnt 0"
    assert pol.evictable <= {k for k, c in pol.chunks.items() if c.ref_cnt == 0}, "an evictable chunk is pinned"


def request(rid, hs, prompt):
    return NS(request_id=rid, num_computed_tokens=0, num_prompt_tokens=prompt, num_tokens=prompt, block_hashes=hs,
              kv_transfer_params={"park_key": rid}, kv_hints=None, skip_reading_prefix_cache=False,
              is_finished=lambda: False)


def lookup(conn, mgr, req, local=0):
    """One scheduler step for a waiting request: the connector's lookup, then the end of the schedule."""
    out = conn.get_num_new_matched_tokens(req, local)
    end_schedule(mgr)
    return out


def blocks_for(n_tokens, first_id=10):
    n = -(-n_tokens // T)
    return NS(blocks=([NS(block_id=first_id + i, is_null=False, block_hash=None) for i in range(n)],))
