"""Two restores admitted together on a sliding-window model leave nothing able to run (vLLM 0.31 KV cache manager,
Gemma 4 31B's KV groups, ParkConnector; CPU only).

Measured in the Phase R pilot 2 K arm on Gemma 4 31B (5 sliding-window groups, window 1024, block 16, and one global
group, block 32; 2,188 GPU blocks): two returning sessions waited while their prefixes came back from disk, both
promotions landed in the same disk burst, and the scheduler admitted both CPU-to-GPU loads in one pass. From then on:
"Running: 0 reqs, Waiting: 2 reqs, Deferred: 2 reqs, GPU KV cache usage: 92.5%", for 40+ minutes.

vLLM's admission undercounts a sliding-window group for a request with a long prefix hit: get_num_blocks_to_allocate
caps the required blocks at max_admission_blocks_per_request (a count of blocks held) and then subtracts the skipped
block positions below the window (a position), so it counts 0. The full-sequence check and the in-flight reservation
(Scheduler._request_remaining_blocks) see only the global group, while the allocation takes the window too: here 719
blocks counted, 1,006 taken, a 33-block reservation for a request that needs 673 more. Two such loads take 92% of the
pool, the next prefill chunk of either does not fit, and the waiting loop never preempts a waiting request that holds
blocks.

ParkConnector's gate: while another request holds an async load that has not started running, a new load is admitted
only if the GPU's free blocks cover what it and every such holder still need, counted per group as the allocator
holds them (a sliding-window group's peak is its admission cap, null blocks are not held). Otherwise the lookup
defers the request (no blocks held), and it is admitted once the holder runs.

Before the change: the second load is admitted and neither request's next chunk can be allocated. After it: the
second request is deferred, the first one's next chunk fits, and the second is admitted once the first runs.

Run: PYTHONPATH=. python tests/test_park_vllm_swa_admit.py
"""
from types import SimpleNamespace as NS

import torch
from vllm import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
from vllm.v1.request import Request, RequestStatus

import layby.vllm.connector as C
from layby.vllm.state import STATE

N_BLOCKS = 2188                 # Gemma 4 31B, 5,735,448,576 B of GPU KV in 2,621,440 B blocks
IN_FLIGHT = 2 * 1024            # two concurrent batches of --max-num-batched-tokens 1024
PROMPT, HIT = 23000, 21952      # a returning session: its prefix restored from the CPU tier (1,006 GPU blocks)

init_none_hash(sha256)
hasher = get_request_block_hasher(16, sha256)


def gemma_kv():
    swa = [KVCacheGroupSpec([f"swa{g}"], SlidingWindowSpec(block_size=16, num_kv_heads=16, head_size=256,
                                                            dtype=torch.bfloat16, sliding_window=1024))
           for g in range(5)]
    glob = KVCacheGroupSpec(["global"], FullAttentionSpec(block_size=32, num_kv_heads=4, head_size=512,
                                                          dtype=torch.bfloat16))
    cfg = KVCacheConfig(num_blocks=N_BLOCKS, kv_cache_tensors=[], kv_cache_groups=swa + [glob])
    return KVCacheManager(cfg, max_model_len=32768, scheduler_block_size=32, hash_block_size=16,
                          max_in_flight_tokens=IN_FLIGHT)


class Base:
    """The offloading scheduler, reduced to its answers: every prefix is a CPU-tier hit of HIT tokens."""

    def __init__(self):
        self._req_status, self.config = {}, NS(kv_group_configs=[])

    def on_new_request(self, request):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens, max_num_new_tokens=None):
        return HIT, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return NS(load_jobs={})

    def request_finished(self, request):
        return False, None


def connector(kv):
    STATE.__init__()
    STATE.block_tokens = STATE.hash_tokens = STATE.chunk_tokens = 16
    conn = object.__new__(C.ParkConnector)
    conn.connector_scheduler, conn.connector_worker = Base(), None
    conn._kv_cache_manager = kv
    conn.__dict__["_bounding_group_ids"] = ()          # every group is offloaded: no bound on the load
    # the request-tracking state ParkConnector.__init__ sets
    conn._waiting, conn._deferred, conn._local, conn._giveups, conn._holding = {}, {}, {}, {}, {}
    conn._loads, conn._finished, conn._early, conn._hints_on = {}, {}, {}, False
    conn._last_step, conn._last_prune = None, float("inf")
    return conn


def request(rid, seed):
    r = Request(rid, [seed * 100000 + i for i in range(PROMPT)], SamplingParams(max_tokens=384), None,
                block_hasher=hasher)
    r.status = RequestStatus.WAITING
    return r


def remaining(kv, r):
    """Scheduler._request_remaining_blocks: what vLLM reserves for a load in flight."""
    return kv.coordinator.get_num_blocks_to_allocate(
        request_id=r.request_id, num_tokens=r.num_tokens, new_computed_blocks=kv.empty_kv_cache_blocks.blocks,
        num_encoder_tokens=0, total_computed_tokens=r.num_computed_tokens,
        num_local_computed_tokens=r.num_computed_tokens, num_tokens_main_model=r.num_tokens, apply_admission_cap=True,
        prefill_end=r.num_prompt_tokens)


def admit(conn, kv, r, inflight):
    """The scheduler's waiting loop for a fresh request with an async external hit (scheduler.py, schedule())."""
    conn.on_new_request(r)
    ext, load_async = conn.get_num_new_matched_tokens(r, 0)
    if ext is None:
        return "deferred"
    assert load_async and ext == HIT, (ext, load_async)
    reserved = sum(remaining(kv, x) for x in inflight)
    blocks = kv.allocate_slots(r, 0, num_external_computed_tokens=ext, delay_cache_blocks=True,
                               full_sequence_must_fit=True, reserved_blocks=reserved, has_scheduled_reqs=False)
    if blocks is None:
        return "refused"
    conn.update_state_after_alloc(r, blocks, ext)
    r.status, r.num_computed_tokens = RequestStatus.WAITING_FOR_REMOTE_KVS, ext
    inflight.append(r)
    return "loading"


def loaded(kv, r):
    """_update_waiting_for_remote_kv: the load completed, the request waits for its next prefill chunk."""
    kv.cache_blocks(r, r.num_computed_tokens)
    r.status = RequestStatus.WAITING


def next_chunk(kv, r):
    return kv.allocate_slots(r, min(1024, r.num_tokens - r.num_computed_tokens), full_sequence_must_fit=True,
                             has_scheduled_reqs=False)


def scheduled(conn, ids):
    """build_connector_meta for a step that runs these requests."""
    conn.build_connector_meta(NS(num_scheduled_tokens={i: 1024 for i in ids}, scheduled_new_reqs=[],
                                 scheduled_cached_reqs=NS(is_context_phase=lambda rid: True),
                                 scheduled_encoder_input_stats=None, total_num_scheduled_tokens=1024 * len(ids),
                                 finished_req_ids=set(), preempted_req_ids=set()))


kv = gemma_kv()
conn = connector(kv)
A, B = request("A", 1), request("B", 2)
inflight = []

# --- 1. vLLM's own count: the window is not counted for a long hit ----------------------------------------------
free0 = kv.block_pool.get_num_free_blocks()
assert admit(conn, kv, A, inflight) == "loading"
took = free0 - kv.block_pool.get_num_free_blocks()
assert took == 1006, f"A's restore took {took} blocks (686 global + 5 x 64 window)"
assert remaining(kv, A) == 33, f"vLLM reserves {remaining(kv, A)} blocks for A (global only)"

# --- 2. a second restore while A's load is in flight ---------------------------------------------------------------
second = admit(conn, kv, B, inflight)
if second == "loading":
    loaded(kv, A); loaded(kv, B)
    a, b = next_chunk(kv, A), next_chunk(kv, B)
    raise AssertionError(
        f"B's restore was admitted beside A's: GPU KV {kv.usage:.1%} held by two waiting requests, "
        f"{kv.block_pool.get_num_free_blocks()} blocks free, A's next chunk {'fits' if a else 'does not fit'}, "
        f"B's {'fits' if b else 'does not fit'}; nothing running can free a block")
assert second == "deferred", second
assert B.request_id not in conn._holding and kv.usage < 0.5, "a deferred request must hold no GPU blocks"

# --- 3. A runs; B is admitted once A no longer waits ------------------------------------------------------------------
loaded(kv, A)
assert next_chunk(kv, A) is not None, "A's next prefill chunk does not fit"
A.status = RequestStatus.RUNNING
inflight.remove(A)
scheduled(conn, ["A"])
assert A.request_id not in conn._holding, "A runs: it no longer holds a restore"
third = admit(conn, kv, B, inflight)
# past the gate; vLLM itself refuses while A's prefill holds the room (A runs, so it finishes or is preempted)
assert third == "refused", f"B after A started running: {third}, {kv.block_pool.get_num_free_blocks()} blocks free"
kv.free(A)
conn.request_finished(A, [])
assert admit(conn, kv, B, inflight) == "loading", "B not admitted after A finished"

# --- 4. no holder, no gate: the first restore of an idle engine goes through as in vLLM ------------------------------
kv2 = gemma_kv()
conn2 = connector(kv2)
assert admit(conn2, kv2, request("C", 3), []) == "loading"
# --- 5. an aborted holder (the client's turn timeout) releases its place at once -------------------------------------
D = request("D", 4)
assert admit(conn2, kv2, D, []) == "deferred", "D should wait behind C's restore"
conn2.request_finished(conn2._holding["C"], [])
assert not conn2._holding, conn2._holding
print("swa admit ok: a second restore that would leave both waiting is deferred; admitted once the first runs")
