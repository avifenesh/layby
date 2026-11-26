"""T2: frequency-gated disk writes (layby.vllm.freq_tier) on vLLM 0.31's native fs tier, CPU only.

1. FrequencyFilter reproduces NVIDIA Dynamo KVBM's FrequencyFilter (lib/llm/src/block_manager/offload/filter.rs, its
   own unit tests ported: basic, decay by size bound, time-based decay 4 -> 3 -> 2 -> 1 -> pruned, multi-round prune)
   and admits on the n-th sighting for n = 2 and 3, with decay over a fixed interval, and without decay.
2. FreqFsTier writes only admitted chunks, completes a job with nothing admitted without I/O, skips a chunk on disk or
   in flight without counting it.
3. Through vLLM's real tiering manager, CPU tier with stock LRU and real files under a temp directory: with request
   counting, a session's first request writes nothing, its next request (the same prefix plus new chunks) writes the
   prefix it reused and not its new tail; the files are the native tier's (FileMapper paths). With store counting, a
   prefix that stays in the CPU tier is never written; after the CPU tier evicted it and it is stored again, it is.
Run: PYTHONPATH=. python tests/test_freq_tier.py
"""
import mmap
import os
import shutil
import tempfile
from types import SimpleNamespace as NS

import numpy as np
from vllm.v1.kv_offload.base import LookupResult, OffloadPolicy, ReqContext, ScheduleEndContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager, TieringOffloadingManager

import layby.vllm.freq_tier as FT
from layby.vllm.freq_tier import FrequencyFilter, FreqFsTier

clock = [1000.0]
tick = lambda: clock[0]


# --- 1. the filter ---------------------------------------------------------------------------------------------
def kvbm(min_freq, period, max_entries):
    """KVBM's filter, parameterized by its own count threshold (min_offload_frequency): n = log2(min) + 1."""
    f = FrequencyFilter(2, period, max_entries, clock=tick)
    f.threshold = min_freq
    return f


f = kvbm(2, 3600, 100)                                   # test_basic_frequency_filter
assert [f.should_offload(h) for h in (0, 0, 1, 2)] == [False, True, False, False]

f = kvbm(4, 3600, 2)                                     # test_decay (size-bound prune)
assert not f.should_offload(0) and not f.should_offload(0)
assert not f.should_offload(1) and not f.should_offload(1)
assert f.should_offload(0)                               # 4 >= 4
assert not f.should_offload(2)                           # 3 entries > 2: prune rounds until 2 fit
assert not f.should_offload(1)

f = kvbm(4, 0.25, 100)                                   # test_time_based_decay
assert [f.should_offload(0) for _ in range(3)] == [False, False, True]
for want in (3, 2, 1):
    clock[0] += 0.25 + (0.05 if want == 3 else 0.0)
    f.decay()
    assert f.counts[0] == want, (want, f.counts)
clock[0] += 0.25
f.decay()
assert 0 not in f.counts

f = kvbm(10, 3600, 2)                                    # test_multi_prune_decay
f.should_offload(0); f.should_offload(1)
assert len(f.counts) == 2
f.should_offload(2)
assert not f.counts, f.counts
f.should_offload(3)
assert len(f.counts) == 1

# n-th sighting, decay every 600 s on a fixed interval, no decay
f = FrequencyFilter(2, 600.0, clock=tick)
assert f.threshold == 2 and [f.should_offload("a") for _ in range(2)] == [False, True]
f = FrequencyFilter(3, 600.0, clock=tick)
assert f.threshold == 4 and [f.should_offload("a") for _ in range(3)] == [False, False, True]
f = FrequencyFilter(2, 600.0, clock=tick)
t0 = clock[0]
f.should_offload("a")                                    # count 1
clock[0] = t0 + 599.0
assert f.should_offload("a")                             # still within the first interval: second sighting writes
f.should_offload("b")                                    # b: 1 at t0 + 599
clock[0] = t0 + 601.0                                    # the interval boundary at t0 + 600 passed: b forgotten
assert not f.should_offload("b") and f.counts["b"] == 1, "a count survived its decay step"
f = FrequencyFilter(3, 600.0, clock=tick)
t0 = clock[0]
f.should_offload("c"); f.should_offload("c")             # 2
clock[0] = t0 + 600.0                                    # one step: 1
assert not f.should_offload("c") and f.counts["c"] == 2  # third sighting, but the decay cost one: 1 -> 2 < 4
assert f.should_offload("c")                             # 4
f = FrequencyFilter(2, 0, clock=tick)                    # decay off (SGLang)
f.should_offload("d")
clock[0] += 1e7
assert f.should_offload("d"), "a count decayed with decay off"
print("filter ok: KVBM's semantics, n-th sighting, fixed-interval decay")

# --- 2. the tier, mocked I/O -----------------------------------------------------------------------------------
stored, noio = [], []
orig_submit, orig_finished = FileSystemTierManager.submit_store, FileSystemTierManager.get_finished_jobs
FileSystemTierManager.submit_store = lambda self, j: stored.append(list(j.keys))
FileSystemTierManager.get_finished_jobs = lambda self: self._fin


class Pool:
    def enqueue_store(self, jid, n, tasks):
        noio.append(jid)


t = object.__new__(FreqFsTier)
t.filter, t.count_mode, t._pool, t._fin = FrequencyFilter(2, 600.0, clock=tick), "request", Pool(), []
t._init_index()
ctx = ReqContext(req_id="r", kv_transfer_params={})
job = lambda i, keys: TransferJob(job_id=i, keys=keys, chunk_ids=np.arange(len(keys)), is_promotion=False, req_context=ctx)
assert t.on_new_request(ctx).policy == OffloadPolicy.REQUEST_LEVEL
t.submit_store(job(1, ["a", "b", "c"]))                 # first sighting of each: no I/O
assert stored == [] and noio == [1]
t.submit_store(job(2, ["a", "b", "c", "d"]))            # a b c second sighting: written; d first
assert stored == [["a", "b", "c"]] and t._inflight == {"a", "b", "c"}
t.submit_store(job(3, ["a", "b", "d"]))                 # a b in flight: skipped, not counted; d second: written
assert stored[-1] == ["d"] and t.filter.counts["a"] == 2, t.filter.counts
t._fin = [JobResult(job_id=2, success=True), JobResult(job_id=3, success=False)]
t.get_finished_jobs()
assert t._written == {"a", "b", "c"} and not t._inflight and not t._writing
t.submit_store(job(4, ["a", "b", "c"]))                 # on disk: no I/O, no sighting
assert noio[-1] == 4 and t.filter.counts["a"] == 2
t.count_mode = "store"
assert t.on_new_request(ctx).policy == OffloadPolicy.CHUNK_LEVEL
FileSystemTierManager.submit_store, FileSystemTierManager.get_finished_jobs = orig_submit, orig_finished
print("tier ok: admitted chunks only, no-I/O completion, on-disk chunks not counted")

# --- 3. through vLLM's tiering manager, real files -------------------------------------------------------------
B, T = 4096, 16


def build(n_chunks, root, count):
    prim = object.__new__(CPUPrimaryTierOffloadingManager)
    CPUOffloadingManager.__init__(prim, num_chunks=n_chunks, cache_policy="lru")
    prim.complete_read, prim.prepare_write, prim.complete_write = prim.complete_load, prim.prepare_store, prim.complete_store
    prim._kv_memoryview = memoryview(mmap.mmap(-1, n_chunks * B)).cast("B", (n_chunks, B))
    par = NS(tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, is_parallelism_agnostic=True)
    spec = NS(blocks_per_chunk=1, tokens_per_hash=T, offload_prompt_only=False, tokens_per_block=[T],
              kv_events_config=NS(enable_kv_cache_events=False, self_describing_kv_events=False),
              config=NS(groups=[NS(group_id=0, tokens_per_block=T, layer_names=["l0"])], model=NS(name="test/m", dtype="bf16"),
                        cache=NS(tokens_per_hash=T), parallel=par, canonical_layout=False, replicated_layout=False))
    fs = FreqFsTier(spec, prim.get_kv_memoryview(), "FreqFsTier", root_dir=root, n_read_threads=2, n_write_threads=2,
                    min_hits=2, decay_s=600, count=count)
    return TieringOffloadingManager(prim, [fs]), prim, fs


def keys(s, n):
    return [make_offload_key(f"{s}{i:03d}".encode().ljust(16, b"."), 0) for i in range(n)]


def settle(mgr, fs):
    fs.drain_jobs()
    mgr._process_finished_jobs()
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))


def request(mgr, fs, rid, ks):
    """One request offering its chunks for storing (as the offloading scheduler does from chunk 0), then finishing."""
    ctx = ReqContext(req_id=rid, kv_transfer_params={})
    mgr.on_new_request(ctx)
    out = mgr.prepare_store(ks, ctx)
    assert out is not None, f"{rid}: store refused"
    if out.keys_to_store:                                  # the scheduler completes only stores it issued
        mgr.complete_store(out.keys_to_store, ctx)
    settle(mgr, fs)
    mgr.on_request_finished(ctx)
    settle(mgr, fs)


def on_disk(fs, ks):
    return [os.path.exists(fs.file_mapper.get_file_name(k)) for k in ks]


root = tempfile.mkdtemp(prefix="pp_t2_", dir=os.path.expanduser("~/.cache"))
try:
    # request counting (SGLang's hit count): the reused prefix goes to disk on the next turn, the new tail does not
    mgr, prim, fs = build(16, os.path.join(root, "req"), "request")
    A = keys("a", 3)
    request(mgr, fs, "a-0", A)
    assert on_disk(fs, A) == [False] * 3, "first sighting wrote"
    A2 = A + keys("b", 2)
    request(mgr, fs, "a-1", A2)
    assert on_disk(fs, A2) == [True, True, True, False, False], on_disk(fs, A2)
    assert fs._written == set(A), fs._written
    assert all(prim.lookup(k, ReqContext(req_id="x", kv_transfer_params={})) is LookupResult.HIT for k in A2), "the CPU tier lost chunks"
    request(mgr, fs, "a-2", A2 + keys("c", 1))
    assert on_disk(fs, A2) == [True] * 5
    # the native lookup reads the files the filter let through
    ctx = ReqContext(req_id="probe", kv_transfer_params={})
    assert fs.lookup(A[0], ctx).name in ("HIT", "RETRY")
    print("tiering ok (request counting): turn 1 writes nothing, turn 2 writes the reused prefix")

    # store counting (KVBM's sighting): a prefix resident in the CPU tier is sighted once
    mgr, prim, fs = build(4, os.path.join(root, "store"), "store")
    P = keys("p", 2)
    request(mgr, fs, "p-0", P)
    request(mgr, fs, "p-1", P)
    assert on_disk(fs, P) == [False, False], "a resident prefix was written"
    request(mgr, fs, "q-0", keys("q", 4))                  # evicts p from the 4-chunk CPU tier
    assert all(prim.lookup(k, ReqContext(req_id="x", kv_transfer_params={})) is LookupResult.MISS for k in P), "p still resident"
    request(mgr, fs, "p-2", P)                             # p stored again: second sighting
    assert on_disk(fs, P) == [True, True], on_disk(fs, P)
    print("tiering ok (store counting): written only on re-entry into the CPU tier")
finally:
    shutil.rmtree(root, ignore_errors=True)
print("T2 ok")
