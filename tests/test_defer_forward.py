"""The deferral cap decides on the expected remaining wait, not the elapsed one (vLLM 0.31 offloading connector
scheduler, tiering manager, CPU primary tier with ParkCachePolicy and ParkFsTier; CPU only).

Measured on GLM 5.3 Flash under heavy load on a 0.55 GB/s disk (LANE.md 2026-10-09): the elapsed cap took p99 to
the no-disk level but raised p95 1.34x and p50 1.47x over no cap, because 325 and 256 requests each waited the
whole cap (about 125 s) and then recomputed a ~58k-token prompt in full, flooding a prefill-bound engine.

ParkConnector's rule now: for a deferred request, the remaining wait is the bytes the disk tier must still read before
and including this request's promotion (unfinished load jobs submitted up to it, in submit order, then the promotions
the tiering manager holds until the end of the step) at the measured drain rate of the load queue (load bytes completed
per second while loads are unfinished, Speeds "drain"; the per-transfer disk speed until a load completed). The request
keeps waiting while remaining <= the recompute cost of the tokens not ready now (priced as layby.rule prices a recompute)
and gives up as soon as remaining > that cost, which can be the first lookup. Safety: if the elapsed wait exceeds the
first remaining estimate by more than the recompute cost, give up. No estimate (no load job holds the chunk): the
elapsed cap as before (tests/test_defer_cap.py).

Measured on the same setup with the per-transfer speed as the rate (2026-10-09): the estimate at first deferral ran
p50 1.6 to 1.8 s against actual waits of 10 to 12 s (p90 7 to 9 s against 39 to 52 s), so the forward branch never fired
and every give-up came from the safety branch, which is the elapsed cap again. The disk speed is a busy transfer's;
the queue drains slower under store jobs, parallel read threads and a twin engine on the device.

Scenarios, each failing on the elapsed cap and passing on the forward rule:
  1. forward: a promotion queued behind a large load gives up at its first lookup (elapsed cap: waited)
  2. next in line: a small promotion at the head of the queue keeps waiting past the elapsed cap and is admitted with
     the loaded prefix when it lands (elapsed cap: gave up and recomputed); the landed row records the estimate
  3. safety: a promotion that never completes gives up once the wait passes its estimate by the recompute cost
  4. consistency: a request that gave up runs and finishes while its promotion and the one ahead land; the CPU
     tier's counts stay consistent (check() of tests/test_park_vllm_double_hint.py) and the give-up settles
  5. drain: the per-transfer speed says a promotion reads in 16 s (under its 27 s recompute) but the queue measured a
     tenth of that rate: the request gives up through the forward branch at its first lookup (per-transfer rate: it
     waited for the safety bound); the landed row before it holds est0 and link0 beside the wait

Run: PYTHONPATH=. python tests/test_defer_forward.py
"""
import mmap
import os
import shutil
import sys
import tempfile
import traceback
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
from layby.rule import recompute_cost
from layby.vllm.state import STATE

B = 4096                                    # bytes per chunk
T = 16                                      # tokens per block, per chunk and per hash
F.probe_disk = lambda *a, **k: None         # no disk probe in a unit test

clock = [__import__("time").monotonic()]    # the connector's and the disk tier's clock, moved by hand
C.time = F.time = NS(monotonic=lambda: clock[0])


def build(n_chunks, root):
    STATE.__init__()
    STATE.chunk_tokens = STATE.block_tokens = STATE.hash_tokens = T
    STATE.bytes_per_token = B / T
    # one chunk per second from disk; a token recomputes in 0.1 s: a 5-chunk prompt costs about 8 s to recompute,
    # its 4-chunk promotion 4 s to read alone
    STATE.speeds.d["disk_gbps"] = B * 1e-9
    STATE.speeds.d["disk_probe_bytes"] = float("inf")   # the test disk's own writes must not replace that speed
    STATE.speeds.d["f0"] = 0.1
    prim = object.__new__(CPUPrimaryTierOffloadingManager)
    CPUOffloadingManager.__init__(prim, num_chunks=n_chunks, cache_policy="ParkCachePolicy",
                                  cache_policy_module_path="layby.vllm.policy")
    prim.complete_read, prim.prepare_write, prim.complete_write = prim.complete_load, prim.prepare_store, prim.complete_store
    buf = mmap.mmap(-1, n_chunks * B)
    prim._kv_memoryview = memoryview(buf).cast("B", (n_chunks, B))
    par = NS(tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, is_parallelism_agnostic=True)
    spec = NS(blocks_per_chunk=1, tokens_per_hash=T, offload_prompt_only=False, tokens_per_block=[T],
              extra_config={},
              kv_events_config=NS(enable_kv_cache_events=False, self_describing_kv_events=False),
              config=NS(groups=[NS(group_id=0, tokens_per_block=T, layer_names=["l0"],
                                   sliding_window_size_in_chunks=None)],
                        model=NS(name="test/m", dtype="bf16"),
                        cache=NS(tokens_per_hash=T), parallel=par, canonical_layout=False, replicated_layout=False))
    fs = F.ParkFsTier(spec, prim.get_kv_memoryview(), "ParkFsTier", root_dir=root, n_read_threads=2, n_write_threads=2)
    mgr = TieringOffloadingManager(prim, [fs])
    spec.get_manager = lambda: mgr
    kvc = KVCacheConfig(num_blocks=128, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(
        layer_names=["l0"], kv_cache_spec=FullAttentionSpec(block_size=T, num_kv_heads=1, head_size=8, dtype=torch.bfloat16))])
    vc = NS(cache_config=NS(prefix_cache_retention_interval=None, enable_prefix_caching=True), speculative_config=None,
            parallel_config=NS(world_size=1, decode_context_parallel_size=1), num_hash_lookahead_tokens=0)
    cs = OffloadingConnectorScheduler(spec, vc, kvc)
    conn = object.__new__(C.ParkConnector)
    conn.connector_scheduler, conn.connector_worker = cs, None
    conn.__dict__["_bounding_group_ids"] = ()
    conn._waiting, conn._deferred, conn._local, conn._giveups, conn._est0 = {}, {}, {}, {}, {}
    conn._ready, conn._prefetched = {}, {}
    conn._loads, conn._finished, conn._early, conn._hints_on = {}, {}, {}, False
    # the disk holds every load until the test releases it; ParkFsTier's bookkeeping sees the submit as usual
    held = {}
    pool_enqueue = fs._pool.enqueue_load
    fs._pool.enqueue_load = lambda job_id, n, tasks: held.__setitem__(job_id, (n, list(tasks)))

    def release(job_id):
        n, tasks = held.pop(job_id)
        pool_enqueue(job_id, n, tasks)
        settle(mgr, fs)
    return conn, cs, mgr, prim, fs, prim._policy, held, release


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
              skip_reading_external_kv=False, resumable=False, num_in_flight_tokens=0, is_finished=lambda: False)


def lookup(conn, cs, mgr, req, local=0):
    out = conn.get_num_new_matched_tokens(req, local)
    mgr.on_schedule_end(ScheduleEndContext(new_req_ids=(), preempted_req_ids=()))
    return out


def blocks_for(n_tokens, first_id=10):
    n = -(-n_tokens // T)
    return NS(blocks=([NS(block_id=first_id + i, is_null=False, block_hash=None) for i in range(n)],))


def cap(n):
    """The recompute cost of n tokens as the connector prices it now."""
    L = STATE.live
    return recompute_cost(n, L.rate("pf", clock[0]), L.rate("n_adm", clock[0]), STATE.params())[1]


def read_s(n_chunks):
    return n_chunks * B / (STATE.speeds.gbps("disk") * 1e9)


def park(mgr, fs, prim, pol, root):
    """Session a (4 chunks) and session b (16 chunks) written to disk and evicted from the CPU tier."""
    HA, HB = hashes("a", 4), hashes("b", 16)
    A, Bk = [key(h) for h in HA], [key(h) for h in HB]
    store(mgr, "a-1", A)
    fs.queue_write(A)
    settle(mgr, fs); settle(mgr, fs)
    store(mgr, "b-1", Bk)
    fs.queue_write(Bk)
    settle(mgr, fs); settle(mgr, fs)
    assert all(k in fs._on_disk for k in A + Bk), "sessions a and b are not on disk"
    store(mgr, "c-1", [key(h) for h in hashes("c", 24)])
    settle(mgr, fs)
    assert not any(k in pol.chunks for k in A + Bk), "a or b still in the CPU tier"
    check(prim, pol)
    return HA, A, HB, Bk


def scenario_forward(root):
    conn, cs, mgr, prim, fs, pol, held, release = build(24, root)
    HA, A, HB, Bk = park(mgr, fs, prim, pol, root)
    # b's promotion (16 s of disk) is queued first
    rb = request("rb", HB + hashes("p", 1), 17 * T)
    conn.on_new_request(rb)
    assert lookup(conn, cs, mgr, rb)[0] is None
    assert len(fs._loading) == 1, "b's promotion was not submitted"
    # a's promotion (4 s) is queued behind it: it lands in about 20 s, recomputing a's 5 chunks costs about 8 s
    r0 = request("r0", HA + hashes("p", 1), 5 * T)
    conn.on_new_request(r0)
    rem, c = read_s(20), cap(5 * T)
    assert rem > c, (rem, c)
    # the first lookup: a's promotion is pending in the tiering manager behind b's 16 submitted chunks
    out = lookup(conn, cs, mgr, r0)
    assert out == (0, False), f"remaining {rem:.1f} s > recompute {c:.1f} s, yet r0 is still deferred: {out}"
    assert len(fs._loading) == 2 and len(held) == 2, "a's promotion was not submitted"
    assert STATE.counts["defer_giveup"] == 1
    conn.update_state_after_alloc(r0, blocks_for(0), 0)
    row = STATE.rlog[-1]
    assert row.get("giveup") is True and row["branch"] == "forward", row
    assert abs(row["rem"] - rem) < 1e-6 and row["est0"] == row["rem"] and row["cap"] < row["rem"], row
    assert row["link0"] == row["est0"], row            # no load completed yet: the drain rate is the link speed
    assert row["wait"] < 1e-6, row
    # b is still deferred: its own promotion is next in line (16 s) and recomputing 17 chunks costs more (27 s)
    assert lookup(conn, cs, mgr, rb)[0] is None
    assert STATE.counts["defer_giveup"] == 1
    check(prim, pol)
    print("  forward: a promotion behind a 16 s queue gives up at its first lookup")


def scenario_next_in_line(root):
    conn, cs, mgr, prim, fs, pol, held, release = build(24, root)
    HA, A, HB, Bk = park(mgr, fs, prim, pol, root)
    r0 = request("r0", HA + hashes("p", 1), 5 * T)
    conn.on_new_request(r0)
    assert lookup(conn, cs, mgr, r0)[0] is None
    (job_id,), = [list(fs._loading)]
    assert lookup(conn, cs, mgr, r0)[0] is None, "4 s of disk against 8 s of recompute: should wait"
    c0 = cap(5 * T)
    assert read_s(4) < c0, (read_s(4), c0)
    clock[0] += c0 + 2.0                                # past the elapsed cap
    assert read_s(4) <= cap(5 * T), "the next-in-line estimate must stay under the recompute cost"
    out = lookup(conn, cs, mgr, r0)
    assert out[0] is None, f"next in line (4 s away) gave up {c0 + 2:.1f} s into the wait: {out}"
    assert STATE.counts["defer_giveup"] == 0
    release(job_id)                                     # the promotion lands
    assert all(k in pol.chunks and pol.chunks[k].is_ready for k in A), "a's promotion did not land"
    out = lookup(conn, cs, mgr, r0)
    assert out == (4 * T, True), f"expected the loaded prefix, got {out}"
    conn.update_state_after_alloc(r0, blocks_for(4 * T), 4 * T)
    row = STATE.rlog[-1]
    assert not row.get("giveup") and row["ext"] == 4 * T and row["rec"] == T, row
    assert abs(row["est0"] - read_s(4)) < 1e-6 and abs(row["est_wait"] - (c0 + 2.0)) < 1e-3, row
    assert abs(row["wait"] - (c0 + 2.0)) < 1e-3, row
    check(prim, pol)
    print("  next in line: waited past the elapsed cap, admitted with the loaded prefix; the row holds the estimate")


def scenario_safety(root):
    conn, cs, mgr, prim, fs, pol, held, release = build(24, root)
    HA, A, HB, Bk = park(mgr, fs, prim, pol, root)
    r0 = request("r0", HA + hashes("p", 1), 5 * T)
    conn.on_new_request(r0)
    assert lookup(conn, cs, mgr, r0)[0] is None
    assert lookup(conn, cs, mgr, r0)[0] is None
    est0 = read_s(4)
    clock[0] += cap(5 * T) + 1.0                        # past the elapsed cap, not past the estimate plus the cost
    assert clock[0] - conn._deferred["r0"] <= est0 + cap(5 * T)
    out = lookup(conn, cs, mgr, r0)
    assert out[0] is None, f"gave up before the safety bound: {out}"
    clock[0] += est0 + 1.0                              # the promotion never lands: the estimate is wrong by more than the cost
    assert clock[0] - conn._deferred["r0"] > est0 + cap(5 * T)
    out = lookup(conn, cs, mgr, r0)
    assert out == (0, False), f"a stuck promotion is still deferred past its estimate plus the recompute cost: {out}"
    conn.update_state_after_alloc(r0, blocks_for(0), 0)
    row = STATE.rlog[-1]
    assert row.get("giveup") is True and row["branch"] == "safety", row
    assert abs(row["est0"] - est0) < 1e-6 and row["wait"] > row["est0"] + row["cap"], row
    assert abs(row["rem"] - est0) < 1e-6, row           # the queue still says 4 s: the estimate, not the queue, caught it
    check(prim, pol)
    print("  safety: a promotion that never lands gives up once the wait passes its estimate by the recompute cost")


def scenario_consistency(root):
    conn, cs, mgr, prim, fs, pol, held, release = build(24, root)
    HA, A, HB, Bk = park(mgr, fs, prim, pol, root)
    rb = request("rb", HB + hashes("p", 1), 17 * T)
    conn.on_new_request(rb)
    assert lookup(conn, cs, mgr, rb)[0] is None
    r0 = request("r0", HA + hashes("p", 1), 5 * T)
    conn.on_new_request(r0)
    assert lookup(conn, cs, mgr, r0) == (0, False), "r0 should give up at once behind b"
    conn.update_state_after_alloc(r0, blocks_for(0), 0)
    jb, ja = list(fs._loading)
    check(prim, pol)
    # r0 runs and finishes while both promotions are still in flight, then they land for a request that stopped waiting
    conn.request_finished(r0, [10, 11, 12, 13, 14])
    mgr.on_request_finished(cs._req_status["r0"].req_context)
    check(prim, pol)
    clock[0] += 20.0
    release(jb)
    check(prim, pol)
    out = lookup(conn, cs, mgr, rb)
    assert out == (16 * T, True), out
    conn.update_state_after_alloc(rb, blocks_for(16 * T), 16 * T)
    (job_id, job), = cs._current_batch_load_jobs.items()
    assert all(pol.chunks[k].ref_cnt == 1 for k in Bk), "the load did not pin the ready chunks"
    check(prim, pol)
    cs._current_batch_load_jobs.clear()
    conn.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=OffloadingWorkerMetadata(completed_jobs={job_id: 1})))
    assert all(pol.chunks[k].ref_cnt == 0 for k in Bk)
    check(prim, pol)
    release(ja)
    assert all(k in pol.chunks and pol.chunks[k].is_ready for k in A), "a's promotion did not land"
    check(prim, pol)
    # the give-up settles once its promotion resolved: the saving is the wait it skipped less the recompute it paid
    conn._settle_giveups(clock[0])
    assert STATE.live.acc["n_giveup"] == 1.0 and not conn._giveups, (STATE.live.acc, conn._giveups)
    # a later turn of a finds its prefix in the CPU tier with no deferral
    r1 = request("r1", HA + hashes("p", 1), 5 * T)
    conn.on_new_request(r1)
    assert lookup(conn, cs, mgr, r1) == (4 * T, True)
    conn.update_state_after_alloc(r1, blocks_for(4 * T), 4 * T)
    assert "est0" not in STATE.rlog[-1] and not STATE.rlog[-1].get("giveup"), STATE.rlog[-1]
    check(prim, pol)
    print("  consistency: promotions land after the give-up and after the request finished; counts match")


def scenario_drain(root):
    conn, cs, mgr, prim, fs, pol, held, release = build(24, root)
    HA, A, HB, Bk = park(mgr, fs, prim, pol, root)
    assert fs.drain_rate() is None, "no load completed yet"
    # a's promotion lands 40 s after its submit: the queue drained 4 chunks in 40 s, a tenth of the link speed
    r0 = request("r0", HA + hashes("p", 1), 5 * T)
    conn.on_new_request(r0)
    assert lookup(conn, cs, mgr, r0)[0] is None, "4 s of disk against 8 s of recompute: should wait"
    (ja,), = [list(fs._loading)]
    clock[0] += 10 * read_s(4)
    release(ja)
    rate = fs.drain_rate()
    assert rate is not None and abs(rate - 4 * B / (10 * read_s(4))) < 1e-6, (rate, 4 * B / (10 * read_s(4)))
    assert abs(STATE.speeds.gbps("disk") * 1e9 - B / read_s(1)) < 1e-6, "the link speed is the per-transfer speed still"
    out = lookup(conn, cs, mgr, r0)
    assert out == (4 * T, True), out
    conn.update_state_after_alloc(r0, blocks_for(4 * T), 4 * T)
    row = STATE.rlog[-1]
    assert not row.get("giveup") and abs(row["wait"] - 10 * read_s(4)) < 1e-3, row
    assert abs(row["est0"] - read_s(4)) < 1e-6 and row["link0"] == row["est0"] and abs(row["est_wait"] - row["wait"]) < 1e-3, row
    # b: the link speed says 16 s, under the 27 s recompute of its 17 chunks, the drain rate says 160 s
    rb = request("rb", HB + hashes("p", 1), 17 * T)
    conn.on_new_request(rb)
    c, rem = cap(17 * T), 16 * B / rate
    assert read_s(16) < c < rem, (read_s(16), c, rem)
    out = lookup(conn, cs, mgr, rb)
    assert out == (0, False), f"the queue drains 16 chunks in {rem:.0f} s > recompute {c:.1f} s, yet rb is deferred: {out}"
    assert STATE.counts["defer_giveup"] == 1
    conn.update_state_after_alloc(rb, blocks_for(0), 0)
    row = STATE.rlog[-1]
    assert row.get("giveup") is True and row["branch"] == "forward", row
    assert abs(row["rem"] - rem) < 1e-3 and row["est0"] == row["rem"] and abs(row["link0"] - read_s(16)) < 1e-6, row
    assert row["wait"] < 1e-6, row
    check(prim, pol)
    print("  drain: the link speed says 16 s, the measured drain rate 160 s: gives up at the first lookup")


if __name__ == "__main__":
    failed = []
    for sc in (scenario_forward, scenario_next_in_line, scenario_safety, scenario_consistency, scenario_drain):
        root = tempfile.mkdtemp(prefix="pp_dfwd_", dir=os.path.expanduser("~/.cache"))
        try:
            sc(root)
        except Exception:   # noqa: BLE001
            failed.append(sc.__name__)
            print(f"  {sc.__name__} FAILED")
            traceback.print_exc()
        finally:
            shutil.rmtree(root, ignore_errors=True)
    if failed:
        print("defer forward FAILED:", ", ".join(failed))
        sys.exit(1)
    print("defer forward ok: give up on the remaining wait at the drain rate, keep waiting when the promotion is near, "
          "safety bound, consistent tier")
