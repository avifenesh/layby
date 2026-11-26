"""A waiting request's prefix on disk is promoted to the CPU tier while it queues, so the scheduler admits it
without a deferral (ParkConnector._prefetch; vLLM 0.31 tiering manager, CPU primary with ParkCachePolicy,
ParkFsTier; CPU only).

Measured in Phase R (Gemma fast heavy, wrx, 2026-11-12 diagnostic): 84 admissions waited for a disk restore that
started only when the scheduler reached them (first deferral to admission p50 4.4 s, p90 12.7 s), after queueing
for 0.16 s at the median and 7.8 s at p90: the restore and the queue wait ran one after the other.

Before the change: the first scheduler lookup of a request whose prefix is on disk defers it (returns None) while
the promotion starts (tests/test_defer_cap.py, part 1). After it: the prefetch at the step it arrives starts the
promotion, which evicts an idle session (not the held request); once the disk serves it, the first scheduler lookup
loads the whole prefix with no deferral, and the admission counts a disk restore that cost no wait (defer 0). A
request whose held chunks would not fit in the CPU tier's room behind the ones ahead of it is not prefetched.

Run: PYTHONPATH=. python tests/test_prefetch.py
"""
import os
import shutil
import tempfile

from scaffold_vllm import T, blocks_for, build, check, end_schedule, hashes, key, lookup, request, settle, store
from layby.vllm.state import STATE

root = tempfile.mkdtemp(prefix="pp_pref_", dir=os.path.expanduser("~/.cache"))
try:
    conn, cs, mgr, prim, fs, pol = build(8, root)

    # session a: 4 chunks written to disk, then evicted from the CPU tier by session c
    A = [key(h) for h in hashes("a", 4)]
    store(mgr, "a-1", A)
    fs.queue_write(A)
    settle(mgr, fs); settle(mgr, fs)
    assert all(k in fs._on_disk for k in A), "session a is not on disk"
    C8 = [key(h) for h in hashes("c", 8)]
    store(mgr, "c-1", C8)
    settle(mgr, fs)
    assert not any(k in pol.chunks for k in A), "session a still in the CPU tier"

    # a's next turn arrives: a's prefix plus one new chunk
    r0 = request("r0", hashes("a", 4) + hashes("p", 1), 5 * T)
    conn.on_new_request(r0)
    assert "r0" in pol.held
    conn._prefetch()
    assert conn._prefetched.get("r0") == 4 and STATE.counts["prefetch"] == 4, (conn._prefetched, STATE.counts)
    end_schedule(mgr)                            # the step ends: the tiering manager submits the promotion
    settle(mgr, fs)
    assert all(k in pol.chunks and pol.chunks[k].ref_cnt == 0 for k in A), "the prefetch did not land"
    assert all(pol.hinted[k] < 0 for k in A), "prefetched chunks are not held for r0"
    assert sum(k in pol.chunks for k in C8) == 4, "the promotion should evict 4 of c's idle chunks"
    check(prim, pol)

    # the scheduler reaches r0: the whole prefix loads now, no deferral
    out = lookup(conn, mgr, r0)
    assert out == (4 * T, True), f"expected the 64-token prefix ready with no deferral, got {out}"
    assert "r0" not in conn._deferred
    n0, d0 = STATE.live.acc["n_defer"], STATE.live.acc["defer"]
    conn.update_state_after_alloc(r0, blocks_for(5 * T), 4 * T)
    assert abs(STATE.live.acc["n_defer"] - (n0 + 1)) < 1e-3 and STATE.live.acc["defer"] == d0, \
        "a disk restore the prefetch hid counts as defer 0"
    assert "r0" not in pol.held and "r0" not in conn._prefetched
    assert "defer" not in STATE.rlog[-1], STATE.rlog[-1]

    fs.shutdown()

    # the GPU prefix cache still holds the first 2 of a's 4 chunks: vLLM loads only past them, so the prefetch
    # promotes chunks 2 and 3 only (before: all 4, measured on wrx 2026-11-13: 57,688 promotions, about 63 per
    # waiting request, most of them for prefixes the GPU still held)
    conn, cs, mgr, prim, fs, pol = build(8, root + "/g")
    A = [key(h) for h in hashes("a", 4)]
    store(mgr, "a-1", A)
    fs.queue_write(A)
    settle(mgr, fs); settle(mgr, fs)
    store(mgr, "c-1", [key(h) for h in hashes("c", 8)])
    settle(mgr, fs)
    assert not any(k in pol.chunks for k in A)
    HA = hashes("a", 4)
    from types import SimpleNamespace as NS
    conn._pool = NS(get_cached_block=lambda h, gids: [object()] if h in HA[:2] else None)
    r3 = request("r3", HA + hashes("p", 1), 5 * T)
    conn.on_new_request(r3)
    conn._prefetch()
    assert conn._prefetched.get("r3") == 2, conn._prefetched
    end_schedule(mgr)
    settle(mgr, fs)
    assert [k in pol.chunks for k in A] == [False, False, True, True], [k in pol.chunks for k in A]
    check(prim, pol)
    fs.shutdown()

    # room: in an 8-chunk tier, a waiting request of 4 chunks and one of 5 behind it; the second is not prefetched
    conn, cs, mgr, prim, fs, pol = build(8, root)
    r1 = request("r1", hashes("x", 4), 4 * T)
    r2 = request("r2", hashes("y", 5), 5 * T)
    conn.on_new_request(r1)
    conn.on_new_request(r2)
    conn._prefetch()
    assert "r1" in conn._prefetched and "r2" not in conn._prefetched, conn._prefetched
    check(prim, pol)
    # a hybrid model: the prefetch promotes the full-attention prefix up to the first chunk no tier has, and of the
    # Mamba states (window 1) only the one where that prefix ends
    from types import SimpleNamespace as NS
    from vllm.v1.kv_offload.base import LookupResult as LR
    F6, M6 = [("f", i) for i in range(6)], [("m", i) for i in range(6)]
    disk = set(F6[:4] + M6)
    seen = []
    mgr2 = NS(lookup=lambda k, c: (seen.append(k), LR.HIT_PENDING if k in disk else LR.MISS)[1])
    prim2 = NS(lookup=lambda k, c: LR.MISS)
    FA, MB = NS(sliding_window_size_in_chunks=None, tokens_per_chunk=T), NS(sliding_window_size_in_chunks=1, tokens_per_chunk=T)
    conn.connector_scheduler = NS(config=NS(kv_group_configs=[FA, MB]))
    rs2 = NS(group_states=[NS(offload_keys=F6), NS(offload_keys=M6)], req_context=None)
    n = conn._prefetch_req(rs2, mgr2, prim2)
    assert n == 5 and seen == F6[:5] + [("m", 3)], (n, seen)
finally:
    fs.shutdown() if "fs" in dir() else None
    shutil.rmtree(root, ignore_errors=True)
print("prefetch ok: a waiting request's disk prefix lands during the queue wait and admits with no deferral")
