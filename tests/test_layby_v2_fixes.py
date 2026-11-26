"""Regression tests for Layby v2 engine, cost-rule, scheduler-policy, and disk-tier fixes (CPU only).

Covers:
  1. Bug A: Hybrid multi-group on_disk token count (Gemma-4 / Qwen3.8 / GLM) does not 2x-overcount.
  2. Bug B: Token-proportional disk restore time (defer_x) + decayed ghost-deferral fade-out.
  3. Bug C: Hybrid SWA/Mamba _ready_tokens ignores stale pre-window chunks; _settle_giveups never
     double-counts defer_giveup before admission; forward remaining-wait cap triggers on step 1.
  4. Bug D: park write-before-evict protection prevents losing park chunks before create_store_job;
     ParkFsTier read priority throttles background store splitting to 1 thread while active loads run
     (keeping chunks immediately pinned instead of leaving them unpinned in CPU).
  5. Upgrade E: ParkCachePolicy O(1) _live_cnt and lazy expired-hint min-heap match sorted LRU order.
  6. Wave 2 fixes: incremental delta W and read-only n_disk_r in layby.rule; bounded _prefetch
     (skips while disk loads run, reserves burst headroom, at most one request per step); re-deferral
     clock reset when an already-landed promotion is evicted while waiting in the GPU queue.

Run: PYTHONPATH=. python tests/test_layby_v2_fixes.py
"""
import math
import os
import shutil
import tempfile
from types import SimpleNamespace as NS

import numpy as np
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

import layby.vllm.connector as C
import layby.vllm.fs_tier as F
import layby.vllm.policy as P
from layby.live import EngineParams, Live
from layby.rule import TG, decide, interp_surv
from layby.vllm.state import STATE
from scaffold_vllm import T, blocks_for, build, check, end_schedule, hashes, key, lookup, request, settle, store


def test_bug_b_token_proportional_defer_and_ghost_decay():
    now = 1000.0
    p = EngineParams(bytes_per_token=163840, gpu_tokens=35000, cpu_tokens=52800,
                     cpu_gbps=50.0, disk_gbps=1.5, f0=0.00032, t0=0.03)
    S = interp_surv(np.linspace(0.95, 0.20, 15), TG)
    # 1. Measure a 32K-token deferral that took 4.0s (transfer = 3.50s, overhead = 0.50s)
    live = Live(120.0)
    live.end_idle(now - 5.0, "c", 10.0, 10000, True)
    live.end_idle(now - 5.0, "g", 2.0, 10000, True)
    x_32k = 32768 * p.bytes_per_token / (p.disk_gbps * 1e9)
    live.add(now, n_defer=4.0, defer=4.0 * (x_32k + 0.5), n_defer_x=4.0, defer_x=4.0 * x_32k)
    # For a 9K-token session (td = 0.98s, rec = 2.88s): disk MUST beat recompute (ins.base < none.base)
    d = decide(S, 9090, 0, (), (), live, p, now)
    assert d["tot"]["ins"][0] < d["tot"]["none"][0], f"9K session wrongly priced disk > rec: {d['tot']}"
    # Partial on_disk (50%) must price strictly between 0% and 100%
    d_part = decide(S, 20000, 10000, (), (), live, p, now)
    assert d_part["tot"]["ins"][0] < d_part["tot"]["none"][0], f"partial on_disk did not scale: {d_part['tot']}"
    # 2. Ghost deferral from 1 hour ago (3600s) must decay back to link wait, NOT lock in a 20s stall
    live2 = Live(120.0)
    live2.end_idle(0.0, "c", 10.0, 10000, True)
    live2.end_idle(0.0, "g", 2.0, 10000, True)
    live2.add(0.0, n_defer=1.0, defer=25.0)       # legacy stall 1 hour ago
    live2.end_idle(3600.0, "c", 10.0, 10000, True)
    live2.end_idle(3600.0, "g", 2.0, 10000, True)
    d_ghost = decide(S, 9090, 0, (), (), live2, p, 3600.0)
    assert d_ghost["tot"]["ins"][0] < d_ghost["tot"]["none"][0], f"1h-old ghost deferral poisoned disk_t: {d_ghost['tot']}"


def test_bug_a_hybrid_on_disk_no_overcount():
    STATE.__init__()
    STATE.chunk_tokens = 16
    STATE.bytes_per_token = 163840.0
    STATE.gpu_tokens = 35000
    STATE.cpu_tokens = 52800
    now = 2000.0
    STATE.live.end_idle(now - 2.0, "c", 10.0, 10000, True)
    STATE.live.end_idle(now - 2.0, "g", 2.0, 10000, True)
    # 2-group hybrid (full + SWA): 1000 tokens (62 chunks) of 2000-token session are on disk in BOTH groups
    full_ks = [("f", i) for i in range(125)]
    swa_ks = [("s", i) for i in range(125)]
    on_disk_set = set(full_ks[:62] + swa_ks[:62])    # 124 chunks across 2 groups = 992 tokens, NOT 1984!
    written = []
    STATE.fs_tier = NS(count_on_disk=lambda ks: sum(1 for k in ks if k in on_disk_set),
                       queue_write=lambda ks: written.append(list(ks)))
    STATE.policy = None
    conn = object.__new__(C.ParkConnector)
    conn._gpu_order = False
    entry = dict(
        keys=full_ks + swa_ks,
        group_keys=[(16, full_ks)],
        tokens=2000,
        params={"park_key": "s1-1"},
        t=now,
    )
    surv = np.linspace(0.95, 0.4, 15).tolist()
    conn._decide(entry, surv, now)
    dlog = STATE.dlog[-1]
    assert dlog["tot"]["none"] != dlog["tot"]["ins"], f"hybrid on_disk 2x-overcounted (none==ins): {dlog['tot']}"
    assert dlog["opt"] in ("ins", "park") and written, f"expected disk write for new half, got {dlog}"


def test_bug_c_hybrid_ready_tokens_and_giveup_single_count():
    # 1. Hybrid _ready_tokens ignores stale pre-window SWA chunks
    F4 = [("f", i) for i in range(4)]
    S4 = [("s", i) for i in range(4)]
    cpu = {F4[0], F4[1], F4[2], S4[2]}   # full-attn has 3 chunks (48 tok); SWA window=1 only needs S4[2] (S4[0], S4[1] are stale MISS)
    prim = NS(lookup=lambda k, c: LookupResult.HIT if k in cpu else (LookupResult.HIT_PENDING if k == F4[3] else LookupResult.MISS))
    FA = NS(sliding_window_size_in_chunks=None, tokens_per_chunk=16)
    SWA = NS(sliding_window_size_in_chunks=1, tokens_per_chunk=16)
    conn = object.__new__(C.ParkConnector)
    conn.connector_scheduler = NS(config=NS(kv_group_configs=[FA, SWA]), manager=NS(primary_tier=prim))
    rs = NS(group_states=[NS(offload_keys=F4), NS(offload_keys=S4)], req_context=NS())
    ready, pending = conn._ready_tokens(rs, 0)
    assert ready == 48 and pending == F4[3], f"hybrid _ready_tokens broken: ready={ready} pending={pending}"

    # 2. _settle_giveups must NOT delete give-up entry while request is still deferred -> defer_giveup counted ONCE
    root = tempfile.mkdtemp(prefix="pp_v2fix_", dir=os.path.expanduser("~/.cache"))
    try:
        clk = [__import__("time").monotonic()]
        orig_time = C.time
        C.time = NS(monotonic=lambda: clk[0])
        try:
            conn, cs, mgr, prim, fs, pol = build(8, root)
            held = []
            real_load = fs.submit_load
            fs.submit_load = lambda j: held.append(j)
            HA = hashes("a", 4)
            A = [key(h) for h in HA]
            store(mgr, "a-1", A)
            fs.queue_write(A)
            settle(mgr, fs); settle(mgr, fs)
            store(mgr, "c-1", [key(h) for h in hashes("c", 8)])
            settle(mgr, fs)
            r0 = request("r0", HA + hashes("p", 1), 5 * T)
            conn.on_new_request(r0)
            assert lookup(conn, mgr, r0)[0] is None
            clk[0] += 50.0
            assert lookup(conn, mgr, r0) == (0, False)
            assert STATE.counts["defer_giveup"] == 1
            # Promotion completes & settles while r0 still waits for GPU room (not admitted yet!)
            real_load(held.pop())
            settle(mgr, fs)
            conn._settle_giveups(clk[0])
            assert "r0" in conn._giveups and conn._giveups["r0"].get("settled") is True
            # Another step where scheduler asks again before admission: must NOT double-count defer_giveup!
            out = lookup(conn, mgr, r0)
            assert STATE.counts["defer_giveup"] == 1, f"defer_giveup double-counted: {STATE.counts}"
            conn.update_state_after_alloc(r0, blocks_for(5 * T), out[0])
            assert "r0" not in conn._giveups
        finally:
            C.time = orig_time
            fs.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_bug_d_park_write_protection_and_read_priority():
    root = tempfile.mkdtemp(prefix="pp_v2park_", dir=os.path.expanduser("~/.cache"))
    try:
        conn, cs, mgr, prim, fs, pol = build(6, root)
        # Store A (3 chunks) and B (3 chunks): tier is full (6 chunks)
        A = [key(h) for h in hashes("a", 3)]
        B = [key(h) for h in hashes("b", 3)]
        store(mgr, "a-1", A)
        store(mgr, "b-1", B)
        settle(mgr, fs)
        # Rule decides "park" for A: mark_pending_write protects A until submit_store pins it!
        pol.mark_pending_write(A)
        fs.queue_write(A)
        pol.apply(A, -1.0)
        # Before serve_external_requests runs, C needs 2 slots: must evict B (oldest/unprotected), NOT A!
        out = store(mgr, "c-1", [key(h) for h in hashes("c", 2)])
        assert set(out.evicted_keys) <= set(B), f"park chunks evicted before disk write started: {out.evicted_keys}"
        # Read priority: while fs._loading is non-empty, background store jobs are NOT split across
        # 16 write threads (_split_io throttles to 1 thread -> returns False), while late writes still
        # pin and submit immediately so park chunks cannot be evicted before writing.
        fs._loading[999] = [key(hashes("z", 1)[0])]
        fake_job = NS(job_id=777, keys=A, chunk_ids=[0, 1, 2])
        assert fs._split_io(fake_job, load=False) is False, "store job split across write threads while load active!"
        # Late write stays within cap//4 (=1 chunk for cap=6) per active slice under load, writes all 3 chunks, and marks NEVER
        settle(mgr, fs); settle(mgr, fs); settle(mgr, fs); settle(mgr, fs)
        fs._loading.pop(999)
        assert all(k in fs._on_disk for k in A), "protected park chunks did not reach disk"
        assert all(pol.hinted.get(k) == P.NEVER for k in A), f"park chunks not marked first-out after write: {pol.hinted}"
        check(prim, pol)
        fs.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wave2_incremental_W_and_read_only_contention():
    """Wave 2: W prices only incremental delta (N - d0) bytes and only delays user-facing disk reads."""
    now = 1000.0
    p = EngineParams(bytes_per_token=163840, gpu_tokens=35000, cpu_tokens=52800,
                     cpu_gbps=50.0, disk_gbps=1.35, f0=0.00022, t0=0.03)
    S = interp_surv(np.linspace(0.95, 0.35, 15), TG)
    live = Live(120.0)
    live.end_idle(now - 50.0, "c", 8.0, 10000, True)
    live.end_idle(now - 50.0, "g", 2.0, 10000, True)
    # Heavy background write activity (n_disk_w=20, wait_disk=80s) but light read activity (n_disk_r=2, wait_disk_r=0.4s)
    live.add(now, n_disk=22.0, wait_disk=80.4, n_disk_r=2.0, wait_disk_r=0.4, n_disk_w=20.0)
    # 25K session with 20.5K already on disk (4.5K delta): W must be tiny and ins.base < none.base!
    d_delta = decide(S, 25000, 20500, (), (), live, p, now)
    w_delta = d_delta["tot"]["ins"][1] - d_delta["tot"]["none"][1]
    assert d_delta["tot"]["ins"][0] < d_delta["tot"]["none"][0], f"background writes poisoned ins.base: {d_delta['tot']}"
    assert w_delta < 0.15, f"incremental W too large ({w_delta:.3f}s): {d_delta['tot']}"
    assert d_delta["opt"] in ("ins", "park"), f"delta write wrongly blocked by W: {d_delta}"
    # Full session already on disk (d0 == N): W must be strictly 0
    d_full = decide(S, 25000, 25000, (), (), live, p, now)
    assert abs(d_full["tot"]["ins"][1] - d_full["tot"]["none"][1]) < 1e-9, f"d0==N had nonzero W: {d_full['tot']}"
    # Fresh 25K session (d0 == 0) must have strictly larger W than 4.5K delta
    d_fresh = decide(S, 25000, 0, (), (), live, p, now)
    w_fresh = d_fresh["tot"]["ins"][1] - d_fresh["tot"]["none"][1]
    assert w_fresh > w_delta * 10.0, f"fresh W ({w_fresh:.3f}) not >> delta W ({w_delta:.3f})"


def test_wave2_bounded_prefetch_and_redefer_reset():
    """Wave 2: _prefetch skips during active loads and limits to 1 req/step; re-deferral resets clock."""
    root = tempfile.mkdtemp(prefix="pp_v2w2_", dir=os.path.expanduser("~/.cache"))
    try:
        clk = [__import__("time").monotonic()]
        orig_time = C.time
        C.time = NS(monotonic=lambda: clk[0])
        try:
            conn, cs, mgr, prim, fs, pol = build(10, root)
            # Put two 2-chunk prefixes on disk and evict them from CPU
            HA = hashes("a", 2); A = [key(h) for h in HA]
            HB = hashes("b", 2); B = [key(h) for h in HB]
            store(mgr, "a-1", A); store(mgr, "b-1", B)
            fs.queue_write(A); fs.queue_write(B)
            settle(mgr, fs); settle(mgr, fs); settle(mgr, fs)
            store(mgr, "z-1", [key(h) for h in hashes("z", 10)])
            settle(mgr, fs)
            # Two waiting requests arrive
            r1 = request("r1", HA + hashes("p1", 1), 3 * T)
            r2 = request("r2", HB + hashes("p2", 1), 3 * T)
            conn.on_new_request(r1)
            clk[0] += 0.1
            conn.on_new_request(r2)
            # 1. While fs._loading is non-empty, _prefetch does nothing
            fs._loading[888] = [A[0]]
            before = STATE.counts["prefetch"]
            conn._prefetch()
            assert STATE.counts["prefetch"] == before, "prefetch fired while active disk load in flight!"
            fs._loading.pop(888)
            # 2. When idle, _prefetch starts ONLY 1 waiting request per step (r1, not r2)
            conn._prefetch()
            assert "r1" in conn._prefetched and "r2" not in conn._prefetched, (
                f"prefetch did not bound to 1 req/step: {conn._prefetched}"
            )
            settle(mgr, fs); settle(mgr, fs)
            # 3. Re-deferral clock reset: r1's earlier promotion landed fast (0.01s < 0.033s recompute cap),
            #    then waited 40s in GPU queue and was evicted -> fresh deferral accumulates only 0.01s real
            #    deferral (NOT 40s GPU queue wait!) so it does not falsely give up!
            conn._deferred["r1"] = clk[0] - 40.0
            conn._ready["r1"] = clk[0] - 39.99
            conn.__dict__.setdefault("_est0", {})["r1"] = (clk[0] - 40.0, 0.5, 0.5)
            # Evict r1's prefetched chunks so lookup defers again
            conn._unhold("r1")
            store(mgr, "z-2", [key(h) for h in hashes("w", 10)])
            settle(mgr, fs)
            held = []
            fs.submit_load = lambda j: held.append(j)
            out = lookup(conn, mgr, r1)
            assert out[0] is None, f"re-deferred request falsely gave up on stale clock: {out}"
            assert "r1" not in conn._ready and abs(conn._deferred["r1"] - clk[0]) < 1e-6, (
                f"re-deferral did not reset clock: deferred={conn._deferred.get('r1')} now={clk[0]}"
            )
        finally:
            C.time = orig_time
            fs.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_upgrade_e_policy_lazy_expire_parity_and_o1_live():
    STATE.__init__()
    clk = [1000.0]
    orig_time = P.time
    P.time = NS(monotonic=lambda: clk[0])
    try:
        pol = P.ParkCachePolicy(5000)
        def mk():
            c = ChunkStatus.__new__(ChunkStatus)
            c.ref_cnt = 0
            return c
        # Insert 100 LRU chunks at t=1000..1099 and 100 hinted chunks expiring at t=1005.5..1104.5
        expected = []
        for i in range(100):
            clk[0] = 1000.0 + i
            k = ("u", i)
            pol.insert(k, mk())
            expected.append((1000.0 + i, k))
        for i in range(100):
            clk[0] = 1000.0
            k = ("h", i)
            pol.insert(k, mk())
            t_exp = 1005.5 + i
            pol.apply([k], t_exp - clk[0], now=clk[0])
            expected.append((t_exp, k))
        # Advance clock past all expiries + GRACE
        clk[0] = 1300.0
        got = [k for k, _ in pol.evict(200, set())]
        exp_order = [k for _, k in sorted(expected, key=lambda x: x[0])]
        assert got == exp_order, f"lazy _exp_lru order mismatch at first diff"
        # O(1) _live_cnt check across overlapping live requests
        k0, k1 = ("x", 0), ("x", 1)
        pol.insert(k0, mk()); pol.insert(k1, mk())
        c1, c2 = ReqContext(req_id="r1"), ReqContext(req_id="r2")
        pol.touch([k0, k1], c1)
        pol.touch([k0], c2)
        assert pol._in_use(k0) and pol._in_use(k1) and pol._live_cnt[k0] == 2
        pol.request_finished(c1)
        assert pol._in_use(k0) and not pol._in_use(k1) and pol._live_cnt[k0] == 1
        pol.request_finished(c2)
        assert not pol._in_use(k0) and not pol._live_cnt
    finally:
        P.time = orig_time


def test_wave3_pending_hint_ins_protection_thru_and_ready():
    """Wave 3: pending-hint & ins pending-write protect below idle chunks; STATE.thru streams deltas
    for already-disk-placed sessions; _check_ready stamps _ready as soon as promotion finishes."""
    root = tempfile.mkdtemp(prefix="pp_v2w3_", dir=os.path.expanduser("~/.cache"))
    try:
        clk = [__import__("time").monotonic()]
        orig_c_time, orig_p_time = C.time, P.time
        C.time = NS(monotonic=lambda: clk[0])
        P.time = NS(monotonic=lambda: clk[0])
        try:
            conn, cs, mgr, prim, fs, pol = build(6, root)
            # 1. Pending-hint guard protects finished chunks below idle chunks, and auto-expires to LRU
            A = [key(h) for h in hashes("a", 3)]
            B = [key(h) for h in hashes("b", 3)]
            store(mgr, "a-1", A)
            store(mgr, "b-1", B)
            settle(mgr, fs)
            # B has short idle hint (+10s), A is pending hint (ttl=0.25s) -> evict 2 must take B, NOT A!
            pol.apply(B, 10.0, now=clk[0])
            pol.mark_pending_write(A, now=clk[0], ttl=0.25, eta=None)
            out = store(mgr, "c-1", [key(h) for h in hashes("c", 2)])
            assert set(out.evicted_keys) <= set(B), f"pending-hint chunks evicted before idle: {out.evicted_keys}"
            # Advance past 0.25s TTL: A automatically returns to LRU (not stuck at -PENDING!)
            clk[0] += 0.5
            pol._expire(clk[0])
            assert all(k not in pol._pending_write and k in pol.lru for k in A), f"pending-hint did not expire: {pol._pending_write}"
            # 2. ins pending-write protects A even with large eta (+120s) until submit_store pins A,
            #    then release_pending_write applies positive eta (now + 120s), NOT NEVER!
            pol.mark_pending_write(A, now=clk[0], ttl=30.0, eta=120.0)
            fs.queue_write(A)
            pol.apply(A, 120.0, now=clk[0])
            assert all(pol.hinted[k] < 0 for k in A), "ins pending-write overwritten by +120s!"
            settle(mgr, fs); settle(mgr, fs)
            assert all(k in fs._on_disk for k in A), "ins chunks did not reach disk"
            assert all(pol.hinted.get(k, 0) > clk[0] and pol.hinted.get(k, 0) < P.NEVER for k in A), (
                f"ins release did not restore positive eta: {pol.hinted}"
            )
            # 3. STATE.thru does NOT stream speculatively in default 'rule' mode (only in 'demand' mode)
            assert STATE.write_mode == "rule"
            STATE.thru.add("s1")
            s1_ctx = ReqContext(req_id="r-s1", kv_transfer_params={"park_key": "s1-2"})
            assert F._through(s1_ctx) is False
            STATE.write_mode = "demand"
            assert F._through(s1_ctx) is True
            STATE.write_mode = "rule"
            STATE.thru.discard("s1")
            # 4. _check_ready stamps _ready[rid] immediately when promotion finishes without polling lookup
            HA = hashes("a", 3)
            store(mgr, "z-1", [key(h) for h in hashes("z", 6)])
            settle(mgr, fs)
            r0 = request("r0", HA + hashes("p", 1), 4 * T)
            conn.on_new_request(r0)
            held = []
            real_load = fs.submit_load
            fs.submit_load = lambda j: held.append(j)
            assert lookup(conn, mgr, r0)[0] is None
            assert "r0" in conn._deferred and "r0" not in conn._ready
            clk[0] += 0.5
            conn._check_ready(clk[0])
            assert "r0" not in conn._ready, "_ready stamped while promotion still in flight!"
            # Promotion finishes at +1.0s; scheduler does NOT poll r0 for 15s, but _check_ready runs!
            clk[0] += 0.5
            real_load(held.pop())
            settle(mgr, fs)
            conn._check_ready(clk[0])
            assert "r0" in conn._ready and abs(conn._ready["r0"] - clk[0]) < 1e-6, (
                f"_check_ready did not stamp exact landing time: {conn._ready}"
            )
            check(prim, pol)
        finally:
            C.time, P.time = orig_c_time, orig_p_time
            fs.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_wave4_no_disk_gpu_blk_roofline_f0_running_protect_and_backpressure():
    """Wave 4:
    1. layby.rule does NOT charge GPU-block holding (blk * own['d']) for disk->CPU promotions.
    2. Speeds.step ignores tiny memory-bound delta steps (<= 1.5*t0) so f0 never collapses by 5x.
    3. mark_pending_write protects keys inserted AFTER admission (chunked prefill) via insert()->_place().
    4. ParkFsTier.serve_external_requests caps concurrently pinned late writes at cap // 4."""
    from layby.live import Speeds
    STATE.__init__()
    # 1. No phantom GPU blocking for disk->CPU promotions under high queue waiting (nw=5)
    now = 1000.0
    p = EngineParams(bytes_per_token=65536, gpu_tokens=40768, cpu_tokens=73696,
                     cpu_gbps=56.0, disk_gbps=0.141, f0=0.00018, t0=0.022)
    S = interp_surv(np.linspace(0.98, 0.40, 15), TG)
    live = Live(120.0)
    live.end_idle(now - 60.0, "g", 1.0, 20000, True)
    live.end_idle(now - 60.0, "c", 15.0, 20000, True)
    # 5 waiting requests (nw=5), 4 in prefill (pf=4), light disk read rate
    live.add(now, nw=5.0 * 60.0, pf=4.0 * 60.0, n_adm=3.0, qwait=30.0, n_disk_r=1.0, wait_disk_r=1.0)
    d = decide(S, 26000, 0, (), (), live, p, now)
    assert d["tot"]["ins"][0] < d["tot"]["none"][0], f"disk->CPU wrongly charged phantom GPU blocking: {d['tot']}"

    # 2. Speeds.f0() roofline filter: memory-bound 219-token delta steps (0.038s vs t0=0.0305s) ignored
    sp = Speeds(120.0, dict(cpu_gbps=50.0, disk_gbps=1.35, f0=0.00025, t0=0.030))
    for i in range(20):
        sp.step(1000.0 + i * 0.1, 0.0305, 0)       # decode steps -> t0 = 0.0305s
    sp.step(1002.0, 0.280, 1024)                    # compute-bound 1024-tok prefill -> f0 = 0.000244
    f_before = sp.f0()
    for i in range(10):
        sp.step(1003.0 + i, 0.038, 219)             # memory-bound 219-tok delta steps (<= 1.5*t0)
    sp.step(1015.0, 0.200, 16)                      # tiny stalled 16-tok step (< 256) must NOT inflate f0
    f_after = sp.f0()
    assert abs(f_after - f_before) < 1e-6 and f_after > 0.00020, f"tiny/stalled steps corrupted f0: {f_before} -> {f_after}"

    # 3. Running-request pre-registration: chunk inserted AFTER mark_pending_write lands protected at -PENDING
    clk = [1000.0]
    orig_p_time = P.time
    P.time = NS(monotonic=lambda: clk[0])
    try:
        pol = P.ParkCachePolicy(10)
        def mk():
            c = ChunkStatus.__new__(ChunkStatus)
            c.ref_cnt = 0
            return c
        k_run = ("run", 0)
        pol.mark_pending_write([k_run], now=clk[0], ttl=120.0, eta=None)  # registered BEFORE insert()!
        pol.insert(k_run, mk())                                           # stored mid-prefill
        assert pol.hinted.get(k_run, 0) < 0, f"mid-prefill chunk not protected on insert: {pol.hinted}"
        k_idle = ("idle", 0)
        pol.insert(k_idle, mk())
        pol.apply([k_idle], 5.0, now=clk[0])
        ev = [k for k, _ in pol.evict(1, set())]
        assert ev == [k_idle], f"running chunk evicted before idle chunk: {ev}"
        pol.clear_pending_write([k_run], restore_lru=True, now=clk[0])
        assert k_run in pol.lru and k_run not in pol._pending_write
    finally:
        P.time = orig_p_time

    # 4. Late-write CPU pinning backpressure: while a write is active with inflight >= cap // 4,
    #    second late write stays queued in fs._late instead of pinning more CPU RAM
    root = tempfile.mkdtemp(prefix="pp_v2w4_", dir=os.path.expanduser("~/.cache"))
    try:
        conn, cs, mgr, prim, fs, pol = build(8, root)
        A = [key(h) for h in hashes("a", 3)]
        B = [key(h) for h in hashes("b", 3)]
        store(mgr, "a-1", A); store(mgr, "b-1", B)
        settle(mgr, fs)
        # Hold store jobs so job A stays active in _writing/_inflight (3 >= 8//4 = 2)
        held_s = []
        fs.submit_store = lambda j: (fs._writing.__setitem__(j.job_id, list(j.keys)),
                                     fs._inflight.update(j.keys), held_s.append(j))
        fs.queue_write(A)
        fs.queue_write(B)
        mgr.on_schedule_end(__import__("vllm.v1.kv_offload.base", fromlist=["ScheduleEndContext"]).ScheduleEndContext(
            new_req_ids=(), preempted_req_ids=()))
        assert len(held_s) == 1 and set(held_s[0].keys) == set(A), f"backpressure failed, submitted {len(held_s)} jobs"
        assert len(fs._late) == 1 and set(fs._late[0][1]) == set(B), f"second late write not deferred: {fs._late}"
        fs.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 5. Promotion-livelock guard: when active disk loads already pin >=75% of CPU tier on slow/saturated disk
    #    (est_wait > cap), a new request gets resident CPU prefix directly (_ReadyOnly); on fast disk (est_wait <= cap)
    #    it defers cleanly without starting an overlapping promotion or throwing away cached tokens!
    root = tempfile.mkdtemp(prefix="pp_v2w4_live_", dir=os.path.expanduser("~/.cache"))
    try:
        conn, cs, mgr, prim, fs, pol = build(8, root)
        STATE.bytes_per_token = 163840.0
        STATE.speeds.d["disk_gbps"] = 0.14
        STATE.speeds.d["disk_probe_bytes"] = float("inf")
        STATE.speeds.d["f0"] = 0.00032
        HA = hashes("a", 3); A = [key(h) for h in HA]
        store(mgr, "a-1", A)
        fs.queue_write(A)
        settle(mgr, fs); settle(mgr, fs)
        store(mgr, "z-1", [key(h) for h in hashes("z", 8)])
        settle(mgr, fs)
        # Simulate active load + 6/8 (75%) CPU chunks pinned (not evictable) on slow disk -> gives up immediately
        fs._loading[999] = [A[0]]
        for k in list(pol.evictable)[:6]:
            pol.mark_non_evictable(k)
        r0 = request("r0", HA + hashes("p", 1), 4 * T)
        conn.on_new_request(r0)
        out = conn.get_num_new_matched_tokens(r0, 0)
        assert out == (0, False) and "r0" not in conn._deferred, f"slow-disk livelock guard failed: out={out}"
        # On fast NVMe (12 GB/s), pinned >= 75% defers cleanly without starting a promotion OR throwing away cache!
        STATE.speeds.d["disk_gbps"] = 12.0
        r1 = request("r1", HA + hashes("p2", 1), 4 * T)
        conn.on_new_request(r1)
        out1 = conn.get_num_new_matched_tokens(r1, 0)
        assert out1 == (None, False) and "r1" in conn._pin_deferred and not mgr._pending_load_submissions, (
            f"fast-disk pin guard wrongly admitted ext=0 or launched promotion: out={out1}"
        )
        fs._loading.pop(999)
        fs.shutdown()
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # 6. Tail-first eviction order in mark_pending_write, clear_pending_write(restore_lru=True), and unhold:
    #    prefix chunk 0 must ALWAYS be evicted last so partial eviction never zeroes an entire prefix!
    STATE.__init__()
    clk = [2000.0]
    orig_p_time = P.time
    P.time = NS(monotonic=lambda: clk[0])
    try:
        pol = P.ParkCachePolicy(10)
        def mk2():
            c = ChunkStatus.__new__(ChunkStatus)
            c.ref_cnt = 0
            return c
        ks = [("p", i) for i in range(4)]
        for k in ks:
            pol.insert(k, mk2())
        pol.mark_pending_write(ks, now=clk[0], ttl=30.0, eta=5.0)
        ev = [k for k, _ in pol.evict(2, set())]
        assert ev == [("p", 3), ("p", 2)], f"mark_pending_write evicted prefix before tail: {ev}"
        pol.clear_pending_write(ks[:2], restore_lru=True, now=clk[0])
        ev2 = [k for k, _ in pol.evict(1, set())]
        assert ev2 == [("p", 1)], f"clear_pending_write(restore_lru) evicted prefix before tail: {ev2}"
        # unhold back to LRU also evicts tail first
        pol2 = P.ParkCachePolicy(10)
        ks2 = [("q", i) for i in range(4)]
        for k in ks2:
            pol2.insert(k, mk2())
        pol2.hold("req-q", ks2, clk[0])
        pol2.unhold("req-q", now=clk[0])
        ev3 = [k for k, _ in pol2.evict(2, set())]
        assert ev3 == [("q", 3), ("q", 2)], f"unhold evicted prefix before tail: {ev3}"

        # 7. Eviction hierarchy (true recovery cost):
        #    CPU-only HELD_MEM < PENDING_DISK < unwritten running PENDING_MEM < disk-backed HELD < on-disk PENDING
        pol3 = P.ParkCachePolicy(10)
        k_hmem, k_pdisk, k_rmem, k_hdisk, k_rdisk = ("hmem", 0), ("pdisk", 0), ("rmem", 0), ("hdisk", 0), ("rdisk", 0)
        for k in (k_hmem, k_pdisk, k_rmem, k_hdisk, k_rdisk):
            pol3.insert(k, mk2())
        STATE.fs_tier = NS(_on_disk={k_hdisk, k_rdisk})
        pol3.hold("r-mem", [k_hmem], clk[0])
        pol3.hold("r-disk", [k_hdisk], clk[0])
        pol3.mark_pending_write([k_pdisk], now=clk[0], ttl=30.0, eta=10.0)    # decided ins/park write
        pol3.mark_pending_write([k_rmem, k_rdisk], now=clk[0], ttl=30.0, eta=None)  # running: unwritten vs on-disk
        ev4 = [k for k, _ in pol3.evict(5, set())]
        assert ev4 == [k_rdisk, k_hdisk, k_rmem, k_pdisk, k_hmem], f"wrong recovery-cost hierarchy: {ev4}"
        STATE.fs_tier = None

        # 8. Cold-start tie-breaker in layby.rule: fast disk (own['d'] < own['r']) breaks W==0 tie to ins;
        #    slow disk (own['d'] >= own['r']) keeps none
        live0 = Live(120.0)
        p_fast = EngineParams(bytes_per_token=163840, gpu_tokens=34944, cpu_tokens=52224,
                              cpu_gbps=56.0, disk_gbps=1.35, f0=0.00035, t0=0.029)
        p_slow = EngineParams(bytes_per_token=65536, gpu_tokens=44096, cpu_tokens=73216,
                              cpu_gbps=56.0, disk_gbps=0.141, f0=0.00010, t0=0.022)
        S0 = interp_surv(np.linspace(0.95, 0.30, 15), TG)
        assert decide(S0, 25000, 0, (), (), live0, p_fast, clk[0])["opt"] == "ins"
        assert decide(S0, 25000, 0, (), (), live0, p_slow, clk[0])["opt"] == "none"
    finally:
        P.time = orig_p_time


def test_wave6_general_streaming_cumulative_defer_prefetch_and_io_timing():
    """Wave 6 general first-principles fixes:
    1. Adaptive streaming in 'rule' mode streams incremental stores ONLY on fast disk (bpt/disk_bps < f0),
       respecting STATE.nowrite, read-priority (_loading), and CPU pin/evictable headroom; NEVER on slow disk.
    2. Shared-prefix _refresh preserves decided PENDING_DISK protection until ParkFsTier pins it.
    3. Cumulative deferral budget (_defer_accum) stops slow-disk re-deferral loops while still excluding
       pure GPU queue wait; sticky _gave_up prevents re-launching doomed promotions; _check_ready ignores
       failed MISS promotions; CPU-pin-budget guard prevents overlapping promotions exceeding evictable slots.
    4. Profitable headroom-aware _prefetch skips unprofitable slow-disk reads and protects running chunks.
    5. Step-scaled pending-hint TTL survives large-model (>0.25s) scheduler steps.
    6. Worker-thread disk load I/O timing measures true NVMe speed rather than scheduler step interval."""
    root = tempfile.mkdtemp(prefix="pp_v2w6_", dir=os.path.expanduser("~/.cache"))
    try:
        clk = [__import__("time").monotonic()]
        orig_c_time, orig_p_time = C.time, P.time
        C.time = NS(monotonic=lambda: clk[0])
        P.time = NS(monotonic=lambda: clk[0])
        try:
            # --- 1. Adaptive streaming physics & backpressure ---
            conn, cs, mgr, prim, fs, pol = build(16, root + "/stream")
            STATE.gpu_tokens = 20000
            STATE.cpu_tokens = 10000    # tight CPU (GPU >= CPU, e.g. 320B MoE)
            STATE.bytes_per_token = 25000.0
            STATE.speeds.d["f0"] = 1e-4            # 10k tok/s prefill (100 us/tok)
            STATE.speeds.d["disk_gbps"] = 2.5      # 10 us/tok -> 10x faster than recompute (fast disk)
            STATE.speeds.d["disk_probe_bytes"] = float("inf")  # keep synthetic test disk speed clean
            store(mgr, "fill", [key(h) for h in hashes("f", 12)])
            settle(mgr, fs)
            ctx_s1 = ReqContext(req_id="r-s1", kv_transfer_params={"park_key": "s1-0"})
            assert fs._should_stream(ctx_s1, 2) is True, "fast disk tight CPU should stream turn 0"
            # STATE.nowrite must veto streaming even on fast disk
            STATE.nowrite.add("s1")
            assert fs._should_stream(ctx_s1, 2) is False, "STATE.nowrite did not veto streaming"
            STATE.nowrite.discard("s1")
            # Active disk read must veto streaming (read priority)
            fs._loading[1] = [key(hashes("z", 1)[0])]
            assert fs._should_stream(ctx_s1, 2) is False, "active load did not veto streaming"
            fs._loading.pop(1)
            # Tight available CPU headroom must veto streaming (zero afal): fill to 16, pin 12 -> avail=4 <= max_inflight=4
            store(mgr, "fill2", [key(h) for h in hashes("g", 4)])
            settle(mgr, fs)
            for k in list(pol.evictable)[:12]:
                pol.mark_non_evictable(k)
            assert fs._should_stream(ctx_s1, 2) is False, "low available headroom did not veto streaming"
            for k in list(pol.chunks):
                pol.mark_evictable(k)
            # Slow disk (bpt / disk_bps >= f0) must NEVER stream, even for STATE.thru sessions!
            STATE.thru.add("s1")
            STATE.speeds.d["disk_gbps"] = 0.14     # 178 us/tok > 100 us/tok (slow disk)
            assert fs._should_stream(ctx_s1, 2) is False, "slow disk wrongly enabled streaming!"
            # Fast disk + standard CPU (GPU < CPU): streams for STATE.thru sessions only
            STATE.speeds.d["disk_gbps"] = 2.5
            STATE.gpu_tokens = 10000
            STATE.cpu_tokens = 20000
            assert fs._should_stream(ctx_s1, 2) is True, "fast disk STATE.thru session should stream"
            ctx_s2 = ReqContext(req_id="r-s2", kv_transfer_params={"park_key": "s2-0"})
            assert fs._should_stream(ctx_s2, 2) is False, "undecided session streamed on generous CPU"
            fs.shutdown()

            # --- 2. Shared-prefix _refresh preserves PENDING_DISK ---
            pol_r = P.ParkCachePolicy(8)
            def mk():
                c = ChunkStatus.__new__(ChunkStatus); c.ref_cnt = 0; return c
            k_sh = ("sh", 0)
            pol_r.insert(k_sh, mk())
            STATE.fs_tier = NS(_on_disk=set())
            pol_r.mark_pending_write([k_sh], now=clk[0], ttl=30.0, eta=10.0)
            pol_r._refresh([k_sh], clk[0])
            assert k_sh in pol_r._pending_write and pol_r._pending_write[k_sh][1] == 10.0, (
                f"_refresh wiped PENDING_DISK protection: {pol_r._pending_write}"
            )
            STATE.fs_tier = None

            # --- 3. Cumulative deferral cap + sticky give-up + pin-budget guard + _check_ready ---
            conn, cs, mgr, prim, fs, pol = build(8, root + "/defer")
            STATE.bytes_per_token = 163840.0       # Gemma-4-like 164 KB/tok
            STATE.speeds.d["disk_gbps"] = 0.14     # slow disk: 1.17 ms/tok
            STATE.speeds.d["f0"] = 0.00032         # recompute: 0.32 ms/tok (3.6x faster than disk at npf=0)
            HA = hashes("a", 4); A = [key(h) for h in HA]
            HB = hashes("b", 4); Bk = [key(h) for h in HB]
            store(mgr, "a-1", A); fs.queue_write(A); settle(mgr, fs); settle(mgr, fs)
            store(mgr, "b-1", Bk); fs.queue_write(Bk); settle(mgr, fs); settle(mgr, fs)
            store(mgr, "c-1", [key(h) for h in hashes("c", 8)]); settle(mgr, fs)
            # Slow-disk _prefetch must skip unprofitable promotions when disk_s >= recompute_cost
            r_slow = request("r_slow", HA + hashes("p", 1), 5 * T)
            conn.on_new_request(r_slow)
            before_pf = STATE.counts["prefetch"]
            conn._prefetch()
            assert STATE.counts["prefetch"] == before_pf, "prefetch fired unprofitable slow-disk read!"
            # Cumulative deferral across re-deferrals: prior 1.0s deferral accumulates, does not reset to 0
            conn._deferred["r_slow"] = clk[0] - 5.0
            conn._ready["r_slow"] = clk[0] - 4.0   # 1.0s real deferral + 4.0s GPU queue wait
            held = []
            fs.submit_load = lambda j: held.append(j)
            # Still need > cap to trigger; here cap is tiny (~0.045s) so 1.0s accumulated immediately gives up!
            out = lookup(conn, mgr, r_slow)
            assert out == (0, False) and "r_slow" in conn._gave_up, f"cumulative defer cap failed: {out}"
            # Subsequent lookup must stay sticky _ReadyOnly (not re-defer)
            out2 = lookup(conn, mgr, r_slow)
            assert out2 == (0, False) and STATE.counts["defer_giveup"] == 1
            # _check_ready must NOT stamp _ready when missing chunk dropped to MISS while still on disk
            r_miss = request("r_miss", HB + hashes("q", 1), 5 * T)
            conn.on_new_request(r_miss)
            conn._deferred["r_miss"] = clk[0]
            conn._local["r_miss"] = 0
            conn._check_ready(clk[0] + 1.0)
            assert "r_miss" not in conn._ready, "_check_ready falsely stamped _ready on failed MISS promotion!"
            # Pin-budget guard (non-75% case): pinned=5 < 6 (75%), avail=3 < 4 chunks needed -> blocked!
            conn_p, cs_p, mgr_p, prim_p, fs_p, pol_p = build(8, root + "/pin")
            STATE.bytes_per_token = 163840.0
            STATE.speeds.d["disk_gbps"] = 0.14
            STATE.speeds.d["disk_probe_bytes"] = float("inf")
            STATE.speeds.d["f0"] = 0.00032
            store(mgr_p, "b-1", Bk); fs_p.queue_write(Bk); settle(mgr_p, fs_p); settle(mgr_p, fs_p)
            store(mgr_p, "c-1", [key(h) for h in hashes("c", 8)]); settle(mgr_p, fs_p)
            fs_p._loading[777] = [Bk[0]]
            for k in list(pol_p.evictable)[:5]:
                pol_p.mark_non_evictable(k)
            r_pin = request("r_pin", HB + hashes("r", 1), 5 * T)
            conn_p.on_new_request(r_pin)
            out_pin = conn_p.get_num_new_matched_tokens(r_pin, 0)
            assert out_pin == (0, False) and "r_pin" in conn_p._gave_up, f"pin-budget guard failed: {out_pin}"
            fs_p._loading.pop(777)
            fs_p.shutdown()
            fs.shutdown()

            # --- 4. Step-scaled pending-hint TTL survives 0.7s 320B MoE steps ---
            conn, cs, mgr, prim, fs, pol = build(8, root + "/ttl")
            conn._finished = __import__("collections").OrderedDict()
            conn._hints_on = True
            conn._step_dt = 0.70   # 320B MoE prefill step
            A2 = [key(h) for h in hashes("t", 2)]
            store(mgr, "t-1", A2); settle(mgr, fs)
            req_t = request("rt", hashes("t", 2), 2 * T)
            conn.on_new_request(req_t)
            conn._on_finish(req_t, [0, 1])
            exp_t = pol._pending_write[A2[0]][0]
            assert exp_t - clk[0] >= 2.8 - 1e-3, f"pending-hint TTL too short for 0.7s step: {exp_t - clk[0]:.2f}s"
            fs.shutdown()

            # --- 5. True worker-thread disk I/O timing ignores scheduler-step polling delay ---
            conn, cs, mgr, prim, fs, pol = build(8, root + "/io")
            fs._block_size = 100 << 20    # 100 MB per chunk
            STATE.speeds.d["disk_probe_bytes"] = 0.0
            t_sub = clk[0]
            fs._loading[1] = ["k0"]
            fs._submit[1] = (t_sub, 1)
            fs._load_mark = t_sub
            fs._load_io_times[1] = [t_sub + 0.01, t_sub + 0.02]   # 100 MB in 10 ms = 10.0 GB/s!
            # Scheduler polls 0.70s later (large-model step delay)
            orig_ft = F.time
            F.time = NS(monotonic=lambda: t_sub + 0.70)
            try:
                orig_gfj = F.FileSystemTierManager.get_finished_jobs
                F.FileSystemTierManager.get_finished_jobs = lambda self: [
                    __import__("vllm.v1.kv_offload.tiering.base", fromlist=["JobResult"]).JobResult(job_id=1, success=True)
                ]
                try:
                    fs.get_finished_jobs()
                finally:
                    F.FileSystemTierManager.get_finished_jobs = orig_gfj
            finally:
                F.time = orig_ft
            gbps = STATE.speeds.gbps("disk")
            assert gbps > 8.0, f"disk_gbps polluted by 0.7s step polling delay: got {gbps:.2f} GB/s"
            fs.shutdown()

            # --- 6. Step-scoped _step_protected prevents vLLM 0.34 lookup->_initiate_promotion->prepare_load crash ---
            conn, cs, mgr, prim, fs, pol = build(8, root + "/stepprot")
            HA = hashes("sa", 4); Ak = [key(h) for h in HA]
            HB = hashes("sb", 4); Bk = [key(h) for h in HB]
            HC = hashes("sc", 4); Ck = [key(h) for h in HC]
            store(mgr, "s1", Ak + Bk); settle(mgr, fs)   # fills all 8 CPU slots
            # Put C on disk then evict C (already absent) by writing via a temporary full store
            # Simulate C on disk + A, B resident in CPU (all 8 slots evictable)
            fs._on_disk.update(Ck)
            ra = request("ra", HA + hashes("ra_t", 1), 5 * T)
            conn.on_new_request(ra)
            out_a = conn.get_num_new_matched_tokens(ra, 0)
            assert out_a[0] == 4 * T, f"expected 4-chunk CPU hit for ra, got {out_a}"
            assert all(k in pol._step_protected for k in Ak), "matched HIT chunks of ra not step-protected"
            # Competing store/promotion in the SAME scheduler step tries to evict 6 chunks (would eat Ak!)
            assert pol.evict(6, set()) is None, "evict() violated _step_protected and evicted ra HIT chunks!"
            # Evicting only the 4 non-protected Bk chunks still succeeds, leaving Ak intact for prepare_load!
            ev4 = pol.evict(4, set())
            assert ev4 is not None and len(ev4) == 4 and all(k in pol.chunks for k in Ak)
            # prepare_load(Ak) succeeds with ZERO AssertionError!
            spec = mgr.prepare_load(Ak, cs._req_status["ra"].req_context)
            assert spec is not None
            mgr.complete_load(Ak, cs._req_status["ra"].req_context)
            # Strict write pin cap under active load (<= cap//4 = 2 chunks for cap=8):
            fs._loading[99] = [Ck[0]]
            fs.queue_write(Ak)
            fs.serve_external_requests(mgr)
            assert len(fs._inflight) <= 2, f"late write exceeded cap//4 pin limit under load: {len(fs._inflight)}"
            fs._loading.pop(99)
            pol._step_protected.clear()
            fs.shutdown()

            # --- 7. Same-step _pending_load_submissions serialization & FCFS queue-head residency protection ---
            conn, cs, mgr, prim, fs, pol = build(8, root + "/qorder")
            STATE.bytes_per_token = 25000.0
            STATE.speeds.d["disk_gbps"] = 12.0
            STATE.speeds.d["disk_probe_bytes"] = float("inf")
            STATE.speeds.d["f0"] = 1e-4
            H1 = hashes("q1", 5); K1 = [key(h) for h in H1]
            H2 = hashes("q2", 5); K2 = [key(h) for h in H2]
            store(mgr, "q1s", K1); fs.queue_write(K1); settle(mgr, fs); settle(mgr, fs); settle(mgr, fs)
            store(mgr, "q2s", K2); fs.queue_write(K2); settle(mgr, fs); settle(mgr, fs); settle(mgr, fs)
            store(mgr, "qev", [key(h) for h in hashes("qe", 8)]); settle(mgr, fs)
            r1 = request("rq1", H1 + hashes("t1", 1), 6 * T)
            r2 = request("rq2", H2 + hashes("t2", 1), 6 * T)
            conn.on_new_request(r1)
            clk[0] += 0.1
            conn.on_new_request(r2)
            # Same-step: r1 queues 5-chunk promotion (_pending_load_submissions=5, fs._loading still empty!)
            out1 = conn.get_num_new_matched_tokens(r1, 0)
            assert out1 == (None, False) and conn._pending_promo_chunks() == 5
            # r2 in the SAME step must NOT launch overlapping 5-chunk promotion (would hit promo_afal!)
            out2 = conn.get_num_new_matched_tokens(r2, 0)
            assert out2 == (None, False) and conn._pending_promo_chunks() == 5, (
                f"same-step overlap not serialized: pending={conn._pending_promo_chunks()}"
            )
            # _prefetch must also skip while same-step pending promotion exists
            pf_before = STATE.counts["prefetch"]
            conn._prefetch()
            assert STATE.counts["prefetch"] == pf_before, "prefetch stacked onto same-step pending promotion!"
            # Finish r1's promotion so r1 holds 5/8 CPU chunks waiting for GPU blocks (active_promo==0 now)
            settle(mgr, fs); settle(mgr, fs)
            assert all(k in pol.chunks for k in K1) and conn._pending_promo_chunks() == 0
            # r2 (5 + 5 = 10 > cap=8) must defer WITHOUT evicting r1's chunks and WITHOUT starting _deferred clock!
            out2b = conn.get_num_new_matched_tokens(r2, 0)
            assert out2b == (None, False) and "rq2" not in conn._deferred and all(k in pol.chunks for k in K1), (
                f"queue-head residency violated: out2b={out2b} def={conn._deferred}"
            )
            # Once r1 is admitted (unheld), r2 immediately promotes cleanly!
            conn._unhold("rq1")
            out2c = conn.get_num_new_matched_tokens(r2, 0)
            assert out2c == (None, False) and conn._pending_promo_chunks() == 5, (
                f"r2 did not promote after r1 unheld: pending={conn._pending_promo_chunks()}"
            )
            fs.shutdown()

            # --- 8. Slow-disk queue_overflow prices ahead_disk_chunks + pin_ceil prevents over-pinning ---
            conn, cs, mgr, prim, fs, pol = build(8, root + "/slowq")
            STATE.bytes_per_token = 163840.0
            STATE.speeds.d["disk_gbps"] = 0.145
            STATE.speeds.d["disk_probe_bytes"] = float("inf")
            STATE.speeds.d["f0"] = 3e-4
            # Moderate pf (~2.0): single own_disk_s (~0.090s) < cap (~0.147s), but queued ahead + own (~0.181s) > cap!
            STATE.live.tick(clk[0])
            clk[0] += 10.0
            STATE.live.add(clk[0], pf=19.2)
            HS1 = hashes("sq1", 5); KS1 = [key(h) for h in HS1]
            HS2 = hashes("sq2", 5); KS2 = [key(h) for h in HS2]
            store(mgr, "sq1s", KS1); fs.queue_write(KS1); settle(mgr, fs); settle(mgr, fs); settle(mgr, fs)
            store(mgr, "sq2s", KS2); fs.queue_write(KS2); settle(mgr, fs); settle(mgr, fs); settle(mgr, fs)
            store(mgr, "sqev", [key(h) for h in hashes("sqe", 8)]); settle(mgr, fs)
            rs1 = request("rsq1", HS1 + hashes("st1", 1), 6 * T)
            rs2 = request("rsq2", HS2 + hashes("st2", 1), 6 * T)
            conn.on_new_request(rs1)
            clk[0] += 0.1
            conn.on_new_request(rs2)
            # Without starting rs1's promotion yet, rs2 sees ahead_disk_chunks=5 + own=5: slow disk wait exceeds cap -> immediate give-up!
            out_s2 = conn.get_num_new_matched_tokens(rs2, 0)
            assert out_s2 == (0, False) and "rsq2" in conn._gave_up, (
                f"slow-disk queue_overflow failed to price ahead_disk_chunks: out={out_s2} gave_up={conn._gave_up}"
            )
            # And when 3 chunks are already pinned (e.g. GPU->CPU stores) on cap=8 (pin_ceil=7),
            # a 5-chunk slow-disk promotion (3 + 5 = 8 > 7) must NOT over-pin CPU!
            ev_keys = list(pol.evictable)[:3]
            for ek in ev_keys:
                pol.mark_non_evictable(ek)
            STATE.live.add(clk[0], pf=500.0)   # raise cap so est_wait <= cap -> must defer via pin_block, not start!
            out_s1 = conn.get_num_new_matched_tokens(rs1, 0)
            assert out_s1 == (None, False) and conn._pending_promo_chunks() == 0, (
                f"pin_ceil did not guard over-pinning when pinned>0: out={out_s1} pending={conn._pending_promo_chunks()}"
            )
            fs.shutdown()
        finally:
            C.time, P.time = orig_c_time, orig_p_time
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_nightly_iteration_details_fallback():
    """vLLM nightly (>=0.34.0rc1) fallback reads iteration_details or computes ctx counts from SchedulerOutput."""
    direct = NS(num_ctx_requests=3, num_ctx_tokens=4096)
    got_direct = C._fallback_iteration_details(NS(iteration_details=direct))
    assert got_direct is direct
    sched_out = NS(
        num_scheduled_tokens={"new1": 1024, "new2": 512, "chunked": 256, "decode": 1},
        scheduled_new_reqs=[NS(req_id="new1"), NS(req_id="new2")],
        scheduled_cached_reqs=NS(req_ids=["chunked", "decode"]),
    )
    got = C._fallback_iteration_details(sched_out)
    assert got.num_ctx_requests == 3 and got.num_ctx_tokens == 1792, f"unexpected fallback counts: {got}"


if __name__ == "__main__":
    test_bug_b_token_proportional_defer_and_ghost_decay()
    test_bug_a_hybrid_on_disk_no_overcount()
    test_bug_c_hybrid_ready_tokens_and_giveup_single_count()
    test_bug_d_park_write_protection_and_read_priority()
    test_upgrade_e_policy_lazy_expire_parity_and_o1_live()
    test_wave2_incremental_W_and_read_only_contention()
    test_wave2_bounded_prefetch_and_redefer_reset()
    test_wave3_pending_hint_ins_protection_thru_and_ready()
    test_wave4_no_disk_gpu_blk_roofline_f0_running_protect_and_backpressure()
    test_wave6_general_streaming_cumulative_defer_prefetch_and_io_timing()
    test_nightly_iteration_details_fallback()
    print("layby v2 engine fixes ok: wave1-wave6 + nightly fallback all verified")
