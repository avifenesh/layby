"""ParkConnector: vLLM 0.30's OffloadingConnector plus the park cost rule.

The connector is the engine adapter for layby.rule.decide. On the scheduler side (EngineCore process,
scheduler thread) it measures what the rule needs and makes the placement decision for each finished
request; transfers, the CPU tier and the disk tier stay vLLM's (with layby.vllm.policy and
layby.vllm.fs_tier as the CPU eviction policy and the disk tier).

Measured here, fed to layby.vllm.state.STATE:
  admission     wait from add to first allocation (Live n_adm, qwait), and for requests the offload tier
                deferred while promoting their prefix, the wait from the first deferral (Live n_defer,
                defer: the measured cost of a restore from disk); waiting and prefilling request
                counts over time (Live nw, pf)
  deferral cap  a deferral runs until it has cost what giving up would: once the time since the first
                deferral exceeds the recompute of the tokens it waits on (prompt - local hit - tokens ready
                now, at the measured prefill speed, priced as layby.rule prices a recompute), the base lookup
                runs with every chunk not ready read as a miss, so the scheduler admits the request with the
                prefix the CPU tier can load now and recomputes the rest (counts defer_giveup, rlog
                giveup=True). The promotion stays in flight and lands in the CPU tier when the disk delivers
                it; the seconds that saved are measured then (Live n_giveup, giveup_saved)
  steps         time between consecutive schedules and the tokens prefilled in them (Speeds.step: t0, f0)
  CPU link      per load job: submit to completion, transfer bytes and time from the worker (Speeds.link,
                Live n_cpu / wait_cpu / busy_cpu)
  GPU residency the block pool's metrics collector is wrapped: a cached block whose last request
                finished is idle; eviction ends the period as an event, a prefix hit as censored
  misses        at admission, the latency a request loses to KV evicted from the GPU and from the CPU
                tier, against the time since that eviction (Live.miss, the source of the holding price)

Queue hold: from its arrival to its admission (or its end), a request's chunks are held in the CPU tier at its
queue rank (ParkCachePolicy.hold): evicted after every idle chunk, the last-arrived first. A preempted request is
held again at the head of the queue. Counts held.

Prefetch: each step, the waiting requests whose held chunks fit in the CPU tier's room (queue order) have the
disk -> CPU promotion of their prefix past what the GPU prefix cache holds (and of each windowed group's window at its
end) started once, so a disk restore runs during the queue wait. An admission
whose prefetch restored its prefix before the scheduler reached it is a disk restore that cost no wait (defer 0).
Counts prefetch (chunks).

Decision: when a request finishes its curve may already be there (kv_transfer_params["park_surv"]);
otherwise the request is remembered under its park_key and decided when POST /hint delivers the curve.
The decision sets the CPU tier's predicted next use of the request's chunks (park_eta; first out for
park and drop) and, for ins and park, writes them to disk (ParkFsTier.queue_write).
"""
import time
from collections import OrderedDict

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import get_block_hash
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.utils import compute_iteration_details

from layby.rule import TG, decide, interp_surv, recompute_cost
from layby.vllm import server
from layby.vllm.state import STATE

logger = init_logger("vllm." + __name__)   # under vLLM's handler: a module name outside "vllm" prints nothing
_failed: set = set()


def _safe(name, fn, *args):
    """Run park bookkeeping; on an internal error log it once and let vLLM go on (the request keeps
    vLLM's default placement)."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        if name not in _failed:
            _failed.add(name)
            logger.exception("park: %s failed; continuing with vLLM defaults for it", name)
        return None
HORIZON = 3600.0          # the residency curves' last age bin; older eviction records cannot matter


class _GpuResidency:
    """Wraps BlockPool.metrics_collector to see every prefix-cache eviction and hit."""

    def __init__(self, inner, pool):
        self.inner, self.pool = inner, pool

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def on_block_allocated(self, block):
        STATE.g_idle.pop(block.block_id, None)        # an idle block handed out again without a hash
        if self.inner:
            self.inner.on_block_allocated(block)

    def on_block_accessed(self, block):
        t = STATE.g_idle.pop(block.block_id, None)
        if t is not None:
            STATE.ended[("g", False)].append(time.monotonic() - t)
        if self.inner:
            self.inner.on_block_accessed(block)

    def on_block_evicted(self, block):
        now = time.monotonic()
        t = STATE.g_idle.pop(block.block_id, None)
        STATE.ended[("g", True)].append(now - t if t is not None else 0.0)
        if block.block_hash is not None:
            STATE.g_evicted[get_block_hash(block.block_hash)] = now
        if self.inner:
            self.inner.on_block_evicted(block)

    def drain_events(self):
        return self.inner.drain_events() if self.inner else []

    def reset(self):
        if self.inner:
            self.inner.reset()

    def __bool__(self):
        return True


class _ReadyOnly:
    """The offloading manager with every chunk that is not ready now read as a miss. The base lookup run
    through it returns the prefix the CPU tier can load this step instead of deferring the request; it
    stops at the first miss, so it starts no promotion past the one in flight."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def lookup(self, key, req_context, **kw):
        r = self.inner.lookup(key, req_context, **kw)
        return r if r is LookupResult.HIT else LookupResult.MISS


def _request_keys(req_status):
    """The request's offloaded chunk keys across every KV group (a hybrid model such as GLM 5.3 Flash keeps MLA
    latents, indexer keys and linear-attention checkpoints in separate groups, and a load needs all of them), in
    prefix order by relative position, so placing them last-first evicts every group's tail before any prefix."""
    states = req_status.group_states
    if len(states) == 1:
        return list(states[0].offload_keys)
    items = []
    for gi, gs in enumerate(states):
        n = len(gs.offload_keys)
        items += [((i + 1) / n, gi, k) for i, k in enumerate(gs.offload_keys)]
    items.sort(key=lambda x: (x[0], x[1]))
    return [k for _, _, k in items]


def session_of(park_key):
    """The session of a park_key (\"<session>-<turn>\", as the replay client and the sidecar send it)."""
    return str(park_key).rsplit("-", 1)[0] if park_key is not None else None


class ParkConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if role != KVConnectorRole.SCHEDULER:
            return
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        cs = self.connector_scheduler
        g0 = cs.config.kv_group_configs[0]
        STATE.block_tokens = vllm_config.cache_config.block_size
        STATE.hash_tokens = STATE.block_tokens      # tokens per request.block_hashes entry
        try:   # vLLM >= 0.31: prefix_match_unit can hash finer than the block
            from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
            STATE.hash_tokens = resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)[1]
        except Exception:   # noqa: BLE001  (older vLLM: hashes per block)
            pass
        STATE.chunk_tokens = g0.tokens_per_chunk
        primary = getattr(cs.manager, "primary_tier", cs.manager)
        view = primary.get_kv_memoryview()
        STATE.bytes_per_token = view.strides[0] / STATE.chunk_tokens
        STATE.cpu_tokens = primary._num_chunks * STATE.chunk_tokens
        STATE.gpu_tokens = kv_cache_config.num_blocks * STATE.block_tokens
        self._waiting: dict[str, float] = {}           # request id -> time added, until admitted
        self._running: dict[str, list[int]] = {}       # running request id -> per-group protected key counts
        self._deferred: dict[str, float] = {}          # request id -> first lookup that deferred it
        self._defer_ready: dict[str, int] = {}         # request id -> tokens already in CPU at first deferral
        self._est0: dict[str, tuple[float, float, float]] = {}  # request id -> (t, first remaining est, link est)
        self._ready: dict[str, float] = {}             # request id -> first lookup after its deferral that could load
        self._prefetched: dict[str, int] = {}          # waiting request id -> promotions its prefetch started
        # request id -> its give-up (the deferral it stopped), until admitted and settled
        self._giveups: dict[str, dict] = {}
        # request id -> the GPU-local prefix hit (tokens) vLLM passed to the last lookup. vLLM calls
        # update_state_after_alloc before it sets request.num_computed_tokens, so a new request reads 0 there
        self._local: dict[str, int] = {}
        self._finished: "OrderedDict[str, dict]" = OrderedDict()   # park_key -> request awaiting a curve
        self._early: dict[str, tuple[float, list]] = {}               # park_key -> curve that beat the finish
        self._loads: dict[int, tuple[float, int]] = {}  # load job id -> (submit time, chunks)
        self._last_step = None                          # (time, prefill tokens, had work)
        self._last_prune = time.monotonic()
        STATE.write_mode = extra.get("park_write", "rule")
        self._gpu_order = bool(extra.get("park_gpu_order", False))
        self._gpu_demote: list[list[tuple[int, object]]] = []
        if STATE.write_mode == "demand":
            # KD write-back: a chunk the rule kept in RAM only is written to disk when the CPU tier evicts it
            from layby.vllm import demote
            if demote.install(cs.manager) is None:
                logger.error("park: write-on-evict needs TieringOffloadingSpec, ParkCachePolicy and ParkFsTier; "
                             "demand mode will not spill")
        port = extra.get("park_port")
        self._hints_on = False            # without a hint port, curves can only ride on the request
        if port:
            try:
                server.start(int(port), str(extra.get("park_host", "127.0.0.1")))
                self._hints_on = True
                if extra.get("park_prof"):
                    from layby import prof
                    prof.start(self._load_tag)    # samples this (the scheduler) thread; GET /prof, /prof/windows
                logger.info("park: hint port %s", port)
            except OSError as e:
                logger.error("park: hint port %s unavailable (%s); serving without late hints", port, e)
        logger.info("park: gpu %d tokens, cpu %d tokens, %d tokens/chunk, %.0f bytes/token",
                    STATE.gpu_tokens, STATE.cpu_tokens, STATE.chunk_tokens, STATE.bytes_per_token)

    # --- GPU residency -------------------------------------------------------------------
    def _load_tag(self):
        """Engine load when a profile window closes: requests waiting, step time, prefill speed, disk deferral."""
        now = time.monotonic()
        L = STATE.live
        nd = L.rate("n_defer", now)
        return dict(waiting=len(self._waiting), deferred=len(self._deferred), t0=round(STATE.speeds.t0(), 4),
                    f0=round(STATE.speeds.f0() * 1e6, 1), defer=round(L.rate("defer", now) / nd, 2) if nd > 0 else 0.0,
                    busy_disk=round(L.rate("busy_disk", now), 2))

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        super().bind_gpu_block_pool(gpu_block_pool)
        gpu_block_pool.metrics_collector = _GpuResidency(gpu_block_pool.metrics_collector, gpu_block_pool)
        self._pool = gpu_block_pool

    # --- admission and misses -----------------------------------------------------------
    def on_new_request(self, request) -> None:
        now = time.monotonic()
        self._waiting[request.request_id] = now
        super().on_new_request(request)
        _safe("hold", self._hold, request.request_id, now)

    def _hold(self, rid: str, rank: float) -> None:
        """Queue hold: the CPU tier keeps a waiting request's chunks by its queue rank until admission
        (ParkCachePolicy.hold)."""
        pol = STATE.policy
        rs = self.connector_scheduler._req_status.get(rid)
        if pol is None or rs is None or not hasattr(pol, "hold"):
            return
        rs.update_offload_keys()
        pol.hold(rid, self._hold_keys(rs), rank)
        self._prefetched.pop(rid, None)        # a preempted request is prefetched again
        STATE.counts["held"] += 1

    def _hold_keys(self, rs) -> list:
        """The chunks a load of this request can use, in prefix order (by end token across groups): every
        full-attention chunk, and for each windowed group (sliding window, Mamba state) only the window that ends
        where the full-attention prefix in the CPU tier or on disk ends; older windows are never loaded."""
        cs = self.connector_scheduler
        mgr = cs.manager
        prim = getattr(mgr, "primary_tier", mgr)
        fs = STATE.fs_tier
        ctx = rs.req_context
        present = lambda k: prim.lookup(k, ctx) is not LookupResult.MISS or (fs is not None and k in fs._on_disk)
        groups = list(zip(cs.config.kv_group_configs, rs.group_states))
        out, bound = [], None
        for g, gs in groups:
            if g.sliding_window_size_in_chunks is not None:
                continue
            ks = list(gs.offload_keys)
            out += ks
            last = next((i for i in range(len(ks) - 1, -1, -1) if present(ks[i])), -1)
            t = (last + 1) * g.tokens_per_chunk
            bound = t if bound is None else min(bound, t)
        for g, gs in groups:
            if g.sliding_window_size_in_chunks is None:
                continue
            ks = list(gs.offload_keys)
            if bound is None:
                end = next((i + 1 for i in range(len(ks) - 1, -1, -1) if present(ks[i])), 0)
            else:
                end = min(bound // g.tokens_per_chunk, len(ks))
            w = (g.load_window_size_in_chunks(end * g.tokens_per_chunk) if hasattr(g, "load_window_size_in_chunks")
                 else g.sliding_window_size_in_chunks) or 0
            out += ks[max(end - w, 0):end]
        pos = getattr(ctx, "get_offload_key_position", None)
        if pos is not None and len(groups) > 1:
            out.sort(key=lambda k: pos(k) or 0)
        return out

    def _unhold(self, rid: str) -> None:
        pol = STATE.policy
        if pol is not None and hasattr(pol, "unhold"):
            pol.unhold(rid)

    def _prefetch(self) -> None:
        """Prefetch: start the disk -> CPU promotion of waiting requests' prefixes while they queue, so a restore
        from disk overlaps the queue wait instead of starting when the scheduler reaches the request. In queue
        order, while the chunks held by the requests up to this one fit in the CPU tier's room (its capacity less
        the chunks in use), once per request; the scheduler's own lookup at admission handles anything that is not
        ready. The tiering manager submits the promotions at the end of this schedule."""
        pol = STATE.policy
        if pol is None or not getattr(pol, "held", None) or STATE.fs_tier is None:
            return
        cs = self.connector_scheduler
        mgr = cs.manager
        prim = getattr(mgr, "primary_tier", None)
        if prim is None:
            return
        fs = STATE.fs_tier
        # do not stack prefetch reads onto an already-busy disk read queue (avoids CPU thrashing and
        # disk contention with active admission promotions)
        if getattr(fs, "_loading", None):
            return
        now = time.monotonic()
        # leave headroom for running requests' CPU stores so prefetched waiting chunks are not evicted
        # before admission (which caused repeated disk re-reads when filling 100% of a tight CPU tier)
        reserve = min(int(STATE.burst_now(now)), pol.cache_capacity // 3)
        room = pol.cache_capacity - (len(pol.chunks) - len(pol.evictable)) - reserve
        used = 0
        for rid, (_, keys) in sorted(pol.held.items(), key=lambda kv: kv[1][0]):
            used += len(keys)
            if used > room:
                break
            if rid in self._prefetched:
                continue
            rs = cs._req_status.get(rid)
            if rs is None or rs.transfer_jobs:
                continue
            n = self._prefetch_req(rs, mgr, prim)
            self._prefetched[rid] = n
            STATE.counts["prefetch"] += n
            if n > 0:
                break   # prefetch one waiting request per step so head-of-queue lands first without ping-pong

    def _prefetch_req(self, rs, mgr, prim) -> int:
        """Look up a waiting request's chunks as the scheduler will (the tiering manager starts a promotion for each
        one only the disk has): full-attention groups along the prefix up to the first chunk no tier has, then each
        windowed group (sliding window, Mamba state) over the window that ends at that prefix. Returns the
        promotions started."""
        groups = list(zip(self.connector_scheduler.config.kv_group_configs, rs.group_states))
        n = 0
        local = self._gpu_prefix_tokens(rs)    # vLLM loads only past the prefix its GPU cache still holds

        def one(k):
            nonlocal n
            before = prim.lookup(k, rs.req_context)
            r = mgr.lookup(k, rs.req_context)
            if before is LookupResult.MISS and r is LookupResult.HIT_PENDING:
                n += 1
            return r

        bound = None                                # tokens of the prefix every full-attention group has
        for g, gs in groups:
            if g.sliding_window_size_in_chunks is not None:
                continue
            start = local // g.tokens_per_chunk
            hit = start
            for k in gs.offload_keys[start:]:
                if one(k) is LookupResult.MISS:
                    break
                hit += 1
            bound = hit * g.tokens_per_chunk if bound is None else min(bound, hit * g.tokens_per_chunk)
        for g, gs in groups:
            if g.sliding_window_size_in_chunks is None:
                continue
            ks = gs.offload_keys
            end = len(ks) if bound is None else min(bound // g.tokens_per_chunk, len(ks))
            w = (g.load_window_size_in_chunks(end * g.tokens_per_chunk) if hasattr(g, "load_window_size_in_chunks")
                 else g.sliding_window_size_in_chunks)
            for k in ks[max(end - (w or 0), 0):end]:
                one(k)
        return n

    def _gpu_prefix_tokens(self, rs) -> int:
        """Tokens of the request's prefix whose blocks the GPU prefix cache holds now, in every full-attention group
        (vLLM's local hit; the offload lookup starts past it). 0 without a bound block pool."""
        pool = getattr(self, "_pool", None)
        req = getattr(rs, "req", None)
        if pool is None or req is None or not hasattr(pool, "get_cached_block"):
            return 0
        gids = [g.group_idx for g in self.connector_scheduler.config.kv_group_configs
                if g.sliding_window_size_in_chunks is None]
        if not gids:
            return 0
        n = 0
        for h in req.block_hashes:
            if pool.get_cached_block(h, gids) is None:
                break
            n += 1
        return n * STATE.hash_tokens

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        rid = request.request_id
        self._local[rid] = int(num_computed_tokens)
        pol, fs = STATE.policy, STATE.fs_tier
        # promotion-livelock guard: if active disk loads already pin >=75% of the CPU tier, do not
        # start yet another concurrent promotion for a new request (which pins 100% of CPU RAM and
        # triggers promotion_allocation_failures livelocks); serve any resident CPU prefix directly
        if (rid not in self._deferred and pol is not None and fs is not None
                and getattr(fs, "_loading", None)
                and (len(pol.chunks) - len(pol.evictable)) >= max(pol.cache_capacity * 3 // 4, 1)):
            cs = self.connector_scheduler
            proxy = _ReadyOnly(cs.manager)
            cs.manager = proxy
            try:
                out = super().get_num_new_matched_tokens(request, num_computed_tokens)
            finally:
                cs.manager = proxy.inner
            return out
        out = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if out[0] is not None:
            if rid in self._deferred and rid not in self._ready:
                self._ready[rid] = time.monotonic()      # the promotion it waited on has landed
            return out
        now = time.monotonic()
        if rid in self._ready:
            # an earlier promotion already landed and waited in the GPU queue before being evicted:
            # reset the deferral clock so the fresh promotion is not falsely killed by an old timestamp
            self._ready.pop(rid, None)
            self._deferred.pop(rid, None)
            self.__dict__.get("_est0", {}).pop(rid, None)
        t0 = self._deferred.get(rid)
        if t0 is None:
            # the offload tier is promoting part of the prefix (disk -> CPU): the request waits steps
            self._deferred[rid] = t0 = now
            rs = self.connector_scheduler._req_status.get(rid)
            if rs is not None:
                dr = self.__dict__.setdefault("_defer_ready", {})
                dr[rid] = self._ready_tokens(rs, int(num_computed_tokens))[0]
        alt = _safe("defer cap", self._defer_cap, request, int(num_computed_tokens), now - t0, now)
        return out if alt is None else alt

    def _defer_cap(self, request, local: int, waited: float, now: float):
        """The bounded deferral: give up and return what the base can load now (0 if nothing) when the remaining
        wait for the promotion exceeds the recompute of the tokens it waits on (forward), when the wait so far
        exceeds the first remaining estimate by more than that recompute (safety), or, with no estimate, when
        the wait so far exceeds it (fallback). None: keep waiting."""
        cs = self.connector_scheduler
        rid = request.request_id
        rs = cs._req_status.get(rid)
        if rs is None or rs.transfer_jobs:
            return None     # the base holds a request with its own transfers in flight; that is not a promotion
        ready, key = self._ready_tokens(rs, local)
        n = max(request.num_prompt_tokens - local - ready, 0)
        L = STATE.live
        x, cap = recompute_cost(n, L.rate("pf", now), L.rate("n_adm", now), STATE.params())
        rem, link = self._remaining(key)
        est0_map = self.__dict__.setdefault("_est0", {})
        est = est0_map.get(rid)
        if rem is not None and est is None:
            est = est0_map[rid] = (now, rem, link)
        if rem is not None and rem > cap:
            branch = "forward"
        elif est is not None and waited > est[1] + cap:
            branch = "safety"
        elif est is None and rem is None and waited > cap:
            branch = "fallback"
        else:
            return None
        proxy = _ReadyOnly(cs.manager)
        cs.manager = proxy
        try:
            out = cs.get_num_new_matched_tokens(
                request, local, max_num_new_tokens=self._max_loadable_tokens(request, local))
        finally:
            cs.manager = proxy.inner
        if out[0] is None:
            return None     # another request is loading these chunks to the GPU: that wait ends with its load
        g = self._giveups.get(rid)
        if g is None or g.get("admitted"):
            # one give-up per deferral: a request the scheduler could not fit after giving up gives up again
            # next step without a second count (a preempted request's new deferral is a new give-up)
            STATE.counts["defer_giveup"] += 1
            g = self._giveups[rid] = dict(t=now, ctx=rs.req_context)
        g.update(x=x, waited=waited, cap=cap, ready=ready, n=n, key=key, rem=rem, branch=branch,
                 est0=None if est is None else est[1], link0=None if est is None else est[2])
        logger.debug("park: request %s gave up a %.1f s deferral (%s: rem %s s, cap %.2f s, %d tokens ready, "
                     "%d to recompute)", rid, waited, branch, None if rem is None else round(rem, 2), cap, ready, n)
        return out

    def _remaining(self, key):
        """Seconds until the promotion for key lands: bytes ahead on the disk load queue (submitted loads plus
        promotions the tiering manager still holds until step end) at the queue's measured drain rate, and at
        the per-transfer disk speed. (None, None) when no load job reads key."""
        fs = STATE.fs_tier
        if key is None or fs is None or not hasattr(fs, "load_queue_ahead"):
            return None, None
        mgr = self.connector_scheduler.manager
        pend = getattr(mgr, "_pending_load_submissions", None) or {}
        pending = ()
        for i, tier in enumerate(getattr(mgr, "secondary_tiers", ())):
            if tier is fs:
                pending = [e.keys for e in pend.get(i, {}).values()]
                break
        ahead = fs.load_queue_ahead(key, pending)
        if ahead is None:
            return None, None
        link_bps = STATE.speeds.gbps("disk") * 1e9
        link = ahead / link_bps if link_bps > 0 else None
        drain = fs.drain_rate() if hasattr(fs, "drain_rate") else None
        bps = drain if drain else link_bps
        return (ahead / bps if bps > 0 else None), link

    def _ready_tokens(self, rs, local: int):
        """Tokens past the local hit that every KV group can load now, and the first chunk key that is not
        ready (preferring HIT_PENDING). For hybrid windowed groups (SWA, Mamba), checks only the window ending
        at the full-attention prefix rather than stale pre-window chunks."""
        mgr = self.connector_scheduler.manager
        prim = getattr(mgr, "primary_tier", mgr)
        groups = list(zip(self.connector_scheduler.config.kv_group_configs, rs.group_states))
        ready, pending = None, None
        for g, gs in groups:
            if getattr(g, "sliding_window_size_in_chunks", None) is not None:
                continue
            tc = g.tokens_per_chunk
            start = local // tc
            n = 0
            for key in gs.offload_keys[start:]:
                r = prim.lookup(key, rs.req_context)
                if r is not LookupResult.HIT:
                    if pending is None or r is LookupResult.HIT_PENDING:
                        pending = key
                    break
                n += 1
            tokens = max(tc * (start + n) - local, 0)
            ready = tokens if ready is None else min(ready, tokens)
        bound = local + (ready or 0)
        for g, gs in groups:
            sw = getattr(g, "sliding_window_size_in_chunks", None)
            if sw is None:
                continue
            tc = g.tokens_per_chunk
            ks = gs.offload_keys
            end = len(ks) if ready is None else min(bound // tc, len(ks))
            w = (g.load_window_size_in_chunks(end * tc) if hasattr(g, "load_window_size_in_chunks") else sw) or 0
            lo = max(end - w, 0)
            for key in ks[lo:end]:
                r = prim.lookup(key, rs.req_context)
                if r is not LookupResult.HIT:
                    if r is LookupResult.HIT_PENDING or pending is None:
                        pending = key
                    ready = 0
                    break
        return ready or 0, pending

    def _settle_giveups(self, now: float) -> None:
        """A give-up's saving is known once the promotion it waited on resolves: the chunk is ready in the CPU
        tier (or gone, the promotion failed or its slot was evicted). Keep settled entries while the request is
        still deferred (not yet admitted) so defer_giveup is never counted twice for one deferral."""
        if not self._giveups:
            return
        mgr = self.connector_scheduler.manager
        prim = getattr(mgr, "primary_tier", mgr)
        for rid, g in list(self._giveups.items()):
            if g.get("settled"):
                if rid not in self._deferred:
                    del self._giveups[rid]
                continue
            if g["key"] is not None and prim.lookup(g["key"], g["ctx"]) is LookupResult.HIT_PENDING:
                continue
            STATE.live.add(now, n_giveup=1.0, giveup_saved=(now - g["t"]) - g["x"])
            g["settled"] = True
            if rid not in self._deferred:
                del self._giveups[rid]

    def _check_ready(self, now: float) -> None:
        """Stamp _ready[rid] as soon as a deferred request's disk promotion completes in primary_tier,
        even if vLLM's FCFS scheduler is blocked on an earlier queued request and does not poll rid
        again for several seconds (prevents GPU queue wait from polluting defer telemetry)."""
        if not self._deferred:
            return
        cs = self.connector_scheduler
        mgr = cs.manager
        prim = getattr(mgr, "primary_tier", mgr)
        for rid in list(self._deferred):
            if rid in self._ready or rid in self._giveups:
                continue
            rs = cs._req_status.get(rid)
            if rs is None or rs.transfer_jobs:
                continue
            _, key = self._ready_tokens(rs, self._local.get(rid, 0))
            if key is None or prim.lookup(key, rs.req_context) is not LookupResult.HIT_PENDING:
                self._ready[rid] = now

    def _admit(self, request, num_external_tokens):
        rid = request.request_id
        self._unhold(rid)
        pol = STATE.policy
        rs = self.connector_scheduler._req_status.get(rid)
        if pol is not None and rs is not None and hasattr(pol, "mark_pending_write"):
            now_a = time.monotonic()
            rs.update_offload_keys()
            lens = [len(gs.offload_keys) for gs in rs.group_states]
            keys = _request_keys(rs)
            self.__dict__.setdefault("_running", {})[rid] = lens
            if keys:
                pol.mark_pending_write(keys, now_a, ttl=120.0, eta=None)
        pf = self._prefetched.pop(rid, 0)
        t = self._waiting.pop(rid, None)
        td = self._deferred.pop(rid, None)
        tr = self._ready.pop(rid, None)
        dr = self.__dict__.get("_defer_ready", {}).pop(rid, 0)
        est = self.__dict__.get("_est0", {}).pop(rid, None)
        local = self._local.pop(rid, None)
        g = self._giveups.get(rid)
        if g is not None:
            if g.get("admitted"):
                g = None             # an earlier admission's give-up (the request was preempted since)
            else:
                g["admitted"] = True
                if g.get("settled"):
                    del self._giveups[rid]
        p = STATE.params()
        landed = None
        if td is not None:
            now = time.monotonic()
            # a restore from disk costs the promotion: first deferral to the first lookup that could load (the wait
            # for admission after that is the queue every request pays, priced by the rule as Q); after a give-up,
            # the wait until it gave up plus the recompute. Also record defer_x (expected transfer seconds of the
            # promoted tokens) so layby.rule scales disk restore proportionally to session size N and on_disk f0.
            defer_w = (tr if tr is not None else now) - td + (g["x"] if g is not None else 0.0)
            promo_toks = g["n"] if g is not None else max(num_external_tokens - dr, 0)
            defer_x = promo_toks * p.bytes_per_token / (p.disk_gbps * 1e9) if (promo_toks > 0 and p.disk_gbps > 0) else 0.0
            kw = dict(n_defer=1.0, defer=defer_w)
            if defer_x > 0.0:
                kw.update(n_defer_x=1.0, defer_x=defer_x)
            STATE.live.add(now, **kw)
            if g is None:
                landed = dict(wait=(tr if tr is not None else now) - td)
                if est is not None:
                    landed.update(est0=est[1], link0=est[2], est_wait=(tr if tr is not None else now) - est[0])
        elif pf and num_external_tokens > 0:
            # a restore from disk its prefetch finished during the queue wait: it cost the admission nothing
            promo_toks = min(pf * STATE.chunk_tokens, num_external_tokens)
            defer_x = promo_toks * p.bytes_per_token / (p.disk_gbps * 1e9) if (promo_toks > 0 and p.disk_gbps > 0) else 0.0
            kw = dict(n_defer=1.0, defer=0.0)
            if defer_x > 0.0:
                kw.update(n_defer_x=1.0, defer_x=defer_x)
            STATE.live.add(time.monotonic(), **kw)
        if t is not None:
            now = time.monotonic()
            STATE.live.add(now, n_adm=1.0, qwait=now - t)
            self._account_misses(request, num_external_tokens, now,
                                 request.num_computed_tokens if local is None else max(local, request.num_computed_tokens),
                                 g, qwait=now - t, defer=None if td is None else now - td,
                                 promo=None if td is None or tr is None else tr - td, landed=landed)

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        out = super().update_state_after_alloc(request, blocks, num_external_tokens)
        _safe("admission", self._admit, request, num_external_tokens)
        return out

    def _account_misses(self, request, ext: int, now: float, local: int, giveup=None, qwait=None, defer=None,
                        promo=None, landed=None) -> None:
        """The latency this admission loses to KV the GPU and the CPU tier evicted (and how long
        ago): the load of what comes from outside the GPU, at the CPU link's measured speed and wait,
        and the recompute of what comes from nowhere, at the measured prefill speed; each counted with
        what it delays (link busy fraction, requests in prefill), as in the simulator."""
        bt, ct = STATE.block_tokens, STATE.chunk_tokens
        prompt = request.num_prompt_tokens
        rec = max(prompt - local - ext, 0)
        params = getattr(request, "kv_transfer_params", None) or {}
        row = dict(t=round(now, 3), sess=session_of(params.get("park_key")), P=prompt, local=local, ext=ext, rec=rec)
        if qwait is not None:
            row["qwait"] = round(qwait, 3)          # add to admission
        if defer is not None:
            row["defer"] = round(defer, 3)          # first deferral (a promotion from disk) to admission
        if promo is not None:
            row["promo"] = round(promo, 3)          # first deferral to the promotion landing
        if giveup is not None:
            row.update(giveup=True, wait=round(giveup["waited"], 3), cap=round(giveup["cap"], 3), ready=giveup["ready"],
                       branch=giveup.get("branch"),
                       rem=None if giveup.get("rem") is None else round(giveup["rem"], 3),
                       est0=None if giveup.get("est0") is None else round(giveup["est0"], 3),
                       link0=None if giveup.get("link0") is None else round(giveup["link0"], 3))
        elif landed is not None:
            row.update({k: round(v, 3) for k, v in landed.items() if v is not None})
        STATE.rlog.append(row)
        del STATE.rlog[:-20000]
        if prompt - local <= 0:
            return
        ht = STATE.hash_tokens
        hashes = request.block_hashes[local // ht: prompt // ht]
        tg = [STATE.g_evicted[h] for h in hashes if h in STATE.g_evicted]
        p = STATE.params()
        L = STATE.live
        n_cpu = L.rate("n_cpu", now)
        wait_cpu = L.rate("wait_cpu", now) / n_cpu if n_cpu > 0 else 0.0
        u = min(L.rate("busy_cpu", now), 1.0)
        npf = L.rate("pf", now)
        load_t = ext * p.bytes_per_token / (p.cpu_gbps * 1e9) + (wait_cpu if ext else 0.0)
        rec_t = rec * p.f0
        full = load_t * (1 + u) + rec_t * (1 + npf)
        if tg:
            L.miss(now, "g", now - max(tg), full)
            L.add(now, miss_g=full)
        pol = STATE.policy
        req_status = self.connector_scheduler._req_status.get(request.request_id)
        if pol is not None and req_status is not None and rec > 0:
            keys = [k for g, gs in zip(self.connector_scheduler.config.kv_group_configs, req_status.group_states)
                    for k in gs.offload_keys[(local + ext) // g.tokens_per_chunk: prompt // g.tokens_per_chunk]]
            ev = [k for k in keys if k in pol.c_evicted]
            kept = [k for k in ev if k not in pol.c_first]
            if kept:
                # loss beyond what loading the same tokens from a CPU copy would have cost, for the share of the
                # missing chunks the tier evicted under pressure; chunks the rule placed first out (park, drop)
                # were lost by that decision, which prices them itself, so they do not raise the holding price
                lost_c = max(full - rec * p.bytes_per_token / (p.cpu_gbps * 1e9), 0.0) * len(kept) / len(ev)
                L.miss(now, "c", now - max(pol.c_evicted[k] for k in kept), lost_c)
                L.add(now, miss_c=lost_c)

    # --- steps and the CPU link --------------------------------------------------------
    def build_connector_meta(self, scheduler_output):
        now = time.monotonic()
        _safe("idle", STATE.flush_idle, now)
        _safe("hints", self._drain_hints, now)
        _safe("giveups", self._settle_giveups, now)
        _safe("ready", self._check_ready, now)
        _safe("step", self._step, scheduler_output, now)
        for rid in getattr(scheduler_output, "preempted_req_ids", None) or ():
            # back in the queue at its head (vLLM resumes preempted requests first): ranks before every arrival
            self.__dict__.get("_running", {}).pop(rid, None)
            _safe("hold", self._hold, rid, -now)
        _safe("prefetch", self._prefetch)
        if getattr(self, "_gpu_demote", None):
            _safe("gpu order", self._apply_gpu_order)
        meta = super().build_connector_meta(scheduler_output)
        _safe("protect running", self._protect_running, scheduler_output, now)
        for job_id, job in meta.load_jobs.items():
            self._loads[job_id] = (now, len(job.src_spec.block_ids))
        if now - self._last_prune > 60.0:
            _safe("prune", self._prune, now)
        return meta

    def _protect_running(self, scheduler_output, now: float) -> None:
        """Keep newly stored CPU chunks of currently running requests protected below idle chunks
        until finish (still evictable if needed, zero pinning): prevents vLLM 0.31 from evicting a
        running request's early prefill chunks during its own decode before the cost rule runs."""
        running = self.__dict__.get("_running")
        pol = STATE.policy
        if not running or pol is None or not hasattr(pol, "mark_pending_write"):
            return
        cs = self.connector_scheduler
        for rid in getattr(scheduler_output, "num_scheduled_tokens", None) or ():
            prev = running.get(rid)
            if prev is None:
                continue
            rs = cs._req_status.get(rid)
            if rs is None:
                continue
            new_keys = []
            for gi, gs in enumerate(rs.group_states):
                ks = gs.offload_keys
                n = len(ks)
                old = prev[gi] if gi < len(prev) else 0
                if n > old:
                    new_keys.extend(ks[old:n])
                    if gi < len(prev):
                        prev[gi] = n
                    else:
                        prev.append(n)
            if new_keys:
                pol.mark_pending_write(new_keys, now, ttl=120.0, eta=None)

    def _step(self, scheduler_output, now):
        it = compute_iteration_details(scheduler_output)
        if self._last_step is not None:
            t, pt, work = self._last_step
            dt = now - t
            if work:
                STATE.speeds.step(now, dt, pt)
            STATE.live.add(now, nw=len(self._waiting) * dt, pf=(len(self._waiting) + it.num_ctx_requests) * dt)
        work = scheduler_output.total_num_scheduled_tokens > 0
        self._last_step = (now, it.num_ctx_tokens, work)

    def update_connector_output(self, connector_output):
        _safe("cpu link", self._links, connector_output)
        super().update_connector_output(connector_output)

    def _links(self, connector_output):
        now = time.monotonic()
        wm = connector_output.kv_connector_worker_meta
        if wm is not None and hasattr(wm, "transfer_stats"):
            ld = wm.transfer_stats.load
            if not ld.is_empty() and ld.time > 0:
                STATE.speeds.link(now, "cpu", ld.bytes, ld.time)
            gbps = STATE.speeds.gbps("cpu")
            chunk_bytes = STATE.bytes_per_token * STATE.chunk_tokens
            for job_id in getattr(wm, "completed_jobs", {}):
                sub = self._loads.pop(job_id, None)
                if sub is not None:
                    x = sub[1] * chunk_bytes / (gbps * 1e9)
                    STATE.live.add(now, n_cpu=1.0, busy_cpu=x, wait_cpu=max(now - sub[0] - x, 0.0))

    # --- decisions ---------------------------------------------------------------------
    def request_finished(self, request, block_ids):
        self._local.pop(request.request_id, None)
        out = super().request_finished(request, block_ids)
        _safe("finish", self._on_finish, request, block_ids)
        return out

    def request_finished_all_groups(self, request, block_ids):
        self._local.pop(request.request_id, None)
        out = super().request_finished_all_groups(request, block_ids)
        _safe("finish", self._on_finish, request, block_ids)
        return out

    def _on_finish(self, request, block_ids) -> None:
        now = time.monotonic()
        rid = request.request_id
        self._unhold(rid)          # finished or aborted while waiting
        self.__dict__.get("_running", {}).pop(rid, None)
        self._prefetched.pop(rid, None)
        self._waiting.pop(rid, None)
        self._deferred.pop(rid, None)
        self._ready.pop(rid, None)
        self.__dict__.get("_defer_ready", {}).pop(rid, None)
        self.__dict__.get("_est0", {}).pop(rid, None)
        groups = block_ids if block_ids and isinstance(block_ids[0], (list, tuple)) else [block_ids or []]
        ids = groups[0]
        pool = getattr(self, "_pool", None)
        for bid in ids or ():
            if pool is None or pool.blocks[bid].ref_cnt <= 1:     # no other running request holds it
                STATE.g_idle[bid] = now
        req_status = self.connector_scheduler._req_status.get(rid)
        if req_status is None:
            return
        params = request.kv_transfer_params
        if params is None:
            params = {}
            req_status.req_context.kv_transfer_params = params
        key = str(params.get("park_key") or rid)
        # per full-attention KV group keys for exact on_disk token count on hybrid models (Gemma-4, Qwen3.8, GLM)
        g_cfgs = getattr(getattr(self.connector_scheduler, "config", None), "kv_group_configs", ())
        group_keys = [(g.tokens_per_chunk, list(gs.offload_keys))
                      for g, gs in zip(g_cfgs, req_status.group_states)
                      if getattr(g, "sliding_window_size_in_chunks", None) is None]
        if not group_keys and g_cfgs:
            group_keys = [(g.tokens_per_chunk, list(gs.offload_keys))
                          for g, gs in zip(g_cfgs, req_status.group_states)]
        entry = dict(keys=_request_keys(req_status), group_keys=group_keys, tokens=request.num_tokens,
                     params=params, t=now)
        if getattr(self, "_gpu_order", False) and pool is not None:
            entry["gpu"] = [(b, pool.blocks[b].block_hash) for g in groups for b in g
                            if pool.blocks[b].block_hash is not None]
        early = self._early.pop(key, None)
        if params.get("park_surv") is not None or early is not None:
            self._decide(entry, params["park_surv"] if params.get("park_surv") is not None else early[1], now)
        elif self._hints_on:
            self._finished[key] = entry
            self._finished.move_to_end(key)
            # tighten running-request protection to a short ~250 ms pending-hint guard while awaiting /hint
            pol = STATE.policy
            if pol is not None and hasattr(pol, "mark_pending_write"):
                pol.mark_pending_write(entry["keys"], now, ttl=0.25, eta=None)
        else:
            pol = STATE.policy
            if pol is not None and hasattr(pol, "clear_pending_write"):
                pol.clear_pending_write(entry["keys"], restore_lru=True, now=now)

    def _drain_hints(self, now: float) -> None:
        while not STATE.hints.empty():
            key, surv = STATE.hints.get_nowait()
            STATE.counts["hints"] += 1
            entry = self._finished.pop(key, None)
            if entry is None:
                STATE.counts["no_entry"] += 1      # not finished yet (or unknown): keep for the finish
                self._early[key] = (now, surv)
                continue
            STATE.counts["late"] += 1
            self._decide(entry, surv, now)

    def _decide(self, entry, surv, now: float) -> None:
        STATE.flush_idle(now)
        pol, fs = STATE.policy, STATE.fs_tier
        keys = entry["keys"]
        S = interp_surv(np.nan_to_num(np.asarray(surv, float), nan=0.5), TG)
        if fs is None:
            on_disk = 0
        elif entry.get("group_keys"):
            on_disk = min(fs.count_on_disk(gks) * tc for tc, gks in entry["group_keys"])
        else:
            on_disk = fs.count_on_disk(keys) * STATE.chunk_tokens
        idle_c = (np.zeros(0), np.zeros(0))
        if pol is not None and pol.idle_since:
            t = np.fromiter(pol.idle_since.values(), float, len(pol.idle_since))
            idle_c = (now - t, np.full(len(t), float(STATE.chunk_tokens)))
        d = decide(S, entry["tokens"], on_disk, STATE.idle_g(now), idle_c, STATE.live, STATE.params(), now,
                   write=STATE.write_mode)
        sess = session_of(entry["params"].get("park_key"))
        if STATE.write_mode == "through" and sess is not None:
            (STATE.nowrite.discard if d["disk"] else STATE.nowrite.add)(sess)
        if STATE.write_mode == "demand" and sess is not None:
            (STATE.thru.add if d["disk"] else STATE.thru.discard)(sess)
            if d.get("spill"):
                for k in keys:
                    STATE.spill[k] = sess
            else:
                for k in keys:
                    STATE.spill.pop(k, None)
        if fs is None and d["disk"]:
            d["opt"] = "none" if d["opt"] == "ins" else "drop"     # no disk tier: the same CPU placement
            d["disk"] = False
        STATE.counts[d["opt"]] += 1
        entry["params"]["park_eta"] = d["eta"]
        if d["disk"] and fs is not None:
            need = fs.queue_write(keys)
            if need is None:
                need = keys
            if pol is not None:
                if hasattr(pol, "clear_pending_write"):
                    need_set = set(need)
                    done_keys = [k for k in keys if k not in need_set]
                    if done_keys:
                        pol.clear_pending_write(done_keys)
                if need and hasattr(pol, "mark_pending_write"):
                    pol.mark_pending_write(need, now, eta=d["eta"])
        elif pol is not None and hasattr(pol, "clear_pending_write"):
            pol.clear_pending_write(keys)
        if pol is not None:
            pol.apply(keys, d["eta"], now)
        if getattr(self, "_gpu_order", False) and d["opt"] in ("park", "drop") and entry.get("gpu"):
            self._gpu_demote.append(entry["gpu"])
        STATE.last_decisions.append(dict(opt=d["opt"], eta=round(d["eta"], 2), tokens=entry["tokens"],
                                         Q=round(d["Q"], 3), price_c=round(d["price_c"], 4)))
        del STATE.last_decisions[:-100]
        L = STATE.live
        STATE.dlog.append(dict(t=round(now, 3), sess=sess, N=entry["tokens"], opt=d["opt"], price_c=round(d["price_c"], 5),
                               tot={k: (round(x, 4), round(y, 4)) for k, (x, y) in d["tot"].items()},
                               p60=round(1.0 - float(np.interp(60.0, TG, S)), 4),
                               p300=round(1.0 - float(np.interp(300.0, TG, S)), 4),
                               med=float(TG[np.argmax(S <= 0.5)]) if (S <= 0.5).any() else None,
                               ne_c=round(L.rate("ne_c", now), 4), w_c=round(L.rate("w_c", now), 3),
                               npf=round(L.rate("pf", now), 3)))
        del STATE.dlog[:-20000]

    def _apply_gpu_order(self) -> None:
        """Move each queued session's free cached blocks to the front of the free queue, tail first."""
        pool = self._pool
        q = pool.free_block_queue
        pending, self._gpu_demote = self._gpu_demote, []
        for blocks in pending:
            move = []
            for bid, h in reversed(blocks):                 # the session's tail goes first
                b = pool.blocks[bid]
                if b.ref_cnt == 0 and not b.is_null and b.block_hash == h and b.prev_free_block is not None:
                    move.append(b)
            for b in move:
                q.remove(b)
            q.prepend_n(move)
            STATE.counts["gpu_demoted"] += len(move)

    def _prune(self, now: float) -> None:
        self._last_prune = now
        for d in (STATE.g_evicted, getattr(STATE.policy, "c_evicted", {})):
            old = [k for k, t in d.items() if now - t > HORIZON]
            for k in old:
                del d[k]
        if STATE.policy is not None:
            STATE.policy.c_first.intersection_update(STATE.policy.c_evicted)
        while self._finished:
            k, e = next(iter(self._finished.items()))
            if now - e["t"] <= HORIZON:
                break
            del self._finished[k]
        self._early = {k: v for k, v in self._early.items() if now - v[0] <= HORIZON}
        self._loads = {j: v for j, v in self._loads.items() if now - v[0] <= HORIZON}
        self._giveups = {k: g for k, g in self._giveups.items() if now - g["t"] <= HORIZON}
        if "_est0" in self.__dict__:
            self._est0 = {k: v for k, v in self._est0.items() if now - v[0] <= HORIZON}
