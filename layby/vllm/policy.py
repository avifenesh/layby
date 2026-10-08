"""Park CPU-tier eviction policy for vLLM 0.30's OffloadingConnector, with residency telemetry.

Load with kv_connector_extra_config {"eviction_policy": "ParkCachePolicy", "cache_policy_module_path":
"layby.vllm.policy"} (see park/vllm/__init__.py). Ported from tools/park_cpu_policy.py (phase D), plus:
  - every request's chunks are tracked while it is live, hinted or not, so a decision that arrives
    after the request finished (layby.vllm.connector) can still place them (apply());
  - idle periods feed layby.vllm.state.STATE.live: a chunk goes idle when its last live request
    finishes; the period ends in eviction (evict) or reuse (touch by a new request).

Ordering, as before: a hinted chunk has a predicted next use (now + park_eta, never for park_eta < 0)
and lives in a heap; an unhinted chunk is ordered LRU and its next use is estimated the LRU way, now +
(time since last use). Eviction takes the chunk predicted to be needed last. A hint GRACE seconds past
due without a touch returns the chunk to LRU order.

Write-on-evict (layby.vllm.demote, park_write "demand"): with a demoter installed, a victim the rule wants on
disk is kept resident and pinned by its disk write while the next victim frees the slot; once written it goes
first at the next eviction. Without a demoter, evict() is unchanged.
"""
import heapq
import itertools
import time
from collections import OrderedDict
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus

from layby.vllm.state import STATE

NEVER = 1e18

try:   # vLLM >= 0.31: one head-to-tail order across KV groups
    from vllm.v1.kv_offload.cpu.policies.base import order_request_keys as _order_keys
except ImportError:
    def _order_keys(key_groups, req_context):
        return [k for g in key_groups for k in g]


def _stock_lru_on_evictable() -> bool:
    """Whether this vLLM's LRUCachePolicy ranks a chunk by when it became evictable (0.30) or by when it was
    inserted (0.31): probed on the stock policy, so an unhinted ParkCachePolicy evicts in the stock order."""
    try:
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
        pol = LRUCachePolicy(4)
        a, b = ChunkStatus.__new__(ChunkStatus), ChunkStatus.__new__(ChunkStatus)
        a.ref_cnt, b.ref_cnt = -1, 0
        pol.insert("a", a); pol.insert("b", b)
        a.ref_cnt = 0; pol.mark_evictable("a")
        return [k for k, _ in pol.evict(1, set())] == ["b"]
    except Exception:  # noqa: BLE001
        return False


_LRU_ON_EVICTABLE = _stock_lru_on_evictable()
GRACE = 2.0


class ParkCachePolicy(CachePolicy):
    def __init__(self, cache_capacity: int):
        super().__init__(cache_capacity)
        self.chunks: dict[OffloadKey, ChunkStatus] = {}
        self.evictable: set[OffloadKey] = set()
        self.hinted: dict[OffloadKey, float] = {}          # key -> predicted next use
        self.heap: list[tuple[float, int, OffloadKey]] = []  # (-next_use, seq, key), lazy
        # key -> seq of its one current heap entry. An entry is current by its seq, not by its time: first out is
        # the constant NEVER, so a chunk hinted first out again (a session's next turn) would revive its old entry
        # and evict() would return the chunk twice (its slot freed twice, the manager's evictable count off by one)
        self.hseq: dict[OffloadKey, int] = {}
        self.expiry: list[tuple[float, int, OffloadKey]] = []  # (next_use, seq, key), lazy min-heap
        self.lru: "OrderedDict[OffloadKey, float]" = OrderedDict()  # unhinted: key -> last use
        self.seq = itertools.count()
        self.live: dict[str, tuple[float | None, set[OffloadKey]]] = {}  # req id -> (park_eta, touched keys)
        self.idle_since: dict[OffloadKey, float] = {}   # idle chunk -> when its last live request finished
        self.c_evicted: dict[OffloadKey, float] = {}    # chunk -> when it was evicted while idle
        # write-on-evict (layby.vllm.demote): victims kept resident until their disk write completes
        self.demoter = None
        self.demoting: set[OffloadKey] = set()          # chosen as victims, being written, not evictable
        self.pending_demote: list[OffloadKey] = []      # demoted by the current evict, write not yet submitted
        STATE.policy = self

    # --- bookkeeping -------------------------------------------------------------------
    def _set_hint(self, key: OffloadKey, t: float) -> None:
        self.lru.pop(key, None)
        self.hinted[key] = t
        s = next(self.seq)
        self.hseq[key] = s
        heapq.heappush(self.heap, (-t, s, key))
        if t < NEVER:
            heapq.heappush(self.expiry, (t, s, key))

    def _set_lru(self, key: OffloadKey, now: float) -> None:
        self.hinted.pop(key, None)
        self.hseq.pop(key, None)
        self.lru[key] = now
        self.lru.move_to_end(key)

    def _expire(self, now: float) -> None:
        """Move hinted chunks whose predicted use is GRACE past due back to LRU order."""
        moved = []
        while self.expiry and self.expiry[0][0] < now - GRACE:
            t, s, key = heapq.heappop(self.expiry)
            if self.hseq.get(key) == s:
                del self.hinted[key], self.hseq[key]
                moved.append((t, key))
        if moved:
            # keep the LRU list ordered by last use: rebuild with the expired chunks in place
            items = list(self.lru.items()) + [(k, t) for t, k in moved]
            items.sort(key=lambda kv: kv[1])
            self.lru = OrderedDict(items)

    def _in_use(self, key: OffloadKey) -> bool:
        return any(key in ks for _, ks in self.live.values())

    def _drop(self, key: OffloadKey) -> None:
        self.demoting.discard(key)
        self.idle_since.pop(key, None)
        self.chunks.pop(key, None)
        self.evictable.discard(key)
        self.hinted.pop(key, None)
        self.hseq.pop(key, None)
        self.lru.pop(key, None)

    # --- CachePolicy -----------------------------------------------------------------------
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        return self.chunks.get(key)

    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        self.chunks[key] = chunk
        if chunk.ref_cnt == 0:
            self.evictable.add(key)
        now = time.monotonic()
        self._set_lru(key, now)
        if not self._in_use(key):
            self.idle_since[key] = now      # e.g. the final store of a request that already finished
        self.c_evicted.pop(key, None)

    def remove(self, key: OffloadKey) -> None:
        self._drop(key)

    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        now = time.monotonic()
        params = (req_context.kv_transfer_params if req_context is not None else None) or {}
        eta = params.get("park_eta")
        keys = list(keys)
        if req_context is not None:
            # in use until the request finishes; remember the keys for the hint
            entry = self.live.setdefault(req_context.req_id, (None if eta is None else float(eta), set()))
            entry[1].update(keys)
        self._refresh(keys, now)

    def _refresh(self, keys: list, now: float) -> None:
        # last key first, so a request's prefix ends most recent: vLLM reuses a prefix only as a contiguous run
        # from the first chunk, so the tail must go before the prefix (as vLLM's own LRU policy does)
        for key in reversed(keys):
            if key in self.chunks:
                self.demoting.discard(key)       # a request came back during its disk write: keep it as a live chunk
                t = self.idle_since.pop(key, None)
                if t is not None:
                    STATE.ended[("c", False)].append(now - t)              # reuse ends the idle period
                self._set_lru(key, now)

    def on_request_finished(self, key_groups, insertion_only_keys, reused_keys, req_context: ReqContext) -> None:
        """vLLM >= 0.31 moves a request's recency update to its finish (the base hook calls touch(), which would
        mark the finished request live again, so it would never go idle and its hint would be overwritten).
        Here: recency first, then the finish (idle periods and the hint)."""
        keys = _order_keys(key_groups, req_context)
        self._refresh(keys, time.monotonic())
        params = req_context.kv_transfer_params or {}
        eta = params.get("park_eta")
        entry = self.live.setdefault(req_context.req_id, (None if eta is None else float(eta), set()))
        entry[1].update(keys)
        self.request_finished(req_context)

    def request_finished(self, req_context: ReqContext) -> None:
        entry = self.live.pop(req_context.req_id, None)
        if entry is None:
            return
        eta, keys = entry
        params = req_context.kv_transfer_params or {}
        if params.get("park_eta") is not None:      # set by the connector at finish
            eta = float(params["park_eta"])
        now = time.monotonic()
        idle = [k for k in keys if k in self.chunks and not self._in_use(k)]
        for key in idle:
            self.idle_since[key] = now
        if eta is not None:
            self.apply(idle, eta, now)

    def apply(self, keys, eta: float, now: float | None = None) -> int:
        """Place idle chunks by a predicted next use: now + eta, or first out for eta < 0."""
        now = time.monotonic() if now is None else now
        t = NEVER if eta < 0 else now + eta
        n = 0
        for key in reversed(list(keys)):    # equal predictions evict in hint order: the tail first, the prefix last
            if key in self.chunks and not self._in_use(key):
                self._set_hint(key, t)
                n += 1
        return n

    def clear(self) -> None:
        self.chunks.clear(); self.evictable.clear(); self.hinted.clear(); self.heap.clear(); self.lru.clear()
        self.hseq.clear()
        self.expiry.clear(); self.live.clear(); self.idle_since.clear(); self.c_evicted.clear()
        self.demoting.clear(); self.pending_demote.clear()

    def _top_hinted(self, protected: set[OffloadKey], skipped: list):
        while self.heap:
            neg_t, s, key = self.heap[0]
            if self.hseq.get(key) != s:
                heapq.heappop(self.heap)  # stale
                continue
            if key not in self.evictable or key in protected:
                skipped.append(heapq.heappop(self.heap))
                continue
            return key, -neg_t
        return None, None

    def evict(self, n: int, protected: set[OffloadKey]) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        if n == 0:
            return []
        if len(self.evictable) < n:
            return None
        now = time.monotonic()
        self._expire(now)
        out, skipped = [], []
        lru_iter = iter(self.lru.items())          # not modified until the loop ends (_drop runs after it)
        lru_next = next(lru_iter, None)
        taken = set()
        dem = self.demoter
        room = dem.budget(n, len(self.evictable)) if dem is not None else 0
        demoted = []        # victims kept resident until written to disk; the next victim frees a slot instead
        while len(out) < n:
            while lru_next is not None and (lru_next[0] not in self.evictable or lru_next[0] in protected
                                            or lru_next[0] in taken):
                lru_next = next(lru_iter, None)
            hk, ht = self._top_hinted(protected, skipped)
            lru_est = now + (now - lru_next[1]) if lru_next is not None else None
            if hk is None and lru_next is None:
                break
            if hk is not None and (lru_est is None or ht >= lru_est):
                heapq.heappop(self.heap)
                key = hk
            else:
                key = lru_next[0]
                lru_next = next(lru_iter, None)
            taken.add(key)
            if len(demoted) < room and dem.wants(key):
                demoted.append(key)
            else:
                out.append(key)
        for e in skipped:
            heapq.heappush(self.heap, e)
        if len(out) < n and demoted:
            # no other slot to free: drop demotions (the policy's first choices first), as the stock policy would
            k = min(n - len(out), len(demoted))
            out += demoted[:k]
            del demoted[:k]
            dem.dropped += k
        if len(out) < n:
            for key in out:  # atomic: restore the hinted entries popped above
                if key in self.hinted:
                    self._set_hint(key, self.hinted[key])
            return None
        res = [(key, self.chunks[key]) for key in out]
        STATE.note_burst(now, n)
        for key in out:
            t = self.idle_since.get(key)
            STATE.ended[("c", True)].append(now - t if t is not None else 0.0)
            STATE.spill.pop(key, None)
            self.c_evicted[key] = now
            self._drop(key)
        for key in demoted:
            # stays in the CPU tier (a lookup still hits it) but cannot be evicted until its write completes;
            # demoted once: if the write fails, the chunk goes at the next eviction
            self.evictable.discard(key)
            self.demoting.add(key)
            self.pending_demote.append(key)
            STATE.spill.pop(key, None)
        if len(self.heap) > 4 * len(self.hinted) + 1024:  # compact stale heap entries
            self.heap = [(-t, next(self.seq), k) for k, t in self.hinted.items()]
            self.hseq = {k: s for _, s, k in self.heap}
            heapq.heapify(self.heap)
            self.expiry = [(t, s, k) for (nt, s, k) in self.heap for t in (-nt,) if t < NEVER]
            heapq.heapify(self.expiry)
        return res

    def mark_evictable(self, key: OffloadKey) -> None:
        if key in self.chunks:
            self.evictable.add(key)
            if key in self.demoting:
                # its disk write ended: it was chosen as a victim, so it goes first at the next eviction
                self.demoting.discard(key)
                self._set_hint(key, NEVER)
                return
            if _LRU_ON_EVICTABLE and key not in self.hinted:
                # as this vLLM's own LRU policy: an unhinted chunk ranks by when it became evictable (its store or
                # its last user finished), not by when it was inserted
                self._set_lru(key, time.monotonic())

    def mark_non_evictable(self, key: OffloadKey) -> None:
        self.evictable.discard(key)

    def release_demotions(self, keys) -> None:
        """Demotions whose write could not be submitted: evictable again, first out."""
        for key in keys:
            if key in self.demoting and key in self.chunks and self.chunks[key].ref_cnt == 0:
                self.demoting.discard(key)
                self.evictable.add(key)
                self._set_hint(key, NEVER)


_base_request_finished = CPUOffloadingManager.on_request_finished


def _request_finished(self, req_context: ReqContext) -> None:
    # the base first: on vLLM >= 0.31 it applies the request's recency (ParkCachePolicy.on_request_finished),
    # which must come before the hint; on 0.30 it does nothing for the policy
    _base_request_finished(self, req_context)
    policy = getattr(self, "_policy", None)
    if isinstance(policy, ParkCachePolicy):
        policy.request_finished(req_context)


CPUOffloadingManager.on_request_finished = _request_finished
