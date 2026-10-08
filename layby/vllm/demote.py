"""Write-on-evict (demotion) for the KD design, without patching vLLM.

vLLM's CPU tier frees an evicted chunk's slot inside prepare_store and hands it to the new store in the same
call, so there is no point at which a victim could still be copied out. Here the eviction policy does it:

  - ParkCachePolicy.evict, when a demoter is installed, keeps a victim the demoter wants on disk (spill-marked,
    no disk copy) resident instead of freeing it, and takes the next chunk in eviction order in its place. The
    caller's allocation is filled from other evictable chunks. If there are not enough of them, demotions turn
    back into plain drops, so a store succeeds exactly when the stock policy would let it.
  - Right after the primary tier's prepare_store returns, the wrapper installed here submits one disk store
    job for the demoted chunks (create_store_job pins them through the normal cascade-read ref count).
  - When the write completes, complete_read drops the ref count to 0 and the policy marks the chunk first out:
    the next eviction frees its slot. A request that returns while the write runs still hits it in the CPU tier.

Only victims are pinned, only while their own write runs, and only as many as leave the CPU tier at least
`reserve` evictable chunks: the larger of this eviction, the largest recent eviction burst, and the chunks the
tier evicts while a demoted slot is held (eviction rate x the measured disk job time).
"""
import itertools
import time

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import ReqContext

from layby.vllm.state import STATE

logger = init_logger(__name__)


class Demoter:
    """Decides which eviction victims are written to the disk tier before their slot is reused."""

    def __init__(self, fs_tier):
        self.fs = fs_tier
        self.demoted = 0          # chunks written on eviction
        self.dropped = 0          # spill-marked victims dropped because no other slot could be freed

    def wants(self, key) -> bool:
        fs = self.fs
        return key in STATE.spill and key not in fs._on_disk and key not in fs._inflight

    def budget(self, n: int, evictable: int) -> int:
        """How many victims this eviction of n chunks may demote, given the evictable chunks before it."""
        now = time.monotonic()
        L = STATE.live
        nd = L.rate("n_disk", now)
        job = L.rate("wait_disk", now) / nd if nd > 0 else 0.0
        chunk_t = STATE.chunk_tokens * STATE.bytes_per_token / (STATE.speeds.gbps("disk") * 1e9)
        hold = job + (len(self.fs._inflight) + 1) * chunk_t          # how long a demoted slot stays pinned
        reserve = max(n, STATE.burst_now(now), L.rate("ne_c", now) * hold)
        cap = int(evictable - n - reserve)
        if cap <= 0:
            return 0
        # the copy must be able to pay (the guard of the write-ahead version, kept for a like-for-like A/B):
        # writing a batch of this call's size must take less than recomputing it, priced as layby.rule does
        s = min(cap, n)
        x = s * STATE.chunk_tokens * STATE.speeds.f0()
        if job + s * chunk_t >= x * (1 + L.rate("pf", now) + L.rate("n_adm", now) * x):
            return 0
        return cap


def install(manager) -> Demoter | None:
    """Turn on write-on-evict for a TieringOffloadingManager whose primary tier runs ParkCachePolicy and whose
    secondary tiers include the ParkFsTier. Returns the demoter, or None if the setup does not fit."""
    from layby.vllm.policy import ParkCachePolicy

    primary = getattr(manager, "primary_tier", None)
    fs = STATE.fs_tier
    pol = getattr(primary, "_policy", None)
    if not isinstance(pol, ParkCachePolicy) or fs is None or fs not in manager.secondary_tiers:
        return None
    tier_idx = manager.secondary_tiers.index(fs)
    dem = Demoter(fs)
    seq = itertools.count()

    def flush():
        keys, pol.pending_demote = pol.pending_demote, []
        try:
            ctx = ReqContext(req_id=f"park-demote-{next(seq)}", kv_transfer_params={"park_disk": True})
            job = manager.create_store_job(keys, ctx, tier_idx)     # pins: ref_cnt 0 -> 1 until the write ends
            fs.submit_store(job)
            dem.demoted += len(keys)
            STATE.counts["spill"] += len(keys)
        except Exception:   # noqa: BLE001
            logger.exception("park: demotion of %d chunks failed; dropping them instead", len(keys))
            pol.release_demotions(keys)

    orig = primary.prepare_store     # bound method; prepare_write is an alias of it bound at __init__

    def prepare_store(keys, req_context):
        out = orig(keys, req_context)
        if pol.pending_demote:
            flush()
        return out

    # promotions allocate through prepare_write and can evict too
    primary.prepare_store = primary.prepare_write = prepare_store
    pol.demoter = dem
    STATE.demoter = dem
    return dem
