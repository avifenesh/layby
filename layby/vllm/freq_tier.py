"""T2: frequency-gated disk writes on vLLM's native fs tier.

Arm T writes every chunk the CPU tier stores through to vLLM's FileSystemTierManager. Arm T2 runs the same tier
class, the same I/O code (batch_store_block: temp file, os.replace, O_DIRECT where the filesystem takes it), the
same file layout and the same lookups; only the admission differs. A chunk is written on its n-th sighting, with
counts that decay over time, as SGLang HiCache write_through_selective and NVIDIA Dynamo KVBM's FrequencyFilter do.
CPU tier: vLLM's own (stock connector, LRU or ARC as T); Layby's connector and policy are not involved.

Config (secondary_tiers entry, beside the stock OffloadingConnector):
  {"type": "FreqFsTier", "module_path": "layby.vllm.freq_tier", "root_dir": ..., "n_read_threads": 16,
   "n_write_threads": 16, "min_hits": 2, "decay_s": 600, "count": "request"}

min_hits (n)  the sighting on which a chunk is written: 2 (the default of both systems) or 3.
decay_s       every decay_s seconds every count drops by one and counts at 0 are forgotten (KVBM's flush interval,
              600 s there). 0 turns decay off (SGLang, whose hit counts never decay).
count         what one sighting is:
              "request" (default): one request whose KV covers the chunk. The tier asks vLLM for request-level
              offloading, so every request offers it all its chunks, also those the CPU tier already holds; this is
              SGLang's hit count (one per insert of a request's sequence into the radix tree), the system the plan
              validates T2 against (same write count within 10% on the pilot workload).
              "store": one store of the chunk into the CPU tier (vLLM's cascade of a newly stored chunk); this is
              KVBM's sighting (one per host-to-disk offload request, which KVBM issues when a block enters its host
              pool). A chunk that stays in the CPU tier is sighted once, so it reaches disk only after the CPU tier
              evicted it and a later request stored it again.
max_entries   the count map's size bound (KVBM: 1,000,000); above it, decay steps run until it fits and the decay
              clock restarts, as KVBM does.

The count follows KVBM's FrequencyFilter exactly: a first sighting sets the count to 1, every later one doubles it,
and the chunk is written once the count reaches 2^(n-1). So the n-th sighting within one decay step writes it, and
a decay step costs one sighting only while the count is small (a doubled count survives more steps), as in KVBM.
For n = 2 the doubling and SGLang's +1 count agree (1 then 2).

A chunk already on disk, or being written, is skipped without a sighting, as KVBM skips a block already in its
target pool. A job whose chunks are all held back completes at once without I/O through the tier's own thread pool
(as ParkFsTier does), so the tiering manager's bookkeeping is the same as for a written job.

Deviations from the two systems (also in the paper's baseline appendix):
  - SGLang gates host (L2) writes with the same count, and L3 writes follow host writes; here only the disk tier is
    gated, the CPU tier keeps vLLM's write-through of every chunk (as T).
  - SGLang counts per radix node (a run of tokens) and at insert time (a request's finish, or a chunked prefill);
    here per offload chunk, at the moment the request offers the chunk for storing (its prefill or decode step), so
    a written prefix reaches disk earlier within the same request.
  - KVBM decays on a timer task; here the decay steps that fell due are applied at the next sighting (the counts
    any sighting reads are the same). KVBM's size-bound prune runs asynchronously after the notify; here at once.
  - vLLM's store back-pressure detector, if configured, would see the no-I/O completions as fast writes. It is off
    by default and must stay off for T and T2 alike.
"""
import dataclasses
import time

from vllm.v1.kv_offload.base import OffloadPolicy
from vllm.v1.kv_offload.tiering.base import RequestOffloadingContext
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)   # under vLLM's handler: a module name outside "vllm" prints nothing

_CAP = 1 << 62      # KVBM's i64 saturating multiply, kept in range
STATS_EVERY_S = 60.0   # the counts go to the server log this often (chunks admitted to disk: the T2 write count)


class FrequencyFilter:
    """KVBM's FrequencyFilter (lib/llm/src/block_manager/offload/filter.rs) with the threshold given as a sighting
    number n: should_offload(key) is True from the n-th sighting within the decay window."""

    def __init__(self, min_hits=2, decay_s=600.0, max_entries=1_000_000, clock=time.monotonic):
        if int(min_hits) < 1:
            raise ValueError("min_hits must be at least 1")
        self.min_hits = int(min_hits)
        self.threshold = 1 << (self.min_hits - 1)
        self.decay_s = float(decay_s or 0.0)
        self.max_entries = int(max_entries)
        self.clock = clock
        self.counts: dict = {}
        self._tick = clock()             # start of the current decay interval

    def _decrement_and_prune(self, k=1):
        self.counts = {key: c - k for key, c in self.counts.items() if c > k}

    def decay(self, now=None):
        """Apply the decay steps that fell due by now: each step lowers every count by one."""
        if self.decay_s <= 0:
            return
        now = self.clock() if now is None else now
        k = int((now - self._tick) // self.decay_s)
        if k > 0:
            self._decrement_and_prune(k)
            self._tick += k * self.decay_s

    def should_offload(self, key, now=None) -> bool:
        now = self.clock() if now is None else now
        self.decay(now)
        c = self.counts.get(key)
        c = 1 if c is None else min(c * 2, _CAP)
        self.counts[key] = c
        if len(self.counts) > self.max_entries:
            # the answer is the count before the prune, as in KVBM (its prune runs after should_offload returns)
            while len(self.counts) > self.max_entries:
                self._decrement_and_prune()
            self._tick = now             # KVBM resets its flush interval after an oversize prune
        return c >= self.threshold


class FreqFsTier(FileSystemTierManager):
    """vLLM's fs tier with a frequency filter on what it writes (module docstring)."""

    def __init__(self, *args, min_hits=2, decay_s=600.0, count="request", max_entries=1_000_000, **kwargs):
        super().__init__(*args, **kwargs)
        if count not in ("request", "store"):
            raise ValueError(f"FreqFsTier count must be 'request' or 'store', not {count!r}")
        self.filter = FrequencyFilter(min_hits, decay_s, max_entries)
        self.count_mode = count
        self._init_index()
        logger.info("FreqFsTier: write on sighting %d (count %s), decay every %s s", self.filter.min_hits, count,
                    self.filter.decay_s or "never")

    def _init_index(self):
        self._written = set()            # chunks this tier wrote (job completed with success)
        self._inflight = set()           # chunks in a write job
        self._writing = {}               # store job id -> its keys
        self.stats = dict(sighted=0, admitted=0, held=0, skipped_on_disk=0, jobs_written=0, jobs_noio=0)
        self._stats_t = time.monotonic()   # last stats log line (the pilot's write-count audit reads them)

    def on_new_request(self, req_context) -> RequestOffloadingContext:
        if self.count_mode == "request":
            return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        return RequestOffloadingContext()

    def submit_store(self, job_metadata) -> None:
        now = time.monotonic()
        if now - self._stats_t >= STATS_EVERY_S:
            self._stats_t = now
            logger.info("FreqFsTier stats: %s", self.stats)
        keys = list(job_metadata.keys)
        keep = []
        for i, k in enumerate(keys):
            if k in self._written or k in self._inflight:
                self.stats["skipped_on_disk"] += 1
                continue
            self.stats["sighted"] += 1
            if self.filter.should_offload(k, now):
                keep.append(i)
        self.stats["admitted"] += len(keep)
        self.stats["held"] += len(keys) - len(keep)
        if not keep:
            self.stats["jobs_noio"] += 1
            self._pool.enqueue_store(job_metadata.job_id, 1, [lambda: None])
            return
        if len(keep) < len(keys):
            job_metadata = dataclasses.replace(job_metadata, keys=[keys[i] for i in keep],
                                               chunk_ids=job_metadata.chunk_ids[keep])
        self.stats["jobs_written"] += 1
        self._writing[job_metadata.job_id] = list(job_metadata.keys)
        self._inflight.update(job_metadata.keys)
        super().submit_store(job_metadata)

    def get_finished_jobs(self):
        results = list(super().get_finished_jobs())
        for r in results:
            keys = self._writing.pop(r.job_id, None)
            if keys is not None:
                self._inflight.difference_update(keys)
                if r.success:
                    self._written.update(keys)
        return results
