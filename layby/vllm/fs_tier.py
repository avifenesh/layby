"""Park disk tier for vLLM 0.30: a secondary tier that holds only what the cost rule places on disk.

Grew from an earlier disk tier of the training repo, plus:
  - late writes: when the rule picks ins or park for a request that already finished, its chunks are
    still in the CPU tier; queue_write() asks for them and serve_external_requests() writes them through
    the tiering manager's parent interface (lookup, create_store_job, own submit_store);
  - link telemetry: the disk link is busy while any disk job is in flight (the pool's per-job time is
    summed over its parallel I/O threads, so it is not link time). Busy wall time and the bytes moved in
    it give the link speed (Speeds.link("disk")) and busy fraction; a job's wait is its time in flight
    beyond its own bytes at that speed (Live n_disk / wait_disk / busy_disk).

As before, a request that arrives with kv_transfer_params {"park_disk": true} is written at REQUEST_LEVEL
as it runs; every other store completes at once without I/O, and lookups answer from an in-memory index
of the chunks this tier wrote.

Config (secondary_tiers entry):
  {"type": "ParkFsTier", "module_path": "layby.vllm.fs_tier", "root_dir": ..., "n_read_threads": 16,
   "n_write_threads": 16}
"""
import dataclasses
import itertools
import time

from vllm.v1.kv_offload.base import LookupResult, OffloadPolicy, ReqContext, RequestOffloadingContext
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

from layby.live import PROBE_BYTES, probe_disk
from layby.vllm.state import STATE
from vllm.logger import init_logger

logger = init_logger(__name__)

LATE_TTL = 30.0   # seconds a late write keeps waiting for a chunk the CPU tier has not stored yet


def _parked(req_context: ReqContext | None) -> bool:
    params = (req_context.kv_transfer_params if req_context is not None else None) or {}
    return bool(params.get("park_disk"))


def _through(req_context: ReqContext | None) -> bool:
    """Chunks written as they are stored. park_write "through": every session's, except those the rule keeps
    off disk. "demand": only sessions the rule placed on disk (ins, park) on their earlier turns."""
    if STATE.write_mode not in ("through", "demand"):
        return False
    params = (req_context.kv_transfer_params if req_context is not None else None) or {}
    key = params.get("park_key")
    sess = None if key is None else str(key).rsplit("-", 1)[0]
    if STATE.write_mode == "through":
        return sess is None or sess not in STATE.nowrite
    return sess is not None and sess in STATE.thru


class ParkFsTier(FileSystemTierManager):
    parked_jobs = 0
    skipped_jobs = 0
    late_jobs = 0
    dedup_jobs = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_disk = set()           # chunks whose write completed
        self._inflight = set()          # chunks being written
        self._writing = {}              # store job id -> its keys
        self._loading = {}              # load job id -> its keys
        self._submit = {}               # job id -> (submit time, n keys) for link telemetry
        self._mark = None               # start of the not yet accounted part of the busy period
        self._bytes = 0                 # bytes completed since _mark
        self._late = []                 # [(queued at, keys)] waiting for a late write
        self._seq = itertools.count()
        STATE.fs_tier = self
        root = kwargs.get("root_dir")
        if root:
            # the disk's measured read speed seeds the estimate: with an untested default the rule may never write,
            # so it would never measure the disk (a cold start the engine cannot leave)
            try:
                g = probe_disk(root, cold=bool(getattr(self, "_use_o_direct", False)))
                if g:
                    STATE.speeds.d["disk_gbps"] = g
                    STATE.speeds.d["disk_probe_bytes"] = PROBE_BYTES
                    logger.info("park: disk probe %.2f GB/s read under %s", g, root)
            except OSError as e:
                logger.warning("park: disk probe failed (%s); disk speed starts at the default", e)

    # --- index -----------------------------------------------------------------------------
    def lookup(self, key, req_context: ReqContext) -> LookupResult:
        return LookupResult.HIT if key in self._on_disk else LookupResult.MISS

    def count_on_disk(self, keys) -> int:
        return sum(1 for k in keys if k in self._on_disk)

    def get_finished_jobs(self):
        results = super().get_finished_jobs()
        now = time.monotonic()
        for r in results:
            sub = self._submit.pop(r.job_id, None)
            if sub is not None:
                t_sub, n = sub
                nbytes = n * self._block_size
                self._bytes += nbytes
                x = nbytes / (STATE.speeds.gbps("disk") * 1e9)
                STATE.live.add(now, n_disk=1.0, wait_disk=max(now - t_sub - x, 0.0))
            keys = self._writing.pop(r.job_id, None)
            if keys is not None:
                self._inflight.difference_update(keys)
                if r.success:
                    self._on_disk.update(keys)
            keys = self._loading.pop(r.job_id, None)
            if keys is not None and not r.success:
                # a failed load: the keys past the ones that loaded are no longer trusted
                ok = set(r.successful_keys or ())
                self._on_disk.difference_update([k for k in keys if k not in ok])
        if self._mark is not None and (results or not self._submit):
            span = now - self._mark
            if span > 0:
                if self._bytes:
                    STATE.speeds.link(now, "disk", self._bytes, span)
                STATE.live.add(now, busy_disk=span)
            self._bytes = 0
            self._mark = now if self._submit else None
        return results

    def submit_load(self, job_metadata) -> None:
        self._loading[job_metadata.job_id] = list(job_metadata.keys)
        self._track(job_metadata)
        super().submit_load(job_metadata)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        if _parked(req_context):
            return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        return RequestOffloadingContext()

    def submit_store(self, job_metadata) -> None:
        if _parked(job_metadata.req_context) or _through(job_metadata.req_context):
            # write only chunks not on disk and not being written: a request-level tier gets every chunk of
            # the request again on each turn (vLLM cascades the chunks already in the CPU tier), which
            # rewrote whole contexts per turn and saturated the disk. The parent releases the job's chunks
            # by job id, so writing a subset is safe.
            keys = list(job_metadata.keys)
            keep = [i for i, k in enumerate(keys) if k not in self._on_disk and k not in self._inflight]
            if not keep:
                ParkFsTier.dedup_jobs += 1
                self._pool.enqueue_store(job_metadata.job_id, 1, [lambda: None])
                return
            if len(keep) < len(keys):
                job_metadata = dataclasses.replace(job_metadata, keys=[keys[i] for i in keep],
                                                   chunk_ids=job_metadata.chunk_ids[keep])
            ParkFsTier.parked_jobs += 1
            self._writing[job_metadata.job_id] = list(job_metadata.keys)
            self._inflight.update(job_metadata.keys)
            self._track(job_metadata)
            return super().submit_store(job_metadata)
        ParkFsTier.skipped_jobs += 1
        # complete through the normal pool path with no I/O
        self._pool.enqueue_store(job_metadata.job_id, 1, [lambda: None])

    def _track(self, job_metadata) -> None:
        now = time.monotonic()
        if self._mark is None:
            self._mark = now                 # the link goes busy
        self._submit[job_metadata.job_id] = (now, len(job_metadata.keys))

    # --- late writes -------------------------------------------------------------------------
    def queue_write(self, keys) -> None:
        keys = [k for k in keys if k not in self._on_disk and k not in self._inflight]
        if keys:
            self._late.append((time.monotonic(), keys))

    def has_pending_work(self) -> bool:
        return bool(self._late) or super().has_pending_work()

    def serve_external_requests(self, parent) -> None:
        super().serve_external_requests(parent)
        if not self._late:
            return
        now = time.monotonic()
        keep = []
        for t, keys in self._late:
            ctx = ReqContext(req_id=f"park-write-{next(self._seq)}", kv_transfer_params={"park_disk": True})
            parent.on_new_request(ctx)
            hit, wait = [], []
            for k in keys:
                if k in self._on_disk or k in self._inflight:
                    continue
                r = parent.lookup(k, ctx)
                if r == LookupResult.HIT:
                    hit.append(k)
                elif now - t < LATE_TTL:
                    wait.append(k)          # not stored on CPU yet (final stores run after finish)
            if hit:
                job = parent.create_store_job(hit, ctx)
                ParkFsTier.late_jobs += 1
                self.submit_store(job)
            parent.on_request_finished(ctx)
            if wait:
                keep.append((t, wait))
        self._late = keep
