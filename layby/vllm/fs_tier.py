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

Parallel I/O: each load (promotion) and store job is split into up to n_read_threads / n_write_threads tasks of
consecutive chunks, where vLLM 0.31's fs tier runs a job as one task (one thread reading or writing the job's
files one after another). Phase R, Gemma fast heavy (wrx, 2026-11-12): restores ran at about 0.7 GB/s at the
median on a disk capped at 1.35 GB/s for reads.

Arm PT (pilot only): "write_all": true writes every chunk the CPU tier stores, as vLLM's own fs tier does in arm T,
with no cost rule (run it beside the stock OffloadingConnector and LRU). It compares this tier's I/O path (its
in-memory index for lookups, its dedup of chunks on disk or in flight, its link telemetry) with the native one. The
default (false) leaves the tier as above.
"""
import dataclasses
import functools
import itertools
import threading
import time

from vllm.v1.kv_offload.base import LookupResult, OffloadPolicy, ReqContext, RequestOffloadingContext
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.fs import io as fs_io

from layby.live import PROBE_BYTES, probe_disk
from layby.vllm.state import STATE
from vllm.logger import init_logger

logger = init_logger("vllm." + __name__)   # under vLLM's handler: a module name outside "vllm" prints nothing

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


def _sess_of(req_context: ReqContext | None) -> str | None:
    params = (req_context.kv_transfer_params if req_context is not None else None) or {}
    key = params.get("park_key")
    return None if key is None else str(key).rsplit("-", 1)[0]


class ParkFsTier(FileSystemTierManager):
    parked_jobs = 0
    skipped_jobs = 0
    late_jobs = 0
    dedup_jobs = 0

    def __init__(self, *args, write_all=False, **kwargs):
        super().__init__(*args, **kwargs)
        # parallel I/O within a job: vLLM 0.31's fs tier runs each job's chunk files as one task on one thread
        # (queue depth 1), so a session's restore read its files one after another while the other read threads
        # idled. Each job is split into up to this many tasks over consecutive chunks
        self._split_load = max(int(kwargs.get("n_read_threads", 16)), 1)
        self._split_store = max(int(kwargs.get("n_write_threads", 16)), 1)
        self._progress_lock = threading.Lock()
        self._write_all = bool(write_all)   # arm PT: every stored chunk, no rule
        self._on_disk = set()           # chunks whose write completed
        self._inflight = set()          # chunks being written
        self._writing = {}              # store job id -> its keys
        self._loading = {}              # load job id -> its keys, in submit order while unfinished
        self._load_key_job = {}         # key -> the unfinished load job that reads it
        self._load_mark = None          # start of the not yet accounted load-queue busy period
        self._load_bytes = 0            # load bytes completed since _load_mark
        self._load_io_times = {}        # job id -> [t_start, t_end] measured inside worker threads
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
        loads_done = False
        io_start, io_end = None, None
        lkj = getattr(self, "_load_key_job", None)
        io_times = getattr(self, "_load_io_times", None)
        lock = getattr(self, "_progress_lock", None)
        for r in results:
            sub = self._submit.pop(r.job_id, None)
            is_load = r.job_id in self._loading
            is_write = r.job_id in self._writing
            t_io = None
            if is_load and io_times is not None:
                if lock is not None:
                    with lock:
                        t_io = io_times.pop(r.job_id, None)
                else:
                    t_io = io_times.pop(r.job_id, None)
                if t_io is not None and t_io[1] > t_io[0]:
                    io_start = t_io[0] if io_start is None else min(io_start, t_io[0])
                    io_end = t_io[1] if io_end is None else max(io_end, t_io[1])
            if sub is not None:
                t_sub, n = sub
                nbytes = n * self._block_size
                self._bytes += nbytes
                x = nbytes / (STATE.speeds.gbps("disk") * 1e9)
                t_done = t_io[1] if (t_io is not None and t_io[1] >= t_sub) else now
                wait = max(t_done - t_sub - x, 0.0)
                kw = dict(n_disk=1.0, wait_disk=wait)
                if is_load:
                    kw.update(n_disk_r=1.0, wait_disk_r=wait)
                elif is_write:
                    kw.update(n_disk_w=1.0)
                STATE.live.add(now, **kw)
            keys = self._writing.pop(r.job_id, None)
            if keys is not None:
                self._inflight.difference_update(keys)
                if r.success:
                    self._on_disk.update(keys)
            keys = self._loading.pop(r.job_id, None)
            if keys is not None:
                loads_done = True
                if hasattr(self, "_load_bytes"):
                    self._load_bytes += len(keys) * self._block_size
                if lkj is not None:
                    for k in keys:
                        if lkj.get(k) == r.job_id:
                            del lkj[k]
                if not r.success:
                    # a failed load: the keys past the ones that loaded are no longer trusted
                    ok = set(r.successful_keys or ())
                    self._on_disk.difference_update([k for k in keys if k not in ok])
        if self._mark is not None and (results or not self._submit):
            span = now - self._mark
            if span > 0:
                # only feed combined bytes into disk speed before reads have run; once reads run,
                # read throughput is measured directly below so full-duplex R+W does not inflate read speed
                if self._bytes and STATE.speeds.bps("drain") is None:
                    STATE.speeds.link(now, "disk", self._bytes, span)
                STATE.live.add(now, busy_disk=span)
            self._bytes = 0
            self._mark = now if self._submit else None
        lmark = getattr(self, "_load_mark", None)
        if lmark is not None and (loads_done or not self._loading):
            span = now - lmark
            if span > 0:
                STATE.speeds.link(now, "drain", self._load_bytes, span)
                if self._load_bytes:
                    # prefer exact worker-thread I/O span when non-zero (avoids scheduler-step polling bias
                    # on large models where a step takes ~0.5-1s while NVMe finishes in ~0.06s)
                    io_span = (io_end - io_start) if (io_start is not None and io_end is not None and io_end > io_start) else span
                    STATE.speeds.link(now, "disk", self._load_bytes, min(io_span, span))
            self._load_bytes = 0
            self._load_mark = now if self._loading else None
        return results

    def drain_rate(self):
        """Bytes per second the load queue drains at: load bytes completed per wall-clock second with unfinished
        loads, decayed over STATE.tw. None until a load job completed."""
        return STATE.speeds.bps("drain")

    def submit_load(self, job_metadata) -> None:
        keys = list(job_metadata.keys)
        self._loading[job_metadata.job_id] = keys
        if hasattr(self, "_load_key_job"):
            for k in keys:
                self._load_key_job[k] = job_metadata.job_id
        if getattr(self, "_load_mark", None) is None and hasattr(self, "_load_mark"):
            self._load_mark = time.monotonic()
        self._track(job_metadata)
        if not self._split_io(job_metadata, load=True):
            super().submit_load(job_metadata)

    def load_queue_ahead(self, key, pending=()):
        """Bytes the disk must still read before key is in the CPU tier: unfinished load jobs submitted up to and
        including the one that reads key, plus any promotions held by the tiering manager until step end.
        None: no load job reads key."""
        lkj = getattr(self, "_load_key_job", None)
        job = lkj.get(key) if lkj is not None else None
        if job is not None:
            total = 0
            for jid, keys in self._loading.items():
                total += len(keys) * self._block_size
                if jid == job:
                    return total
            return None
        total = sum(len(keys) for keys in self._loading.values()) * self._block_size
        for keys in pending:
            total += len(keys) * self._block_size
            if key in keys:
                return total
        return None

    def _split_io(self, job, load: bool) -> bool:
        """Run a job's chunk files as up to n_read_threads (loads) or n_write_threads (stores) tasks of consecutive
        chunks; the pool completes the job when every task has. A load that fails keeps the chunks before the first
        failed one, as the base's single task does. False: not split (one chunk, or a vLLM without these fields)."""
        keys = list(job.keys)
        n = len(keys)
        # read priority: while an urgent promotion load is in flight, run background stores on 1 thread
        # instead of competing across all 16 write threads (while still pinning chunks immediately via submit_store)
        max_k = getattr(self, "_split_load" if load else "_split_store", 1)
        if not load and self._loading:
            max_k = 1
        k = min(max_k, n)
        min_k = 1 if load else 2
        if k < min_k or not all(hasattr(self, a) for a in ("_pool", "file_mapper", "_job_block_counts", "_block_size",
                                                           "_primary_kv_view", "_use_o_direct", "_load_progress",
                                                           "_progress_lock")):
            return False
        job_id = job.job_id
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(cid) * self._block_size for cid in job.chunk_ids]
        bounds = [n * i // k for i in range(k + 1)]
        view, size, odirect = self._primary_kv_view, self._block_size, self._use_o_direct
        self._job_block_counts[job_id] = n
        if load:
            if hasattr(self, "_load_job_keys"):
                self._load_job_keys[job_id] = keys
            io_times = self.__dict__.setdefault("_load_io_times", {})

            def part(lo, hi):
                def task():
                    t0 = time.monotonic()
                    try:
                        fs_io.batch_load_block(paths[lo:hi], view, offsets[lo:hi], size, odirect)
                    except OSError as exc:
                        done = lo + getattr(exc, "num_succeeded", 0)
                        with self._progress_lock:   # the job keeps the chunks before its first failure
                            cur = self._load_progress.get(job_id)
                            self._load_progress[job_id] = done if cur is None else min(cur, done)
                        raise
                    finally:
                        t1 = time.monotonic()
                        with self._progress_lock:
                            prev = io_times.get(job_id)
                            if prev is None:
                                io_times[job_id] = [t0, t1]
                            else:
                                if t0 < prev[0]:
                                    prev[0] = t0
                                if t1 > prev[1]:
                                    prev[1] = t1
                return task
            self._pool.enqueue_load(job_id, k, [part(bounds[i], bounds[i + 1]) for i in range(k)])
        else:
            if hasattr(self, "_store_jobs"):
                self._store_jobs[job_id] = job
            elif getattr(self, "events", None) is not None and hasattr(self, "_store_job_keys"):
                self._store_job_keys[job_id] = keys
            tasks = [functools.partial(fs_io.batch_store_block, paths[bounds[i]:bounds[i + 1]], view,
                                       offsets[bounds[i]:bounds[i + 1]], size, odirect) for i in range(k)]
            self._pool.enqueue_store(job_id, k, tasks)
        return True

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        if _parked(req_context):
            return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        return RequestOffloadingContext()

    def _should_stream(self, req_context: ReqContext | None, n_keep: int) -> bool:
        """Adaptive streaming in default 'rule' mode: write incremental CHUNK_LEVEL prefill stores directly
        to disk when disk speed strictly beats GPU recompute, disk reads are idle, CPU pinning headroom is
        safe (zero afal risk), and either the cost rule already placed this session on disk (STATE.thru) or
        tight CPU capacity risks evicting running prompts before turn end (without overriding STATE.nowrite)."""
        if STATE.write_mode != "rule" or n_keep <= 0:
            return False
        pol = STATE.policy
        if pol is None or pol.cache_capacity <= 0:
            return False
        bpt = STATE.bytes_per_token
        d_gbps = STATE.speeds.gbps("disk")
        f0 = STATE.speeds.f0()
        if bpt <= 0 or d_gbps <= 0 or f0 <= 0:
            return False
        # hardware physics: only stream when disk transfer per token strictly beats GPU recompute (fast disk)
        if bpt / (d_gbps * 1e9) >= f0:
            return False
        # read-priority and zero-afal CPU headroom guard
        if self._loading:
            return False
        cap = pol.cache_capacity
        max_inflight = max(cap // 4, 1)
        if len(self._inflight) + n_keep > max_inflight:
            return False
        pinned = len(pol.chunks) - len(pol.evictable)
        avail = cap - pinned
        if avail <= max(cap // 2, max_inflight):
            return False
        sess = _sess_of(req_context)
        if sess is not None and sess in STATE.nowrite:
            return False
        if sess is not None and sess in STATE.thru:
            return True
        # cold / turn-0 CPU overflow protection: if GPU >= CPU or active running/pinned chunks already occupy
        # >50% of CPU while the rule is writing the majority of turns (or during initial cold start), stream
        # incremental chunks so long prompts are not evicted from CPU before post-turn late_write runs
        c = STATE.counts
        n_disk = c["ins"] + c["park"]
        n_dec = n_disk + c["none"] + c["drop"]
        if n_dec >= 4 and n_disk < n_dec * 0.5:
            return False
        gpu_tok, cpu_tok = STATE.gpu_tokens, STATE.cpu_tokens
        if gpu_tok > 0 and cpu_tok > 0 and gpu_tok >= cpu_tok:
            return True
        pinned = len(pol.chunks) - len(pol.evictable)
        pmem = pol.pending_mem_count() if hasattr(pol, "pending_mem_count") else 0
        return STATE.evictions > 0 and (pinned + pmem) > cap // 2

    def submit_store(self, job_metadata) -> None:
        parked = _parked(job_metadata.req_context)
        thru = _through(job_metadata.req_context)
        write_all = getattr(self, "_write_all", False)
        keys = list(job_metadata.keys)
        keep = [i for i, k in enumerate(keys) if k not in self._on_disk and k not in self._inflight]
        stream = (not (parked or thru or write_all)) and self._should_stream(job_metadata.req_context, len(keep))
        if parked or thru or write_all or stream:
            # write only chunks not on disk and not being written: a request-level tier gets every chunk of
            # the request again on each turn (vLLM cascades the chunks already in the CPU tier), which
            # rewrote whole contexts per turn and saturated the disk. The parent releases the job's chunks
            # by job id, so writing a subset is safe.
            pol = STATE.policy
            if parked and pol is not None and hasattr(pol, "release_pending_write"):
                # late-write chunks are now pinned (ref_cnt >= 1) by create_store_job: apply deferred ins/park hint
                pol.release_pending_write(keys)
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
            if not self._split_io(job_metadata, load=False):
                super().submit_store(job_metadata)
            return
        ParkFsTier.skipped_jobs += 1
        # complete through the normal pool path with no I/O
        self._pool.enqueue_store(job_metadata.job_id, 1, [lambda: None])

    def _track(self, job_metadata) -> None:
        now = time.monotonic()
        if self._mark is None:
            self._mark = now                 # the link goes busy
        self._submit[job_metadata.job_id] = (now, len(job_metadata.keys))

    # --- late writes -------------------------------------------------------------------------
    def queue_write(self, keys) -> list:
        keys = [k for k in keys if k not in self._on_disk and k not in self._inflight]
        if keys:
            self._late.append((time.monotonic(), keys))
        return keys

    def has_pending_work(self) -> bool:
        return bool(self._late) or super().has_pending_work()

    def serve_external_requests(self, parent) -> None:
        super().serve_external_requests(parent)
        if not self._late:
            return
        now = time.monotonic()
        pol = STATE.policy
        # backpressure on slow disks: never pin more than ~25% of the CPU tier with concurrent late
        # store jobs (always allow at least 1 active write so large single turns never stall)
        max_inflight = max(pol.cache_capacity // 4, 1) if pol is not None else 0
        keep = []
        for idx, (t, keys) in enumerate(self._late):
            pinned = (len(pol.chunks) - len(pol.evictable)) if pol is not None else 0
            idle_all = (pinned == 0 and not self._loading and not self._inflight)
            if not idle_all and max_inflight and pol is not None:
                write_room = max_inflight - len(self._inflight)
                total_room = max(pol.cache_capacity // 2, 1) - pinned
                if min(write_room, total_room) <= 0:
                    keep.extend(self._late[idx:])
                    break
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
                # under any active load/pins, strictly enforce <=25% write pinning AND <=50% total CPU pinning
                # so slow-disk late writes can never pile up with promotions to cause CPU allocation failures
                if not idle_all and max_inflight and pol is not None:
                    pinned_now = len(pol.chunks) - len(pol.evictable)
                    budget = min(max_inflight - len(self._inflight), max(pol.cache_capacity // 2, 1) - pinned_now)
                    if budget <= 0:
                        wait = hit + wait
                        hit = []
                    elif len(hit) > budget:
                        wait = hit[budget:] + wait
                        hit = hit[:budget]
            if hit:
                job = parent.create_store_job(hit, ctx)
                ParkFsTier.late_jobs += 1
                self.submit_store(job)
            parent.on_request_finished(ctx)
            if wait:
                keep.append((t, wait))
        self._late = keep
