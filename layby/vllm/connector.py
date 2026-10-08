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
  steps         time between consecutive schedules and the tokens prefilled in them (Speeds.step: t0, f0)
  CPU link      per load job: submit to completion, transfer bytes and time from the worker (Speeds.link,
                Live n_cpu / wait_cpu / busy_cpu)
  GPU residency the block pool's metrics collector is wrapped: a cached block whose last request
                finished is idle; eviction ends the period as an event, a prefix hit as censored
  misses        at admission, the latency a request loses to KV evicted from the GPU and from the CPU
                tier, against the time since that eviction (Live.miss, the source of the holding price)

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
from vllm.v1.utils import compute_iteration_details

from layby.rule import TG, decide, interp_surv
from layby.vllm import server
from layby.vllm.state import STATE

logger = init_logger(__name__)
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
        self._deferred: dict[str, float] = {}          # request id -> first lookup that deferred it
        # request id -> the GPU-local prefix hit (tokens) vLLM passed to the last lookup. vLLM calls
        # update_state_after_alloc before it sets request.num_computed_tokens, so a new request reads 0 there
        self._local: dict[str, int] = {}
        self._finished: "OrderedDict[str, dict]" = OrderedDict()   # park_key -> request awaiting a curve
        self._early: dict[str, tuple[float, list]] = {}               # park_key -> curve that beat the finish
        self._loads: dict[int, tuple[float, int]] = {}  # load job id -> (submit time, chunks)
        self._last_step = None                          # (time, prefill tokens, had work)
        self._last_prune = time.monotonic()
        STATE.write_mode = extra.get("park_write", "rule")
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
                server.start(int(port))
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
        self._waiting[request.request_id] = time.monotonic()
        super().on_new_request(request)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._local[request.request_id] = int(num_computed_tokens)
        out = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if out[0] is None and request.request_id not in self._deferred:
            # the offload tier is promoting part of the prefix (disk -> CPU): the request waits steps
            self._deferred[request.request_id] = time.monotonic()
        return out

    def _admit(self, request, num_external_tokens):
        t = self._waiting.pop(request.request_id, None)
        td = self._deferred.pop(request.request_id, None)
        local = self._local.pop(request.request_id, None)
        if td is not None:
            now = time.monotonic()
            STATE.live.add(now, n_defer=1.0, defer=now - td)
        if t is not None:
            now = time.monotonic()
            STATE.live.add(now, n_adm=1.0, qwait=now - t)
            self._account_misses(request, num_external_tokens, now,
                                 request.num_computed_tokens if local is None else max(local, request.num_computed_tokens))

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        out = super().update_state_after_alloc(request, blocks, num_external_tokens)
        _safe("admission", self._admit, request, num_external_tokens)
        return out

    def _account_misses(self, request, ext: int, now: float, local: int) -> None:
        """The latency this admission loses to KV the GPU and the CPU tier evicted (and how long
        ago): the load of what comes from outside the GPU, at the CPU link's measured speed and wait,
        and the recompute of what comes from nowhere, at the measured prefill speed; each counted with
        what it delays (link busy fraction, requests in prefill), as in the simulator."""
        bt, ct = STATE.block_tokens, STATE.chunk_tokens
        prompt = request.num_prompt_tokens
        rec = max(prompt - local - ext, 0)
        params = getattr(request, "kv_transfer_params", None) or {}
        STATE.rlog.append(dict(t=round(now, 3), sess=session_of(params.get("park_key")), P=prompt, local=local,
                               ext=ext, rec=rec))
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
            tc = [pol.c_evicted[k] for k in keys if k in pol.c_evicted]
            if tc:
                # loss beyond what loading the same tokens from a CPU copy would have cost
                lost_c = max(full - rec * p.bytes_per_token / (p.cpu_gbps * 1e9), 0.0)
                L.miss(now, "c", now - max(tc), lost_c)
                L.add(now, miss_c=lost_c)

    # --- steps and the CPU link --------------------------------------------------------
    def build_connector_meta(self, scheduler_output):
        now = time.monotonic()
        _safe("idle", STATE.flush_idle, now)
        _safe("hints", self._drain_hints, now)
        _safe("step", self._step, scheduler_output, now)
        meta = super().build_connector_meta(scheduler_output)
        for job_id, job in meta.load_jobs.items():
            self._loads[job_id] = (now, len(job.src_spec.block_ids))
        if now - self._last_prune > 60.0:
            _safe("prune", self._prune, now)
        return meta

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
        self._waiting.pop(request.request_id, None)
        self._deferred.pop(request.request_id, None)
        ids = block_ids[0] if block_ids and isinstance(block_ids[0], (list, tuple)) else block_ids
        pool = getattr(self, "_pool", None)
        for bid in ids or ():
            if pool is None or pool.blocks[bid].ref_cnt <= 1:     # no other running request holds it
                STATE.g_idle[bid] = now
        req_status = self.connector_scheduler._req_status.get(request.request_id)
        if req_status is None:
            return
        params = request.kv_transfer_params
        if params is None:
            params = {}
            req_status.req_context.kv_transfer_params = params
        key = str(params.get("park_key") or request.request_id)
        entry = dict(keys=_request_keys(req_status), tokens=request.num_tokens,
                     params=params, t=now)
        early = self._early.pop(key, None)
        if params.get("park_surv") is not None or early is not None:
            self._decide(entry, params["park_surv"] if params.get("park_surv") is not None else early[1], now)
        elif self._hints_on:
            self._finished[key] = entry
            self._finished.move_to_end(key)

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
        on_disk = fs.count_on_disk(keys) * STATE.chunk_tokens if fs is not None else 0
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
        if pol is not None:
            pol.apply(keys, d["eta"], now)
        if d["disk"] and fs is not None:
            fs.queue_write(keys)
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

    def _prune(self, now: float) -> None:
        self._last_prune = now
        for d in (STATE.g_evicted, getattr(STATE.policy, "c_evicted", {})):
            old = [k for k, t in d.items() if now - t > HORIZON]
            for k in old:
                del d[k]
        while self._finished:
            k, e = next(iter(self._finished.items()))
            if now - e["t"] <= HORIZON:
                break
            del self._finished[k]
        self._early = {k: v for k, v in self._early.items() if now - v[0] <= HORIZON}
        self._loads = {j: v for j, v in self._loads.items() if now - v[0] <= HORIZON}
