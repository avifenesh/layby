"""Scheduler hooks of the SGLang park adapter (registered by layby.sglang.plugin through SGLang's
HookRegistry). Each one returns at once unless this scheduler built a ParkRadixCache.

  SchedulerRequestReceiver._pull_raw_reqs  AFTER   leader (the rank that pulls requests): drain the
        hint queue and the storage threads' events, run the rule for late hints, and append a
        ParkTick (rank 0's monotonic time, placements) to the pulled list; the list is then
        broadcast to every TP rank with the requests
  SchedulerRequestReceiver.recv_requests   AFTER   every rank: strip the ticks, set the tick clock
        and apply the placements in the same order; leader: note each new request's arrival
  ScheduleBatch.prepare_for_extend         AFTER   leader: a request's first prefill is its
        admission: wait since arrival (Live n_adm, qwait) and the latency lost to KV the GPU and
        host memory evicted, against the time since that eviction (Live.miss, the holding price)
  Scheduler.run_batch                      BEFORE  leader: step timing between launches (Speeds.step:
        t0, f0) and the waiting and prefilling request counts over time (Live nw, pf)
  Scheduler.on_idle                        BEFORE  leader: an idle loop ends the step being timed
"""
import time

from layby.sglang import policy
from layby.sglang.cache import apply_placement, decide_entry
from layby.sglang.state import HORIZON, PRUNE_EVERY, STATE, logger, safe


class ParkTick:
    """Rides the request broadcast from TP rank 0, so every rank applies the same clock and
    placements at the same point of its loop."""

    def __init__(self, t, placements):
        self.t, self.placements = t, placements


# --- leader: tick production -----------------------------------------------------------
def pull_after(result, *args, **kwargs):
    if STATE.cache is None:
        return None
    STATE.leader = result is not None
    if result is None:
        return None
    now = time.monotonic()
    safe("tick", _produce, now)
    tick = ParkTick(now, STATE.outbox)
    STATE.outbox = []
    result.append(tick)
    return result


def _produce(now):
    if STATE.port and STATE.server is None:
        from layby.vllm import server
        try:
            STATE.server = server.start(STATE.port, state=STATE)
            logger.info("park: hint port %s", STATE.port)
        except OSError as e:
            STATE.server = False
            logger.error("park: hint port %s unavailable (%s); serving without late hints", STATE.port, e)
    _drain_threads()
    while not STATE.hints.empty():
        key, surv = STATE.hints.get_nowait()
        STATE.counts["hints"] += 1
        entry = STATE.entries.get(key)
        if entry is None or entry.get("decided"):
            STATE.counts["no_entry"] += 1              # not finished yet (or unknown): keep for the finish
            STATE.early[key] = (now, surv)
            continue
        STATE.counts["late"] += 1
        decide_entry(key, entry, surv, now)
    if now - STATE.last_prune > PRUNE_EVERY:
        for d in (STATE.g_evicted, STATE.c_evicted):
            old = [k for k, t in d.items() if now - t > HORIZON]
            for k in old:
                del d[k]
        STATE.early = {k: v for k, v in STATE.early.items() if now - v[0] <= HORIZON}
        STATE.arrivals = {k: t for k, t in STATE.arrivals.items() if now - t <= HORIZON}


def _drain_threads():
    """Storage thread events: per job its wait beyond its own bytes at the link speed, per busy
    period the link speed and busy time (as park/vllm/fs_tier.py)."""
    q = STATE.thread_events
    while not q.empty():
        ev = q.get_nowait()
        if not STATE.leader:
            continue
        if ev[0] == "disk_job":
            _, t_end, t_sub, nbytes = ev
            x = nbytes / (STATE.speeds.gbps("disk") * 1e9)
            STATE.live.add(t_end, n_disk=1.0, wait_disk=max(t_end - t_sub - x, 0.0))
        else:
            _, t_end, span, nbytes = ev
            if span > 0:
                if nbytes:
                    STATE.speeds.link(t_end, "disk", nbytes, span)
                STATE.live.add(t_end, busy_disk=span)


# --- every rank: tick consumption --------------------------------------------------------
def recv_after(result, *args, **kwargs):
    if not result:
        return None
    rest = [r for r in result if not isinstance(r, ParkTick)]
    if len(rest) != len(result):
        for tick in result:
            if isinstance(tick, ParkTick):
                safe("tick", _consume, tick)
    if STATE.cache is not None and STATE.leader and rest:
        safe("arrivals", _arrivals, rest)
    return rest


def _consume(tick):
    policy.on_tick_clock(tick.t)
    for key, t_pred, disk in tick.placements:
        safe("placement", apply_placement, key, t_pred, disk)
    if tick.t - STATE.last_prune > PRUNE_EVERY:
        _prune(tick.t)


def _prune(now):
    """Every rank, on the tick clock: records of nodes that left the tree, requests that never got
    a curve, ticks older than the residency horizon."""
    STATE.last_prune = now
    arena = STATE.cache.tree_core._node_arena
    for k in [k for k in STATE.next_use if k not in arena]:
        del STATE.next_use[k]
    STATE.disk_ok.intersection_update(arena.keys())
    while STATE.entries:
        k, e = next(iter(STATE.entries.items()))
        if now - e["t"] <= HORIZON:
            break
        del STATE.entries[k]
    i = 0
    while i + 1 < len(STATE.tick_t) and STATE.tick_t[i + 1] < now - HORIZON:
        i += 1
    if i:
        del STATE.tick_c[:i], STATE.tick_t[:i]
    for d in (STATE.g_idle, STATE.c_idle):
        for k in [k for k in d if k not in arena]:
            del d[k]
    STATE.cache.park_prune(HORIZON)


def _arrivals(reqs):
    from sglang.srt.managers.io_struct import BatchTokenizedGenerateReqInput, TokenizedGenerateReqInput
    now = time.monotonic()
    for r in reqs:
        if isinstance(r, TokenizedGenerateReqInput):
            STATE.arrivals[r.rid] = now
        elif isinstance(r, BatchTokenizedGenerateReqInput):
            for x in r:
                STATE.arrivals[x.rid] = now


# --- leader: admission, misses, steps ----------------------------------------------------
def prepare_for_extend_after(result, batch, *args, **kwargs):
    if STATE.cache is None or not STATE.leader or not STATE.arrivals:
        return None
    for req in batch.reqs:
        t = STATE.arrivals.pop(req.rid, None)
        if t is not None:
            safe("admission", _admit, req, t)
    return None


def _admit(req, t):
    now = time.monotonic()
    STATE.live.add(now, n_adm=1.0, qwait=now - t)
    _account_misses(req, now)


def _account_misses(req, now):
    """The latency this admission loses to KV the GPU and host memory evicted (and how long ago):
    the load of what comes from host memory and storage, at the measured link speeds and waits, and
    the recompute of what comes from nowhere, at the measured prefill speed; each counted with what
    it delays (CPU link busy fraction, requests in prefill), as in park/vllm/connector.py."""
    prompt = len(req.origin_input_ids)
    dev = req.cached_tokens_device
    host, sto = req.cached_tokens_host, req.cached_tokens_storage
    ext = host + sto
    rec = max(prompt - dev - ext, 0)
    if prompt - dev <= 0 or not (STATE.g_evicted or STATE.c_evicted):
        return
    cache, ps = STATE.cache, STATE.page_size
    hashes = cache.park_prompt_hashes(req)
    p, L = STATE.params(), STATE.live
    n_cpu = L.rate("n_cpu", now)
    wait_cpu = L.rate("wait_cpu", now) / n_cpu if n_cpu > 0 else 0.0
    n_disk = L.rate("n_disk", now)
    wait_disk = L.rate("wait_disk", now) / n_disk if n_disk > 0 else 0.0
    u = min(L.rate("busy_cpu", now), 1.0)
    npf = L.rate("pf", now)
    bpt = p.bytes_per_token
    load_t = ext * bpt / (p.cpu_gbps * 1e9) + (wait_cpu if ext else 0.0)
    load_t += sto * bpt / (p.disk_gbps * 1e9) + (wait_disk if sto else 0.0)
    rec_t = rec * p.f0
    full = load_t * (1 + u) + rec_t * (1 + npf)
    tg = [STATE.g_evicted[h] for h in hashes[dev // ps:] if h in STATE.g_evicted]
    if tg:
        L.miss(now, "g", now - max(tg), full)
        L.add(now, miss_g=full)
    if rec > 0:
        tc = [STATE.c_evicted[h] for h in hashes[(dev + ext) // ps:] if h in STATE.c_evicted]
        if tc:
            # loss beyond what loading the same tokens from a host copy would have cost
            lost_c = max(full - rec * bpt / (p.cpu_gbps * 1e9), 0.0)
            L.miss(now, "c", now - max(tc), lost_c)
            L.add(now, miss_c=lost_c)


def run_batch_before(sched, batch, *args, **kwargs):
    if STATE.cache is None or not STATE.leader:
        return None
    safe("step", _step, sched, batch)
    return None


def _step(sched, batch):
    now = time.monotonic()
    pt, npre = 0, 0
    if batch.forward_mode.is_extend():
        lens = list(getattr(batch, "extend_lens", None) or ())
        pt = sum(n for n in lens if n > 1)
        npre = sum(1 for n in lens if n > 1)
    _mark(sched, now, npre)
    if STATE.step_last is not None:
        t, p_tokens = STATE.step_last
        STATE.speeds.step(now, now - t, p_tokens)
    STATE.step_last = (now, pt)


def _mark(sched, now, npre):
    """Waiting and prefilling requests over time since the last mark."""
    if STATE.step_mark is not None:
        t, npre0 = STATE.step_mark
        dt = now - t
        nw = len(sched.waiting_queue)
        STATE.live.add(now, nw=nw * dt, pf=(nw + npre0) * dt)
    STATE.step_mark = (now, npre)


def on_idle_before(sched, *args, **kwargs):
    if STATE.cache is None or not STATE.leader:
        return None
    STATE.step_last = None                    # the next launch does not time a step across idle
    safe("idle", _mark, sched, time.monotonic(), 0)
    return None


HOOKS = (
    ("sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver._pull_raw_reqs",
     pull_after, "AFTER"),
    ("sglang.srt.managers.scheduler_components.request_receiver.SchedulerRequestReceiver.recv_requests",
     recv_after, "AFTER"),
    ("sglang.srt.managers.schedule_batch.ScheduleBatch.prepare_for_extend", prepare_for_extend_after, "AFTER"),
    ("sglang.srt.managers.scheduler.Scheduler.run_batch", run_batch_before, "BEFORE"),
    ("sglang.srt.managers.scheduler.Scheduler.on_idle", on_idle_before, "BEFORE"),
)
