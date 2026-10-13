"""CT: Continuum's KV time-to-live (arXiv 2511.02230) on vLLM, a baseline arm. Our port; the policy is layby.continuum.

vLLM's OffloadingConnector with a GPU pin: when a request whose response ends in a tool call finishes, the program's
(session's) KV blocks are pinned in the GPU block pool for the TTL of eq. 2 (layby.continuum.ContinuumTTL); otherwise
vLLM's default eviction. The CPU tier is vLLM's own (run it with the C0 configuration: CPUOffloadingSpec, LRU or ARC
as tuned). No disk tier, as in the paper.

  --kv-transfer-config '{"kv_connector": "ContinuumConnector", "kv_connector_module_path": "layby.vllm.continuum",
    "kv_role": "kv_both", "kv_connector_extra_config": {"cpu_bytes_to_use": ..., "offload_prompt_only": false,
      "eviction_policy": "lru", "ct_eta": 0.83, "ct_port": 8765}}'

ct_eta   the workload's memoryfulness (layby.continuum.memoryfulness of the replayed sessions; default 1.0, the
         paper's cold-start assumption). ct_k: the record threshold (default 100, the paper's). ct_window: seconds
         of the sliding window for T (default 120, the window of every other live estimate in this adapter).
ct_port  optional: GET /live (counts, T, R, the records' sizes) and GET /log (one row per pin and per admission).

What the client sends per request, in kv_transfer_params (the replay client fills them from the workload):
  park_key   "<session>-<turn>": the program id is the part before the last "-" (as for Layby)
  park_kind  "tool" when the response ends in a tool call (pinned), anything else is not pinned
  park_tool  the name of the tool called after this response (the paper parses it from the output)
  park_last  true on a program's last request (never pinned)

Mechanics. A pin is one extra reference on each of the request's cached GPU blocks (BlockPool.touch; the partial
tail block has no prefix-cache hash, no later request can hit it, so it is not held), taken in
request_finished before the scheduler frees the request, so the blocks leave the free queue and cannot be evicted.
The next request of the program gets them as a prefix-cache hit. A pin ends (BlockPool.free_blocks, tail first, so
the blocks go back to the free queue as vLLM frees a request) when
  - the program's next request is admitted (the paper's line 26: it now holds its own references);
  - the TTL expired and no request of the program is waiting (section 5.2, unpin_requests);
  - vLLM scheduled nothing in a step while requests waited, with no connector lookup pending and no CPU load in
    flight: the pin of the latest program arrival goes, the waiting program's own pin included (section 5.2,
    deadlock prevention; the reference scheduler unpins when allocation fails with an empty running queue). An
    unpinned block stays in the prefix cache, evictable, so the waiting program can still hit it.

Measured here: the duration of each tool call (a program's finish to its next request's arrival, server side), T
(the mean admission wait of requests whose program's KV was evicted from the GPU: a returning request whose GPU
program's KV from its last finish left the GPU prefix cache (the last cached block of a KV group lost its hash),
over the last ct_window seconds), and R (Prefill-Reload: the session's tokens times the measured seconds per token of the connector's CPU to
GPU load jobs, submit to completion; before the first load, the KV bytes per token at 20 GB/s, the adapter's
PCIe-class default).

Deviations from the paper (also in the paper's baseline appendix):
  - Scheduling is vLLM's: no program-level FCFS and no TTL-aware priority (section 4.3); the plan ports the pin only.
  - R is measured live from the engine's own CPU loads, not profiled offline against context length (section 5.2).
  - eta is configured from the workload's sessions (as sim.baselines.ContinuumTTL computes it), not learned online;
    it floors at 0 (the paper allows eta < 0 and reports none).
  - The tool name and the tool-call flag come from the client (kv_transfer_params) instead of a parser of the
    model's output (section 5.1); the replay's outputs are random tokens, so the workload's recorded tool names stand
    in for the parsed ones.
  - The deadlock check runs after vLLM's schedule, not inside it: a step that scheduled no tokens while requests
    were waiting, with no connector lookup pending and no CPU load in flight, unpins one pin; vLLM retries next
    step, so pins go one per step until something is scheduled (the paper unpins in a loop inside the failed
    allocation). The engine's allocator is the judge: no block count is estimated here, which a hybrid model
    (sliding-window groups, PR #60045's admission count) would get wrong. The signal is not a failed lookup: vLLM
    0.31 drains waiting requests that already hold KV blocks first and stops at the first it cannot allocate, and
    such a request (its CPU load done) gets no connector lookup, so the requests behind it are never looked up
    (the start block's Gemma CT stall, 2026-10-13: 0 running, 7 waiting, one pin of 1,109 of 2,188 blocks).
  - T_default is recomputed per request from the current T and that request's R (the paper derives one fixed value
    from the same formula).
Differences from the simulator's port (sim/baselines.py ContinuumTTL), which this follows where the paper is silent:
the simulator's global records also hold human gaps and its candidates are a fixed geometric grid; here only tool
durations are recorded and the candidates are the recorded durations, as the paper says.
"""
import math
import queue
import time
from collections import deque

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.logger import init_logger

from layby.continuum import ContinuumTTL
from layby.live import Speeds

logger = init_logger("vllm." + __name__)   # under vLLM's handler: a module name outside "vllm" prints nothing
_failed: set = set()


def _safe(name, fn, *args):
    """Run CT bookkeeping; on an internal error log it once and let vLLM go on with default eviction."""
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        if name not in _failed:
            _failed.add(name)
            logger.exception("continuum: %s failed; continuing with vLLM defaults for it", name)
        return None


def session_of(park_key):
    return str(park_key).rsplit("-", 1)[0] if park_key is not None else None


def _params(request):
    return getattr(request, "kv_transfer_params", None) or {}


def _common_prefix(a, b) -> int:
    """Length of the longest common prefix of two token lists (slice compares run in C)."""
    n = min(len(a), len(b))
    if a[:n] == b[:n]:
        return n
    lo, hi = 0, n                         # a[:lo] == b[:lo], a[:hi] != b[:hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid
    return lo


class CTState:
    """What the hint server needs (layby.vllm.server): a hint queue (unused here), snapshot(), the logs."""

    def __init__(self):
        self.hints = queue.SimpleQueue()
        self.counts = dict(pins=0, unpin_admit=0, unpin_ttl=0, unpin_pressure=0, unpin_replace=0, no_pin=0,
                           tool_records=0, evicted_returns=0)
        self.dlog: list = []
        self.rlog: list = []
        self.conn = None

    def snapshot(self):
        c = self.conn
        out = dict(counts=dict(self.counts))
        if c is not None:
            now = time.monotonic()
            out.update(T=c._T(now), s_per_token=c._spt(), pinned=len(c._pins),
                       pinned_blocks=sum(len(p["blocks"]) for p in c._pins.values()),
                       records=dict(all=len(c.ttl.all), tools={k: len(v) for k, v in c.ttl.by_tool.items()}),
                       eta=c.ttl.eta)
        return out


CT_STATE = CTState()


class ContinuumConnector(OffloadingConnector):
    def __init__(self, vllm_config, role, kv_cache_config):
        super().__init__(vllm_config, role, kv_cache_config)
        if role != KVConnectorRole.SCHEDULER:
            return
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._setup(extra, vllm_config.cache_config.block_size, kv_cache_config,
                    self.connector_scheduler.config.kv_group_configs[0].tokens_per_chunk)
        port = extra.get("ct_port")
        if port:
            from layby.vllm import server
            try:
                server.start(int(port), state=CT_STATE)
                logger.info("continuum: telemetry port %s", port)
            except OSError as e:
                logger.error("continuum: port %s unavailable (%s)", port, e)

    def _setup(self, extra, block_tokens, kv_cache_config, chunk_tokens, clock=time.monotonic):
        """Scheduler-side state (split from __init__ so a test can build the connector without a vLLM config)."""
        self.clock = clock
        self.ttl = ContinuumTTL(eta=float(extra.get("ct_eta", 1.0)), k=int(extra.get("ct_k", 100)))
        self.window = float(extra.get("ct_window", 120.0))
        self.block_tokens = int(block_tokens)
        self.chunk_tokens = int(chunk_tokens)
        nb = getattr(kv_cache_config, "num_blocks", 0) or 0
        size = sum(getattr(t, "size", 0) for t in getattr(kv_cache_config, "kv_cache_tensors", ()) or ())
        self.bytes_per_token = size / (nb * self.block_tokens) if nb else 0.0
        self.speeds = Speeds(self.window, dict(cpu_gbps=20.0, disk_gbps=1.5, f0=1 / 6000, t0=0.025))
        self._pins: dict[str, dict] = {}            # session -> {blocks, expiry, arrival, tau}
        self._waiting: dict[str, tuple] = {}        # request id -> (time added, session, request)
        self._local: dict[str, int] = {}            # request id -> GPU prefix hit at its lookup
        self._live: set = set()                     # request ids arrived and not finished
        self._pending: set = set()                  # request ids whose connector lookup is still pending
        self._last: dict[str, tuple] = {}           # session -> (finish time, tool or None, tokens held)
        self._held: dict[str, list] = {}            # session -> token ids of its last finished request
        self._marks: dict[str, list] = {}           # session -> (block id, hash) of the last cached block per group
        self._arrival: dict[str, float] = {}        # session -> its program's first arrival
        self._waits: deque = deque()                # (time, admission wait) of returns whose KV was evicted
        self._loads: dict[int, tuple[float, int]] = {}   # load job id -> (submit time, tokens)
        self._load_s = 0.0                          # decayed seconds and tokens of completed loads
        self._load_tok = 0.0
        self._load_t = None
        self._pool = getattr(self, "_pool", None)
        CT_STATE.conn = self

    # --- estimates -----------------------------------------------------------------------
    def _T(self, now):
        """Mean admission wait of returns whose KV was evicted, over the last window seconds; 0 before any."""
        while self._waits and now - self._waits[0][0] > self.window:
            self._waits.popleft()
        return sum(w for _, w in self._waits) / len(self._waits) if self._waits else 0.0

    def _spt(self):
        """Seconds per token of a CPU-to-GPU load (submit to completion), decayed over the window."""
        if self._load_tok > 0:
            return self._load_s / self._load_tok
        return self.bytes_per_token / (self.speeds.gbps("cpu") * 1e9)

    # --- vLLM hooks ----------------------------------------------------------------------
    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        super().bind_gpu_block_pool(gpu_block_pool)
        self._pool = gpu_block_pool

    def on_new_request(self, request) -> None:
        _safe("arrival", self._arrive, request)
        super().on_new_request(request)

    def _arrive(self, request):
        now = self.clock()
        sess = session_of(_params(request).get("park_key"))
        self._waiting[request.request_id] = (now, sess, request)
        self._live.add(request.request_id)
        if sess is None:
            return
        self._arrival.setdefault(sess, now)
        last = self._last.get(sess)
        if last is not None and last[1] is not None:
            # the tool call's duration: its program's finish to this arrival (server side, section 5.1)
            self.ttl.record(last[1], now - last[0])
            CT_STATE.counts["tool_records"] += 1
            self._last[sess] = (last[0], None, last[2])     # one record per finish

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        self._local[request.request_id] = int(num_computed_tokens)
        out = super().get_num_new_matched_tokens(request, num_computed_tokens)
        if out is not None and out[0] is None:              # None: the lookup is still pending, vLLM skips it
            self._pending.add(request.request_id)
        else:
            self._pending.discard(request.request_id)
        return out

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        out = super().update_state_after_alloc(request, blocks, num_external_tokens)
        _safe("admission", self._admit, request)
        return out

    def _admit(self, request):
        now = self.clock()
        self._pending.discard(request.request_id)
        w = self._waiting.pop(request.request_id, None)
        local = self._local.pop(request.request_id, None)
        if w is None:
            return
        t_add, sess, _ = w
        if sess is None:
            return
        last = self._last.get(sess)
        # evicted: the KV its program held at its last finish left the GPU prefix cache, read off the blocks
        # themselves: the last cached block of each KV group lost its hash (BlockPool evicts by resetting it). The
        # hit length is no measure: vLLM 0.31 rounds a Gemma 4 hit down by four 32-token pages with nothing evicted
        # (every intact return hit floor32(held) - 128 in the start block), and a chat template that re-renders the
        # reply's tail shortens the hit too (pilot 2). Without marks (no pool bound), a block short of the shared
        # prefix counts.
        shared = last[2] if last else 0
        prev, cur = self._held.get(sess), getattr(request, "all_token_ids", None)
        if last is not None and prev is not None and cur is not None:
            shared = min(shared, _common_prefix(prev, cur))
        marks = self._marks.get(sess)
        if last is None:
            evicted = False
        elif marks and self._pool is not None:
            evicted = any(self._pool.blocks[bid].block_hash != h for bid, h in marks)
        else:
            evicted = (local if local is not None else 0) + self.block_tokens <= shared
        if evicted:
            self._waits.append((now, now - t_add))
            CT_STATE.counts["evicted_returns"] += 1
        CT_STATE.rlog.append(dict(t=round(now, 3), sess=sess, wait=round(now - t_add, 4), local=local,
                                  held=last[2] if last else None, shared=shared if last else None, evicted=evicted,
                                  pinned=sess in self._pins))
        del CT_STATE.rlog[:-20000]
        if sess in self._pins:
            # the program's next request is admitted: it holds its own references now (Algorithm 1, line 26)
            self._unpin(sess)
            CT_STATE.counts["unpin_admit"] += 1

    def build_connector_meta(self, scheduler_output):
        meta = super().build_connector_meta(scheduler_output)
        now = self.clock()
        for job_id, job in getattr(meta, "load_jobs", {}).items():
            self._loads[job_id] = (now, len(job.src_spec.block_ids) * self.chunk_tokens)
        blocked = (not getattr(scheduler_output, "total_num_scheduled_tokens", 0) and bool(self._live)
                   and not self._pending and not self._loads)
        _safe("unpin", self._step, now, blocked)
        return meta

    def _step(self, now, blocked=False):
        waiting = {s for _, s, _ in self._waiting.values() if s is not None}
        for sess, p in list(self._pins.items()):
            if p["expiry"] < now and sess not in waiting:
                self._unpin(sess)
                CT_STATE.counts["unpin_ttl"] += 1
        if blocked:
            self._relieve()

    def _relieve(self):
        """Deadlock prevention (section 5.2): vLLM could not allocate the head waiting request and ran nothing, so
        the pin of the latest program arrival goes, the waiting program's own pin included; one per step."""
        if not self._pins or self._pool is None:
            return
        self._unpin(max(self._pins, key=lambda s: self._pins[s]["arrival"]))
        CT_STATE.counts["unpin_pressure"] += 1

    def update_connector_output(self, connector_output):
        _safe("cpu link", self._links, connector_output)
        super().update_connector_output(connector_output)

    def _links(self, connector_output):
        wm = connector_output.kv_connector_worker_meta
        done = getattr(wm, "completed_jobs", None) if wm is not None else None
        if not done:
            return
        now = self.clock()
        if self._load_t is not None and now > self._load_t:
            d = math.exp(-(now - self._load_t) / self.window)
            self._load_s *= d; self._load_tok *= d
        self._load_t = now
        for job_id in done:
            sub = self._loads.pop(job_id, None)
            if sub is not None and sub[1] > 0:
                self._load_s += now - sub[0]
                self._load_tok += sub[1]

    def request_finished(self, request, block_ids):
        out = super().request_finished(request, block_ids)
        _safe("finish", self._finish, request, [block_ids])
        return out

    def request_finished_all_groups(self, request, block_ids):
        out = super().request_finished_all_groups(request, block_ids)
        _safe("finish", self._finish, request, block_ids)
        return out

    def _finish(self, request, groups):
        now = self.clock()
        self._waiting.pop(request.request_id, None)
        self._local.pop(request.request_id, None)
        self._pending.discard(request.request_id)
        self._live.discard(request.request_id)
        params = _params(request)
        sess = session_of(params.get("park_key"))
        if sess is None:
            return
        if sess in self._pins:            # a pin the program never came back to (should not happen): replace it
            self._unpin(sess)
            CT_STATE.counts["unpin_replace"] += 1
        tool_call = params.get("park_kind") == "tool" and not params.get("park_last")
        tool = (params.get("park_tool") or "?") if tool_call else None
        self._last[sess] = (now, tool, int(request.num_tokens))
        toks = getattr(request, "all_token_ids", None)
        if toks is not None:
            self._held[sess] = list(toks)
        if self._pool is not None:
            marks = []
            for ids in groups:
                for bid in reversed(list(ids or ())):
                    b = self._pool.blocks[bid]
                    if not b.is_null and b.block_hash is not None:
                        marks.append((bid, b.block_hash))
                        break
            self._marks[sess] = marks
        if params.get("park_last"):
            self._last.pop(sess, None); self._arrival.pop(sess, None); self._held.pop(sess, None)
            self._marks.pop(sess, None)
        if not tool_call or self._pool is None:
            CT_STATE.counts["no_pin"] += 1
            return
        T = self._T(now)
        R = int(request.num_tokens) * self._spt()
        tau = self.ttl.ttl(tool, T, R)
        row = dict(t=round(now, 3), sess=sess, tool=tool, N=int(request.num_tokens), T=round(T, 4), R=round(R, 4),
                   src=self.ttl.source(tool), tau=round(tau, 3))
        CT_STATE.dlog.append(row)
        del CT_STATE.dlog[:-20000]
        if tau <= 0.0:
            CT_STATE.counts["no_pin"] += 1
            return
        seen, blocks = set(), []
        for ids in groups:
            for bid in ids or ():
                if bid in seen:
                    continue
                seen.add(bid)
                b = self._pool.blocks[bid]
                if not b.is_null and b.block_hash is not None:     # a block the next request can hit
                    blocks.append(b)
        if not blocks:
            return
        self._pool.touch(blocks)          # one more reference each: out of the free queue until unpinned
        self._pins[sess] = dict(blocks=blocks, expiry=now + tau, arrival=self._arrival.get(sess, now), tau=tau)
        CT_STATE.counts["pins"] += 1

    def _unpin(self, sess) -> int:
        """Drop the pin's references (tail first, as vLLM frees a request). Returns the blocks that became free."""
        p = self._pins.pop(sess, None)
        if p is None:
            return 0
        blocks = p["blocks"]
        self._pool.free_blocks(reversed(blocks))
        return sum(1 for b in blocks if b.ref_cnt == 0)
