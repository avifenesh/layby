"""Per-scheduler park state for SGLang (one scheduler process per TP rank).

Extends layby.vllm.state.ParkState (Live, Speeds, hint queue, counts, params()) with what the SGLang
adapter keeps. Two kinds of state, kept apart because TP ranks must evict identically:

  every rank   the tick clock (rank 0's monotonic time, broadcast with the requests), the logical
               access counter at each tick (to read a node's last use as tick time), the predicted
               next use per tree node, finished requests awaiting a curve, the disk placement sets;
               all of it changes only through the broadcast ticks or through tree operations every
               rank runs the same way
  leader only  (TP rank 0) the telemetry fed to Live and Speeds, the open idle periods, eviction
               times per page hash, early hints, decisions; the rule runs here and its placements
               reach every rank in the next tick

Touched from the scheduler thread only, except the hint queue (hint server thread) and thread_events
(the storage backup and prefetch threads).
"""
import logging
import queue
from collections import OrderedDict

import numpy as np

from layby.live import AG
from layby.vllm.state import ParkState

logger = logging.getLogger("layby.sglang")
_failed: set = set()

NEVER = 1e18              # predicted next use of a first-out copy
GRACE = 2.0               # a predicted use this far past due without a touch falls back to LRU order
HORIZON = 3600.0          # the residency curves' last age bin; older records cannot matter
PRUNE_EVERY = 60.0        # seconds between prunes of the per-node and per-page records


def safe(name, fn, *args, **kwargs):
    """Run park bookkeeping; on an internal error log it once and let SGLang go on (the request keeps
    SGLang's default placement)."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        if name not in _failed:
            _failed.add(name)
            logger.exception("park: %s failed; continuing with SGLang defaults for it", name)
        return None


class SglState(ParkState):
    def __init__(self):
        super().__init__()
        self.cache = None                     # the ParkRadixCache of this scheduler, once built
        self.leader = True                    # this rank pulls requests (TP rank 0); set every tick
        self.port = None                      # hint port (leader only); None: curves ride on requests
        self.server = None
        self.page_size = 1
        # every rank
        self.clock = 0.0                      # rank 0's monotonic time at the last tick
        self.tick_c: list = []                # logical access counter at each tick ...
        self.tick_t: list = []                # ... and the tick's clock
        self.next_use: dict = {}              # node id -> (predicted next use, counter when set)
        self.entries: "OrderedDict[str, dict]" = OrderedDict()   # park_key -> finished request
        self.leaf_of: dict = {}               # request id -> its inserted leaf, finish to release
        self.disk_pages: set = set()          # page hashes written to storage by this server
        self.disk_ok: set = set()             # node ids whose pending host copy goes on to storage
        self.last_prune = 0.0
        # leader only
        self.g_idle: dict = {}                # node id -> (idle since, tokens) while on the GPU
        self.c_idle: dict = {}                # node id -> (idle since, tokens) while in host memory
        self.g_evicted: dict = {}             # page hash -> time the GPU evicted it
        self.c_evicted: dict = {}             # page hash -> time host memory evicted it
        self.early: dict = {}                 # park_key -> (time, curve) that beat the finish
        self.outbox: list = []                # placements for the next tick: (key, next use, disk)
        self.arrivals: dict = {}              # request id -> time the scheduler received it
        self.step_mark = None                 # (time, prefill requests in the batch) at the last mark
        self.step_last = None                 # (time, prefill tokens) of the last batch launched
        self.thread_events: "queue.SimpleQueue" = queue.SimpleQueue()
        self.counts.update(gone=0, write_skipped=0, write_late=0, write_lost=0)

    def _open(self, d, now):
        if not d:
            return (np.zeros(0), np.zeros(0))
        v = np.array(list(d.values()), float)
        return (now - v[:, 0], v[:, 1])

    def idle_g(self, now):
        return self._open(self.g_idle, now)

    def idle_c(self, now):
        return self._open(self.c_idle, now)

    def snapshot(self):
        now = self.now()
        L, p = self.live, self.params()
        r = {k: L.rate(k, now) for k in L.acc}
        speeds = dict(cpu_gbps=p.cpu_gbps, disk_gbps=p.disk_gbps, f0=p.f0, t0=p.t0)
        if self.speeds.acc.get("s_d2h", 0.0) > 0:
            speeds["d2h_gbps"] = self.speeds.gbps("d2h")
        return dict(engine="sglang", rates=r, speeds=speeds,
                    capacity=dict(gpu_tokens=self.gpu_tokens, cpu_tokens=self.cpu_tokens,
                                  bytes_per_token=self.bytes_per_token, page_size=self.page_size),
                    idle=dict(gpu_nodes=len(self.g_idle), cpu_nodes=len(self.c_idle),
                              hinted_nodes=len(self.next_use), awaiting_curve=len(self.entries),
                              disk_pages=len(self.disk_pages)),
                    resid_g=dict(zip(("1s", "10s", "60s"), np.interp([1, 10, 60], AG, L.resid("g", now)).round(3).tolist())),
                    resid_c=dict(zip(("10s", "60s", "300s"), np.interp([10, 60, 300], AG, L.resid("c", now)).round(3).tolist())),
                    counts=self.counts, last=self.last_decisions[-20:])


STATE = SglState()
