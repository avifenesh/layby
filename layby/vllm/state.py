"""Per-engine park state shared by the connector, the CPU policy and the disk tier (one EngineCore
process). Everything here is touched from the scheduler thread only, except the hint queue, which the
hint server thread fills."""
import queue
import time

import math

import numpy as np

from layby.live import EngineParams, Live, Speeds


class ParkState:
    def __init__(self):
        self.tw = 120.0                       # estimator smoothing window, seconds
        self.live = Live(self.tw)
        # fallbacks until the engine has measured itself: PCIe-class CPU link, NVMe-class disk, a
        # prefill speed of the order of a mid-size model; replaced by measurements as transfers run
        self.speeds = Speeds(self.tw, dict(cpu_gbps=20.0, disk_gbps=1.5, f0=1 / 6000, t0=0.025))
        self.block_tokens = 16                # GPU block size in tokens
        self.chunk_tokens = 16                # offload chunk size in tokens
        self.bytes_per_token = 0              # KV bytes per token (all layers)
        self.gpu_tokens = 0
        self.cpu_tokens = 0
        self.policy = None                    # ParkCachePolicy of the CPU tier
        self.fs_tier = None                   # ParkFsTier, if a disk tier is configured
        self.demoter = None                   # layby.vllm.demote.Demoter (write_mode "demand")
        self.write_mode = "rule"              # "through": every chunk to disk unless the rule says a disk copy cannot pay
        self.nowrite = set()                  # sessions the rule keeps off disk (write_mode "through")
        self.thru = set()                     # sessions written through to disk (write_mode "demand": ins, park)
        self.spill = {}                       # chunk -> session, written to disk when the CPU tier evicts it ("demand": none)
        self.evictions = 0                    # CPU-tier evict() calls so far
        self.burst = (0.0, 0.0)               # (time, chunks): the largest recent single eviction of the CPU tier, decayed over tw
        self.g_idle: dict = {}                # GPU block hash -> time its last user finished
        self.g_evicted: dict = {}             # GPU block hash -> time it was evicted while idle
        # idle periods that ended since the last flush: (tier, evicted) -> ages. vLLM reports them per block
        # and per chunk (thousands per admission); folding them into the estimators once per step keeps
        # that out of the scheduler's hot path
        self.ended = {("g", False): [], ("g", True): [], ("c", False): [], ("c", True): []}
        self.hints: "queue.SimpleQueue" = queue.SimpleQueue()   # (key, surv) from the hint server
        self.counts = dict(none=0, ins=0, park=0, drop=0, hints=0, late=0, no_entry=0, spill=0)
        self.last_decisions: list = []
        # research logs (bounded): one row per placement decision with its cost terms, one per admission with
        # where its prompt came from; served at /log so a run can be read decision by decision
        self.dlog: list = []
        self.rlog: list = []

    @staticmethod
    def now() -> float:
        return time.monotonic()

    def params(self) -> EngineParams:
        return EngineParams.measured(self.speeds, self.bytes_per_token, self.gpu_tokens, self.cpu_tokens)

    def idle_g(self, now):
        if not self.g_idle:
            return (np.zeros(0), np.zeros(0))
        t = np.fromiter(self.g_idle.values(), float, len(self.g_idle))
        return (now - t, np.full(len(t), float(self.block_tokens)))

    def note_burst(self, now, n):
        self.evictions += 1
        t, b = self.burst
        self.burst = (now, max(float(n), b * math.exp(-(now - t) / self.live.tw)))

    def burst_now(self, now):
        t, b = self.burst
        return b * math.exp(-(now - t) / self.live.tw)

    def flush_idle(self, now):
        for (tier, ev), ages in self.ended.items():
            if ages:
                w = self.block_tokens if tier == "g" else self.chunk_tokens
                self.live.end_idle_many(now, tier, ages, w, ev)
                ages.clear()

    def snapshot(self):
        now = self.now()
        self.flush_idle(now)
        L, p = self.live, self.params()
        r = {k: L.rate(k, now) for k in L.acc}
        return dict(rates=r, speeds=dict(cpu_gbps=p.cpu_gbps, disk_gbps=p.disk_gbps, f0=p.f0, t0=p.t0),
                    capacity=dict(gpu_tokens=self.gpu_tokens, cpu_tokens=self.cpu_tokens,
                                  bytes_per_token=self.bytes_per_token, chunk_tokens=self.chunk_tokens),
                    idle=dict(gpu_blocks=len(self.g_idle), cpu_chunks=len(self.policy.idle_since) if self.policy else 0),
                    resid_g=dict(zip(("1s", "10s", "60s"), np.interp([1, 10, 60], _AG(), L.resid("g", now)).round(3).tolist())),
                    resid_c=dict(zip(("10s", "60s", "300s"), np.interp([10, 60, 300], _AG(), L.resid("c", now)).round(3).tolist())),
                    counts=self.counts, disk=self._disk(), last=self.last_decisions[-20:])

    def _disk(self):
        fs = self.fs_tier
        if fs is None:
            return {}
        t = type(fs)
        d = dict(written_jobs=t.parked_jobs, dedup_jobs=t.dedup_jobs, late_jobs=t.late_jobs,
                 skipped_jobs=t.skipped_jobs, chunks_on_disk=len(fs._on_disk), chunks_writing=len(fs._inflight))
        if self.demoter is not None:
            d.update(demoted=self.demoter.demoted, demote_dropped=self.demoter.dropped,
                     demoting=len(self.policy.demoting) if self.policy else 0)
        return d


def _AG():
    from layby.live import AG
    return AG


STATE = ParkState()
