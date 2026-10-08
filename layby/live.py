"""Live engine measurements for the park cost rule (engine-agnostic).

An engine adapter (vLLM, SGLang, a gateway) feeds events into Live as they happen; the cost rule reads
decayed rates and Kaplan-Meier residency curves from it. Every quantity is measured, not configured:
link speeds and prefill speed come from Speeds, capacities from the engine config.
"""
import math
import numpy as np

AG = np.concatenate([[0.0], np.geomspace(0.05, 3600, 48)])   # idle-age bin edges for residency estimates


class Live:
    """What a running engine can measure about itself, as averages decayed over TW seconds: the
    latency arriving requests lose to GPU misses and to CPU misses (seconds per second), link busy
    fractions, the mean number of requests waiting for or in prefill, and how long idle KV stays on
    the GPU and in the CPU tier: a token-weighted Kaplan-Meier curve over idle age, where an idle
    period ends in eviction (event) or reuse (censored). A decision rule reads these live."""

    def __init__(self, tw):
        self.tw, self.t, self.t0 = tw, 0.0, None
        self.acc = dict(age_g=0.0, w_g=0.0, age_c=0.0, w_c=0.0, ne_g=0.0, ne_c=0.0, miss_g=0.0, miss_c=0.0,
                        busy_cpu=0.0, busy_disk=0.0, pf=0.0,
                        n_cpu=0.0, wait_cpu=0.0, n_disk=0.0, wait_disk=0.0, nw=0.0, n_adm=0.0, qwait=0.0,
                        n_defer=0.0, defer=0.0, n_giveup=0.0, giveup_saved=0.0)
        self.km = {k: np.zeros(len(AG) - 1) for k in ("ev_g", "ce_g", "ev_c", "ce_c", "lg", "lc")}

    def tick(self, now):
        if self.t0 is None:
            self.t0 = self.t = now
        if now > self.t:
            d = math.exp(-(now - self.t) / self.tw)
            for k in self.acc:
                self.acc[k] *= d
            for v in self.km.values():
                v *= d
            self.t = now

    def add(self, now, **kw):
        self.tick(now)
        for k, v in kw.items():
            self.acc[k] += v

    def end_idle(self, now, tier, age, w, evicted):
        """An idle period of w tokens on tier g or c ended at this age, by eviction or by reuse."""
        self.tick(now)
        b = min(max(int(np.searchsorted(AG, age, side="right")) - 1, 0), len(AG) - 2)
        self.km[("ev_" if evicted else "ce_") + tier][b] += w
        if evicted:
            self.acc["age_" + tier] += age * w; self.acc["w_" + tier] += w; self.acc["ne_" + tier] += 1

    def end_idle_many(self, now, tier, ages, w, evicted):
        """end_idle for many periods of w tokens each (ages: array), in one update."""
        if len(ages) == 0:
            return
        self.tick(now)
        ages = np.asarray(ages, float)
        b = np.clip(np.searchsorted(AG, ages, side="right") - 1, 0, len(AG) - 2)
        np.add.at(self.km[("ev_" if evicted else "ce_") + tier], b, w)
        if evicted:
            self.acc["age_" + tier] += float(ages.sum()) * w
            self.acc["w_" + tier] += w * len(ages)
            self.acc["ne_" + tier] += len(ages)

    def miss(self, now, tier, since_evict, loss):
        """A returning request lost `loss` seconds (its own and what it delays) to KV evicted from the
        tier `since_evict` seconds ago."""
        self.tick(now)
        b = min(max(int(np.searchsorted(AG, since_evict, side="right")) - 1, 0), len(AG) - 2)
        self.km["l" + tier][b] += loss

    def lam(self, tier, x, now):
        """Loss per evicted token from returns within x seconds of the eviction. Holding n extra tokens
        on a tier that evicts e tokens/s moves every eviction n/e seconds earlier, so it costs
        e * lam(n/e) seconds per second."""
        self.tick(now)
        w = self.acc["w_" + tier]
        if w <= 0 or x <= 0:
            return 0.0
        L = self.km["l" + tier]
        cum = np.concatenate([[0.0], np.cumsum(L)])
        return float(np.interp(x, AG, cum)) / w

    def rate(self, k, now):
        """Decayed sum over the decayed window length (no start-up bias)."""
        self.tick(now)
        span = self.tw * (1 - math.exp(-(now - self.t0) / self.tw))
        return self.acc[k] / span if span > 0 else 0.0

    def tau(self, tier, now):
        """Mean idle age at eviction for tier g (GPU) or c (CPU); inf before the first eviction."""
        self.tick(now)
        w = self.acc["w_" + tier]
        return self.acc["age_" + tier] / w if w > 0 else math.inf

    def resid(self, tier, now, idle=()):
        """P(idle KV is still on the tier at age AG[j]), j = 0..; idle: (age, tokens) of periods still
        open, counted as censored at their current age."""
        self.tick(now)
        ev, ce = self.km["ev_" + tier], self.km["ce_" + tier].copy()
        if isinstance(idle, tuple) and len(idle) == 2 and isinstance(idle[0], np.ndarray):
            ages, w = idle                                       # vectorized form: (ages, tokens) arrays
            np.add.at(ce, np.clip(np.searchsorted(AG, ages, side="right") - 1, 0, len(AG) - 2), w)
            idle = ()
        for age, w in idle:
            ce[min(max(int(np.searchsorted(AG, age, side="right")) - 1, 0), len(AG) - 2)] += w
        risk = np.cumsum((ev + ce)[::-1])[::-1]
        h = np.where(risk > 0, ev / np.maximum(risk, 1e-12), 0.0)
        return np.concatenate([[1.0], np.cumprod(1 - h)])


PROBE_BYTES = 512 << 20


def probe_disk(root, total=PROBE_BYTES, files=32, threads=16, cold=False):
    """Read speed in GB/s of a disk tier under root: write `files` files of total/files bytes with `threads` threads
    (as the tier does), fsync them, read them back, delete them. A buffered tier reads recent writes from the OS
    page cache, so the default measures that warm read-back; cold=True drops the files from the cache first (the
    device itself, as an O_DIRECT tier sees it). The disk link's starting estimate, until the engine's own
    transfers are measured."""
    import os
    from concurrent.futures import ThreadPoolExecutor
    import time as _t
    os.makedirs(root, exist_ok=True)
    size = total // files
    buf = os.urandom(size)
    paths = [os.path.join(root, f".park_probe_{os.getpid()}_{i}") for i in range(files)]

    def wr(p):
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, buf); os.fsync(fd)
            if cold:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def rd(p):
        fd = os.open(p, os.O_RDONLY)
        try:
            n = 0
            while True:
                b = os.read(fd, 16 << 20)
                if not b:
                    return n
                n += len(b)
        finally:
            os.close(fd)
    try:
        with ThreadPoolExecutor(threads) as ex:
            list(ex.map(wr, paths))
            t = _t.perf_counter()
            n = sum(ex.map(rd, paths))
            secs = _t.perf_counter() - t
        return n / secs / 1e9 if secs > 0 else None
    finally:
        for p in paths:
            try:
                os.unlink(p)
            except OSError:
                pass


class Speeds:
    """Engine speeds measured from the engine's own transfers and steps, decayed over tw seconds.

    link(k, nbytes, secs): a transfer of nbytes on link k ("cpu" for CPU<->GPU, "disk" for disk<->CPU)
    that ran for secs (from start of copy to completion, excluding queue wait).
    step(secs, prefill_tokens): one engine step of secs that prefilled prefill_tokens tokens. The fixed
    per-step cost t0 is the decayed mean of steps with no prefill; f0 is the extra time per prefilled
    token on top of t0.
    """

    def __init__(self, tw, defaults):
        self.tw, self.t = tw, None
        self.d = dict(defaults)                 # fallbacks before the first measurement
        self.acc = {}

    def _add(self, now, **kw):
        if self.t is not None and now > self.t:
            f = math.exp(-(now - self.t) / self.tw)
            for k in self.acc:
                self.acc[k] *= f
        self.t = now if self.t is None else max(self.t, now)
        for k, v in kw.items():
            self.acc[k] = self.acc.get(k, 0.0) + v

    def link(self, now, k, nbytes, secs):
        if secs > 0:
            self._add(now, **{f"b_{k}": nbytes, f"s_{k}": secs})

    def step(self, now, secs, prefill_tokens):
        if prefill_tokens > 0:
            self._add(now, pt=prefill_tokens, ps=secs, pn=1.0)
        else:
            self._add(now, ds=secs, dn=1.0)

    def gbps(self, k):
        """Measured link throughput; until the link has moved as many bytes as its probe did (d[k + "_probe_bytes"]),
        the probe's (or default) value: a few small transfers measure latency, not throughput, and queueing is priced
        separately by the link's measured wait."""
        s = self.acc.get(f"s_{k}", 0.0)
        if s <= 0 or self.acc.get(f"b_{k}", 0.0) < self.d.get(f"{k}_probe_bytes", 0.0):
            return self.d[f"{k}_gbps"]
        return self.acc[f"b_{k}"] / s / 1e9

    def t0(self):
        n = self.acc.get("dn", 0.0)
        return self.acc["ds"] / n if n > 0 else self.d["t0"]

    def f0(self):
        pt = self.acc.get("pt", 0.0)
        if pt <= 0:
            return self.d["f0"]
        return max(self.acc["ps"] - self.acc["pn"] * self.t0(), 0.0) / pt


class EngineParams:
    """What the cost rule needs to know about the engine: capacities from its config, speeds measured."""

    def __init__(self, bytes_per_token, gpu_tokens, cpu_tokens, cpu_gbps, disk_gbps, f0, t0, att=0.0):
        self.bytes_per_token, self.gpu_tokens, self.cpu_tokens = bytes_per_token, gpu_tokens, cpu_tokens
        self.cpu_gbps, self.disk_gbps, self.f0, self.t0, self.att = cpu_gbps, disk_gbps, f0, t0, att

    @classmethod
    def measured(cls, speeds, bytes_per_token, gpu_tokens, cpu_tokens):
        return cls(bytes_per_token, gpu_tokens, cpu_tokens, speeds.gbps("cpu"), speeds.gbps("disk"),
                   speeds.f0(), speeds.t0())
