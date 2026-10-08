#!/usr/bin/env python3
"""Published baselines as engine_sim rules (decide(view) -> eta, disk, pre, warm_eta, gpu_pin).

ContinuumTTL  Continuum (arXiv 2511.02230): after a turn that ends in a tool call, pin the session's KV on
              the GPU for tau* = argmax_tau P(tau, f) (T eta + R) - tau. P(., f) is the empirical CDF of earlier
              gaps after tool f in this run (the global CDF when f has 100 samples or fewer; Exp(1) when the
              global one has too), T the mean queueing delay (here the measured admission wait), eta the
              workload's memoryfulness -Corr(k, N - k) over (turn index, turns left), R the time to restore
              the session's KV without the pin (CPU tier load, as the cost rule measures it). Human turns are
              not pinned. CPU tier: LRU write-through; no disk.
ChoiJoshi     Choi and Joshi (arXiv 2608.30830), host-retain branch: keep the KV on the GPU until the break-even
              time t1 = beta2 / alpha1, where alpha1 = N / C_gpu is the HBM share held per second and beta2 the
              host-resume cost (the CPU tier load time); then the host keeps it (LRU). No per-request prediction,
              no disk.
KeepAlive     Serverless in the Wild (ATC'20) hybrid histogram, as a keep-alive: pin on the GPU until the 99th
              percentile of this user's earlier gaps (at least 3 of them), else no pin. CPU tier LRU, no disk.

All three keep vLLM's LRU order in the CPU tier (eta None) and only add a GPU pin; engine_sim holds a pinned
session's idle blocks until the pin ends unless nothing else can be evicted (the earliest pin goes first).
"""
import math
from collections import defaultdict

import numpy as np

KIB = 144
GRID = np.geomspace(0.05, 1800, 120)


def memoryfulness(W):
    """-Corr(k, N - k) over every turn of the workload (Continuum's eta)."""
    k, left = [], []
    for s in W["sessions"]:
        n = len(s["turns"])
        for j in range(n):
            k.append(j); left.append(n - j)
    c = np.corrcoef(k, left)[0, 1] if len(k) > 2 else 0.0
    return float(max(-c, 0.0)) if np.isfinite(c) else 0.0


def _restore_c(v):
    """Time to bring the session's KV back from the CPU tier: link wait, transfer, step (as sim.rule)."""
    p, live, now, N = v["p"], v["live"], v["now"], v["tokens"]
    lam = live.rate("n_cpu", now)
    wait = live.rate("wait_cpu", now) / lam if lam > 0 else 0.0
    return wait + N * KIB * 1024 / (p.cpu_gbps * 1e9) + p.t0


class _History:
    """Earlier gaps seen in this run: the previous turn's gap is known when the next turn of the session ends."""

    def __init__(self):
        self.last = {}                       # session -> (turn dict of its previous finished turn)
        self.by_tool = defaultdict(list)
        self.by_user = defaultdict(list)
        self.all = []

    def observe(self, v, user):
        k, t = v["sess"], v["turn"]
        prev = self.last.get(k)
        if prev is not None and prev.get("gap_after") is not None:
            g = float(prev["gap_after"])
            self.all.append(g)
            if prev.get("kind") == "tool":
                self.by_tool[prev.get("tool") or "?"].append(g)
            self.by_user[user].append(g)
        self.last[k] = t


class ContinuumTTL:
    def __init__(self, W):
        self.eta = memoryfulness(W)
        self.users = [s.get("user") or s["id"] for s in W["sessions"]]
        self.h = _History()

    def cdf(self, tool):
        xs = self.h.by_tool.get(tool) or []
        if len(xs) <= 100:
            xs = self.h.all
        if len(xs) <= 100:
            return 1.0 - np.exp(-GRID)                  # the paper's default when there is too little history
        xs = np.sort(xs)
        return np.searchsorted(xs, GRID, side="right") / len(xs)

    def decide(self, v):
        self.h.observe(v, self.users[v["sess"]])
        t = v["turn"]
        if t.get("kind") != "tool":
            return dict(eta=None, disk=False, pre=None, warm_eta=None)
        live, now = v["live"], v["now"]
        na = live.rate("n_adm", now)
        T = live.rate("qwait", now) / na if na > 0 else 0.0
        R = _restore_c(v)
        u = self.cdf(t.get("tool") or "?") * (T * self.eta + R) - GRID
        j = int(np.argmax(u))
        pin = float(GRID[j]) if u[j] > 0 else 0.0
        return dict(eta=None, disk=False, pre=None, warm_eta=None, gpu_pin=pin)


class ChoiJoshi:
    def __init__(self, W=None):
        pass

    def decide(self, v):
        N, p = v["tokens"], v["p"]
        t1 = _restore_c(v) * p.gpu_tokens / max(N, 1)
        return dict(eta=None, disk=False, pre=None, warm_eta=None, gpu_pin=t1)


class KeepAlive:
    def __init__(self, W):
        self.users = [s.get("user") or s["id"] for s in W["sessions"]]
        self.h = _History()

    def decide(self, v):
        user = self.users[v["sess"]]
        self.h.observe(v, user)
        xs = self.h.by_user.get(user) or []
        pin = float(np.quantile(xs, 0.99)) if len(xs) >= 3 else 0.0
        return dict(eta=None, disk=False, pre=None, warm_eta=None, gpu_pin=pin)


class WriteThroughDisk:
    """The disk tier without prediction: every session is written to disk as it finishes, LRU everywhere (the
    tiered default of vLLM's TieringOffloadingSpec and SGLang HiCache write-through to storage)."""

    def __init__(self, W=None):
        pass

    def decide(self, v):
        return dict(eta=None, disk=True, pre=None, warm_eta=None)
