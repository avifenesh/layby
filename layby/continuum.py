"""Continuum's KV time-to-live (Li et al., arXiv 2511.02230v7, section 4), engine-agnostic.

After a request that ends in a tool call, Continuum pins the program's KV on the GPU for

    tau* = argmax_tau  P(tau, f) (T eta + R) - tau                                      (eq. 2)

  P(tau, f)  the empirical CDF of the recorded durations S[f] of tool f: (1/|S[f]|) sum_{t in S[f]} 1[t <= tau];
             a duration is the server-side interval from a request's finish to the arrival of its program's next
             request (section 5.1);
  T          the average queueing delay of requests that were evicted (a sliding window), initialized to 0;
  eta        the workload's memoryfulness, -Corr(k, N - k) over (requests served, requests left);
  R          Prefill-Reload(r): the time to rebuild r's KV on the GPU, a reload from CPU when offloading is on.

Eq. 2 is eq. 1, P(tau, f) Benefit(r) - Cost(tau, r) with Benefit = MemUsage/M (Prefill-Reload + T eta) and Cost =
MemUsage/M tau, divided by MemUsage/M. Candidates: tau = 0 and every distinct recorded duration. Cold start (K = 100):
with |S| <= K records over all tools, the TTL is T_default, the optimum of eq. 2 when durations are Exp(1) and eta = 1:
argmax (1 - e^-tau)(T + R) - tau = ln(T + R) if T + R > 1, else 0. With |S| > K but |S[f]| <= K, P is the CDF of
the global records S; otherwise that of S[f].

This module holds the records and the TTL; the engine adapter (layby.vllm.continuum) measures T and R, pins and
unpins. Differences from the paper's system are listed in layby.vllm.continuum.
"""
import math
from collections import defaultdict

import numpy as np

K = 100   # the paper's threshold on the number of records (section 4.2)


def default_ttl(T, R):
    """T_default: eq. 2 with Exp(1) durations and eta = 1. d/dtau [(1 - e^-tau) B - tau] = B e^-tau - 1 = 0."""
    B = T + R
    return math.log(B) if B > 1.0 else 0.0


def best_ttl(durations, B):
    """argmax over tau in {0} and the distinct durations of ECDF(tau) * B - tau (eq. 2 with B = T eta + R).
    Ties go to the smaller tau."""
    xs = np.sort(np.asarray(durations, float))
    if len(xs) == 0 or B <= 0:
        return 0.0
    u = np.unique(xs)
    P = np.searchsorted(xs, u, side="right") / len(xs)
    gain = P * B - u
    j = int(np.argmax(gain))
    return float(u[j]) if gain[j] > 0.0 else 0.0


class ContinuumTTL:
    """Tool-call records and the TTL of eq. 2 with the paper's cold-start rule."""

    def __init__(self, eta=1.0, k=K):
        self.eta = float(eta)
        self.k = int(k)
        self.by_tool = defaultdict(list)
        self.all = []

    def record(self, tool, duration):
        d = max(float(duration), 0.0)
        self.by_tool[tool].append(d)
        self.all.append(d)

    def source(self, tool):
        """Which records set the TTL of tool f now: 'default', 'global' or 'tool'."""
        if len(self.all) <= self.k:
            return "default"
        return "tool" if len(self.by_tool.get(tool, ())) > self.k else "global"

    def ttl(self, tool, T, R):
        src = self.source(tool)
        if src == "default":
            return default_ttl(T, R)
        xs = self.by_tool[tool] if src == "tool" else self.all
        return best_ttl(xs, T * self.eta + R)


def memoryfulness(sessions):
    """Continuum's eta = -Corr(k, N - k) over every request of every program (k requests served before it, N in its
    program). sessions: a list of turn counts or of dicts with "turns". Floors at 0 as sim.baselines does (the paper
    allows eta < 0 but reports none)."""
    k, left = [], []
    for s in sessions:
        n = s if isinstance(s, int) else len(s["turns"])
        for j in range(n):
            k.append(j); left.append(n - j)
    if len(k) <= 2:
        return 0.0
    c = np.corrcoef(k, left)[0, 1]
    return float(max(-c, 0.0)) if np.isfinite(c) else 0.0
