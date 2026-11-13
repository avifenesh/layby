"""A request that arrived and waits for admission keeps its CPU-tier chunks ahead of idle sessions, and the queue's
next request keeps them longest (ParkCachePolicy.hold, from ParkConnector.on_new_request; CPU only).

Measured in Phase R (wrx, 2026-11-12 diagnostic logs): agent sessions come back right after a turn ends and then wait
for GPU room. Their chunks still carried the hint of the previous finish (a return in about 0.05 s), expired to LRU
2 s later and, being the oldest, were evicted first. The request next in line lost its prefix: returns that waited
more than 20 s in the queue recomputed 86% of their cached tokens on Qwen slow pressure and 31% on Gemma fast heavy,
against 13% and 3% for returns admitted within 1 s.

Before the change: with A and B waiting (A first) and C idle, eviction takes A's chunks first, then B's, then C's.
After it: C's first, then B's (the later arrival), then A's; held chunks do not expire; a promotion that lands for
a waiting request is held; a preempted request ranks before every arrival; admission returns the chunks to LRU;
a chunk two waiting requests share stays held until both are admitted.

Run: PYTHONPATH=. python tests/test_queue_hold.py
"""
from types import SimpleNamespace as NS

from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

import layby.vllm.connector as C
import layby.vllm.policy as P
from layby.vllm.state import STATE

STATE.chunk_tokens = STATE.block_tokens = 16
clock = [1000.0]
P.time = NS(monotonic=lambda: clock[0])


def chunk():
    c = ChunkStatus.__new__(ChunkStatus)
    c.ref_cnt = 0
    return c


def keys(s, n=4):
    return [(s, i) for i in range(n)]


def setup():
    pol = P.ParkCachePolicy(100)
    for j, s in enumerate("ABC"):
        clock[0] = 1000.0 + j                    # A finished first, then B, then C
        for k in keys(s):
            pol.insert(k, chunk())
        pol.apply(keys(s), 0.05)                 # each predicted back in 0.05 s
    clock[0] = 1010.0                            # the hints expired: LRU by their time, A oldest
    return pol


def order(pol, n=12):
    return "".join(k[0] for k, _ in pol.evict(n, set()))


# before: no hold, the session that came back first loses its chunks first
pol = setup()
assert order(pol) == "AAAABBBBCCCC", order(pol)

# A arrived at 1003, B at 1004, C has not come back
pol = setup()
pol.hold("ra", keys("A"), 1003.0)
pol.hold("rb", keys("B"), 1004.0)
clock[0] = 1100.0                                # waiting for 90 s: held chunks do not expire
got = order(pol)
assert got == "CCCCBBBBAAAA", got

# a promotion that lands for a waiting request is held, not LRU
pol = setup()
pol.hold("ra", keys("A") + [("A", 9)], 1003.0)
pol.insert(("A", 9), chunk())
assert pol.hinted[("A", 9)] < 0, pol.hinted.get(("A", 9))
got = order(pol, 13)
assert got.endswith("AAAAA") and got[:8] == "BBBBCCCC", got

# a preempted request goes to the head of the queue: its chunks outlast every waiting arrival's
pol = setup()
pol.hold("ra", keys("A"), 1003.0)
pol.hold("rc", keys("C"), -1005.0)              # preempted at 1005 (ParkConnector holds it at -now)
assert order(pol) == "BBBBAAAACCCC"

# admission: the chunks go back to LRU, in use from now (most recent)
pol = setup()
pol.hold("ra", keys("A"), 1003.0)
pol.unhold("ra")
assert not pol.held and not pol.held_t
assert order(pol) == "BBBBCCCCAAAA"

# a chunk two waiting requests share stays held until both are admitted; another request's decision leaves it held
pol = setup()
shared = ("A", 0)
pol.hold("ra", keys("A"), 1003.0)
pol.hold("rb", [shared] + keys("B"), 1004.0)
pol.unhold("ra")
pol.apply([shared], 500.0)
assert pol.hinted[shared] < 0 and set(pol.held_t[shared]) == {"rb"}
got = [k for k, _ in pol.evict(12, set())]
assert len(set(got)) == 12 and got[-1] == shared, got

# the heap's compaction keeps held chunks off the expiry heap
pol = setup()
for i in range(400):
    # each hold leaves 4 stale heap entries below the current ones (ranks rise): 1600 > 4 x 12 + 1024
    pol.hold("ra", keys("A"), 1003.0 + i * 1e-3)
pol.evict(1, set())                              # compacts
assert len(pol.heap) < 100, len(pol.heap)
clock[0] = 1200.0
got = order(pol, 11)
assert got.endswith("AAAA"), got

# the connector holds at arrival and releases at admission and at the finish
STATE.policy = pol = setup()
conn = object.__new__(C.ParkConnector)
rs = NS(update_offload_keys=lambda: None, group_states=[NS(offload_keys=keys("A"))])
conn.connector_scheduler = NS(_req_status={"ra": rs})
conn._prefetched = {}
conn._hold("ra", 1003.0)
assert set(pol.held) == {"ra"} and STATE.counts["held"] == 1
conn._unhold("ra")
assert not pol.held
print("queue hold ok: waiting requests outlast idle sessions, the queue's head longest; admission releases")
