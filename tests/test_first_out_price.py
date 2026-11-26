"""CPU-tier evictions of chunks the rule placed first out (park, drop) stay out of the holding price and the residency
of kept copies.

Measured in Phase R (Qwen slow pressure, wrx, 2026-11-12 diagnostic): the rule dropped 96 of 351 sessions it expected
back within about 6 s, at a CPU holding price 3.5x the price behind its keep decisions; those returns recomputed 85%
of their tokens. The holding price e * lam(n / e) learned from every eviction and every miss on an evicted chunk,
including the chunks a drop had placed first out, so each drop that came back as a miss raised the price that
produced the next drop.

Before the change: a first-out eviction adds its tokens to the evicted weight w_c and an eviction event to the
residency curve. After it: it is censored for the residency, adds nothing to w_c, still counts as an eviction event
(ne_c), and the policy remembers it in c_first (the connector leaves misses on it out of the holding price).

Run: PYTHONPATH=. python tests/test_first_out_price.py
"""
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus

import layby.vllm.policy as P
from layby.vllm.state import STATE

STATE.chunk_tokens = 16
STATE.block_tokens = 16


def fresh():
    for v in STATE.ended.values():
        v.clear()
    STATE.live.acc.update(w_c=0.0, ne_c=0.0, age_c=0.0)
    for k in ("ev_c", "ce_c", "lc"):
        STATE.live.km[k][:] = 0.0
    pol = P.ParkCachePolicy(100)
    return pol


def add(pol, keys):
    for k in keys:
        pol.insert(k, ChunkStatus.__new__(ChunkStatus))
        pol.evictable.add(k)


now = STATE.now()

# a dropped session (first out) is evicted
pol = fresh()
drop = [("d", i) for i in range(4)]
add(pol, drop)
pol.apply(drop, -1.0)
got = [k for k, _ in pol.evict(4, set())]
assert sorted(got) == sorted(drop), got
STATE.flush_idle(now)
assert STATE.live.acc["w_c"] == 0.0, STATE.live.acc["w_c"]          # before: 64 (4 chunks x 16 tokens)
assert STATE.live.km["ev_c"].sum() == 0.0                           # before: 64 as eviction events
assert STATE.live.km["ce_c"].sum() == 64.0                          # censored instead
assert STATE.live.acc["ne_c"] == 4.0                                # still eviction events for first-out stays
assert pol.c_first == set(drop), pol.c_first
assert set(pol.c_evicted) == set(drop)

# a kept (LRU) session evicted under pressure counts as before
pol = fresh()
kept = [("k", i) for i in range(4)]
add(pol, kept)
got = [k for k, _ in pol.evict(2, set())]
assert len(got) == 2 and not pol.c_first, pol.c_first
STATE.flush_idle(now)
assert STATE.live.acc["w_c"] == 32.0, STATE.live.acc["w_c"]
assert STATE.live.km["ev_c"].sum() == 32.0
assert STATE.live.acc["ne_c"] == 2.0

# a re-inserted chunk is no longer first-out evicted
pol = fresh()
add(pol, drop)
pol.apply(drop, -1.0)
pol.evict(1, set())
k = next(iter(pol.c_first))
pol.insert(k, ChunkStatus.__new__(ChunkStatus))
assert k not in pol.c_first and k not in pol.c_evicted
print("first-out price ok")
