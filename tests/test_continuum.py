"""CT: Continuum's TTL policy (layby.continuum) and its vLLM pin (layby.vllm.continuum), CPU only.

1. The TTL: T_default = ln(T + R) when T + R > 1, else 0 (eq. 2 with Exp(1) durations, eta = 1); eq. 2 on recorded
   durations equals a brute-force maximum over a fine grid; the record thresholds (K = 100) pick default, global and
   per-tool records as section 4.2 says; memoryfulness is 1 for fixed-length programs.
2. The pin, on vLLM 0.31's real BlockPool with the offloading connector's own hooks stubbed: a tool-call finish pins
   the request's blocks (out of the free queue after the scheduler frees the request); a human turn or a last request
   does not pin; the program's next request records the tool's duration and its admission ends the pin; an expired
   TTL ends a pin unless the program has a request waiting; when the oldest waiting request does not fit, pins go
   latest program arrival first, and the waiting program's own pin stays.
Run: PYTHONPATH=. python tests/test_continuum.py
"""
import math
from types import SimpleNamespace as NS

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.core.block_pool import BlockPool

from layby.continuum import ContinuumTTL, best_ttl, default_ttl, memoryfulness
import layby.vllm.continuum as CC

# --- 1. the TTL -------------------------------------------------------------------------------------------------
assert default_ttl(0.0, 0.5) == 0.0 and default_ttl(0.4, 0.6) == 0.0
assert abs(default_ttl(2.0, 3.0) - math.log(5.0)) < 1e-12
grid = np.linspace(0, 20, 200001)
B = 5.0
assert abs(default_ttl(0.0, B) - grid[np.argmax((1 - np.exp(-grid)) * B - grid)]) < 1e-3

rng = np.random.default_rng(0)
for trial in range(50):
    xs = rng.lognormal(0.5, 1.2, rng.integers(1, 300))
    Bv = float(rng.uniform(0.1, 30))
    tau = best_ttl(xs, Bv)
    cand = np.concatenate([[0.0], np.unique(xs)])
    gain = np.searchsorted(np.sort(xs), cand, side="right") / len(xs) * Bv - cand
    assert abs((np.searchsorted(np.sort(xs), tau, side="right") / len(xs) * Bv - tau) - max(gain.max(), 0.0)) < 1e-9
    # no tau on a fine grid does better (the ECDF is a step function: the optimum sits on a recorded duration)
    g = np.linspace(0, xs.max() * 1.1, 4001)
    assert (np.searchsorted(np.sort(xs), g, side="right") / len(xs) * Bv - g).max() <= max(gain.max(), 0.0) + 1e-9
assert best_ttl([5.0, 6.0], 1.0) == 0.0                   # pinning never pays: tau = 0

c = ContinuumTTL(eta=0.5)
assert c.source("Bash") == "default"
assert abs(c.ttl("Bash", 2.0, 3.0) - math.log(5.0)) < 1e-12   # cold start ignores eta (the paper sets eta = 1 there)
for i in range(101):
    c.record("Read", 1.0)
assert c.source("Bash") == "global" and c.source("Read") == "tool"
for i in range(60):
    c.record("Bash", 30.0)
assert c.source("Bash") == "global"                       # 60 records <= K
assert c.ttl("Bash", 2.0, 3.0) == 1.0                     # global: P(1) = 101/161, gain 0.627 * 4 - 1 > 0; 30 s loses
for i in range(41):
    c.record("Bash", 30.0)
assert c.source("Bash") == "tool" and c.ttl("Bash", 2.0, 3.0) == 0.0     # every Bash call takes 30 s: B = 4 < 30
assert c.ttl("Bash", 60.0, 10.0) == 30.0                  # B = 40 > 30
assert abs(memoryfulness([5, 5, 5]) - 1.0) < 1e-9 and abs(memoryfulness([dict(turns=[0] * 4)] * 3) - 1.0) < 1e-9
print("ttl ok: eq. 2, T_default, record thresholds, memoryfulness")

# --- 2. the pin ---------------------------------------------------------------------------------------------------
OffloadingConnector.on_new_request = lambda self, r: None
OffloadingConnector.get_num_new_matched_tokens = lambda self, r, n: (0, False)
OffloadingConnector.update_state_after_alloc = lambda self, r, b, n: None
OffloadingConnector.request_finished = lambda self, r, b: (False, None)
OffloadingConnector.request_finished_all_groups = lambda self, r, b: (False, None)
OffloadingConnector.build_connector_meta = lambda self, so: NS(load_jobs={})
OffloadingConnector.update_connector_output = lambda self, o: None

BT = 16
clock = [100.0]
pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=BT)
# the test sets block hashes by hand, outside the pool's hash map: reallocation just drops them
pool._remove_cached_block_hashes = lambda b: (b.reset_hash(), [])[1]
conn = object.__new__(CC.ContinuumConnector)
conn._setup({"ct_eta": 1.0}, BT, NS(num_blocks=64, kv_cache_tensors=[NS(size=64 * BT * 1000)]), BT, clock=lambda: clock[0])
conn._pool = pool
assert conn.bytes_per_token == 1000.0
FREE0 = pool.get_num_free_blocks()


def req(rid, key, tokens, **kw):
    return NS(request_id=rid, num_tokens=tokens, kv_transfer_params=dict(park_key=key, **kw))


def run(r, local=0):
    """Arrive, look up, get blocks (a GPU hit on `local` tokens of earlier blocks is not modelled: new blocks)."""
    conn.on_new_request(r)
    conn.get_num_new_matched_tokens(r, local)
    blocks = pool.get_new_blocks(-(-r.num_tokens // BT))
    for i, b in enumerate(blocks):
        if (i + 1) * BT <= r.num_tokens:                  # full blocks enter the prefix cache, a partial tail does not
            b.set_block_hash(f"{r.request_id}-{i}".encode(), BT)
    conn.update_state_after_alloc(r, NS(blocks=(blocks,)), 0)
    return blocks


def finish(r, blocks):
    conn.request_finished(r, [b.block_id for b in blocks])
    pool.free_blocks(reversed(blocks))                    # the scheduler frees the request after the connector hook


def step():
    conn.build_connector_meta(NS())


# a tool turn pins: T = 0 and R = 160 tokens * 1000 B / 20 GB/s, so T + R < 1 and the cold-start TTL is 0; make the
# measured reload slow (1 s per 160 tokens) so that T_default = ln(R) > 0
conn._load_s, conn._load_tok = 2.0, 160.0                 # 12.5 ms per token: R = 2 s for 160 tokens
a0 = req("a0", "a-0", 160, park_kind="tool", park_tool="Bash")
ba = run(a0)
finish(a0, ba)
tau = math.log(2.0)
assert "a" in conn._pins and abs(conn._pins["a"]["tau"] - tau) < 1e-9, conn._pins
assert all(b.ref_cnt == 1 for b in ba), [b.ref_cnt for b in ba]
assert pool.get_num_free_blocks() == FREE0 - len(ba), "pinned blocks are still in the free queue"
assert CC.CT_STATE.dlog[-1]["src"] == "default"

# a human turn and a last request do not pin
h0 = req("h0", "h-0", 64, park_kind="human")
bh = run(h0); finish(h0, bh)
l0 = req("l0", "l-0", 64, park_kind="tool", park_tool="Bash", park_last=True)
bl = run(l0); finish(l0, bl)
assert "h" not in conn._pins and "l" not in conn._pins
assert pool.get_num_free_blocks() == FREE0 - len(ba)

# the TTL holds the pin until it expires; the next request records the tool's duration and its admission unpins
clock[0] += tau * 0.5
step()
assert "a" in conn._pins
a1 = req("a1", "a-1", 176, park_kind="tool", park_tool="Bash")
conn.on_new_request(a1)
assert len(conn.ttl.by_tool["Bash"]) == 1 and abs(conn.ttl.by_tool["Bash"][0] - tau * 0.5) < 1e-9, conn.ttl.by_tool
clock[0] += tau                                           # expired while the program's request waits: kept
step()
assert "a" in conn._pins, "a pin was dropped while its program's request waited"
pool.touch(ba)                                            # the admission's prefix hit takes its own references
conn.get_num_new_matched_tokens(a1, 160)
conn.update_state_after_alloc(a1, NS(blocks=([],)), 0)
assert "a" not in conn._pins and all(b.ref_cnt == 1 for b in ba)
assert CC.CT_STATE.counts["unpin_admit"] == 1
assert CC.CT_STATE.rlog[-1]["evicted"] is False           # 160 of 160 tokens hit on the GPU
pool.free_blocks(reversed(ba))
assert pool.get_num_free_blocks() == FREE0

# an expired pin of a program with nothing waiting goes at the next step
b0 = req("b0", "b-0", 640, park_kind="tool", park_tool="Edit")       # R = 8 s: TTL ln 8
bb = run(b0); finish(b0, bb)
assert "b" in conn._pins
clock[0] += conn._pins["b"]["tau"] + 0.01
step()
assert "b" not in conn._pins and pool.get_num_free_blocks() == FREE0 and CC.CT_STATE.counts["unpin_ttl"] == 1

# the partial tail block (no prefix-cache hash) is not held
q0 = req("q0", "q-0", 170, park_kind="tool", park_tool="Edit")
bq = run(q0); finish(q0, bq)
assert len(conn._pins["q"]["blocks"]) == 10 and bq[-1].ref_cnt == 0 and all(b.ref_cnt == 1 for b in bq[:-1])
clock[0] += conn._pins["q"]["tau"] + 0.01
step()
assert "q" not in conn._pins and pool.get_num_free_blocks() == FREE0

# deadlock prevention: x then y then z pinned (program arrival order); a waiting request needs more than is free
pins = {}
for s in ("x", "y", "z"):
    clock[0] += 0.01                                      # well inside the TTL (ln 2 s)
    r = req(f"{s}0", f"{s}-0", 160, park_kind="tool", park_tool="Grep")
    b = run(r); finish(r, b)
    pins[s] = b
free = pool.get_num_free_blocks()
assert free == FREE0 - 30
big = req("w0", "x-1", (free + 5) * BT, park_kind="tool", park_tool="Grep")   # x's next turn: its own pin covers 10
conn.on_new_request(big)
conn.get_num_new_matched_tokens(big, 0)
step()
# need = free + 5 - 10 (x's own pin) < free: fits without unpinning anyone
assert set(conn._pins) == {"x", "y", "z"}, set(conn._pins)
conn._waiting.clear()
big2 = req("w1", "w-0", (free + 15) * BT)                 # a new program: needs 15 blocks more than free
conn.on_new_request(big2)
step()
assert set(conn._pins) == {"x"}, f"expected z then y unpinned (latest arrival first), left {set(conn._pins)}"
assert CC.CT_STATE.counts["unpin_pressure"] == 2
assert pool.get_num_free_blocks() == free + 20
print("pin ok: tool turns pinned for the TTL, unpinned at admission, at expiry, and latest-arrival first under pressure")
print("CT ok")
