"""CT: Continuum's TTL policy (layby.continuum) and its vLLM pin (layby.vllm.continuum), CPU only.

1. The TTL: T_default = ln(T + R) when T + R > 1, else 0 (eq. 2 with Exp(1) durations, eta = 1); eq. 2 on recorded
   durations equals a brute-force maximum over a fine grid; the record thresholds (K = 100) pick default, global and
   per-tool records as section 4.2 says; memoryfulness is 1 for fixed-length programs.
2. The pin, on vLLM 0.31's real BlockPool with the offloading connector's own hooks stubbed: a tool-call finish pins
   the request's blocks (out of the free queue after the scheduler frees the request); a human turn or a last request
   does not pin; the program's next request records the tool's duration and its admission ends the pin; an expired
   TTL ends a pin unless the program has a request waiting; when the oldest waiting request does not fit, pins go
   latest program arrival first, one per scheduler step in which vLLM looked a waiting request up, could not
   allocate it and ran nothing; the waiting program's own pin is a victim too (the Gemma stall of pilot 2).
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

# deadlock prevention (section 5.2): the scheduler looked up the head waiting request, could not allocate it and ran
# nothing, so a pin goes, latest program arrival first, one per step (vLLM retries the request next step)
def tried(r):
    """vLLM looked r up (its connector lookup finished) and did not allocate it: no update_state_after_alloc."""
    if r.request_id not in conn._waiting:
        conn.on_new_request(r)
    conn.get_num_new_matched_tokens(r, 0)


OffloadingConnector.get_num_new_matched_tokens = lambda self, r, n: (getattr(r, "ext", 0), False)

# the stall: a program's next request does not extend its pinned prefix (its prompt shrank) and needs more than is
# free; the pin's TTL expired, but its program waits, so only deadlock prevention can free it, the program's own pin
# included (the paper names no exception; an unpinned block stays in the prefix cache, evictable)
s0 = req("s0", "s-0", 50 * BT, park_kind="tool", park_tool="Read")
bs = run(s0); finish(s0, bs)
assert "s" in conn._pins and pool.get_num_free_blocks() == FREE0 - 50
clock[0] += conn._pins["s"]["tau"] + 1.0
s1 = req("s1", "s-1", 30 * BT, park_kind="tool", park_tool="Edit")   # 30 blocks > 13 free; old estimate 30 - 50 < 0
tried(s1)
step()
assert "s" not in conn._pins, f"the waiting program's own pin was kept while nothing could run: {set(conn._pins)}"
assert pool.get_num_free_blocks() == FREE0 and CC.CT_STATE.counts["unpin_pressure"] == 1
b1 = pool.get_new_blocks(30); conn.update_state_after_alloc(s1, NS(blocks=(b1,)), 0)   # admitted next step
conn._waiting.pop("s1", None); pool.free_blocks(reversed(b1))
assert pool.get_num_free_blocks() == FREE0

pins = {}
for s in ("x", "y", "z"):
    clock[0] += 0.01                                      # well inside the TTL (ln 2 s)
    r = req(f"{s}0", f"{s}-0", 160, park_kind="tool", park_tool="Grep")
    b = run(r); finish(r, b)
    pins[s] = b
free = pool.get_num_free_blocks()
assert free == FREE0 - 30
# a request that was scheduled this step, or a lookup still pending (None), releases nothing
w = req("w0", "w-0", (free + 15) * BT)
tried(w)
conn.build_connector_meta(NS(total_num_scheduled_tokens=7))       # something ran: vLLM waits for it to finish
assert set(conn._pins) == {"x", "y", "z"}, set(conn._pins)
w.ext = None
tried(w)
step()
assert set(conn._pins) == {"x", "y", "z"}, "a pending connector lookup is not an allocation failure"
del w.ext
step()                                                    # no lookup this step: nothing to relieve
assert set(conn._pins) == {"x", "y", "z"}
tried(w); step()
assert set(conn._pins) == {"x", "y"}, f"expected z (latest arrival) first, left {set(conn._pins)}"
tried(w); step()
assert set(conn._pins) == {"x"}, f"expected y next, left {set(conn._pins)}"
bw = pool.get_new_blocks(free + 15)                       # now it fits: admitted, x keeps its pin
conn.update_state_after_alloc(w, NS(blocks=(bw,)), 0)
step()
assert set(conn._pins) == {"x"} and CC.CT_STATE.counts["unpin_pressure"] == 3
print("pin ok: tool turns pinned for the TTL, unpinned at admission, at expiry, and latest-arrival first under pressure")
print("CT ok")
