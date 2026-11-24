"""Unhinted ParkCachePolicy evicts in the same order as vLLM's LRUCachePolicy, including chunks whose store finished
after later inserts (they rank by when they became evictable)."""
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
import layby.vllm.policy as P
from layby.vllm.state import STATE

STATE.chunk_tokens = 16


def chunk(ref):
    c = ChunkStatus.__new__(ChunkStatus)
    c.ref_cnt = ref
    return c


def ins(pol, k, c):
    try:
        pol.insert(k, c, ReqContext(req_id="r"))
    except TypeError:
        pol.insert(k, c)


def drive(pol):
    ctx = ReqContext(req_id="r")
    for i in range(4):
        ins(pol, ("a", i), chunk(-1))            # being stored: not evictable yet
    for i in range(4):
        ins(pol, ("b", i), chunk(0))
    for i in range(4):
        pol.get(("a", i)).ref_cnt = 0            # a's store completes after b was inserted
        pol.mark_evictable(("a", i))
    if hasattr(pol, "touch"):
        pol.touch([("b", 0), ("b", 1)], None)
    else:
        pol.on_request_finished([[("b", 0), ("b", 1)]], set(), {("b", 0), ("b", 1)}, ctx)
    return [k for k, _ in pol.evict(8, set())]


want = drive(LRUCachePolicy(100))
got = drive(P.ParkCachePolicy(100))
assert got == want, (got, want)
print("lru parity ok", got)
