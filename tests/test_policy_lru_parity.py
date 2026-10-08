"""Unhinted ParkCachePolicy evicts in the same order as vLLM's LRUCachePolicy, including chunks whose store finished
after later inserts (they rank by when they became evictable)."""
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
import layby.vllm.policy as P
from layby.vllm.state import STATE

STATE.chunk_tokens = 16


def chunk(ref):
    c = ChunkStatus.__new__(ChunkStatus)
    c.ref_cnt = ref
    return c


def drive(pol):
    for i in range(4):
        pol.insert(("a", i), chunk(-1))          # being stored: not evictable yet
    for i in range(4):
        pol.insert(("b", i), chunk(0))
    for i in range(4):
        pol.get(("a", i)).ref_cnt = 0            # a's store completes after b was inserted
        pol.mark_evictable(("a", i))
    pol.touch([("b", 0), ("b", 1)], None)
    return [k for k, _ in pol.evict(8, set())]


want = drive(LRUCachePolicy(100))
got = drive(P.ParkCachePolicy(100))
assert got == want, (got, want)
print("lru parity ok", got)
