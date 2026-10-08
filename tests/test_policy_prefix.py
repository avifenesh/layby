"""ParkCachePolicy evicts a session's tail before its prefix, hinted or not (vLLM reuses only a contiguous prefix)."""
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
import layby.vllm.policy as P
from layby.vllm.state import STATE

STATE.chunk_tokens = 16
pol = P.ParkCachePolicy(100)
keys = [("s", i) for i in range(10)]
for k in keys:
    pol.insert(k, ChunkStatus.__new__(ChunkStatus)); pol.evictable.add(k)
pol.touch(keys, None)                                   # a request reuses the whole session
got = [k for k, _ in pol.evict(3, set())]
assert got == keys[::-1][:3], got
pol2 = P.ParkCachePolicy(100)
for k in keys:
    pol2.insert(k, ChunkStatus.__new__(ChunkStatus)); pol2.evictable.add(k)
pol2.apply(keys, -1.0)                                  # park/drop: first out, tail first
got = [k for k, _ in pol2.evict(4, set())]
assert got == keys[::-1][:4], got
print("prefix ok")
