"""park_gpu_order: a park or drop decision moves the session's free cached GPU blocks to the front of vLLM's free
queue, its tail first, and leaves blocks that were reused since the finish where they are.

Uses vLLM's own FreeKVCacheBlockQueue and KVCacheBlock. Run: PYTHONPATH=. python tests/test_gpu_order.py
"""
from types import SimpleNamespace as NS

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock

from layby.vllm.connector import ParkConnector as PC
from layby.vllm.state import STATE

blocks = [KVCacheBlock(i) for i in range(12)]
_hfield = "_block_hash" if hasattr(blocks[0], "_block_hash") else "block_hash"
for b in blocks:
    setattr(b, _hfield, f"h{b.block_id}".encode())
q = FreeKVCacheBlockQueue(list(blocks))           # LRU order: 0 is evicted first, 11 last
pool = NS(blocks=blocks, free_block_queue=q)

# Session S finished with blocks 8, 9, 10, 11 (its prefix first). Block 10 was then evicted and reused by another
# request's data before the decision applied: its hash changed, so it must not move.
session = [(b, blocks[b].block_hash) for b in (8, 9, 10, 11)]
q.remove(blocks[10]); setattr(blocks[10], _hfield, b"other"); q.append(blocks[10])
me = NS(_pool=pool, _gpu_demote=[session])
before = STATE.counts["gpu_demoted"]
PC._apply_gpu_order(me)

order = [b.block_id for b in q.get_all_free_blocks()]
assert order[:3] == [11, 9, 8], f"session tail-first at the front expected, got {order}"
assert order.index(10) > 3, f"a block reused since the finish moved: {order}"
assert STATE.counts["gpu_demoted"] - before == 3
assert q.num_free_blocks == 12 and me._gpu_demote == []
print("ok:", order)
