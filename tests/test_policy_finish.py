"""A finished request's chunks go idle and take the cost rule's hint, on vLLM 0.30 and 0.31 alike.

vLLM 0.31 applies a request's recency at its finish (CachePolicy.on_request_finished, whose default calls touch).
Before the fix, that touch marked the finished request live again: its chunks never went idle, the hint the rule
gave them was skipped (apply() leaves in-use chunks alone) and overwritten by LRU, and the live table grew per request."""
from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

from layby.vllm.state import STATE

STATE.chunk_tokens = 16
m = CPUOffloadingManager(num_chunks=16, cache_policy="ParkCachePolicy", cache_policy_module_path="layby.vllm.policy")
pol = m._policy
keys = [make_offload_key(bytes([65 + i]) * 16, 0) for i in range(6)]
for turn in range(2):
    ctx = ReqContext(req_id=f"r{turn}", kv_transfer_params={"park_key": f"s-{turn}"})
    m.on_new_request(ctx)
    ks = keys[: 3 + 3 * turn]
    out = m.prepare_store(ks, ctx)
    m.complete_store(out.keys_to_store, ctx)
    # the connector decides as the request finishes: it hints the chunks and records park_eta on the request
    pol.apply(ks, -1.0)
    ctx.kv_transfer_params["park_eta"] = -1.0
    m.on_request_finished(ctx)
assert not pol.live, f"finished requests still live: {list(pol.live)}"
assert all(k in pol.idle_since for k in keys), "finished chunks not idle"
assert all(pol.hinted.get(k) == 1e18 for k in keys), "the rule's hint was lost at finish"
assert pol.apply(keys, 30.0) == len(keys), "a late hint skipped idle chunks"
print("finish ok")
