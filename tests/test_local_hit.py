"""Miss accounting must use the GPU-local prefix hit vLLM passed to the lookup.

vLLM 0.30/0.31 call the connector's update_state_after_alloc before they set request.num_computed_tokens, so a new
request reads 0 there. The adapter used that 0 as the local hit: every admission whose prompt held a block the GPU
had once evicted counted as a GPU miss (and, with a CPU policy, its once-evicted CPU chunks as CPU misses), which
inflated the miss rates the holding price is built from. A request fully served from the GPU must add no miss.
Run: PYTHONPATH=. python tests/test_local_hit.py [module]   (module default layby.vllm.connector)
"""
import importlib, sys, time
from types import SimpleNamespace as NS

mod = importlib.import_module(sys.argv[1] if len(sys.argv) > 1 else "layby.vllm.connector")
state_mod = importlib.import_module(mod.__name__.rsplit(".", 1)[0] + ".state")
STATE = state_mod.STATE
PC = mod.ParkConnector

STATE.hash_tokens = getattr(STATE, "hash_tokens", None) or STATE.block_tokens   # set by the connector's __init__
P = 4096
hashes = [f"h{i}".encode() for i in range(P // STATE.hash_tokens)]
now = time.monotonic()
for h in hashes:
    STATE.g_evicted[h] = now - 30.0            # every block was evicted once, 30 s ago, then recomputed and cached
STATE.policy = None

req = NS(request_id="r1", num_computed_tokens=0, num_prompt_tokens=P, block_hashes=hashes, kv_transfer_params={})
me = NS(_waiting={}, _deferred={}, _local={}, _giveups={}, connector_scheduler=NS(_req_status={}))
me._account_misses = lambda *a: PC._account_misses(me, *a)
me._waiting["r1"] = now - 0.01
# the scheduler's lookup saw the whole prompt as a GPU prefix hit (block-aligned), nothing external; this is what
# ParkConnector.get_num_new_matched_tokens records before it asks the offloading tiers
me._local["r1"] = P
before = STATE.live.acc["miss_g"]
PC._admit(me, req, 0)                       # update_state_after_alloc: request.num_computed_tokens is still 0
after = STATE.live.acc["miss_g"]
assert after == before, f"a fully GPU-served admission counted a GPU miss ({after - before:.3f} s): local hit read as 0"
print("ok: no miss for a GPU-served admission")
