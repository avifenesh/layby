"""_request_keys merges every KV group's chunk keys in prefix order by relative position."""
from types import SimpleNamespace as NS
from layby.vllm.connector import _request_keys

one = NS(group_states=(NS(offload_keys=["a0", "a1", "a2"]),))
assert _request_keys(one) == ["a0", "a1", "a2"]
two = NS(group_states=(NS(offload_keys=["m0", "m1", "m2", "m3"]), NS(offload_keys=["k0", "k1"])))
got = _request_keys(two)
assert set(got) == {"m0", "m1", "m2", "m3", "k0", "k1"} and len(got) == 6, got
assert got.index("k0") < got.index("m3") and got.index("m0") < got.index("k1"), got
assert got[-1] == "k1" and got[-2] == "m3", got      # both groups' tails come last
print("request keys ok", got)
