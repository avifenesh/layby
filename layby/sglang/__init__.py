"""SGLang 0.5.21 adapter for the park cost rule. See park/sglang/README.md.

Pieces:
  plugin.py   the sglang.srt.plugins entry point: registers the park eviction policy, the park radix
              cache backend and the scheduler hooks
  state.py    per-scheduler state (Live, Speeds, hint queue, tick clock, predicted next use per node)
  policy.py   the eviction order: GPU LRU, host tier by predicted next use (first out for park, drop)
  cache.py    ParkRadixCache: residency and link telemetry, the decision, the storage placement filter
  hooks.py    scheduler hooks: the TP-broadcast tick (clock, placements), admission and miss losses,
              step timing
The hint port is park/vllm/server.py, served with this adapter's state.
"""
