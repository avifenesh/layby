"""vLLM 0.30 adapter for the park cost rule.

Engine config (all modules load in the EngineCore process; put the repo on PYTHONPATH):
  --kv-transfer-config '{"kv_connector": "ParkConnector", "kv_connector_module_path": "layby.vllm.connector",
    "kv_role": "kv_both", "kv_connector_extra_config": {
      "spec_name": "TieringOffloadingSpec", "cpu_bytes_to_use": ..., "offload_prompt_only": false,
      "eviction_policy": "ParkCachePolicy", "cache_policy_module_path": "layby.vllm.policy",
      "secondary_tiers": [{"type": "ParkFsTier", "module_path": "layby.vllm.fs_tier", "root_dir": ...}],
      "park_port": 8765}}'

Pieces:
  state.py      the per-engine telemetry (layby.live.Live, layby.live.Speeds) and the hint queue
  connector.py  OffloadingConnector subclass: admission waits, queue lengths, step and link timing,
                GPU residency (through the block pool's metrics collector), miss losses, and the
                decision at request finish or when a late hint arrives
  policy.py     the CPU-tier eviction policy: predicted next use per chunk, CPU residency events
  fs_tier.py    the disk tier: writes only what the rule places on disk, also after the request ended
  server.py     a local HTTP port for hints ({key, surv}) and a telemetry snapshot (GET /live)

Baseline arms of the evaluation (default off; none of them changes Layby's own configuration):
  freq_tier.py  T2: vLLM's native fs tier writing a chunk only on its n-th sighting, counts decaying over time
                (SGLang write_through_selective, Dynamo KVBM's FrequencyFilter)
  continuum.py  CT: Continuum's GPU time-to-live pin after tool calls (layby.continuum), CPU tier only
  fs_tier.py    PT: ParkFsTier with "write_all": every chunk, no rule, to compare its I/O path with vLLM's
Curves for the K_km and K_ewma controls come from layby.curves (precomputed per turn, sent like any other curve).

A request is identified to the sidecar by kv_transfer_params["park_key"] (the request id otherwise). Its
idle-period curve arrives either with the request (kv_transfer_params["park_surv"], 15 values of
P(T > E2[j])) or after it finished (POST /hint), when the sidecar has scored the response.
"""
