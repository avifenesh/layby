# Changelog

Newest first. Paper versions map to tags: arXiv v1 will be `v1.0.0`, tagged on the commit the arXiv v1 submission
was built from. The archive DOI is copied here, into `CITATION.cff` and into the README once the tag is archived.

## 1.0.0 (2026-10-10)

The release the paper describes:

- The cost rule (`layby/rule.py`), live estimators and the in-process profiler (`layby/live.py`, `layby/prof.py`).
- Engine adapters: vLLM 0.30 and 0.31 (`layby/vllm/`: `ParkConnector`, `ParkCachePolicy`, `ParkFsTier`, the hint
  server) and SGLang 0.5.21 (`layby/sglang/`: the `park` plugin, radix cache backend and eviction policy).
- The sidecar proxy (`layby/sidecar/`) that tracks sessions, scores them with Layby-Dwell and posts curves.
- Layby-Dwell inference (`layby_dwell/`), the model card, release file list and quantization study (`model/`).
  Weights are on the Hugging Face Hub as `avifenesh/layby-dwell`, not in git.
- ReturnBench: four content-free pools, the builders with pinned sources and hashes, source licenses (`bench/`), the
  simulator with baselines and the `returnbench` CLI (`sim/`), the simulator reference run (`reference/`).
- The real-engine harness (`engine/`): replay client, arm runners for vLLM and SGLang, the LMCache baseline arms,
  the smoke gate, pooling with a session bootstrap, and the three public workloads with their manifest.
- CPU tests for the adapters (`tests/`).
- Results in the README: Qwen3-8B on vLLM 0.30 and SGLang 0.5.21 (A100), GLM 5.3 Flash FP8 on vLLM 0.31.0
  (8x RTX PRO 6000), Qwen3-8B on vLLM 0.31.0 against LMCache (RTX PRO 6000), and the 33-cell simulator table.

Not released: the author's private agent replay (the public workloads in `engine/workloads/` stand in for it).
