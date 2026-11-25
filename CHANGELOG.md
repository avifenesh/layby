# Changelog

Newest first. Paper versions map to tags: arXiv v1 maps to `v1.0.0`, and Layby v2 maps to `v2.0`. The archive DOI is
copied here, into `CITATION.cff` and into the README once a tag is archived.

## 2.0.0 (2026-11-25)

Model weights: [Avifenesh/layby-dwell](https://huggingface.co/Avifenesh/layby-dwell) (`v2.0` on `main`; legacy 421M
`v1.0` archived at branch `v1` and tag `v1.0`).

- **Layby-Dwell v2 (`33.8M`, CPU-ready, zero `laya` dependency):** two-tower `v2_gated` architecture built on
  `jhu-clsp/ettin-encoder-32m` (10x384 ModernBERT + top-4 `LayerMix` + `[CLS]` + `AttentionPool`) and a
  `PeriodicNumericEncoder` over 32 causal server-view numeric features with gated bilinear residual fusion
  (`layby_dwell/v2.py`). 12.5x smaller (`135 MB` vs `1,685 MB`), runs directly on CPU in `10.6-29.8 ms` with
  `0 MiB` GPU VRAM (`4.1 ms` GPU BF16), cuts human `ECE@300s` by 3.3x (`0.0104` vs `0.0340`), and matches/beats v1
  on ReturnBench 33-cell (`0.660`, worst `1.039`; slow disk `0.817` vs `0.823`) and BurstGPT v2 OOD.
- **Offline config-only Hub loading & v1 revision support:** `DwellV2` builds its encoder from `encoder/config.json`
  (`AutoModel.from_config`) without downloading base weights; `Dwell(..., revision="v1")` loads archived v1 weights.
- **Cost-rule & live estimator upgrades (`layby/rule.py`, `layby/live.py`):** read-only disk counters
  (`n_disk_r`, `wait_disk_r`), GPU roofline `Speeds.f0()` filter, token-proportional `defer_x` with ghost decay,
  incremental delta `W`, and fast-disk cold-start tie-breaker.
- **vLLM 0.34.0 adapter (`layby/vllm/`):** step-scoped HIT protection (`_step_protected`, fixing the `cpu/manager.py:242`
  eviction race that crashed stock `write-through` on `GLM-5.3-Flash`), FCFS queue-ordered promotion serialization
  with CPU pin ceiling, hardware-gated fast-NVMe streaming, worker-thread I/O timing, fixed `_ReadyOnly.lookup()`,
  and 4-tier CPU eviction ordering (`27/27` unit tests passing).
- **SGLang 0.5.23 + HiCache adapter (`layby/sglang/`):** `dec_lock_ref`/`dec_host_lock_ref` idle residency hooks,
  clean `f0` excluding H->D load-back stalls and overlap idle ticks, pending write-through placement, prompt-ancestor
  fallback under `write_through_selective`, and `CPU:GPU` write-policy guidance (`23/23` unit tests passing).
- **Sidecar (`layby/sidecar/`):** auto-detects v2 (zero `laya`) and v1 checkpoints, bounded LRU tip tracking, and
  optional rolling per-kind online temperature recalibration (`--online-calib`).
- **Real-hardware results in README & paper:** RTX PRO 6000 Blackwell benchmarks across untouched held-out
  `GLM-5.3-Flash` 320B MoE TP=4 (`seed=2027`, `4.69x` faster mean than `no-disk`, `2.16x` faster than `LMCache
  0.5.7.post1 ARC`, `2.17x` faster than `freq-gated`, `8.60x` faster than `Continuum`), `Gemma-4-31B`,
  `Qwen3.8-27B`, and `SGLang 0.5.23 + HiCache` (`Qwen3-8B` across `2.0 GB/s`, `1.31 GB/s`, and `141 MB/s`).

## 1.0.0 (2026-10-10)

Archive: [10.5281/zenodo.23285214](https://doi.org/10.5281/zenodo.23285214) (all versions:
[10.5281/zenodo.23285213](https://doi.org/10.5281/zenodo.23285213)). Model weights: [10.57967/hf/10852](https://doi.org/10.57967/hf/10852).

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
