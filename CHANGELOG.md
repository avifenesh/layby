# Changelog

Newest first. Paper versions map to tags: arXiv v1 maps to `v1.0.0`, and Layby v2 maps to `v2.0`. The archive DOI is
copied here, into `CITATION.cff` and into the README once a tag is archived.

## 2.1.1 (2026-11-26)

- **Paper-reproducibility artifacts (`data/`, `figs.py`, release asset `layby.pdf`):** added `data/v2_results.json`, `data/returnbench_v2_s0.json`, `data/ablation_rows.jsonl`, `data/baselines_rows.jsonl`, and `figs.py` (with repository-relative path resolution) so every paper table and figure regenerates directly from the release. Attached `layby.pdf` to the GitHub release.
- **Audit-verified reporting (`README.md`, `model/MODEL_CARD.md`, `data/v2_results.json`):** removed the invalid `layby+calib` benchmark comparison after verifying `engine/replay.py` calls `/park/score` (not `/v1/chat/completions`), leaving `OnlineCalib` untouched (`boundaries = 0`, bit-identical curves). Pooled identical repeats honestly as multi-repeat `layby` averages alongside repeat spreads, aligned all-turn linear vs. resumed-turn nearest-rank table conventions, aligned ReturnBench Table 8 / Fig. 7 / Table 12 to published 33.8M v2 curves (`data/returnbench_v2_s0.json`), aligned full `eval_uhold` (`n=184,148`) ablations, and scoped CPU latencies across server, ARM64, laptop, and isolated 8-thread runs.
- **Online calibration bin-edge parity (`layby/sidecar/proxy.py`, `tests/test_dwell_v2_sidecar.py`):** fixed `np.searchsorted(E2, gap, side="left")` in `OnlineCalib.observe_return()` to match training bin boundaries.
- **Analyzer CLI options (`engine/analyze_arms.py`):** added `--all-turns` and `--pct {linear,nearest}` flags so both all-turn linear and resumed-turn nearest-rank table conventions reproduce directly from raw JSONL logs.

## 2.1.0 (2026-11-26)

- **vLLM nightly (`>=0.34.0rc1`) compatibility (`layby/vllm/connector.py`):** added `_fallback_iteration_details` fallback when `vllm.v1.utils.compute_iteration_details` is absent on vLLM nightly (reads `SchedulerOutput.iteration_details` or derives context request/token counts from scheduled new and chunked cached requests), covered by `test_nightly_iteration_details_fallback` in `tests/test_layby_v2_fixes.py`.
- **Arm runner sidecar forwarding (`engine/run_arms.sh`, `engine/run_arms_sglang.sh`):** forward `${SIDECAR_EXTRA:-}` to `layby.sidecar.proxy`.
- **Cross-hardware and ablation evaluations:** added 1x RTX PRO 6000, 1x RTX 5090 Laptop 24 GB, and NVIDIA DGX Spark GB10 ARM64 (`Inkling 2 Mini` 26.4B NVFP4) evaluations plus model, slow-disk, and fast-NVMe ablations (superseded in `v2.1.1` with pooled multi-repeat `layby` reporting after confirming `--online-calib` is a no-op under `/park/score`).

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
