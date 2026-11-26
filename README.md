# Layby

Layby decides where an idle LLM session's KV cache waits until the session comes back: on the GPU, in CPU RAM, on
disk, or nowhere. Agent and chat sessions go idle between turns for seconds to minutes. Their KV is large, and
recomputing it on return is what makes the next turn slow.

Model: [Avifenesh/layby-dwell](https://huggingface.co/Avifenesh/layby-dwell) on Hugging Face (`v2.0` on `main`; legacy `v1.0` archived at branch `v1`). Paper: arXiv, link to
follow. Author: [Avi Fenesh](https://github.com/avifenesh).

Layby has four parts:

- **A cost rule** (`layby/rule.py`). For each finished turn it compares the expected cost of keeping the session's
  KV in RAM only, also writing it to disk, moving it to disk and out of RAM first, or dropping it. Every term is a
  probability from a return-time curve times a cost the engine measures live: link speeds, queue waits, prefill
  speed, eviction rates and residency. There are no tuned constants.
- **Layby-Dwell v2**, the curve. A standalone **33.8M** (`135 MB`) two-tower model built on
  `jhu-clsp/ettin-encoder-32m` with zero `laya` dependency. It reads what a serving engine sees of a session (the
  tool calls and their arguments, the last assistant texts and human message, token counts, timing) and returns
  `P(next request later than t)` at 15 horizons from 0.5 s to 30 min. It runs directly on CPU next to the engine
  (`10.6-19.4 ms` p50 live, **0 MiB GPU VRAM**; or `4.1 ms` on GPU BF16), so text never leaves the host; the rule
  itself only needs the curve. Weights: [Avifenesh/layby-dwell](https://huggingface.co/Avifenesh/layby-dwell); model
  card and release notes: `model/`. Inference: `layby_dwell/`.
- **Engine adapters** for vLLM 0.30, 0.31, 0.34, and nightly (`>=0.34.0rc1`) (`layby/vllm/`) and SGLang 0.5.21 and
  0.5.23 (`layby/sglang/`), plus a sidecar proxy (`layby/sidecar/`) that tracks sessions, scores them on CPU or GPU,
  optionally adapts per-kind calibration online (`--online-calib`), and sends the curves to the engine.
- **ReturnBench**, a benchmark on public agent and chat traces with a simulator and baselines (`bench/`, `sim/`), and
  the real-engine harness (`engine/`).

## Install

    pip install -e .                 # rule, simulator, ReturnBench
    pip install -e ".[dwell]"        # Layby-Dwell v2 inference (torch, transformers, safetensors; zero laya)
    pip install -e ".[dwell-v1]"     # optional: legacy laya package for archived v1 (421M) weights
    pip install -e ".[sidecar]"      # the sidecar proxy
    pip install -e ".[engine]"       # the real-engine replay client

Tested on: Linux x86_64 and Linux ARM64 (DGX Spark GB10). The rule, simulator, ReturnBench, and `Layby-Dwell v2`
inference run on CPU (Python 3.10+). The vLLM adapter ran inside `vllm/vllm-openai:v0.30.0`, `v0.31.0`, `v0.34.0`, and
nightly `0.34.0rc1` (with `lmcache 0.5.7.post1` for comparisons); the SGLang adapter ran inside
`lmsysorg/sglang:v0.5.21` and `lmsysorg/sglang:v0.5.23-cu130-runtime`. GPUs: NVIDIA A100 40/80 GB, NVIDIA RTX PRO 6000
Blackwell 96 GB (1x and 4x TP=4), NVIDIA RTX 5090 Laptop GPU 24 GB, and NVIDIA DGX Spark (GB10, 121 GiB unified memory,
ARM64) on Ubuntu 24.04 with CUDA 13.0.

## Quick start: the bench on a CPU

Two cells of ReturnBench, under a minute on 8 cores (`bench/README.md`):

    returnbench eval --rules wt cost --cells wildchat:120 swechat:18 --workloads 2 --seeds 1

It prints this (its rows are a subset of the shipped reference run, `reference/rows.jsonl`, and match it bit for bit).
In the tool's output, `C0` is no-disk, `wt` is write-through and `cost` is Layby's cost rule:

    geomean over cells of the p95 TTFT ratio (per cell: geomean over workloads)
    rule      cells  p95/C0  worst  @0.5  @1.5  @3   p95/wt  @0.5  @1.5  @3
    C0            6    1.00   1.00  1.00  1.00  1.00     1.80  0.98  2.27  2.63
    wt            6    0.55   1.04  1.02  0.44  0.38     1.00  1.00  1.00  1.00
    cost          6    0.55   0.84  0.80  0.47  0.44     0.99  0.79  1.07  1.15

The full 33-cell table takes about 6 minutes on 8 cores; its command, expected output and checker are in
`reference/README.md`. On a GPU box, `engine/smoke.sh` is the install proof for the vLLM adapter: it boots the server
with the adapter in write-through mode and runs `engine/smoke_restore.py`, whose client log (`$OUT/SM.client.log`)
ends with the line `PASS` when a session evicted from the GPU comes back from the CPU tier and then from disk with
the same greedy output, or `FAIL`.

## Quick start: vLLM

Layby replaces the offloading connector, its CPU-tier policy and its disk tier:

    vllm serve MODEL --enable-prefix-caching --kv-transfer-config '{
      "kv_connector": "ParkConnector", "kv_connector_module_path": "layby.vllm.connector", "kv_role": "kv_both",
      "kv_connector_extra_config": {"cpu_bytes_to_use": 30064771072, "offload_prompt_only": false,
        "eviction_policy": "ParkCachePolicy", "cache_policy_module_path": "layby.vllm.policy",
        "spec_name": "TieringOffloadingSpec",
        "secondary_tiers": [{"type": "ParkFsTier", "module_path": "layby.vllm.fs_tier", "root_dir": "KV_DIR",
                             "n_read_threads": 16, "n_write_threads": 16}],
        "park_port": 8765}}'

`KV_DIR` is a directory on the disk tier (the arm runners empty it before each arm). Then put the sidecar in front of
the server. With `Layby-Dwell v2` (`33.8M`), run it on CPU (`--device cpu`) for **0 MiB GPU VRAM** overhead:

    python -m layby.sidecar.proxy --upstream http://127.0.0.1:8000 --listen 8100 \
      --ckpt DWELL/model.safetensors --calib DWELL/config.json --device cpu \
      --engine vllm --hint-url http://127.0.0.1:8765/hint

Clients talk to port 8100. A client that already knows its curve can skip the sidecar and send it with the request
(`kv_transfer_params: {"park_key": ..., "park_surv": [15 values]}`) or post it later (`POST /hint`).

## Quick start: SGLang

Install this repo into SGLang's environment. That registers the SGLang plugin `park`:

    SGLANG_PLUGINS=park SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python \
    SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=KV_DIR \
    sglang serve --model-path MODEL --enable-hierarchical-cache \
      --hicache-write-policy write_through_selective --hicache-storage-backend file \
      --radix-cache-backend park --radix-eviction-policy park --radix-eviction-policy-config '{"port": 8765}'

Use `write_through_selective` when host RAM is close to GPU KV (`CPU <= 1.5x GPU`), or `write_through` when host RAM
has headroom (`CPU >= 2x GPU`). The sidecar works the same way with `--engine sglang`.

## Results

### Layby v2 real-hardware results (vLLM 0.34.0, LMCache 0.5.7.post1 ARC, Continuum, SGLang 0.5.23 + HiCache)

All v2 runs below use the published **Layby-Dwell v2 (`33.8M`)** checkpoint running on CPU (**0 MiB extra GPU VRAM**,
`10.6-19.4 ms` p50 live latency, `0` hint errors across >8,000 turns) on a 4x NVIDIA RTX PRO 6000 Blackwell 96 GB
node (`64` vCPU, `923 GiB` RAM, local NVMe measuring `9.5-12.5 GB/s` O_DIRECT read). All arms run inside `cgroup v2`
isolation with O_DIRECT (vLLM) or `sync_file_range` + `posix_fadvise(DONTNEED)` (SGLang) so disk reads hit real
storage bandwidth rather than the OS page cache.

The arms:

- **no-disk**: stock vLLM or SGLang with a CPU tier and no disk tier (LRU).
- **write-through**: stock vLLM writing every chunk through to the disk tier (LRU).
- **freq-gated**: vLLM disk writes gated by chunk reuse count (`min_hits=2` or `3`, 600 s decay).
- **lmcache**: LMCache `0.5.7.post1` with ARC L1/L2 eviction and disk tier (`vllm/vllm-openai:v0.34.0`).
- **continuum**: Continuum (arXiv `2511.02230`), GPU TTL pinning after tool calls, CPU tier as no-disk, no disk.
- **hicache-selective / hicache-file / hicache-write-back**: SGLang `0.5.23` HiCache with file L3 storage under
  `write_through_selective`, `write_through`, or `write_back`.
- **layby-v2**: Layby v2 cost rule + adapters with live CPU `Layby-Dwell v2` (`33.8M`) curves.

#### 1. Untouched strictly disjoint held-out benchmark: `GLM-5.3-Flash` 320B MoE TP=4 (`vLLM 0.34.0`, `seed=2027`)

24 strictly disjoint held-out sessions (`0/24` session overlap with development workloads, `0%` overlap with
Layby-Dwell training, `398` turns, `p50 max_ctx = 123K`, `max = 190K`), 4x RTX PRO 6000 TP=4 (`16 GiB` GPU KV,
`10 GiB` CPU KV, native NVMe `12.5 GB/s`), frozen v2 code:

| arm | served (err) | p50 TTFT | p95 TTFT | p99 TTFT | mean TTFT | max TTFT | recomputed tok | disk written | peak VRAM / GPU | makespan |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| continuum (no disk) | 398 (0) | 195.28 s | 293.49 s | 303.63 s | 167.91 s | 312.17 s | 32.93M | 0.0 GB | 92,026 MiB | 79.33 m |
| no-disk | 398 (0) | 103.90 s | 166.26 s | 174.20 s | 91.57 s | 176.71 s | 33.11M | 0.0 GB | 92,026 MiB | 63.81 m |
| freq-gated | 398 (0) | 45.15 s | 91.25 s | 119.20 s | 42.31 s | 143.92 s | 15.21M | 90.3 GB | 92,540 MiB | 40.93 m |
| lmcache (0.5.7.post1 ARC) | 398 (0) | 33.51 s | 111.98 s | 122.18 s | 42.23 s | 129.69 s | 16.28M | 222.3 GB | 92,889 MiB (+0.9 GiB) | 43.07 m |
| **layby-v2 (CPU 33.8M)** | **398 (0)** | **16.47 s** | **48.92 s** | **52.46 s** | **19.52 s** | **63.12 s** | **9.00M** | 151.8 GB | **92,026 MiB (+0)** | **34.42 m** |

On this untouched 320B MoE workload, **layby-v2** is **4.69x faster mean** (`3.40x` p95, `1.85x` makespan) than
`no-disk`, **2.16x faster mean** (`2.29x` p95, `31.7%` less disk, `-3.5 GiB` total VRAM) than `lmcache 0.5.7.post1 ARC`,
**2.17x faster mean** (`15.3x` fewer allocation failures) than `freq-gated`, and **8.60x faster mean** than `continuum`.

#### 2. Full `vLLM 0.34.0` multi-model matrix (2 frozen repeats per cell)

| workload (model, disk regime, turns) | no-disk mean / p95 | write-through mean / p95 | lmcache ARC mean / p95 | freq-gated mean / p95 | continuum mean / p95 | **layby-v2 (2-run avg) mean / p95** |
|---|---:|---:|---:|---:|---:|---:|
| `glm53-fast-real` `seed=2026` (GLM-5.3-Flash 320B TP=4, native NVMe `11.6 GB/s`, 437t) | 108.86 s / 192.22 s | crashed turn 76* | 73.73 s / 157.72 s | 49.19 s / 101.70 s | cap timeout (44 cens) | **29.06 s / 66.07 s** (35.6 m wall) |
| `gemma-fast-heavy` (Gemma-4-31B FP8, fast NVMe `1.35 GB/s`, 443t) | 7.32 s / 38.26 s | 3.85 s / 18.88 s | 5.66 s / 28.71 s | 4.42 s / 22.10 s | 7.24 s / 40.96 s | **3.81 s / 17.87 s** (#1 every metric) |
| `qwen-fast-heavy` (Qwen3.8-27B FP8, fast NVMe `1.31 GB/s`, 488t) | 2.43 s / 10.49 s | **1.83 s / 7.39 s** | 2.19 s / 9.12 s | 1.89 s / 8.35 s | 2.76 s / 13.01 s | **1.85 s / 7.69 s** (ties write-through) |
| `qwen-slow-pressure` (Qwen3.8-27B FP8, slow disk `141 MB/s`, 366t) | 12.46 s / 39.50 s | 12.72 s / 58.21 s | 11.03 s / 40.53 s | 15.17 s / 62.94 s | 14.68 s / 48.50 s | **10.58 s / 35.39 s** (`4.9x` less disk, `0` afal) |
| `gemma-slow-pressure` (Gemma-4-31B FP8, slow disk `146 MB/s`, 321t) | 24.80 s / 82.20 s | 24.19 s / 97.93 s | cap timeout (5 cens, 190 s p95) | 26.52 s / 84.54 s | 27.32 s / 90.44 s | 26.19 s / **76.82 s** (r1 67.6 s / r2 86.0 s) |

*\*Stock `vLLM 0.34.0` `write-through` crashed fatally at turn 76 (`AssertionError: Chunk ... not found in cache` in `cpu/manager.py:242`); `layby-v2`'s step-scoped HIT protection prevented the race with 0 errors across all repeats.*

#### 3. `SGLang 0.5.23 + HiCache` real-hardware matrix (`Qwen/Qwen3-8B` bf16, `page=64`, `fadvise_sfr`)

| regime (NVMe cap, GPU / CPU KV, turns) | arm | mean TTFT | p95 TTFT | p99 TTFT | max TTFT | recomputed tok | disk written |
|---|---|---:|---:|---:|---:|---:|---:|
| **Gen4 NVMe KV-pressure** (`2.0 GB/s`, 5 / 5 GiB, 366t) | hicache-selective (LRU) | 3.60 s | 13.33 s | 17.23 s | 21.55 s | 2.40M | 49.9 GB |
| **Gen4 NVMe KV-pressure** (`2.0 GB/s`, 5 / 5 GiB, 366t) | **layby-v2 (r1 / r2, CPU 33.8M)** | **3.27 / 3.40 s** | **11.91 / 13.53 s** | **16.39 / 16.98 s** | **20.92 / 19.63 s** | **2.31 / 2.28M** | **49.2 / 47.5 GB** |
| **Capped `1.31 GB/s` KV-pressure** (5 / 5 GiB, 366t) | no-disk-selective (LRU) | 3.85 s | 14.21 s | 18.44 s | 21.97 s | 2.67M | 0.0 GB |
| **Capped `1.31 GB/s` KV-pressure** (5 / 5 GiB, 366t) | hicache-write-back (LRU) | OOM turn 21 | crashed | crashed | crashed | - | 10.9 GB |
| **Capped `1.31 GB/s` KV-pressure** (5 / 5 GiB, 366t) | hicache-selective (LRU) | **3.33 s** | 13.61 s | 17.47 s | 19.75 s | **2.22M** | 50.3 GB |
| **Capped `1.31 GB/s` KV-pressure** (5 / 5 GiB, 366t) | **layby-v2 (r1 / r2)** | 3.53 / 3.82 s | **13.50** / 14.18 s | **17.23 / 17.45 s** | 21.54 / 21.50 s | 2.35 / 2.52M | **42.8 / 37.4 GB** |
| **Slow disk `141 MB/s`** (5 / 10 GiB, 366t) | hicache-file (`write_through`) | 3.28 s | 12.69 s | 16.23 s | 18.28 s | 1.80M | 94.2 GB |
| **Slow disk `141 MB/s`** (5 / 10 GiB, 366t) | hicache-selective | 3.43 s | 13.80 s | 17.51 s | 19.80 s | 2.17M | 52.8 GB |
| **Slow disk `141 MB/s`** (5 / 10 GiB, 366t) | hicache-write-back | OOM turn 21 | crashed | crashed | crashed | - | - |
| **Slow disk `141 MB/s`** (5 / 10 GiB, 366t) | no-disk (`write_through`) | **2.88 s** | **11.26 s** | 18.47 s | 22.72 s | 1.36M | 0.0 GB |
| **Slow disk `141 MB/s`** (5 / 10 GiB, 366t) | **layby-v2 (`write_through`, auto 0 GB)** | 3.00 s | 12.75 s | 18.08 s | 22.49 s | **1.31M** | **0.0 GB** |

On SGLang `0.5.23`, `layby-v2` beats `hicache-selective` across every metric on Gen4 NVMe (`2.0 GB/s`, where L3 read
beats 8B bf16 GPU recompute), cuts disk writes `15-26%` at `1.31 GB/s`, and automatically suppresses 100% of disk
writes (`0.0 GB`, lowest recompute of all 7 arms) on `141 MB/s` slow disk, with `0` crashes while `hicache-write-back`
OOM-crashed at turn 21 under both KV pressure and slow disk.

#### 4. `Layby-Dwell v2` (`33.8M`) vs archived `v1` (`421M`)

| metric (untouched held-out splits, 3-seed medians) | archived `v1` (`421M`) | **`Layby-Dwell v2` (`33.8M`)** |
|---|---:|---:|
| parameters / safetensors size | 421M / 1,685 MB | **33.8M / 135 MB (12.5x smaller)** |
| CPU latency (`bs=1`) / peak VRAM | 1.7-2.9 s (GPU required) / 2,616 MB | **10.6-29.8 ms / 0 MB on CPU** (`427 MB` GPU BF16) |
| official 33-cell ReturnBench (`p95` vs no-disk) | 0.659 (worst 1.043, 0.5 GB/s: 0.823) | **0.660 (worst 1.039, 0.5 GB/s: 0.817)** |
| `eval_uhold` human NLL / `ECE@300s` (`n=18,091`) | 1.5980 / 0.0340 | **1.5439 / 0.0104 (3.3x lower ECE)** |
| `eval_uhold` `swechat:tool` / tool Spearman `rho` | 0.7589 / 0.774 | **0.6490 / 0.790** (published seed 0: **0.798**) |
| BurstGPT v2 `eval_ood` NLL / `ECE@60s` (`n=20,000`) | 2.1235 / 0.0405 | **2.0871 / 0.0148** |

Details, per-slice breakdowns, and the archived v1 model card are in `model/MODEL_CARD.md` and `model/RELEASE.md`.

#### 5. Cross-hardware evaluation, Online Calibration, and ablations (v2.1)

Adding runtime `--online-calib` (rolling per-kind temperature recalibration on top of the published 33.8M `Layby-Dwell v2`
weights, zero retraining) improves every single-GPU regime across four hardware tiers and five model families:

**Cross-hardware single-GPU comparison (`366` turns / `351` resumed turns, `O_DIRECT`):**

| hardware tier & model | policy | mean TTFT | p50 TTFT | p95 TTFT | p99 TTFT | max TTFT | disk written |
|---|---|---:|---:|---:|---:|---:|---:|
| **1x RTX PRO 6000 (96 GB)** `Qwen3-8B` BF16 (Gen4 `2.0 GB/s`, 5/5 GiB) | **Layby + Online Calibration** | **1.91 s** | **0.26 s** | **8.96 s** | **14.24 s** | **16.36 s** | 100.0 GB |
| 1x RTX PRO 6000 (96 GB) `Qwen3-8B` BF16 | Layby v2 (Static) | 2.44 s | 0.36 s | 11.66 s | 17.82 s | 22.29 s | 101.7 GB |
| 1x RTX PRO 6000 (96 GB) `Qwen3-8B` BF16 | Write-Through (All Disk) | 3.04 s | 0.29 s | 13.44 s | 35.38 s | 139.07 s | 111.9 GB |
| 1x RTX PRO 6000 (96 GB) `Qwen3-8B` BF16 | LMCache 0.5.7.post1 (ARC) | 3.17 s | 0.39 s | 12.97 s | 17.29 s | 22.60 s | 93.4 GB |
| 1x RTX PRO 6000 (96 GB) `Qwen3-8B` BF16 | No Disk (CPU-Only LRU) | 3.98 s | 1.20 s | 16.94 s | 20.73 s | 22.27 s | 0.0 GB |
| **1x RTX 5090 Laptop (24 GB)** `Qwen3-8B` BF16 (Gen4 `2.0 GB/s`, 4.75/5 GiB) | **Layby + Online Calibration** | **23.75 s** | 10.57 s | **95.51 s** | **105.87 s** | **117.97 s** | 100.5 GB |
| 1x RTX 5090 Laptop (24 GB) `Qwen3-8B` BF16 | Layby v2 (Static) | 27.62 s | 11.45 s | 98.70 s | 129.18 s | 157.33 s | 94.9 GB |
| 1x RTX 5090 Laptop (24 GB) `Qwen3-8B` BF16 | Write-Through (All Disk) | 31.03 s | **5.65 s** | 120.66 s | 272.00 s | 490.68 s | 110.3 GB |
| 1x RTX 5090 Laptop (24 GB) `Qwen3-8B` BF16 | LMCache 0.5.7.post1 (ARC) | 35.03 s | 25.75 s | 99.61 s | 109.30 s | 125.76 s | 109.8 GB |
| 1x RTX 5090 Laptop (24 GB) `Qwen3-8B` BF16 | No Disk (CPU-Only LRU) | 39.24 s | 29.22 s | 103.21 s | 118.49 s | 122.64 s | 0.0 GB |
| **DGX Spark GB10 ARM64 (121 GiB)** `Inkling 2 Mini` 26.4B NVFP4 (native `8.3 GB/s`, 10/2 GiB) | Write-Through (All Disk)* | **6.18 s** | **1.16 s** | **37.68 s** | **58.27 s** | 98.81 s | 67.1 GB |
| DGX Spark GB10 ARM64 `Inkling 2 Mini` 26.4B NVFP4 | **Layby + Online Calibration** | **7.77 s** | 2.00 s | 46.47 s | 61.69 s | 99.49 s | 59.1 GB |
| DGX Spark GB10 ARM64 `Inkling 2 Mini` 26.4B NVFP4 | Layby v2 (Static) | 8.56 s | 1.96 s | 46.28 s | 61.31 s | **65.05 s** | 58.6 GB |
| DGX Spark GB10 ARM64 `Inkling 2 Mini` 26.4B NVFP4 | Frequency-Gated Disk (Reuse >= 2) | 18.54 s | 3.82 s | 70.77 s | 83.22 s | 107.84 s | 41.6 GB |
| DGX Spark GB10 ARM64 `Inkling 2 Mini` 26.4B NVFP4 | No Disk (CPU-Only LRU) | 23.10 s | 3.10 s | 79.09 s | 91.10 s | 95.67 s | 0.0 GB |

*\*On DGX Spark `Inkling 2 Mini` (vLLM nightly `0.34.0rc1`, 48 looped FlashAttn + 24 SConv-SWA layers, ARM64 CPU sidecar),
`LMCache 0.5.7.post1` failed to import (`ImportError: get_kv_cache_spec_kind`), while Layby ran all arms cleanly (`2.97x`
faster than No Disk, `2.39x` faster than Frequency-Gated, `-9.2%` from Online Calibration). Write-Through led on native
`8.3 GB/s` NVMe because Inkling 2's 40 sliding-window layers compact offloaded KV to `~44 KiB/tok` while `ParkConnector`
currently estimates `128 KiB/tok` from the primary chunk stride (see Limits).*

**Zero-gap Blackwell Online Calibration overlay (`vLLM 0.34.0`, resumed mean / p95 TTFT):**
- `Qwen3.8-27B` fast NVMe (`1.31 GB/s`): **`1.74 s / 7.69 s`** (`81.0 GB` disk), beating both `write-through` (`1.83 s`) and static `layby-v2` (`1.85 s`).
- `Gemma-4-31B` fast NVMe (`1.35 GB/s`): **`3.69 s / 17.58 s`** (beats static `3.81 / 17.87 s` and `write-through` `3.85 / 18.88 s`).
- `Gemma-4-31B` slow disk (`146 MB/s`): **`24.35 s / 80.45 s`** (median `9.64 s`, `28.2 GB`), closing the static mean gap vs. `no-disk` (`24.80 s / 82.20 s`).
- `Qwen3.8-27B` slow disk (`141 MB/s`): **`10.06 s / 39.36 s`** (`52.90 s` p99, `58.67 s` max, `14.4 GB` disk), beating `no-disk` (`12.46 s`).
- `GLM-5.3-Flash 320B` TP=4 (native `11.6-12.5 GB/s` NVMe): `24.40 / 64.62 s` held-out (`seed=2027`) and `33.03 / 78.83 s` dev (`seed=2026`), beating all external baselines (`42.2-167.9 s`), while static `layby-v2` (`19.52 / 48.92 s` and `29.06 / 66.07 s`) stays fastest on 12 GB/s native NVMe.

**Slow-disk neural vs. heuristic ladder (`Qwen3-8B`, `141 MB/s`, 5/10 GiB, `0.0 GB` written by all Layby/no-disk arms):**
- **Layby + Online Calibration (Neural)**: **`2.04 s` mean / `9.05 s` p95 / `12.90 s` p99**
- Layby v2 (Static Neural): `2.07 s` mean / `9.86 s` p95 / `14.23 s` p99
- Layby Cost Rule Only (Kaplan-Meier, No Neural): `2.23 s` mean / `9.18 s` p95 / `14.55 s` p99
- Layby Cost Rule Only (EWMA, No Neural): `2.33 s` mean / `9.75 s` p95 / `15.55 s` p99
- No Disk (CPU-Only LRU): `2.98 s` mean / `13.21 s` p95 / `17.23 s` p99
- Write-Through (All to Disk): `10.53 s` mean / `54.02 s` p95 / `274.28 s` p99 (`77.2 GB` written)

**Architecture & real-engine component ablations (summary):**
- On `eval_uhold` (184,148 unseen-user turns) + 33-cell ReturnBench, removing the tabular tower (`Text Encoder Only`) degrades NLL `+0.599` (`2.1852` vs `1.5862`) and breaks worst-cell safety (`1.324` vs `1.039`); removing the text tower (`Tabular Only`) worsens human `ECE@300s` `3.1x` (`0.0323` vs `0.0104`); removing `LayerMix` + gated bilinear fusion (`Plain Concatenation`) worsens `ECE@300s` `1.9x` (`0.0202`) and worst cell (`1.084`). Adding an auxiliary prefix-reuse head (`aux=0.10/0.25`) regresses `wait_poll` NLL (`+0.391`) and trails published `aux=0` weights live (`2.00 s` vs `1.91 s`).
- On `Qwen3-8B` Gen4 NVMe (`2.0 GB/s`), the 8-arm real-engine decomposition confirms that the cost rule + adapter alone capture most of the fast-NVMe mean gain (`1.94 s` EWMA, `1.97 s` Kaplan-Meier, `2.65 s` adapter without curves, `3.98 s` stock no-disk), while the neural encoder + Online Calibration delivers the best combined tail and mean (`1.91 s` r1 / `2.02 s` r2 under 4-GPU contention, `14.24 s` p99 vs `15.56-15.89 s` heuristics and `25.06 s` frequency-gated) and separates clearly on slow disk (`-12.4%` mean, `-17.0%` p99 vs EWMA).

---

### Earlier v1 results (vLLM 0.30-0.31, SGLang 0.5.21, Layby-Dwell v1)

Time to first token of returning turns, p95, as a ratio to another arm. Below 1 is better. The v1 LMCache arms below
are **lmcache** (LMCache 0.5.5 defaults) and **lmcache-retain** (LMCache with `--l2-prefetch-policy retain` and a
0.95 L1 eviction watermark).

Three repeats, one box each, session bootstrap 95% intervals. Qwen3-8B on one A100, 16 GiB GPU KV, 28 GiB CPU tier.

| engine, workload | layby vs no-disk | write-through vs no-disk | layby vs write-through |
|---|---|---|---|
| vLLM 0.30, private agent replay (18 sessions) | 0.69 [0.60, 0.78] | 0.57 [0.49, 0.64] | 1.22 [1.13, 1.28] |
| vLLM 0.30, ReturnBench mix36 (36 sessions, heavy load) | 0.16 [0.14, 0.18] | 0.15 [0.13, 0.17] | 1.11 [0.97, 1.22] |
| SGLang 0.5.21, private agent replay | 0.93 | 0.92 | 1.01 |
| SGLang 0.5.21, ReturnBench mix36 | 0.34 | 0.15 | 2.28 |

These rows ran an adapter with two bugs since fixed (local hit, hint heap) and no wait cap. A rerun of the
private replay with the fixed adapter on vLLM 0.31.0, one repeat per arm on each of three A100 boxes:

| box | layby vs no-disk | write-through vs no-disk | layby vs write-through | p95 no-disk / write-through / layby (s) |
|---|---|---|---|---|
| two SXM4 80GB boxes, disk 3.5 and 5.5 GB/s (pooled) | 0.64 [0.58, 0.74] | 0.71 [0.64, 0.79] | 0.89 [0.84, 1.04] | 4.29 / 2.96 / 2.85 and 4.92 / 3.55 / 2.93 |
| PCIe 40GB box, disk 0.54 GB/s | | | | 8.02 / 4.52 / 13.49 |

On fast disks the gap to write-through is gone. On the slower box Layby lost: see Limits.

In the simulator, over 33 cells of ReturnBench (11 pool and load settings, disks at 0.5, 1.5 and 3 GB/s), the cost
rule's p95 is 0.66 of no-disk against 0.68 for write-through and 0.60 for an oracle that knows each return time.
Write-through is best on fast disks (0.54 at 3 GB/s against 0.57) and worst on slow ones: up to 1.46 of no-disk
at 0.5 GB/s, where it floods the disk. The cost rule is never worse than 1.04 of no-disk in any cell.

GLM 5.3 Flash FP8 on vLLM 0.31.0, Nebius nodes with 8x RTX PRO 6000: two TP4 engines (the two repeats of each row)
share a network SSD read with O_DIRECT at 0.55 GB/s, so no page cache helps. ReturnBench at real length, contexts up
to 260k tokens. Heavy is 150 sessions, half is 75. Node 1 ran the adapter before the local-hit fix (lesson 4 below).
Node 2 ran the heavy load again with the fixed adapter, with and without the wait cap (lesson 5). The cap is the
default; layby without it is the uncapped operating point.

| load | layby vs no-disk | write-through vs no-disk | layby vs write-through |
|---|---|---|---|
| heavy, node 1, before the fix | 0.67 [0.60, 0.82] | 1.17 [1.15, 1.19] | 0.58 [0.52, 0.70] |
| heavy, node 2, layby with the wait cap (default) | 0.79 [0.73, 0.85] | 1.35 [1.31, 1.42] | 0.59 [0.54, 0.63] |
| heavy, node 2, layby without the cap | 0.59 [0.52, 0.70] | 1.35 [1.31, 1.42] | 0.44 [0.38, 0.53] |
| half, node 1 (75 sessions) | 0.95 [0.82, 1.08] | 3.99 [3.28, 4.74] | 0.24 [0.20, 0.28] |

Node 2 per engine (A / B), resumed-turn TTFT in seconds:

| arm | p50 | p95 | p99 | recomputed |
|---|---|---|---|---|
| no-disk | 54.3 / 62.5 | 275 / 278 | 286 / 286 | 61% / 62% |
| write-through | 58.3 / 58.7 | 394 / 353 | 460 / 480 | 56% / 56% |
| layby, no cap | 13.6 / 10.1 | 153 / 167 | 1919 / 829 | 39% / 35% |
| layby, wait cap (default) | 14.6 / 20.8 | 222 / 208 | 283 / 290 | 39% / 40% |

On a slow shared disk, writing every chunk is worse than having no disk at all. Layby writes only what pays: with the
fixed adapter and no cap it cuts heavy-load p95 to 0.59 of no-disk, with the cap to 0.79, and it leaves half load where it
was. The simulator predicted 0.80, 1.27 and 0.63 for the heavy load before the disk arms ran. Without the cap the p99
is a starvation tail: 33 of 1,908 turns on engine A waited 30 to 42 minutes while a disk promotion of their prefix
was stuck behind a saturated disk. The cap removes that tail (p99 283 and 290 s against 286 for no-disk) and costs 1.34x at
p95 and 1.47x at p50 against the uncapped adapter, because each give-up recomputes a prompt of about 58k tokens on a
prefill-bound engine.

Qwen3-8B on vLLM 0.31.0, one RTX PRO 6000 per VM, the A100 rounds' capacities (16 GiB GPU KV, 28 GiB CPU tier),
ReturnBench mix36, and a disk tier on the VM's network SSD read with O_DIRECT at 0.48 GB/s (three repeats per arm):

| arm | p50 (s) | p95 (s) | p99 (s) |
|---|---|---|---|
| no-disk | 0.12 / 0.12 / 0.12 | 5.7 / 7.8 / 5.6 | 10.4 / 15.3 / 10.9 |
| write-through | 0.13 / 0.12 / 0.12 | 8.9 / 13.4 / 8.1 | 91 / 165 / 143 |
| layby | 0.13 / 0.12 / 0.12 | 3.5 / 3.8 / 3.6 | 6.7 / 10.6 / 6.8 |
| layby before the local-hit fix | 0.15 / 0.14 | 5.6 / 5.2 | 9.8 / 12.3 |
| lmcache | 12.3 / 15.3 / 13.6 | 51 / 63 / 55 | 63 / 83 / 69 |
| lmcache-retain | 0.18 / 0.17 / 0.18 | 7.6 / 7.5 / 6.6 | 16.3 / 14.6 / 15.0 |

| pair | p95 ratio [95% CI] | p50 ratio |
|---|---|---|
| layby vs no-disk | 0.59 [0.51, 0.68] | 1.01 |
| layby vs write-through | 0.37 [0.28, 0.56] | 1.01 |
| write-through vs no-disk | 1.61 [1.08, 2.09] | 1.00 |
| lmcache-retain vs no-disk | 1.21 [1.08, 1.36] | 1.46 |
| layby vs lmcache-retain | 0.49 [0.44, 0.57] | 0.70 |
| layby vs lmcache | 0.07 [0.06, 0.07] | 0.01 |
| layby vs layby before the fix (two repeats) | 0.68 [0.61, 0.75] | 0.88 |

## Limits

- **Small-model disk crossover on SGLang `0.5.23` (`Qwen3-8B` bf16 at `1.31 GB/s`).** On an RTX PRO 6000 Blackwell GPU,
  `Qwen3-8B` bf16 prefill takes `105 us/tok` while `147.5 KB/tok` KV reads over `1.31 GB/s` NVMe take `120 us/tok`.
  Below `2.0 GB/s`, L3 disk cannot beat GPU recompute on mean latency (`layby-v2` still cuts disk writes `15-26%` and
  wins tail latency, and once NVMe reaches `2.0 GB/s` Gen4 speeds it beats `hicache-selective` across every metric).
  Similarly, in SGLang, `write_through_selective` is needed when `CPU == GPU` while `write_through` wins when
  `CPU >= 2x GPU`.
- **Ultra-slow disk on Gemma-4-31B (`gemma-slow-pressure`, `146 MB/s`).** Static `layby-v2` leads all comparators on
  2-run average p95 (`76.82 s` vs `no-disk` `82.20 s`, `freq-gated` `84.54 s`, `continuum` `90.44 s`, `write-through`
  `97.93 s`, `lmcache` `190.16 s` timeout) while `no-disk` had slightly lower mean (`24.80 s` vs `26.19 s`). Enabling
  `--online-calib` closes the mean gap (`24.35 s` mean, `80.45 s` p95, median cut to `9.64 s`), though `no-disk` keeps
  a lower median (`5.66 s`).
- **Hybrid multi-group SWA `bytes_per_token` overestimation on `Inkling 2 Mini` (DGX Spark).** On architectures where
  most layers are short sliding-window (40 SWA-1024 + 8 global layers in `Inkling 2 Mini`), offloaded KV averages
  `~44 KiB/tok` while `ParkConnector` currently derives `bytes_per_token` from the primary chunk stride (`128 KiB/tok`),
  causing the cost rule to overestimate disk I/O cost by ~2.9x. On unthrottled `8.3 GB/s` NVMe, that conservative
  estimate caused Layby to skip 52 turns (`7.77 s` mean vs `6.18 s` write-through, still `2.97x` faster than `no-disk`
  and `2.39x` faster than `freq-gated`). Per-group hybrid byte accounting is future work.
- **Online calibration on ultra-fast 12 GB/s native NVMe (`GLM-5.3-Flash` TP=4).** Online per-kind temperature
  recalibration wins across every 1-GPU Gen4 and slow-disk workload, whereas static `layby-v2` (`19.52 s` held-out /
  `29.06 s` dev) stays faster than `--online-calib` (`24.40 s / 33.03 s`) on 12 GB/s native NVMe.
- **SGLang adapter model scope.** `ParkRadixCache` targets full-attention models on SGLang's Python TreeCore
  (`SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python`, applied equally to all SGLang arms in benchmarks); hybrid SWA and
  Mamba models in SGLang remain future work.
- **The disk in the v1 A100 runs was mostly RAM.** vLLM's file tier falls back from O_DIRECT to buffered I/O on the
  overlay filesystems of those boxes, so recent writes were read back from the OS page cache. On real O_DIRECT disks
  (the Nebius and v2 Blackwell rounds), write-through degrades under contention while Layby adapts automatically.
- **Write-back on eviction is a negative result on vLLM 0.30.** vLLM frees an evicted CPU slot at once, so writing
  ahead of eviction pins chunks and starves the CPU tier (p95 1.67x write-through).
- **The adapter's own cost is within noise.** With the rule switched off, the vLLM adapter matches stock vLLM within
  run-to-run spread (recompute within 1%).
- **The private replay is not shipped.** It is the author's own sessions. The public workloads in
  `engine/workloads/` reproduce the setup.
- **GLM 5.3 Flash needs TP4 or less on RTX PRO 6000.** The sm_120 sparse MLA decode kernel has no build for 8 heads
  per GPU, so TP8 does not start.

## Adapter lessons

Each one cost real-engine runs before it was found. All eight are fixed, and each has a regression test.

1. **Do not rewrite what is already on disk.** vLLM sends a request-level tier every chunk of the request again on
   each turn. `ParkFsTier` now writes only chunks that are neither on disk nor in flight (`tests/test_park_vllm_fs.py`).
2. **Keep telemetry out of the scheduler's hot path.** vLLM calls block-pool hooks once per block (~1,900 blocks on a
   30k-token turn). Events are buffered and folded in once per step (`tests/test_live_batch.py`), with an in-process
   sampling profiler at `/prof` and `/prof/windows`.
3. **Evict a session's tail before its prefix.** vLLM reuses a cached prefix only as a contiguous run from chunk 0.
   `ParkCachePolicy` walks keys tail first and matches vLLM's LRU order (`tests/test_policy_prefix.py`,
   `tests/test_policy_lru_parity.py`).
4. **Read the local hit where vLLM reports it.** vLLM calls `update_state_after_alloc` before setting
   `num_computed_tokens`, so reading the block-aligned local hit from `get_num_new_matched_tokens` eliminated phantom
   miss penalties (`tests/test_local_hit.py`).
5. **Bound a deferral at what giving up costs.** Giving up on a disk promotion once wait exceeds recompute cost
   removes 30-minute starvation tails on slow shared disks (`tests/test_defer_cap.py`).
6. **Protect same-step HIT chunks from promotion allocation (vLLM 0.34.0).** Under tight CPU RAM (`GLM-5.3-Flash`
   320B MoE), promoting a queued request's chunks on the same scheduler step can evict an unpinned HIT chunk before
   `prepare_load()` crashes `EngineCore` (`cpu/manager.py:242`, killing stock `write-through` at turn 76). Step-scoped
   HIT protection (`_step_protected`) and worker-thread I/O timing (`_load_io_times`) eliminate the race
   (`tests/test_layby_v2_fixes.py`).
7. **Serialize overlapping promotions and stream fast NVMe cleanly (vLLM 0.34.0).** Concurrent disk promotions larger
   than CPU capacity ping-pong chunks in and out of RAM; FCFS queue-ordered promotion serialization with a CPU pin
   ceiling plus hardware-gated fast-NVMe streaming cut `GLM-5.3-Flash` makespan by >2x (`tests/test_layby_v2_fixes.py`).
8. **Hook write-through lock release and exclude load-back stalls in SGLang 0.5.23.** Async D->H write-through keeps
   `lock_ref > 0` across `on_release()` until `dec_lock_ref()`, and H->D load-back stalls inflate `f0` by 8x if
   counted as prefill compute. Hooking `dec_lock_ref`/`dec_host_lock_ref`, filtering `hicache_consumer_index >= 0`,
   and falling back to the deepest surviving prompt ancestor under `write_through_selective` restored accurate
   residency, prefill speed, and hint coverage (`tests/test_park_sglang.py`).

## Names

The placement layer started as "park", and the engine-facing names keep it: the vLLM classes `ParkConnector`,
`ParkCachePolicy` and `ParkFsTier`, the SGLang plugin, backend and eviction policy `park`, and the request fields
`park_key`, `park_surv`, `park_eta`, `park_disk`. The Python package is `layby`.

## Layout

```
layby/          the cost rule, live estimators, profiler, vLLM and SGLang adapters, sidecar
layby_dwell/    Layby-Dwell v2 (zero-laya) and v1 inference (features, batching, precisions)
model/          model card, release file list (v2.0 and archived v1.0), quantization study
sim/            simulator, baselines, ReturnBench evaluation (python -m sim, or returnbench)
bench/          ReturnBench pools, builders, licenses (bench/README.md)
engine/         real-engine harness: replay client, arm runners, smoke gate, pooling (engine/REPRODUCE.md)
reference/      the simulator reference run
tests/          CPU tests (vLLM 0.30/0.31/0.34, SGLang 0.5.21/0.5.23, Layby-Dwell v2)
```

## Tests

    PYTHONPATH=. python tests/test_dwell_v2_sidecar.py              # Layby-Dwell v2 (zero laya) + sidecar
    PYTHONPATH=. python -m pytest tests/test_layby_v2_fixes.py      # vLLM 0.34.0 v2 fixes (27 tests)
    PYTHONPATH=. python -m pytest tests/test_park_sglang.py         # SGLang 0.5.23 adapter (23 tests)

## Citation

The paper is "Layby: Placing Idle KV Cache by Predicted Return Time and Measured Cost" (2026; arXiv id to follow).
`CITATION.cff` at the repository root carries the same entry for GitHub and Zenodo. The code is archived on
Zenodo: [10.5281/zenodo.23285213](https://doi.org/10.5281/zenodo.23285213) (all versions; v1.0.0 is
[10.5281/zenodo.23285214](https://doi.org/10.5281/zenodo.23285214)). The Layby-Dwell model is
[10.57967/hf/10852](https://doi.org/10.57967/hf/10852) on the Hugging Face Hub.

```bibtex
@misc{fenesh2026layby,
  title  = {Layby: Placing Idle KV Cache by Predicted Return Time and Measured Cost},
  author = {Avi Fenesh},
  year   = {2026},
  note   = {arXiv id to follow}
}
```

## License

Code: Apache-2.0 (`LICENSE`). Layby-Dwell v2 weights are Apache-2.0, matching the base
`jhu-clsp/ettin-encoder-32m` weights (and archived v1 weights match Apache-2.0 `convaiinnovations/laya`). The
ReturnBench pools carry their source datasets' licenses (`bench/LICENSES.md`).
