---
license: apache-2.0
base_model: jhu-clsp/ettin-encoder-32m
library_name: transformers
pipeline_tag: text-classification
tags:
  - kv-cache
  - llm-serving
  - inference
  - survival-analysis
  - vllm
  - sglang
  - agents
datasets:
  - allenai/WildChat-4.8M
  - SALT-NLP/SWE-chat
---

# Layby-Dwell v2

Layby-Dwell predicts how long an LLM session stays idle after a response ends. It reads what a
serving engine can see at that moment and returns a calibrated survival curve: the probability that
the session's next request has not arrived yet, at 15 horizons from 0.5 s to 30 min.

Code, engine adapters and the ReturnBench benchmark: [github.com/avifenesh/layby](https://github.com/avifenesh/layby) (`v2.1.0`), archived on Zenodo as
[10.5281/zenodo.23285213](https://doi.org/10.5281/zenodo.23285213).
Paper: arXiv, link to follow. Author: [Avi Fenesh](https://github.com/avifenesh), [Tiyuvta](https://tiyuvta.ai) ([research notes](https://tiyuvta.ai/research/), [partner deployment studies](https://tiyuvta.ai/partners/)).

It exists for one decision: where to keep the session's KV cache while it waits. Keep it on the GPU,
move it to CPU memory, write it to disk, or let it go. Layby, the placement rule it was built for,
turns the curve and the engine's own measured hardware costs into that choice. There are no tuned
constants.

> **v2.0 release (`main`, tag `v2.0`):** `Layby-Dwell v2` replaces the 421M `v1` model with a
> standalone **33.8M** (`135 MB`) two-tower model built on
> [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) (Apache-2.0). It
> removes the `laya` dependency, runs on **CPU in 11-29 ms with 0 MiB GPU VRAM** (or 4.1 ms on GPU),
> cuts human calibration error by 3.3x, and matches or beats `v1` on ReturnBench and held-out splits.
> Legacy `v1` (`421M`, `convaiinnovations/laya`) remains archived at branch `v1` / tag `v1.0`
> (`Dwell("avifenesh/layby-dwell", revision="v1")`).

## Architecture (`v2_gated`)

- **Text tower:** [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m)
  (10-layer, 384-dim ModernBERT, `max_len = 1536`) over compact delta text (`kind`, `tool`, `program`,
  `model`, `TOOL_CALL`, `ASSISTANT`, `USER`) + learnable top-4 `LayerMix` + `[CLS]` + `AttentionPool`.
- **Numeric tower (`PeriodicNumericEncoder`):** 32 causal server-view numeric features with explicit
  missing masks, 8 learned Fourier frequencies per feature, and 2 residual SwiGLU blocks (256-dim).
- **Gated bilinear residual head:** `z = z_num + sigmoid(gate) * z_delta` over concatenated
  `[h_txt, h_num, h_num * txt_proj]`, producing 16 right-censored discrete survival bin logits with
  per-kind temperature calibration (`_all: 1.0261`, `human: 1.0051`, `tool: 1.0309`).
- **Size:** `33,755,701` parameters (`14.4M` non-embedding), `135 MB` fp32 safetensors, zero `laya`
  dependency. One forward pass per state; no text is generated.

## Input

A server-view state: a JSON object (or Python `dict`) built at the moment a response ends. Any
OpenAI-compatible server can fill every field. `layby_dwell.Tracker` builds it from the request and
response stream.

| field | what it is |
|---|---|
| `kind` | `tool` if the response asks for tool calls, else `human` |
| `model`, `tool`, `program`, `arg_chars` | the served model, the first tool call's name, the program of a shell command, argument size |
| `ctx_tokens`, `turn`, `session_age_s`, `hour`, `dow` | context size, position in the session, clock (UTC) |
| `recent` | the session's last 8 idle periods: kind, tool, seconds |
| `meta` | previous gap and kind, same-kind gap and its EWMA, command size, timeout, background flag |
| `tool_history`, `meta.program_history`, `meta.user_kind_history` | this user's earlier gaps for the same tool, program and kind: count, mean, p90, share under 1 s |
| `tool_call`, `assistant`, `user` | the tool calls, the current turn's assistant text and the last user message, truncated |

Sessions are linked by a hash of the conversation prefix. A user is whatever key the server has
(API key, tenant, header). History fields are optional: the model was trained with telemetry dropout
(`p_id = 0.30`, `p_history = 0.15`) and degrades gracefully on cold-start traffic, though session
history remains valuable for long human waits (see Limits).

## Output

`P(T > h)` for `h` in `[0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 120, 300, 600, 1800]` seconds, where
`T` is the time until the session's next request. Probabilities are strictly non-increasing and
calibrated per kind with temperatures stored in `config.json`.

## Use

```python
from layby_dwell import Dwell, Tracker

# v2 (33.8M, zero laya dependency; runs comfortably on CPU with 0 MiB GPU VRAM, or GPU bf16)
dwell = Dwell("avifenesh/layby-dwell", device="cpu")
tracker = Tracker()

# in the serving path, per request and response:
sess = tracker.on_request(body, user)
state, kind = tracker.on_response(sess, body, resp_message, usage, user)
surv = dwell.survival([state], [kind])[0]     # 15 values of P(T > h)
```

To reproduce legacy v1 (`421M`, requires `pip install -e ".[dwell,dwell-v1]"`):

```python
dwell_v1 = Dwell("avifenesh/layby-dwell", revision="v1")
```

Score at the end of each response, off the critical path. Batch the boundaries that end together:
`survival` sorts states by length and pads each batch only to its longest state (rounded to 16).

### With vLLM and SGLang

The Layby adapters run the placement rule inside the engine and take the curve as a hint keyed by
request:

- **vLLM 0.30, 0.31, 0.34, and nightly (`>=0.34.0rc1`):** `ParkConnector` (`layby.vllm.connector`, an
  `OffloadingConnector` subclass) with `ParkCachePolicy` (`layby.vllm.policy`) and `ParkFsTier`
  (`layby.vllm.fs_tier`), configured through `--kv-transfer-config`.
- **SGLang 0.5.21 and 0.5.23:** the `park` plugin (`layby.sglang`, `SGLANG_PLUGINS=park`,
  `--radix-cache-backend park --radix-eviction-policy park`) on HiCache with a file storage backend.

The Layby sidecar (`python -m layby.sidecar.proxy --device cpu`) is an OpenAI-compatible proxy that
handles session tracking, CPU scoring (0 MiB GPU VRAM), optional online temperature recalibration
(`--online-calib`), and hint delivery for either engine.

## Training

Trained from [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m)
(`1b8ba06455dd44f80fc9c1ca9e22806157a57379`) on `3,236,086` cleaned, secret-scrubbed server-view
turns for 1 epoch with ScheduleFree AdamW, FP32 master weights under BF16 autocast, `max_len = 1536`,
inverse-subsampling importance weighting normalized by kind, and server-telemetry dropout
(`p_id = 0.30`, `p_history = 0.15`). Target: exact right-censored 16-bin discrete survival negative
log-likelihood (a strictly proper scoring rule). The published weights are Seed 0 of the 3-seed
champion run (`rung4_32m_all_corpora`).

### Data

Only datasets whose licenses allow commercial use, with 0 secret regex hits across 100% of rows and
strict zero-leakage splits (0 unseen-user overlap, 0 cross-split sessions, 0/9,987 ReturnBench
`replay_pop` turns in training, and future weeks 12-15 of FreeInference held out):

- [GitHub Copilot Coding Agent Traces 2026](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md) (CC BY 4.0): agent sessions, timestamps only.
- [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) and SWE-chat v2 (ODC-BY): Claude Code and Codex transcripts.
- [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) (ODC-BY): chat conversations with exact request timestamps.
- [TraceLab](https://github.com/uw-syfi/TraceLab) (CC BY 4.0): coding-agent traces (broken 0 ms clock artifacts removed).
- [BurstGPT v2](https://github.com/HPMLL/BurstGPT) (`BurstGPT_3.csv`, CC BY 4.0): chronological 80% session train split (10% dev / 10% eval OOD held out).
- The author's own Claude Code and Codex sessions, and users' data under the terms they agreed to.

## Evaluation

### Head-to-head vs. archived `v1` (`421M`) on strictly untouched splits

All `v2` offline metrics below are 3-seed medians on strictly untouched evaluation splits:

| Metric / Benchmark | Archived `v1` (`421M`) | **`Layby-Dwell v2` (`33.8M`)** | Improvement |
|---|---:|---:|---|
| Parameters (total / non-embedding) | 421M / 369M | **33.8M / 14.4M** | **12.5x smaller (25.6x non-emb)** |
| Checkpoint size (fp32 safetensors) | 1,685 MB | **135 MB** | **12.5x smaller** |
| Peak serving VRAM (GPU BF16) | 2,616 MB | **427 MB** (or **0 MB** on CPU) | **6.1x less VRAM** |
| Official 33-cell ReturnBench (`p95` vs no-disk) | 0.659 (worst 1.043) | **0.660 (worst 1.039)** | Matches overall, beats `v1` on worst cell |
| ReturnBench slow disk (`0.5 GB/s` vs no-disk) | 0.823 | **0.817** | Beats `v1` on all 3 seeds (write-through: 0.986) |
| Untouched `eval_uhold` human NLL (`n=18,091`) | 1.5980 | **1.5439** | **-0.0541 NLL** |
| Untouched `eval_uhold` human `ECE@300s` | 0.0340 | **0.0104** | **3.3x lower calibration error** |
| Untouched `eval_uhold` `swechat:tool` (`n=7,260`) | 0.7589 | **0.6490** | **-0.1099 NLL** |
| Untouched `eval_uhold` `ask_user` / `subagent` NLL | 2.1127 / 1.8073 | **1.6967 / 1.6127** | **-0.4160 / -0.1946 NLL** |
| Untouched `eval_uhold` tool Spearman `rho` | 0.774 | **0.790** (published seed 0: **0.798**) | **+0.016 to +0.024 rank gain** |
| Untouched BurstGPT v2 `eval_ood` (`n=20,000`) | 2.1235 (`ece60` 0.0405) | **2.0871 (`ece60` 0.0148)** | **-0.0364 NLL, 2.7x lower 60s ECE** |
| AgentX 390-session replay (`0.5 GB/s`) | 1.012 (wt: 1.158) | **1.004** | Eliminates write-through blow-up |

### Architecture & objective ablations (`eval_uhold`, 184,148 unseen-user turns + 33 ReturnBench cells)

One-epoch ablations on the same 3.24M-row corpus confirm every architectural module (`aux=0.10` is 3-seed median):

| Model variant | Params | All NLL | Tool NLL | Human NLL | Human `ECE@300s` | `wait_poll` NLL | RB p95 (worst) | RB `@0.5` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Layby-Dwell v2 (`aux=0`, published)** | **33.8M** | **1.5862** | **1.6046** | 1.4375 | 0.0104 | **1.8843** | 0.660 (1.039) | 0.817 |
| Joint-Prefix (`aux=0.10`, 3 seeds) | 33.8M | 1.6068 | 1.6282 | 1.4338 | 0.0115 | 2.2126 | 0.659 (1.039) | 0.815 |
| Stronger Aux (`aux=0.25`) | 33.8M | 1.6027 | 1.6233 | 1.4359 | 0.0115 | 2.0448 | 0.657 (1.039) | 0.817 |
| Long-Tail Upweight (`2x >60s`) | 33.8M | 1.6006 | 1.6214 | 1.4326 | 0.0128 | 2.3348 | 0.660 (1.073) | **0.806** |
| Compact 18.5M (`Ettin-17M`) | 18.5M | 1.5930 | 1.6129 | **1.4318** | **0.0093** | 2.3914 | 0.659 (1.067) | 0.818 |
| Plain Concat (no `LayerMix` / gate) | 32.1M | 1.6435 | 1.6664 | 1.4577 | 0.0202 | 2.2865 | 0.661 (1.084) | 0.819 |
| Tabular Telemetry Only (no text) | 1.4M | 1.6684 | 1.6906 | 1.4888 | 0.0323 | 1.8861 | 0.667 (1.061) | 0.828 |
| Text Encoder Only (no telemetry) | 32.5M | 2.1852 | 2.2540 | 1.6288 | 0.0514 | 2.6953 | 0.678 (1.324) | 0.873 |

### Real-engine serving (CPU sidecar, 0 MiB GPU VRAM)

Running this exact checkpoint (`rung4_32m_all_corpora`, seed 0) on host CPU across 4 hardware tiers and 5 model families (`O_DIRECT` / `fadvise`, 0 hint errors across >8,000 live turns, 0 MiB extra GPU VRAM):

- **4x RTX PRO 6000 Blackwell, vLLM 0.34.0, untouched held-out `GLM-5.3-Flash` 320B MoE TP=4 (`seed=2027`, 128K-190K ctx, 398 turns, `12.5 GB/s` NVMe):**
  `19.52 s` mean TTFT (`48.92 s` p95), **4.69x faster mean** than `no-disk` (`91.57 s`), **2.16x faster** than `LMCache 0.5.7.post1 ARC` (`42.23 s`, with 31.7% less disk and 3.5 GiB less VRAM), **2.17x faster** than `freq-gated` (`42.31 s`), and **8.60x faster** than `Continuum` (`167.91 s`).
- **1x RTX PRO 6000 Blackwell (`96 GB`), vLLM 0.34.0 with `--online-calib`:**
  - `Qwen3-8B` BF16 (`2.0 GB/s`): `#1` across every metric at **`1.91 s` mean / `8.96 s` p95 / `14.24 s` p99** vs `3.98 s` `no-disk` (**2.08x**), `3.17 s` `LMCache ARC`, and `3.04 s` `write-through` (`35.38 s` p99 / `139.07 s` max).
  - `Qwen3.8-27B` FP8 (`1.31 GB/s`): `#1` overall mean (**`1.74 s`** vs `1.83 s` `write-through`, `2.43 s` `no-disk`); `Gemma-4-31B` FP8 (`1.35 GB/s`): `#1` mean and p95 (**`3.69 s` / `17.58 s`** vs `3.85 s` / `18.88 s` `write-through`); `Gemma-4-31B` slow disk (`146 MB/s`): beats `no-disk` on both mean and p95 (**`24.35 s` / `80.45 s`** vs `24.80 s` / `82.20 s`, while `LMCache` timed out at `190.2 s` p95).
- **1x RTX 5090 Laptop (`24 GB`), vLLM 0.34.0 (`Qwen3-8B` BF16, `2.0 GB/s`):**
  `--online-calib` leads mean and all tails at **`23.75 s` mean / `95.51 s` p95 / `105.87 s` p99 / `117.97 s` max** vs `39.24 s` `no-disk` (**1.65x**), `35.03 s` `LMCache ARC` (**1.47x**), and `31.03 s` `write-through` (`272.0 s` p99 / `490.7 s` max).
- **NVIDIA DGX Spark GB10 ARM64 (`121 GiB`), vLLM nightly `0.34.0rc1` (`Inkling 2 Mini` 26.4B looped SWA MoE NVFP4, native `8.3 GB/s` NVMe):**
  Runs natively on ARM64 CPU (`117 ms` median, 0 MiB VRAM). `--online-calib` reaches **`7.77 s` mean** (**2.97x faster** than `no-disk` `23.10 s`, **2.39x faster** than `freq-gated` `18.54 s`), while `LMCache 0.5.7.post1` failed to import on vLLM nightly; `write-through` led mean (`6.18 s`) due to single-stride overestimation of Inkling 2's compact hybrid-SWA KV.
- **Slow-disk neural vs. heuristic ladder (`141 MB/s`, `Qwen3-8B`, `0.0 GB` written by Layby/no-disk):**
  Monotonic latency drop purely via CPU eviction order: `no-disk` (`2.98 s` mean / `17.23 s` p99) -> EWMA (`2.33 s` / `15.55 s`) -> Kaplan-Meier (`2.23 s` / `14.55 s`) -> static neural (`2.07 s` / `14.23 s`) -> neural `--online-calib` (**`2.04 s` / `12.90 s`**, **12.4% faster mean and 17.0% faster p99 than EWMA**). Stock `write-through` collapses to `10.53 s` mean / `274.3 s` p99 (`77.2 GB` written).
- **SGLang 0.5.23 + HiCache (`Qwen3-8B` BF16, 366 turns):**
  Beats `hicache-selective` across every metric at `2.0 GB/s` (`3.27-3.40 s` vs `3.60 s`), cuts disk writes `15-26%` at `1.31 GB/s`, and auto-suppresses disk writes (`0.0 GB`) on `141 MB/s` slow disk while `hicache-write-back` OOM-crashes at turn 21.

Full real-hardware tables across all hardware tiers and model families are in the [Layby repository README](https://github.com/avifenesh/layby).

## Speed

| device | precision | one state (`bs=1`) | batch of 8 | VRAM |
|---|---|---|---|---|
| CPU x86, 8 threads (`L = 128 / 256 / 512`) | fp32 | **20.9 / 24.3 / 29.8 ms** | | **0 MB** |
| CPU x86, live sidecar (64-vCPU host, mixed lengths) | fp32 | **10.6 to 19.4 ms p50** | | **0 MB** |
| CPU ARM64, live sidecar (DGX Spark GB10) | fp32 | **117 ms p50** | | **0 MB** |
| RTX 5090 (`L = 256 / 1536`, PyTorch `sdpa`) | bf16 | **4.07 / 4.26 ms** | **1,960 / 488 st/s** (`L=256/1024`) | **427 MB** |

Unlike `v1` (1.7-2.9 s on CPU, GPU-required), `v2` is fast enough to run directly on CPU inside the
inference router or sidecar process with zero GPU contention. On GPU, serve in `bf16` (the default).
Weight-only 8-bit options (`precision="int8w"`, `"fp8w"`) remain available via `torchao`; see
`model/QUANT.md` for the v1 quantization study.

## Limits

- **Session history matters for long human waits.** Without session linking (stateless scoring),
  long-horizon human recall degrades sharply. Run `Tracker` so prior session gaps are populated.
- **Temporal harness drift on tool latencies.** On the single slice where archived `v1` edged `v2`
  (`freeinference:tool` future weeks 12-15: `1.6336` vs `1.6828`, `+0.0492` NLL driven by `wait_poll`
  latency shift), `v2` still ranked tools better (`rho = 0.790-0.798` vs `0.774`) and beat `v1` on
  `freeinference:human` (`-0.0690` NLL). For deployments with shifting agent harnesses, enable online
  per-kind temperature recalibration (`--online-calib` in `layby/sidecar/proxy.py`).
- **English-centric text fields:** non-English corpora were not specifically benchmarked.
- **Forecast, not a guarantee:** use the survival curve inside a cost-aware rule that prices wrong
  calls, as Layby does, rather than as a hard eviction timeout.
- **Data handling:** scoring reads prompt and response text locally; it stores nothing, but treat it
  like any other component on the inference host.

## License

Apache-2.0, matching the base [jhu-clsp/ettin-encoder-32m](https://huggingface.co/jhu-clsp/ettin-encoder-32m) weights.

## Citation

```bibtex
@misc{fenesh2026layby,
  title  = {Layby: Placing Idle KV Cache by Predicted Return Time and Measured Cost},
  author = {Avi Fenesh},
  year   = {2026},
  note   = {arXiv id to follow}
}
```
