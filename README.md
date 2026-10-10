# Layby

Layby decides where an idle LLM session's KV cache waits until the session comes back: on the GPU, in CPU RAM, on
disk, or nowhere. Agent and chat sessions go idle between turns for seconds to minutes. Their KV is large, and
recomputing it on return is what makes the next turn slow.

Model: [Avifenesh/layby-dwell](https://huggingface.co/Avifenesh/layby-dwell) on Hugging Face. Paper: arXiv, link to
follow. Author: [Avi Fenesh](https://github.com/avifenesh), [Tiyuvta](https://tiyuvta.ai).

Layby has four parts:

- **A cost rule** (`layby/rule.py`). For each finished turn it compares the expected cost of keeping the session's
  KV in RAM only, also writing it to disk, moving it to disk and out of RAM first, or dropping it. Every term is a
  probability from a return-time curve times a cost the engine measures live: link speeds, queue waits, prefill
  speed, eviction rates and residency. There are no tuned constants.
- **Layby-Dwell**, the curve. A model that reads what a serving engine sees of a session (the tool calls and their
  arguments, the last assistant texts and human message, token counts, timing) and returns P(next request later
  than t) at 15 horizons from 0.5 s to 30 min. It runs next to the engine, so text never leaves the host; the rule
  itself only needs the curve. Weights: [Avifenesh/layby-dwell](https://huggingface.co/Avifenesh/layby-dwell); model
  card and release notes: `model/`. Inference: `layby_dwell/`.
- **Engine adapters** for vLLM 0.30 and 0.31 (`layby/vllm/`) and SGLang 0.5.21 (`layby/sglang/`), plus a sidecar
  proxy (`layby/sidecar/`) that tracks sessions, scores them and sends the curves to the engine.
- **ReturnBench**, a benchmark on public agent and chat traces with a simulator and baselines (`bench/`, `sim/`), and
  the real-engine harness (`engine/`).

## Install

    pip install -e .                 # rule, simulator, ReturnBench
    pip install -e ".[dwell]"        # Layby-Dwell inference (torch, transformers)
    pip install -e ".[sidecar]"      # the sidecar proxy
    pip install -e ".[engine]"       # the real-engine replay client

Tested on: Linux x86_64. The rule, simulator and ReturnBench need Python 3.10 or later and numpy, and run on a CPU.
The vLLM adapter ran inside the `vllm/vllm-openai:v0.30.0` and `v0.31.0` images, the SGLang adapter inside
`lmsysorg/sglang:v0.5.21` and in a Python 3.12 venv with `sglang[all]==0.5.21` (torch 2.13.0+cu130). GPUs: A100 40
and 80 GB (Qwen3-8B rounds) and RTX PRO 6000 on Ubuntu 24.04 with CUDA 13.0 (the GLM round). Driver and CUDA versions
of the A100 boxes: see `engine/REPRODUCE.md`.

## Quick start: the bench on a CPU

Two cells of ReturnBench, under a minute on 8 cores (`bench/README.md`):

    returnbench eval --rules wt cost --cells wildchat:120 swechat:18 --workloads 2 --seeds 1

It prints this (its rows are a subset of the shipped reference run, `reference/rows.jsonl`, and match it bit for bit):

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

`KV_DIR` is a directory on the disk tier (the arm runners empty it before each arm). Then put the sidecar in front of the server. It tags each request with a session key, scores each finished turn with
Layby-Dwell and posts the curve to the adapter's hint port:

    python -m layby.sidecar.proxy --upstream http://127.0.0.1:8000 --listen 8100 \
      --ckpt DWELL/model.safetensors --calib DWELL/config.json --engine vllm --hint-url http://127.0.0.1:8765/hint

Clients talk to port 8100. A client that already knows its curve can skip the sidecar and send it with the request
(`kv_transfer_params: {"park_key": ..., "park_surv": [15 values]}`) or post it later (`POST /hint`).

## Quick start: SGLang

Install this repo into SGLang's environment. That registers the SGLang plugin `park`:

    SGLANG_PLUGINS=park SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python sglang serve --model-path MODEL \
      --enable-hierarchical-cache --hicache-write-policy write_through --hicache-storage-backend file \
      --radix-cache-backend park --radix-eviction-policy park --radix-eviction-policy-config '{"port": 8765}'

The sidecar works the same way with `--engine sglang`.

## Results

Time to first token of returning turns, p95, as a ratio to stock vLLM or SGLang with a CPU tier and no disk (C0).
Below 1 is better. Three repeats, one box each, session bootstrap 95% intervals. K is Layby. T is the stock engine
writing every chunk through to a disk tier. Qwen3-8B on one A100, 16 GiB GPU KV, 28 GiB CPU tier.

| engine, workload | K vs C0 | T vs C0 | K vs T |
|---|---|---|---|
| vLLM 0.30, private agent replay (18 sessions) | 0.69 [0.60, 0.78] | 0.57 [0.49, 0.64] | 1.22 [1.13, 1.28] |
| vLLM 0.30, ReturnBench mix36 (36 sessions, heavy load) | 0.16 [0.14, 0.18] | 0.15 [0.13, 0.17] | 1.11 [0.97, 1.22] |
| SGLang 0.5.21, private agent replay | 0.93 | 0.92 | 1.01 |
| SGLang 0.5.21, ReturnBench mix36 | 0.34 | 0.15 | 2.28 |

These rows ran an adapter with two bugs since fixed (local hit, hint heap) and no wait cap. A rerun of the
private replay with the fixed adapter on vLLM 0.31.0, one repeat per arm on each of three A100 boxes:

| box | K vs C0 | T vs C0 | K vs T | p95 C0 / T / K (s) |
|---|---|---|---|---|
| two SXM4 80GB boxes, disk 3.5 and 5.5 GB/s (pooled) | 0.64 [0.58, 0.74] | 0.71 [0.64, 0.79] | 0.89 [0.84, 1.04] | 4.29 / 2.96 / 2.85 and 4.92 / 3.55 / 2.93 |
| PCIe 40GB box, disk 0.54 GB/s | | | | 8.02 / 4.52 / 13.49 |

On fast disks the gap to write-through is gone. On the slower box Layby lost: see Limits.

In the simulator, over 33 cells of ReturnBench (11 pool and load settings, disks at 0.5, 1.5 and 3 GB/s), the cost rule's p95 is 0.66 of C0 against 0.68 for write-through and 0.60 for an oracle that knows each
return time. Write-through is best on fast disks (0.54 at 3 GB/s against 0.57) and worst on slow ones: up to 1.46 of C0
at 0.5 GB/s, where it floods the disk. The cost rule is never worse than 1.04 of C0 in any cell.

GLM 5.3 Flash FP8 on vLLM 0.31.0, Nebius nodes with 8x RTX PRO 6000: two TP4 engines (the two repeats of each row)
share a network SSD read with O_DIRECT at 0.55 GB/s, so no page cache helps. ReturnBench at real length, contexts up
to 260k tokens. Heavy is 150 sessions, half is 75. Node 1 ran the adapter before the local-hit fix (lesson 4 below).
Node 2 ran the heavy load again with the fixed adapter, with and without the wait cap (lesson 5). The cap is the
default; K without it is the uncapped operating point.

| load | K vs C0 | T vs C0 | K vs T |
|---|---|---|---|
| heavy, node 1, before the fix | 0.67 [0.60, 0.82] | 1.17 [1.15, 1.19] | 0.58 [0.52, 0.70] |
| heavy, node 2, K with the wait cap (default) | 0.79 [0.73, 0.85] | 1.35 [1.31, 1.42] | 0.59 [0.54, 0.63] |
| heavy, node 2, K without the cap | 0.59 [0.52, 0.70] | 1.35 [1.31, 1.42] | 0.44 [0.38, 0.53] |
| half, node 1 (75 sessions) | 0.95 [0.82, 1.08] | 3.99 [3.28, 4.74] | 0.24 [0.20, 0.28] |

Node 2 per engine (A / B), resumed-turn TTFT in seconds:

| arm | p50 | p95 | p99 | recomputed |
|---|---|---|---|---|
| C0, no disk | 54.3 / 62.5 | 275 / 278 | 286 / 286 | 61% / 62% |
| T, write-through | 58.3 / 58.7 | 394 / 353 | 460 / 480 | 56% / 56% |
| K, no cap | 13.6 / 10.1 | 153 / 167 | 1919 / 829 | 39% / 35% |
| K, wait cap (default) | 14.6 / 20.8 | 222 / 208 | 283 / 290 | 39% / 40% |

On a slow shared disk, writing every chunk is worse than having no disk at all. Layby writes only what pays: with the
fixed adapter and no cap it cuts heavy-load p95 to 0.59 of C0, with the cap to 0.79, and it leaves half load where it
was. The simulator predicted 0.80, 1.27 and 0.63 for the heavy load before the disk arms ran. Without the cap the p99
is a starvation tail: 33 of 1,908 turns on engine A waited 30 to 42 minutes while a disk promotion of their prefix
was stuck behind a saturated disk. The cap removes that tail (p99 283 and 290 s against 286 for C0) and costs 1.34x at
p95 and 1.47x at p50 against the uncapped adapter, because each give-up recomputes a prompt of about 58k tokens on a
prefill-bound engine. A 30-minute wait is a failed request in production, so the cap ships as the default. An earlier
return-time guard (recompute before the first lookup when a restore would be slower) cut the tail on node 1 but
recomputed four of five disk returns; it is not in this release.

The paper (arXiv, link to follow) has the method, the full tables and the engineering details.

Qwen3-8B on vLLM 0.31.0, one RTX PRO 6000 per VM, the A100 rounds' capacities (16 GiB GPU KV, 28 GiB CPU tier),
ReturnBench mix36, and a disk tier on the VM's network SSD read with O_DIRECT at 0.48 GB/s. Three repeats per arm:
repeats 1 and 2 on two VMs, one each; repeat 3 of every arm on one further VM. K is Layby after the local-hit fix
(lesson 4); its repeat 3 shares the VM with the other arms' repeat 3, and its repeats 1 and 2 ran on VMs of the same
type, not the same box. The adapter before the fix (two repeats) is kept as its own row. This round ran without the
wait cap, which was built after it. LM is LMCache 0.5.5 with its defaults; LMR is LMCache with
`--l2-prefetch-policy retain` and a 0.95 L1 eviction watermark, the best setting we found for it.

| arm | p50 (s) | p95 (s) | p99 (s) |
|---|---|---|---|
| C0, no disk | 0.12 / 0.12 / 0.12 | 5.7 / 7.8 / 5.6 | 10.4 / 15.3 / 10.9 |
| T, write-through | 0.13 / 0.12 / 0.12 | 8.9 / 13.4 / 8.1 | 91 / 165 / 143 |
| K, Layby | 0.13 / 0.12 / 0.12 | 3.5 / 3.8 / 3.6 | 6.7 / 10.6 / 6.8 |
| K before the local-hit fix | 0.15 / 0.14 | 5.6 / 5.2 | 9.8 / 12.3 |
| LM, LMCache defaults | 12.3 / 15.3 / 13.6 | 51 / 63 / 55 | 63 / 83 / 69 |
| LMR, LMCache retain | 0.18 / 0.17 / 0.18 | 7.6 / 7.5 / 6.6 | 16.3 / 14.6 / 15.0 |

| pair | p95 ratio [95% CI] | p50 ratio |
|---|---|---|
| K vs C0 | 0.59 [0.51, 0.68] | 1.01 |
| K vs T | 0.37 [0.28, 0.56] | 1.01 |
| T vs C0 | 1.61 [1.08, 2.09] | 1.00 |
| LMR vs C0 | 1.21 [1.08, 1.36] | 1.46 |
| K vs LMR | 0.49 [0.44, 0.57] | 0.70 |
| K vs LM | 0.07 [0.06, 0.07] | 0.01 |
| K vs K before the fix (two repeats) | 0.68 [0.61, 0.75] | 0.88 |

LMCache's default prefetch policy deletes a chunk from its CPU tier once the request that restored it from disk ends,
so a returning session reads the disk again; its counters showed more than half of its hits served from disk while
the CPU tier was half empty. `retain` fixes most of that. LMCache could not run GLM 5.3 Flash here: its connector
stored nothing for that model, and at equal GPU KV the engine ran out of memory at warmup.

## Limits

- **The disk in the A100 runs was mostly RAM.** vLLM's file tier falls back from O_DIRECT to buffered I/O on the
  overlay filesystems of those boxes, so recent writes were read back from the OS page cache (about 125 GB of host
  RAM); the cold disk read at about 60 MB/s. Write-through (T) there acts as a larger RAM tier, which is why it wins.
  On a real slow disk (the GLM round) the order flips: T is worse than no disk and Layby is best.
- **A slow disk under CPU-tier pressure can beat the rule.** On a PCIe A100 whose disk ran at 0.54 GB/s, the
  fixed adapter recomputed 1.52M prompt tokens against 0.95M for write-through, p95 13.5 s against 4.5 s. Most of
  it came from sessions the rule left in the CPU tier to spill at eviction: the slower GPU kept more sessions in
  flight, and the eviction-time writes fell behind. The rule prices a spill as a delay to others, not the chance
  it is not done when the session returns. Pricing that would move those sessions to an early write; not tested.
- **Write-back on eviction (KD) is a negative result on vLLM 0.30.** vLLM frees an evicted CPU slot at once, so the
  adapter writes chunks ahead of eviction, and the write pins them. Bursts of pinned chunks starve the CPU tier, and
  the arm recomputes about as much as having no disk (p95 1.67x T). A clean version needs write-on-evict inside vLLM's
  tiering manager.
- **The adapter's own cost is within noise.** With the rule switched off (K0), the vLLM adapter matches stock vLLM on
  the same card within the spread of two stock runs (p95 200.6 s against 197.4 and 206.5 s, recompute within 1%).
- **The private replay is not shipped.** It is the author's own sessions. The public workloads in
  `engine/workloads/` reproduce the setup.
- **GLM 5.3 Flash needs TP4 or less on RTX PRO 6000.** The sm_120 sparse MLA decode kernel has no build for 8 heads
  per GPU, so TP8 does not start.

## Adapter lessons

Each one cost real-engine runs before it was found. All five are fixed, and each has a test that fails on the old
code.

1. **Do not rewrite what is already on disk.** vLLM sends a request-level tier every chunk of the request again on
   each turn. The first disk tier wrote whole contexts again per turn and saturated the disk. It now writes only
   chunks that are neither on disk nor in flight (`tests/test_park_vllm_fs.py`).
2. **Keep telemetry out of the scheduler's hot path.** vLLM calls the block-pool hooks once per block. A returning
   30k-token session touches about 1,900 blocks, and a decayed-estimator update per block cost 20 to 40 ms of
   scheduler time per admission. Events are now buffered and folded in once per step (`tests/test_live_batch.py`).
   The adapter serves an in-process sampling profiler (`/prof`, `/prof/windows`) because containers often forbid
   ptrace.
3. **Evict a session's tail before its prefix.** vLLM reuses a cached prefix only as a contiguous run from the first
   chunk. The first CPU policy refreshed and hinted chunks prefix first, so it evicted chunk 0 first and stranded
   the rest of the session: with the rule off it recomputed 2.02M tokens against 1.10M for stock vLLM. It now walks
   keys last first and copies the running vLLM's own LRU order (`tests/test_policy_prefix.py`,
   `tests/test_policy_lru_parity.py`).
4. **Read the local hit where vLLM reports it.** vLLM 0.30 and 0.31 call `update_state_after_alloc` before they set
   `num_computed_tokens`, so the adapter saw a local hit of 0 on every new request and booked GPU-served prefixes as
   GPU misses and their once-evicted CPU chunks as CPU misses. The inflated miss loss raised the holding price of a
   CPU copy and the rule dropped sessions about to return: 436 of about 880 decisions on the Qwen3-8B slow-disk round,
   85 after the fix. The adapter now records the block-aligned local hit vLLM passes to
   `get_num_new_matched_tokens` and uses it at admission. The fix took Layby's p50 from 1.19 to 1.01 of C0 and its
   p95 from 0.83 to 0.59 (`tests/test_local_hit.py`).
5. **Bound a deferral at what giving up costs.** vLLM holds a request for whole scheduler steps while the disk tier
   promotes its prefix to RAM. With two engines on one 0.55 GB/s disk, 33 of 1,908 resumed turns waited 30 to 42
   minutes for their first token (p99 1,919 s against 286 s with no disk tier). Predicting the disk cost before the
   first lookup and recomputing instead cut the p99 but recomputed four of five disk returns and raised p50 and p95.
   The adapter now lets the promotion run and gives up only once the wait exceeds the recompute of the tokens it
   waits on, at the measured prefill speed: the request is admitted with the prefix RAM can load now and recomputes
   the rest; the promotion lands in RAM when the disk delivers it (`tests/test_defer_cap.py`). On the GLM heavy load
   that took p99 to 283 and 290 s (C0: 286) and cost 1.34x at p95 and 1.47x at p50 against the uncapped adapter. The
   cap is the default; the uncapped adapter is the other operating point. A forward-looking cap, which gives up when
   the expected remaining wait (bytes queued ahead over the load queue's measured drain rate) exceeds the recompute
   cost, estimated the wait well (measured wait 1.2 to 1.4 times the estimate at the median) and gave up after about
   7 s instead of 125 s, but did not beat the elapsed cap: p95 0.82 of no disk on its node against 0.79, p50 0.26
   against 0.32, p99 1.3 times no disk's. A give-up costs a full long-prompt recompute either way. It is not in this
   release.

## Names

The placement layer started as "park", and the engine-facing names keep it: the vLLM classes `ParkConnector`,
`ParkCachePolicy` and `ParkFsTier`, the SGLang plugin, backend and eviction policy `park`, and the request fields
`park_key`, `park_surv`, `park_eta`, `park_disk`. The Python package is `layby`.

## Layout

```
layby/          the cost rule, live estimators, profiler, vLLM and SGLang adapters, sidecar
layby_dwell/    Layby-Dwell inference (features, batching, precisions)
model/          model card, release file list, quantization study
sim/            simulator, baselines, ReturnBench evaluation (python -m sim, or returnbench)
bench/          ReturnBench pools, builders, licenses (bench/README.md)
engine/         real-engine harness: replay client, arm runners, smoke gate, pooling (engine/REPRODUCE.md)
reference/      the simulator reference run
tests/          CPU tests (vLLM 0.30 and 0.31, SGLang 0.5.21)
```

## Tests

    PYTHONPATH=. python tests/test_park_vllm_fs.py          # and the other vLLM tests, in a vLLM 0.30 or 0.31 env
    PYTHONPATH=. python -m pytest tests/test_park_sglang.py # in an SGLang 0.5.21 env with this repo installed

## Citation

The paper is "Layby: placing idle LLM sessions' KV cache by when they come back" (2026; arXiv id to follow).
`CITATION.cff` at the repository root carries the same entry for GitHub and Zenodo. The code is archived on
Zenodo: [10.5281/zenodo.23285213](https://doi.org/10.5281/zenodo.23285213) (all versions; v1.0.0 is
[10.5281/zenodo.23285214](https://doi.org/10.5281/zenodo.23285214)).

```
@misc{fenesh2026layby,
  title  = {Layby: placing idle LLM sessions' KV cache by when they come back},
  author = {Avi Fenesh},
  year   = {2026},
  note   = {arXiv id to follow}
}
```

## License

Code: Apache-2.0 (`LICENSE`). The model is Apache-2.0, like the Laya and ModernBERT weights it starts from. The
ReturnBench pools carry their source datasets' licenses (`bench/LICENSES.md`).
