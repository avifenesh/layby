# Reproduce the real-engine runs

Every run replays a content-free workload (token counts, gaps, kinds and Layby-Dwell curves; no text) against a real
server and records each turn's time to first token. An arm is one server configuration. A repeat is one box. Arms are
compared inside a repeat, then pooled over repeats with a session bootstrap.

Arms (run_arms.sh, run_arms_sglang.sh):

| arm | what runs |
|---|---|
| C0 | stock engine, CPU tier only, no disk, LRU |
| T | stock engine, CPU tier plus a disk tier written through, LRU |
| K | Layby: the cost rule in the engine, curves posted per turn |
| K0 | the Layby adapter with no curves (the rule never runs): the adapter's own overhead |
| KW | K in write-through mode (vLLM only) |
| KD | K with write-back on CPU eviction (vLLM only; a negative result, see the README) |

Workloads are in `workloads/`, with bytes, sha256, source pools and licenses per file in `workloads/MANIFEST.json`
(`python scripts/verify_manifest.py` checks them; `box_run.sh` runs the check first). The private replay used in the
paper (the author's own sessions) is not shipped.

Paths. Every script takes its locations from environment variables and falls back to the values the rounds used:
`LAYBY_ROOT` (this repo; default the checkout the script lives in), `REPLAY` (the workload file; default
`$HOME/replay.json`), `OUT` (outputs; default `$HOME/out` for the Qwen3-8B boxes, `$DATA/out` for the Nebius VM),
`KV_DIR` (the disk tier directory, emptied before each arm; default `$HOME/kvdisk` or `$DATA/disk_NAME`), `DATA`
(the VM's data mount; default `/data`), `HF_HOME` (model cache in `node_prep.sh`; default `$DATA/hf`), `VENV`
(default `$HOME/venv`). The rounds ran as root, so `$HOME` was root's home.

| file | sessions | turns | contexts |
|---|---|---|---|
| `mix36_w0.json` | 36 (SWE-chat, TraceLab Claude, WildChat) | 916 | scaled to 32k tokens |
| `mix150_real.json` | 150 (60 SWE-chat, 25 TraceLab, 65 WildChat) | 7,858 | real, up to 260k tokens |
| `mix36_real.json` | 36 | 1,649 | real |

Build your own from the ReturnBench pools:

    # 32k-scaled pools -> engine workload; this exact command rebuilds workloads/mix36_w0.json
    python engine/build_scaled_workload.py mix36_w0.json 36 0 \
      bench/pools/swechat.json bench/pools/tracelab_claude.json bench/pools/wildchat.json
    # real-length pools (bench/REAL_LENGTH.md): bench/build/build_replay_pop.py --no-scale --max-turns 0 --max-hist 0
    # --ctx-cap 262144, then
    python engine/build_real_workload.py mix150_real.json 150 \
      --pools swechat.real.json:swechat.real.refs.parquet tracelab_claude.real.json:tracelab_claude.real.refs.parquet \
              wildchat.real.json:wildchat.real.refs.parquet \
      --probs CURVES.parquet

## 1. vLLM 0.30, Qwen3-8B, one A100 per repeat

Hardware: one A100 (40 or 80 GB). Image `vllm/vllm-openai:v0.30.0`. GPU KV 16 GiB, CPU tier 28 GiB, disk tier on the
box's local disk. Check every card for thermal throttling (`nvidia-smi --query-gpu=clocks.sm,temperature.gpu,
clocks_throttle_reasons.active`); a throttled card runs about 4x slower and its repeat is not comparable.

To add: driver and CUDA version of the A100 boxes (the image is pinned; the host driver was not recorded).

    # on each box, inside the image, from a checkout of this repo
    cp engine/workloads/mix36_w0.json $HOME/replay.json   # REPLAY
    bash engine/box_run.sh ""   C0 T K      # repeat 1
    bash engine/box_run.sh .2   T K C0      # repeat 2 on a second box
    bash engine/box_run.sh .3   K C0 T      # repeat 3 on a third box

Outputs land in `$OUT` (default `$HOME/out`): `ARM.jsonl` per turn, `ARM.env.json` (GPUs, driver, torch and CUDA
versions, repo commit, server version), `ARM.server.log`, `ARM.park_live.json` telemetry, `ARM.prof.json` and
`ARM.profw.json` scheduler-thread profiles. Pool:

    mkdir pooled && cp box1/{C0,T,K}.jsonl box2/{C0,T,K}.2.jsonl box3/{C0,T,K}.3.jsonl pooled/
    python engine/pool_arms.py pooled --pair K:C0 --pair K:T --pair T:C0 --out pooled/pooled.json

## 2. SGLang 0.5.21, Qwen3-8B, one A100 per repeat

Image `lmsysorg/sglang:v0.5.21`. Install this repo into the image's own environment (it registers the SGLang plugin
entry point `park`), then:

    pip install aiohttp -e .                                  # from a checkout of this repo
    VENV=$(dirname $(dirname $(command -v sglang))) WINDOW=0 \
      bash engine/run_arms_sglang.sh $HOME/replay.json $HOME/out 17179869184 30064771072 $HOME/kvdisk K T C0

`run_arms_sglang.sh` runs every arm on SGLang's Python radix tree core (the Layby backend needs it, so the baselines use
it too).

## 3. vLLM 0.31.0, GLM 5.3 Flash FP8, Nebius 8x RTX PRO 6000

VM: Nebius `gpu-rtx6000-a` preset `8gpu-192vcpu-1744gb`, image `ubuntu24.04-cuda13.0`, 1200 GiB network SSD boot
disk. Two engines at TP4 (GPUs 0-3 and 4-7). TP8 does not start on this card: the sm_120 sparse MLA decode kernel has
no 8-heads-per-rank build. FP8 KV, `--prefix-match-unit 256`, eager mode, GPU KV 4 GiB per GPU (522,156 tokens per
engine), CPU tier 28 GiB per engine, disk tier on the boot disk (O_DIRECT; it measured 0.55 GB/s).

    # as root on the VM, from a checkout of this repo under $HOME (node_prep.sh mounts $HOME and $DATA into the container)
    bash engine/node_prep.sh                                  # container `park`, deps, model download; PREP_DONE
    docker exec park bash $PWD/engine/smoke.sh                # restore gate; $OUT/SM.client.log ends PASS
    docker exec -d park bash $PWD/engine/engine.sh A 0,1,2,3 8000 28765 C0 T K C0.3 T.3 K.3
    docker exec -d park bash $PWD/engine/engine.sh B 4,5,6,7 8001 28766 C0.2 T.2 K.2 C0.4 T.4 K.4

Both engines run the same arm at the same time, so each disk-using arm shares the disk with its own twin. Each arm is
bounded by `WINDOW=4500` seconds. Outputs are in `$OUT/A` and `$OUT/B` (`OUT` defaults to `$DATA/out`); A holds
repeats 1 and 3, B holds 2 and 4. Pool as in section 1.

The smoke gate a new model or card must pass before a run: a 120k-token session evicted from the GPU comes back as
external prefix-cache hits covering the whole prefix, first from the CPU tier and then from disk, with the same greedy
output (`engine/smoke_restore.py`).

## Telemetry

The Layby adapter serves `GET /live` (rates, measured link speeds, decision counts, disk tier counters), `GET /prof`
and `GET /prof/windows` (a sampling profile of the engine's scheduler thread, whole run and per minute with the
engine's load) on its hint port. `run_arms.sh` saves all three at the end of each Layby arm.
