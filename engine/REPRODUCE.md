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

Workloads are in `workloads/`. The private replay used in the paper (the author's own sessions) is not shipped.

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

    # on each box, this repo at /root/layby
    cp /root/layby/engine/workloads/mix36_w0.json /root/replay.json
    bash /root/layby/engine/box_run.sh ""   C0 T K      # repeat 1
    bash /root/layby/engine/box_run.sh .2   T K C0      # repeat 2 on a second box
    bash /root/layby/engine/box_run.sh .3   K C0 T      # repeat 3 on a third box

Outputs land in `/root/out` (`ARM.jsonl` per turn, `ARM.server.log`, `ARM.park_live.json` telemetry, `ARM.prof.json` and
`ARM.profw.json` scheduler-thread profiles). Pool:

    mkdir pooled && cp box1/{C0,T,K}.jsonl box2/{C0,T,K}.2.jsonl box3/{C0,T,K}.3.jsonl pooled/
    python engine/pool_arms.py pooled --pair K:C0 --pair K:T --pair T:C0

## 2. SGLang 0.5.21, Qwen3-8B, one A100 per repeat

Image `lmsysorg/sglang:v0.5.21`. Install this repo into the image's own environment (it registers the SGLang plugin
entry point `park`), then:

    pip install aiohttp -e /root/layby
    VENV=$(dirname $(dirname $(command -v sglang))) WINDOW=0 \
      bash /root/layby/engine/run_arms_sglang.sh /root/replay.json /root/out 17179869184 30064771072 /root/kvdisk K T C0

`run_arms_sglang.sh` runs every arm on SGLang's Python radix tree core (the Layby backend needs it, so the baselines use
it too).

## 3. vLLM 0.31.0, GLM 5.3 Flash FP8, Nebius 8x RTX PRO 6000

VM: Nebius `gpu-rtx6000-a` preset `8gpu-192vcpu-1744gb`, image `ubuntu24.04-cuda13.0`, 1200 GiB network SSD boot
disk. Two engines at TP4 (GPUs 0-3 and 4-7). TP8 does not start on this card: the sm_120 sparse MLA decode kernel has
no 8-heads-per-rank build. FP8 KV, `--prefix-match-unit 256`, eager mode, GPU KV 4 GiB per GPU (522,156 tokens per
engine), CPU tier 28 GiB per engine, disk tier on the boot disk (O_DIRECT; it measured 0.55 GB/s).

    # as root on the VM, this repo at /root/layby
    bash /root/layby/engine/node_prep.sh                      # container `park`, deps, model download; PREP_DONE
    docker exec park bash /root/layby/engine/smoke.sh         # restore gate; /data/out/SM.client.log ends PASS
    docker exec -d park bash /root/layby/engine/engine.sh A 0,1,2,3 8000 28765 C0 T K C0.3 T.3 K.3
    docker exec -d park bash /root/layby/engine/engine.sh B 4,5,6,7 8001 28766 C0.2 T.2 K.2 C0.4 T.4 K.4

Both engines run the same arm at the same time, so each disk-using arm shares the disk with its own twin. Each arm is
bounded by `WINDOW=4500` seconds. Outputs are in `/data/out/A` and `/data/out/B`; A holds repeats 1 and 3, B holds
2 and 4. Pool as in section 1.

The smoke gate a new model or card must pass before a run: a 120k-token session evicted from the GPU comes back as
external prefix-cache hits covering the whole prefix, first from the CPU tier and then from disk, with the same greedy
output (`engine/smoke_restore.py`).

## Telemetry

The Layby adapter serves `GET /live` (rates, measured link speeds, decision counts, disk tier counters), `GET /prof`
and `GET /prof/windows` (a sampling profile of the engine's scheduler thread, whole run and per minute with the
engine's load) on its hint port. `run_arms.sh` saves all three at the end of each Layby arm.
