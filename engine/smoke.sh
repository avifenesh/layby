#!/usr/bin/env bash
# smoke.sh: restore gate for GLM 5.3 Flash FP8 on vLLM 0.31.0 at TP4 (eager) with the Layby adapter in write-through
# mode, so a session evicted from the GPU must come back from the CPU tier and then from disk as external prefix hits.
# Run inside the `park` container (node_prep.sh): docker exec park bash /root/layby/engine/smoke.sh.
# Results in /data/out/SM.*; the client log ends with PASS or FAIL. TP8 fails on RTX PRO 6000 (sm_120 sparse MLA
# decode has no 8-heads-per-rank kernel), TP4 (16 heads) works.
set -u
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VENV=/root/venv ENGINE_DIR=/root/layby/engine REPO=/root/layby PORT=${PORT:-8000} PARK_PORT=${PARK_PORT:-28765}
export PYTHONHASHSEED=0 MODEL=${MODEL:-zai-org/GLM-5.3-Flash} MAXLEN=${MAXLEN:-262144} BOOT_WAIT=${BOOT_WAIT:-360}
export REPLAY_TIMEOUT=3000 VOCAB=${VOCAB:-150000}
export SERVE_ARGS="--tensor-parallel-size ${TP:-4} --enforce-eager --kv-cache-dtype fp8 --prefix-match-unit 256 --revision ${REV:-eb9eb208} --max-num-seqs 32"
export OFF_EXTRA=${OFF_EXTRA:-',"blocks_per_chunk":1,"park_write":"through"'}
export SMOKE_ARGS="--prompt ${PROMPT:-120000} --gpu-tokens ${GPU_TOKENS:-560000} --cpu-tokens ${CPU_TOKENS:-1100000} --filler 30000"
mkdir -p /data/out /data/disk_smoke
$ENGINE_DIR/run_arms.sh /root/replay.json /data/out ${KV_BYTES:-4294967296} ${CPU_BYTES:-68719476736} /data/disk_smoke SM \
  >> /data/out/runner.log 2>&1
echo SMOKE_DONE >> /data/out/runner.log
