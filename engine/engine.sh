#!/usr/bin/env bash
# engine.sh NAME GPUS PORT PARK_PORT ARM...: one TP4 GLM 5.3 Flash engine on GPUS running the arms in order with
# run_arms.sh on /root/replay.json (ReturnBench mix150_real). Outputs in /data/out/NAME; ends with ALL_ARMS_DONE.
set -u
N=$1; export CUDA_VISIBLE_DEVICES=$2 PORT=$3 PARK_PORT=$4; shift 4
export VENV=/root/venv ENGINE_DIR=/root/layby/engine REPO=/root/layby PYTHONHASHSEED=0
export MODEL=zai-org/GLM-5.3-Flash MAXLEN=262144 BOOT_WAIT=360 VOCAB=150000
export WINDOW=${WINDOW:-4500} REPLAY_TIMEOUT=${REPLAY_TIMEOUT:-5700}
export SERVE_ARGS="--tensor-parallel-size 4 --enforce-eager --kv-cache-dtype fp8 --prefix-match-unit 256 --revision eb9eb208 --max-num-seqs 32"
export OFF_EXTRA=',"blocks_per_chunk":1'
O=/data/out/$N; mkdir -p $O
for arm in "$@"; do
  /root/layby/engine/run_arms.sh /root/replay.json $O ${KV_BYTES:-4294967296} ${CPU_BYTES:-30064771072} /data/disk_$N $arm >> $O/runner.log 2>&1
done
echo ALL_ARMS_DONE >> $O/runner.log
