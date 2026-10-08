#!/usr/bin/env bash
# engine.sh NAME GPUS PORT PARK_PORT ARM...: one TP4 GLM 5.3 Flash engine on GPUS running the arms in order with
# run_arms.sh on REPLAY (ReturnBench mix150_real). Outputs in $OUT/NAME; ends with ALL_ARMS_DONE.
# Env, all optional: LAYBY_ROOT (this repo; default the checkout this script lives in), REPLAY (default
# $HOME/replay.json, where node_prep.sh puts the workload), DATA (the VM's data mount, default /data), OUT (default
# $DATA/out), KV_DIR (the disk tier directory, default $DATA/disk_NAME), VENV (default $HOME/venv), KV_BYTES,
# CPU_BYTES, WINDOW, REPLAY_TIMEOUT.
set -u
N=$1; export CUDA_VISIBLE_DEVICES=$2 PORT=$3 PARK_PORT=$4; shift 4
LAYBY_ROOT=${LAYBY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DATA=${DATA:-/data}; REPLAY=${REPLAY:-$HOME/replay.json}; OUT=${OUT:-$DATA/out}; KV_DIR=${KV_DIR:-$DATA/disk_$N}
export VENV=${VENV:-$HOME/venv} ENGINE_DIR=$LAYBY_ROOT/engine REPO=$LAYBY_ROOT PYTHONHASHSEED=0
export MODEL=zai-org/GLM-5.3-Flash MAXLEN=262144 BOOT_WAIT=360 VOCAB=150000
export WINDOW=${WINDOW:-4500} REPLAY_TIMEOUT=${REPLAY_TIMEOUT:-5700}
export SERVE_ARGS="--tensor-parallel-size 4 --enforce-eager --kv-cache-dtype fp8 --prefix-match-unit 256 --revision eb9eb208 --max-num-seqs 32"
export OFF_EXTRA=',"blocks_per_chunk":1'
O=$OUT/$N; mkdir -p $O
for arm in "$@"; do
  $LAYBY_ROOT/engine/run_arms.sh $REPLAY $O ${KV_BYTES:-4294967296} ${CPU_BYTES:-30064771072} $KV_DIR $arm >> $O/runner.log 2>&1
done
echo ALL_ARMS_DONE >> $O/runner.log
