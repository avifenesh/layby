#!/usr/bin/env bash
# Runs on a rented box (vllm/vllm-openai:v0.30.0 image): one repeat of the real-engine A/B, arms in the given order.
# Usage: box_run.sh REPEAT ARM...   (REPEAT suffix: "" for the first box, ".2", ".3").
# Env, all optional: LAYBY_ROOT (this repo; default the checkout this script lives in), REPLAY (the workload to replay;
# default $HOME/replay.json, e.g. a copy of engine/workloads/mix36_w0.json), OUT (outputs; default $HOME/out), KV_DIR
# (the disk tier directory; default $HOME/kvdisk), VENV (default $HOME/venv). The rounds ran as root inside the
# container, so these defaults were root's home.
set -u
R=$1; shift
LAYBY_ROOT=${LAYBY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
REPLAY=${REPLAY:-$HOME/replay.json}; OUT=${OUT:-$HOME/out}; KV_DIR=${KV_DIR:-$HOME/kvdisk}; VENV=${VENV:-$HOME/venv}
mkdir -p $VENV/bin $OUT $KV_DIR
pip install -q aiohttp
ln -sf $(command -v vllm) $VENV/bin/vllm; ln -sf $(command -v python3) $VENV/bin/python
export VENV ENGINE_DIR=$LAYBY_ROOT/engine REPO=$LAYBY_ROOT PORT=8000 PARK_PORT=28765 MODEL=Qwen/Qwen3-8B
export HF_HUB_ENABLE_HF_TRANSFER=0
python3 $LAYBY_ROOT/scripts/verify_manifest.py $LAYBY_ROOT/engine/workloads/MANIFEST.json >> $OUT/runner.log 2>&1 || echo "workload manifest check FAILED" >> $OUT/runner.log
python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-8B'))" > $OUT/model.log 2>&1
for arm in "$@"; do
  $LAYBY_ROOT/engine/run_arms.sh $REPLAY $OUT 17179869184 30064771072 $KV_DIR "$arm$R" >> $OUT/runner.log 2>&1
done
echo ALL_ARMS_DONE >> $OUT/runner.log
