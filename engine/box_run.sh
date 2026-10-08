#!/usr/bin/env bash
# Runs on a rented box (vllm/vllm-openai:v0.30.0 image): one repeat of the real-engine A/B, arms in the given order.
# Usage: box_run.sh REPEAT ARM...   (REPEAT suffix: "" for the first box, ".2", ".3"). Expects this repo at
# /root/layby and the workload at /root/replay.json (e.g. engine/workloads/mix36_w0.json).
set -u
R=$1; shift
mkdir -p /root/venv/bin /root/out /root/kvdisk
pip install -q aiohttp
ln -sf $(command -v vllm) /root/venv/bin/vllm; ln -sf $(command -v python3) /root/venv/bin/python
export VENV=/root/venv ENGINE_DIR=/root/layby/engine REPO=/root/layby PORT=8000 PARK_PORT=28765 MODEL=Qwen/Qwen3-8B
export HF_HUB_ENABLE_HF_TRANSFER=0
python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-8B'))" > /root/out/model.log 2>&1
for arm in "$@"; do
  /root/layby/engine/run_arms.sh /root/replay.json /root/out 17179869184 30064771072 /root/kvdisk "$arm$R" >> /root/out/runner.log 2>&1
done
echo ALL_ARMS_DONE >> /root/out/runner.log
