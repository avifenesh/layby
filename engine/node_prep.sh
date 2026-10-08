#!/usr/bin/env bash
# node_prep.sh: prepare a GPU VM (Ubuntu with Docker and NVIDIA runtime) for the GLM 5.3 Flash round. Run as root.
# Expects this repo at /root/layby. Starts a long-lived container `park` from vllm/vllm-openai:$VLLM_TAG with /data and
# /root mounted, installs the client deps, and downloads the model. Logs to /root/prep.log; ends with PREP_DONE or
# PREP_FAILED. The workloads already carry their Layby-Dwell curves, so no scoring is needed here.
set -u
VLLM_TAG=${VLLM_TAG:-v0.31.0}
MODEL=${MODEL:-zai-org/GLM-5.3-Flash}
REV=${REV:-eb9eb208}
exec >> /root/prep.log 2>&1
echo "=== prep $(date -u +%H:%M:%S) image $VLLM_TAG model $MODEL@$REV"
mkdir -p /data/hf /data/out
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2
docker pull vllm/vllm-openai:$VLLM_TAG || { echo PREP_FAILED pull; exit 1; }
docker rm -f park >/dev/null 2>&1
docker run -d --name park --gpus all --ipc host --network host --shm-size 64g \
  -e HF_HOME=/data/hf -e PYTHONHASHSEED=0 -v /data:/data -v /root:/root --entrypoint sleep \
  vllm/vllm-openai:$VLLM_TAG infinity || { echo PREP_FAILED run; exit 1; }
docker exec park bash -lc 'pip install -q aiohttp numpy && mkdir -p /root/venv/bin && ln -sf $(command -v vllm) /root/venv/bin/vllm && ln -sf $(command -v python3) /root/venv/bin/python && cp /root/layby/engine/workloads/mix150_real.json /root/replay.json' \
  || { echo PREP_FAILED deps; exit 1; }
echo "=== download $(date -u +%H:%M:%S)"
docker exec park bash -lc "hf download $MODEL --revision $REV --max-workers 32 > /data/out/download.log 2>&1" \
  || { echo PREP_FAILED download; tail -5 /data/out/download.log; exit 1; }
du -sh /data/hf
echo "PREP_DONE $(date -u +%H:%M:%S)"
