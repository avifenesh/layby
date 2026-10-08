#!/usr/bin/env bash
# node_prep.sh: prepare a GPU VM (Ubuntu with Docker and NVIDIA runtime) for the GLM 5.3 Flash round. Run as root.
# Starts a long-lived container `park` from vllm/vllm-openai:$VLLM_TAG with $DATA and $HOME mounted at the same paths,
# installs the client deps, copies the workload to $HOME/replay.json and downloads the model. Logs to $HOME/prep.log;
# ends with PREP_DONE or PREP_FAILED. The workloads already carry their Layby-Dwell curves, so no scoring is needed here.
# Env, all optional: LAYBY_ROOT (this repo; default the checkout this script lives in; keep it under $HOME or $DATA so
# the container sees it at the same path), DATA (the VM's data mount, default /data), HF_HOME (model cache, default
# $DATA/hf), OUT (default $DATA/out), VLLM_TAG, MODEL, REV.
set -u
VLLM_TAG=${VLLM_TAG:-v0.31.0}
MODEL=${MODEL:-zai-org/GLM-5.3-Flash}
REV=${REV:-eb9eb208}
LAYBY_ROOT=${LAYBY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
DATA=${DATA:-/data}; HF_HOME=${HF_HOME:-$DATA/hf}; OUT=${OUT:-$DATA/out}
exec >> $HOME/prep.log 2>&1
echo "=== prep $(date -u +%H:%M:%S) image $VLLM_TAG model $MODEL@$REV"
mkdir -p $HF_HOME $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -2
docker pull vllm/vllm-openai:$VLLM_TAG || { echo PREP_FAILED pull; exit 1; }
docker rm -f park >/dev/null 2>&1
docker run -d --name park --gpus all --ipc host --network host --shm-size 64g \
  -e HF_HOME=$HF_HOME -e PYTHONHASHSEED=0 -v $DATA:$DATA -v $HOME:$HOME --entrypoint sleep \
  vllm/vllm-openai:$VLLM_TAG infinity || { echo PREP_FAILED run; exit 1; }
docker exec park bash -lc "pip install -q aiohttp numpy && mkdir -p $HOME/venv/bin && ln -sf \$(command -v vllm) $HOME/venv/bin/vllm && ln -sf \$(command -v python3) $HOME/venv/bin/python && cp $LAYBY_ROOT/engine/workloads/mix150_real.json $HOME/replay.json" \
  || { echo PREP_FAILED deps; exit 1; }
echo "=== download $(date -u +%H:%M:%S)"
docker exec park bash -lc "hf download $MODEL --revision $REV --max-workers 32 > $OUT/download.log 2>&1" \
  || { echo PREP_FAILED download; tail -5 $OUT/download.log; exit 1; }
du -sh $HF_HOME
echo "PREP_DONE $(date -u +%H:%M:%S)"
