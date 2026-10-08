#!/bin/bash
# The arm runner of run_arms.sh for SGLang 0.5.21 (HiCache), same capacities and the same replay.
#   C0 : HiCache host tier only (no storage), LRU, write_through
#   T  : C0 + file L3 storage, every node written through to storage
#   K  : the Layby cost rule in the engine (layby.sglang: --radix-cache-backend park, park eviction
#        order, storage filter, hint port); the client tags turns with custom_params park_key and
#        posts each turn's v6 curve to the hint port when it ends (replay.py --park-hints v6)
# A repeat carries a suffix (C0.2 runs arm C0 again into C0.2.*).
# Usage: run_arms_sglang.sh WORKLOAD OUTDIR GPU_KV_BYTES CPU_BYTES DISK_DIR ARM...
# Capacities as run_arms.sh: GPU_KV_BYTES becomes --max-total-tokens (GPU_KV_BYTES / BPT tokens),
# CPU_BYTES the host pool as --hicache-ratio (CPU_BYTES / GPU_KV_BYTES), DISK_DIR the file backend's
# directory (SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR, emptied before each arm).
# Env: MODEL, VENV (default /root/.venv-sglang, with this repo installed: uv pip install -e REPO, which registers the
#      SGLang plugin entry point `park`), ENGINE_DIR (this directory, default the script's own), BPT (KV bytes per token, all layers; default 147456,
#      Qwen3-8B bf16: 36 layers x 8 KV heads x 128 x 2 x 2 bytes), PAGE (page size, default 64),
#      TREE_CORE (default python for every arm: the park backend needs the Python TreeCore, so the
#      baselines use it too), SERVE_ARGS (extra sglang serve args), PORT (default 30000),
#      PARK_PORT (default 8765), WINDOW (replay --window seconds; unset: whole workload).
# The plugin loads only in arm K (SGLANG_PLUGINS=park; other arms SGLANG_PLUGINS=none).
# The server runs in its own session (setsid); cleanup kills only that process group.
set -u
W=$1; O=$2; KV=$3; CPU=$4; DISK=$5; shift 5
MODEL=${MODEL:-Qwen/Qwen3-8B}
mkdir -p $O
VENV=${VENV:-/root/.venv-sglang}; ENGINE_DIR=${ENGINE_DIR:-$(cd "$(dirname "$0")" && pwd)}
export PATH=$VENV/bin:$PATH
BPT=${BPT:-147456}; PAGE=${PAGE:-64}
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=${TREE_CORE:-python}
PORT=${PORT:-30000}; PARK_PORT=${PARK_PORT:-8765}
TOKENS=$(python3 -c "print(int($KV) // int($BPT))")
RATIO=$(python3 -c "print(round(int($CPU) / int($KV), 6))")
WIN=${WINDOW:+--window $WINDOW}
for arm in "$@"; do
  storage=""; park=""; client=""; plugins=none
  case ${arm%%.*} in
    C0) ;;
    T) storage="--hicache-storage-backend file" ;;
    K) storage="--hicache-storage-backend file"; plugins=park
       park="--radix-cache-backend park --radix-eviction-policy park --radix-eviction-policy-config {\"port\":$PARK_PORT}"
       client="--park-hints v6 --park-port $PARK_PORT" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  rm -rf $DISK; mkdir -p $DISK
  echo "=== arm $arm $(date -u +%H:%M:%S)"
  SGLANG_PLUGINS=$plugins SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$DISK setsid $VENV/bin/sglang serve \
    --model-path $MODEL --port $PORT --context-length 40960 --max-running-requests 64 \
    --page-size $PAGE --max-total-tokens $TOKENS --enable-metrics \
    --enable-hierarchical-cache --hicache-ratio $RATIO --hicache-write-policy write_through \
    $storage $park ${SERVE_ARGS:-} > $O/$arm.server.log 2>&1 &
  SP=$!
  for i in $(seq 1 180); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  curl -sf localhost:$PORT/health >/dev/null || { echo "server failed"; tail -30 $O/$arm.server.log; kill -9 -- -$SP 2>/dev/null; wait $SP 2>/dev/null; continue; }
  timeout 5400 $VENV/bin/python $ENGINE_DIR/replay.py $W $O/$arm.jsonl --engine sglang --url http://127.0.0.1:$PORT \
    --model $MODEL $client $WIN > $O/$arm.client.log 2>&1
  rc=$?
  [ -n "$client" ] && curl -sf localhost:$PARK_PORT/live > $O/$arm.park_live.json
  echo "arm $arm rc=$rc $(grep -c ttft $O/$arm.jsonl) turns"
  kill $SP; sleep 10
  kill -9 -- -$SP 2>/dev/null   # what the server left behind (scheduler, detokenizer): its own process group only
  wait $SP 2>/dev/null; sleep 3
done
rm -rf $DISK
echo ALL_DONE
