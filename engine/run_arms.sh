#!/bin/bash
# Real-engine arm runner for vLLM (0.30 for Qwen3-8B, 0.31 for GLM 5.3 Flash). Each arm boots a server, replays a
# workload against it (replay.py) and records per-turn TTFT, then tears it down. Tiers: GPU -> CPU (pinned) -> disk.
#   C0 : stock vLLM, CPU tier only (no disk), LRU
#   T  : stock vLLM tiering, CPU + disk, every chunk written through to disk, LRU
#   K  : Layby: the cost rule in the engine (layby.vllm: ParkConnector, ParkCachePolicy, ParkFsTier); the client
#        tags turns with kv_transfer_params park_key and posts each turn's Layby-Dwell curve to the hint port when
#        it ends (replay.py --park-hints v6), as the sidecar (layby.sidecar.proxy) would
#   K0 : arm K with no curves posted (the rule never runs, nothing written to disk): the adapter's own overhead
#   KW : arm K in write-through mode: every chunk to disk as stored (as T) unless a disk restore of the session
#        would be slower than recomputing it; the rule orders the CPU tier
#   KD : arm K in demand mode: ins/park sessions written through, none sessions written to disk just before the
#        CPU tier evicts them (if the write can pay), drop never (a negative result on vLLM 0.30, see README)
#   SM : smoke: arm K0's server and smoke_restore.py (SMOKE_ARGS) instead of the replay
# A repeat carries a suffix (C0.2 runs arm C0 again into C0.2.*).
# Usage: run_arms.sh WORKLOAD OUTDIR GPU_KV_BYTES CPU_BYTES DISK_DIR ARM...
# Env: MODEL (default Qwen/Qwen3-8B), VENV (dir with bin/vllm and bin/python, default /root/.venv), ENGINE_DIR (this
#      directory, default the script's own), REPO (the layby repo root, default ENGINE_DIR/..), SERVE_ARGS (extra
#      vllm serve args), OFF_EXTRA (extra kv_connector_extra_config entries, each starting with a comma), MAXLEN
#      (default 40960), BOOT_WAIT (5 s polls, default 180), REPLAY_TIMEOUT (s, default 5400), VOCAB (replay token
#      ids, default 150000), WINDOW (replay --window seconds), PORT (default 8000), PARK_PORT (hint port, default
#      8765). The server runs in its own session (setsid); cleanup kills only that process group.
set -u
W=$1; O=$2; KV=$3; CPU=$4; DISK=$5; shift 5
MODEL=${MODEL:-Qwen/Qwen3-8B}
mkdir -p $O
VENV=${VENV:-/root/.venv}; ENGINE_DIR=${ENGINE_DIR:-$(cd "$(dirname "$0")" && pwd)}
export PATH=$VENV/bin:$PATH
REPO=${REPO:-$(dirname $ENGINE_DIR)}
export PYTHONPATH=$ENGINE_DIR:$REPO:${PYTHONPATH:-}
PARK_PORT=${PARK_PORT:-8765}
export VLLM_USE_FLASHINFER_SAMPLER=0
unset PYTORCH_CUDA_ALLOC_CONF   # expandable segments break the pinned CPU tier (vLLM refuses to start)
PORT=${PORT:-8000}
for arm in "$@"; do
  pol='"eviction_policy":"lru"'; src=none; conn='"kv_connector":"OffloadingConnector"'; extra=""; client=""
  tier=",\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"fs\",\"root_dir\":\"$DISK\",\"n_read_threads\":16,\"n_write_threads\":16}]"
  case ${arm%%.*} in
    C0) tier="" ;;
    T) ;;
    K|K0|KW|KD|SM) pol='"eviction_policy":"ParkCachePolicy","cache_policy_module_path":"layby.vllm.policy"'
       tier=",\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"ParkFsTier\",\"module_path\":\"layby.vllm.fs_tier\",\"root_dir\":\"$DISK\",\"n_read_threads\":16,\"n_write_threads\":16}]"
       conn='"kv_connector":"ParkConnector","kv_connector_module_path":"layby.vllm.connector"'
       extra=",\"park_port\":$PARK_PORT,\"park_prof\":1"; client="--park-hints v6 --park-port $PARK_PORT"
       [ ${arm%%.*} = K0 ] && client=""                                  # no curves: the rule never runs
       [ ${arm%%.*} = SM ] && client=""                                  # smoke: restore gate, no replay
       [ ${arm%%.*} = KW ] && extra="$extra,\"park_write\":\"through\""   # write-through mode
       [ ${arm%%.*} = KD ] && extra="$extra,\"park_write\":\"demand\"" ;;  # disk-decided write-through, spill on CPU eviction
    *) echo "unknown arm $arm"; continue ;;
  esac
  rm -rf $DISK; mkdir -p $DISK
  SHM0=$(ls /dev/shm | grep '^vllm_offload_' | sort)   # other servers' CPU tiers: never touched
  OFF="{$conn,\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU,\"offload_prompt_only\":false,$pol$tier$extra${OFF_EXTRA:-}}}"
  echo "=== arm $arm $(date -u +%H:%M:%S)"
  setsid $VENV/bin/vllm serve $MODEL --port $PORT --max-model-len ${MAXLEN:-40960} --enable-prefix-caching \
    --kv-cache-memory-bytes $KV --max-num-seqs 64 --enable-prompt-tokens-details ${SERVE_ARGS:-} \
    --kv-transfer-config "$OFF" > $O/$arm.server.log 2>&1 &
  SP=$!
  trap "kill -9 -- -$SP 2>/dev/null" EXIT INT TERM   # a killed runner takes its server group with it
  for i in $(seq 1 ${BOOT_WAIT:-180}); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  curl -sf localhost:$PORT/health >/dev/null || { echo "server failed"; tail -30 $O/$arm.server.log; kill -9 -- -$SP 2>/dev/null; wait $SP 2>/dev/null
    comm -13 <(echo "$SHM0") <(ls /dev/shm | grep '^vllm_offload_' | sort) | sed 's|^|/dev/shm/|' | xargs -r rm -f; continue; }
  if [ ${arm%%.*} = SM ]; then
    timeout ${REPLAY_TIMEOUT:-5400} $VENV/bin/python $ENGINE_DIR/smoke_restore.py --url http://127.0.0.1:$PORT --model $MODEL \
      --vocab ${VOCAB:-150000} ${SMOKE_ARGS:-} > $O/$arm.client.log 2>&1
  else
  timeout ${REPLAY_TIMEOUT:-5400} $VENV/bin/python $ENGINE_DIR/replay.py $W $O/$arm.jsonl --url http://127.0.0.1:$PORT --cpu-hint $src --prefetch $src --model $MODEL --vocab ${VOCAB:-150000} ${WINDOW:+--window $WINDOW} $client > $O/$arm.client.log 2>&1
  fi
  rc=$?
  case ${arm%%.*} in K|K0|KW|KD|SM) curl -sf localhost:$PARK_PORT/live > $O/$arm.park_live.json
                              curl -sf localhost:$PARK_PORT/prof > $O/$arm.prof.json
                              curl -sf localhost:$PARK_PORT/prof/windows > $O/$arm.profw.json ;; esac
  echo "arm $arm rc=$rc $(grep -c ttft $O/$arm.jsonl) turns"
  kill $SP; sleep 10
  # hard-kill what the server left behind (engine core): its own process group only, then free its CPU tier
  kill -9 -- -$SP 2>/dev/null
  wait $SP 2>/dev/null; sleep 3
  # a killed server leaks its CPU tier in /dev/shm: remove only files this arm created
  comm -13 <(echo "$SHM0") <(ls /dev/shm | grep '^vllm_offload_' | sort) | sed 's|^|/dev/shm/|' | xargs -r rm -f
done
rm -rf $DISK
echo ALL_DONE
