#!/bin/bash
# The arm runner of run_arms.sh for SGLang 0.5.21 (HiCache), same capacities and the same replay.
#   C0 : HiCache host tier only (no storage), LRU, write_through
#   T  : C0 + file L3 storage, every node written through to storage
#   K  : the Layby cost rule in the engine (layby.sglang: --radix-cache-backend park, park eviction
#        order, storage filter, hint port) with live scoring: the sidecar (layby.sidecar.proxy --engine sglang)
#        starts once the server is healthy, through SERVER_WRAP (the arm's cgroup and GPU); the client tags turns
#        with custom_params park_key and sends each finished turn's state from SIDE_FILE to the sidecar, which
#        scores it and posts the curve to the hint port (replay.py --park-hints v6 --side-file --sidecar). Needs
#        SIDE_FILE, SIDECAR_CKPT, SIDECAR_CALIB (exit 4 before any arm without them); LIVE_SCORING=off (outside
#        Phase R only) posts the precomputed v6 curve instead.
# A repeat carries a suffix (C0.2 runs arm C0 again into C0.2.*).
# Usage: run_arms_sglang.sh WORKLOAD OUTDIR GPU_KV_BYTES CPU_BYTES DISK_DIR ARM...
# Capacities as run_arms.sh: GPU_KV_BYTES becomes --max-total-tokens (GPU_KV_BYTES / BPT tokens),
# CPU_BYTES the host pool as --hicache-ratio (CPU_BYTES / GPU_KV_BYTES), DISK_DIR the file backend's
# directory (SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR, emptied before each arm), DISK_BYTES (env, required for T and
# K*: exit 4) its LRU file evictor's cap (SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE, bytes). After the replay the tier
# directory's size goes to ARM.tier_bytes (du -sb). Engine death: before replay_end the server process (kill -0), its
# /health and the sidecar process (K) are checked; a dead one prints "SERVER DIED" and makes rc 86.
# Env: MODEL, VENV (default $HOME/.venv-sglang, with this repo installed: uv pip install -e REPO, which registers the
#      SGLang plugin entry point `park`), ENGINE_DIR (this directory, default the script's own), BPT (KV bytes per token, all layers; default 147456,
#      Qwen3-8B bf16: 36 layers x 8 KV heads x 128 x 2 x 2 bytes), PAGE (page size, default 64),
#      TREE_CORE (default python for every arm: the park backend needs the Python TreeCore, so the
#      baselines use it too), SERVE_ARGS (extra sglang serve args), PORT (default 30000),
#      PARK_PORT (default 8765), HARD_CAP (replay --hard-cap seconds; unset: no cap), REPLAY_TIMEOUT (s, default 5400;
#      must exceed HARD_CAP by 300 s, as the cap writes the censored rows), BOOT_WAIT (5 s polls, default 180).
#      Arms K_km and K_ewma: arm K fed the control curves surv_km and surv_ewma (layby.curves annotate).
#      Live scoring (arm K): SIDE_FILE, SIDECAR_CKPT, SIDECAR_CALIB, SIDECAR_PORT (default PARK_PORT + 1),
#      SIDECAR_DEVICE, SIDECAR_WAIT (5 s polls, default 60); REPO (the layby repo root, default ENGINE_DIR/..) is put
#      on PYTHONPATH for the sidecar.
#      Phase R (engine/phase_r/orchestrate.py): SERVER_WRAP (a command prefix for the server, e.g.
#      "phase_r/cg_exec.sh CGROUP"), ARM_HOOK (called as "ARM_HOOK EVENT ARM OUTDIR" with replay_start once the server
#      is healthy and replay_end after the replay), PARK_HINTS (the curve arm K posts, default v6), WRITE_POLICY
#      (--hicache-write-policy, default write_through; write_through_selective for HC-sel). Every arm needs the
#      HiCacheFile fadvise patch (phase_r/sglang/apply_patch.sh); the orchestrator checks it before each arm.
# The plugin loads only in arm K (SGLANG_PLUGINS=park; other arms SGLANG_PLUGINS=none).
# The server runs in its own session (setsid); cleanup kills only that process group.
set -u
W=$1; O=$2; KV=$3; CPU=$4; DISK=$5; shift 5
MODEL=${MODEL:-Qwen/Qwen3-8B}
mkdir -p $O
VENV=${VENV:-$HOME/.venv-sglang}; ENGINE_DIR=${ENGINE_DIR:-$(cd "$(dirname "$0")" && pwd)}
export PATH=$VENV/bin:$PATH
BPT=${BPT:-147456}; PAGE=${PAGE:-64}
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=${TREE_CORE:-python}
PORT=${PORT:-30000}; PARK_PORT=${PARK_PORT:-8765}
TOKENS=$(python3 -c "print(int($KV) // int($BPT))")
RATIO=$(python3 -c "print(round(int($CPU) / int($KV), 6))")
WIN=${HARD_CAP:+--hard-cap $HARD_CAP}
RT=${REPLAY_TIMEOUT:-5400}
if [ -n "${HARD_CAP:-}" ] && [ "$(python3 -c "print(int(float('$RT') >= float('$HARD_CAP') + 300))")" != 1 ]; then
  echo "REPLAY_TIMEOUT $RT must exceed HARD_CAP $HARD_CAP by 300 s"; exit 2
fi
LIVE=${LIVE_SCORING:-on}
for arm in "$@"; do   # preflight (exit 4 before any arm)
  case ${arm%%.*} in
    K) if [ "$LIVE" != off ]; then
         [ -n "${SIDE_FILE:-}" ] && [ -f "$SIDE_FILE" ] || { echo "PREFLIGHT: arm $arm needs SIDE_FILE (live scoring side file) ${SIDE_FILE:-unset}"; exit 4; }
         [ -n "${SIDECAR_CKPT:-}" ] && [ -f "$SIDECAR_CKPT" ] && [ -n "${SIDECAR_CALIB:-}" ] && [ -f "$SIDECAR_CALIB" ] || {
           echo "PREFLIGHT: arm $arm needs SIDECAR_CKPT and SIDECAR_CALIB (the Layby-Dwell release)"; exit 4; }
       fi ;;
  esac
  case ${arm%%.*} in
    T|K|K_km|K_ewma) [[ "${DISK_BYTES:-}" =~ ^[0-9]+$ ]] || { echo "PREFLIGHT: arm $arm needs DISK_BYTES (disk-tier capacity in bytes), got '${DISK_BYTES:-}'"; exit 4; } ;;
  esac
done
REPO=${REPO:-$(dirname "$ENGINE_DIR")}
alive() {   # alive SERVER_PID [SIDECAR_PID]: what died during the replay ("" when everything is up)
  local why=""
  kill -0 "$1" 2>/dev/null || why="server process $1 gone"
  curl -sf -m 10 localhost:$PORT/health >/dev/null || why="${why:+$why; }server /health failed"
  [ -n "${2:-}" ] && { kill -0 "$2" 2>/dev/null || why="${why:+$why; }sidecar process $2 gone"; }
  echo "$why"
}
for arm in "$@"; do
  storage=""; park=""; client=""; plugins=none; SCP=""
  wp=${WRITE_POLICY:-write_through}
  case ${arm%%.*} in
    C0) ;;
    T) storage="--hicache-storage-backend file" ;;
    K|K_km|K_ewma) storage="--hicache-storage-backend file"; plugins=park
       wp=${WRITE_POLICY:-write_through_selective}
       park="--radix-cache-backend park --radix-eviction-policy park --radix-eviction-policy-config {\"port\":$PARK_PORT}"
       case ${arm%%.*} in K_km) hints=km ;; K_ewma) hints=ewma ;; *) hints=${PARK_HINTS:-v6} ;; esac
       client="--park-hints $hints --park-port $PARK_PORT" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  rm -rf $DISK; mkdir -p $DISK
  echo "=== arm $arm $(date -u +%H:%M:%S)"
  SGLANG_PLUGINS=$plugins SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=$DISK SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE=${DISK_BYTES:-0} setsid ${SERVER_WRAP:-} $VENV/bin/sglang serve \
    --model-path $MODEL --port $PORT --context-length 40960 --max-running-requests 64 \
    --page-size $PAGE --max-total-tokens $TOKENS --enable-metrics \
    --enable-hierarchical-cache --hicache-ratio $RATIO --hicache-write-policy $wp \
    $storage $park ${SERVE_ARGS:-} > $O/$arm.server.log 2>&1 &
  SP=$!
  for i in $(seq 1 ${BOOT_WAIT:-180}); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  curl -sf localhost:$PORT/health >/dev/null || { echo "server failed"; tail -30 $O/$arm.server.log; kill -9 -- -$SP 2>/dev/null; wait $SP 2>/dev/null; continue; }
  # shellcheck disable=SC2064  # the trap takes this arm's process groups now
  trap "kill -9 -- -$SP 2>/dev/null" EXIT INT TERM   # a killed runner takes its server group with it
  if [ ${arm%%.*} = K ] && [ "$LIVE" != off ]; then   # live scoring: the sidecar, in the arm's cgroup, on its GPU
    SCPORT=${SIDECAR_PORT:-$((PARK_PORT + 1))}
    PYTHONPATH=$REPO:${PYTHONPATH:-} setsid ${SERVER_WRAP:-} $VENV/bin/python -m layby.sidecar.proxy \
      --upstream http://127.0.0.1:$PORT --listen $SCPORT --engine sglang --hint-url http://127.0.0.1:$PARK_PORT/hint \
      --ckpt "$SIDECAR_CKPT" --calib "$SIDECAR_CALIB" ${SIDECAR_DEVICE:+--device $SIDECAR_DEVICE} \
      ${SIDECAR_ENCODER:+--encoder $SIDECAR_ENCODER} ${SIDECAR_EXTRA:-} > $O/$arm.sidecar.log 2>&1 &
    SCP=$!
    # shellcheck disable=SC2064  # the trap takes this arm's process groups now
    trap "kill -9 -- -$SP -$SCP 2>/dev/null" EXIT INT TERM
    for _ in $(seq 1 ${SIDECAR_WAIT:-60}); do curl -sf localhost:$SCPORT/park/stats >/dev/null && break; kill -0 $SCP 2>/dev/null || break; sleep 5; done
    curl -sf localhost:$SCPORT/park/stats >/dev/null || { echo "server failed: sidecar did not start"; tail -30 $O/$arm.sidecar.log
      kill -9 -- -$SP -$SCP 2>/dev/null; wait $SP 2>/dev/null; continue; }
    client="$client --side-file $SIDE_FILE --sidecar http://127.0.0.1:$SCPORT"
  fi
  [ -n "${ARM_HOOK:-}" ] && $ARM_HOOK replay_start "$arm" "$O"
  timeout $RT $VENV/bin/python $ENGINE_DIR/replay.py $W $O/$arm.jsonl --engine sglang --url http://127.0.0.1:$PORT \
    --model $MODEL $client $WIN > $O/$arm.client.log 2>&1
  rc=$?
  dead=$(alive $SP "$SCP")
  if [ -n "$dead" ]; then echo "arm $arm: SERVER DIED during the replay: $dead"; [ $rc = 0 ] && rc=86; fi
  du -sb $DISK 2>/dev/null | cut -f1 > $O/$arm.tier_bytes
  [ -n "${ARM_HOOK:-}" ] && $ARM_HOOK replay_end "$arm" "$O"
  [ -n "$client" ] && curl -sf localhost:$PARK_PORT/live > $O/$arm.park_live.json
  [ -n "$SCP" ] && curl -sf localhost:${SIDECAR_PORT:-$((PARK_PORT + 1))}/park/stats > $O/$arm.sidecar_stats.json || true
  echo "arm $arm rc=$rc $(grep -c ttft $O/$arm.jsonl) turns"
  [ -n "$SCP" ] && kill -9 -- -$SCP 2>/dev/null
  kill $SP; sleep 10
  kill -9 -- -$SP 2>/dev/null   # what the server left behind (scheduler, detokenizer): its own process group only
  wait $SP 2>/dev/null; sleep 3
done
rm -rf $DISK
echo ALL_DONE
