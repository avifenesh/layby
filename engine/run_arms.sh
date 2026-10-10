#!/bin/bash
# Real-engine arm runner for vLLM (0.30 for Qwen3-8B, 0.31 for GLM 5.3 Flash). Each arm boots a server, replays a
# workload against it (replay.py) and records per-turn TTFT, then tears it down. Tiers: GPU -> CPU (pinned) -> disk.
#   C0 : stock vLLM, CPU tier only (no disk), LRU
#   T  : stock vLLM tiering, CPU + disk, every chunk written through to disk, LRU
#   T2 : T with frequency-gated disk writes (layby.vllm.freq_tier FreqFsTier: a chunk goes to disk on its n-th
#        sighting, counts decaying): T2_MIN_HITS (n, default 2), T2_DECAY_S (default 600; 0 no decay), T2_COUNT
#        (request, SGLang's hit count, default; or store, KVBM's)
#   CT : Continuum's GPU time-to-live pin after tool calls (layby.vllm.continuum ContinuumConnector), CPU tier as C0,
#        no disk; CT_ETA (the workload's memoryfulness, layby.continuum.memoryfulness; required), telemetry on
#        PARK_PORT; the client sends park_kind, park_tool and park_last (replay.py --continuum)
#   PT : T through Layby's own file tier (ParkFsTier "write_all": every chunk, no rule) beside the stock connector and
#        LRU (pilot only: its I/O path against the native one)
#   K  : Layby: the cost rule in the engine (layby.vllm: ParkConnector, ParkCachePolicy, ParkFsTier) with live scoring:
#        the sidecar (layby.sidecar.proxy) starts once the server is healthy, through SERVER_WRAP (inside the arm's
#        cgroup, on the arm's GPUs), and the client tags turns with kv_transfer_params park_key and, when a turn ends,
#        sends its server-view state from SIDE_FILE to the sidecar's /park/score, which scores it with the frozen
#        model and posts the curve to the hint port (replay.py --park-hints v6 --side-file --sidecar). Needs
#        SIDE_FILE, SIDECAR_CKPT and SIDECAR_CALIB (the Layby-Dwell release); without them the runner exits 4 before
#        any arm. LIVE_SCORING=off (outside Phase R only) posts the precomputed v6 curve instead.
#   K_km, K_ewma : arm K fed the no-model control curves surv_km and surv_ewma (layby.curves annotate) instead of v6
#   K0 : arm K with no curves posted (the rule never runs, nothing written to disk): the adapter's own overhead
#   KW : arm K in write-through mode: every chunk to disk as stored (as T) unless a disk restore of the session
#        would be slower than recomputing it; the rule orders the CPU tier
#   KD : arm K in demand mode: ins/park sessions written through, none sessions written to disk just before the
#        CPU tier evicts them (if the write can pay), drop never (a negative result on vLLM 0.30, see README)
#   SM : smoke: arm K0's server and smoke_restore.py (SMOKE_ARGS) instead of the replay
#   LM : LMCache 0.5.x (its multiprocess server and vLLM connector), the external baseline: L1 = CPU_BYTES of CPU
#        memory with LRU (--l1-size-gb as the exact float CPU_BYTES / 2^30, which LMCache turns back into the same
#        bytes), L2 = the disk dir with O_DIRECT and LRU eviction at DISK_BYTES (max_capacity_gb, exact the same way),
#        every chunk
#        written through (LMCache's default store policy). LM_PORT (default 6555; the HTTP port is LM_PORT + 100),
#        LM_CHUNK (tokens, default 256: a multiple of the engine's block size). LMS: the same stack under
#        smoke_restore.py. LMR: LM with LMCache keeping what it restores from disk in L1 (--l2-prefetch-policy retain;
#        the default deletes a prefetched chunk from L1 when its reader ends, so a session's next turn reads the disk
#        again) and L1 eviction at 0.95 (default 0.8). Needs `lmcache` on PATH in the server's environment.
# A repeat carries a suffix (C0.2 runs arm C0 again into C0.2.*).
# Fairness (BENCH_PLAN sections 5 and 6): every arm but LM gets the same cpu_bytes_to_use (CPU_BYTES) and
# offload_prompt_only false, and LM's L1 the same bytes (checked before any arm: exit 4 if LMCache would round it, or
# if OFF_EXTRA sets cpu_bytes_to_use); every disk tier (T, T2, PT, K*) is vLLM's fs tier class, which opens its files
# with O_DIRECT where the filesystem takes it (the harness flags a buffered fallback); LM's L2 runs with use_direct_io.
# Disk-tier capacity DISK_BYTES (required for every arm with a disk tier; exit 4): LM's L2 evicts at it. vLLM's fs
# tier (and Layby's tiers built on it) has no capacity bound and never deletes, so it cannot be held to DISK_BYTES;
# instead the runner writes the tier directory's size after the replay to ARM.tier_bytes (du -sb; with no eviction the
# size at the end is the largest), and the harness invalidates an arm whose tier exceeded DISK_BYTES (health.py).
# Engine death: after the replay, before replay_end, the server process (kill -0) and its /health, the LMCache server
# process (LM) and the sidecar process (K) are checked; a dead one prints "SERVER DIED" and makes rc 86.
# Usage: run_arms.sh WORKLOAD OUTDIR GPU_KV_BYTES CPU_BYTES DISK_DIR ARM...
# Env: MODEL (default Qwen/Qwen3-8B), VENV (dir with bin/vllm and bin/python, default $HOME/.venv), ENGINE_DIR (this
#      directory, default the script's own), REPO (the layby repo root, default ENGINE_DIR/..), SERVE_ARGS (extra
#      vllm serve args), OFF_EXTRA (extra kv_connector_extra_config entries, each starting with a comma), MAXLEN
#      (default 40960), BOOT_WAIT (5 s polls, default 180), REPLAY_TIMEOUT (s, default 5400), VOCAB (replay token
#      ids, default 150000), HARD_CAP (replay --hard-cap seconds), PORT (default 8000), PARK_PORT (hint port, default
#      8765). The server runs in its own session (setsid); cleanup kills only that process group.
#      Phase R (engine/phase_r/orchestrate.py): SERVER_WRAP (a command prefix for the server and the LMCache server,
#      e.g. "phase_r/cg_exec.sh CGROUP" to start them inside the arm's cgroup), ARM_HOOK (called as
#      "ARM_HOOK EVENT ARM OUTDIR" with replay_start once the server is healthy and replay_end after the replay),
#      PARK_HINTS (the curve arm K's client posts, default v6). DISK_BYTES (disk-tier capacity, bytes). Live scoring
#      (arm K): SIDE_FILE, SIDECAR_CKPT, SIDECAR_CALIB, SIDECAR_PORT (default PARK_PORT + 1), SIDECAR_DEVICE (default
#      the sidecar's own choice: cuda when present), SIDECAR_WAIT (5 s polls, default 60). EVICTION (the CPU-tier
#      eviction of the arms on vLLM's
#      own policy, C0 T T2 CT PT: lru or arc, default lru), FS_THREADS (read and write threads of every disk tier,
#      default 16). REPLAY_TIMEOUT must exceed HARD_CAP by 300 s (the cap writes the censored rows).
set -u
W=$1; O=$2; KV=$3; CPU=$4; DISK=$5; shift 5
MODEL=${MODEL:-Qwen/Qwen3-8B}
mkdir -p $O
VENV=${VENV:-$HOME/.venv}; ENGINE_DIR=${ENGINE_DIR:-$(cd "$(dirname "$0")" && pwd)}
export PATH=$VENV/bin:$PATH
REPO=${REPO:-$(dirname $ENGINE_DIR)}
export PYTHONPATH=$ENGINE_DIR:$REPO:${PYTHONPATH:-}
PARK_PORT=${PARK_PORT:-8765}
export VLLM_USE_FLASHINFER_SAMPLER=0
unset PYTORCH_CUDA_ALLOC_CONF   # expandable segments break the pinned CPU tier (vLLM refuses to start)
PORT=${PORT:-8000}
RT=${REPLAY_TIMEOUT:-5400}
if [ -n "${HARD_CAP:-}" ] && [ "$(python3 -c "print(int(float('$RT') >= float('$HARD_CAP') + 300))")" != 1 ]; then
  echo "REPLAY_TIMEOUT $RT must exceed HARD_CAP $HARD_CAP by 300 s"; exit 2
fi
# preflight (exit 4 before any arm): live scoring inputs for K, DISK_BYTES for every disk arm, equal CPU-tier bytes
LIVE=${LIVE_SCORING:-on}
for arm in "$@"; do
  case ${arm%%.*} in
    K) if [ "$LIVE" != off ]; then
         [ -n "${SIDE_FILE:-}" ] && [ -f "$SIDE_FILE" ] || { echo "PREFLIGHT: arm $arm needs SIDE_FILE (live scoring side file) ${SIDE_FILE:-unset}"; exit 4; }
         [ -n "${SIDECAR_CKPT:-}" ] && [ -f "$SIDECAR_CKPT" ] && [ -n "${SIDECAR_CALIB:-}" ] && [ -f "$SIDECAR_CALIB" ] || {
           echo "PREFLIGHT: arm $arm needs SIDECAR_CKPT and SIDECAR_CALIB (the Layby-Dwell release)"; exit 4; }
       fi ;;
  esac
  case ${arm%%.*} in
    T|T2|PT|K|K_km|K_ewma|K0|KW|KD|SM|LM|LMS|LMR)
      [[ "${DISK_BYTES:-}" =~ ^[0-9]+$ ]] || { echo "PREFLIGHT: arm $arm needs DISK_BYTES (disk-tier capacity in bytes), got '${DISK_BYTES:-}'"; exit 4; } ;;
  esac
done
case "${OFF_EXTRA:-}" in *cpu_bytes_to_use*) echo "PREFLIGHT: OFF_EXTRA sets cpu_bytes_to_use; every arm gets CPU_BYTES"; exit 4 ;; esac
# LMCache takes GiB as a float and sizes int(gb * 2^30): the shortest float repr of bytes / 2^30 is exact
L1GB=$(python3 -c "print(repr(int('$CPU') / 2**30))")
[ "$(python3 -c "print(int(float('$L1GB') * (1 << 30)))")" = "$CPU" ] || { echo "PREFLIGHT: LMCache L1 $L1GB GiB is not CPU_BYTES $CPU"; exit 4; }
L2GB=""
if [ -n "${DISK_BYTES:-}" ]; then
  L2GB=$(python3 -c "print(repr(int('$DISK_BYTES') / 2**30))")
  [ "$(python3 -c "print(int(float('$L2GB') * 1024**3))")" = "$DISK_BYTES" ] || { echo "PREFLIGHT: LMCache L2 $L2GB GiB is not DISK_BYTES"; exit 4; }
fi
# alive SERVER_PID [LMCACHE_PID] [SIDECAR_PID]: what died during the replay ("" when everything is up)
alive() {
  local why=""
  kill -0 "$1" 2>/dev/null || why="server process $1 gone"
  curl -sf -m 10 localhost:$PORT/health >/dev/null || why="${why:+$why; }server /health failed"
  [ -n "${2:-}" ] && { kill -0 "$2" 2>/dev/null || why="${why:+$why; }LMCache server process $2 gone"; }
  [ -n "${3:-}" ] && { kill -0 "$3" 2>/dev/null || why="${why:+$why; }sidecar process $3 gone"; }
  echo "$why"
}
TH=${FS_THREADS:-16}
FSIO="\"root_dir\":\"$DISK\",\"n_read_threads\":$TH,\"n_write_threads\":$TH"
for arm in "$@"; do
  pol="\"eviction_policy\":\"${EVICTION:-lru}\""; src=none; conn='"kv_connector":"OffloadingConnector"'; extra=""; client=""; hints=""; SCP=""
  tier=",\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"fs\",$FSIO}]"
  case ${arm%%.*} in
    C0) tier="" ;;
    T) ;;
    T2) tier=",\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"FreqFsTier\",\"module_path\":\"layby.vllm.freq_tier\",$FSIO,\"min_hits\":${T2_MIN_HITS:-2},\"decay_s\":${T2_DECAY_S:-600},\"count\":\"${T2_COUNT:-request}\"}]" ;;
    CT) [ -n "${CT_ETA:-}" ] || { echo "arm $arm: CT_ETA unset (layby.continuum.memoryfulness of the workload)"; continue; }
       tier=""; conn='"kv_connector":"ContinuumConnector","kv_connector_module_path":"layby.vllm.continuum"'
       extra=",\"ct_eta\":$CT_ETA,\"ct_port\":$PARK_PORT"; client="--continuum" ;;
    PT) tier=",\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"ParkFsTier\",\"module_path\":\"layby.vllm.fs_tier\",$FSIO,\"write_all\":true}]" ;;
    LM|LMS|LMR) conn='"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector"'
       tier=""; pol=""; extra="\"lmcache.mp.port\":${LM_PORT:-6555}" ;;
    K|K_km|K_ewma|K0|KW|KD|SM) pol='"eviction_policy":"ParkCachePolicy","cache_policy_module_path":"layby.vllm.policy"'
       tier=",\"spec_name\":\"TieringOffloadingSpec\",\"secondary_tiers\":[{\"type\":\"ParkFsTier\",\"module_path\":\"layby.vllm.fs_tier\",$FSIO}]"
       conn='"kv_connector":"ParkConnector","kv_connector_module_path":"layby.vllm.connector"'
       case ${arm%%.*} in K_km) hints=km ;; K_ewma) hints=ewma ;; *) hints=${PARK_HINTS:-v6} ;; esac
       extra=",\"park_port\":$PARK_PORT,\"park_prof\":1"; client="--park-hints $hints --park-port $PARK_PORT"
       [ ${arm%%.*} = K0 ] && client=""                                  # no curves: the rule never runs
       [ ${arm%%.*} = SM ] && client=""                                  # smoke: restore gate, no replay
       [ ${arm%%.*} = KW ] && extra="$extra,\"park_write\":\"through\""   # write-through mode
       [ ${arm%%.*} = KD ] && extra="$extra,\"park_write\":\"demand\"" ;;  # disk-decided write-through, spill on CPU eviction
    *) echo "unknown arm $arm"; continue ;;
  esac
  rm -rf $DISK; mkdir -p $DISK
  SHM0=$(ls /dev/shm | grep '^vllm_offload_' | sort)   # other servers' CPU tiers: never touched
  OFF="{$conn,\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"cpu_bytes_to_use\":$CPU,\"offload_prompt_only\":false,$pol$tier$extra${OFF_EXTRA:-}}}"
  LMP=""
  case ${arm%%.*} in LM|LMS|LMR)
    OFF="{$conn,\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$extra}}"
    LMX=""; [ ${arm%%.*} = LMR ] && LMX="--l2-prefetch-policy retain --eviction-trigger-watermark 0.95 --eviction-ratio 0.1"
    L2="{\"type\":\"nixl_store_dynamic\",\"backend\":\"POSIX\",\"backend_params\":{\"file_path\":\"$DISK\",\"use_direct_io\":\"true\",\"max_capacity_gb\":\"$L2GB\"},\"persist_enabled\":false,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.9,\"eviction_ratio\":0.1}}"
    setsid ${SERVER_WRAP:-} lmcache server --port ${LM_PORT:-6555} --http-port $(( ${LM_PORT:-6555} + 100 )) --chunk-size ${LM_CHUNK:-256} --separate-object-groups \
      --l1-size-gb $L1GB --eviction-policy LRU $LMX --l2-adapter "$L2" > $O/$arm.lmcache.log 2>&1 &
    LMP=$!; sleep 20 ;;
  esac
  echo "=== arm $arm $(date -u +%H:%M:%S)"
  setsid ${SERVER_WRAP:-} $VENV/bin/vllm serve $MODEL --port $PORT --max-model-len ${MAXLEN:-40960} --enable-prefix-caching \
    --kv-cache-memory-bytes $KV --max-num-seqs 64 --enable-prompt-tokens-details ${SERVE_ARGS:-} \
    --kv-transfer-config "$OFF" > $O/$arm.server.log 2>&1 &
  SP=$!
  trap "kill -9 -- -$SP ${LMP:+-$LMP} 2>/dev/null" EXIT INT TERM   # a killed runner takes its server groups with it
  for i in $(seq 1 ${BOOT_WAIT:-180}); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 5; done
  curl -sf localhost:$PORT/health >/dev/null || { echo "server failed"; tail -30 $O/$arm.server.log; kill -9 -- -$SP ${LMP:+-$LMP} 2>/dev/null; wait $SP 2>/dev/null
    comm -13 <(echo "$SHM0") <(ls /dev/shm | grep '^vllm_offload_' | sort) | sed 's|^|/dev/shm/|' | xargs -r rm -f; continue; }
  if [ ${arm%%.*} = K ] && [ "$LIVE" != off ]; then   # live scoring: the sidecar, in the arm's cgroup, on its GPUs
    SCPORT=${SIDECAR_PORT:-$((PARK_PORT + 1))}
    setsid ${SERVER_WRAP:-} $VENV/bin/python -m layby.sidecar.proxy --upstream http://127.0.0.1:$PORT --listen $SCPORT \
      --engine vllm --hint-url http://127.0.0.1:$PARK_PORT/hint --ckpt "$SIDECAR_CKPT" --calib "$SIDECAR_CALIB" \
      ${SIDECAR_DEVICE:+--device $SIDECAR_DEVICE} > $O/$arm.sidecar.log 2>&1 &
    SCP=$!
    # shellcheck disable=SC2064  # the trap takes this arm's process groups now
    trap "kill -9 -- -$SP ${LMP:+-$LMP} -$SCP 2>/dev/null" EXIT INT TERM
    for _ in $(seq 1 ${SIDECAR_WAIT:-60}); do curl -sf localhost:$SCPORT/park/stats >/dev/null && break; kill -0 $SCP 2>/dev/null || break; sleep 5; done
    curl -sf localhost:$SCPORT/park/stats >/dev/null || { echo "server failed: sidecar did not start"; tail -30 $O/$arm.sidecar.log
      kill -9 -- -$SP ${LMP:+-$LMP} -$SCP 2>/dev/null; wait $SP 2>/dev/null
      # shellcheck disable=SC2010  # the same /dev/shm cleanup as below
      comm -13 <(echo "$SHM0") <(ls /dev/shm | grep '^vllm_offload_' | sort) | sed 's|^|/dev/shm/|' | xargs -r rm -f; continue; }
    client="$client --side-file $SIDE_FILE --sidecar http://127.0.0.1:$SCPORT"
  fi
  [ -n "${ARM_HOOK:-}" ] && $ARM_HOOK replay_start "$arm" "$O"
  if [ ${arm%%.*} = SM ] || [ ${arm%%.*} = LMS ]; then
    timeout $RT $VENV/bin/python $ENGINE_DIR/smoke_restore.py --url http://127.0.0.1:$PORT --model $MODEL \
      --vocab ${VOCAB:-150000} ${SMOKE_ARGS:-} > $O/$arm.client.log 2>&1
  else
  timeout $RT $VENV/bin/python $ENGINE_DIR/replay.py $W $O/$arm.jsonl --url http://127.0.0.1:$PORT --cpu-hint $src --prefetch $src --model $MODEL --vocab ${VOCAB:-150000} ${HARD_CAP:+--hard-cap $HARD_CAP} $client > $O/$arm.client.log 2>&1
  fi
  rc=$?
  dead=$(alive $SP "$LMP" "$SCP")
  if [ -n "$dead" ]; then echo "arm $arm: SERVER DIED during the replay: $dead"; [ $rc = 0 ] && rc=86; fi
  du -sb $DISK 2>/dev/null | cut -f1 > $O/$arm.tier_bytes
  [ -n "${ARM_HOOK:-}" ] && $ARM_HOOK replay_end "$arm" "$O"
  case ${arm%%.*} in K|K_km|K_ewma|K0|KW|KD|SM) curl -sf localhost:$PARK_PORT/live > $O/$arm.park_live.json
                              curl -sf localhost:$PARK_PORT/prof > $O/$arm.prof.json
                              curl -sf localhost:$PARK_PORT/prof/windows > $O/$arm.profw.json ;;
    CT) curl -sf localhost:$PARK_PORT/live > $O/$arm.ct_live.json
        curl -sf localhost:$PARK_PORT/log > $O/$arm.ct_log.json ;; esac
  echo "arm $arm rc=$rc $(grep -c ttft $O/$arm.jsonl) turns"
  [ -n "$SCP" ] && kill -9 -- -$SCP 2>/dev/null
  kill $SP; sleep 10
  # hard-kill what the server left behind (engine core): its own process group only, then free its CPU tier
  kill -9 -- -$SP 2>/dev/null
  [ -n "$LMP" ] && { kill -- -$LMP 2>/dev/null; sleep 5; kill -9 -- -$LMP 2>/dev/null; }
  wait $SP 2>/dev/null; sleep 3
  # a killed server leaks its CPU tier in /dev/shm: remove only files this arm created
  comm -13 <(echo "$SHM0") <(ls /dev/shm | grep '^vllm_offload_' | sort) | sed 's|^|/dev/shm/|' | xargs -r rm -f
done
rm -rf $DISK
echo ALL_DONE
