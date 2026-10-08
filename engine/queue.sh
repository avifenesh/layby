#!/usr/bin/env bash
# queue.sh ARM[:TAG] ...: once no arm runner is active on this box (deadline 2 h of waiting), run each ARM in order
# with run_arms.sh on the box's own replay. ARM:TAG first keeps the box's existing ARM outputs as ARM.TAG.*.
# Ends with ALL_ARMS_DONE.
# Env, all optional, the same as box_run.sh: LAYBY_ROOT (default the checkout this script lives in), REPLAY (default
# $HOME/replay.json), OUT (default $HOME/out), KV_DIR (default $HOME/kvdisk), VENV (default $HOME/venv).
set -u
end=$(( $(date +%s) + 2*3600 ))
until ! pgrep -f '[b]ox_run.sh|[r]un_arms3.sh|[a]fter_k.sh|[a]fter_arm.sh' >/dev/null; do
  [ "$(date +%s)" -lt "$end" ] || { echo deadline; exit 1; }; sleep 30
done
LAYBY_ROOT=${LAYBY_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
REPLAY=${REPLAY:-$HOME/replay.json}; OUT=${OUT:-$HOME/out}; KV_DIR=${KV_DIR:-$HOME/kvdisk}
sed -i 's/^ALL_ARMS_DONE$/ARMS_DONE_BEFORE_QUEUE/' $OUT/runner.log
export VENV=${VENV:-$HOME/venv} ENGINE_DIR=$LAYBY_ROOT/engine REPO=$LAYBY_ROOT PORT=8000 PARK_PORT=28765 MODEL=Qwen/Qwen3-8B
for spec in "$@"; do
  A=${spec%%:*}; TAG=""; [ "$spec" != "$A" ] && TAG=${spec#*:}
  if [ -n "$TAG" ]; then
    for f in $OUT/$A.jsonl $OUT/$A.client.log $OUT/$A.server.log $OUT/$A.park_live.json $OUT/$A.prof.json; do
      [ -e "$f" ] && mv "$f" "${f/\/$A./\/$A.$TAG.}"
    done
    echo "=== queue: $A outputs kept as $A.$TAG.*" >> $OUT/runner.log
  fi
  $LAYBY_ROOT/engine/run_arms.sh $REPLAY $OUT 17179869184 30064771072 $KV_DIR $A >> $OUT/runner.log 2>&1
done
echo ALL_ARMS_DONE >> $OUT/runner.log
