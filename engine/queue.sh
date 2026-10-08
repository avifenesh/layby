#!/usr/bin/env bash
# queue.sh ARM[:TAG] ...: once no arm runner is active on this box (deadline 2 h of waiting), run each ARM in order
# with run_arms.sh on the box's own replay. ARM:TAG first keeps the box's existing ARM outputs as ARM.TAG.*.
# Ends with ALL_ARMS_DONE.
set -u
end=$(( $(date +%s) + 2*3600 ))
until ! pgrep -f '[b]ox_run.sh|[r]un_arms3.sh|[a]fter_k.sh|[a]fter_arm.sh' >/dev/null; do
  [ "$(date +%s)" -lt "$end" ] || { echo deadline; exit 1; }; sleep 30
done
sed -i 's/^ALL_ARMS_DONE$/ARMS_DONE_BEFORE_QUEUE/' /root/out/runner.log
export VENV=/root/venv ENGINE_DIR=/root/layby/engine REPO=/root/layby PORT=8000 PARK_PORT=28765 MODEL=Qwen/Qwen3-8B
for spec in "$@"; do
  A=${spec%%:*}; TAG=""; [ "$spec" != "$A" ] && TAG=${spec#*:}
  if [ -n "$TAG" ]; then
    for f in /root/out/$A.jsonl /root/out/$A.client.log /root/out/$A.server.log /root/out/$A.park_live.json /root/out/$A.prof.json; do
      [ -e "$f" ] && mv "$f" "${f/\/$A./\/$A.$TAG.}"
    done
    echo "=== queue: $A outputs kept as $A.$TAG.*" >> /root/out/runner.log
  fi
  /root/layby/engine/run_arms.sh /root/replay.json /root/out 17179869184 30064771072 /root/kvdisk $A >> /root/out/runner.log 2>&1
done
echo ALL_ARMS_DONE >> /root/out/runner.log
