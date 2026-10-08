#!/usr/bin/env bash
# stop_arms.sh [now]: stop pending after_* scripts; with "now" also stop the running arm (runner, replay, server) and
# clean this box's vLLM shared-memory CPU tiers.
pkill -f '[a]fter_k.sh'; pkill -f '[a]fter_arm.sh'; pkill -f '[q]ueue.sh'
if [ "${1:-}" = now ]; then
  pkill -f '[b]ox_run.sh'; pkill -f '[r]un_arms3.sh'; pkill -f '[r]eplay.py'; pkill -f '[v]llm serve'
  sleep 8; pkill -9 -f '[v]llm serve'; pkill -9 -f '[E]ngineCore'; sleep 3
  for f in /dev/shm/vllm_offload_*; do [ -e "$f" ] && rm -f -- "$f"; done
fi
pgrep -fa '[v]llm serve|[r]eplay.py|[b]ox_run|[a]fter_|[q]ueue.sh' | cut -c1-80
echo stopped ${1:-pending}
