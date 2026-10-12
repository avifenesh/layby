#!/usr/bin/env bash
# arm_hook.sh EVENT ARM OUTDIR: the runners' ARM_HOOK. Appends "EVENT EPOCH_SECONDS" to OUTDIR/ARM.markers, so the
# harness can cut the arm's telemetry to the replay window (boot excluded: loading the weights under memory.max is
# not part of the measured window). Events: replay_start (server healthy, replay about to start), replay_end.
#
# replay_start first reclaims the arm cgroup's unmapped page cache (memory.stat file - max(shmem, file_mapped): the
# clean pages of the weights read at boot) through memory.reclaim, so the window starts without it and the
# page-cache rule (health.py) sees only what the arm reads through the cache during the replay. With memory.swap.max
# 0 the kernel can only drop file pages. The cgroup is ARM_CGROUP, else the directory after cg_exec.sh in
# SERVER_WRAP; before and after are logged to OUTDIR/ARM.reclaim ("reclaim BEFORE_BYTES AFTER_BYTES").
set -u
unmapped() {
  awk '$1 == "file" {f = $2} $1 == "shmem" {s = $2} $1 == "file_mapped" {m = $2}
       END {x = (s > m) ? s : m; v = f - x; print (v > 0) ? v : 0}' "$1/memory.stat"
}
if [ "$1" = replay_start ]; then
  cg="${ARM_CGROUP:-}"
  if [ -z "$cg" ]; then
    read -r w c _ <<< "${SERVER_WRAP:-}"
    case "${w:-}" in *cg_exec.sh) cg="${c:-}" ;; esac
  fi
  if [ -n "$cg" ] && [ -r "$cg/memory.stat" ] && [ -w "$cg/memory.reclaim" ]; then
    before=$(unmapped "$cg")
    [ "$before" -gt 0 ] && { echo "$before" > "$cg/memory.reclaim"; } 2>/dev/null
    printf 'reclaim %s %s\n' "$before" "$(unmapped "$cg")" >> "$3/$2.reclaim"
  fi
fi
printf '%s %s\n' "$1" "$(date +%s.%N)" >> "$3/$2.markers"
