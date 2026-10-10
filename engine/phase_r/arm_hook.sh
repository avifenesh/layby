#!/usr/bin/env bash
# arm_hook.sh EVENT ARM OUTDIR: the runners' ARM_HOOK. Appends "EVENT EPOCH_SECONDS" to OUTDIR/ARM.markers, so the
# harness can cut the arm's telemetry to the replay window (boot excluded: loading the weights under memory.max is
# not part of the measured window). Events: replay_start (server healthy, replay about to start), replay_end.
set -u
printf '%s %s\n' "$1" "$(date +%s.%N)" >> "$3/$2.markers"
