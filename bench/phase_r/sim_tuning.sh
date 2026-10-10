#!/usr/bin/env bash
# K_ewma's alpha in the simulator on the Phase R tuning workloads (BENCH_PLAN section 8), CPU only.
# Rows go to OUT/alpha_{plan,ref}.jsonl; `python -m sim table ROWS --per-cell` summarizes them.
# Capacities "plan": GPU KV 0.15 P (at least one max-length context, which vLLM needs to start), CPU tier 0.4 P,
# P the builder's peak live KV of the model's measured workload. "ref": the profile's reference engine.
# Disks: Qwen3-8B 0.5 (slow, io.max 450 MiB/s) and 1.5 GB/s (medium, r = 1.7 at the profile's prefill);
# GLM 5.3 Flash 0.275 (slow, r = 0.46) and 0.55 GB/s (medium, r = 0.93), the measured 2026-10-08 disk speeds.
# Usage: sim_tuning.sh WORKLOAD_DIR KM.json OUT [SEEDS]
set -euo pipefail
WD=$1; KM=$2; OUT=$3; SEEDS=${4:-3}
PY=${PY:-python3}
RULES="wt cost_km cost_ewma_a0.3 cost_ewma_a0.5 cost_ewma_a0.8"
mkdir -p "$OUT"
run() {  # rows profile workload disks [--set ...]
  local rows=$1 prof=$2 w=$3 disks=$4; shift 4
  nice -n 19 "$PY" -m sim eval --profile "$prof" --cells "$WD/$w:0" --workloads 1 --seeds "$SEEDS" --disk $disks \
    --km "$KM" --rules $RULES --procs 6 --out "$OUT/$rows" "$@" > /dev/null
}
rm -f "$OUT/alpha_plan.jsonl" "$OUT/alpha_ref.jsonl"
run alpha_plan.jsonl qwen3_8b qwen3_8b.tuning.json "0.5 1.5" --set gpu_tokens=32768 cpu_tokens=79302
run alpha_plan.jsonl glm53_flash glm53_flash_heavy.tuning.json "0.275 0.55" --set gpu_tokens=423916 cpu_tokens=1130444
run alpha_plan.jsonl glm53_flash glm53_flash_half.tuning.json "0.275 0.55" --set gpu_tokens=262144 cpu_tokens=368556
run alpha_ref.jsonl qwen3_8b qwen3_8b.tuning.json "0.5 1.5"
run alpha_ref.jsonl glm53_flash glm53_flash_heavy.tuning.json "0.275 0.55"
run alpha_ref.jsonl glm53_flash glm53_flash_half.tuning.json "0.275 0.55"
