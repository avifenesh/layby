# Phase R CPU preparation (2026-10-11)

Everything here ran on CPU, before any GPU run. Plan: BENCH_PLAN draft 4.2, sections 3 to 8.

## K_km curves

`km_public_train.json`: per-kind Kaplan-Meier curves P(T > E2[j]) fitted with `python -m layby.curves fit-km` on the
public-source rows (SWE-chat, WildChat, Copilot agent traces) of the Layby-Dwell v6 training split
(`cls5/train.parquet`). One wait per row, uncapped. A wait that never returned is censored at its gap.
`km_train_waits.py` builds the fitting pool and checks it: 0 sessions and 0 row refs overlap with any Phase R workload.

| kind | waits | censored |
|---|---|---|
| tool | 538,869 | 0 |
| human | 389,403 | 158,128 |
| workflow | 32,520 | 0 |
| all (workloads with no kind: cc-traces) | 960,792 | 158,128 |

## K_ewma alpha

`sim_tuning.sh`: the simulator on the tuning workloads (run as built, every session at its start), C0 and the rule
with each curve, median over 3 seeds, then a 5-seed rerun. Capacities as the plan: GPU KV 0.15 P (floored at one
max-length request, which vLLM needs to start), CPU tier 0.4 P, P from the model's measured workload. Slow and medium
disks: Qwen3-8B 0.5 and 1.5 GB/s (r = 0.57 and 1.7 at the reference engine's prefill), GLM 5.3 Flash 0.275 and
0.55 GB/s (r = 0.46 and 0.93; the speeds measured on 2026-10-08), GLM with the constants of the pre-registered GLM
prediction (`sim.eval_pop.PROFILES`).

p95 TTFT ratio to C0 (3 seeds; pooled = geometric mean over the 6 cells):

| workload | disk GB/s | a 0.3 | a 0.5 | a 0.8 | K_km | wt |
|---|---|---|---|---|---|---|
| qwen3_8b.tuning | 0.5 | 0.991 | 0.972 | 0.998 | 1.051 | 1.282 |
| qwen3_8b.tuning | 1.5 | 0.607 | 0.614 | 0.625 | 0.637 | 0.613 |
| glm53_flash_heavy.tuning | 0.275 | 0.817 | 0.802 | 0.751 | 0.832 | 0.907 |
| glm53_flash_heavy.tuning | 0.55 | 0.618 | 0.601 | 0.589 | 0.618 | 0.650 |
| glm53_flash_half.tuning | 0.275 | 0.874 | 0.879 | 0.865 | 0.899 | 0.912 |
| glm53_flash_half.tuning | 0.55 | 0.752 | 0.755 | 0.745 | 0.762 | 0.751 |
| pooled, 3 seeds | | 0.764 | 0.759 | **0.750** | 0.786 | 0.826 |
| pooled, 5 seeds | | 0.762 | 0.760 | **0.754** | 0.772 | 0.829 |

Choice: alpha 0.8 (best pooled in both runs). The three alphas lie within about 1.5%, inside seed noise; Qwen
alone would pick 0.5. All Phase R workloads carry `surv_ewma` at alpha 0.8 (`layby.curves annotate --alpha 0.8`).
At the reference engine's larger capacities the tuning workloads barely queue (C0 p50 0.1 s on Qwen) and the alphas
tie.

## Capacities

`capacities.json` next to the workloads holds every number. P is the workload builder's peak live KV estimate, not a
simulator C0 replay. Disk tier = 1.2 x unique KV (new plus output tokens, a keep cut counted as new).

| model (measured workload) | P tokens | GPU KV tokens (0.15 P) | CPU tier (0.4 P) |
|---|---|---|---|
| Qwen3-8B | 198,256 | 29,738, floored to 32,768 (4.50 GiB) | 79,302 (10.9 GiB) |
| GLM heavy | 2,826,109 | 423,916 (3.24 GiB per rank) | 1,130,444 (31.9 GiB, 4 ranks) |
| GLM half | 921,391 | 138,209, floored to 262,144 (2.0 GiB per rank) | 368,556 (10.4 GiB) |

## cc-traces (no model)

`qwen3_8b.heldout_cc`, p95 ratio to C0, 3 seeds, at the plan capacities / at the reference engine:

| disk GB/s | wt | wt2 | K_km (pooled curve) |
|---|---|---|---|
| 0.5 | 1.09 / 1.02 | 1.08 / 1.02 | 1.01 / 1.01 |
| 1.5 | 1.00 / 0.95 | 1.00 / 0.97 | 0.99 / 1.01 |
| 3 | 0.96 / 0.93 | 0.97 / 0.92 | 0.97 / 1.01 |

The workload is prefill-bound in the simulator (19.9M unique tokens, mostly compaction restarts; C0 p50 167 s and
69 s), so no placement moves it by more than 9%.

## Workloads v2: Qwen3.8-27B and Gemma 4 31B (2026-10-11)

The owner replaced Qwen3-8B and GLM 5.3 Flash with Qwen3.8-27B and Gemma 4 31B (FP8 weights, one RTX PRO 6000 per
engine, Nebius). The v1 outputs are in `workloads/archive_v1/`. Same partitions, seen set (`used.parquet`, no input
changed since it was built) and checks; context cap 131,072 (default until the compatibility check reports
max-model-len), the same restart rule.

Steps: `service_model.py` (constants, `models_phase_r.json`), `build_workloads.py --models`, `side_files.py`,
`finalize_workloads.py` (surv_km, surv_ewma at alpha 0.8, CT_ETA, `capacities.json`), `km_train_waits.py` (re-check:
0 sessions and 0 refs of the new workloads in the K_km fitting pool).

KV per model (BF16 KV, from config.json; Qwen/Qwen3.8-27B-FP8@017b9c7, RedHatAI/gemma-4-31B-it-FP8-block@d724254):

| model | growing KV per token | fixed per session |
|---|---|---|
| Qwen3.8-27B | 64 KiB (16 full-attention layers x 4 KV heads x 256) | 153,944,064 B recurrent state (48 Gated DeltaNet layers: fp32 SSM 48 x 128 x 128 + conv) |
| Gemma 4 31B | 80 KiB (10 global layers x 4 KV heads x 512, K = V cached in both slots) | min(c, 1,024) x 800 KiB (50 sliding layers x 16 KV heads x 256), 800 MiB at most |

Service time: ESTIMATES. The Qwen3-8B fit (C0 0.649, TPOT 0.0190, B 0.0266 per 1k prompt) scaled by prefill FLOPs per
token (C0, B) and decode bytes per sequence step (TPOT) at the fit's mean prompt (23,015 tokens), no FP8 compute
credit: Qwen3.8-27B x2.13 / x1.49, Gemma 4 31B x2.74 / x1.80. Heavy load = mean requests in flight 1.43, that of the
v1 Qwen3-8B measured workload under its fit. Replace with the pilot's microbenchmarks.

Disk tier (1.2 x unique KV) is sized at an upper bound until the pilot measures the bytes the engines write: Gemma's
sliding layers counted for every token, and for Qwen3.8 one recurrent state per 784-token block (vLLM 0.31.0 sets the
attention block to 784 tokens so its page equals the 3.21 MB per-layer mamba page; the compat boot log confirms the
state size). `disk_bytes_turn_end` (one state and the window's new tokens per turn end) is the lower figure.
GPU KV is floored at 1.1 x one 131,072-token request (Qwen 9.62 GB, Gemma 12.73 GB), above 0.15 P for both models.

Re-sized 2026-10-11 (coordinator's decision): heavy P = the GPU KV floor / 0.15 (Qwen3.8 64.1 GB, Gemma 84.9 GB
targets; `build_workloads.py --size-by-kv 7.3333`), CPU tier 0.4 P, light 0.5 P, sweeps 1.5, 2 and 3 P; tuning, pilot
and measured run to completion (estimated makespan in each manifest), the 20- and 40-minute stages keep their
lengths at the heavy P. Disk tier 1.2x unique KV at turn ends (`disk_bytes`; `disk_bytes_upper` for the pilot check).
Qwen3.8 prefill is measured (compat `prefill.json`, b1 9,009 tok/s at 23K, vs 16,983 for Qwen3-8B from its C0 runs:
C0 and B x1.885); its TPOT and all of Gemma's service time are still the FLOP and bytes estimates.
`pool_all.{json,refs.parquet,side.parquet}`: every turn of every candidate session of the three partitions (12,418
sessions, 520,605 turns), so a later re-sizing takes v6 curves from one scored file.

Bounded sessions (2026-10-11, coordinator): starts uniform in the first 5 minutes (`--duration 300`), each session cut
at its last turn that the service model ends within 55 minutes of its start (`--session-cap 3300`), sessions added
until the heavy P target holds again (light and sweeps the same way), so a run to completion ends near 60 minutes.
The GPU KV floor is the KV vLLM 0.31.0 needs to admit one 131,072-token request, from the compat boots (pool x
131,072 / pool tokens): Qwen3.8 8.94 GB, Gemma 4 25.0 GB (its sliding groups reserve the window plus the batch
budget during prefill, so the floor is 2.2x the session KV at the cap). Heavy P = 1.1 x that / 0.15: Qwen3.8
65.6 GB, Gemma 4 183.4 GB. Gemma's prefill is measured too (b1 4,447 tok/s at 23K, Triton attention: C0 and B x3.82);
both TPOTs remain estimates. CT_ETA is the measured memoryfulness of each workload; where it is 0.0 (floored), CT's
TTL uses the reload cost alone (T x eta + R = R).
