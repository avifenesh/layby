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

## surv_v6, GPU floors, cells (2026-10-11, after the compat check)

surv_v6: `attach_surv_v6.py` joins each workload's side rows to `pool_all.surv_v6.parquet` (Layby-Dwell v6 @101cc44,
sidecar scorer, bf16; receipt `surv_v6_receipt.json`, copied next to the workloads) by feat_ref, after checking the
receipt's sha256 of the pool side file and scores and that every joined row's server-view state and kind equal the
pool's. 33,796 turns with a wait got a curve; 0 public turns without one; the 2,247 cc-traces turns have no text and
stay null. `NAME.side.jsonl` (mode 600) is the live sidecar's input. Sanity (`surv_v6.json`): tool P(T > 60 s)
0.02 to 0.06 predicted vs 0.01 to 0.05 observed; human 0.71 to 0.81 vs 0.60 to 0.79; human P(T > 300 s) 0.40 to 0.47
vs 0.25 to 0.37: the workloads keep a session only up to its first wait that never returned, so long waits are
under-represented against the model's (non-returns included) prediction.

GPU KV floor with a connector: vLLM 0.31.0 keeps the hybrid KV manager on with OffloadingConnector (SupportsHMA), so a
sliding group holds window + in-flight tokens per request. One 131,072-token request needs 9,541 blocks x 2,621,440 B =
25.01 GB on Gemma 4 and 174 x 51,380,224 B = 8.94 GB on Qwen3.8, with or without a connector; the formula reproduces
every compat log figure (49,895 and 301,887 tokens, num_gpu_blocks 21,975, concurrency 2.303218 and 6.810345). vLLM's
"GPU KV cache size: N tokens" is concurrency x max_model_len, so pool / N (430 KB/token for Gemma) is not a
reservation. No re-size: Gemma GPU KV 27.81 GB (1.11x a 131K request, above 20 GiB) = 0.15 P, Qwen 10.05 GB (1.12x).
Recorded in models_phase_r.json and capacities.json (gpu_request_check).

Cells (`make_cells.py`, engine/phase_r/cells): qwen and gemma slow heavy, slow light, medium heavy; pilots slow (every
arm plus PT, for RSS), medium (C0 T K) and kdisk (`kdisk_workload.py`: the pilot workload cut at 20 minutes, CPU tier
1.5x the largest session, 0.21 (Qwen) and 0.12 (Gemma) of the peak live tier bytes; pass: `check_k_disk.py`, with
replay `--verify-disk` comparing disk-restored turns against a cold recompute). Medium carries a PLACEHOLDER io.max
read cap of 1 GB/s: at the NRD disk's nominal 2 GiB/s the compat inputs classify both models fast.

## Draft 6 re-size: measured service model, KV pressure, throttled slow disk (2026-10-12, after pilot 1)

Service model (`fit_service.py`, outputs in `~/.cache/layby_phase_r/fit/fit_MODEL.json`): the iteration-level
simulator (`sim/engine_sim.py`) set up as the pilot's C0 arm (chunked prefill 8,192, max_num_seqs 64, GPU KV in tokens
of growing KV with a per-request reserve: Qwen's recurrent state 2,349 tokens, Gemma's window 10,240 tokens; CPU tier
at the tier's measured bytes per token; CPU to GPU at the measured 56 GB/s). Prefill and decode terms come from the
pilot microbenchmark (Qwen F0 92.3 us/token, ATT 0.0137 per 1K tokens of position, T0 18.1 ms, BW 276 GB/s; Gemma
F0 118 us, ATT 0.0757); scale factors KP (prefill) and KD (decode) are fitted on the C0 runs by grid search on served
count, resumed-turn TTFT p50 and p95 and makespan. Queueing is not a term: it comes from the scheduler and the KV
capacity.

| model | KP, KD | run | served (sim / measured) | resumed p50 s | resumed p95 s | makespan s | per-turn TTFT median abs log error; within 2x |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 1.00, 0.75 | C0 r2 (uncapped) | 709 / 709 | 90.8 / 88.1 | 142.9 / 144.1 | 6,542 / 6,490 | 0.13; 81% |
| Qwen3.8-27B | | C0 r9 (cap 3,600 s) | 531 / 537 | 110.7 / 109.0 | 147.2 / 147.4 | - | 0.11; 97% |
| Gemma 4 31B | 1.00, 1.00 | C0 (cap 3,600 s) | 387 / 405 | 513.9 / 526.3 | 640.9 / 678.6 | - | 0.045; 96% |

The microbenchmark terms alone (KP = KD = 1) give Qwen p50 +16%, p95 +17%, makespan +10%. Every calibration point is
a saturated run: the fit at low load rests on the microbenchmark terms (KP = 1 on both models). The fit sizes
workloads only; H6's predictions stay microbenchmark-only.

Load scan (`scan_load.py`, measured partition, sessions in builder order, each cut where its projected C0 end passes
3,000 s): C0's tiers stop holding the working set between GPU KV / P of 0.22 and 0.19 on Qwen (recompute share 16% to
66%, resumed TTFT p50 5.8 s to 65 s, mean in flight 5 to 11) and between 0.45 and 0.32 on Gemma (p50 1.6 s to 68 s).
At GPU KV = 0.15 P (Qwen N 32: p50 106 s, 18 in flight; Gemma N 60: p50 423 s, 49 in flight) C0 recomputes 95% or
more of its prompt tokens and its TTFT is queueing: KV pressure 0.15 and a drained C0 queue cannot both hold on these
models (the GPU KV floor of 1.1 x one 131K request fixes P at 7.3 such requests). The coordinator's rule picks the KV
pressure; a C0 in-flight target of about 3 (Qwen 16 sessions, GPU/P 0.25, p50 4.7 s) would be the drained choice.

Builder (`build_workloads.py --service-fit`): sessions start uniform over 300 s, each is cut at its last turn whose
projected C0 end is within 3,000 s of the run start (tuning stages: 5/6 of the stage length), and sessions are added
in partition order until the projected peak live KV reaches 7.333 x the GPU request bytes (the smallest such count;
where none reaches it, the count with the largest P, flagged `target_reached false`). `kdisk_workload.py
--service-fit` cuts the pilot workload at 20 minutes of the projection. Side files, surv_v6 (pool scores, 0 public
turns without a curve), surv_km, surv_ewma, CT_ETA and capacities as before; disjointness checks 0 overlap
(partitions, seen set, users, K_km pool). The v2 workloads moved to `workloads/archive_v2/`.

Slow disk: network_ssd with io.max 210 MiB/s read and write (r 0.29 Qwen, 0.26 Gemma at the cell prefill rate; 0.34
and 0.44 at the 32K rate). Cells and the pilot-2 blocks: `make_cells.py` (its docstring).

### Drained load (coordinator and owner, 2026-10-12, replaces the KV-pressure sizing above)

`build_workloads.py --size-by-inflight 3`: the measured workload takes the session count whose projected C0 mean
requests in flight is closest to 3, each count projected at the capacities of its own P (a fixed point,
`sim_sizing.project_at_rule`); every other workload is sized to that P (extras to their multiple) at those capacities.
Capacities (`finalize_workloads.py`): GPU KV max(1.1x the GPU KV for one 131K request, 0.15 P), CPU tier 0.55 P minus GPU
KV (GPU + CPU = 0.55 P, the original 0.15 + 0.40), at least 0.05 P.

| model | sessions / turns | P GB | GPU share | CPU share | C0 in flight | C0 resumed p50 s | C0 recompute share |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 9 / 399 | 24.6 | 0.40 (floor) | 0.15 | 2.8 | 7.4 | 0.39 |
| Gemma 4 31B | 9 / 411 | 38.6 | 0.71 (floor) | 0.05 (floor; total 0.76) | 2.8 | 0.3 | 0.05 |

The next count up (10 sessions) gives 4.2 in flight on both (Qwen p50 27 s, recompute 0.76). Every target P was
reached. kdisk workloads come from the KV-pressure pilot workloads (`workloads/archive_v3`): the drained pilot
workload cannot put a 1.5x-largest-session CPU tier under pressure. The KV-pressure workloads are in
`workloads/archive_v3`, the v2 ones in `archive_v2`.

### 32K context cap (owner, 2026-10-12)

At 131,072 the GPU KV floor (1.1x one max-length request) took 0.40 P (Qwen) and 0.71 P (Gemma) at the drained load,
and the CPU tier was smaller than one max-length session on both models. The owner set `--max-model-len 32768`;
sessions past the cap restart by the same compaction rule (the newest 0.16 x CAP tokens of history plus the turn's new
tokens). One request at 32,768 needs 48 blocks x 51,380,224 B = 2.47 GB on Qwen3.8 and 6,469 blocks x 2,621,440 B =
16.96 GB on Gemma 4 (the formula that reproduces the compat logs; Gemma's sliding groups still reserve window + 2 x 8,192
batch tokens per request, so its floor barely moves with the cap). vLLM 0.31.0's align-mode Mamba manager holds one
state block per group for a running request (two while it checkpoints a prefill), so the simulator's 2,349-token
recurrent reserve stands.

CPU tier minimum: a disk restore is staged through the CPU tier, so the tier holds at least 1.5 max-length contexts at
the tier's measured bytes per token (pilot 1: Qwen 72,472 B, Gemma 131,714 B), whole tier chunks
(`sim_sizing.cpu_min_bytes`, the kdisk rule): Qwen 3.60 GB, Gemma 6.47 GB. The drained-load fixed point projects at
this rule.

| model | sessions / turns | P GB | GPU KV | CPU tier | C0 in flight | C0 resumed p50 s | C0 recompute share |
|---|---|---|---|---|---|---|---|
| Qwen3.8-27B | 20 / 826 | 19.9 | 2.99 GB, 0.15 P | 7.98 GB, 0.40 P | 2.7 | 2.5 | 0.28 |
| Gemma 4 31B | 20 / 801 | 38.8 | 18.65 GB, 0.48 P (floor) | 6.47 GB, 0.17 P (minimum; total 0.65 P) | 2.8 | 0.1 | 0.12 |

Light: Qwen 9 sessions / 309 turns, Gemma 11 / 486. kdisk (from the new pilot workloads, 20 minutes): Qwen 17 / 272
(CPU tier 3.60 GB, 0.19 of the peak tier bytes), Gemma 19 / 309 (6.47 GB, 0.16). Cells carry MAXLEN from
capacities.json `max_model_len`. Checks: 0 overlap (partitions, seen set, users, K_km pool), 0 public turns without a
v6 curve, 31 of 31 cells and blocks pass the dry run. The 131K drained workloads are in `workloads/archive_v4`.
