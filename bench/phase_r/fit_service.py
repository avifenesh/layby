#!/usr/bin/env python3
"""C0 service model per serving model, fitted on the pilot's C0 runs (BENCH_PLAN draft 6: load and session cut from
measured service times). Replaces the scaled service-time estimates of service_model.py for workload sizing.

The model is the iteration-level engine simulator (sim/engine_sim.py) set up as the pilot's C0 arm: vLLM 0.31.0,
chunked prefill budget 8,192 tokens, max_num_seqs 64, FCFS, GPU prefix cache with LRU eviction, write-through CPU
tier (LRU, offload_prompt_only false), no disk. Queueing is not a fitted term: it comes out of the scheduler, the GPU
KV capacity (the running set holds every running request's context plus a per-request reserve) and the CPU tier's
hit pattern. The terms per model:

  prefill  step time += sum over chunks of c x F0 x (1 + ATT x (chunk start + c / 2) / 1000): F0 (s per token at
           position 0) and ATT (attention growth per 1,000 tokens of position) from the pilot's batch-1 prefill
           microbenchmark (2K, 8K, 32K tokens, least squares), each step also pays T0
  decode   step time += T0 + decode KV bytes / BW: T0 and BW from the pilot's TPOT microbenchmark (batch 1, 8, 16,
           32 at about 8.3K tokens of context; least squares on TPOT = T0 + batch x context x KV bytes per token / BW)
  loads    CPU -> GPU at the pilot's measured offload rate (vllm:kv_offload_total_bytes / time, CPU_to_GPU)
  capacity GPU KV bytes / growing KV bytes per token (tokens), and per running request the recurrent state (Qwen)
           or the sliding window (Gemma) in tokens of growing KV; CPU tier bytes / the tier's measured bytes per token
Then two scale factors, KP on the prefill terms and KD on the decode terms, are fitted on the C0 runs: a grid search
that minimises the summed squared log error of, per run, the served count by the run's cap, the resumed-turn TTFT
p50 and p95 among served turns, and (uncapped runs) the makespan. The fit's error is reported per run and per turn
(matched session and turn: median absolute log ratio of TTFT and of end-to-end time).

This fit sizes workloads only. It is not the H6 simulator calibration (BENCH_PLAN section 1: microbenchmarks only):
the H6 predictions use KP = KD = 1.

Usage: fit_service.py OUT.json --model NAME --workload pilot.json --micro micro.json --gpu-bytes B --cpu-bytes B
       --grow-bpt B --tier-bpt B --req-bytes B --cpu-gbps G --run LABEL=ROWS.jsonl:CAP_S [--run ...]
       [--kp 0.8,0.9,...] [--kd 1,1.5,...] [--procs N]
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from sim.engine_sim import Params, simulate  # noqa: E402

BUDGET = 8192
MAX_SEQS = 64
TPOT_CTX = 8320          # the TPOT microbenchmark's context: an 8,192-token prompt plus half its output


def micro_terms(micro, grow_bpt):
    """F0, ATT, T0, BW from the pilot microbenchmark (micro_MODEL.json)."""
    tp = sorted((int(b), v["tpot_ms_median"] / 1000.0) for b, v in micro["tpot"].items())
    X = np.array([[1.0, b * TPOT_CTX * grow_bpt] for b, _ in tp])
    y = np.array([t for _, t in tp])
    (t0, inv_bw), *_ = np.linalg.lstsq(X, y, rcond=None)
    pf = sorted((int(n), float(np.median(v["b1_ttft_s"]))) for n, v in micro["prefill"].items())
    X = np.array([[n, n * n / 2000.0] for n, _ in pf])
    y = np.array([t - math.ceil(n / BUDGET) * t0 for n, t in pf])
    (f0, f0att), *_ = np.linalg.lstsq(X, y, rcond=None)
    return dict(t0=float(t0), bw=float(1.0 / inv_bw), f0=float(f0), att=float(f0att / f0),
                tpot_points=tp, prefill_points=pf, tpot_ctx=TPOT_CTX)


def params(base, kp=1.0, kd=1.0, until=None):
    p = Params(gpu_tokens=base["gpu_tokens"], cpu_tokens=base["cpu_tokens"], budget=BUDGET, max_seqs=MAX_SEQS,
               t0=base["t0"] * kd, bw=base["bw"] / kd, f0=base["f0"] * kp, att=base["att"],
               cpu_gbps=base["cpu_gbps"], kib=base["tier_bpt"] / 1024.0, hbm_kib=base["grow_bpt"] / 1024.0,
               req_tokens=base["req_tokens"], req_tokens_prefill=base.get("req_tokens_prefill"),
               **{k: base[k] for k in ("win_groups", "win_lead", "win_block", "win_page_tokens") if k in base},
               until=until)
    return p


def sim_c0(W, base, kp=1.0, kd=1.0, until=None):
    rows, info = simulate(W, None, params(base, kp, kd, until))
    return rows, info


def inflight(intervals, t_end):
    """Mean number of requests in flight over [0, t_end] from (start, end) intervals."""
    if t_end <= 0:
        return 0.0
    return float(sum(max(0.0, min(e, t_end) - max(s, 0.0)) for s, e in intervals) / t_end)


def measured(path, cap):
    R = [json.loads(x) for x in open(path)]
    R = [r for r in R if "session" in r and "turn" in r]
    t0 = 0.0                                  # "sent" is seconds since the replay start, as the simulator's clock
    served = [r for r in R if not r.get("censored") and not r.get("failed") and r.get("ttft") is not None
              and r["sent"] - t0 + r["e2e"] <= cap + 1e-6]
    res = [r["ttft"] for r in served if r["turn"] > 0]
    end = max(r["sent"] - t0 + r["e2e"] for r in served)
    hit = sum((r.get("usage") or {}).get("prompt_tokens_details", {}).get("cached_tokens") or 0 for r in served)
    return dict(turns=len(R), served=len(served), resumed_p50=float(np.median(res)), resumed_p95=float(np.quantile(res, .95)),
                makespan=float(end), inflight=inflight([(r["sent"] - t0, r["sent"] - t0 + r["e2e"]) for r in served], min(end, cap)),
                prompt_tokens_M=sum(r["prompt_tokens"] for r in served) / 1e6, cached_tokens_M=hit / 1e6,
                per_turn={(r["session"], r["turn"]): (r["ttft"], r["e2e"]) for r in served})


def simulated(rows, cap):
    served = [r for r in rows if r["t_done"] <= cap + 1e-6]
    res = [r["ttft"] for r in served if r["turn"] > 0]
    end = max(r["t_done"] for r in served)
    return dict(served=len(served), resumed_p50=float(np.median(res)), resumed_p95=float(np.quantile(res, .95)),
                makespan=float(end), inflight=inflight([(r["t_send"], r["t_done"]) for r in served], min(end, cap)),
                prompt_tokens_M=sum(r["P"] for r in served) / 1e6, cached_tokens_M=sum(r["hit"] + r["ext"] for r in served) / 1e6,
                per_turn={(r["session"], r["turn"]): (r["ttft"], r["t_done"] - r["t_send"]) for r in served})


def errors(m, s, capped):
    e = dict(served=math.log(s["served"] / m["served"]), p50=math.log(s["resumed_p50"] / m["resumed_p50"]),
             p95=math.log(s["resumed_p95"] / m["resumed_p95"]))
    if not capped:
        e["makespan"] = math.log(s["makespan"] / m["makespan"])
    keys = sorted(set(m["per_turn"]) & set(s["per_turn"]))
    lt = [abs(math.log(max(s["per_turn"][k][0], 1e-3) / max(m["per_turn"][k][0], 1e-3))) for k in keys]
    le = [abs(math.log(max(s["per_turn"][k][1], 1e-3) / max(m["per_turn"][k][1], 1e-3))) for k in keys]
    return e, dict(matched_turns=len(keys), ttft_median_abs_log=float(np.median(lt)), e2e_median_abs_log=float(np.median(le)),
                   ttft_within_x2=float(np.mean([x <= math.log(2) for x in lt])))


def _job(args):
    W, base, kp, kd, caps = args
    rows, info = sim_c0(W, base, kp, kd, until=max(caps))
    return kp, kd, rows, info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workload", required=True)
    ap.add_argument("--micro", required=True)
    ap.add_argument("--gpu-bytes", type=float, required=True)
    ap.add_argument("--cpu-bytes", type=float, required=True)
    ap.add_argument("--grow-bpt", type=float, required=True)
    ap.add_argument("--tier-bpt", type=float, required=True)
    ap.add_argument("--req-bytes", type=float, required=True, help="GPU KV a running request holds beyond its "
                    "growing KV: the recurrent state (Qwen) or the sliding window at its maximum (Gemma)")
    ap.add_argument("--req-prefill-bytes", type=float, help="GPU KV a request holds beyond its growing KV while "
                    "its prefill runs (sim_sizing.req_reserve_tokens; default: --req-bytes)")
    ap.add_argument("--models", help="models_phase_r.json: the GPU reserve from the model's kv spec "
                    "(sim_sizing.sim_reserve_params; replaces --req-bytes and --req-prefill-bytes)")
    ap.add_argument("--budget", type=int, default=BUDGET, help="--max-num-batched-tokens of the runs")
    ap.add_argument("--max-seqs", type=int, default=MAX_SEQS, help="--max-num-seqs of the runs")
    ap.add_argument("--cpu-gbps", type=float, required=True)
    ap.add_argument("--run", action="append", required=True, help="LABEL=ROWS.jsonl:CAP_S (CAP_S 0: uncapped)")
    ap.add_argument("--kp", default="0.8,0.9,1.0,1.1,1.2,1.3,1.45,1.6")
    ap.add_argument("--kd", default="0.75,1.0,1.5,2.0")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    globals().update(BUDGET=a.budget, MAX_SEQS=a.max_seqs)   # the runs' engine settings (forked workers inherit them)

    W = json.load(open(a.workload))
    mt = micro_terms(json.load(open(a.micro)), a.grow_bpt)
    base = dict(mt, gpu_tokens=int(a.gpu_bytes // a.grow_bpt), cpu_tokens=int(a.cpu_bytes // a.tier_bpt),
                req_tokens=int(round(a.req_bytes / a.grow_bpt)),
                req_tokens_prefill=None if a.req_prefill_bytes is None else int(round(a.req_prefill_bytes / a.grow_bpt)),
                cpu_gbps=a.cpu_gbps, tier_bpt=a.tier_bpt,
                grow_bpt=a.grow_bpt)
    if a.models:
        from sim_sizing import sim_reserve_params
        kv = next(m for m in json.load(open(a.models)) if m["name"] == a.model)["kv"]
        base.update(sim_reserve_params(kv))
    runs = []
    for r in a.run:
        lab, rest = r.split("=", 1)
        path, cap = rest.rsplit(":", 1)
        cap = float(cap)
        runs.append(dict(label=lab, path=path, cap=cap if cap > 0 else math.inf, capped=cap > 0))
    for r in runs:
        r["m"] = measured(r["path"], r["cap"])
    caps = [r["cap"] for r in runs]
    grid = [(float(kp), float(kd)) for kp in a.kp.split(",") for kd in a.kd.split(",")]
    jobs = [(W, base, kp, kd, [c if c < math.inf else 1e9 for c in caps]) for kp, kd in grid]
    results = []
    with ProcessPoolExecutor(a.procs) as ex:
        for kp, kd, rows, info in ex.map(_job, jobs):
            per, loss = [], 0.0
            for r in runs:
                s = simulated(rows, r["cap"])
                e, t = errors(r["m"], s, r["capped"])
                loss += sum(x * x for x in e.values())
                per.append(dict(run=r["label"], log_err=e, turn_err=t,
                                sim={k: v for k, v in s.items() if k != "per_turn"}))
            results.append(dict(kp=kp, kd=kd, loss=loss, runs=per))
            print(f"kp {kp:.2f} kd {kd:.2f} loss {loss:.4f} " + " | ".join(
                f"{x['run']}: " + " ".join(f"{k} {v:+.3f}" for k, v in x["log_err"].items()) for x in per), flush=True)
    best = min(results, key=lambda x: x["loss"])
    unit = next(x for x in results if x["kp"] == 1.0 and x["kd"] == 1.0) if any(
        x["kp"] == 1.0 and x["kd"] == 1.0 for x in results) else None
    out = dict(
        model=a.model, what=__doc__.split("\n\n")[0], use="workload sizing only; not the H6 calibration",
        workload=os.path.basename(a.workload), micro=os.path.basename(a.micro),
        base=base, fit=dict(kp=best["kp"], kd=best["kd"], loss=best["loss"]),
        sim_params=dict(vars(params(base, best["kp"], best["kd"]))),
        measured=[dict(run=r["label"], cap_s=None if not r["capped"] else r["cap"],
                       **{k: v for k, v in r["m"].items() if k != "per_turn"}) for r in runs],
        best=best, micro_only=unit, grid=[dict(kp=x["kp"], kd=x["kd"], loss=x["loss"]) for x in results])
    with open(a.out + ".tmp", "w") as f:
        json.dump(out, f, indent=1)
    os.replace(a.out + ".tmp", a.out)
    print("best", json.dumps(dict(kp=best["kp"], kd=best["kd"], loss=round(best["loss"], 4))), flush=True)


if __name__ == "__main__":
    main()
