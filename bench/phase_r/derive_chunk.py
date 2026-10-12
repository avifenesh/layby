#!/usr/bin/env python3
"""Gemma's chunked-prefill size from measurements (owner, 2026-10-12: no guessed constants).

vLLM 0.31.0 gives a prefilling Gemma 4 request the sliding window plus two in-flight chunks in each of its 5 sliding
groups, so the GPU KV the engine needs for one max-length request (its start check, and the floor of the capacity
rule) grows with --max-num-batched-tokens C. For each measured C this script:
  1. floor(C) = 1.1 x (one max_model_len request's global KV + the prefill reserve at C), sim_sizing.req_reserve_bytes;
     printed beside vLLM's own figure (pool / "Maximum concurrency" at the cells' 18.65 GB pool, --vllm-concurrency)
  2. the C0 service model at C: the microbenchmark at C (micro_cC.json: prefill and TPOT on the pilot-2 VM),
     scaled by the KP and KD refitted on the measured Gemma C0 runs with the corrected reserve (--fit)
  3. the drained load at C: the measured partition's candidates (build_workloads.py --dump-candidates), the largest
     session count whose projected C0 resumed turns are admitted on arrival at the median, each count projected at the capacities
     of its own P (BENCH_PLAN draft 6.1 rule: GPU max(0.15 P, floor), CPU 0.55 P - GPU, at least 0.05 P and 1.5
     max-length contexts), the builder's search (build_workloads.assemble_inflight)
Rule (the plan's pre-registered GPU share): the largest C whose floor(C) <= 0.15 P(C). Writes --out JSON.

Usage: derive_chunk.py CANDIDATES.json --fit FIT.json --models models_phase_r.json --micro C=micro_cC.json ...
       [--vllm-concurrency C=X ...] [--target 3] [--end-cap 3000] --out OUT.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_service  # noqa: E402
from sim_sizing import cpu_min_bytes, load_fit, project_at_rule, req_reserve_bytes, sim_reserve_params  # noqa: E402


def drained(C, fit, kv, floor, rule, target, end_cap):
    """The builder's drained-load search (build_workloads.assemble_drained): the largest count whose projected C0
    resumed turns are admitted on arrival at the median."""
    memo = {}

    def at(n):
        if n not in memo:
            memo[n] = project_at_rule([dict(x) for x in C[:n]], fit, end_cap, kv, floor, rule)
            st = memo[n][3]
            print(f"    n {n}: queue p50 {st['resumed_queue_p50_s']} s, in flight {st['mean_inflight']}, "
                  f"P {st['peak_live_kv_bytes'] / 1e9:.1f} GB, C0 p50 {st['resumed_ttft_p50_s']} s, "
                  f"recompute {st['recompute_share']}", flush=True)
        return memo[n]

    ok = lambda n: (at(n)[3]["resumed_queue_p50_s"] or 0.0) <= 0.0   # noqa: E731
    lo, hi = 1, 2
    while hi < len(C) and ok(hi):
        lo, hi = hi, min(len(C), hi * 2)
    if ok(hi):
        lo = hi
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    _, _, _, st, caps = at(lo)
    return lo, st, caps


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("candidates")
    ap.add_argument("--fit", required=True, help="fit_service.py output refitted with the prefill reserve (KP, KD)")
    ap.add_argument("--models", required=True)
    ap.add_argument("--model", default="gemma4_31b")
    ap.add_argument("--micro", action="append", required=True, help="C=micro_cC.json")
    ap.add_argument("--vllm-concurrency", action="append", default=[], help="C=X: vLLM's 'Maximum concurrency' at "
                    "--pool-bytes and max_model_len (the check of the floor formula)")
    ap.add_argument("--pool-bytes", type=float, default=18653904896)
    ap.add_argument("--target", type=float, default=3.0)
    ap.add_argument("--end-cap", type=float, default=3000.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    M = next(m for m in json.load(open(a.models)) if m["name"] == a.model)
    kv, cap = M["kv"], M["cap"]
    g = kv["grow_bytes_per_token"]
    glob = -(-cap // 32) * kv["gpu_request_check"]["block_bytes"]        # the global group: 32-token pages
    base = load_fit(a.fit)
    kp, kd = base["kp"], base["kd"]
    cand = json.load(open(a.candidates))["sessions"]
    conc = {int(k): float(v) for k, v in (x.split("=") for x in a.vllm_concurrency)}
    rule0 = dict(gpu_share=0.15, fast_total=0.55, cpu_floor=0.05, cpu_min=cpu_min_bytes(kv, cap))
    rows = []
    for spec in sorted(a.micro, key=lambda x: int(x.split("=")[0])):
        c, path = spec.split("=", 1)
        c = int(c)
        fit_service.BUDGET = c
        mt = fit_service.micro_terms(json.load(open(path)), g)
        pf, dc = req_reserve_bytes(kv, c, cap)
        req = glob + pf
        floor = 1.1 * req
        sp = dict(base["params"], budget=c, t0=mt["t0"] * kd, bw=mt["bw"] / kd, f0=mt["f0"] * kp, att=mt["att"],
                  **sim_reserve_params(kv))
        fit = dict(base, params=sp)
        print(f"C {c}: request {req / 1e9:.2f} GB, floor {floor / 1e9:.2f} GB, prefill b1 at 32K "
              f"{json.load(open(path))['prefill'].get('32000', {}).get('b1_tok_s')}", flush=True)
        n, st, caps = drained(cand, fit, kv, floor, rule0, a.target, a.end_cap)
        P = st["peak_live_kv_bytes"]
        row = dict(chunk=c, request_bytes=req, request_bytes_vllm=(a.pool_bytes / conc[c]) if c in conc else None,
                   floor_bytes=floor, sessions=n, P_bytes=P, gpu_plan_bytes=0.15 * P, floor_fits=floor <= 0.15 * P,
                   gpu_share=caps["gpu_share"], cpu_share=caps["cpu_share"], total_share=caps["total_share"],
                   mean_inflight=st["mean_inflight"], resumed_ttft_p50_s=st["resumed_ttft_p50_s"],
                   resumed_ttft_p95_s=st.get("resumed_ttft_p95_s"), recompute_share=st["recompute_share"],
                   makespan_s=st.get("makespan_s"), micro_terms={k: mt[k] for k in ("t0", "bw", "f0", "att")})
        rows.append(row)
        print("  " + json.dumps({k: row[k] for k in ("chunk", "sessions", "floor_fits", "gpu_share", "mean_inflight",
                                                     "resumed_ttft_p50_s", "recompute_share")}), flush=True)
    fits = [r["chunk"] for r in rows if r["floor_fits"]]
    out = dict(rule="the largest C whose floor (1.1 x one max-length request at C) <= 0.15 P(C) at the drained load",
               chosen=max(fits) if fits else None, kp=kp, kd=kd, fit=os.path.basename(a.fit),
               candidates=os.path.basename(a.candidates), target_inflight=a.target, rows=rows)
    json.dump(out, open(a.out, "w"), indent=1)
    print("chosen", out["chosen"], flush=True)


if __name__ == "__main__":
    main()
