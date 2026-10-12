#!/usr/bin/env python3
"""After build_workloads.py: K_km and K_ewma curves, CT_ETA, KV capacities, per model of MODELS.json.

For every workload NAME.json in DIR whose name starts with a model's name:
  curves   layby.curves annotate (surv_km from --km, surv_ewma at --alpha); surv_v6 stays null (score_side_files.py
           fills it on a GPU)
  ct_eta   layby.continuum.memoryfulness of the workload's sessions (Continuum's eta; CT needs it per workload)
  kv       the builder's accounting (manifest load): P = peak live KV bytes, unique KV bytes, the disk tier
Capacities per model, from its measured workload's P (BENCH_PLAN section 5, draft 6): GPU KV max(0.15 P, 1.1 x the GPU KV
the engine needs for one request at the context cap, kv.gpu_request_bytes, so the engine can start), CPU tier 0.55 P
minus the GPU KV (GPU + CPU = 0.55 P, the original 0.15 + 0.40), at least 0.05 P and at least 1.5 max-length contexts
at the tier's measured bytes per token (sim_sizing.cpu_min_bytes; a disk restore is staged through the CPU tier). Per workload: disk tier
1.2 x max(1, the model's measured disk_write_factor: bytes the tier wrote per unique KV byte in pilot 2's T arms) x
unique KV at turn ends (one recurrent state and the window's new tokens per turn end; the coordinator's
2026-10-11 decision); disk_bytes_upper is 1.2 x the builder's upper bound (the sliding layers' KV of every token, a
recurrent state at every KV block boundary), for a check against the bytes the engines write in the pilot.

Writes NAME.json (annotated), NAME.manifest.json (curves, ct_eta, capacities, new bytes and sha256) and
DIR/capacities.json, each through a hidden temporary name and a rename.

Usage: finalize_workloads.py DIR --models models.json --km km_public_train.json [--alpha 0.8]
"""
import argparse
import glob
import hashlib
import json
import os
import sys

from layby.continuum import memoryfulness
from layby.curves import annotate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_sizing import cpu_min_bytes  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--models", required=True)
ap.add_argument("--km", required=True)
ap.add_argument("--alpha", type=float, default=0.8)
ap.add_argument("--gpu-share", type=float, default=0.15)
ap.add_argument("--fast-total", type=float, default=0.55, help="GPU KV + CPU tier as a share of P (draft 6)")
ap.add_argument("--cpu-floor", type=float, default=0.05)
a = ap.parse_args()


def put_json(path, obj, **kw):
    d, b = os.path.split(path)
    tmp = os.path.join(d, f".{b}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, **kw)
    os.replace(tmp, path)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


KM = json.load(open(a.km))
km_sha = sha(a.km)
models = json.load(open(a.models))
caps = dict(_note=(
    "Phase R capacities (BENCH_PLAN section 5), bytes. P: the workload builder's peak live KV (largest summed KV "
    "of started, unfinished sessions on the builder's service-time model: grow x context + recurrent state + "
    "min(context, window) x window bytes), not a C0 replay. Per model, from its measured workload: GPU KV "
    "max(0.15 P, 1.1 x gpu_request_bytes: the GPU KV the engine needs to admit one max_model_len request), CPU tier 0.55 P minus "
    "the GPU KV, at least 0.05 P and at least 1.5 max_model_len contexts at tier_bytes_per_token (draft 6). Per workload: disk_bytes = 1.2 x "
    "unique KV at turn ends (one recurrent state and the window's new tokens per turn end); disk_bytes_upper = 1.2 x "
    "the upper bound (sliding layers' KV for every token, a recurrent state per KV block, vLLM's mamba align mode), "
    "ct_eta = layby.continuum.memoryfulness. KV constants from the Hugging Face config.json "
    "(bench/phase_r/service_model.py); service times: each model's service_model.status says which parts are measured "
    "(compat prefill microbenchmark) and which are estimates scaled from the Qwen3-8B fit. Sizing: the heavy P is the "
    "GPU KV floor / 0.15 (build_workloads.py --size-by-kv), sessions run to completion (est_makespan_s)."), models={}, workloads={})
for M in models:
    k = M["kv"]
    names = sorted(os.path.basename(p)[: -len(".json")] for p in glob.glob(os.path.join(a.dir, f"{M['name']}.*.json"))
                   if not p.endswith((".manifest.json",)))
    mans = {n: json.load(open(os.path.join(a.dir, f"{n}.manifest.json"))) for n in names}
    meas = mans[f"{M['name']}.measured"]["load"]
    P = meas["peak_live_kv_bytes"]
    floor = int(1.1 * k.get("gpu_request_bytes", k["longest_request_bytes"]))
    gpu = max(int(a.gpu_share * P), floor)
    cmin = cpu_min_bytes(k, M["cap"])
    cpu = max(int(a.fast_total * P) - gpu, int(a.cpu_floor * P), cmin) // 4096 * 4096
    mc = dict(measured_workload=f"{M['name']}.measured", hf=M["hf"], max_model_len=M["cap"], ctx_cap_source=M["cap_source"],
              kv_dtype=k["dtype"], grow_bytes_per_token=k["grow_bytes_per_token"],
              state_bytes_per_session=k["state_bytes_per_session"], window_tokens=k["window_tokens"],
              window_bytes_per_token=k["window_bytes_per_token"], window_bytes_per_session_max=k["window_bytes_per_session_max"],
              state_block_tokens=k.get("state_block_tokens"), state_block_source=k.get("state_block_source"),
              longest_request_bytes=k["longest_request_bytes"], gpu_request_bytes=k.get("gpu_request_bytes"),
              gpu_request_source=k.get("gpu_request_source"), gpu_request_check=k.get("gpu_request_check"), P_bytes=P, P_grow_tokens=meas["peak_live_kv_tokens"],
              gpu_kv_bytes_plan=int(0.15 * P), gpu_kv_floor_bytes=floor, gpu_kv_bytes=gpu, gpu_kv_floored=gpu == floor,
              cpu_bytes=cpu, gpu_share=round(gpu / P, 4), cpu_share=round(cpu / P, 4),
              cpu_floored=cpu == int(a.cpu_floor * P) // 4096 * 4096, cpu_min_bytes=cmin, cpu_at_min=cpu == cmin // 4096 * 4096,
              tier_bytes_per_token=k["tier_bytes_per_token"], capacity_rule=dict(
                  gpu_share=a.gpu_share, fast_total=a.fast_total, cpu_floor=a.cpu_floor,
                  cpu_min="1.5 x max_model_len x tier_bytes_per_token, whole tier chunks (a disk restore is staged through the CPU tier)"), service_model=dict(c0=M["c0"], tpot=M["tpot"], b_per_1k_prompt=M["b"],
                                                          inflight=M["inflight"], status=M["service_assumption"]["status"]))
    caps["models"][M["name"]] = mc
    for n in names:
        p = os.path.join(a.dir, f"{n}.json")
        W = json.load(open(p))
        r = annotate(W, KM, a.alpha)
        eta = round(memoryfulness(W["sessions"]), 4)
        L = mans[n]["load"]
        ub = L["unique_kv_bytes"]
        wc = dict(model=M["name"], sessions=L["sessions"], turns=L["turns"], peak_live_kv_bytes=L["peak_live_kv_bytes"],
                  peak_live_kv_grow_tokens=L["peak_live_kv_tokens"], peak_over_model_P=round(L["peak_live_kv_bytes"] / P, 3),
                  unique_kv_grow_tokens=L["unique_kv_tokens"], unique_kv_bytes=ub, disk_bytes=int(1.2 * max(1.0, k.get("disk_write_factor", 1.0)) * ub),
                  unique_kv_bytes_upper=L["unique_kv_bytes_upper"], disk_bytes_upper=int(1.2 * L["unique_kv_bytes_upper"]),
                  est_makespan_s=L["est_makespan_s"], run_to_completion=L.get("run_to_completion", False),
                  ct_eta=eta, est_mean_inflight=L["est_mean_inflight"])
        if k["state_bytes_per_session"]:
            wc.update(state_snapshots=L["state_snapshots"], state_blocks=L["state_blocks"])
        caps["workloads"][n] = wc
        put_json(p, W)       # surv_v6 already attached (attach_surv_v6.py) stays in W and its manifest record is kept
        man = mans[n]
        man["bytes"], man["sha256"] = os.path.getsize(p), sha(p)
        v6 = (man.get("curves") or {}).get("surv_v6")
        man["curves"] = dict(surv_km=f"layby.curves annotate, bench/phase_r/km_public_train.json sha256 {km_sha}",
                             surv_ewma=f"alpha {a.alpha} (chosen in the simulator on the tuning workloads, 2026-10-10)",
                             turns_with_curves=r["turns"], ewma_from_own_waits=r["ewma_own"],
                             surv_v6=v6 if isinstance(v6, dict) else "null until scored from the side file (GPU)")
        man["ct_eta"] = eta
        man["capacities"] = dict(model=mc, workload=wc)
        put_json(os.path.join(a.dir, f"{n}.manifest.json"), man, indent=1)
        print(n, json.dumps(wc), flush=True)
    print(M["name"], json.dumps({x: mc[x] for x in ("P_bytes", "gpu_kv_bytes", "gpu_kv_floored", "cpu_bytes")}), flush=True)
put_json(os.path.join(a.dir, "capacities.json"), caps, indent=1)
