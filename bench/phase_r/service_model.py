#!/usr/bin/env python3
"""Per-model constants for the Phase R workloads: KV accounting from the Hugging Face config.json, and a service-time
model for sizing (build_workloads.py --models). Writes a JSON list of model specs.

KV accounting (bytes, BF16 KV cache: the FP8 checkpoints carry no kv_cache_scheme, so the engines' kv-cache-dtype
"auto" keeps the model dtype):
  grow   bytes per token of the part that grows with the context: the full-attention layers' K and V
         (layers x KV heads x head dim x 2 x 2 bytes). Gemma 4's global layers have K = V in the checkpoint, but the
         engines load the K weights into both the K and the V slots (vLLM gemma4.py), so both are cached.
  state  bytes per session that do not grow: Qwen3.8's Gated DeltaNet layers hold a recurrent state (SSM state
         V heads x key dim x value dim in mamba_ssm_dtype, float32, plus the causal-conv state (kernel - 1) x
         conv channels in the model dtype) per layer, fixed whatever the context.
  window, window_bytes
         Gemma 4's sliding-window layers: K and V of the newest `window` tokens (layers x KV heads x head dim x 2
         x 2 bytes per token), so a session holds min(context, window) x window_bytes.
A session at context c holds grow x c + state + min(c, window) x window_bytes.
  state_block_tokens (--state-block NAME=TOKENS:SOURCE)
         the KV block size at which an engine may keep a recurrent state per block (vLLM's mamba "align" prefix
         caching sets the attention block so its page equals the state's page); the builder's upper bound on unique
         KV counts one state per block.

Service time (seconds per turn at the heavy load, the form build_workloads.py sizes with): C0 + TPOT x out_len + B x
prompt / 1000, least squares on the C0 arms of the 2026-10 Qwen3-8B heavy rounds (BF16, vLLM, one RTX PRO 6000 on
Nebius): C0 0.649, TPOT 0.0190, B 0.0266. The new models have no measured run yet, so each coefficient is the Qwen3-8B
one scaled by a cost ratio at the fit's mean prompt (C_REF tokens):
  C0, B  x  prefill FLOPs per prompt token: 2 x linear-layer weights (the FP8 tensors of the checkpoint; Qwen3-8B
            its non-embedding weights) + 4 x sum over attention layers of (query heads x head dim x the tokens it
            attends: c for full layers, min(c, window) for sliding ones) + the Gated DeltaNet state update
            (4 x V heads x key dim x value dim per layer).
  TPOT   x  bytes a decode step reads per sequence: (linear weights + lm_head) / B_REF sequences in flight + the
            sequence's KV (+ the recurrent state, read and written; + the window).
No credit is taken for FP8 tensor-core speed (FP8 weights count at 1 byte for decode reads only), so prefill is
estimated slow, not fast. INFLIGHT_REF (the v1 Qwen3-8B measured workload's mean requests in flight under its fit)
is kept for the v1 sizing rule; v2 sizes by KV pressure (build_workloads.py --size-by-kv).
Measured prefill: --prefill NAME=prefill.json (the compatibility check's microbenchmark, {"LEN": {"b1_tok_s"}},
one request at a time) replaces the FLOP ratio of C0 and B by REF_PREFILL_TPS / (the model's b1 tok/s at C_REF,
log-linear in the prompt length between the measured points). REF_PREFILL_TPS is Qwen3-8B's solo-like prefill rate
on the same GPU type, from the fit's own C0 runs: p90 of uncached tokens / TTFT over turns with at least 8,192
uncached tokens (16,859 / 16,983 / 17,131 tok/s in the three runs, about 22K tokens of context; the median taken).
TPOT stays the bytes estimate until a decode microbenchmark exists. Each model records which parts are measured.
--max-len NAME=FILE takes the max-model-len the compatibility check booted with (default --max-model-len).

Usage: service_model.py OUT.json --ref-config qwen3_8b.config.json --config NAME=CONFIG.json:REPO@REV:F8_PARAMS:BF16_LM_HEAD
       [--prefill NAME=prefill.json] [--max-len NAME=boot_maxlen] [--max-model-len 131072] [--inflight-ref 1.4317]
"""
import argparse
import hashlib
import json
import math
import os

FIT = dict(c0=0.649, tpot=0.0190, b=0.0266, source="least squares on the C0 arms of the 2026-10 Qwen3-8B heavy rounds "
           "(nebius_q8 p1, p2, p4: 2,748 turns; BF16, vLLM, 1x RTX PRO 6000 Nebius); work/svc_fit2.py")
C_REF = 23015        # mean prompt tokens of those turns
REF_PREFILL_TPS = 16983.0   # Qwen3-8B, the fit's C0 runs (docstring)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def text(c):
    return c.get("text_config", c)


def ref_arch(c):
    h, L, i = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"]
    q, kv, d = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
    lin = L * (3 * h * i + h * q * d * 2 + h * kv * d * 2)
    return dict(lin=lin, head=h * c["vocab_size"] * 2, full=[(L, q, d)], swa=[], gdn=0.0, grow=L * kv * d * 2 * 2,
                state=0, window=0, window_bytes=0, layers=L)


def arch(c, f8, head_bytes):
    t = text(c)
    lt = t["layer_types"]
    q = t["num_attention_heads"]
    a = dict(lin=float(f8), head=float(head_bytes), full=[], swa=[], gdn=0.0, grow=0, state=0, window=0,
             window_bytes=0, layers=t["num_hidden_layers"])
    nf = sum(x == "full_attention" for x in lt)
    if t.get("model_type", "").startswith("qwen3_5"):          # Qwen3.8: Gated DeltaNet + gated full attention
        nl = sum(x == "linear_attention" for x in lt)
        hv, dk, dv, hk = t["linear_num_value_heads"], t["linear_key_head_dim"], t["linear_value_head_dim"], t["linear_num_key_heads"]
        ssm = hv * dk * dv * (4 if t.get("mamba_ssm_dtype") == "float32" else 2)
        conv = (t["linear_conv_kernel_dim"] - 1) * (2 * hk * dk + hv * dv) * 2
        a.update(full=[(nf, q, t["head_dim"])], grow=nf * t["num_key_value_heads"] * t["head_dim"] * 2 * 2,
                 state=nl * (ssm + conv), gdn=nl * 4.0 * hv * dk * dv,
                 detail=dict(full_layers=nf, kv_heads=t["num_key_value_heads"], head_dim=t["head_dim"],
                             gdn_layers=nl, ssm_state_bytes_per_layer=ssm, conv_state_bytes_per_layer=conv,
                             ssm_dtype=t.get("mamba_ssm_dtype"), max_position_embeddings=t["max_position_embeddings"]))
    elif t.get("model_type", "").startswith("gemma4"):         # Gemma 4: 5 sliding : 1 global, K = V on global
        ns = sum(x == "sliding_attention" for x in lt)
        gk, gd = t["num_global_key_value_heads"], t["global_head_dim"]
        sk, sd = t["num_key_value_heads"], t["head_dim"]
        a.update(full=[(nf, q, gd)], swa=[(ns, q, sd)], grow=nf * gk * gd * 2 * 2, window=t["sliding_window"],
                 window_bytes=ns * sk * sd * 2 * 2,
                 detail=dict(global_layers=nf, global_kv_heads=gk, global_head_dim=gd, k_eq_v=t["attention_k_eq_v"],
                             k_eq_v_cached_as="K and V slots (the K weights load into both; vLLM gemma4.py)",
                             sliding_layers=ns, sliding_kv_heads=sk, sliding_head_dim=sd,
                             sliding_window=t["sliding_window"], max_position_embeddings=t["max_position_embeddings"]))
    else:
        raise ValueError(f"unknown model_type {t.get('model_type')}")
    return a


def flops(a, c):
    att = sum(n * q * d * c for n, q, d in a["full"]) + sum(n * q * d * min(c, a["window"]) for n, q, d in a["swa"])
    return 2.0 * a["lin"] + 4.0 * att + a["gdn"]


def decode_bytes(a, c, wbytes, b):
    return wbytes / b + a["grow"] * c + 2 * a["state"] + a["window_bytes"] * min(c, a["window"])


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--ref-config", required=True)
ap.add_argument("--config", action="append", required=True,
                help="NAME=CONFIG.json:REPO@REV:F8_PARAMS:LM_HEAD_PARAMS (the FP8 checkpoint's F8_E4M3 tensor count "
                     "from the Hub's safetensors metadata; the BF16 lm_head)")
ap.add_argument("--max-model-len", type=int, default=131072)
ap.add_argument("--inflight-ref", type=float, required=True)
ap.add_argument("--prefill", action="append", default=[], help="NAME=prefill.json")
ap.add_argument("--max-len", action="append", default=[], help="NAME=FILE holding the booted max-model-len")
ap.add_argument("--state-block", action="append", default=[], help="NAME=TOKENS:SOURCE")
ap.add_argument("--gpu-request", action="append", default=[],
                help="NAME=BYTES:SOURCE, the GPU KV the engine needs for one max-length request (vLLM's start check: "
                     "its KV pool / its maximum concurrency at max-model-len, from the compatibility boot)")
a = ap.parse_args()
blocks = {x.split("=", 1)[0]: x.split("=", 1)[1].split(":", 1) for x in a.state_block}
gpureq = {x.split("=", 1)[0]: x.split("=", 1)[1].split(":", 1) for x in a.gpu_request}

rc = json.load(open(a.ref_config))
R = ref_arch(rc)
ref_w = R["lin"] * 2 + R["head"]                     # BF16
f_ref, d_ref = flops(R, C_REF), decode_bytes(R, C_REF, ref_w, a.inflight_ref)
prefill = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in a.prefill}
maxlen = {x.split("=", 1)[0]: x.split("=", 1)[1] for x in a.max_len}


def tps_at(path, c):
    """b1 tok/s at prompt length c, log-linear between the measured lengths (clamped at the ends)."""
    P = sorted((int(k), float(v["b1_tok_s"])) for k, v in json.load(open(path)).items())
    if c <= P[0][0]:
        return P[0][1]
    for (x0, y0), (x1, y1) in zip(P, P[1:]):
        if c <= x1:
            f = math.log(c / x0) / math.log(x1 / x0)
            return y0 + f * (y1 - y0)
    return P[-1][1]

out = []
for spec in a.config:
    name, rest = spec.split("=", 1)
    path, repo, f8, head = rest.split(":")
    c = json.load(open(path))
    A = arch(c, float(f8), float(head) * 2)
    w = A["lin"] * 1 + A["head"]                       # FP8 linear weights, BF16 lm_head
    rp, rd = flops(A, C_REF) / f_ref, decode_bytes(A, C_REF, w, a.inflight_ref) / d_ref
    src, measured = "ESTIMATE: FLOP ratio at C_REF (no measured run)", []
    tps = None
    if name in prefill and os.path.exists(prefill[name]):
        tps = tps_at(prefill[name], C_REF)
        rp, src = REF_PREFILL_TPS / tps, (f"MEASURED: {REF_PREFILL_TPS:.0f} / b1 prefill {tps:.0f} tok/s at {C_REF} "
                                         f"tokens (compat prefill.json sha256 {sha(prefill[name])})")
        measured.append("prefill (C0, B)")
    cap = int(open(maxlen[name]).read().strip()) if name in maxlen and os.path.exists(maxlen[name]) else a.max_model_len
    longest = A["grow"] * cap + A["state"] + A["window_bytes"] * min(cap, A["window"])
    out.append(dict(
        name=name, hf=repo, config_sha256=sha(path), cap=cap,
        cap_source=("compat check boot_maxlen" if name in maxlen and os.path.exists(maxlen[name])
                    else "default 131,072 (compat check not in yet)"),
        kv=dict(dtype="bf16", grow_bytes_per_token=A["grow"], state_bytes_per_session=A["state"],
                window_tokens=A["window"], window_bytes_per_token=A["window_bytes"],
                window_bytes_per_session_max=A["window_bytes"] * A["window"], longest_request_bytes=longest,
                state_block_tokens=int(blocks[name][0]) if name in blocks else None,
                state_block_source=blocks[name][1] if name in blocks else None,
                gpu_request_bytes=int(float(gpureq[name][0])) if name in gpureq else longest,
                gpu_request_source=gpureq[name][1] if name in gpureq else "longest_request_bytes (session KV at the cap)",
                arch=A["detail"]),
        c0=round(FIT["c0"] * rp, 4), tpot=round(FIT["tpot"] * rd, 5), b=round(FIT["b"] * rp, 5),
        inflight=a.inflight_ref,
        service_assumption=dict(
            status=("measured: " + ", ".join(measured) + "; " if measured else "") +
                   ("ESTIMATE (replace with the pilot's microbenchmarks): TPOT" + ("" if measured else ", C0, B")),
            measured=measured,
            ref_model="Qwen3-8B BF16 (Qwen/Qwen3-8B config sha256 " + sha(a.ref_config) + ")", ref_fit=FIT,
            c_ref_tokens=C_REF, prefill_ratio=round(rp, 4), prefill_ratio_source=src, decode_ratio=round(rd, 4),
            ref_prefill_flops_per_token=f_ref, prefill_flops_per_token=flops(A, C_REF),
            ref_decode_bytes_per_seq_step=d_ref, decode_bytes_per_seq_step=decode_bytes(A, C_REF, w, a.inflight_ref),
            linear_params_fp8=float(f8), lm_head_params_bf16=float(head), fp8_compute_credit=False,
            ref_prefill_tok_s=REF_PREFILL_TPS, prefill_tok_s_at_c_ref=round(tps if tps else REF_PREFILL_TPS / rp, 0),
            flop_ratio=round(flops(A, C_REF) / f_ref, 4))))
    print(name, json.dumps({k: v for k, v in out[-1].items() if k not in ("service_assumption",)} |
                           {"ratios": [round(rp, 3), round(rd, 3)]}))
json.dump(out, open(a.out, "w"), indent=1)
