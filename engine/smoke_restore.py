#!/usr/bin/env python3
"""Smoke gate for a KV offload stack: does a session evicted from the GPU come back from the CPU tier (and from disk)
as external prefix-cache hits, with the same greedy output?

Sends session A (a long random-token prompt) and keeps its first --gen greedy tokens; then sends filler sessions until
A's GPU blocks must be gone (filler tokens >= --gpu-tokens), resends A and reads the engine's prefix-cache counters
(local vs external hit tokens) from /metrics around that request; then, with --cpu-tokens, sends enough filler to push
A out of the CPU tier too and repeats (the restore must come from disk). Prints one JSON line per phase and PASS/FAIL.

Usage: smoke_restore.py --url http://127.0.0.1:8000 --model M --prompt 60000 --gpu-tokens N [--cpu-tokens N]
"""
import argparse, json, random, re, sys, time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--url", default="http://127.0.0.1:8000"); ap.add_argument("--model", required=True)
ap.add_argument("--prompt", type=int, default=60000); ap.add_argument("--gen", type=int, default=64)
ap.add_argument("--gpu-tokens", type=int, required=True); ap.add_argument("--cpu-tokens", type=int, default=0)
ap.add_argument("--filler", type=int, default=30000); ap.add_argument("--vocab", type=int, default=150000)
a = ap.parse_args()


def post(prompt, n):
    body = json.dumps({"model": a.model, "prompt": prompt, "max_tokens": n, "temperature": 0.0, "ignore_eos": True,
                       "return_token_ids": True}).encode()
    req = urllib.request.Request(a.url + "/v1/completions", body, {"Content-Type": "application/json"})
    t = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=3600))
    ch = r["choices"][0]
    return time.time() - t, ch.get("token_ids") or ch.get("text"), r.get("usage", {})


def metrics():
    txt = urllib.request.urlopen(a.url + "/metrics", timeout=60).read().decode()
    out = {}
    for name in ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total",
                 "vllm:external_prefix_cache_hits_total", "vllm:external_prefix_cache_queries_total"):
        out[name] = sum(float(m) for m in re.findall(r"^" + re.escape(name) + r"(?:\{[^}]*\})? ([0-9.e+]+)$", txt, re.M))
    return out


def toks(seed, n):
    rng = random.Random(seed)
    return [rng.randrange(1000, a.vocab) for _ in range(n)]


def phase(name, prompt, ref):
    m0 = metrics()
    dt, out, usage = post(prompt, a.gen)
    m1 = metrics()
    d = {k.split(":")[1]: m1[k] - m0[k] for k in m0}
    rec = dict(phase=name, ttft_s=round(dt, 2), local_hits=d["prefix_cache_hits_total"],
               external_hits=d["external_prefix_cache_hits_total"], prompt=len(prompt), same_output=out == ref,
               cached=(usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    print(json.dumps(rec), flush=True)
    return rec


A = toks(1, a.prompt)
dt, ref, _ = post(A, a.gen)
print(json.dumps(dict(phase="cold", ttft_s=round(dt, 2), prompt=len(A))), flush=True)
time.sleep(2)
phase("gpu_hit", A, ref)
sent, i = 0, 100
while sent < a.gpu_tokens * 1.2:
    post(toks(i, a.filler), 1); sent += a.filler; i += 1
time.sleep(5)
r1 = phase("cpu_restore", A, ref)
ok = r1["external_hits"] >= 0.9 * (a.prompt - a.prompt % 256 - 256) and r1["same_output"]
if a.cpu_tokens:
    sent = 0
    while sent < (a.gpu_tokens + a.cpu_tokens) * 1.2:
        post(toks(i, a.filler), 1); sent += a.filler; i += 1
    time.sleep(10)
    r2 = phase("disk_restore", A, ref)
    ok = ok and r2["external_hits"] >= 0.9 * (a.prompt - a.prompt % 256 - 256) and r2["same_output"]
print("PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
