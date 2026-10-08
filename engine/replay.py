#!/usr/bin/env python3
"""Closed-loop replay of agent sessions against a vLLM server (OpenAI completions API) or an SGLang
server (native /generate, --engine sglang).

Each session sends its turns in order. Turn i's prompt is the full token history: earlier prompts,
the model's own outputs (returned token ids), and the new tokens of this turn. The next turn is sent
`gap_after` seconds after this turn finishes, so waits caused by the server delay the session the
way they would a real agent. Output length is fixed per turn (ignore_eos). The move/stay hint for
the idle period after a turn rides on that turn's request as vllm_xargs.park_hint.

Records per turn: session, turn, kind, prompt tokens, hint, TTFT, end-to-end time, queue start.
Scrapes /metrics before and after.

SGLang (--engine sglang): /generate with input_ids and sampling_params (max_new_tokens, ignore_eos,
temperature 0), streamed for TTFT; the output token ids come from the stream (cumulative by default,
incremental with --incremental-streaming-output, both handled). --park-hints tags each turn with
sampling_params.custom_params.park_key and posts its curve to the park.sglang hint port. The vLLM-only
options (--hint, --cpu-hint, --prefetch) are refused. usage holds the turn's meta_info token counts.

--window SECONDS bounds the run in wall time from the start: a session stops before a turn sent past
the window, and at once when a turn's gap_after (or its start) would put the next send past it,
writing {"session", "turn", "truncated_at_window": true} for the turn it does not send. The ended
turn's park hint is still posted (the session simply does not return within the run).

Usage: replay.py replay.json OUT.jsonl --url http://127.0.0.1:8000 --hint none|gbm|oracle
       [--model Qwen/Qwen3-8B] [--vocab 150000] [--engine vllm|sglang] [--window SECONDS]
"""
import argparse, asyncio, json, random, time
import aiohttp

ap = argparse.ArgumentParser()
ap.add_argument("workload"); ap.add_argument("out")
ap.add_argument("--url", default="http://127.0.0.1:8000")
ap.add_argument("--hint", default="none", choices=["none", "gbm", "oracle"])
ap.add_argument("--cpu-hint", default="none", choices=["none", "gbm", "oracle", "laya", "laya2", "ens", "stk", "gbm_s", "laya2_s", "ens_s", "stk_s", "stk_p", "v6_p"],
                help="send kv_transfer_params.park_eta for the CPU-tier ParkCachePolicy")
ap.add_argument("--prefetch", default="none", choices=["none", "gbm", "oracle", "laya", "laya2", "ens", "stk", "gbm_s", "laya2_s", "ens_s", "stk_s", "stk_p", "v6_p"],
                help="warm the session's prefix before its predicted return")
ap.add_argument("--reload-gbps", type=float, default=20.0)
ap.add_argument("--park-hints", help="curve name: tag each turn with kv_transfer_params.park_key and, when it ends, "
                "POST its surv_NAME curve to the park adapter's hint port (park.vllm), as a sidecar would")
ap.add_argument("--park-port", type=int, default=8765)
ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
ap.add_argument("--window", type=float, help="run length in seconds from the start; sessions stop at it")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--vocab", type=int, default=150000)
a = ap.parse_args()
if a.engine == "sglang" and (a.hint != "none" or a.cpu_hint != "none" or a.prefetch != "none"):
    ap.error("--hint, --cpu-hint and --prefetch are vLLM-only; SGLang takes --park-hints")

W = json.load(open(a.workload))
out = open(a.out, "w")
t_zero = None


def toks(rng, n):
    return [rng.randrange(1000, a.vocab) for _ in range(n)]


def eta_of(t):
    if a.cpu_hint == "none":
        return None
    return t[f"eta_{a.cpu_hint}"]


async def turn(sess, i, t, history, hint, park_key=None):
    if a.engine == "sglang":
        return await turn_sglang(sess, t, history, park_key)
    body = {"model": a.model, "prompt": history, "max_tokens": t["out_len"], "ignore_eos": True,
            "temperature": 0.0, "stream": True, "return_token_ids": True,
            "stream_options": {"include_usage": True}}
    if hint != "none":
        body["vllm_xargs"] = {"park_hint": hint}
    eta = eta_of(t)
    if eta is not None:
        body["kv_transfer_params"] = {"park_eta": eta, "park_disk": bool(eta < 0)}
    if a.park_hints:
        body["kv_transfer_params"] = {"park_key": park_key}
    t0 = time.time()
    ttft, out_ids, usage = None, [], None
    async with sess.post(f"{a.url}/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=3600)) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                ids = ch.get("token_ids") or []
                if ids and ttft is None:
                    ttft = time.time() - t0
                out_ids.extend(ids)
    t1 = time.time()
    return out_ids, dict(ttft=ttft, e2e=t1 - t0, sent=t0 - t_zero, usage=usage)


def merge_ids(acc, ids):
    """SGLang streams the output ids so far (cumulative), or only the new ones (incremental)."""
    if len(ids) >= len(acc) and ids[:len(acc)] == acc:
        return list(ids)
    return acc + list(ids)


async def turn_sglang(sess, t, history, park_key=None):
    sp = {"max_new_tokens": t["out_len"], "ignore_eos": True, "temperature": 0.0}
    if a.park_hints:
        sp["custom_params"] = {"park_key": park_key}
    body = {"input_ids": history, "sampling_params": sp, "stream": True}
    t0 = time.time()
    ttft, out_ids, meta = None, [], None
    async with sess.post(f"{a.url}/generate", json=body, timeout=aiohttp.ClientTimeout(total=3600)) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            ids = ev.get("output_ids") or []
            if ids and ttft is None:
                ttft = time.time() - t0
            out_ids = merge_ids(out_ids, ids)
            meta = ev.get("meta_info") or meta
    t1 = time.time()
    usage = None
    if meta:
        usage = {k: meta[k] for k in ("prompt_tokens", "completion_tokens", "cached_tokens",
                                      "cached_tokens_details") if k in meta}
    return out_ids, dict(ttft=ttft, e2e=t1 - t0, sent=t0 - t_zero, usage=usage)


async def warm(http, history, eta):
    body = {"model": a.model, "prompt": history, "max_tokens": 1, "temperature": 0.0}
    if eta is not None:
        body["kv_transfer_params"] = {"park_eta": eta, "park_disk": bool(eta < 0)}
    t0 = time.time()
    try:
        async with http.post(f"{a.url}/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=600)) as r:
            await r.read()
        return time.time() - t0
    except Exception:
        return None


hint_tasks: set = set()


def post_hint(http, key, surv):
    task = asyncio.create_task(park_hint(http, key, surv))
    hint_tasks.add(task)
    task.add_done_callback(hint_tasks.discard)


def past_window(delay=0.0):
    """True when a send `delay` seconds from now falls past the run window."""
    return a.window is not None and time.time() + delay - t_zero > a.window


def truncate(s, i):
    out.write(json.dumps(dict(session=s["id"], turn=i, truncated_at_window=True)) + "\n"); out.flush()


async def park_hint(http, key, surv):
    """What the sidecar does when a response ends: send the idle-period curve to the engine adapter.
    A turn the model did not score gets 0.5 everywhere (as the simulator's curve_of)."""
    surv = surv if surv is not None else [0.5] * 15
    try:
        async with http.post(f"http://127.0.0.1:{a.park_port}/hint", json={"key": key, "surv": surv},
                             timeout=aiohttp.ClientTimeout(total=10)) as r:
            await r.read()
    except Exception as e:  # noqa: BLE001  (a lost hint leaves the request on LRU order)
        out.write(json.dumps(dict(park_hint_error=repr(e)[:200], key=key)) + "\n"); out.flush()


def warm_at(t):
    """Seconds after the turn ends to start a prefetch, or None. Uses the workload's precomputed
    pre_<src> when present (confidence-gated decision), else the older lead-based rule."""
    if a.prefetch == "none" or t["gap_after"] is None:
        return None
    if f"pre_{a.prefetch}" in t:
        return t[f"pre_{a.prefetch}"]
    lead = t["ctx"] * 144 * 1024 / (a.reload_gbps * 1e9) + 0.3
    target = t["gap_after"] if a.prefetch == "oracle" else t["q10_gbm"]
    if target is None or target < lead + 1.0:
        return None
    return target - lead


async def run_session(http, s):
    rng = random.Random(s["seed"])
    if past_window(s["start"]):
        return truncate(s, 0)
    await asyncio.sleep(s["start"])
    history = []
    for i, t in enumerate(s["turns"]):
        if past_window():
            return truncate(s, i)
        history = history + toks(rng, t["new_tokens"])
        hint = "none" if a.hint == "none" else t[f"hint_{a.hint}"]
        try:
            try:
                out_ids, m = await turn(http, i, t, history, hint, f"{s['id']}-{i}")
            except aiohttp.ClientConnectionError:  # a dropped keep-alive connection: resend once, flag the row
                out_ids, m = await turn(http, i, t, history, hint, f"{s['id']}-{i}")
                m["retried"] = 1
        except Exception as e:  # one failed turn ends its session, not the replay
            out.write(json.dumps(dict(session=s["id"], turn=i, error=repr(e)[:300])) + "\n"); out.flush()
            break
        history = history + out_ids
        if a.park_hints and t["gap_after"] is not None:
            post_hint(http, f"{s['id']}-{i}", t.get(f"surv_{a.park_hints}"))
        rec = dict(session=s["id"], pool=s["pool"], turn=i, kind=t["kind"], prompt_tokens=len(history) - len(out_ids),
                   out_tokens=len(out_ids), hint=hint, prev_hint=(s["turns"][i - 1][f"hint_{a.hint}"] if i and a.hint != "none" else "none"),
                   prev_gap=(s["turns"][i - 1]["gap_after"] if i else None), **m)
        out.write(json.dumps(rec) + "\n"); out.flush()
        if t["gap_after"] is None:
            break
        if past_window(t["gap_after"]):
            return truncate(s, i + 1)
        w = warm_at(t)
        if w is not None and w < t["gap_after"]:
            await asyncio.sleep(w)
            eta = eta_of(t)
            asyncio.create_task(warm(http, list(history), None if eta is None else (-1.0 if eta < 0 else max(eta - w, 0.5))))
            out.write(json.dumps(dict(session=s["id"], turn=i, warm_at=w)) + "\n"); out.flush()
            await asyncio.sleep(t["gap_after"] - w)
        else:
            await asyncio.sleep(t["gap_after"])


async def scrape(http, tag):
    async with http.get(f"{a.url}/metrics") as r:
        txt = await r.text()
    if a.engine == "sglang":
        keep = [l for l in txt.splitlines() if l.startswith("sglang:") and any(k in l for k in (
            "cache_hit", "cached_tokens", "hicache", "load_back", "backup", "prefetch", "evict", "dropped",
            "prompt_tokens", "generation_tokens", "token_usage", "num_requests", "retract"))]
    else:
        keep = [l for l in txt.splitlines() if l.startswith("vllm:") and any(k in l for k in (
            "prefix_cache", "kv_offload", "external", "num_preemptions", "prompt_tokens", "generation_tokens",
            "kv_cache_usage", "request_success", "connector"))]
    out.write(json.dumps({"metrics": tag, "lines": keep}) + "\n"); out.flush()


async def main():
    global t_zero
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as http:
        await scrape(http, "before")
        t_zero = time.time()
        await asyncio.gather(*(run_session(http, s) for s in W["sessions"]))
        out.write(json.dumps({"wall_s": time.time() - t_zero}) + "\n")
        if hint_tasks:
            await asyncio.gather(*list(hint_tasks))   # hints of the last turns before the window
        await scrape(http, "after")

asyncio.run(main())
