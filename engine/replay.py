#!/usr/bin/env python3
"""Closed-loop replay of agent sessions against a vLLM server (OpenAI completions API) or an SGLang
server (native /generate, --engine sglang).

Each session sends its turns in order. Turn i's prompt is the full token history: earlier prompts,
the model's own outputs (returned token ids), and the new tokens of this turn. The next turn is sent
`gap_after` seconds after this turn finishes, so waits caused by the server delay the session the
way they would a real agent. Output length is fixed per turn (ignore_eos). The move/stay hint for
the idle period after a turn rides on that turn's request as vllm_xargs.park_hint.

Run to completion (Phase R): every session runs to its last turn. --hard-cap SECONDS bounds the run in wall
time from the start; at the cap, in-flight requests are cancelled and every turn not served by then (the
in-flight one and all later ones, including sessions not yet started) is written as censored:
{"censored": true, "ttft": Infinity, "e2e": Infinity}. Without --hard-cap the run waits for every session.

Failed turns: an HTTP error, a timeout (--turn-timeout, default 3600 s), a stream with no token, or any other
error on a turn writes the turn as failed ({"failed": true, "ttft": Infinity, "error": ...}) and the session
goes on: the next turn is sent gap_after seconds after the failure. The failed turn's new tokens stay in the
history (the client sent them); its output does not exist, so nothing is appended for it, and no park hint is
posted for it (no response ended). A dropped keep-alive connection is resent once first (row flag "retried").

Records per turn: session, turn, kind, tool and prog (when the workload has them), the previous turn's kind,
tool and gap, prompt tokens, hint, TTFT, end-to-end time, send time, and concurrency (requests in flight at
send time, this one included). The last rows hold {"concurrency_series": [[t, n], ...]} (every change of the
in-flight count, t in seconds from the start) and {"wall_s", "served", "failed", "censored"}. Infinity is
written as the JSON token Infinity (Python's json reads it). Scrapes /metrics before and after. Writes
OUT.env.json beside the output (GPUs, driver, torch and CUDA versions, repo commit, server version; every
field optional).

Open loop (--open-loop-rate R --requests N [--seed S], H5's throughput sweep): arrivals form a Poisson process at
R per second from the seed; each arrival sends the next turn of a session whose previous turn has finished (picked
at random from the seed's second stream; an arrival with no ready session is dropped and counted). Turns of one
session stay in order and build the same history as in the closed loop; workload gaps and starts are not used.
The run stops after N sends and waits for the in-flight ones (censored if still in flight at --hard-cap). Rows are
the same, plus "arrival" (scheduled seconds from the start); the summary adds the rate, requests sent and dropped
arrivals. The output must be named CELL__ARM__HOST__rate<R>.jsonl, as analyze_phase_r.py reads it.

Health check: before anything is sent, the workload's session and turn counts are printed; with
--expect-sessions / --expect-turns a mismatch exits with status 2 and sends nothing.

Park hints (--park-hints NAME): each turn is tagged with park_key "SESSION-TURN" and, when it ends, the turn's
precomputed curve surv_NAME from the workload is posted to the adapter's hint port, as a sidecar would.
The curve names: v6 (Layby-Dwell), km and ewma (the K_km and K_ewma controls: surv_km and surv_ewma, added to the
workload by layby.curves annotate). Before anything is sent, the turns with a wait but no surv_NAME are counted; more
than 1% of them (any at all for km and ewma, which annotate gives every such turn) exits with status 2.
Continuum (--continuum, arm CT, vLLM only): each turn carries kv_transfer_params park_key "SESSION-TURN", park_kind
(the turn's kind: "tool" when its response ends in a tool call), park_tool (the workload's tool name for the call
after this response) and park_last (true on the session's last turn), as layby.vllm.continuum reads them. No curve is
posted. Not with --park-hints.
Live scoring (--side-file PATH --sidecar URL, with --park-hints NAME): when a turn ends, the turn's real
server-view state is read from the side file and sent to the sidecar's POST URL/park/score, which scores it
with the frozen model, sends the curve to the adapter itself and returns it; the replay does not post the
precomputed curve. Both curves are logged per turn ({"curve": "SESSION-TURN", "pre": [...], "live": [...],
"max_abs_diff": d, "score_s": seconds}). A turn with no side-file entry, or whose scoring call fails, gets the
precomputed curve posted as before and is flagged ("live_missing" / "live_error"). Side file: JSONL, one row
per turn: {"session": workload session id, "turn": i, "state": server-view state (JSON string or object),
"kind": "tool"|"human" (optional, defaults to the turn's kind)}.

SGLang (--engine sglang): /generate with input_ids and sampling_params (max_new_tokens, ignore_eos,
temperature 0), streamed for TTFT; the output token ids come from the stream (cumulative by default,
incremental with --incremental-streaming-output, both handled). --park-hints tags each turn with
sampling_params.custom_params.park_key and posts its curve to the park.sglang hint port. The vLLM-only
options (--hint, --cpu-hint, --prefetch) are refused. usage holds the turn's meta_info token counts.

Usage: replay.py replay.json OUT.jsonl --url http://127.0.0.1:8000 --hint none|gbm|oracle
       [--model Qwen/Qwen3-8B] [--vocab 150000] [--engine vllm|sglang] [--hard-cap SECONDS]
       [--expect-sessions N --expect-turns M] [--park-hints v6|km|ewma [--side-file SIDE.jsonl --sidecar URL]]
       [--continuum]
       [--open-loop-rate R --requests N [--seed 0]]
"""
import argparse, asyncio, json, os, random, subprocess, sys, time
import aiohttp

INF = float("inf")

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
ap.add_argument("--continuum", action="store_true", help="arm CT: send park_key, park_kind, park_tool and park_last "
                "in kv_transfer_params (layby.vllm.continuum); vLLM only")
ap.add_argument("--side-file", help="live scoring: JSONL of per-turn server-view states, keyed by session and turn")
ap.add_argument("--sidecar", help="live scoring: the sidecar's base URL (POST URL/park/score)")
ap.add_argument("--engine", default="vllm", choices=["vllm", "sglang"])
ap.add_argument("--hard-cap", type=float, help="seconds from the start; turns not served by then are censored")
ap.add_argument("--turn-timeout", type=float, default=3600.0, help="seconds per request before the turn fails")
ap.add_argument("--expect-sessions", type=int, help="health check: exact session count of the workload")
ap.add_argument("--expect-turns", type=int, help="health check: exact turn count of the workload")
ap.add_argument("--open-loop-rate", type=float, help="open loop: Poisson arrivals at this many requests per second")
ap.add_argument("--requests", type=int, help="open loop: requests to send")
ap.add_argument("--seed", type=int, default=0, help="open loop: arrival and session-pick seed")
ap.add_argument("--model", default="Qwen/Qwen3-8B")
ap.add_argument("--vocab", type=int, default=150000)
a = ap.parse_args()
if a.engine == "sglang" and (a.hint != "none" or a.cpu_hint != "none" or a.prefetch != "none"):
    ap.error("--hint, --cpu-hint and --prefetch are vLLM-only; SGLang takes --park-hints")
if a.continuum and (a.engine != "vllm" or a.park_hints):
    ap.error("--continuum is the vLLM CT arm: not with --engine sglang nor with --park-hints")
if bool(a.side_file) != bool(a.sidecar):
    ap.error("--side-file and --sidecar go together")
if a.side_file and not a.park_hints:
    ap.error("live scoring needs --park-hints NAME (the precomputed curve it is compared with)")
if a.open_loop_rate is not None:
    if a.open_loop_rate <= 0 or not a.requests or a.requests <= 0:
        ap.error("--open-loop-rate needs a positive rate and --requests N")
    if a.prefetch != "none":
        ap.error("--prefetch uses the closed loop's gaps; not with --open-loop-rate")
    _parts = os.path.basename(a.out)[:-6].split("__") if a.out.endswith(".jsonl") else []
    try:
        _ok = len(_parts) == 4 and all(_parts) and _parts[3].startswith("rate") and float(_parts[3][4:]) == a.open_loop_rate
    except ValueError:
        _ok = False
    if not _ok:
        ap.error(f"open-loop output must be named CELL__ARM__HOST__rate{a.open_loop_rate:g}.jsonl "
                 "(analyze_phase_r.py reads the rate from the name)")
elif a.requests is not None:
    ap.error("--requests is for --open-loop-rate")

W = json.load(open(a.workload))


def health_check():
    """Print the workload's counts and stop before sending anything if they differ from the expected ones."""
    n_s = len(W["sessions"]); n_t = sum(len(s["turns"]) for s in W["sessions"])
    print(f"workload {os.path.basename(a.workload)}: sessions {n_s} turns {n_t} resumed {n_t - n_s}", flush=True)
    bad = [f"{what} {got} != expected {want}" for what, got, want in
           (("sessions", n_s, a.expect_sessions), ("turns", n_t, a.expect_turns)) if want is not None and got != want]
    if bad:
        print("HEALTH CHECK FAILED: " + "; ".join(bad), file=sys.stderr, flush=True)
        sys.exit(2)
    return n_s, n_t


N_SESSIONS, N_TURNS = health_check()
STRICT_CURVES = ("km", "ewma")         # layby.curves annotate gives every turn with a wait its curve


def curve_check():
    """With --park-hints NAME: turns with a wait (the ones a curve is posted for) that lack surv_NAME. Over 1% of
    them (any for km and ewma) stops the run before anything is sent: a missing curve would be posted as 0.5."""
    need = [t for s in W["sessions"] for t in s["turns"] if t.get("gap_after") is not None]
    miss = sum(f"surv_{a.park_hints}" not in t for t in need)
    print(f"curves surv_{a.park_hints}: {len(need) - miss} of {len(need)} turns with a wait", flush=True)
    if miss and (a.park_hints in STRICT_CURVES or miss > 0.01 * len(need)):
        print(f"HEALTH CHECK FAILED: {miss} turns with a wait lack surv_{a.park_hints}", file=sys.stderr, flush=True)
        sys.exit(2)


if a.park_hints:
    curve_check()


def load_side(path):
    side = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            st = r["state"]
            side[(r["session"], int(r["turn"]))] = (st if isinstance(st, str) else json.dumps(st), r.get("kind"))
    have = sum((s["id"], i) in side for s in W["sessions"] for i in range(len(s["turns"])))
    print(f"side file {os.path.basename(path)}: {len(side)} rows, {have} of {N_TURNS} workload turns covered", flush=True)
    return side


SIDE = load_side(a.side_file) if a.side_file else None
out = open(a.out, "w")
t_zero = None
inflight = 0
conc_series = []                       # (seconds from the start, requests in flight) at every change
progress = {}                          # session id -> index of the next turn without a row
counts = dict(served=0, failed=0, censored=0)


def emit(rec):
    out.write(json.dumps(rec) + "\n"); out.flush()


def toks(rng, n):
    return [rng.randrange(1000, a.vocab) for _ in range(n)]


def eta_of(t):
    if a.cpu_hint == "none":
        return None
    return t[f"eta_{a.cpu_hint}"]


def _conc(delta):
    global inflight
    inflight += delta
    conc_series.append((round(time.time() - t_zero, 3), inflight))
    return inflight


async def turn(sess, i, t, history, hint, park_key=None, last=False):
    """One request; counts itself in flight from send to end (cancellation included)."""
    conc = _conc(+1)
    try:
        if a.engine == "sglang":
            out_ids, m = await turn_sglang(sess, t, history, park_key)
        else:
            out_ids, m = await turn_vllm(sess, t, history, hint, park_key, last)
    finally:
        _conc(-1)
    m["concurrency"] = conc
    return out_ids, m


def ct_params(t, park_key, last):
    """What layby.vllm.continuum reads per request (arm CT)."""
    return {"park_key": park_key, "park_kind": t["kind"], "park_tool": t.get("tool"), "park_last": bool(last)}


async def turn_vllm(sess, t, history, hint, park_key=None, last=False):
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
    if a.continuum:
        body["kv_transfer_params"] = ct_params(t, park_key, last)
    t0 = time.time()
    ttft, out_ids, usage = None, [], None
    async with sess.post(f"{a.url}/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=a.turn_timeout)) as r:
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
    async with sess.post(f"{a.url}/generate", json=body, timeout=aiohttp.ClientTimeout(total=a.turn_timeout)) as r:
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


def post_hint(http, s, i, t):
    key = f"{s['id']}-{i}"
    pre = t.get(f"surv_{a.park_hints}")
    coro = park_hint(http, key, pre) if SIDE is None else live_hint(http, s["id"], i, t, key, pre)
    task = asyncio.create_task(coro)
    hint_tasks.add(task)
    task.add_done_callback(hint_tasks.discard)


async def park_hint(http, key, surv):
    """What the sidecar does when a response ends: send the idle-period curve to the engine adapter.
    A turn the model did not score gets 0.5 everywhere (as the simulator's curve_of)."""
    surv = surv if surv is not None else [0.5] * 15
    try:
        async with http.post(f"http://127.0.0.1:{a.park_port}/hint", json={"key": key, "surv": surv},
                             timeout=aiohttp.ClientTimeout(total=10)) as r:
            await r.read()
    except Exception as e:  # noqa: BLE001  (a lost hint leaves the request on LRU order)
        emit(dict(park_hint_error=repr(e)[:200], key=key))


async def live_hint(http, sid, i, t, key, pre):
    """Live scoring: the sidecar scores the turn's real state and sends the curve to the adapter itself.
    Both curves are logged; a missing state or a failed call falls back to the precomputed curve."""
    rec = dict(curve=key, session=sid, turn=i, pre=pre)
    got = SIDE.get((sid, i))
    if got is None:
        rec["live_missing"] = True
        emit(rec)
        return await park_hint(http, key, pre)
    state, kind = got
    t0 = time.time()
    try:
        async with http.post(f"{a.sidecar.rstrip('/')}/park/score",
                             json={"key": key, "state": state, "kind": kind or t["kind"]},
                             timeout=aiohttp.ClientTimeout(total=60)) as r:
            r.raise_for_status()
            live = (await r.json(content_type=None))["surv"]
    except Exception as e:  # noqa: BLE001
        rec.update(live_error=repr(e)[:200], score_s=round(time.time() - t0, 4))
        emit(rec)
        return await park_hint(http, key, pre)
    rec.update(live=live, score_s=round(time.time() - t0, 4))
    if pre is not None and len(pre) == len(live):
        rec["max_abs_diff"] = max(abs(float(x) - float(y)) for x, y in zip(pre, live))
    emit(rec)


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


def meta_of(s, i):
    """Fields every row of turn i carries, served or not: what the analysis bins by."""
    t = s["turns"][i]
    p = s["turns"][i - 1] if i else None
    return dict(session=s["id"], pool=s.get("pool"), turn=i, kind=t["kind"], tool=t.get("tool"), prog=t.get("prog"),
                prev_kind=p["kind"] if p else None, prev_tool=p.get("tool") if p else None,
                prev_gap=p["gap_after"] if p else None)


async def serve_turn(http, s, st, i, extra=None):
    """Send turn i of session s on the session's state st (rng, history) and write its row. Returns True when
    served. A failed turn writes a failed row; its new tokens stay in the history, it adds no output."""
    t = s["turns"][i]
    st["history"] = st["history"] + toks(st["rng"], t["new_tokens"])
    history = st["history"]
    hint = "none" if a.hint == "none" else t[f"hint_{a.hint}"]
    last = i == len(s["turns"]) - 1 or t.get("gap_after") is None
    try:
        try:
            out_ids, m = await turn(http, i, t, history, hint, f"{s['id']}-{i}", last)
        except aiohttp.ClientConnectionError:  # a dropped keep-alive connection: resend once, flag the row
            out_ids, m = await turn(http, i, t, history, hint, f"{s['id']}-{i}", last)
            m["retried"] = 1
        if m["ttft"] is None:
            raise RuntimeError("stream ended without a token")
    except Exception as e:  # a failed turn: recorded as infinite TTFT, the session goes on
        err = f"HTTP {e.status} {e.message}" if isinstance(e, aiohttp.ClientResponseError) else repr(e)
        emit(dict(meta_of(s, i), prompt_tokens=len(history), out_tokens=0, hint=hint, ttft=INF, e2e=INF,
                  failed=True, error=err[:300], sent=None, concurrency=None, **(extra or {})))
        counts["failed"] += 1
        return False
    st["history"] = history + out_ids
    if a.park_hints and t["gap_after"] is not None:
        post_hint(http, s, i, t)
    emit(dict(meta_of(s, i), prompt_tokens=len(history), out_tokens=len(out_ids), hint=hint,
              prev_hint=(s["turns"][i - 1][f"hint_{a.hint}"] if i and a.hint != "none" else "none"), **m,
              **(extra or {})))
    counts["served"] += 1
    return True


async def run_session(http, s):
    st = dict(rng=random.Random(s["seed"]), history=[])
    progress[s["id"]] = 0
    await asyncio.sleep(s["start"])
    for i, t in enumerate(s["turns"]):
        served = await serve_turn(http, s, st, i)
        progress[s["id"]] = i + 1
        if t["gap_after"] is None:
            break
        w = warm_at(t) if served else None
        if w is not None and w < t["gap_after"]:
            await asyncio.sleep(w)
            eta = eta_of(t)
            asyncio.create_task(warm(http, list(st["history"]), None if eta is None else (-1.0 if eta < 0 else max(eta - w, 0.5))))
            emit(dict(session=s["id"], turn=i, warm_at=w))
            await asyncio.sleep(t["gap_after"] - w)
        else:
            await asyncio.sleep(t["gap_after"])


async def open_loop(http):
    """Open loop: arrivals form a Poisson process at --open-loop-rate from --seed (the same arrival times in every
    arm); each arrival sends the next turn of a session that is ready (its previous turn finished, turns left),
    picked at random from a second stream of the same seed. An arrival with no ready session is dropped and
    counted. Stops after --requests sends, then waits for the in-flight ones (up to the hard cap; still in flight
    at the cap: censored). Gaps and session starts of the workload are not used."""
    arr, pick = random.Random(a.seed), random.Random(a.seed + 1)
    st = {s["id"]: dict(rng=random.Random(s["seed"]), history=[], next=0, busy=False) for s in W["sessions"]}
    tasks, inflight_turns = set(), {}
    sent = dropped = 0
    t_arr = 0.0

    async def one(s, i):
        try:
            await serve_turn(http, s, st[s["id"]], i, extra=dict(arrival=round(t_arr_of[(s["id"], i)], 4)))
        finally:
            st[s["id"]]["busy"] = False
            inflight_turns.pop((s["id"], i), None)

    t_arr_of = {}
    while sent < a.requests:
        t_arr += arr.expovariate(a.open_loop_rate)
        if a.hard_cap is not None and t_arr > a.hard_cap:
            break
        delay = t_zero + t_arr - time.time()
        if delay > 0:
            await asyncio.sleep(delay)
        ready = [s for s in W["sessions"] if not st[s["id"]]["busy"] and st[s["id"]]["next"] < len(s["turns"])]
        if not ready:
            if all(not v["busy"] for v in st.values()):
                break                                     # the workload is used up
            dropped += 1
            continue
        s = pick.choice(ready)
        i = st[s["id"]]["next"]
        st[s["id"]].update(next=i + 1, busy=True)
        t_arr_of[(s["id"], i)] = t_arr
        inflight_turns[(s["id"], i)] = s
        task = asyncio.create_task(one(s, i))
        tasks.add(task)
        sent += 1
    left = None if a.hard_cap is None else max(a.hard_cap - (time.time() - t_zero), 0.0)
    done, pending = await asyncio.wait(tasks, timeout=left) if tasks else (set(), set())
    for t in pending:
        t.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    for (sid, i), s in list(inflight_turns.items()):
        emit(dict(meta_of(s, i), ttft=INF, e2e=INF, censored=True, arrival=round(t_arr_of[(sid, i)], 4)))
        counts["censored"] += 1
    return dict(open_loop_rate=a.open_loop_rate, requests=sent, arrivals_dropped=dropped), bool(pending)


def censor_rest():
    """At the hard cap: every turn without a row (in flight, waiting, or in a session not yet started)."""
    for s in W["sessions"]:
        for i in range(progress.get(s["id"], 0), len(s["turns"])):
            emit(dict(meta_of(s, i), ttft=INF, e2e=INF, censored=True))
            counts["censored"] += 1


async def scrape(http, tag):
    try:
        async with http.get(f"{a.url}/metrics", timeout=aiohttp.ClientTimeout(total=30)) as r:
            txt = await r.text()
    except Exception as e:  # noqa: BLE001  (a dead server after the run: keep the turn rows)
        emit({"metrics": tag, "error": repr(e)[:200], "lines": []})
        return
    if a.engine == "sglang":
        keep = [l for l in txt.splitlines() if l.startswith("sglang:") and any(k in l for k in (
            "cache_hit", "cached_tokens", "hicache", "load_back", "backup", "prefetch", "evict", "dropped",
            "prompt_tokens", "generation_tokens", "token_usage", "num_requests", "retract"))]
    else:
        keep = [l for l in txt.splitlines() if l.startswith("vllm:") and any(k in l for k in (
            "prefix_cache", "kv_offload", "external", "num_preemptions", "prompt_tokens", "generation_tokens",
            "kv_cache_usage", "request_success", "connector"))]
    emit({"metrics": tag, "lines": keep})


def _cmd(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


async def write_env(http):
    """OUT.env.json beside OUT.jsonl: what ran where. Every field is optional and a failure here never fails the run:
    GPUs (name, driver, memory) from nvidia-smi if present, torch and CUDA versions if torch imports, this repo's
    commit from git, the server's /version answer (vLLM and SGLang both serve it)."""
    env = dict(time_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), engine=a.engine, url=a.url, model=a.model,
               workload=os.path.basename(a.workload), sessions=N_SESSIONS, turns=N_TURNS, hard_cap=a.hard_cap,
               turn_timeout=a.turn_timeout, live_scoring=bool(SIDE), python=sys.version.split()[0],
               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    smi = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    if smi:
        rows = [[c.strip() for c in l.split(",")] for l in smi.splitlines() if l.strip()]
        env["gpus"] = [dict(name=r[0], driver=r[1], memory=r[2]) for r in rows if len(r) >= 3]
    try:
        import torch
        env["torch"], env["cuda"] = torch.__version__, torch.version.cuda
    except Exception:
        pass
    env["commit"] = _cmd(["git", "-C", os.path.dirname(os.path.abspath(__file__)), "rev-parse", "HEAD"])
    try:
        async with http.get(f"{a.url}/version", timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status == 200:
                j = await r.json(content_type=None)
                env["server_version"] = j.get("version", j) if isinstance(j, dict) else j
    except Exception:
        pass
    try:
        with open((a.out[:-6] if a.out.endswith(".jsonl") else a.out) + ".env.json", "w") as f:
            json.dump(env, f, indent=1); f.write("\n")
    except Exception as e:
        print("env.json not written:", repr(e)[:200], file=sys.stderr)


async def main():
    global t_zero
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn) as http:
        await write_env(http)
        await scrape(http, "before")
        t_zero = time.time()
        conc_series.append((0.0, 0))
        extra = {}
        if a.open_loop_rate:
            extra, cap_hit = await open_loop(http)
        else:
            tasks = [asyncio.create_task(run_session(http, s)) for s in W["sessions"]]
            done, pending = await asyncio.wait(tasks, timeout=a.hard_cap)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for t in done:
                if not t.cancelled() and t.exception() is not None:   # a replay bug, not a failed turn: say so loudly
                    print("session task error:", repr(t.exception())[:300], file=sys.stderr)
            censor_rest()
            cap_hit = bool(pending)
        wall = time.time() - t_zero
        if hint_tasks:
            await asyncio.gather(*list(hint_tasks))   # hints of the last served turns
        emit({"concurrency_series": conc_series})
        emit(dict(wall_s=wall, hard_cap=a.hard_cap, cap_hit=cap_hit, **counts, **extra))
        await scrape(http, "after")

asyncio.run(main())
