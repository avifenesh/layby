#!/usr/bin/env python3
"""park sidecar: an OpenAI-compatible proxy in front of an engine that gives the engine's park adapter
the idle-period curve of every finished response.

Per request: link it to its session (layby.sidecar.session), tag it with a park_key the engine adapter
sees, and forward it, streaming or not, while collecting the response (text, reasoning, tool calls,
usage). When the response ends: build the v6 server-view state, score it with Laya (batched, on a
background task), and send {key, surv} to the engine adapter.

Live scoring for replays (POST /park/score {"key", "state", "kind"}): a client that already holds the turn's
server-view state (engine/replay.py --side-file) gets it scored by the same batched path, the curve sent to the
engine adapter under its key, and {"key", "surv"} back once the adapter has it.

Engines (--engine):
  vllm    park key in kv_transfer_params; hints to layby.vllm's hint port (POST /hint)
  sglang  park key in custom_params (SGLang's OpenAI API passes it to the scheduler as
          sampling_params.custom_params); hints to layby.sglang's hint port (POST /hint, same API)

Usage: python -m layby.sidecar.proxy --upstream http://127.0.0.1:8000 --listen 8100
         --ckpt DWELL/model.safetensors --calib DWELL/config.json   (DWELL: the Layby-Dwell release)
         [--engine vllm|sglang] [--hint-url http://127.0.0.1:8765/hint] [--device cuda|cpu]
Content stays on this host: states are scored in process and only curves leave the proxy.
"""
import argparse
import asyncio
from collections import OrderedDict, deque
import hashlib
import json
import logging
import math
import time
import uuid

import aiohttp
from aiohttp import web
import numpy as np

from layby.rule import E2
from layby.sidecar.session import Tracker

log = logging.getLogger("layby.sidecar")


class OnlineCalib:
    """Rolling realized-gap calibration monitor + optional adaptive per-kind temperature.

    Stores recent (logits, surv) keyed by boundary key; when Tracker links the session's next request
    and measures the realized gap, records exact discrete survival NLL and 300s calibration error,
    and (when enabled) adapts per-kind temperature on a sliding window to absorb workload/harness drift.
    """

    def __init__(self, base_temps: dict[str, float], enabled: bool = False, window: int = 500, max_pending: int = 20000):
        self.base_temps = dict(base_temps or {"_all": 1.0})
        self.temps = dict(self.base_temps)
        self.enabled = bool(enabled)
        self.window = int(window)
        self.max_pending = int(max_pending)
        self._pending: "OrderedDict[str, tuple[np.ndarray, np.ndarray, str]]" = OrderedDict()
        self._obs: dict[str, deque] = {}   # kind -> deque[(logits, surv, bin_idx, gap)]
        self._updates = 0

    def record_prediction(self, key: str, logits: np.ndarray, surv: np.ndarray, kind: str) -> None:
        if key is None or np.isnan(logits).any():
            return
        self._pending[key] = (np.asarray(logits, np.float32), np.asarray(surv, np.float32), str(kind))
        self._pending.move_to_end(key)
        while len(self._pending) > self.max_pending:
            self._pending.popitem(last=False)

    def observe_return(self, key: str, gap: float, kind: str) -> None:
        item = self._pending.pop(key, None)
        if item is None:
            return
        lg, surv, pred_kind = item
        kd = str(kind or pred_kind)
        b = int(np.searchsorted(E2, gap, side="right"))
        q = self._obs.setdefault(kd, deque(maxlen=self.window))
        q.append((lg, surv, b, float(gap)))
        self._updates += 1
        if self.enabled and len(q) >= 40 and self._updates % 20 == 0:
            self._refit(kd)

    def _refit(self, kd: str) -> None:
        q = self._obs.get(kd)
        if not q or len(q) < 40:
            return
        Z = np.stack([x[0] for x in q], axis=0).astype(np.float64)
        bins = np.array([x[2] for x in q], dtype=np.int64)
        # Lightweight 1-D grid + local refinement over log-temperature (no autograd needed, <0.5 ms)
        base = float(self.base_temps.get(kd, self.base_temps.get("_all", 1.0)))
        best_t, best_l = base, math.inf
        for cand in np.geomspace(max(0.25, base * 0.4), min(4.0, base * 2.5), 25):
            z = Z / cand
            z -= z.max(axis=1, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=1))
            nll = float(np.mean(lse - z[np.arange(len(bins)), bins]))
            # Mild L2 prior toward offline-calibrated temperature so small windows stay stable
            reg = 0.02 * (math.log(cand / base) ** 2)
            if nll + reg < best_l:
                best_l, best_t = nll + reg, float(cand)
        self.temps[kd] = best_t

    def summary(self) -> dict:
        out = {"enabled": self.enabled, "temps": {k: round(v, 4) for k, v in self.temps.items()}, "by_kind": {}}
        for kd, q in self._obs.items():
            if not q:
                continue
            survs = np.stack([x[1] for x in q], axis=0)
            bins = np.array([x[2] for x in q], dtype=np.int64)
            gaps = np.array([x[3] for x in q], dtype=np.float64)
            # Reconstruct discrete bin probabilities from monotonic survival curve
            S1 = np.concatenate([np.ones((len(q), 1)), survs, np.zeros((len(q), 1))], axis=1)
            P = np.clip(S1[:, :-1] - S1[:, 1:], 1e-7, 1.0)
            nll = float(-np.mean(np.log(P[np.arange(len(bins)), bins])))
            j300 = int(np.argmin(np.abs(E2 - 300.0)))
            p300 = survs[:, j300]
            y300 = (gaps > 300.0).astype(np.float64)
            out["by_kind"][kd] = {
                "n": len(q),
                "nll": round(nll, 4),
                "cal_err_300s": round(float(abs(p300.mean() - y300.mean())), 4),
            }
        return out


class VllmAdapter:
    def __init__(self, hint_url):
        self.hint_url = hint_url

    def tag(self, body, key):
        kv = body.get("kv_transfer_params")
        body["kv_transfer_params"] = dict(kv or {}, park_key=key)

    async def send(self, http, items):
        payload = [{"key": k, "surv": [float(x) for x in s]} for k, s in items]
        async with http.post(self.hint_url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as r:
            await r.read()


class SglangAdapter(VllmAdapter):
    def tag(self, body, key):
        body["custom_params"] = dict(body.get("custom_params") or {}, park_key=key)


ADAPTERS = {"vllm": VllmAdapter, "sglang": SglangAdapter}


class Sidecar:
    def __init__(self, a):
        self.a = a
        self.tracker = Tracker()
        self.adapter = ADAPTERS[a.engine](a.hint_url)
        self.queue: asyncio.Queue = asyncio.Queue()
        self.scorer = None
        self.calib = None
        self.score_ms: deque = deque(maxlen=1000)
        self.stats = dict(requests=0, boundaries=0, hints=0, hint_errors=0, unlinked=0, batches=0)

    async def start(self, app):
        from layby.sidecar.scorer import Scorer          # heavy import: torch, transformers
        self.scorer = await asyncio.to_thread(
            Scorer,
            self.a.ckpt,
            self.a.calib,
            self.a.device,
            getattr(self.a, "max_len", None),
            getattr(self.a, "precision", None),
            getattr(self.a, "encoder", None),
        )
        self.calib = OnlineCalib(self.scorer.temp, enabled=bool(getattr(self.a, "online_calib", False)))
        self.http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
        app["scorer_task"] = asyncio.create_task(self.score_loop())

    async def stop(self, app):
        app["scorer_task"].cancel()
        await self.http.close()

    async def score_loop(self):
        coalesce_s = max(float(getattr(self.a, "coalesce_ms", 0.0)), 0.0) / 1000.0
        while True:
            batch = [await self.queue.get()]
            if coalesce_s > 0.0 and len(batch) < self.a.batch:
                deadline = time.monotonic() + coalesce_s
                while len(batch) < self.a.batch:
                    rem = deadline - time.monotonic()
                    if rem <= 0.0:
                        break
                    try:
                        batch.append(await asyncio.wait_for(self.queue.get(), timeout=rem))
                    except asyncio.TimeoutError:
                        break
            while not self.queue.empty() and len(batch) < self.a.batch:
                batch.append(self.queue.get_nowait())
            keys, states, kinds, futs = zip(*batch)
            try:
                t0 = time.perf_counter()
                temps = self.calib.temps if self.calib is not None else None
                Z, S = await asyncio.to_thread(self.scorer.score_with_logits, list(states), list(kinds), temps)
                self.score_ms.append((time.perf_counter() - t0) * 1000.0)
                self.stats["batches"] += 1
                if self.calib is not None:
                    for k, lg, sv, kd in zip(keys, Z, S, kinds):
                        self.calib.record_prediction(k, lg, sv, kd)
                await self.adapter.send(self.http, list(zip(keys, S.tolist())))
                self.stats["hints"] += len(batch)
                for f, row in zip(futs, S.tolist()):
                    if f is not None and not f.done():
                        f.set_result(row)
            except Exception as e:  # noqa: BLE001  (a lost hint leaves the engine on its default order)
                self.stats["hint_errors"] += len(batch)
                log.warning("hint batch failed: %r", e)
                for f in futs:
                    if f is not None and not f.done():
                        f.set_exception(e)

    @staticmethod
    def user_of(req, body):
        u = body.get("user")
        if u:
            return str(u)
        auth = req.headers.get("Authorization", "")
        return hashlib.blake2b(auth.encode(), digest_size=8).hexdigest() if auth else "anon"

    async def chat(self, req):
        body = await req.json()
        self.stats["requests"] += 1
        user = self.user_of(req, body)
        sess = self.tracker.on_request(body, user)
        if sess.last_returned is not None and self.calib is not None:
            self.calib.observe_return(*sess.last_returned)
        key = uuid.uuid4().hex
        fwd = dict(body)
        self.adapter.tag(fwd, key)
        stream = bool(body.get("stream"))
        if stream:
            fwd["stream_options"] = dict(body.get("stream_options") or {}, include_usage=True)
        headers = {k: v for k, v in req.headers.items() if k.lower() in ("authorization", "content-type")}
        async with self.http.post(self.a.upstream + req.path, json=fwd, headers=headers,
                                  timeout=aiohttp.ClientTimeout(total=None)) as up:
            if not stream:
                data = await up.read()
                resp = web.Response(body=data, status=up.status, content_type="application/json")
                if up.status == 200:
                    d = json.loads(data)
                    msg = (d.get("choices") or [{}])[0].get("message") or {}
                    self.boundary(sess, body, msg, d.get("usage"), user, key)
                return resp
            out = web.StreamResponse(status=up.status, headers={"Content-Type": "text/event-stream"})
            await out.prepare(req)
            acc = dict(content="", reasoning_content="", tool_calls={}, usage=None)
            buf = b""
            async for chunk in up.content.iter_any():
                await out.write(chunk)
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    self.parse_sse(line, acc)
            await out.write_eof()
            if up.status == 200:
                calls = [acc["tool_calls"][i] for i in sorted(acc["tool_calls"])]
                msg = {"content": acc["content"] or None, "reasoning_content": acc["reasoning_content"] or None,
                       "tool_calls": calls}
                self.boundary(sess, body, msg, acc["usage"], user, key)
            return out

    @staticmethod
    def parse_sse(line, acc):
        line = line.strip()
        if not line.startswith(b"data:") or line == b"data: [DONE]":
            return
        try:
            ev = json.loads(line[5:])
        except ValueError:
            return
        if ev.get("usage"):
            acc["usage"] = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            acc["content"] += d.get("content") or ""
            acc["reasoning_content"] += d.get("reasoning_content") or d.get("reasoning") or ""
            for tc in d.get("tool_calls") or []:
                c = acc["tool_calls"].setdefault(tc.get("index", 0), {"function": {"name": "", "arguments": ""}})
                f = tc.get("function") or {}
                c["function"]["name"] += f.get("name") or ""
                c["function"]["arguments"] += f.get("arguments") or ""

    def boundary(self, sess, body, msg, usage, user, key):
        state, kind = self.tracker.on_response(sess, body, msg, usage, user, key=key)
        self.stats["boundaries"] += 1
        self.queue.put_nowait((key, state, kind, None))

    async def score(self, req):
        """Score a state the client built (a replay's side file) under the client's key; answer with the curve
        after the adapter has it."""
        d = await req.json()
        self.stats["requests"] += 1
        fut = asyncio.get_running_loop().create_future()
        self.queue.put_nowait((str(d["key"]), d["state"], d.get("kind") or "human", fut))
        try:
            surv = await fut
        except Exception as e:  # noqa: BLE001
            return web.json_response({"key": d["key"], "error": repr(e)[:200]}, status=502)
        return web.json_response({"key": d["key"], "surv": surv})

    async def passthrough(self, req):
        data = await req.read()
        headers = {k: v for k, v in req.headers.items() if k.lower() in ("authorization", "content-type")}
        async with self.http.request(req.method, self.a.upstream + req.path_qs, data=data or None,
                                     headers=headers) as up:
            return web.Response(body=await up.read(), status=up.status,
                                content_type=up.content_type or "application/json")

    async def park_stats(self, req):
        ms = np.fromiter(self.score_ms, float) if self.score_ms else np.zeros(0)
        return web.json_response(dict(
            self.stats,
            sessions=self.tracker.n_sessions,
            active_tips=len(self.tracker.by_tip),
            tips_evicted=self.tracker.tips_evicted,
            queued=self.queue.qsize(),
            model_version=getattr(self.scorer, "model_version", None),
            device=str(getattr(self.scorer, "dev", "")),
            score_ms_p50=round(float(np.median(ms)), 2) if len(ms) else None,
            score_ms_p95=round(float(np.percentile(ms, 95)), 2) if len(ms) else None,
            online_calib=self.calib.summary() if self.calib is not None else None,
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--listen", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--engine", default="vllm", choices=sorted(ADAPTERS))
    ap.add_argument("--hint-url", default="http://127.0.0.1:8765/hint")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--encoder", default=None, help="optional local encoder dir for Layby-Dwell v2")
    ap.add_argument("--device")
    ap.add_argument("--precision", default=None, choices=["bf16", "fp32", "int8w", "fp8w"])
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32, help="states scored per forward pass at most")
    ap.add_argument("--coalesce-ms", type=float, default=0.0, help="optional micro-batch coalesce wait in ms")
    ap.add_argument("--online-calib", action="store_true", help="enable rolling online per-kind temperature calibration")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    sc = Sidecar(a)
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(sc.start)
    app.on_cleanup.append(sc.stop)
    app.router.add_post("/v1/chat/completions", sc.chat)
    app.router.add_get("/park/stats", sc.park_stats)
    app.router.add_post("/park/score", sc.score)
    app.router.add_route("*", "/{tail:.*}", sc.passthrough)
    web.run_app(app, host=a.host, port=a.listen)


if __name__ == "__main__":
    main()
