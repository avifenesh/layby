#!/usr/bin/env python3
"""park sidecar: an OpenAI-compatible proxy in front of an engine that gives the engine's park adapter
the idle-period curve of every finished response.

Per request: link it to its session (layby.sidecar.session), tag it with a park_key the engine adapter
sees, and forward it, streaming or not, while collecting the response (text, reasoning, tool calls,
usage). When the response ends: build the v6 server-view state, score it with Laya (batched, on a
background task), and send {key, surv} to the engine adapter.

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
import hashlib
import json
import logging
import uuid

import aiohttp
from aiohttp import web

from layby.sidecar.session import Tracker

log = logging.getLogger("layby.sidecar")


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
        self.stats = dict(requests=0, boundaries=0, hints=0, hint_errors=0, unlinked=0)

    async def start(self, app):
        from layby.sidecar.scorer import Scorer          # heavy import: torch, transformers
        self.scorer = await asyncio.to_thread(Scorer, self.a.ckpt, self.a.calib, self.a.device)
        self.http = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0))
        app["scorer_task"] = asyncio.create_task(self.score_loop())

    async def stop(self, app):
        app["scorer_task"].cancel()
        await self.http.close()

    async def score_loop(self):
        while True:
            batch = [await self.queue.get()]
            while not self.queue.empty() and len(batch) < self.a.batch:
                batch.append(self.queue.get_nowait())
            keys, states, kinds = zip(*batch)
            try:
                S = await asyncio.to_thread(self.scorer.score, list(states), list(kinds))
                await self.adapter.send(self.http, list(zip(keys, S.tolist())))
                self.stats["hints"] += len(batch)
            except Exception as e:  # noqa: BLE001  (a lost hint leaves the engine on its default order)
                self.stats["hint_errors"] += len(batch)
                log.warning("hint batch failed: %r", e)

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
        state, kind = self.tracker.on_response(sess, body, msg, usage, user)
        self.stats["boundaries"] += 1
        self.queue.put_nowait((key, state, kind))

    async def passthrough(self, req):
        data = await req.read()
        headers = {k: v for k, v in req.headers.items() if k.lower() in ("authorization", "content-type")}
        async with self.http.request(req.method, self.a.upstream + req.path_qs, data=data or None,
                                     headers=headers) as up:
            return web.Response(body=await up.read(), status=up.status,
                                content_type=up.content_type or "application/json")

    async def park_stats(self, req):
        return web.json_response(dict(self.stats, sessions=self.tracker.n_sessions, queued=self.queue.qsize()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--listen", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--engine", default="vllm", choices=sorted(ADAPTERS))
    ap.add_argument("--hint-url", default="http://127.0.0.1:8765/hint")
    ap.add_argument("--ckpt", required=True); ap.add_argument("--calib", required=True)
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int, default=32, help="states scored per forward pass at most")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    sc = Sidecar(a)
    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(sc.start)
    app.on_cleanup.append(sc.stop)
    app.router.add_post("/v1/chat/completions", sc.chat)
    app.router.add_get("/park/stats", sc.park_stats)
    app.router.add_route("*", "/{tail:.*}", sc.passthrough)
    web.run_app(app, host=a.host, port=a.listen)


if __name__ == "__main__":
    main()
