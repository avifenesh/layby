"""Local HTTP port of the park adapter, served from a daemon thread in the engine's scheduler process.

  POST /hint  {"key": str, "surv": [15 floats]}   the idle-period curve of a finished (or running)
              request; queued and applied by the scheduler thread at its next step
  GET  /live  telemetry snapshot (rates, measured speeds, residency, decision counts) as JSON
  GET  /log   every placement decision with its cost terms and every admission with its prompt sources
  GET  /health

The engine adapter passes its state (layby.vllm.state.STATE by default; park.sglang.state.STATE for
SGLang): the server only needs its hint queue and snapshot().
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from layby.vllm.state import STATE


class _Handler(BaseHTTPRequestHandler):
    state = STATE

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"ok": True})
        if self.path.startswith("/prof"):
            from layby import prof
            if "windows" in self.path:
                return self._send(200, prof.windows())
            return self._send(200, prof.report())
        if self.path == "/log":
            return self._send(200, dict(decisions=list(self.state.dlog), returns=list(self.state.rlog)))
        if self.path == "/live":
            # read-only and approximate: the scheduler thread may be updating the estimators
            try:
                return self._send(200, self.state.snapshot())
            except Exception as e:   # noqa: BLE001  (a racing update; ask again)
                return self._send(503, {"error": repr(e)})
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/hint":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length") or 0)
            d = json.loads(self.rfile.read(n) or b"{}")
            items = d if isinstance(d, list) else [d]
            for it in items:
                surv = [float(x) for x in it["surv"]]
                if len(surv) != 15:
                    raise ValueError("surv needs 15 values (P(T > E2[j]))")
                self.state.hints.put((str(it["key"]), surv))
        except (KeyError, ValueError, TypeError) as e:
            return self._send(400, {"error": repr(e)})
        self._send(200, {"queued": len(items)})


def start(port: int, host: str = "127.0.0.1", state=None):
    handler = _Handler if state is None else type("_StateHandler", (_Handler,), {"state": state})
    srv = ThreadingHTTPServer((host, port), handler)
    threading.Thread(target=srv.serve_forever, name="park-hints", daemon=True).start()
    return srv
