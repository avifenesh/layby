"""The Phase R hooks in the real runners (engine/run_arms.sh, engine/run_arms_sglang.sh), driven with stub engines:
SERVER_WRAP prefixes the server command (so the engine starts inside the arm's cgroup), ARM_HOOK writes replay_start
after the server is healthy and replay_end after the replay, PARK_HINTS reaches arm K's client, WRITE_POLICY reaches
SGLang's --hicache-write-policy. The stub server answers /health; the stub python records the replay's arguments and
writes one turn. CPU only; about 15 s per runner (the runners' own shutdown sleeps).

Run: python engine/phase_r/tests/test_runner_hooks.py
"""
import os
import socket
import subprocess
import sys
import tempfile
import unittest

PR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINE = os.path.dirname(PR)
sys.path.insert(0, PR)
import health  # noqa: E402
from common import read_text  # noqa: E402

SCRATCH = os.environ.get("LAYBY_TEST_DIR", os.path.expanduser("~/.cache/layby-phase-r-test"))

SERVER = """#!/usr/bin/env python3
import http.server, json, os, sys
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"server_argv": args, "wrapped": os.environ.get("WRAPPED")}) + "\\n")
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""
CLIENT = """#!/usr/bin/env python3
import json, os, sys
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"client_argv": sys.argv[1:]}) + "\\n")
out = sys.argv[3]   # replay.py WORKLOAD OUT.jsonl ...
with open(out, "w") as f:
    f.write(json.dumps({"session": 0, "turn": 0, "ttft": 0.1}) + "\\n")
"""


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class RunnerHooks(unittest.TestCase):
    def setUp(self):
        os.makedirs(SCRATCH, exist_ok=True)
        self.d = tempfile.mkdtemp(dir=SCRATCH)
        self.venv = os.path.join(self.d, "venv")
        os.makedirs(os.path.join(self.venv, "bin"))
        for name, body in (("vllm", SERVER), ("sglang", SERVER), ("lmcache", SERVER), ("python", CLIENT)):
            p = os.path.join(self.venv, "bin", name)
            with open(p, "w") as f:
                f.write(body)
            os.chmod(p, 0o755)
        self.log = os.path.join(self.d, "stub.jsonl")
        self.out = os.path.join(self.d, "out")
        os.makedirs(self.out)
        self.wl = os.path.join(self.d, "w.json")
        with open(self.wl, "w") as f:
            f.write('{"sessions": []}')

    def run_runner(self, script, arm, extra_env):
        env = dict(os.environ, VENV=self.venv, ENGINE_DIR=ENGINE, REPO=os.path.dirname(ENGINE), STUB_LOG=self.log,
                   PORT=str(free_port()), PARK_PORT=str(free_port()), BOOT_WAIT="12",
                   SERVER_WRAP="env WRAPPED=yes", ARM_HOOK=os.path.join(PR, "arm_hook.sh"), **extra_env)
        r = subprocess.run(["bash", os.path.join(ENGINE, script), self.wl, self.out, str(1 << 30), str(2 << 30),
                            os.path.join(self.d, "disk"), arm], env=env, capture_output=True, text=True, timeout=180)
        import json
        rows = [json.loads(x) for x in read_text(self.log).splitlines()]
        return r, rows

    def check(self, r, rows, arm):
        self.assertIn("ALL_DONE", r.stdout, r.stdout + r.stderr)
        self.assertIn(f"arm {arm} rc=0 1 turns", r.stdout)
        server = [x for x in rows if "server_argv" in x and x["server_argv"][0] == "serve"]
        self.assertEqual(len(server), 1)
        self.assertEqual(server[0]["wrapped"], "yes")            # the server ran under SERVER_WRAP
        m = health.read_markers(os.path.join(self.out, f"{arm}.markers"))
        self.assertEqual(sorted(m), ["replay_end", "replay_start"])
        self.assertLessEqual(m["replay_start"], m["replay_end"])
        return server[0]["server_argv"], [x for x in rows if "client_argv" in x][0]["client_argv"]

    def test_vllm_runner(self):
        r, rows = self.run_runner("run_arms.sh", "K", {"PARK_HINTS": "ewma"})
        argv, client = self.check(r, rows, "K")
        self.assertEqual(argv[0], "serve")
        self.assertIn("--park-hints", client)
        self.assertEqual(client[client.index("--park-hints") + 1], "ewma")

    def test_lmcache_server_joins_the_arms_cgroup(self):
        # arm LM starts an LMCache server beside vLLM; it holds the L1 tier, so it must be charged to the arm's cgroup
        r, rows = self.run_runner("run_arms.sh", "LM", {"LM_PORT": str(free_port())})
        self.check(r, rows, "LM")
        lm = [x for x in rows if "server_argv" in x and x["server_argv"][0] == "server"]
        self.assertEqual(len(lm), 1)
        self.assertEqual(lm[0]["wrapped"], "yes")

    def test_sglang_runner(self):
        r, rows = self.run_runner("run_arms_sglang.sh", "T", {"WRITE_POLICY": "write_through_selective"})
        argv, _ = self.check(r, rows, "T")
        self.assertEqual(argv[argv.index("--hicache-write-policy") + 1], "write_through_selective")
        self.assertEqual(argv[argv.index("--hicache-storage-backend") + 1], "file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
