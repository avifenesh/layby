"""The Phase R hooks in the real runners (engine/run_arms.sh, engine/run_arms_sglang.sh), driven with stub engines:
SERVER_WRAP prefixes the server command (so the engine starts inside the arm's cgroup), ARM_HOOK writes replay_start
after the server is healthy and replay_end after the replay, PARK_HINTS reaches arm K's client, WRITE_POLICY reaches
SGLang's --hicache-write-policy; the baseline and control arms (T2, CT, PT, K_km, K_ewma) get their configurations. The stub server answers /health; the stub python records the replay's arguments and
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

    def tearDown(self):
        import shutil
        shutil.rmtree(self.d, ignore_errors=True)

    def run_runner(self, script, arm, extra_env):
        env = dict(os.environ, VENV=self.venv, ENGINE_DIR=ENGINE, REPO=os.path.dirname(ENGINE), STUB_LOG=self.log,
                   PORT=str(free_port()), PARK_PORT=str(free_port()), BOOT_WAIT="12",
                   SERVER_WRAP="env WRAPPED=yes", ARM_HOOK=os.path.join(PR, "arm_hook.sh"), **extra_env)
        r = subprocess.run(["bash", os.path.join(ENGINE, script), self.wl, self.out, str(1 << 30), str(2 << 30),
                            os.path.join(self.d, "disk")] + ([arm] if isinstance(arm, str) else list(arm)), env=env,
                           capture_output=True, text=True, timeout=180 * (1 if isinstance(arm, str) else len(arm)))
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

    def test_vllm_baseline_and_control_arms(self):
        # T2, CT, PT, K_km in one runner call: each arm's kv-transfer-config and client flags, with the plan's
        # fairness settings (same CPU bytes, offload_prompt_only false, every disk tier on vLLM's fs tier class)
        import json
        r, _ = self.run_runner("run_arms.sh", "T2", {})
        self.assertIn("ALL_DONE", r.stdout, r.stdout + r.stderr)
        r, rows = self.run_runner("run_arms.sh", ["T2", "CT", "PT", "K_km", "K_ewma"],
                                  {"T2_MIN_HITS": "3", "T2_COUNT": "store", "CT_ETA": "0.83", "EVICTION": "arc",
                                   "FS_THREADS": "32"})
        self.assertIn("ALL_DONE", r.stdout, r.stdout + r.stderr)
        servers = [x["server_argv"] for x in rows if "server_argv" in x]
        clients = [x["client_argv"] for x in rows if "client_argv" in x]
        self.assertEqual(len(servers), 6)
        self.assertEqual(len(clients), 6)
        cfg = {}
        for argv, cl, arm in zip(servers, clients, ["T2", "T2", "CT", "PT", "K_km", "K_ewma"]):
            c = json.loads(argv[argv.index("--kv-transfer-config") + 1])
            x = c["kv_connector_extra_config"]
            self.assertEqual(x["cpu_bytes_to_use"], 2 << 30)
            self.assertIs(x["offload_prompt_only"], False)
            for t in x.get("secondary_tiers", []):
                self.assertEqual(t["root_dir"], os.path.join(self.d, "disk"))
            cfg.setdefault(arm, []).append((c, x, cl))
        (c, x, cl), (c3, x3, _) = cfg["T2"]
        self.assertEqual(c["kv_connector"], "OffloadingConnector")
        self.assertEqual(x["eviction_policy"], "lru")
        self.assertEqual(x["secondary_tiers"], [{"type": "FreqFsTier", "module_path": "layby.vllm.freq_tier",
                                                 "root_dir": os.path.join(self.d, "disk"), "n_read_threads": 16,
                                                 "n_write_threads": 16, "min_hits": 2, "decay_s": 600,
                                                 "count": "request"}])
        t3 = x3["secondary_tiers"][0]
        self.assertEqual((t3["min_hits"], t3["count"], t3["n_read_threads"], x3["eviction_policy"]), (3, "store", 32, "arc"))
        c, x, cl = cfg["CT"][0]
        self.assertEqual((c["kv_connector"], c["kv_connector_module_path"]), ("ContinuumConnector", "layby.vllm.continuum"))
        self.assertNotIn("secondary_tiers", x)
        self.assertEqual((x["ct_eta"], x["eviction_policy"]), (0.83, "arc"))
        self.assertIn("--continuum", cl)
        self.assertNotIn("--park-hints", cl)
        c, x, cl = cfg["PT"][0]
        self.assertEqual(c["kv_connector"], "OffloadingConnector")
        self.assertEqual(x["secondary_tiers"][0]["type"], "ParkFsTier")
        self.assertIs(x["secondary_tiers"][0]["write_all"], True)
        self.assertNotIn("--park-hints", cl)
        for arm, name in (("K_km", "km"), ("K_ewma", "ewma")):
            c, x, cl = cfg[arm][0]
            self.assertEqual(c["kv_connector"], "ParkConnector")
            self.assertEqual(cl[cl.index("--park-hints") + 1], name)
        # CT with no eta is refused (its TTL needs the workload's memoryfulness), and the replay timeout must
        # leave the hard cap time to write its censored rows
        r, _ = self.run_runner("run_arms.sh", "CT", {})
        self.assertIn("arm CT: CT_ETA unset", r.stdout)
        r, _ = self.run_runner("run_arms.sh", "C0", {"HARD_CAP": "1000", "REPLAY_TIMEOUT": "1200"})
        self.assertEqual(r.returncode, 2)
        self.assertIn("must exceed HARD_CAP", r.stdout)

    def test_sglang_runner(self):
        r, rows = self.run_runner("run_arms_sglang.sh", "T", {"WRITE_POLICY": "write_through_selective"})
        argv, _ = self.check(r, rows, "T")
        self.assertEqual(argv[argv.index("--hicache-write-policy") + 1], "write_through_selective")
        self.assertEqual(argv[argv.index("--hicache-storage-backend") + 1], "file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
