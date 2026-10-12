"""The Phase R hooks in the real runners (engine/run_arms.sh, engine/run_arms_sglang.sh), driven with stub engines:
SERVER_WRAP prefixes the server command (so the engine starts inside the arm's cgroup), ARM_HOOK writes replay_start
after the server is healthy and replay_end after the replay, PARK_HINTS reaches arm K's client, WRITE_POLICY reaches
SGLang's --hicache-write-policy; the baseline and control arms (T2, CT, PT, K_km, K_ewma) get their configurations.
Arm K starts the live-scoring sidecar under SERVER_WRAP and passes the side file to the replay, and is refused
(exit 4) without one; a server (or LMCache server) that dies during the replay makes the arm's rc non-zero; LMCache's
L1 and L2 sizes are the exact CPU and disk bytes. The stub server answers /health (GET /die ends it); the stub python
records the replay's arguments and writes one turn, or, run as the sidecar, answers /park/stats. CPU only; about 15 s
per runner (the runners' own shutdown sleeps).

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
    f.write(json.dumps({"server_argv": args, "wrapped": os.environ.get("WRAPPED"),
                        "max_size": os.environ.get("SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE")}) + "\\n")
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        if self.path == "/die":
            os._exit(0)
    def log_message(self, *a):
        pass
http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
"""
CLIENT = """#!/usr/bin/env python3
import json, os, sys, urllib.request
if sys.argv[1:3] == ["-m", "layby.sidecar.proxy"]:   # the live-scoring sidecar: a stub server on --listen
    with open(os.environ["STUB_LOG"], "a") as f:
        f.write(json.dumps({"sidecar_argv": sys.argv[3:], "wrapped": os.environ.get("WRAPPED")}) + "\\n")
    sys.argv[1:] = ["sidecar", "--port", sys.argv[sys.argv.index("--listen") + 1]]
    exec(open(os.path.join(os.path.dirname(__file__), "vllm")).read())
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps({"client_argv": sys.argv[1:]}) + "\\n")
for port in filter(None, os.environ.get("STUB_DIE", "").split(",")):   # a server dies during the replay
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/die", timeout=5).read()
    except Exception:
        pass
out = sys.argv[3]   # replay.py WORKLOAD OUT.jsonl ...
with open(out, "w") as f:
    f.write(json.dumps({"session": 0, "turn": 0, "ttft": 0.1}) + "\\n")
"""


_given = set()


def free_port():
    # each call closes its socket, so the kernel can hand the same port out twice; LMCache also binds LM_PORT + 100
    while True:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        if not _given & {p, p - 100, p + 100}:
            _given.add(p)
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

    def run_runner(self, script, arm, extra_env, cpu=2 << 30):
        env = dict(os.environ, VENV=self.venv, ENGINE_DIR=ENGINE, REPO=os.path.dirname(ENGINE), STUB_LOG=self.log,
                   PORT=str(free_port()), PARK_PORT=str(free_port()), BOOT_WAIT="12", DISK_BYTES=str(5 << 30),
                   SERVER_WRAP="env WRAPPED=yes", ARM_HOOK=os.path.join(PR, "arm_hook.sh"))
        env.update(extra_env)
        env = {k: v for k, v in env.items() if v is not None}            # None: unset
        r = subprocess.run(["bash", os.path.join(ENGINE, script), self.wl, self.out, str(1 << 30), str(cpu),
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

    def live(self):
        """The live-scoring inputs of arm K: a side file and the Layby-Dwell release (stub files)."""
        env = {}
        for k, name in (("SIDE_FILE", "side.jsonl"), ("SIDECAR_CKPT", "model.safetensors"), ("SIDECAR_CALIB", "config.json")):
            env[k] = os.path.join(self.d, name)
            open(env[k], "w").close()
        return env

    def test_vllm_runner(self):
        r, rows = self.run_runner("run_arms.sh", "K", dict(self.live(), PARK_HINTS="ewma"))
        argv, client = self.check(r, rows, "K")
        self.assertEqual(argv[0], "serve")
        self.assertIn("--park-hints", client)
        self.assertEqual(client[client.index("--park-hints") + 1], "ewma")

    def test_k_scores_live_through_the_sidecar_in_the_arms_cgroup(self):
        # arm K starts layby.sidecar.proxy under SERVER_WRAP (the arm's cgroup) and the replay sends it the side file
        for script, engine in (("run_arms.sh", "vllm"), ("run_arms_sglang.sh", "sglang")):
            open(self.log, "w").close()
            env = self.live()
            sport = free_port()
            r, rows = self.run_runner(script, "K", dict(env, SIDECAR_PORT=str(sport)))
            _, client = self.check(r, rows, "K")
            sc = [x for x in rows if "sidecar_argv" in x]
            self.assertEqual(len(sc), 1, r.stdout)
            self.assertEqual(sc[0]["wrapped"], "yes")
            a = sc[0]["sidecar_argv"]
            self.assertEqual((a[a.index("--engine") + 1], a[a.index("--ckpt") + 1], a[a.index("--listen") + 1]),
                             (engine, env["SIDECAR_CKPT"], str(sport)))
            self.assertEqual(client[client.index("--side-file") + 1], env["SIDE_FILE"])
            self.assertEqual(client[client.index("--sidecar") + 1], f"http://127.0.0.1:{sport}")
        # the controls post their precomputed curves: no sidecar, no side file
        open(self.log, "w").close()
        r, rows = self.run_runner("run_arms.sh", "K_ewma", {})
        _, client = self.check(r, rows, "K_ewma")
        self.assertNotIn("--side-file", client)
        self.assertFalse([x for x in rows if "sidecar_argv" in x])

    def test_k_without_side_file_is_refused_before_any_arm(self):
        for script in ("run_arms.sh", "run_arms_sglang.sh"):
            open(self.log, "w").close()
            r, rows = self.run_runner(script, ["C0", "K"], {})
            self.assertEqual(r.returncode, 4, r.stdout)
            self.assertIn("needs SIDE_FILE", r.stdout)
            self.assertFalse(rows)                                     # nothing started, not even C0
            env = self.live()
            os.remove(env["SIDECAR_CALIB"])
            r, _ = self.run_runner(script, "K", env)
            self.assertEqual(r.returncode, 4)
            r, _ = self.run_runner(script, "T", {"DISK_BYTES": None})   # a disk arm needs its capacity
            self.assertEqual(r.returncode, 4)
            self.assertIn("needs DISK_BYTES", r.stdout)

    def test_server_death_during_the_replay_fails_the_arm(self):
        # the engine (or LMCache's server) exits while the replay runs: the runner must not report rc=0
        port = free_port()
        r, rows = self.run_runner("run_arms.sh", "C0", {"PORT": str(port), "STUB_DIE": str(port)})
        self.assertIn("SERVER DIED during the replay", r.stdout)
        self.assertIn("arm C0 rc=86", r.stdout)
        self.assertEqual(sorted(health.read_markers(os.path.join(self.out, "C0.markers"))), ["replay_end", "replay_start"])
        lm = free_port()
        r, _ = self.run_runner("run_arms.sh", "LM", {"LM_PORT": str(lm), "STUB_DIE": str(lm)})
        self.assertIn("LMCache server process", r.stdout)
        self.assertIn("arm LM rc=86", r.stdout)
        port = free_port()
        r, _ = self.run_runner("run_arms_sglang.sh", "C0", {"PORT": str(port), "STUB_DIE": str(port)})
        self.assertIn("arm C0 rc=86", r.stdout)
        r, _ = self.run_runner("run_arms.sh", "C0", {})                  # a healthy server: rc 0
        self.assertIn("arm C0 rc=0", r.stdout)

    def test_lmcache_gets_the_exact_cpu_and_disk_bytes(self):
        # 1.5 GiB of CPU tier: --l1-size-gb used to floor to whole GiB (1), so LM ran with less CPU tier than the
        # other arms; LMCache sizes int(gb * 2^30), which must give back the same bytes
        import json
        cpu, disk = 3 << 29, (7 << 30) + (5 << 20)
        r, rows = self.run_runner("run_arms.sh", "LM", {"LM_PORT": str(free_port()), "DISK_BYTES": str(disk)}, cpu=cpu)
        self.check(r, rows, "LM")
        a = [x for x in rows if "server_argv" in x and x["server_argv"][0] == "server"][0]["server_argv"]
        self.assertEqual(int(float(a[a.index("--l1-size-gb") + 1]) * (1 << 30)), cpu)
        l2 = json.loads(a[a.index("--l2-adapter") + 1])
        self.assertEqual(int(float(l2["backend_params"]["max_capacity_gb"]) * 1024 ** 3), disk)
        self.assertEqual(open(os.path.join(self.out, "LM.tier_bytes")).read().strip(), "0")   # the tier's du
        r, _ = self.run_runner("run_arms.sh", "C0", {"OFF_EXTRA": ',"cpu_bytes_to_use":1'})
        self.assertEqual(r.returncode, 4)

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
        self.assertEqual([x["max_size"] for x in rows if "server_argv" in x], [str(5 << 30)])   # disk tier cap
        self.assertEqual(argv[argv.index("--hicache-write-policy") + 1], "write_through_selective")
        self.assertEqual(argv[argv.index("--hicache-storage-backend") + 1], "file")


if __name__ == "__main__":
    unittest.main(verbosity=2)
