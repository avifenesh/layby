"""The orchestrator end to end in dry-run on the example cells: every command printed, none executed, exit codes for
a clean block, a regime mismatch, a host fault (GPU throttle, fio drift) and A/A mode; the two-engine GLM cell in
lockstep rounds.

Run: python engine/phase_r/tests/test_dry_run.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

PR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PR)
import orchestrate  # noqa: E402
from common import load_json  # noqa: E402

EX = os.path.join(PR, "examples")
SCRATCH = os.environ.get("LAYBY_TEST_DIR", os.path.expanduser("~/.cache/layby-phase-r-test"))


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = orchestrate.main(argv)
    return rc, buf.getvalue()


class DryRun(unittest.TestCase):
    def setUp(self):
        os.makedirs(SCRATCH, exist_ok=True)
        self.out = tempfile.mkdtemp(dir=SCRATCH)
        self.tmp = tempfile.mkdtemp(dir=SCRATCH)

    def tearDown(self):
        shutil.rmtree(self.out, ignore_errors=True)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def cell(self, name, **patch):
        with open(os.path.join(EX, name)) as f:
            c = json.load(f)
        c["pilot"] = os.path.join(EX, c["pilot"])
        for k, v in patch.items():
            c[k] = v
        p = os.path.join(self.tmp, name)
        with open(p, "w") as f:
            json.dump(c, f)
        return p

    def runs(self):
        with open(os.path.join(self.out, "runs.jsonl")) as f:
            return [json.loads(x) for x in f]

    def test_clean_block_runs_every_arm_once_and_executes_nothing(self):
        rc, out = run(["run", "--cell", self.cell("cell_main_vllm_slow_heavy.json"), "--host", "h1", "--repeat", "2",
                       "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK, out[-2000:])
        r = self.runs()
        self.assertEqual(sorted(x["label"] for x in r), sorted(["C0", "T", "LM", "K", "K_ewma", "K_km"]))
        self.assertTrue(all(x["valid"] and x["dry_run"] for x in r))
        lines = out.splitlines()
        self.assertTrue(all(ln.startswith("DRY ") for ln in lines if ln.strip()), [ln for ln in lines if not ln.startswith("DRY")][:3])
        # the steps the plan requires, in every round
        n = len(r)
        self.assertEqual(sum("echo 3 > /proc/sys/vm/drop_caches" in ln for ln in lines), 2 * n)
        self.assertEqual(sum("fstrim -v /mnt/layby/A" in ln for ln in lines), n)
        self.assertEqual(sum("cgroup.kill" in ln for ln in lines), n + 1)        # n arms + the regime probe cgroup
        self.assertEqual(sum("/io.max" in ln for ln in lines), n + 1)
        self.assertEqual(sum("bash /root/layby/engine/run_arms.sh" in ln for ln in lines), n)
        self.assertTrue(any("SERVER_WRAP=/root/layby/engine/phase_r/cg_exec.sh /sys/fs/cgroup/layby/" in ln for ln in lines))
        self.assertTrue(any("PARK_HINTS=ewma" in ln for ln in lines))
        # the order is the seeded one, and positions are logged
        b = load_json(os.path.join(self.out, "main-vllm-slow-heavy", "h1", "r2", "a1", "block.json"))
        import order
        self.assertEqual([r[0][1] for r in b["rounds"]],
                         order.arm_order(["C0", "T", "LM", "K", "K_ewma", "K_km"], 20261010, "main-vllm-slow-heavy", 2))
        self.assertEqual(b["regime"]["A"]["regime"], "slow")
        self.assertEqual(b["next"], "done")
        # root and GPU actions are tagged so a reader of commands.log sees what needs privileges
        log = open(os.path.join(self.out, "main-vllm-slow-heavy", "h1", "r2", "commands.log")).read()
        self.assertIn("[root] echo 3 > /proc/sys/vm/drop_caches", log)
        self.assertIn("[gpu] docker exec", log)

    def test_memory_max_uses_each_arms_own_rss(self):
        rc, out = run(["run", "--cell", self.cell("cell_main_vllm_slow_heavy.json"), "--host", "h1", "--out", self.out,
                       "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK)
        got = {}
        for ln in out.splitlines():
            if ln.endswith("/memory.max") and ".r1a1.A." in ln:
                w = ln.split()
                got[ln.rsplit("/", 2)[-2].rsplit(".", 1)[-1]] = int(w[w.index("echo") + 1])
        G, M = 1 << 30, 1 << 20
        pinned, slack = 28 * G, 256 * M
        self.assertEqual(got, {"C0": pinned + 12 * G + slack, "T": pinned + 12 * G + 512 * M + slack,
                               "LM": pinned + (11 + 3) * G + slack,       # vLLM engine plus its LMCache server
                               "K": pinned + 13 * G + slack, "K_ewma": pinned + 13 * G + slack,
                               "K_km": pinned + 13 * G + slack})
        lm = [x for x in self.runs() if x["label"] == "LM"][0]
        env = load_json(os.path.join(lm["dir"], lm["prefix"] + ".harness_env.json"))
        self.assertEqual(env["rss"], {"vllm": 11 * G, "lmcache_server": 3 * G})

    def test_missing_arm_rss_is_a_config_error(self):
        pil = load_json(os.path.join(EX, "pilot_qwen3_8b_vllm.json"))
        del pil["arm_rss_bytes"]["K_km"]
        pp = os.path.join(self.tmp, "pilot.json")
        with open(pp, "w") as f:
            json.dump(pil, f)
        p = self.cell("cell_main_vllm_slow_heavy.json")
        c = load_json(p)
        c["pilot"] = pp
        with open(p, "w") as f:
            json.dump(c, f)
        rc, _ = run(["run", "--cell", p, "--host", "h1", "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_CONFIG)
        self.assertFalse(os.path.exists(os.path.join(self.out, "runs.jsonl")))
        # an A/A block runs only C0's two copies, which use C0's entry
        rc, out = run(["run", "--cell", p, "--aa", "C0", "--host", "h1", "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK)
        self.assertIn(f"echo {28 * (1 << 30) + 12 * (1 << 30) + (256 << 20)} > /sys/fs/cgroup/layby/"
                      "main-vllm-slow-heavy-AA.r1a1.A.C0_aa1/memory.max", out)

    def test_regime_mismatch_stops_before_any_arm(self):
        # without io.max the NVMe reads 3.6 GB/s: medium, not slow
        p = self.cell("cell_main_vllm_slow_heavy.json", io_max=None, io_max_none_reason="test")
        rc, out = run(["run", "--cell", p, "--host", "h1", "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_CONFIG)
        self.assertFalse(os.path.exists(os.path.join(self.out, "runs.jsonl")))
        self.assertNotIn("run_arms.sh", out)

    def test_gpu_throttle_is_a_host_fault(self):
        c = load_json(os.path.join(EX, "cell_main_vllm_slow_heavy.json"))
        dr = dict(c["dry_run"], gpu_mask="0x40")
        rc, out = run(["run", "--cell", self.cell("cell_main_vllm_slow_heavy.json", dry_run=dr), "--host", "h1",
                       "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_HOST_FAULT)
        self.assertNotIn("run_arms.sh", out)   # the first arm's health check stops the block
        self.assertEqual([x["status"] for x in self.runs()], ["health_failed"])

    def test_fio_drift_from_first_probe_is_a_host_fault(self):
        hosts = os.path.join(self.out, "hosts", "h1")
        os.makedirs(hosts)
        json.dump({"seq_read_Bps": 4.5e9}, open(os.path.join(hosts, "first_probe.A.json"), "w"))   # now 3.6e9: -20%
        rc, _ = run(["run", "--cell", self.cell("cell_main_vllm_slow_heavy.json"), "--host", "h1", "--out", self.out,
                     "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_HOST_FAULT)
        h = [json.loads(x) for x in open(os.path.join(self.out, "runs.jsonl"))]
        self.assertEqual(h[0]["status"], "health_failed")

    def test_aa_mode(self):
        rc, _ = run(["run", "--cell", self.cell("cell_main_vllm_slow_heavy.json"), "--aa", "C0", "--host", "h1",
                     "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK)
        r = self.runs()
        self.assertEqual(sorted(x["label"] for x in r), ["C0_aa1", "C0_aa2"])
        self.assertTrue(all(x["code"] == "C0" and x["cell"] == "main-vllm-slow-heavy-AA" for x in r))

    def test_two_engine_node_runs_lockstep_rounds(self):
        rc, out = run(["run", "--cell", self.cell("cell_glm_slow_heavy_2slot.json"), "--host", "n1", "--out",
                       self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK)
        r = self.runs()
        self.assertEqual(sorted(x["label"] for x in r), ["C0", "K", "K_ewma", "T"])
        self.assertEqual({x["slot"] for x in r}, {"A", "B"})
        b = load_json(os.path.join(self.out, "large-glm-slow-heavy", "n1", "r1", "a1", "block.json"))
        self.assertEqual(len(b["rounds"]), 2)
        self.assertTrue(all(len(rd) == 2 for rd in b["rounds"]))
        self.assertIn("CUDA_VISIBLE_DEVICES=4,5,6,7", out)
        self.assertIn("cpuset.mems", out)

    def test_ram_tier_sglang_cell(self):
        rc, out = run(["run", "--cell", self.cell("cell_main_sglang_fast_heavy.json"), "--host", "h1", "--out",
                       self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK, out[-1500:])
        self.assertIn("mount -t tmpfs -o size=68719476736,mode=0755 layby_ram_tier /mnt/layby/ram", out)
        self.assertNotIn("/io.max", out)
        self.assertIn("apply_patch.sh /root/.venv-sglang/bin/python check", out)
        self.assertIn("WRITE_POLICY=write_through_selective", out)
        # memory.max carries the RAM tier: 28 GiB pinned + 12 GiB RSS + 256 MiB + 64 GiB
        self.assertIn(f"echo {(28 + 12 + 64) * (1 << 30) + (256 << 20)} > /sys/fs/cgroup/layby/", out)

    def test_plan_command(self):
        rc, out = run(["plan", "--cell", self.cell("cell_glm_slow_heavy_2slot.json"), "--repeat", "2"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.strip().splitlines()), 2)

    def test_bad_spec(self):
        p = self.cell("cell_main_vllm_slow_heavy.json", arms=[{"label": "K.x"}])
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)
        p = self.cell("cell_main_vllm_slow_heavy.json", io_max=None)
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)


if __name__ == "__main__":
    unittest.main(verbosity=2)
