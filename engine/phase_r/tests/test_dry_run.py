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
        arms = ["C0", "T", "T2", "LM", "CT", "K", "K_ewma", "K_km"]          # the plan's headline cell set
        self.assertEqual(sorted(x["label"] for x in r), sorted(arms))
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
        cmd = {ln.split(" > ")[0].split()[-1]: ln for ln in lines if "bash /root/layby/engine/run_arms.sh" in ln}
        self.assertEqual(sorted(cmd), sorted(arms))
        self.assertIn("CT_ETA=0.83", cmd["CT"])
        self.assertIn("T2_MIN_HITS=2", cmd["T2"])
        self.assertTrue(all("HARD_CAP=9000" in ln and "REPLAY_TIMEOUT=10800" in ln for ln in cmd.values()))
        # the order is the seeded one, and positions are logged
        b = load_json(os.path.join(self.out, "main-vllm-slow-heavy", "h1", "r2", "a1", "block.json"))
        import order
        self.assertEqual([r[0][1] for r in b["rounds"]],
                         order.arm_order(arms, 20261010, "main-vllm-slow-heavy", 2))
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
                               "T2": pinned + 12 * G + 512 * M + slack, "CT": pinned + 12 * G + slack,
                               "LM": pinned + (11 + 3) * G + slack,       # vLLM engine plus its LMCache server
                               "K": pinned + (13 + 2) * G + slack,        # engine plus the live-scoring sidecar
                               "K_ewma": pinned + 13 * G + slack,
                               "K_km": pinned + 13 * G + slack})
        lm = [x for x in self.runs() if x["label"] == "LM"][0]
        env = load_json(os.path.join(lm["dir"], lm["prefix"] + ".harness_env.json"))
        self.assertEqual(env["rss"], {"vllm": 11 * G, "lmcache_server": 3 * G})
        k = [x for x in self.runs() if x["label"] == "K"][0]
        self.assertEqual(load_json(os.path.join(k["dir"], k["prefix"] + ".harness_env.json"))["rss"],
                         {"engine": 13 * G, "sidecar": 2 * G})

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
        self.assertEqual(sorted(x["label"] for x in r if x["status"] == "ran"), ["C0", "K", "K_ewma", "T", "T2"])
        self.assertEqual({x["slot"] for x in r}, {"A", "B"})
        b = load_json(os.path.join(self.out, "large-glm-slow-heavy", "n1", "r1", "a1", "block.json"))
        self.assertEqual([len(rd) for rd in b["rounds"]], [2, 2, 2])       # 5 arms over 2 slots, plus a filler
        # the filler: C0's runner in the empty slot, unscored (not a verdict), indexed as such
        fill = [x for x in r if x["status"] == "filler"]
        self.assertEqual([x["label"] for x in fill], ["_filler"])
        self.assertIn([fill[0]["slot"], "_filler"], b["rounds"][-1])
        self.assertNotIn("_filler", b["verdicts"])
        self.assertEqual(len(b["fillers"]), 1)
        self.assertEqual(b["next"], "done")
        cmds = [ln for ln in out.splitlines() if "bash /root/layby/engine/run_arms.sh" in ln]
        self.assertEqual(len(cmds), 6)
        fc = [ln for ln in cmds if f"large-glm-slow-heavy.r1a1.{fill[0]['slot']}._filler" in ln]
        self.assertEqual(len(fc), 1)
        self.assertEqual(fc[0].split(" > ")[0].split()[-1], "C0")         # the filler runs arm C0's configuration
        # every measured arm's verdict records its neighbour (overlap seconds come from the markers on a real run)
        last = [lab for _, lab in b["rounds"][-1] if lab != "_filler"][0]
        self.assertEqual(b["verdicts"][last]["covariates"]["neighbour"]["labels"], ["_filler"])
        self.assertIn("CUDA_VISIBLE_DEVICES=4,5,6,7", out)
        self.assertIn("cpuset.mems", out)

    def test_ram_tier_sglang_cell(self):
        rc, out = run(["run", "--cell", self.cell("cell_main_sglang_fast_heavy.json"), "--host", "h1", "--out",
                       self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK, out[-1500:])
        self.assertIn("mount -t tmpfs -o size=68719476736,mode=0755 layby_ram_tier /mnt/layby/ram", out)
        self.assertNotIn("/io.max", out)
        self.assertIn("apply_patch.sh /root/.venv-sglang/bin/python check fdatasync", out)
        self.assertIn("WRITE_POLICY=write_through_selective", out)
        # memory.max carries the RAM tier: 28 GiB pinned + 12 GiB RSS + 256 MiB + 64 GiB
        self.assertIn(f"echo {(28 + 12 + 64) * (1 << 30) + (256 << 20)} > /sys/fs/cgroup/layby/", out)

    def test_k_scores_live_and_needs_its_inputs(self):
        rc, out = run(["run", "--cell", self.cell("cell_main_vllm_slow_heavy.json"), "--host", "h1", "--out", self.out,
                       "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK)
        cmd = {ln.split(" > ")[0].split()[-1]: ln for ln in out.splitlines() if "bash /root/layby/engine/run_arms.sh" in ln}
        self.assertIn("SIDE_FILE=/root/phase_r/measured.side.jsonl", cmd["K"])
        self.assertIn("SIDECAR_CKPT=/root/dwell/model.safetensors", cmd["K"])
        self.assertIn("SIDECAR_PORT=28766", cmd["K"])                       # park_port + 1
        for arm in ("K_ewma", "K_km", "C0"):
            self.assertNotIn("SIDE_FILE=", cmd[arm])
        self.assertTrue(all(f"DISK_BYTES={140 << 30}" in ln for ln in cmd.values()))
        # no live_scoring in the cell: refused before anything runs (exit 4)
        out2 = tempfile.mkdtemp(dir=self.tmp)
        p = self.cell("cell_main_vllm_slow_heavy.json", live_scoring=None)
        self.assertEqual(run(["run", "--cell", p, "--host", "h1", "--out", out2, "--dry-run"])[0],
                         orchestrate.EXIT_CONFIG)
        self.assertFalse(os.path.exists(os.path.join(out2, "runs.jsonl")))
        # without K the cell needs no live inputs
        c = load_json(os.path.join(EX, "cell_main_vllm_slow_heavy.json"))
        arms = [a for a in c["arms"] if a["code"] != "K"]
        self.assertEqual(run(["plan", "--cell", self.cell("cell_main_vllm_slow_heavy.json", live_scoring=None,
                                                          arms=arms)])[0], 0)

    def test_live_inputs_checked_where_the_engine_runs(self):
        from common import Runner
        c = orchestrate.load_cell(self.cell("cell_main_vllm_slow_heavy.json"))
        c["runner"]["mode"] = "host"
        c["live_scoring"] = {"side_file": os.path.join(self.tmp, "side.jsonl"), "ckpt": os.path.join(self.tmp, "m"),
                             "calib": os.path.join(self.tmp, "c")}
        open(c["live_scoring"]["ckpt"], "w").close()
        open(c["live_scoring"]["calib"], "w").close()
        probs = orchestrate.live_problems(Runner(), c)
        self.assertEqual(len(probs), 1)
        self.assertIn("side_file", probs[0])
        open(c["live_scoring"]["side_file"], "w").close()
        self.assertEqual(orchestrate.live_problems(Runner(), c), [])

    def test_cpu_tier_bytes_equal_across_arms(self):
        # 28.5 GiB is not a whole GiB: LMCache's L1 gets the exact float, so every arm has the same CPU tier
        p = self.cell("cell_main_vllm_slow_heavy.json",
                      capacities={"gpu_kv_bytes": "16G", "cpu_bytes": "28.5G", "disk_bytes": "140G"})
        c = orchestrate.load_cell(p)
        tiers = orchestrate.check_cpu_tiers(c)
        self.assertEqual(set(tiers.values()), {int(28.5 * (1 << 30))})
        self.assertEqual(orchestrate.lmcache_gb(int(28.5 * (1 << 30))), ("28.5", int(28.5 * (1 << 30))))
        c = load_json(os.path.join(EX, "cell_main_vllm_slow_heavy.json"))
        runner = dict(c["runner"], env=dict(c["runner"]["env"], OFF_EXTRA=',"cpu_bytes_to_use":1'))
        self.assertEqual(run(["plan", "--cell", self.cell("cell_main_vllm_slow_heavy.json", runner=runner)])[0],
                         orchestrate.EXIT_CONFIG)
        runner = dict(c["runner"], env=dict(c["runner"]["env"], LIVE_SCORING="off"))
        self.assertEqual(run(["plan", "--cell", self.cell("cell_main_vllm_slow_heavy.json", runner=runner)])[0],
                         orchestrate.EXIT_CONFIG)

    def test_capacity_and_chunk_are_required(self):
        p = self.cell("cell_main_vllm_slow_heavy.json", capacities={"gpu_kv_bytes": "16G", "cpu_bytes": "28G"})
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)
        p = self.cell("cell_main_vllm_slow_heavy.json", kv_chunk_bytes=None)
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)
        p = self.cell("cell_main_sglang_fast_heavy.json", ram_tier_bytes="32G")   # smaller than disk_bytes 64G
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)

    def test_regime_read_rate_is_the_slower_of_seq_and_chunk_reads(self):
        # RAM tier: sequential 20 GB/s, but random reads of one 9 MiB page at 5 GB/s: r = 1.7, medium, not fast
        c = load_json(os.path.join(EX, "cell_main_sglang_fast_heavy.json"))
        dr = dict(c["dry_run"], ram_probe=dict(c["dry_run"]["ram_probe"], rand_read_Bps=5.0e9))
        rc, out = run(["run", "--cell", self.cell("cell_main_sglang_fast_heavy.json", dry_run=dr), "--host", "h1",
                       "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_CONFIG)
        self.assertNotIn("run_arms_sglang.sh", out)
        reg = load_json(os.path.join(self.out, "main-sglang-fast-heavy", "h1", "r1", "a1", "regime.json"))["A"]
        self.assertEqual((reg["regime"], reg["read_path"]["used"]), ("medium", 5.0e9))
        self.assertIn("--bs=9437184", out)                                  # the RAM probe reads one tier page
        self.assertEqual(orchestrate.restore_read_Bps({"seq_read_Bps": 2.0, "rand_read_Bps": 3.0}), 2.0)

    def test_pilot_must_match_the_engine(self):
        p = self.cell("cell_main_sglang_fast_heavy.json")
        c = load_json(p)
        c["pilot"] = os.path.join(EX, "pilot_qwen3_8b_vllm.json")
        with open(p, "w") as f:
            json.dump(c, f)
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)

    def test_hicache_pilot(self):
        # HC-wt's L3 write rate with each patch variant and stock SGLang, inside a cgroup with the cell's io.max
        p = self.cell("cell_main_sglang_fast_heavy.json")
        rc, out = run(["hicache-pilot", "--cell", p, "--host", "h1", "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK, out[-1500:])
        self.assertIn("hicache_write_pilot.py --dir /mnt/layby/ram/kvtier --bytes 4294967296 --page-bytes 9437184", out)
        self.assertIn("cg_exec.sh /sys/fs/cgroup/layby/main-sglang-fast-heavy.hicache_pilot.A", out)
        res = load_json(os.path.join(self.out, "main-sglang-fast-heavy", "h1", "hicache_pilot.json"))
        self.assertEqual(sorted(res["slots"]["A"]["variants"]), ["fdatasync", "stock", "sync_file_range"])
        # under io.max on a disk cell the pilot's cgroup carries the throttle
        c = load_json(p)
        c.update(regime="slow", io_max={"rbps": "450M", "wbps": "450M"}, ram_tier_bytes=None,
                 disk={"mount": "/mnt/layby/A", "wipe": "fstrim"})
        with open(p, "w") as f:
            json.dump(c, f)
        rc, out = run(["hicache-pilot", "--cell", p, "--host", "h2", "--out", self.out, "--dry-run"])
        self.assertEqual(rc, orchestrate.EXIT_OK, out[-1500:])
        self.assertIn(f"echo '259:1 rbps={450 << 20} wbps={450 << 20}' > /sys/fs/cgroup/layby/main-sglang-fast-heavy"
                      f".hicache_pilot.A/io.max", out)
        # the measured runs switch variant by the cell key; a vLLM cell has no HiCache
        p2 = self.cell("cell_main_sglang_fast_heavy.json", sglang_patch_variant="sync_file_range")
        rc, out = run(["run", "--cell", p2, "--host", "h1", "--out", self.out, "--dry-run"])
        self.assertIn("apply_patch.sh /root/.venv-sglang/bin/python check sync_file_range", out)
        self.assertEqual(run(["plan", "--cell", self.cell("cell_main_sglang_fast_heavy.json",
                                                          sglang_patch_variant="nosync")])[0], orchestrate.EXIT_CONFIG)
        self.assertEqual(run(["hicache-pilot", "--cell", self.cell("cell_main_vllm_slow_heavy.json"), "--host", "h1",
                              "--out", self.out, "--dry-run"])[0], orchestrate.EXIT_CONFIG)

    def test_neighbour_overlap(self):
        d = self.tmp
        def ctx(label, t0, t1, filler=False):
            with open(os.path.join(d, f"c.{label}.h.markers"), "w") as f:
                f.write(f"replay_start {t0}\nreplay_end {t1}\n")
            return {"dir": d, "prefix": f"c.{label}.h", "arm": {"label": label, "filler": filler}}
        a, b = ctx("K", 100.0, 1100.0), ctx("_filler", 150.0, 900.0, True)
        ov = orchestrate.Block.overlap(None, a, [a, b])
        self.assertEqual((ov["overlap_s"], ov["overlap_frac"], ov["labels"], ov["filler"]), (750.0, 0.75, ["_filler"], [True]))
        self.assertIsNone(orchestrate.Block.overlap(None, a, [a]))           # one engine: no neighbour

    def test_plan_command(self):
        rc, out = run(["plan", "--cell", self.cell("cell_glm_slow_heavy_2slot.json"), "--repeat", "2"])
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.strip().splitlines()), 3)

    def test_bad_spec(self):
        p = self.cell("cell_main_vllm_slow_heavy.json", arms=[{"label": "K.x"}])
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)
        p = self.cell("cell_main_vllm_slow_heavy.json", io_max=None)
        self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG)
        # arm codes the runner does not know, CT without its eta, a T2 count mode neither system has, a replay
        # timeout that would kill the replay before the hard cap writes its censored rows
        for arms, extra in (([{"label": "X", "code": "T3"}], {}), ([{"label": "HC", "code": "T2"}], {"engine": "sglang"}),
                            ([{"label": "CT", "code": "CT"}], {}),
                            ([{"label": "T2", "code": "T2", "env": {"T2_COUNT": "block"}}], {}),
                            ([{"label": "C0", "code": "C0"}], {"hard_cap_s": 10600})):
            p = self.cell("cell_main_vllm_slow_heavy.json", arms=arms, **extra)
            self.assertEqual(run(["plan", "--cell", p])[0], orchestrate.EXIT_CONFIG, (arms, extra))
        p = self.cell("cell_main_vllm_slow_heavy.json", arms=[{"label": "C0", "code": "C0.2"}])   # a repeat suffix
        self.assertEqual(run(["plan", "--cell", p])[0], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
