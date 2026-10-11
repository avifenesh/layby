"""cgroup helpers, the shell wrappers, the telemetry sampler, page-cache residency and fio parsing; CPU only, no root.

Run: python engine/phase_r/tests/test_tools.py
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

PR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PR)
import cgroup  # noqa: E402
import pagecache  # noqa: E402
import probe  # noqa: E402
from common import GiB, MiB, Runner, parse_size  # noqa: E402

SCRATCH = os.environ.get("LAYBY_TEST_DIR", os.path.expanduser("~/.cache/layby-phase-r-test"))


class CpuTier(unittest.TestCase):
    def test_cpu_tier_must_be_page_aligned(self):
        # 2026-10-11 pilot: LMCache 0.5.5's NIXL L2 adapter fails at startup (prepXferDlist NIXL_ERR_NOT_FOUND) when
        # the L1 size is not a whole number of pages; Qwen's cpu_bytes 26793659596 (3,276 B past a page) killed arm LM
        # before any request. The cell is refused before any arm instead.
        import orchestrate
        def cell(cpu):
            return {"capacities": {"cpu_bytes": str(cpu)}, "runner": {"env": {}}, "slots": [{}],
                    "arms": [{"label": "C0", "code": "C0", "env": {}}, {"label": "LM", "code": "LM", "env": {}}]}
        with self.assertRaises(ValueError):
            orchestrate.check_cpu_tiers(cell(26793659596))
        self.assertEqual(set(orchestrate.check_cpu_tiers(cell(26793656320)).values()), {26793656320})


class Cgroup(unittest.TestCase):
    def test_budget(self):
        # pinned 28 GiB + RSS 12 GiB + 256 MiB; the RAM tier adds its size
        self.assertEqual(cgroup.budget(28 * GiB, 12 * GiB), 40 * GiB + 256 * MiB)
        self.assertEqual(cgroup.budget(28 * GiB, 12 * GiB, ram_tier_bytes=64 * GiB), 104 * GiB + 256 * MiB)

    def test_budget_is_a_whole_number_of_pages(self):
        # the kernel rounds memory.max down to a page and the read-back check compares exactly: the 2026-10-11 pilot
        # (cpu_bytes 26793659596, not page aligned) failed every arm's health check on "memory.max 61421830144 !=
        # 61421833420"
        b = cgroup.budget(26793659596, 32 * GiB)
        self.assertEqual(b % 4096, 0)
        self.assertTrue(26793659596 + 32 * GiB + 256 * MiB <= b < 26793659596 + 32 * GiB + 256 * MiB + 4096)

    def test_verify_reads_the_kernels_io_max_format(self):
        # the kernel prints rbps, wbps, riops and wiops; only rbps and wbps are set, and they are what is compared
        class R:
            dry = False
            def read(self, path, fake=None):
                return {"memory.max": f"{41 * GiB}\n", "cpuset.cpus.effective": "2-25\n",
                        "io.max": "253:32 rbps=1000000000 wbps=max riops=max wiops=max\n"}[os.path.basename(path)]
        self.assertEqual(cgroup.verify(R(), "/x", 41 * GiB, "2-25", [("253:32", 1000000000, None)]), [])
        self.assertTrue(cgroup.verify(R(), "/x", 41 * GiB, "2-25", [("253:32", 900000000, None)]))

    def test_io_max_line(self):
        self.assertEqual(cgroup.io_max_line("259:1", 450 * MiB, 450 * MiB), "259:1 rbps=471859200 wbps=471859200")
        self.assertEqual(cgroup.io_max_line("8:0", None, 1000), "8:0 rbps=max wbps=1000")

    def test_parsers(self):
        p = cgroup.parse_pressure("some avg10=1.50 avg60=0.20 avg300=0.00 total=1234\n"
                                  "full avg10=0.00 avg60=0.00 avg300=0.00 total=99\n")
        self.assertEqual(p["full"]["total"], 99)
        self.assertEqual(p["some"]["avg10"], 1.5)
        s = cgroup.parse_io_stat("259:1 rbytes=10 wbytes=20 rios=1 wios=2 dbytes=0 dios=0\n")
        self.assertEqual(s["259:1"]["wbytes"], 20)
        self.assertEqual(cgroup.parse_kv("oom 0\noom_kill 3\n")["oom_kill"], 3)

    def test_create_writes_every_limit(self):
        rn = Runner(dry_run=True, echo=False)
        p = cgroup.create(rn, "cell.r1a1.A.K", 41 * GiB, "2-25", "0", [("259:1", 471859200, 471859200)])
        writes = [t for k, t in rn.history if t.startswith("echo")]
        for want in ("+cpu +cpuset +io +memory' > /sys/fs/cgroup/cgroup.subtree_control",
                     f"echo 2-25 > {p}/cpuset.cpus", f"echo 0 > {p}/cpuset.mems",
                     f"echo {41 * GiB} > {p}/memory.max", f"echo 0 > {p}/memory.swap.max",
                     f"echo '259:1 rbps=471859200 wbps=471859200' > {p}/io.max"):
            self.assertTrue(any(want in w for w in writes), (want, writes))
        self.assertTrue(all(k == "root" for k, t in rn.history if t.startswith(("echo", "mkdir"))))
        self.assertEqual(cgroup.verify(rn, p, 41 * GiB, "2-25", [("259:1", 471859200, 471859200)]), [])

    def test_cpus(self):
        self.assertEqual(cgroup._cpus("0-2,5"), {0, 1, 2, 5})

    def test_this_host_is_cgroup_v2(self):
        rn = Runner(dry_run=False, echo=False)
        self.assertEqual(cgroup.check_v2(rn), [], "cgroup v2 with cpu, cpuset, io and memory expected here")

    def test_block_device_of_a_real_path(self):
        rn = Runner(dry_run=False, echo=False)
        d = cgroup.block_device(rn, os.path.expanduser("~"))
        if d["majmin"] is not None:
            self.assertRegex(d["majmin"], r"^\d+:\d+$")
            self.assertFalse(os.path.exists(f"/sys/class/block/{d['name']}/partition"), "a partition, not its disk")


class Shell(unittest.TestCase):
    def test_cg_exec_empty_runs_unconfined(self):
        r = subprocess.run([os.path.join(PR, "cg_exec.sh"), "", "echo", "hi"], capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout), (0, "hi\n"))

    def test_cg_exec_refuses_when_it_cannot_join(self):
        d = tempfile.mkdtemp(dir=SCRATCH)   # not a cgroup: writing cgroup.procs is refused unless root; then 97
        os.chmod(d, 0o500)
        r = subprocess.run([os.path.join(PR, "cg_exec.sh"), d, "echo", "ran"], capture_output=True, text=True)
        os.chmod(d, 0o700)
        if os.geteuid() != 0:
            self.assertEqual(r.returncode, 97)
            self.assertNotIn("ran", r.stdout)

    def test_arm_hook_markers(self):
        d = tempfile.mkdtemp(dir=SCRATCH)
        h = os.path.join(PR, "arm_hook.sh")
        subprocess.run([h, "replay_start", "K", d], check=True)
        subprocess.run([h, "replay_end", "K", d], check=True)
        import health
        m = health.read_markers(os.path.join(d, "K.markers"))
        self.assertEqual(sorted(m), ["replay_end", "replay_start"])
        self.assertLessEqual(m["replay_start"], m["replay_end"])


class Telemetry(unittest.TestCase):
    def test_samples_this_cgroup(self):
        with open("/proc/self/cgroup") as f:
            own = f.read().strip().split("::")[-1]
        cgp = "/sys/fs/cgroup" + own
        d = tempfile.mkdtemp(dir=SCRATCH)
        with open(os.path.join(d, "f.bin"), "wb") as f:
            f.write(b"x" * (1 << 20))
        out = os.path.join(tempfile.mkdtemp(dir=SCRATCH), "tel.jsonl")
        dev = os.listdir("/sys/block")[0]
        subprocess.run([sys.executable, os.path.join(PR, "telemetry.py"), "--out", out, "--interval", "0.05",
                        "--cgroup", f"me={cgp}", "--dev", dev, "--tier", f"t={d}", "--cachestat-every", "0",
                        "--samples", "3"], check=True, timeout=60)
        with open(out) as f:
            rows = [json.loads(x) for x in f]
        self.assertEqual(len(rows), 3)
        self.assertIn("memory.pressure", rows[0]["cg"]["me"])
        self.assertGreater(rows[0]["mem_available"], 0)
        self.assertIn(dev, rows[0]["disk"])
        self.assertEqual(rows[0]["tier"]["t"]["files"], 1)


class PageCache(unittest.TestCase):
    def test_mincore_agrees_with_cachestat(self):
        d = tempfile.mkdtemp(dir=SCRATCH)
        p = os.path.join(d, "a.bin")
        with open(p, "wb") as f:
            f.write(os.urandom(4 << 20))
        fd = os.open(p, os.O_RDONLY)
        try:
            os.read(fd, 4 << 20)
            cs = pagecache.cachestat_fd(fd)
            mc = pagecache.mincore_fd(fd, 4 << 20)
        finally:
            os.close(fd)
        if cs is None:
            self.skipTest("no cachestat on this kernel")
        self.assertEqual(cs["nr_cache"], mc)
        self.assertEqual(mc * pagecache.PAGE, 4 << 20)


class Fio(unittest.TestCase):
    def test_parse(self):
        out = "fio: warning\n" + json.dumps({"jobs": [{"read": {"io_bytes": 5, "bw_bytes": 4.7e8, "iops": 112.0,
                                                                "clat_ns": {"percentile": {"99.000000": 3.1e6}}},
                                                       "write": {"io_bytes": 0}}]})
        p = probe.parse_fio(out)
        self.assertEqual(p["read"]["Bps"], 4.7e8)
        self.assertAlmostEqual(p["read"]["p99_us"], 3100.0)
        self.assertNotIn("write", p)

    def test_full_probe_commands(self):
        rn = Runner(dry_run=True, echo=False)
        r = probe.run(rn, "/mnt/t", "full", wrap=["/x/cg_exec.sh", "/sys/fs/cgroup/layby/p"],
                      fake={"seq_read_Bps": 1e9, "seq_write_Bps": 5e8})
        cmds = [t for _, t in rn.history if " fio --name" in " " + t]
        self.assertEqual(len(cmds), 3)
        self.assertTrue(all("--direct=1" in c and "--bs=4M" in c or "randread" in c for c in cmds))
        self.assertTrue(all(c.startswith("/x/cg_exec.sh /sys/fs/cgroup/layby/p fio") for c in cmds))
        self.assertEqual((r["seq_read_Bps"], r["seq_write_Bps"]), (1e9, 5e8))
        self.assertAlmostEqual(probe.drift({"seq_read_Bps": 1e9}, {"seq_read_Bps": 0.9e9}), 0.1)

    def test_sizes(self):
        self.assertEqual(parse_size("450M"), 450 * MiB)
        self.assertEqual(parse_size("4GiB"), 4 * GiB)
        self.assertEqual(parse_size(123), 123)


if __name__ == "__main__":
    os.makedirs(SCRATCH, exist_ok=True)
    unittest.main(verbosity=2)
