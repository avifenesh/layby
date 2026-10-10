"""Invalidation rules after an arm and the health checks before it (health.py, BENCH_PLAN sections 5 and 7), on
synthetic cgroup, PSI and GPU telemetry.

Run: python engine/phase_r/tests/test_invalidation.py
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import health  # noqa: E402
from common import GiB  # noqa: E402

T0 = 1_800_000_000.0     # replay_start
WIN = 1000.0             # a 1000 s replay window
FIELDS = health.GPU_FIELDS + health.COUNTER_FIELDS


def cg(oom=0, oom_kill=0):
    return {"memory.events": {"oom": oom, "oom_kill": oom_kill}, "io.stat": {"259:1": {"rbytes": 0, "wbytes": 0}},
            "memory.stat": {"file": 0}, "memory.peak": 1}


def tel(mem_full_s=0.0, io_full_s=0.0, before=100.0):
    """Telemetry rows every 10 s from boot (T0 - before) to the end, PSI full totals rising evenly in the window."""
    rows = []
    t = T0 - before
    while t <= T0 + WIN + 20:
        frac = min(max((t - T0) / WIN, 0.0), 1.0)
        boot_stall = 50e6 if t >= T0 else 0.0   # 50 s of memory stall while loading weights, before the window
        rows.append({"t": t, "cg": {"K": {"memory.pressure": {"full": {"total": int(boot_stall + mem_full_s * 1e6 * frac)}},
                                          "io.pressure": {"full": {"total": int(io_full_s * 1e6 * frac)}}}}})
        t += 10
    return rows


def gpu_row(t, mask=0, hw_thermal_us=0, power_cap_us=0, sm=2400):
    d = {f: "0" for f in FIELDS}
    d["timestamp"] = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime(t)) + ".000"
    d["index"] = "0"
    d["clocks.sm"] = str(sm)
    d["temperature.gpu"] = "60"
    d["clocks_event_reasons.active"] = hex(mask)
    d["clocks_event_reasons_counters.hw_thermal_slowdown"] = str(hw_thermal_us)
    d["clocks_event_reasons_counters.sw_power_cap"] = str(power_cap_us)
    d["mask"] = mask
    return d


def gpu(mask_at=None, mask=0, thermal_jump_at=None, cap_us_per_s=0):
    rows = []
    for k in range(-5, int(WIN) + 5, 5):
        t = T0 + k
        m = mask if mask_at is not None and abs(t - mask_at) < 3 else 0
        th = 2_000_000 if thermal_jump_at is not None and t >= thermal_jump_at else 0
        rows.append(gpu_row(t, m, th, int(cap_us_per_s * max(k, 0))))
    return rows


MARK = {"replay_start": T0, "replay_end": T0 + WIN}


def verdict(**kw):
    args = dict(markers=MARK, cg_before=cg(), cg_after=cg(), tel_rows=tel(), cg_name="K", gpu_rows=gpu(),
                runner_rc=0, runner_log="arm K rc=0 7858 turns", server_log="", regime="slow", engine="vllm")
    args.update(kw)
    return health.verdict(**args)


class AfterArm(unittest.TestCase):
    def test_clean_arm_is_valid(self):
        v = verdict()
        self.assertTrue(v["valid"], v["reasons"])
        self.assertAlmostEqual(v["covariates"]["window_s"], WIN)
        self.assertEqual(v["covariates"]["gpu0"]["sm_clock_median"], 2400)

    def test_oom_and_oom_kill_invalidate(self):
        self.assertFalse(verdict(cg_after=cg(oom=1))["valid"])
        v = verdict(cg_after=cg(oom_kill=2))
        self.assertFalse(v["valid"])
        self.assertIn("oom_kill +2", v["reasons"][0])

    def test_memory_psi_full_over_one_percent(self):
        self.assertTrue(verdict(tel_rows=tel(mem_full_s=9.0))["valid"])          # 0.9% of 1000 s
        v = verdict(tel_rows=tel(mem_full_s=11.0))                               # 1.1%
        self.assertFalse(v["valid"])
        self.assertIn("psi full memory", v["reasons"][0])

    def test_boot_stall_outside_window_does_not_count(self):
        v = verdict()   # tel() puts 50 s of memory stall before replay_start
        self.assertTrue(v["valid"])
        self.assertLess(v["covariates"]["psi_full_memory_frac"], 0.001)

    def test_io_psi_logged_not_excluded_by_default(self):
        v = verdict(tel_rows=tel(io_full_s=400.0))
        self.assertTrue(v["valid"])
        self.assertAlmostEqual(v["covariates"]["psi_full_io_frac"], 0.4, places=2)
        v = verdict(tel_rows=tel(io_full_s=400.0), settings={"psi_resources": ["memory", "io"]})
        self.assertFalse(v["valid"])

    def test_hw_slowdown_in_window_invalidates(self):
        v = verdict(gpu_rows=gpu(mask_at=T0 + 500, mask=0x8))
        self.assertFalse(v["valid"])
        self.assertIn("hw_slowdown", v["reasons"][0])

    def test_thermal_reasons_invalidate(self):
        for m in (0x20, 0x40, 0x80):
            self.assertFalse(verdict(gpu_rows=gpu(mask_at=T0 + 100, mask=m))["valid"], hex(m))

    def test_event_outside_window_ignored(self):
        self.assertTrue(verdict(gpu_rows=gpu(mask_at=T0 + WIN + 4, mask=0x40))["valid"])

    def test_power_cap_logged_not_excluded(self):
        v = verdict(gpu_rows=gpu(mask_at=T0 + 300, mask=0x4, cap_us_per_s=100_000))
        self.assertTrue(v["valid"])
        self.assertGreater(v["covariates"]["gpu0"]["sw_power_cap_s"], 0)
        self.assertGreater(v["covariates"]["gpu0"]["sw_power_cap_samples"], 0)

    def test_idle_reason_is_not_a_throttle(self):
        self.assertTrue(verdict(gpu_rows=gpu(mask_at=T0 + 300, mask=0x1))["valid"])

    def test_thermal_counter_increase_between_samples_invalidates(self):
        # a short thermal event between two 1 s samples shows only in the cumulative counter
        v = verdict(gpu_rows=gpu(thermal_jump_at=T0 + 600))
        self.assertFalse(v["valid"])
        self.assertIn("hw_thermal_slowdown counter", v["reasons"][0])

    def test_engine_crash(self):
        self.assertFalse(verdict(runner_rc=1)["valid"])
        self.assertFalse(verdict(runner_log="server failed\n")["valid"])
        self.assertFalse(verdict(markers={"replay_start": T0})["valid"])   # no replay_end: the run never finished

    def test_vllm_odirect_fallback(self):
        log = "WARNING O_DIRECT is not supported at '/mnt/x'; falling back to buffered I/O"
        self.assertFalse(verdict(server_log=log)["valid"])
        self.assertTrue(verdict(server_log=log, regime="fast")["valid"])     # RAM tier: no disk to bypass
        self.assertTrue(verdict(server_log=log, engine="sglang")["valid"])


class BeforeArm(unittest.TestCase):
    first = {"seq_read_Bps": 1.0e9}

    def pre(self, **kw):
        args = dict(gpu_rows=[gpu_row(T0)], gpu_procs_list=[], probe_now={"seq_read_Bps": 1.0e9},
                    probe_first=self.first, mem_avail=100 * GiB, budget=40 * GiB,
                    workload=(150, 7858, "abc"), expect={"sessions": 150, "turns": 7858}, sglang_patch=None)
        args.update(kw)
        return health.pre_arm(**args)

    def test_clean(self):
        self.assertEqual(self.pre(), [])

    def test_fio_within_ten_percent(self):
        self.assertEqual(self.pre(probe_now={"seq_read_Bps": 0.91e9}), [])
        bad = self.pre(probe_now={"seq_read_Bps": 0.89e9})
        self.assertEqual([b["check"] for b in bad], ["fio"])
        self.assertTrue(bad[0]["host_level"])
        self.assertEqual([b["check"] for b in self.pre(probe_now={"seq_read_Bps": 1.11e9})], ["fio"])
        self.assertEqual([b["check"] for b in self.pre(probe_now={"error": "fio rc=1"})], ["fio"])

    def test_free_memory_budget_plus_8_gib(self):
        self.assertEqual(self.pre(mem_avail=48 * GiB), [])
        self.assertEqual([b["check"] for b in self.pre(mem_avail=48 * GiB - 1)], ["free_memory"])

    def test_gpu_throttle_and_busy(self):
        bad = self.pre(gpu_rows=[gpu_row(T0, mask=0x40)], gpu_procs_list=["1234, python, 9000 MiB"])
        self.assertEqual(sorted(b["check"] for b in bad), ["gpu_busy", "gpu_throttle"])
        self.assertEqual(self.pre(gpu_rows=[gpu_row(T0, mask=0x4)]), [])   # power cap: not a failure

    def test_workload_counts_exact(self):
        self.assertEqual([b["check"] for b in self.pre(workload=(150, 7857, "abc"))], ["workload_counts"])
        bad = self.pre(expect={"sessions": 150, "turns": 7858, "sha256": "def"})
        self.assertEqual([b["check"] for b in bad], ["workload_sha256"])

    def test_sglang_patch_required(self):
        self.assertEqual(self.pre(sglang_patch="PATCHED 0.5.21 abc /x"), [])
        self.assertEqual([b["check"] for b in self.pre(sglang_patch="NOT_PATCHED 0.5.21 hunks=0 /x")],
                         ["sglang_patch"])


class Gpu(unittest.TestCase):
    def test_decode_and_parse(self):
        self.assertEqual(health.decode(0x4 | 0x40), ["sw_power_cap", "hw_thermal_slowdown"])
        self.assertEqual(health.mask_of("0x0000000000000008"), 8)
        line = ", ".join(["2026/10/10 21:00:00.123", "0", "2400", "12000", "2617", "55", "300.1", "600.0", "97",
                          "80000", "5", "16", "0x0000000000000004"]) + "\n"
        rows = health.parse_gpu_csv(line, health.GPU_FIELDS)
        self.assertEqual(rows[0]["mask"], 4)
        self.assertEqual(health.parse_gpu_csv("garbage\n", health.GPU_FIELDS), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
