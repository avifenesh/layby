"""fio probes of a disk tier (BENCH_PLAN sections 5 and 7; research brief 3.3).

  full : O_DIRECT 4 MiB sequential write, then sequential read, then random read; before and after every block
  quick: O_DIRECT 4 MiB sequential read; before every arm (the 10% health rule compares it with the host's first
         full probe)
A probe can run inside an arm's cgroup (wrap = cg_exec.sh CGROUP), so io.max applies and the measured number is the
throttled one the regime is computed from. The RAM tier (tmpfs) is probed buffered (direct=False).

Defaults: 4 MiB blocks (the size Nebius advises for full bandwidth; the plan fixes 4 MiB), queue depth 16 for the
sequential jobs and 32 for random reads, libaio. Random reads use the tier's chunk size when the cell gives it
(kv_chunk_bytes), since that is the read the engine issues, else 4 KiB (an IOPS probe).
"""
import json
import os

from common import parse_size

DEFAULTS = {"ioengine": "libaio", "size": "4G", "runtime_full": 30, "runtime_quick": 8, "bs": "4M",
            "iodepth_seq": 16, "iodepth_rand": 32, "rand_bs": "4k"}
FILE = ".layby_fio_probe"


def fio_cmd(job, filename, rw, bs, runtime, size, direct, ioengine, iodepth):
    cmd = ["fio", f"--name={job}", f"--filename={filename}", f"--rw={rw}", f"--bs={bs}", f"--size={size}",
           f"--direct={1 if direct else 0}", f"--ioengine={ioengine}", f"--iodepth={iodepth}", "--numjobs=1",
           "--time_based", f"--runtime={runtime}", "--ramp_time=2", "--group_reporting", "--output-format=json"]
    if rw in ("write", "randwrite"):
        cmd.append("--end_fsync=1")
    return cmd


def parse_fio(text):
    """bytes/s, IOPS and p99 completion latency (us) of the single job in fio's JSON (warnings before it skipped)."""
    i = text.find("{")
    if i < 0:
        raise ValueError("no JSON in fio output")
    j = json.loads(text[i:])["jobs"][0]
    out = {}
    for d in ("read", "write"):
        x = j.get(d) or {}
        if x.get("io_bytes", 0) > 0:
            p = (x.get("clat_ns") or {}).get("percentile") or {}
            out[d] = {"Bps": float(x["bw_bytes"]), "iops": float(x["iops"]),
                      "p99_us": p.get("99.000000", 0) / 1000.0}
    return out


def _fake(rw, values):
    d = "write" if "write" in rw else "read"
    key = {"write": "seq_write_Bps", "read": "seq_read_Bps", "randread": "rand_read_Bps"}[rw]
    bps = values.get(key, 1.0e9)
    return json.dumps({"jobs": [{d: {"io_bytes": 1, "bw_bytes": bps, "iops": bps / (4 << 20),
                                     "clat_ns": {"percentile": {"99.000000": 2.0e6}}}}]})


def run(runner, tier_dir, kind="full", cfg=None, wrap=None, direct=True, rand_bs=None, fake=None):
    """Probe tier_dir; returns {'kind','direct','seq_read_Bps',...,'raw': {...}}. wrap: command prefix (list)."""
    c = dict(DEFAULTS, **(cfg or {}))
    fake = fake or {}
    fn = os.path.join(tier_dir, FILE)
    rt = c["runtime_full"] if kind == "full" else c["runtime_quick"]
    jobs = [("seq_write", "write", c["bs"], c["iodepth_seq"]), ("seq_read", "read", c["bs"], c["iodepth_seq"]),
            ("rand_read", "randread", rand_bs or c["rand_bs"], c["iodepth_rand"])]
    if kind == "quick":
        jobs = [("seq_read", "read", c["bs"], c["iodepth_seq"])]
    res = {"kind": kind, "dir": tier_dir, "direct": direct, "wrapped": bool(wrap), "raw": {}}
    for job, rw, bs, qd in jobs:
        cmd = (list(wrap) if wrap else []) + fio_cmd(job, fn, rw, bs, rt, c["size"], direct, c["ioengine"], qd)
        r = runner.run(cmd, kind="root" if wrap else "plain", fake=_fake(rw, fake), timeout=rt + 600)
        if not r.ok:
            res["error"] = f"{job}: rc={r.rc} {r.err[-500:]}"
            break
        p = parse_fio(r.out)
        d = "write" if "write" in rw else "read"
        res[f"{job}_Bps"] = p[d]["Bps"]
        res[f"{job}_iops"] = p[d]["iops"]
        res[f"{job}_p99_us"] = p[d]["p99_us"]
        res["raw"][job] = r.out if len(r.out) < 200000 else r.out[:200000]
    runner.run(["rm", "-f", fn], kind="root", fake="")
    res["bytes_size"] = parse_size(c["size"])
    return res


def drift(first, now, key="seq_read_Bps"):
    """Relative change of a probe value from the host's first probe (0.07 = 7% slower or faster)."""
    a, b = first.get(key), now.get(key)
    if not a or b is None:
        return None
    return abs(b - a) / a
