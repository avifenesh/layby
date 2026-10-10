#!/usr/bin/env python3
"""Continuous host telemetry during an arm: one JSON line per interval until SIGTERM.

Each line: t (epoch s), mem_available, per cgroup (--cgroup NAME=PATH) memory.current, memory.stat (file, anon,
shmem, file_dirty, file_writeback, pgmajfault, workingset_refault_file), memory.events, memory.pressure and
io.pressure (avg10 and total us, some and full), io.stat (bytes and ios per device); per block device (--dev NAME)
/proc/diskstats read and write bytes, I/O counts and busy ms (iostat's inputs); and every --cachestat-every seconds,
page-cache residency of each disk tier directory (--tier NAME=DIR: files, bytes, cached, dirty).
GPU telemetry is a separate nvidia-smi -lms process (orchestrate.py starts both).

Usage: telemetry.py --out FILE [--interval 1] [--cgroup NAME=PATH]... [--dev NAME]... [--tier NAME=DIR]...
                    [--cachestat-every 10]
"""
import argparse
import json
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cgroup import parse_io_stat, parse_kv, parse_pressure  # noqa: E402
import pagecache  # noqa: E402

STAT_KEYS = ("file", "anon", "shmem", "file_mapped", "file_dirty", "file_writeback", "pgmajfault",
             "workingset_refault_file")


def rd(p):
    try:
        with open(p) as f:
            return f.read()
    except OSError:
        return ""


def cg_sample(p):
    st = parse_kv(rd(os.path.join(p, "memory.stat")))
    cur = rd(os.path.join(p, "memory.current")).strip()
    return {"memory.current": int(cur) if cur.isdigit() else None,
            "memory.stat": {k: st.get(k) for k in STAT_KEYS},
            "memory.events": parse_kv(rd(os.path.join(p, "memory.events"))),
            "memory.pressure": parse_pressure(rd(os.path.join(p, "memory.pressure"))),
            "io.pressure": parse_pressure(rd(os.path.join(p, "io.pressure"))),
            "io.stat": parse_io_stat(rd(os.path.join(p, "io.stat")))}


def diskstats(devs):
    out = {}
    for ln in rd("/proc/diskstats").splitlines():
        f = ln.split()
        if len(f) >= 14 and f[2] in devs:
            out[f[2]] = {"rd_ios": int(f[3]), "rd_bytes": int(f[5]) * 512, "wr_ios": int(f[7]),
                         "wr_bytes": int(f[9]) * 512, "in_flight": int(f[11]), "busy_ms": int(f[12])}
    return out


def mem_available():
    for ln in rd("/proc/meminfo").splitlines():
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) * 1024
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--cgroup", action="append", default=[])
    ap.add_argument("--dev", action="append", default=[])
    ap.add_argument("--tier", action="append", default=[])
    ap.add_argument("--cachestat-every", type=float, default=10.0)
    ap.add_argument("--samples", type=int, default=0, help="stop after N samples (tests); 0 = until SIGTERM")
    a = ap.parse_args(argv)
    cgs = dict(x.split("=", 1) for x in a.cgroup)
    tiers = dict(x.split("=", 1) for x in a.tier)
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    last_cs, n = 0.0, 0
    with open(a.out, "a") as out:
        while not stop["now"]:
            t = time.time()
            row = {"t": t, "mem_available": mem_available(), "cg": {k: cg_sample(p) for k, p in cgs.items()},
                   "disk": diskstats(set(a.dev))}
            if tiers and t - last_cs >= a.cachestat_every:
                row["tier"] = {k: pagecache.dir_cached_bytes(d) for k, d in tiers.items() if os.path.isdir(d)}
                last_cs = t
            out.write(json.dumps(row) + "\n")
            out.flush()
            n += 1
            if a.samples and n >= a.samples:
                break
            # sleep to the next tick; a signal ends the sleep early through the stop flag on the next check
            time.sleep(max(0.0, a.interval - (time.time() - t)))


if __name__ == "__main__":
    main()
