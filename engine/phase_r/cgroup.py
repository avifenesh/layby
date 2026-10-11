"""cgroup v2 isolation for one engine process (BENCH_PLAN section 5, "Page cache, one rule for every arm").

Layout: /sys/fs/cgroup/layby/<name>, one child per engine per arm, created fresh before the arm and removed after.
Limits on the child:
  memory.max      = pinned tiers + measured engine RSS + 256 MiB of page-cache allowance (+ the RAM tier in the fast
                    regime, whose tmpfs pages are charged to the engine that writes them)
  memory.swap.max = 0 (no pages hide in swap)
  cpuset.cpus / cpuset.mems: the engine's cores and NUMA node(s)
  io.max          = "MAJ:MIN rbps=R wbps=W" on the whole-disk device under the disk tier (io.max refuses partitions,
                    so a partition resolves to its parent disk). Buffered writeback is charged to the cgroup that
                    dirtied the pages only when memory and io are both enabled on it, which they are.
The engine joins the child through cg_exec.sh (it writes 0 to cgroup.procs, so it works from inside a privileged
container started with --cgroupns=host, where PIDs differ from the host's).

Direct cgroupfs writes, not systemd-run: the engines run inside a docker container (engine/node_prep.sh), where
systemd-run has no systemd to talk to. The parent sits at the top level beside system.slice; systemd leaves cgroups it
did not create alone.
"""
import os
import time

from common import MiB

CG_ROOT = "/sys/fs/cgroup"
PARENT = "layby"
SLACK = 256 * MiB
CONTROLLERS = ("cpu", "cpuset", "io", "memory")


PAGE = 4096


def budget(pinned_bytes, rss_bytes, slack=SLACK, ram_tier_bytes=0):
    """memory.max for one engine: pinned tiers + measured RSS + page-cache allowance (+ RAM tier, fast regime),
    rounded up to a whole page (the kernel rounds memory.max down to a page; verify() reads it back exactly)."""
    b = int(pinned_bytes) + int(rss_bytes) + int(slack) + int(ram_tier_bytes)
    return -(-b // PAGE) * PAGE


def path_of(name, root=CG_ROOT):
    return os.path.join(root, PARENT, name)


def io_max_line(majmin, rbps=None, wbps=None):
    r = "max" if rbps is None else str(int(rbps))
    w = "max" if wbps is None else str(int(wbps))
    return f"{majmin} rbps={r} wbps={w}"


def block_device(runner, path, override=None):
    """The whole-disk block device under path: {'source','fstype','name','majmin','ram'}.

    override (from the cell spec's disk.device) is used as given, for hosts where the mount source does not name the
    device (LVM over several disks, for example: give the dm device)."""
    if override:
        name = os.path.basename(override)
        mm = runner.read(f"/sys/class/block/{name}/dev", fake="259:1\n").strip()
        return {"source": override, "fstype": "given", "name": name, "majmin": mm, "ram": False}
    r = runner.run(["findmnt", "-no", "SOURCE,FSTYPE", "--target", path], fake="/dev/nvme1n1 ext4\n")
    src, fstype = (r.out.split() + ["", ""])[:2]
    if fstype in ("tmpfs", "ramfs", "overlay") or not src.startswith("/dev/"):
        return {"source": src, "fstype": fstype, "name": None, "majmin": None, "ram": fstype in ("tmpfs", "ramfs")}
    name = os.path.basename(os.path.realpath(src)) if not runner.dry else os.path.basename(src)
    sysd = f"/sys/class/block/{name}"
    if not runner.dry and os.path.exists(os.path.join(sysd, "partition")):
        name = os.path.basename(os.path.dirname(os.path.realpath(sysd)))   # partition -> parent disk
    mm = runner.read(f"/sys/class/block/{name}/dev", fake="259:1\n").strip()
    return {"source": src, "fstype": fstype, "name": name, "majmin": mm, "ram": False}


def check_v2(runner):
    """Problems that stop the harness: cgroup v2 not mounted, or a controller missing."""
    problems = []
    r = runner.run(["stat", "-f", "-c", "%T", CG_ROOT], fake="cgroup2fs\n")
    if r.out.strip() != "cgroup2fs":
        problems.append(f"{CG_ROOT} is {r.out.strip() or 'missing'}, not cgroup2fs (cgroup v2 unified hierarchy)")
        return problems
    have = runner.read(os.path.join(CG_ROOT, "cgroup.controllers"), fake=" ".join(CONTROLLERS)).split()
    for c in CONTROLLERS:
        if c not in have:
            problems.append(f"controller {c} not available at the cgroup root")
    return problems


def ensure_parent(runner):
    """Create /sys/fs/cgroup/layby and enable the controllers for its children."""
    want = " ".join("+" + c for c in CONTROLLERS)
    runner.write(os.path.join(CG_ROOT, "cgroup.subtree_control"), want)
    parent = os.path.join(CG_ROOT, PARENT)
    runner.mkdir(parent)
    runner.write(os.path.join(parent, "cgroup.subtree_control"), want)
    return parent


def create(runner, name, mem_max, cpus, mems, io=None):
    """Fresh child cgroup with the arm's limits; io: list of (majmin, rbps, wbps). Returns its path."""
    ensure_parent(runner)
    p = path_of(name)
    if not runner.dry and os.path.isdir(p):
        destroy(runner, p)
    runner.mkdir(p)
    runner.write(os.path.join(p, "cpuset.cpus"), cpus)
    runner.write(os.path.join(p, "cpuset.mems"), mems)
    runner.write(os.path.join(p, "memory.max"), int(mem_max))
    runner.write(os.path.join(p, "memory.swap.max"), 0)
    for majmin, rbps, wbps in io or []:
        runner.write(os.path.join(p, "io.max"), io_max_line(majmin, rbps, wbps))
    return p


def verify(runner, p, mem_max, cpus, io=None):
    """Read the limits back; returns a list of mismatches (an empty list is a pass)."""
    bad = []
    got = runner.read(os.path.join(p, "memory.max"), fake=f"{int(mem_max)}\n").strip()
    if got != str(int(mem_max)):
        bad.append(f"memory.max {got} != {int(mem_max)}")
    got = runner.read(os.path.join(p, "cpuset.cpus.effective"), fake=f"{cpus}\n").strip()
    if not runner.dry and _cpus(got) != _cpus(cpus):
        bad.append(f"cpuset.cpus.effective {got} != {cpus}")
    want_io = {mm: io_max_line(mm, r, w) for mm, r, w in io or []}
    text = runner.read(os.path.join(p, "io.max"), fake="\n".join(want_io.values()))
    lines = {ln.split()[0]: ln.strip() for ln in text.splitlines() if ln.strip()}

    def rw(line):   # the kernel also prints riops and wiops; rbps and wbps are the limits set here
        kv = dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)
        return kv.get("rbps", "max"), kv.get("wbps", "max")
    for mm, line in want_io.items():
        if mm in lines and rw(lines[mm]) == rw(line):
            continue
        if not (line.endswith("rbps=max wbps=max") and mm not in lines):
            bad.append(f"io.max {lines.get(mm)} != {line}")
    return bad


def _cpus(s):
    out = set()
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def destroy(runner, p, timeout=30):
    """Kill whatever is left in the cgroup (cgroup.kill, Linux 5.14+), wait for it to empty (bounded), remove it."""
    if not runner.dry and not os.path.isdir(p):
        return True
    runner.write(os.path.join(p, "cgroup.kill"), 1)
    if not runner.dry:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if "populated 0" in runner.read(os.path.join(p, "cgroup.events")):
                break
            time.sleep(0.2)
        else:
            return False
    runner.rmdir(p)
    return True


def parse_kv(text):
    out = {}
    for ln in text.splitlines():
        k, _, v = ln.partition(" ")
        if k:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = v.strip()
    return out


def parse_pressure(text):
    """'some avg10=0.00 avg60=0.00 avg300=0.00 total=123' lines -> {'some': {...}, 'full': {...}}; total in us."""
    out = {}
    for ln in text.splitlines():
        parts = ln.split()
        if not parts:
            continue
        out[parts[0]] = {k: (int(v) if k == "total" else float(v)) for k, v in (x.split("=") for x in parts[1:])}
    return out


def parse_io_stat(text):
    out = {}
    for ln in text.splitlines():
        parts = ln.split()
        if parts:
            out[parts[0]] = {k: int(v) for k, v in (x.split("=") for x in parts[1:])}
    return out


def snapshot(runner, p):
    """Every counter the invalidation rules and the telemetry use, read once."""
    def rd(f, fake=""):
        try:
            return runner.read(os.path.join(p, f), fake=fake)
        except (FileNotFoundError, PermissionError, OSError):
            return ""
    return {
        "t": time.time(),
        "memory.current": _int(rd("memory.current", "0")),
        "memory.peak": _int(rd("memory.peak", "0")),
        "memory.stat": parse_kv(rd("memory.stat", "file 0\nanon 0\nshmem 0\nfile_dirty 0\nfile_writeback 0")),
        "memory.events": parse_kv(rd("memory.events", "low 0\nhigh 0\nmax 0\noom 0\noom_kill 0")),
        "memory.pressure": parse_pressure(rd("memory.pressure", "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
                                                                "full avg10=0.00 avg60=0.00 avg300=0.00 total=0")),
        "io.pressure": parse_pressure(rd("io.pressure", "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\n"
                                                        "full avg10=0.00 avg60=0.00 avg300=0.00 total=0")),
        "io.stat": parse_io_stat(rd("io.stat")),
        "cpu.stat": parse_kv(rd("cpu.stat", "usage_usec 0")),
    }


def _int(s):
    s = (s or "").strip()
    return int(s) if s.isdigit() else None
