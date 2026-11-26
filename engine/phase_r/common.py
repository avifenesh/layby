"""Command execution for the Phase R harness, with a dry-run mode.

Every action that touches the host goes through Runner: shell commands, cgroup and /proc writes, background processes.
Each one is logged (commands.log under the run directory) with its kind: root (needs root: cgroup, drop_caches,
mount, mkfs, fstrim), gpu (starts or queries a GPU process), or plain. In dry-run nothing is executed: the command is
printed with its kind and a canned result is returned (the `fake` argument, or an empty string), so the orchestrator
walks every step of a block on any machine.
"""
import json
import os
import shlex
import subprocess
import sys
import time

GiB = 1 << 30
MiB = 1 << 20


class Result:
    def __init__(self, rc, out, err=""):
        self.rc, self.out, self.err = rc, out, err

    @property
    def ok(self):
        return self.rc == 0


class DryProc:
    """Stands in for a Popen in dry-run."""
    pid = 0
    returncode = 0

    def poll(self):
        return 0

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


class Runner:
    def __init__(self, dry_run=False, log_path=None, echo=None):
        self.dry = dry_run
        self.log_path = log_path
        self.echo = dry_run if echo is None else echo
        self.history = []   # (kind, command) of every action, for tests and the dry-run listing

    def set_log(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.log_path = path

    def _note(self, kind, text):
        self.history.append((kind, text))
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}Z [{kind}] {text}"
        if self.echo:
            print(("DRY " if self.dry else "") + line[21:], flush=True)
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(line + "\n")

    @staticmethod
    def _fmt(cmd):
        return cmd if isinstance(cmd, str) else shlex.join(str(c) for c in cmd)

    def run(self, cmd, kind="plain", fake="", fake_rc=0, timeout=None, check=False, env=None):
        """Run cmd (list, or str for sh -c); returns Result. kind: plain, root or gpu."""
        text = self._fmt(cmd)
        self._note(kind, text)
        if self.dry:
            return Result(fake_rc, fake)
        try:
            p = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout,
                               env=env)
        except subprocess.TimeoutExpired as e:
            return Result(124, e.stdout or "", f"timeout after {timeout}s")
        if check and p.returncode != 0:
            raise RuntimeError(f"command failed rc={p.returncode}: {text}\n{p.stderr[-2000:]}")
        return Result(p.returncode, p.stdout, p.stderr)

    def write(self, path, value, kind="root"):
        """Write value to a kernel interface file (cgroup, /proc/sys)."""
        self._note(kind, f"echo {shlex.quote(str(value))} > {path}")
        if self.dry:
            return True
        with open(path, "w") as f:
            f.write(str(value))
        return True

    def read(self, path, fake=""):
        if self.dry:
            return fake
        with open(path) as f:
            return f.read()

    def mkdir(self, path, kind="root"):
        self._note(kind, f"mkdir -p {path}")
        if not self.dry:
            os.makedirs(path, exist_ok=True)

    def rmdir(self, path, kind="root"):
        self._note(kind, f"rmdir {path}")
        if not self.dry:
            try:
                os.rmdir(path)
            except FileNotFoundError:
                pass

    def popen(self, cmd, stdout_path, kind="plain", env=None):
        """Start a background process writing to stdout_path; returns a Popen (or DryProc)."""
        self._note(kind, f"{self._fmt(cmd)} > {stdout_path} 2>&1 &")
        if self.dry:
            return DryProc()
        os.makedirs(os.path.dirname(stdout_path), exist_ok=True)
        out = open(stdout_path, "ab")
        return subprocess.Popen(cmd, shell=isinstance(cmd, str), stdout=out, stderr=subprocess.STDOUT, env=env,
                                start_new_session=True)

    def stop(self, proc, grace=10):
        """Terminate a background process by its handle (never by pattern), bounded."""
        if isinstance(proc, DryProc) or proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=grace)


def dump(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=str)
    os.replace(tmp, path)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def read_text(path, errors="strict"):
    with open(path, errors=errors) as f:
        return f.read()


def append_jsonl(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True, default=str) + "\n")


def parse_size(v):
    """Bytes from an int or a string with a binary suffix (K, M, G, T; KiB, MiB also accepted)."""
    if isinstance(v, (int, float)):
        return int(v)
    s = str(v).strip().upper().removesuffix("IB").removesuffix("B")
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    if s and s[-1] in mult:
        return int(float(s[:-1]) * mult[s[-1]])
    return int(s)


def log(msg):
    print(f"[phase_r {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)
