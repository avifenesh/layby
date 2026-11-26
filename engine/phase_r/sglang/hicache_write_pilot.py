#!/usr/bin/env python3
"""HC-wt's L3 write throughput with each page-cache patch variant and with stock SGLang (BENCH_PLAN section 5 pilot).

HiCacheFile.set (SGLang v0.5.21's own hicache_storage.py: tests/fixtures/hicache_storage_v0.5.21.py, unmodified) is
the call SGLang's HiCache makes for every page it writes to the file L3 tier. This tool writes --bytes of pages of
--page-bytes through it, from --threads threads, into --dir, once per variant:
  stock            the unmodified file (buffered writes, pages stay in the page cache)
  fdatasync        hicache_fadvise.patch (write, fdatasync, DONTNEED; the default every SGLang arm runs with)
  sync_file_range  hicache_fadvise_sfr.patch (write, sync_file_range on the data, DONTNEED)
and prints one JSON object per variant: set_Bps (bytes over the time until the last set returned, what the backup
queue sees), durable_Bps (the same bytes over the time until a final syncfs of the directory returned, so a buffered
variant pays for its writeback too), the p50 and p99 time of one set, and the page cache the variant left behind.
orchestrate.py hicache-pilot runs it inside a cgroup carrying the cell's io.max (cg_exec.sh), so the throttle applies,
and records the variants side by side; the pilot decides which variant the measured runs use (cell key
"sglang_patch_variant").

Imports the module with stubs for the two SGLang modules it needs (envs and the LRU file evictor) and real torch when
importable, else a small numpy stand-in for the tensor calls the module makes. CPU only; needs `patch`.

Usage: hicache_write_pilot.py --dir DIR [--variant stock|fdatasync|sync_file_range]... [--bytes 4G]
                              [--page-bytes 9437184] [--threads 1]
"""
import argparse
import concurrent.futures
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.dirname(HERE)
FIXTURE = os.path.join(PR, "tests", "fixtures", "hicache_storage_v0.5.21.py")
PATCHES = {"stock": None, "fdatasync": os.path.join(HERE, "hicache_fadvise.patch"),
           "sync_file_range": os.path.join(HERE, "hicache_fadvise_sfr.patch")}

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def numpy_torch():
    """The torch surface hicache_storage.py touches in set/get: uint8, Tensor, numel, element_size, view, contiguous,
    numpy."""
    import numpy as np
    t = types.ModuleType("torch")
    t.uint8 = np.uint8

    class Tensor:
        def __init__(self, a):
            self.a = a

        def numel(self):
            return self.a.size

        def element_size(self):
            return self.a.itemsize

        def contiguous(self):
            return Tensor(np.ascontiguousarray(self.a))

        def view(self, dtype=None):
            return Tensor(self.a.view(dtype))

        def numpy(self):
            return self.a
    t.Tensor = Tensor
    t.from_numpy = Tensor
    return t


def variant_file(variant, workdir):
    """A copy of the v0.5.21 file with the variant's patch applied (stock: unmodified)."""
    pkg = os.path.join(workdir, variant)
    d = os.path.join(pkg, "sglang", "srt", "mem_cache")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "hicache_storage.py")
    shutil.copy(FIXTURE, path)
    if PATCHES[variant]:
        with open(PATCHES[variant]) as f:
            r = subprocess.run(["patch", "--forward", "-p2", "-d", pkg], stdin=f, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"patch {variant}: {r.stdout}{r.stderr}")
    return path


def load_hicache(path, store_dir):
    """Import hicache_storage.py from path with stubbed sglang.srt.environ and the LRU evictor."""
    for k in [k for k in sys.modules if k == "sglang" or k.startswith("sglang.")]:
        del sys.modules[k]
    if not HAVE_TORCH:
        sys.modules["torch"] = numpy_torch()

    class _Env:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v
    envs = types.SimpleNamespace(SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=_Env(store_dir),
                                 SGLANG_HICACHE_FILE_BACKEND_ENABLE_METADATA_CACHE=_Env(False),
                                 SGLANG_HICACHE_FILE_BACKEND_METADATA_TTL=_Env(None))

    class LRUFileEvictor:
        def __init__(self, *a, **k):
            pass

        def reserve(self, *a, **k):
            return True

        def commit(self, *a, **k):
            pass

        def touch(self, *a, **k):
            pass

        def abort(self, *a, **k):
            pass

        def clear(self):
            pass
    names = ["sglang", "sglang.srt", "sglang.srt.environ", "sglang.srt.mem_cache", "sglang.srt.mem_cache.storage",
             "sglang.srt.mem_cache.storage.file", "sglang.srt.mem_cache.storage.file.lru_file_evictor"]
    for n in names:
        sys.modules[n] = types.ModuleType(n)
    sys.modules["sglang.srt.environ"].envs = envs
    sys.modules["sglang.srt.mem_cache.storage.file.lru_file_evictor"].LRUFileEvictor = LRUFileEvictor
    spec = importlib.util.spec_from_file_location("sglang.srt.mem_cache.hicache_storage", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def make_store(m):
    cfg = m.HiCacheStorageConfig(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                                 is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=True,
                                 model_name="Qwen/Qwen3-8B")
    return m.HiCacheFile(cfg)


def tensor(nbytes, fill):
    import numpy as np
    a = (np.arange(nbytes, dtype=np.uint32) * 2654435761 + fill).astype(np.uint8)
    if HAVE_TORCH:
        import torch
        return torch.from_numpy(a)
    return sys.modules["torch"].Tensor(a)


def measure(variant, store_dir, total, page, threads, workdir):
    sys.path.insert(0, PR)
    import pagecache
    m = load_hicache(variant_file(variant, workdir), store_dir)
    store = make_store(m)
    n = max(1, total // page)
    src = tensor(page, 7)
    lat = []

    def one(i):
        t = time.perf_counter()
        ok = store.set(f"{variant}-{i}", src)
        lat.append(time.perf_counter() - t)
        return ok
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(threads) as ex:
        oks = list(ex.map(one, range(n)))
    t_set = time.perf_counter() - t0
    syncfs(store_dir)
    t_dur = time.perf_counter() - t0
    left = pagecache.dir_cached_bytes(store_dir)
    lat.sort()
    nbytes = n * page
    return {"variant": variant, "pages": n, "page_bytes": page, "bytes": nbytes, "threads": threads,
            "failed_sets": oks.count(False), "set_s": t_set, "durable_s": t_dur, "set_Bps": nbytes / t_set,
            "durable_Bps": nbytes / t_dur, "set_p50_s": lat[len(lat) // 2], "set_p99_s": lat[min(len(lat) - 1, int(0.99 * len(lat)))],
            "cached_after": left["cached"], "torch": HAVE_TORCH}


def syncfs(path):
    """syncfs(2) on path's filesystem (os has no wrapper); os.sync() where libc lacks it."""
    import ctypes
    fd = os.open(path, os.O_RDONLY)
    try:
        f = getattr(ctypes.CDLL(None, use_errno=True), "syncfs", None)
        if f is None or f(fd) != 0:
            os.sync()
    finally:
        os.close(fd)


def parse_size(v):
    s = str(v).strip().upper().removesuffix("IB").removesuffix("B")
    mult = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    return int(float(s[:-1]) * mult[s[-1]]) if s and s[-1] in mult else int(s)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="a directory on the disk tier's filesystem (emptied per variant)")
    ap.add_argument("--variant", action="append", choices=sorted(PATCHES), help="default: all three")
    ap.add_argument("--bytes", default="4G")
    ap.add_argument("--page-bytes", default=str(64 * 147456), help="one HiCache page (default Qwen3-8B, page size 64)")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    total, page = parse_size(a.bytes), parse_size(a.page_bytes)
    out = []
    work = tempfile.mkdtemp(prefix="hicache_pilot_")
    try:
        for v in a.variant or ["stock", "fdatasync", "sync_file_range"]:
            d = os.path.join(a.dir, f"hicache_pilot_{v}")
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
            try:
                r = measure(v, d, total, page, a.threads, work)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            print(json.dumps(r), flush=True)
            out.append(r)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


if __name__ == "__main__":
    main()
