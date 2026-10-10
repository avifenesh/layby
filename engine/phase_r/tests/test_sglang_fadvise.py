"""The HiCacheFile page-cache patch (engine/phase_r/sglang/hicache_fadvise.patch) leaves no page of a stored KV page
in the page cache, after a write and after a read.

The test applies the patch to SGLang v0.5.21's own hicache_storage.py (tests/fixtures/hicache_storage_v0.5.21.py,
unmodified from github.com/sgl-project/sglang tag v0.5.21, python/sglang/srt/mem_cache/hicache_storage.py, Apache-2.0,
sha256 f54c2e54fd2e7546a0814dba922f6264362049fcbf1e390dde738144838f5733), imports it with stubs for the two SGLang modules it needs (envs and the LRU evictor), and runs
HiCacheFile.set and HiCacheFile.get on a real file on a disk-backed filesystem. Residency is read with cachestat(2)
(mincore(2) on older kernels). The unpatched module runs the same steps and must leave pages cached, which shows the
test detects the problem the patch fixes.

Uses real torch when importable (CPU only), else a small numpy stand-in for the four tensor calls the module makes.
Needs a disk-backed directory: LAYBY_TEST_DIR (default ~/.cache/layby-phase-r-test); tmpfs cannot drop pages, so the
test skips there.

Run: python engine/phase_r/tests/test_sglang_fadvise.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.dirname(HERE)
sys.path.insert(0, PR)
import pagecache  # noqa: E402

FIXTURE = os.path.join(HERE, "fixtures", "hicache_storage_v0.5.21.py")
PATCH = os.path.join(PR, "sglang", "hicache_fadvise.patch")
PAGE_BYTES = 8 << 20   # one stored page of 8 MiB (Qwen3-8B at page size 64 is 9 MiB)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


def _numpy_torch():
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


def _load(path, store_dir):
    """Import hicache_storage.py from path with stubbed sglang.srt.environ and the LRU evictor."""
    for k in [k for k in sys.modules if k == "sglang" or k.startswith("sglang.")]:
        del sys.modules[k]
    if not HAVE_TORCH:
        sys.modules["torch"] = _numpy_torch()
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


def _tensor(nbytes, fill):
    import numpy as np
    a = (np.arange(nbytes, dtype=np.uint32) * 2654435761 + fill).astype(np.uint8)
    if HAVE_TORCH:
        import torch
        return torch.from_numpy(a)
    return sys.modules["torch"].Tensor(a)


def _bytes(t):
    return t.numpy().tobytes() if HAVE_TORCH else t.a.tobytes()


def _run(module_path, store_dir):
    """set one page, warm the file, get it back; residency of the stored file after each step."""
    m = _load(module_path, store_dir)
    cfg = m.HiCacheStorageConfig(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1, attn_cp_rank=0, attn_cp_size=1,
                                 is_mla_model=False, enable_storage_metrics=False, is_page_first_layout=True,
                                 model_name="Qwen/Qwen3-8B")
    store = m.HiCacheFile(cfg)
    src = _tensor(PAGE_BYTES, 7)
    assert store.set("k0", src)
    path = os.path.join(store_dir, "k0" + store.config_suffix + ".bin")
    after_set = pagecache.file_stat(path)
    with open(path, "rb") as f:   # an unrelated buffered read pulls the file into the cache
        while f.read(1 << 20):
            pass
    warmed = pagecache.cached_bytes(path)
    dst = _tensor(PAGE_BYTES, 0)
    got = store.get("k0", dst)
    after_get = pagecache.cached_bytes(path)
    return {"after_set": after_set["cached"], "dirty_after_set": after_set["dirty"], "warmed": warmed,
            "after_get": after_get, "roundtrip": got is not None and _bytes(dst) == _bytes(src),
            "method": after_set["method"]}


class FadviseTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = os.environ.get("LAYBY_TEST_DIR", os.path.expanduser("~/.cache/layby-phase-r-test"))
        os.makedirs(base, exist_ok=True)
        if _fstype(base) in ("tmpfs", "ramfs"):
            raise unittest.SkipTest(f"{base} is RAM-backed; page drop cannot be observed")
        cls.tmp = tempfile.mkdtemp(dir=base)
        pkg = os.path.join(cls.tmp, "pkg")
        os.makedirs(os.path.join(pkg, "sglang", "srt", "mem_cache"))
        cls.orig = os.path.join(pkg, "sglang", "srt", "mem_cache", "hicache_storage.py")
        shutil.copy(FIXTURE, cls.orig)
        cls.patched_pkg = os.path.join(cls.tmp, "patched")
        shutil.copytree(pkg, cls.patched_pkg)
        r = subprocess.run(["patch", "--forward", "-p2", "-d", cls.patched_pkg], stdin=open(PATCH),
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        cls.patched = os.path.join(cls.patched_pkg, "sglang", "srt", "mem_cache", "hicache_storage.py")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _store(self, name):
        d = os.path.join(self.tmp, name)
        os.makedirs(d)
        return d

    def test_patch_has_both_hunks(self):
        self.assertEqual(open(self.patched).read().count("layby-fadvise"), 2)

    def test_unpatched_leaves_pages_cached(self):
        r = _run(self.orig, self._store("store_orig"))
        self.assertTrue(r["roundtrip"])
        self.assertGreater(r["after_set"], PAGE_BYTES // 2, r)   # the written page sits in the cache
        self.assertGreater(r["after_get"], PAGE_BYTES // 2, r)   # and a read leaves it there

    def test_patched_drops_pages(self):
        r = _run(self.patched, self._store("store_patched"))
        self.assertTrue(r["roundtrip"])
        self.assertEqual(r["after_set"], 0, r)
        self.assertGreater(r["warmed"], PAGE_BYTES // 2, r)       # the warm-up read did cache it
        self.assertEqual(r["after_get"], 0, r)
        print(f"\n  patched ({r['method']}, torch={HAVE_TORCH}): cached after set {r['after_set']}, "
              f"after warm read {r['warmed']}, after get {r['after_get']}")


class ApplyScript(unittest.TestCase):
    """apply_patch.sh on a fake installed SGLang (the v0.5.21 file plus dist-info metadata on PYTHONPATH)."""

    def _site(self, version):
        site = tempfile.mkdtemp(dir=os.environ.get("LAYBY_TEST_DIR", os.path.expanduser("~/.cache/layby-phase-r-test")))
        d = os.path.join(site, "sglang", "srt", "mem_cache")
        os.makedirs(d)
        open(os.path.join(site, "sglang", "__init__.py"), "w").close()
        shutil.copy(FIXTURE, os.path.join(d, "hicache_storage.py"))
        md = os.path.join(site, f"sglang-{version}.dist-info")
        os.makedirs(md)
        with open(os.path.join(md, "METADATA"), "w") as f:
            f.write(f"Metadata-Version: 2.1\nName: sglang\nVersion: {version}\n")
        return site

    def _run(self, site, mode):
        env = dict(os.environ, PYTHONPATH=site)
        r = subprocess.run(["bash", os.path.join(PR, "sglang", "apply_patch.sh"), sys.executable, mode], env=env,
                           capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip()

    def test_apply_check_idempotent(self):
        site = self._site("0.5.21")
        self.assertEqual(self._run(site, "check")[0], 1)
        rc, out = self._run(site, "apply")
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.startswith("PATCHED 0.5.21"), out)
        self.assertTrue(self._run(site, "apply")[1].startswith("ALREADY_PATCHED"))
        rc, out = self._run(site, "check")
        self.assertEqual(rc, 0)
        with open(os.path.join(site, "sglang", "srt", "mem_cache", "hicache_storage.py")) as f:
            self.assertEqual(f.read().count("layby-fadvise"), 2)
        shutil.rmtree(site)

    def test_refuses_other_versions(self):
        site = self._site("0.5.20")
        rc, out = self._run(site, "apply")
        self.assertEqual(rc, 2)
        self.assertTrue(out.startswith("WRONG_VERSION"), out)
        shutil.rmtree(site)


def _fstype(path):
    best, fs = "", ""
    for line in open("/proc/self/mounts"):
        p = line.split()
        if path.startswith(p[1]) and len(p[1]) > len(best):
            best, fs = p[1], p[2]
    return fs


if __name__ == "__main__":
    unittest.main(verbosity=2)
