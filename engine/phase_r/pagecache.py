"""Page-cache residency of files: cachestat(2) (Linux 6.5+), with mincore(2) as the fallback.

cached_bytes(path) -> bytes of the file held in the page cache now.
dir_cached_bytes(root) -> (files, file bytes, cached bytes, dirty bytes) over every regular file under root.

Used by the telemetry sampler (cached bytes of the disk tier during an arm) and by the SGLang fadvise test.
"""
import ctypes
import ctypes.util
import errno
import mmap
import os

PAGE = os.sysconf("SC_PAGE_SIZE")
_NR_CACHESTAT = 451   # the same number on x86_64 and aarch64 (unified syscall table since 6.5)
_libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
_libc.syscall.restype = ctypes.c_long
_libc.mmap.restype = ctypes.c_void_p
_libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
_libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]


class _Range(ctypes.Structure):
    _fields_ = [("off", ctypes.c_uint64), ("len", ctypes.c_uint64)]


class _Stat(ctypes.Structure):
    _fields_ = [("nr_cache", ctypes.c_uint64), ("nr_dirty", ctypes.c_uint64), ("nr_writeback", ctypes.c_uint64),
                ("nr_evicted", ctypes.c_uint64), ("nr_recently_evicted", ctypes.c_uint64)]


_has_cachestat = None


def cachestat_fd(fd):
    """dict of page counts from cachestat(2) over the whole file, or None if the kernel lacks it."""
    global _has_cachestat
    if _has_cachestat is False:
        return None
    rng, st = _Range(0, 0), _Stat()
    rc = _libc.syscall(_NR_CACHESTAT, ctypes.c_int(fd), ctypes.byref(rng), ctypes.byref(st), ctypes.c_uint(0))
    if rc != 0:
        e = ctypes.get_errno()
        if e in (errno.ENOSYS, errno.EPERM):
            _has_cachestat = False
            return None
        raise OSError(e, os.strerror(e))
    _has_cachestat = True
    return {k: getattr(st, k) for k, _ in _Stat._fields_}


def mincore_fd(fd, size):
    """Resident page count of the file by mapping it and calling mincore(2)."""
    if size == 0:
        return 0
    addr = _libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    if addr in (None, ctypes.c_void_p(-1).value):
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e))
    try:
        n = (size + PAGE - 1) // PAGE
        vec = (ctypes.c_ubyte * n)()
        if _libc.mincore(addr, size, vec) != 0:
            e = ctypes.get_errno()
            raise OSError(e, os.strerror(e))
        return sum(b & 1 for b in vec)
    finally:
        _libc.munmap(addr, size)


def file_stat(path):
    """{'cached': bytes, 'dirty': bytes, 'method': 'cachestat'|'mincore'} for one file."""
    fd = os.open(path, os.O_RDONLY)
    try:
        st = cachestat_fd(fd)
        if st is not None:
            return {"cached": st["nr_cache"] * PAGE, "dirty": st["nr_dirty"] * PAGE, "method": "cachestat"}
        return {"cached": mincore_fd(fd, os.fstat(fd).st_size) * PAGE, "dirty": None, "method": "mincore"}
    finally:
        os.close(fd)


def cached_bytes(path):
    return file_stat(path)["cached"]


def dir_cached_bytes(root):
    """Totals over every regular file under root; files that vanish while walking (an engine evicting) are skipped."""
    files = size = cached = dirty = 0
    for d, _, names in os.walk(root):
        for n in names:
            p = os.path.join(d, n)
            try:
                s = os.stat(p)
                st = file_stat(p)
            except (FileNotFoundError, PermissionError):
                continue
            files += 1; size += s.st_size; cached += st["cached"]; dirty += st["dirty"] or 0
    return {"files": files, "bytes": size, "cached": cached, "dirty": dirty}


if __name__ == "__main__":
    import json, sys
    for p in sys.argv[1:]:
        print(json.dumps({"path": p, **(dir_cached_bytes(p) if os.path.isdir(p) else file_stat(p))}))
