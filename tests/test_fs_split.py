"""ParkFsTier runs a job's chunk files on several I/O threads, not one (vLLM 0.31 tiering manager, CPU primary with
ParkCachePolicy, ParkFsTier; CPU only, real files under ~/.cache).

vLLM 0.31's fs tier enqueues each load (promotion) or store job as a single task: one thread opens, reads or writes
and closes the job's files one after another while the other read threads idle. Phase R, Gemma fast heavy (wrx,
2026-11-12): restores ran at about 0.7 GB/s at the median, on a disk capped at 1.35 GB/s for reads.

Before the change: an 8-chunk store and an 8-chunk promotion each go to the pool as 1 task. After it: 4 tasks each
(n_read_threads = n_write_threads = 4), the promoted bytes equal the written ones, and a promotion that fails on its
6th file keeps the 5 chunks before it in the CPU tier and drops the rest from the disk index, as the base's single
task does.

Run: PYTHONPATH=. python tests/test_fs_split.py
"""
import os
import shutil
import tempfile

from scaffold_vllm import B, build, check, end_schedule, hashes, key, settle
from vllm.v1.kv_offload.base import LookupResult, ReqContext

root = tempfile.mkdtemp(prefix="pp_split_", dir=os.path.expanduser("~/.cache"))
fs = None
try:
    conn, cs, mgr, prim, fs, pol = build(8, root, n_threads=4)
    calls = []
    for name in ("enqueue_load", "enqueue_store"):
        real = getattr(fs._pool, name)
        setattr(fs._pool, name, (lambda real, name: lambda job_id, n, tasks: (calls.append((name, n)), real(job_id, n, tasks)))(real, name))
    view = prim.get_kv_memoryview().cast("B")      # flat bytes: chunk i at [i * B, (i + 1) * B)

    def put(sess, n, tag):
        ks = [key(h) for h in hashes(sess, n)]
        ctx = ReqContext(req_id=f"r-{sess}", kv_transfer_params={"park_key": sess})
        mgr.on_new_request(ctx)
        out = mgr.prepare_store(ks, ctx)
        for i, cid in enumerate(out.store_spec.chunk_ids):
            view[int(cid) * B:(int(cid) + 1) * B] = bytes([tag + i]) * B     # each chunk's bytes name it
        mgr.complete_store(out.keys_to_store, ctx)
        mgr.on_request_finished(ctx)
        return ks

    def promote(ks, rid):
        ctx = ReqContext(req_id=rid, kv_transfer_params={})
        mgr.on_new_request(ctx)
        res = [mgr.lookup(k, ctx) for k in ks]
        assert LookupResult.MISS not in res, res
        end_schedule(mgr)
        settle(mgr, fs)
        return ctx

    A = put("a", 8, 10)
    fs.queue_write(A)
    settle(mgr, fs); settle(mgr, fs)
    assert all(k in fs._on_disk for k in A), "a not on disk"
    assert ("enqueue_store", 4) in calls, f"the 8-chunk store was not split into 4 tasks: {calls}"
    put("c", 8, 100)                                  # evicts a from the CPU tier, overwrites its slots
    settle(mgr, fs)                                   # c's cascade to the disk tier ends: c is evictable
    assert not any(k in pol.chunks for k in A)
    calls.clear()
    promote(A, "ra")
    assert calls == [("enqueue_load", 4)], f"the 8-chunk promotion was not split into 4 tasks: {calls}"
    for i, k in enumerate(A):
        assert k in pol.chunks, f"chunk {i} of a did not land"
        cid = pol.chunks[k].chunk_id
        assert bytes(view[cid * B:(cid + 1) * B]) == bytes([10 + i]) * B, f"chunk {i} of a came back with other bytes"
    check(prim, pol)

    # a promotion whose 6th file is gone: the first 5 chunks stay, the rest leave the disk index
    Bk = put("b", 8, 50)
    fs.queue_write(Bk)
    settle(mgr, fs); settle(mgr, fs)
    put("d", 8, 150)
    settle(mgr, fs)
    os.remove(fs.file_mapper.get_file_name(Bk[5]))
    promote(Bk, "rb")
    assert all(k in pol.chunks for k in Bk[:5]), [k in pol.chunks for k in Bk]
    assert not any(k in pol.chunks for k in Bk[5:]), [k in pol.chunks for k in Bk]
    assert all(k in fs._on_disk for k in Bk[:5]) and not any(k in fs._on_disk for k in Bk[5:])
    for i, k in enumerate(Bk[:5]):
        c = pol.chunks[k].chunk_id
        assert bytes(view[c * B:(c + 1) * B]) == bytes([50 + i]) * B
    check(prim, pol)
finally:
    if fs is not None:
        fs.shutdown()
    shutil.rmtree(root, ignore_errors=True)
print("fs split ok: stores and promotions run as 4 tasks, bytes intact, a failed file keeps the chunks before it")
