import numpy as np
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.tiering.base import TransferJob, JobResult
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
import layby.vllm.fs_tier as F

stored, noio = [], []
FileSystemTierManager.submit_store = lambda self, j: stored.append(list(j.keys))
FileSystemTierManager.get_finished_jobs = lambda self: self._fin
class Pool:
    def enqueue_store(self, jid, n, tasks): noio.append(jid)
t = object.__new__(F.ParkFsTier)
t._on_disk, t._inflight, t._writing, t._loading, t._submit, t._mark, t._bytes, t._late = set(), set(), {}, {}, {}, None, 0, []
t._pool, t._block_size, t._fin = Pool(), 1 << 20, []
ctx = ReqContext(req_id="r", kv_transfer_params={"park_disk": True})
job = lambda i, keys: TransferJob(job_id=i, keys=keys, chunk_ids=np.arange(len(keys)), is_promotion=False, req_context=ctx)
t.submit_store(job(1, ["a", "b", "c"]))
t.submit_store(job(2, ["a", "b", "c", "d"]))          # a b c in flight: writes d only
assert stored == [["a", "b", "c"], ["d"]], stored
t._fin = [JobResult(job_id=1, success=True), JobResult(job_id=2, success=True)]
t.get_finished_jobs()
assert t._on_disk == {"a", "b", "c", "d"} and not t._inflight and not t._writing
t._fin = []
t.get_finished_jobs()                                    # empty batch: no error
t.submit_store(job(3, ["a", "b", "c", "d"]))          # all on disk: no I/O
assert noio == [3] and len(stored) == 2
print("dedup ok")

# write-through mode: every session's chunks are written unless the rule keeps the session off disk
from layby.vllm.state import STATE
STATE.write_mode = "through"
plain = lambda key: ReqContext(req_id="q", kv_transfer_params={"park_key": key})
t.submit_store(TransferJob(job_id=4, keys=["e"], chunk_ids=np.arange(1), is_promotion=False, req_context=plain("s1-3")))
assert stored[-1] == ["e"], stored
STATE.nowrite.add("s1")
t.submit_store(TransferJob(job_id=5, keys=["f"], chunk_ids=np.arange(1), is_promotion=False, req_context=plain("s1-4")))
assert stored[-1] == ["e"] and noio[-1] == 5
STATE.write_mode = "rule"
print("through ok")

# arm PT: write_all writes every stored chunk with no rule (default write mode), the default tier writes none of them
assert STATE.write_mode == "rule"
t.submit_store(TransferJob(job_id=6, keys=["g"], chunk_ids=np.arange(1), is_promotion=False, req_context=plain("s2-0")))
assert stored[-1] == ["e"] and noio[-1] == 6, "the default tier wrote a chunk the rule did not place"
t._write_all = True
t.submit_store(TransferJob(job_id=7, keys=["g", "e"], chunk_ids=np.arange(2), is_promotion=False, req_context=plain("s2-0")))
assert stored[-1] == ["g"], stored                       # e is in flight from job 4: deduplicated
print("write_all ok")
