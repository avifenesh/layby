"""Two returning sessions restored together on a sliding-window model both finish under ParkConnector on vLLM 0.31.0
with PR #60045 (engine/phase_r/vllm/pr60045_swa_admission.patch), the engine every Phase R vLLM arm runs. CPU only.

Measured in the Phase R pilot 2 K arm on Gemma 4 31B (5 sliding-window groups, window 1024, block 16, and one global
group, block 32; 2,188 GPU blocks): two returning sessions waited while their prefixes came back from disk, both
promotions landed in the same disk burst, and the scheduler admitted both CPU-to-GPU loads in one pass. From then on:
"Running: 0 reqs, Waiting: 2 reqs, Deferred: 2 reqs, GPU KV cache usage: 92.5%", for 40+ minutes. The cause is in
vLLM, not Layby: stock 0.31.0 counts 0 sliding-window blocks for a long prefix hit in its full-sequence check and in
the in-flight reservation (Scheduler._request_remaining_blocks), so it admits the second load into a pool neither
request can take its next chunk from, and the waiting loop never preempts. The fix is vLLM PR #60045, applied to every
arm by the orchestrator; Layby has no admission gate of its own (stock admission plus the fix, as in the other arms).

The test drives ParkConnector (over a stand-in for the offloading scheduler: every prefix is a CPU-tier hit, loaded
asynchronously) through the scheduler's KV calls on vLLM's real KV cache manager (the driver from
engine/phase_r/tests/test_vllm_swa_admission.py). On the patched engine the second load waits while the first holds its
blocks, and both finish. On the stock engine the same run stalls with both loads holding blocks, which shows the test
detects the problem. The patched and stock methods are compiled from the installed vLLM file in memory; site-packages
are not touched.

Run: PYTHONPATH=. python tests/test_park_vllm_swa_admit.py
"""
import os
import sys
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "engine", "phase_r", "tests"))
import test_vllm_swa_admission as V  # noqa: E402

import layby.vllm.connector as C  # noqa: E402
from layby.vllm.state import STATE  # noqa: E402


class Base:
    """The offloading scheduler, reduced to its answers: every prefix is a CPU-tier hit of HIT tokens."""

    def __init__(self):
        self._req_status, self.config = {}, NS(kv_group_configs=[])

    def on_new_request(self, request):
        pass

    def get_num_new_matched_tokens(self, request, num_computed_tokens, max_num_new_tokens=None):
        return V.HIT, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output):
        return NS(load_jobs={})

    def request_finished(self, request):
        return False, None


def connector(kv):
    STATE.__init__()
    STATE.block_tokens = STATE.hash_tokens = STATE.chunk_tokens = 16
    conn = object.__new__(C.ParkConnector)
    conn.connector_scheduler, conn.connector_worker = Base(), None
    conn._kv_cache_manager = kv
    conn.__dict__["_bounding_group_ids"] = ()          # every group is offloaded: no bound on the load
    # the request-tracking state ParkConnector.__init__ sets
    conn._waiting, conn._deferred, conn._local, conn._giveups = {}, {}, {}, {}
    conn._loads, conn._finished, conn._early, conn._hints_on = {}, {}, {}, False
    conn._last_step, conn._last_prune = None, float("inf")
    return conn


def run(patched):
    with V.engine(patched):
        kv = V.gemma_kv()
        A, B = V.request("A", 1), V.request("B", 2)
        conn = connector(kv)
        for r in (A, B):
            conn.on_new_request(r)
        e = V.Engine(kv, [A, B], connector=conn)
        e.step()
        first = (V.window_held(kv, A), V.window_held(kv, B), kv.block_pool.get_num_free_blocks())
        return first, e.run(), e.done


# --- the engine every vLLM arm runs: the second load waits for the first, both finish ---------------------------------
(a_held, b_held, free), out, done = run(patched=True)
assert a_held == [64] * 5 and b_held == [0] * 5, (a_held, b_held, "B's load must wait while A's holds its blocks")
assert out == "done" and sorted(done) == ["A", "B"], out

# --- the detector: stock vLLM 0.31.0 admits both loads and nothing can run --------------------------------------------
(a_held, b_held, free), out, done = run(patched=False)
assert a_held == b_held == [64] * 5 and free == V.N_BLOCKS - 1 - 2 * 1006, (a_held, b_held, free)
assert out.startswith("stuck") and not done, out
print(f"swa admit ok: on vLLM 0.31.0 + PR #60045 both restores finish under ParkConnector; stock vLLM: {out}")
