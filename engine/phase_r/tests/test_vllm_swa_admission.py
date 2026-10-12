"""vLLM PR #60045 (engine/phase_r/vllm/pr60045_swa_admission.patch) keeps two async KV loads on a sliding-window model
from leaving the engine with nothing it can run. No Layby code: vLLM's real KV cache manager and the scheduler's own
reservation (Scheduler._request_remaining_blocks), with the stock OffloadingConnector's answer for a returning session
(its prefix is a CPU-tier hit, loaded asynchronously). CPU only.

KV layout of Gemma 4 31B: 5 sliding-window groups (block 16, window 1024) and one full-attention group (block 32),
2,188 blocks, --max-num-batched-tokens 1024 with async scheduling (2,048 tokens in flight). Two returning sessions of
23,000 tokens, 21,952 of them restored. Stock 0.31.0 counts 0 window blocks for such a hit, admits both loads (2 x
1,006 blocks) and then neither next chunk fits: 0 running, both waiting, and the waiting loop never preempts. With the
patch the second load waits until the first runs, and both finish. The patched count equals the window blocks the
allocator holds at the request's peak.

The stock and patched get_num_blocks_to_allocate are compiled from the installed vLLM file (the patch applied or
reversed in memory, apply_patch.method), so site-packages are not touched. PR60045_TEST_STOCK=1 runs the patched-engine
tests on the stock method instead: they fail, which shows they detect the bug. apply_patch.py itself is tested on a fake
installed vLLM (the installed 0.31.0 file plus dist-info metadata on PYTHONPATH).

Needs vLLM 0.31.0 importable (CPU torch is enough). Run: python engine/phase_r/tests/test_vllm_swa_admission.py
"""
import contextlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS

HERE = os.path.dirname(os.path.abspath(__file__))
PR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PR, "vllm"))
import apply_patch  # noqa: E402

try:
    import torch
    from vllm import SamplingParams
    from vllm.utils.hashing import sha256
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, SlidingWindowSpec
    from vllm.v1.request import Request, RequestStatus
    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False

N_BLOCKS, MAX_LEN, IN_FLIGHT, CHUNK = 2188, 32768, 2 * 1024, 1024
PROMPT, HIT = 23000, 21952


@contextlib.contextmanager
def engine(patched):
    """vLLM with (or without) PR #60045's get_num_blocks_to_allocate."""
    if os.environ.get("PR60045_TEST_STOCK") == "1":
        patched = False
    real = SingleTypeKVCacheManager.get_num_blocks_to_allocate
    SingleTypeKVCacheManager.get_num_blocks_to_allocate = apply_patch.method(patched)
    try:
        yield
    finally:
        SingleTypeKVCacheManager.get_num_blocks_to_allocate = real


def gemma_kv():
    swa = [KVCacheGroupSpec([f"swa{g}"], SlidingWindowSpec(block_size=16, num_kv_heads=16, head_size=256,
                                                            dtype=torch.bfloat16, sliding_window=1024))
           for g in range(5)]
    full = KVCacheGroupSpec(["full"], FullAttentionSpec(block_size=32, num_kv_heads=4, head_size=512,
                                                        dtype=torch.bfloat16))
    cfg = KVCacheConfig(num_blocks=N_BLOCKS, kv_cache_tensors=[], kv_cache_groups=swa + [full])
    return KVCacheManager(cfg, max_model_len=MAX_LEN, scheduler_block_size=32, hash_block_size=16,
                          max_in_flight_tokens=IN_FLIGHT)


def request(rid, seed):
    init_none_hash(sha256)
    r = Request(rid, [seed * 100000 + i for i in range(PROMPT)], SamplingParams(max_tokens=16), None,
                block_hasher=get_request_block_hasher(16, sha256))
    r.status = RequestStatus.WAITING
    return r


def remaining(kv, r):
    """The scheduler's own in-flight reservation (Scheduler._request_remaining_blocks, no spec decode)."""
    return Scheduler._request_remaining_blocks(NS(kv_cache_manager=kv, max_model_len=MAX_LEN,
                                                  _spec_decode_step_blocks=lambda: 0), r)


def window_held(kv, r):
    return [sum(not b.is_null for b in m.req_to_blocks.get(r.request_id, ()))
            for m in kv.coordinator.single_type_managers[:5]]


class Engine:
    """Scheduler.schedule()'s KV calls for this workload, step by step, with two batches in flight: a waiting request
    with no blocks asks the connector (a CPU-tier hit of HIT tokens, loaded asynchronously) and is admitted with the
    reservation of the loads in flight; a loaded or running request takes its next chunk; the waiting loop stops at
    the first request it cannot allocate (it never preempts). A load completes at the end of its step. A request
    leaves (its blocks freed) once its prompt is computed and its last chunk is processed."""

    def __init__(self, kv, reqs, connector=None):
        self.kv, self.waiting, self.running, self.inflight = kv, list(reqs), [], []
        self.connector = connector
        self.last = {}            # request id -> tokens of the previous step, still in flight
        self.done, self.peak = [], {}

    def ext_hit(self, r):
        if self.connector is None:
            return HIT, True
        return self.connector.get_num_new_matched_tokens(r, 0)

    def step(self):
        kv, scheduled, loading = self.kv, {}, []
        for r in list(self.running):
            n = min(CHUNK, r.num_tokens - r.num_computed_tokens)
            if n and kv.allocate_slots(r, n) is not None:
                scheduled[r.request_id] = (r, n)
        for r in list(self.waiting):
            if r.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                continue
            if r.num_computed_tokens == 0:       # fresh: the connector's async hit
                ext, load_async = self.ext_hit(r)
                if ext is None:
                    continue                     # the connector asks to wait (no blocks held)
                assert load_async and ext == HIT, (ext, load_async)
                reserved = sum(remaining(kv, x) for x in self.inflight)
                blocks = kv.allocate_slots(r, 0, num_external_computed_tokens=ext, delay_cache_blocks=True,
                                           full_sequence_must_fit=True, reserved_blocks=reserved,
                                           has_scheduled_reqs=bool(self.running))
                if blocks is None:
                    break
                if self.connector is not None:
                    self.connector.update_state_after_alloc(r, blocks, ext)
                r.status, r.num_computed_tokens = RequestStatus.WAITING_FOR_REMOTE_KVS, ext
                self.inflight.append(r)
                loading.append(r)
                continue
            n = min(CHUNK, r.num_tokens - r.num_computed_tokens)   # loaded: its next chunk
            if kv.allocate_slots(r, n, full_sequence_must_fit=True, has_scheduled_reqs=bool(self.running)) is None:
                break
            self.waiting.remove(r)
            self.running.append(r)
            r.status = RequestStatus.RUNNING
            scheduled[r.request_id] = (r, n)
        if self.connector is not None:
            self.connector.build_connector_meta(NS(
                num_scheduled_tokens={i: n for i, (_, n) in scheduled.items()}, scheduled_new_reqs=[],
                scheduled_cached_reqs=NS(is_context_phase=lambda rid: True), scheduled_encoder_input_stats=None,
                total_num_scheduled_tokens=sum(n for _, n in scheduled.values()), finished_req_ids=set(),
                preempted_req_ids=set()))
        for r, n in scheduled.values():          # _update_after_schedule
            r.num_computed_tokens += n
            r.num_in_flight_tokens += n
            if r.num_computed_tokens < r.num_tokens and r not in self.inflight:
                self.inflight.append(r)          # _inflight_prefills: still prefilling after this chunk
            elif r.num_computed_tokens >= r.num_tokens and r in self.inflight:
                self.inflight.remove(r)
        for r in self.running + self.waiting:
            p = self.peak.setdefault(r.request_id, [0] * 5)
            self.peak[r.request_id] = [max(a, b) for a, b in zip(p, window_held(self.kv, r))]
        for rid, (r, n) in self.last.items():    # the previous step's output
            r.num_in_flight_tokens -= n
            if r.num_computed_tokens >= r.num_tokens and r.num_in_flight_tokens == 0:
                self.finish(r)
        self.last = {i: x for i, x in scheduled.items() if x[0] in self.running}
        for r in loading:                        # _update_waiting_for_remote_kv
            kv.cache_blocks(r, r.num_computed_tokens)
            r.status = RequestStatus.WAITING
        return scheduled, loading

    def finish(self, r):
        self.running.remove(r)
        self.kv.free(r)
        if self.connector is not None:
            self.connector.request_finished(r, [])
        self.done.append(r.request_id)

    def run(self, steps=200):
        for i in range(steps):
            n_done = len(self.done)
            scheduled, loading = self.step()
            if not scheduled and not loading and len(self.done) == n_done and not self.running:
                # nothing ran, loaded or finished, and nothing is running: the next step is the same
                return f"stuck at step {i}: 0 running, {len(self.waiting)} waiting, {self.kv.usage:.1%} KV used, " \
                       f"{self.kv.block_pool.get_num_free_blocks()} blocks free"
            if not self.waiting and not self.running:
                return "done"
        return "out of steps"


@unittest.skipUnless(HAVE_VLLM, "needs vLLM 0.31.0")
class Admission(unittest.TestCase):
    def test_stock_admits_both_loads_and_stalls(self):
        # the detector: stock 0.31.0 reaches the state seen on the Gemma K arm
        with engine(patched=False):
            kv = gemma_kv()
            A, B = request("A", 1), request("B", 2)
            e = Engine(kv, [A, B])
            e.step()
            self.assertEqual(remaining(kv, A), 33)                 # the full-attention group only
            self.assertEqual(N_BLOCKS - 1 - kv.block_pool.get_num_free_blocks(), 2 * 1006)
            self.assertTrue(e.run().startswith("stuck at step 0"))

    def test_second_load_waits_and_both_finish(self):
        with engine(patched=True):
            kv = gemma_kv()
            A, B = request("A", 1), request("B", 2)
            e = Engine(kv, [A, B])
            e.step()
            self.assertEqual(window_held(kv, A), [64] * 5)
            self.assertEqual(remaining(kv, A), 33 + 5 * 66, "reservation: A's next window blocks are counted")
            self.assertEqual(window_held(kv, B), [0] * 5, "B's load waits while A's holds its blocks")
            out = e.run()
            self.assertEqual(out, "done", out)
            self.assertEqual(sorted(e.done), ["A", "B"])

    def test_admission_count_matches_the_allocators_peak(self):
        # what the full-sequence check counts per window group at admission is what the allocator holds at the
        # request's peak (stepped chunk by chunk with two batches in flight)
        with engine(patched=True):
            kv = gemma_kv()
            A = request("A", 1)
            counted = [m.get_num_blocks_to_allocate("A", A.num_tokens, [], HIT, 0, A.num_tokens,
                                                    apply_admission_cap=True)
                       for m in kv.coordinator.single_type_managers[:5]]
            e = Engine(kv, [A])
            self.assertEqual(e.run(), "done")
            self.assertEqual(e.peak["A"], counted)
            self.assertEqual(counted, [130] * 5)


@unittest.skipUnless(HAVE_VLLM, "needs vLLM 0.31.0")
class ApplyScript(unittest.TestCase):
    """apply_patch.py on a fake installed vLLM (the installed 0.31.0 file plus dist-info metadata on PYTHONPATH)."""

    def _site(self, version):
        site = tempfile.mkdtemp(dir=os.environ.get("LAYBY_TEST_DIR", os.path.expanduser("~/.cache/layby-phase-r-test")))
        d = os.path.join(site, "vllm", "v1", "core")
        os.makedirs(d)
        open(os.path.join(site, "vllm", "__init__.py"), "w").close()
        target, hunks = apply_patch.parse()
        with open(apply_patch.installed_file(target)) as f:
            text = f.read()
        if apply_patch.state(text, hunks) == "patched":
            text = apply_patch.apply_text(text, [(n, o) for o, n in hunks])
        with open(os.path.join(site, target), "w") as f:
            f.write(text)
        md = os.path.join(site, f"vllm-{version}.dist-info")
        os.makedirs(md)
        with open(os.path.join(md, "METADATA"), "w") as f:
            f.write(f"Metadata-Version: 2.1\nName: vllm\nVersion: {version}\n")
        self.addCleanup(shutil.rmtree, site)
        return site, os.path.join(site, target)

    def _run(self, site, mode):
        env = dict(os.environ, PYTHONPATH=site)
        r = subprocess.run([sys.executable, os.path.join(PR, "vllm", "apply_patch.py"), mode], env=env,
                           capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout.strip()

    def test_apply_check_idempotent(self):
        site, f = self._site("0.31.0")
        rc, out = self._run(site, "check")
        self.assertEqual(rc, 1, out)
        self.assertTrue(out.startswith("NOT_PATCHED 0.31.0 has=none"), out)
        rc, out = self._run(site, "apply")
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.startswith("PATCHED 0.31.0 pr60045"), out)
        self.assertIn(f, out)
        rc, out = self._run(site, "apply")
        self.assertEqual(rc, 0, out)
        self.assertTrue(out.startswith("ALREADY_PATCHED 0.31.0 pr60045"), out)
        self.assertEqual(self._run(site, "check")[0], 0)
        with open(f) as fh:
            text = fh.read()
        self.assertIn("num_live_blocks = min(", text)
        self.assertNotIn("num_required_blocks, self.max_admission_blocks_per_request", text)

    def test_refuses_other_versions_and_edited_files(self):
        site, f = self._site("0.31.1")
        rc, out = self._run(site, "apply")
        self.assertEqual((rc, out.split()[0]), (2, "WRONG_VERSION"), out)
        site, f = self._site("0.31.0")
        with open(f) as fh:
            text = fh.read()
        with open(f, "w") as fh:
            fh.write(text.replace("num_req_blocks = len(self.req_to_blocks.get(request_id, ()))", "num_req_blocks = 0"))
        rc, out = self._run(site, "apply")
        self.assertEqual((rc, out.split()[0]), (2, "PATCH_DOES_NOT_APPLY"), out)


if __name__ == "__main__":
    unittest.main()
