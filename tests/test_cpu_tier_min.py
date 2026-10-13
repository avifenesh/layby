"""The CPU tier holds one max-length disk restore beside one max-length parked write (owner, 2026-10-12). At 1.5
contexts pilot 2's Gemma restores failed over to recompute whenever a write held the tier."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bench", "phase_r"))
from sim_sizing import CPU_ALIGN, capacities, cpu_min_bytes  # noqa: E402

GEMMA = dict(tier_bytes_per_token=131714, tier_chunk_bytes=2621440)
QWEN = dict(tier_bytes_per_token=72472, tier_chunk_bytes=51380224)


def test_restore_beside_write_in_whole_chunks():
    per = -(-32768 * GEMMA["tier_bytes_per_token"] // GEMMA["tier_chunk_bytes"])    # 1,647 chunks per context
    assert cpu_min_bytes(GEMMA, 32768) == 2 * per * GEMMA["tier_chunk_bytes"]
    assert cpu_min_bytes(QWEN, 32768) == 2 * 47 * QWEN["tier_chunk_bytes"]


def test_minimum_binds_and_aligns():
    c = capacities(18.5e9, 5.74e9, cpu_min=cpu_min_bytes(GEMMA, 32768))
    assert c["cpu_at_min"] and c["cpu_bytes"] >= cpu_min_bytes(GEMMA, 32768) and c["cpu_bytes"] % CPU_ALIGN == 0
