"""Layby's vLLM modules log under vLLM's "vllm" logger tree. vLLM attaches its handler to "vllm" only, so a logger
named after the module ("layby.vllm.freq_tier") prints nothing: in the 2026-10-11 pilot the T2 arm's write-count
lines (FreqFsTier stats) and every other Layby log line were missing from the server logs. Static check, no vLLM needed.

  python -m pytest -q tests/test_vllm_logger_names.py
"""
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "layby", "vllm")


def test_every_init_logger_is_under_the_vllm_tree():
    bad = []
    for f in sorted(os.listdir(ROOT)):
        if not f.endswith(".py"):
            continue
        for m in re.finditer(r"init_logger\(([^)]*)\)", open(os.path.join(ROOT, f)).read()):
            if m.group(1).strip() != '"vllm." + __name__':
                bad.append((f, m.group(1)))
    assert not bad, bad
