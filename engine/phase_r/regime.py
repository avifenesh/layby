"""Regime of a cell from two measured ratios (BENCH_PLAN section 1), fixed before any arm outcome is seen.

  r = disk restore speed / prefill speed, both in tokens per second:
      (read bytes/s on the tier's real read path / KV bytes per token) / the engine's prefill tok/s at the cell's load
  w = write-through's offered write rate / the disk's write bandwidth (bytes/s each)

  slow   if r < 0.5 or w > 1    (restoring is much slower than recomputing, or the disk cannot absorb write-through)
  fast   if r >= 2 and w < 0.5
  medium otherwise

Inputs: the pilot JSON (one per model and engine; produced by the pilot's microbenchmarks and C0 dry runs):
  {"model": "Qwen/Qwen3-8B", "engine": "vllm", "kv_bytes_per_token": 147456,
   "engine_rss_bytes": 12884901888,                       # engine RSS beyond the pinned tiers, peak over a C0 run
   "loads": {"heavy": {"prefill_tok_s": 21000.0, "wt_offered_write_Bps": 6.1e8},
             "light": {"prefill_tok_s": 24000.0, "wt_offered_write_Bps": 2.2e8}}}
and the disk probe (probe.py): read and write bytes/s, measured inside the arm's cgroup when io.max is set, so the
throttle is part of the number.
"""

SLOW_R, FAST_R = 0.5, 2.0
SLOW_W, FAST_W = 1.0, 0.5


def ratios(read_Bps, write_Bps, kv_bytes_per_token, prefill_tok_s, offered_write_Bps):
    restore_tok_s = read_Bps / kv_bytes_per_token
    r = restore_tok_s / prefill_tok_s
    w = offered_write_Bps / write_Bps if write_Bps > 0 else float("inf")
    return r, w


def classify(r, w):
    if r < SLOW_R or w > SLOW_W:
        return "slow"
    if r >= FAST_R and w < FAST_W:
        return "fast"
    return "medium"


def load_inputs(pilot, load):
    try:
        L = pilot["loads"][load]
        return float(pilot["kv_bytes_per_token"]), float(L["prefill_tok_s"]), float(L["wt_offered_write_Bps"])
    except KeyError as e:
        raise ValueError(f"pilot JSON lacks {e} for load {load!r}") from None


def cell_regime(pilot, load, read_Bps, write_Bps):
    bpt, pf, off = load_inputs(pilot, load)
    r, w = ratios(read_Bps, write_Bps, bpt, pf, off)
    return {"r": r, "w": w, "regime": classify(r, w), "read_Bps": read_Bps, "write_Bps": write_Bps,
            "kv_bytes_per_token": bpt, "prefill_tok_s": pf, "wt_offered_write_Bps": off, "load": load}


def bandwidth_bounds(pilot, load):
    """Read and write bandwidths (bytes/s) at the regime boundaries for this load, to choose io.max:
    slow holds when read < slow_read_below or write < slow_write_below; fast needs read >= fast_read_at_least and
    write > fast_write_above."""
    bpt, pf, off = load_inputs(pilot, load)
    return {"slow_read_below": SLOW_R * pf * bpt, "slow_write_below": off / SLOW_W,
            "fast_read_at_least": FAST_R * pf * bpt, "fast_write_above": off / FAST_W}


def check(declared, pilot, load, read_Bps, write_Bps):
    """(ok, detail): the cell's declared regime matches the one its measured ratios give."""
    d = cell_regime(pilot, load, read_Bps, write_Bps)
    d["declared"] = declared
    return d["regime"] == declared, d
