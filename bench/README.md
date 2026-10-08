# ReturnBench

ReturnBench scores where a serving engine should keep a session's KV cache while it waits for the
session's next request. Agent and chat sessions come back: a tool result in 50 ms, a person in two
minutes, sometimes never. Between turns the KV can stay on the GPU, sit in a CPU tier, go to disk, or be
dropped and recomputed. The choice decides the time to first token (TTFT) of the turn that comes back.

The bench replays real sessions from four public trace datasets through a simulated engine and reports
p50 and p95 TTFT of returning turns, as a ratio to vLLM's default (C0). Everything runs on a CPU. The
full reference table takes about 6 minutes on 8 cores.

```
pip install -e .
returnbench eval --rules wt cost orc          # the 33-cell reference table
returnbench eval --rules wt cost --cells wildchat:120 swechat:18 --workloads 2 --seeds 1   # a quick look
```

## What is measured

A **cell** is a pool, a session count n and a disk speed. A **workload** draws n sessions from the pool
and starts them uniformly over 600 s; each session then runs closed loop: a turn is sent, it finishes,
the session waits its real idle gap, the next turn is sent. Each workload runs under 3 engine seeds
(step-time jitter). Per rule and workload: the median over seeds of p50 and p95 TTFT of every turn after
a session's first. Per cell: the geometric mean over 8 workloads of the ratio to C0. The summary is the
geometric mean over cells, overall and by disk speed, plus the same against write-through-to-disk (wt).

The reference set is 33 cells: 11 (pool, n) pairs times disks of 0.5, 1.5 and 3 GB/s.

| Pool | n |
|---|---|
| swechat | 6, 12, 18 |
| tracelab_claude | 12, 18, 25 |
| wildchat | 60, 120, 211 |
| swechat + tracelab_claude + wildchat | 18, 36 |

## The engine model

`sim/engine_sim.py` is an iteration-level model of Qwen3-8B on vLLM 0.30 with chunked prefill:

- GPU KV of 116,496 tokens (16 GiB). A finished request's blocks stay cached; new allocations evict the
  least recently freed session first, tail blocks first.
- A 28 GiB CPU tier, written through at finish, loaded back over one 20 GB/s link.
- A disk tier at 0.5, 1.5 or 3 GB/s. A disk hit is promoted to the CPU tier, then loaded to the GPU.
  Writes and reads share the disk link.
- FCFS scheduling with a 2,048-token budget per step. A request that does not fit blocks the queue.
- Step time: a fixed 25 ms, plus decode KV reads at 600 GB/s, plus 1/6000 s per prefilled token, with
  5% lognormal jitter. These are the calibration constants of the Layby work (`CAL` in
  `sim/eval_pop.py`).

KV is 144 KiB per token (Qwen3-8B, bf16). Every policy sees the same engine; only placement differs.

## Pools

| Pool | Source | Sessions | Turns | Users | Kinds | Median gap | Median context | Curves |
|---|---|---|---|---|---|---|---|---|
| `swechat` | SWE-chat (Claude Code sessions) | 60 | 2,044 | 5 | 1,904 tool, 140 human | 0.0 s | 25.9K | yes |
| `tracelab_claude` | TraceLab (Claude Code) | 25 | 655 | 2 | 607 tool, 48 human | 0.1 s | 24.1K | yes |
| `wildchat` | WildChat-4.8M (ChatGPT) | 211 | 1,746 | 132 | 1,746 human | 69 s | 3.2K | yes |
| `copilot` | GitHub Copilot coding agent 2026 | 200 | 5,542 | 200 (one per session) | 4,691 tool, 540 human, 311 workflow | 0.8 s | 24.0K | no |

Sessions are test-split sessions of held-out users: users the Layby-Dwell model never saw in training
or validation (Copilot has no user ids, so its pool takes test-split sessions of any user; the split is
chronological, so they all come after every training session). Each pool is in `bench/pools/POOL.json`.
Sources, versions and licenses: [bench/LICENSES.md](LICENSES.md).

### Schema

```json
{"pool": "swechat", "gap_cap": null, "ctx_cap": 32768,
 "sessions": [
   {"id": "s000", "pool": "swechat", "seed": 1000, "start": 382.18, "user": "swechat:u000",
    "turns": [
      {"new_tokens": 15913, "out_len": 128, "ctx": 16041, "kind": "tool", "tool": "Read", "prog": null,
       "gap_after": 0.014, "surv_v6": [0.028, 0.013, "... 15 values"]}]}]}
```

- `new_tokens`: prompt tokens the turn adds; `out_len`: tokens it generates (384 after a human wait,
  128 after a tool wait); `ctx`: the context after the turn.
- `kind`: what the session waits for after this turn: `human`, `tool` or `workflow` (an agent
  continuing on its own).
- `tool`: the tool the model just called; `prog`: the first program of a shell command.
- `gap_after`: seconds until the session's next request. `null` on the last turn. **This is the answer.
  Only the oracle reads it.**
- `surv_v6`: P(gap > h) at h = 0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 120, 300, 600, 1800 s, from
  Layby-Dwell, computed from what the server sees when the turn ends.
- `start`, `seed`: the replay start offset and a per-session seed (the eval redraws starts).

## Policies

| Name | What it does |
|---|---|
| `C0` | vLLM's default: LRU GPU cache, write-through LRU CPU tier, no disk tier. Every ratio is against it. |
| `wt` | Every session written to disk at finish, LRU everywhere. What vLLM's tiered offloading and SGLang HiCache write-through do. |
| `cost` | The Layby cost rule: the cheapest in expectation of none, ins, park and drop, from the survival curve and costs the engine measures while it runs. No fitted constants. `sim/rule.py`. |
| `orc` | The cost rule fed a step curve at the true gap: perfect prediction, same costs. |
| `stk` | Thresholds on the curve with 7 constants tuned in simulation. |
| `cont` | Continuum (arXiv 2511.02230): GPU pin after a tool call for a TTL from the per-tool gap CDF. |
| `cj` | Choi and Joshi (arXiv 2608.30830): GPU hold until the break-even time. |
| `ka` | Serverless in the Wild (ATC '20): GPU keep-alive to the user's 99th percentile gap. |
| `c_nopark`, `c_disk`, `o_disk` | Ablations of the cost rule and the oracle over option subsets. |

## Reference results

`returnbench eval --rules wt stk cost orc`, 33 cells, 8 workloads, 3 seeds. Geometric mean over cells
of the p95 TTFT ratio. Rows: `reference/rows.jsonl`; per cell: `reference/table.txt`.

| Rule | p95 vs C0 | Worst cell | @0.5 GB/s | @1.5 | @3 | p95 vs wt | @0.5 | @1.5 | @3 |
|---|---|---|---|---|---|---|---|---|---|
| C0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.46 | 1.01 | 1.68 | 1.84 |
| wt | 0.68 | 1.46 | 0.99 | 0.60 | 0.54 | 1.00 | 1.00 | 1.00 | 1.00 |
| stk | 0.78 | 1.65 | 0.96 | 0.72 | 0.68 | 1.14 | 0.97 | 1.21 | 1.24 |
| cost | 0.66 | 1.04 | 0.82 | 0.61 | 0.57 | 0.96 | 0.83 | 1.03 | 1.04 |
| orc | 0.60 | 1.00 | 0.71 | 0.57 | 0.53 | 0.88 | 0.72 | 0.95 | 0.98 |

How to read it:

- Most of the gain over C0 comes from having a disk tier at all. On a fast disk, writing everything (wt)
  is within 2 to 5% of the oracle.
- On a slow disk, writing everything floods the link: wt reaches 1.46x C0's p95 in the worst cell.
- The cost rule is at most 4% worse than C0 in any cell; wt and stk reach 1.46 and 1.65. On a
  0.5 GB/s disk it cuts p95 by 18%, where wt gains nothing and stk 4%. It trails wt by 3 to 4% on fast
  disks.
- The oracle bounds what prediction can add: 0.88 of wt overall, 0.72 on the slow disk.

The run is deterministic: the same command gives the same rows, bit for bit. The exact command, the commit and
date of the shipped files, the row fields and the diff check are in [reference/README.md](../reference/README.md).

## Add your own policy

A policy is a class with `decide(view)`. The simulator calls it when a turn finishes:

```python
class MyPolicy:
    def __init__(self, W):            # W: the workload (sessions); optional
        ...

    def decide(self, v):
        t = v["turn"]                 # the turn that just finished (schema above); never read t["gap_after"]
        return dict(
            eta=None,                 # CPU tier order: None = LRU; seconds to the predicted next use
                                      # (the tier evicts the latest predicted first); < 0 = first out
            disk=False,               # write the session's KV to disk now
            pre=None,                 # fire a 1-token prefetch this many seconds after the turn ends
            warm_eta=None,            # eta the prefetch carries
            gpu_pin=0.0,              # optional: hold the session's idle GPU blocks this many seconds
        )
```

`v` also holds `tokens` (the session's KV size), `now`, `on_disk` (tokens already on disk), `n_wait`
(requests waiting), `idle_g` and `idle_c` ((age, tokens) of idle sessions on the GPU and in the CPU
tier), `link_cpu` and `link_disk` (when each link frees up), `p` (engine parameters) and `live`
(measured rates and residency curves: `sim/live.py`). A policy may also define `gpu_key(sess, now)`
(smallest evicted first) and `cpu_key(sess, now, last_use)` (largest evicted first).

```
returnbench eval --policy mine=mypkg.mymod:MyPolicy --rules wt cost
```

`examples/human_to_disk.py` is a ten-line example.

## Rebuild the pools from the source data

`bench/build/build_all.sh` downloads the four datasets at pinned versions, checks their hashes,
extracts idle boundaries, applies the same split and user hold-out as the Layby training pipeline, and
writes the pools. Steps from the boundary files on reproduce the shipped pools exactly on all four
sources; TraceLab also reproduces end to end from the raw download. See [bench/build/README.md](build/README.md).

The survival curves need Layby-Dwell. Without it the rebuilt pools have no curves, and only `C0`, `wt`,
`cont`, `cj` and `ka` can run on them.

## Limits

- A simulator, not an engine. The Layby adapters run the cost rule inside vLLM and SGLang; those runs
  are not part of this bench.
- One model and one engine shape (Qwen3-8B, 16 GiB GPU KV, 28 GiB CPU tier).
- Small held-out user sets on the agent pools: 5 users in `swechat`, 2 in `tracelab_claude`.
- `copilot` ships without curves.
- Output lengths are fixed per kind (384 after a human wait, 128 after a tool wait); prompt lengths
  and gaps are real.

## Layout

The repository map is the Layout section of the top-level [README](../README.md). The bench's parts: `sim/` (the
simulator and CLI), `bench/pools/` (the four pools, listed with hashes in `bench/MANIFEST.json`), `bench/build/`
(extractors and builders), `reference/` (the reference run, `reference/README.md`), `examples/` (a policy to start
from).
