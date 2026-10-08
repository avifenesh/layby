# The simulator reference run

`rows.jsonl` and `table.txt` are the ReturnBench reference table of the README and `bench/README.md`: 33 cells, 8
workloads, 3 seeds, rules C0, wt, stk, cost and orc, on the shipped pools (`bench/pools/`, `bench/MANIFEST.json`).

Command (about 6 minutes on 8 cores; `--out` appends, so start from no file):

    returnbench eval --rules wt stk cost orc --per-cell --out reference/rows.jsonl > reference/table.txt

Commit: `75e4b3e543606fb94b1f0953aaba3154a4cbe7e6` (2026-10-08), the commit that added both files; the simulator has not
changed since. The run is deterministic: the same command on the same pools gives the same rows, bit for bit (numpy
seeds, no wall-clock input).

## Files

`rows.jsonl`: 3,960 rows, one per (cell, workload, seed, rule). Fields:

| field | meaning |
|---|---|
| `pool` | the pool, or pools joined with `+` |
| `n` | sessions in the workload |
| `w` | workload index (the numpy seed that draws the sessions and their start times) |
| `seed` | engine seed (step-time jitter) |
| `disk` | disk link speed, GB/s |
| `rule` | the policy (`C0`, `wt`, `stk`, `cost`, `orc`; `bench/README.md` defines them) |
| `p50`, `p95`, `p99` | TTFT quantiles in seconds over every turn after a session's first |
| `warms` | 1-token prefetch requests the policy fired |

`table.txt`: `returnbench table reference/rows.jsonl --per-cell` output: per cell and rule, p50 and p95 in seconds
and their ratios to C0, then the summary (geometric mean over cells of the p95 ratio to C0 and to wt, overall and
per disk speed). Its last block is the table in `bench/README.md`.

## Check a rerun

    mkdir -p rerun
    returnbench eval --rules wt stk cost orc --per-cell --out rerun/rows.jsonl > rerun/table.txt
    diff reference/rows.jsonl rerun/rows.jsonl && diff reference/table.txt rerun/table.txt && echo SAME

`diff` prints nothing and the line ends with `SAME` on a pass; any changed row is a failure (`diff` exits 1 and shows
the row). The tolerance is zero: this is a simulator with fixed seeds, not a measurement.

Quick check (seconds): the README's two-cell quick look writes rows that are a subset of this file.

    returnbench eval --rules wt cost --cells wildchat:120 swechat:18 --workloads 2 --seeds 1 --out rerun/quick.jsonl
    grep -c -x -F -f rerun/quick.jsonl reference/rows.jsonl     # prints 36 when every quick row is in the reference

A rebuilt pool (`bench/build/build_all.sh` against a later upstream revision) changes these rows; rerun the command
and commit the new files with the new `bench/MANIFEST.json` hashes.

## License

The rows are simulator output over the ReturnBench pools and carry the pools' source licenses (`bench/LICENSES.md`:
ODC-BY 1.0 for SWE-chat and WildChat, CC BY 4.0 for TraceLab); the code that produced them is Apache-2.0.
