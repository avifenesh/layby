# Building the pools

`build_all.sh` rebuilds all four pools from the public datasets. Run it in an empty directory:

```
mkdir work && cd work && ../bench/build/build_all.sh
```

Steps:

1. **Download**, pinned: SWE-chat revision `f66cca95`, WildChat-4.8M revision `c827c6df`, TraceLab
   release `v0.0.2`, the Copilot release `ghcp-coding-agent-2026`. `SHA256SUMS.raw` checks the files
   that are not pinned by a git revision.
2. **Extract idle boundaries**: one JSON line per idle period (the turn ended, the session waits).
   - `extract_transcripts.py`: Claude Code transcripts (SWE-chat). Users from `sessions.parquet`.
   - `extract_tracelab.py`: TraceLab's DuckDB.
   - `extract_wildchat.py`: WildChat, 8% of conversations by a hash of the conversation id. Gaps are
     the next request's `created` minus this response's `timestamp`.
   - `extract_copilot.py`: Copilot traces, 25% of sessions by a hash of the session id.
   Extractor output order is not fixed, so `SHA256SUMS.bnd` holds the hash of each file after
   `LC_ALL=C sort`. The script compares them.
3. **Split and pool** (`split.py`, `build_replay_pop.py`): per source, sessions ordered by first idle
   time, the last 20% are test; 10% of users by hash are held out of training. A pool takes test-split
   sessions of held-out users with at least 6 turns (4 for WildChat), 40 turns at most, contexts scaled
   under 32,768 tokens. Copilot has no user ids: 200 test-split sessions of any user.
4. **Curves** (needs Layby-Dwell): each turn's state is scored and the 15 survival probabilities are
   attached with `attach_curves.py`, joined on the `feat_ref` column of `refs/POOL.refs.parquet`. The
   state builder and the model are released with Layby-Dwell. This step is the only one that needs a GPU.
5. **Sanitize** (`sanitize.py`): only lengths, gaps, kinds, tool and program names and curves leave.
   User ids become pool-local indices, MCP tool names become `mcp`, rare program names `other`.

Checked: TraceLab end to end (raw DuckDB to pool, identical). For SWE-chat, WildChat and Copilot,
`SHA256SUMS.bnd` is the hash of the boundary files the shipped pools were built from, and steps 3 and 5
reproduce the shipped pools from them exactly. Their extractors have not been re-run from the raw
download yet. The `copilot` pool carries
tool names and per-session users; no reference cell uses it.
