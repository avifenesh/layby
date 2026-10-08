# Real-length pools

The reference pools scale each session's context under 32,768 tokens and stop at 40 turns or 38,000 history tokens.
That is a trial setting. Real agent sessions run far longer, so the real-engine bench uses the same sessions at
their true lengths.

## What limited length

`build_replay_pop.py` had three limits:

- `--ctx-cap 32768` scaled the whole context curve down;
- `--max-turns 40` cut sessions after 40 turns;
- a fixed 38,000-token history limit stopped the rest.

New flags, with the old defaults kept so the reference pools rebuild byte for byte (checked on SWE-chat: identical):

- `--no-scale`: contexts keep their true per-turn growth. The session ends at the last turn whose history fits
  `--ctx-cap`, the model's window.
- `--max-turns 0` takes every turn. `--max-hist 0` drops the history limit.

Output lengths stay 384 tokens after a human wait and 128 after a tool wait.

Command per pool, from the feature table:

    build_replay_pop.py feat5.parquet SRC SRC.real.json --refs-out SRC.real.refs.parquet --gap-cap inf \
      --min-turns 6 --no-scale --ctx-cap 262144 --max-turns 0 --max-hist 0

WildChat takes `--min-turns 4`. Copilot takes `--any-user --sessions 200`.

## Pools at real length (cap 262,144 tokens)

| pool | sessions | turns | turns per session p50 / p90 / max | max context p50 / p90 / max | token-turns | session gap sum, gaps capped at 300 s, p50 / p90 / max |
|---|---|---|---|---|---|---|
| SWE-chat | 60 | 6,346 | 84 / 230 / 302 | 123,650 / 256,833 / 260,337 | 0.83B | 19 / 77 / 209 min |
| TraceLab Claude | 25 | 918 | 26 / 81 / 132 | 77,939 / 182,747 / 260,452 | 0.09B | 2 / 49 / 82 min |
| WildChat | 211 | 1,777 | 6 / 14 / 51 | 4,258 / 27,904 / 213,827 | 0.03B | 10 / 33 / 112 min |
| Copilot | 199 | 9,747 | 31 / 110 / 416 | 64,422 / 168,344 / 260,428 | 0.79B | 9 / 38 / 125 min |

Before the cap, real session peaks reach 967k tokens on SWE-chat, 389k on TraceLab and 903k on Copilot. The cap
ends those sessions where the model's window ends.

## Workloads (engine format)

`engine/build_real_workload.py` draws sessions like `eval_pop.workload`. Each pool gets an even
share, and a small pool hands its remainder to the others. Starts are uniform over 600 s, gaps are capped at
300 s, and `surv_v6` is attached per turn.

| file | sessions | turns | max context p50 / p90 / max | token-turns | peak live KV | replay length |
|---|---|---|---|---|---|---|
| `engine/workloads/mix150_real.json` | 150 (60 SWE-chat, 25 TraceLab, 65 WildChat) | 7,858 | 52,956 / 253,336 / 260,452 | 928M | 4.50M tokens | 3.6 h, session p50 9 min |
| `engine/workloads/mix36_real.json` | 36 (12 each) | 1,649 | 49,942 / 170,887 / 260,452 | 165M | 1.43M tokens | 1.5 h, session p50 11 min |

- **Peak live KV** is the largest, over time, of the summed current context of every started, unfinished session.
  It ignores service time, which only stretches the timeline.
- **Bytes** are tokens times the model's KV per token. At about 11 KB per token (GLM 5.3 Flash, MLA layers only,
  linear state extra), 4.5M tokens is about 50 GB. At 320 KiB per token (Hy3 BF16) it is about 1.4 TB.
- **Replay length** is wall time from the gaps alone. Prefill of 100k to 260k-token prompts adds to it.

## Curves

Layby-Dwell's state uses `ctx_tokens` from the source boundary (`feat5.parquet` ctx, the real context), never the
replay's scaled one. So the existing v6 curves stay valid for the turns the 32k pools already had. Only the turns the
old limits cut need scoring:

| pool | turns needing scores |
|---|---|
| SWE-chat | 4,302 of 6,346 |
| TraceLab Claude | 263 of 918 |
| WildChat | 31 of 1,777 |
| Copilot | all 9,747 |

That is 14,343 states in all, built with the Layby-Dwell state builder over the pools' refs. They hold public-dataset
text, so they are not shipped; the workloads in `engine/workloads/` carry the resulting curves on every turn.

Scoring needs a GPU: about 3 min at about 100 states/s, against about 10 h on CPU (layby_dwell `Dwell.score`, or the
scoring script of the training repo with `--view server`). Then rebuild the workloads with
`engine/build_real_workload.py ... --probs <existing curves> <new curves>`.

## Replay client

`engine/replay.py` handles these lengths. All 150 histories build in 9 s at 0.4 GB RSS. The largest prompt is 262,127
tokens, 1.9 MB of JSON.

`engine/run_arms.sh` takes these as environment variables: `MAXLEN` (at least the cap plus output; 262144 for these
runs), `REPLAY_TIMEOUT` (above the replay length, or bound the arm with `WINDOW` seconds) and `BOOT_WAIT`.
