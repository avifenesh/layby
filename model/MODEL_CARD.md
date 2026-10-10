---
license: apache-2.0
base_model: convaiinnovations/laya
library_name: transformers
pipeline_tag: text-classification
tags:
  - kv-cache
  - llm-serving
  - inference
  - survival-analysis
  - vllm
  - sglang
  - agents
datasets:
  - allenai/WildChat-4.8M
  - SALT-NLP/SWE-chat
---

# Layby-Dwell

Layby-Dwell predicts how long an LLM session stays idle after a response ends. It reads what a
serving engine can see at that moment and returns a survival curve: the probability that the
session's next request has not arrived yet, at 15 horizons from 0.5 s to 30 min.

Code, engine adapters and the ReturnBench benchmark: [github.com/avifenesh/layby](https://github.com/avifenesh/layby), archived on Zenodo as
[10.5281/zenodo.23285213](https://doi.org/10.5281/zenodo.23285213).
Paper: arXiv, link to follow. Author: [Avi Fenesh](https://github.com/avifenesh), [Tiyuvta](https://tiyuvta.ai).

It exists for one decision: where to keep the session's KV cache while it waits. Keep it on the GPU,
move it to CPU memory, write it to disk, or let it go. Layby, the placement rule it was built for,
turns the curve and the engine's own measured costs into that choice. There are no tuned constants.

- Encoder: ModernBERT-large (395M) with a small scoring head, 421M parameters, fp32 weights (1.7 GB).
- Fine-tuned from [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya) with its
  RLCD recipe (strictly proper scoring rule, calibrated probabilities).
- One forward pass per state. No text is generated.

## Input

A server-view state: a JSON object built at the moment a response ends. A generic OpenAI-compatible
server can fill every field. `layby_dwell.Tracker` builds it from the request and response stream.

| field | what it is |
|---|---|
| `kind` | `tool` if the response asks for tool calls, else `human` |
| `model`, `tool`, `program`, `arg_chars` | the served model, the first tool call's name, the program of a shell command, argument size |
| `ctx_tokens`, `turn`, `session_age_s`, `hour`, `dow` | context size, position in the session, clock (UTC) |
| `recent` | the session's last 8 idle periods: kind, tool, seconds |
| `meta` | previous gap and kind, same-kind gap and its EWMA, command size, timeout, background flag |
| `tool_history`, `meta.program_history`, `meta.user_kind_history` | this user's earlier gaps for the same tool, program and kind: count, mean, p90, share under 1 s |
| `tool_call`, `assistant`, `user` | the tool calls, the current turn's assistant text and the last user message, truncated |

Sessions are linked by a hash of the conversation prefix. A user is whatever key the server has
(API key, tenant, header). History fields are optional: the model was trained with them dropped at
random and degrades gracefully without them, but session history matters (see Limits).

## Output

`P(T > h)` for `h` in `[0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 120, 300, 600, 1800]` seconds, where
`T` is the time until the session's next request. Probabilities are calibrated per kind with
temperatures fitted after training (stored in `config.json`).

## Use

```python
from layby_dwell import Dwell, Tracker

dwell = Dwell("avifenesh/layby-dwell")        # GPU if present, bf16
tracker = Tracker()

# in the serving path, per request and response:
sess = tracker.on_request(body, user)
state, kind = tracker.on_response(sess, body, resp_message, usage, user)
surv = dwell.survival([state], [kind])[0]     # 15 values of P(T > h)
```

Score at the end of each response, off the critical path. Batch the boundaries that end together:
`survival` sorts states by length and pads each batch only to its longest state.

### With vLLM and SGLang

The Layby adapters run the placement rule inside the engine and take the curve as a hint keyed by
request:

- vLLM 0.30 and 0.31: `ParkConnector` (`layby.vllm.connector`, an `OffloadingConnector` subclass) with the CPU
  tier's `ParkCachePolicy` (`layby.vllm.policy`) and the `ParkFsTier` disk tier (`layby.vllm.fs_tier`),
  configured through `--kv-transfer-config`.
- SGLang 0.5.21: the `park` plugin (`layby.sglang`, `SGLANG_PLUGINS=park`, `--radix-cache-backend park
  --radix-eviction-policy park`) on HiCache with a storage backend.

The curve rides on the request (`park_surv`, with `park_key` in `kv_transfer_params` or
`custom_params`) or arrives after the response through the adapter's local hint port
(`POST /hint {"key": ..., "surv": [...]}`). The Layby sidecar (`python -m layby.sidecar.proxy`) is an OpenAI-compatible proxy that
does the tracking, scoring and posting for either engine.

## Training

Fine-tuning chain from base Laya, each round starting from the previous one: wait-time rounds v1 to
v4, then v5 (1.67M rows, one epoch), then this model (v6): 300K rows in the server view, schedule-free
AdamW, max length 1024 tokens, one RTX 5090, 6.1 h. The server view hides what a generic server
cannot know: no data source, workflow turns shown as human, and per row the client name (p 0.5),
the identity block (p 0.3), harness-parsed fields (p 0.3) and all session history (p 0.15) dropped.

Target: the seconds until the session's next request, binned into 16 levels. Sessions that never
come back are censored, and their target spreads over the levels still possible.

### Data

- [GitHub Copilot Coding Agent Traces 2026](https://github.com/Azure/AzurePublicDataset/blob/master/GitHubCopilotCodingAgentDataset2026.md) (CC BY 4.0): agent sessions, timestamps only, no content.
- [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat) (ODC-BY): Claude Code and Codex transcripts from 252 users.
- [WildChat-4.8M](https://huggingface.co/datasets/allenai/WildChat-4.8M) (ODC-BY): chat conversations with exact request times.
- [TraceLab](https://github.com/uw-syfi/TraceLab) (CC BY 4.0): coding-agent traces, content-free.
- The author's own Claude Code and Codex sessions.
- Users' data, under the terms they agreed to.

Only datasets whose licenses allow commercial use. 10% of users (by hash) on every source with real
user ids were held out of training and validation to measure unseen users.

## Evaluation

### Curve quality on held-out users (negative log-likelihood of the true gap, lower is better)

| pool | Layby-Dwell | same recipe, all sources | source left out of training |
|---|---|---|---|
| SWE-chat | 0.583 | 0.613 | 0.677 |
| TraceLab-Claude | 1.015 | 1.105 | 1.231 |
| WildChat | 2.383 | 2.440 | 2.363 |

The last column retrains from base Laya without that source. A source the model never saw costs
some calibration, not the decisions (next table).

### Placement on held-out users (ReturnBench, simulator)

p95 time to first token against vLLM's default (CPU tier, LRU, no disk), geometric mean over 33
cells (3 pools, 2 to 4 load levels, disk at 0.5, 1.5 and 3 GB/s; 8 workloads x 3 seeds each):

| policy | p95 vs default | worst cell |
|---|---|---|
| write everything to disk, LRU | 0.68 | 1.46 |
| Layby rule with Layby-Dwell | 0.66 | 1.04 |
| same rule with the true gap (oracle) | 0.60 | 1.00 |

Leave-one-source-out: the rule driven by curves from models that never saw the pool's source gives
0.67 against 0.69 with the all-sources control (27 cells).

### Real engine

vLLM 0.30, Qwen3-8B, one A100, 16 GiB GPU KV, 28 GiB CPU tier, an 18-session agent replay (671
turns): Layby-Dwell under a threshold rule cut p95 time to first token by 22 to 25% against the
default, over three repeats. Results with the Layby rule on vLLM and SGLang are in the paper.

## Speed

Public-pool states are long: 740 tokens at the median, 1,000 at p90 (1,024 max).

| device | precision | one state | batch of 8 | batch of 32 |
|---|---|---|---|---|
| CPU, 8 threads (shared, loaded host) | fp32 | 1.7 to 2.9 s | | 0.2 to 0.3 states/s |
| RTX 4090 | bf16 | 23.0 ms | 93.1 states/s | 83.6 states/s |
| RTX 4090 | fp32 | 27.7 ms | 40.2 states/s | 40.4 states/s |
| RTX 4090 | int8 weights | 26.5 ms | 90.3 states/s | 81.2 states/s |
| RTX 4090 | fp8 weights | 29.1 ms | 95.2 states/s | 88.6 states/s |

GPU timings are medians of 3 runs on 1,024 public states, tokenizing included. CPU is too slow to serve; run
the scorer on a GPU next to the engine, or on a small GPU of its own. Serve in bf16, the precision every
evaluation used. int8 weights (`precision="int8w"`) halve the encoder's weight memory at the same speed and
pick the same cost-rule option as bf16 for 99.75% of public states. fp8 weights (`"fp8w"`, FP8 GPUs only) do so
for 99.54%. fp32 differs from bf16 by up to 0.0135 over 4,445 states. 8-bit activations are not offered: int8
dynamic is 7x slower, and fp8 dynamic returns NaN for 2% of states. Details: [QUANT.md](QUANT.md).

## Limits

- Session history carries most of the signal for long human waits. Without session linking
  (stateless), recall of "back after 300 s or more" on human waits drops from 0.68 to 0.35. Run the
  tracker.
- Weak on traffic unlike its sources: on a small gateway sample (556 tool waits) it never called a
  60 s tool wait correctly. Calibrate on your own traffic before trusting long horizons there.
- Copilot automated turns are hard to call; a per-cohort guardrail should hold them.
- English-centric text fields; other languages were not measured.
- The curve is a forecast, not a guarantee. Use it inside a rule that prices a wrong call, as Layby
  does, not as a hard timeout.
- It sees prompt and response text. It stores nothing, but treat scoring like any other component
  that reads traffic.

## License

Apache-2.0, like the base Laya and ModernBERT weights.

## Citation

```
@misc{fenesh2026layby,
  title  = {Layby: placing idle LLM sessions' KV cache by when they come back},
  author = {Avi Fenesh},
  year   = {2026},
  note   = {arXiv id to follow}
}
```
