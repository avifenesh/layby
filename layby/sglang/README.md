# park on SGLang 0.5.21

The SGLang adapter for the park cost rule (`layby.rule.decide`). It mirrors the vLLM 0.30 adapter in
`layby/vllm/`: the engine measures itself, the rule places each finished request's KV, the host tier
evicts by predicted next use, and storage holds only what the rule puts on disk. Target: SGLang 0.5.21
from PyPI with HiCache (host memory tier, optional L3 storage).

## Install

```
uv venv .venv-sglang -p 3.12
uv pip install -p .venv-sglang/bin/python --prerelease=allow "sglang[all]==0.5.21"   # pins cuda-tile 1.6.0rc5
uv pip install -p .venv-sglang/bin/python -e . --no-deps                            # park, with its plugin entry point
```

`sglang[all]==0.5.21` brings torch 2.13.0+cu130 and sglang-kernel 0.4.7 (CUDA 13). The editable
install registers the entry point `park = layby.sglang.plugin:register` in group `sglang.srt.plugins`.
SGLang runs it in the main process (before it parses arguments) and in every scheduler process.

## Launch

```
SGLANG_PLUGINS=park SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=/nvme/park \
sglang serve --model-path Qwen/Qwen3-8B --page-size 64 --max-total-tokens N \
  --enable-hierarchical-cache --hicache-ratio R --hicache-write-policy write_through \
  --hicache-storage-backend file \
  --radix-cache-backend park --radix-eviction-policy park \
  --radix-eviction-policy-config '{"port": 8765}'
```

- `--radix-cache-backend park` builds `ParkRadixCache` (a `UnifiedRadixCache` subclass) on the
  Python TreeCore. The Rust TreeCore, SGLang's default, keeps eviction native and has no
  `node_by_id`. The backend refuses to start without `--radix-eviction-policy park`, and refuses
  hybrid SWA or Mamba models.
- `--radix-eviction-policy-config` takes `port`: the local hint port (TP rank 0, 127.0.0.1). Without
  it, curves can only ride on the request.
- Without `--hicache-storage-backend` the rule has no disk option: ins becomes none, and park becomes
  drop, as in the vLLM adapter.
- `SGLANG_PLUGINS=park` loads only this plugin. Any other value (the runner uses `none`) leaves it
  out. The hooks do nothing unless the park backend is in use.

Requests carry `custom_params`, which SGLang passes to the scheduler as
`req.sampling_params.custom_params`:
- `park_key`: the key the sidecar uses for the hint (the request id otherwise).
- `park_surv`: optional. 15 values of P(T > E2[j]) for the idle period after this request. When
  present, the decision is made at finish.

Native `/generate` takes `custom_params` in `sampling_params`. The OpenAI chat and completions APIs
take it as a top-level field. The sidecar (`layby/sidecar/proxy.py --engine sglang`) sets it.

Hint port, the same API as `layby/vllm/server.py` (the same server, given this adapter's state):
`POST /hint {"key", "surv": [15]}` or a list of them, `GET /live` (telemetry snapshot), `GET /health`.

## Arms

`engine/run_arms_sglang.sh WORKLOAD OUTDIR GPU_KV_BYTES CPU_BYTES DISK_DIR ARM...` runs C0 (HiCache
host tier, LRU, write_through), T (+ file storage, every node written through) and K (park). It uses
the same capacity arguments as `engine/run_arms.sh`: GPU KV bytes become `--max-total-tokens` (bytes / BPT),
host bytes become `--hicache-ratio`, and the disk directory becomes the file backend directory.
Every arm uses the Python TreeCore (`TREE_CORE`) and page size 64 (`PAGE`). Replay:
`engine/replay.py --engine sglang` (native `/generate`, token ids in and out, TTFT from the stream).
`--window SECONDS` bounds a run.

## What each piece measures

| SGLang 0.5.21 point | how | feeds |
|---|---|---|
| `SchedulerRequestReceiver._pull_raw_reqs` | AFTER hook, leader | drains hints and storage-thread events, runs the rule for late hints, appends a `ParkTick` (rank 0 monotonic time, placements) to the pulled list |
| `SchedulerRequestReceiver.recv_requests` | AFTER hook, every rank | strips ticks after the TP broadcast: tick clock, counter-to-time log, placements; leader records each request's arrival |
| `ScheduleBatch.prepare_for_extend` | AFTER hook, leader | first prefill = admission: `n_adm`, `qwait`; miss losses (`Live.miss` g and c, `miss_g`, `miss_c`) from `cached_tokens_device/host/storage` and the prompt's page hashes against GPU and host eviction times |
| `Scheduler.run_batch` | BEFORE hook, leader | `Speeds.step` (time between launches, prefill tokens of the previous batch: t0, f0); `nw`, `pf` (waiting and prefilling requests over time) |
| `Scheduler.on_idle` | BEFORE hook, leader | an idle loop ends the step being timed |
| tree core `kv_events` (`KVCacheEventRecorder`) | instance wrapper | GPU remove (demote, or delete without a host copy) and host remove (host leaf eviction): KM residency events with idle age, eviction time per page hash; CPU store (write-through ack, prefetch insert): host idle start. Works with KV events disabled |
| tree core `_split_node` | instance wrapper | the suffix keeps its hint (restamped); the prefix inherits the open idle periods (split by tokens) and a pending disk placement |
| `UnifiedRadixCache.insert_req` / `insert` | override | the inserted leaf of a finished request; the entry under `park_key`; decision at finish when a curve is there |
| `UnifiedRadixCache.on_release` | override | GPU and host idle periods start for unlocked nodes on the request's path |
| `UnifiedRadixCache.inc_lock_ref` | override | a lock on a path ends its idle periods (reuse, censored) |
| `UnifiedRadixCache.loading_check` + controller `load` / `start_loading` | override + instance wrappers | H->D link: bytes over the CUDA event time (`Speeds.link "cpu"`), wait from the load request to completion beyond its own transfer (`n_cpu`, `wait_cpu`, `busy_cpu`) |
| `UnifiedRadixCache._log_write_ack_metrics` | override | D->H speed (`d2h_gbps` in `/live`; the rule prices restores, H->D) |
| controller `_page_backup` / `_page_transfer` / `write_storage` | instance wrappers (storage threads) | disk link busy wall time and bytes (`Speeds.link "disk"`, `busy_disk`); each job's wait (`n_disk`, `wait_disk`) |
| `UnifiedRadixCache.write_backup_storage` | override | storage filter: a write-through ack writes to L3 only nodes the rule placed on disk |
| `--radix-eviction-policy park` (`ParkStrategy.get_priority`) | `_EVICTION_POLICY_FACTORIES` | GPU heap: LRU on the logical counter. Host heap: predicted next use (hint), first out for park and drop, LRU estimate `now + (now - last use)` otherwise. A hint ends on a touch or GRACE (2 s) past due |

Decision flow: at finish, or when `POST /hint` brings the curve, the leader runs `decide` with the
request's tokens, the tokens of its path already in storage, and the open GPU and host idle periods.
The placement (predicted next use, or first out; disk or not) goes into the next tick. Every rank then
sets the predicted next use on the idle nodes of the path. A node shared with a session that has not
returned keeps the earlier use. For ins and park, the path's host copies are written to storage now
(late writes). A copy still in flight is written when its write-through ack lands.

## TP and other limits

- TP=1 is the tested target. For TP>1 the design keeps every rank's tree identical. Placements and
  the clock come only from rank 0, inside the request broadcast (`ParkTick`). Priorities read the tick
  clock, and the tick at which SGLang's logical access counter passed a node's `last_access_time`.
  They never read a per-rank wall clock. Telemetry and decisions run on rank 0 only (Live, Speeds,
  idle periods, eviction times). The hint port binds on rank 0 only. Each rank writes its own KV shard
  to storage, as SGLang does.
- Not supported: DP attention (the tick would ride the work/control split), pipeline parallelism (PP
  stages > 0 receive requests point-to-point), `--scheduler-recv-interval` > 1 (skipped receives
  carry no tick), the `write_back` policy (host duplicates are reclaimed under it), EAGLE bigram keys
  (page hashes differ), and hybrid SWA or Mamba models.
- The file backend must not evict (leave `SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE` unset). The adapter
  counts pages it wrote as on disk.
- Page hashes per node are SGLang's storage chain hashes. With storage off they are computed lazily
  on eviction and kept on the node, as the KV events recorder does. Use a page size of 16 or more:
  the eviction-time records are per page.

## CPU checks (no GPU)

```
CUDA_VISIBLE_DEVICES= .venv-sglang/bin/python -m pytest -q tests/test_park_sglang.py
```

The checks cover: eviction order and placement on mock nodes; the plugin entry point and every hook
target resolving and wrapped in the installed 0.5.21; a real `ParkRadixCache` on the Python TreeCore
with a CPU allocator (insert, split, match, evict, residency events, the tick path end to end);
storage and load link telemetry; the hint port; the sidecar tag; and `replay.py` against mock SGLang
and vLLM servers, with and without `--window`.

## GPU smoke checklist (rented box)

1. Install as above. `python -c "import sglang"`, and confirm `sglang-kernel` loads on the card's
   arch.
2. Start arm K with a small model and `--log-level info`. The log should show `Applied 1 hook(s)` for
   the 5 targets, `Tree cache initialized: source=registered('park') impl=ParkRadixCache`,
   `Init Unified Radix Cache ... Tree Core: UnifiedTreeCore`, `park: cache up`, `park: host ...
   storage True`, and `park: hint port 8765`.
3. `curl localhost:8765/health`, then `curl localhost:8765/live`: capacities match the flags
   (gpu_tokens = `--max-total-tokens` rounding, cpu_tokens = ratio x gpu, bytes_per_token = model KV
   bytes per token per rank).
4. Send one `/generate` with `custom_params.park_key` and `park_surv` (15 values). `/live` counts one
   decision. Send a second request with only `park_key`, then POST its curve: `late` = 1.
5. Run a short replay (`run_arms_sglang.sh` with WINDOW=300, arms C0 T K) at a GPU size that forces
   host eviction. In `K.park_live.json`: `ne_g` and `ne_c` > 0, `resid_c` not all 1, `n_cpu` and
   `n_disk` > 0, `speeds.cpu_gbps` and `disk_gbps` measured (not the 20 / 1.5 defaults), `miss_g` or
   `miss_c` > 0, the decision mix not all one option, `write_late` > 0 and `write_skipped` > 0. The
   storage dir holds files in K and grows faster in T.
6. Correctness under eviction: the K server log has no `park: ... failed` line and no SGLang
   assertion. Outputs are deterministic (temperature 0, ignore_eos), so a fixed prompt gives the same
   ids in C0 and K.
7. Overhead: the K step time (`/live` t0) is within noise of C0 at the same load. The Python TreeCore
   is in every arm, so the baselines pay it too.
8. TP=2, only if TP>1 is in scope: K starts, `/live` is served once (rank 0), and a replay with heavy
   eviction raises no TP divergence assertion (`write_back duplicate-reclaim`, rank consensus checks).
