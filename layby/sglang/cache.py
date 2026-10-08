"""ParkRadixCache: SGLang 0.5.21's UnifiedRadixCache plus the park cost rule (--radix-cache-backend park).

The cache is the engine adapter for layby.rule.decide, the port of park/vllm/connector.py and
park/vllm/fs_tier.py. Tree, transfers, the host tier and the L3 storage stay SGLang's (HiCache), with
layby.sglang.policy as the eviction order and the storage write filtered to what the rule places on
disk.

Measured here (leader rank), fed to layby.sglang.state.STATE:
  GPU residency  a node whose last request released it is idle; a GPU removal (demote to host, or
                 delete without a host copy) ends the period as an event, a new lock (reuse) as
                 censored; the KV events recorder is wrapped, so this works with KV events disabled
  CPU residency  the same for nodes with a host copy: a host eviction ends the period as an event
  eviction times per page hash (GPU and host), read by the admission hook for miss losses
  CPU link       H->D load-back acks: bytes over the CUDA event time (Speeds.link "cpu"), and wait
                 from the load request to completion beyond the transfer itself (Live n_cpu,
                 wait_cpu, busy_cpu); D->H write-through acks give a D->H speed for the snapshot
  disk link      storage backup and prefetch jobs (threads): busy wall time and the bytes moved in it
                 (Speeds.link "disk", Live busy_disk), and each job's wait (Live n_disk, wait_disk)

Decision: when a request finishes its curve may already be there (custom_params park_surv);
otherwise it waits under its park_key (custom_params park_key, the request id otherwise) for POST
/hint. The leader decides; the placement (predicted next use, disk) reaches every rank in the next
tick and sets the path's predicted next use (layby.sglang.policy.place); for ins and park the path's
host copies are written to storage (late writes; a host copy still in flight goes on when it lands).
Write-through acks write nothing else to storage.
"""
import threading
import time

import numpy as np

from sglang.srt.disaggregation.kv_events import StorageMedium
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.mem_cache.utils import compute_node_hash_values, get_hash_str

from layby.rule import TG, decide, interp_surv
from layby.sglang import policy
from layby.sglang.state import NEVER, STATE, logger, safe

FULL = policy.FULL


class _Events:
    """Wraps the tree core's KVCacheEventRecorder: sees every placement event, then forwards it."""

    def __init__(self, inner, cache):
        self.inner, self.cache = inner, cache

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def record_store(self, node, medium=None, **kw):
        if medium == StorageMedium.CPU:
            safe("cpu store", self.cache._park_cpu_store, node)
        return self.inner.record_store(node, medium, **kw)

    def record_remove(self, node, medium=None):
        safe("eviction", self.cache._park_remove, node, medium)
        return self.inner.record_remove(node, medium)


class ParkRadixCache(UnifiedRadixCache):
    def __init__(self, params):
        self._park_cap = None
        super().__init__(params)
        tc = self.tree_core
        tc.kv_events = _Events(tc.kv_events, self)
        split = tc._split_node

        def _split_node(key, child, split_len):
            was = safe("split", policy.valid_before_split, child)
            new_node, action = split(key, child, split_len)
            safe("split", policy.on_split, new_node, child, bool(was))
            return new_node, action

        tc._split_node = _split_node
        STATE.cache = self
        STATE.page_size = self.page_size
        alloc = self.token_to_kv_pool_allocator
        STATE.gpu_tokens = int(getattr(alloc, "size_full", alloc.size))
        logger.info("park: cache up, gpu %d tokens, page %d tokens", STATE.gpu_tokens, self.page_size)

    # --- HiCache: capacities and link telemetry --------------------------------------
    def init_hicache(self, server_args, params) -> None:
        super().init_hicache(server_args, params)
        cc = self.cache_controller
        if cc is None:
            return
        host = cc.mem_pool_host
        hp = host.anchor_entry.host_pool if hasattr(host, "anchor_entry") else host
        STATE.cpu_tokens = int(hp.size)
        STATE.bytes_per_token = float(hp.size_per_token)
        self._park_load_req: list = []                 # times of load requests not yet launched
        self._park_load_sub: dict = {}                 # id(ack) -> submit time of its oldest load
        self._park_backup_sub: dict = {}               # storage backup operation id -> submit time
        self._park_disk = dict(n=0, mark=None, bytes=0)  # disk busy period (backup and prefetch threads)
        self._park_disk_lock = threading.Lock()

        load, start_loading, write_storage = cc.load, cc.start_loading, cc.write_storage

        def _load(*a, **k):
            out = load(*a, **k)
            if out is not None:
                self._park_load_req.append(time.monotonic())
            return out

        def _submitted(n):
            if len(cc.ack_load_queue) > n:
                t = min(self._park_load_req) if self._park_load_req else time.monotonic()
                self._park_load_sub[id(cc.ack_load_queue[-1])] = t
            self._park_load_req.clear()

        def _start_loading(*a, **k):
            n = len(cc.ack_load_queue)
            out = start_loading(*a, **k)
            safe("cpu link", _submitted, n)
            return out

        def _write_storage(*a, **k):
            op_id = write_storage(*a, **k)
            self._park_backup_sub[op_id] = time.monotonic()
            return op_id

        cc.load, cc.start_loading, cc.write_storage = _load, _start_loading, _write_storage
        if getattr(cc, "enable_storage", False):
            backup, transfer = cc._page_backup, cc._page_transfer

            def _page_backup(op):
                t = self._park_backup_sub.pop(op.id, None)
                return self._park_disk_job(backup, op, t)

            def _page_transfer(op):
                return self._park_disk_job(transfer, op, getattr(op, "start_time", None))

            cc._page_backup, cc._page_transfer = _page_backup, _page_transfer
        logger.info("park: host %d tokens, %.0f bytes/token, storage %s", STATE.cpu_tokens,
                    STATE.bytes_per_token, bool(getattr(cc, "enable_storage", False)))

    def _park_disk_job(self, fn, op, t_sub):
        """A storage job on a controller thread: the disk link is busy while any job runs. Events go
        to the scheduler thread through STATE.thread_events."""
        d = self._park_disk
        t0 = time.monotonic()
        with self._park_disk_lock:
            if d["n"] == 0:
                d["mark"] = t0
            d["n"] += 1
        c0 = getattr(op, "completed_tokens", 0) or 0
        try:
            return fn(op)
        finally:
            safe("disk link", self._park_disk_done, op, t_sub, t0, c0)

    def _park_disk_done(self, op, t_sub, t0, c0) -> None:
        d = self._park_disk
        now = time.monotonic()
        nbytes = 0.0
        try:
            if hasattr(op, "request_id"):
                # a prefetch: its completed tokens arrive later through the sync thread; it read the
                # pages the hit query found unless the scheduler terminated it
                tokens = 0 if op.is_terminated() else len(op.hash_value) * STATE.page_size
            else:
                tokens = max((getattr(op, "completed_tokens", 0) or 0) - c0, 0)
            nbytes = tokens * STATE.bytes_per_token
        finally:
            leader = STATE.leader                       # only the leader drains the events
            if leader:
                STATE.thread_events.put(("disk_job", now, t_sub if t_sub is not None else t0, nbytes))
            with self._park_disk_lock:
                d["bytes"] += nbytes
                d["n"] -= 1
                if d["n"] == 0:
                    if leader:
                        STATE.thread_events.put(("disk_busy", now, now - d["mark"], d["bytes"]))
                    d["bytes"] = 0

    def loading_check(self, finish_count=None) -> None:
        cc = self.cache_controller
        before = list(cc.ack_load_queue) if cc is not None else []
        super().loading_check(finish_count)
        if before:
            done = before[:len(before) - len(cc.ack_load_queue)]
            for ack in done:
                safe("cpu link", self._park_load_ack, ack)

    def _park_load_ack(self, ack) -> None:
        t_sub = self._park_load_sub.pop(id(ack), None)
        if not STATE.leader or not ack.timing_enabled or ack.num_bytes <= 0:
            return
        now = time.monotonic()
        secs = ack.start_event.elapsed_time(ack.finish_event) / 1000.0
        STATE.speeds.link(now, "cpu", ack.num_bytes, secs)
        x = ack.num_bytes / (STATE.speeds.gbps("cpu") * 1e9)
        wait = max(now - t_sub - x, 0.0) if t_sub is not None else 0.0
        STATE.live.add(now, n_cpu=1.0, busy_cpu=x, wait_cpu=wait)

    def _log_write_ack_metrics(self, ack) -> None:
        super()._log_write_ack_metrics(ack)
        if STATE.leader and ack.timing_enabled and ack.num_bytes > 0:
            secs = safe("d2h link", ack.start_event.elapsed_time, ack.finish_event)
            if secs:
                STATE.speeds.link(time.monotonic(), "d2h", ack.num_bytes, secs / 1000.0)

    def park_prune(self, horizon) -> None:
        """Submit times of storage backups that never reached the backup thread."""
        old = time.monotonic() - horizon
        sub = getattr(self, "_park_backup_sub", None)
        if sub:
            for k in [k for k, t in sub.items() if t < old]:
                del sub[k]
        if getattr(self, "_park_load_sub", None):
            live = {id(a) for a in self.cache_controller.ack_load_queue}
            self._park_load_sub = {k: t for k, t in self._park_load_sub.items() if k in live}

    # --- finish, release, reuse -------------------------------------------------------
    def insert(self, params):
        r = super().insert(params)
        if self._park_cap is not None:
            self._park_cap.append(r)
        return r

    def insert_req(self, req, *, up_to: int, **kwargs) -> None:
        self._park_cap = []
        try:
            super().insert_req(req, up_to=up_to, **kwargs)
        finally:
            cap, self._park_cap = self._park_cap, None
        safe("finish", self._park_finish, req, up_to, cap)

    def _park_finish(self, req, up_to, cap) -> None:
        if not cap or cap[0] is None or cap[0].last_device_node is None:
            return
        leaf = cap[0].last_device_node
        if self.tree_core.is_root(leaf):
            return
        STATE.leaf_of[req.rid] = leaf
        if not req.finished():
            return
        cp = getattr(req.sampling_params, "custom_params", None) or {}
        key = str(cp.get("park_key") or req.rid)
        surv = cp.get("park_surv")
        if surv is None and STATE.port is None:
            return                     # no curve now and none can come later
        entry = dict(leaf=leaf, tokens=int(up_to), t=STATE.clock)
        STATE.entries[key] = entry
        STATE.entries.move_to_end(key)
        if STATE.leader:
            early = STATE.early.pop(key, None)
            if surv is not None or early is not None:
                decide_entry(key, entry, surv if surv is not None else early[1], time.monotonic())

    def on_release(self, req, *, inserted: bool) -> None:
        super().on_release(req, inserted=inserted)
        safe("release", self._park_release, req)

    def _park_release(self, req) -> None:
        leaf = STATE.leaf_of.pop(req.rid, None)
        if leaf is None or not STATE.leader:
            return
        now = time.monotonic()
        for node in self.park_path(leaf):
            cd = node.component_data[FULL]
            if cd.lock_ref > 0:
                continue                                 # another running request holds it
            w = float(len(node.key))
            if not node.evicted and node.id not in STATE.g_idle:
                STATE.g_idle[node.id] = (now, w)
            if node.backuped and cd.host_lock_ref == 0 and node.id not in STATE.c_idle:
                STATE.c_idle[node.id] = (now, w)

    def inc_lock_ref(self, node_id, skip_lock_components=()):
        out = super().inc_lock_ref(node_id, skip_lock_components)
        if STATE.leader and (STATE.g_idle or STATE.c_idle):
            safe("reuse", self._park_reuse, node_id)
        return out

    def _park_reuse(self, node_id) -> None:
        """A lock on a path ends the idle periods on it: reuse, censored at its age."""
        now = time.monotonic()
        L = STATE.live
        for node in self.park_path(node_id):
            for tier, d in (("g", STATE.g_idle), ("c", STATE.c_idle)):
                r = d.pop(node.id, None)
                if r is not None:
                    L.end_idle(now, tier, now - r[0], float(len(node.key)), False)

    # --- residency events -------------------------------------------------------------
    def _park_cpu_store(self, node) -> None:
        """A host copy landed (write-through ack, or a storage prefetch): idle from now if no request
        holds the node."""
        if not STATE.leader:
            return
        cd = node.component_data[FULL]
        if cd.lock_ref == 0 and cd.host_lock_ref == 0 and node.id not in STATE.c_idle:
            STATE.c_idle[node.id] = (time.monotonic(), float(len(node.key)))

    def _park_remove(self, node, medium) -> None:
        gpu = medium is None or medium == StorageMedium.GPU
        gone = not gpu or not node.backuped              # the node leaves the tree
        if gone:
            STATE.next_use.pop(node.id, None)
            STATE.disk_ok.discard(node.id)
        if not STATE.leader:
            return
        now = time.monotonic()
        tier, idle, ev = ("g", STATE.g_idle, STATE.g_evicted) if gpu else ("c", STATE.c_idle, STATE.c_evicted)
        r = idle.pop(node.id, None)
        w = float(len(node.key))
        STATE.live.end_idle(now, tier, now - r[0] if r is not None else 0.0, w, True)
        if gpu and gone:
            STATE.c_idle.pop(node.id, None)              # deleted without a host copy
        for h in self.park_hashes(node):
            ev[h] = now

    # --- paths and pages --------------------------------------------------------------
    def park_path(self, node_id):
        """Nodes from node_id up to the root (excluded); [] if the node left the tree."""
        node = self.tree_core._node_arena.get(node_id)
        root = self.tree_core.root_node
        out = []
        while node is not None and node is not root:
            out.append(node)
            node = node.parent
        return out

    def park_hashes(self, node):
        """Page hashes of a node (SGLang's storage chain hash); computed and kept on the node when
        storage is off, as the KV events recorder does."""
        if node.hash_value is None:
            missing = []
            cur = node
            while cur is not None and cur.parent is not None and cur.hash_value is None:
                missing.append(cur)
                cur = cur.parent
            for n in reversed(missing):
                n.hash_value = compute_node_hash_values(n, self.page_size)
        return node.hash_value or []

    def park_prompt_hashes(self, req):
        ps = self.page_size
        n = len(req.origin_input_ids) // ps * ps
        if n <= 0:
            return []
        return get_hash_str(list(req.origin_input_ids[:n]), None, page_size=ps)

    def park_on_disk(self, path) -> int:
        ps = self.page_size
        return sum(ps for node in path for h in (node.hash_value or ()) if h in STATE.disk_pages)

    # --- storage placement ------------------------------------------------------------
    def write_backup_storage(self, node_id) -> None:
        """Called at each write-through ack: only a node the rule placed on disk goes to storage."""
        if node_id in STATE.disk_ok:
            STATE.disk_ok.discard(node_id)
            return self._park_write(node_id)
        STATE.counts["write_skipped"] += 1

    def _park_write(self, node_id) -> None:
        super().write_backup_storage(node_id)
        node = self.tree_core._node_arena.get(node_id)
        if node is not None and node.hash_value:
            STATE.disk_pages.update(node.hash_value)

    def park_write_path(self, path) -> None:
        """Late writes for an ins or park placement: host copies go to storage now; a node whose host
        copy is not there yet goes on when its write-through ack lands."""
        if not self.enable_storage:
            return
        for node in path:
            hv = self.park_hashes(node)
            if hv and all(h in STATE.disk_pages for h in hv):
                continue
            if node.write_through_pending_id is not None or (not node.backuped and not node.evicted):
                STATE.disk_ok.add(node.id)                 # the D->H copy is in flight or not issued
            elif node.backuped:
                self._park_write(node.id)
                STATE.counts["write_late"] += 1
            else:
                STATE.counts["write_lost"] += 1            # gone from host memory before the decision


def decide_entry(key, entry, surv, now) -> None:
    """Leader: run the rule for a finished request and queue its placement for the next tick."""
    if entry.get("decided"):
        return
    entry["decided"] = True
    cache = STATE.cache
    path = cache.park_path(entry["leaf"])
    if not path:
        STATE.counts["gone"] += 1
        STATE.outbox.append((key, None, False))       # every rank drops the entry
        return
    S = interp_surv(np.nan_to_num(np.asarray(surv, float), nan=0.5), TG)
    on_disk = cache.park_on_disk(path) if cache.enable_storage else 0
    d = decide(S, entry["tokens"], on_disk, STATE.idle_g(now), STATE.idle_c(now), STATE.live,
               STATE.params(), now)
    if not cache.enable_storage and d["disk"]:
        d["opt"] = "none" if d["opt"] == "ins" else "drop"     # no disk tier: the same CPU placement
        d["disk"] = False
    STATE.counts[d["opt"]] += 1
    t_pred = NEVER if d["eta"] < 0 else now + d["eta"]
    STATE.outbox.append((key, t_pred, bool(d["disk"])))
    STATE.last_decisions.append(dict(opt=d["opt"], eta=round(d["eta"], 2), tokens=entry["tokens"],
                                     Q=round(d["Q"], 3), price_c=round(d["price_c"], 4)))
    del STATE.last_decisions[:-100]


def apply_placement(key, t_pred, disk) -> None:
    """Every rank, from a tick: place the finished request's path."""
    entry = STATE.entries.pop(key, None)
    if entry is None or t_pred is None:
        return
    cache = STATE.cache
    path = cache.park_path(entry["leaf"])
    policy.place(path, t_pred)
    if disk:
        cache.park_write_path(path)


def park_cache_factory(ctx):
    """--radix-cache-backend park: a UnifiedRadixCache on the Python TreeCore (the Rust core keeps
    eviction native and has no node_by_id), with the park eviction strategy."""
    from sglang.srt.mem_cache.registry import create_unified_radix_cache

    if (ctx.params.eviction_policy or "").lower() != "park":
        raise ValueError("--radix-cache-backend park needs --radix-eviction-policy park")
    if ctx.is_hybrid_swa or ctx.is_hybrid_ssm:
        raise ValueError("--radix-cache-backend park supports full-attention models only")
    ctx.params.tree_core_backend = "python"
    return create_unified_radix_cache(ctx, cache_class=ParkRadixCache)
