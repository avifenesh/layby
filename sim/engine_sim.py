#!/usr/bin/env python3
"""Iteration-level simulator of the phase C/D engine, for scoring placement and prefetch decisions on p95.

The engine evals showed the resumed-turn tail is queueing: tail turns arrive with about 10 requests in
flight while the 116k-token GPU KV holds about four 25k-token contexts, and many recompute after gaps
under a second because their blocks were evicted while they waited. A per-warm score cannot see that, so
this models the engine the replay ran against (Qwen3-8B, vLLM 0.30.0, chunked prefill, 16 GiB GPU KV,
28 GiB write-through CPU tier, gated disk tier) closely enough to rank decision sources by p50 and p95.

  scheduler   FCFS. Each iteration: one token per decoding request, then prefill chunks for running
              requests, then admission of waiting requests while the token budget and KV allow (the
              first request that does not fit blocks the rest, as in vLLM V1).
  GPU cache   a finished request's blocks stay cached; new allocations evict the least recently freed
              session first, tail blocks first (vLLM frees blocks in reverse order).
  CPU tier    write-through at finish; capacity in tokens. LRU (C0) or ParkCachePolicy: a hinted session
              is predicted next at touch + park_eta (never for park), unhinted the LRU way
              (now + time since last use); eviction drops the session predicted latest, whole (losing a
              head chunk makes the rest unmatchable); a hint 2 s past due falls back to LRU.
  disk        a request with park_eta < 0 writes its session; a disk hit is promoted to the CPU tier at
              DISK_GBPS, then loaded to the GPU at CPU_GBPS.
  loads       one CPU->GPU link and one disk link, serialized; a request prefills after its load.
  step time   T0 + decode KV bytes / BW + prefill tokens * F0 * (1 + ATT * mean position / 1000).

Use simulate(W, None, Params()) for C0 (LRU, no disk tier) and simulate(W, "x", Params(), rule) for a
policy; sim.eval_pop drives it over sampled workloads.
"""
import heapq, math, random
import numpy as np

KIB = 144
NEVER = 1e18
GRACE = 2.0


class Params:
    def __init__(self, **kw):
        self.gpu_tokens = 116496
        self.cpu_tokens = 30064771072 // (KIB * 1024)
        self.budget = 2048          # max_num_batched_tokens
        self.t0 = 0.020             # per-iteration fixed cost (weights read, launch)
        self.bw = 800e9             # effective HBM bandwidth for decode KV reads
        self.f0 = 1 / 9000          # seconds per prefill token at position 0
        self.att = 0.03             # attention growth per 1000 tokens of position
        self.cpu_gbps = 20.0
        self.disk_gbps = 1.5
        self.disk_write = True      # parked writes share the disk link with loads (False: free writes)
        self.load_gate = None       # skip a warm when this many requests wait (None: never skip)
        self.jitter = 0.0           # lognormal sigma on each step time (run-to-run variation)
        self.defer = False          # disk-tier arms: a request's first lookup returns RETRY (one step lost)
        self.lookup_s = 0.0         # scheduler time per lookup call (waiting requests the loop reaches)
        self.seed = 0
        self.tw = 120.0             # Live averaging window, seconds (estimator smoothing, not a decision constant)
        self.window = None          # run length in seconds: no send after it (None: every turn is sent)
        self.__dict__.update(kw)


class Req:
    __slots__ = ("sess", "turn", "P", "O", "warm", "eta", "t_send", "t_first", "t_done", "hit", "ext",
                 "rem", "out", "ready", "alloc", "kind", "looked", "disk", "t_adm")

    def __init__(self, sess, turn, P, O, warm, eta, t, kind):
        self.sess, self.turn, self.P, self.O, self.warm, self.eta, self.t_send = sess, turn, P, O, warm, eta, t
        self.t_first = self.t_done = None
        self.hit = self.ext = 0
        self.rem = P
        self.out = 0
        self.ready = 0.0
        self.alloc = 0
        self.kind = kind
        self.looked = False
        self.disk = None             # write to disk at finish; None: when eta < 0
        self.t_adm = None


from .live import AG, Live   # shared with the engine adapters


def simulate(W, src, p, rule=None):
    """Replay W's sessions closed-loop through the simulated engine; src None is C0 (LRU, no hints).
    rule: an object with decide(view) -> dict(eta, disk, pre, warm_eta), called when a turn finishes
    with the live engine state; it replaces the replay's eta_<src>/pre_<src> (src must not be None)."""
    S = W["sessions"]
    live = Live(p.tw)
    # baseline rules may order evictions themselves: gpu_key(sess, now) smallest goes first, cpu_key(sess, now,
    # last_use) largest goes first; default LRU on the GPU and predicted next use in the CPU tier
    gpu_key = rule is not None and hasattr(rule, "gpu_key")
    cpu_key = rule is not None and hasattr(rule, "cpu_key")
    n = len(S)
    H = [0] * n                      # session prefix length after its last finished request
    gpu_idle = {}                    # session -> cached tokens on GPU (idle)
    gpu_free_t = {}                  # session -> time its blocks were freed (LRU)
    cpu = {}                         # session -> tokens in the CPU tier
    cpu_hint = {}                    # session -> predicted next use (ParkCachePolicy) or None
    cpu_last = {}                    # session -> last use
    g_ev_t, c_ev_t = {}, {}          # session -> last time the GPU (CPU) tier evicted its tokens
    gpu_pin = {}                     # session -> time until which its idle GPU blocks are held (baselines)
    disk = {}                        # session -> tokens whose disk write completed
    disk_writing = {}                # session -> tokens written or being written
    running_sess = {}                # session -> tokens computed by a running request (shared blocks)
    alloc = 0
    link_cpu = link_disk = 0.0
    waiting, running = [], []
    ev = []                          # (time, seq, kind, payload)
    seq = 0
    out = []
    warms = 0
    skipped = 0
    rnd = random.Random(int(p.seed))
    lookup_time = 0.0
    io = dict(disk_read=0, disk_write=0, cpu_load=0)   # tokens moved
    wlog = {}                        # (session, turn) -> the warm before it: [sent, admitted, done]

    def push(t, kind, payload):
        nonlocal seq
        heapq.heappush(ev, (t, seq, kind, payload)); seq += 1

    for i, s in enumerate(S):
        push(s["start"], "send", (i, 0))

    def gpu_evict(need, keep, now):
        """Free cached idle blocks (never session `keep`'s) until `need` more tokens fit."""
        idle = sum(gpu_idle.values())
        while alloc + idle + need > p.gpu_tokens:
            cand = [x for x in gpu_idle if x != keep]
            if not cand:
                return False
            # a pinned session's blocks wait out their pin unless nothing else is left (then the pin with the
            # earliest end goes, as Continuum's deadlock victim)
            free = [x for x in cand if gpu_pin.get(x, -1.0) <= now]
            if free:
                v = min(free, key=lambda x: rule.gpu_key(x, now) if gpu_key else gpu_free_t[x])
            else:
                v = min(cand, key=lambda x: gpu_pin[x])
            over = alloc + idle + need - p.gpu_tokens
            live.end_idle(now, "g", now - gpu_free_t[v], min(gpu_idle[v], over), True)
            g_ev_t[v] = now
            if gpu_idle[v] <= over:
                idle -= gpu_idle.pop(v)
            else:
                gpu_idle[v] -= over; idle -= over   # tail blocks go first; the prefix stays matchable
        return True

    def cpu_pred(k, now):
        h = cpu_hint.get(k)
        if h is not None and src is not None:
            if h < NEVER and now > h + GRACE:
                cpu_hint[k] = None; cpu_last[k] = h
                return now + (now - h)
            return h
        return now + (now - cpu_last[k])

    def cpu_store(k, tokens, eta, now, protect):
        cpu.pop(k, None)
        while cpu and sum(cpu.values()) + tokens > p.cpu_tokens:
            cand = [x for x in cpu if x not in protect] or list(cpu)
            v = max(cand, key=lambda x: rule.cpu_key(x, now, cpu_last[x]) if cpu_key else cpu_pred(x, now))
            live.end_idle(now, "c", now - cpu_last[v], cpu[v], True)
            c_ev_t[v] = now
            del cpu[v]; cpu_hint.pop(v, None)
        cpu[k] = tokens
        cpu_last[k] = now
        cpu_hint[k] = None if (src is None or eta is None) else (NEVER if eta < 0 else now + eta)

    def turn_of(i, j):
        return S[i]["turns"][j]

    def send(i, j, now, warm=False, eta=None):
        nonlocal warms
        t = turn_of(i, j)
        if warm:
            r = Req(i, j, H[i], 1, True, eta, now, t["kind"])
            warms += 1
        else:
            e = t.get(f"eta_{src}") if src else None
            r = Req(i, j, H[i] + t["new_tokens"], t["out_len"], False, e, now, t["kind"])
        waiting.append(r)

    def admit(r, now):
        """Prefix lookup and KV allocation at the moment the scheduler takes the request."""
        nonlocal alloc, link_cpu, link_disk
        k = r.sess
        own = gpu_idle.get(k, 0)
        if not gpu_evict(r.P + r.O - min(own, r.P + r.O), k, now):
            return False
        g = min(r.P, max(own, running_sess.get(k, 0)))
        if own:
            live.end_idle(now, "g", now - gpu_free_t[k], own, False)
        gpu_idle.pop(k, None)
        gpu_pin.pop(k, None)
        alloc += r.P + r.O
        r.alloc = r.P + r.O
        c, d = cpu.get(k, 0), disk.get(k, 0)
        if k in cpu:
            if not any(x.sess == k for x in running):
                live.end_idle(now, "c", now - cpu_last[k], cpu[k], False)
            cpu_last[k] = now
            if src is not None:
                cpu_hint[k] = None if r.eta is None else (NEVER if r.eta < 0 else now + r.eta)
        ext = max(c, d)
        r.hit = g
        if ext > g:
            L = (min(ext, r.P) - g) * KIB * 1024
            if c >= ext:
                live.add(now, busy_cpu=L / (p.cpu_gbps * 1e9), n_cpu=1.0, wait_cpu=max(link_cpu - now, 0.0))
                link_cpu = max(link_cpu, now) + L / (p.cpu_gbps * 1e9); r.ready = link_cpu
                io["cpu_load"] += L // (KIB * 1024)
            else:
                # promoted disk -> CPU, then loaded CPU -> GPU (the second hop is not queued on link_cpu)
                live.add(now, busy_disk=L / (p.disk_gbps * 1e9), n_disk=1.0, wait_disk=max(link_disk - now, 0.0))
                link_disk = max(link_disk, now) + L / (p.disk_gbps * 1e9); r.ready = link_disk + L / (p.cpu_gbps * 1e9)
                io["disk_read"] += L // (KIB * 1024)
            r.ext = min(ext, r.P) - g
        r.rem = r.P - g - r.ext
        if r.rem <= 0:
            r.rem = 1                    # vLLM always recomputes the last prompt token
        if not r.warm:
            # latency this request loses to its own prefix missing from the GPU, and the part of it a
            # full CPU copy would have avoided
            pre = min(H[k], r.P)
            rec = max(0, pre - g - r.ext)
            lost = max(r.ready - now, 0.0) + rec * p.f0 * (1 + p.att * (g + r.ext + rec / 2) / 1000)
            if pre > g:
                in_cpu = c >= pre
                lost_c = 0.0 if in_cpu else max(0.0, lost - (pre - g) * KIB * 1024 / (p.cpu_gbps * 1e9))
                live.add(now, miss_g=lost, miss_c=lost_c)
                # the same loss with what it delays: link waits by the link's busy fraction, recompute by
                # the requests in prefill
                u = live.rate("busy_cpu" if c >= ext else "busy_disk", now)
                rec_t = rec * p.f0 * (1 + p.att * (g + r.ext + rec / 2) / 1000)
                full = max(r.ready - now, 0.0) * (1 + min(u, 1.0)) + rec_t * (1 + live.rate("pf", now))
                if g_ev_t.get(k, -1.0) >= gpu_free_t.get(k, math.inf):
                    live.miss(now, "g", now - g_ev_t[k], full)
                if not in_cpu and c_ev_t.get(k, -1.0) >= cpu_last.get(k, math.inf):
                    live.miss(now, "c", now - c_ev_t[k], max(0.0, full - (pre - g) * KIB * 1024 / (p.cpu_gbps * 1e9)))
        return True

    now = 0.0
    while ev or waiting or running:
        if not waiting and not running:
            now = max(now, ev[0][0])
        while ev and ev[0][0] <= now:
            _, _, kind, pl = heapq.heappop(ev)
            if kind == "send":
                send(pl[0], pl[1], now)
            elif kind == "disk":
                disk[pl[0]] = max(disk.get(pl[0], 0), pl[1])
            elif kind == "warm":
                i, j, eta = pl
                if p.load_gate is not None and len(waiting) >= p.load_gate:
                    skipped += 1
                else:
                    send(i, j, now, warm=True, eta=eta)
        # schedule one iteration
        budget = p.budget
        dec = [r for r in running if r.rem == 0 and now >= r.ready]
        budget -= len(dec)
        chunks = []
        for r in running:
            if r.rem > 0 and now >= r.ready and budget > 0:
                c = min(r.rem, budget); chunks.append((r, c)); budget -= c
        defer = p.defer and src is not None
        lookups = deferred = 0
        wi = 0
        while wi < len(waiting) and budget > 0:
            r = waiting[wi]
            lookups += 1
            if defer and not r.looked:
                r.looked = True; wi += 1; deferred += 1   # async disk lookup in flight: skipped this step
                continue
            if not admit(r, now):
                break
            waiting.pop(wi); running.append(r); r.t_adm = now
            live.add(now, n_adm=1.0, qwait=now - r.t_send)
            running_sess[r.sess] = max(running_sess.get(r.sess, 0), r.hit)
            if now >= r.ready:
                c = min(r.rem, budget); chunks.append((r, c)); budget -= c
        if not dec and not chunks:
            if deferred:
                now += p.t0 + lookups * p.lookup_s   # an idle step while deferred lookups resolve
                lookup_time += lookups * p.lookup_s
                continue
            nxt = [r.ready for r in running if r.ready > now]
            t_ev = ev[0][0] if ev else math.inf
            t_next = min(nxt + [t_ev])
            if t_next == math.inf:
                break
            now = t_next
            continue
        ctx_bytes = sum((r.P + r.out) for r in dec) * KIB * 1024
        pre = sum(c * p.f0 * (1 + p.att * (r.P - r.rem + c / 2) / 1000) for r, c in chunks)
        lk = lookups * p.lookup_s if defer else 0.0
        dt = p.t0 + ctx_bytes / p.bw + pre + lk
        lookup_time += lk
        t_prev = now
        now += dt * rnd.lognormvariate(0, p.jitter) if p.jitter else dt
        live.add(now, pf=(len(waiting) + sum(1 for r in running if r.rem > 0)) * (now - t_prev), nw=len(waiting) * (now - t_prev))
        for r, c in chunks:
            r.rem -= c
            running_sess[r.sess] = max(running_sess.get(r.sess, 0), r.P - r.rem)
            if r.rem == 0:
                r.out = 1; r.t_first = now
        for r in dec:
            r.out += 1
        for r in [r for r in running if r.rem == 0 and r.out >= r.O]:
            running.remove(r)
            r.t_done = now
            k = r.sess
            alloc -= r.alloc
            tot = r.P + (0 if r.warm else r.O)
            gpu_idle[k] = max(gpu_idle.get(k, 0), tot); gpu_free_t[k] = now
            if not any(x.sess == k for x in running):
                running_sess.pop(k, None)
            dec_r = None
            if rule is not None and not r.warm:
                dec_r = rule.decide(dict(sess=k, turn=turn_of(k, r.turn), tokens=tot, now=now, p=p, live=live,
                                         link_cpu=link_cpu, link_disk=link_disk, n_wait=len(waiting),
                                         on_disk=max(disk.get(k, 0), disk_writing.get(k, 0)),
                                         idle_g=[(now - gpu_free_t[x], gpu_idle[x]) for x in gpu_idle if x != k],
                                         idle_c=[(now - cpu_last[x], cpu[x]) for x in cpu
                                                 if x != k and not any(y.sess == x for y in running)]))
                r.eta, r.disk = dec_r["eta"], dec_r["disk"]
                if dec_r.get("gpu_pin"):
                    gpu_pin[k] = now + dec_r["gpu_pin"]
            cpu_store(k, tot, r.eta, now, {x.sess for x in running})
            to_disk = r.disk if r.disk is not None else (r.eta is not None and r.eta < 0)
            if to_disk and src is not None:
                if p.disk_write:
                    # REQUEST_LEVEL write of the session; blocks already on disk are skipped
                    new = tot - max(disk.get(k, 0), disk_writing.get(k, 0))
                    if new > 0:
                        io["disk_write"] += new
                        live.add(now, busy_disk=new * KIB * 1024 / (p.disk_gbps * 1e9), n_disk=1.0, wait_disk=max(link_disk - now, 0.0))
                        link_disk = max(link_disk, now) + new * KIB * 1024 / (p.disk_gbps * 1e9)
                        disk_writing[k] = tot
                        push(link_disk, "disk", (k, tot))
                else:
                    disk[k] = tot
            if r.warm:
                wlog[(k, r.turn)] = [r.t_send, r.t_adm, now]
                continue
            H[k] = tot
            out.append(dict(session=S[k]["id"], turn=r.turn, kind=r.kind, ttft=r.t_first - r.t_send,
                            hit=r.hit, ext=r.ext, P=r.P, queue=r.t_adm - r.t_send, load=max(r.ready - r.t_adm, 0.0),
                            warm=None if (k, r.turn) not in wlog else [x - r.t_send for x in wlog[(k, r.turn)]]))
            t = turn_of(k, r.turn)
            if t["gap_after"] is None or r.turn + 1 >= len(S[k]["turns"]):
                continue
            if p.window is not None and now + t["gap_after"] > p.window:
                continue                     # the session does not return within the run (as replay.py --window)
            push(now + t["gap_after"], "send", (k, r.turn + 1))
            if dec_r is not None:
                w, we = dec_r.get("pre"), dec_r.get("warm_eta")
            else:
                w = t.get(f"pre_{src}") if src else None
                e = t.get(f"eta_{src}")
                we = None if e is None or w is None else (-1.0 if e < 0 else max(e - w, 0.5))
            if w is not None and w < t["gap_after"]:
                push(now + w, "warm", (k, r.turn + 1, we))
    return out, dict(warms=warms, skipped=skipped, wall=now, lookup_time=lookup_time, **{k: v * KIB * 1024 / 1e9 for k, v in io.items()})


def summarize(rows):
    v = np.array([r["ttft"] for r in rows if r["turn"] > 0])
    tok = dict(queried=sum(r["P"] for r in rows) / 1e6, gpu=sum(r["hit"] for r in rows) / 1e6, cpu=sum(r["ext"] for r in rows) / 1e6)
    tok["recomp"] = tok["queried"] - tok["gpu"] - tok["cpu"]
    rec = np.mean([r["P"] - r["hit"] - r["ext"] > 0.5 * r["P"] for r in rows if r["turn"] > 0])
    return dict(n=len(v), p50=float(np.quantile(v, .5)), p95=float(np.quantile(v, .95)), p99=float(np.quantile(v, .99)),
                recompute=float(rec), tok=tok)


