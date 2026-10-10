#!/usr/bin/env python3
"""Phase R workloads: tuning, pilot and measured, disjoint by session (and by user where the source has users),
disjoint from every session the model or an earlier engine round has seen, one set per serving model.

Candidate pool, per public source with real gaps (SWE-chat, TraceLab Claude Code, WildChat, Copilot agent traces):
every extracted session, minus the sessions in USED (used_sessions.py), minus, on sources with user ids, every
session of a user who is not in the held-out user set or who has any session in USED. A session is kept to its
first boundary that never returned (the session ends there) and needs at least --min-turns turns (4 on WildChat).
SemiAnalysis cc-traces (--cc, extract_cctraces.py) is a separate held-out source: it has no text and no turn kind,
so it gets its own workload per model.

Partition: a stable hash of (source, user) on sources with users and of (source, session) otherwise sends each unit
to tuning, pilot or measured (--split fractions). Within a partition sessions are taken in hash order, round robin
over the sources (an even share per source).

Per model (--model NAME,CAP,LIVE,C0,TPOT,B): turns follow build_replay_pop.py (output 384 tokens after a human
wait, 128 otherwise; new tokens from the real context curve). When a turn would exceed the context CAP, the
oldest history is truncated: the turn's prompt starts over (keep = 0) with the newest --retain x CAP tokens of the
history plus the turn's new tokens, so prefix reuse resumes after the cut, as after a client-side compaction.
The scored state keeps the real, unscaled context. Gaps are capped at --gap-cap (gap_uncapped keeps the real one).

Sizing: a turn's end-to-end time at the heavy load is estimated as C0 + TPOT x out_len + B x prompt/1000 seconds
(least squares on the C0 arms of the 2026-10 heavy rounds); a session lasts the sum of its turn times and capped
gaps. A session longer than --duration is cut at its last turn that ends within it. Sessions are added until the
summed session time reaches LIVE x duration (LIVE: the mean number of live sessions measured in those rounds), and
each starts uniformly in [0, duration - its length], so the run to completion is estimated at --duration seconds.
Tuning also gets the shorter stages of section 8 (--stages, nested prefixes of the tuning order).

Outputs in OUT: NAME.json (engine/replay.py format, content-free, surv_v6 null until the frozen model scores the
side file), NAME.refs.parquet (key, session, turn, src, source_session, feat_ref: which source boundary each turn
is; the side-file builder joins text to it), NAME.manifest.json, and pool.json (counts and checks).

Usage: build_workloads.py OUT --feat feat5.parquet --used used.parquet --cc cc_requests.parquet \
       --model qwen3_8b,32768,8.5,0.649,0.0190,0.0266 --model glm53_flash,262144,63,18.6,0.443,1.185
"""
import argparse
import hashlib
import json
import os

import numpy as np
import pandas as pd

SRC = ["swechat", "tracelab_claude", "wildchat", "copilot"]
HAS_USERS = {"swechat", "tracelab_claude", "wildchat"}
SOURCES = {   # pinned public versions (bench/LICENSES.md, bench/build/SHA256SUMS.raw)
    "swechat": dict(name="SWE-chat", url="https://huggingface.co/datasets/SALT-NLP/SWE-chat",
                    revision="f66cca95b14caaa4177f7ed5eaa424608dadcffa", license="ODC-BY-1.0",
                    sha256={"sessions.parquet": "2ada63973b182b691318916ca8c813e694091400e43744eca9cad3da2d958a95"}),
    "tracelab_claude": dict(name="TraceLab public coding-agent trace (Claude Code)", url="https://github.com/uw-syfi/TraceLab",
                            revision="release v0.0.2", license="CC-BY-4.0",
                            sha256={"syfi_coding_trace.duckdb": "a7bab286bc640844560850965ccf47975cf66407154132abaab90f27ec9be744"}),
    "wildchat": dict(name="WildChat-4.8M", url="https://huggingface.co/datasets/allenai/WildChat-4.8M",
                     revision="c827c6df8fcf008219ffaffa4d1dd77491099367", license="ODC-BY-1.0", sha256={}),
    "copilot": dict(name="GitHub Copilot Coding Agent Traces 2026", url="https://github.com/Azure/AzurePublicDataset",
                    revision="release ghcp-coding-agent-2026", license="CC-BY-4.0",
                    sha256={f"date.2026-06-0{i}.tar.gz": h for i, h in enumerate([
                        "cacad69f23d3c87ac54d788fec3f8e9c84df82c0cc9638d693340423c14e68e9",
                        "95b0450041095dbe88181b88841c64b1e42f55a0127aa7abe91c1c9d3b424240",
                        "622a5b0742eeb196d69d9a5ee5f6457cf82d432ddc435b9eb76eb5bb4363f1d0",
                        "fd81d45e0149cae10aa193f2c2536ae709a5eba61b5aa886e534587e80473028",
                        "ababba256966dc8bb005c98fbb4b72faea5301e17dcd949fd5b69e71b8239c50",
                        "9546e3fd156d379e2420b8a0ada496ff507cf74aab32b9fc97bd1b6b353fa182",
                        "8b6da125b486aea3eb37c752adf6be14d61dd1e6a584f5967bca7077fafe8c3c"], start=1)}),
    "cc_traces": dict(name="SemiAnalysis CC Traces, Weka, no subagents (May 12 2026)",
                      url="https://huggingface.co/datasets/semianalysisai/cc-traces-weka-no-subagents-051226",
                      revision="0ae681ae27a0e3e716b344cb21f1b01bb1313d52", license="Apache-2.0",
                      sha256={"traces.jsonl": "fcac6e738d3457dbb97e97168fec8b720b42bb281a44af1e933ab6cd01a4041d"}),
}
TAG = {"swechat": "swe", "tracelab_claude": "tra", "wildchat": "wil", "copilot": "cop", "cc_traces": "ccx"}

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--feat", required=True)
ap.add_argument("--used", required=True)
ap.add_argument("--cc")
ap.add_argument("--model", action="append", required=True)
ap.add_argument("--duration", type=float, default=2700.0)
ap.add_argument("--stages", type=float, nargs="*", default=[1200.0, 2400.0])
ap.add_argument("--split", type=float, nargs=2, default=[0.25, 0.15], help="tuning and pilot fractions")
ap.add_argument("--retain", type=float, default=0.16)
ap.add_argument("--gap-cap", type=float, default=300.0)
ap.add_argument("--min-turns", type=int, default=6)
ap.add_argument("--min-turns-chat", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--used-report", help="used_sessions.py report (per use: refs, sessions) to copy into the manifests")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def h01(*parts):
    d = hashlib.blake2b("|".join(map(str, ("phase-r",) + parts)).encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") / 2.0**64


# ---------------------------------------------------------------- candidate pool
F = pd.read_parquet(a.feat, columns=["src", "split", "uhold", "session", "user", "kind", "ref", "returned", "gap",
                                     "ctx", "t", "tool", "prog"], filters=[("src", "in", SRC)])
for c in F.columns:
    if str(F[c].dtype).startswith("string"):
        F[c] = F[c].astype(object)
F["returned"] = F.returned.astype(bool)
F["uhold"] = F.uhold.astype(bool)
U = pd.read_parquet(a.used)
used = set(zip(U.src, U.session))
F["used"] = [x in used for x in zip(F.src, F.session)]
su = F[F.used].groupby("src").user.agg(lambda s: set(s.dropna()))
F["user_seen"] = [u in su.get(s, set()) for s, u in zip(F.src, F.user)]
F = F.sort_values(["src", "session", "t"]).reset_index(drop=True)

pool, counts = {}, {}
for s in SRC:
    d = F[F.src == s]
    c = {"sessions_extracted": int(d.session.nunique()),
         "users_extracted": int(d.user.nunique()) if s in HAS_USERS else None,
         "sessions_seen": int(d[d.used].session.nunique())}
    d = d[~d.used]
    c["after_minus_seen"] = int(d.session.nunique())
    if s in HAS_USERS:
        c["after_heldout_users"] = int(d[d.uhold].session.nunique())
        d = d[d.uhold & ~d.user_seen]
        c["after_heldout_unseen_users"] = int(d.session.nunique())
    keep = {}
    mt = a.min_turns_chat if s == "wildchat" else a.min_turns
    for sess, g in d.groupby("session", sort=True):
        nr = np.flatnonzero(~g.returned.values)
        g = g.iloc[: nr[0] + 1] if len(nr) else g
        if len(g) >= mt:
            keep[sess] = g
    c["after_min_turns"] = len(keep)
    c["users_in_pool"] = len({g.user.iloc[0] for g in keep.values()}) if s in HAS_USERS else None
    counts[s] = c
    pool[s] = keep
    print(s, c, flush=True)
del F

cc = None
if a.cc:
    cc = pd.read_parquet(a.cc).sort_values(["trace", "idx"])
    import pyarrow.parquet as pq
    ft = pq.read_table(a.feat, columns=["session", "user"])          # every source, private ones included
    ids = set(cc.trace.unique())
    hit = sum(len(ids & set(ft[c].to_pandas().astype(str).unique())) for c in ("session", "user"))
    del ft
    counts["cc_traces"] = {"sessions_extracted": len(ids), "requests": int(len(cc)),
                           "sessions_seen": int(len(ids & {x for s, x in used})), "ids_in_feature_table": hit}

# ---------------------------------------------------------------- partition
PARTS = ["tuning", "pilot", "measured"]


def part_of(src, unit):
    u = h01(src, unit)
    return PARTS[0] if u < a.split[0] else PARTS[1] if u < a.split[0] + a.split[1] else PARTS[2]


order = {p: {} for p in PARTS}
for s in SRC:
    items = []
    for sess, g in pool[s].items():
        unit = g.user.iloc[0] if s in HAS_USERS else sess
        items.append((h01("order", s, sess), sess, part_of(s, unit), unit))
    for p in PARTS:
        order[p][s] = [(sess, unit) for _, sess, pp, unit in sorted(items) if pp == p]
for p in PARTS:
    print(p, {s: len(order[p][s]) for s in SRC}, flush=True)


def interleave(lists):
    out, i = [], 0
    while any(i < len(x) for x in lists):
        out += [x[i] for x in lists if i < len(x)]
        i += 1
    return out


# ---------------------------------------------------------------- turns per model
def spec(m):
    n, cap, live, c0, tpot, b = m.split(",")
    return dict(name=n, cap=int(cap), live=float(live), c0=float(c0), tpot=float(tpot), b=float(b))


def est(M, prompt, out_len):
    return M["c0"] + M["tpot"] * out_len + M["b"] * prompt / 1000.0


def place(M, hist, new, out_len):
    """Apply the context cap. Returns (keep, new_tokens, ctx_after): keep None means the history is reused whole."""
    if hist + new + out_len <= M["cap"]:
        return None, new, hist + new + out_len
    n = min(hist, int(a.retain * M["cap"])) + new
    n = min(n, M["cap"] - out_len)
    return 0, n, n + out_len


def turns_feat(M, g):
    c = np.maximum.accumulate(g.ctx.ffill().fillna(8192).values.astype(float))
    T, hist, prev_real, prev_out = [], 0, 0, 0
    for j, r in enumerate(g.itertuples()):
        out_len = 384 if r.kind == "human" else 128
        cx = int(c[j])
        new_real = max(64, cx - prev_real - prev_out) if j else max(256, cx - out_len)
        keep, new, ctx = place(M, hist, new_real, out_len)
        gap = float(r.gap) if r.returned else None
        T.append(dict(new_tokens=int(new), out_len=out_len, kind=r.kind, ctx=int(ctx),
                      gap_after=None if gap is None else min(gap, a.gap_cap), gap_uncapped=gap,
                      tool=r.tool if isinstance(r.tool, str) else None, prog=r.prog if isinstance(r.prog, str) else None,
                      keep=keep, ctx_real=cx, new_real=int(new_real), _ref=r.ref, _why="cap" if keep is not None else None))
        hist, prev_real, prev_out = ctx, cx, out_len
    return T


def turns_cc(M, g):
    T, hist, prev_in, prev_out, prev_nb = [], 0, None, 0, 0
    for r in g.itertuples():
        out_len = max(1, int(r.out_tok or 1))
        shared = min(int(r.prefix_blocks) * 64, prev_in) if prev_in is not None else 0
        if prev_in is None:
            keep_src, new_real = None, int(r.in_tok)
        elif r.prefix_blocks >= prev_nb - 1:        # every full block of the previous prompt is reused: append
            keep_src, new_real = None, max(1, int(r.in_tok) - prev_in - prev_out)
        else:                                       # the prefix breaks (compaction, edit, model switch)
            keep_src, new_real = shared, max(1, int(r.in_tok) - shared)
        if keep_src is not None:                    # replay the break, then apply the context cap
            hist = min(hist, keep_src)
        keep, new, ctx = place(M, hist, new_real, out_len)
        why = "cap" if keep is not None else None
        if keep is None and keep_src is not None:
            keep, why = hist, "source"
        gap = None if pd.isna(r.gap_after) else float(r.gap_after)
        T.append(dict(new_tokens=int(new), out_len=out_len, kind=None, ctx=int(ctx),
                      gap_after=None if gap is None else min(gap, a.gap_cap), gap_uncapped=gap, tool=None, prog=None,
                      keep=keep, ctx_real=int(r.in_tok) + out_len, new_real=int(new_real), _ref=f"{r.trace}:{r.idx}",
                      _why=why))
        hist, prev_in, prev_out, prev_nb = ctx, int(r.in_tok), out_len, int(r.n_blocks)
    return T


def cut(M, T, limit):
    """Turns that end within `limit` seconds of the session start; returns (turns, length, cut)."""
    tm, out = 0.0, []
    for t in T:
        e = est(M, t["ctx"] - t["out_len"], t["out_len"])
        if out and tm + e > limit:
            break
        out.append(dict(t, _e2e=e))
        tm += e
        if t["gap_after"] is None:
            break
        tm += t["gap_after"]
    if out:
        out[-1] = dict(out[-1], gap_after=None)     # the replay ends the session here
    length = sum(t["_e2e"] for t in out) + sum(t["gap_after"] or 0.0 for t in out)
    return out, length, len(out) < len(T)


def assemble(M, name, items, duration, build_turns, src_of):
    rng = np.random.default_rng(abs(hash_int(name)) % 2**32)
    sessions, refs, target, total = [], [], M["live"] * duration, 0.0
    for k, (src, sess, unit, g) in enumerate(items):
        full = build_turns(M, g)
        T, L, was_cut = cut(M, full, duration)
        if len(T) < 3:
            continue
        sid = f"{TAG[src]}{len(sessions):04d}"
        start = float(rng.uniform(0, max(0.0, duration - L)))
        turns = []
        why = [t["_why"] for t in T]
        for i, t in enumerate(T):
            refs.append((f"{sid}:{i}", sid, i, src, sess, t["_ref"]))
            turns.append({x: v for x, v in t.items() if not x.startswith("_")} | {"surv_v6": None})
        sessions.append(dict(id=sid, pool=src, seed=1000 + len(sessions), start=start, turns=turns,
                             _src_session=sess, _unit=unit, _len=L, _cut=was_cut, _turns_src=len(full), _why=why))
        total += L
        if total >= target:
            break
    return sessions, refs, total


def hash_int(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "big")


def load(sessions, M, duration):
    ev = []
    for s in sessions:
        tm, cur = s["start"], 0
        for t in s["turns"]:
            ev.append((tm, t["ctx"] - cur)); cur = t["ctx"]
            tm += est(M, t["ctx"] - t["out_len"], t["out_len"]) + (t["gap_after"] or 0.0)
        ev.append((tm, -cur))
    ev.sort()
    peak = float(np.max(np.cumsum([d for _, d in ev])))
    ends = [s["start"] + s["_len"] for s in sessions]
    mx = np.array([max(t["ctx"] for t in s["turns"]) for s in sessions])
    T = [t for s in sessions for t in s["turns"]]
    return dict(sessions=len(sessions), turns=len(T),
                sessions_per_source={k: int(v) for k, v in pd.Series([s["pool"] for s in sessions]).value_counts().items()},
                turns_per_session_p50_p90_max=[float(np.median([len(s["turns"]) for s in sessions])),
                                               float(np.percentile([len(s["turns"]) for s in sessions], 90)),
                                               int(max(len(s["turns"]) for s in sessions))],
                max_ctx_p50_p90_max=[float(np.median(mx)), float(np.percentile(mx, 90)), int(mx.max())],
                kinds={k: int(v) for k, v in pd.Series([t["kind"] for t in T]).value_counts(dropna=False).items()},
                cap_truncations=int(sum(w == "cap" for s in sessions for w in s["_why"])),
                source_prefix_breaks=int(sum(w == "source" for s in sessions for w in s["_why"])),
                sessions_cut_at_duration=int(sum(s["_cut"] for s in sessions)),
                token_turns_M=round(sum(t["ctx"] for t in T) / 1e6, 1),
                prompt_tokens_M=round(sum(t["ctx"] - t["out_len"] for t in T) / 1e6, 1),
                gaps_capped=int(sum(t["gap_uncapped"] is not None and t["gap_uncapped"] > a.gap_cap for t in T)),
                peak_live_kv_tokens=int(peak),
                est_makespan_s=round(max(ends), 0), est_mean_live_sessions=round(sum(s["_len"] for s in sessions) / duration, 2),
                est_duration_target_s=duration)


def write(M, name, sessions, refs, extra):
    W = dict(gap_cap=a.gap_cap, ctx_cap=M["cap"], model=M["name"], retain=a.retain,
             src=f"Layby Phase R {name}", curve="surv_v6 (null until scored from the side file)",
             sessions=[{k: v for k, v in s.items() if not k.startswith("_")} for s in sessions])
    p = os.path.join(a.out, f"{name}.json")
    json.dump(W, open(p, "w"))
    R = pd.DataFrame(refs, columns=["key", "session", "turn", "src", "source_session", "feat_ref"])
    R.to_parquet(os.path.join(a.out, f"{name}.refs.parquet"), index=False)
    L = load(sessions, M, extra["duration"])
    man = dict(workload=f"{name}.json", model=M["name"], ctx_cap=M["cap"], partition=extra["partition"],
               bytes=os.path.getsize(p), sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
               load=L, service_model=dict(c0=M["c0"], tpot=M["tpot"], b_per_1k_prompt=M["b"], live=M["live"]),
               sessions=[dict(id=s["id"], source=s["pool"], source_session=s["_src_session"], turns=len(s["turns"]),
                              turns_in_source=s["_turns_src"], cut_at_duration=s["_cut"]) for s in sessions])
    json.dump(man, open(os.path.join(a.out, f"{name}.manifest.json"), "w"), indent=1)
    print(name, json.dumps(L), flush=True)
    return man


# ---------------------------------------------------------------- build
models = [spec(m) for m in a.model]
built = {}
for M in models:
    for p in PARTS:
        items = [(src, sess, unit, pool[src][sess]) for src, (sess, unit) in
                 interleave([[(s, x) for x in order[p][s]] for s in SRC])]
        name = f"{M['name']}.{p}"
        S, R, _ = assemble(M, name, items, a.duration, turns_feat, None)
        built[name] = write(M, name, S, R, dict(duration=a.duration, partition=p))
        if p == "tuning":
            for d in a.stages:
                nm = f"{M['name']}.tuning_{int(d / 60)}min"
                S2, R2, _ = assemble(M, nm, items, d, turns_feat, None)
                built[nm] = write(M, nm, S2, R2, dict(duration=d, partition=p))
    if cc is not None:
        items = sorted(((h01("order", "cc_traces", tr), tr, g) for tr, g in cc.groupby("trace")), key=lambda x: x[0])
        items = [("cc_traces", tr, tr, g) for _, tr, g in items]
        name = f"{M['name']}.heldout_cc"
        S, R, _ = assemble(M, name, items, a.duration, turns_cc, None)
        built[name] = write(M, name, S, R, dict(duration=a.duration, partition="heldout_cc"))

# ---------------------------------------------------------------- checks
chk = {}
by_part = {}
for name, man in built.items():
    by_part.setdefault(man["partition"], set()).update((s["source"], s["source_session"]) for s in man["sessions"])
for x in PARTS + ["heldout_cc"]:
    for y in PARTS + ["heldout_cc"]:
        if x < y:
            chk[f"overlap_sessions:{x}:{y}"] = len(by_part.get(x, set()) & by_part.get(y, set()))
allw = set().union(*by_part.values())
chk["overlap_with_seen_sessions"] = len(allw & used)
chk["sources_used"] = sorted({s for s, _ in allw})
chk["private_sources_present"] = sorted({s for s, _ in allw} - set(SRC) - {"cc_traces"})
units = {p: {(s, u) for s in SRC for (_, u) in order[p][s] if s in HAS_USERS} for p in PARTS}
for x in PARTS:
    for y in PARTS:
        if x < y:
            chk[f"overlap_users:{x}:{y}"] = len(units[x] & units[y])
json.dump(dict(counts=counts, partitions={p: {s: len(order[p][s]) for s in SRC} for p in PARTS}, checks=chk,
               args=vars(a)), open(os.path.join(a.out, "pool.json"), "w"), indent=1)
print(json.dumps(chk))
rep = json.load(open(a.used_report)) if a.used_report else None
for name, man in built.items():
    srcs = sorted({s["source"] for s in man["sessions"]})
    mine = {(s["source"], s["source_session"]) for s in man["sessions"]}
    man["sources"] = {s: SOURCES[s] for s in srcs}
    man["disjointness"] = dict(
        rule="no session in v6's training lineage (every split of every model dataset version), in any ReturnBench "
             "pool or in any earlier engine workload; on sources with user ids, only held-out users with no seen "
             "session; tuning, pilot and measured partitioned by user (or by session where there are no users)",
        sessions_in_seen_set=len(mine & used), cross_partition=chk, seen_set_sizes=counts, seen_set_uses=rep,
        cc_traces_ids_in_feature_table=counts.get("cc_traces", {}).get("ids_in_feature_table") if "cc_traces" in srcs else None)
    json.dump(man, open(os.path.join(a.out, f"{name}.manifest.json"), "w"), indent=1)
assert chk["overlap_with_seen_sessions"] == 0 and not chk["private_sources_present"]
assert all(v == 0 for k, v in chk.items() if k.startswith("overlap_"))
