#!/usr/bin/env python3
"""Replay pool of one held-out population (content-free: lengths, gaps, kinds) for the simulator.

Sessions: test-split sessions of source SRC whose user is in the uhold set (users no model saw in
train or val), with at least --min-turns turns. Construction: output
384 tokens after a human wait and 128 after a tool wait, the context curve scaled under --ctx-cap over
the first --max-turns turns, history under 38,000 tokens, gaps capped at --gap-cap, starts uniform over
--start-window. No decision fields: rules decide from the attached curves at simulation time.
Real-length mode (--no-scale): the context keeps its true per-turn growth and the session ends at the last
turn whose context fits --ctx-cap (the model's window); --max-turns 0 and --max-hist 0 lift those limits.
--refs-out writes (sid, turn, feat_ref): which boundary each replay turn is, for scoring it with the
model and attaching the curve (attach_curves.py --key feat_ref).

Usage: build_replay_pop.py SPLIT.parquet SRC OUT.json --refs-out REFS.parquet [--min-turns 6] [--gap-cap inf]
       [--any-user] [--sessions N]
--any-user takes test-split sessions of every user, for sources without held-out users (Copilot): the
split is chronological, so these sessions come after every training session but users may overlap.
"""
import argparse, hashlib, json
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("feat"); ap.add_argument("src"); ap.add_argument("out")
ap.add_argument("--refs-out", required=True)
ap.add_argument("--min-turns", type=int, default=6)
ap.add_argument("--any-user", action="store_true", help="test-split sessions of every user (sources with no held-out users)")
ap.add_argument("--sessions", type=int, default=0, help="sample at most this many sessions (0: all)")
ap.add_argument("--ctx-cap", type=int, default=32768)
ap.add_argument("--gap-cap", type=float, default=300.0)
ap.add_argument("--max-turns", type=int, default=40, help="0: every turn")
ap.add_argument("--max-hist", type=int, default=38000, help="history token limit; 0: none (--ctx-cap still applies)")
ap.add_argument("--no-scale", action="store_true", help="real contexts: no scaling under --ctx-cap; stop at the cap")
ap.add_argument("--start-window", type=float, default=600.0)
a = ap.parse_args()

df = pd.read_parquet(a.feat, columns=["src", "split", "uhold", "session", "kind", "ref", "returned", "gap", "ctx", "t",
                                     "tool", "prog", "user"])
for c in df.columns:
    if str(df[c].dtype).startswith("string"):
        df[c] = df[c].astype(object)
te = df[(df.split == "test") & (df.uhold | a.any_user) & (df.src == a.src)].copy()
te["returned"] = te.returned.astype(bool)
n = te.groupby("session").size()
keep = n[n >= a.min_turns].index
if a.sessions and len(keep) > a.sessions:
    keep = pd.Series(keep).sample(a.sessions, random_state=0)
te = te[te.session.isin(keep)].sort_values(["session", "t"]).reset_index(drop=True)

rng = np.random.default_rng(0)
sessions, refs = [], []
for k, (sess, s) in enumerate(te.groupby("session", sort=True)):
    c = np.maximum.accumulate(s.ctx.ffill().fillna(8192).values.astype(float))
    mt = a.max_turns if a.max_turns > 0 else len(s)
    if not a.no_scale:
        c = c * min(1.0, a.ctx_cap / c[:mt].max())
    turns, prev, hist, rr = [], 0, 0, []
    for j, r in enumerate(s.head(mt).itertuples()):
        out_len = 384 if r.kind == "human" else 128
        cx = int(c[j])
        new = max(64, cx - prev - turns[-1]["out_len"]) if turns else max(256, cx - out_len)
        if a.max_hist and hist + new + out_len > a.max_hist:
            break
        if a.no_scale and hist + new + out_len > a.ctx_cap:
            break
        hist += new + out_len
        gap = float(min(r.gap, a.gap_cap)) if r.returned else None
        turns.append(dict(new_tokens=int(new), out_len=out_len, kind=r.kind, ctx=cx, gap_after=gap,
                          tool=r.tool if isinstance(r.tool, str) else None, prog=r.prog if isinstance(r.prog, str) else None))
        rr.append((f"s{k:03d}", j, r.ref))
        prev = cx
        if not r.returned:
            break
    if len(turns) >= 3:
        turns[-1]["gap_after"] = None   # the replay ends the session here
        refs += rr
        u = s.user.iloc[0]
        sessions.append(dict(id=f"s{k:03d}", pool=a.src, seed=1000 + k, start=float(rng.uniform(0, a.start_window)), turns=turns,
                             user=hashlib.blake2b(str(u).encode(), digest_size=6).hexdigest() if isinstance(u, str) else None))
pd.DataFrame(refs, columns=["sid", "turn", "feat_ref"]).to_parquet(a.refs_out, index=False)
json.dump(dict(gap_cap=a.gap_cap, ctx_cap=a.ctx_cap, src=a.src, sessions=sessions), open(a.out, "w"))
T = [t for s in sessions for t in s["turns"]]
print(f"{a.src}: sessions {len(sessions)} turns {len(T)} human {sum(t['kind'] == 'human' for t in T)}")
