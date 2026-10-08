#!/usr/bin/env python3
"""Boundary files of one source -> the table build_pool.py reads (split and user hold-out).

Same rules as the Layby training pipeline, so the bench sessions come out identical:
  dedupe   a boundary seen twice (same file and line) keeps the longest gap;
  split    per source, sessions ordered by first idle time: first 70% train, next 10% val, last 20% test;
  uhold    10% of users by hash (blake2b, 4 bytes, mod 10 == 0) on sources with real user ids
           (swechat, tracelab_*, wildchat). The model never trained on these users.
The bench pools are test-split sessions of uhold users (Copilot has no user ids: test-split sessions
of any user, all later than every training session).

Usage: split.py OUT.parquet BOUNDARIES.jsonl [...]
"""
import hashlib, json, sys
import pandas as pd

NO_USER = "unknown"   # every boundary of the four public sources has a user; this never fires on them


def rows(paths):
    for p in paths:
        for l in open(p):
            b = json.loads(l)
            yield dict(src=b["src"], kind=b["kind"], returned=b["returned"], gap=b["gap_s"], t=b["idle_start"],
                       user=b.get("user") or NO_USER, session=b["session"], tool=b.get("tool"), prog=b.get("prog"),
                       ctx=b.get("ctx_tokens"), ref=b.get("ref") or f"{b['file']}:{b['line_a']}:{b.get('line_b')}",
                       key=(b["file"], b["line_a"]))


def main():
    out, paths = sys.argv[1], sys.argv[2:]
    best = {}
    for r in rows(paths):
        o = best.get(r["key"])
        if o is None or r["gap"] > o["gap"]:
            best[r["key"]] = r
    df = pd.DataFrame(best.values()).drop(columns=["key"])
    df["kind"] = df["kind"].replace({"subagent_tool": "tool"})
    df = df.sort_values(["src", "session", "t"]).reset_index(drop=True)
    first = df.groupby(["src", "session"])["t"].transform("min")
    df["split"] = "train"
    for src, idx in df.groupby("src").groups.items():
        f = first.loc[idx]
        q70, q80 = f.quantile(0.7), f.quantile(0.8)
        df.loc[idx[f.values > q70], "split"] = "val"
        df.loc[idx[f.values > q80], "split"] = "test"
    real_user = df.src.str.startswith(("swechat", "tracelab_", "wildchat"))
    hu = df.user.astype(str).map(lambda u: int(hashlib.blake2b(u.encode(), digest_size=4).hexdigest(), 16) % 10 == 0)
    df["uhold"] = real_user & hu
    df["ctx"] = pd.to_numeric(df["ctx"], errors="coerce")
    for c in ("kind", "tool", "prog", "src", "user", "session", "ref"):
        df[c] = df[c].astype("string")
    df = df.sort_values(["t"]).reset_index(drop=True)   # the row order the pipeline wrote (ties keep their order)
    df.to_parquet(out, index=False)
    print(df.groupby(["src", "split", "kind"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
