#!/usr/bin/env python3
"""Make a replay pool safe to publish: only lengths, gaps, kinds, tool names and curves leave.

A turn keeps new_tokens, out_len, kind, ctx, gap_after, tool, prog and surv_* (model survival curves).
Anything else is dropped. Checks on the way:
  - every value is a number, None, a list of numbers, or a string from a closed vocabulary;
  - kind is one of human, tool, workflow;
  - tool: built-in agent tool names pass; MCP tools (mcp__server__tool) become "mcp", since the server
    name says what a user had installed; anything else that is not a plain identifier becomes "other";
  - prog: the common executables in PROGS pass, anything else becomes "other" (the extractor's
    first-token parse sometimes catches words from heredocs or prose);
  - user: replaced by POOL:uNNN in order of first appearance (no hash of the source id leaves).
No rule in sim/ reads prog, and tool only feeds ContinuumTTL's per-tool CDF, which needs more than 100
samples per tool; the merged names never reach that, so these maps change no result.

Usage: sanitize.py IN.json POOL OUT.json
"""
import json, re, sys

KINDS = {"human", "tool", "workflow"}
PROGS = {"git", "ls", "echo", "grep", "find", "cat", "gh", "python3", "python", "python-script", "bun", "sed",
         "go", "ssh", "wc", "mkdir", "for", "pytest", "sleep", "docker", "curl", "make", "rm", "which",
         "chmod", "head", "tail", "cargo", "node", "npm", "npx", "pnpm", "cp", "mv", "perl", "pwd", "rsync",
         "du", "diff", "date", "bash", "ruff", "seq", "test", "pdflatex", "pdfinfo", "open"}
TURN_NUM = ("new_tokens", "out_len", "ctx", "gap_after")
IDENT = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,40}")


def tool_name(t):
    if t is None:
        return None
    if t.startswith("mcp__"):
        return "mcp"
    return t if IDENT.fullmatch(t) else "other"


def num(x, k):
    assert x is None or isinstance(x, (int, float)), (k, x)
    return x


def main():
    src, pool, out = sys.argv[1:4]
    W = json.load(open(src))
    users, S = {}, []
    for s in W["sessions"]:
        turns = []
        for t in s["turns"]:
            assert t["kind"] in KINDS, t["kind"]
            u = {k: num(t.get(k), k) for k in TURN_NUM}
            u["kind"] = t["kind"]
            u["tool"] = tool_name(t.get("tool"))
            p = t.get("prog")
            u["prog"] = None if p is None else (p if p in PROGS else "other")
            for k, v in t.items():
                if k.startswith("surv_"):
                    assert v is None or all(isinstance(x, (int, float)) for x in v), k
                    u[k] = v
            turns.append(u)
        r = dict(id=s["id"], pool=pool, seed=int(s["seed"]), start=float(s["start"]), turns=turns)
        if s.get("user") is not None:
            r["user"] = users.setdefault(s["user"], f"{pool}:u{len(users):03d}")
        S.append(r)
    cap = W["gap_cap"]
    json.dump(dict(pool=pool, gap_cap=None if cap == float("inf") else cap, ctx_cap=W["ctx_cap"], sessions=S), open(out, "w"))
    T = [t for s in S for t in s["turns"]]
    print(f"{pool}: sessions {len(S)} turns {len(T)} users {len(users)} "
          f"curves {sum(any(k.startswith('surv_') and t[k] is not None for k in t) for t in T)}")


if __name__ == "__main__":
    main()
