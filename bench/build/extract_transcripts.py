#!/usr/bin/env python3
"""Idle boundaries from local agent transcripts (Claude Code and Codex), content-free.

Boundary kinds (known when the cache goes idle):
  tool     : model emitted a tool call -> tool result arrived (keyed by call id)
  human    : turn finished (final assistant text / task_complete) -> next message that wakes the session
  workflow : same, in a subagent session
Human boundaries with no later message are right-censored at the file's last event.

ret_by says what actually came back (a label annotation for evaluation slices, never a feature of the
same row): human, command (slash or shell command typed by the user), notification (background task or
system event), continuation (context compaction restart), cron (the same prompt text 3+ times in the
file: a loop or schedule), peer (another agent session), delegation (Codex heartbeat or delegation).
bg_pending is a decision-time feature: background tasks launched and not yet reported back.
Each record carries a pointer (file, line) to both ends so content features can be computed later
without copying text here.

Usage: extract_transcripts.py claude|codex ROOT [--src NAME] [--users sessions.parquet] > boundaries.jsonl
  --src    source name (default claude_code / codex)
  --users  parquet with session_id, user_id (SWE-chat): sets the user per file stem
"""
import collections, hashlib, io, json, os, re, subprocess, sys
from datetime import datetime
from pathlib import Path

SKIP = {"cd", "pushd", "popd", "export", "source", ".", "set", "unset", "alias", "true", "ulimit", "umask",
        "trap", "local", "declare", "eval", "exec", "time", "builtin", "command", "then", "do", "else", "fi"}
PREFIX = {"sudo", "nice", "ionice", "timeout", "env", "nohup", "stdbuf", "taskset", "systemd-run"}


def dic(x):
    return x if isinstance(x, dict) else {}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def program(cmd):
    """First real program of a shell command: skips cd/export/source segments and wrapper prefixes."""
    for seg in re.split(r"&&|\|\||;|\||\n|\(|\)|`", cmd):
        toks = [t for t in seg.strip().split()]
        i = 0
        for i, t in enumerate(toks):
            wrapper = t in PREFIX or t.startswith("-") or t.lstrip("-").isdigit() or re.fullmatch(r"\d+[smh]", t)
            if not wrapper and not ("=" in t and not t.startswith(("'", '"'))):
                break
        else:
            continue
        t = toks[i].strip("'\"")
        if t and t not in SKIP and not t.startswith(("#", "{", "}", "$")):
            return os.path.basename(t)
    return None


def open_lines(f):
    if str(f).endswith(".zst"):
        return io.StringIO(subprocess.run(["zstd", "-dc", str(f)], capture_output=True, text=True).stdout)
    return f.open()


def stem(f):
    return f.name.split(".jsonl")[0]


def arg_features(name, args):
    a = args if isinstance(args, str) else json.dumps(args or {})
    feat = {"arg_len": len(a)}
    try:
        d = json.loads(a) if isinstance(a, str) else a
    except ValueError:
        d = {}
    cmd = None
    if isinstance(d, dict):
        cmd = d.get("command") or d.get("cmd")
        if isinstance(cmd, list):
            cmd = " ".join(map(str, cmd))
        feat["timeout"] = d.get("timeout") or d.get("timeout_ms")
        feat["background"] = bool(d.get("run_in_background"))
    if isinstance(cmd, str):
        feat["prog"] = program(cmd)
        feat["cmd_len"] = len(cmd)
    return feat


def claude_ret_by(r, c, rep):
    """What woke the session, from a user line that is not a tool result. None: not a wake-up."""
    ok = dic(r.get("origin")).get("kind")
    text = c if isinstance(c, str) else " ".join(b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else ""
    head = text.lstrip()[:60]
    if ok == "task-notification" or head.startswith("<task-notification"):
        return "notification"
    if ok in ("peer", "coordinator"):
        return "peer"
    if ok == "auto-continuation" or head.startswith("This session is being continued"):
        return "continuation"
    if r.get("isMeta"):
        return None
    if ok != "human" and not isinstance(c, str):
        return None
    if head.startswith(("<command-name", "<local-command", "<bash-input", "/")):
        return "command"
    if head.startswith(("<system-reminder", "[SYSTEM NOTIFICATION", "<ci-monitor-event", "<system")):
        return "notification"
    if len(text) >= 40 and rep[hashlib.blake2b(text.encode(), digest_size=8).digest()] >= 3:
        return "cron"
    return "human"


def claude(root, src="claude_code", users=None):
    users = users or {}
    for f in Path(root).rglob("*.jsonl"):
        sub = "subagents" in f.parts
        pend_tools, pend_human, last_t, model, sess, ctx = {}, None, None, None, f.stem, None
        bg, bg_ids = 0, set()
        try:
            lines = f.read_text(errors="replace").splitlines()
        except OSError:
            continue
        rows, rep = [], collections.Counter()
        for line in lines:
            try:
                r = json.loads(line)
            except ValueError:
                r = None
            if not isinstance(r, dict):
                r = None
            rows.append(r)
            if r and r.get("type") == "user":
                c = dic(r.get("message")).get("content")
                if isinstance(c, str) and len(c) >= 40:
                    rep[hashlib.blake2b(c.encode(), digest_size=8).digest()] += 1
        user = users.get(f.stem) or users.get(sess)
        base = dict(src=src, session=sess, file=str(f), **({"user": user} if user else {}))
        for i, r in enumerate(rows):
            if not r or "timestamp" not in r or r.get("type") not in ("assistant", "user"):
                continue
            t = ts(r["timestamp"]); last_t = t
            msg = dic(r.get("message"))
            c = msg.get("content")
            blocks = [b for b in c if isinstance(b, dict)] if isinstance(c, list) else []
            if r["type"] == "assistant":
                model = msg.get("model") or model
                u = dic(msg.get("usage"))
                if u:
                    ctx = sum(u.get(k) or 0 for k in ("input_tokens", "cache_read_input_tokens",
                                                       "cache_creation_input_tokens", "output_tokens"))
                uses = [b for b in blocks if b.get("type") == "tool_use"]
                for b in uses:
                    if isinstance(b.get("input"), dict) and b["input"].get("run_in_background"):
                        bg_ids.add(b.get("id"))
                    pend_tools[b.get("id")] = (t, i, b.get("name"), arg_features(b.get("name"), b.get("input")), ctx, bg)
                if not uses and any(b.get("type") == "text" for b in blocks):
                    pend_human = (t, i, ctx, bg)
            else:
                results = [b for b in blocks if b.get("type") == "tool_result"]
                for b in results:
                    p = pend_tools.pop(b.get("tool_use_id"), None)
                    if b.get("tool_use_id") in bg_ids:
                        bg += 1
                    if p:
                        yield dict(base, kind="subagent_tool" if sub else "tool", returned=True, ret_by="tool",
                                   gap_s=t - p[0], idle_start=p[0], model=model, tool=p[2], **p[3], ctx_tokens=p[4],
                                   bg_pending=p[5], line_a=p[1], line_b=i, is_error=bool(b.get("is_error")))
                if results:
                    continue
                rb = claude_ret_by(r, c, rep)
                if rb is None:
                    continue
                if rb == "notification" and bg:
                    bg -= 1
                if pend_human:
                    yield dict(base, kind="workflow" if sub else "human", returned=True, ret_by=rb,
                               gap_s=t - pend_human[0], idle_start=pend_human[0], model=model, ctx_tokens=pend_human[2],
                               bg_pending=pend_human[3], line_a=pend_human[1], line_b=i)
                pend_human = None
        if pend_human and last_t is not None:
            yield dict(base, kind="workflow" if sub else "human", returned=False, ret_by=None, gap_s=last_t - pend_human[0],
                       idle_start=pend_human[0], model=model, ctx_tokens=pend_human[2], bg_pending=pend_human[3],
                       line_a=pend_human[1])


def codex_ret_by(m):
    head = (m or "").lstrip()[:40]
    if head.startswith(("<heartbeat", "<codex_delegation", "<realtime_delegation")):
        return "delegation"
    if head.startswith("<subagent_notification") or head.startswith("<task-notification"):
        return "notification"
    return "human"


def codex(root, src="codex", users=None):
    users = users or {}
    for f in list(Path(root).rglob("*.jsonl")) + list(Path(root).rglob("*.jsonl.zst")):
        pend_tools, pend_human, last_t, model, sess, ctx, sub = {}, None, None, None, stem(f), None, False
        try:
            lines = open_lines(f)
        except OSError:
            continue
        user = users.get(sess)
        base = dict(src=src, session=sess, file=str(f), **({"user": user} if user else {}))
        for i, line in enumerate(lines):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if not isinstance(r, dict):
                continue
            p = r.get("payload") if isinstance(r.get("payload"), dict) else {}
            typ, st = r.get("type"), p.get("type")
            if typ == "session_meta":
                sub = sub or "subagent" in json.dumps(p.get("source")) or p.get("thread_source") == "subagent"
            if "timestamp" not in r:
                continue
            t = ts(r["timestamp"]); last_t = t
            if typ == "event_msg" and st == "token_count":
                lu = dic(dic(p.get("info")).get("last_token_usage"))
                ctx = lu.get("total_tokens") or ctx
            elif typ == "turn_context":
                model = p.get("model") or model
            elif typ == "response_item" and st in ("function_call", "custom_tool_call"):
                name = p.get("name")
                pend_tools[p.get("call_id")] = (t, i, name, arg_features(name, p.get("arguments") or p.get("input")), ctx)
            elif typ == "response_item" and st in ("function_call_output", "custom_tool_call_output"):
                q = pend_tools.pop(p.get("call_id"), None)
                if q:
                    yield dict(base, kind="subagent_tool" if sub else "tool", returned=True, ret_by="tool", gap_s=t - q[0],
                               idle_start=q[0], model=model, tool=q[2], **q[3], ctx_tokens=q[4], line_a=q[1], line_b=i)
            elif typ == "event_msg" and st == "task_complete":
                pend_human = (t, i, ctx)
            elif typ == "event_msg" and st == "user_message":
                if pend_human:
                    yield dict(base, kind="workflow" if sub else "human", returned=True, ret_by=codex_ret_by(p.get("message")),
                               gap_s=t - pend_human[0], idle_start=pend_human[0], model=model, ctx_tokens=pend_human[2],
                               line_a=pend_human[1], line_b=i)
                pend_human = None
        if pend_human and last_t is not None:
            yield dict(base, kind="workflow" if sub else "human", returned=False, ret_by=None, gap_s=last_t - pend_human[0],
                       idle_start=pend_human[0], model=model, ctx_tokens=pend_human[2], line_a=pend_human[1])


if __name__ == "__main__":
    kw = {}
    if "--src" in sys.argv:
        kw["src"] = sys.argv[sys.argv.index("--src") + 1]
    if "--users" in sys.argv:
        import pandas as pd
        u = pd.read_parquet(sys.argv[sys.argv.index("--users") + 1], columns=["session_id", "user_id", "owner_id"])
        u = u.assign(user_id=u.user_id.replace("", None).fillna("owner:" + u.owner_id.astype(str))).dropna()
        kw["users"] = dict(zip(u.session_id.astype(str), u.user_id.astype(str)))
    gen = {"claude": claude, "codex": codex}[sys.argv[1]]
    for b in gen(sys.argv[2], **kw):
        if b["gap_s"] >= 0:
            print(json.dumps(b))
