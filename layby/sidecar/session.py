"""Session sidecar: turns an OpenAI-compatible request/response stream into Laya v6 server-view states.

A boundary is the moment a response ends and the session's KV goes idle. For each one the tracker
builds the state Layby-Dwell was trained on (the training repo's state builder), in the server view a
generic engine can fill (score_states.server): no src, no harness-parsed fields (prev_ret_by,
bg_pending), no skeleton (TraceLab only). Field by field, as offline:

  kind            tool when the response asks for tool calls, else human
  model, tool, program, arg_chars, meta.cmd_chars/timeout_s/background/n_tools
                  from the request model and the first tool call (features.arg_features)
  ctx_tokens      prompt + completion tokens of this response (usage)
  turn            boundaries before this one in the session
  session_age_s, meta.since_prev_s
                  time since the session's first / previous boundary
  hour, dow       UTC clock at the boundary
  recent          the last 8 boundaries of the session, oldest first: [kind, tool, gap seconds]
  meta.prev_*     previous boundary's gap and kind; previous same-kind gap, its EWMA (alpha 0.3 over
                  log1p gaps, pandas adjust=True) and count
  tool_history, meta.program_history, meta.user_kind_history
                  this user's earlier gaps for the same tool / program / kind across sessions: count,
                  mean (of log1p, shown back in seconds), 90th percentile and share under 1 s over the
                  last 200 (at least 3)
  tool_call, assistant, user
                  "name {json args}" of the tool calls (" | " joined), the last 3 assistant texts of the
                  current turn, the last human message, cut as in training

Sessions are linked by prompt prefix: a request continues the session whose last response is the last
assistant message in the request (matched by a hash of the conversation up to and including it).
A boundary's gap becomes known when the session's next request arrives; only then does it enter the
session's history and the user's histories.
"""
import hashlib
import json
import math
import time
from collections import deque
from datetime import datetime, timezone

import numpy as np

from layby.sidecar.features import arg_features

LOG1 = math.log1p(1.0)


def cut(s, n):
    s = s if isinstance(s, str) else ""
    return s if len(s) <= n else s[: n // 2] + " … " + s[-n // 2:]


def num(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return round(float(x), nd)


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") in (None, "text"))
    return ""


def _canon(m):
    """A message reduced to what every client keeps when it sends the conversation back."""
    calls = [(c.get("function", {}).get("name"), _norm_args(c.get("function", {}).get("arguments")))
             for c in (m.get("tool_calls") or [])]
    return json.dumps([m.get("role"), _text(m.get("content")).strip(), calls], ensure_ascii=False)


def _norm_args(a):
    try:
        return json.dumps(json.loads(a) if isinstance(a, str) else a, sort_keys=True, ensure_ascii=False)
    except (ValueError, TypeError):
        return str(a)


def conv_hash(messages):
    h = hashlib.blake2b(digest_size=16)
    for m in messages:
        h.update(_canon(m).encode())
        h.update(b"\x00")
    return h.hexdigest()


class _Hist:
    """One user's gaps for one tool / program / kind: count, running mean of log1p, last 200."""
    __slots__ = ("n", "s", "last")

    def __init__(self):
        self.n, self.s, self.last = 0, 0.0, deque(maxlen=200)

    def add(self, lg):
        self.n += 1; self.s += lg; self.last.append(lg)

    def view(self, keep_none):
        if self.n == 0:
            return None
        h = {"n": self.n, "mean_s": num(np.expm1(self.s / self.n))}
        if len(self.last) >= 3:
            a = np.fromiter(self.last, float)
            h["p90_s"] = num(np.expm1(np.quantile(a, 0.9)))
            h["share_under_1s"] = num(float((a < LOG1).mean()), 2)
        elif keep_none:
            h["p90_s"], h["share_under_1s"] = None, None
        return h


class Boundary:
    __slots__ = ("t", "kind", "tool", "prog", "user", "gap")

    def __init__(self, t, kind, tool, prog, user):
        self.t, self.kind, self.tool, self.prog, self.user, self.gap = t, kind, tool, prog, user, None


class Session:
    def __init__(self, sid, t0):
        self.sid, self.t0 = sid, t0
        self.bounds: list[Boundary] = []        # every boundary, in order


class Tracker:
    def __init__(self, clock=time.time):
        self.clock = clock
        self.by_tip: dict[str, Session] = {}    # hash of conversation up to the last response -> session
        self.hist: dict[tuple, _Hist] = {}      # (user, field, value) -> gaps
        self.n_sessions = 0

    # --- requests ------------------------------------------------------------------------
    def on_request(self, body, user):
        """Link a request to its session; the previous boundary's gap is now known. Returns the session."""
        now = self.clock()
        msgs = body.get("messages") or []
        last_a = max((i for i, m in enumerate(msgs) if m.get("role") == "assistant"), default=None)
        sess = self.by_tip.pop(conv_hash(msgs[: last_a + 1]), None) if last_a is not None else None
        if sess is None:
            self.n_sessions += 1
            sess = Session(f"s{self.n_sessions}", now)
        elif sess.bounds and sess.bounds[-1].gap is None:
            b = sess.bounds[-1]
            b.gap = max(now - b.t, 0.0)
            lg = math.log1p(b.gap)
            for key in ((b.user, "tool", b.tool), (b.user, "prog", b.prog), (b.user, "kind", b.kind)):
                self.hist.setdefault(key, _Hist()).add(lg)
        return sess

    # --- responses -------------------------------------------------------------------------
    def on_response(self, sess, body, resp, usage, user):
        """The response ended: record the boundary and return its server-view state (JSON) and kind."""
        now = self.clock()
        msgs = body.get("messages") or []
        calls = resp.get("tool_calls") or []
        kind = "tool" if calls else "human"
        tool = prog = None
        feat = {}
        if calls:
            f0 = calls[0].get("function", {})
            tool = f0.get("name")
            feat = arg_features(tool, f0.get("arguments") or "{}")
            prog = feat.get("prog")
        prev = sess.bounds[-1] if sess.bounds else None
        known = [b for b in sess.bounds if b.gap is not None]
        same = [b for b in known if b.kind == kind]
        ts = datetime.fromtimestamp(now, timezone.utc)
        rec = [[b.kind, b.tool, num(b.gap)] for b in known[-8:]]
        th = self.hist.get((user, "tool", tool))
        meta = {"cmd_chars": num(feat.get("cmd_len"), 0), "timeout_s": num(feat.get("timeout"), 0),
                "n_tools": num(len(calls), 0),
                "background": bool(feat["background"]) if feat.get("background") is not None else None,
                "since_prev_s": num(now - prev.t) if prev else None,
                "prev_gap_s": num(prev.gap) if prev and prev.gap is not None else None,
                "prev_kind": prev.kind if prev else None,
                "prev_same_kind_s": num(same[-1].gap) if same else None,
                "ewma_same_kind_s": num(np.expm1(_ewma([math.log1p(b.gap) for b in same]))) if same else None,
                "n_same_kind": num(sum(1 for b in sess.bounds if b.kind == kind), 0),
                "program_history": _hv(self.hist.get((user, "prog", prog)), False),
                "user_kind_history": _hv(self.hist.get((user, "kind", kind)), False)}
        meta = {k: v for k, v in meta.items() if v is not None} or None
        asst_texts = _turn_assistant_texts(msgs) + [x for x in (resp.get("reasoning_content"), _text(resp.get("content"))) if x]
        s = {"kind": kind, "model": body.get("model"), "tool": tool, "program": prog,
             "arg_chars": num(feat.get("arg_len"), 0), "ctx_tokens": num(_ctx(usage), 0),
             "turn": len(sess.bounds), "session_age_s": num(now - (sess.bounds[0].t if sess.bounds else now), 0),
             "hour": ts.hour, "dow": ts.weekday(), "recent": rec, "tool_history": _hv(th, True), "meta": meta,
             "tool_call": cut(" | ".join(f"{c.get('function', {}).get('name')} {_dump_args(c.get('function', {}).get('arguments'))}" for c in calls), 700),
             "assistant": cut(" ".join(asst_texts[-3:]), 700),
             "user": cut(_last_user(msgs), 400)}
        state = json.dumps({k: v for k, v in s.items() if not (v is None or (isinstance(v, (str, list)) and len(v) == 0))},
                           ensure_ascii=False, default=str)
        sess.bounds.append(Boundary(now, kind, tool, prog, user))
        tip = msgs + [{"role": "assistant", "content": resp.get("content"), "tool_calls": calls}]
        self.by_tip[conv_hash(tip)] = sess
        return state, kind


def _hv(h, keep_none):
    return h.view(keep_none) if h is not None else None


def _ewma(xs, alpha=0.3):
    w = (1 - alpha) ** np.arange(len(xs))[::-1]
    return float((w * np.asarray(xs)).sum() / w.sum())


def _ctx(usage):
    u = usage or {}
    v = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
    return v or None


def _dump_args(a):
    try:
        return json.dumps(json.loads(a) if isinstance(a, str) else a, ensure_ascii=False)
    except (ValueError, TypeError):
        return str(a)


def _last_user(msgs):
    for m in reversed(msgs):
        if m.get("role") == "user":
            t = _text(m.get("content"))
            if t:
                return t
    return ""


def _turn_assistant_texts(msgs):
    """Assistant texts since the last human message (the current turn, before this response), each
    message's reasoning before its text, as the transcript blocks were."""
    out = []
    for m in reversed(msgs):
        if m.get("role") == "user" and _text(m.get("content")):
            break
        if m.get("role") == "assistant":
            out = [x for x in (m.get("reasoning_content"), _text(m.get("content"))) if x] + out
    return out
