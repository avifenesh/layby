#!/usr/bin/env python3
"""GitHub Copilot coding-agent traces 2026 (Azure/AzurePublicDataset, CC-BY) -> wait boundaries.

Content-free. Each session has turns; each turn has llm_calls and tool_batches. Both carry an END
timestamp (ms precision) and duration_ms, so start = ts - duration. Every successful LLM call is a
boundary: the KV of its prompt + completion goes idle at its end, and the session returns when the
next LLM call of the session starts.
  kind  human     the next call is initiator 'user' (a new user turn)
        tool      a tool batch ran between the two calls
        workflow  the agent continued with no tool batch in between (auto continuation)
  gap = next.start - call.end. The last call of a session is right-censored at the end of the
  trace window. Overlapping calls (next starts before this one ends) are dropped.
No user id exists, so user = session (history features stay per session).

Sampling: sessions kept with probability --frac (hash of session id), stable across runs.
Sessions split across date shards are merged by session id.

Usage: extract_copilot.py DIR OUT_BND.jsonl [--frac 0.25]
"""
import argparse, collections, glob, gzip, hashlib, io, json, tarfile
from datetime import datetime, timezone

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("out")
ap.add_argument("--frac", type=float, default=0.25)
a = ap.parse_args()


def keep(sid):
    return int(hashlib.blake2b(sid.encode(), digest_size=4).hexdigest(), 16) / 2**32 < a.frac


def ts(s):
    # '2026-06-01T17:44:20.419000000Z' -> epoch seconds (ns truncated to us)
    d, frac = s.rstrip("Z").split(".") if "." in s else (s.rstrip("Z"), "0")
    return datetime.fromisoformat(d).replace(tzinfo=timezone.utc).timestamp() + float("0." + frac[:6])


calls = collections.defaultdict(list)   # sid -> [(start, end, initiator, ok, model, ctx)]
batches = collections.defaultdict(list)  # sid -> [(start, end, n_fn, first_fn)]
end_window, nsess = 0.0, set()
for path in sorted(glob.glob(f"{a.dir}/*.tar.gz")):
    with tarfile.open(path, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.endswith(".jsonl.gz"):
                continue
            for line in io.BytesIO(gzip.decompress(tf.extractfile(m).read())):
                s = json.loads(line)
                sid = s["session_id"]
                for t in s["turns"]:
                    for k in t["llm_calls"]:
                        if k.get("timestamp"):
                            end_window = max(end_window, ts(k["timestamp"]))
                if not keep(sid):
                    continue
                nsess.add(sid)
                for t in s["turns"]:
                    for k in t["llm_calls"]:
                        if not k.get("timestamp") or k.get("duration_ms") is None:
                            continue
                        e = ts(k["timestamp"]); st = e - k["duration_ms"] / 1000
                        tok = k.get("tokens") or {}
                        ctx = (tok.get("prompt") or 0) + (tok.get("completion") or 0)
                        calls[sid].append((st, e, k.get("initiator_type") or "", k.get("result") == "Success",
                                           k.get("model"), ctx))
                    for b in t["tool_batches"]:
                        if not b.get("timestamp"):
                            continue
                        e = ts(b["timestamp"]); st = e - (b.get("duration_ms") or 0) / 1000
                        fns = b.get("function_calls") or []
                        batches[sid].append((st, e, len(fns), fns[0].get("name") if fns else None))

cnt = collections.Counter()
with open(a.out, "w") as out:
    for sid, cs in calls.items():
        cs.sort()
        bs = sorted(batches.get(sid, []))
        for i, (st, e, init, ok, model, ctx) in enumerate(cs):
            if not ok or not ctx:
                cnt["skip_fail"] += 1
                continue
            rec = dict(src="copilot", idle_start=e, model=model, session=sid, user="copilot:" + sid,
                       ctx_tokens=ctx, file="copilot", line_a=f"{sid}:{i}", line_b=None, ref=f"copilot:{sid}:{i}")
            if i + 1 < len(cs):
                nst, _, ninit = cs[i + 1][:3]
                if nst < e - 0.05:
                    cnt["skip_overlap"] += 1
                    continue
                tb = [b for b in bs if b[1] >= e - 0.05 and b[0] <= nst + 0.05]
                kind = "human" if ninit == "user" else ("tool" if tb else "workflow")
                rec.update(kind=kind, returned=True, gap_s=max(0.0, nst - e))
                if tb:
                    rec.update(tool=tb[0][3], tools=[b[3] for b in tb], n_fn=sum(b[2] for b in tb),
                               tool_run_s=round(sum(b[1] - b[0] for b in tb), 3))
            else:
                rec.update(kind="human", returned=False, gap_s=max(0.0, end_window - e))
            cnt[(rec["kind"], rec["returned"])] += 1
            out.write(json.dumps(rec) + "\n")
print(len(nsess), "sessions", dict(cnt))
