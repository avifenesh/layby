#!/usr/bin/env python3
"""WildChat (allenai, ODC-BY) -> human-wait boundaries with content. APPROXIMATE LABELS.

WildChat stamps assistant turns at the moment the backend received the full response. WildChat-4.8M
also keeps `created` on each assistant turn: the OpenAI request creation time, i.e. when the user's
message was sent (1 s resolution). When present:
    T = created(i+1) - ts(i)                                    (exact to the second)
Otherwise (WildChat-1M) for assistant turn i followed by turn i+1 in the same conversation:
    T ~= ts(i+1) - ts(i) - gen(i+1)
where gen(i+1) = 1.0 s + tokens(i+1) / rate(model) is an assumed generation time (tokens = chars/4;
rate 60 tok/s for gpt-3.5, 25 tok/s for gpt-4 family). The label is noisy by tens of seconds, so
use it for coarse horizons (>= 60 s), not for sub-minute prefetch timing. label_exact marks rows.
Context is usage.prompt_tokens + completion_tokens when the record has usage, else chars / 4.
The last assistant turn of a conversation is right-censored at the dataset's last timestamp.

Sampling: conversations are kept with probability --frac (hash of conversation id), so the same
sample comes out on every run. Users are hashed_ip.

Usage: extract_wildchat.py DIR OUT_BND.jsonl OUT_CONTENT.parquet --frac 0.08
"""
import argparse, glob, hashlib, json
import pandas as pd
import pyarrow.parquet as pq
from datetime import timezone


def utc(d):
    return d.replace(tzinfo=timezone.utc).timestamp() if d.tzinfo is None else d.timestamp()

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("out_bnd"); ap.add_argument("out_content")
ap.add_argument("--frac", type=float, default=0.08)
a = ap.parse_args()


def keep(cid):
    return int(hashlib.blake2b(cid.encode(), digest_size=4).hexdigest(), 16) / 2**32 < a.frac


def rate(model):
    return 25.0 if "gpt-4" in (model or "") and "turbo" not in (model or "") else 60.0


def clip(s, n=2000):
    s = s or ""
    return s if len(s) <= n else s[: n // 2] + " ... " + s[-n // 2:]


files = sorted(glob.glob(f"{a.dir}/*.parquet"))
end = 0.0
convs = []
for f in files:
    t = pq.read_table(f, columns=["conversation_hash", "model", "conversation", "turn"])
    for cid, model, conv, turn in zip(*[t.column(c).to_pylist() for c in ("conversation_hash", "model", "conversation", "turn")]):
        asst = [m for m in conv if m.get("role") == "assistant" and m.get("timestamp")]
        if asst:
            end = max(end, max(utc(m["timestamp"]) for m in asst))
        if keep(cid):
            convs.append((cid, model, conv))

content = []
with open(a.out_bnd, "w") as out:
    for cid, model, conv in convs:
        user_ip, last_user = None, ""
        turns = []  # (asst ts, asst text, user text before it, user ip)
        for m in conv:
            if m.get("role") == "user":
                last_user = m.get("content") or ""
                user_ip = m.get("hashed_ip") or user_ip
            elif m.get("role") == "assistant" and m.get("timestamp"):
                u = m.get("usage") or {}
                tok = (u.get("prompt_tokens") or 0) + (u.get("completion_tokens") or 0)
                turns.append((utc(m["timestamp"]), m.get("content") or "", last_user, m.get("created"), tok))
        for i, (ts, text, utext, _, tok) in enumerate(turns):
            ref = f"wildchat:{cid}:{i}"
            ctx = tok or sum(len(x[1]) + len(x[2]) for x in turns[: i + 1]) // 4
            rec = dict(src="wildchat", kind="human", idle_start=ts, model=model, session=cid, user=user_ip,
                       ctx_tokens=ctx, file="wildchat", line_a=f"{cid}:{i}", line_b=None, ref=ref)
            if i + 1 < len(turns):
                nts, ntext, _, ncreated, _ = turns[i + 1]
                if ncreated:
                    rec.update(returned=True, gap_s=max(0.0, ncreated - ts), raw_gap_s=nts - ts, label_exact=True)
                else:
                    gen = 1.0 + (len(ntext) / 4) / rate(model)
                    rec.update(returned=True, gap_s=max(0.0, nts - ts - gen), raw_gap_s=nts - ts, label_exact=False)
            else:
                rec.update(returned=False, gap_s=max(0.0, end - ts))
            out.write(json.dumps(rec) + "\n")
            content.append((ref, "", clip(text), clip(utext)))
pd.DataFrame(content, columns=["ref", "args_text", "asst_text", "user_text"]).to_parquet(a.out_content, index=False)
print(len(convs), "conversations,", len(content), "boundaries")
