#!/usr/bin/env python3
"""Every public-source session the Layby-Dwell model or an earlier engine round has seen, by source id.

Inputs (all local, none shipped):
  --feat F [F ...]        boundary feature tables with columns src, session, ref (the first is the current one;
                          older ones resolve refs of older dataset versions)
  --cls DIR [DIR ...]     model dataset versions, each with train/val/test.parquet (column ref). Every split counts
                          as seen: train and val shaped the weights, test chose checkpoints and calibration.
  --pool TAG:POOL.json:REFS.parquet
                          a replay pool (build_replay_pop.py output) and its refs (sid, turn, feat_ref). TAG names
                          the pool family (for example scaled or real) that workloads draw from.
  --workload FILE:TAG     an engine workload drawn from the TAG pools. Its session ids are the pool file's first
                          three letters plus the pool session id (build_real_workload.py, eval_pop.workload).
                          Each mapped session is checked turn by turn (kind, and new_tokens unless the workload was
                          rescaled) against the pool session.
Output: OUT.parquet (src, session, used_by), one row per (session, use); prints counts per source and use.

Usage: used_sessions.py OUT.parquet --feat feat5.parquet [feat_old.parquet] --cls cls cls2 ... \
       --pool scaled:swechat.json:swechat.refs.parquet ... --workload mix36_w0.json:scaled ...
"""
import argparse
import json
import os

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

PUBLIC = ("swechat", "swechat_codex", "tracelab_claude", "tracelab_codex", "wildchat", "copilot")

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--feat", nargs="+", required=True)
ap.add_argument("--cls", nargs="*", default=[])
ap.add_argument("--pool", nargs="*", default=[])
ap.add_argument("--workload", nargs="*", default=[])
a = ap.parse_args()

# ref -> (src, session) over the public sources, current table first (Arrow joins: millions of refs)
maps = []
for f in a.feat:
    t = pq.read_table(f, columns=["src", "session", "ref"], filters=[("src", "in", list(PUBLIC))])
    t = pa.table({c: pc.cast(t[c], pa.string()) for c in ("src", "session", "ref")})
    maps.append(t)
ref_map = maps[0]
for t in maps[1:]:
    known = pc.is_in(t["ref"], value_set=ref_map["ref"])
    ref_map = pa.concat_tables([ref_map, t.filter(pc.invert(known))])
ref_map = ref_map.group_by("ref").aggregate([("src", "min"), ("session", "min")]).rename_columns(["ref", "src", "session"])
print(f"public refs known: {ref_map.num_rows}", flush=True)

rows, report = [], []


def resolve(refs, use):
    r = pa.table({"ref": pc.unique(pa.array(pd.Series(refs, dtype=str).values, pa.string()))})
    j = r.join(ref_map, "ref", join_type="left outer")
    hit = j.filter(pc.is_valid(j["src"]))
    miss = j.filter(pc.invert(pc.is_valid(j["src"])))["ref"]
    s = hit.select(["src", "session"]).to_pandas().drop_duplicates()
    s["used_by"] = use
    rows.append(s)
    # a ref that names a public source but has no feature row is a hole in the check: count it
    pub_miss = int(pc.sum(pc.match_substring_regex(miss, r"^(wildchat|copilot|tracelab|public/)")).as_py() or 0)
    report.append(dict(use=use, refs=r.num_rows, public_resolved=hit.num_rows, public_unresolved=pub_miss,
                       public_sessions=len(s)))
    print(json.dumps(report[-1]), flush=True)
    return s


for d in a.cls:
    for split in ("train", "val", "test"):
        p = os.path.join(d, f"{split}.parquet")
        if os.path.exists(p):
            resolve(pd.read_parquet(p, columns=["ref"]).ref, f"model:{os.path.basename(d.rstrip('/'))}:{split}")

pools = {}
for spec in a.pool:
    tag, js, refs = spec.split(":")
    P = json.load(open(js))
    R = pd.read_parquet(refs)
    name = os.path.basename(js)
    pools.setdefault(tag, {})[name[:3]] = (name, {s["id"]: s for s in P["sessions"]}, R)
    resolve(R.feat_ref, f"pool:{tag}:{name}")

for spec in a.workload:
    path, tag = spec.rsplit(":", 1)
    W = json.load(open(path))
    feat_refs, bad = [], 0
    for s in W["sessions"]:
        name, S, R = pools[tag][s["id"][:3]]
        ps = S[s["id"][3:]]
        same = len(ps["turns"]) >= len(s["turns"]) and all(
            x["kind"] == y["kind"] for x, y in zip(s["turns"], ps["turns"]))
        bad += not same
        feat_refs += list(R[R.sid == s["id"][3:]].feat_ref)
    got = resolve(feat_refs, f"workload:{os.path.basename(path)}")
    report[-1].update(workload_sessions=len(W["sessions"]), signature_mismatch=bad)
    if len(got) != len(W["sessions"]):
        report[-1]["note"] = f"{len(W['sessions'])} workload sessions map to {len(got)} source sessions"

U = pd.concat(rows, ignore_index=True)
U.to_parquet(a.out, index=False)
os.chmod(a.out, 0o600)
json.dump(report, open(os.path.splitext(a.out)[0] + ".report.json", "w"), indent=1)
print(U.drop_duplicates(["src", "session"]).groupby("src").size().rename("sessions seen (any use)").to_string())
