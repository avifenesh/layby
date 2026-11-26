#!/usr/bin/env python3
"""Side files for the Phase R workloads: per turn, the server-view state the frozen Layby-Dwell model scores.

For every NAME.refs.parquet in DIR (build_workloads.py), the turns' source boundaries (feat_ref) are built into
states by the training state builder (--builder: build_cls_v5.py FEAT CONTENT TRACELAB OUT --refs REFS) from
the rows of the sessions involved only, so a state is the one the model was trained on: the turn's real text,
real context, session history and the user's earlier waits. The server view then drops the data source and shows
an agent's own continuation (workflow) as human, as layby_dwell.bench.server does.

Writes DIR/NAME.side.parquet: key (session:turn, the workload's), session, turn, src, source_session, feat_ref,
kind (server view: tool or human), state (JSON), scorable. Source rows without text or turn kind (cc-traces)
get scorable false and no state. The side files hold public-source text: they stay local and never enter a
workload or the repository. Prints the share of public-source turns without a state (the plan fails a workload
over 1%).

Every output is written under a temporary name and renamed, so a copy taken while the builder runs never sees a
partial side file. --only PREFIX ... limits the run to workloads whose name starts with a prefix. A cached state
build in WORKDIR is reused only when it was built for the same boundaries.

Usage: side_files.py DIR --feat feat5.parquet --content content5.parquet --tracelab tracelab5.bnd.jsonl \
       --builder build_cls_v5.py [--python PY] [--work WORKDIR] [--only PREFIX ...]
"""
import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd
import pyarrow.parquet as pq


def server(state):
    """layby_dwell.bench.server: the state a generic OpenAI-compatible server can build."""
    d = json.loads(state)
    d.pop("src", None)
    if d.get("kind") == "workflow":
        d["kind"] = "human"
    if d.get("recent"):
        d["recent"] = [["human" if x and x[0] == "workflow" else x[0]] + list(x[1:]) for x in d["recent"]]
    m = d.get("meta") or {}
    if m.get("prev_kind") == "workflow":
        m["prev_kind"] = "human"
    return json.dumps(d, ensure_ascii=False)


ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--feat", required=True)
ap.add_argument("--content", required=True)
ap.add_argument("--tracelab", required=True)
ap.add_argument("--builder", required=True)
ap.add_argument("--python", default=sys.executable)
ap.add_argument("--work")
ap.add_argument("--only", nargs="*", help="workload name prefixes (default: every NAME.refs.parquet in DIR)")
a = ap.parse_args()
work = a.work or os.path.join(a.dir, "_states")
os.makedirs(work, mode=0o700, exist_ok=True)

files = sorted(f for f in glob.glob(os.path.join(a.dir, "*.refs.parquet"))
               if not a.only or os.path.basename(f).startswith(tuple(a.only)))
R = {f: pd.read_parquet(f) for f in files}
pub = pd.concat([r[r.src != "cc_traces"] for r in R.values()])
refs = pd.DataFrame({"feat_ref": sorted(set(pub.feat_ref))})
pairs = pub[["src", "source_session"]].drop_duplicates()
print(f"{len(refs)} public boundaries from {len(pairs)} sessions", flush=True)

out_dir = os.path.join(work, "cls")
rp = os.path.join(work, "refs.parquet")
cached = os.path.exists(os.path.join(out_dir, "replay.parquet")) and os.path.exists(rp)
if cached and set(pd.read_parquet(rp).feat_ref) != set(refs.feat_ref):
    raise SystemExit(f"{work} holds states built for other boundaries: pass a new --work")
if not cached:
    sess = sorted(set(pairs.source_session))
    ft = pq.read_table(a.feat, filters=[("session", "in", sess), ("src", "in", sorted(set(pairs.src)))])
    f = ft.to_pandas()
    want = set(zip(pairs.src, pairs.source_session))           # built once (per row it is quadratic)
    f = f[[(s, x) in want for s, x in zip(f.src, f.session)]]
    fp = os.path.join(work, "feat.parquet")
    f.to_parquet(fp, index=False)
    ct = pq.read_table(a.content, filters=[("ref", "in", sorted(set(f.ref)))])
    cp = os.path.join(work, "content.parquet")
    ct.to_pandas().to_parquet(cp, index=False)
    refs.to_parquet(rp, index=False)
    for p in (fp, cp, rp):
        os.chmod(p, 0o600)
    print(f"feature rows {len(f)}, content rows {ct.num_rows}", flush=True)
    subprocess.run([a.python, a.builder, fp, cp, a.tracelab, out_dir, "--refs", rp], check=True)

S = pd.read_parquet(os.path.join(out_dir, "replay.parquet"), columns=["ref", "kind", "state"]).drop_duplicates("ref")
S["state"] = [server(x) for x in S.state]
S["kind"] = S.kind.where(S.kind != "workflow", "human")
S = S.set_index("ref")
summary = {}
for f, r in R.items():
    name = os.path.basename(f)[: -len(".refs.parquet")]
    d = r.copy()
    m = d.feat_ref.map(S.state)
    d["kind"] = d.feat_ref.map(S.kind)
    d["state"] = m
    d["scorable"] = d.state.notna()
    p = os.path.join(a.dir, f"{name}.side.parquet")
    tmp = os.path.join(a.dir, f".{name}.side.parquet.tmp")      # hidden: no *.side.parquet glob matches it
    d.to_parquet(tmp, index=False)
    os.chmod(tmp, 0o600)
    os.replace(tmp, p)
    pubm = d[d.src != "cc_traces"]
    summary[name] = dict(turns=len(d), public_turns=len(pubm), public_without_state=int((~pubm.scorable).sum()),
                         share_without_state=round(float((~pubm.scorable).mean()), 5) if len(pubm) else None,
                         unscorable_source_turns=int((d.src == "cc_traces").sum()))
    print(name, summary[name], flush=True)
sp = os.path.join(a.dir, "side_files.json")
if os.path.exists(sp):                 # keep the entries of workloads this run did not rebuild
    summary = {k: v for k, v in json.load(open(sp)).items() if k not in summary} | summary
with open(sp + ".tmp", "w") as fh:
    json.dump(dict(sorted(summary.items())), fh, indent=1)
os.replace(sp + ".tmp", sp)
