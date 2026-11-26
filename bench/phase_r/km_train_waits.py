#!/usr/bin/env python3
"""The waits K_km is fitted on: the public sources' rows of the Layby-Dwell v6 training split, as a pool for
`python -m layby.curves fit-km`, plus a check that none of their sessions is in a Phase R workload.

Rows: CLS/train.parquet (the dataset v6 was trained from; v6 drew its 300K training rows from this split), sources
--src only (default SWE-chat, WildChat, Copilot: the public sources the Phase R workloads draw from). Each row is
one wait: kind, gap (seconds, uncapped) and returned (false: the session never came back, a wait censored at its
gap). Rows are grouped by source session (FEAT maps ref to session) into pool sessions whose turns carry kind,
gap_after and returned; the order inside a session does not matter to the KM fit.

Disjointness: every (source, session) of these rows is compared with every (src, source_session) in the Phase R
workloads' refs files (WDIR/*.refs.parquet), and every row ref with their feat_ref. Any overlap exits 3.

Output: OUT.json (the pool; ids and gaps from public sources, kept outside git) and OUT.report.json (counts).
Usage: km_train_waits.py OUT.json --cls ~/park-predictor-data/cls5 --feat ~/park-predictor-data/feat5.parquet \
       --workloads ~/.cache/layby_phase_r/workloads
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--cls", required=True)
ap.add_argument("--feat", required=True)
ap.add_argument("--workloads", required=True)
ap.add_argument("--src", nargs="+", default=["swechat", "wildchat", "copilot"])
a = ap.parse_args()

T = pq.read_table(os.path.join(a.cls, "train.parquet"), columns=["ref", "src", "kind", "gap", "returned"],
                  filters=[("src", "in", a.src)]).to_pandas()
F = pq.read_table(a.feat, columns=["ref", "src", "session"], filters=[("src", "in", a.src)]).to_pandas()
F = F.drop_duplicates("ref")
J = T.merge(F[["ref", "session"]], on="ref", how="left")
unmapped = int(J.session.isna().sum())
if unmapped:
    sys.exit(f"{unmapped} training rows have no session in {a.feat}")

refs = pd.concat([pd.read_parquet(p, columns=["src", "source_session", "feat_ref"])
                  for p in sorted(glob.glob(os.path.join(a.workloads, "*.refs.parquet")))], ignore_index=True)
wl_sess = set(zip(refs.src, refs.source_session.astype(str)))
tr_sess = set(zip(J.src, J.session.astype(str)))
overlap_sessions = len(wl_sess & tr_sess)
overlap_refs = int(J.ref.isin(set(refs.feat_ref)).sum())

sessions = []
for (src, sess), g in J.groupby(["src", "session"], sort=True):
    sessions.append(dict(id=f"{src}:{sess}", pool=src,
                         turns=[dict(kind=k, gap_after=float(x), returned=bool(r))
                                for k, x, r in zip(g.kind, g.gap, g.returned)]))
json.dump(dict(src="Layby-Dwell v6 training split (cls5/train), public sources " + ", ".join(a.src),
               sessions=sessions), open(a.out, "w"))
os.chmod(a.out, 0o600)
rep = dict(rows=len(J), sessions=len(sessions),
           rows_by_src_kind={f"{s}:{k}": int(n) for (s, k), n in J.groupby(["src", "kind"]).size().items()},
           censored_by_src={s: int((~g.returned.astype(bool)).sum()) for s, g in J.groupby("src")},
           workload_refs=len(refs), workload_sessions=len(wl_sess),
           overlap_sessions=overlap_sessions, overlap_refs=overlap_refs)
json.dump(rep, open(os.path.splitext(a.out)[0] + ".report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
if overlap_sessions or overlap_refs:
    sys.exit(3)
