#!/usr/bin/env python3
"""Attach surv_v6 to the Phase R workloads from one scored pool: the v6 curves of pool_all (score_side_parquet.py on
the compat box, the sidecar's scorer, receipt surv_v6_receipt.json) joined to each workload's turns.

Every workload turn is a source boundary (feat_ref, side_files.py); pool_all.side.parquet holds every turn of every
candidate session, one row per feat_ref, and the scores file holds that row's curve under the pool's key. So a
workload side row maps to its curve through feat_ref -> pool key. The join is checked, not assumed: the workload's
server-view state and kind must equal the pool row's for every scorable turn (a mismatch stops the run, nothing
written), and the scores file must be the one the receipt names (sha256 of the pool side file and of the scores).

Per workload NAME.json with NAME.side.parquet beside it: every turn with a wait (gap_after not None) whose side row
is scorable gets surv_v6 (15 values, P(T > h) at the receipt's horizons). Turns without a state (cc-traces, no
text) keep surv_v6 null. Public turns with a wait and no curve are counted; any count above 1% of a workload's
public waits stops the run before a file is written (the plan fails the build there; the expected count is 0).
NAME.json is rewritten with its curve field set, NAME.manifest.json gets bytes, sha256 and a curves.surv_v6 record.

Sanity check per workload and wait kind: mean predicted P(T > 60 s) and P(T > 300 s) against the observed share on
the workload's gaps, capped (gap_after; the cap is 300 s, so at 300 s it is the share at the cap) and uncapped
(gap_uncapped). Written to DIR/surv_v6.json with the counts and the receipt; the receipt is copied to
DIR/surv_v6_receipt.json. With --side-jsonl, NAME.side.jsonl (the live sidecar's format, engine/replay.py
--side-file: session, turn, state, kind) is written for every workload with scorable turns, mode 600 (public-source
text: it stays local and goes only to the run hosts).

Usage: attach_surv_v6.py DIR --scores pool_all.surv_v6.parquet --receipt surv_v6_receipt.json [--pool pool_all]
       [--only NAME ...] [--side-jsonl] [--dry-run]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd

MAX_MISSING = 0.01
SCRIPT = "bench/phase_r/attach_surv_v6.py"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def put(path, write, mode=None):
    """Write through a hidden temporary name and a rename; keep the old file's mode unless one is given."""
    d, b = os.path.split(path)
    tmp = os.path.join(d, f".{b}.tmp")
    with open(tmp, "w") as f:
        write(f)
    m = mode if mode is not None else (os.stat(path).st_mode & 0o777 if os.path.exists(path) else 0o644)
    os.chmod(tmp, m)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--scores", required=True, help="pool_all.surv_v6.parquet (key, session, turn, kind, src, partition, surv_v6)")
    ap.add_argument("--receipt", required=True, help="surv_v6_receipt.json written with the scores")
    ap.add_argument("--pool", default="pool_all", help="the scored pool's name in DIR (its side file is the join)")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--side-jsonl", action="store_true", help="also write NAME.side.jsonl for live scoring")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    rec = json.load(open(a.receipt))
    H = rec["horizons_s"]
    i60, i300 = H.index(60), H.index(300)
    pool_side = os.path.join(a.dir, f"{a.pool}.side.parquet")
    want = rec["workloads"][a.pool]
    got = dict(side_sha256=sha256(pool_side), file_sha256=sha256(a.scores))
    for k, v in got.items():
        if want[k] != v:
            sys.exit(f"{k}: receipt {want[k]} but the file has {v}: these scores are not for this pool side file")
    sc = pd.read_parquet(a.scores, columns=["key", "kind", "surv_v6"]).set_index("key")
    ps = pd.read_parquet(pool_side, columns=["key", "feat_ref", "kind", "state"])
    if not ps.feat_ref.is_unique or not ps.key.is_unique:
        sys.exit("pool side file: feat_ref or key not unique")
    ps = ps.set_index("feat_ref")

    names = a.only or sorted(os.path.basename(p)[: -len(".side.parquet")]
                             for p in glob.glob(os.path.join(a.dir, "*.side.parquet")))
    names = [n for n in names if n != a.pool and os.path.exists(os.path.join(a.dir, f"{n}.json"))]
    out, report, failed = {}, {}, []
    for name in names:
        side = pd.read_parquet(os.path.join(a.dir, f"{name}.side.parquet"),
                               columns=["session", "turn", "src", "feat_ref", "kind", "state", "scorable"])
        ok = side[side.scorable]
        pk = ps.reindex(ok.feat_ref)
        bad_join = int(pk.key.isna().sum())
        bad_state = int((pk.state.values != ok.state.values).sum()) - bad_join
        bad_kind = int((pk.kind.values != ok.kind.values).sum()) - bad_join
        keys = pk.key.values
        scored = sc.reindex(keys)
        bad_score_kind = int((scored.kind.values != ok.kind.values).sum()) - int(scored.kind.isna().sum())
        if bad_join or bad_state or bad_kind or bad_score_kind:
            sys.exit(f"{name}: join check failed (feat_ref not in pool {bad_join}, state differs {bad_state}, kind "
                     f"differs {bad_kind}, scored kind differs {bad_score_kind}); nothing written")
        curve = {}
        for s, t, row in zip(ok.session, ok.turn, scored.surv_v6.values):
            curve[(s, int(t))] = None if row is None else np.asarray(row, dtype=float)
        src_of = {(s, int(t)): src for s, t, src in zip(side.session, side.turn, side.src)}

        W = json.load(open(os.path.join(a.dir, f"{name}.json")))
        n_wait = n_set = pub_wait = pub_missing = nostate = 0
        by = {}
        for s in W["sessions"]:
            for i, t in enumerate(s["turns"]):
                if t.get("gap_after") is None:
                    continue
                n_wait += 1
                src = src_of.get((s["id"], i))
                public = src != "cc_traces"           # a turn with no side row counts as public (it should have one)
                pub_wait += public
                row = curve.get((s["id"], i))
                if row is None or len(row) != len(H) or np.isnan(row).any():
                    t["surv_v6"] = None
                    pub_missing += public
                    nostate += not public
                    continue
                t["surv_v6"] = [float(x) for x in row]
                n_set += 1
                g, gu = float(t["gap_after"]), float(t.get("gap_uncapped", t["gap_after"]))
                b = by.setdefault(t.get("kind") or "?", np.zeros(7))
                b += [1, row[i60], g > 60, gu > 60, row[i300], g >= W.get("gap_cap", 300.0), gu > 300]
        share = pub_missing / pub_wait if pub_wait else 0.0
        sanity = {}
        for k, v in sorted(by.items()):
            n = v[0]
            sanity[k] = dict(waits=int(n), pred_p_gt_60=round(v[1] / n, 4), obs_gt_60_capped=round(v[2] / n, 4),
                             obs_gt_60_uncapped=round(v[3] / n, 4), pred_p_gt_300=round(v[4] / n, 4),
                             obs_at_cap_300_capped=round(v[5] / n, 4), obs_gt_300_uncapped=round(v[6] / n, 4))
        report[name] = dict(turns=int(sum(len(s["turns"]) for s in W["sessions"])), turns_with_wait=n_wait,
                            surv_v6_set=n_set, public_turns_with_wait=pub_wait, public_without_curve=pub_missing,
                            share_without_curve=round(share, 5), no_state_turns_with_wait=nostate, sanity=sanity)
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "sanity"}), flush=True)
        if share > MAX_MISSING:
            failed.append(name)
        out[name] = (W, ok)

    if failed:
        sys.exit(f"FAILED: over {MAX_MISSING:.0%} of public turns with a wait lack a curve in {failed}; nothing written")
    if a.dry_run:
        print(json.dumps({n: r["sanity"] for n, r in report.items()}, indent=1))
        return
    m = rec["model"]
    prov = dict(model=f"{m['repo']}@{m['revision']}", model_sha256=m["model_sha256"], config_sha256=m["config_sha256"],
                scorer=rec["scorer"], dtype=rec["dtype"], batch=rec["batch"], max_len=rec["max_len"],
                scores_file=os.path.basename(a.scores), scores_sha256=got["file_sha256"],
                pool_side_sha256=got["side_sha256"], receipt="surv_v6_receipt.json",
                join="side feat_ref -> pool_all key (state and kind checked equal)", script=SCRIPT)
    for name, (W, ok) in out.items():
        p = os.path.join(a.dir, f"{name}.json")
        W["curve"] = f"surv_v6 (Layby-Dwell v6, {m['repo']}@{m['revision']}, sidecar scorer, {rec['dtype']})"
        put(p, lambda f: json.dump(W, f))
        mp = os.path.join(a.dir, f"{name}.manifest.json")
        man = json.load(open(mp))
        man["bytes"], man["sha256"] = os.path.getsize(p), sha256(p)
        r = report[name]
        man.setdefault("curves", {})["surv_v6"] = dict(prov, turns_with_curve=r["surv_v6_set"],
                                                       public_without_curve=r["public_without_curve"],
                                                       no_state_turns=r["no_state_turns_with_wait"])
        put(mp, lambda f: json.dump(man, f, indent=1))
        if a.side_jsonl and len(ok):
            sj = os.path.join(a.dir, f"{name}.side.jsonl")
            put(sj, lambda f: f.writelines(json.dumps(dict(session=s, turn=int(t), state=st, kind=k), ensure_ascii=False) + "\n"
                                           for s, t, st, k in zip(ok.session, ok.turn, ok.state, ok.kind)), mode=0o600)
            man["side_jsonl"] = dict(rows=len(ok), bytes=os.path.getsize(sj), sha256=sha256(sj))
            put(mp, lambda f: json.dump(man, f, indent=1))
    shutil.copyfile(a.receipt, os.path.join(a.dir, "surv_v6_receipt.json"))
    summ = dict(receipt=rec, provenance=prov, workloads=report)
    put(os.path.join(a.dir, "surv_v6.json"), lambda f: json.dump(summ, f, indent=1))
    print(f"wrote {len(out)} workloads, surv_v6.json and surv_v6_receipt.json in {a.dir}", flush=True)


if __name__ == "__main__":
    main()
