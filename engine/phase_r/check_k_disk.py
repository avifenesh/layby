#!/usr/bin/env python3
"""Pass criterion of the K disk-path pilot (cells pilot-*-kdisk): under CPU pressure the rule sends sessions to disk
and they come back from it unchanged.

PASS when all of these hold for the K arm:
  1. the rule placed sessions on disk: park_live.json counts ins + park > 0;
  2. K wrote chunks: park_live.json disk.chunks_on_disk + disk.written_jobs > 0 (chunks on disk at the end, or write
     jobs completed);
  3. K restored from disk: at least --min-verified replay rows carry a "verify" record (replay.py --verify-disk: the
     turn ran alone and the engine's disk tier was read during it);
  4. every verified turn's greedy output equals its cold recompute (verify.identical true); or, with
     --reference-rows (BENCH_PLAN draft 6, for a model whose stock engine is batch-variant: Gemma 4 with online fp8
     gave 18 of 20 identical in the pilot's stock batched-prefix control), the same probe on stock vLLM's own disk tier
     (arm T, same workload, same --verify-disk) is the reference: K fails only when its share of verified turns that
     differ from the cold recompute is higher than the reference's at one-sided Fisher exact p < 0.05, and the
     reference needs at least --min-verified verified turns too.
Prints one JSON object (the counts, each check, PASS or FAIL) and exits 0 on PASS, 1 on FAIL.

Usage: check_k_disk.py --rows CELL.K.HOST.jsonl --park-live CELL.K.HOST.park_live.json [--min-verified 1]
       [--reference-rows CELL.T.HOST.jsonl] [--out CHECK.json]
"""
import argparse
import json
import math
import sys

ALPHA = 0.05


def fisher_greater(a, n1, b, n2):
    """One-sided Fisher exact p-value that the first group's rate (a of n1) exceeds the second's (b of n2)."""
    k, n = a + b, n1 + n2
    den = math.comb(n, k)
    return sum(math.comb(n1, x) * math.comb(n2, k - x) for x in range(a, min(k, n1) + 1)) / den


def verified(rows):
    return [r["verify"] for r in rows if isinstance(r.get("verify"), dict)]


def check(rows, live, min_verified=1, reference=None):
    c = live.get("counts") or {}
    d = live.get("disk") or {}
    ver = verified(rows)
    res = dict(ins=int(c.get("ins", 0)), park=int(c.get("park", 0)), none=int(c.get("none", 0)),
               drop=int(c.get("drop", 0)), written_jobs=int(d.get("written_jobs", 0)),
               chunks_on_disk=int(d.get("chunks_on_disk", 0)), verified=len(ver),
               identical=sum(bool(v.get("identical")) for v in ver),
               disk_read_bytes=sum(float(v.get("disk_read_bytes") or 0) for v in ver),
               not_identical=[{k: v.get(k) for k in ("first_diff", "out_tokens", "disk_read_bytes")}
                              for v in ver if not v.get("identical")])
    res["checks"] = {"rule_chose_disk": res["ins"] + res["park"] > 0,
                     "chunks_written": res["chunks_on_disk"] + res["written_jobs"] > 0,
                     "disk_restores_verified": res["verified"] >= min_verified,
                     "outputs_identical": res["verified"] > 0 and res["identical"] == res["verified"]}
    if reference is not None:
        rv = verified(reference)
        ri = sum(bool(v.get("identical")) for v in rv)
        kd, rd = res["verified"] - res["identical"], len(rv) - ri
        p = fisher_greater(kd, res["verified"], rd, len(rv)) if res["verified"] and rv else None
        res["reference"] = dict(verified=len(rv), identical=ri, differ=rd, k_differ=kd, fisher_p_k_higher=p,
                                alpha=ALPHA, rule="stock vLLM's disk tier (arm T) on the same probe")
        del res["checks"]["outputs_identical"]
        res["checks"]["reference_verified"] = len(rv) >= min_verified
        res["checks"]["mismatch_not_above_stock"] = p is not None and p >= ALPHA
    res["verdict"] = "PASS" if all(res["checks"].values()) else "FAIL"
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", required=True)
    ap.add_argument("--park-live", required=True)
    ap.add_argument("--min-verified", type=int, default=1)
    ap.add_argument("--reference-rows", help="the stock reference arm's rows (T, same workload and probe)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rows = [json.loads(ln) for ln in open(a.rows) if ln.strip()]
    ref = [json.loads(ln) for ln in open(a.reference_rows) if ln.strip()] if a.reference_rows else None
    res = check(rows, json.load(open(a.park_live)), a.min_verified, ref)
    txt = json.dumps(res, indent=1)
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
