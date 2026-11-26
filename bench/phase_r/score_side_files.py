#!/usr/bin/env python3
"""Fill surv_v6 in the Phase R workloads: score each workload's side file with the frozen Layby-Dwell v6 model.

The scoring path is the live sidecar's (layby.sidecar.scorer.Scorer, the class layby.sidecar.proxy loads): the
release's model.safetensors and config.json (per-kind temperatures), base Laya's tokenizer and encoder config,
max_len 1024, bf16 autocast on GPU, batches of at most --batch states (the proxy's default 32) in the side file's
order. So the live check of the measured runs (replay.py --side-file, max |live - precomputed|) compares the same
model, calibration and dtype.

For every NAME.side.parquet in DIR with a NAME.json beside it: the scorable rows (server-view state, server kind) are
scored, and every workload turn with a wait (gap_after not None) whose side row has a curve gets surv_v6 (15 values,
P(T > h) at h = 0.5 ... 1800 s). Turns with no state (cc-traces: no text) keep surv_v6 null. Public turns with a
wait but no curve are counted per workload; over 1% stops the run before any file is written. NAME.json is rewritten
as build_workloads.py writes it, and NAME.manifest.json gets the new bytes and sha256 and a curves.surv_v6 record.

Writes DIR/surv_v6.json (content-free): per workload the counts, the scoring time, and the sanity check (mean
predicted P(T > 60 s) per wait kind against the observed share of waits over 60 s), plus the model files' sha256,
the software versions, the GPU, peak GPU memory and a batch-composition check (the same states scored alone and in
batches: the floor of the live check's max |live - precomputed|).

Usage: score_side_files.py DIR [--model DIR] [--laya DIR] [--revision 101cc44] [--batch 32] [--device cuda]
       [--only NAME ...] [--dry-run]
--model defaults to the Hugging Face snapshot of avifenesh/layby-dwell at --revision (from the local cache when it
holds it), --laya to base Laya at the revision the release names (config.json base_model). The side files hold
public-source text: they stay local, and nothing they contain is written to the workloads or the receipt.
"""
import argparse
import glob
import hashlib
import importlib.metadata as md
import json
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = "avifenesh/layby-dwell"
H60 = 10                     # index of h = 60 s in the release's horizons
MAX_MISSING = 0.01


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def version(pkg):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def resolve(a):
    from huggingface_hub import snapshot_download
    model = a.model or snapshot_download(REPO, revision=a.revision,
                                         allow_patterns=["config.json", "model.safetensors", "encoder/*", "tokenizer/*"])
    cfg = json.load(open(os.path.join(model, "config.json")))
    if a.laya:
        laya = a.laya
    else:
        base, _, rev = cfg.get("base_model", "convaiinnovations/laya@").partition("@")
        laya = snapshot_download(base, revision=rev or None)
    return model, laya, cfg


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("dir")
    ap.add_argument("--model", help="local copy of the Layby-Dwell release (default: the Hub snapshot at --revision)")
    ap.add_argument("--revision", default="101cc44")
    ap.add_argument("--laya", help="base Laya directory (default: the revision in the release's base_model)")
    ap.add_argument("--batch", type=int, default=32, help="states per forward pass (layby.sidecar.proxy --batch)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", nargs="*", help="workload names (default: every NAME.side.parquet)")
    ap.add_argument("--noise-states", type=int, default=256, help="states rescored one at a time for the batch check")
    ap.add_argument("--dry-run", action="store_true", help="score and report, write nothing")
    a = ap.parse_args()

    model_dir, laya_dir, cfg = resolve(a)
    os.environ["LAYA_DIR"] = laya_dir
    import torch
    from layby.sidecar.scorer import Scorer

    ckpt, calib = os.path.join(model_dir, "model.safetensors"), os.path.join(model_dir, "config.json")
    t0 = time.time()
    sc = Scorer(ckpt, calib, a.device)
    load_s = time.time() - t0
    if sc.dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(sc.dev)
    H = list(sc.H)
    assert H[H60] == 60, H

    names = a.only or sorted(os.path.basename(p)[: -len(".side.parquet")]
                             for p in glob.glob(os.path.join(a.dir, "*.side.parquet")))
    scored, report, failed = {}, {}, []
    total_states = total_s = 0.0
    noise = None
    for name in names:
        side = pd.read_parquet(os.path.join(a.dir, f"{name}.side.parquet"),
                               columns=["session", "turn", "src", "kind", "state", "scorable"])
        W = json.load(open(os.path.join(a.dir, f"{name}.json")))
        ok = side[side.scorable].reset_index(drop=True)
        S = np.full((len(ok), len(H)), np.nan, np.float32)
        if sc.dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        for b in range(0, len(ok), a.batch):
            S[b:b + a.batch] = sc.score(ok.state[b:b + a.batch].tolist(), ok.kind[b:b + a.batch].tolist())
        if sc.dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t1
        total_states += len(ok); total_s += dt
        if noise is None and len(ok):
            n = min(a.noise_states, len(ok))
            one = np.vstack([sc.score([ok.state[i]], [ok.kind[i]]) for i in range(n)])
            d = np.abs(one - S[:n])
            noise = dict(workload=name, states=n, batch=a.batch, max_abs_diff=float(np.nanmax(d)),
                         p99_abs_diff=float(np.nanpercentile(d.max(1), 99)), mean_abs_diff=float(np.nanmean(d)))
        curve = {(s, int(t)): row for s, t, row in zip(ok.session, ok.turn, S)}
        kinds = {(s, int(t)): (k, src) for s, t, k, src in zip(side.session, side.turn, side.kind, side.src)}

        n_wait = n_set = pub_wait = pub_missing = nostate_wait = 0
        by = {}
        for s in W["sessions"]:
            for i, t in enumerate(s["turns"]):
                if t.get("gap_after") is None:
                    continue
                n_wait += 1
                k, src = kinds.get((s["id"], i), (None, None))
                row = curve.get((s["id"], i))
                public = src is not None and src != "cc_traces"
                pub_wait += public
                if row is None or np.isnan(row).any():
                    t["surv_v6"] = None
                    if public or src is None:
                        pub_missing += 1
                    else:
                        nostate_wait += 1
                    continue
                t["surv_v6"] = [float(x) for x in row]
                n_set += 1
                g = t.get("gap_uncapped", t["gap_after"])
                b_ = by.setdefault(t.get("kind") or "?", [0, 0.0, 0])
                b_[0] += 1; b_[1] += float(row[H60]); b_[2] += g > 60
        share = pub_missing / pub_wait if pub_wait else 0.0
        sanity = {k: dict(waits=v[0], pred_p_gt_60=round(v[1] / v[0], 4), obs_p_gt_60=round(v[2] / v[0], 4))
                  for k, v in sorted(by.items())}
        report[name] = dict(side_rows=len(side), states_scored=len(ok), nan_states=int(np.isnan(S).any(1).sum()),
                            turns_with_wait=n_wait, surv_v6_set=n_set, public_turns_with_wait=pub_wait,
                            public_without_curve=pub_missing, share_without_curve=round(share, 5),
                            no_state_turns_with_wait=nostate_wait, score_s=round(dt, 2),
                            states_per_s=round(len(ok) / dt, 1) if dt and len(ok) else None, sanity=sanity)
        print(name, json.dumps({k: v for k, v in report[name].items() if k != "sanity"}), json.dumps(sanity), flush=True)
        if share > MAX_MISSING:
            failed.append(name)
        scored[name] = W

    peak = dict(allocated_gib=round(torch.cuda.max_memory_allocated(sc.dev) / 2**30, 3),
                reserved_gib=round(torch.cuda.max_memory_reserved(sc.dev) / 2**30, 3),
                device=torch.cuda.get_device_name(sc.dev)) if sc.dev.type == "cuda" else None
    pooled = {}
    for r in report.values():
        for k, v in r["sanity"].items():
            p = pooled.setdefault(k, [0, 0.0, 0.0])
            p[0] += v["waits"]; p[1] += v["pred_p_gt_60"] * v["waits"]; p[2] += v["obs_p_gt_60"] * v["waits"]
    pooled = {k: dict(waits=v[0], pred_p_gt_60=round(v[1] / v[0], 4), obs_p_gt_60=round(v[2] / v[0], 4))
              for k, v in sorted(pooled.items())}
    rec = dict(model=dict(repo=REPO, revision=a.revision, dir=model_dir, model_sha256=sha256(ckpt),
                          config_sha256=sha256(calib), temperature=cfg["layby_dwell"]["temperature"],
                          base_laya=laya_dir),
               scorer="layby.sidecar.scorer.Scorer (the live sidecar's path)", dtype="bf16 autocast" if sc.dev.type == "cuda"
               else "fp32", batch=a.batch, max_len=sc.cfg["max_len"],
               software=dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                             transformers=version("transformers"), laya=version("laya")),
               gpu_memory=peak, load_s=round(load_s, 1), states=int(total_states), score_s=round(total_s, 1),
               states_per_s=round(total_states / total_s, 1) if total_s else None, batch_noise=noise,
               pooled_sanity=pooled, workloads=report, failed=failed)
    print("pooled", json.dumps(pooled), "noise", json.dumps(noise), "gpu", json.dumps(peak),
          f"throughput {rec['states_per_s']} states/s", flush=True)
    if failed:
        print(f"FAILED: over {MAX_MISSING:.0%} of public turns with a wait lack a curve in {failed}; nothing written",
              file=sys.stderr, flush=True)
        sys.exit(2)
    if a.dry_run:
        return
    for name, W in scored.items():
        p = os.path.join(a.dir, f"{name}.json")
        W["curve"] = f"surv_v6 (Layby-Dwell v6, {REPO}@{a.revision}, sidecar scorer, {rec['dtype']})"
        tmp = p + ".tmp"
        json.dump(W, open(tmp, "w"))
        os.chmod(tmp, os.stat(p).st_mode & 0o777)
        os.replace(tmp, p)
        mp = os.path.join(a.dir, f"{name}.manifest.json")
        man = json.load(open(mp))
        man["bytes"], man["sha256"] = os.path.getsize(p), sha256(p)
        man.setdefault("curves", {})["surv_v6"] = dict(
            model=f"{REPO}@{a.revision}", model_sha256=rec["model"]["model_sha256"],
            config_sha256=rec["model"]["config_sha256"], scorer=rec["scorer"], dtype=rec["dtype"], batch=a.batch,
            turns_with_curve=report[name]["surv_v6_set"], public_without_curve=report[name]["public_without_curve"],
            no_state_turns=report[name]["no_state_turns_with_wait"],
            script="bench/phase_r/score_side_files.py")
        tmp = mp + ".tmp"
        json.dump(man, open(tmp, "w"), indent=1)
        os.chmod(tmp, os.stat(mp).st_mode & 0o777)
        os.replace(tmp, mp)
    json.dump(rec, open(os.path.join(a.dir, "surv_v6.json"), "w"), indent=1)
    print(f"wrote {len(scored)} workloads and {os.path.join(a.dir, 'surv_v6.json')}", flush=True)


if __name__ == "__main__":
    main()
