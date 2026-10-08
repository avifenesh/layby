#!/usr/bin/env python3
"""Check the data manifests: every listed file exists next to its manifest with the listed byte count and sha256.

    python scripts/verify_manifest.py                      # bench/MANIFEST.json and engine/workloads/MANIFEST.json
    python scripts/verify_manifest.py PATH/MANIFEST.json   # one or more manifests
    python scripts/verify_manifest.py --update ...         # rewrite bytes and sha256 from the files (after a rebuild)

Prints one line per file and exits 1 on any mismatch or missing file. Paths in a manifest are relative to its directory.
"""
import argparse, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = [os.path.join(ROOT, "bench", "MANIFEST.json"), os.path.join(ROOT, "engine", "workloads", "MANIFEST.json")]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifests", nargs="*", default=DEFAULT)
    ap.add_argument("--update", action="store_true", help="write the files' current bytes and sha256 into the manifests")
    a = ap.parse_args()
    bad = 0
    for mp in a.manifests:
        m = json.load(open(mp))
        base = os.path.dirname(os.path.abspath(mp))
        for e in m["files"]:
            p = os.path.join(base, e["path"])
            if not os.path.exists(p):
                print(f"MISSING {os.path.relpath(p, ROOT)}"); bad += 1; continue
            size, digest = os.path.getsize(p), sha256(p)
            if a.update:
                e["bytes"], e["sha256"] = size, digest
                print(f"UPDATED {os.path.relpath(p, ROOT)} {size} {digest}")
                continue
            ok = size == e["bytes"] and digest == e["sha256"]
            print(f"{'OK' if ok else 'FAIL'} {os.path.relpath(p, ROOT)} {size} bytes sha256 {digest[:16]}...")
            bad += not ok
        if a.update:
            with open(mp, "w") as f:
                json.dump(m, f, indent=1); f.write("\n")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
