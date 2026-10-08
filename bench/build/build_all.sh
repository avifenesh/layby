#!/usr/bin/env bash
# Rebuild the four ReturnBench pools from the public source data, pinned to the versions the reference
# pools were built from. Run from an empty work directory; the raw data lands under ./public (about 40 GB).
# Needs: python with pandas, pyarrow, duckdb; the hf CLI (SWE-chat is gated: accept its terms on the
# Hugging Face page first); gh; curl; zstd.
# Step 4 (survival curves) needs the Layby-Dwell model; see README.md. Without it the pools come out
# with no curves, and only the rules that do not read curves (C0, wt, cont, cj, ka) can run on them.
set -euo pipefail
B=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-python3}
OUT=${OUT:-pools}

# 1. source data, pinned
mkdir -p public
hf download SALT-NLP/SWE-chat --repo-type dataset --revision f66cca95b14caaa4177f7ed5eaa424608dadcffa --local-dir public/swechat
hf download allenai/WildChat-4.8M --repo-type dataset --revision c827c6df8fcf008219ffaffa4d1dd77491099367 --local-dir public/wildchat48
gh release download ghcp-coding-agent-2026 -R Azure/AzurePublicDataset -D public/copilot2026 --pattern 'date.*.tar.gz' --skip-existing
curl -L --fail -o public/tracelab.duckdb https://github.com/uw-syfi/TraceLab/releases/download/v0.0.2/syfi_coding_trace.duckdb
sha256sum -c "$B/SHA256SUMS.raw"

# 2. idle boundaries (one JSON line per idle period: kind, gap, context, tool, user; no text)
$PY "$B/extract_transcripts.py" claude public/swechat/transcripts --src swechat --users public/swechat/sessions.parquet > swechat.bnd.jsonl
$PY "$B/extract_tracelab.py" public/tracelab.duckdb > tracelab.bnd.jsonl
$PY "$B/extract_wildchat.py" public/wildchat48/data wildchat.bnd.jsonl wildchat.text.parquet --frac 0.08
rm -f wildchat.text.parquet        # the extractor also writes message text for content models; the bench does not use it
$PY "$B/extract_copilot.py" public/copilot2026 copilot.bnd.jsonl --frac 0.25
for f in swechat tracelab wildchat copilot; do
  echo "$(LC_ALL=C sort -S 2G $f.bnd.jsonl | sha256sum | cut -c1-64)  $f.bnd.jsonl"
done > bnd.sums
grep -v '^#' "$B/SHA256SUMS.bnd" | diff - bnd.sums && echo "boundaries match the reference"

# 3. chronological split, user hold-out, replay pools
mkdir -p "$OUT" refs
for x in "swechat swechat 6" "tracelab tracelab_claude 6" "wildchat wildchat 4"; do
  set -- $x
  $PY "$B/split.py" $1.split.parquet $1.bnd.jsonl
  $PY "$B/build_replay_pop.py" $1.split.parquet $2 $2.raw.json --refs-out refs/$2.refs.parquet --gap-cap inf --min-turns $3
done
$PY "$B/split.py" copilot.split.parquet copilot.bnd.jsonl
$PY "$B/build_replay_pop.py" copilot.split.parquet copilot copilot.raw.json --refs-out refs/copilot.refs.parquet --gap-cap inf --any-user --sessions 200

# 4. survival curves (optional): score every turn's server-view state with Layby-Dwell, write
#    PROBS.parquet (ref, p0..p14), then:
#    $PY "$B/attach_curves.py" POOL.raw.json refs/POOL.refs.parquet PROBS.parquet v6 POOL.raw.json --key feat_ref

# 5. publishable pools
for p in swechat tracelab_claude wildchat copilot; do
  $PY "$B/sanitize.py" $p.raw.json $p "$OUT/$p.json"
done
