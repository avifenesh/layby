#!/usr/bin/env bash
# apply_patch.sh PYTHON apply|check: the HiCacheFile page-cache patch (hicache_fadvise.patch) on the SGLang installed in
# PYTHON's environment. Every SGLang arm runs with it (BENCH_PLAN section 5). The patch is written for SGLang 0.5.21
# and is refused on any other version.
#   apply: patch the installed hicache_storage.py unless it already carries the patch; prints PATCHED or
#          ALREADY_PATCHED and the file's sha256.
#   check: exit 0 and print PATCHED with the sha256 if both patch hunks are present, else exit 1.
# Exit codes: 0 ok, 1 not patched (check), 2 wrong version or patch failure.
set -euo pipefail
PY=${1:?usage: apply_patch.sh PYTHON apply|check}; MODE=${2:?usage: apply_patch.sh PYTHON apply|check}
HERE=$(cd "$(dirname "$0")" && pwd)
WANT=${SGLANG_VERSION_WANT:-0.5.21}
# find_spec does not import sglang (importing it pulls in torch and CUDA)
SITE=$("$PY" -c 'import importlib.util,os; s=importlib.util.find_spec("sglang"); print(os.path.dirname(os.path.dirname(s.origin)))')
VER=$("$PY" -c 'import importlib.metadata as m; print(m.version("sglang"))' 2>/dev/null || echo unknown)
F=$SITE/sglang/srt/mem_cache/hicache_storage.py
[ -f "$F" ] || { echo "NO_FILE $F"; exit 2; }
hunks=$(grep -c 'layby-fadvise' "$F" || true)
case $MODE in
  check)
    if [ "$hunks" = 2 ]; then echo "PATCHED $VER $(sha256sum "$F" | cut -d' ' -f1) $F"; exit 0; fi
    echo "NOT_PATCHED $VER hunks=$hunks $F"; exit 1 ;;
  apply)
    [ "$VER" = "$WANT" ] || { echo "WRONG_VERSION sglang $VER, patch is for $WANT"; exit 2; }
    if [ "$hunks" = 2 ]; then echo "ALREADY_PATCHED $VER $(sha256sum "$F" | cut -d' ' -f1) $F"; exit 0; fi
    [ "$hunks" = 0 ] || { echo "PARTIAL hunks=$hunks $F"; exit 2; }
    patch --forward --dry-run -p2 -d "$SITE" < "$HERE/hicache_fadvise.patch" >/dev/null || { echo "PATCH_DOES_NOT_APPLY $F"; exit 2; }
    patch --forward -p2 -d "$SITE" < "$HERE/hicache_fadvise.patch" >/dev/null
    rm -f "$SITE"/sglang/srt/mem_cache/__pycache__/hicache_storage.*.pyc
    echo "PATCHED $VER $(sha256sum "$F" | cut -d' ' -f1) $F" ;;
  *) echo "usage: apply_patch.sh PYTHON apply|check"; exit 2 ;;
esac
