#!/usr/bin/env bash
# apply_patch.sh PYTHON apply|check [VARIANT]: the HiCacheFile page-cache patch on the SGLang installed in PYTHON's
# environment. Every SGLang arm runs with it (BENCH_PLAN section 5). Written for SGLang 0.5.21; refused on any other
# version. VARIANT (default fdatasync):
#   fdatasync        hicache_fadvise.patch: a write ends with fdatasync, then posix_fadvise(DONTNEED)
#   sync_file_range  hicache_fadvise_sfr.patch: sync_file_range(WAIT_BEFORE | WRITE | WAIT_AFTER) on the file's data
#                    instead of fdatasync, then DONTNEED (the pilot, orchestrate.py hicache-pilot, picks it when
#                    fdatasync slows HC-wt's L3 writes under the cell's io.max)
# Both drop the pages after a read too.
#   apply: patch the installed hicache_storage.py unless it already carries this variant (the other variant is
#          reversed first); prints PATCHED or ALREADY_PATCHED, the variant and the file's sha256.
#   check: exit 0 and print PATCHED with the variant and the sha256 if this variant's hunks are all present, else
#          exit 1.
# Exit codes: 0 ok, 1 not patched with this variant (check), 2 wrong version, unknown variant or patch failure.
set -euo pipefail
PY=${1:?usage: apply_patch.sh PYTHON apply|check [VARIANT]}; MODE=${2:?usage: apply_patch.sh PYTHON apply|check [VARIANT]}
VARIANT=${3:-fdatasync}
HERE=$(cd "$(dirname "$0")" && pwd)
WANT=${SGLANG_VERSION_WANT:-0.5.21}
declare -A PATCH=([fdatasync]=hicache_fadvise.patch [sync_file_range]=hicache_fadvise_sfr.patch)
[ -n "${PATCH[$VARIANT]:-}" ] || { echo "UNKNOWN_VARIANT $VARIANT (fdatasync or sync_file_range)"; exit 2; }
# find_spec does not import sglang (importing it pulls in torch and CUDA)
SITE=$("$PY" -c 'import importlib.util,os; s=importlib.util.find_spec("sglang"); print(os.path.dirname(os.path.dirname(s.origin)))')
VER=$("$PY" -c 'import importlib.metadata as m; print(m.version("sglang"))' 2>/dev/null || echo unknown)
F=$SITE/sglang/srt/mem_cache/hicache_storage.py
[ -f "$F" ] || { echo "NO_FILE $F"; exit 2; }
state() {   # which variant the file carries: none, fdatasync, sync_file_range or partial (marker lines per variant)
  local all sfr
  all=$(grep -c 'layby-fadvise' "$F" || true); sfr=$(grep -c 'layby-fadvise-sfr' "$F" || true)
  case "$all/$sfr" in 0/0) echo none ;; 2/0) echo fdatasync ;; 3/2) echo sync_file_range ;; *) echo "partial($all/$sfr)" ;; esac
}
st=$(state)
case $MODE in
  check)
    if [ "$st" = "$VARIANT" ]; then echo "PATCHED $VER $VARIANT $(sha256sum "$F" | cut -d' ' -f1) $F"; exit 0; fi
    echo "NOT_PATCHED $VER want=$VARIANT has=$st $F"; exit 1 ;;
  apply)
    [ "$VER" = "$WANT" ] || { echo "WRONG_VERSION sglang $VER, patch is for $WANT"; exit 2; }
    if [ "$st" = "$VARIANT" ]; then echo "ALREADY_PATCHED $VER $VARIANT $(sha256sum "$F" | cut -d' ' -f1) $F"; exit 0; fi
    case $st in
      none) ;;
      fdatasync|sync_file_range)   # the other variant: reverse it first
        patch --reverse --dry-run -p2 -d "$SITE" < "$HERE/${PATCH[$st]}" >/dev/null || { echo "CANNOT_REVERSE $st $F"; exit 2; }
        patch --reverse -p2 -d "$SITE" < "$HERE/${PATCH[$st]}" >/dev/null ;;
      *) echo "PARTIAL $st $F"; exit 2 ;;
    esac
    patch --forward --dry-run -p2 -d "$SITE" < "$HERE/${PATCH[$VARIANT]}" >/dev/null || { echo "PATCH_DOES_NOT_APPLY $F"; exit 2; }
    patch --forward -p2 -d "$SITE" < "$HERE/${PATCH[$VARIANT]}" >/dev/null
    rm -f "$SITE"/sglang/srt/mem_cache/__pycache__/hicache_storage.*.pyc
    echo "PATCHED $VER $VARIANT $(sha256sum "$F" | cut -d' ' -f1) $F" ;;
  *) echo "usage: apply_patch.sh PYTHON apply|check [VARIANT]"; exit 2 ;;
esac
