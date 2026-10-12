"""PYTHON apply_patch.py apply|check: vLLM PR #60045 (pr60045_swa_admission.patch, the sliding-window admission count)
on the vLLM installed in PYTHON's environment. Every vLLM arm runs with it, stock baselines and Layby alike. Written
for vLLM 0.31.0; apply refuses any other version.
  apply: patch the installed single_type_kv_cache_manager.py unless it already carries the change; prints PATCHED or
         ALREADY_PATCHED, the version, the file's sha256 and its path.
  check: exit 0 and print PATCHED with the sha256 if every hunk of the change is present, else exit 1.
Exit codes: 0 ok, 1 not patched (check), 2 wrong version, missing file or a file the patch does not apply to.

The hunks are applied in Python (exact text, each hunk's old side must occur once), so the engine's environment needs
no patch(1). Does not import vllm (importing it pulls in torch and CUDA); find_spec only locates the package.

method(patched): for CPU tests, SingleTypeKVCacheManager.get_num_blocks_to_allocate with or without the change, compiled
from the installed file (the patch applied or reversed in memory), so a test can swap it in without touching
site-packages.
"""
import hashlib
import importlib.metadata
import importlib.util
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PATCH = os.path.join(HERE, "pr60045_swa_admission.patch")
WANT = os.environ.get("VLLM_VERSION_WANT", "0.31.0")
NAME = "pr60045"


def parse(path=PATCH):
    """(target path relative to the site dir, [(old text, new text) per hunk]) of a one-file unified diff."""
    target, hunks, cur = None, [], None
    with open(path) as f:
        lines = f.read().split("\n")
    for ln in lines:
        if ln.startswith("+++ "):
            target = ln[4:].split("\t")[0][2:]   # strip b/
        elif ln.startswith("@@"):
            cur = ([], [])
            hunks.append(cur)
        elif cur is not None and ln[:1] in (" ", "-", "+"):
            if ln[0] != "+":
                cur[0].append(ln[1:])
            if ln[0] != "-":
                cur[1].append(ln[1:])
        elif cur is not None and ln == "":
            cur = None   # end of the diff
    return target, [("\n".join(o) + "\n", "\n".join(n) + "\n") for o, n in hunks]


def state(text, hunks):
    """none (every hunk's old side present once), patched (every new side present), or partial."""
    if all(new in text for _, new in hunks):
        return "patched"
    if all(text.count(old) == 1 for old, _ in hunks):
        return "none"
    return "partial"


def apply_text(text, hunks):
    for old, new in hunks:
        if text.count(old) != 1:
            raise ValueError("hunk does not apply")
        text = text.replace(old, new)
    return text


def installed_file(target):
    s = importlib.util.find_spec("vllm")
    return os.path.join(os.path.dirname(os.path.dirname(s.origin)), target)


def method(patched=True):
    target, hunks = parse()
    path = installed_file(target)
    with open(path) as f:
        text = f.read()
    st = state(text, hunks)
    if patched and st == "none":
        text = apply_text(text, hunks)
    elif not patched and st == "patched":
        text = apply_text(text, [(new, old) for old, new in hunks])
    elif st == "partial":
        raise ValueError(f"{path}: neither stock nor patched")
    ns = {"__name__": "vllm.v1.core._pr60045_single_type_kv_cache_manager"}
    exec(compile(text, path, "exec"), ns)
    return ns["SingleTypeKVCacheManager"].get_num_blocks_to_allocate


def main(mode):
    target, hunks = parse()
    try:
        ver = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        ver = "unknown"
    f = installed_file(target)
    if not os.path.isfile(f):
        print(f"NO_FILE {f}")
        return 2
    with open(f) as fh:
        text = fh.read()
    st = state(text, hunks)
    sha = lambda t: hashlib.sha256(t.encode()).hexdigest()
    if mode == "check":
        if st == "patched":
            print(f"PATCHED {ver} {NAME} {sha(text)} {f}")
            return 0
        print(f"NOT_PATCHED {ver} has={st} {f}")
        return 1
    if mode != "apply":
        print("usage: apply_patch.py apply|check")
        return 2
    if ver != WANT:
        print(f"WRONG_VERSION vllm {ver}, patch is for {WANT}")
        return 2
    if st == "patched":
        print(f"ALREADY_PATCHED {ver} {NAME} {sha(text)} {f}")
        return 0
    if st != "none":
        print(f"PATCH_DOES_NOT_APPLY {st} {f}")
        return 2
    text = apply_text(text, hunks)
    with open(f + ".layby-tmp", "w") as fh:
        fh.write(text)
    os.replace(f + ".layby-tmp", f)
    cache = os.path.join(os.path.dirname(f), "__pycache__")
    stem = os.path.basename(f)[:-3] + "."
    for p in os.listdir(cache) if os.path.isdir(cache) else []:
        if p.startswith(stem) and p.endswith(".pyc"):
            os.remove(os.path.join(cache, p))
    print(f"PATCHED {ver} {NAME} {sha(text)} {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
