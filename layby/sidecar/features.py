"""Tool-call features, shared with the training repo's offline extractor (copied verbatim so
the sidecar computes exactly what Laya was trained on)."""
import json
import os
import re

SKIP = {"cd", "pushd", "popd", "export", "source", ".", "set", "unset", "alias", "true", "ulimit", "umask",
        "trap", "local", "declare", "eval", "exec", "time", "builtin", "command", "then", "do", "else", "fi"}
PREFIX = {"sudo", "nice", "ionice", "timeout", "env", "nohup", "stdbuf", "taskset", "systemd-run"}


def program(cmd):
    """First real program of a shell command: skips cd/export/source segments and wrapper prefixes."""
    for seg in re.split(r"&&|\|\||;|\||\n|\(|\)|`", cmd):
        toks = [t for t in seg.strip().split()]
        i = 0
        for i, t in enumerate(toks):
            wrapper = t in PREFIX or t.startswith("-") or t.lstrip("-").isdigit() or re.fullmatch(r"\d+[smh]", t)
            if not wrapper and not ("=" in t and not t.startswith(("'", '"'))):
                break
        else:
            continue
        t = toks[i].strip("'\"")
        if t and t not in SKIP and not t.startswith(("#", "{", "}", "$")):
            return os.path.basename(t)
    return None


def arg_features(name, args):
    a = args if isinstance(args, str) else json.dumps(args or {})
    feat = {"arg_len": len(a)}
    try:
        d = json.loads(a) if isinstance(a, str) else a
    except ValueError:
        d = {}
    cmd = None
    if isinstance(d, dict):
        cmd = d.get("command") or d.get("cmd")
        if isinstance(cmd, list):
            cmd = " ".join(map(str, cmd))
        feat["timeout"] = d.get("timeout") or d.get("timeout_ms")
        feat["background"] = bool(d.get("run_in_background"))
    if isinstance(cmd, str):
        feat["prog"] = program(cmd)
        feat["cmd_len"] = len(cmd)
    return feat
