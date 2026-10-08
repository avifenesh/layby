"""In-process sampling profiler for the engine thread (containers often forbid ptrace, so py-spy cannot attach).

start(tag) samples the calling thread's Python stack every PERIOD seconds from a daemon thread (it lives as long
as the process) and counts each function inclusively (anywhere on the stack) and exclusively (the leaf), over the
whole run and per WINDOW-second window. tag() (optional) describes the engine's load when a window closes (queue
length, step time, ...), so the slow windows can be picked out. report() gives the whole-run top entries;
windows(n) the last n windows with their tags and top exclusive entries.
"""
import collections
import sys
import threading
import time

PERIOD = 0.005
WINDOW = 60.0
KEEP = 240                      # windows kept (4 h at 60 s)
_incl, _excl = collections.Counter(), collections.Counter()
_n = [0]
_win = collections.deque(maxlen=KEEP)
_cur = dict(t0=time.time(), n=0, incl=collections.Counter(), excl=collections.Counter())


def _name(f):
    c = f.f_code
    return f"{c.co_filename.rsplit('/', 1)[-1]}:{c.co_firstlineno} {c.co_name}"


def _sample(ident):
    f = sys._current_frames().get(ident)
    if f is None:
        return
    _n[0] += 1
    _cur["n"] += 1
    leaf = _name(f)
    _excl[leaf] += 1
    _cur["excl"][leaf] += 1
    seen = set()
    while f is not None:
        k = _name(f)
        if k not in seen:
            seen.add(k)
            _incl[k] += 1
            _cur["incl"][k] += 1
        f = f.f_back


def _close(tag):
    try:
        t = tag() if tag else {}
    except Exception as e:      # noqa: BLE001  (a racing read of engine state)
        t = {"error": repr(e)}
    n = max(_cur["n"], 1)
    _win.append(dict(t=round(_cur["t0"], 1), samples=_cur["n"], tag=t,
                     excl=[(k, round(v / n, 3)) for k, v in _cur["excl"].most_common(12)],
                     incl=[(k, round(v / n, 3)) for k, v in _cur["incl"].most_common(25)]))
    _cur.update(t0=time.time(), n=0, incl=collections.Counter(), excl=collections.Counter())


def _run(ident, main, tag):
    while main.is_alive():
        time.sleep(PERIOD)
        _sample(ident)
        if time.time() - _cur["t0"] >= WINDOW:
            _close(tag)


def start(tag=None):
    main = threading.current_thread()
    threading.Thread(target=_run, args=(main.ident, main, tag), name="park-prof", daemon=True).start()


def report(top=40):
    n = max(_n[0], 1)
    return dict(samples=_n[0], inclusive=[(k, round(v / n, 3)) for k, v in _incl.most_common(top)],
                exclusive=[(k, round(v / n, 3)) for k, v in _excl.most_common(top)])


def windows(last=60):
    return list(_win)[-last:]
