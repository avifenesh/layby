"""The park cost rule: where to keep a session's KV for the idle period after a turn.

Inputs: the predictor's survival curve for the idle period (P(T > E2[j]), T = seconds to the session's
next request) and what the engine measures about itself (park.live.Live, park.live.EngineParams).
No fitted constants: every term is a probability from the curve times a cost the engine measures.

  g(a), c(a)     P(idle KV is still on the GPU / in the CPU tier a seconds after its last use), a
                 token-weighted Kaplan-Meier curve over recent idle periods;
  k_c, k_d, k_r  cost of restoring the session's KV from the CPU tier, from disk (via CPU), or by
                 recompute: its own time (with the link's mean queue wait) plus the time it delays
                 others: a transfer of x delays the about lam * (wait + x) transfers that arrive behind
                 it by x, and its N allocated GPU tokens slow admission by N / C for x, delaying each
                 waiting request as much; a recompute of x delays the requests in prefill and those
                 admitted while it runs by x;
  price          holding n more tokens on a tier that evicts e tokens/s moves every eviction n/e
                 seconds earlier: e * lam(n/e) seconds per second, lam(x) the measured loss per evicted
                 token from returns within x seconds of the eviction. A copy in an LRU tier is gone once
                 about C tokens newer than it came in, so it pays the price for at most C / e seconds.

Placement: the cheapest in expectation of none (CPU copy only), ins (also write to disk, keep the CPU
hint), park (write to disk, first out of the CPU tier) and drop (first out of the CPU tier, no disk
write). A first-out copy stays until the tier's next eviction. Requests wait Q (the mean admission
wait) before the engine looks up their KV, so the return at T costs (1 - g(T + Q)) times the restore
from wherever the KV is then: the CPU tier, disk (ins, park) or the disk copy the session already has
plus recompute (none, drop). eta (the CPU tier's predicted next use) is the median return; park and
drop give -1 (first out).
"""
import math
import numpy as np

from .live import AG

E2 = np.array([0.5, 1, 2, 3, 5, 8, 12, 20, 30, 45, 60, 120, 300, 600, 1800])
TG = np.geomspace(0.05, 3600, 160)
OPTIONS = ("none", "ins", "park", "drop")


def interp_surv(P, t, H=E2):
    """P(T > t) from P at horizons H (one row), log-time interpolation, P(T > 0) = 1."""
    t = np.atleast_1d(np.asarray(t, float))
    grid = np.concatenate([[1e-3], H]); Pg = np.concatenate([[1.0], P])
    lt = np.log(np.clip(t, 1e-3, H[-1])); lg = np.log(grid)
    k = np.clip(np.searchsorted(lg, lt, side="right") - 1, 0, len(grid) - 2)
    w = (lt - lg[k]) / (lg[k + 1] - lg[k])
    return np.exp((1 - w) * np.log(np.maximum(Pg[k], 1e-9)) + w * np.log(np.maximum(Pg[k + 1], 1e-9)))


def decide(S, N, on_disk, idle_g, idle_c, live, p, now, options=OPTIONS):
    """S: P(T > TG) for the idle period (interp_surv of the curve on TG). N: the session's KV tokens.
    on_disk: tokens of it already on disk. idle_g, idle_c: (age, tokens) of the idle periods still open
    on the GPU and in the CPU tier. Returns dict(opt, eta, disk, first_out, tot)."""
    Sg = np.concatenate([[1.0], S])
    m = Sg[:-1] - Sg[1:]                                            # T as atoms between grid points
    tm = np.sqrt(np.concatenate([[TG[0] / 2], TG[:-1]]) * TG)
    m_inf = S[-1]                                                   # not back within the horizon
    HZ = TG[-1]
    b = N * p.bytes_per_token
    npf = live.rate("pf", now)
    na = live.rate("n_adm", now)
    Q = live.rate("qwait", now) / na if na > 0 else 0.0             # mean wait for admission
    tc_, td_ = b / (p.cpu_gbps * 1e9), b / (p.disk_gbps * 1e9)
    # per link: transfers per second and their mean queue wait; a transfer of x seconds delays the
    # ones that arrive while it waits and runs, about lam * (wait + x) of them, by x each
    lnk = {}
    for k in ("cpu", "disk"):
        lam = live.rate("n_" + k, now)
        lnk[k] = (lam, live.rate("wait_" + k, now) / lam if lam > 0 else 0.0)
    dly = lambda k, x: lnk[k][0] * (lnk[k][1] + x) * x
    rec = lambda n: n * p.f0 * (1 + p.att * n / 2000)
    # a load holds the request's N allocated GPU tokens while it runs: admission, bound by GPU
    # capacity C, slows by N / C for that long, which delays each waiting request as much
    blk = live.rate("nw", now) * N / p.gpu_tokens
    # a restore from disk: where the engine measures it, the time a request whose prefix is on disk waits
    # from its first lookup to admission (an engine may defer such a request for whole scheduler steps
    # while the disk tier promotes it); else the disk link's wait and transfer
    nd = live.rate("n_defer", now)
    defer = live.rate("defer", now) / nd if nd > 0 else None
    disk_t = (lambda x: defer) if defer is not None else (lambda x: lnk["disk"][1] + x)
    own = dict(c=lnk["cpu"][1] + tc_ + p.t0, d=disk_t(td_) + tc_ + p.t0, r=rec(N) + p.t0)
    # a recompute of x seconds takes the prefill budget from the requests in prefill and from those
    # that arrive (na per second) while it runs: about npf + na * x requests, each delayed by x
    prf = lambda x: (npf + na * x) * x
    ext = dict(c=dly("cpu", tc_) + blk * own["c"], d=dly("disk", td_) + dly("cpu", tc_) + blk * own["d"], r=prf(own["r"]))
    # without a new disk write, a miss past the CPU tier loads the disk copy the session already has
    # (the engine takes any external prefix) and recomputes the rest
    d0 = min(on_disk, N)
    if d0 > 0:
        f0 = d0 / N
        own["o"] = disk_t(f0 * td_) + f0 * tc_ + rec(N - d0) + p.t0
        ext["o"] = dly("disk", f0 * td_) + dly("cpu", f0 * tc_) + blk * (own["o"] - rec(N - d0)) + prf(rec(N - d0))
    else:
        own["o"], ext["o"] = own["r"], ext["r"]
    Rg = live.resid("g", now, idle_g); Rc = live.resid("c", now, idle_c)
    cumc = np.concatenate([[0.0], np.cumsum(np.diff(AG) * (Rc[1:] + Rc[:-1]) / 2)])
    g = lambda a: np.interp(a, AG, Rg)
    Gc = lambda a: np.interp(a, AG, cumc) + np.maximum(a - AG[-1], 0) * Rc[-1]
    ec = live.rate("w_c", now)
    price_c = ec * live.lam("c", N / ec, now) if ec > 0 else 0.0     # seconds lost per second held
    W = 0.0 if on_disk >= N else dly("disk", td_)                    # a disk write delays the link's other users
    ne = live.rate("ne_c", now)        # a first-out copy stays in the CPU tier until the tier's next eviction
    # in an LRU tier a copy is evicted once about C tokens newer than it came in, so while it sits there
    # the tier evicts at most about C tokens: at e tokens/s that bounds the time it costs price_c to C / e
    hc = p.cpu_tokens / ec if ec > 0 else math.inf

    def restore(a, opt, kind):
        """Expected own time or delay to others (kind) of bringing back KV idle for a seconds and not
        on the GPU: from the CPU tier while it is there, else disk (ins, park) or what the session
        already has on disk plus recompute (none, drop)."""
        k = own if kind == "own" else ext
        miss = k["o"] if opt in ("none", "drop") else k["d"]
        ca = np.exp(-ne * np.asarray(a, float)) if opt in ("park", "drop") else np.interp(a, AG, Rc)
        return ca * k["c"] + (1 - ca) * miss

    best, tot = None, {}
    Ta = tm + Q                                                     # the return is admitted (KV checked) at T + Q
    for opt in options:
        cn = (1 - g(Ta)) * (restore(Ta, opt, "own") + restore(Ta, opt, "ext"))
        base = float((m * cn).sum())
        first = opt in ("park", "drop")
        mem = 0.0 if first else price_c * float((m * np.minimum(Gc(Ta), hc)).sum() + m_inf * min(Gc(HZ), hc))
        place = mem + (W if opt in ("ins", "park") else 0.0)
        tot[opt] = (base, place)
        if best is None or base + place < best[0]:
            best = (base + place, opt)
    opt = best[1]
    first = opt in ("park", "drop")
    # the CPU tier evicts the copy predicted back last; rank by the median return
    eta = -1.0 if first else (float(TG[np.argmax(S <= 0.5)]) if (S <= 0.5).any() else 1800.0)
    return dict(opt=opt, eta=eta, disk=opt in ("ins", "park"), first_out=first, tot=tot,
                Q=Q, price_c=price_c, wait_disk=lnk["disk"][1], npf=npf)
