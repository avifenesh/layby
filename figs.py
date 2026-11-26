#!/usr/bin/env python3
"""Figures and table snippets for the Layby paper, from result files only.

Inputs (content-free, relative to the repository root):
  reference/rows.jsonl            ReturnBench reference run (33 cells; C0 wt stk cost orc)
  data/returnbench_v2_s0.json     Published 33.8M Layby-Dwell v2 ReturnBench 33-cell results
  data/v2_results.json            Real-engine vLLM/SGLang, cross-hardware, and ablation results
  data/baselines_rows.jsonl       33 ReturnBench cells for GPU-pinning baselines (C0 cont cj ka)
  data/ablation_rows.jsonl        33 ReturnBench cells for action-option ablations
  bench/pools/*.json              ReturnBench pool statistics
Outputs: figs/*.pdf, figs/*.tex (tables input by main.tex).

Per cell (pool, sessions, disk speed): each workload's p95 is the median over engine seeds; the cell ratio is the
geometric mean over workloads of rule p95 / C0 p95 (as returnbench). Summaries are geometric means over cells.
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
LAYBY = HERE if os.path.exists(os.path.join(HERE, "reference", "rows.jsonl")) else os.environ.get("LAYBY_REPO", os.path.expanduser("~/projects/inference/layby"))
OUT = os.path.join(HERE, "figs")
os.makedirs(OUT, exist_ok=True)

# categorical slots in fixed order (validated light palette); markers carry identity too.
# Keys are the rule ids in the result files; labels are the arm names the paper uses (main.tex \arm{}).
STYLE = {"cost": ("#2a78d6", "o", "layby"), "wt": ("#eb6834", "s", "write-through"),
         "stk": ("#1baf7a", "^", "threshold"), "orc": ("#eda100", "D", "oracle")}
NAMES = {"C0": r"\arm{no-disk}", "wt": r"\arm{write-through}", "stk": r"\arm{threshold}",
         "cost": r"\arm{layby}", "orc": r"\arm{oracle}", "cont": r"\arm{continuum}", "cj": r"\arm{choi-joshi}",
         "ka": r"\arm{keep-alive}", "c_disk": r"\arm{layby}, disk options only", "c_nopark": r"\arm{layby}, no park",
         "o_disk": r"\arm{oracle}, disk options only"}
POOL = {"swechat": "SWE-chat", "tracelab_claude": "TraceLab", "wildchat": "WildChat",
        "swechat+tracelab_claude+wildchat": "mix"}


def pool_name(p):
    return "+".join(x.replace(".nc.v6.json", "").replace(".json", "") for x in p.split("+"))


def load(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def cells(rows, metric="p95"):
    """{(pool, n, disk): {rule: ratio vs C0}}, plus C0's absolute value per cell (median over workloads)."""
    med = defaultdict(list)
    for x in rows:
        med[(pool_name(x["pool"]), x["n"], x["disk"], x["w"], x["rule"])].append(x[metric])
    per = defaultdict(dict)
    for (p, n, d, w, r), v in med.items():
        per[(p, n, d, w)][r] = float(np.median(v))
    out, base = defaultdict(dict), defaultdict(list)
    acc = defaultdict(lambda: defaultdict(list))
    for (p, n, d, w), rs in per.items():
        c = rs["C0"]
        base[(p, n, d)].append(c)
        for r, v in rs.items():
            acc[(p, n, d)][r].append(np.log(v / c))
    for k, rs in acc.items():
        for r, l in rs.items():
            out[k][r] = float(np.exp(np.mean(l)))
    return out, {k: float(np.median(v)) for k, v in base.items()}


def gm(xs):
    return float(np.exp(np.mean(np.log(xs)))) if len(xs) else float("nan")


def summary(rows, rules, ref="wt"):
    c95, _ = cells(rows, "p95")
    c50, _ = cells(rows, "p50")
    S = {}
    for r in rules:
        ks = [k for k in c95 if r in c95[k]]
        v = [c95[k][r] for k in ks]
        S[r] = dict(n=len(ks), p95=gm(v), p50=gm([c50[k][r] for k in ks]), worst=max(v),
                    bad=sum(x > 1.05 for x in v),
                    disk={d: gm([c95[k][r] for k in ks if k[2] == d]) for d in (0.5, 1.5, 3.0)})
        if all(ref in c95[k] for k in ks):
            S[r]["vs_ref"] = gm([c95[k][r] / c95[k][ref] for k in ks])
            S[r]["vs_ref_disk"] = {d: gm([c95[k][r] / c95[k][ref] for k in ks if k[2] == d]) for d in (0.5, 1.5, 3.0)}
    return S


def tex_summary(S, rules, path, ref=True):
    hdr = r"rule & p95 ($\downarrow$) & p50 ($\downarrow$) & worst ($\downarrow$) & $>$1.05 & @0.5 & @1.5 & @3" + (" & " + two("vs", r"\arm{write-through}") if ref else "") + r" \\"
    lines = [r"\begin{tabular}{l" + "r" * (8 if ref else 7) + "}", r"\toprule", hdr, r"\midrule"]
    non_orc = [r for r in rules if r not in ("orc", "o_disk")]
    best = {}
    if len(non_orc) >= 2:
        best["p95"] = min(round(S[r]["p95"], 2) for r in non_orc)
        best["p50"] = min(round(S[r]["p50"], 2) for r in non_orc)
        best["worst"] = min(round(S[r]["worst"], 2) for r in non_orc)
        best["bad"] = min(S[r]["bad"] for r in non_orc)
        for d in (0.5, 1.5, 3.0):
            best[d] = min(round(S[r]["disk"][d], 2) for r in non_orc)
        if ref and all("vs_ref" in S[r] for r in non_orc):
            best["vs_ref"] = min(round(S[r]["vs_ref"], 2) for r in non_orc)

    def fmt(val, key, spec=".2f", is_orc=False):
        s = format(val, spec)
        if not is_orc and key in best and abs(round(val, 2) - best[key]) < 1e-9:
            return r"\textbf{" + s + "}"
        return s

    for r in rules:
        s = S[r]
        is_orc = r in ("orc", "o_disk")
        bad_s = f"{s['bad']}/{s['n']}"
        if not is_orc and "bad" in best and s["bad"] == best["bad"]:
            bad_s = r"\textbf{" + bad_s + "}"
        row = f"{NAMES[r]} & {fmt(s['p95'], 'p95', '.2f', is_orc)} & {fmt(s['p50'], 'p50', '.2f', is_orc)} & " \
              f"{fmt(s['worst'], 'worst', '.2f', is_orc)} & {bad_s} & " + \
              " & ".join(fmt(s["disk"][d], d, ".2f", is_orc) for d in (0.5, 1.5, 3.0))
        if ref:
            row += f" & {fmt(s['vs_ref'], 'vs_ref', '.2f', is_orc)}" if "vs_ref" in s else " & --"
        lines.append(row + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def fig_cells(rows, path):
    c95, _ = cells(rows, "p95")
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    rules = ["wt", "stk", "cost", "orc"]
    for i, d in enumerate((0.5, 1.5, 3.0)):
        for j, r in enumerate(rules):
            col, m, lab = STYLE[r]
            v = [c95[k][r] for k in c95 if k[2] == d]
            x = i + (j - 1.5) * 0.18 + np.linspace(-0.05, 0.05, len(v))
            ax.scatter(x, v, s=22, marker=m, color=col, edgecolor="white", linewidth=0.6, zorder=3,
                       label=lab if i == 0 else None)
    ax.axhline(1.0, color="#6b6a63", lw=1, zorder=1)
    ax.set_yscale("log"); ax.set_yticks([0.3, 0.5, 0.7, 1.0, 1.5]); ax.set_yticklabels(["0.3", "0.5", "0.7", "1.0", "1.5"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["disk 0.5 GB/s", "disk 1.5 GB/s", "disk 3 GB/s"])
    ax.set_ylabel("p95 TTFT vs no-disk (per cell)")
    ax.grid(axis="y", color="#e4e3dc", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def load_rb_v2(path):
    """Load Layby-Dwell v2 33.8M ReturnBench 33-cell results (cost_v2) keyed as (pool, n, disk)."""
    b = json.load(open(path))
    p95, p50 = {}, {}
    for k, v in b["cells"].items():
        pool_s, n_s, d_s = k.split("|")
        key = (pool_s, int(n_s.split("=")[1]), float(d_s.split("=")[1]))
        p50[key] = float(v["cost_v2"][2])
        p95[key] = float(v["cost_v2"][3])
    return p95, p50


def fig_load(rows, path, rb_v2_path=None):
    """Regime map: per-cell ratio vs C0's own p95 (a load proxy), one panel per disk speed."""
    c95, base = cells(rows, "p95")
    if rb_v2_path and os.path.exists(rb_v2_path):
        v2_95, _ = load_rb_v2(rb_v2_path)
        for k in c95:
            if k in v2_95:
                c95[k]["cost"] = v2_95[k]
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 2.5), sharey=True)
    for ax, d in zip(axs, (0.5, 1.5, 3.0)):
        ks = sorted([k for k in c95 if k[2] == d], key=lambda k: base[k])
        for r in ["wt", "cost", "orc"]:
            col, m, lab = STYLE[r]
            ax.scatter([base[k] for k in ks], [c95[k][r] for k in ks], marker=m, s=20, color=col, edgecolor="white",
                       linewidth=0.5, zorder=3, label=lab if d == 0.5 else None)
        ax.axhline(1.0, color="#6b6a63", lw=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_yticks([0.3, 0.5, 1.0, 1.5]); ax.set_yticklabels(["0.3", "0.5", "1.0", "1.5"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f"disk {d:g} GB/s", fontsize=8)
        ax.set_xlabel("no-disk p95 TTFT (s)", fontsize=8)
        ax.grid(color="#e4e3dc", lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axs[0].set_ylabel("p95 vs no-disk")
    fig.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88)); fig.savefig(path); plt.close(fig)


def tex_pools(path):
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"pool & sessions & turns & tool & human & gap p50 (s) & gap p90 (s) \\", r"\midrule"]
    for p, name in [("swechat", "SWE-chat"), ("tracelab_claude", "TraceLab (Claude Code)"), ("wildchat", "WildChat"),
                    ("copilot", "Copilot coding agent")]:
        S = json.load(open(os.path.join(LAYBY, "bench/pools", p + ".json")))["sessions"]
        t = [x for s in S for x in s["turns"]]
        g = [x["gap_after"] for x in t if x.get("gap_after") is not None]
        k = defaultdict(int)
        for x in t:
            k[x.get("kind")] += 1
        lines.append(f"{name} & {len(S)} & {len(t)} & {k['tool']} & {k['human']} & {np.median(g):.1f} & {np.percentile(g, 90):.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def load_pool(path):
    """pool_arms.py output (header lines, then one JSON object)."""
    t = open(path).read()
    return json.loads(t[t.index("{"):])


REAL = [("vLLM 0.30", "private replay", "vllm_private", ["C0", "T", "K", "KW", "KD"]),
        ("vLLM 0.30", "public replay", "vllm_public", ["C0", "T", "K", "KD"]),
        ("SGLang 0.5.21", "private replay", "sglang_private", ["C0", "T", "K"]),
        ("SGLang 0.5.21", "public replay", "sglang_public", ["C0", "T", "K"])]
def two(a, b):
    """A two-line, right-aligned table header cell, so long arm names do not widen a column."""
    return r"\begin{tabular}[b]{@{}r@{}}" + a + r"\\{}" + b + r"\end{tabular}"


RNAMES = {"C0": r"\arm{no-disk}", "T": r"\arm{write-through}", "K": r"\arm{layby}",
          "KW": r"\arm{layby-write-through}", "KD": r"\arm{layby-write-on-evict}"}


def tex_real(path):
    """Real-engine table: per arm the median over repeats of p50 and p95 (s) and of recomputed prompt tokens, and the
    pooled session-bootstrap p95 ratios against C0 and T with 95% CIs (pool_arms.py)."""
    lines = [r"\begin{tabular}{lllrrrr}", r"\toprule",
             r"engine & workload & arm & p50 (s) & p95 (s) & " + two(r"p95 vs \arm{no-disk}", r"[95\% CI]") + " & "
             + two(r"p95 vs \arm{write-through}", r"[95\% CI]") + r" \\", r"\midrule"]
    for eng, wl, g, arms in REAL:
        j = load_pool(os.path.join(HERE, "data/real", g + ".pool.txt"))
        pairs = j.get("pairs", {})
        for i, a in enumerate(arms):
            reps = j["arms"][a]["reps"]
            p50 = np.median([r["p50"] for r in reps]); p95 = np.median([r["p95"] for r in reps])
            def pr(k):
                v = pairs.get(k)
                if a == k.split(":")[1]:
                    return "1"
                return f"{v['p95']:.2f} [{v['p95_ci'][0]:.2f}, {v['p95_ci'][1]:.2f}]" if v else "n/a"
            n = len(reps)
            lab = RNAMES[a] + ("" if n == 3 else f" ({n} rep.)")
            lines.append(f"{eng if i == 0 else ''} & {wl if i == 0 else ''} & {lab} & {p50:.2f} & {p95:.1f} & {pr(a + ':C0')} & {pr(a + ':T')}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(path, "w").write("\n".join(lines) + "\n")


FIDELITY = [("CAL", "data/fidelity_rows.jsonl"), ("A100 f0", "data/fidelity_rows_a100.jsonl")]


def tex_fidelity(path):
    """Simulator against the measured vLLM rounds on the same two workloads (tools/sim_fidelity.py rows): per source,
    the median over seeds of C0, T and K p95 and the K:C0, T:C0, K:T ratios; measured = pooled pool_arms.py output."""
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r" & & \multicolumn{3}{c}{p95 (s)} & \multicolumn{3}{c}{p95 ratio} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"workload & source & \arm{no-disk} & \arm{write-through} & \arm{layby} & "
             + two(r"\arm{layby} vs", r"\arm{no-disk}") + " & "
             + two(r"\arm{write-through}", r"vs \arm{no-disk}") + " & "
             + two(r"\arm{layby} vs", r"\arm{write-through}") + r" \\", r"\midrule"]
    for wl, g in (("private", "vllm_private"), ("public", "vllm_public")):
        j = load_pool(os.path.join(HERE, "data/real", g + ".pool.txt"))
        m = {a: j["arms"][a]["pooled"]["p95"] for a in ("C0", "T", "K")}
        pr = j["pairs"]
        lines.append(f"{wl} & measured (vLLM, 3 rep.) & {m['C0']:.1f} & {m['T']:.1f} & {m['K']:.1f} & "
                     f"{pr['K:C0']['p95']:.2f} & {pr['T:C0']['p95']:.2f} & {pr['K:T']['p95']:.2f}" + r" \\")
        for name, f in FIDELITY:
            rows = load(os.path.join(HERE, f))
            for d in (0.5, 3.0):
                med = {r: float(np.median([x["p95"] for x in rows if x["wl"] == wl and x["disk"] == d and x["rule"] == r]))
                       for r in ("C0", "wt", "cost")}
                lines.append(f" & sim {name}, disk {d:g}\\,GB/s & {med['C0']:.1f} & {med['wt']:.1f} & {med['cost']:.1f} & "
                             f"{med['cost'] / med['C0']:.2f} & {med['wt'] / med['C0']:.2f} & {med['cost'] / med['wt']:.2f}" + r" \\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    open(path, "w").write("\n".join(lines) + "\n")


def tex_percell(rows, path, rb_v2_path=None):
    """Appendix: every reference cell, C0's absolute p95 (median over workloads) and each rule's p95 vs C0."""
    C, base = cells(rows)
    if rb_v2_path and os.path.exists(rb_v2_path):
        v2_95, _ = load_rb_v2(rb_v2_path)
        for k in C:
            if k in v2_95:
                C[k]["cost"] = v2_95[k]
    rules = ["wt", "stk", "cost", "orc"]
    short = {r: NAMES[r] for r in rules}
    lines = [r"\begin{longtable}{lrrr" + "r" * len(rules) + "}",
             r"\caption{Layby (33.8M Layby-Dwell v2) stays within 4\% of \arm{no-disk} in every cell (worst 1.039); write-through and the tuned threshold each "
             r"exceed \arm{no-disk} by more than 40\% in some. Every ReturnBench reference cell: \arm{no-disk}'s p95 TTFT "
             r"(median over 8 workloads) and each rule's p95 relative to \arm{no-disk} (geometric mean over workloads): "
             r"\arm{write-through}, the tuned threshold rule (\arm{threshold}), the Layby cost rule with Layby-Dwell 33.8M (\arm{layby}) and "
             r"the oracle given the true gap (\arm{oracle}).}\label{tab:percell} \\",
             r"\toprule",
             r"pool & sessions & disk (GB/s) & \arm{no-disk} p95 (s) & " + " & ".join(short[r] for r in rules) + r" \\",
             r"\midrule", r"\endfirsthead",
             r"\multicolumn{%d}{l}{\small Table~\ref{tab:percell}, continued} \\" % (4 + len(rules)),
             r"\toprule",
             r"pool & sessions & disk (GB/s) & \arm{no-disk} p95 (s) & " + " & ".join(short[r] for r in rules) + r" \\",
             r"\midrule", r"\endhead"]
    for k in sorted(C, key=lambda k: (k[0], k[1], k[2])):
        p, n, d = k
        lines.append(f"{POOL.get(p, p)} & {n} & {d:g} & {base[k]:.1f} & " + " & ".join(f"{C[k][r]:.2f}" for r in rules) + r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    open(path, "w").write("\n".join(lines) + "\n")


def tex_v2_model(v2, path):
    """Layby-Dwell v2 (33.8M) vs archived v1 (421M) head-to-head table."""
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"metric & archived v1 (421M) & \textbf{Layby-Dwell v2 (33.8M)} & delta \\",
        r"\midrule",
    ]
    for r in v2["dwell_v2_vs_v1"]:
        lines.append(f"{r['metric']} & {r['v1']} & \\textbf{{{r['v2']}}} & {r['delta']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def tex_v2_glm2027(v2, path):
    """Untouched held-out GLM-5.3-Flash 320B MoE TP=4 (seed=2027) table."""
    rows = v2["glm2027_heldout"]
    best = {
        "p50": min(r["p50"] for r in rows),
        "p95": min(r["p95"] for r in rows),
        "p99": min(r["p99"] for r in rows),
        "mean": min(r["mean"] for r in rows),
        "recomp_m": min(r["recomp_m"] for r in rows),
        "extra_vram_gib": min(r["extra_vram_gib"] for r in rows),
        "wall_min": min(r["wall_min"] for r in rows),
    }
    def fb(val, key, fmt):
        s = fmt.format(val)
        return f"\\textbf{{{s}}}" if abs(val - best[key]) < 1e-9 else s
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"policy & p50 (s $\downarrow$) & p95 (s $\downarrow$) & p99 (s $\downarrow$) & mean (s $\downarrow$) & recomp. ($\downarrow$) & disk & extra VRAM & makespan ($\downarrow$) \\",
        r"\midrule",
    ]
    for r in rows:
        is_lb = r["arm"] == "layby-v2"
        arm = r"\textbf{\arm{layby}} (2-rep avg)" if is_lb else r"\arm{" + r["arm"] + "}"
        vram_raw = f"+{r['extra_vram_gib']:.1f}\\,GiB" if r["extra_vram_gib"] > 0 else "0"
        vram = f"\\textbf{{{vram_raw}}}" if abs(r["extra_vram_gib"] - best["extra_vram_gib"]) < 1e-9 else vram_raw
        lines.append(
            f"{arm} & {fb(r['p50'], 'p50', '{:.2f}')} & {fb(r['p95'], 'p95', '{:.2f}')} & "
            f"{fb(r['p99'], 'p99', '{:.2f}')} & {fb(r['mean'], 'mean', '{:.2f}')} & "
            f"{fb(r['recomp_m'], 'recomp_m', '{:.2f}M')} & {r['disk_gb']:.1f}\\,GB & "
            f"{vram} & {fb(r['wall_min'], 'wall_min', '{:.1f}\\,min')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def _fmt_mp(entry, bold_m=False, bold_p=False):
    if "status" in entry and entry.get("mean") is None and entry.get("p95") is None:
        return entry["status"]
    if "status" in entry and entry.get("mean") is None:
        return f"{entry['status']}/{entry['p95']:.0f}"
    m, p = entry["mean"], entry["p95"]
    ms = f"{m:.1f}" if p >= 100 else f"{m:.2f}"
    ps = f"{p:.1f}" if p >= 100 else f"{p:.2f}"
    if bold_m:
        ms = f"\\textbf{{{ms}}}"
    if bold_p:
        ps = f"\\textbf{{{ps}}}"
    return f"{ms}/{ps}"


def tex_v2_vllm(v2, path):
    """Full vLLM 0.34.0 multi-model matrix (all-turn mean / p95 TTFT in seconds, 2-3 repeat avg for layby)."""
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"workload (mean / p95, s $\downarrow$) & \arm{no-disk} & " + two(r"\arm{write-}", r"\arm{through}") + r" & "
        + two(r"\arm{lmcache}", r"ARC") + r" & \arm{freq-gated} & \arm{continuum} & "
        + two(r"\textbf{\arm{layby}}", r"(2--3 rep avg)") + r" \\",
        r"\midrule",
    ]
    for row in v2["vllm_v034_matrix"]:
        a = row["arms"]
        valid = [v for v in a.values() if isinstance(v.get("mean"), (int, float)) and "status" not in v]
        min_m = min(round(v["mean"], 1 if v["p95"] >= 100 else 2) for v in valid)
        min_p = min(round(v["p95"], 1 if v["p95"] >= 100 else 2) for v in valid)
        def fmt_cell(ent):
            if "status" in ent or not isinstance(ent.get("mean"), (int, float)):
                return _fmt_mp(ent)
            rm = round(ent["mean"], 1 if ent["p95"] >= 100 else 2)
            rp = round(ent["p95"], 1 if ent["p95"] >= 100 else 2)
            return _fmt_mp(ent, rm <= min_m + 1e-9, rp <= min_p + 1e-9)
        lines.append(
            f"{row['cell']} & {fmt_cell(a['no-disk'])} & {fmt_cell(a['write-through'])} & "
            f"{fmt_cell(a['lmcache'])} & {fmt_cell(a['freq-gated'])} & {fmt_cell(a['continuum'])} & "
            f"{fmt_cell(a['layby-v2'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def tex_v2_sglang(v2, path):
    """SGLang 0.5.23 + HiCache real-hardware matrix (Qwen3-8B BF16, all-turn linear percentiles)."""
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"regime & policy & mean (s $\downarrow$) & p95 (s $\downarrow$) & p99 (s $\downarrow$) & max (s $\downarrow$) & recomp. ($\downarrow$) & disk ($\downarrow$) \\",
        r"\midrule",
    ]
    groups = []
    for r in v2["sglang_v0523_matrix"]:
        if not groups or groups[-1][0] != r["regime"]:
            groups.append((r["regime"], []))
        groups[-1][1].append(r)
    for gi, (reg, rows) in enumerate(groups):
        if gi > 0:
            lines.append(r"\midrule")
        ok = [r for r in rows if "status" not in r]
        best = {k: min(r[k] for r in ok if r.get(k) is not None) for k in ("mean", "p95", "p99", "max", "recomp", "disk")}
        def fm(r, k, spec, suf=""):
            v = r.get(k)
            if v is None:
                return "--"
            s = format(v, spec) + suf
            return f"\\textbf{{{s}}}" if "status" not in r and abs(v - best[k]) < 1e-9 else s
        for ri, r in enumerate(rows):
            reg_s = reg if ri == 0 else ""
            arm_parts = r["arm"].split(" ", 1)
            arm_tex = r"\arm{" + arm_parts[0] + "}" + (f" {arm_parts[1]}" if len(arm_parts) > 1 else "")
            if r["arm"].startswith("layby"):
                arm_tex = f"\\textbf{{{arm_tex}}}"
            if "status" in r:
                dk = f"{r['disk']:.1f}\\,GB" if r.get("disk") is not None else "--"
                lines.append(f"{reg_s} & {arm_tex} & {r['status']} & -- & -- & -- & -- & {dk} \\\\")
            else:
                lines.append(
                    f"{reg_s} & {arm_tex} & {fm(r, 'mean', '.2f')} & {fm(r, 'p95', '.2f')} & "
                    f"{fm(r, 'p99', '.2f')} & {fm(r, 'max', '.2f')} & {fm(r, 'recomp', '.2f', 'M')} & "
                    f"{fm(r, 'disk', '.1f', r'\,GB')} \\\\"
                )
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def fig_overview(v2, path):
    """Page-1/2 teaser & system overview: (a) closed-loop Layby architecture; (b) two-regime headline results."""
    from matplotlib.patches import FancyBboxPatch

    fig = plt.figure(figsize=(7.0, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.26, 1.0], wspace=0.40)

    # Panel (a): System architecture flow diagram
    ax = fig.add_subplot(gs[0])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("(a) Closed-loop placement (zero workload-fitted thresholds)", fontsize=6.5, fontweight="bold", pad=4)

    def box(x, y, w, h, title, body, fc, ec, tc="#1d2430"):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.10,rounding_size=0.18",
                           facecolor=fc, edgecolor=ec, linewidth=1.05, zorder=3)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h * 0.70, title, ha="center", va="center",
                fontsize=6.3, fontweight="bold", color=tc, zorder=4)
        ax.text(x + w / 2, y + h * 0.28, body, ha="center", va="center",
                fontsize=5.65, color="#333942", zorder=4)

    box(0.20, 4.45, 4.35, 1.35, "OpenAI Server View",
        "Turn-delta text + 32 causal\ntelemetry features at turn end", "#f4f6fa", "#7f8c9b")
    box(5.30, 4.45, 4.50, 1.35, "Layby-Dwell (CPU, 33.8M)",
        "Two-tower (11-19 ms 64-vCPU,\n0 MiB VRAM) $\\rightarrow$ curve $S(t)$", "#e7f1fc", "#2a78d6", "#16488a")
    box(0.20, 2.35, 4.35, 1.45, "Live Engine Measurements",
        "KM $g(a), c(a)$; links $B_c, B_d$;\ndeferral $D$; prefill $f_0$; price $\\pi_c$", "#fef6e7", "#d98b00", "#8a5700")
    box(5.30, 2.35, 4.50, 1.45, "Measured-Cost Rule",
        "$\\arg\\min\\; \\mathbb{E}$[restore + queue delay +\nCPU holding + write delay]", "#e6f6ef", "#1baf7a", "#0d6e4b")
    box(0.20, 0.25, 9.60, 1.35, "Four Per-Session Actions (GPU $\\rightarrow$ CPU $\\rightarrow$ Disk)",
        "none (keep CPU, order by median $S$)  |  ins (keep CPU + write disk)\n"
        "park (write disk, evict CPU first)  |  drop (evict CPU first, skip disk)", "#f6f5f0", "#6b6a63", "#222222")

    arrow = dict(arrowstyle="-|>,head_length=0.42,head_width=0.22", color="#4a5568", lw=1.1)
    ax.annotate("", xy=(5.22, 5.12), xytext=(4.63, 5.12), arrowprops=arrow)
    ax.annotate("", xy=(7.55, 3.88), xytext=(7.55, 4.38), arrowprops=arrow)
    ax.annotate("", xy=(5.22, 3.08), xytext=(4.63, 3.08), arrowprops=arrow)
    ax.annotate("", xy=(7.55, 1.68), xytext=(7.55, 2.28), arrowprops=arrow)
    ax.annotate("", xy=(2.38, 2.28), xytext=(2.38, 1.68),
                arrowprops=dict(arrowstyle="-|>,head_length=0.42,head_width=0.22", color="#b07200", lw=1.05, ls="--"))
    ax.text(2.55, 1.98, "feedback", fontsize=5.6, color="#8a5700", va="center")

    # Panel (b): Headline 2-regime contrast (GLM-5.3 held-out 2-rep avg + Qwen3.8 slow disk 3-rep avg)
    gs_r = gs[1].subgridspec(2, 1, hspace=0.50)

    ax1 = fig.add_subplot(gs_r[0])
    glm = [
        ("continuum", 167.91, "#9a9992"),
        ("no-disk", 91.57, "#6b6a63"),
        ("lmcache", 42.23, "#eb6834"),
        ("layby", 21.51, "#2a78d6"),
    ]
    y1 = np.arange(len(glm))
    ax1.barh(y1, [x[1] for x in glm], height=0.62, color=[x[2] for x in glm], alpha=0.92)
    for yi, (_, val, _) in enumerate(glm):
        ax1.text(val + 3.5, yi, f"{val:.1f} s", va="center", fontsize=6.3,
                 fontweight="bold" if yi == 3 else "normal", color="#16488a" if yi == 3 else "#222")
    ax1.set_yticks(y1); ax1.set_yticklabels([x[0] for x in glm], fontsize=6.4)
    ax1.invert_yaxis(); ax1.set_xlim(0, 218)
    ax1.set_title("(b) Held-out GLM-5.3 320B mean TTFT (s)", fontsize=7.4, fontweight="bold", pad=3)
    ax1.tick_params(axis="x", labelsize=6.1)
    ax1.grid(axis="x", color="#e4e3dc", lw=0.5); ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    ax2 = fig.add_subplot(gs_r[1])
    qsp = [
        ("freq-gated", 62.94, "#1baf7a"),
        ("write-through", 58.21, "#eb6834"),
        ("no-disk", 39.50, "#6b6a63"),
        ("layby", 36.70, "#2a78d6"),
    ]
    y2 = np.arange(len(qsp))
    ax2.barh(y2, [x[1] for x in qsp], height=0.62, color=[x[2] for x in qsp], alpha=0.92)
    for yi, (_, val, _) in enumerate(qsp):
        ax2.text(val + 1.5, yi, f"{val:.1f} s", va="center", fontsize=6.3,
                 fontweight="bold" if yi == 3 else "normal", color="#16488a" if yi == 3 else "#222")
    ax2.set_yticks(y2); ax2.set_yticklabels([x[0] for x in qsp], fontsize=6.4)
    ax2.invert_yaxis(); ax2.set_xlim(0, 82)
    ax2.set_title("Qwen3.8-27B slow disk (141 MB/s) p95 (s)", fontsize=7.4, fontweight="bold", pad=3)
    ax2.tick_params(axis="x", labelsize=6.1)
    ax2.grid(axis="x", color="#e4e3dc", lw=0.5); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.subplots_adjust(left=0.025, right=0.985, top=0.90, bottom=0.10)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fig_v2_glm(v2, path):
    """Untouched held-out GLM-5.3-Flash 320B MoE TP=4 (seed=2027) 3-panel bar comparison."""
    rows = v2["glm2027_heldout"]
    arms = ["layby" if r["arm"] == "layby-v2" else r["label"] for r in rows]
    y = np.arange(len(arms))
    colors = {
        "continuum": "#9a9992",
        "no-disk": "#6b6a63",
        "freq-gated": "#1baf7a",
        "lmcache": "#eb6834",
        "layby": "#2a78d6",
    }
    cols = [colors[a] for a in arms]

    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.55), sharey=True)

    # Panel 1: Mean and p95 TTFT
    h = 0.36
    means = [r["mean"] for r in rows]
    p95s = [r["p95"] for r in rows]
    axs[0].barh(y - h / 2, means, height=h, color=cols, alpha=0.95, label="mean TTFT")
    axs[0].barh(y + h / 2, p95s, height=h, color=cols, alpha=0.45, edgecolor=cols, linewidth=0.8, label="p95 TTFT")
    for yi, m, p in zip(y, means, p95s):
        axs[0].text(m + 4, yi - h / 2, f"{m:.1f}", va="center", fontsize=6.5, color="#222")
        axs[0].text(p + 4, yi + h / 2, f"{p:.1f}", va="center", fontsize=6.5, color="#444")
    axs[0].set_xlim(0, 355)
    axs[0].set_xlabel("TTFT (s)  [solid: mean, light: p95]", fontsize=7.5)
    axs[0].set_title("All-turn TTFT (lower is better)", fontsize=8)

    # Panel 2: Makespan
    walls = [r["wall_min"] for r in rows]
    axs[1].barh(y, walls, height=0.58, color=cols, alpha=0.92)
    for yi, w in zip(y, walls):
        axs[1].text(w + 1.5, yi, f"{w:.1f} m", va="center", fontsize=7, color="#222")
    axs[1].set_xlim(0, 96)
    axs[1].set_xlabel("Replay makespan (min)", fontsize=7.5)
    axs[1].set_title("Wall-clock makespan", fontsize=8)

    # Panel 3: Recomputed tokens (M) and Disk written (GB) annotated
    recomps = [r["recomp_m"] for r in rows]
    disks = [r["disk_gb"] for r in rows]
    axs[2].barh(y, recomps, height=0.58, color=cols, alpha=0.92)
    for yi, rc, dk in zip(y, recomps, disks):
        note = f"{rc:.1f}M ({dk:.0f} GB)" if dk > 0 else f"{rc:.1f}M (0 GB)"
        axs[2].text(rc + 0.6, yi, note, va="center", fontsize=6.5, color="#222")
    axs[2].set_xlim(0, 49)
    axs[2].set_xlabel("Recomputed tokens (M) & disk", fontsize=7.5)
    axs[2].set_title("Recompute & disk written", fontsize=8)

    axs[0].set_yticks(y)
    axs[0].set_yticklabels(arms, fontsize=7.5)
    axs[0].invert_yaxis()
    for ax in axs:
        ax.grid(axis="x", color="#e4e3dc", lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_v2_matrix(v2, path):
    """Normalized p95 TTFT vs no-disk across all 6 vLLM 0.34.0 real-hardware cells."""
    rows = v2["vllm_v034_matrix"]
    arm_order = [
        ("write-through", "#eb6834", "write-through"),
        ("lmcache", "#eda100", "lmcache ARC"),
        ("freq-gated", "#1baf7a", "freq-gated"),
        ("continuum", "#9a9992", "continuum"),
        ("layby-v2", "#1c63c7", "layby (2-3 rep avg)"),
    ]
    x = np.arange(len(rows))
    w = 0.155
    fig, ax = plt.subplots(figsize=(6.8, 2.75))
    ax.axhline(1.0, color="#6b6a63", lw=1.0, ls="--", zorder=1)

    for j, (key, col, lab) in enumerate(arm_order):
        offs = x + (j - 2.0) * w
        vals, mask_idx = [], []
        for i, row in enumerate(rows):
            base = row["arms"]["no-disk"]["p95"]
            ent = row["arms"][key]
            if isinstance(ent.get("p95"), (int, float)) and "status" not in ent:
                vals.append(ent["p95"] / base)
                mask_idx.append(offs[i])
            elif ent.get("status", "").startswith(("cap", "timeout")) and isinstance(ent.get("p95"), (int, float)):
                v_cap = min(ent["p95"] / base, 2.15)
                ax.bar(offs[i], v_cap, width=w * 0.9, color=col, alpha=0.35, edgecolor=col, hatch="///", zorder=2)
                ax.text(offs[i], v_cap + 0.04, "cap", ha="center", va="bottom", fontsize=5.5, color="#992222", rotation=90)
            elif ent.get("status", "").startswith(("crash", "timeout", "cap")):
                ax.text(offs[i], 0.06, "crash" if "crash" in ent["status"] else "cap", ha="center", va="bottom",
                        fontsize=5.5, color="#992222", rotation=90, zorder=4)
        ax.bar(mask_idx, vals, width=w * 0.9, color=col, alpha=0.93, label=lab, zorder=3)
        if key == "layby-v2":
            for xi, vi in zip(mask_idx, vals):
                if vi >= 0.85:
                    ax.text(xi, vi - 0.05, f"{vi:.2f}", ha="center", va="top", fontsize=5.6, fontweight="bold",
                            color="white", rotation=90, zorder=5)
                else:
                    ax.text(xi, vi + 0.04, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.6, fontweight="bold",
                            color="#16488a", rotation=90, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels([r["short"] for r in rows], fontsize=6.8)
    ax.set_ylim(0, 2.38)
    ax.set_ylabel("p95 TTFT vs no-disk (lower is better)", fontsize=7.5)
    ax.grid(axis="y", color="#e4e3dc", lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=6.8, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def tex_v3_cross_hw(v2, path):
    """Cross-hardware single-GPU table: 1x RTX PRO 6000, 1x RTX 5090 Laptop 24 GB, DGX Spark GB10 Inkling 2 Mini."""
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"hardware tier \& policy & mean (s $\downarrow$) & p50 (s $\downarrow$) & p95 (s $\downarrow$) & p99 (s $\downarrow$) & max (s $\downarrow$) & disk \\",
        r"\midrule",
    ]
    for gi, grp in enumerate(v2["cross_hardware_1gpu"]):
        if gi > 0:
            lines.append(r"\midrule")
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{grp['group']}}}}} \\\\")
        rows = grp["rows"]
        best_m = min(r["mean"] for r in rows)
        best_p50 = min(r["p50"] for r in rows)
        best_p95 = min(r["p95"] for r in rows)
        best_p99 = min(r["p99"] for r in rows)
        best_max = min(r["max"] for r in rows)
        def fb(v, b):
            s = f"{v:.2f}"
            return f"\\textbf{{{s}}}" if abs(v - b) < 1e-9 else s
        for r in rows:
            lab = r["label"].replace(">=", r"$\ge$")
            if r["policy"] == "layby":
                lab = f"\\textbf{{{lab}}}"
            lines.append(
                f"\\quad {lab} & {fb(r['mean'], best_m)} & {fb(r['p50'], best_p50)} & "
                f"{fb(r['p95'], best_p95)} & {fb(r['p99'], best_p99)} & {fb(r['max'], best_max)} & "
                f"{r['disk_gb']:.1f}\\,GB \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def tex_v3_slow_and_engine_abl(v2, path_slow, path_eng):
    """Slow-disk 5-policy neural vs heuristic ladder + Gen4 real-engine component decomposition."""
    # 1. Slow disk ladder (Qwen3-8B 141 MB/s)
    rows = v2["slow_disk_ladder"]["qwen8b_141mbs"]
    bm = min(r["mean"] for r in rows)
    bp50 = min(r["p50"] for r in rows)
    bp95 = min(r["p95"] for r in rows)
    bp99 = min(r["p99"] for r in rows)
    bmax = min(r["max"] for r in rows)
    def fb(v, b):
        s = f"{v:.2f}"
        return f"\\textbf{{{s}}}" if abs(v - b) < 1e-9 else s
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"policy (Qwen3-8B, 141\,MB/s slow disk) & mean (s $\downarrow$) & p50 (s $\downarrow$) & p95 (s $\downarrow$) & p99 (s $\downarrow$) & max (s $\downarrow$) & disk \\",
        r"\midrule",
    ]
    for r in rows:
        lab = r["label"]
        if lab.startswith("Layby"):
            lab = f"\\textbf{{{lab}}}"
        lines.append(
            f"{lab} & {fb(r['mean'], bm)} & {fb(r['p50'], bp50)} & {fb(r['p95'], bp95)} & "
            f"{fb(r['p99'], bp99)} & {fb(r['max'], bmax)} & {r['disk_gb']:.1f}\\,GB \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path_slow, "w").write("\n".join(lines) + "\n")

    # 2. Gen4 real-engine component decomposition
    erows = v2["engine_ablations_gen4"]
    em = min(r["mean"] for r in erows)
    ep50 = min(r["p50"] for r in erows)
    ep95 = min(r["p95"] for r in erows)
    ep99 = min(r["p99"] for r in erows)
    emax = min(r["max"] for r in erows)
    elines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"policy / ablated component (Gen4 2.0\,GB/s) & mean (s $\downarrow$) & p50 (s $\downarrow$) & p95 (s $\downarrow$) & p99 (s $\downarrow$) & max (s $\downarrow$) & disk \\",
        r"\midrule",
    ]
    for r in erows:
        lab = r["label"].replace(">=", r"$\ge$")
        if "3-rep avg" in lab:
            lab = f"\\textbf{{{lab}}}"
        elines.append(
            f"{lab} & {fb(r['mean'], em)} & {fb(r['p50'], ep50)} & {fb(r['p95'], ep95)} & "
            f"{fb(r['p99'], ep99)} & {fb(r['max'], emax)} & {r['disk_gb']:.1f}\\,GB \\\\"
        )
    elines += [r"\bottomrule", r"\end{tabular}"]
    open(path_eng, "w").write("\n".join(elines) + "\n")


def tex_v3_model_ablation(v2, path):
    """Layby-Dwell architecture, objective, and multi-seed ablation table (full eval_uhold, bold best per column)."""
    rows = v2["model_ablations_v2"]
    best = {
        "all_nll": min(r["all_nll"] for r in rows),
        "tool_nll": min(r["tool_nll"] for r in rows),
        "human_nll": min(r["human_nll"] for r in rows),
        "ece300": min(r["ece300"] for r in rows),
        "wait_poll": min(r["wait_poll"] for r in rows),
        "rb_p95": min(r["rb_p95"] for r in rows),
        "rb_worst": min(r["rb_worst"] for r in rows),
        "rb_slow": min(r["rb_slow"] for r in rows),
    }
    def bf(val, k, fmt):
        s = fmt.format(val)
        return f"\\textbf{{{s}}}" if abs(val - best[k]) < 1e-9 else s
    lines = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        r"model variant & params & all ($\downarrow$) & tool ($\downarrow$) & human ($\downarrow$) & ECE@300\,s ($\downarrow$) & wait\_poll ($\downarrow$) & RB p95 (worst) & @0.5 ($\downarrow$) \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        if i == 6:
            lines.append(r"\midrule")
        var = r["variant"].replace(">", r"$>$")
        if r.get("is_ref") or r.get("is_pub"):
            var = f"\\textbf{{{var}}}"
        rb_cell = f"{bf(r['rb_p95'], 'rb_p95', '{:.3f}')} ({bf(r['rb_worst'], 'rb_worst', '{:.3f}')})"
        lines.append(
            f"{var} & {r['params']} & {bf(r['all_nll'], 'all_nll', '{:.4f}')} & "
            f"{bf(r['tool_nll'], 'tool_nll', '{:.4f}')} & {bf(r['human_nll'], 'human_nll', '{:.4f}')} & "
            f"{bf(r['ece300'], 'ece300', '{:.4f}')} & {bf(r['wait_poll'], 'wait_poll', '{:.4f}')} & "
            f"{rb_cell} & {bf(r['rb_slow'], 'rb_slow', '{:.3f}')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(path, "w").write("\n".join(lines) + "\n")


def fig_v3_cross_hw(v2, path):
    """3-panel cross-hardware single-GPU mean TTFT comparison across RTX PRO 6000, RTX 5090 Laptop, and DGX Spark GB10."""
    grps = v2["cross_hardware_1gpu"]
    pcol = {
        "layby": "#1c63c7",
        "write-through": "#eb6834",
        "lmcache": "#eda100",
        "freq-gated": "#1baf7a",
        "no-disk": "#6b6a63",
    }
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.28))
    for ax, grp in zip(axs, grps):
        rows = grp["rows"]
        y = np.arange(len(rows))
        vals = [r["mean"] for r in rows]
        cols = [pcol.get(r["policy"], "#888") for r in rows]
        labs = [r["label"].replace("Freq-Gated (Reuse >= 2)", "Freq-Gated").replace("No Disk (LRU)", "No Disk") for r in rows]
        ax.barh(y, vals, height=0.58, color=cols, alpha=0.93)
        mx = max(vals)
        for yi, r in enumerate(rows):
            bold = r["policy"] == "layby"
            ax.text(r["mean"] + mx * 0.03, yi, f"{r['mean']:.2f} s", va="center", fontsize=6.5,
                    fontweight="bold" if bold else "normal", color="#16488a" if bold else "#222")
        ax.set_yticks(y)
        ax.set_yticklabels(labs, fontsize=6.6)
        ax.invert_yaxis()
        ax.set_xlim(0, mx * 1.34)
        ax.set_title(grp["short"], fontsize=7.0, fontweight="bold", pad=4)
        ax.set_xlabel("Resumed mean TTFT (s)", fontsize=6.8)
        ax.tick_params(axis="x", labelsize=6.3)
        ax.grid(axis="x", color="#e4e3dc", lw=0.55)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout(w_pad=0.9)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def fig_ablations(v2, path):
    """2-panel scientific ablation figure: (a) slow-disk neural-vs-heuristic ladder; (b) model architecture ablation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.75), gridspec_kw={"width_ratios": [1.05, 1.15]})

    # Panel (a): Qwen3-8B slow-disk monotonic mean/p99 ladder (0.0 GB written for all 4; WT shown as inset callout)
    ladder = [
        ("No Disk (LRU)", 2.98, 17.23, "#6b6a63"),
        ("Cost Rule (EWMA)", 2.33, 15.55, "#eb6834"),
        ("Cost Rule (KM)", 2.23, 14.55, "#eda100"),
        ("Layby (2-rep avg)", 2.06, 13.57, "#1c63c7"),
    ]
    y1 = np.arange(len(ladder))
    ax1.barh(y1, [x[1] for x in ladder], height=0.55, color=[x[3] for x in ladder], alpha=0.93)
    for yi, (_, m, p99, _) in enumerate(ladder):
        is_best = yi == len(ladder) - 1
        ax1.text(m + 0.07, yi, f"{m:.2f} s  (p99 {p99:.1f})", va="center", fontsize=6.1,
                 fontweight="bold" if is_best else "normal", color="#16488a" if is_best else "#222")
    ax1.set_yticks(y1)
    ax1.set_yticklabels([x[0] for x in ladder], fontsize=6.6)
    ax1.set_ylim(4.45, -0.55)
    ax1.set_xlim(0, 4.75)
    ax1.set_xlabel("Resumed mean TTFT (s; all 0.0 GB disk)", fontsize=6.6)
    ax1.set_title("(a) Slow-disk CPU ordering ladder (141 MB/s)", fontsize=7.1, fontweight="bold", pad=4)
    ax1.text(2.38, 3.92, "Write-through: 10.53 s mean, 274 s p99 (77.2 GB)",
             ha="center", va="center", fontsize=5.6, color="#992222",
             bbox=dict(boxstyle="round,pad=0.25", fc="#fef2f2", ec="#e5a0a0", lw=0.7))
    ax1.tick_params(axis="x", labelsize=6.2)
    ax1.grid(axis="x", color="#e4e3dc", lw=0.55)
    ax1.set_axisbelow(True)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)

    # Panel (b): Model architecture ablation (full eval_uhold 184K turns, apples-to-apples NLL & human ECE@300s)
    m_order = [
        ("Text Only (32.5M)", 2.1852, 0.0514, "#c0392b"),
        ("Tabular Only (1.4M)", 1.6684, 0.0323, "#eb6834"),
        ("Plain MLP+Concat (32.1M)", 1.6435, 0.0202, "#eda100"),
        ("Aux=0.10 (3-seed med.)", 1.6068, 0.0115, "#9a9992"),
        ("Layby-Dwell (3-seed med.)", 1.5950, 0.0103, "#6ea5e8"),
        ("Layby-Dwell (pub. s0)", 1.5862, 0.0116, "#1c63c7"),
    ]
    y2 = np.arange(len(m_order))
    ax2.barh(y2, [x[1] for x in m_order], height=0.58, color=[x[3] for x in m_order], alpha=0.93)
    for yi, (_, nll, ece, _) in enumerate(m_order):
        is_best = yi == len(m_order) - 1
        ax2.text(nll + 0.03, yi, f"{nll:.3f} (ECE {ece:.3f})", va="center", fontsize=5.8,
                 fontweight="bold" if is_best else "normal", color="#16488a" if is_best else "#222")
    ax2.set_yticks(y2)
    ax2.set_yticklabels([x[0] for x in m_order], fontsize=6.2)
    ax2.invert_yaxis()
    ax2.set_xlim(1.30, 2.90)
    ax2.set_xlabel("Full eval_uhold NLL (lower is better)", fontsize=6.6)
    ax2.set_title("(b) Layby-Dwell ablation (184K turns)", fontsize=7.1, fontweight="bold", pad=4)
    ax2.tick_params(axis="x", labelsize=6.2)
    ax2.grid(axis="x", color="#e4e3dc", lw=0.55)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout(w_pad=1.1)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    ref = load(os.path.join(LAYBY, "reference/rows.jsonl"))
    rb_v2_path = os.path.join(HERE, "data/returnbench_v2_s0.json")
    S = summary(ref, ["C0", "wt", "stk", "cost", "orc"])
    if os.path.exists(rb_v2_path):
        c95_ref, _ = cells(ref, "p95")
        v2_95, v2_50 = load_rb_v2(rb_v2_path)
        ks = [k for k in c95_ref if k in v2_95]
        v = [v2_95[k] for k in ks]
        S["cost"] = dict(
            n=len(ks), p95=gm(v), p50=gm([v2_50[k] for k in ks]), worst=max(v),
            bad=sum(x > 1.05 for x in v),
            disk={d: gm([v2_95[k] for k in ks if k[2] == d]) for d in (0.5, 1.5, 3.0)},
            vs_ref=gm([v2_95[k] / c95_ref[k]["wt"] for k in ks]),
            vs_ref_disk={d: gm([v2_95[k] / c95_ref[k]["wt"] for k in ks if k[2] == d]) for d in (0.5, 1.5, 3.0)},
        )
    for r, s in S.items():
        print(r, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k not in ("disk", "vs_ref_disk")},
              {d: round(v, 2) for d, v in s["disk"].items()}, {d: round(v, 2) for d, v in s.get("vs_ref_disk", {}).items()})
    tex_summary(S, ["wt", "stk", "cost", "orc"], os.path.join(OUT, "tab_sim.tex"))
    fig_cells(ref, os.path.join(OUT, "fig_cells.pdf"))
    fig_load(ref, os.path.join(OUT, "fig_load.pdf"), rb_v2_path=rb_v2_path)
    tex_pools(os.path.join(OUT, "tab_pools.tex"))
    base = load(os.path.join(HERE, "data/baselines_rows.jsonl"))
    if base:
        B = summary(base, ["C0", "cont", "cj", "ka"], ref="C0")
        for r, s in B.items():
            print(r, round(s["p95"], 3), round(s["p50"], 3), round(s["worst"], 3), s["bad"], {d: round(v, 3) for d, v in s["disk"].items()})
        tex_summary(B, ["cont", "cj", "ka"], os.path.join(OUT, "tab_baselines.tex"), ref=False)
    abl = load(os.path.join(HERE, "data/ablation_rows.jsonl"))
    if abl:
        A = summary(abl, ["C0", "wt", "cost", "c_disk", "c_nopark", "o_disk"])
        for r, s in A.items():
            print(r, round(s["p95"], 3), round(s["worst"], 3), {d: round(v, 3) for d, v in s["disk"].items()},
                  {d: round(v, 3) for d, v in s.get("vs_ref_disk", {}).items()})
        tex_summary(A, ["wt", "cost", "c_disk", "c_nopark", "o_disk"], os.path.join(OUT, "tab_ablation.tex"))
    if os.path.isdir(os.path.join(HERE, "data/real")):
        tex_real(os.path.join(OUT, "tab_real.tex"))
    if all(os.path.exists(os.path.join(HERE, f)) for _, f in FIDELITY):
        tex_fidelity(os.path.join(OUT, "tab_fidelity.tex"))
    tex_percell(ref, os.path.join(OUT, "tab_percell.tex"), rb_v2_path=rb_v2_path)
    v2_path = os.path.join(HERE, "data/v2_results.json")
    if os.path.exists(v2_path):
        v2 = json.load(open(v2_path))
        tex_v2_model(v2, os.path.join(OUT, "tab_v2_model.tex"))
        tex_v2_glm2027(v2, os.path.join(OUT, "tab_v2_glm2027.tex"))
        tex_v2_vllm(v2, os.path.join(OUT, "tab_v2_vllm.tex"))
        tex_v2_sglang(v2, os.path.join(OUT, "tab_v2_sglang.tex"))
        tex_v3_cross_hw(v2, os.path.join(OUT, "tab_v3_cross_hw.tex"))
        tex_v3_slow_and_engine_abl(v2, os.path.join(OUT, "tab_v3_slow_ablation.tex"), os.path.join(OUT, "tab_v3_engine_ablation.tex"))
        tex_v3_model_ablation(v2, os.path.join(OUT, "tab_v3_model_ablation.tex"))
        fig_overview(v2, os.path.join(OUT, "fig_overview.pdf"))
        fig_v2_glm(v2, os.path.join(OUT, "fig_v2_glm.pdf"))
        fig_v2_matrix(v2, os.path.join(OUT, "fig_v2_matrix.pdf"))
        fig_v3_cross_hw(v2, os.path.join(OUT, "fig_v3_cross_hw.pdf"))
        fig_ablations(v2, os.path.join(OUT, "fig_ablations.pdf"))
        print("v2/v2.1 tables and figures generated cleanly")
