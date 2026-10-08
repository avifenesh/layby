"""returnbench: run placement policies on ReturnBench pools in the engine simulator.

  returnbench eval [--rules C0 wt cost ...] [--cells POOL:N ...] [--workloads 8] [--seeds 3]
                   [--disk 0.5 1.5 3] [--policy NAME=module:Class ...] [--out rows.jsonl]
  returnbench table rows.jsonl [--cells] [--ref wt]

eval runs the 33-cell reference set unless --cells is given, prints the summary, and appends one JSON
row per (cell, workload, seed, rule) to --out. table summarizes rows written earlier. C0 always runs:
every ratio is against it.
"""
import argparse, json, sys

from . import eval_pop as E


def fmt_cells(cr):
    lines = []
    for (pool, n, d), rs in cr.items():
        lines.append(f"{pool} n={n} disk {d:g} GB/s")
        for r, (p50, p95, g50, g95) in rs.items():
            lines.append(f"  {r:9s} p50 {p50:6.2f} s  p95 {p95:6.1f} s   vs C0: p50 {g50:.2f}  p95 {g95:.2f}")
    return "\n".join(lines)


def fmt_summary(res, ref):
    disks = sorted({d for e in res.values() for d in e["by_disk"]})
    head = f"{'rule':9s} {'cells':>5s}  p95/C0  worst  " + "  ".join(f"@{d:g}" for d in disks)
    if any("vs_" + ref in e for e in res.values()):
        head += f"   p95/{ref}  " + "  ".join(f"@{d:g}" for d in disks)
    lines = ["geomean over cells of the p95 TTFT ratio (per cell: geomean over workloads)", head]
    for r, e in res.items():
        s = f"{r:9s} {e['cells']:5d}  {e['vs_C0']:6.2f}  {e['worst_vs_C0']:5.2f}  " + "  ".join(f"{e['by_disk'][d]:4.2f}" for d in disks)
        if "vs_" + ref in e:
            s += f"   {e['vs_' + ref]:6.2f}  " + "  ".join(f"{e['by_disk_vs_' + ref][d]:4.2f}" for d in disks)
        lines.append(s)
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="returnbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval", help="run policies on the bench")
    e.add_argument("--rules", nargs="+", default=["C0", "wt", "cost"])
    e.add_argument("--policy", nargs="*", default=[], help="NAME=module:Class, a policy of your own")
    e.add_argument("--cells", nargs="*", help="POOL:N (POOL may join pools with +); default: the 33-cell reference set")
    e.add_argument("--workloads", type=int, default=8); e.add_argument("--seeds", type=int, default=3)
    e.add_argument("--disk", type=float, nargs="+", default=list(E.DISKS))
    e.add_argument("--curve", default="v6", help="survival curve field surv_NAME the curve-driven rules read")
    e.add_argument("--window", type=float, help="run length in seconds (no send after it)")
    e.add_argument("--procs", type=int, default=8); e.add_argument("--out")
    e.add_argument("--ref", default="wt"); e.add_argument("--per-cell", action="store_true")
    t = sub.add_parser("table", help="summarize rows written by eval")
    t.add_argument("rows"); t.add_argument("--ref", default="wt"); t.add_argument("--per-cell", action="store_true")
    a = ap.parse_args(argv)

    if a.cmd == "eval":
        extra = [x for x in a.policy]
        names = [E.register(x) for x in extra]
        rules = ["C0"] + [r for r in a.rules + names if r != "C0"]
        bad = [r for r in rules if r not in E.RULES]
        if bad:
            ap.error(f"unknown rules {bad}; known: {list(E.RULES)}")
        cells = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in a.cells] if a.cells else E.REFERENCE
        rows = E.evaluate(cells, rules, a.workloads, a.seeds, a.disk, a.curve, a.window, a.procs, extra)
        if a.out:
            with open(a.out, "a") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
    else:
        rows = [json.loads(l) for l in open(a.rows)]
    cr, res = E.summary(rows, a.ref)
    if a.per_cell:
        print(fmt_cells(cr)); print()
    print(fmt_summary(res, a.ref))


if __name__ == "__main__":
    sys.exit(main())
