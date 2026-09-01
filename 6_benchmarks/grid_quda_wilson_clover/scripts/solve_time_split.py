#!/usr/bin/env python3
"""Emit the merged Wilson+Clover solve time-split table for one benchmark row.

Companion to summarize.py. Where summarize.py reports every section of a run,
this produces the single table used in the campaign docs: total solve time
broken into its nested sub-times, Wilson and Clover side by side, Grid against
QUDA.

    python3 solve_time_split.py <run_dir> [--grid-backend grid|grid_mixed]
                                          [--grid-label G1] [--quda-label Q1]

<run_dir> is one row directory, e.g.
    runs/2026_8_26_grid_quda_campaign2_16x48/q1_strict_57630112
containing records.jsonl and quda_resource/tunecache.tsv.

Conventions (must match the campaign docs):
  * Grid time is `median_s`; QUDA time is `internal_s` (device-only, excludes
    the PCIe host transfer that dominates QUDA's public-API timings).
  * Both are the median of `solve_repeats` independent full solves.
  * Grid sub-times are the `kind:"timing"` records at `timing_scope`
    "resident_grid" -- clean, resident-device measurements.
  * QUDA has NO comparable sub-times: its per-operation records go through the
    public API and are ~77% cudaMemcpy. The one QUDA per-kernel figure quoted
    is its own autotuner time from quda_resource/tunecache.tsv, selected for
    the row's sloppy precision/reconstruct, non-xpay, non-dagger, parity-1 --
    the same kernel-row convention the GFlop/s columns use.
  * Rows nest: total = iterations x per-iteration; per-iteration = CG matrix
    apply (`normal_pc`) + vector algebra remainder. `normal_pc` is ~4 hops.
"""

import argparse
import json
import os
import re
import sys

# QUDA template-argument encoding, confirmed against Campaign 2 tunecaches.
PRECISION_TAG = {"double": "ArgId", "single": "ArgIf", "half": "ArgIs"}
RECONSTRUCT_TAG = {"no": "_s18", "18": "_s18", "12": "_s12", "8": "_s8"}


def load_records(run_dir):
    """Return (solve, timing) dicts keyed by (action, backend) / (action, op)."""
    path = os.path.join(run_dir, "records.jsonl")
    solve, timing = {}, {}
    with open(path) as handle:
        for line in handle:
            record = json.loads(line)
            kind = record.get("kind")
            if kind == "solve":
                solve[(record["action"], record["backend"])] = record
            elif kind == "timing" and record.get("backend") == "grid":
                timing[(record["action"], record["op"])] = record["median_s"]
    return solve, timing


def quda_kernel_seconds(run_dir, sloppy_precision, sloppy_reconstruct):
    """Autotuned seconds/call for the Wilson and clover dslash of this row.

    Returns {action: seconds}. Missing entries are omitted rather than raising:
    a row whose kernel never got tuned should degrade to "n/a" in the table.
    """
    path = os.path.join(run_dir, "quda_resource", "tunecache.tsv")
    prec = PRECISION_TAG.get(str(sloppy_precision).lower())
    recon = RECONSTRUCT_TAG.get(str(sloppy_reconstruct).lower())
    found = {}
    if prec is None or recon is None or not os.path.exists(path):
        return found
    with open(path) as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 17 or "GB/s" not in fields[16]:
                continue
            name, aux, seconds = fields[1], fields[2], fields[15]
            if "parity=1" not in aux or "dagger" in aux or "xpay" in aux:
                continue
            if prec not in name or recon not in name:
                continue
            if "WilsonCloverPreconditioned" in name:
                found.setdefault("clover", float(seconds))
            elif re.search(r"quda6Wilson", name):
                found.setdefault("wilson", float(seconds))
    return found


def collect(run_dir, grid_backend):
    solve, timing = load_records(run_dir)
    any_quda = next(r for (a, b), r in solve.items() if b == "quda")
    kernels = quda_kernel_seconds(
        run_dir, any_quda.get("sloppy_precision"), any_quda.get("sloppy_reconstruct"))

    data = {}
    for action in ("wilson", "clover"):
        grid = solve.get((action, grid_backend))
        quda = solve.get((action, "quda"))
        if grid is None or quda is None:
            continue
        iterations = grid["iterations"]
        grid_total = grid["median_s"]
        quda_total = quda["internal_s"]
        per_iter = grid_total / iterations
        normal_pc = timing.get((action, "normal_pc"))
        data[action] = {
            "grid_total": grid_total,
            "quda_total": quda_total,
            "grid_iters": iterations,
            "quda_iters": quda["iterations"],
            "grid_per_iter": per_iter,
            "quda_per_iter": quda_total / quda["iterations"],
            "normal_pc": normal_pc,
            "vector_algebra": (per_iter - normal_pc) if normal_pc else None,
            "hop": timing.get((action, "pc_dslash")),
            "quda_hop": kernels.get(action),
            "clover_inv": timing.get((action, "clover_inv")),
            "mat": timing.get((action, "mat")),
        }
    return data


def render(data, grid_label, quda_label):
    seconds = lambda v: "%.4f s" % v
    millis = lambda v: "%.3f ms" % (v * 1e3)
    actions = [a for a in ("wilson", "clover") if a in data]

    header = ["Measurement"]
    for action in actions:
        tag = action.capitalize()
        header += ["%s Grid (%s)" % (tag, grid_label),
                   "%s QUDA (%s)" % (tag, quda_label),
                   "%s ratio" % tag[0]]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]

    def emit(label, grid_key, quda_cell, fmt, bold=False, ratio=True):
        cells = []
        for action in actions:
            d = data[action]
            gv = d.get(grid_key)
            gs = fmt(gv) if gv is not None else "—"
            qv = quda_cell(d)
            qs = fmt(qv) if isinstance(qv, float) else (qv or "—")
            if ratio and isinstance(qv, float) and gv:
                rs = "%.2fx" % (gv / qv)
            else:
                rs = "—"
            cells += [gs, qs, rs]
        if bold:
            cells = ["**%s**" % c for c in cells]
            label = "**%s**" % label
        lines.append("| " + " | ".join([label] + cells) + " |")

    emit("Total solve time", "grid_total", lambda d: d["quda_total"], seconds, bold=True)
    emit("CG iterations", "grid_iters", lambda d: float(d["quda_iters"]), lambda v: "%d" % v)
    emit("Time per iteration", "grid_per_iter", lambda d: d["quda_per_iter"], millis, bold=True)
    emit("↳ CG matrix apply (`normal_pc`)", "normal_pc", lambda d: "not separable", millis)
    emit("↳ vector algebra (remainder)", "vector_algebra", lambda d: "not separable", millis)
    emit("One hop (`pc_dslash`)", "hop", lambda d: d["quda_hop"], millis, bold=True)
    emit("↳ of which local clover (`clover_inv`)", "clover_inv",
         lambda d: "fused into hop" if d.get("clover_inv") else "n/a (no clover term)",
         millis, ratio=False)
    emit("Full operator (`mat`)", "mat", lambda d: "not comparable", millis)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="one row directory containing records.jsonl")
    parser.add_argument("--grid-backend", default="grid", choices=["grid", "grid_mixed"],
                        help="which Grid record to use (default: grid, i.e. G1)")
    parser.add_argument("--grid-label", default="G1")
    parser.add_argument("--quda-label", default="Q1")
    args = parser.parse_args()

    data = collect(args.run_dir, args.grid_backend)
    if not data:
        sys.exit("no paired grid/quda solve records found in %s" % args.run_dir)
    print(render(data, args.grid_label, args.quda_label))


if __name__ == "__main__":
    main()
