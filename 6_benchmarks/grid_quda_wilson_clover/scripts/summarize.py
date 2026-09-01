#!/usr/bin/env python3
"""Summarize Grid/QUDA Wilson(-clover) benchmark JSONL records."""

import argparse
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


CASE_FIELDS = (
    "input",
    "lattice",
    "mpi",
    "mass",
    "csw",
    "precision",
    "cache_state",
    "action",
    "op",
)
PAIR_FIELDS = CASE_FIELDS[:-2] + ("action", "op")


class SummaryError(RuntimeError):
    pass


def load_records(paths):
    records = []
    for path in paths:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise SummaryError(f"cannot read {path}: {exc}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SummaryError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise SummaryError(f"{path}:{line_number}: record must be a JSON object")
            record["_source"] = str(path)
            record["_line"] = line_number
            records.append(record)
    return records


# ---------------------------------------------------------------------
# Stock QUDA dslash_test log parsing.
#
# dslash_test is a separate gtest-based binary (quda/tests/dslash_test.cpp),
# not the shared JSONL harness, so its stdout is parsed on a best-effort
# basis against the literal printfQuda()/gtest formats in
# quda/tests/dslash_test_utils.h (run_test(), verify()) and
# quda/tests/dslash_test.cpp as of the commit this benchmark was built
# against. perlmutter/run_benchmark.sh wraps that stdout with an
# ENV-line provenance header (see write_provenance()) ahead of a
# "--- output ---" marker; parse_dslash_log expects that wrapper.
#
# This has not yet been exercised against a real dslash_test run (its
# build is still pending); spot-check the regexes below the first time a
# real log is available and fix any drift from the assumed format.
# ---------------------------------------------------------------------

ENV_LINE_RE = re.compile(r"^ENV\s+(.*)$")
GTEST_RUN_RE = re.compile(r"\[\s*RUN\s*\]\s*DslashTest\.(\w+)")
GTEST_STATUS_RE = re.compile(r"\[\s*(OK|FAILED|SKIPPED)\s*\]\s*DslashTest\.(\w+)")
# Matches display_test_info()'s data row in dslash_test.cpp:
#   prec recon dtest_type matpc_type dagger xdim/ydim/zdim tdim Lsdim dslash_type niter
DSLASH_INFO_ROW_RE = re.compile(
    r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\d+)\s*$",
    re.MULTILINE,
)
# run_test()'s "%fus per kernel call" (dslash_test_utils.h) -- resident-device,
# transfer-free per-call time for the whole configured --niter loop.
PER_CALL_RE = re.compile(r"([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)\s*us per kernel call")
GFLOPS_RE = re.compile(r"GFLOPS\s*=\s*([0-9.eE+-]+)")
GBYTES_RE = re.compile(r"GBYTES\s*=\s*([0-9.eE+-]+)")
# "%d flops per kernel call, %d flops per site" (dslash_test_utils.h) -- the
# latter is a property of the operator's arithmetic (dtest_type/dslash_type),
# not of which code computes it, so it is valid to pair with a Grid timing
# for the identical operator -- see grid_derived_flops_rows().
FLOPS_PER_SITE_RE = re.compile(r"(\d+)\s*flops per kernel call,\s*(\d+)\s*flops per site")
# verify()'s "Results: ... L2 relative deviation = %e, ..." line.
DEVIATION_RE = re.compile(r"L2 relative deviation\s*=\s*([0-9.eE+-]+)")


def parse_env(text):
    env = {}
    for line in text.splitlines():
        match = ENV_LINE_RE.match(line.strip())
        if not match:
            continue
        for token in match.group(1).split():
            if "=" not in token:
                continue
            key, _, value = token.partition("=")
            env[key] = value
    return env


def gtest_block(text, test_name):
    """Return (output slice, status) for one DslashTest.<test_name> gtest
    block, or (None, None) if that test's [ RUN ] marker is not found."""
    run_match = None
    for match in GTEST_RUN_RE.finditer(text):
        if match.group(1) == test_name:
            run_match = match
            break
    if run_match is None:
        return None, None
    start = run_match.end()
    status = None
    end = len(text)
    for match in GTEST_STATUS_RE.finditer(text, start):
        if match.group(2) == test_name:
            status = match.group(1)
            end = match.start()
            break
    return text[start:end], status


def parse_dslash_log(path):
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SummaryError(f"cannot read {path}: {exc}") from exc

    if "--- output ---" in text:
        provenance_text, _, output_text = text.partition("--- output ---")
    else:
        provenance_text, output_text = text, text
    env = parse_env(provenance_text)

    benchmark_block, benchmark_status = gtest_block(output_text, "benchmark")
    verify_block, verify_status = gtest_block(output_text, "verify")

    record = {
        "kind": "resident_dslash",
        "backend": "quda",
        "timing_scope": "resident_device_quda",
        "action": env.get("ACTION"),
        "op": env.get("DTEST_OP"),
        "precision": env.get("PRECISION"),
        "input": env.get("INPUT"),
        "lattice": env.get("LATT"),
        "mpi": env.get("MPI"),
        "mass": finite_number(env.get("MASS")),
        "csw": finite_number(env.get("CSW")),
        "niter": finite_number(env.get("REPETITIONS")),
        "benchmark_status": benchmark_status or "MISSING",
        "verify_status": verify_status or "MISSING",
        "_source": str(path),
        "_line": None,
    }

    info_match = DSLASH_INFO_ROW_RE.search(benchmark_block or "")
    if info_match:
        record["quda_prec"] = info_match.group(1)
        record["quda_recon"] = info_match.group(2)
        record["dtest_type"] = info_match.group(3)
        record["matpc_type"] = info_match.group(4)
        record["dslash_type"] = info_match.group(11)

    if benchmark_block:
        per_call = PER_CALL_RE.search(benchmark_block)
        if per_call:
            record["seconds"] = float(per_call.group(1)) / 1.0e6
        gflops = GFLOPS_RE.search(benchmark_block)
        if gflops:
            record["gflops"] = float(gflops.group(1))
        gbytes = GBYTES_RE.search(benchmark_block)
        if gbytes:
            record["gbytes"] = float(gbytes.group(1))
        flops_per_site = FLOPS_PER_SITE_RE.search(benchmark_block)
        if flops_per_site:
            record["flops_per_call"] = float(flops_per_site.group(1))
            record["flops_per_site"] = float(flops_per_site.group(2))

    if verify_block:
        deviations = [float(m.group(1)) for m in DEVIATION_RE.finditer(verify_block)]
        if deviations:
            record["deviation"] = max(deviations)

    return record


def finite_number(value):
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def display_value(record, key, fallback="-"):
    value = record.get(key)
    if value is None or value == "":
        return fallback
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def elapsed(record):
    for field, label in (("median_s", "median"), ("mean_s", "mean"), ("seconds", "value")):
        value = finite_number(record.get(field))
        if value is not None:
            return value, label
    samples = record.get("sample_seconds")
    if isinstance(samples, list):
        values = [finite_number(item) for item in samples]
        values = [value for value in values if value is not None]
        if values:
            return statistics.median(values), "median"
    return None, "missing"


def spread(record):
    minimum = finite_number(record.get("min_s"))
    maximum = finite_number(record.get("max_s"))
    stddev = finite_number(record.get("stddev_s"))
    if minimum is not None and maximum is not None:
        result = f"[{minimum:.6g}, {maximum:.6g}]"
        if stddev is not None:
            result += f"; sd={stddev:.3g}"
        return result
    samples = record.get("sample_seconds")
    if isinstance(samples, list):
        values = [finite_number(item) for item in samples]
        values = [value for value in values if value is not None]
        if values:
            result = f"[{min(values):.6g}, {max(values):.6g}]"
            if len(values) > 1:
                result += f"; sd={statistics.stdev(values):.3g}"
            return result
    return "-"


def case_key(record):
    return (record.get("_source"),) + tuple(record.get(field) for field in PAIR_FIELDS)


def pair_backends(records):
    grouped = defaultdict(dict)
    for record in records:
        backend = record.get("backend")
        if backend not in ("grid", "quda"):
            continue
        key = case_key(record)
        if backend in grouped[key]:
            prior = grouped[key][backend]
            raise SummaryError(
                f"duplicate {backend} record for {record.get('action')}/{record.get('op')} "
                f"in {record.get('_source')} (lines {prior.get('_line')} and {record.get('_line')})"
            )
        grouped[key][backend] = record
    return [(pair.get("grid"), pair.get("quda")) for _, pair in sorted(grouped.items(), key=lambda item: repr(item[0]))]


def matching_check(records, timing, op=None):
    """`op` overrides timing's own op field when the correctness marker for
    this row is filed under a different op -- e.g. a `backend: grid_mixed`
    solve record has `op: solve` but its correctness check is filed under
    `op: solve_mixed`, so looking it up with the record's own op would
    silently match the wrong (plain-CG) correctness gate instead."""
    op = timing.get("op") if op is None else op
    for record in records:
        if record.get("kind") != "correctness":
            continue
        if record.get("_source") != timing.get("_source"):
            continue
        if record.get("action") == timing.get("action") and record.get("op") == op and record.get("precision") == timing.get("precision"):
            return "PASS" if record.get("passed") is True else "FAIL"
    return "MISSING"


def correctness_op(record):
    return "solve_mixed" if record.get("backend") == "grid_mixed" else None


def markdown_table(headers, rows):
    if not rows:
        return "_No records._\n"
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines.extend("| " + " | ".join(str(cell).replace("|", "\\|") for cell in row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def record_case(record):
    fields = []
    for name in ("input", "lattice", "mpi", "mass"):
        value = record.get(name)
        if value is not None:
            fields.append(f"{name}={display_value(record, name)}")
    case = ", ".join(fields) if fields else Path(str(record.get("_source", "?"))).name
    source = record.get("_source")
    run_label = Path(str(source)).parent.name if source else None
    return f"run={run_label}, {case}" if run_label else case


def timing_rows(records):
    rows = []
    timings = [record for record in records if record.get("kind") == "timing"]
    for grid, quda in pair_backends(timings):
        exemplar = grid or quda
        assert exemplar is not None
        grid_time, grid_metric = elapsed(grid or {})
        quda_time, quda_metric = elapsed(quda or {})
        ratio = grid_time / quda_time if grid_time is not None and quda_time not in (None, 0.0) else None
        metric = grid_metric if grid_metric == quda_metric else f"{grid_metric}/{quda_metric}"
        rows.append(
            [
                record_case(exemplar),
                display_value(exemplar, "action"),
                display_value(exemplar, "op"),
                display_value(exemplar, "precision"),
                metric,
                f"{grid_time:.6g}" if grid_time is not None else "-",
                spread(grid or {}),
                f"{quda_time:.6g}" if quda_time is not None else "-",
                spread(quda or {}),
                f"{ratio:.4g}" if ratio is not None else "-",
                matching_check(records, exemplar),
            ]
        )
    return rows


def iteration_cell(record):
    """Iteration count as reported, annotated when the measured repeats did not
    all agree so a varying count is never displayed as a single clean number."""
    if not record:
        return "-"
    shown = display_value(record, "iterations")
    if record.get("iterations_constant") is False:
        low = display_value(record, "iterations_min")
        high = display_value(record, "iterations_max")
        return f"{shown} (varies {low}-{high})"
    return shown


def seconds_per_iteration(record, elapsed_seconds):
    """Prefer the harness's internal-time-based per-repeat median (immune to
    QUDA's per-call PCIe transfer); fall back to the wall-clock-based median,
    then the pre-fix unlabeled field name for older records, then dividing
    aggregates only when no reported field is present at all."""
    if record:
        for field in (
            "s_per_iteration_median_internal",
            "s_per_iteration_median_wall",
            "s_per_iteration_median",  # legacy field name, wall-clock-based
        ):
            reported = finite_number(record.get(field))
            if reported is not None and reported > 0.0:
                return reported
    iterations = finite_number((record or {}).get("iterations"))
    if elapsed_seconds is None or iterations in (None, 0.0):
        return None
    return elapsed_seconds / iterations


def solve_attributable(grid, quda):
    """True only when the harness affirmatively marked the pair attributable.
    Absent evidence is not attributable evidence: a record predating the flag,
    or a half-missing pair, is reported as unknown rather than as a clean
    kernel-explained ratio."""
    flags = [record.get("attributable_to_operator_throughput") for record in (grid, quda) if record]
    if not flags or any(flag is None for flag in flags):
        return None
    return all(flag is True for flag in flags)


def solve_rows(records):
    rows = []
    solves = [record for record in records if record.get("kind") == "solve"]
    for grid, quda in pair_backends(solves):
        exemplar = grid or quda
        assert exemplar is not None
        grid_time, grid_metric = elapsed(grid or {})
        quda_time, quda_metric = elapsed(quda or {})
        internal = finite_number((quda or {}).get("internal_s"))
        # QUDA's public API copies the source/solution over PCIe on every call,
        # so `quda_time` (wall) is not a kernel-throughput number by itself --
        # prefer internal_s, which is present on every solve record. Falling
        # back to wall only covers older records that predate this field.
        quda_reference_time = internal if internal is not None else quda_time
        ratio = (
            grid_time / quda_reference_time
            if grid_time is not None and quda_reference_time not in (None, 0.0)
            else None
        )
        grid_tpi = seconds_per_iteration(grid, grid_time)
        quda_tpi = seconds_per_iteration(quda, quda_time)
        attributable = solve_attributable(grid, quda)
        if ratio is None:
            ratio_cell = "-"
        elif attributable is True:
            ratio_cell = f"{ratio:.4g}"
        elif attributable is False:
            ratio_cell = f"{ratio:.4g} (NOT attributable)"
        else:
            ratio_cell = f"{ratio:.4g} (attribution unknown)"
        rows.append(
            [
                record_case(exemplar),
                display_value(exemplar, "action"),
                display_value(exemplar, "precision"),
                grid_metric if grid_metric == quda_metric else f"{grid_metric}/{quda_metric}",
                f"{grid_time:.6g}" if grid_time is not None else "-",
                iteration_cell(grid),
                f"{grid_tpi:.6g}" if grid_tpi is not None else "-",
                display_value(grid or {}, "independent_residual", display_value(grid or {}, "true_residual")),
                f"{quda_time:.6g}" if quda_time is not None else "-",
                iteration_cell(quda),
                f"{quda_tpi:.6g}" if quda_tpi is not None else "-",
                display_value(quda or {}, "independent_residual", display_value(quda or {}, "true_residual")),
                f"{internal:.6g}" if internal is not None else "-",
                ratio_cell,
                matching_check(records, exemplar),
            ]
        )
    return rows


def solve_attribution_problems(records):
    """Solve pairs whose time ratio must not be read as a kernel-throughput
    result, either because the harness said so or because it never said."""
    problems = []
    solves = [record for record in records if record.get("kind") == "solve"]
    for grid, quda in pair_backends(solves):
        if solve_attributable(grid, quda) is True:
            continue
        exemplar = grid or quda
        if exemplar is None:
            continue
        problems.append((exemplar, grid, quda))
    return problems


def mixed_solve_rows(records):
    """Grid's own MixedPrecisionConjugateGradient (backend `grid_mixed`) never
    goes through pair_backends() -- it is a third backend alongside grid/quda,
    not a pair member -- so without this it is silently absent from every
    table and only recoverable by grepping the raw log. Both ratios below are
    wall-clock: grid_mixed has no public-API host transfer, same as plain
    grid, so wall time is the right basis against plain grid; against QUDA it
    is compared to QUDA's internal_s for the same PCIe-contamination reason
    used everywhere else in this script."""
    rows = []
    solves = [record for record in records if record.get("kind") == "solve"]
    grouped = defaultdict(dict)
    for record in solves:
        backend = record.get("backend")
        if backend not in ("grid", "quda", "grid_mixed"):
            continue
        grouped[case_key(record)][backend] = record
    for key, backends in sorted(grouped.items(), key=lambda item: repr(item[0])):
        mixed = backends.get("grid_mixed")
        if mixed is None:
            continue
        grid = backends.get("grid")
        quda = backends.get("quda")
        mixed_time, mixed_metric = elapsed(mixed)
        grid_time, _ = elapsed(grid or {})
        internal = finite_number((quda or {}).get("internal_s"))
        ratio_vs_grid = (
            mixed_time / grid_time if mixed_time is not None and grid_time not in (None, 0.0) else None
        )
        ratio_vs_quda_internal = (
            mixed_time / internal if mixed_time is not None and internal not in (None, 0.0) else None
        )
        restarts = mixed.get("restarts_all")
        restarts_cell = str(restarts[0]) if isinstance(restarts, list) and restarts else "-"
        rows.append(
            [
                record_case(mixed),
                display_value(mixed, "action"),
                display_value(mixed, "precision"),
                mixed_metric,
                f"{mixed_time:.6g}" if mixed_time is not None else "-",
                iteration_cell(mixed),
                restarts_cell,
                display_value(mixed, "independent_residual", display_value(mixed, "true_residual")),
                f"{grid_time:.6g}" if grid_time is not None else "-",
                f"{internal:.6g}" if internal is not None else "-",
                f"{ratio_vs_grid:.4g}" if ratio_vs_grid is not None else "-",
                f"{ratio_vs_quda_internal:.4g}" if ratio_vs_quda_internal is not None else "-",
                matching_check(records, mixed, correctness_op(mixed)),
            ]
        )
    return rows


def clover_ratio_rows(records):
    timings = [record for record in records if record.get("kind") == "timing"]
    grouped = defaultdict(dict)
    for record in timings:
        key = (
            record.get("_source"),
            record.get("input"),
            record.get("lattice"),
            record.get("mpi"),
            record.get("mass"),
            record.get("precision"),
            record.get("cache_state"),
            record.get("op"),
            record.get("backend"),
        )
        grouped[key][str(record.get("action"))] = record

    rows = []
    for _, actions in sorted(grouped.items(), key=lambda item: repr(item[0])):
        wilson = actions.get("wilson")
        clover = actions.get("clover")
        if wilson is None or clover is None:
            continue
        wilson_time, _ = elapsed(wilson)
        clover_time, _ = elapsed(clover)
        ratio = clover_time / wilson_time if wilson_time not in (None, 0.0) and clover_time is not None else None
        rows.append(
            [
                record_case(wilson),
                display_value(wilson, "op"),
                display_value(wilson, "backend"),
                display_value(wilson, "precision"),
                f"{wilson_time:.6g}" if wilson_time is not None else "-",
                f"{clover_time:.6g}" if clover_time is not None else "-",
                f"{ratio:.4g}" if ratio is not None else "-",
            ]
        )
    return rows


def resident_dslash_rows(records):
    rows = []
    for record in records:
        if record.get("kind") != "resident_dslash":
            continue
        seconds = finite_number(record.get("seconds"))
        rows.append(
            [
                Path(str(record.get("_source", "?"))).name,
                display_value(record, "action"),
                display_value(record, "op"),
                display_value(record, "precision"),
                display_value(record, "niter"),
                f"{seconds:.6g}" if seconds is not None else "-",
                display_value(record, "gflops"),
                display_value(record, "gbytes"),
                display_value(record, "deviation"),
                display_value(record, "benchmark_status"),
                display_value(record, "verify_status"),
            ]
        )
    return rows


def resident_dslash_problems(records):
    problems = []
    for record in records:
        if record.get("kind") != "resident_dslash":
            continue
        benchmark_status = record.get("benchmark_status")
        verify_status = record.get("verify_status")
        if benchmark_status != "OK" or verify_status in ("FAILED", "MISSING"):
            problems.append(record)
    return problems


# ---------------------------------------------------------------------
# Grid does not print its own FLOP/s anywhere in this repository's logs --
# confirmed by direct search of the shared harness log, the harness source
# (benchmark_grid_quda_wilson_clover.cc), and real production HMC logs, none
# of which report a Grid-side flop rate. grid_derived_flops_rows() derives
# one instead: it multiplies Grid's own measured wall time for its
# `normal_pc` operator (M_pc^dagger M_pc) by the "flops per site" constant
# a stock QUDA dslash_test run prints for the mathematically identical
# operator (dtest_type MatPCDagMatPC). That constant is a property of the
# operator's arithmetic, not of which code computes it, so pairing it with
# Grid's own time for the same operator is legitimate -- but it requires a
# dslash_test log (see --dslash-log); without one, this table is empty
# rather than silently wrong.
# ---------------------------------------------------------------------

DSLASH_TYPE_TO_GRID_OP = {"wilson": "normal_pc", "clover": "normal_pc"}


def local_volume(record):
    lattice = record.get("lattice")
    mpi = record.get("mpi")
    if not lattice or not mpi:
        return None
    try:
        latt_dims = [int(x) for x in str(lattice).split(".")]
        mpi_dims = [int(x) for x in str(mpi).split(".")]
    except ValueError:
        return None
    if not latt_dims or len(latt_dims) != len(mpi_dims):
        return None
    volume = 1
    for extent, ranks in zip(latt_dims, mpi_dims):
        if ranks <= 0 or extent % ranks != 0:
            return None
        volume *= extent // ranks
    return volume


def resident_dslash_site_flops(records):
    """Map (dslash_type, matching Grid op) -> the resident_dslash record that
    supplies flops_per_site for it, keyed on dslash_type (always parsed from
    the info row) rather than action (only present when the log carries an
    ENV wrapper) so this works for dslash_test logs run outside run_benchmark.sh."""
    lookup = {}
    for record in records:
        if record.get("kind") != "resident_dslash":
            continue
        if record.get("dtest_type") != "MatPCDagMatPC":
            continue
        flops_per_site = finite_number(record.get("flops_per_site"))
        if flops_per_site is None:
            continue
        grid_op = DSLASH_TYPE_TO_GRID_OP.get(record.get("dslash_type"))
        if grid_op is None:
            continue
        key = (record.get("dslash_type"), grid_op)
        lookup.setdefault(key, record)
    return lookup


def grid_derived_flops_rows(records):
    site_flops = resident_dslash_site_flops(records)
    rows = []
    for record in records:
        if record.get("kind") != "timing" or record.get("backend") != "grid":
            continue
        if record.get("op") != "normal_pc":
            continue
        source_record = site_flops.get((record.get("action"), "normal_pc"))
        if source_record is None:
            continue
        flops_per_site = finite_number(source_record.get("flops_per_site"))
        volume = local_volume(record)
        grid_time, metric = elapsed(record)
        if flops_per_site is None or volume is None or grid_time is None or grid_time <= 0:
            continue
        checkerboard_sites = volume / 2.0
        gflops = flops_per_site * checkerboard_sites / grid_time / 1.0e9
        rows.append(
            [
                record_case(record),
                display_value(record, "action"),
                display_value(record, "precision"),
                metric,
                f"{grid_time:.6g}",
                f"{volume:.0f}",
                f"{flops_per_site:.0f}",
                Path(str(source_record.get("_source", "?"))).name,
                f"{gflops:.1f}",
            ]
        )
    rows.sort(key=lambda row: tuple(row))
    return rows


# ---------------------------------------------------------------------
# QUDA autotuner throughput (quda_resource/tunecache.tsv).
#
# Neither the shared JSONL harness nor its log carries FLOPS/GB-s figures --
# that data only exists in QUDA's own per-run autotuning cache, one row per
# distinct kernel/launch-config QUDA's autotuner benchmarked, trailing
# comment `# X Gflop/s, Y GB/s, tuning took Z seconds at <timestamp>`. Each
# run directory gets its own tunecache.tsv (autotuning is per-process), so
# this is located relative to each records.jsonl's own directory, never a
# fixed path. Rows are reported as-is (one per kernel/precision/reconstruct/
# parity/xpay/dagger combination actually tuned) rather than picked down to
# a single "representative" number: a run's tunecache mixes the sloppy-
# precision solve kernel with the full-precision correctness-check kernel,
# and there is no reliable way to tell which is "the" CG kernel from this
# file alone, so collapsing them would risk mislabeling one as the other.
# ---------------------------------------------------------------------

TUNECACHE_COMMENT_RE = re.compile(r"^#\s*([0-9.eE+-]+)\s*Gflop/s,\s*([0-9.eE+-]+)\s*GB/s")
RECONSTRUCT_RE = re.compile(r"QudaReconstructType_s(\d+)E")
RECONSTRUCT_LABELS = {"18": "NO", "12": "12", "8": "8"}
PRECISION_LABELS = {"8": "double", "4": "single", "2": "half"}


def parse_tunecache(path):
    rows = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    for line in lines:
        if not line.strip() or line.startswith("volume"):
            continue
        fields = line.split("\t")
        if len(fields) < 5:
            continue
        name = fields[1].strip()
        aux = fields[2].strip()
        # "policy," is the top-level measured entry for a kernel launch;
        # "policy_kernel=..." rows are that same launch's internal
        # sub-decomposition and would double-count the same work if included.
        if not aux.startswith("policy,") or "Wilson" not in name:
            continue
        comment = fields[-1].strip()
        match = TUNECACHE_COMMENT_RE.match(comment)
        if not match:
            continue
        aux_fields = dict(item.split("=", 1) for item in aux.split(",") if "=" in item)
        recon_match = RECONSTRUCT_RE.search(name)
        if "WilsonCloverPreconditioned" in name:
            kernel = "WilsonCloverPreconditioned"
        elif "WilsonClover" in name:
            kernel = "WilsonClover"
        else:
            kernel = "Wilson"
        precision_code = aux_fields.get("precision")
        recon_code = recon_match.group(1) if recon_match else None
        rows.append(
            {
                "volume": fields[0].strip(),
                "kernel": kernel,
                "precision": PRECISION_LABELS.get(precision_code, precision_code or "-"),
                "reconstruct": RECONSTRUCT_LABELS.get(recon_code, recon_code or "-"),
                "parity": aux_fields.get("parity", "-"),
                "xpay": "xpay" in aux.split(","),
                "dagger": "dagger" in aux.split(","),
                "gflops": float(match.group(1)),
                "gbytes": float(match.group(2)),
            }
        )
    return rows


def flops_rows(records):
    rows = []
    sources = sorted({record.get("_source") for record in records if record.get("_source")})
    for source in sources:
        source_path = Path(source)
        tune_path = source_path.parent / "quda_resource" / "tunecache.tsv"
        if not tune_path.exists():
            continue
        for entry in parse_tunecache(tune_path):
            rows.append(
                [
                    source_path.parent.name,
                    entry["kernel"],
                    entry["precision"],
                    entry["reconstruct"],
                    entry["parity"],
                    "yes" if entry["xpay"] else "-",
                    "yes" if entry["dagger"] else "-",
                    f"{entry['gflops']:.1f}",
                    f"{entry['gbytes']:.1f}",
                ]
            )
    rows.sort(key=lambda row: tuple(row))
    return rows


def auxiliary_rows(records):
    rows = []
    for record in records:
        if record.get("kind") not in ("setup", "upload", "autotune"):
            continue
        value, metric = elapsed(record)
        rows.append(
            [
                record_case(record),
                display_value(record, "kind"),
                display_value(record, "action"),
                display_value(record, "op", display_value(record, "stage")),
                display_value(record, "backend"),
                metric,
                f"{value:.6g}" if value is not None else "-",
            ]
        )
    return rows


def render_summary(records):
    failed = [record for record in records if record.get("kind") == "correctness" and record.get("passed") is not True]
    missing = []
    for record in records:
        if record.get("kind") not in ("timing", "solve"):
            continue
        if matching_check(records, record, correctness_op(record)) == "MISSING":
            identity = (record.get("_source"), record.get("action"), record.get("op"), record.get("precision"))
            if identity not in missing:
                missing.append(identity)
    dslash_problems = resident_dslash_problems(records)
    attribution_problems = solve_attribution_problems(records)

    output = ["# Grid vs QUDA benchmark summary", ""]
    output.extend(
        [
            f"Inputs: {len({record.get('_source') for record in records})}; records: {len(records)}; "
            f"failed correctness checks: {len(failed)}; missing correctness groups: {len(missing)}; "
            f"dslash_test problems: {len(dslash_problems)}; "
            f"non-attributable solve ratios: {len(attribution_problems)}.",
            "",
            "Timing values are seconds per operation/solve. The `metric` column states whether a record supplies a "
            "median, mean, or single value; the summarizer does not relabel means as medians.",
            "",
            "## Fixed-work and public-API timings",
            "",
            markdown_table(
                ["case", "action", "operation", "precision", "metric", "Grid s", "Grid spread", "QUDA s", "QUDA spread", "Grid/QUDA", "check"],
                timing_rows(records),
            ).rstrip(),
            "",
            "## Matched CG solves",
            "",
            "Residual columns are the worst independently evaluated Grid-Schur residual over the measured repeats "
            "(not the warm correctness solve). `s/iter (internal)` is the harness's own per-repeat median where "
            "available; for QUDA this is derived from `QudaInvertParam::secs`, never from the wall-clock public-API "
            "timer, so it is not contaminated by the per-call PCIe transfer. `Grid/QUDA (internal)` divides Grid's "
            "wall time (no transfer step exists on that side) by QUDA's internal_s for the same reason -- it is only "
            "presented as a bare number when the harness marked the pair attributable to operator throughput; "
            "otherwise it carries an explicit annotation and is listed under \"Solve-ratio attribution\" below.",
            "",
            markdown_table(
                ["case", "action", "precision", "metric", "Grid s", "Grid iters", "Grid s/iter (internal)", "Grid residual", "QUDA wall s", "QUDA iters", "QUDA s/iter (internal)", "QUDA residual", "QUDA internal s", "Grid/QUDA (internal)", "check"],
                solve_rows(records),
            ).rstrip(),
            "",
            "## Grid mixed-precision CG solves",
            "",
            "Grid's own `MixedPrecisionConjugateGradient` (backend `grid_mixed`): double-outer/single-inner, "
            "run alongside (not paired with) the plain grid/quda solves above. Both ratio columns are wall-clock: "
            "`vs Grid(double)` compares against plain Grid double CG (no transfer step on either side); "
            "`vs QUDA(internal)` compares against QUDA's own internal_s for the same PCIe-contamination reason as "
            "the table above. Ratio > 1 means the mixed solve was slower than that reference.",
            "",
            markdown_table(
                ["case", "action", "precision", "metric", "Grid-mixed s", "inner iters", "restarts", "residual", "Grid(double) s", "QUDA internal s", "mixed/Grid(double)", "mixed/QUDA(internal)", "check"],
                mixed_solve_rows(records),
            ).rstrip(),
            "",
            "## Clover overhead within each backend",
            "",
            markdown_table(
                ["case", "operation", "backend", "precision", "Wilson s", "Clover s", "Clover/Wilson"],
                clover_ratio_rows(records),
            ).rstrip(),
            "",
            "## Setup, upload, and autotuning",
            "",
            markdown_table(
                ["case", "kind", "action", "stage", "backend", "metric", "seconds"],
                auxiliary_rows(records),
            ).rstrip(),
            "",
            "## Stock QUDA dslash_test (resident-device, separate binary)",
            "",
            "These come from a different process than the shared harness (no host-pointer public-API transfer). "
            "Match rows to the shared harness by action/op/precision only -- never ratio this `seconds` column "
            "directly against the shared harness's `public_api_end_to_end` QUDA timing above. `benchmark`/`verify` "
            "are the gtest per-test outcomes from dslash_test itself; `deviation` is its own CPU-reference L2 check, "
            "independent of this repository's correctness gates.",
            "",
            markdown_table(
                ["log", "action", "op", "precision", "niter", "s/call", "GFLOPS", "GB/s", "deviation", "benchmark", "verify"],
                resident_dslash_rows(records),
            ).rstrip(),
            "",
            "## QUDA autotuned kernel throughput (quda_resource/tunecache.tsv)",
            "",
            "Sourced from each run's own QUDA autotuning cache (located relative to its records.jsonl), not from "
            "the harness log. One row per distinct kernel/precision/reconstruct/parity/xpay/dagger combination "
            "QUDA's autotuner actually benchmarked in that run -- a run typically tunes more than one, since the "
            "sloppy-precision solve kernel and the full-precision correctness-check kernel are both tuned and "
            "both appear here. Not collapsed to a single \"representative\" figure: there is no reliable way from "
            "this file alone to tell which row was the CG inner loop, so pick the row matching the precision/"
            "reconstruct/parity you care about rather than trusting row order.",
            "",
            markdown_table(
                ["run dir", "kernel", "precision", "reconstruct", "parity", "xpay", "dagger", "Gflop/s", "GB/s"],
                flops_rows(records),
            ).rstrip(),
            "",
            "## Grid derived throughput (Grid time x QUDA operator-intrinsic flops/site)",
            "",
            "Grid does not print its own FLOP/s anywhere in this repository's logs (checked: the shared harness "
            "log, the harness source, and real production HMC logs -- confirmed absent by direct search, not "
            "assumed). This table is a DERIVED figure, not a native Grid measurement: it multiplies Grid's own "
            "measured wall time for its `normal_pc` operator (M_pc^dagger M_pc) by the \"flops per site\" constant "
            "a stock QUDA dslash_test run prints for the mathematically identical operator (dtest_type "
            "MatPCDagMatPC). That constant is a property of the operator's arithmetic, not of which code computes "
            "it, so pairing it with Grid's own time for the same operator is legitimate. Requires a dslash_test "
            "log passed via --dslash-log; without one, this table is empty rather than silently wrong.",
            "",
            markdown_table(
                ["case", "action", "precision", "metric", "Grid s", "local volume", "flops/site", "flops/site source", "Grid Gflop/s (derived)"],
                grid_derived_flops_rows(records),
            ).rstrip(),
            "",
        ]
    )

    if attribution_problems:
        output.extend(
            [
                "## Solve-ratio attribution",
                "",
                "These solve pairs must not be quoted as kernel-throughput results. Report convergence and "
                "throughput separately for them.",
                "",
            ]
        )
        for exemplar, grid, quda in attribution_problems:
            attributable = solve_attributable(grid, quda)
            reason = "harness marked non-attributable" if attributable is False else "no attribution flag recorded"
            span = display_value(exemplar, "iteration_relative_span")
            tolerance = display_value(exemplar, "iteration_tolerance")
            output.append(
                f"- NOT-ATTRIBUTABLE `{exemplar.get('_source')}`: {exemplar.get('action')}/solve "
                f"precision={exemplar.get('precision')} ({reason}); "
                f"Grid iters={iteration_cell(grid)}, QUDA iters={iteration_cell(quda)}, "
                f"relative span={span} tolerance={tolerance}"
            )
        output.append("")

    if failed or missing or dslash_problems:
        output.extend(["## Correctness problems", ""])
        for record in failed:
            output.append(
                f"- FAIL `{record.get('_source')}:{record.get('_line')}`: "
                f"{record.get('action')}/{record.get('op')} precision={record.get('precision')} "
                f"relative deviation={record.get('rel_deviation')} tolerance={record.get('tolerance')}"
            )
        for source, action, op, precision in missing:
            output.append(f"- MISSING: `{source}`: {action}/{op} precision={precision}")
        for record in dslash_problems:
            output.append(
                f"- DSLASH_TEST `{record.get('_source')}`: {record.get('action')}/{record.get('op')} "
                f"precision={record.get('precision')} benchmark={record.get('benchmark_status')} "
                f"verify={record.get('verify_status')}"
            )
        output.append("")

    return "\n".join(output)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records", nargs="*", default=[], type=Path,
        help="shared-harness JSONL files (optional if --dslash-log is given)",
    )
    parser.add_argument(
        "--dslash-log", dest="dslash_logs", nargs="*", default=[], type=Path,
        help="stock QUDA dslash_test log file(s), as written by run_benchmark.sh MODE=dslash "
             "(separate resident-device evidence, parsed independently of the shared JSONL harness)",
    )
    parser.add_argument("-o", "--output", type=Path, help="write Markdown here instead of stdout")
    parser.add_argument(
        "--allow-failed-correctness",
        action="store_true",
        help="return success even when a correctness check failed or is missing",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.records and not args.dslash_logs:
        print("summarize.py: ERROR: no input given; pass JSONL records and/or --dslash-log", file=sys.stderr)
        return 2
    try:
        records = load_records(args.records)
        for log_path in args.dslash_logs:
            records.append(parse_dslash_log(log_path))
        rendered = render_summary(records)
    except SummaryError as exc:
        print(f"summarize.py: ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)

    invalid = any(record.get("kind") == "correctness" and record.get("passed") is not True for record in records)
    missing = any(
        record.get("kind") in ("timing", "solve")
        and matching_check(records, record, correctness_op(record)) == "MISSING"
        for record in records
    )
    dslash_invalid = bool(resident_dslash_problems(records))
    return 1 if (invalid or missing or dslash_invalid) and not args.allow_failed_correctness else 0


if __name__ == "__main__":
    raise SystemExit(main())
