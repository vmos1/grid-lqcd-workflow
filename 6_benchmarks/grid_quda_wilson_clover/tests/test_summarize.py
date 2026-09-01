#!/usr/bin/env python3

import importlib.util
import json
import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "summarize.py"
SPEC = importlib.util.spec_from_file_location("grid_quda_summarize", SCRIPT)
SUMMARY = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SUMMARY)


# A hand-built stand-in for what perlmutter/run_benchmark.sh's run_dslash()
# would tee to disk: its provenance/timing-scope/command header wrapped
# around stock QUDA dslash_test's own gtest + printfQuda stdout, following
# the literal format strings in quda/tests/dslash_test_utils.h (run_test(),
# verify()) and quda/tests/dslash_test.cpp (display_test_info()). Not yet
# cross-checked against a real run -- dslash_test is not built here.
SAMPLE_DSLASH_LOG = """--- provenance ---
ENV DATE=2026-08-25T12:00:00-07:00
ENV HOST=nid001000 SLURM_JOB_ID=12345
ENV MODE=dslash ACTION=clover DTEST_OP=normal_pc INPUT=hot PRECISION=strict
ENV LATT=48.48.48.96 MPI=1.2.2.4 MASS=-0.2416 CSW=1.20536588031793
ENV SAMPLES=7 WARMUPS=2 REPETITIONS=20 SOLVE_REPEATS=5 TOL=1e-10 MAXITER=50000
ENV NODES=4 NTASKS=16 NTPN=4 GPUS_PER_TASK=1 CPUS_PER_TASK=32
ENV BIN=/path/bin DTEST=/path/dslash_test CFG=/path/cfg
ENV RECORDS=/path/records.jsonl QUDA_RESOURCE_PATH=/path/quda_resource CACHE_STATE=empty
ENV GRID_SHA=abc123 QUDA_SHA=def456
ENV MODULES=PrgEnv-gnu,cudatoolkit/12.9,craype-accel-nvidia80
--- timing scope ---
resident_device_quda; generated hot gauge; --niter=20; one internal warm-up
--- command ---
srun ... dslash_test ...
--- output ---
[==========] Running 2 tests from 1 test suite.
[ RUN      ] DslashTest.benchmark
running the following test:
prec    recon   dtest_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension    dslash_type    niter
double   18       MatPCDagMatPC           ODD_ODD_ASYM    0     24/ 24/ 24         24              1          CLOVER_WILSON       20
Grid partition info:     X  Y  Z  T
                         1  2  2  4
Tuning...
Executing 20 kernel loops...
done.

123.456700us per kernel call
987654 flops per kernel call, 1234 flops per site 567 bytes per site
GFLOPS = 654.321000
GBYTES = 321.654000
[       OK ] DslashTest.benchmark (2000 ms)
[ RUN      ] DslashTest.verify
running the following test:
prec    recon   dtest_type     matpc_type   dagger   S_dim         T_dimension   Ls_dimension    dslash_type    niter
double   18       MatPCDagMatPC           ODD_ODD_ASYM    0     24/ 24/ 24         24              1          CLOVER_WILSON       20
Tuning...
Executing 2 kernel loops...
done.

Results: reference = 1.234500, QUDA = 1.234501, L2 relative deviation = 3.200000e-07, max deviation = 1.100000e-07
[       OK ] DslashTest.verify (50 ms)
[----------] 2 tests from DslashTest (2050 ms total)
[==========] 2 tests from 1 test suite ran. (2050 ms total)
[  PASSED  ] 2 tests.
"""


class SummarizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture_path = Path(
            "/dev/shm/grid_quda_summarize_test_{}.jsonl".format(os.getpid())
        )
        cls.log_fixture_path = Path(
            "/dev/shm/grid_quda_summarize_test_{}.log".format(os.getpid())
        )

    def write_records(self, records):
        with self.fixture_path.open("w", encoding="utf-8") as fixture:
            for record in records:
                fixture.write(json.dumps(record) + "\n")
        return self.fixture_path

    def write_text(self, text):
        self.log_fixture_path.write_text(text, encoding="utf-8")
        return self.log_fixture_path

    def test_pairs_backends_and_computes_ratios(self):
        common = {"input": "hot", "lattice": "8.8.8.16", "mpi": "1.1.1.1", "mass": -0.2416,
                  "precision": "strict", "cache_state": "warm", "action": "clover", "op": "normal_pc"}
        path = self.write_records([
            {**common, "kind": "timing", "backend": "grid", "mean_s": 4.0, "min_s": 3.9, "max_s": 4.1},
            {**common, "kind": "timing", "backend": "quda", "mean_s": 2.0, "min_s": 1.9, "max_s": 2.1},
            {**common, "kind": "correctness", "passed": True, "rel_deviation": 1e-13, "tolerance": 1e-10},
        ])
        rendered = SUMMARY.render_summary(SUMMARY.load_records([path]))
        self.assertIn("| 2 | PASS |", rendered)
        self.assertIn("mean", rendered)

    def test_failed_correctness_sets_nonzero_status(self):
        common = {"action": "wilson", "op": "dslash", "precision": "strict"}
        path = self.write_records([
            {**common, "kind": "timing", "backend": "grid", "mean_s": 1.0},
            {**common, "kind": "timing", "backend": "quda", "mean_s": 0.5},
            {**common, "kind": "correctness", "passed": False, "rel_deviation": 0.2, "tolerance": 1e-10},
        ])
        with redirect_stdout(io.StringIO()):
            self.assertEqual(SUMMARY.main([str(path)]), 1)
            self.assertEqual(SUMMARY.main([str(path), "--allow-failed-correctness"]), 0)

    def test_missing_correctness_is_visible(self):
        path = self.write_records([
            {"kind": "solve", "action": "wilson", "op": "solve", "backend": "grid",
             "precision": "strict", "mean_s": 2.0, "iterations": 20},
            {"kind": "solve", "action": "wilson", "op": "solve", "backend": "quda",
             "precision": "strict", "mean_s": 1.0, "iterations": 20},
        ])
        records = SUMMARY.load_records([path])
        rendered = SUMMARY.render_summary(records)
        self.assertIn("MISSING", rendered)
        self.assertIn("0.1", rendered)
        self.assertIn("0.05", rendered)

    def test_attributable_solve_ratio_is_presented_plainly(self):
        common = {"kind": "solve", "action": "clover", "op": "solve", "precision": "strict",
                  "iterations_constant": True, "attributable_to_operator_throughput": True}
        path = self.write_records([
            {**common, "backend": "grid", "median_s": 4.0, "iterations": 100,
             "s_per_iteration_median": 0.04, "independent_residual": 3e-11},
            {**common, "backend": "quda", "median_s": 2.0, "iterations": 100,
             "s_per_iteration_median": 0.02, "independent_residual": 4e-11},
            {"kind": "correctness", "action": "clover", "op": "solve", "precision": "strict", "passed": True},
        ])
        rendered = SUMMARY.render_summary(SUMMARY.load_records([path]))
        self.assertIn("| 2 | PASS |", rendered)
        self.assertNotIn("NOT attributable", rendered)
        self.assertNotIn("## Solve-ratio attribution", rendered)
        # The harness's own per-repeat median wins over median_s / iterations.
        self.assertIn("0.04", rendered)

    def test_solve_ratio_uses_internal_time_not_wall(self):
        # Regression test: QUDA's public invertQuda() API copies source/solution
        # over PCIe on every call, so wall time (median_s) is not a kernel-
        # throughput number. Here wall says Grid is faster (1.0/1.2 = 0.8333)
        # while internal says Grid is slower (1.0/0.5 = 2) -- the same sign
        # flip documented at 24^3x48 in the campaign plan. The rendered ratio,
        # and QUDA's rendered s/iter, must come from internal time only.
        common = {"kind": "solve", "action": "wilson", "op": "solve", "precision": "strict",
                  "iterations": 100, "iterations_constant": True,
                  "attributable_to_operator_throughput": True}
        path = self.write_records([
            {**common, "backend": "grid", "median_s": 1.0, "independent_residual": 1e-12},
            {**common, "backend": "quda", "median_s": 1.2, "internal_s": 0.5,
             "s_per_iteration_median_wall": 0.012, "s_per_iteration_median_internal": 0.005,
             "independent_residual": 1e-12},
            {"kind": "correctness", "action": "wilson", "op": "solve", "precision": "strict", "passed": True},
        ])
        rendered = SUMMARY.render_summary(SUMMARY.load_records([path]))
        self.assertIn("| 2 | PASS |", rendered)
        self.assertNotIn("0.8333", rendered)
        self.assertIn("0.005", rendered)
        self.assertNotIn("0.012", rendered)

    def test_non_attributable_solve_ratio_is_flagged(self):
        common = {"kind": "solve", "action": "wilson", "op": "solve", "precision": "strict",
                  "attributable_to_operator_throughput": False, "iteration_relative_span": 0.31,
                  "iteration_tolerance": 0.02}
        path = self.write_records([
            {**common, "backend": "grid", "median_s": 4.0, "iterations": 100,
             "iterations_constant": True, "iterations_min": 100, "iterations_max": 100},
            {**common, "backend": "quda", "median_s": 2.0, "iterations": 131,
             "iterations_constant": False, "iterations_min": 100, "iterations_max": 131},
            {"kind": "correctness", "action": "wilson", "op": "solve", "precision": "strict", "passed": True},
        ])
        rendered = SUMMARY.render_summary(SUMMARY.load_records([path]))
        self.assertIn("2 (NOT attributable)", rendered)
        self.assertIn("## Solve-ratio attribution", rendered)
        self.assertIn("NOT-ATTRIBUTABLE", rendered)
        self.assertIn("varies 100-131", rendered)

    def test_missing_attribution_flag_is_not_treated_as_attributable(self):
        common = {"kind": "solve", "action": "wilson", "op": "solve", "precision": "strict"}
        path = self.write_records([
            {**common, "backend": "grid", "median_s": 4.0, "iterations": 100},
            {**common, "backend": "quda", "median_s": 2.0, "iterations": 100},
            {"kind": "correctness", "action": "wilson", "op": "solve", "precision": "strict", "passed": True},
        ])
        rendered = SUMMARY.render_summary(SUMMARY.load_records([path]))
        self.assertIn("attribution unknown", rendered)
        self.assertIn("no attribution flag recorded", rendered)

    def test_duplicate_backend_is_rejected(self):
        common = {"kind": "timing", "action": "wilson", "op": "mat", "backend": "grid"}
        path = self.write_records([{**common, "mean_s": 1.0}, {**common, "mean_s": 1.1}])
        with self.assertRaises(SUMMARY.SummaryError):
            SUMMARY.render_summary(SUMMARY.load_records([path]))

    def test_parse_dslash_log_extracts_metrics(self):
        path = self.write_text(SAMPLE_DSLASH_LOG)
        record = SUMMARY.parse_dslash_log(path)
        self.assertEqual(record["kind"], "resident_dslash")
        self.assertEqual(record["action"], "clover")
        self.assertEqual(record["op"], "normal_pc")
        self.assertEqual(record["precision"], "strict")
        self.assertEqual(record["benchmark_status"], "OK")
        self.assertEqual(record["verify_status"], "OK")
        self.assertEqual(record["dtest_type"], "MatPCDagMatPC")
        self.assertEqual(record["dslash_type"], "CLOVER_WILSON")
        self.assertAlmostEqual(record["seconds"], 123.4567e-6)
        self.assertAlmostEqual(record["gflops"], 654.321)
        self.assertAlmostEqual(record["gbytes"], 321.654)
        self.assertAlmostEqual(record["deviation"], 3.2e-07)

    def test_dslash_log_renders_and_passes(self):
        path = self.write_text(SAMPLE_DSLASH_LOG)
        with redirect_stdout(io.StringIO()):
            self.assertEqual(SUMMARY.main(["--dslash-log", str(path)]), 0)
        record = SUMMARY.parse_dslash_log(path)
        rendered = SUMMARY.render_summary([record])
        self.assertIn("dslash_test", rendered)
        self.assertIn("clover", rendered)

    def test_dslash_log_failed_verify_sets_nonzero_status(self):
        text = SAMPLE_DSLASH_LOG.replace(
            "[       OK ] DslashTest.verify (50 ms)", "[  FAILED  ] DslashTest.verify (50 ms)"
        )
        path = self.write_text(text)
        record = SUMMARY.parse_dslash_log(path)
        self.assertEqual(record["verify_status"], "FAILED")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(SUMMARY.main(["--dslash-log", str(path)]), 1)
            self.assertEqual(SUMMARY.main(["--dslash-log", str(path), "--allow-failed-correctness"]), 0)

    def test_no_input_is_an_error(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(SUMMARY.main([]), 2)


if __name__ == "__main__":
    unittest.main()
