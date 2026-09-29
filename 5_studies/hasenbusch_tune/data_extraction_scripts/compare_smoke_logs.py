#!/usr/bin/env python3
"""Compare two HMC driver logs from identical-knob runs (e.g. two Grid builds).

Extracts, per monomial, the first NSAMPLES "Force average" values in order of
appearance, the trajectory-average FORCES line, Total H before/after and dH, and
the Plaquette lines, and prints relative differences |a-b|/max(|a|,|b|).

Usage: compare_smoke_logs.py <log_a> <log_b> [--samples N] [--tol-force 1e-9]
                             [--tol-dh 1e-6] [--tol-plaq 1e-8]
Exit 0 if every compared quantity is within tolerance, 1 otherwise.
"""
import argparse
import re
import sys
from collections import OrderedDict

RE_FORCE = re.compile(r"\[(\d+)\]\[(\d+)\] Force average: ([-+0-9.eE]+) (.*?)\s*$")
RE_HBEF = re.compile(r"Total H before trajectory = ([-+0-9.eE]+)")
RE_HAFT = re.compile(r"Total H after trajectory\s*= ([-+0-9.eE]+)\s+dH = ([-+0-9.eE]+)")
RE_PLAQ = re.compile(r"Plaquette: \[ \d+ \] ([-+0-9.eE]+)")
RE_FORCES = re.compile(r"FORCES traj=(\d+) (.*)$")


def parse(path):
    forces = OrderedDict()
    out = {"plaq": [], "forces_traj": OrderedDict()}
    with open(path, errors="replace") as f:
        for line in f:
            m = RE_FORCE.search(line)
            if m:
                key = f"[{m.group(1)}][{m.group(2)}] {m.group(4)}"
                forces.setdefault(key, []).append(float(m.group(3)))
                continue
            m = RE_HBEF.search(line)
            if m:
                out["H_before"] = float(m.group(1))
                continue
            m = RE_HAFT.search(line)
            if m:
                out["H_after"] = float(m.group(1))
                out["dH"] = float(m.group(2))
                continue
            m = RE_PLAQ.search(line)
            if m:
                out["plaq"].append(float(m.group(1)))
                continue
            m = RE_FORCES.search(line)
            if m:
                for kv in m.group(2).split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        try:
                            out["forces_traj"][f"traj{m.group(1)}:{k}"] = float(v)
                        except ValueError:
                            pass
    out["forces"] = forces
    return out


def rel(a, b):
    d = max(abs(a), abs(b))
    return abs(a - b) / d if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_a")
    ap.add_argument("log_b")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--tol-force", type=float, default=1e-9)
    ap.add_argument("--tol-dh", type=float, default=1e-6)
    ap.add_argument("--tol-plaq", type=float, default=1e-8)
    args = ap.parse_args()

    A, B = parse(args.log_a), parse(args.log_b)
    ok = True
    rows = []

    for key in A["forces"]:
        if key not in B["forces"]:
            rows.append((f"force {key}", "missing in B", "", "", "FAIL"))
            ok = False
            continue
        n = min(args.samples, len(A["forces"][key]), len(B["forces"][key]))
        for i in range(n):
            a, b = A["forces"][key][i], B["forces"][key][i]
            r = rel(a, b)
            flag = "ok" if r <= args.tol_force else "FAIL"
            ok &= flag == "ok"
            rows.append((f"force {key} s{i}", f"{a:.12g}", f"{b:.12g}", f"{r:.2e}", flag))

    for k in A["forces_traj"]:
        if k in B["forces_traj"]:
            a, b = A["forces_traj"][k], B["forces_traj"][k]
            r = rel(a, b)
            rows.append((f"FORCES {k}", f"{a:.12g}", f"{b:.12g}", f"{r:.2e}", "info"))

    for k, tol in (("H_before", args.tol_dh), ("H_after", args.tol_dh)):
        if k in A and k in B:
            r = rel(A[k], B[k])
            flag = "ok" if r <= tol else "FAIL"
            ok &= flag == "ok"
            rows.append((k, f"{A[k]:.15g}", f"{B[k]:.15g}", f"{r:.2e}", flag))
    if "dH" in A and "dH" in B:
        # dH is a small difference of large numbers: compare absolutely, scaled by H.
        scale = max(abs(A.get("H_before", 1.0)), 1.0)
        d = abs(A["dH"] - B["dH"]) / scale
        flag = "ok" if d <= args.tol_dh else "FAIL"
        ok &= flag == "ok"
        rows.append(("dH (|diff|/H)", f"{A['dH']:.12g}", f"{B['dH']:.12g}", f"{d:.2e}", flag))

    for i, (a, b) in enumerate(zip(A["plaq"], B["plaq"])):
        r = rel(a, b)
        flag = "ok" if r <= args.tol_plaq else "FAIL"
        ok &= flag == "ok"
        rows.append((f"plaquette[{i}]", f"{a:.15g}", f"{b:.15g}", f"{r:.2e}", flag))

    w = max(len(r[0]) for r in rows) if rows else 10
    print(f"{'quantity':<{w}}  {'A':>18}  {'B':>18}  {'rel':>9}  status")
    for r in rows:
        print(f"{r[0]:<{w}}  {r[1]:>18}  {r[2]:>18}  {r[3]:>9}  {r[4]}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
