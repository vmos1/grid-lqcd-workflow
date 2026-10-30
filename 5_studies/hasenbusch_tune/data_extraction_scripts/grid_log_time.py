#!/usr/bin/env python3
"""Correct Grid's misprinted log timestamps (L198, 2026-10-30).

Grid prints the time since start as `<seconds>.<microseconds> s`, the microseconds padded to six
digits on the WRONG side: 200.013305 s is printed "200.133050 s". The leading zeros of the
fraction move to its end, so a printed fraction ending in k zeros is one of k+1 values (shift
right by 0..k). A fraction that does not end in 0 is always exact. Any time taken as a difference
of two printed stamps can be off by up to 0.9 s per stamp.

StampFixer resolves ambiguous stamps from their exact neighbours. Stamps are non-decreasing, so a
whole run of ambiguous stamps between two exact ones is resolved together: walking forward from
the last exact stamp, each takes its smallest candidate not below the previous value (the
misprint only ever makes a stamp larger). Resolving each stamp alone against the two exact
neighbours is not enough: in 3287.776044 (exact), 3288.282580, 3288.282780, 3288.282800, ...,
3288.805507 (exact) every printed value fits between the neighbours, but the run is
3288.028258, 3288.028278, 3288.028280 (found 2026-10-30 by a review; up to 0.56 s per stamp). If
the walk overshoots the next exact stamp the run keeps its printed values. Feed every stamp in
file order; a value registered with want() is filled in when its run closes (call finish() at end
of file).

Self-test:  python3 grid_log_time.py
"""


def candidates(raw):
    """All values a printed stamp can stand for, largest (as printed) first."""
    sec, _, frac = raw.partition(".")
    if not frac:
        return [float(raw)]
    frac = frac.ljust(6, "0")[:6]
    k = len(frac) - len(frac.rstrip("0"))
    return [int(sec) + int(("0" * z + frac)[:6]) / 1e6 for z in range(k + 1)]


def is_exact(raw):
    _, _, frac = raw.partition(".")
    return bool(frac) and not frac.endswith("0")


def resolve(raw, lo, hi):
    """The candidate of `raw` inside [lo, hi] (None = no bound); the printed value if none fits,
    the largest that fits if several do."""
    c = candidates(raw)
    ok = [v for v in c if (lo is None or v >= lo) and (hi is None or v <= hi)]
    return ok[0] if ok else c[0]


def resolve_run(raws, lo, hi):
    """Values for a run of ambiguous stamps between exact stamps lo and hi (either may be None)."""
    out, prev = [], lo
    for r in raws:
        c = sorted(candidates(r))
        ok = [v for v in c if prev is None or v >= prev]
        v = ok[0] if ok else c[-1]
        out.append(v)
        prev = v
    if hi is not None and out and out[-1] > hi:
        return [float(r) for r in raws]
    return out


class StampFixer:
    def __init__(self):
        self.last_exact, self.run, self.pending = None, [], []

    def _close(self, hi):
        vals = resolve_run(self.run, self.last_exact, hi)
        for d, key, i in self.pending:
            d[key] = vals[i]
        self.run, self.pending = [], []

    def feed(self, raw):
        """Every printed stamp, in file order."""
        if is_exact(raw):
            v = float(raw)
            self._close(v)
            self.last_exact = v
        else:
            self.run.append(raw)

    def want(self, d, key, raw):
        """Set d[key] to the corrected value of `raw` (now if exact, else when its run of
        ambiguous stamps closes). Call before feed(raw) for the same line."""
        d[key] = float(raw)  # final if exact, provisional otherwise
        if not is_exact(raw):
            self.pending.append((d, key, len(self.run)))

    def finish(self):
        self._close(None)


if __name__ == "__main__":
    # misprinted 200.013305 between exact neighbours
    assert abs(resolve("200.133050", 200.012001, 200.014001) - 200.013305) < 1e-9
    # a genuine trailing zero stays
    assert abs(resolve("200.133050", 200.133001, 200.134001) - 200.133050) < 1e-9
    # two leading zeros
    assert abs(resolve("37.125000", 37.001, 37.002) - 37.001250) < 1e-9
    d, f = {}, StampFixer()
    # true 256.089440 printed "256.894400", between exact stamps 256.088123 and 256.091111
    for raw in ["256.088123", "256.894400", "256.091111"]:
        if raw == "256.894400":
            f.want(d, "t", raw)
        f.feed(raw)
    assert abs(d["t"] - 256.089440) < 1e-9, d
    # a run of misprints (2026_10_24_B2_tail24_s301_from2008/hmc.log, lines 24307-24313); the
    # wanted stamp is the third of the run, true value 3288.028280
    d, f = {}, StampFixer()
    run = ["3287.776044", "3288.282580", "3288.282780", "3288.282800", "3288.282830",
           "3288.282840", "3288.805507"]
    for raw in run:
        if raw == "3288.282800":
            f.want(d, "t", raw)
        f.feed(raw)
    assert abs(d["t"] - 3288.028280) < 1e-9, d
    print("grid_log_time: self-test passed")
