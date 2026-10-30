#!/usr/bin/env python3
"""Correct Grid's misprinted log timestamps (L198, 2026-10-30).

Grid prints the time since start as `<seconds>.<microseconds> s`, the microseconds padded to six
digits on the WRONG side: 200.013305 s is printed "200.133050 s". The leading zeros of the
fraction move to its end, so a printed fraction ending in k zeros is one of k+1 values (shift
right by 0..k). A fraction that does not end in 0 is always exact. Any time taken as a difference
of two printed stamps can be off by up to 0.9 s per stamp.

StampFixer resolves an ambiguous stamp from its nearest exact neighbours: the true value lies
between the last exact stamp before it and the first exact stamp after it, and in a log written
line by line those are milliseconds apart, so one candidate survives. Feed every stamp in file
order; a value registered with want() is filled in once the next exact stamp arrives (call
finish() at end of file).

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


class StampFixer:
    def __init__(self):
        self.last_exact, self.pending = None, []

    def feed(self, raw):
        """Every printed stamp, in file order."""
        if is_exact(raw):
            v = float(raw)
            for d, key, r, lo in self.pending:
                d[key] = resolve(r, lo, v)
            self.pending = []
            self.last_exact = v

    def want(self, d, key, raw):
        """Set d[key] to the corrected value of `raw` (now if exact, else at the next exact
        stamp). Call before feed(raw) for the same line."""
        if is_exact(raw):
            d[key] = float(raw)
        else:
            d[key] = float(raw)  # provisional
            self.pending.append((d, key, raw, self.last_exact))

    def finish(self):
        for d, key, r, lo in self.pending:
            d[key] = resolve(r, lo, None)
        self.pending = []


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
    print("grid_log_time: self-test passed")
