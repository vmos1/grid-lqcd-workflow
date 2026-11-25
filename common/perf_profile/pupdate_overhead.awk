# pupdate_overhead.awk -- per force piece: the time of the momentum update that is NOT the
# force itself. Reads the integrator's lines
#   [i][j] P update elapsed time: X ms (force: Y ms) ...
#   [l] batched smeared P update elapsed time: X ms (pullback+Ta: Y ms)
# and prints, per piece: calls, sum X, sum Y, sum (X - Y), mean (X - Y) per call, all in s / ms.
# Usage: awk -f pupdate_overhead.awk hmc.log
/P update elapsed time:/ {
  key = ""
  for (i = 1; i <= NF; i++) {
    if ($i == "P" && $(i+1) == "update") {
      if ($(i-1) == "smeared") key = $(i-3) " pullback"
      else key = $(i-1)
    }
    if ($i == "time:") x = $(i+1)
    if ($i == "(force:" || $i == "(pullback+Ta:") y = $(i+1)
  }
  n[key]++
  sx[key] += x / 1000.0
  sy[key] += y / 1000.0
  if (!(key in seen)) { seen[key] = 1; order[++nk] = key }
}
END {
  printf "%-16s %6s %10s %10s %10s %10s\n", "piece", "calls", "update s", "force s", "other s", "other ms/call"
  for (k = 1; k <= nk; k++) {
    key = order[k]
    o = sx[key] - sy[key]
    printf "%-16s %6d %10.1f %10.1f %10.1f %10.1f\n", key, n[key], sx[key], sy[key], o, 1000.0 * o / n[key]
    tn += n[key]; tx += sx[key]; ty += sy[key]
  }
  printf "%-16s %6d %10.1f %10.1f %10.1f %10.1f\n", "total", tn, tx, ty, tx - ty, 1000.0 * (tx - ty) / tn
}
