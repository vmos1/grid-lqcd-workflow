# mg_solve_sum.awk -- sums the multigrid solve lines of an HMC log per pseudofermion rung:
#   [GridMG PFn] solve i/2 iters N ... seconds S ... vcycle_applies fine F coarse C
# Prints solves, outer iterations, seconds, fine and coarse operator applications, and the
# time per fine application (ms). Usage: awk -f mg_solve_sum.awk hmc.log
/\[GridMG PF[0-9]+\] solve / {
  key = ""
  for (i = 1; i <= NF; i++) {
    if ($i == "[GridMG") key = $(i+1)
    if ($i == "iters") it = $(i+1)
    if ($i == "seconds") s = $(i+1)
    if ($i == "fine") f = $(i+1)
    if ($i == "coarse") c = $(i+1)
  }
  sub(/\]/, "", key)
  n[key]++; si[key] += it; ss[key] += s; sf[key] += f; sc[key] += c
  if (!(key in seen)) { seen[key] = 1; order[++nk] = key }
}
END {
  printf "%-6s %7s %8s %10s %9s %9s %12s %14s\n", "rung", "solves", "iters", "seconds", "fine", "coarse", "ms per fine", "coarse per it"
  for (k = 1; k <= nk; k++) {
    key = order[k]
    printf "%-6s %7d %8d %10.1f %9d %9d %12.3f %14.2f\n", key, n[key], si[key], ss[key], sf[key], sc[key], 1000.0 * ss[key] / sf[key], sc[key] / si[key]
    tn += n[key]; ti += si[key]; ts += ss[key]; tf += sf[key]; tc += sc[key]
  }
  printf "%-6s %7d %8d %10.1f %9d %9d %12.3f %14.2f\n", "total", tn, ti, ts, tf, tc, 1000.0 * ts / tf, tc / ti
}
