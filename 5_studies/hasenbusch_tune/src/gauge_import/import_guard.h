#pragma once
// C1: exact-skip guard for FermOp.ImportGauge(U).
//
// GuardedImportGauge(op, U): when HASEN_GRID_IMPORT_SKIP=1 (read once, default OFF), skip
// op.ImportGauge(U) iff norm2(U - Ulast[op]) == 0.0 exactly, where Ulast[op] is a shadow copy
// of the last field imported through THIS guard into THAT operator object (keyed by &op).
// Bitwise-equal field => identical import => results are bit-identical to the switch-OFF run.
// Switch OFF: plain op.ImportGauge(U), no shadow, no counters.
//
// Why a shadow copy: WilsonFermion::Umu (public) is the DOUBLED -0.5*U store with boundary
// phases, not comparable to U without redoing the doubling (which is the cost we avoid).
//
// Hazards (documented, by design): (1) an ImportGauge into the same operator object that does
// NOT go through this guard leaves the shadow stale; (2) an operator destroyed and a new one
// allocated at the same address inherits the old shadow. Both are absent in the production
// driver as audited (operators live for the whole run; see report).
//
// HASEN_GRID_IMPORT_SKIP_VERBOSE=1: one line per guarded call on the boss rank with cumulative
// skipped/performed counts for that operator.
#include <Grid/Grid.h>
#include <cstdlib>
#include <map>
#include <memory>

namespace Grid {

inline bool ImportSkipEnv(const char *name) {
  const char *e = std::getenv(name);
  return e && e[0] == '1';
}
inline bool ImportSkipEnabled() {
  static const bool on = ImportSkipEnv("HASEN_GRID_IMPORT_SKIP");
  return on;
}
inline bool ImportSkipVerbose() {
  static const bool on = ImportSkipEnv("HASEN_GRID_IMPORT_SKIP_VERBOSE");
  return on;
}

template <class GF>
struct ImportGuardState {
  struct Entry {
    std::unique_ptr<GF> last;
    long skipped = 0;
    long performed = 0;
  };
  std::map<const void *, Entry> reg;
};

template <class GF>
inline ImportGuardState<GF> &ImportGuardRegistry() {
  static ImportGuardState<GF> s;
  return s;
}

template <class Op, class GF>
void GuardedImportGauge(Op &op, const GF &U) {
  if (!ImportSkipEnabled()) {
    op.ImportGauge(U);
    return;
  }
  auto &e = ImportGuardRegistry<GF>().reg[static_cast<const void *>(&op)];
  bool same = false;
  if (e.last && e.last->Grid() == U.Grid()) {
    // Exact device-side test, one link direction at a time (temp = 1/4 of a GaugeField).
    same = true;
    for (int mu = 0; mu < Nd && same; ++mu) {
      auto d = PeekIndex<LorentzIndex>(U, mu);
      d = d - PeekIndex<LorentzIndex>(*e.last, mu);
      if (norm2(d) != 0.0) same = false;
    }
  }
  if (same) {
    ++e.skipped;
  } else {
    op.ImportGauge(U);
    if (!e.last || e.last->Grid() != U.Grid()) e.last.reset(new GF(U.Grid()));
    *e.last = U;
    ++e.performed;
  }
  if (ImportSkipVerbose() && U.Grid()->IsBoss()) {
    std::cout << GridLogMessage << "ImportGuard op=" << static_cast<const void *>(&op)
              << (same ? " SKIP" : " IMPORT") << " cum_skipped=" << e.skipped
              << " cum_performed=" << e.performed << std::endl;
  }
}

}  // namespace Grid
