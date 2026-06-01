// 2+1 flavor Wilson Clover HMC — Hasenbusch mass tuning.
//
// Splits the 2-flavor light quark determinant into 5 ratio levels to measure
// per-level MD forces. Run 10 short trajectories, read back FORCES lines, and
// adjust the 4 intermediate Hasenbusch masses until forces are equalized.
//
// Hasenbusch chain (Nf=2 light):
//   PF0: det(M_l)  / det(M_H1)   <- lightest ratio (hardest CG, smallest force)
//   PF1: det(M_H1) / det(M_H2)
//   PF2: det(M_H2) / det(M_H3)
//   PF3: det(M_H3) / det(M_H4)
//   PF4: det(M_H4) / det(M_PV)   <- heaviest ratio (cheapest CG, largest force)
//
// Constructor convention (verified from TwoFlavourRatio.h):
//   TwoFlavourRatioPseudoFermionAction(NumOp=heavier, DenOp=lighter)
//   represents det(lighter)/det(heavier) for Nf=2.
//
// Strange Nf=1: OneFlavourRatioRationalPseudoFermionAction (full-lattice RHMC).
// Note: TwoFlavourEvenOddRatioPseudoFermionAction asserts ConstEE()==1, which
// WilsonCloverFermion never satisfies — use full-lattice ratio actions here.
//
// Build against install-grid-gpu (no TXQCD fork needed).

#include <Grid/Grid.h>
#include <Grid/parallelIO/IldgIO.h>
#include <iomanip>

using namespace Grid;

// ---------------------------------------------------------------------------
// XML parameter structs
// ---------------------------------------------------------------------------

struct CloverFermionParameters : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(CloverFermionParameters,
                                  double, mass,
                                  double, csw_r,
                                  double, csw_t,
                                  double, StoppingCondition,
                                  double, MDStoppingCondition,
                                  int,    MaxCGIterations);
};

struct RHMCParameters : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(RHMCParameters,
                                  double, lo,
                                  double, hi,
                                  int,    degree,
                                  int,    MaxIter,
                                  double, tolerance,
                                  double, mdtolerance,
                                  int,    precision,
                                  int,    BoundsCheckFreq);
};

// Four intermediate Hasenbusch masses between m_light and m_PV=1.0.
// Tune iteratively until per-level forces are equalized.
struct HasenbuschParameters : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(HasenbuschParameters,
                                  double, m_H1,
                                  double, m_H2,
                                  double, m_H3,
                                  double, m_H4);
};

struct WilsonCloverHasenbuschHMCParameters : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(WilsonCloverHasenbuschHMCParameters,
                                  double,                  gauge_beta,
                                  bool,                    ApplySmearing,
                                  CloverFermionParameters, Light,
                                  HasenbuschParameters,    Hasenbusch,
                                  CloverFermionParameters, Strange,
                                  RHMCParameters,          StrangeRHMC)

  template <class ReaderClass>
  WilsonCloverHasenbuschHMCParameters(Reader<ReaderClass> &Reader) {
    read(Reader, "Action", *this);
  }
};

struct SmearingParam : Serializable {
  GRID_SERIALIZABLE_CLASS_MEMBERS(SmearingParam, double, rho, Integer, Nsmear)
  template <class ReaderClass>
  SmearingParam(Reader<ReaderClass> &Reader) {
    read(Reader, "StoutSmearing", *this);
  }
};

// ---------------------------------------------------------------------------
// ForceNormObserver — prints one FORCES line per trajectory.
//
// Uses deriv_norm_average() from ActionBase: per-site RMS force averaged over
// all MD steps in the trajectory (reset at trajectory start by the integrator).
// Tag "FORCES" makes lines grep-able: grep "^.*FORCES" logfile
// ---------------------------------------------------------------------------

class ForceNormObserver : public HmcObservable<LatticeGaugeField> {
 public:
  struct ActionRef {
    std::string name;
    Action<LatticeGaugeField> *act;
  };

  explicit ForceNormObserver(std::vector<ActionRef> actions)
      : actions_(std::move(actions)) {}

  void TrajectoryComplete(int traj, LatticeGaugeField &,
                          GridSerialRNG &, GridParallelRNG &) override {
    std::cout << GridLogMessage << "FORCES traj=" << traj;
    for (auto &ar : actions_) {
      RealD f = (ar.act->deriv_num > 0) ? ar.act->deriv_norm_average() : 0.0;
      std::cout << " " << ar.name << "="
                << std::scientific << std::setprecision(4) << f;
    }
    std::cout << std::endl;
  }

  void print_parameters() {}

 private:
  std::vector<ActionRef> actions_;
};

// Module wrapper so ForceNormObserver can be registered via
// TheHMC.Resources.AddObservable<ForceNormModule>(&obs).
// HMCModuleBase<Prod> requires virtual Prod* getPtr().
class ForceNormModule
    : public HMCModuleBase<HmcObservable<LatticeGaugeField>> {
 public:
  explicit ForceNormModule(ForceNormObserver *obs) : obs_(obs) {}
  HmcObservable<LatticeGaugeField> *getPtr() override { return obs_; }
  void print_parameters() override { obs_->print_parameters(); }

 private:
  ForceNormObserver *obs_;
};

// ---------------------------------------------------------------------------
// Config reader that tolerates missing ildg-lfn (Chroma/SciDAC LIME files).
// Replicates IldgReader::readConfiguration without GRID_ASSERT(found_ildgLFN).
// ---------------------------------------------------------------------------
static void readChromaLimeConfig(LatticeGaugeField &Umu, FieldMetaData &FieldMetaData_,
                                  const std::string &import_cfg) {
  typedef LatticeGaugeField GaugeField;
  typedef typename GaugeField::vector_object vobj;
  typedef typename vobj::scalar_object       sobj;
  typedef LorentzColourMatrixF fobj;
  typedef LorentzColourMatrixD dobj;

  IldgReader reader;
  reader.open(import_cfg);

  Coordinate dims = Umu.Grid()->FullDimensions();
  GRID_ASSERT(dims.size() == 4);

  ildgFormat        ildgFormat_;
  scidacChecksum    scidacChecksum_;
  usqcdInfo         usqcdInfo_;
  FieldNormMetaData FieldNormMetaData_;

  int found_ildgFormat     = 0;
  int found_ildgLFN        = 0;
  int found_scidacChecksum = 0;
  int found_ildgBinary     = 0;
  int found_FieldMetaData  = 0;
  uint32_t nersc_csum, scidac_csuma, scidac_csumb;
  std::string format;

  while (limeReaderNextRecord(reader.LimeR) == LIME_SUCCESS) {
    uint64_t nbytes = limeReaderBytes(reader.LimeR);

    if (strncmp(limeReaderType(reader.LimeR), ILDG_BINARY_DATA, strlen(ILDG_BINARY_DATA))) {
      std::vector<char> xmlc(nbytes + 1, '\0');
      limeReaderReadData((void *)&xmlc[0], &nbytes, reader.LimeR);
      std::string xmlstring(&xmlc[0]);

      if (!strncmp(limeReaderType(reader.LimeR), ILDG_FORMAT, strlen(ILDG_FORMAT))) {
        XmlReader RD(xmlstring, true, "");
        read(RD, "ildgFormat", ildgFormat_);
        if (ildgFormat_.precision == 64) format = "IEEE64BIG";
        if (ildgFormat_.precision == 32) format = "IEEE32BIG";
        GRID_ASSERT(ildgFormat_.lx == dims[0]);
        GRID_ASSERT(ildgFormat_.ly == dims[1]);
        GRID_ASSERT(ildgFormat_.lz == dims[2]);
        GRID_ASSERT(ildgFormat_.lt == dims[3]);
        found_ildgFormat = 1;
      }
      if (!strncmp(limeReaderType(reader.LimeR), ILDG_DATA_LFN, strlen(ILDG_DATA_LFN))) {
        FieldMetaData_.ildg_lfn = xmlstring;
        found_ildgLFN = 1;
      }
      if (!strncmp(limeReaderType(reader.LimeR), GRID_FORMAT, strlen(ILDG_FORMAT))) {
        XmlReader RD(xmlstring, true, "");
        read(RD, "FieldMetaData", FieldMetaData_);
        format = FieldMetaData_.floating_point;
        GRID_ASSERT(FieldMetaData_.dimension[0] == dims[0]);
        GRID_ASSERT(FieldMetaData_.dimension[1] == dims[1]);
        GRID_ASSERT(FieldMetaData_.dimension[2] == dims[2]);
        GRID_ASSERT(FieldMetaData_.dimension[3] == dims[3]);
        found_FieldMetaData = 1;
      }
      if (!strncmp(limeReaderType(reader.LimeR), SCIDAC_RECORD_XML, strlen(SCIDAC_RECORD_XML))) {
        if (xmlstring.find("usqcdInfo") != std::string::npos) {
          XmlReader RD(xmlstring, true, "");
          read(RD, "usqcdInfo", usqcdInfo_);
        }
      }
      if (!strncmp(limeReaderType(reader.LimeR), SCIDAC_CHECKSUM, strlen(SCIDAC_CHECKSUM))) {
        XmlReader RD(xmlstring, true, "");
        read(RD, "scidacChecksum", scidacChecksum_);
        found_scidacChecksum = 1;
      }
      if (!strncmp(limeReaderType(reader.LimeR), GRID_FIELD_NORM, strlen(GRID_FIELD_NORM))) {
        XmlReader RD(xmlstring, true, "");
        read(RD, GRID_FIELD_NORM, FieldNormMetaData_);
      }
    } else {
      uint64_t offset = ftello(reader.File);
      if (format == std::string("IEEE64BIG")) {
        GaugeSimpleMunger<dobj, sobj> munge;
        BinaryIO::readLatticeObject<vobj, dobj>(Umu, reader.filename, munge, offset, format,
                                                nersc_csum, scidac_csuma, scidac_csumb);
      } else {
        GaugeSimpleMunger<fobj, sobj> munge;
        BinaryIO::readLatticeObject<vobj, fobj>(Umu, reader.filename, munge, offset, format,
                                                nersc_csum, scidac_csuma, scidac_csumb);
      }
      found_ildgBinary = 1;
    }
  }
  reader.close();

  if (!found_ildgLFN)
    std::cout << GridLogMessage << "Note: ildg-lfn record absent (Chroma/SciDAC config); OK." << std::endl;
  GRID_ASSERT(found_ildgBinary);
  GRID_ASSERT(found_ildgFormat);
  GRID_ASSERT(found_scidacChecksum);
  GRID_ASSERT(found_FieldMetaData || found_ildgFormat);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  Grid_init(&argc, &argv);
  std::cout << GridLogMessage << "Grid is setup to use "
            << GridThread::GetThreads() << " threads" << std::endl;

  typedef GenericHMCRunner<MinimumNorm2>         HMCWrapper;
  typedef WilsonImplR                            FermionImplPolicy;
  typedef WilsonCloverFermionD                   FermionAction;
  typedef typename FermionAction::FermionField   FermionField;
  typedef Grid::XmlReader                        Serialiser;

  // Parse --ConfigFile <path> to load a thermalized starting config (ILDG/LIME format).
  // If not given, falls back to StartingType from XML (e.g. ColdStart).
  std::string import_cfg;
  for (int i = 1; i < argc - 1; ++i)
    if (std::string(argv[i]) == "--ConfigFile") { import_cfg = argv[i+1]; break; }

  HMCWrapper TheHMC;
  TheHMC.ReadCommandLine(argc, argv);

  if (TheHMC.ParameterFile.empty()) {
    std::cout << "Input file not specified. Use --ParameterFile.\nAborting"
              << std::endl;
    exit(1);
  }
  Serialiser Reader(TheHMC.ParameterFile);
  WilsonCloverHasenbuschHMCParameters MyParams(Reader);

  TheHMC.Resources.AddFourDimGrid("gauge");
  auto GridPtr_early = TheHMC.Resources.GetCartesian();

  // If a config file is given: read it (ILDG LIME) and write a temp NERSC
  // checkpoint so GenericHMCRunner can load it via CheckpointStart.
  CheckpointerParameters CPparams(Reader);
  RNGModuleParameters    RNGpar(Reader);

  if (!import_cfg.empty()) {
    std::cout << GridLogMessage << "Importing starting config: " << import_cfg << std::endl;

    LatticeGaugeField Utmp(GridPtr_early);
    FieldMetaData fmd;
    readChromaLimeConfig(Utmp, fmd, import_cfg);
    std::cout << GridLogMessage << "Config read OK. Writing temp checkpoint..." << std::endl;

    std::string cfg_tmp = "/tmp/hb_import_cfg";
    std::string rng_tmp = "/tmp/hb_import_rng";

    NerscIO::writeConfiguration(Utmp, cfg_tmp + ".0");

    GridSerialRNG  sRNG_tmp;
    GridParallelRNG pRNG_tmp(GridPtr_early);
    sRNG_tmp.SeedFixedIntegers(RNGpar.getSerialSeeds());
    pRNG_tmp.SeedFixedIntegers(RNGpar.getParallelSeeds());
    NerscIO::writeRNGState(sRNG_tmp, pRNG_tmp, rng_tmp + ".0");

    CPparams.config_prefix = cfg_tmp;
    CPparams.rng_prefix    = rng_tmp;
    std::cout << GridLogMessage << "Temp checkpoint written." << std::endl;
  }

  TheHMC.Resources.LoadNerscCheckpointer(CPparams);
  TheHMC.Resources.SetRNGSeeds(RNGpar);

  typedef PlaquetteMod<HMCWrapper::ImplPolicy>   PlaqObs;
  typedef PolyakovMod<HMCWrapper::ImplPolicy>    PolyakovObs;
  TheHMC.Resources.AddObservable<PlaqObs>();
  TheHMC.Resources.AddObservable<PolyakovObs>();

  WilsonGaugeActionR Waction(MyParams.gauge_beta);

  auto GridPtr   = TheHMC.Resources.GetCartesian();
  auto GridRBPtr = TheHMC.Resources.GetRBCartesian();
  LatticeGaugeField U(GridPtr);

  // Periodic xyz, antiperiodic t (chroma convention)
  std::vector<Complex> boundary = {1, 1, 1, -1};
  FermionAction::ImplParams FermParams(boundary);
  WilsonAnisotropyCoefficients anis;

  // ── Masses ────────────────────────────────────────────────────────────────
  Real m_l  = MyParams.Light.mass;
  Real m_H1 = MyParams.Hasenbusch.m_H1;
  Real m_H2 = MyParams.Hasenbusch.m_H2;
  Real m_H3 = MyParams.Hasenbusch.m_H3;
  Real m_H4 = MyParams.Hasenbusch.m_H4;
  Real m_s  = MyParams.Strange.mass;
  Real pv   = 1.0;

  Real csw_l = MyParams.Light.csw_r;
  Real csw_s = MyParams.Strange.csw_r;

  std::cout << GridLogMessage << "Hasenbusch chain:" << std::endl;
  std::cout << GridLogMessage << "  m_l  = " << m_l  << std::endl;
  std::cout << GridLogMessage << "  m_H1 = " << m_H1 << std::endl;
  std::cout << GridLogMessage << "  m_H2 = " << m_H2 << std::endl;
  std::cout << GridLogMessage << "  m_H3 = " << m_H3 << std::endl;
  std::cout << GridLogMessage << "  m_H4 = " << m_H4 << std::endl;
  std::cout << GridLogMessage << "  m_PV = " << pv   << std::endl;
  std::cout << GridLogMessage << "  m_s  = " << m_s  << std::endl;

  // ── Fermion operators ─────────────────────────────────────────────────────
  // 6 operators for the 5-level light Hasenbusch chain
  FermionAction Op_l  (U, *GridPtr, *GridRBPtr, m_l,  csw_l, csw_l, anis, FermParams);
  FermionAction Op_H1 (U, *GridPtr, *GridRBPtr, m_H1, csw_l, csw_l, anis, FermParams);
  FermionAction Op_H2 (U, *GridPtr, *GridRBPtr, m_H2, csw_l, csw_l, anis, FermParams);
  FermionAction Op_H3 (U, *GridPtr, *GridRBPtr, m_H3, csw_l, csw_l, anis, FermParams);
  FermionAction Op_H4 (U, *GridPtr, *GridRBPtr, m_H4, csw_l, csw_l, anis, FermParams);
  FermionAction Op_PV (U, *GridPtr, *GridRBPtr, pv,   csw_l, csw_l, anis, FermParams);
  // Strange + its PV
  FermionAction Op_s  (U, *GridPtr, *GridRBPtr, m_s,  csw_s, csw_s, anis, FermParams);
  FermionAction Op_sPV(U, *GridPtr, *GridRBPtr, pv,   csw_s, csw_s, anis, FermParams);

  // ── CG solver ────────────────────────────────────────────────────────────
  // Single tolerance for all Hasenbusch levels: force measurement, not production.
  ConjugateGradient<FermionField> CG(MyParams.Light.StoppingCondition,
                                     MyParams.Light.MaxCGIterations);

  // ── RHMC for strange (Nf=1, full-lattice) ────────────────────────────────
  OneFlavourRationalParams OFRp;
  OFRp.lo              = MyParams.StrangeRHMC.lo;
  OFRp.hi              = MyParams.StrangeRHMC.hi;
  OFRp.MaxIter         = MyParams.StrangeRHMC.MaxIter;
  OFRp.tolerance       = MyParams.StrangeRHMC.tolerance;
  OFRp.mdtolerance     = MyParams.StrangeRHMC.mdtolerance;
  OFRp.degree          = MyParams.StrangeRHMC.degree;
  OFRp.precision       = MyParams.StrangeRHMC.precision;
  OFRp.BoundsCheckFreq = MyParams.StrangeRHMC.BoundsCheckFreq;

  // ── Hasenbusch pseudofermion actions ──────────────────────────────────────
  // TwoFlavourRatioPseudoFermionAction(NumOp=heavier, DenOp=lighter)
  //   represents det(lighter)/det(heavier) [Nf=2].
  // Product PF0*...*PF4 = det(M_l)^2/det(M_PV)^2  (telescopes).
  TwoFlavourRatioPseudoFermionAction<FermionImplPolicy> PF0(Op_H1, Op_l,  CG, CG);
  TwoFlavourRatioPseudoFermionAction<FermionImplPolicy> PF1(Op_H2, Op_H1, CG, CG);
  TwoFlavourRatioPseudoFermionAction<FermionImplPolicy> PF2(Op_H3, Op_H2, CG, CG);
  TwoFlavourRatioPseudoFermionAction<FermionImplPolicy> PF3(Op_H4, Op_H3, CG, CG);
  TwoFlavourRatioPseudoFermionAction<FermionImplPolicy> PF4(Op_PV, Op_H4, CG, CG);

  // Strange Nf=1: RHMC, det(M_s)/det(M_PV)
  OneFlavourRatioRationalPseudoFermionAction<FermionImplPolicy>
      Strange(Op_sPV, Op_s, OFRp);

  // ── Force norm observer ───────────────────────────────────────────────────
  ForceNormObserver forceObs({
      {"PF0",     &PF0},
      {"PF1",     &PF1},
      {"PF2",     &PF2},
      {"PF3",     &PF3},
      {"PF4",     &PF4},
      {"Strange", &Strange},
      {"Gauge",   &Waction},
  });
  TheHMC.Resources.AddObservable<ForceNormModule>(&forceObs);

  // ── Integrator levels ─────────────────────────────────────────────────────
  // Flat 2-level: all fermion forces at Level1 (same step rate).
  // Deliberate for tuning — we want all levels evaluated at the same frequency
  // so force norms are directly comparable. In production, use a multi-level
  // integrator once masses are balanced.
  ActionLevel<HMCWrapper::Field> Level1(1);  // all fermion actions
  Level1.push_back(&PF0);
  Level1.push_back(&PF1);
  Level1.push_back(&PF2);
  Level1.push_back(&PF3);
  Level1.push_back(&PF4);
  Level1.push_back(&Strange);

  ActionLevel<HMCWrapper::Field> Level2(4);  // gauge (4 inner steps per fermion step)
  Level2.push_back(&Waction);

  TheHMC.TheAction.push_back(Level1);
  TheHMC.TheAction.push_back(Level2);

  TheHMC.Parameters.initialize(Reader);

  if (!import_cfg.empty()) {
    TheHMC.Parameters.StartingType    = "CheckpointStart";
    TheHMC.Parameters.StartTrajectory = 0;
    std::cout << GridLogMessage << "Starting from imported config (CheckpointStart traj=0)" << std::endl;
  }

  if (MyParams.ApplySmearing) {
    SmearingParam SmPar(Reader);
    Smear_Stout<HMCWrapper::ImplPolicy> Stout(SmPar.rho);
    SmearedConfiguration<HMCWrapper::ImplPolicy> SmearingPolicy(
        GridPtr, SmPar.Nsmear, Stout);
    TheHMC.Run(SmearingPolicy);
  } else {
    TheHMC.Run();
  }

  Grid_finalize();
}
