#ifndef RecoTracker_MkFitCore_interface_Config_h
#define RecoTracker_MkFitCore_interface_Config_h

#include <algorithm>
#include <cmath>
#include <string>  // won't compile on clang gcc for mac OS w/o this!
#include <map>
#include <vector>

namespace mkfit {

  enum PropagationFlagsEnum {
    PF_none = 0,

    PF_use_param_b_field = 0x1,
    PF_apply_material = 0x2
  };

  struct PropagationFlags {
    union {
      struct {
        bool use_param_b_field : 1;
        bool apply_material : 1;
        // Could add: bool use_trig_approx  -- now Config::useTrigApprox = true
        // Could add: int  n_iter : 8       -- now Config::Niter = 5
      };

      unsigned int _raw_;
    };

    PropagationFlags() : _raw_(0) {}

    PropagationFlags(int pfe)
        : use_param_b_field(pfe & PF_use_param_b_field), apply_material(pfe & PF_apply_material) {}
  };

  //------------------------------------------------------------------------------

  using IntVec = std::vector<int>;
  using IntVec_i = IntVec::iterator;

  // Enum for input seed options
  enum seedOpts { simSeeds, cmsswSeeds, findSeeds };
  typedef std::map<std::string, std::pair<seedOpts, std::string> > seedOptsMap;

  // Enum for seed cleaning options
  enum cleanOpts { noCleaning, cleanSeedsN2, cleanSeedsPure, cleanSeedsBadLabel };
  typedef std::map<std::string, std::pair<cleanOpts, std::string> > cleanOptsMap;

  // Enum for cmssw matching options
  enum matchOpts { trkParamBased, hitBased, labelBased };
  typedef std::map<std::string, std::pair<matchOpts, std::string> > matchOptsMap;

  //------------------------------------------------------------------------------

  namespace Config {

    // math general --> from namespace TMath
    constexpr float PI = 3.14159265358979323846;
    constexpr float TwoPI = 6.28318530717958647692;
    constexpr float PIOver2 = Config::PI / 2.0f;
    constexpr float PIOver4 = Config::PI / 4.0f;
    constexpr float PI3Over4 = 3.0f * Config::PI / 4.0f;
    constexpr float InvPI = 1.0f / Config::PI;
    constexpr float RadToDeg = 180.0f / Config::PI;
    constexpr float DegToRad = Config::PI / 180.0f;
    constexpr float sol = 0.299792458;  // speed of light in nm/s
    constexpr double Sqrt2 = 1.4142135623730950488016887242097;
    constexpr double OOSqrt2 = 1.0 / Config::Sqrt2;

    // general parameters of matrices
    constexpr int nParams = 6;

    extern std::string geomPlugin;

    // config for fitting
    constexpr int nLayers = 10;  // default: 10; cmssw tests: 13, 17, 26 (for endcap)

    // New layer constants for common barrel / endcap. I'd prefer those to go
    // into some geometry definition "plugin" -- they belong more into some Geom
    // namespace, too.
    // XXXX This needs to be generalized for other geometries !
    // TrackerInfo more or less has all this information (or could have it).
    extern int nTotalLayers;         // To be set by geometry plugin.
    constexpr int nMaxTrkHits = 64;  // Used for array sizes in MkFitter/Finder, max hits in toy MC
    constexpr int nAvgSimHits = 32;  // Used for reserve() calls for sim hits/states

    constexpr float fRadialSpacing = 4.;
    constexpr float fRadialExtent = 0.01;
    constexpr float fInnerSensorSize = 5.0;  // approximate sensor size in cm
    constexpr float fOuterSensorSize = Config::fInnerSensorSize * 2.;
    constexpr float fEtaDet = 1;  // default: 1; cmssw tests: 2, 2.5

    constexpr float cmsDeltaRad = 2.5;  //fixme! using constant 2.5 cm, to be taken from layer properties

    // config for material effects in cmssw
    constexpr float rangeZME = 300.;
    constexpr int nBinsZME = 300;
    constexpr float rangeRME = 120.;
    constexpr int nBinsRME = 120;

    constexpr float Rl[136] = {
        0.018, 0.031, 0.017, 0.023, 0.018, 0.028, 0.021, 0.040, 0.066, 0.039, 0.069, 0.040, 0.103, 0.098, 0.028, 0.038,
        0.025, 0.034, 0.037, 0.078, 0.048, 0.064, 0.168, 0.085, 0.144, 0.033, 0.157, 0.078, 0.014, 0.032, 0.052, 0.012,
        0.026, 0.038, 0.015, 0.035, 0.061, 0.015, 0.035, 0.043, 0.015, 0.036, 0.033, 0.010, 0.021, 0.022, 0.093, 0.084,
        0.100, 0.194, 0.093, 0.108, 0.200, 0.093, 0.084, 0.100, 0.194, 0.093, 0.108, 0.200, 0.038, 0.075, 0.038, 0.075,
        0.038, 0.075, 0.038, 0.075, 0.038, 0.075, 0.038, 0.075, 0.039, 0.078, 0.039, 0.078, 0.039, 0.078, 0.039, 0.078,
        0.039, 0.078, 0.039, 0.078, 0.046, 0.023, 0.046, 0.023, 0.046, 0.046, 0.023, 0.046, 0.023, 0.046, 0.048, 0.024,
        0.048, 0.024, 0.048, 0.048, 0.024, 0.048, 0.024, 0.048, 0.055, 0.027, 0.055, 0.027, 0.055, 0.055, 0.027, 0.055,
        0.027, 0.055, 0.043, 0.021, 0.043, 0.043, 0.043, 0.021, 0.043, 0.043, 0.040, 0.020, 0.040, 0.040, 0.040, 0.020,
        0.040, 0.040, 0.014, 0.028, 0.028, 0.014, 0.028, 0.028};

    constexpr float Xi[136] = {
        0.039e-03, 0.062e-03, 0.029e-03, 0.037e-03, 0.032e-03, 0.049e-03, 0.044e-03, 0.080e-03, 0.147e-03, 0.086e-03,
        0.162e-03, 0.092e-03, 0.214e-03, 0.207e-03, 0.062e-03, 0.081e-03, 0.051e-03, 0.068e-03, 0.078e-03, 0.155e-03,
        0.110e-03, 0.138e-03, 0.321e-03, 0.166e-03, 0.311e-03, 0.077e-03, 0.371e-03, 0.185e-03, 0.035e-03, 0.069e-03,
        0.104e-03, 0.025e-03, 0.051e-03, 0.072e-03, 0.033e-03, 0.069e-03, 0.114e-03, 0.033e-03, 0.071e-03, 0.083e-03,
        0.033e-03, 0.073e-03, 0.064e-03, 0.021e-03, 0.043e-03, 0.043e-03, 0.216e-03, 0.209e-03, 0.185e-03, 0.309e-03,
        0.216e-03, 0.255e-03, 0.369e-03, 0.216e-03, 0.209e-03, 0.185e-03, 0.309e-03, 0.216e-03, 0.255e-03, 0.369e-03,
        0.083e-03, 0.166e-03, 0.083e-03, 0.166e-03, 0.083e-03, 0.166e-03, 0.083e-03, 0.166e-03, 0.083e-03, 0.166e-03,
        0.083e-03, 0.166e-03, 0.088e-03, 0.175e-03, 0.088e-03, 0.175e-03, 0.088e-03, 0.175e-03, 0.088e-03, 0.175e-03,
        0.088e-03, 0.175e-03, 0.088e-03, 0.175e-03, 0.104e-03, 0.052e-03, 0.104e-03, 0.052e-03, 0.104e-03, 0.104e-03,
        0.052e-03, 0.104e-03, 0.052e-03, 0.104e-03, 0.110e-03, 0.055e-03, 0.110e-03, 0.055e-03, 0.110e-03, 0.110e-03,
        0.055e-03, 0.110e-03, 0.055e-03, 0.110e-03, 0.130e-03, 0.065e-03, 0.130e-03, 0.065e-03, 0.130e-03, 0.130e-03,
        0.065e-03, 0.130e-03, 0.065e-03, 0.130e-03, 0.097e-03, 0.048e-03, 0.097e-03, 0.097e-03, 0.097e-03, 0.048e-03,
        0.097e-03, 0.097e-03, 0.089e-03, 0.045e-03, 0.089e-03, 0.089e-03, 0.089e-03, 0.045e-03, 0.089e-03, 0.089e-03,
        0.030e-03, 0.061e-03, 0.061e-03, 0.030e-03, 0.061e-03, 0.061e-03};

    extern float RlgridME[Config::nBinsZME][Config::nBinsRME];
    extern float XigridME[Config::nBinsZME][Config::nBinsRME];

    // This will become layer dependent (in bits). To be consistent with min_dphi.
    static constexpr int m_nphi = 256;

    // Config for propagation - could/should enter into PropagationFlags?!
    constexpr int Niter = 5;
    constexpr bool useTrigApprox = true;

    // PropagationFlags as used during finding and fitting. Defined for each Geom in its plugin.
    extern bool finding_requires_propagation_to_hit_pos;
    extern PropagationFlags finding_inter_layer_pflags;
    extern PropagationFlags finding_intra_layer_pflags;
    extern PropagationFlags backward_fit_pflags;
    extern PropagationFlags forward_fit_pflags;
    extern PropagationFlags seed_fit_pflags;
    extern PropagationFlags pca_prop_pflags;

    // Config for Bfield. Note: for now the same for CMS-2017 and CylCowWLids.
    constexpr float Bfield = 3.8112;
    constexpr float mag_c1 = 3.8114;
    constexpr float mag_b0 = -3.94991e-06;
    constexpr float mag_b1 = 7.53701e-06;
    constexpr float mag_a = 2.43878e-11;

    // Config for SelectHitIndices
    // Use extra arrays to store phi and q of hits.
    // MT: This would in principle allow fast selection of good hits, if
    // we had good error estimates and reasonable *minimal* phi and q windows.
    // Speed-wise, those arrays (filling AND access, about half each) cost 1.5%
    // and could help us reduce the number of hits we need to process with bigger
    // potential gains.
#ifdef CONFIG_PhiQArrays
    extern bool usePhiQArrays;
#else
    constexpr bool usePhiQArrays = true;
#endif

    // sorting config (bonus,penalty)
    constexpr float validHitBonus_ = 4;
    constexpr float validHitSlope_ = 0.2;
    constexpr float overlapHitBonus_ = 0;  // set to negative for penalty
    constexpr float missingHitPenalty_ = 8;
    constexpr float tailMissingHitPenalty_ = 3;

    // Threading
    extern int numThreadsFinder;
    extern int numThreadsEvents;
    extern int numSeedsPerTask;

    extern seedOpts seedInput;

    // config on seed cleaning
    constexpr float track1GeVradius = 87.6;  // = 1/(c*B)
    constexpr float c_etamax_brl = 0.9;
    constexpr float c_dpt_common = 0.25;
    constexpr float c_dzmax_brl = 0.005;
    constexpr float c_drmax_brl = 0.010;
    constexpr float c_ptmin_hpt = 2.0;
    constexpr float c_dzmax_hpt = 0.010;
    constexpr float c_drmax_hpt = 0.010;
    constexpr float c_dzmax_els = 0.015;
    constexpr float c_drmax_els = 0.015;

    // config on duplicate removal
    extern bool useHitsForDuplicates;
    extern bool removeDuplicates;
    extern const float maxdPhi;
    extern const float maxdPt;
    extern const float maxdEta;
    extern const float minFracHitsShared;
    extern const float maxdR;

    // duplicate removal: tighter version
    extern const float maxd1pt;
    extern const float maxdphi;
    extern const float maxdcth;
    extern const float maxcth_ob;
    extern const float maxcth_fw;

    extern bool useCMSGeom;
    extern bool includePCA;

    extern bool silent;
    extern bool json_verbose;

    // NAN and silly track parameter tracking options
    constexpr bool nan_etc_sigs_enable = false;

    constexpr bool nan_n_silly_check_seeds = true;
    constexpr bool nan_n_silly_print_bad_seeds = false;
    constexpr bool nan_n_silly_fixup_bad_seeds = false;
    constexpr bool nan_n_silly_remove_bad_seeds = true;

    constexpr bool nan_n_silly_check_cands_every_layer = false;
    constexpr bool nan_n_silly_print_bad_cands_every_layer = false;
    constexpr bool nan_n_silly_fixup_bad_cands_every_layer = false;

    constexpr bool nan_n_silly_check_cands_pre_bkfit = true;
    constexpr bool nan_n_silly_check_cands_post_bkfit = true;
    constexpr bool nan_n_silly_print_bad_cands_bkfit = false;

    // ================================================================

    inline float BfieldFromZR(const float z, const float r) {
      return (Config::mag_b0 * z * z + Config::mag_b1 * z + Config::mag_c1) * (Config::mag_a * r * r + 1.f);
    }

#ifndef MPT_SIZE
#if defined(__AVX512F__)
#define MPT_SIZE 16
#elif defined(__AVX__) || defined(__AVX2__)
#define MPT_SIZE 8
#elif defined(__SSE3__)
#define MPT_SIZE 4
#else
#define MPT_SIZE 8
#endif
#endif

  };  // namespace Config

  inline float cdist(float a) { return a > Config::PI ? Config::TwoPI - a : a; }

}  // end namespace mkfit
#endif
