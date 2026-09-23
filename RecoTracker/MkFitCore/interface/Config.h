#ifndef RecoTracker_MkFitCore_interface_Config_h
#define RecoTracker_MkFitCore_interface_Config_h

namespace mkfit {

  namespace Const {
    constexpr float PI = 3.14159265358979323846;
    constexpr float TwoPI = 6.28318530717958647692;
    constexpr float PIOver2 = Const::PI / 2.0f;
    constexpr float PIOver4 = Const::PI / 4.0f;
    constexpr float PI3Over4 = 3.0f * Const::PI / 4.0f;
    constexpr float InvPI = 1.0f / Const::PI;
    constexpr float sol = 0.299792458;  // speed of light in m/ns
    constexpr float sol_over_100 = 0.299792458e-2;

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
  }  // namespace Const

  inline float cdist(float a) { return a > Const::PI ? Const::TwoPI - a : a; }

  //------------------------------------------------------------------------------

  namespace Config {
    // config for fitting
    constexpr int nLayers = 10;  // default/toy: 10; cms-like: 18 (barrel), 27 (endcap)

    // Layer constants for common barrel / endcap.
    // TrackerInfo more or less has all this information.
    constexpr int nMaxTrkHits = 64;  // Used for array sizes in MkFitter/Finder, max hits in toy MC
    constexpr int nAvgSimHits = 32;  // Used for reserve() calls for sim hits/states

    // This will become layer dependent (in bits). To be consistent with min_dphi.
    static constexpr int m_nphi = 256;

    // Config for propagation - could/should enter into PropagationFlags?!
    constexpr int Niter = 5;
    // for prop to plane getS step
    constexpr int nSStepsInProp2Plane = 2;
    // Move to Config.cc, make a command-line option in mkFit.cc to ease profiling comparisons.
    // If making this constexpr again, also fix ifs using it in MkBuilder.cc and MkFinder.cc.
    // constexpr bool usePropToPlane = true;
    // constexpr bool usePtMultScat = true;
    extern bool usePropToPlane;
    extern bool usePtMultScat;

    // MkFinderV2p2 per-layer candidate policy. These exist so the three pieces
    // can be A/B'd from a single build; the intended behaviour is all three on,
    // which is the default. Turning one off recovers what v2p2 did before
    // 2026-09-22: no within-sensitive-region verdict at all, so a layer the
    // track never crosses was recorded as a hole; no hole limits, so
    // maxHolesPerCand / maxConsecHoles were read only by V1/V2; and no
    // candidate-stopping cuts, so neither minPtCut nor the looper stop ever
    // fired. Set from mkFit.cc with --v2p2-wsr / --v2p2-hole-limits /
    // --v2p2-stop-cuts; see RecoTracker/SESSIONS.md S4.
    extern bool v2p2UseWsr;
    extern bool v2p2UseHoleLimits;
    extern bool v2p2UseStopCuts;

    // MkFinderV2p2 in-layer combinatorial search: walk the pre-selected hits
    // forward along the trajectory and take a SEQUENCE of them, rather than the
    // single best-chi2 hit. Off recovers the best-hit hack. See SESSIONS.md S8.
    extern bool v2p2InLayerComb;

    // Keep one beam slot for a continuation that DECLINED the layer, even when
    // it is outranked. A hole is the only move that does not shrink the
    // covariance, so it is the only branch that stays open to the possibility
    // that an earlier hit was wrong -- and the score is computed with an error
    // model known to be 1.4-2x short, so the ranking it produces is not a reason
    // to throw the hedge away. See MkFinderV2p2::select_and_materialise().
    extern bool v2p2ReserveHoleSlot;

    // Best-short. Move a STOPPED TrackCand out of the beam and remember the best
    // of them on the CombCandidate, the way V1's CandCloner does. Two effects:
    // a stopped candidate stops occupying a beam slot it can never use, and a
    // seed whose long candidates degrade still yields the best short track it
    // ever had. Outward search only -- going inward the "tail" is the head of
    // the track and a short track is a different animal.
    extern bool v2p2BestShort;

    // Config for Bfield. Note: for now the same for CMS-phase1 and CylCowWLids.
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
#if defined(MKFIT_STANDALONE)
    extern int numThreadsFinder;
    extern int numThreadsEvents;
    extern int numSeedsPerTask;
#else
    constexpr int numThreadsFinder = 1;
    constexpr int numThreadsEvents = 1;
    constexpr int numSeedsPerTask = 32;
#endif

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
#if defined(MKFIT_STANDALONE)
    extern bool useHitsForDuplicates;
    extern bool removeDuplicates;
#else
    const bool useHitsForDuplicates = true;
#endif
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

    // ================================================================

    inline float bFieldFromZR(const float z, const float r) {
      return (Config::mag_b0 * z * z + Config::mag_b1 * z + Config::mag_c1) * (Config::mag_a * r * r + 1.f);
    }

  };  // namespace Config

  //------------------------------------------------------------------------------

}  // end namespace mkfit
#endif
