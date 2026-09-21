#ifndef RecoTracker_MkFitCore_src_MkBins_h
#define RecoTracker_MkFitCore_src_MkBins_h

#include "RecoTracker/MkFitCore/src/MiniPropagators.h"

namespace mkfit {

  class LayerInfo;
  class LayerOfHits;
  struct MkRZLimits;

  // Reference the window covariance to the LAYER SURFACE instead of to the fixed
  // path length errPropFromPathL_impl() transports it to. See MkBins.cc.
  // Runtime switch so the A/B needs one build.
  extern bool g_mkbins_surface_q;

  struct MkBinTrackCovExtract {
    MPlexQF m_cov_0_0 = { 0.0f };
    MPlexQF m_cov_0_1 = { 0.0f };
    MPlexQF m_cov_1_1 = { 0.0f };
    MPlexQF m_cov_2_2 = { 0.0f };
    // Needed only for the surface referencing above -- the position block of the
    // covariance is what the projection acts on, and these two complete it.
    MPlexQF m_cov_0_2 = { 0.0f };
    MPlexQF m_cov_1_2 = { 0.0f };

    MkBinTrackCovExtract() = default;

    MkBinTrackCovExtract(const MPlexLS &err) {
      init_from_track_errors(err);
    }

    void init_from_track_errors(const MPlexLS &err) {
      m_cov_0_0 = err.ReduceFixedIJ(0, 0);
      m_cov_0_1 = err.ReduceFixedIJ(0, 1);
      m_cov_1_1 = err.ReduceFixedIJ(1, 1);
      m_cov_2_2 = err.ReduceFixedIJ(2, 2);
      m_cov_0_2 = err.ReduceFixedIJ(0, 2);
      m_cov_1_2 = err.ReduceFixedIJ(1, 2);
    }

    MPlexQF calc_err_xy(const MPlexQF &x, const MPlexQF &y) const {
      return x * x * m_cov_0_0 + y * y * m_cov_1_1 + 2.0f * x * y * m_cov_0_1;
    };
  };

  //============================================================================

  struct MkBinLimits {
    MPlexQUH q0, q1, q2, p1, p2; // q0 in center, to detect dead regions and set WSR.m_in_gap = true

    // would it make sense to store hit ids here (in sth like std::fixed_capacity_vector)?
    // std::vector<int> hits;
  };

  //============================================================================

  struct MkBins {
    // To become members ... or go into a helper struct / config.
    static constexpr float DDPHI_PRESEL_FAC = 2.0f;
    static constexpr float DDQ_PRESEL_FAC = 1.2f;
    static constexpr float PHI_BIN_EXTRA_FAC = 2.75f;
    static constexpr float Q_BIN_EXTRA_FAC = 1.6f;

    // Assumed phi half-extent of a hit, in radians -- the phi-side counterpart of
    // LayerOfHits::hit_q_half_length(). Unlike that one it is NOT derived from the
    // hit covariance: it is a single detector-wide constant, with no per-hit,
    // per-layer or geometry dependence. Was an unnamed 0.0123f literal repeated in
    // five places (this file, MkFinderV2p2 pre-selection, MkFinder V2 path).
    //
    // What the phi extent of a strip hit actually is: the module frame has xdir
    // perpendicular to the strips (i.e. essentially azimuthal), ydir along the
    // strips and zdir along the normal -- so ydir and zdir both lie in the (r,z)
    // plane. Consequently the strip *length* contributes to z and r but **nothing
    // to phi**, and the only phi extent is the across-strip pitch term. For a TOB
    // 2S strip at r = 69 cm with 90 um pitch that is sigma_phi ~ 1.3e-4 rad, so
    // DDPHI_PRESEL_FAC * this constant is ~190x the pitch term -- uniformly and
    // hugely over-generous, tilted layers included. That costs scanned hits, not
    // efficiency, so it is a tuning/cleanup item rather than a bug. See
    // RecoTracker/CLAUDE.md for the full q/phi extraction cross-check.
    static constexpr float HIT_PHI_HALF_EXTENT = 0.0123f;

    static constexpr int NEW_MAX_HIT = 6;  // 4 - 6 give about the same # of tracks in quality-val

    mini_propagators::InitialStatePlex m_isp;
    mini_propagators::StatePlex m_sp1, m_sp2;

    MPlexQF m_phi_min, m_phi_max, m_phi_center, m_phi_delta;
    MPlexQF m_q_min, m_q_max, m_q_center;

    MPlexQF m_dphi_track, m_dq_track;  // 3 sigma track errors at initial state

    // debug & ntuple dump -- to be local in functions or ifdef MKFIT_STANDALONE
    // MPlexQF phi_c, dphi;
    // MPlexQF q_c, qmin, qmax;

    int m_n_proc;
    bool m_is_barrel; // set in prop_to_*()

    MPlexQF q_delta() const { return 0.5f * (m_q_max - m_q_min); }

    // -----------------------------------------------------

    MkBins(int n_proc) :
      m_n_proc(n_proc)
    {}

    //zzz MkBins(const MPlexLV &par, const MPlexQI &chg, bool is_barrel, int n_proc = NN) :
    MkBins(const MPlexLV &par, const MPlexQI &chg, int n_proc) :
      m_isp(par, chg), m_n_proc(n_proc)//zz , m_is_barrel(is_barrel)
    {}

    void prop_to_limits(const LayerInfo &li);
    void prop_to_limits(const MkRZLimits &ls);

    void prop_to_limits_in_order(const MkRZLimits &ls);

    void determine_bin_windows(const MkBinTrackCovExtract &cov_ex);
    void surface_reference_dq(const MkBinTrackCovExtract &cov_ex);

    void find_bin_ranges(const LayerOfHits &loh, MkBinLimits &bl);

  };

}  // namespace mkfit

#endif
