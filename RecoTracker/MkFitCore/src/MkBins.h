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

  // Runtime overrides for the phi pre-selection; see MkBins.cc. There is no
  // separate binnor factor any more: the fetch range is DERIVED from the cut,
  // which is what makes "the cut is wider than the fetch" impossible to express.
  extern float g_v2p2_dphi_trk_fac;   // factor on dphi_track, cut AND fetch
  extern float g_v2p2_hit_dphi_rad;   // the cut tolerance itself, radians
  extern int   g_v2p2_phi_extra_bins; // fetch safety margin, whole bins
  // The split dq cut; the q fetch is DERIVED from these, as phi's is from its own.
  extern float g_v2p2_dq_trk_fac;     // multiplies dq_track (itself 3 sigma)
  extern float g_v2p2_dq_hit_fac;     // multiplies hit_q_half_length; floor 1.0
  extern int   g_v2p2_q_extra_bins;   // fetch margin beyond the cut, whole q bins
  // PER-HIT phi extent from the covariance, instead of the flat half-bin
  // constant. When on, g_v2p2_dphi_hit_fac multiplies it and has the same
  // geometric meaning as g_v2p2_dq_hit_fac: a FLOOR OF 1.0 contains the hit's
  // own extent. When off, g_v2p2_hit_dphi_rad is used as a flat tolerance.
  extern bool  g_v2p2_phi_per_hit;
  extern float g_v2p2_dphi_hit_fac;

  struct MkBins {
    // To become members ... or go into a helper struct / config.
    // THE dphi CUT. The hit term is the hit's own phi extent, derived from its
    // covariance (LayerOfHits::hit_phi_half_extent(), the phi counterpart of
    // hit_q_half_length()), times DPHI_HIT_FAC:
    //
    //   ddphi < DPHI_TRK_FAC * dphi_track + DPHI_HIT_FAC * hit_phi_half_extent
    //
    // The per-hit extent is ~4e-5 rad in TB2S, so the cut is carried by the
    // track term, and DPHI_TRK_FAC = 2 is what makes that work: 1.0 loses 41
    // found tracks forward and 582 fully recovered chopped pT5 tracks inward,
    // 1.75 is the lowest free value forward, and 2.0 is free in both directions
    // and reproduces the flat constant to the unit inward. Above |eta| 0.8 the 2
    // also compensates a phi covariance measured 2.2-3.4x short (the material
    // model in the transition and forward region); at central eta, where the
    // covariance is within 1.25 of right, it is containment.
    static constexpr bool  PHI_PER_HIT  = true;
    static constexpr float DPHI_TRK_FAC = 2.0f;
    static constexpr float DPHI_HIT_FAC = 3.0f;

    // Flat phi tolerance, in RADIANS, used only with the per-hit extent
    // switched off. Equals one phi bin at N = 8 by accident of history, not by
    // design -- see the note on HIT_PHI_HALF_EXTENT below.
    static constexpr float PHI_PRESEL_TOLERANCE = 2.0f * 0.0123f;

    // Safety margin the BINNOR adds beyond what the cut can accept, in WHOLE
    // BINS added to the bin INDEX -- so it carries no float-to-bin rounding of
    // its own. It exists because the per-hit reference phi is the Hermite's
    // crossing at the hit's own module plane, which can fall slightly outside
    // the [phi_min, phi_max] span the range is built from.
    static constexpr int PHI_EXTRA_BINS = 1;

    // THE dq CUT, SPLIT. It used to be one factor (EXTRA_DQ) over both terms:
    //
    //   ddq < EXTRA_DQ * dq_trk + EXTRA_DQ * DDQ_PRESEL_FAC * hit_q_half_length
    //
    // which cannot be interpreted, because WHICH TERM BINDS IS A PROPERTY OF THE
    // LAYER: the containment term spans a factor 335 across the detector (0.009
    // cm in the pixel barrel to 3.015 in TB2S at unit factor) while the track
    // term does not. Strips are containment-dominated, pixels are
    // covariance-dominated, and one number cannot sit in the right place for
    // both. Split, each in its own honest unit:
    //
    //   DQ_TRK_FAC  multiplies dq_track, which is itself 3 sigma
    //   DQ_HIT_FAC  multiplies the hit's own half-extent -- FLOOR IS EXACTLY 1.0,
    //               below which the window stops reaching the strip it is trying
    //               to contain. (In the old compound units that floor was the
    //               opaque 1/1.2 = 0.833.)
    //
    // DQ_TRK_FAC is EXTRA_DQ = 1.5 in the old units. DQ_HIT_FAC sits 20 % above
    // its floor: taking it from 1.8 (the old ratio) to 1.2 is worth 3-6 % of
    // build time and is a physics null both ways -- forward -3 found tracks and
    // -15 fakes, inward +55 recovered chopped hits.
    static constexpr float DQ_TRK_FAC = 1.5f;
    static constexpr float DQ_HIT_FAC = 1.2f;

    // Fetch margin beyond what the cut accepts, in WHOLE q bins on the index.
    static constexpr int Q_EXTRA_BINS = 1;

    static constexpr float DDPHI_PRESEL_FAC = 2.0f;
    static constexpr float DDQ_PRESEL_FAC = 1.2f;
    // V2's fetch margins, as in upstream CMSSW; see find_bin_ranges_v2().
    static constexpr float PHI_BIN_EXTRA_FAC = 2.75f;
    static constexpr float Q_BIN_EXTRA_FAC = 1.6f;

    // MISNAMED -- this is HALF A PHI BIN, a binning granule and not a property
    // of any hit. The phi axis is axis_pow2_u1<float, bin_index_t, 16, 8>
    // (HitStructures.h): 256 bins over 2pi, width 0.024544 rad, half of which is
    // 0.012272. The per-hit phi extent derived from the covariance is
    // LayerOfHits::hit_phi_half_extent(); for a TB2S strip it is ~4e-5 rad,
    // about 300x smaller than this.
    //
    // Remaining consumers: PHI_PRESEL_TOLERANCE above, used by MkFinderV2p2 only
    // with the per-hit extent switched off, and V2 -- its cut in MkFinder.cc and
    // its upstream fetch in find_bin_ranges_v2(). The name is kept for V2.
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

    // MkFinderV2p2: fetch derived from the v2p2 cut (the g_v2p2_* globals).
    void find_bin_ranges(const LayerOfHits &loh, MkBinLimits &bl);
    // MkFinder::selectHitIndicesV2: the fetch V2 has in upstream CMSSW, kept
    // verbatim so production V2 is unchanged by the v2p2 window work.
    void find_bin_ranges_v2(const LayerOfHits &loh, MkBinLimits &bl);

  };

}  // namespace mkfit

#endif
