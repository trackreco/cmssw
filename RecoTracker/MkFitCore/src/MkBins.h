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
  // Transitional: reinstate the old hand-rolled range, for A/B only.
  extern bool  g_v2p2_phi_legacy_range;

  struct MkBins {
    // To become members ... or go into a helper struct / config.
    // Per-hit phi tolerance of the pre-selection cut, in RADIANS, flat and
    // detector-wide. Equals one phi bin at N = 8 by accident of history, not by
    // design -- see the note on HIT_PHI_HALF_EXTENT below.
    // TODO: this wants to be PER HIT, from the hit covariance -- the phi
    // counterpart of LayerOfHits::hit_q_half_length(), which does not exist.
    // Measured: it can be tightened 4x for free, which is what using a binning
    // granule as a resolution looks like.
    static constexpr float PHI_PRESEL_TOLERANCE = 2.0f * 0.0123f;

    // Safety margin the BINNOR adds beyond what the cut can accept, in WHOLE
    // BINS added to the bin INDEX -- so it carries no float-to-bin rounding of
    // its own. It exists because the per-hit reference phi is the Hermite's
    // crossing at the hit's own module plane, which can fall slightly outside
    // the [phi_min, phi_max] span the range is built from.
    static constexpr int PHI_EXTRA_BINS = 1;

    static constexpr float DDPHI_PRESEL_FAC = 2.0f;
    static constexpr float DDQ_PRESEL_FAC = 1.2f;
    static constexpr float PHI_BIN_EXTRA_FAC = 2.75f;
    static constexpr float Q_BIN_EXTRA_FAC = 1.6f;

    // MISNAMED, and the name has caused trouble -- read this before using it.
    //
    // This is HALF A PHI BIN, chosen as such: the phi axis is
    // axis_pow2_u1<float, bin_index_t, 16, 8> (HitStructures.h), i.e. 256 bins
    // over 2pi, width 0.024544 rad, half of which is 0.012272. It is a BINNING
    // granule, not a property of a hit, and an earlier commit here naming it
    // "hit phi half extent" and calling it the phi-side counterpart of
    // LayerOfHits::hit_q_half_length() was wrong -- hit_q_half_length IS derived
    // per hit from the covariance, and this is not derived from anything.
    //
    // It has two consumers and only ONE of them is legitimately a bin quantity:
    //
    //   PHI_BIN_EXTRA_FAC, the binnor range (MkBins.cc) -- CORRECT unit. Read
    //     the factor as "bins of margin": 2.75 is 1.38 bins. Measured floor is
    //     1.00 bins, and that whole bin is paying for a missing "+1"; see
    //     g_v2p2_phi_bin_fix in MkBins.cc.
    //
    //   DDPHI_PRESEL_FAC, the per-hit pre-selection cut (MkFinderV2p2, and the
    //     V2 path in MkFinder.cc) -- WRONG quantity. A bin granule is not a
    //     resolution. What belongs there is the hit's own phi extent from its
    //     covariance, the counterpart to hit_q_half_length that does not exist
    //     yet. Measured symptom: the cut can be tightened 4x for free.
    //
    // Was an unnamed 0.0123f literal repeated in five places.
    //
    // What the phi extent of a strip hit actually is: the module frame has xdir
    // perpendicular to the strips (i.e. essentially azimuthal), ydir along the
    // strips and zdir along the normal -- so ydir and zdir both lie in the (r,z)
    // plane. Consequently the strip *length* contributes to z and r but **nothing
    // to phi**, and the only phi extent is the across-strip pitch term. For a TOB
    // 2S strip at r = 69 cm with 90 um pitch that is sigma_phi ~ 1.3e-4 rad. So
    // for the CUT the right value is ~200x smaller than a bin, and comparing the
    // two is comparing a resolution with a binning granule -- which is why the
    // scan finds 4x of slack and why a per-hit phi extent is the real fix.
    // Measured: moving the cut costs no efficiency in either direction, so this
    // is a speed and correctness-of-naming item, not an efficiency one. See
    // RecoTracker/CLAUDE.md for the full q/phi extraction cross-check.
    static constexpr float HIT_PHI_HALF_EXTENT = 0.0123f;

    // Runtime overrides for the three dphi factors, so the phi side can be
    // scanned the way EXTRA_DQ was. Defaults reproduce the constants above
    // exactly. Deliberately THREE knobs and not one: the dq scan showed that a
    // single factor over both terms of a cut cannot be interpreted, because
    // which term binds is a property of the layer.
    //
    //   g_v2p2_dphi_trk_fac  multiplies m_dphi_track, in the CUT and the BINNOR
    //   g_v2p2_hit_dphi_fac  replaces DDPHI_PRESEL_FAC, in the CUT
    //   g_v2p2_bin_dphi_fac  replaces PHI_BIN_EXTRA_FAC, in the BINNOR
    //
    // The binnor pair is not optional bookkeeping: the cut can only reject hits
    // the binnor already fetched, so raising hit_dphi_fac above bin_dphi_fac, or
    // dphi_trk_fac above 1 without the binnor following, is a silent NO-OP and
    // the resulting flatness is an artefact of the fetch, not physics.

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
