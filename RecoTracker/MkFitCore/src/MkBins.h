#ifndef RecoTracker_MkFitCore_src_MkBins_h
#define RecoTracker_MkFitCore_src_MkBins_h

#include "RecoTracker/MkFitCore/src/MiniPropagators.h"
#include "RecoTracker/MkFitCore/src/V2p2Config.h"

namespace mkfit {

  class LayerInfo;
  class LayerOfHits;
  struct MkRZLimits;

  struct MkBinTrackCovExtract {
    MPlexQF m_cov_0_0 = { 0.0f };
    MPlexQF m_cov_0_1 = { 0.0f };
    MPlexQF m_cov_1_1 = { 0.0f };
    MPlexQF m_cov_2_2 = { 0.0f };
    // Needed only for surface_reference_dq() -- the position block of the
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
    // V2 (MkFinder::selectHitIndicesV2) constants, as in upstream CMSSW. The
    // v2p2 window is configured in V2p2Config.h.
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
    // Remaining consumers: V2 -- its cut in MkFinder.cc and its upstream fetch
    // in find_bin_ranges_v2(). Config::V2p2::Window::dphi_flat_rad defaults to
    // twice this value. The name is kept for V2.
    static constexpr float HIT_PHI_HALF_EXTENT = 0.0123f;

    // V2 pqueue cap. v2p2 uses Config::V2p2::InLayer::max_presel_hits.
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

    // MkFinderV2p2: fetch derived from the v2p2 cut, Config::V2p2::Window.
    void find_bin_ranges(const LayerOfHits &loh, MkBinLimits &bl);
    // MkFinder::selectHitIndicesV2: the fetch V2 has in upstream CMSSW, kept
    // verbatim so production V2 is unchanged by the v2p2 window work.
    void find_bin_ranges_v2(const LayerOfHits &loh, MkBinLimits &bl);

  };

}  // namespace mkfit

#endif
