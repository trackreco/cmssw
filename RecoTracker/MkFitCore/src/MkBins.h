#ifndef RecoTracker_MkFitCore_src_MkBins_h
#define RecoTracker_MkFitCore_src_MkBins_h

#include "RecoTracker/MkFitCore/src/MiniPropagators.h"

namespace mkfit {

  class LayerInfo;
  class LayerOfHits;

  //============================================================================

  struct MkRZLimits {
    float m_rin = 0, m_rout = 0, m_zmin = 0, m_zmax = 0;

    MkRZLimits() = default;
    MkRZLimits(const LayerInfo &li) { setup(li); }
    MkRZLimits(const LayerInfo &li1, const LayerInfo &li2) { setup(li1, li2); }

    void setup(const LayerInfo &li);
    void setup(const LayerInfo &li1, const LayerInfo &li2);
  };

  //============================================================================

  struct MkBinTrackCovExtract {
    MPlexQF m_cov_0_0;
    MPlexQF m_cov_0_1;
    MPlexQF m_cov_1_1;
    MPlexQF m_cov_2_2;

    MkBinTrackCovExtract() = default;

    MkBinTrackCovExtract(const MPlexLS &err);

    MPlexQF calc_err_xy(const MPlexQF &x, const MPlexQF &y) const {
      return x * x * m_cov_0_0 + y * y * m_cov_1_1 + 2.0f * x * y * m_cov_0_1;
    };
  };

  //============================================================================

  struct MkBinLimits {
    MPlexQUH q0, q1, q2, p1, p2; // q0 in center, to detect dead regions and set WSR.m_in_gap = true
    // std::vector<int> hits; // ??? hmmh, tricky
  };

  //============================================================================

  struct MkBins {
    // To become members ... or go into a helper struct / config.
    static constexpr float DDPHI_PRESEL_FAC = 2.0f;
    static constexpr float DDQ_PRESEL_FAC = 1.2f;
    static constexpr float PHI_BIN_EXTRA_FAC = 2.75f;
    static constexpr float Q_BIN_EXTRA_FAC = 1.6f;

    static constexpr int NEW_MAX_HIT = 6;  // 4 - 6 give about the same # of tracks in quality-val

    mini_propagators::InitialStatePlex m_isp;
    mini_propagators::StatePlex m_sp1, m_sp2;

    MPlexQF m_phi_min, m_phi_max, m_phi_center, m_phi_delta;
    MPlexQF m_q_min, m_q_max, m_q_center;

    MPlexQF m_dphi_track, m_dq_track;  // 3 sigma track errors at initial state

    // debug & ntuple dump -- to be local in functions
    // MPlexQF phi_c, dphi;
    // MPlexQF q_c, qmin, qmax;

    int m_n_proc;
    bool m_is_barrel;
    bool m_is_outward;

    MPlexQF q_delta() const { return 0.5f * (m_q_max - m_q_min); }

    // -----------------------------------------------------

    MkBins(bool is_barrel, bool is_outward, int n_proc = NN) :
      m_n_proc(n_proc), m_is_barrel(is_barrel), m_is_outward(is_outward)
    {}

    MkBins(const MPlexLV &par, const MPlexQI &chg, bool is_barrel, int n_proc = NN) :
      m_isp(par, chg), m_n_proc(n_proc), m_is_barrel(is_barrel)
    {}

    void prop_to_limits(const LayerInfo &li);
    void prop_to_limits(const MkRZLimits &ls);

    void prop_to_final_edge(const MkRZLimits &ls);

    void determine_bin_windows(const MkBinTrackCovExtract &cov_ex);

    void find_bin_ranges(const LayerOfHits &loh, MkBinLimits &bl);

  };

}  // namespace mkfit

#endif
