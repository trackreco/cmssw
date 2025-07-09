#include "RecoTracker/MkFitCore/src/MkBins.h"

#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

namespace mkfit {

  namespace mp = mini_propagators;

  void MkRZLimits::setup(const LayerInfo &li) {
    m_rin = li.rin();
    m_rout = li.rout();
    m_zmin = li.zmin();
    m_zmax = li.zmax();
  }

  void MkRZLimits::setup(const LayerInfo &li1, const LayerInfo &li2) {
    m_rin = std::min(li1.rin(), li2.rin());
    m_rout = std::max(li1.rout(), li2.rout());
    m_zmin = std::min(li1.zmin(), li2.zmin());
    m_zmax = std::max(li1.zmax(), li2.zmax());
  }

  //==============================================================================

  MkBinTrackCovExtract::MkBinTrackCovExtract(const MPlexLS &err) :
    m_cov_0_0(err.ReduceFixedIJ(0, 0)),
    m_cov_0_1(err.ReduceFixedIJ(0, 1)),
    m_cov_1_1(err.ReduceFixedIJ(1, 1)),
    m_cov_2_2(err.ReduceFixedIJ(2, 2))
  {}

  //==============================================================================

  void MkBins::prop_to_limits(const LayerInfo &li) {
    // Positions 1 and 2 should really be by "propagation order", 1 is the closest/
    // This should also work for backward propagation so not exactly trivial.
    // Also, do not really need propagation to center. Well, to be checked, and
    // to figure out error scaling factors / correction functions.
    if (m_is_barrel) {
      m_isp.propagate_to_r(mp::PA_Exact, li.rin(), m_sp1, true, m_n_proc);
      m_isp.propagate_to_r(mp::PA_Exact, li.rout(), m_sp2, true, m_n_proc);
    } else {
      m_isp.propagate_to_z(mp::PA_Exact, li.zmin(), m_sp1, true, m_n_proc);
      m_isp.propagate_to_z(mp::PA_Exact, li.zmax(), m_sp2, true, m_n_proc);
    }
  }

  void MkBins::prop_to_limits(const MkRZLimits &ls) {
    // Implementation for MkFinderV2p2.
    // m_isp is at the previous hit.

    // Need inward/outward hint. Also, could move m_isp to the first stop / edge.
    // Also, propagate to outer from the inward, not from the initial, now that
    // is not in the center o the layer (though this might need to be fixed, esp if we
    // apply the material there -- as we really should, at least in sub-det transitions where
    // majority of services are).
    // But then we need to calc dq_track, dphi_track before.

    if (m_is_barrel) {
      m_isp.propagate_to_r(mp::PA_Exact, ls.m_rin, m_sp1, true, m_n_proc);
      m_isp = m_sp1;
      m_isp.propagate_to_r(mp::PA_Exact, 0.5f * (ls.m_rin + ls.m_rout), m_sp2, true, m_n_proc);
      m_isp = m_sp2;
      // m_isp is now at the layer center ... for checks etc ... can skip it later.
      m_isp.propagate_to_r(mp::PA_Exact, ls.m_rout, m_sp2, true, m_n_proc);
    } else {
      m_isp.propagate_to_z(mp::PA_Exact, ls.m_zmin, m_sp1, true, m_n_proc);
      m_isp = m_sp1;
      m_isp.propagate_to_z(mp::PA_Exact, 0.5f * (ls.m_zmin + ls.m_zmax), m_sp2, true, m_n_proc);
      m_isp = m_sp2;
      // m_isp is now at the layer center ... for checks etc ... can skip it later.
      m_isp.propagate_to_z(mp::PA_Exact, ls.m_zmax, m_sp2, true, m_n_proc);
    }
  }

  void MkBins::determine_bin_windows(const MkBinTrackCovExtract &cov_ex) {
    // Below made members for debugging
    // MPlexQF phi_c, dphi_min, dphi_max;
    // phi_c = Matriplex::fast_atan2(m_isp.y, m_isp.x);  // calculated below as difference

    MPlexQF xp1, xp2;
    xp1 = Matriplex::fast_atan2(m_sp1.y, m_sp1.x);
    xp2 = Matriplex::fast_atan2(m_sp2.y, m_sp2.x);
    Matriplex::min_max(xp1, xp2, m_phi_min, m_phi_max);
    // Matriplex::min_max(Matriplex::fast_atan2(m_sp1.y, m_sp1.x), Matriplex::fast_atan2(m_sp2.y, m_sp2.x), m_phi_min, m_phi_max);
    m_phi_delta = m_phi_max - m_phi_min;
    m_phi_center = 0.5f * (m_phi_max + m_phi_min);
    for (int ii = 0; ii < NN; ++ii) {
      if (ii < m_n_proc) {
        if (m_phi_delta[ii] > Const::PI) {
          std::swap(m_phi_max[ii], m_phi_min[ii]);
          m_phi_delta[ii] = Const::TwoPI - m_phi_delta[ii];
          m_phi_center[ii] = Const::PI - m_phi_center[ii];
        }
        m_phi_delta *= 0.5f;
        // printf("phi_c: %f  p1: %f  p2: %f   m_phi_min: %f  m_phi_max: %f   dphi: %f\n",
        //       m_phi_center[ii], xp1[ii], xp2[ii], m_phi_min[ii], m_phi_max[ii], m_phi_delta[ii]);
      }
    }

    // Calculate dphi_track, dq_track differs for barrel/endcap
    MPlexQF r2_c = m_isp.x * m_isp.x + m_isp.y * m_isp.y;
    MPlexQF r2inv_c = 1.0f / r2_c;
    MPlexQF dphidx_c = -m_isp.y * r2inv_c;
    MPlexQF dphidy_c = m_isp.x * r2inv_c;
    m_dphi_track = 3.0f * cov_ex.calc_err_xy(dphidx_c, dphidy_c).abs().sqrt();

    // MPlexQF qmin, qmax;
    if (m_is_barrel) {
      Matriplex::min_max(m_sp1.z, m_sp2.z, m_q_min, m_q_max);
      m_q_center = m_isp.z;
      m_dq_track = 3.0f * Matriplex::abs(cov_ex.m_cov_2_2).sqrt();
    } else {
      Matriplex::min_max(Matriplex::hypot(m_sp1.x, m_sp1.y), Matriplex::hypot(m_sp2.x, m_sp2.y), m_q_min, m_q_max);
      m_q_center = Matriplex::sqrt(r2_c);
      m_dq_track = 3.0f * (r2inv_c * cov_ex.calc_err_xy(m_isp.x, m_isp.y).abs()).sqrt();
    }
  }

  void MkBins::find_bin_ranges(const LayerOfHits &loh, MkBinLimits &bl) {
    for (int i = 0; i < NN; ++i) {
      if (i < m_n_proc) {
        // Clamp crazy sizes. This actually only happens when prop-fail flag is set.
        // const float dphi_clamp = 0.1;
        // if (dphi_min[i] > 0.0f || dphi_min[i] < -dphi_clamp) dphi_min[i] = -dphi_clamp;
        // if (dphi_max[i] < 0.0f || dphi_max[i] > dphi_clampf) dphi_max[i] = dphi_clamp;
        bl.p1[i] = loh.phiBinChecked(m_phi_min[i] - m_dphi_track[i] - PHI_BIN_EXTRA_FAC * 0.0123f);
        bl.p2[i] = loh.phiBinChecked(m_phi_max[i] + m_dphi_track[i] + PHI_BIN_EXTRA_FAC * 0.0123f);

        bl.q0[i] = loh.qBinChecked(m_q_center[i]);
        bl.q1[i] = loh.qBinChecked(m_q_min[i] - m_dq_track[i] - Q_BIN_EXTRA_FAC * 0.5f * loh.layer_info().q_bin());
        bl.q2[i] = loh.qBinChecked(m_q_max[i] + m_dq_track[i] + Q_BIN_EXTRA_FAC * 0.5f * loh.layer_info().q_bin()) + 1;
      }
    }
  }

}  // namespace mkfit
