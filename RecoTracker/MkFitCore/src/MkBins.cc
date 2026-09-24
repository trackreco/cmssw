#include "RecoTracker/MkFitCore/src/MkBins.h"
#include "RecoTracker/MkFitCore/src/MkRZLimits.h"

#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include <algorithm>
#include <cmath>

namespace mkfit {

  using namespace Config::V2p2;

  // Largest phi half-width a range can have: just under pi, since at pi the two
  // endpoints coincide and the arc degenerates to a point.
  static constexpr float kMaxHalfPhiWindow = 3.14f;

  namespace mp = mini_propagators;

  //==============================================================================

  void MkBins::prop_to_limits_in_order(const MkRZLimits &ls) {
    // The second implementation for MkFinderV2p2.
    // m_isp is at the previous hit.
    // Propagate to final edge of the layer limits.
    // To be post-processed by finding inner point via hermite.
    // Also, full error propagation needs to be done for track_dphi / dq.
    // Compare the difference.
    //
    // There is some worry sp1 and sp2 are used later on so as to expect one to be larger.
    // It shouldn't matter for bin edges, there we min/max stuff.
    // It might impact ordering of hits, but if we get hermite from start to end,
    // oh ... well, yes, we need to be careful if we want them in order.
    // Cross check how t parameter behaves, ie, if it goes from 0 to 1 when
    // ds is negative and getting more negative with distance.
    // It matters for hermite, which point is "first".
    // Also, we don't really need sp1 ... the initial point is fine.
    // Only the "time" will be extended
    // So ... which do we keep, how do we name them?

    m_is_barrel = ls.m_is_barrel;
    if (m_is_barrel) {
      float r;
      r = ls.m_is_outward ? ls.m_rin : ls.m_rout;
      m_isp.propagate_to_r(mp::PA_Exact, r, m_sp1, true, m_n_proc);
      m_isp = m_sp1;
      r = ls.m_is_outward ? ls.m_rout : ls.m_rin;
      m_isp.propagate_to_r(mp::PA_Exact, r, m_sp2, true, m_n_proc);
    } else {
      float z1 = ls.m_zmin, z2 = ls.m_zmax;
      bool is_pos = ls.m_layer_info_1->layer_type() == LayerInfo::EndCapPos;
      if ((is_pos && ! ls.m_is_outward) || ( ! is_pos && ls.m_is_outward))
        std::swap(z1, z2);

      m_isp.propagate_to_z(mp::PA_Exact, z1, m_sp1, true, m_n_proc);
      m_isp = m_sp1;
      m_isp.propagate_to_z(mp::PA_Exact, z2, m_sp2, true, m_n_proc);
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
        m_phi_delta[ii] *= 0.5f;
        // printf("phi_c: %f  p1: %f  p2: %f   m_phi_min: %f  m_phi_max: %f   dphi: %f\n",
        //       m_phi_center[ii], xp1[ii], xp2[ii], m_phi_min[ii], m_phi_max[ii], m_phi_delta[ii]);
      }
    }

    // Calculate dphi_track, dq_track differs for barrel/endcap
    MPlexQF r2_c = m_isp.x * m_isp.x + m_isp.y * m_isp.y;
    MPlexQF r2inv_c = 1.0f / r2_c;

    // sigma_phi = J sigma_xy J^T with J = (-y, x)/r^2, so |J| = 1/r. J is taken at
    // the layer crossing with the smaller radius, where sigma_phi is largest. See
    // doc/MkFinderV2p2-DesignNotes.md, "Search window".
    MPlexQF jx, jy, r2inv_j;
    {
      const MPlexQF r2_1 = m_sp1.x * m_sp1.x + m_sp1.y * m_sp1.y;
      const MPlexQF r2_2 = m_sp2.x * m_sp2.x + m_sp2.y * m_sp2.y;
      for (int i = 0; i < m_n_proc; ++i) {
        const bool first = r2_1[i] <= r2_2[i];
        jx[i] = first ? m_sp1.x[i] : m_sp2.x[i];
        jy[i] = first ? m_sp1.y[i] : m_sp2.y[i];
        r2inv_j[i] = 1.0f / (first ? r2_1[i] : r2_2[i]);
      }
    }
    MPlexQF dphidx_c = -jy * r2inv_j;
    MPlexQF dphidy_c = jx * r2inv_j;
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

    if (Diag::mkbins_surface_q)
      surface_reference_dq(cov_ex);
  }

  //----------------------------------------------------------------------------
  // surface_reference_dq() -- dq_track referenced to the layer surface (radial
  // normal in the barrel, z in the endcap), evaluated at m_sp2 where cov_ex
  // lives. Off by default (Diag::mkbins_surface_q); the per-hit version with the
  // module normal is MkFinderV2p2::surface_referenced_dq(). See
  // doc/MkFinderV2p2-DesignNotes.md, "Search window".
  //----------------------------------------------------------------------------

  void MkBins::surface_reference_dq(const MkBinTrackCovExtract &cov_ex) {
    // Clamp on the amplification. g = cot(theta) for a radial barrel track, so 20
    // is |eta| ~ 3.7, beyond the tracker.
    constexpr float kMaxSlope = 20.0f;

    for (int i = 0; i < m_n_proc; ++i) {
      const float x = m_sp2.x[i], y = m_sp2.y[i];
      const float r2 = x * x + y * y;
      if (r2 <= 0.0f)
        continue;
      const float rinv = 1.0f / std::sqrt(r2);
      const float nx = x * rinv, ny = y * rinv;   // radial unit vector

      const float pr = nx * m_sp2.px[i] + ny * m_sp2.py[i];  // p . n_radial
      const float pz = m_sp2.pz[i];

      const float c00 = cov_ex.m_cov_0_0[i], c01 = cov_ex.m_cov_0_1[i];
      const float c11 = cov_ex.m_cov_1_1[i], c22 = cov_ex.m_cov_2_2[i];
      const float c02 = cov_ex.m_cov_0_2[i], c12 = cov_ex.m_cov_1_2[i];

      float var;
      if (m_is_barrel) {
        // v = e_z - (p_z / (p.n)) * n ; note |p| cancels out of the ratio.
        if (pr == 0.0f)
          continue;
        float g = pz / pr;
        g = std::clamp(g, -kMaxSlope, kMaxSlope);
        const float v0 = -g * nx, v1 = -g * ny;
        var = v0 * v0 * c00 + v1 * v1 * c11 + c22 + 2.0f * (v0 * v1 * c01 + v0 * c02 + v1 * c12);
      } else {
        // w = r^ - ((r^.p) / p_z) * e_z
        if (pz == 0.0f)
          continue;
        float ginv = pr / pz;
        ginv = std::clamp(ginv, -kMaxSlope, kMaxSlope);
        var = nx * nx * c00 + ny * ny * c11 + ginv * ginv * c22 +
              2.0f * (nx * ny * c01 - ginv * nx * c02 - ginv * ny * c12);
      }

      if (var > 0.0f)
        m_dq_track[i] = 3.0f * std::sqrt(var);
    }
  }

  void MkBins::find_bin_ranges(const LayerOfHits &loh, MkBinLimits &bl) {
    for (int i = 0; i < NN; ++i) {
      if (i < m_n_proc) {
        // Fetch what the cut can accept, using the layer's largest hit extent since
        // the hit is not known yet, then extend by whole bins on the bin index.
        // The half-width is clamped below pi: a larger one wraps to a small arc.
        // See doc/MkFinderV2p2-DesignNotes.md, "Search window".
        const float phi_hit_term = Window::phi_per_hit
                                 ? Window::dphi_hit_fac * loh.max_hit_phi_half_extent()
                                 : Window::dphi_flat_rad;
        const float cut_dphi = std::min(Window::dphi_trk_fac * m_dphi_track[i] + phi_hit_term,
                                        kMaxHalfPhiWindow);
        auto pr = loh.phiRangeBins(m_phi_min[i] - cut_dphi, m_phi_max[i] + cut_dphi);
        bl.p1[i] = loh.phiMaskApply(pr.begin - Window::phi_extra_bins);
        bl.p2[i] = loh.phiMaskApply(pr.end   + Window::phi_extra_bins);

        // Same for q. The q axis is bounded, so the extension clamps.
        bl.q0[i] = loh.qBinChecked(m_q_center[i]);
        const float cut_dq = Window::dq_trk_fac * m_dq_track[i] +
                             Window::dq_hit_fac * loh.max_hit_q_half_length();
        auto qr = loh.qRangeBins(m_q_min[i] - cut_dq, m_q_max[i] + cut_dq);
        const int nq = (int)loh.qNBins();
        int qb = (int)qr.begin - Window::q_extra_bins;
        int qe = (int)qr.end   + Window::q_extra_bins;
        bl.q1[i] = (unsigned short)(qb < 0 ? 0 : qb);
        bl.q2[i] = (unsigned short)(qe > nq ? nq : qe);
      }
    }
  }

}  // namespace mkfit
