#include "RecoTracker/MkFitCore/src/MkBins.h"
#include "RecoTracker/MkFitCore/src/MkRZLimits.h"

#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include <algorithm>
#include <cmath>

namespace mkfit {

  bool g_mkbins_surface_q = false;

  // Phi pre-selection, cut and fetch. ONE tolerance, not two factors: the
  // binnor must fetch everything the cut can accept, so the fetch range is
  // derived from the cut rather than tuned against it. That is what makes the
  // old failure mode -- a cut wider than the fetch, silently accepting nothing
  // extra -- impossible to express.
  float g_v2p2_dphi_trk_fac   = 1.0f;
  float g_v2p2_hit_dphi_rad   = MkBins::PHI_PRESEL_TOLERANCE;
  int   g_v2p2_phi_extra_bins = MkBins::PHI_EXTRA_BINS;

  // Transitional, for A/B against the old behaviour only. The old range was
  // hand-rolled as a pair of phiBinChecked() calls with NO "+1", and the scan
  // loops consume [p1, p2) half-open, so the bin holding the upper edge was
  // never scanned -- on every range, not occasionally. PHI_BIN_EXTRA_FAC was
  // carrying a whole spare bin to cover that: measured free at 1.00 bins of
  // margin and lossy at 0.88. Delete this once the A/B is recorded.
  bool  g_v2p2_phi_legacy_range = false;

  // Q fetch margin, in units of half a q_bin -- i.e. Q_BIN_EXTRA_FAC, runtime.
  // Exists to test whether the q side has the decoupling the phi side had: the
  // CUT accepts EXTRA_DQ * DDQ_PRESEL_FAC * hit_q_half_length, which in TB2S at
  // EXTRA_DQ = 3 is 9.05 cm, against a fetch margin of 1.6 * 0.5 * 6.0 = 4.8.
  // If that is live, half the cut's reach was never fetched.
  float g_v2p2_q_bin_extra_fac = MkBins::Q_BIN_EXTRA_FAC;

  float g_v2p2_dq_trk_fac   = MkBins::DQ_TRK_FAC;
  float g_v2p2_dq_hit_fac   = MkBins::DQ_HIT_FAC;
  int   g_v2p2_q_extra_bins = MkBins::Q_EXTRA_BINS;
  bool  g_v2p2_q_legacy_range = false;

  // Largest representable phi half-width: a hair under pi, since at pi the two
  // endpoints coincide and the arc degenerates to a point.
  static constexpr float kMaxHalfPhiWindow = 3.14f;

  namespace mp = mini_propagators;

  //==============================================================================

  void MkBins::prop_to_limits(const LayerInfo &li) {
    // Positions 1 and 2 should really be by "propagation order", 1 is the closest/
    // This should also work for backward propagation so not exactly trivial.
    // Also, do not really need propagation to center. Well, to be checked, and
    // to figure out error scaling factors / correction functions.
    m_is_barrel = li.is_barrel();
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

    m_is_barrel = ls.m_is_barrel;
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
        m_phi_delta *= 0.5f;
        // printf("phi_c: %f  p1: %f  p2: %f   m_phi_min: %f  m_phi_max: %f   dphi: %f\n",
        //       m_phi_center[ii], xp1[ii], xp2[ii], m_phi_min[ii], m_phi_max[ii], m_phi_delta[ii]);
      }
    }

    // Calculate dphi_track, dq_track differs for barrel/endcap
    MPlexQF r2_c = m_isp.x * m_isp.x + m_isp.y * m_isp.y;
    MPlexQF r2inv_c = 1.0f / r2_c;

    // sigma_phi = J sigma_xy J^T with J = grad phi = (-y, x)/r^2, so |J| = 1/r and
    // sigma_phi is LARGEST at the smallest radius the track crosses inside the layer.
    // One window has to cover the whole layer, so evaluate J there -- the conservative
    // end -- rather than at whichever end the propagation happened to stop at.
    //
    // This also fixes an inconsistency: cov_ex is the covariance at m_sp2, while m_isp
    // is left at m_sp1 by prop_to_limits_in_order(), so J and sigma were being taken at
    // different points. |J| = 1/r makes that a direct scale error, and rout/rin is 1.34
    // at pixel layer 0, 1.27 at the inner TOB double layer.
    //
    // Note it is a no-op for an OUTWARD search, where m_sp1 is already at rin; it only
    // widens INWARD searches, which is where the window was too tight.
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

    if (g_mkbins_surface_q)
      surface_reference_dq(cov_ex);
  }

  //----------------------------------------------------------------------------
  // Reference the q variance to the LAYER SURFACE.
  //
  // errPropFromPathL_impl() takes no plane -- it transports the curvilinear
  // jacobian to a fixed PATH LENGTH s -- and MkBinTrackCovExtract then reads
  // err(2,2) verbatim as the q variance. So the window covariance describes the
  // spread of where the track is AFTER A GIVEN DISTANCE, not the spread of where
  // it crosses the layer. The Kalman update does carry this term, in
  // jacCurv2Loc's cosz block (KalmanUtilsMPlex.cc); the window never has.
  //
  // The correction is a pure linear map on the POSITION block: slide each sample
  // along the momentum until it meets the surface,
  //
  //     dx_s = (I - p^ n^T / (n^.p^)) dx
  //
  // with n the surface normal (radial for a barrel cylinder, z for an endcap
  // disc). For a radial barrel track it amplifies sigma_q by exactly 1/sin^2(t)
  // -- unity at eta = 0, 8.3x at |eta| = 2 -- and leaves sigma_phi untouched,
  // since the whole correction lies in the (r,z) plane.
  //
  // Everything is evaluated at m_sp2, which is where pea propagated to and hence
  // where cov_ex lives. (The dphi jacobian above deliberately uses the smaller
  // radius instead; that is a conservative choice for a 1/r scale factor, not a
  // consistency requirement.)
  //
  // The two limits recover the old code exactly: g -> 0 (normal incidence on a
  // barrel) gives Var = C22, and 1/g -> 0 (normal incidence on a disc) gives the
  // old radial projection.
  //----------------------------------------------------------------------------

  void MkBins::surface_reference_dq(const MkBinTrackCovExtract &cov_ex) {
    // Amplification clamp. g = cot(theta) for a radial barrel track, so 20 is
    // |eta| ~ 3.7 -- beyond the tracker, i.e. it only ever catches degenerate
    // lanes (grazing incidence, failed propagation) and never a real operating
    // point.
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
        // Clamp crazy sizes. This actually only happens when prop-fail flag is set.
        // const float dphi_clamp = 0.1;
        // if (dphi_min[i] > 0.0f || dphi_min[i] < -dphi_clamp) dphi_min[i] = -dphi_clamp;
        // if (dphi_max[i] < 0.0f || dphi_max[i] > dphi_clampf) dphi_max[i] = dphi_clamp;
        // Fetch exactly what the cut can accept, then extend by whole BINS.
        // Keeping the extender in bin units is the point: it is added to the bin
        // INDEX, so it introduces no float-to-bin rounding of its own.
        // PRECONDITION of the range helper, and it is ours to enforce. A range
        // on a circle is an arc; a half-width at or above pi is not an arc and
        // wraps to an arbitrary SMALL one, silently. This is what the old
        // commented-out "clamp crazy sizes ... only happens when prop-fail flag
        // is set" was reaching for -- it is a precondition, not a workaround.
        const float cut_dphi = std::min(g_v2p2_dphi_trk_fac * m_dphi_track[i] + g_v2p2_hit_dphi_rad,
                                        kMaxHalfPhiWindow);
        if (g_v2p2_phi_legacy_range) {
          const float old_dphi = g_v2p2_dphi_trk_fac * m_dphi_track[i] +
                                 PHI_BIN_EXTRA_FAC * HIT_PHI_HALF_EXTENT;
          bl.p1[i] = loh.phiBinChecked(m_phi_min[i] - old_dphi);
          bl.p2[i] = loh.phiBinChecked(m_phi_max[i] + old_dphi);
        } else {
          auto pr = loh.phiRangeBins(m_phi_min[i] - cut_dphi, m_phi_max[i] + cut_dphi);
          bl.p1[i] = loh.phiMaskApply(pr.begin - g_v2p2_phi_extra_bins);
          bl.p2[i] = loh.phiMaskApply(pr.end   + g_v2p2_phi_extra_bins);
        }

        bl.q0[i] = loh.qBinChecked(m_q_center[i]);
        if (g_v2p2_q_legacy_range) {
          // Old: margin keyed on the BIN WIDTH, unrelated to what the cut
          // accepts. In TB2S that fetched 4.8 cm against a cut reaching 9.05 at
          // EXTRA_DQ = 3 -- half the cut's reach was never pulled.
          const float q_margin = m_dq_track[i] + g_v2p2_q_bin_extra_fac * 0.5f * loh.layer_info().q_bin();
          bl.q1[i] = loh.qBinChecked(m_q_min[i] - q_margin);
          bl.q2[i] = loh.qBinChecked(m_q_max[i] + q_margin) + 1;
        } else {
          // Fetch exactly what the cut can accept, using the layer's WORST-CASE
          // hit extent since the per-hit one is not known until the hit is in
          // hand, then extend by whole bins on the INDEX. The q axis is bounded,
          // so the extension CLAMPS where the phi one wraps.
          const float cut_dq = g_v2p2_dq_trk_fac * m_dq_track[i] +
                               g_v2p2_dq_hit_fac * loh.max_hit_q_half_length();
          auto qr = loh.qRangeBins(m_q_min[i] - cut_dq, m_q_max[i] + cut_dq);
          const int nq = (int)loh.qNBins();
          int qb = (int)qr.begin - g_v2p2_q_extra_bins;
          int qe = (int)qr.end   + g_v2p2_q_extra_bins;
          bl.q1[i] = (unsigned short)(qb < 0 ? 0 : qb);
          bl.q2[i] = (unsigned short)(qe > nq ? nq : qe);
        }
      }
    }
  }

}  // namespace mkfit
