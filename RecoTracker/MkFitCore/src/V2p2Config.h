#ifndef RecoTracker_MkFitCore_src_V2p2Config_h
#define RecoTracker_MkFitCore_src_V2p2Config_h

#include "RecoTracker/MkFitCore/interface/Config.h"

//==============================================================================
// Tunable parameters of MkFinderV2p2 and of the v2p2 window in MkBins.
//
// Each knob is declared once, with its default, through MKFIT_V2P2_KNOB:
//   - CMSSW build:      constexpr. No mutable global state, and branches on the
//                       switches fold away at compile time.
//   - standalone build: a mutable variable, set by mkFit.cc options and by the
//                       shell drivers in standalone/RdfTrace/ValProp.cc. The
//                       definitions are in V2p2Config.cc, which includes this
//                       header with MKFIT_V2P2_CONFIG_DEFINE set.
//
// These are meant to move into IterationParams, per iteration and settable from
// the JSON configs; the defaults here then become the IterationParams defaults.
//
// V2's frozen constants stay in MkBins: they are production V2 as in upstream
// CMSSW and are not tuned here.
//==============================================================================

#if !defined(MKFIT_STANDALONE)
#define MKFIT_V2P2_KNOB(type, name, value) constexpr type name = value
#elif defined(MKFIT_V2P2_CONFIG_DEFINE)
#define MKFIT_V2P2_KNOB(type, name, value) type name = value
#else
#define MKFIT_V2P2_KNOB(type, name, value) extern type name
#endif

namespace mkfit {

  // Coefficients of the layer-step score, V2p2Score.h. One set per direction,
  // which is the minimum that expresses the head/body asymmetry described
  // there; per-layer and per-transition modifiers belong here too and are
  // deliberately not invented yet -- there is no measurement behind them.
  struct V2p2ScoreParams {
    float hit_bonus = 30.0f;     // per hit taken
    float overlap_bonus = 0.0f;  // per hit beyond the first in a layer
    float chi2_weight = 1.0f;    // per unit of chi2
    float miss_penalty = 8.0f;   // a real hole
    float edge_penalty = 0.0f;   // crossing the boundary -- absence is explained
    float gap_penalty = 0.0f;    // inactive module -- absence is explained
    float stop_penalty = 8.0f;   // out of hole budget

    // Likelihood mode. Per-layer hit efficiency; the only free number in it, and
    // a measurable one rather than a tuned one.
    float hit_eff = 0.99f;
  };

  namespace Config::V2p2 {

    //--------------------------------------------------------------------------
    // Pre-selection window: the per-hit dq / dphi cut in
    // MkFinderV2p2::preselect_hit_batch(), and the binnor fetch derived from it
    // in MkBins::find_bin_ranges().
    //
    //   ddq   < dq_trk_fac   * dq_trk     + dq_hit_fac   * hit_q_half_length
    //   ddphi < dphi_trk_fac * dphi_track + dphi_hit_fac * hit_phi_half_extent
    //
    // dq_track and dphi_track are 3 sigma. dq_trk is dq_track referenced to the
    // hit's module surface when surface_q is on.
    namespace Window {
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
      //   dq_trk_fac  multiplies dq_track, which is itself 3 sigma
      //   dq_hit_fac  multiplies the hit's own half-extent -- FLOOR IS EXACTLY 1.0,
      //               below which the window stops reaching the strip it is trying
      //               to contain. (In the old compound units that floor was the
      //               opaque 1/1.2 = 0.833.)
      //
      // dq_trk_fac is EXTRA_DQ = 1.5 in the old units. dq_hit_fac sits 20 % above
      // its floor: taking it from 1.8 (the old ratio) to 1.2 is worth 3-6 % of
      // build time and is a physics null both ways -- forward -3 found tracks and
      // -15 fakes, inward +55 recovered chopped hits.
      MKFIT_V2P2_KNOB(float, dq_trk_fac, 1.5f);
      MKFIT_V2P2_KNOB(float, dq_hit_fac, 1.2f);

      // Reference the q error to the HIT'S OWN MODULE PLANE before the cut, see
      // MkFinderV2p2::surface_referenced_dq(). Exact for tilted and flat layers.
      MKFIT_V2P2_KNOB(bool, surface_q, true);

      // Fetch margin beyond what the cut accepts, in WHOLE q bins on the index.
      // NOT removable yet, although it costs 22 % of build time without the line
      // pre-cut: the per-hit q cut uses the SURFACE-REFERENCED dq (growing
      // ~cosh^2(eta), ~14x at |eta| 2 in the pixel barrel) while the fetch uses
      // the raw m_dq_track. At 0 the cut is wider than the fetch at high eta and
      // the inward search loses 1373 of 43358 fully recovered chopped pT5 tracks,
      // all in pixel-barrel hits of disc-touching tracks. Forward is free.
      MKFIT_V2P2_KNOB(int, q_extra_bins, 1);

      // THE dphi CUT. The hit term is the hit's own phi extent, derived from its
      // covariance (LayerOfHits::hit_phi_half_extent(), the phi counterpart of
      // hit_q_half_length()), times dphi_hit_fac.
      //
      // The per-hit extent is ~4e-5 rad in TB2S, so the cut is carried by the
      // track term, and dphi_trk_fac = 2 is what makes that work: 1.0 loses 41
      // found tracks forward and 582 fully recovered chopped pT5 tracks inward,
      // 1.75 is the lowest free value forward, and 2.0 is free in both directions
      // and reproduces the flat constant to the unit inward. Above |eta| 0.8 the 2
      // also compensates a phi covariance measured 2.2-3.4x short (the material
      // model in the transition and forward region); at central eta, where the
      // covariance is within 1.25 of right, it is containment.
      MKFIT_V2P2_KNOB(float, dphi_trk_fac, 2.0f);
      MKFIT_V2P2_KNOB(bool, phi_per_hit, true);
      MKFIT_V2P2_KNOB(float, dphi_hit_fac, 3.0f);

      // Flat phi tolerance, in RADIANS, used only with phi_per_hit off. Equals
      // one phi bin at N = 8 by accident of history, not by design -- see the note
      // on MkBins::HIT_PHI_HALF_EXTENT.
      MKFIT_V2P2_KNOB(float, dphi_flat_rad, 2.0f * 0.0123f);

      // Safety margin the BINNOR adds beyond what the cut can accept, in WHOLE
      // BINS added to the bin INDEX -- so it carries no float-to-bin rounding of
      // its own. It was meant for the per-hit reference phi, the Hermite's
      // crossing at the hit's own module plane, falling slightly outside the
      // [phi_min, phi_max] span the range is built from. Measured at the current
      // window: 0 is free forward (-1 found track) and inward (+1 chopped hit),
      // and one spare bin costs 20 % of build time, since every fetched hit
      // costs a plane solve per candidate.
      MKFIT_V2P2_KNOB(int, phi_extra_bins, 0);
    }  // namespace Window

    //--------------------------------------------------------------------------
    // LINE PRE-CUT (MkFinderV2p2::select_hits). Before the plane solve, the
    // track between its two layer crossings m_sp1 and m_sp2 is taken as a
    // straight line in (qbar, q) and (qbar, phi), evaluated at the hit's own
    // qbar, and the hit is dropped if it is outside a tolerance that is looser
    // than the real dq / dphi cut. Tolerances, with g the line's slope:
    //   q:   dq_slack * dq_trk_fac * dq_track * (1 + g^2)
    //        + dq_hit_fac * hit_q_half_length + qbar_fac * |g| * hit_qbar_half_extent
    //   phi: dphi_slack * dphi_trk_fac * dphi_track + the real cut's hit term
    //        + qbar_fac * |g_phi| * hit_qbar_half_extent
    // (1 + g^2) references dq_track to the layer surface, as the real cut does
    // per hit. The qbar terms cover a tilted strip, whose centroid r is uncertain
    // along the strip; they are applied in the barrel only. See
    // doc/MkFinderV2p2-DesignNotes.md, "Line pre-cut".
    namespace PreCut {
      MKFIT_V2P2_KNOB(bool, q, true);
      MKFIT_V2P2_KNOB(bool, phi, true);
      MKFIT_V2P2_KNOB(float, dq_slack, 2.0f);
      MKFIT_V2P2_KNOB(float, dphi_slack, 1.5f);
      MKFIT_V2P2_KNOB(float, qbar_fac, 1.2f);
    }  // namespace PreCut

    //--------------------------------------------------------------------------
    // In-layer combinatorial search.
    namespace InLayer {
      // Walk the pre-selected hits forward along the trajectory and take a
      // SEQUENCE of them, rather than the single best-chi2 hit. Off recovers the
      // best-hit path. Production pairs it with maxCandsPerSeed = 3.
      MKFIT_V2P2_KNOB(bool, comb, true);

      // Reduction cap, PER SUB-LAYER. It sits UPSTREAM of the in-layer
      // combinatorial search, so it bounds what that search can ever see, and one
      // number cannot be right everywhere: overlap availability runs from 2-3 % of
      // pixel-barrel crossings to 50-56 % of TFPX. Per layer is where it should
      // end up.
      MKFIT_V2P2_KNOB(int, max_presel_hits, 6);

      // Most hits one in-layer path may take.
      MKFIT_V2P2_KNOB(int, max_sec_depth, 4);
      // Array size for a path's hit chain; max_sec_depth must not exceed it.
      constexpr int max_sec_depth_limit = 8;

      // Keep one beam slot for a continuation that DECLINED the layer, even when
      // it is outranked. A hole is the only move that does not shrink the
      // covariance, so it is the only branch that stays open to the possibility
      // that an earlier hit was wrong. See MkFinderV2p2::select_and_materialise().
      MKFIT_V2P2_KNOB(bool, reserve_hole_slot, false);

      // Best-short. Move a STOPPED TrackCand out of the beam and remember the best
      // of them on the CombCandidate, the way V1's CandCloner does. Outward search
      // only -- going inward the "tail" is the head of the track.
      MKFIT_V2P2_KNOB(bool, best_short, false);
    }  // namespace InLayer

    //--------------------------------------------------------------------------
    // Per-layer candidate policy, and hit acceptance.
    namespace Policy {
      // Within-sensitive-region verdict; with it off a layer the track never
      // crosses is recorded as a hole.
      MKFIT_V2P2_KNOB(bool, use_wsr, true);
      // Edge fuzz of the WSR, in sigma of the track's q error.
      MKFIT_V2P2_KNOB(float, wsr_n_sigma, 5.0f);

      // maxHolesPerCand / maxConsecHoles from IterationParams.
      MKFIT_V2P2_KNOB(bool, use_hole_limits, true);

      // Candidate-stopping cuts at pickup: minPtCut from IterationParams, and the
      // looper stop below. Forward search only for the looper stop.
      MKFIT_V2P2_KNOB(bool, use_stop_cuts, true);
      // Stop once the transverse angle between position and momentum exceeds
      // this, for pT below looper_max_pt and r above looper_min_r.
      MKFIT_V2P2_KNOB(float, looper_max_angle, Const::PIOver2 - 0.2f);
      MKFIT_V2P2_KNOB(float, looper_max_pt, 1.2f);
      MKFIT_V2P2_KNOB(float, looper_min_r, 25.0f);

      // Kalman chi2 acceptance of a hit, best-hit and in-layer paths alike.
      MKFIT_V2P2_KNOB(float, hit_chi2_cut, 30.0f);
    }  // namespace Policy

    //--------------------------------------------------------------------------
    // Layer-step score, V2p2Score.h.
    namespace Score {
      // 0 = the linear form, kept as the A/B reference. 1 = the log-likelihood
      // ratio. See v2p2_layer_step_score().
      MKFIT_V2P2_KNOB(int, mode, 0);

      // Outward: trailing holes sit at large radius where a track legitimately runs
      // out of detector. Inward: they sit at small radius, where the track has to
      // have come from. Same numbers to start with, so the first A/B measures the
      // SELECTION and not a simultaneous change of the penalty.
      MKFIT_V2P2_KNOB(V2p2ScoreParams, fwd, {});
      MKFIT_V2P2_KNOB(V2p2ScoreParams, bkw, {});

      // TERM ABLATION of the likelihood. rho and eps enter the score IDENTICALLY
      // -- both multiply n_hits -- so replacing log_rho by a constant is EXACTLY a
      // global shift of eps, and the only thing it can remove is rho's VARIATION.
      // Set the constant to the measured mean and the ablation is variation-only.
      MKFIT_V2P2_KNOB(bool, use_rho, true);
      MKFIT_V2P2_KNOB(float, rho_const, 0.0f);   // ln(hits/cm^2), used when use_rho is false
      MKFIT_V2P2_KNOB(bool, use_detv, true);
      MKFIT_V2P2_KNOB(float, detv_const, 0.0f);  // ln(det V) per hit, used when use_detv is false
    }  // namespace Score

    //--------------------------------------------------------------------------
    // Diagnostic instruments, default off. Not physics configuration.
    namespace Diag {
      // Make the MC-matched hit win its layer and bypass the chi2 acceptance cut.
      // Meaningful only in a MKFIT_TRACE build. See PrimTCandRep::bKey.
      MKFIT_V2P2_KNOB(bool, force_mc, false);

      // Reference the window covariance to the LAYER CYLINDER in MkBins, see
      // MkBins::surface_reference_dq(). Superseded by Window::surface_q for the
      // cut; kept because it is the route by which a corrected dq_track reaches
      // the trace.
      MKFIT_V2P2_KNOB(bool, mkbins_surface_q, false);
    }  // namespace Diag

#if defined(MKFIT_STANDALONE)
    // Running means of the likelihood terms, so the ablation constants above are
    // measured and not guessed. Not thread safe: switch accumulation on only in a
    // serialised (trace) build, which is what MkFitTbb.h gives under TBB_DEBUG.
    namespace ScoreStats {
      MKFIT_V2P2_KNOB(bool, accum, false);
      MKFIT_V2P2_KNOB(long, n_hits, 0);
      MKFIT_V2P2_KNOB(double, sum_log_rho, 0.0);  // weighted by n_hits
      MKFIT_V2P2_KNOB(double, sum_log_detv, 0.0);
    }  // namespace ScoreStats

    // The OLD compound dq knob, kept as a setter so recorded scans reproduce
    // (--v2p2-extra-dq, val_extra_dq, test/v2p2-*-dq.sh): it writes dq_trk_fac and
    // dq_hit_fac in the old 1 : 1.2 ratio. That ratio is NOT the default any more
    // (1.5 : 1.2), so no argument reproduces the defaults.
    void set_extra_dq(float f);
#endif

  }  // namespace Config::V2p2

}  // namespace mkfit

#undef MKFIT_V2P2_KNOB

#endif
