#ifndef RecoTracker_MkFitCore_src_V2p2Score_h
#define RecoTracker_MkFitCore_src_V2p2Score_h

#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/src/V2p2Config.h"

#include <cmath>

namespace mkfit {

  //============================================================================
  // The MkFinderV2p2 score.
  //
  // NOT track_score_func, deliberately. That one stays as it is for V1/V2 and
  // for the final output; this is a second, independent function for the layer
  // processing, and it differs from it in three ways that are structural rather
  // than a matter of tuning.
  //
  // 1. IT IS ADDITIVE OVER LAYER STEPS, BY CONSTRUCTION.
  //    global(cand) = seed term + sum over its layer steps of delta().
  //    That is what makes the end-of-layer selection a SINGLE FLAT SORT over
  //    every competing path of a CombCandidate -- the in-layer alternatives of
  //    every PrimTCandRep, plus the ones that declined the layer, plus the ones
  //    already stopped -- instead of a two-level arbitration with an arbitrary
  //    offset between the levels. "Score globally" and "score locally and add an
  //    offset" are then the same thing, which is the point.
  //
  //    track_score_func is additive only by accident and only sometimes:
  //    phase2:LstIntoPix happens to be (constant bonus, and it sets
  //    tailPenalty == missingHitPenalty so the inside/tail reshuffle cancels),
  //    while the default scorer is not -- its bonus is
  //    validHitSlope_ * nfoundhits + validHitBonus_, so bonus * nfoundhits is
  //    QUADRATIC in the hit count, and its two hole penalties differ (8 vs 3) so
  //    the retroactive inside/tail reclassification changes the total.
  //
  // 2. THERE IS NO TAIL-HOLE CONCEPT, AND THAT IS THE POINT.
  //    TrackCand splits holes into "inside" and "tail", reclassifying
  //    retroactively (addHitIdx does nInsideMinusOneHits_ += nTailMinusOneHits_
  //    whenever a hit is found), because the global formula is evaluated ONCE at
  //    the end and therefore has to ask whether a hole was later closed. Scoring
  //    each step as it happens makes that question both unanswerable and
  //    unnecessary.
  //
  //    What the tail penalty was standing in for is real, though, and it is
  //    DIRECTION-DEPENDENT: going outward, the trailing holes are at large
  //    radius, where a track legitimately runs out of detector, and they should
  //    be cheap; going inward, the trailing holes are at SMALL radius -- the
  //    head of the track, where it has to have come from -- and they are the
  //    most expensive holes there are. Getting this the wrong way round was a
  //    large phase-1 failure. Note the current code cannot even express the
  //    difference: SteeringParams::m_track_scorer is per REGION and
  //    findTracksStandardv2p2 hands the same one to IT_FwdSearch and
  //    IT_BkwSearch. Worse, mergeCandsAndBestShortOne scores with
  //    getScoreCand(f, c) -- penalizeTailMissHits defaults to FALSE -- and v2p2
  //    scores nowhere else, so in v2p2 today a trailing hole costs nothing at
  //    all, in either direction. An inward candidate that attaches no pixel hit
  //    whatsoever pays zero for it.
  //
  //    So the replacement is a hole penalty that is a function of DIRECTION and
  //    LAYER rather than of position in the hit-addition sequence -- which is
  //    also the natural home for a pixel-to-outer-tracker modifier.
  //
  // 3. IT TAKES A FEATURE STRUCT, NOT A FIXED ARGUMENT LIST.
  //    track_score_func(nfoundhits, ntailholes, noverlaphits, nmisshits, chi2,
  //    pt, inFindCandidates) cannot be given a new quantity without changing
  //    every implementation of it, ours and CMSSW's. Per-hit precision
  //    (q_half_length), which WSR class a hole was, and the layer transition all
  //    need to be in the score and none of them can get there. A struct also
  //    makes the features RECORDABLE: dumping them into the trace next to truth
  //    is what turns the choice of function into an offline fit rather than a
  //    hand-tune, and it is the only reason a learned or Pareto-optimised scorer
  //    is reachable from here at all.
  //============================================================================

  // Why a candidate took no hit in this layer. Mirrors the Hit::kHit*Idx values
  // that get recorded, but as a dense enum so it can index a parameter table.
  enum V2p2HoleKind_e {
    V2P2_NoHole = 0,  // hits were taken
    V2P2_Miss,        // sensitive region crossed, nothing found -- a real hole
    V2P2_Edge,        // crossing within dq of the layer boundary
    V2P2_Gap,         // crossing in an endcap r hole
    V2P2_Stop,        // out of hole budget
    V2P2_HoleKind_Count
  };

  // One layer step: everything that happened to one candidate between its state
  // at layer check-in and its state at the end of the layer. Kept a POD so it can
  // go into the trace unchanged.
  struct LayerStepFeatures {
    // What was taken.
    float chi2_sum = 0.0f;       // over the hits of this step
    float chi2_max = 0.0f;
    // Precision of the best hit taken, as the q half-length in cm -- small is
    // good. LayerOfHits::HitInfo carries it per hit already, so a P macro-pixel
    // (0.042 cm) and a 2S strip (2.5125 cm) are distinguishable without any
    // stereo bit, and in a 2S stack, where a stereo bit would be meaningless,
    // it correctly says the two sensors are equal.
    float q_half_len_best = 0.0f;
    // Sum over the hits taken of ln(det V), V being the 2x2 residual covariance
    // in the module frame -- the very matrix chi2 is formed with. chi2 is
    // dimensionless, so it has already divided the precision out; this is what
    // puts it back. Two hits at chi2 = 1, one a P macro-pixel and one a 2S strip,
    // are not equally informative and only this term says so.
    float log_det_v_sum = 0.0f;
    // ln of the local hit density in the search window, hits per cm^2. Per layer
    // step, so common to the hits of one path. A good-looking hit in a crowded
    // window is more likely to be an accident, and this is what makes the bar
    // rise with occupancy -- which is most of what a fitted chi2 cut in 1/pT and
    // theta was absorbing, occupancy varying with both.
    float log_rho = 0.0f;
    short n_hits = 0;            // hits taken in this layer: 0, 1, 2, ...
    short n_overlap = 0;         // = max(n_hits - 1, 0)
    short hole_kind = V2P2_NoHole;

    // Where. layer_from is the layer of the candidate's last found hit, so the
    // pair names the actual PROPAGATION, not plan adjacency -- a candidate that
    // missed a layer steps across two.
    short layer_from = -1;
    short layer_to = -1;
    bool  is_outward = true;
    bool  is_pixel = false;
    bool  is_barrel = true;

    // What the candidate is, for terms that cannot be per-hit.
    //
    // Three facets of "per layer" and they are genuinely different (maintainer):
    // the STEP -- how deep into the search we are, which `step` carries; the
    // LAYER itself, a local property, which layer_to carries; and the
    // TRANSITION from the previous layer to this one, which is a property of the
    // step and not of either layer. The transition is the one expected to matter
    // most -- entering the outer tracker from the pixels is a different
    // proposition from moving between two outer-tracker layers -- and it is why
    // layer_from is taken from the last FOUND hit rather than from the previous
    // plan entry: a candidate that missed a layer steps across two, and keying
    // on plan adjacency would name a propagation that never happened.
    float pt = 0.0f;
    short step = 0;              // HoTs so far, i.e. layers this candidate has traversed
    short n_found_so_far = 0;
    short n_holes_so_far = 0;
  };

  // The coefficients, the score mode and the term-ablation switches are in
  // V2p2Config.h, Config::V2p2::Score.

  // THE LOG-LIKELIHOOD RATIO, against "this layer produced no hit" as the
  // reference, which is why the hole scores exactly zero and needs no penalty of
  // its own.
  //
  //   take hit j :  ln(eps/(1-eps)) - ln(2pi) - chi2_j/2 - ln(det V_j)/2 - ln rho
  //   take none  :  0
  //
  // The first two terms are constants, the middle two come from the Gaussian
  // density of the residual, and the last from modelling the other hits in the
  // window as uniform at density rho. Dimensions cancel: det V is in cm^4 and rho
  // in hits/cm^2.
  //
  // What it buys over a tuned linear form is that the hit-versus-hole break-even
  // is DERIVED. At 30 um pixel resolution and rho = 1/cm^2 it sits near chi2 = 29;
  // for a 2S strip, near 17. Nobody chose either.
  //
  // It assumes chi2 is trustworthy, which it is only to the 1.4-2x the covariance
  // is still short by -- that biases the chi2 term without changing its shape.
  // And "background is uniform at rho" ignores that other tracks make CORRELATED
  // hits. Both are nameable and measurable, which the fitted constants are not.
  inline float v2p2_layer_step_loglh(const LayerStepFeatures &f, const V2p2ScoreParams &p) {
    if (f.n_hits == 0)
      return 0.0f;   // the reference hypothesis, whatever kind of hole it was
    const float c_eps = std::log(p.hit_eff / (1.0f - p.hit_eff)) - 1.8378771f;  // ln(2pi)
#if defined(MKFIT_STANDALONE)
    namespace ss = Config::V2p2::ScoreStats;
    if (ss::accum) {
      ss::n_hits += f.n_hits;
      ss::sum_log_rho += (double) f.n_hits * f.log_rho;
      ss::sum_log_detv += f.log_det_v_sum;
    }
#endif
    namespace sc = Config::V2p2::Score;
    // log_rho is per layer STEP, so it multiplies n_hits; log_det_v_sum is
    // already a sum over the hits taken, so its replacement must be scaled.
    const float lrho = sc::use_rho ? f.log_rho : sc::rho_const;
    const float ldetv = sc::use_detv ? f.log_det_v_sum : f.n_hits * sc::detv_const;
    return f.n_hits * (c_eps - lrho) - 0.5f * (f.chi2_sum + ldetv);
  }

  inline float v2p2_layer_step_score(const LayerStepFeatures &f) {
    namespace sc = Config::V2p2::Score;
    const V2p2ScoreParams &p = f.is_outward ? sc::fwd : sc::bkw;

    if (sc::mode == 1)
      return v2p2_layer_step_loglh(f, p);

    float s = p.hit_bonus * f.n_hits + p.overlap_bonus * f.n_overlap - p.chi2_weight * f.chi2_sum;

    switch (f.hole_kind) {
      case V2P2_Miss: s -= p.miss_penalty; break;
      case V2P2_Edge: s -= p.edge_penalty; break;
      case V2P2_Gap:  s -= p.gap_penalty;  break;
      case V2P2_Stop: s -= p.stop_penalty; break;
      default: break;
    }
    return s;
  }

  // Hit::kHit*Idx -> V2p2HoleKind_e, so the fake-hit decision and the score stay
  // one decision made in one place.
  inline V2p2HoleKind_e v2p2_hole_kind(int fake_hit_idx) {
    switch (fake_hit_idx) {
      case Hit::kHitMissIdx:  return V2P2_Miss;
      case Hit::kHitEdgeIdx:  return V2P2_Edge;
      case Hit::kHitInGapIdx: return V2P2_Gap;
      case Hit::kHitStopIdx:  return V2P2_Stop;
      default:                return V2P2_Miss;
    }
  }

}  // namespace mkfit

#endif
