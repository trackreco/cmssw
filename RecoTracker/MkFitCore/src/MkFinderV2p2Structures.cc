#include "MkFinderV2p2Structures.h"
#include "MkFinderV2p2.h"
#include "PropagationMPlex.h"
#include "KalmanUtilsMPlex.h"
#include "MatriplexPackers.h"

// NOTE: this is ON here and OFF in MkFinderV2p2.cc (`//#define DEBUG`), so a
// g_debug run prints the Kalman side and not the search side. Left as it is
// rather than "fixed", because which half you want is a choice -- but it is a
// choice, not the accident the one-character asymmetry looks like.
#define DEBUG
#include "Debug.h"

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

namespace mkfit {

  //----------------------------------------------------------------------------
  // SecTCandRep storage -- decision and plan.
  //
  // In-layer combinatorial search produces, for every PrimTCandRep, a set of
  // partial extensions: hits from both sub-layers and from module overlaps,
  // taken in propagation order, with Kalman updates applied along the way. Which
  // of those paths materialise into the CombCandidate as TrackCands is decided
  // only at end_layer(). So what has to be stored is a DECISION TREE -- exactly
  // the shape CombCandidate::m_hots already uses for its HoTNodes, and for the
  // same reason: every node needs a way back to its predecessor, nothing needs a
  // way forward, and the whole thing is reclaimed in one go.
  //
  // A plain std::vector<SecTCandRep> is the simpler and cleaner implementation of
  // that. A node's index IS its handle, a single int m_parent_idx gives the way
  // back, appending is a push_back, and end_layer() is a size rewind that keeps
  // capacity -- after a few layers the arena is at high water and stops
  // allocating. Nothing else is needed: no per-node bookkeeping, no free list.
  //
  // Design points to keep when this gets built:
  //
  //  - ONE arena per MkFinderV2p2 (i.e., per thread), NOT one per CCandRep. A
  //    Matriplex batch will want to pack lanes from SecTCandReps drawn across
  //    several PrimTCandReps and several CCandReps, so a single base pointer must
  //    reach all of them. (MkFinder carries const HoTNode *m_HoTNodeArr[NN] --
  //    NN separate bases chased scalar-ly -- precisely because m_hots is
  //    per-CombCandidate.)
  //
  //  - Reach nodes by INDEX, never by pointer. Then the vector may grow freely:
  //    the single-allocation requirement for slurpIn is instantaneous (the
  //    offsets are computed right before the gather), unlike CcPool's, which is
  //    hard because raw TrackCand*s outlive it.
  //
  //  - Expand the frontier breadth-first BY TREE DEPTH, not depth-first per rep.
  //    That is what fills NN lanes from the whole depth-d frontier at once, and
  //    it keeps parent_idx < child_idx unconditionally true, so a forward sweep
  //    over the arena is always a valid topological order.
  //
  //  - The frontier is search state, not node state: a std::vector<int> of live
  //    leaves, double-buffered per sub-layer / overlap step, carrying the hit
  //    cursor (skipping a hit is a cursor advance) and the running hole count.
  //    The node itself carries only parent, hit, chi2 and the updated state --
  //    what is needed to register hits into the CombCandidate at end of layer.
  //
  //  - Materialise survivors at EVERY end_layer(). Holding to that as an
  //    invariant is what makes the bulk rewind correct and means finished or
  //    dropped CombCandidates never need to be weeded out of the arena.
  //
  // Note the lifetime asymmetry against m_hots, which is why this is a separate
  // arena rather than an extension of HoTNode: sizeof(HoTNode) is 12 B and it
  // lives for the whole event, while a SecTCandRep is ~136 B (a TrackState alone
  // is 112) and lives for one layer.
  //----------------------------------------------------------------------------

  void PropErrsArgs::do_propagation_stuff() {
    MPlexHV dummy {0.0f};
    propagateHelixToPlaneMPlex(tsErr, tsPar, tsChg, dummy, dummy, &sPerp,
                               propErr, propPar, outFailFlag,
                               N_filled, prop_config->finding_inter_layer_pflags, nullptr);
  }

  void KalmanOpArgs::do_kalman_stuff() {
    // bool debug = true;

    dprintf("do_kalman_stuff\n");
    for (int i = 0; i < N_filled; ++i) {
      dprintf("  %d: %f %f %f : %f %f %f : %f\n", i, tsXyz.x[i], tsXyz.y[i], tsXyz.z[i],
              tsXyz.inv_pt[i], vdt::fast_atan2(tsXyz.py[i], tsXyz.px[i]), tsXyz.theta[i],
              sPerp[i]);
    }
    // m_solve_plane is set exactly for the steps that stay WITHIN one layer -- the
    // second and later hits of an in-layer path -- so those take the intra-layer
    // flags. With Config::usePropToPlane on (which phase-2 sets) the two sets are
    // configured identically today, so this is naming rather than behaviour; it
    // stops being free the moment they diverge.
    propagateHelixToPlaneMPlex(tsErr, tsPar, tsChg, plPnt, plNrm, m_solve_plane ? nullptr : &sPerp,
                               propErr, propPar, outFailFlag, N_filled,
                               m_solve_plane ? prop_config->finding_intra_layer_pflags
                                             : prop_config->finding_inter_layer_pflags,
                               nullptr);

#ifdef MKFIT_TRACE_KALMAN_DEBUG
    // Charge as it goes into the update. Propagation does not change it, but the
    // update can -- through a curvature flip, formalized by
    // kalmanCheckChargeFlip() below -- and a flip is important information, so
    // keep the pre-update value for propagated_state and let updated_state carry
    // the post-update one. Comparing the two then shows the flip.
    const MPlexQI chg_pre_update = tsChg;
#endif

    kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                              propErr, propPar, tsChg, msErr, msPar, plNrm, plDir, plPnt,
                              tsErr, tsPar, tsChi2, N_filled);
    kalmanCheckChargeFlip(tsPar, tsChg, N_filled);

    // The original -- but Chi2 only.
    // kalmanPropagateAndComputeChi2Plane(tsErr, tsPar, tsChg, msErr, msPar, plNrm, plDir, plPnt,
    //                       nullptr,
    //                       tsChi2,
    //                       propPar,
    //                       outFailFlag,
    //                       N_filled,
    //                       prop_config->finding_intra_layer_pflags,
    //                       prop_config->finding_requires_propagation_to_hit_pos);

    // Update prim candidate state for best hit -- to be generalized
    dprintf("Kalman post-update check:\n");
    for (int i = 0; i < N_filled; ++i) {
#ifdef MKFIT_TRACE
      bool is_mc = false;
      if (tr_hitmatch_ids[i] >= 0)
        is_mc = mp_event->tr_hitmatch(tr_hitmatch_ids[i]).mc_match;
      // Keyed selection: normally the chi2 itself. With forcing, an MC-matched
      // hit is shifted below every non-matched one but still ordered against
      // other MC-matched hits by its own chi2, so "the best chi2 one" wins.
      const float key = (g_v2p2_force_mc && is_mc) ? (tsChi2[i] - 1.0e6f) : tsChi2[i];
      if (key < ptcp[i]->bKey) {
        ptcp[i]->bKey = key;
        ptcp[i]->bIsMc = is_mc;
#else
      if (tsChi2[i] < ptcp[i]->bChi2) {
#endif
        dprintf("  Updating for i=%d, old-chi2 %f, new %f\n", i, ptcp[i]->bChi2, tsChi2[i]);
        tsPar.copyOut(i, ptcp[i]->bState.parArray_nc());
        tsErr.copyOut(i, ptcp[i]->bState.errArray_nc());
        ptcp[i]->bState.charge = tsChg[i];
        ptcp[i]->bHot = hot[i];
        ptcp[i]->bChi2 = tsChi2[i];
#ifdef MKFIT_TRACE
        ptcp[i]->b_tr_hitmatch_id = tr_hitmatch_ids[i];
#endif
      }

#ifdef MKFIT_TRACE
      // Create TrKalmanUpdate for EVERY hit that went through Kalman
      int hm_id = tr_hitmatch_ids[i];
      TrHitMatch &hm = mp_event->tr_hitmatch(hm_id);

      TrKalmanUpdate &ku = mp_event->trace_kalmanupdate({ -1, hm_id, hm.state_id });
      ku.chi2 = tsChi2[i];

      // Do not know this yet -- how will I know?
      // ku.chi2_trk = 0.0f;  // Could track cumulative if needed
      // What does accepted mean? pass_chi2 cut / score cut ... when we have it.
      // ku.accepted = (ptcp[i]->b_tr_hitmatch_id == hm_id);
      // ku.state_id_out = ku.accepted ? ptcp[i]->tcand().m_trace_state_id : -1;

#ifdef MKFIT_TRACE_KALMAN_DEBUG
      // Full pre- and post-update states, for every hit including the rejected
      // ones (for an accepted hit the post-update state is also reachable as
      // trCandStates_[state_id_out].state).
      // Charges straddle the update on purpose: pre-update on the propagated
      // state, post-update on the updated one.
      propPar.copyOut(i, ku.propagated_state.parArray_nc());
      propErr.copyOut(i, ku.propagated_state.errArray_nc());
      ku.propagated_state.charge = chg_pre_update[i];

      tsPar.copyOut(i, ku.updated_state.parArray_nc());
      tsErr.copyOut(i, ku.updated_state.errArray_nc());
      ku.updated_state.charge = tsChg[i];
#endif

      // Link forward from HitMatch
      hm.kalman_id = ku.id;
#endif

      if (mp_out) {
        ItemOut o;
        o.ptc = ptcp[i];
        o.parent_idx = parent_idx[i];
        o.hit_pos = hit_pos[i];
        o.hot = hot[i];
        o.hit_in_layer = hit_in_layer[i];
        o.chi2 = tsChi2[i];
        tsPar.copyOut(i, o.state.parArray_nc());
        tsErr.copyOut(i, o.state.errArray_nc());
        o.state.charge = tsChg[i];
        o.state.valid = true;
#ifdef MKFIT_TRACE
        o.tr_hitmatch_id = tr_hitmatch_ids[i];
#endif
        mp_out->push_back(o);
      }
    }
  }

}
