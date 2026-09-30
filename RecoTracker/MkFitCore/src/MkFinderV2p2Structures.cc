#include "MkFinderV2p2Structures.h"
#include "MkFinderV2p2.h"
#include "PropagationMPlex.h"
#include "KalmanUtilsMPlex.h"
#include "MatriplexPackers.h"

// DEBUG is on here and off in MkFinderV2p2.cc, so a g_debug run prints the
// Kalman side and not the search side.
#define DEBUG
#include "Debug.h"

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

namespace mkfit {

  void KalmanOpArgs::do_kalman_stuff() {
    // bool debug = true;

    dprintf("do_kalman_stuff\n");
    for (int i = 0; i < N_filled; ++i) {
      dprintf("  %d: %f %f %f : %f %f %f : %f\n", i, tsXyz.x[i], tsXyz.y[i], tsXyz.z[i],
              tsXyz.inv_pt[i], vdt::fast_atan2(tsXyz.py[i], tsXyz.px[i]), tsXyz.theta[i],
              sPerp[i]);
    }
    // m_solve_plane marks steps within one layer (second and later hits of an
    // in-layer path), which take the intra-layer propagation flags.
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
                              tsErr, tsPar, tsChi2, N_filled, nullptr, nullptr, false,
                              // det V is read only by the likelihood score.
                              Config::V2p2::Score::mode == 1 ? &tsDetV : nullptr);
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
      const float key = (Config::V2p2::Diag::force_mc && is_mc) ? (tsChi2[i] - 1.0e6f) : tsChi2[i];
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
        o.det_v = tsDetV[i];
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
