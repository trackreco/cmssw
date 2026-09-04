#include "MkFinderV2p2Structures.h"
#include "PropagationMPlex.h"
#include "KalmanUtilsMPlex.h"
#include "MatriplexPackers.h"

#define DEBUG
#include "Debug.h"

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

namespace mkfit {

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
    propagateHelixToPlaneMPlex(tsErr, tsPar, tsChg, plPnt, plNrm, &sPerp,
                               propErr, propPar, outFailFlag,
                               N_filled, prop_config->finding_inter_layer_pflags, nullptr);
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
      if (tsChi2[i] < ptcp[i]->bChi2) {
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

      // Link forward from HitMatch
      hm.kalman_id = ku.id;
#endif
    }
  }

}
