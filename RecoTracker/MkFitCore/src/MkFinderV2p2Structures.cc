#include "MkFinderV2p2Structures.h"
#include "PropagationMPlex.h"
#include "KalmanUtilsMPlex.h"
#include "MatriplexPackers.h"

#define DEBUG
#include "Debug.h"

namespace mkfit {

  void KalmanOpArgs::do_kalman_stuff() {
    bool debug = true;

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
        tsErr.copyOut(i, ptcp[i]->bState.errors.Array());
        tsPar.copyOut(i, ptcp[i]->bState.parameters.Array());
        ptcp[i]->bState.charge = tsChg[i];
        ptcp[i]->bHot = hot[i];
        ptcp[i]->bChi2 = tsChi2[i];
      }
    }
  }

}
