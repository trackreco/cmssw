#ifndef RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h
#define RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include "HotTub.h"
#include "MiniPropagators.h"

#include <queue>

namespace mkfit {

    struct CCandRep;

    //----------------------------------------------------------------------------

    struct SecTCandRep : public HotTubItem {
    };

    //----------------------------------------------------------------------------

    struct PrimTCandRep {
      CCandRep *mp_ccrep;
      int m_origin_tcand_index; // TrackCand index in CombCandidate
      float m_s_to_boundary; // Or some such thing ... to be seen.

      struct PQE { // Priority-Queue Entry
        float score;
        unsigned int hit_orig_index;
        unsigned int hit_index;
        int layer;
        // float dalpha
        // Hit &
        // ModuleInfo &

        // state on new hit, inv_pt, inv_k, theta on hit on previous layer / last hit
        mini_propagators::InitialState mixed_state;

        bool operator<(const PQE& o) const { return score < o.score; }
      };
      // Need to sub-class it to be able to call reserve on the vec
      std::priority_queue<PQE, std::vector<PQE>> m_pqueue;
      int m_pqueue_size = 0;

      std::vector<PQE> m_layer_hits;
      std::vector<PQE> m_layer_sec_hits;

      PrimTCandRep(CCandRep *ccr, int orig_idx) {
        mp_ccrep = ccr;
        m_origin_tcand_index = orig_idx;
        m_s_to_boundary = 0.0f;
      }

      TrackCand& tcand();

      // these go to SecTCandRep:
      // score, params, new hits[4];
      // index in the CombCand -- thing we will update as needed

      // Hack for best hit
      TrackState bState;
      HitOnTrack bHot;
      float bChi2 = 999.999f;
    };

    //----------------------------------------------------------------------------

    struct CCandRep : public HotTubConsumer<SecTCandRep> {
      CombCandidate &m_ccand;

      std::vector<PrimTCandRep> m_pTcs; // for now, could be in another hot-tub

    #if defined(MKFIT_STANDALONE)
      // Tuning & Debugging. Managed in MkFinderV2p2 processing.
      int m_seed_mc_label = -1;
      int m_mc_layer_sequence = -1; // counts layers WITH mc hits
      int m_n_mc_hits_in_layer = -1;
      // int m_n_mc_hits_in_layer_sec = -1;
    #endif

      CCandRep(HotTub<SecTCandRep> &htub, CombCandidate& ccand) :
        HotTubConsumer<SecTCandRep>(htub),
        m_ccand(ccand)
      {
         // QQQQ reserve also in begin_next_Ccrep_in_layer()
         // QQQQ clear in end_layer() -- might want to reuse the objects more
        m_pTcs.reserve(ccand.capacity());
      }
    };

    inline TrackCand& PrimTCandRep::tcand() { return mp_ccrep->m_ccand[m_origin_tcand_index]; }

    //----------------------------------------------------------------------------
    // BaseArgs

    struct BaseArgs {

      const PropagationConfig *prop_config = nullptr;

      mini_propagators::InitialStatePlex tsXyz;
      MPlexLS tsErr { 0.0f }; // input (on prev hit) and output (on current hit) [ts - track-state]
      MPlexLV tsPar { 0.0f }; // ""
      MPlexQI tsChg { 0 };    // "" Kalman update can flip it through curvature flip

      MPlexQF sPerp { 0.0f }; // path-length in transverse plane, calculated from alpha. p2plane really needs 3D path.

      MPlexLS propErr { 0.0f }; // intermediate: propagated from tsErr and used as input to Kalman
      MPlexLV propPar { 0.0f }; // input: pre-propagated as part of hit pre-selection

      MPlexQI outFailFlag { 0 }; // dummy, can be detected in pre-propagation, no other errors detected / reported

      int N_filled = 0;
    };

    //----------------------------------------------------------------------------
    // PropErrsArgs

    struct PropErrsArgs : public BaseArgs {

      void reset() { N_filled = 0; }

      void item_begin() {}
      bool item_finished() { return ++N_filled == NN; }

      void load_state_err_chg(const TrackBase &tb) {
        // tsXyz initialized manually, already in plex form
        tsPar.copyIn(N_filled, tb.posArray()); // propToPlane needs initial parameters, too
        tsErr.copyIn(N_filled, tb.errArray());
        tsChg[N_filled] = tb.charge();
      }

      void compute_pars() {
        // Parameters are stored in StatePlex -- so we can vectorize translation to pt, phi, theta.
        // Some stuff could be passed over as it won't change before update: pt, theta, k_inv
        // They are passed in output-parameters as propagation also needs input pars.
        propPar.aij(0, 0) = tsXyz.x;
        propPar.aij(1, 0) = tsXyz.y;
        propPar.aij(2, 0) = tsXyz.z;
        propPar.aij(3, 0) = tsXyz.inv_pt;
        propPar.aij(4, 0) = Matriplex::fast_atan2(tsXyz.py, tsXyz.px);
        propPar.aij(5, 0) = tsXyz.theta;

        sPerp = tsXyz.dalpha / ( tsXyz.inv_pt * tsXyz.inv_k);
      }

      void do_propagation_stuff();
    };

    //----------------------------------------------------------------------------
    // KalmanOpArgs

    struct KalmanOpArgs : public BaseArgs {

      PrimTCandRep *ptcp[NN];
      HitOnTrack    hot[NN];

      MPlexHS msErr { 0.0f }; // input measurement / hit [ms - measurement state]
      MPlexHV msPar { 0.0f }; // ""
      MPlexHV plNrm { 0.0f }; // input detector plane [pl - plane]
      MPlexHV plDir { 0.0f }; // ""
      MPlexHV plPnt { 0.0f }; // ""

      MPlexQF tsChi2 { 0.0f };   // output

      void reset() { N_filled = 0; }

      // There will be some more of this state, also secondary or who knows what.
      void item_begin(PrimTCandRep *ptc, HitOnTrack ht) { ptcp[N_filled] = ptc; hot[N_filled] = ht; }
      bool item_finished() { return ++N_filled == NN; }

      void load_state_err_chg(const mini_propagators::InitialState &state_on_hit, const TrackBase &tb) {
        tsXyz.copyIn(N_filled, state_on_hit);
        tsPar.copyIn(N_filled, tb.posArray()); // propToPlane needs initial parameters, too
        tsErr.copyIn(N_filled, tb.errArray());
        tsChg[N_filled] = tb.charge();
      }

      void load_hit_module(const Hit &hit, const ModuleInfo & mi) {
        msErr.copyIn(N_filled, hit.errArray());
        msPar.copyIn(N_filled, hit.posArray());
        plNrm.copyIn(N_filled, mi.zdir.Array());
        plDir.copyIn(N_filled, mi.xdir.Array());
        plPnt.copyIn(N_filled, mi.pos.Array());
      }

      void compute_pars() {
        // Parameters are stored in StatePlex -- so we can vectorize translation to pt, phi, theta.
        // Some stuff could be passed over as it won't change before update: pt, theta, k_inv
        // They are passed in output-parameters as propagation also needs input pars.
        propPar.aij(0, 0) = tsXyz.x;
        propPar.aij(1, 0) = tsXyz.y;
        propPar.aij(2, 0) = tsXyz.z;
        propPar.aij(3, 0) = tsXyz.inv_pt;
        propPar.aij(4, 0) = Matriplex::fast_atan2(tsXyz.py, tsXyz.px);
        propPar.aij(5, 0) = tsXyz.theta;

        sPerp = tsXyz.dalpha / ( tsXyz.inv_pt * tsXyz.inv_k);
      }

      void do_kalman_stuff();
    };

}

#endif
