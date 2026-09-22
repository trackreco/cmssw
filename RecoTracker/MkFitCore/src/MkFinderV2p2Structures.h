#ifndef RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h
#define RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include "MiniPropagators.h"

#include <queue>

namespace mkfit {

#ifdef MKFIT_TRACE
    class Event;
#endif

    struct CCandRep;

    //----------------------------------------------------------------------------

    // Intermediate storage of post-kalman-update with enough state to properly
    // update the TrackCand. Will potentially have multiple hits.
    // NOT YET USED. Storage is decided (see MkFinderV2p2Structures.cc): one
    // growable std::vector<SecTCandRep> per MkFinderV2p2, nodes reached by index,
    // a single int parent field, bulk rewind at end_layer(). The parent link is
    // still missing here because the expansion that needs it is not built.
    //
    // A SecTCandRepPQ wrapper (a std::priority_queue<SecTCandRep>) used to sit
    // next to this and has been removed: it was never referenced anywhere, and it
    // contradicts the arena decision -- a pqueue copies its values, which aliases
    // node ownership the moment nodes carry parent links. SecTCandRep::operator<
    // went with it, as it existed only to order that heap.
    struct SecTCandRep {
      TrackCand *m_tcand; // or, PrimTCandRep? Or either? To accomodate missed/glazed layers?
      TrackState m_state;
      HitOnTrack m_hot;
      float m_chi2;
      float m_score;
#ifdef MKFIT_TRACE
      int m_tr_hitmatch_id = -1;
#endif
    };

    //----------------------------------------------------------------------------

    struct PrimTCandRep {
      CCandRep *mp_ccrep;
      int m_origin_tcand_index; // TrackCand index in CombCandidate

      struct PQE { // Priority-Queue Entry
        float score;
        unsigned int hit_orig_index;
        unsigned int hit_index;
        int layer;

#ifdef MKFIT_TRACE
        int tr_hitmatch_id;
#endif
        // float dalpha
        // Hit &
        // ModuleInfo &

        // state on new hit, inv_pt, inv_k, theta on hit on previous layer / last hit
        mini_propagators::InitialState mixed_state;

        // We want the worst / highest score (dphi) at the top -- so we can replace it.
        bool operator<(const PQE& o) const { return score < o.score; }
      };
      // Need to sub-class it to be able to call reserve on the vec
      std::priority_queue<PQE, std::vector<PQE>> m_pqueue;
      int m_pqueue_size = 0;

      // m_layer_sec_hits is declared and never filled: the second sub-layer is
      // not processed at all today (OT_as_single_entry = false, so the plan emits
      // singles and has_second_layer() is always false).
      //
      // WHY TWO VECTORS AND NOT ONE (maintainer, 2026-09-21). An earlier note
      // here argued these should merge into a single dalpha-ordered list, on the
      // grounds that the sub-layers of a pair are nested and radially
      // INTERLEAVED: L4 spans r(22.14, 28.73) and L5 r(22.39, 28.54), offset
      // 2.5 mm out of a 6.5 cm shell because TBPS is tilted, and which member
      // sits at larger r flips module by module -- so a track's L5 hit can sit at
      // SHORTER path length than its L4 hit and the two cannot be processed in
      // radial sequence.
      //
      // That observation stands; the conclusion drawn from it does not. The two
      // sensors of a PS module are BONDED -- one stack, not two independent
      // layers -- and they are not equivalent measurements: P (macro-pixel,
      // 1.5 mm) measures q about 16x better than S (strip, 24 mm). So the useful
      // asymmetry is not path-length order at all:
      //
      //   - PRE-SELECT THE PAIR ON THE P HIT. It is the better q measurement, and
      //     the mkFit propagator runs in both directions, so reaching the S hit
      //     from a P-anchored state costs nothing and needs no ordering decision.
      //     Ordering by path length would instead force a decision that the
      //     geometry does not actually determine, and overlap hits make the
      //     ordering still less determinate.
      //   - P WANTS NARROWER q BINS than S, which is a per-sub-layer binnor
      //     property. Merging into one list throws that away; keeping them apart
      //     is what makes the finer binning buy less pre-selection work.
      //
      // Hence the pair stays distinguishable even after the layer merge ("fat
      // layers"), via a STEREO BIT ON Hit rather than via two LayerOfHits. That
      // is the piece to build before this vector can go away.
      //
      // And the OT ENDCAP is the hard case, not the barrel: every TEDD disc is PS
      // on the inside (lower r, out to ~650 mm) and 2S outside, so a single disc
      // carries both module types at different radii and the P/S split is not a
      // layer property there at all.
      std::vector<PQE> m_layer_hits;
      std::vector<PQE> m_layer_sec_hits;

      PrimTCandRep(CCandRep *ccr, int orig_idx) {
        mp_ccrep = ccr;
        m_origin_tcand_index = orig_idx;
      }

      CombCandidate& ccand();
      TrackCand& tcand();

      // these go to SecTCandRep:
      // score, params, chi2, new hits[4 or more for doulbe layers];
      // index in the CombCand -- thing we will update as needed

      // Hack for best hit
      TrackState bState;
      HitOnTrack bHot;
      float bChi2 = 999.999f;
#ifdef MKFIT_TRACE
      int b_tr_hitmatch_id = -1;
      // Truth-forcing diagnostic (g_v2p2_force_mc). The best-hit choice is made
      // on bKey, which is normally just bChi2; with forcing on, an MC-matched
      // hit gets a key below any non-matched one so it always wins its layer,
      // while bChi2 keeps the REAL chi2 so nothing downstream is falsified.
      // bIsMc then lets the acceptance cut be bypassed for it. This exists to
      // separate "the true hit was never available" from "the ranking or the
      // pruning threw it away" -- it is an oracle, never a production path.
      float bKey = 999.999f;
      bool  bIsMc = false;
#endif
    }; // end struct PrimTCandRep

    //----------------------------------------------------------------------------

    struct CCandRep {
      CombCandidate &m_ccand;

      std::vector<PrimTCandRep> m_primTCs; // for now, could live in the shared arena

      // We could also keep track of the TrackCands that do not enter layer
      // processing at all -- either already stopped or missing this layer.
      // Store indices into m_ccand, the way m_primTCs does, rather than
      // pointers: indices stay valid regardless of how the CombCandidate grows.
      // Only worth doing together with SecTCandRep and the selection / merging
      // step -- the best-hit path has no use for it.
      // std::vector<int> m_otherTCs;

      // int m_num_primTCs_to_kalman = 0; // to be improved

    #if defined(MKFIT_STANDALONE)
      // Tuning & Debugging. Managed in MkFinderV2p2 processing.
      int m_seed_mc_label = -1;
      int m_mc_layer_sequence = -1; // counts layers WITH mc hits
      int m_n_mc_hits_in_layer = -1;
      // int m_n_mc_hits_in_layer_sec = -1;
    #endif

      CCandRep(CombCandidate& ccand) :
        m_ccand(ccand)
      {
         // QQQQ reserve also in begin_next_Ccrep_in_layer()
         // QQQQ clear in end_layer() -- might want to reuse the objects more
        m_primTCs.reserve(ccand.capacity());
      }
    }; // end struct CCandRep

    inline CombCandidate& PrimTCandRep::ccand() { return mp_ccrep->m_ccand; }
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

#ifdef MKFIT_TRACE
      int tr_hitmatch_ids[NN];
      void set_tr_hitmatch_id(int id) { tr_hitmatch_ids[N_filled] = id; }
      const Event *mp_event = nullptr;
#endif

      void reset() { N_filled = 0; }

      // There will be some more of this state, also secondary or who knows what.
      void item_begin(PrimTCandRep *ptc, HitOnTrack ht) { ptcp[N_filled] = ptc; hot[N_filled] = ht; }
      bool item_finished() { return ++N_filled == NN; }

      void load_state_err_chg(const mini_propagators::InitialState &params_on_hit, const TrackState &prev_state) {
        tsXyz.copyIn(N_filled, params_on_hit);
        tsPar.copyIn(N_filled, prev_state.parArray()); // propToPlane needs initial parameters, too
        tsErr.copyIn(N_filled, prev_state.errArray());
        tsChg[N_filled] = prev_state.charge;
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
