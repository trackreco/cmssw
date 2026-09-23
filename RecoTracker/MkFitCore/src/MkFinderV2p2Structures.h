#ifndef RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h
#define RecoTracker_MkFitCore_src_MkFinderV2p2Structures_h

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"

#include "MiniPropagators.h"
#include "V2p2Score.h"

#include <queue>
#include <vector>

namespace mkfit {

#ifdef MKFIT_TRACE
    class Event;
#endif

    struct CCandRep;
    struct PrimTCandRep;

    //----------------------------------------------------------------------------

    // One node of the in-layer combinatorial tree: a candidate that has taken a
    // particular SEQUENCE of hits within this layer, with the Kalman updates
    // applied along the way.
    //
    // It is a PERSISTENT TREE with parent links only -- walked child to parent,
    // appended to within a layer, discarded whole at the end of it. There is no
    // way forward because nothing needs one: what the node exists for is the way
    // BACK, so a surviving leaf can register its hits and chi2s into the
    // CombCandidate. Storage is one plain std::vector<SecTCandRep> per
    // MkFinderV2p2, nodes reached by INDEX; the index is the handle, so the
    // vector may grow freely, and end-of-layer is a size rewind that keeps
    // capacity. There is deliberately no free list: a node with live children
    // cannot be released, and recycling its slot would silently overwrite the
    // parent state of an undecided path.
    //
    // ONE arena per finder, not one per CCandRep: a Matriplex batch draws lanes
    // from several PrimTCandReps and several CCandReps, so a single base must
    // reach all of them. (MkFinder carries const HoTNode *m_HoTNodeArr[NN], NN
    // separate bases chased scalar-ly, precisely because m_hots is
    // per-CombCandidate.)
    //
    // Note the lifetime asymmetry against CombCandidate::m_hots, which is why
    // this is a separate arena and not an extension of HoTNode: a HoTNode is
    // 12 B and lives for the whole event, a SecTCandRep is ~140 B -- a
    // TrackState alone is 112 -- and lives for one layer.
    struct SecTCandRep {
      PrimTCandRep *m_ptc;      // the candidate this path extends
      int   m_parent_idx;       // arena index of the predecessor; -1 = the PrimTCandRep itself
      int   m_hit_pos;          // position in m_ptc->m_layer_hits of THIS hit.
                                // The traversal cursor is m_hit_pos + 1 and is therefore
                                // not stored: it is search state, derivable from the hit.
      HitOnTrack m_hot;
      TrackState m_state;       // AFTER the update at m_hot
      float m_chi2;             // of this hit alone
      // The whole path through this layer, accumulated parent -> child. This is
      // what the score is a function of, and it is one step's worth however many
      // hits the path took: a two-hit path is ONE layer step, not two.
      LayerStepFeatures m_feat;
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
      // WHAT REPLACES THE TWO VECTORS (decided 2026-09-21, maintainer): ONE
      // merged list per rep, ordered by STEP DISTANCE along the trajectory and
      // sorted once on the initial Hermite path length. `dalpha` is a sufficient
      // and cheaper key than path_length(): both sub-layers are reached from the
      // same origin state with the same k, so dalpha is monotone in arc length.
      // The reduction stays per sub-layer -- one bounded pqueue each, so
      // NEW_MAX_HIT remains a per-sensor budget and a shower in one sensor cannot
      // starve the other -- and only the drained lists merge.
      //
      // Why step distance and not "the P hit first". An earlier draft here
      // argued precision-first: P (macro-pixel, 1.5 mm) measures q about 16x
      // better than S (strip, 24 mm), so anchor the pair on P and the S window
      // shrinks. RETRACTED, and the arithmetic is the reason -- the 16x is what
      // the P UPDATE gains, not what the S WINDOW gains. The S window's q term is
      // EXTRA_DQ * DDQ_PRESEL_FAC * q_half_length = 3 * 1.2 * 0.80 = 2.89 cm in
      // TBPS, against a track term of order 0.1-0.3 cm, so shrinking the track
      // term 16x moves that window by a few percent. And measurement killed it
      // outright: the FINE sensor is the one more often MISSING (B-only 4.1 % of
      // TBPS crossings, 5.2 % in the TEDD PS region, about twice A-only), so a
      // P-anchored pre-selection either loses those crossings or needs a branch,
      // and branches are poison in the vectorized kernel. Step ordering covers
      // A-only, B-only, both and overlaps with no branch at all.
      //
      // Per-hit precision is still wanted, but as a WEIGHT in the reduction key,
      // never as a stage -- and it needs no new array and no stereo bit on Hit:
      // LayerOfHits::HitInfo::q_half_length already is it (TBPS 0.042 vs 0.803,
      // TEDD 0.074 vs 1.178, TB2S 2.5125 for both, where a stereo bit would be
      // meaningless). The reduction key is ddphi alone today, which is what lets
      // a 0.042 cm P hit and a 2.5 cm strip hit compete on equal terms.
      //
      // The observation that motivated the retracted draft still stands and is
      // why the ordering has to be MEASURED rather than assumed: the sub-layers
      // of a pair are nested and radially INTERLEAVED. L4 spans r(22.14, 28.73)
      // and L5 r(22.39, 28.54), offset 2.5 mm out of a 6.5 cm shell because TBPS
      // is tilted, and which member sits at larger r flips module by module -- so
      // a track's L5 hit can sit at SHORTER path length than its L4 hit.
      //
      // And the OT ENDCAP is the hard case, not the barrel: every TEDD disc is PS
      // on the inside (lower r, out to ~650 mm) and 2S outside, so a single disc
      // carries both module types at different radii and "how precise is this
      // hit" is a per-hit question there, not a per-layer one. q_half_length
      // answers it per hit; a per-layer flag could not.
      std::vector<PQE> m_layer_hits;
      std::vector<PQE> m_layer_sec_hits;

      // Did this candidate produce any in-layer path? Set by expand_in_layer();
      // read at end of layer, where a candidate with none is the one that
      // DECLINED the layer and competes as a hole.
      bool m_has_sec_nodes = false;

      PrimTCandRep(CCandRep *ccr, int orig_idx) {
        mp_ccrep = ccr;
        m_origin_tcand_index = orig_idx;
      }

      // Within-sensitive-region verdict for THIS candidate on THIS layer, set by
      // MkFinderV2p2::determine_wsr() once sp1/sp2 and dq_track are known. It is
      // per candidate, not per layer: two candidates crossing the same layer can
      // differ, one passing through the middle and one clipping the z end.
      //
      //   WSR_Inside  -- the whole crossing is comfortably inside sensitive q,
      //                  so a missing hit is a genuine hole;
      //   WSR_Edge    -- the crossing is within dq of a boundary, or the track
      //                  turns around inside the layer, so a missing hit is not
      //                  evidence of anything;
      //   WSR_Outside -- the track does not reach the layer. The layer is then
      //                  SKIPPED: no hits scanned and no HoT of any kind added,
      //                  which is the point -- the layer plans are deliberately
      //                  inclusive (the transition plans list BPix + FPix + TOB +
      //                  TEC, i.e. the union over tracks), so 67.6 % of layer
      //                  searches are on layers the track never crosses and every
      //                  one of them used to record a hole.
      WSR_Result m_wsr;

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

      // Arena indices of every in-layer path belonging to this CombCandidate,
      // across all of its PrimTCandReps. The selection at end of layer is a
      // single flat sort over these plus one entry per TrackCand that declined
      // the layer or was already stopped -- that flat sort is the whole reason
      // the score is additive, and the reason these are collected per CCandRep
      // rather than per PrimTCandRep.
      //
      // It has to be per CCandRep and resolved at END OF LAYER, not per NN batch:
      // begin_next_Ccrep_in_layer() pushes all of a CombCandidate's TrackCands
      // into the pre-select queue at once, and the queue is drained NN at a time,
      // so one CombCandidate's alternatives can straddle a batch boundary.
      std::vector<int> m_sec_nodes;

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
      // Provenance of each lane, for the in-layer expansion: which arena node this
      // extension grew from (-1 = the PrimTCandRep's own incoming state) and which
      // entry of m_layer_hits it takes. Unused by the best-hit path.
      int           parent_idx[NN];
      int           hit_pos[NN];

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

      // One result per lane, appended when mp_out is set. The best-hit path needs
      // only the winner and keeps it in PrimTCandRep::b*; the expansion needs
      // every outcome, because a hit that loses on chi2 at depth 1 may still be
      // the right second hit of a two-hit path.
      struct ItemOut {
        PrimTCandRep *ptc;
        int   parent_idx;
        int   hit_pos;
        HitOnTrack hot;
        unsigned int hit_in_layer;   // index WITHIN the LayerOfHits, for hit_q_half_length()
        float chi2;
        TrackState state;
#ifdef MKFIT_TRACE
        int   tr_hitmatch_id;
#endif
      };
      std::vector<ItemOut> *mp_out = nullptr;

      // Let propagate-to-plane SOLVE for the crossing (sPerp == nullptr) instead
      // of being handed one. Depth 0 of the expansion starts from the
      // PrimTCandRep's own state, for which the Hermite has already produced the
      // crossing, so it uses the cheap path. Past the first update the Hermite's
      // two endpoints no longer describe the trajectory, so the solve is used --
      // it is the cheaper thing to WRITE; the one-point Hermite
      // (Hermite3D::calculate_coeffs(sp, inv_k, dalpha), which exists and is
      // still uncalled) is the cheaper thing to RUN and is the intended
      // replacement.
      bool m_solve_plane = false;

      void reset() { N_filled = 0; }

      // There will be some more of this state, also secondary or who knows what.
      unsigned int hit_in_layer[NN];

      void item_begin(PrimTCandRep *ptc, HitOnTrack ht, int par_idx = -1, int hpos = -1,
                      unsigned int hil = 0) {
        ptcp[N_filled] = ptc; hot[N_filled] = ht;
        parent_idx[N_filled] = par_idx; hit_pos[N_filled] = hpos;
        hit_in_layer[N_filled] = hil;
      }
      bool item_finished() { return ++N_filled == NN; }

      // Start from a TrackState directly, for the solve-plane path: there is no
      // pre-solved crossing, so tsXyz is not filled and compute_pars() must be
      // skipped -- propPar is a pure output there, where on the sPerp path it is
      // an input carrying the already-known intersection.
      void load_state_err_chg(const TrackState &st) {
        tsPar.copyIn(N_filled, st.parArray());
        tsErr.copyIn(N_filled, st.errArray());
        tsChg[N_filled] = st.charge;
      }

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
