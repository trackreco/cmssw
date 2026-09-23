#ifndef RecoTracker_MkFitCore_src_MkFinderV2p2_h
#define RecoTracker_MkFitCore_src_MkFinderV2p2_h

#include "RecoTracker/MkFitCore/interface/SteeringParams.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"
#include "RecoTracker/MkFitCore/interface/MkJob.h"

#include "MkRZLimits.h"
#include "MkFinderV2p2Structures.h"

#include "MkBins.h"

#include <atomic>
#include <functional>
#include <list>
#include <utility>
#include <vector>

namespace mkfit {

  // Diagnostic oracles, both default OFF and both meaningful only in a
  // MKFIT_TRACE build. See PrimTCandRep::bKey.
  //   g_v2p2_force_mc -- make the MC-matched hit win its layer and bypass the
  //                      chi2 < 30 acceptance cut.
  extern bool g_v2p2_force_mc;
  // Pre-selection dq allowance, MkFinderV2p2.cc. Runtime so it can be scanned.
  extern float g_v2p2_extra_dq;
  // Reference the pre-selection q error to the HIT'S OWN MODULE PLANE. Exact for
  // tilted and flat layers alike -- see MkFinderV2p2.cc.
  extern bool  g_v2p2_surface_q;
  // Per-sub-layer hit reduction cap; see MkFinderV2p2.cc.
  extern int   g_v2p2_max_presel_hits;
  // Most hits one in-layer path may take; see MkFinderV2p2.cc.
  extern int   g_v2p2_max_sec_depth;


  // Did the per-layer policy actually fire? quality-val cannot answer that: a cut
  // that never fires and a cut that fires on candidates which were doomed anyway
  // both show up as "no change". Atomic because the search runs one finder per
  // thread; summed over threads and events, printed and reset by mkFit.cc
  // alongside the quality-val summary.
  struct V2p2PolicyCounters {
    std::atomic<long> n_quadrant_skip{0};  // pull-in: coarse rz check said the layer is behind us
    std::atomic<long> n_stop_minpt{0};     // pull-in: pT below minPtCut
    std::atomic<long> n_stop_looper{0};    // pull-in: past pi/2 - 0.2 to the radial direction
    std::atomic<long> n_wsr_inside{0};     // crossing wholly in sensitive q
    std::atomic<long> n_wsr_edge{0};       // within dq of a boundary, or turning inside the layer
    std::atomic<long> n_wsr_outside{0};    // does not reach the layer
    std::atomic<long> n_wsr_in_gap{0};     // ... because of the endcap r hole
    std::atomic<long> n_layer_skipped{0};  // WSR_Outside and therefore no HoT added
    std::atomic<long> n_hole{0};           // kHitMissIdx -- a real hole, counted against the limits
    std::atomic<long> n_hot_edge{0};       // kHitEdgeIdx -- not counted
    std::atomic<long> n_hot_gap{0};        // kHitInGapIdx -- not counted
    std::atomic<long> n_stop_holes{0};     // kHitStopIdx -- out of hole budget
    std::atomic<long> n_ccand_retired{0};  // every TrackCand under it stopped
    std::atomic<long> n_sec_nodes{0};      // in-layer tree nodes built
    std::atomic<long> n_sec_deep{0};       // ... of those, at depth 2 or more
    std::atomic<long> n_path_taken{0};     // layers where a path was registered
    std::atomic<long> n_extra_hits{0};     // hits taken beyond the first in one layer
    std::atomic<long> n_sel_entries{0};    // competitors entering the end-of-layer selection
    std::atomic<long> n_sel_kept{0};       // ... and surviving it
    std::atomic<long> n_selections{0};     // end-of-layer selections run
    std::atomic<long> n_same_module{0};    // extra hit from the SAME module -- not an overlap
    std::atomic<long> n_diff_module{0};    // extra hit from another module -- a genuine overlap
    std::atomic<long> n_same_module_vetoed{0};  // extensions refused for sharing a module
    std::atomic<long> n_hole_slot_reserved{0};  // beam slots given to an outranked decliner
    std::atomic<long> n_best_short_offered{0};  // stopped candidates removed from the beam
    std::atomic<long> n_best_short_taken{0};    // ... and that became the seed's best short
    // Matriplex lane occupancy of the Kalman batches: is the expansion still
    // vectorised, or is it running ragged tails? mean = lanes / (calls * NN).
    std::atomic<long> n_kalman_calls{0};
    std::atomic<long> n_kalman_lanes{0};
    std::atomic<long> n_kalman_calls_d0{0};   // depth 0 only
    std::atomic<long> n_kalman_lanes_d0{0};

    void reset();
    void print(const char *tag) const;
  };
  extern V2p2PolicyCounters g_v2p2_policy_counters;

  class FindingFoos;
  class IterationParams;
  class IterationLayerConfig;
  class SteeringParams;
  struct LayerControl;
  class LayerInfo;
  class Event;

  class MkJob;

  class MkFinderV2p2 {
    friend class MkBuilder;

    //-------------------------------------------------------------------------
    class BatchManager {
    //-------------------------------------------------------------------------
      friend class MkFinderV2p2;

      EventOfCombCandidates *mp_eoccs = nullptr;
      int m_begin;
      int m_end;

      int m_n_dormant;
      int m_n_finding;
      // int m_n_to_finalize;
      int m_n_finished;

      // Not needed, have the list of CCandReps
      // // Cursors into EOCCS for activation
      // int m_Cc_pos;  // current CombCand index
      // int m_pTc_pos; // current primary TrackCand index

      // something for the derived ones --- teritary, to handle in-layer combinatorials
      // or maybe we'll need a sub-manager for those? Or will the finder do that
    public:
      CombCandidate& ccand(int i) const { return (*mp_eoccs)[i]; }
      CombCandidate* ccand_ptr(int i) const { return &(*mp_eoccs)[i]; }

      void setup(EventOfCombCandidates &eoccs, int seed_begin, int seed_end) {
        mp_eoccs = &eoccs;
        m_begin = seed_begin;
        m_end = seed_end;
        m_n_dormant = 0;
        m_n_finding = 0;
        m_n_finished = 0;
        reset_for_new_layer();
      }
      void release() {
        mp_eoccs = nullptr;
      }

      void reset_for_new_layer() {
        // Prepare for next layer / extraction.
        // We have list of active CCandReps now
        // m_Cc_pos = m_begin;
        // m_pTc_pos = 0;
      }

      int n_total() const { return m_end - m_begin; }
      int n_dormant() const { return m_n_dormant; }
      int n_finding() const { return m_n_finding; }
      int n_finished() const { return m_n_finished; }
      bool are_all_ccands_finished() const { return n_finished() == n_total(); }
      bool has_dormant_ccands() const { return n_dormant() > 0; }

      // Iteration over all CombCandidates for top-level administrative tasks.
      class iterator {
        CombCandidate *m_ccand;
      public:
        iterator(CombCandidate *bm) : m_ccand(bm) {}
        CombCandidate& operator*() { return *m_ccand; }
        iterator& operator++() {
          ++m_ccand;
          return *this;
        }
        bool operator!=(const iterator &i) const { return m_ccand != i.m_ccand; }
      };

      iterator begin() const { return iterator(ccand_ptr(m_begin)); }
      iterator end() const { return iterator(ccand_ptr(m_end - 1) + 1); }
    //-------------------------------------------------------------------------
    }; // end class BatchManager
    //-------------------------------------------------------------------------

  public:
    MkFinderV2p2() = default;

    //----------------------------------------------------------------------------

    void setup(const MkJob *job, EventOfCombCandidates &eoccs, int seed_begin, int seed_end,
               SteeringParams::iterator &sp_it, const Event *ev);
              //  int region, // sp_it->region()
              //  const PropagationConfig &pc, // trk_info.prop_config() and m_job->m_trk_info
              //  const IterationConfig &ic, // m_job->m_iter_config
              //  const IterationParams &ip, // m_job->params_cur();
              //  const IterationLayerConfig &ilc, // m_job->m_iter_config.m_layer_configs[curr_layer]
              //  const SteeringParams &sp, // m_job->steering_params(region)
              //  const std::vector<bool> *ihm, // m_job->get_mask_for_layer(curr_layer)
              //  bool infwd);               // m_job->m_in_fwd
    void release();

    int awaken_candidates();

    void begin_layer();

    bool any_Ccreps_to_begin() const { return m_active_ccreps_pos != m_active_ccreps.end(); }
    void begin_next_Ccrep_in_layer();

    bool enough_work_for_batch() const { return (int) m_cand_queue.size() >= NN; }
    bool any_work_for_batch() const { return ! m_cand_queue.empty(); }
    void process_layer_batch();

    void end_layer();

    void process_layer();

    // Mostly for debug printouts in MkBuilder steering code.
    const BatchManager& batch_mgr() const { return m_batch_mgr; }

    //----------------------------------------------------------------------------

  private:
    //----------------------------------------------------------------------------
    // Per-pass state of process_layer_batch().
    //
    // TWO batch widths are in play, and keeping them apart is most of what makes
    // the layer pass readable: LayerBatch is NN CANDIDATES wide, HitBatch is NN
    // (candidate, hit) PAIRS wide. Everything in LayerBatch is per candidate and
    // indexed by the same i; everything in HitBatch is per scanned hit and
    // indexed by h, with prim_idcs[h] naming the candidate it belongs to.
    //
    // This is also the carrier the earlier procedural split was missing. The
    // abandoned sketch at the bottom of MkFinderV2p2.cc names exactly these
    // phases and is annotated "can't quite work" -- without an explicit batch
    // object each phase needed a dozen arguments, so it stayed a monolith.
    struct LayerBatch {
      int N_proc = 0;
      PrimTCandRep *ptc[NN];             // the candidates in this pass
      MkBins B { 0 };                    // isp + the two bounding-surface crossings
      MkBinTrackCovExtract TCE;          // position block of the window covariance
      MkBinLimits BL_p, BL_s;            // binnor ranges, primary / secondary layer
      mini_propagators::Hermite3D H;     // cubic through sp1, sp2 -- the trajectory model
      // Local hit density per candidate, ln(hits / cm^2), for the likelihood
      // score. Counted over the bins actually walked, so it is independent of the
      // pre-selection cut and cannot be circular.
      int   n_scanned[NN] = {0};
      float log_rho[NN] = {0.0f};
#ifdef MKFIT_TRACE
      int tr_layersearch_ids[NN];
#endif
    };

    struct HitBatch {
      int fill_pos = 0;
      MPlexQI  prim_idcs;                // -> LayerBatch::ptc
      MPlexQUI hit_idcs;                 // index within the LayerOfHits
      MPlexQUI hit_orig_idcs;            // index in the event hit vector
      mini_propagators::InitialStatePlex is_plex;
#ifdef MKFIT_TRACE_PROP_COMPARE
      mini_propagators::StatePlex h_plex;  // PA_Line cross-check against the Hermite
#endif
    };

    void prop_to_layer_edges(LayerBatch &b);
    void determine_search_windows(LayerBatch &b);
    void determine_wsr(LayerBatch &b);
    void select_hits(LayerBatch &b);
    void preselect_hit_batch(LayerBatch &b, HitBatch &hb, const LayerOfHits &L, int N_proc_hits,
                             bool is_sec_layer);
    // Re-reference the pre-selection q error from a fixed path length onto the
    // hit's own module plane. Static: a pure function of the state, the module
    // normal and the covariance. See MkFinderV2p2.cc for the derivation.
    static float surface_referenced_dq(float dq_track_fallback,
                                       const MkBinTrackCovExtract &TCE, int pi,
                                       const mini_propagators::StatePlex &h3_state, int h,
                                       const MPlex3V &module_norm, bool is_barrel);
    // Candidate-stopping cuts applied at pull-in, where only the candidate's own
    // state is needed. Returns the reason, or SR_NotStopped to keep going.
    TrackCand::StopReason_e stop_cuts_at_pickup(const TrackCand &tc) const;
    // Which fake HoT a candidate that took no hit in this layer gets: the hole
    // limits, the WSR override, and the gap case, in V1's order.
    int fake_hit_index(const TrackCand &tc, const WSR_Result &wsr) const;

    void prepare_kalman_workload(LayerBatch &b);
    void kalman_update(LayerBatch &b);
    void process_kalman_results(LayerBatch &b);

    // The in-layer combinatorial search, replacing the two phases above when
    // Config::v2p2InLayerComb is on. expand_in_layer() grows the SecTCandRep
    // tree breadth-first by depth; materialise_in_layer() picks a path out of it
    // and registers it into the CombCandidate.
    void expand_in_layer(LayerBatch &b);
    void select_and_materialise(CCandRep &ccrep);
    void offer_best_short(CombCandidate &ccand, const TrackCand &tc) const;
    // The direction-, layer- and candidate-dependent part of a layer step, filled
    // once per path root and carried down the tree.
    void fill_step_geometry(LayerStepFeatures &f, const PrimTCandRep &ptc, float log_rho) const;
    // Turn the Kalman results accumulated in m_sec_out into arena nodes, keeping
    // those that pass the chi2 cut. Returns the arena range that was appended.
    std::pair<int, int> harvest_sec_nodes(const LayerBatch &b);

    //----------------------------------------------------------------------------
    // Job / batch-of-seeds control variables and globel references
    const MkJob *mp_job = nullptr;
    SteeringParams::iterator *mp_steeringparams_iter = nullptr;
    const Event *mp_event = nullptr;

    BatchManager m_batch_mgr;

    //----------------------------------------------------------------------------
    // Per-(di)layer state & control
    std::list<CCandRep> m_active_ccreps;
    // Current CombCand and next TrackCand to go through layer initialization, i.e.,
    // propagation to layer limits, Binnor creation and extraction of bin-indices, and
    // pre-selection of hits.
    std::list<CCandRep>::iterator m_active_ccreps_pos; // Current CombCand to be processed or is in processing.

    // Pre-selection queue -- list of pTcs to do initial prop + Binnor + hit extraction for.
    // Elements are slots in the pTC hot-tub.
    std::list<PrimTCandRep*> m_cand_queue;

    // Per-(di)layer geometrical state
    MkRZLimits m_rz_limits;

    // The in-layer combinatorial tree. ONE arena for the whole finder, reached by
    // index, rewound (not deallocated) once the layer batch has materialised, so
    // after a few layers it is at high water and stops allocating.
    std::vector<SecTCandRep> m_sec_arena;
    // Kalman outcomes of the depth currently being expanded, drained into the
    // arena by harvest_sec_nodes().
    std::vector<KalmanOpArgs::ItemOut> m_sec_out;

    // One competitor in the end-of-layer selection. node_idx >= 0 is an in-layer
    // path; otherwise the candidate declined the layer, and add_fake says whether
    // that costs it a HoT (a hole) or nothing at all (the layer was out of
    // reach, or it was already stopped).
    struct SelEntry {
      int   tcand_idx;
      int   node_idx;
      int   fake_hit;
      bool  add_fake;
      float score;
    };
    std::vector<SelEntry> m_sel;
    std::vector<TrackCand> m_new_cands;
  };

} // end namespace mkfit

#endif
