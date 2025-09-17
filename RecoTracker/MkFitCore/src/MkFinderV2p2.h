#ifndef RecoTracker_MkFitCore_src_MkFinderV2p2_h
#define RecoTracker_MkFitCore_src_MkFinderV2p2_h

#include "RecoTracker/MkFitCore/interface/SteeringParams.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/TrackStructures.h"
#include "RecoTracker/MkFitCore/interface/MkJob.h"

#include "MkBase.h"
#include "MkFinderV2p2Structures.h"

#include <functional>
#include <list>

namespace mkfit {

  class FindingFoos;
  class IterationParams;
  class IterationLayerConfig;
  class SteeringParams;
  struct LayerControl;
  class Event;

  class MkJob;

  class MkFinderV2p2 : public MkBase { /// XXXXXX do we need mkbase ???
    friend class MkBuilder;


    // Unrolled indices of CombCands and TrackCands.
    struct BatchInfo {

    };

    class BatchManager {
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
    }; // end class BatchManager

  public:
    MkFinderV2p2() : m_hot_tub(2 * NN) // XX to be checked, probably some more
    {}

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
    // int unroll_candidates();

    void begin_layer();

    bool any_Ccreps_to_begin() const { return m_active_ccreps_pos != m_active_ccreps.end(); }
    void begin_next_Ccrep_in_layer();

    bool enough_work_for_pre_select() const { return (int) m_pre_select_queue.size() >= NN; }
    bool any_work_for_pre_select() const { return ! m_pre_select_queue.empty(); }
    // ??? void pre_select_hits(int layer, MkBinLimits &BL);
    void process_pre_select();

    void end_layer();

    void process_layer();

    // Mostly for debug printouts in MkBuilder steering code.
    const BatchManager& batch_mgr() const { return m_batch_mgr; }

    //----------------------------------------------------------------------------


    //----------------------------------------------------------------------------

  private:
    //----------------------------------------------------------------------------
    // Candidate search related variables -- first for alignment along with MkBase

    MPlexQF m_Chi2;

    // Hit errors / parameters for hit matching, update.
    MPlexHS m_msErr{0.0f};
    MPlexHV m_msPar{0.0f};

    CombCandidate *m_CombCand[NN];
    // const TrackCand *m_TrkCand[NN]; // hmmh, could get all data through this guy ... but scattered
    // storing it in now for bkfit debug printouts
    TrackCand *m_TrkCand[NN];

    //----------------------------------------------------------------------------
    // Copy in / out functions

    //----------------------------------------------------------------------------
    // Job / batch-of-seeds control variables and globel references
    const MkJob *mp_job = nullptr;
    SteeringParams::iterator *mp_steeringparams_iter = nullptr;
    const Event *mp_event = nullptr;

    BatchManager m_batch_mgr;

    //----------------------------------------------------------------------------
    // Per-(di)layer state & control
    HotTub<SecTCandRep> m_hot_tub;

    std::list<CCandRep> m_active_ccreps;
    // Current CombCand and next TrackCand to go through layer initialization, i.e.,
    // propagation to layer limits, Binnor creation and extraction of bin-indices, and
    // pre-selection of hits.
    std::list<CCandRep>::iterator m_active_ccreps_pos; // Current CombCand to be processed or is in processing.
    int m_active_ccreps_tC_pos;                        // Index of next TrackCand to be processed.

    int m_n_Ccs_to_finalize; // or something, with indices or referenes or iterators.

    // Pre-selection queue -- list of pTcs to do initial prop + Binnor + hit extraction for.
    // Elements are slots in the pTC hot-tub.
    std::list<PrimTCandRep*> m_pre_select_queue;

    //
  };

} // end namespace mkfit

#endif
