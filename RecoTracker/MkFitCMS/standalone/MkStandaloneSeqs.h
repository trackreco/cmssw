#ifndef RecoTracker_MkFitCMS_standalone_MkStandaloneSeqs_h
#define RecoTracker_MkFitCMS_standalone_MkStandaloneSeqs_h

#include <vector>
#include <map>

namespace mkfit {

  class EventOfHits;
  class Track;
  class TrackExtra;
  typedef std::vector<Track> TrackVec;
  typedef std::vector<TrackExtra> TrackExtraVec;

  class Event;

  namespace StdSeq {

    void loadHitsAndBeamSpot(Event &ev, EventOfHits &eoh);

    void handle_duplicates(Event *event);

    void dump_simtracks(Event *event);
    void track_print(Event *event, const Track &t, const char *pref);

    // Validation quality & ROOT
    //--------------------------

    struct Quality {
      int m_cnt = 0, m_cnt1 = 0, m_cnt2 = 0, m_cnt_8 = 0, m_cnt1_8 = 0, m_cnt2_8 = 0, m_cnt_nomc = 0;

      void quality_val(Event *event);
      void quality_reset();
      void quality_process(Event *event, Track &tkcand, const int itrack, std::map<int, int> &cmsswLabelToPos);
      void quality_print();
    };

    void root_val_dumb_cmssw(Event *event);
    void root_val(Event *event);

    void prep_recotracks(Event *event);
    void prep_simtracks(Event *event);
    void prep_cmsswtracks(Event *event);
    void prep_reftracks(Event *event, TrackVec &tracks, TrackExtraVec &extras, const bool realigntracks);
    // sort hits by layer, init track extras, align track labels if true
    void prep_tracks(Event *event, TrackVec &tracks, TrackExtraVec &extras, const bool realigntracks);
    void score_tracks(TrackVec &tracks);  // if track score is not already assigned

    // Carryover from MkBuilder.h
    //----------------------------

    // void create_seeds_from_sim_tracks();
    // void find_seeds();
    // void fit_seeds();
    // void PrepareSeeds();

    // This one is here mostly to keep code for printing overlap hit truth - DUMP_OVERLAP_RTTS.
    // quality_store_tracks(const EventOfCombCandidates &eoccs, TrackVec & tracks);

    // Carryover from MkFitter.h
    //---------------------------

    // void FitTracks(const int N_proc, const Event * ev, const PropagationFlags pflags);
    // void FitTracksSteered(const bool is_barrel[], const int N_proc, const Event * ev, const PropagationFlags pflags);
    // void CollectFitValidation(const int hi, const int N_proc, const Event * ev) const;

  }  // namespace StdSeq

}  // namespace mkfit

#endif
