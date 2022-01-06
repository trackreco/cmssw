#ifndef RecoTracker_MkFitCMS_standalone_MkStandaloneSeqs_h
#define RecoTracker_MkFitCMS_standalone_MkStandaloneSeqs_h

namespace mkfit {

  class EventOfHits;
  class Event;

  namespace StdSeq {

    void LoadHitsAndBeamSpot(Event &ev, EventOfHits &eoh);

    void handle_duplicates(Event *m_event);


    // Carryover from MkBuilder.h

    // void create_seeds_from_sim_tracks();
    // void find_seeds();
    // void fit_seeds();

    // --------

    /* CCCC
    void quality_val();
    void quality_reset();
    void quality_process(Track& tkcand, const int itrack, std::map<int,int> & cmsswLabelToPos);
    void quality_print();
    void track_print(const Track &t, const char* pref);

    void quality_store_tracks(TrackVec & tracks);

    void root_val_dumb_cmssw();
    void root_val();

    void prep_recotracks();
    void prep_simtracks();
    void prep_cmsswtracks();
    void prep_reftracks(TrackVec& tracks, TrackExtraVec& extras, const bool realigntracks);
    void prep_tracks(TrackVec& tracks, TrackExtraVec& extras, const bool realigntracks); // sort hits by layer, init track extras, align track labels if true
    void score_tracks(TrackVec& tracks); // if track score not already assigned

    void PrepareSeeds();
    */


  }  // namespace StdSeq

}  // namespace mkfit

#endif
