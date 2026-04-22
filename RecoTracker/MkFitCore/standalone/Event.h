#ifndef RecoTracker_MkFitCore_standalone_Event_h
#define RecoTracker_MkFitCore_standalone_Event_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/BeamSpot.h"
#include "Validation.h"

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#endif

#include <mutex>

namespace mkfit {

  struct DataFile;

  class Event {
  public:
    explicit Event(int evtID, int nLayers);
    Event(Validation &v, int evtID, int nLayers);

    void reset(int evtID);
    void validate();
    void printStats(const TrackVec &, TrackExtraVec &);

    int evtID() const { return evtID_; }
    void resetLayerHitMap(bool resetSimHits);

    void write_out(DataFile &data_file);
    void read_in(DataFile &data_file, FILE *in_fp = 0);
    int write_tracks(FILE *fp, const TrackVec &tracks);
    int read_tracks(FILE *fp, TrackVec &tracks, bool skip_reading = false);

    void setInputFromCMSSW(std::vector<HitVec> hits, TrackVec seeds);

    int use_seeds_from_cmsswtracks();  //special mode --> use only seeds which generated cmssw reco track
    int clean_cms_simtracks();
    int clean_cms_seedtracks(
        TrackVec *seed_ptr = nullptr);    //operates on seedTracks_; returns the number of cleaned seeds
    int clean_cms_seedtracks_badlabel();  //operates on seedTracks_, removes those with label == -1;
    void relabel_bad_seedtracks();
    void relabel_cmsswtracks_from_seeds();

    int select_tracks_iter(unsigned int n = 0);  //for cmssw input

    void fill_hitmask_bool_vectors(int track_algo, std::vector<std::vector<bool>> &layer_masks);
    void fill_hitmask_bool_vectors(std::vector<int> &track_algo_vec, std::vector<std::vector<bool>> &layer_masks);

    Validation &validation_;

    // For seed access in deep data dumpers.
    struct SimInfoFromHits {
      int label = -1, n_hits = 0, n_valid, n_match = 0;
      int n_pix = 0, n_pix_match = 0;
      int n_strip = 0, n_strip_match = 0;
      float good_frac() const { return (float)n_match / n_valid; }
      int n_invalid() const { return n_hits - n_valid; }
      int n_pix_bad() const { return n_pix - n_pix_match; }
      int n_strip_bad() const { return n_strip - n_strip_match; }
      bool is_set() const { return label >= 0; }
    };
    SimInfoFromHits simInfoForTrack(const Track &s) const;
    SimInfoFromHits simInfoForTrack(Track &s, bool relabel);

    int countSimHitsInLayer(int label, int layer) const;
    int countPixelHits(const Track &track) const;
    int countPixelLayers(const Track &track) const;
    int lastPixelLayer(const Track &track) const;

    void setCurrentSeedTracks(const TrackVec &seeds);
    void resetCurrentSeedTracks();
    const Track &currentSeed(int i) const { return (*currentSeedTracks_)[i]; }
    SimInfoFromHits simInfoForCurrentSeed(int i) const { return currentSeedSimFromHits_[i]; }
    const TrackVec& currentSeedTracks() const { return *currentSeedTracks_; }

    void relabelSeedTracksSequentially();

    void print_tracks(const TrackVec &tracks, bool print_hits) const;

    size_t memUsage() const;
    void printMemUsage() const;

  private:
    int evtID_;

  public:
    BeamSpot beamSpot_;  // XXXX Read/Write of BeamSpot + file-version bump or extra-section to be added.
    std::vector<HitVec> layerHits_;
    std::vector<std::vector<uint64_t>> layerHitMasks_;  //aligned with layerHits_
    MCHitInfoVec simHitsInfo_;

    TrackVec simTracks_, seedTracks_, candidateTracks_, fitTracks_;
    TrackVec cmsswTracks_;
    // validation sets these, so needs to be mutable
    mutable TrackExtraVec simTracksExtra_, seedTracksExtra_, candidateTracksExtra_, fitTracksExtra_;
    mutable TrackExtraVec cmsswTracksExtra_;

    TSVec simTrackStates_;

    const TrackVec *currentSeedTracks_ = nullptr;
    mutable std::vector<SimInfoFromHits> currentSeedSimFromHits_;

  #ifdef MKFIT_TRACE
    // Not thread safe within event, multiple Events ok.
    mutable std::vector<TrCandMeta> trCandMetas_;
    mutable std::vector<TrCandStage> trCandStages_;
    mutable std::vector<TrCandState> trCandStates_;
    mutable std::vector<TrHitMatch> trHitMatches_;
    mutable std::vector<TrKalmanUpdate> trKalmanUpdates_;

    mutable TrackVec trSeeds_;
    /* *** for multiple iteration tracing ***
      could be made std::vector<TrackVec> trSeedsPerIter_;
      but then would also need:
      struct IterTraceInfo {
        int meta_begin = 0;
        int state_begin = 0;
        int seed_begin = 0;  // Index into trSeedsPerIter_
        int iteration_idx = 0;
        int search_direction = 0;
      };
      mutable std::vector<IterTraceInfo> trIterInfos_;
    */

    mutable SeedVecInsp seedVecInsp_; // describe mixture of input seeds for HLT setup

    TrCandMeta& tr_candmeta(int i) const { return trCandMetas_[i]; }
    TrCandStage& tr_candstage(int i) const { return trCandStages_[i]; }
    TrCandState& tr_candstate(int i) const { return trCandStates_[i]; }
    TrHitMatch& tr_hitmatch(int i) const { return trHitMatches_[i]; }
    TrKalmanUpdate& tr_kalmanupdate(int i) const { return trKalmanUpdates_[i]; }

    TrCandMeta& trace_candmeta(TrCandMeta && cm) const {
      int s = trCandMetas_.size();
      auto &t = trCandMetas_.emplace_back(cm);
      t.id = s;
      return t;
    }
    TrCandStage& trace_candstage(TrCandStage && cs) const {
      int s = trCandStages_.size();
      auto &t = trCandStages_.emplace_back(cs);
      t.id = s;
      return t;
    }
    TrCandState& trace_candstate(TrCandState && cs) const {
      int s = trCandStates_.size();
      auto &t = trCandStates_.emplace_back(cs);
      t.id = s;
      return t;
    }
    TrHitMatch& trace_hitmatch(TrHitMatch && hm) const {
      int s = trHitMatches_.size();
      auto &t = trHitMatches_.emplace_back(hm);
      t.id = s;
      return t;
    }
    TrKalmanUpdate& trace_kalmanupdate(TrKalmanUpdate && ku) const {
      int s = trKalmanUpdates_.size();
      auto &t = trKalmanUpdates_.emplace_back(ku);
      t.id = s;
      return t;
    }

    int trace_new_cand_meta(int event, int seed_index) {
      auto &cm = trace_candmeta({ -1, event, seed_index });
      return cm.id;
    }
    // always do stage and initial state together
    // int trace_new_cand_stage(int meta_id, int parent_stage_id, int stage) {
    //   auto &cstg = trace_candstage({ -1, meta_id, parent_stage_id, stage });
    //   return cstg.id;
    // }
    std::pair<int,int>
    trace_new_cand_stage_and_state(int meta_id, int parent_stage_id, int stage, int layer, const EBiVec3 &state) const {
      assert(stage >= 0 && stage <= 2 && "stage expected to be between 0 and 2");
      auto &cstage = trace_candstage({ -1, meta_id, parent_stage_id, stage });
      auto &cstate = trace_candstate({ -1, -1, meta_id, cstage.id, layer, 0, state });
      cstage.root_state_id = cstate.id;
      return { cstage.id, cstate.id };
    }
    int trace_new_cand_state(int parent_state_id, int layer, const EBiVec3 &state) const {
      auto &pcs = trCandStates_[parent_state_id];
      pcs.has_children = true;
      auto &cs = trace_candstate({ -1, parent_state_id, pcs.meta_id, pcs.stage_id, layer, pcs.step + 1, state });
      return cs.id;
    }

    // Aggregators, maps
    void build_trace_maps_etc();

    std::vector<int> trRootCands_;
    std::unordered_map<int, std::vector<int>> trChildrenByCand_;
    std::unordered_map<int, std::vector<int>> trHitMatchesByCand_;
    std::unordered_map<int, std::vector<int>> trKalmanUpdatesByCand_;

    std::vector<SimInfoFromHits> trSIFHforSeedByMeta_;
    std::vector<SimInfoFromHits> trSIFHforCandByMeta_;
  #endif

    static std::mutex printmutex;
  };

  typedef std::vector<Event> EventVec;

  struct DataFileHeader {
    int f_magic = 0xBEEF;
    int f_format_version = 7;  //last update with ph2 geom
    int f_sizeof_track = sizeof(Track);
    int f_sizeof_hit = sizeof(Hit);
    int f_sizeof_hot = sizeof(HitOnTrack);
    int f_n_layers = -1;
    int f_n_events = -1;

    int f_extra_sections = 0;

    DataFileHeader() = default;
  };

  struct DataFile {
    enum ExtraSection {
      ES_SimTrackStates = 0x1,
      ES_Seeds = 0x2,
      ES_CmsswTracks = 0x4,
      ES_HitIterMasks = 0x8,
      ES_BeamSpot = 0x10
    };

    FILE *f_fp = 0;
    long f_pos = sizeof(DataFileHeader);

    DataFileHeader f_header;

    std::mutex f_next_ev_mutex;

    // ----------------------------------------------------------------

    bool hasSimTrackStates() const { return f_header.f_extra_sections & ES_SimTrackStates; }
    bool hasSeeds() const { return f_header.f_extra_sections & ES_Seeds; }
    bool hasCmsswTracks() const { return f_header.f_extra_sections & ES_CmsswTracks; }
    bool hasHitIterMasks() const { return f_header.f_extra_sections & ES_HitIterMasks; }
    bool hasBeamSpot() const { return f_header.f_extra_sections & ES_BeamSpot; }

    int openRead(const std::string &fname, int expected_n_layers);
    void openWrite(const std::string &fname, int n_layers, int n_ev, int extra_sections = 0);

    void rewind();

    int advancePosToNextEvent(FILE *fp);

    void skipNEvents(int n_to_skip);

    void close();
    void CloseWrite(int n_written);  //override nevents in the header and close
  };

  void print(std::string pfx, int itrack, const Track &trk, const Event &ev);

  void print(std::string pfx, int itrack, const Track &trk, int hit_begin, int hit_end, const Event &ev);

  void print(std::string pfx, const TrackVec &tvec, const Event &ev);

  void print(std::string pfx, const Event::SimInfoFromHits &si);

#ifdef MKFIT_TRACE
  void print(std::string pfx, const ::EBiVec3 &s);
  void print(std::string pfx, const TrCandMeta &cm, const Event *ev);
  void print(std::string pfx, const TrCandStage &cs);
  void print(std::string pfx, const TrCandState &cs);
  void print(std::string pfx, const TrHitMatch &hm);
  void print(std::string pfx, const TrKalmanUpdate &ku);
#endif

}  // end namespace mkfit
#endif
