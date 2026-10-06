#ifndef RecoTracker_MkFitCMS_interface_MkStdSeqs_h
#define RecoTracker_MkFitCMS_interface_MkStdSeqs_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/DeadRegion.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

namespace mkfit {

  class EventOfHits;
  class IterationConfig;
  class TrackerInfo;
  class MkJob;
  class TrackCand;

  namespace StdSeq {

    void loadDeads(EventOfHits &eoh, const std::vector<DeadVec> &deadvectors);

    void cmssw_LoadHits_Begin(EventOfHits &eoh, const std::vector<const HitVec *> &orig_hitvectors);
    void cmssw_LoadHits_End(EventOfHits &eoh);

    // Not used anymore. Left here if we want to experiment again with
    // COPY_SORTED_HITS in class LayerOfHits.
    void cmssw_Map_TrackHitIndices(const EventOfHits &eoh, TrackVec &seeds);
    void cmssw_ReMap_TrackHitIndices(const EventOfHits &eoh, TrackVec &out_tracks);

    int clean_cms_seedtracks_iter(TrackVec &seeds, const IterationConfig &itrcfg, const BeamSpot &bspot);

    void remove_duplicates(TrackVec &tracks);

    void clean_duplicates(TrackVec &tracks, const IterationConfig &itconf);
    void clean_duplicates_sharedhits(TrackVec &tracks, const IterationConfig &itconf);
    void clean_duplicates_sharedhits_pixelseed(TrackVec &tracks, const IterationConfig &itconf);

    // Removes the tracks of flagged seeds that added fewer than min_added_hits found
    // hits to the seed's. A seed is flagged when its score, looked up by the track's
    // label in q_by_label, is in [score_lo, score_hi); the score is the seeder's
    // cleaning score, or its fake score with on_fake_score. Applied to the final
    // tracks, after the duplicate cleaner. Returns the number of tracks removed.
    struct SeedFlagCut {
      int min_added_hits = 0;  // 0 = off
      float score_lo = 0.35f;
      float score_hi = 1e30f;
      bool on_fake_score = true;
      // score_lo grows at low pT and in an |eta| band, where true seeds score higher (multiple scattering):
      // score_lo * max(1, pt_ref / pT) * (eta_fac if eta_lo <= |eta| < eta_hi); the track's pT and eta
      float pt_ref = 0;  // 0 = no pT dependence
      float eta_lo = 0, eta_hi = 0, eta_fac = 1;
    };
    int remove_flagged_seed_tracks(TrackVec &tracks, const std::vector<SeedQuality> &q_by_label, const SeedFlagCut &cut);

    // Quality filters used directly (not through IterationConfig)

    template <class TRACK>
    bool qfilter_nan_n_silly(const TRACK &t, const MkJob &) {
      return !(t.hasNanNSillyValues());
    }

  }  // namespace StdSeq

}  // namespace mkfit

#endif
