#include "RecoTracker/MkFitCMS/standalone/Shell.h"

#include "RecoTracker/MkFitCore/src/Debug.h"

#include "RecoTracker/MkFitCMS/interface/runFunctions.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/MkBuilder.h"
#include "RecoTracker/MkFitCore/src/MkFitter.h"
#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"
#include "RecoTracker/MkFitCMS/standalone/MkStandaloneSeqs.h"

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"

#include "RecoTracker/MkFitCore/standalone/Event.h"

#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

namespace mkfit {

  void Shell::RunLSTintoPix(SeedSelect_e seed_select, int selected_seed, int count) {
    const IterationConfig &itconf = Config::ItrInfo[m_it_index];
    IterationMaskIfc mask_ifc;
    m_event->fill_hitmask_bool_vectors(itconf.m_track_algorithm, mask_ifc.m_mask_vector);

    const TrackerInfo &trackerInfo = Config::TrkInfo;

    m_tracks.clear();

    if (seed_select != SS_PreSet) {
      m_seeds.clear();
      int n_algo = 0; // seeds are grouped by algo
      for (auto &s : m_event->seedTracks_) {
        if (s.algoint() == itconf.m_track_algorithm) {
          if (seed_select == SS_UseAll || seed_select == SS_IndexPostCleaning) {
            m_seeds.push_back(s);
          } else if (seed_select == SS_Label && s.label() == selected_seed) {
            m_seeds.push_back(s);
            if (--count <= 0)
              break;
          } else if (seed_select == SS_IndexPreCleaning && n_algo >= selected_seed) {
            m_seeds.push_back(s);
            if (--count <= 0)
              break;
          }
          ++n_algo;
        } else if (n_algo > 0)
          break;
      }
    }

    printf("Shell::RunLSTintoPix running over %d seeds\n", (int) m_seeds.size());

    for (int is = 0; auto &s : m_seeds) {
      // Chomp off pixels, pretending we have T5s
      // s.sortHitsByR(m_event->layerHits_);
      s.sortHitsByLayer();

      print("seed-post-sort", is, s, *m_event);

      std::vector<HitOnTrack> ohits;
      s.swapOutAndResetHits(ohits);
      for (auto &oh : ohits) {
        if (trackerInfo[oh.layer].is_pixel())
          continue;
        s.addHitIdx(oh, 0.0f);
      }

      print("seed-post-pix-removal", is, s, *m_event);

      ++is;
    }

    {
      const EventOfHits &eoh = *m_eoh;
      const IterationMaskIfcBase &it_mask_ifc = mask_ifc;
      MkBuilder &builder = *m_builder;
      TrackVec &seeds = m_seeds;
      TrackVec &out_tracks = m_tracks;

      MkJob job({trackerInfo, itconf, eoh, eoh.refBeamSpot(), &it_mask_ifc});

      builder.begin_event(&job, m_event, __func__);

      // Check nans in seeds.
      builder.seed_post_cleaning(seeds);

      m_event->setCurrentSeedTracks(seeds);

      builder.find_tracks_load_seeds(seeds, false); // seeds not sorted

      job.switch_to_backward();

      builder.compactifyHitStorageForBestCand(false, 99); // do not remove anything
      builder.backwardFit(true); // true: use_prop_to_plane
      builder.beginBkwSearch();

      // On a barrel track, index 1, pt 9.8
      // - selectHitIndicesV1 & V2
      //   . no scale misses the hit on layer 3 (as overlap)
      //   . scale 10 gets it right
      //   . scale 100 picks up an extra hit in layer 3
      // - selectHitIndicesV2
      //   . no scale misses the hit on layer 3 (as overlap)
      //   . scale 10 gets it right
      //   . scale 100 picks up an extra hit in layer 3
      // On a track going through 3 pixel disks both fail, index 0, pt 0.8
      builder.ref_eocc_nc().scaleErrors(100.0f);

      print("post-bkfit-n-scale state", builder.ref_eocc()[0][0].state());

      // builder.findTracksCloneEngine(SteeringParams::IT_BkwSearch);
      builder.findTracksStandardv2p2(SteeringParams::IT_BkwSearch);

      filter_candidates_func post_filter;
      post_filter = StdSeq::qfilter_nan_n_silly<TrackCand>;
      // post_filter is always at least doing nan_n_silly filter.
      builder.filter_comb_cands(post_filter, true);

      builder.endBkwSearch();

      builder.export_best_comb_cands(out_tracks, false /*true*/); // do not remove missing hits

      builder.end_event();
    }

    print("END", m_tracks, *m_event);
  }
}
