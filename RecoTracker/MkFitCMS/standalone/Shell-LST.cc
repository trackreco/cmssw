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

#include "TFile.h"
#include "TTree.h"

namespace mkfit {

  //===========================================================================
  #pragma region Event Loops
  //===========================================================================

  void Shell::LoopNEvents(int N_events) {
    N_events = std::min(N_events, m_evs_in_file);
    for (int ev = 1; ev <= N_events; ++ev) {

      printf("\n##### BEG Event %d #####\n\n", ev);

      GoToEvent(ev);

      // find max pT5 index (seeds are: pT5, T5, p)
      int min_seed = 0; // for now always 0
      int max_seed = 0;
      auto const &seeds = m_event->seedTracks_;
      int ns = seeds.size();
      for (int si = 0; si < ns; ++si) {
        auto const &s = seeds[si];
        auto hot = s.getHitOnTrack(0);
        auto &li = tracker_info()->layer(hot.layer);
        if ( ! li.is_pixel() || s.algoint() != 4) {
          max_seed = si - 1;
          break;
        }
      }
      int num_seeds = max_seed - min_seed + 1;
      printf("Selected %d LST pT5 seeds [%d, %d].\n", num_seeds, min_seed, max_seed);

      if (num_seeds > 0) {
        RunLSTintoPix(mkfit::Shell::SS_IndexPreCleaning, min_seed, num_seeds);
      }

      printf("\n##### END Event %d #####\n", ev);
    }
  }

  void Shell::LoopNEventsHlt(int N_events, const int wanted_algo, const bool dump_all) {

    struct Dump {
      int ev;
      int label;
      int seed_index;
      float pt, eta, phi;
      float rpt, reta, rphi;
      int npix;
      int npix_good;
      int nstrip;
      int nstrip_good;
    } d;

    TFile *F = TFile::Open("t5.root", "RECREATE");
    TTree *T = new TTree("T","T5 into pix");
    T->Branch("d",&d.ev,"ev/I:label:seed_index:pt/F:eta:phi:rpt:reta:rphi:npix/I:npix_good:nstrip:nstrip_good");

    StdSeq::Quality::s_quality_sum.quality_reset();

    N_events = std::min(N_events, m_evs_in_file);

    for (int ev = 1; ev <= N_events; ++ev) {

      printf("\n##### BEG Event %d ##### HLT seeds\n\n", ev);

      GoToEvent(ev);
      m_event->relabelSeedTracksSequentially();

      const bool wanted_first_is_pix = true;
      const bool wanted_last_is_strip = true;
      bool in_wanted = false;
      bool in_algo = false;

      bool first_is_pix = false;
      bool last_is_strip = false;
      int n_match = 0;

      auto do_print = [&]() {
        printf("END of Seed pattern first_is_pix = %d, last_is_strip = %d ==> N = %d\n",
                first_is_pix, last_is_strip, n_match);
      };

      auto do_match = [&](bool f_is_pix, bool l_is_strp)->bool {
        return (f_is_pix == first_is_pix && l_is_strp == last_is_strip);
      };

      auto do_reset = [&](bool f_is_pix, bool l_is_strp) {
        first_is_pix = f_is_pix;
        last_is_strip = l_is_strp;
        n_match = 1;
      };

      // find max pT5 index (seeds are: pT5, T5, p)
      int min_seed = -1;
      int num_seeds = 0;
      auto const &seeds = m_event->seedTracks_;
      int ns = seeds.size();
      for (int si = 0; si < ns; ++si) {
        auto const &s = seeds[si];

        const LayerInfo &lbeg = tracker_info()->layer(s.getHitOnTrack(0).layer);
        const LayerInfo &lend = tracker_info()->layer(s.getLastHitLyr());
        const bool f_is_pix = lbeg.is_pixel();
        const bool l_is_strp = ! lend.is_pixel();

        if (s.algoint() != wanted_algo) {
          if (in_algo) {
            break;
          }
          continue;
        } else {
          if (!in_algo) {
            in_algo = true;
            // Make sure to fail the match next.
            first_is_pix = ! f_is_pix;
            last_is_strip = ! l_is_strp;
          }
        }

        if (do_match(f_is_pix, l_is_strp)) {
          if (in_wanted)
            ++num_seeds;
           ++n_match;
        } else {
          if (n_match) // n_match == 0 for first entry here
            do_print();
          if (in_wanted && !dump_all)
            break;

          if (f_is_pix == wanted_first_is_pix && l_is_strp == wanted_last_is_strip) {
            in_wanted = true;
            min_seed = si;
            num_seeds = 1;
          } else {
            in_wanted = false;
          }

          do_reset(f_is_pix, l_is_strp);
        }
      }
      do_print();

      printf("Selected %d LST pT5 seeds [%d, %d].\n", num_seeds, min_seed, min_seed + num_seeds);

      const bool print_seed_summary = false;
      const bool print_seed_details = false;
      if (print_seed_summary) {
        for (int si = min_seed; si < min_seed + num_seeds; ++si) {
          const Track &t = m_event->seedTracks_[si];
          auto sifh = m_event->simInfoForTrack(t);
          if (sifh.is_set()) {
            printf("  si=%d n_hits=%d n_match=%d n_pix=%d n_pix_match=%d frac=%f lab=%d\n",
                  si, sifh.n_hits, sifh.n_match, sifh.n_pix, sifh.n_pix_match, sifh.good_frac(), sifh.label);
          } else {
            printf("  si=%d no label\n", si);
          }
          if (print_seed_details) {
            mkfit::print("  SEED", si, t, *event());
          }
        }
      }

      // HACK -- seed validation -- comment out the RunLSTintoPix() call below.
      // m_tracks = m_event->seedTracks_;

      if (num_seeds > 0) {
        RunLSTintoPix(mkfit::Shell::SS_IndexPreCleaning, min_seed, num_seeds);
      }

      StdSeq::Quality qval;
      qval.quality_val(m_event);

      const bool print_bad_seed_vector = true; // depends on selection below
      std::vector<int> bad_seeds;

      int NT = m_tracks.size();
      for (int i = 0; i < NT; ++i) {
        const Track &t = m_tracks[i];
        auto sifh = m_event->simInfoForTrack(t);
        if (!sifh.is_set()) // only take tracks with sim match
          continue;

        if (print_bad_seed_vector) {
          if (sifh.n_pix_match == 0 && std::abs(t.momEta()) < 1.0f) { // catching cases where we add no pixel hits
            printf("bad seed %d  pt=%.3f, eta=%.3f, phi=%3f\n", t.label(), t.pT(), t.momEta(), t.momPhi());
            bad_seeds.push_back(t.label());
          }
        }

        const Track &s = m_event->simTracks_[sifh.label];
        d.ev = ev;
        d.label = sifh.label;
        d.seed_index = t.label();
        d.pt = s.pT();
        d.eta = s.momEta();
        d.phi = s.momPhi();
        d.rpt = t.pT();
        d.reta = t.momEta();
        d.rphi = t.momPhi();
        d.npix = d.npix_good = d.nstrip = d.nstrip_good = 0;
        for (int hi = 0; hi < t.nTotalHits(); ++hi) {
          auto hot = t.getHitOnTrack(hi);
          // printf(" %d", hot.index);
          if (hot.index < 0)
            continue;
          const Hit &h = m_event->layerHits_[hot.layer][hot.index];
          int hl = m_event->simHitsInfo_[h.mcHitID()].mcTrackID_;
          // printf(" (%d)", hl);
          if (tracker_info()->layer(hot.layer).is_pixel()) {
            if (hl == sifh.label)
              ++d.npix_good;
            ++d.npix;
          } else {
            if (hl == sifh.label)
              ++d.nstrip_good;
            ++d.nstrip;
          }
        }
        T->Fill();
      }

      if (print_bad_seed_vector) {
        printf("s.SetSeedsFromIdcs({ ");
        int nbs = bad_seeds.size();
        if (nbs > 0) printf("%d", bad_seeds[0]);
        for (int bsi = 1; bsi < nbs; ++bsi)
          printf(", %d", bad_seeds[bsi]);
        printf(" })\n");
      }

      printf("\n##### END Event %d #####\n", ev);
    }

    StdSeq::Quality::s_quality_sum.quality_print();

    T->Write();
    F->Close();
    delete F;
  }

  #pragma endregion Event Loops

  //===========================================================================
  #pragma region RunLSTintoPix
  //===========================================================================

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
      // s.sortHitsByLayer();

      // print("seed-post-sort", is, s, *m_event);

      const bool clear_out_pixel_hits = false;
      if (clear_out_pixel_hits) {
        std::vector<HitOnTrack> ohits;
        s.swapOutAndResetHits(ohits);
        for (auto &oh : ohits) {
          if (trackerInfo[oh.layer].is_pixel())
            continue;
          s.addHitIdx(oh, 0.0f);
        }
      }

      // QQQQ test hacking seed charge if it is zero
      if (s.charge() == 0) {
        auto sifh = m_event->simInfoForTrack(s);
        if (sifh.is_set()) {
          int chg = m_event->simTracks_[sifh.label].charge();
          printf("ZERO-CHG: setting charge to %d for seed %d\n", chg, is);
          s.setCharge(chg);
        } else {
          printf("ZERO-CHG: failed getting label from hits for seed %d\n", is);
        }
      }
      // print("seed-post-pix-removal", is, s, *m_event);

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

      builder.findTracksStandardv2p2();

      job.switch_to_backward();

      builder.compactifyHitStorageForBestCand(false, 99); // do not remove anything
      builder.backwardFit(); // prop_to_plane depends on Config::usePropToPlane

      builder.beginBkwSearch();

      // QQQQQ
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
      // builder.ref_eocc_nc().scaleErrors(100.0f);

      // print("post-bkfit-n-scale state", builder.ref_eocc()[0][0].state());

      // builder.findTracksCloneEngine(SteeringParams::IT_BkwSearch);
      builder.findTracksStandardv2p2(SteeringParams::IT_BkwSearch);

      filter_candidates_func post_filter;
      post_filter = StdSeq::qfilter_nan_n_silly<TrackCand>;
      // post_filter is always at least doing nan_n_silly filter.
      builder.filter_comb_cands(post_filter, true);

      builder.endBkwSearch();

      builder.export_best_comb_cands(out_tracks, false /*true*/); // do not remove missing hits

      m_event->candidateTracks_ = m_tracks; // For quality-val

      // Do not clear ... useful for debugging / printouts!
      // m_event->resetCurrentSeedTracks();

      builder.end_event();
    }

    // print("END LST into Pix", m_tracks, *m_event);
  }

  #pragma endregion RunLSTintoPix

}
