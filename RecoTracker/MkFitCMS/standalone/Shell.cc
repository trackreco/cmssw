#include "RecoTracker/MkFitCMS/standalone/Shell.h"

#include "RecoTracker/MkFitCore/src/Debug.h"

// #include "RecoTracker/MkFitCore/src/Matriplex/MatriplexCommon.h"

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

#include "TROOT.h"
#include "TRint.h"

#include "TFile.h"
#include "TTree.h"

#ifdef WITH_REVE
#include "TRandom.h"
#include "ROOT/REveJetCone.hxx"
#include "ROOT/REveManager.hxx"
#include "ROOT/REveScene.hxx"
#include "ROOT/REveBoxSet.hxx"
#endif

#include "oneapi/tbb/task_arena.h"

#include <vector>
#include <set>

// clang-format off

namespace {
  constexpr int algos[] = {4, 22, 23, 5, 24, 7, 8, 9, 10, 6};  // 10 iterations
  constexpr int n_algos = sizeof(algos) / sizeof(int);
  constexpr int end_algo = 24 + 1;

  const char* b2a(bool b) { return b ? "true" : "false"; }

  std::vector<mkfit::DeadVec> loc_dummy_deadvec;
}

namespace mkfit {

  Shell::Shell()
    : m_deadvectors(loc_dummy_deadvec)
  {
    // Constructor for use in plain ROOT, easy to crash!
  }

  Shell::Shell(std::vector<DeadVec> &dv, const std::string &in_file, int start_ev)
    : m_deadvectors(dv)
  {
    m_eoh = new EventOfHits(Config::TrkInfo);
    m_builder = new MkBuilder(Config::silent);

    m_backward_fit = Config::backwardFit;

    m_data_file = new DataFile;
    m_event = new Event(0, Config::TrkInfo.n_layers());

    if ( ! in_file.empty() && Config::nEvents > 0) {
      m_evs_in_file = m_data_file->openRead(in_file, Config::TrkInfo.n_layers());
      GoToEvent(start_ev);
    } else {
      printf("Shell initialized but the %s, running on an empty Event.\n",
      in_file.empty() ? "input-file not specified" : "requested number of events to process is 0");
    }
  }

  Shell::~Shell() {
    delete m_event;
    delete m_data_file;
    delete m_builder;
    delete m_eoh;
    delete gApplication;
  }

  void Shell::Run() {
    std::vector<const char *> argv = { "mkFit", "-l" };
    int argc = argv.size();
    gApplication = new TRint("mkFit-shell", &argc, const_cast<char**>(argv.data()));

    char buf[256];
    sprintf(buf, "mkfit::Shell &s = * (mkfit::Shell*) %p;", this);
    gROOT->ProcessLine(buf);
    printf("Shell &s variable is set: ");
    gROOT->ProcessLine("s");

    gApplication->Run(true);
    printf("Shell::Run finished\n");
  }

  void Shell::Status() {
    printf("On event %d, selected iteration index %d, algo %d - %s\n"
          "  debug = %s, use_dead_modules = %s\n"
           "  clean_seeds = %s, backward_fit = %s, remove_duplicates = %s\n",
           m_event->evtID(), m_it_index, algos[m_it_index], TrackBase::algoint_to_cstr(algos[m_it_index]),
           b2a(g_debug), b2a(Config::useDeadModules),
           b2a(m_clean_seeds), b2a(m_backward_fit), b2a(m_remove_duplicates));
  }

  TrackerInfo* Shell::tracker_info() { return &Config::TrkInfo; }

  //===========================================================================
  // Event navigation / processing
  //===========================================================================

  void Shell::GoToEvent(int eid) {
    if (eid < 1) {
      fprintf(stderr, "Requested event %d is less than 1 -- 1 is the first event, %d is total number of events in file\n",
             eid, m_evs_in_file);
      throw std::runtime_error("event out of range");
    }
    if (eid > m_evs_in_file) {
      fprintf(stderr, "Requested event %d is grater than total number of events in file %d\n",
             eid, m_evs_in_file);
      throw std::runtime_error("event out of range");
    }

    int pos = m_event->evtID();
    if (eid > pos) {
      m_data_file->skipNEvents(eid - pos - 1);
    } else {
      m_data_file->rewind();
      m_data_file->skipNEvents(eid - 1);
    }
    m_event->resetCurrentSeedTracks(); // left after ProcessEvent() for debugging etc
    m_event->reset(eid);
    m_event->read_in(*m_data_file);
    StdSeq::loadHitsAndBeamSpot(*m_event, *m_eoh);
    if (Config::useDeadModules) {
      StdSeq::loadDeads(*m_eoh, m_deadvectors);
    }

    printf("At event %d\n", eid);
  }

  void Shell::NextEvent(int skip) {
    GoToEvent(m_event->evtID() + skip);
  }

  void Shell::ProcessEvent(SeedSelect_e seed_select, int selected_seed, int count) {
    // count is mostly used for SS_IndexPreCleaning and SS_IndexPostCleaning.
    // It is also honoured for SS_Label, as (especially without cleaning) there might be several
    // seeds with the same label.
    // There are no checks for upper bounds, ie, if requested seeds beyond the first one exist.

    const IterationConfig &itconf = Config::ItrInfo[m_it_index];
    IterationMaskIfc mask_ifc;
    m_event->fill_hitmask_bool_vectors(itconf.m_track_algorithm, mask_ifc.m_mask_vector);

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

    printf("Shell::ProcessEvent running over %d seeds\n", (int) m_seeds.size());

    // Equivalent to run_OneIteration(...) without MkBuilder::release_memory().
    // If seed_select == SS_IndexPostCleaning the given seed is picked after cleaning.
    {
      const TrackerInfo &trackerInfo = Config::TrkInfo;
      const EventOfHits &eoh = *m_eoh;
      const IterationMaskIfcBase &it_mask_ifc = mask_ifc;
      MkBuilder &builder = *m_builder;
      TrackVec &seeds = m_seeds;
      TrackVec &out_tracks = m_tracks;
      bool do_seed_clean = m_clean_seeds && seed_select != SS_PreSet;
      bool do_backward_fit = m_backward_fit;
      bool do_remove_duplicates = m_remove_duplicates;

      MkJob job({trackerInfo, itconf, eoh, eoh.refBeamSpot(), &it_mask_ifc});

      builder.begin_event(&job, m_event, __func__);

      // Seed cleaning not done on all iterations.
      do_seed_clean = m_clean_seeds && itconf.m_seed_cleaner;

      if (do_seed_clean) {
        itconf.m_seed_cleaner(seeds, itconf, eoh.refBeamSpot());
        printf("Shell::ProcessEvent post seed-cleaning: %d seeds\n", (int) m_seeds.size());
      } else {
        printf("Shell::ProcessEvent no seed-cleaning\n");
      }

      // Check nans in seeds -- this should not be needed when Slava fixes
      // the track parameter coordinate transformation.
      builder.seed_post_cleaning(seeds);

      if (seed_select == SS_IndexPostCleaning) {
        int seed_size = (int) seeds.size();
        if (selected_seed >= 0 && selected_seed < seed_size) {
          if (selected_seed + count >= seed_size) {
            count = seed_size - selected_seed;
            printf("  -- selected seed_index + count > seed vector size after cleaning -- trimming count to %d\n",
                   count);
          }
          for (int i = 0; i < count; ++i)
            seeds[i] = seeds[selected_seed + i];
          seeds.resize(count);
        } else {
          seeds.clear();
        }
      }

      if (seeds.empty()) {
        if (seed_select != SS_UseAll)
          printf("Shell::ProcessEvent requested seed not found among seeds of the selected iteration.\n");
        else
          printf("Shell::ProcessEvent no seeds found.\n");
        return;
      }

      if (itconf.m_requires_seed_hit_sorting) {
        for (auto &s : seeds)
          s.sortHitsByLayer();  // sort seed hits for the matched hits (I hope it works here)
      }

      m_event->setCurrentSeedTracks(seeds);

      builder.find_tracks_load_seeds(seeds, do_seed_clean);

      // Hmmh, can make this configurable through Shell / MkJob / MkBuilder?
      // builder.findTracksCloneEngine();
      printf("Shell::ProcessEvent running MkBuilder::findTracksStandardv2p2() -- HARDCODED for now.\n");
      builder.findTracksStandardv2p2();

      printf("Shell::ProcessEvent post fwd search: %d comb-cands\n", builder.ref_eocc().size());

      // Pre backward-fit filtering.
      filter_candidates_func pre_filter;
      if (do_backward_fit && itconf.m_pre_bkfit_filter)
        pre_filter = [&](const TrackCand &tc, const MkJob &jb) -> bool {
          return itconf.m_pre_bkfit_filter(tc, jb) && StdSeq::qfilter_nan_n_silly<TrackCand>(tc, jb);
        };
      else if (itconf.m_pre_bkfit_filter)
        pre_filter = itconf.m_pre_bkfit_filter;
      else if (do_backward_fit)
        pre_filter = StdSeq::qfilter_nan_n_silly<TrackCand>;
      // pre_filter can be null if we are not doing backward fit as nan_n_silly will be run below.
      if (pre_filter)
        builder.filter_comb_cands(pre_filter, true);

      printf("Shell::ProcessEvent post pre-bkf-filter (%s) and nan-filter (%s) filter: %d comb-cands\n",
             b2a(bool(itconf.m_pre_bkfit_filter)), b2a(do_backward_fit), builder.ref_eocc().size());

      job.switch_to_backward();

      if (do_backward_fit) {
        if (itconf.m_backward_search) {
          builder.compactifyHitStorageForBestCand(itconf.m_backward_drop_seed_hits, itconf.m_backward_fit_min_hits);
        }

        builder.backwardFit();

        if (itconf.m_backward_search) {
          builder.beginBkwSearch();
          builder.findTracksCloneEngine(SteeringParams::IT_BkwSearch);
        }

        printf("Shell::ProcessEvent post backward fit / search: %d comb-cands\n", builder.ref_eocc().size());
      }

      // Post backward-fit filtering.
      filter_candidates_func post_filter;
      if (do_backward_fit && itconf.m_post_bkfit_filter)
        post_filter = [&](const TrackCand &tc, const MkJob &jb) -> bool {
          return itconf.m_post_bkfit_filter(tc, jb) && StdSeq::qfilter_nan_n_silly<TrackCand>(tc, jb);
        };
      else
        post_filter = StdSeq::qfilter_nan_n_silly<TrackCand>;
      // post_filter is always at least doing nan_n_silly filter.
      builder.filter_comb_cands(post_filter, true);

      printf("Shell::ProcessEvent post post-bkf-filter (%s) and nan-filter (true): %d comb-cands\n",
             b2a(do_backward_fit && itconf.m_post_bkfit_filter), builder.ref_eocc().size());

      if (do_backward_fit && itconf.m_backward_search)
        builder.endBkwSearch();

      builder.export_best_comb_cands(out_tracks, true);

      if (do_remove_duplicates && itconf.m_duplicate_cleaner) {
        itconf.m_duplicate_cleaner(out_tracks, itconf);
      }

      printf("Shell::ProcessEvent post remove-duplicates: %d comb-cands\n", (int) out_tracks.size());

      // Do not clear ... useful for debugging / printouts!
      // m_event->resetCurrentSeedTracks();

      builder.end_event();
    }

    printf("Shell::ProcessEvent found %d tracks, number of seeds at end %d\n",
           (int) m_tracks.size(), (int) m_seeds.size());
  }

  //===========================================================================
  // Iteration selection
  //===========================================================================

  void Shell::SelectIterationIndex(int itidx) {
    if (itidx < 0 || itidx >= n_algos) {
      fprintf(stderr, "Requested iteration index out of range [%d, %d)", 0, n_algos);
      throw std::runtime_error("iteration index out of range");
    }
    m_it_index = itidx;
  }

  void Shell::SelectIterationAlgo(int algo) {
    for (int i = 0; i < n_algos; ++i) {
      if (algos[i] == algo) {
        m_it_index = i;
        return;
      }
    }
    fprintf(stderr, "Requested algo %d not found", algo);
    throw std::runtime_error("algo not found");
  }

  void Shell::PrintIterations() {
    printf("Shell::PrintIterations selected index = %d, %d iterations hardcoded as\n",
            m_it_index, n_algos);
    for (int i = 0; i < n_algos; ++i)
      printf("%d %2d %s\n", i, algos[i], TrackBase::algoint_to_cstr(algos[i]));
  }

  //===========================================================================
  // Flags / status setters
  //===========================================================================

  void Shell::SetDebug(bool b) { g_debug = b; }
  void Shell::SetCleanSeeds(bool b) { m_clean_seeds = b; }
  void Shell::SetBackwardFit(bool b) { m_backward_fit = b; }
  void Shell::SetRemoveDuplicates(bool b) { m_remove_duplicates = b; }
  void Shell::SetUseDeadModules(bool b) { Config::useDeadModules = b; }

  //===========================================================================
  // Analysis helpers
  //===========================================================================

  /*
    sim tracks are written to .bin files with a label equal to its own index.
    reco tracks labels are seed indices.
    seed labels are sim track indices
    --
    mkfit labels are seed indices in given iteration after cleaning (at seed load-time).
          This is no longer true -- was done like that in branch where this code originated from.
          It seems the label is the same as seed label.
          CombCandidate does have m_seed_origin_index -- index of seed after cleaning.
  */

  int Shell::LabelFromHits(Track &t, bool replace, float good_frac) {
    auto sifh = m_event->simInfoForTrack(t);
    bool success = sifh.good_frac()>= good_frac;
    int relabel = success ? sifh.label : -1;
    // printf("found_hits=%d, best_lab %d (%d hits), existing label=%d (replace flag=%s)\n",
    //        t.nFoundHits(), sifh.label, sifh.n_match, t.label(), b2a(replace));
    if (replace)
        t.setLabel(relabel);
    return relabel;
  }

  void Shell::FillByLabelMaps_CkfBase() {
    Event &ev = *m_event;
    const int track_algo = Config::ItrInfo[m_it_index].m_track_algorithm;

    m_ckf_map.clear();
    m_sim_map.clear();
    m_seed_map.clear();
    m_mkf_map.clear();

    // Pick ckf tracks with right algo and a good label.
    int rec_algo_match = 0;
    for (auto &t : ev.cmsswTracks_) {
      if (t.algoint() != track_algo)
        continue;
      ++rec_algo_match;
      int label = LabelFromHits(t, false, 0.5);
      if (label >= 0) {
        m_ckf_map.insert(std::make_pair(label, &t));
      }
    }

    // Pick sim tracks with labels found by ckf.
    for (auto &t : ev.simTracks_) {
      if (t.label() >= 0 && m_ckf_map.find(t.label()) != m_ckf_map.end()) {
        m_sim_map.insert(std::make_pair(t.label(), &t));
      }
    }

    // Pick seeds with right algo and a label found by ckf.
    for (auto &t : ev.seedTracks_) {
      if (t.algoint() == track_algo && t.label() >= 0 && m_ckf_map.find(t.label()) != m_ckf_map.end()) {
        m_seed_map.insert(std::make_pair(t.label(), &t));
      }
    }
    // Some seeds seem to be labeled -1, try fixing when not otherwise found.
    for (auto &t : ev.seedTracks_) {
      if (t.algoint() == track_algo && t.label() == -1) {
        int lab = LabelFromHits(t, true, 0.5);
        if (lab >= 0 && m_seed_map.find(lab) == m_seed_map.end()) {
          if (m_ckf_map.find(lab) != m_ckf_map.end()) {
            m_seed_map.insert(std::make_pair(t.label(), &t));
            printf("Saved seed with label -1 -> %d\n", lab);
          }
        }
      }
    }

    // Pick mkfit tracks, label by 
    for (auto &t : m_tracks) {
      int label = LabelFromHits(t, false, 0.5);
      if (label >= 0) {
        m_mkf_map.insert(std::make_pair(label, &t));
      }
    }

    printf("Shell::FillByLabelMaps reporting tracks with label >= 0, algo=%d (%s): "
           "ckf: %d of %d (same algo=%d)), sim: %d of %d, seed: %d of %d, mkfit: %d w/label of %d\n",
           track_algo, TrackBase::algoint_to_cstr(track_algo),
           (int) m_ckf_map.size(), (int) ev.cmsswTracks_.size(), rec_algo_match,
           (int) m_sim_map.size(), (int) ev.simTracks_.size(),
           (int) m_seed_map.size(), (int) m_seeds.size(),
           (int) m_mkf_map.size(), (int) m_tracks.size()
    );
  }

  bool Shell::CheckMkFitLayerPlanVsReferenceHits(const Track &mkft, const Track &reft, const std::string &name) {
    // Check if all hit-layers of a reference track reft are in the mkfit layer plan.
    // Returns true if all layers are in the plan.
    // String name is printed in front of label, expected to be SIMK or CKF.
    const IterationConfig &itconf = Config::ItrInfo[m_it_index];
    auto lp = itconf.m_steering_params[ mkft.getEtaRegion() ].m_layer_plan;
    bool ret = true;
    for (int hi = 0; hi < reft.nTotalHits(); ++hi) {
      auto hot = reft.getHitOnTrack(hi);
      if (std::find_if(lp.begin(), lp.end(), [=](auto &x){ return x.m_layer == hot.layer; }) == lp.end())
      {
        printf("CheckMkfLayerPlanVsCkfHits: layer %d not in layer plan for region %d, %s label=%d\n",
                hot.layer, mkft.getEtaRegion(), name.c_str(), reft.label());
        ret = false;
      }
    }
    return ret;
  }

  //===========================================================================
  // Analysis drivers / main functions / Comparators
  //===========================================================================

  void Shell::Compare() {
    Event &ev = *m_event;
    const IterationConfig &itconf = Config::ItrInfo[m_it_index];

    FillByLabelMaps_CkfBase();

    printf("------------------------------------------------------\n");

    const bool print_all_def = false;
    int mkf_cnt=0, less_hits=0, more_hits=0;

    // TOBTEC: look for rec-seeds with hits in tob1 and 2 only.
    int tot_cnt = 0, no_mkf_cnt = 0;

    for (auto& [l, simt_ptr]: m_sim_map)
    {
      auto &simt = * simt_ptr;
      auto &ckft = * m_ckf_map[l];
      auto mi = m_mkf_map.find(l);

      bool print_all = print_all_def;

      // TOBTEC: look for rec-seeds with hits in tob1 and 2 only.
      bool select = true;
      {
        auto &ckf_seed = ev.seedTracks_[ckft.label()];
        for (int hi = 0; hi < ckf_seed.nTotalHits(); ++hi) {
          const HitOnTrack hot = ckf_seed.getHitOnTrack(hi);
          if (hot.index >= 0 && (hot.layer < 10 || hot.layer > 13)) {
            select = false;
            break;
          }
        }
      }
      if ( ! select) continue;

      ++tot_cnt;
      //print_all = true;

      if (mi != m_mkf_map.end())
      {
        auto &mkft = * mi->second;
        mkf_cnt++;
        if (mkft.nFoundHits() < ckft.nFoundHits()) ++less_hits;
        if (mkft.nFoundHits() > ckft.nFoundHits()) ++more_hits;

        CheckMkFitLayerPlanVsReferenceHits(mkft, ckft, "CKF");
        // CheckMkFitLayerPlanVsReferenceHits(mkft, simt, "SIM");

        (void) print_all;
        if (/* itconf.m_track_algorithm == 10 ||*/ print_all) {
          // ckf label is wrong when validation is on (even quality val) for mixedTriplet, pixelless and tobtec
          // as seed tracks get removed for non-mkfit iterations and indices from rec-tracks are no longer valid.
          auto &ckf_seed = ev.seedTracks_[ckft.label()];
          auto &mkf_seed = m_seeds[mkft.label()];
          print("ckf  ", 0, ckft, ev);
          print("mkfit", 0, mkft, ev);
          print("sim  ", 0, simt, ev);
          print("ckf seed", 0, ckf_seed, ev);
          print("mkf seed", 0, mkf_seed, ev);
          printf("------------------------------------------------------\n");

          TrackVec ssss;
          ssss.push_back(mkf_seed);

          IterationSeedPartition pppp(1);
          IterationConfig::get_seed_partitioner("phase1:1:debug")(Config::TrkInfo, ssss, *m_eoh, pppp);

          printf("------------------------------------------------------\n");
          printf("\n");
        }
      }
      else
      {
        printf("\n!!!!! No mkfit track with this label.\n\n");
        ++no_mkf_cnt;

        auto &ckf_seed = ev.seedTracks_[ckft.label()];
        print("ckf ", 0, ckft, ev);
        print("sim ", 0, simt, ev);
        print("ckf seed", 0, ckf_seed, ev);
        auto smi = m_seed_map.find(l);
        if (smi != m_seed_map.end())
          print("seed with matching label", 0, *smi->second, ev);
        printf("------------------------------------------------------\n");
      }
    }

    printf("mkFit found %d, matching_label=%d, less_hits=%d, more_hits=%d  [algo=%d (%s)]\n",
           (int) ev.fitTracks_.size(), mkf_cnt, less_hits, more_hits,
           itconf.m_track_algorithm, TrackBase::algoint_to_cstr(itconf.m_track_algorithm));

    if (tot_cnt > 0) {
      printf("\ntobtec tob1/2 tot=%d no_mkf=%d (%f%%)\n",
            tot_cnt, no_mkf_cnt, 100.0 * no_mkf_cnt / tot_cnt);
    } else {
      printf("\nNo CKF tracks with seed hits in TOB1/2 found (need iteration idx 8, TobTec?)\n");
    }

    printf("-------------------------------------------------------------------------------------------\n");
    printf("-------------------------------------------------------------------------------------------\n");
    printf("\n");
  }

  //===========================================================================
  // Seed study prototype
  //===========================================================================

  int Shell::select_seeds_for_algo(int algo, TrackVec &seeds) {
    int n_algo = 0; // seeds are grouped by algo
    for (auto &s : m_event->seedTracks_) {
      if (s.algoint() == algo) {
        seeds.push_back(s);
        ++n_algo;
      } else if (n_algo > 0)
        break;
    }
    return n_algo;
  }

  // SeedEntry, to be extended with parameter matching information
  struct SeedE { int index; float frac; };
  // SeedEntryVector
  // struct SeedEV {};
  using SeedEV = std::vector<SeedE>;
  // SeedQualities
  struct SeedQs {
    SeedEV v99, v80, v65, v50, v25;

    void add_seed(int idx, float frac) {
      if (frac > 0.99f)       v99.push_back({idx,frac});
      else if (frac >= 0.8f)  v80.push_back({idx,frac});
      else if (frac >= 0.65f) v65.push_back({idx,frac});
      else if (frac >= 0.5f)  v50.push_back({idx,frac});
      else if (frac >= 0.25f) v25.push_back({idx,frac});
    }

    SeedE best_seed() {
      if ( ! v99.empty()) return v99[0];
      if ( ! v80.empty()) return v80[0];
      if ( ! v65.empty()) return v65[0];
      if ( ! v50.empty()) return v50[0];
      if ( ! v25.empty()) return v25[0];
      return { -1, 0.0f };
    }
  };
  // SeedeXtendedInfo
  struct SeedXI {
    SeedQs orig;
    SeedQs clnd;
  };
  using SXIMap = std::map<int, SeedXI>;

  void Shell::StudySimAndSeeds() {
    std::unordered_map<int,int> seed_counts;
    auto frac_or_0 = [](int a, int b) -> double { return b ? ((double) a / b) : 0.0; };

    std::vector<int> algo_to_idx(end_algo, -1);
    for (int i = 0; i < Config::nItersCMSSW; ++i) {
      int a = algos[i];
      algo_to_idx[a] = i;
    }

    int n_sim   = m_event->simTracks_.size();
    int n_seed  = m_event->seedTracks_.size();
    int n_cmssw = m_event->cmsswTracks_.size();
    for (int si = 0; si < n_seed; ++si) {
      const Track &t = m_event->seedTracks_[si];
      ++seed_counts[ t.algoint() ];
    }
    int n_seed_sum = 0, n_seed_sum_post_clean = 0;
    std::set<int> algos_left;
    for (auto [a, v] : seed_counts) algos_left.insert(a);
    printf("Event %4d | N_sim = %5d | N_seed = %5d | N_cmssw = %5d |\n", m_event->evtID(), n_sim, n_seed, n_cmssw);
    printf("  Seeds by index / algo:\n");

    for (int i = 0; i < Config::nItersCMSSW; ++i) {
      int a = algos[i];
      n_seed_sum += seed_counts[a];
      algos_left.erase(a);

      const IterationConfig &itconf = Config::ItrInfo[i];
      assert(a == itconf.m_track_algorithm);
      m_seeds.clear();
      int ns2 = select_seeds_for_algo(itconf.m_track_algorithm, m_seeds);
      assert(ns2 == seed_counts[a]);

      SXIMap sxi_map;

      // Count number of good seeds for each track
      std::map<int, int> lbl_to_good_seed;
      int max_good_seeds = 0;
      for (int ti = 0; ti < (int) m_seeds.size(); ++ti) {
        Track &t = m_seeds[ti];
        auto sifh = m_event->simInfoForTrack(t, true);
        if (sifh.good_frac() >= 1.0f) {
          ++lbl_to_good_seed[t.label()];
          max_good_seeds = std::max(lbl_to_good_seed[t.label()], max_good_seeds);
        }
        if (sifh.label >= 0) {
          sxi_map[sifh.label].orig.add_seed(ti, sifh.good_frac());
        }
      }
      printf("    %d / %2d -> %5d (%5d, %2d)", i, a,
             seed_counts[a], (int) lbl_to_good_seed.size(), max_good_seeds);

      TrackVec orig_seeds = m_seeds;

      std::map<int, int> lbl_to_good_cl_seed;
      int max_good_cl_seeds = 0;
      if (itconf.m_seed_cleaner) {
        itconf.m_seed_cleaner(m_seeds, itconf, m_eoh->refBeamSpot());
        for (int ti = 0; ti < (int) m_seeds.size(); ++ti) {
          Track &t = m_seeds[ti];
          auto sifh = m_event->simInfoForTrack(t, true);
          if (sifh.good_frac() >= 1.0f) {
            ++lbl_to_good_cl_seed[t.label()];
            max_good_cl_seeds = std::max(lbl_to_good_cl_seed[t.label()], max_good_cl_seeds);
          }
          if (sifh.label >= 0) {
            sxi_map[sifh.label].clnd.add_seed(ti, sifh.good_frac());
          }
        }
        int ns_post_clean = m_seeds.size();
        n_seed_sum_post_clean += ns_post_clean;
        printf(" -> post-cleaning %5d (%5d, %2d) [%0.3f]",
              ns_post_clean, (int) lbl_to_good_cl_seed.size(), max_good_cl_seeds,
              frac_or_0(ns_post_clean, seed_counts[a]));
      }
      printf("\n");

      if ( ! itconf.m_seed_cleaner)
        continue;

      for (auto [lab, sxi] : sxi_map) {
        if ( ! sxi.orig.v99.empty() && sxi.clnd.v99.empty()) {
          Track &s = orig_seeds[sxi.orig.v99[0].index];
          printf("Lost 99p seed label %d, n_h=%d,  pt=%.3f, eta=%.3f\n", lab, s.nTotalHits(), s.pT(), s.momEta());
          auto se = sxi.clnd.best_seed();
          if (se.index >= 0) {
            Track &cs = m_seeds[se.index];
            printf("  Best cleaned n_h=%d, %.3f\n", cs.nTotalHits(), se.frac);
          }
        }
      }
    }
    if ( ! algos_left.empty()) {
      printf("  Additional algos, not in mkFit 10-index mapping\n");
      for (auto a : algos_left) {
        n_seed_sum += seed_counts[a];
        printf("        %2d -> %5d\n", a, seed_counts[a]);
      }
    }
    printf("  Total       %5d -> post-cleaning %5d [%0.3f]\n",
           n_seed_sum, n_seed_sum_post_clean, frac_or_0(n_seed_sum_post_clean, n_seed_sum));
  }

  void Shell::PreSelectSeeds(int iter_idx, Shell::seed_selector_func selector) {
    const IterationConfig &itconf = Config::ItrInfo[iter_idx];
    int a = algos[iter_idx];
    assert(a == itconf.m_track_algorithm);

    m_seeds.clear();
    select_seeds_for_algo(itconf.m_track_algorithm, m_seeds);
    if (itconf.m_seed_cleaner) {
      itconf.m_seed_cleaner(m_seeds, itconf, m_eoh->refBeamSpot());
    }

    // Select seeds with non-neg label and all good hits. Yay.
    TrackVec selected_seeds;
    for (int ti = 0; ti < (int) m_seeds.size(); ++ti) {
      Track &t = m_seeds[ti];
      auto sifh = m_event->simInfoForTrack(t, true);
      if (sifh.label >= 0 && sifh.good_frac() >= 1.0f && selector(t)) {
        selected_seeds.push_back(t);
      }
    }
    m_seeds.swap(selected_seeds);
    printf("Selected %d seeds.\n", (int) m_seeds.size());
  }

  void Shell::WriteSimTree() {
    TFile *F = TFile::Open("s.root", "RECREATE");
    TTree *T = new TTree("T", "mkfit sim-seed stuff");

    TrackVec * tvp = & m_event->simTracks_;
    TBranch *bv = T->Branch("s", tvp);

    const long long N_EVENTS = 10;
    for (int ev = 1; ev <= N_EVENTS; ++ev) {
        GoToEvent(ev);

        bv->Fill();
    }
    T->SetEntries();
    T->Write();
    F->Close();
    delete F;
  }

  void Shell::ReadSimTree() {
    TFile *F = TFile::Open("s.root", "READ");
    TTree *T = (TTree*) F->Get("T");

    TrackVec tv, *tvp = &tv;
    T->SetBranchAddress("s", &tvp);

    const long long N_EVENTS = 10;
    for (int ev = 1; ev <= N_EVENTS; ++ev) {
        T->GetEntry(ev - 1);
        printf("Ev:%d N_sim=%d\n", ev, (int) tv.size());
    }

    F->Close();
    delete F;
  }

  //===========================================================================
  // Visualization helpers
  //===========================================================================

#ifdef WITH_REVE

  void Shell::ShowTracker() {
    namespace REX = ROOT::Experimental;
    auto eveMng = REX::REveManager::Create();
    eveMng->AllowMultipleRemoteConnections(false, false);

    {
      REX::REveElement *holder = new REX::REveElement("Jets");

      int N_Jets = 4;
      TRandom &r = *gRandom;

      //const Double_t kR_min = 240;
      const Double_t kR_max = 250;
      const Double_t kZ_d   = 300;
      for (int i = 0; i < N_Jets; i++)
      {
          auto jet = new REX::REveJetCone(Form("Jet_%d",i ));
          jet->SetCylinder(2*kR_max, 2*kZ_d);
          jet->AddEllipticCone(r.Uniform(-0.5, 0.5), r.Uniform(0, TMath::TwoPi()),
                              0.1, 0.2);
          jet->SetFillColor(kRed + 4);
          jet->SetLineColor(kBlack);
          jet->SetMainTransparency(90);

          holder->AddElement(jet);
      }
      eveMng->GetEventScene()->AddElement(holder);
    }

    auto &ti = Config::TrkInfo;
    for (int l = 0; l < ti.n_layers(); ++l) {
      auto &li = ti[l];
      auto* bs = new REX::REveBoxSet(Form("Layer %d", l));
      bs->Reset(REX::REveBoxSet::kBT_InstancedScaledRotated, true, li.n_modules());
      bs->SetMainColorPtr(new Color_t);
      bs->UseSingleColor();
      if (li.is_pixel())
        bs->SetMainColor(li.is_barrel() ? kBlue - 3 : kCyan - 3);
      else
        bs->SetMainColor(li.is_barrel() ? kMagenta - 3 : kGreen - 3);

      float t[16];
      t[3] = t[7] = t[11] = 0;
      t[15] = 1;
      for (int m = 0; m < li.n_modules(); ++m) {
        auto &mi = li.module_info(m);
        auto &si = li.module_shape(mi.shapeid);

        auto &x = mi.xdir;
        t[0] = x[0] * si.dx1;
        t[1] = x[1] * si.dx1;
        t[2] = x[2] * si.dx1;
        auto y = mi.calc_ydir();
        t[4] = y[0] * si.dy;
        t[5] = y[1] * si.dy;
        t[6] = y[2] * si.dy;
        auto &z = mi.zdir;
        t[8] = z[0] * si.dz;
        t[9] = z[1] * si.dz;
        t[10] = z[2] * si.dz;
        auto &p = mi.pos;
        t[12] = p[0];
        t[13] = p[1];
        t[14] = p[2];

        bs->AddInstanceMat4(t);
      }
      bs->SetMainTransparency(60);
      bs->RefitPlex();

      eveMng->GetEventScene()->AddElement(bs);
    }
    eveMng->Show();
  }

#endif // WITH_REVE

}
