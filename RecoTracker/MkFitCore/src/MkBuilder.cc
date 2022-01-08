#include <memory>
#include <limits>

#include "RecoTracker/MkFitCore/interface/MkBuilder.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

#include "Pool.h"
#include "CandCloner.h"
#include "FindingFoos.h"
#include "MkFitter.h"
#include "MkFinder.h"

#include "Ice/IceRevisitedRadix.h"

//#define DEBUG
#include "Debug.h"

#include "tbb/parallel_for.h"
#include "tbb/parallel_for_each.h"

// Set this to select a single track for deep debugging:
//#define SELECT_SEED_LABEL -494

namespace mkfit {

  //==============================================================================
  // Execution context -- Pools of helper objects
  //==============================================================================

  struct ExecutionContext {
    ExecutionContext() = default;
    ~ExecutionContext() = default;

    Pool<CandCloner> m_cloners;
    Pool<MkFitter> m_fitters;
    Pool<MkFinder> m_finders;

    void populate(int n_thr) {
      m_cloners.populate(n_thr - m_cloners.size());
      m_fitters.populate(n_thr - m_fitters.size());
      m_finders.populate(n_thr - m_finders.size());
    }
  };

  ExecutionContext g_exe_ctx;

}  // end namespace mkfit

//------------------------------------------------------------------------------

#define CLONER(_n_) std::unique_ptr<CandCloner, decltype(retcand)> _n_(g_exe_ctx.m_cloners.GetFromPool(), retcand)
#define FITTER(_n_) std::unique_ptr<MkFitter, decltype(retfitr)> _n_(g_exe_ctx.m_fitters.GetFromPool(), retfitr)
#define FINDER(_n_) std::unique_ptr<MkFinder, decltype(retfndr)> _n_(g_exe_ctx.m_finders.GetFromPool(), retfndr)

namespace {
  using namespace mkfit;
  auto retcand = [](CandCloner *cloner) { g_exe_ctx.m_cloners.ReturnToPool(cloner); };
  [[maybe_unused]] auto retfitr = [](MkFitter *mkfttr) { g_exe_ctx.m_fitters.ReturnToPool(mkfttr); };
  auto retfndr = [](MkFinder *mkfndr) { g_exe_ctx.m_finders.ReturnToPool(mkfndr); };

  // Range of indices processed within one iteration of a TBB parallel_for.
  struct RangeOfSeedIndices {
    int m_rng_beg, m_rng_end;
    int m_beg, m_end;

    RangeOfSeedIndices(int rb, int re) : m_rng_beg(rb), m_rng_end(re) { reset(); }

    void reset() {
      m_end = m_rng_beg;
      next_chunk();
    }

    bool valid() const { return m_beg < m_rng_end; }

    int n_proc() const { return m_end - m_beg; }

    void next_chunk() {
      m_beg = m_end;
      m_end = std::min(m_end + NN, m_rng_end);
    }

    RangeOfSeedIndices &operator++() {
      next_chunk();
      return *this;
    }
  };

  // Region of seed indices processed in a single TBB parallel for.
  struct RegionOfSeedIndices {
    int m_reg_beg, m_reg_end, m_vec_cnt;

    RegionOfSeedIndices(const IntVec &seedEtaSeparators, int region) {
      m_reg_beg = (region == 0) ? 0 : seedEtaSeparators[region - 1];
      m_reg_end = seedEtaSeparators[region];
      m_vec_cnt = (m_reg_end - m_reg_beg + NN - 1) / NN;
    }

    int count() const { return m_reg_end - m_reg_beg; }

    tbb::blocked_range<int> tbb_blk_rng_std(int thr_hint = -1) const {
      if (thr_hint < 0)
        thr_hint = Config::numSeedsPerTask;
      return tbb::blocked_range<int>(m_reg_beg, m_reg_end, thr_hint);
    }

    tbb::blocked_range<int> tbb_blk_rng_vec() const {
      return tbb::blocked_range<int>(0, m_vec_cnt, std::max(1, Config::numSeedsPerTask / NN));
    }

    RangeOfSeedIndices seed_rng(const tbb::blocked_range<int> &i) const {
      return RangeOfSeedIndices(m_reg_beg + NN * i.begin(), std::min(m_reg_beg + NN * i.end(), m_reg_end));
    }
  };

#ifdef DEBUG
  void pre_prop_print(int ilay, MkBase *fir) {
    const float pt = 1.f / fir->getPar(0, 0, 3);
    std::cout << "propagate to lay=" << ilay << " start from x=" << fir->getPar(0, 0, 0)
              << " y=" << fir->getPar(0, 0, 1) << " z=" << fir->getPar(0, 0, 2)
              << " r=" << getHypot(fir->getPar(0, 0, 0), fir->getPar(0, 0, 1))
              << " px=" << pt * std::cos(fir->getPar(0, 0, 4)) << " py=" << pt * std::sin(fir->getPar(0, 0, 4))
              << " pz=" << pt / std::tan(fir->getPar(0, 0, 5)) << " pT=" << pt << std::endl;
  }

  void post_prop_print(int ilay, MkBase *fir) {
    std::cout << "propagate to lay=" << ilay << " arrive at x=" << fir->getPar(0, 1, 0) << " y=" << fir->getPar(0, 1, 1)
              << " z=" << fir->getPar(0, 1, 2) << " r=" << getHypot(fir->getPar(0, 1, 0), fir->getPar(0, 1, 1))
              << std::endl;
  }

  void print_seed(const Track &seed) {
    std::cout << "MX - found seed with label=" << seed.label() << " nHits=" << seed.nFoundHits()
              << " chi2=" << seed.chi2() << " posEta=" << seed.posEta() << " posPhi=" << seed.posPhi()
              << " posR=" << seed.posR() << " posZ=" << seed.z() << " pT=" << seed.pT() << std::endl;
  }

  void print_seed2(const TrackCand &seed) {
    std::cout << "MX - found seed with nFoundHits=" << seed.nFoundHits() << " chi2=" << seed.chi2() << " x=" << seed.x()
              << " y=" << seed.y() << " z=" << seed.z() << " px=" << seed.px() << " py=" << seed.py()
              << " pz=" << seed.pz() << " pT=" << seed.pT() << std::endl;
  }

  void print_seeds(const TrackVec &seeds) {
    std::cout << "found total seeds=" << seeds.size() << std::endl;
    for (auto &&seed : seeds) {
      print_seed(seed);
    }
  }

  void print_seeds(const EventOfCombCandidates &event_of_comb_cands) {
    for (int iseed = 0; iseed < event_of_comb_cands.m_size; iseed++) {
      print_seed2(event_of_comb_cands.m_candidates[iseed].front());
    }
  }
#endif

  bool sortCandByScore(const TrackCand &cand1, const TrackCand &cand2) {
    return mkfit::sortByScoreTrackCand(cand1, cand2);
  }

}  // end unnamed namespace

//------------------------------------------------------------------------------
// Constructor and destructor
//------------------------------------------------------------------------------

namespace mkfit {

  MkBuilder *MkBuilder::make_builder() { return new MkBuilder; }

  void MkBuilder::populate() { g_exe_ctx.populate(Config::numThreadsFinder); }

  MkBuilder::MkBuilder() {}

  MkBuilder::~MkBuilder() {}

  //------------------------------------------------------------------------------
  // Common functions
  //------------------------------------------------------------------------------

  void MkBuilder::begin_event(MkJob *job, Event *ev, const char *build_type) {
    m_nan_n_silly_per_layer_count = 0;

    m_job = job;
    m_event = ev;

    m_seedEtaSeparators.resize(m_job->num_regions());
    m_seedMinLastLayer.resize(m_job->num_regions());
    m_seedMaxLastLayer.resize(m_job->num_regions());

    for (int i = 0; i < m_job->num_regions(); ++i) {
      m_seedEtaSeparators[i] = 0;
      m_seedMinLastLayer[i] = 9999;
      m_seedMaxLastLayer[i] = 0;
    }

    if (!Config::silent) {
      std::cout << "MkBuilder building tracks with '" << build_type << "'"
                << ", iteration_index=" << job->m_iter_config.m_iteration_index
                << ", track_algorithm=" << job->m_iter_config.m_track_algorithm << std::endl;
    }
  }

  void MkBuilder::end_event() {
    m_job = nullptr;
    m_event = nullptr;
  }

  void MkBuilder::release_memory() {
    TrackVec tmp;
    m_tracks.swap(tmp);
    m_event_of_comb_cands.ReleaseMemory();
  }

  void MkBuilder::import_seeds(const TrackVec &in_seeds, std::function<insert_seed_foo> insert_seed) {
    // bool debug = true;

    const int size = in_seeds.size();

    IterationSeedPartition part(size);

    m_job->m_iter_config.m_partition_seeds(m_job->m_trk_info, in_seeds, m_job->m_event_of_hits, part);

    RadixSort radix;
    radix.Sort(&part.m_sort_score[0], size);

    for (int i = 0; i < size; ++i) {
      int j = radix.GetRanks()[i];

      const Track &S = in_seeds[j];
      HitOnTrack hot = S.getLastHitOnTrack();

      int reg = part.m_region[j];

      ++m_seedEtaSeparators[reg];

      m_seedMinLastLayer[reg] = std::min(m_seedMinLastLayer[reg], hot.layer);
      m_seedMaxLastLayer[reg] = std::max(m_seedMaxLastLayer[reg], hot.layer);

      // MT-OPTIMIZE-MEM: should delay this and pool-allocate memory for HoTs once
      // population for each region is known.
      insert_seed(S, reg);
    }

    for (int i = 0; i < m_job->num_regions(); ++i) {
      if (m_seedMinLastLayer[i] == 9999)
        m_seedMinLastLayer[i] = -1;
      if (m_seedMaxLastLayer[i] == 0)
        m_seedMaxLastLayer[i] = -1;
    }

    // MIMI -- we in principle support any number of regions now.
    dprintf(
        "MkBuilder::import_seeds finished import of %d seeds (last seeding layer min, max):\n"
        "  ec- = %d(%d,%d), t- = %d(%d,%d), brl = %d(%d,%d), t+ = %d(%d,%d), ec+ = %d(%d,%d).\n",
        size,
        m_seedEtaSeparators[0],
        m_seedMinLastLayer[0],
        m_seedMaxLastLayer[0],
        m_seedEtaSeparators[1],
        m_seedMinLastLayer[1],
        m_seedMaxLastLayer[1],
        m_seedEtaSeparators[2],
        m_seedMinLastLayer[2],
        m_seedMaxLastLayer[2],
        m_seedEtaSeparators[3],
        m_seedMinLastLayer[3],
        m_seedMaxLastLayer[3],
        m_seedEtaSeparators[4],
        m_seedMinLastLayer[4],
        m_seedMaxLastLayer[4]);

    // Sum up region counts to contain actual separator indices, fix min/max layers.
    for (int i = 1; i < m_job->num_regions(); ++i) {
      m_seedEtaSeparators[i] += m_seedEtaSeparators[i - 1];
    }

    // MT-OPTIMIZE-MEM: Allocate mem, assign to CombCands as per region.
    // Reset() that is called in find_tracks_load_seeds() should be called here with additional
    // information from this seed partition and max expected num-of hits in given iteration / region.
    // Or, at least estimated from N_max_cands and N_layers in given eta-region.

    //dump seeds
    dcall(print_seeds(m_event_of_comb_cands));
  }

  //------------------------------------------------------------------------------

  int MkBuilder::filter_comb_cands(std::function<filter_track_cand_foo> filter) {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    int i = 0, place_pos = 0;

    // printf ("MkBuilder::filter_comb_cands Entering filter size eoccsm_size=%d\n", eoccs.m_size);

    IntVec removed_cnts(m_job->num_regions());
    while (i < eoccs.m_size) {
      if (filter(eoccs[i].front())) {
        if (place_pos != i)
          std::swap(eoccs[place_pos], eoccs[i]);
        ++place_pos;
      } else {
        assert(eoccs[i].front().getEtaRegion() < m_job->num_regions());
        ++removed_cnts[eoccs[i].front().getEtaRegion()];
      }
      ++i;
    }

    int n_removed = 0;
    for (int reg = 0; reg < m_job->num_regions(); ++reg) {
      // printf ("MkBuilder::filter_comb_cands reg=%d: n_rem_was=%d removed_in_r=%d n_rem=%d, es_was=%d es_new=%d\n",
      //         reg, n_removed, removed_cnts[reg], n_removed + removed_cnts[reg],
      //         m_seedEtaSeparators[reg], m_seedEtaSeparators[reg] - n_removed - removed_cnts[reg]);

      n_removed += removed_cnts[reg];
      m_seedEtaSeparators[reg] -= n_removed;
    }

    eoccs.ResizeAfterFiltering(n_removed);

    // printf ("MkBuilder::filter_comb_cands n_removed = %d, eoccsm_size=%d\n", n_removed, eoccs.m_size);

    return n_removed;
  }

  void MkBuilder::find_min_max_hots_size() {
    const EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    int min[5], max[5], gmin = 0, gmax = 0;
    int i = 0;
    for (int reg = 0; reg < 5; ++reg) {
      min[reg] = 9999;
      max[reg] = 0;
      for (; i < m_seedEtaSeparators[reg]; i++) {
        min[reg] = std::min(min[reg], eoccs[i].m_hots_size);
        max[reg] = std::max(max[reg], eoccs[i].m_hots_size);
      }
      gmin = std::max(gmin, min[reg]);
      gmax = std::max(gmax, max[reg]);
    }
    printf(
        "MkBuilder::find_min_max_hots_size MIN %3d -- [ %3d | %3d | %3d | %3d | %3d ]   MAX %3d -- [ %3d | %3d | %3d | "
        "%3d | %3d ]\n",
        gmin,
        min[0],
        min[1],
        min[2],
        min[3],
        min[4],
        gmax,
        max[0],
        max[1],
        max[2],
        max[3],
        max[4]);
  }

  void MkBuilder::select_best_comb_cands(bool clear_m_tracks, bool remove_missing_hits) {
    if (clear_m_tracks)
      m_tracks.clear();
    export_best_comb_cands(m_tracks, remove_missing_hits);
  }

  void MkBuilder::export_best_comb_cands(TrackVec &out_vec, bool remove_missing_hits) {
    const EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    out_vec.reserve(out_vec.size() + eoccs.m_size);
    for (int i = 0; i < eoccs.m_size; i++) {
      // See MT-RATS comment below.
      assert(!eoccs[i].empty() && "BackwardFitBH requires output tracks to align with seeds.");

      // Take the first candidate, if it exists.
      if (!eoccs[i].empty()) {
        const TrackCand &bcand = eoccs[i].front();
        out_vec.emplace_back(bcand.exportTrack(remove_missing_hits));
      }
    }
  }

  void MkBuilder::export_tracks(TrackVec &out_vec) {
    out_vec.reserve(out_vec.size() + m_tracks.size());
    for (auto &t : m_tracks) {
      out_vec.emplace_back(t);
    }
  }

  //------------------------------------------------------------------------------
  // PrepareSeeds
  //------------------------------------------------------------------------------

  void MkBuilder::seed_post_cleaning(TrackVec &tv) {
#ifdef SELECT_SEED_LABEL
    {  // Select seed with the defined label for detailed debugging.
      for (int i = 0; i < (int)tv.size(); ++i) {
        if (tv[i].label() == SELECT_SEED_LABEL) {
          printf("Preselect seed with label %d - found on pos %d\n", SELECT_SEED_LABEL, i);
          if (i != 0)
            tv[0] = tv[i];
          tv.resize(1);
          print("Label", tv[0].label(), tv[0], true);
          break;
        }
      }
      if (tv.size() != 1) {
        printf("Preselect seed with label %d - NOT FOUND. Cleaning out seeds.\n", SELECT_SEED_LABEL);
        tv.clear();
      }
    }
#endif

    if (Config::nan_n_silly_check_seeds) {
      int count = 0;

      for (int i = 0; i < (int)tv.size(); ++i) {
        bool silly = tv[i].hasSillyValues(Config::nan_n_silly_print_bad_seeds,
                                          Config::nan_n_silly_fixup_bad_seeds,
                                          "Post-cleaning seed silly value check and fix");
        if (silly) {
          ++count;
          if (Config::nan_n_silly_remove_bad_seeds) {
            // XXXX MT
            // Could do somethin smarter here: setStopped ?  check in seed cleaning ?
            tv.erase(tv.begin() + i);
            --i;
          }
        }
      }

      if (count > 0 && !Config::silent) {
        printf("Nan'n'Silly detected %d silly seeds (fix=%d, remove=%d).\n",
               count,
               Config::nan_n_silly_fixup_bad_seeds,
               Config::nan_n_silly_remove_bad_seeds);
      }
    }
  }

  //------------------------------------------------------------------------------
  // FindTracksBestHit
  //------------------------------------------------------------------------------

  void MkBuilder::find_tracks_load_seeds_BH(const TrackVec &in_seeds) {
    // bool debug = true;

    m_tracks.reserve(in_seeds.size());
    m_tracks.clear();

    import_seeds(in_seeds, [&](const Track &seed, int region) {
      m_tracks.push_back(seed);
      m_tracks.back().setNSeedHits(seed.nTotalHits());
      m_tracks.back().setEtaRegion(region);
    });

    //dump seeds
    dcall(print_seeds(m_tracks));
  }

  void MkBuilder::FindTracksBestHit(SteeringParams::IterationType_e iteration_dir) {
    // bool debug = true;

    TrackVec &cands = m_tracks;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      if (iteration_dir == SteeringParams::IT_BkwSearch && !m_job->steering_params(region).has_bksearch_plan()) {
        printf("No backward search plan for region %d\n", region);
        return;
      }

      // XXXXXX Select endcap / barrel only ...
      // if (region != TrackerInfo::Reg_Endcap_Neg && region != TrackerInfo::Reg_Endcap_Pos)
      // if (region != TrackerInfo::Reg_Barrel)
      //   return;

      const SteeringParams &st_par = m_job->steering_params(region);
      const TrackerInfo &trk_info = m_job->m_trk_info;

      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      tbb::parallel_for(rosi.tbb_blk_rng_vec(), [&](const tbb::blocked_range<int> &blk_rng) {
        FINDER(mkfndr);

        RangeOfSeedIndices rng = rosi.seed_rng(blk_rng);

        std::vector<int> trk_idcs(NN);  // track indices in Matriplex
        std::vector<int> trk_llay(NN);  // last layer on input track

        while (rng.valid()) {
          dprint(std::endl << "processing track=" << rng.m_beg << ", label=" << cands[rng.m_beg].label());

          int prev_layer = 9999;

          for (int i = rng.m_beg, ii = 0; i < rng.m_end; ++i, ++ii) {
            int llay = cands[i].getLastHitLyr();
            trk_llay[ii] = llay;
            prev_layer = std::min(prev_layer, llay);

            dprintf("  %2d %2d %2d lay=%3d prev_layer=%d\n", ii, i, cands[i].label(), llay, prev_layer);
          }
          int curr_tridx = 0;

          auto layer_plan_it = st_par.make_iterator(iteration_dir);

          dprintf("Made iterator for %d, first layer=%d ... end layer=%d\n",
                  iteration_dir,
                  layer_plan_it.layer(),
                  layer_plan_it.last_layer());

          assert(layer_plan_it.is_pickup_only());

          int curr_layer = layer_plan_it.layer();

          mkfndr->Stopped.SetVal(0);

          // Loop over layers, starting from after the seed.
          // Consider inverting loop order and make layer outer, need to
          // trade off hit prefetching with copy-out of candidates.
          while (++layer_plan_it) {
            prev_layer = curr_layer;
            curr_layer = layer_plan_it.layer();
            mkfndr->Setup(m_job->m_iter_config.m_params,
                          m_job->m_iter_config.m_layer_configs[curr_layer],
                          m_job->get_mask_for_layer(curr_layer));

            dprint("at layer " << curr_layer);
            const LayerOfHits &layer_of_hits = m_job->m_event_of_hits.m_layers_of_hits[curr_layer];
            const LayerInfo &layer_info = trk_info.m_layers[curr_layer];
            const FindingFoos &fnd_foos = FindingFoos::get_finding_foos(layer_info.is_barrel());

            // Pick up seeds that become active on current layer -- unless already fully loaded.
            if (curr_tridx < rng.n_proc()) {
              int prev_tridx = curr_tridx;

              for (int i = rng.m_beg, ii = 0; i < rng.m_end; ++i, ++ii) {
                if (trk_llay[ii] == prev_layer)
                  trk_idcs[curr_tridx++] = i;
              }
              if (curr_tridx > prev_tridx) {
                dprintf("added %d seeds, started with %d\n", curr_tridx - prev_tridx, prev_tridx);

                mkfndr->InputTracksAndHitIdx(cands, trk_idcs, prev_tridx, curr_tridx, false, prev_tridx);
              }
            }

            if (layer_plan_it.is_pickup_only())
              continue;

            dcall(pre_prop_print(curr_layer, mkfndr.get()));

            (mkfndr.get()->*fnd_foos.m_propagate_foo)(
                layer_info.m_propagate_to, curr_tridx, Config::finding_inter_layer_pflags);

            dcall(post_prop_print(curr_layer, mkfndr.get()));

            mkfndr->SelectHitIndices(layer_of_hits, curr_tridx);

            // Stop low-pT tracks that can not reach the current barrel layer.
            if (layer_info.is_barrel()) {
              const float r_min_sqr = layer_info.m_rin * layer_info.m_rin;
              for (int i = 0; i < curr_tridx; ++i) {
                if (!mkfndr->Stopped[i]) {
                  if (mkfndr->RadiusSqr(i, MkBase::iP) < r_min_sqr) {
                    if (region == TrackerInfo::Reg_Barrel) {
                      mkfndr->Stopped[i] = 1;
                      mkfndr->OutputTrackAndHitIdx(cands[rng.m_beg + i], i, false);
                    }
                    mkfndr->XWsrResult[i].m_wsr = WSR_Outside;
                    mkfndr->XHitSize[i] = 0;
                  }
                } else {  // make sure we don't add extra work for AddBestHit
                  mkfndr->XWsrResult[i].m_wsr = WSR_Outside;
                  mkfndr->XHitSize[i] = 0;
                }
              }
            }

            // make candidates with best hit
            dprint("make new candidates");

            mkfndr->AddBestHit(layer_of_hits, curr_tridx, fnd_foos);

            // Stop tracks that have reached N_max_holes.
            for (int i = 0; i < curr_tridx; ++i) {
              if (!mkfndr->Stopped[i] && mkfndr->BestHitLastHoT(i).index == -2) {
                mkfndr->Stopped[i] = 1;
                mkfndr->OutputTrackAndHitIdx(cands[rng.m_beg + i], i, false);
              }
            }

          }  // end of layer loop

          mkfndr->OutputNonStoppedTracksAndHitIdx(cands, trk_idcs, 0, curr_tridx, false);

          ++rng;
        }  // end of loop over candidates in a tbb chunk
      });  // end parallel_for over candidates in a region
    });    // end of parallel_for_each over regions
  }

  //------------------------------------------------------------------------------
  // FindTracksCombinatorial: Standard TBB and CloneEngine TBB
  //------------------------------------------------------------------------------

  void MkBuilder::find_tracks_load_seeds(const TrackVec &in_seeds) {
    // This will sort seeds according to iteration configuration.

    // m_tracks can be used for BkFit.
    m_tracks.clear();

    m_event_of_comb_cands.Reset((int)in_seeds.size(), m_job->max_max_cands());

    import_seeds(in_seeds, [&](const Track &seed, int region) { m_event_of_comb_cands.InsertSeed(seed, region); });
  }

  int MkBuilder::find_tracks_unroll_candidates(std::vector<std::pair<int, int>> &seed_cand_vec,
                                               int start_seed,
                                               int end_seed,
                                               int layer,
                                               int prev_layer,
                                               bool pickup_only,
                                               SteeringParams::IterationType_e iteration_dir) {
    int silly_count = 0;

    seed_cand_vec.clear();

    for (int iseed = start_seed; iseed < end_seed; ++iseed) {
      CombCandidate &ccand = m_event_of_comb_cands[iseed];

      if (ccand.m_state == CombCandidate::Dormant && ccand.m_pickup_layer == prev_layer) {
        ccand.m_state = CombCandidate::Finding;
      }
      if (!pickup_only && ccand.m_state == CombCandidate::Finding) {
        bool active = false;
        for (int ic = 0; ic < (int)ccand.size(); ++ic) {
          if (ccand[ic].getLastHitIdx() != -2) {
            // Check if the candidate is close to it's max_r, pi/2 - 0.2 rad (11.5 deg)
            if (iteration_dir == SteeringParams::IT_FwdSearch && ccand[ic].pT() < 1.2) {
              const float dphi = std::abs(ccand[ic].posPhi() - ccand[ic].momPhi());
              if (ccand[ic].posRsq() > 625.f && dphi > 1.371f && dphi < 4.512f) {
                // printf("Stopping cand at r=%f, posPhi=%.1f momPhi=%.2f pt=%.2f emomEta=%.2f\n",
                //        ccand[ic].posR(), ccand[ic].posPhi(), ccand[ic].momPhi(), ccand[ic].pT(), ccand[ic].momEta());
                ccand[ic].addHitIdx(-2, layer, 0.0f);
                continue;
              }
            }

            active = true;
            seed_cand_vec.push_back(std::pair<int, int>(iseed, ic));
            ccand[ic].resetOverlaps();

            if (Config::nan_n_silly_check_cands_every_layer) {
              if (ccand[ic].hasSillyValues(Config::nan_n_silly_print_bad_cands_every_layer,
                                           Config::nan_n_silly_fixup_bad_cands_every_layer,
                                           "Per layer silly check"))
                ++silly_count;
            }
          }
        }
        if (!active) {
          ccand.m_state = CombCandidate::Finished;
        }
      }
    }

    if (Config::nan_n_silly_check_cands_every_layer && silly_count > 0) {
      m_nan_n_silly_per_layer_count += silly_count;
    }

    return seed_cand_vec.size();
  }

  void MkBuilder::find_tracks_handle_missed_layers(MkFinder *mkfndr,
                                                   const LayerInfo &layer_info,
                                                   std::vector<std::vector<TrackCand>> &tmp_cands,
                                                   const std::vector<std::pair<int, int>> &seed_cand_idx,
                                                   const int region,
                                                   const int start_seed,
                                                   const int itrack,
                                                   const int end) {
    // XXXX-1 If I miss a layer, insert the original track into tmp_cands
    // AND do not do it in FindCandidates as the position can be badly
    // screwed by then. See comment there, too.
    // One could also do a pre-check ... so as not to use up a slot.

    // bool debug = true;

    for (int ti = itrack; ti < end; ++ti) {
      TrackCand &cand = m_event_of_comb_cands.m_candidates[seed_cand_idx[ti].first][seed_cand_idx[ti].second];
      WSR_Result &w = mkfndr->XWsrResult[ti - itrack];

      // XXXX-4 Low pT tracks can miss a barrel layer ... and should be stopped
      const float cand_r =
          std::hypot(mkfndr->getPar(ti - itrack, MkBase::iP, 0), mkfndr->getPar(ti - itrack, MkBase::iP, 1));

      dprintf("WSR Check label %d, seed %d, cand %d score %f -> wsr %d, in_gap %d\n",
              cand.label(),
              seed_cand_idx[ti].first,
              seed_cand_idx[ti].second,
              cand.score(),
              w.m_wsr,
              w.m_in_gap);

      if (layer_info.is_barrel() && cand_r < layer_info.m_rin) {
        // Fake outside so it does not get processed in FindTracks Std/CE... and
        // create a stopped replica in barrel and original copy if there is
        // still chance to hit endcaps.
        dprintf("Barrel cand propagated to r=%f ... layer is %f - %f\n", cand_r, layer_info.m_rin, layer_info.m_rout);

        mkfndr->XHitSize[ti - itrack] = 0;
        w.m_wsr = WSR_Outside;

        tmp_cands[seed_cand_idx[ti].first - start_seed].push_back(cand);
        if (region == TrackerInfo::Reg_Barrel) {
          dprintf(" creating extra stopped held back candidate\n");
          tmp_cands[seed_cand_idx[ti].first - start_seed].back().addHitIdx(-2, layer_info.m_layer_id, 0);
        }
      } else if (w.m_wsr == WSR_Outside) {
        dprintf(" creating extra held back candidate\n");
        tmp_cands[seed_cand_idx[ti].first - start_seed].push_back(cand);

        // This can fire for Standard finding when candidates from a given seed are
        // split between two iterations of the vecotrized loop over seeds as the
        // output vector is shared between finding and the outside xtras here.
        // if (tmp_cands[seed_cand_idx[ti].first - start_seed].size() > m_event_of_comb_cands.m_candidates[seed_cand_idx[ti].first].size())
        //   printf("XXXXXXX %d %d for seed %d, index %d\n", (int) tmp_cands[seed_cand_idx[ti].first - start_seed].size(), (int) m_event_of_comb_cands.m_candidates[seed_cand_idx[ti].first].size(),
        //          seed_cand_idx[ti].first, seed_cand_idx[ti].first - start_seed);
      } else if (w.m_wsr == WSR_Edge) {
        // XXXX-2 Additionally, if I miss/hit by epsilon, here would be a
        // good place to clone the extra track that goes straight into
        // the "other" sub-section - ecap/brl (for the transition
        // region). Here the jump-to-layer-on-miss I wanted to add to
        // LayerInfo will be needed.
        // And, stop the track if told to do so !!!
      }

      // XXXX-3 mind the gap - as in w.m_in_gap !!!!
      // Those don't really need to be taken out ... but think it through.
    }
  }

  //------------------------------------------------------------------------------
  // FindTracksCombinatorial: Standard TBB
  //------------------------------------------------------------------------------

  void MkBuilder::FindTracksStandard(SteeringParams::IterationType_e iteration_dir) {
    // debug = true;

    EventOfCombCandidates &eoccs = m_event_of_comb_cands;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      if (iteration_dir == SteeringParams::IT_BkwSearch && !m_job->steering_params(region).has_bksearch_plan()) {
        printf("No backward search plan for region %d\n", region);
        return;
      }

      const TrackerInfo &trk_info = m_job->m_trk_info;
      const SteeringParams &st_par = m_job->steering_params(region);
      const IterationParams &params = m_job->params();

      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      // adaptive seeds per task based on the total estimated amount of work to divide among all threads
      const int adaptiveSPT =
          clamp(Config::numThreadsEvents * eoccs.m_size / Config::numThreadsFinder + 1, 4, Config::numSeedsPerTask);
      dprint("adaptiveSPT " << adaptiveSPT << " fill " << rosi.count() << "/" << eoccs.m_size << " region " << region);

      // loop over seeds
      tbb::parallel_for(rosi.tbb_blk_rng_std(adaptiveSPT), [&](const tbb::blocked_range<int> &seeds) {
        FINDER(mkfndr);

        const int start_seed = seeds.begin();
        const int end_seed = seeds.end();
        const int n_seeds = end_seed - start_seed;

        std::vector<std::vector<TrackCand>> tmp_cands(n_seeds);
        for (size_t iseed = 0; iseed < tmp_cands.size(); ++iseed) {
          tmp_cands[iseed].reserve(2 * params.maxCandsPerSeed);  //factor 2 seems reasonable to start with
        }

        std::vector<std::pair<int, int>> seed_cand_idx;
        seed_cand_idx.reserve(n_seeds * params.maxCandsPerSeed);

        auto layer_plan_it = st_par.make_iterator(iteration_dir);

        dprintf("Made iterator for %d, first layer=%d ... end layer=%d\n",
                iteration_dir,
                layer_plan_it.layer(),
                layer_plan_it.last_layer());

        assert(layer_plan_it.is_pickup_only());

        int curr_layer = layer_plan_it.layer(), prev_layer;

        dprintf("\nMkBuilder::FindTracksStandard region=%d, seed_pickup_layer=%d, first_layer=%d\n",
                region,
                curr_layer,
                layer_plan_it.next_layer());

        auto &iter_params = (iteration_dir == SteeringParams::IT_BkwSearch) ? m_job->m_iter_config.m_backward_params
                                                                            : m_job->m_iter_config.m_params;

        // Loop over layers, starting from after the seed.
        while (++layer_plan_it) {
          prev_layer = curr_layer;
          curr_layer = layer_plan_it.layer();
          mkfndr->Setup(
              iter_params, m_job->m_iter_config.m_layer_configs[curr_layer], m_job->get_mask_for_layer(curr_layer));

          dprintf("\n* Processing layer %d\n", curr_layer);

          const LayerOfHits &layer_of_hits = m_job->m_event_of_hits.m_layers_of_hits[curr_layer];
          const LayerInfo &layer_info = trk_info.m_layers[curr_layer];
          const FindingFoos &fnd_foos = FindingFoos::get_finding_foos(layer_info.is_barrel());

          int theEndCand = find_tracks_unroll_candidates(seed_cand_idx,
                                                         start_seed,
                                                         end_seed,
                                                         curr_layer,
                                                         prev_layer,
                                                         layer_plan_it.is_pickup_only(),
                                                         iteration_dir);

          if (layer_plan_it.is_pickup_only() || theEndCand == 0)
            continue;

          // vectorized loop
          for (int itrack = 0; itrack < theEndCand; itrack += NN) {
            int end = std::min(itrack + NN, theEndCand);

            dprint("processing track="
                   << itrack << ", label="
                   << eoccs.m_candidates[seed_cand_idx[itrack].first][seed_cand_idx[itrack].second].label());

            //fixme find a way to deal only with the candidates needed in this thread
            mkfndr->InputTracksAndHitIdx(eoccs.m_candidates, seed_cand_idx, itrack, end, false);

            //propagate to layer
            dcall(pre_prop_print(curr_layer, mkfndr.get()));

            (mkfndr.get()->*fnd_foos.m_propagate_foo)(
                layer_info.m_propagate_to, end - itrack, Config::finding_inter_layer_pflags);

            dcall(post_prop_print(curr_layer, mkfndr.get()));

            dprint("now get hit range");
            mkfndr->SelectHitIndices(layer_of_hits, end - itrack);

            find_tracks_handle_missed_layers(
                mkfndr.get(), layer_info, tmp_cands, seed_cand_idx, region, start_seed, itrack, end);

            // if(Config::dumpForPlots) {
            //std::cout << "MX number of hits in window in layer " << curr_layer << " is " <<  mkfndr->getXHitEnd(0, 0, 0)-mkfndr->getXHitBegin(0, 0, 0) << std::endl;
            //}

            dprint("make new candidates");
            mkfndr->FindCandidates(layer_of_hits, tmp_cands, start_seed, end - itrack, fnd_foos);

          }  //end of vectorized loop

          // sort the input candidates
          for (int is = 0; is < n_seeds; ++is) {
            dprint("dump seed n " << is << " with N_input_candidates=" << tmp_cands[is].size());

            std::sort(tmp_cands[is].begin(), tmp_cands[is].end(), sortCandByScore);
          }

          // now fill out the output candidates
          for (int is = 0; is < n_seeds; ++is) {
            if (!tmp_cands[is].empty()) {
              eoccs[start_seed + is].clear();

              // Put good candidates into eoccs, process -2 candidates.
              int n_placed = 0;
              bool first_short = true;
              for (int ii = 0; ii < (int)tmp_cands[is].size() && n_placed < params.maxCandsPerSeed; ++ii) {
                TrackCand &tc = tmp_cands[is][ii];

                // See if we have an overlap hit available, but only if we have a true hit in this layer
                // and pT is above the pTCutOverlap
                if (tc.pT() > params.pTCutOverlap && tc.getLastHitLyr() == curr_layer && tc.getLastHitIdx() >= 0) {
                  CombCandidate &ccand = eoccs[start_seed + is];

                  HitMatch *hm = ccand[tc.originIndex()].findOverlap(
                      tc.getLastHitIdx(), layer_of_hits.GetHit(tc.getLastHitIdx()).detIDinLayer());

                  if (hm) {
                    tc.addHitIdx(hm->m_hit_idx, curr_layer, hm->m_chi2);
                    tc.incOverlapCount();

                    // --- ROOT text tree dump of all found overlaps
                    // static bool first = true;
                    // if (first)
                    // {
                    //   // ./mkFit ... | perl -ne 'if (/^ZZZ_EXTRA/) { s/^ZZZ_EXTRA //og; print; }' > extra.rtt
                    //   printf("ZZZ_EXTRA label/I:can_idx/I:layer/I:pt/F:eta/F:phi/F:"
                    //          "chi2/F:chi2_extra/F:module/I:module_extra/I:extra_label/I\n");
                    //   first = false;
                    // }

                    // const Hit       &h    = layer_of_hits.GetHit(tc.getLastHitIdx());
                    // const MCHitInfo &mchi = m_event->simHitsInfo_[h.mcHitID()];
                    // // label/I:can_idx/I:layer/I:pt/F:eta/F:phi/F:chi2_orig/F:chi2/F:chi2_extra/F:module/I:module_extra/I
                    // printf("ZZZ_EXTRA %d %d %d %f %f %f %f %f %u %u %d\n",
                    //        tc.label(), tc.originIndex(), curr_layer, tc.pT(), tc.posEta(), tc.posPhi(),
                    //        tc.chi2(), hm->m_chi2, layer_of_hits.GetHit(tc.getLastHitIdx()).detIDinLayer(), hm->m_module_id, mchi.mcTrackID());
                  }
                }

                if (tc.getLastHitIdx() != -2) {
                  eoccs[start_seed + is].emplace_back(tc);
                  ++n_placed;
                } else if (first_short) {
                  first_short = false;
                  if (tc.score() > eoccs[start_seed + is].m_best_short_cand.score()) {
                    eoccs[start_seed + is].m_best_short_cand = tc;
                  }
                }
              }

              tmp_cands[is].clear();
            }
          }

        }  // end of layer loop

        // final sorting
        for (int iseed = start_seed; iseed < end_seed; ++iseed) {
          eoccs[iseed].MergeCandsAndBestShortOne(m_job->params(), true, true);
        }
      });  // end parallel-for over chunk of seeds within region
    });    // end of parallel-for-each over eta regions

    // debug = false;
  }

  //------------------------------------------------------------------------------
  // FindTracksCombinatorial: CloneEngine TBB
  //------------------------------------------------------------------------------

  void MkBuilder::FindTracksCloneEngine(SteeringParams::IterationType_e iteration_dir) {
    // debug = true;

    EventOfCombCandidates &eoccs = m_event_of_comb_cands;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      if (iteration_dir == SteeringParams::IT_BkwSearch && !m_job->steering_params(region).has_bksearch_plan()) {
        printf("No backward search plan for region %d\n", region);
        return;
      }

      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      // adaptive seeds per task based on the total estimated amount of work to divide among all threads
      const int adaptiveSPT =
          clamp(Config::numThreadsEvents * eoccs.m_size / Config::numThreadsFinder + 1, 4, Config::numSeedsPerTask);
      dprint("adaptiveSPT " << adaptiveSPT << " fill " << rosi.count() << "/" << eoccs.m_size << " region " << region);

      tbb::parallel_for(rosi.tbb_blk_rng_std(adaptiveSPT), [&](const tbb::blocked_range<int> &seeds) {
        CLONER(cloner);
        FINDER(mkfndr);

        cloner->Setup(m_job->params());

        // loop over layers
        find_tracks_in_layers(*cloner, mkfndr.get(), iteration_dir, seeds.begin(), seeds.end(), region);

        cloner->Release();
      });
    });

    // debug = false;
  }

  void MkBuilder::find_tracks_in_layers(CandCloner &cloner,
                                        MkFinder *mkfndr,
                                        SteeringParams::IterationType_e iteration_dir,
                                        const int start_seed,
                                        const int end_seed,
                                        const int region) {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    const TrackerInfo &trk_info = m_job->m_trk_info;
    const SteeringParams &st_par = m_job->steering_params(region);
    const IterationParams &params = m_job->params();

    const int n_seeds = end_seed - start_seed;

    std::vector<std::pair<int, int>> seed_cand_idx, seed_cand_update_idx;
    seed_cand_idx.reserve(n_seeds * params.maxCandsPerSeed);
    seed_cand_update_idx.reserve(n_seeds * params.maxCandsPerSeed);

    std::vector<std::vector<TrackCand>> extra_cands(n_seeds);
    for (int ii = 0; ii < n_seeds; ++ii)
      extra_cands[ii].reserve(params.maxCandsPerSeed);

    cloner.begin_eta_bin(&eoccs, &seed_cand_update_idx, &extra_cands, start_seed, n_seeds);

    // Loop over layers, starting from after the seed.
    // Note that we do a final pass with curr_layer = -1 to update parameters
    // and output final tracks.

    auto layer_plan_it = st_par.make_iterator(iteration_dir);

    dprintf("Made iterator for %d, first layer=%d ... end layer=%d\n",
            iteration_dir,
            layer_plan_it.layer(),
            layer_plan_it.last_layer());

    assert(layer_plan_it.is_pickup_only());

    int curr_layer = layer_plan_it.layer(), prev_layer;

    dprintf(
        "\nMkBuilder::find_tracks_in_layers region=%d, seed_pickup_layer=%d, first_layer=%d; start_seed=%d, "
        "end_seed=%d\n",
        region,
        curr_layer,
        layer_plan_it.next_layer(),
        start_seed,
        end_seed);

    auto &iter_params = (iteration_dir == SteeringParams::IT_BkwSearch) ? m_job->m_iter_config.m_backward_params
                                                                        : m_job->m_iter_config.m_params;

    // Loop over layers according to plan.
    while (++layer_plan_it) {
      prev_layer = curr_layer;
      curr_layer = layer_plan_it.layer();
      mkfndr->Setup(
          iter_params, m_job->m_iter_config.m_layer_configs[curr_layer], m_job->get_mask_for_layer(curr_layer));

      const bool pickup_only = layer_plan_it.is_pickup_only();

      dprintf("\n\n* Processing layer %d, %s\n\n", curr_layer, pickup_only ? "pickup only" : "full finding");

      const LayerInfo &layer_info = trk_info.m_layers[curr_layer];
      const LayerOfHits &layer_of_hits = m_job->m_event_of_hits.m_layers_of_hits[curr_layer];
      const FindingFoos &fnd_foos = FindingFoos::get_finding_foos(layer_info.is_barrel());

      const int theEndCand = find_tracks_unroll_candidates(
          seed_cand_idx, start_seed, end_seed, curr_layer, prev_layer, pickup_only, iteration_dir);

      dprintf("  Number of candidates to process: %d\n", theEndCand);

      // Don't bother messing with the clone engine if there are no candidates
      // (actually it crashes, so this protection is needed).
      // If there are no cands on this iteration, there won't be any later on either,
      // by the construction of the seed_cand_idx vector.
      // XXXXMT There might be cases in endcap where all tracks will miss the
      // next layer, but only relevant if we do geometric selection before.

      if (pickup_only || theEndCand == 0)
        continue;

      cloner.begin_layer(curr_layer);

      //vectorized loop
      for (int itrack = 0; itrack < theEndCand; itrack += NN) {
        const int end = std::min(itrack + NN, theEndCand);

#ifdef DEBUG
        dprintf("\nProcessing track=%d, start_seed=%d, n_seeds=%d, theEndCand=%d, end=%d, nn=%d, end_eq_tec=%d\n",
                itrack,
                start_seed,
                n_seeds,
                theEndCand,
                end,
                end - itrack,
                end == theEndCand);
        dprintf("  (seed,cand): ");
        for (int i = itrack; i < end; ++i)
          dprintf("(%d,%d)  ", seed_cand_idx[i].first, seed_cand_idx[i].second);
        dprintf("\n");
#endif

        mkfndr->InputTracksAndHitIdx(eoccs.m_candidates, seed_cand_idx, itrack, end, false);

#ifdef DEBUG
        for (int i = itrack; i < end; ++i)
          dprintf("  track %d, idx %d is from seed %d\n", i, i - itrack, mkfndr->Label(i - itrack, 0, 0));
#endif

        // propagate to current layer
        (mkfndr->*fnd_foos.m_propagate_foo)(
            layer_info.m_propagate_to, end - itrack, Config::finding_inter_layer_pflags);

        dprint("now get hit range");

#ifdef DUMPHITWINDOW
        mkfndr->m_event = m_event;
#endif

        mkfndr->SelectHitIndices(layer_of_hits, end - itrack);

        find_tracks_handle_missed_layers(
            mkfndr, layer_info, extra_cands, seed_cand_idx, region, start_seed, itrack, end);

        // if (Config::dumpForPlots) {
        //std::cout << "MX number of hits in window in layer " << curr_layer << " is " <<  mkfndr->getXHitEnd(0, 0, 0)-mkfndr->getXHitBegin(0, 0, 0) << std::endl;
        // }

        // copy_out the propagated track params, errors only.
        // Do not, keep cands at last valid hit until actual update,
        // this requires change to propagation flags used in MkFinder::UpdateWithLastHit()
        // from intra-layer to inter-layer.
        // mkfndr->CopyOutParErr(eoccs.m_candidates, end - itrack, true);

        dprint("make new candidates");
        cloner.begin_iteration();

        mkfndr->FindCandidatesCloneEngine(layer_of_hits, cloner, start_seed, end - itrack, fnd_foos);

        cloner.end_iteration();
      }  //end of vectorized loop

      cloner.end_layer();

      // Update loop of best candidates. CandCloner prepares the list of those
      // that need update (excluding all those with negative last hit index).

      const int theEndUpdater = seed_cand_update_idx.size();

      for (int itrack = 0; itrack < theEndUpdater; itrack += NN) {
        const int end = std::min(itrack + NN, theEndUpdater);

        mkfndr->InputTracksAndHitIdx(eoccs.m_candidates, seed_cand_update_idx, itrack, end, true);

        mkfndr->UpdateWithLastHit(layer_of_hits, end - itrack, fnd_foos);

        // copy_out the updated track params, errors only (hit-idcs and chi2 already set)
        mkfndr->CopyOutParErr(eoccs.m_candidates, end - itrack, false);
      }

      // Check if cands are sorted, as expected.
      /*
    for (int iseed = start_seed; iseed < end_seed; ++iseed)
    {
      auto & cc = eoccs[iseed];

      for (int i = 0; i < ((int) cc.size()) - 1; ++i)
      {
        if (cc[i].score() < cc[i+1].score())
        {
          printf("CloneEngine - NOT SORTED: layer=%d, iseed=%d (size=%llu)-- %d : %d smaller than %d : %d\n",
                 curr_layer, iseed, cc.size(), i, cc[i].score(), i+1, cc[i+1].score());
        }
      }
    }
    */

    }  // end of layer loop

    cloner.end_eta_bin();

    // final sorting
    for (int iseed = start_seed; iseed < end_seed; ++iseed) {
      eoccs[iseed].MergeCandsAndBestShortOne(m_job->params(), true, true);
    }
  }

  //==============================================================================
  // BackwardFit
  //==============================================================================

  // MT-RATS - eta separators can be screwed after copy out with possibly empty CombCands.
  // I added asserts to two applicable places above (both here in MkBuilder.cc).
  // One could also re-calculate / adjust m_seedEtaSeparators, during export iself, probably.
  // Or use separate seed / track vectors for every region -- which would be prettier.

  void MkBuilder::BackwardFitBH() {
    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      tbb::parallel_for(rosi.tbb_blk_rng_vec(), [&](const tbb::blocked_range<int> &blk_rng) {
        FINDER(mkfndr);

        RangeOfSeedIndices rng = rosi.seed_rng(blk_rng);

        while (rng.valid()) {
          // final backward fit
          fit_cands_BH(mkfndr.get(), rng.m_beg, rng.m_end, region);

          ++rng;
        }
      });
    });
  }

  void MkBuilder::fit_cands_BH(MkFinder *mkfndr, int start_cand, int end_cand, int region) {
    const SteeringParams &st_par = m_job->steering_params(region);

    for (int icand = start_cand; icand < end_cand; icand += NN) {
      const int end = std::min(icand + NN, end_cand);

      // printf("Pre Final fit for %d - %d\n", icand, end);
      // for (int i = icand; i < end; ++i) { const Track &t = eoccs[i][0];
      //   printf("  %4d with q=%+d chi2=%7.3f pT=%7.3f eta=% 7.3f x=%.3f y=%.3f z=%.3f nHits=%2d  label=%4d findable=%d\n",
      //          i, t.charge(), t.chi2(), t.pT(), t.momEta(), t.x(), t.y(), t.z(), t.nFoundHits(), t.label(), t.isFindable());
      // }

      bool chi_debug = false;
#ifdef DEBUG_BACKWARD_FIT_BH
    redo_fit:
#endif

      // input candidate tracks
      mkfndr->BkFitInputTracks(m_tracks, icand, end);

      // perform fit back to first layer on track
      mkfndr->BkFitFitTracksBH(m_job->m_event_of_hits, st_par, end - icand, chi_debug);

      // now move one last time to PCA
      if (Config::includePCA) {
        mkfndr->BkFitPropTracksToPCA(end - icand);
      }

#ifdef DEBUG_BACKWARD_FIT_BH
      // Dump tracks with pT > 2 and chi2/dof > 20. Assumes MPT_SIZE=1.
      if (!chi_debug && 1.0f / mkfndr->Par[MkBase::iP].At(0, 3, 0) > 2.0f &&
          mkfndr->Chi2(0, 0, 0) / (eoccs[icand][0].nFoundHits() * 3 - 6) > 20.0f) {
        chi_debug = true;
        printf("CHIHDR Event %d, Cand %3d, pT %f, chipdof %f ### NOTE x,y,z in cm, sigmas, deltas in mum ### !!!\n",
               m_event->evtID(),
               icand,
               1.0f / mkfndr->Par[MkBase::iP].At(0, 3, 0),
               mkfndr->Chi2(0, 0, 0) / (eoccs[icand][0].nFoundHits() * 3 - 6));
        printf(
            "CHIHDR %3s %10s %10s %10s %10s %10s %11s %11s %11s %10s %10s %10s %10s %11s %11s %11s %10s %10s %10s %10s "
            "%10s %11s %11s\n",
            "lyr",
            "chi2",
            "x_h",
            "y_h",
            "z_h",
            "r_h",
            "sx_h",
            "sy_h",
            "sz_h",
            "x_t",
            "y_t",
            "z_t",
            "r_t",
            "sx_t",
            "sy_t",
            "sz_t",
            "pt",
            "phi",
            "theta",
            "phi_h",
            "phi_t",
            "d_xy",
            "d_z");
        goto redo_fit;
      }
#endif

      // copy out full set of info at last propagated position
      mkfndr->BkFitOutputTracks(m_tracks, icand, end, Config::includePCA);

      // printf("Post Final fit for %d - %d\n", icand, end);
      // for (int i = icand; i < end; ++i) { const Track &t = eoccs[i][0];
      //   printf("  %4d with q=%+d chi2=%7.3f pT=%7.3f eta=% 7.3f x=%.3f y=%.3f z=%.3f nHits=%2d  label=%4d findable=%d\n",
      //          i, t.charge(), t.chi2(), t.pT(), t.momEta(), t.x(), t.y(), t.z(), t.nFoundHits(), t.label(), t.isFindable());
      // }
    }
  }

  //------------------------------------------------------------------------------

  void MkBuilder::BackwardFit() {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;

    tbb::parallel_for_each(m_job->regions_begin(), m_job->regions_end(), [&](int region) {
      const RegionOfSeedIndices rosi(m_seedEtaSeparators, region);

      // adaptive seeds per task based on the total estimated amount of work to divide among all threads
      const int adaptiveSPT =
          clamp(Config::numThreadsEvents * eoccs.m_size / Config::numThreadsFinder + 1, 4, Config::numSeedsPerTask);
      dprint("adaptiveSPT " << adaptiveSPT << " fill " << rosi.count() << "/" << eoccs.m_size << " region " << region);

      tbb::parallel_for(rosi.tbb_blk_rng_std(adaptiveSPT), [&](const tbb::blocked_range<int> &cands) {
        FINDER(mkfndr);

        fit_cands(mkfndr.get(), cands.begin(), cands.end(), region);
      });
    });
  }

  void MkBuilder::fit_cands(MkFinder *mkfndr, int start_cand, int end_cand, int region) {
    EventOfCombCandidates &eoccs = m_event_of_comb_cands;
    const SteeringParams &st_par = m_job->steering_params(region);

    int step = NN;

    for (int icand = start_cand; icand < end_cand; icand += step) {
      int end = std::min(icand + NN, end_cand);

      // Check if we need to fragment this for SlurpIn to work.
      // Would actually prefer to do memory allocator for HoTNode storage.
      // This has been "fixed" by copying Cands back into original container, not swapping the contents
      // with vectors created in another thread (and thus not in the same memory pool, apparently), see
      // CandCloner::ProcessSeedRange(). Standard building does not have this problem.
      /*
    step = NN;
    {
       int end_c = icand + 1;
       while (end_c < end)
       {
          // Still crashes with 0x1fffffff and 0x1ffffff, 0x1fffff works (~2000 breaks over 5k high PU events)
          if (std::abs(&eoccs[icand][0] - &eoccs[end_c][0]) > 0x1fffff)
          {
             end  = end_c;
             step = end - icand;
             if ( ! Config::silent)
               printf("XXYZZ MkBuilder::fit_cands Breaking up candidates with offset outside of 32-bit range, step=%d.\n", step);
             break;
          }
          ++end_c;
       }
    }
    */

      // printf("Pre Final fit for %d - %d\n", icand, end);
      // for (int i = icand; i < end; ++i) { const Track &t = eoccs[i][0];
      //   printf("  %4d with q=%+d chi2=%7.3f pT=%7.3f eta=% 7.3f x=%.3f y=%.3f z=%.3f nHits=%2d  label=%4d findable=%d\n",
      //          i, t.charge(), t.chi2(), t.pT(), t.momEta(), t.x(), t.y(), t.z(), t.nFoundHits(), t.label(), t.isFindable());
      // }

      bool chi_debug = false;
#ifdef DEBUG_BACKWARD_FIT
      chi_debug = true;
      static bool first = true;
      if (first) {
        // ./mkFit ... | perl -ne 'if (/^BKF_OVERLAP/) { s/^BKF_OVERLAP //og; print; }' > bkf_ovlp.rtt
        printf(
            "BKF_OVERLAP event/I:label/I:prod_type/I:is_findable/I:layer/I:is_stereo/I:is_barrel/I:"
            "pt/F:eta/F:phi/F:chi2/F:isnan/I:isfin/I:gtzero/I:hit_label/I:"
            "sx_t/F:sy_t/F:sz_t/F:d_xy/F:d_z/F\n");
        first = false;
      }
      mkfndr->m_event = m_event;
#endif

      // input tracks
      mkfndr->BkFitInputTracks(eoccs, icand, end);

      // fit tracks back to first layer
      mkfndr->BkFitFitTracks(m_job->m_event_of_hits, st_par, end - icand, chi_debug);

      // now move one last time to PCA
      if (Config::includePCA) {
        mkfndr->BkFitPropTracksToPCA(end - icand);
      }

      mkfndr->BkFitOutputTracks(eoccs, icand, end, Config::includePCA);

      // printf("Post Final fit for %d - %d\n", icand, end);
      // for (int i = icand; i < end; ++i) { const Track &t = eoccs[i][0];
      //   printf("  %4d with q=%+d chi2=%7.3f pT=%7.3f eta=% 7.3f x=%.3f y=%.3f z=%.3f nHits=%2d  label=%4d findable=%d\n",
      //          i, t.charge(), t.chi2(), t.pT(), t.momEta(), t.x(), t.y(), t.z(), t.nFoundHits(), t.label(), t.isFindable());
      // }
    }
  }

}  // end namespace mkfit
