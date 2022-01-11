#include "RecoTracker/MkFitCMS/standalone/MkStandaloneSeqs.h"
#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"

#include "RecoTracker/MkFitCore/src/Debug.h"

#include "tbb/parallel_for.h"

namespace mkfit {

  namespace StdSeq {

    //=========================================================================
    // Hit processing
    //=========================================================================

    void LoadHitsAndBeamSpot(Event &ev, EventOfHits &eoh) {
      eoh.Reset();

      // fill vector of hits in each layer
      // XXXXMT: Does it really makes sense to multi-thread this?
      tbb::parallel_for(tbb::blocked_range<int>(0, ev.layerHits_.size()), [&](const tbb::blocked_range<int> &layers) {
        for (int ilay = layers.begin(); ilay < layers.end(); ++ilay) {
          eoh.SuckInHits(ilay, ev.layerHits_[ilay]);
        }
      });
      eoh.SetBeamSpot(ev.beamSpot_);
    }

    void handle_duplicates(Event *event) {
      // Mark tracks as duplicates; if within CMSSW, remove duplicate tracks from fit or candidate track collection
      if (Config::removeDuplicates) {
        if (Config::quality_val || Config::sim_val || Config::cmssw_val) {
          find_duplicates(event->candidateTracks_);
          if (Config::backwardFit)
            find_duplicates(event->fitTracks_);
        }
        // For the MEIF benchmarks and the stress tests, no validation flags are set so we will enter this block
        else {
          // Only care about the candidate tracks here; no need to run the duplicate removal on both candidate and fit tracks
          find_duplicates(event->candidateTracks_);
        }
      }
    }

    //=========================================================================
    // Random stuff
    //=========================================================================

    void dump_simtracks(Event *event) {
      // Ripped out of MkBuilder::begin_event, ifdefed under DEBUG

      std::vector<Track> &simtracks = event->simTracks_;

      for (int itrack = 0; itrack < (int)simtracks.size(); ++itrack) {
        // bool debug = true;
        Track track = simtracks[itrack];
        // if (track.label() != itrack) {
        //   dprintf("Bad label for simtrack %d -- %d\n", itrack, track.label());
        // }

        dprint("MX - simtrack with nHits=" << track.nFoundHits() << " chi2=" << track.chi2() << " pT=" << track.pT()
                                           << " phi=" << track.momPhi() << " eta=" << track.momEta());
      }

      for (int itrack = 0; itrack < (int)simtracks.size(); ++itrack) {
        for (int ihit = 0; ihit < simtracks[itrack].nFoundHits(); ++ihit) {
          dprint("track #" << itrack << " hit #" << ihit
                           << " hit pos=" << simtracks[itrack].hitsVector(event->layerHits_)[ihit].position()
                           << " phi=" << simtracks[itrack].hitsVector(event->layerHits_)[ihit].phi());
        }
      }
    }

    void track_print(Event *event, const Track &t, const char *pref) {
      printf("%s with q=%+i pT=%7.3f eta=% 7.3f nHits=%2d  label=%4d\nState:\n",
             pref,
             t.charge(),
             t.pT(),
             t.momEta(),
             t.nFoundHits(),
             t.label());

      print(t.state());

      printf("Hits:\n");
      for (int ih = 0; ih < t.nTotalHits(); ++ih) {
        int lyr = t.getHitLyr(ih);
        int idx = t.getHitIdx(ih);
        if (idx >= 0) {
          const Hit &hit = event->layerHits_[lyr][idx];
          printf("    hit %2d lyr=%2d idx=%4d pos r=%7.3f z=% 8.3f   mc_hit=%4d mc_trk=%4d\n",
                 ih,
                 lyr,
                 idx,
                 hit.r(),
                 hit.z(),
                 hit.mcHitID(),
                 hit.mcTrackID(event->simHitsInfo_));
        } else
          printf("    hit %2d        idx=%i\n", ih, t.getHitIdx(ih));
      }
    }

    //------------------------------------------------------------------------------
    // Non-ROOT validation
    //------------------------------------------------------------------------------

    void Quality::quality_val(Event *event) {
      quality_reset();

      std::map<int, int> cmsswLabelToPos;
      if (Config::dumpForPlots && Config::readCmsswTracks) {
        for (size_t itrack = 0; itrack < event->cmsswTracks_.size(); itrack++) {
          cmsswLabelToPos[event->cmsswTracks_[itrack].label()] = itrack;
        }
      }

      for (size_t itrack = 0; itrack < event->candidateTracks_.size(); itrack++) {
        quality_process(event, event->candidateTracks_[itrack], itrack, cmsswLabelToPos);
      }

      quality_print();
    }

    void Quality::quality_reset() { m_cnt = m_cnt1 = m_cnt2 = m_cnt_8 = m_cnt1_8 = m_cnt2_8 = m_cnt_nomc = 0; }

    void Quality::quality_process(Event *event, Track &tkcand, const int itrack, std::map<int, int> &cmsswLabelToPos) {
      // KPM: Do not use this method for validating CMSSW tracks if we ever build a DumbCMSSW function for them to print out...
      // as we would need to access seeds through map of seed ids...

      // initialize track extra (input original seed label)
      const auto label = tkcand.label();
      TrackExtra extra(label);

      // track_print(tkcand, "XXX");

      // access temp seed trk and set matching seed hits
      const auto &seed = event->seedTracks_[itrack];
      extra.findMatchingSeedHits(tkcand, seed, event->layerHits_);

      // set mcTrackID through 50% hit matching after seed
      extra.setMCTrackIDInfo(
          tkcand, event->layerHits_, event->simHitsInfo_, event->simTracks_, false, (Config::seedInput == simSeeds));
      const int mctrk = extra.mcTrackID();

      //  int mctrk = tkcand.label(); // assumes 100% "efficiency"

      const float pT = tkcand.pT();
      float pTmc = 0.f, etamc = 0.f, phimc = 0.f;
      float pTr = 0.f;
      int nfoundmc = -1;

      if (mctrk < 0 || static_cast<size_t>(mctrk) >= event->simTracks_.size()) {
        ++m_cnt_nomc;
        dprint("XX bad track idx " << mctrk << ", orig label was " << label);
      } else {
        auto &simtrack = event->simTracks_[mctrk];
        pTmc = simtrack.pT();
        etamc = simtrack.momEta();
        phimc = simtrack.momPhi();
        pTr = pT / pTmc;

        nfoundmc = simtrack.nUniqueLayers();

        ++m_cnt;
        if (pTr > 0.9 && pTr < 1.1)
          ++m_cnt1;
        if (pTr > 0.8 && pTr < 1.2)
          ++m_cnt2;

        if (tkcand.nFoundHits() >= 0.8f * nfoundmc) {
          ++m_cnt_8;
          if (pTr > 0.9 && pTr < 1.1)
            ++m_cnt1_8;
          if (pTr > 0.8 && pTr < 1.2)
            ++m_cnt2_8;
        }

        // perl -ne 'print if m/FOUND_LABEL\s+[-\d]+/o;' | sort -k2 -n
        // grep "FOUND_LABEL" | sort -n -k 8,8 -k 2,2
        // printf("FOUND_LABEL %6d  pT_mc= %8.2f eta_mc= %8.2f event= %d\n", label, pTmc, etamc, event->evtID());
      }

#ifdef SELECT_SEED_LABEL
      if (label == SELECT_SEED_LABEL)
        track_print(tkcand, "MkBuilder::quality_process SELECT_SEED_LABEL:");
#endif

      float pTcmssw = 0.f, etacmssw = 0.f, phicmssw = 0.f;
      int nfoundcmssw = -1;
      if (Config::dumpForPlots && Config::readCmsswTracks) {
        if (cmsswLabelToPos.count(label)) {
          auto &cmsswtrack = event->cmsswTracks_[cmsswLabelToPos[label]];
          pTcmssw = cmsswtrack.pT();
          etacmssw = cmsswtrack.momEta();
          phicmssw = cmsswtrack.swimPhiToR(tkcand.x(), tkcand.y());  // to get rough estimate of diff in phi
          nfoundcmssw = cmsswtrack.nUniqueLayers();
        }
      }

      if (!Config::silent && Config::dumpForPlots) {
        std::lock_guard<std::mutex> printlock(Event::printmutex);
        printf(
            "MX - found track with chi2= %6.3f nFoundHits= %2d pT= %7.4f eta= %7.4f phi= %7.4f nfoundmc= %2d pTmc= "
            "%7.4f etamc= %7.4f phimc= %7.4f nfoundcmssw= %2d pTcmssw= %7.4f etacmssw= %7.4f phicmssw= %7.4f lab= %d\n",
            tkcand.chi2(),
            tkcand.nFoundHits(),
            pT,
            tkcand.momEta(),
            tkcand.momPhi(),
            nfoundmc,
            pTmc,
            etamc,
            phimc,
            nfoundcmssw,
            pTcmssw,
            etacmssw,
            phicmssw,
            label);
      }
    }

    void Quality::quality_print() {
      if (!Config::silent) {
        std::lock_guard<std::mutex> printlock(Event::printmutex);
        std::cout << "found tracks=" << m_cnt << "  in pT 10%=" << m_cnt1 << "  in pT 20%=" << m_cnt2
                  << "     no_mc_assoc=" << m_cnt_nomc << std::endl;
        std::cout << "  nH >= 80% =" << m_cnt_8 << "  in pT 10%=" << m_cnt1_8 << "  in pT 20%=" << m_cnt2_8
                  << std::endl;
      }
    }

    //------------------------------------------------------------------------------
    // Root validation
    //------------------------------------------------------------------------------

    void root_val_dumb_cmssw(Event *event) {
      // get labels correct first
      event->relabel_bad_seedtracks();
      event->relabel_cmsswtracks_from_seeds();

      //collection cleaning
      if (Config::nItersCMSSW > 0)
        event->select_tracks_iter(Config::nItersCMSSW);

      // set the track collections to each other
      event->candidateTracks_ = event->cmsswTracks_;
      event->fitTracks_ = event->candidateTracks_;

      // prep the tracks + extras
      prep_simtracks(event);
      prep_recotracks(event);

      // validate
      event->Validate();
    }

    void root_val(Event *event) {
      // score the tracks
      score_tracks(event->seedTracks_);
      score_tracks(event->candidateTracks_);

      // deal with fit tracks
      if (Config::backwardFit) {
        score_tracks(event->fitTracks_);
      } else
        event->fitTracks_ = event->candidateTracks_;

      // sort hits + make extras, align if needed
      prep_recotracks(event);
      if (Config::cmssw_val)
        prep_cmsswtracks(event);

      // validate
      event->Validate();
    }

    void prep_recotracks(Event *event) {
      // seed tracks extras always needed
      if (Config::sim_val || Config::sim_val_for_cmssw) {
        prep_tracks(event, event->seedTracks_, event->seedTracksExtra_, true);
      } else if (Config::cmssw_val)  // seed tracks are not validated, labels used for maps --> do NOT align index and labels!
      {
        prep_tracks(event, event->seedTracks_, event->seedTracksExtra_, false);
      }

      // make extras + align index == label() for candidate tracks
      prep_tracks(event, event->candidateTracks_, event->candidateTracksExtra_, true);
      prep_tracks(event, event->fitTracks_, event->fitTracksExtra_, true);
    }

    void prep_simtracks(Event *event) {
      // First prep sim tracks to have hits sorted, then mark unfindable if too short
      prep_reftracks(event, event->simTracks_, event->simTracksExtra_, false);

      // Now, make sure sim track shares at least four hits with a single cmssw seed.
      // This ensures we factor out any weakness from CMSSW

      // First, make a make a map of [lyr][hit idx].vector(seed trk labels)
      LayIdxIDVecMapMap seedHitIDMap;
      std::map<int, int> labelNHitsMap;
      std::map<int, int> labelAlgoMap;
      std::map<int, std::vector<int>> labelSeedHitsMap;
      for (const auto &seedtrack : event->seedTracks_) {
        for (int ihit = 0; ihit < seedtrack.nTotalHits(); ihit++) {
          const auto lyr = seedtrack.getHitLyr(ihit);
          const auto idx = seedtrack.getHitIdx(ihit);

          if (lyr < 0 || idx < 0)
            continue;  // standard check
          seedHitIDMap[lyr][idx].push_back(seedtrack.label());
          labelSeedHitsMap[seedtrack.label()].push_back(lyr);
        }
        labelNHitsMap[seedtrack.label()] = seedtrack.nTotalHits();
        labelAlgoMap[seedtrack.label()] = seedtrack.algoint();
      }

      // Then, loop over sim tracks, and add up how many lyrs they possess of a single seed track
      unsigned int count = 0;
      for (auto &simtrack : event->simTracks_) {
        if (simtrack.isNotFindable())
          continue;  // skip ones we already know are bad
        TrkIDLaySetMap seedIDMap;
        for (int ihit = 0; ihit < simtrack.nTotalHits(); ihit++) {
          const auto lyr = simtrack.getHitLyr(ihit);
          const auto idx = simtrack.getHitIdx(ihit);

          if (lyr < 0 || idx < 0)
            continue;  // standard check

          if (!seedHitIDMap.count(lyr))
            continue;  // ensure seed hit map has at least one entry for this layer
          if (!seedHitIDMap.at(lyr).count(idx))
            continue;  // ensure seed hit map has at least one entry for this idx

          for (const auto label : seedHitIDMap.at(lyr).at(idx)) {
            const auto &seedLayers = labelSeedHitsMap[label];
            if (std::find(seedLayers.begin(), seedLayers.end(), lyr) != seedLayers.end())  //seed check moved here
              seedIDMap[label].emplace(lyr);
          }
        }

        // now see if one of the seedIDs matched has at least 4 hits!
        bool isSimSeed = false;
        for (const auto &seedIDpair : seedIDMap) {
          if ((int)seedIDpair.second.size() == labelNHitsMap[seedIDpair.first]) {
            isSimSeed = true;
            if (Config::mtvRequireSeeds)
              simtrack.setAlgoint(labelAlgoMap[seedIDpair.first]);
            if (Config::mtvRequireSeeds)
              event->simTracksExtra_[count].addAlgo(labelAlgoMap[seedIDpair.first]);
            //break;
          }
        }
        if (Config::mtvLikeValidation) {
          // Apply MTV selection criteria and then return
          if (simtrack.prodType() != Track::ProdType::Signal || simtrack.charge() == 0 || simtrack.posR() > 2.5 ||
              std::abs(simtrack.z()) > 30 || std::abs(simtrack.momEta()) > 3.0)
            simtrack.setNotFindable();
          else if (Config::mtvRequireSeeds && !isSimSeed)
            simtrack.setNotFindable();
        } else {
          // set findability based on bool isSimSeed
          if (!isSimSeed)
            simtrack.setNotFindable();
        }
        count++;
      }
    }

    void prep_cmsswtracks(Event *event) { prep_reftracks(event, event->cmsswTracks_, event->cmsswTracksExtra_, true); }

    void prep_reftracks(Event *event, TrackVec &tracks, TrackExtraVec &extras, const bool realigntracks) {
      prep_tracks(event, tracks, extras, realigntracks);

      // mark cmsswtracks as unfindable if too short
      for (auto &track : tracks) {
        const int nlyr = track.nUniqueLayers();
        if (nlyr < Config::cmsSelMinLayers)
          track.setNotFindable();
      }
    }

    void prep_tracks(Event *event, TrackVec &tracks, TrackExtraVec &extras, const bool realigntracks) {
      for (size_t i = 0; i < tracks.size(); i++) {
        extras.emplace_back(tracks[i].label());
      }
      if (realigntracks)
        event->validation_.alignTracks(tracks, extras, false);
    }

    void score_tracks(TrackVec &tracks) {
      for (auto &track : tracks) {
        track.setScore(getScoreCand(track));
      }
    }

  }  // namespace StdSeq

}  // namespace mkfit

//==============================
// Carryover from MkBuilder.cc
//==============================

//------------------------------------------------------------------------------
// Seeding functions: importing, finding and fitting
//------------------------------------------------------------------------------

/*

void MkBuilder::create_seeds_from_sim_tracks()
{
  // Import from simtrack snatching first Config::nlayers_per_seed hits.
  //
  // Reduce number of hits, pick endcap over barrel when both are available.
  //   This is done by assumin endcap hit is after the barrel, no checks.

  // bool debug = true;

  TrackerInfo &trk_info = Config::TrkInfo;
  TrackVec    &sims     = m_event->simTracks_;
  TrackVec    &seeds    = m_event->seedTracks_;

  const int size = sims.size();
  seeds.clear();       // Needed when reading from file and then recreating from sim.
  seeds.reserve(size);

  dprintf("MkBuilder::create_seeds_from_sim_tracks processing %d simtracks.\n", size);

  for (int i = 0; i < size; ++i)
  {
    const Track &src = sims[i];

    dprintf("  [%d] pT=%f eta=%f n_hits=%d lbl=%d\n", i, src.pT(), src.momEta(), src.nFoundHits(), src.label());

    if (src.isNotFindable())
    {
      dprintf("  [%d] not findable.\n", i);
      continue;
    }

    int h_sel = 0, h = 0;
    const HitOnTrack *hots = src.getHitsOnTrackArray();
    HitOnTrack  new_hots[ Config::nlayers_per_seed_max ];

    // Exit condition -- need to check one more hit after Config::nlayers_per_seed
    // good hits are found.
    bool last_hit_check = false;

    while ( ! last_hit_check && h < src.nTotalHits())
    {
      assert (hots[h].index >= 0 && "Expecting input sim tracks (or seeds later) to not have holes");

      if (h_sel == Config::nlayers_per_seed) last_hit_check = true;

      // Check if hit is on a sibling layer given the previous one.
      if (h_sel > 0 && trk_info.are_layers_siblings(new_hots[h_sel - 1].layer, hots[h].layer))
      {
        dprintf("    [%d] Sibling layers %d %d ... overwriting with new one\n", i,
                new_hots[h_sel - 1].layer, hots[h].layer);

        new_hots[h_sel - 1] = hots[h];
      }
      // Drop further hits on the same layer. It seems hard to select the best one (in any way).
      else if (h_sel > 0 && new_hots[h_sel - 1].layer == hots[h].layer)
      {
        dprintf("    [%d] Hits on the same layer %d ... keeping the first one\n", i, hots[h].layer);
      }
      else if ( ! last_hit_check)
      {
        new_hots[h_sel++] = hots[h];
      }

      ++h;
    }

    if (h_sel < Config::nlayers_per_seed)
    {
      printf("MkBuilder::create_seeds_from_sim_tracks simtrack %d only yielded %d hits. Skipping ...\n",
             src.label(), h_sel);
      continue;
    }

    seeds.emplace_back( Track(src.state(), 0, src.label(), h_sel, new_hots) );

    dprintf("  [%d->%d] Seed nh=%d, last_lay=%d, last_idx=%d\n", i, (int) seeds.size() - 1,
            seeds.back().nTotalHits(), seeds.back().getLastHitLyr(), seeds.back().getLastHitIdx());
    // dprintf("  "); for (int i=0; i<dst.nTotalHits();++i) printf(" (%d/%d)", dst.getHitIdx(i), dst.getHitLyr(i)); printf("\n");
  }

  dprintf("MkBuilder::create_seeds_from_sim_tracks finished processing of %d sim tracks - created %d seeds.\n",
          size, (int) seeds.size());
}


void MkBuilder::find_seeds()
{
  fprintf(stderr, "__FILE__::__LINE__ Needs fixing for B/E support, search for XXMT4K\n");
  exit(1);

#ifdef DEBUG
  bool debug(false);
#endif
  TripletIdxConVec seed_idcs;

  //double time = dtime();
  findSeedsByRoadSearch(seed_idcs,m_event_of_hits.m_layers_of_hits,m_event->layerHits_[1].size(),m_event);
  //time = dtime() - time;

  // use this to initialize tracks
  // XXMT4K  ... configurable input layers ... or hardcode something else for endcap.
  // Could come from TrackerInfo ...
  // But what about transition ... TrackerInfo as well or arbitrary combination of B/E seed layers ????

  const LayerOfHits &loh0 = m_event_of_hits.m_layers_of_hits[0];
  const LayerOfHits &loh1 = m_event_of_hits.m_layers_of_hits[1];
  const LayerOfHits &loh2 = m_event_of_hits.m_layers_of_hits[2];

  // make seed tracks
  TrackVec & seedtracks = m_event->seedTracks_;
  seedtracks.resize(seed_idcs.size());
  for (size_t iseed = 0; iseed < seedtracks.size(); iseed++)
  {
    auto & seedtrack = seedtracks[iseed];
    seedtrack.setLabel(iseed);

    // use to set charge
    const Hit & hit0 = loh0.GetHit(seed_idcs[iseed][0]);
    const Hit & hit1 = loh1.GetHit(seed_idcs[iseed][1]);
    const Hit & hit2 = loh2.GetHit(seed_idcs[iseed][2]);

    seedtrack.setCharge(calculateCharge(hit0,hit1,hit2));

    for (int ihit = 0; ihit < Config::nlayers_per_seed; ihit++)
    {
      // XXMT4K  - ihit to layer[ihit]
      seedtrack.addHitIdx(seed_idcs[iseed][ihit], ihit, 0.0f);
    }

    for (int ihit = Config::nlayers_per_seed; ihit < Config::nLayers; ihit++)
    {
      seedtrack.setHitIdxLyr(ihit, -1, -1);
    }

    dprint("iseed: " << iseed << " mcids: " << hit0.mcTrackID(m_event->simHitsInfo_) << " " <<
	   hit1.mcTrackID(m_event->simHitsInfo_) << " " << hit1.mcTrackID(m_event->simHitsInfo_));
  }
}

} // end namespace mkfit

namespace
{
  void fill_seed_layer_sig(const Track& trk, int n_hits, bool is_brl[])
  {
    const TrackerInfo &trk_info = Config::TrkInfo;

    for (int i = 0; i < n_hits; ++i)
    {
      is_brl[i] = trk_info.m_layers[ trk.getHitLyr(i) ].is_barrel();
    }
  }

  bool are_seed_layer_sigs_equal(const Track& trk, int n_hits, const bool is_brl_ref[])
  {
    const TrackerInfo &trk_info = Config::TrkInfo;

    for (int i = 0; i < n_hits; ++i)
    {
      if(trk_info.m_layers[ trk.getHitLyr(i) ].is_barrel() != is_brl_ref[i]) return false;
    }

    return true;
  }
}

namespace mkfit {

void MkBuilder::fit_seeds()
{
  // Expect seeds to be sorted in eta (in some way) and that Event::seedEtaSeparators_[]
  // array holds starting indices of 5 eta regions.
  // Within each region it vectorizes the fit as long as layer indices of all seeds match.
  // See layer_sig_change label below.
  // Alternatively, we could be using the layer plan (but it might require addition of
  // a new flag in LayerControl (well, should really change those bools to a bitfield).

  // debug = true;

  TrackVec& seedtracks = m_event->seedTracks_;

  dcall(print_seeds(seedtracks));

  tbb::parallel_for_each(m_regions.begin(), m_regions.end(),
  [&](int reg)
  {
    RegionOfSeedIndices rosi(m_seedEtaSeparators, reg);

    tbb::parallel_for(rosi.tbb_blk_rng_vec(),
      [&](const tbb::blocked_range<int>& blk_rng)
    {
      // printf("TBB seeding krappe -- range = %d to %d - extent = %d ==> %d to %d - extent %d\n",
      //        i.begin(), i.end(), i.end() - i.begin(), beg, std::min(end,theEnd), std::min(end,theEnd) - beg);

      // printf("Seed info pos(  x       y       z        r;     eta    phi)   mom(  pt      pz;     eta    phi)\n");

      FITTER( mkfttr );

      RangeOfSeedIndices rng = rosi.seed_rng(blk_rng);

      while (rng.valid())
      {
#ifdef DEBUG
        // MT dump seed so i see if etas are about right
        for (int i = rng.m_beg; i < rng.m_end; ++i)
        {
          auto &t = seedtracks[i];
          auto &dst = t;
          dprintf("Seed %4d lbl=%d pos(%+7.3f %+7.3f %+7.3f; %+7.3f %+6.3f %+6.3f) mom(%+7.3f %+7.3f; %+6.3f %+6.3f)\n",
                  i, t.label(), t.x(), t.y(), t.z(), t.posR(), t.posEta(), t.posPhi(),
                  t.pT(), t.pz(), t.momEta(), t.momPhi());
          dprintf("  Idx/lay for above track:"); for (int i=0; i<dst.nTotalHits();++i) dprintf(" (%d/%d)", dst.getHitIdx(i), dst.getHitLyr(i)); dprintf("\n");
        }
#endif

        // We had seeds sorted in eta_mom ... but they have dZ displacement ...
        // so they can go through "semi random" barrel/disk pattern close to
        // transition region for overall layers 2 and 3 where eta of barrel is
        // larger than transition region.
        // E.g., for 10k tracks in endcap/barrel the break happens ~250 times,
        // often several times witin the same NN range (5 time is not rare with NN=8).
        //
        // Sorting on eta_pos of the last seed hit yields ~50 breaks on the same set.
        // This is being used now (in import_seeds()).
        //
        // In the following we make sure seed range passed to vectorized
        // function has compatible layer signatures (barrel / endcap).

      layer_sig_change:

        bool is_brl[Config::nlayers_per_seed_max];

        fill_seed_layer_sig(seedtracks[rng.m_beg], Config::nlayers_per_seed, is_brl);

        for (int i = rng.m_beg + 1; i < rng.m_end; ++i)
        {
          if ( ! are_seed_layer_sigs_equal(seedtracks[i], Config::nlayers_per_seed, is_brl))
          {
            dprintf("Breaking seed range due to different layer signature at %d (%d, %d)\n", i, rng.m_beg, rng.m_end);

            fit_one_seed_set(seedtracks, rng.m_beg, i, mkfttr.get(), is_brl);

            rng.m_beg = i;
            goto layer_sig_change;
          }
        }

        fit_one_seed_set(seedtracks, rng.m_beg, rng.m_end, mkfttr.get(), is_brl);

        ++rng;
      }
    });
  });
}

inline void MkBuilder::fit_one_seed_set(TrackVec& seedtracks, int itrack, int end,
                                        MkFitter *mkfttr, const bool is_brl[])
{
  // debug=true;

  mkfttr->SetNhits(Config::nlayers_per_seed);
  mkfttr->InputTracksAndHits(seedtracks, m_event_of_hits.m_layers_of_hits, itrack, end);

  if (Config::cf_seeding) mkfttr->ConformalFitTracks(false, itrack, end);

  mkfttr->FitTracksSteered(is_brl, end - itrack, m_event, Config::seed_fit_pflags);

  mkfttr->OutputFittedTracksAndHitIdx(m_event->seedTracks_, itrack, end, false);
}


//------------------------------------------------------------------------------
// PrepareSeeds
//------------------------------------------------------------------------------

void MkBuilder::PrepareSeeds()
{
  // {
  //   TrackVec  &tv = m_event->seedTracks_;
  //   char pref[80];
  //   for (int i = 0; i < (int) tv.size(); ++i)
  //   {
  //     sprintf(pref, "Pre-cleaning seed silly value check event=%d index=%d:", m_event->evtID(), i);
  //     tv[i].hasSillyValues(true, false, pref);
  //   }
  // }

  if (Config::seedInput == simSeeds)
  {
    if (Config::useCMSGeom)
    {
      m_event->clean_cms_simtracks();

      // printf("\n* Simtracks after cleaning:\n");
      // m_event->print_tracks(m_event->simTracks_, true);
      // printf("\n");
    }
    // create_seeds_from_sim_tracks();

    seed_post_cleaning(m_event->seedTracks_);
  }
  else if (Config::seedInput == cmsswSeeds)
  {
    m_event->relabel_bad_seedtracks();

    // want to make sure we mark which sim tracks are findable based on cmssw seeds BEFORE seed cleaning
    if (Config::sim_val || Config::quality_val)
    {
      prep_simtracks();
    }

    // need to make a map of seed ids to cmssw tk ids BEFORE seeds are sorted
    if (Config::cmssw_val)
    {
      m_event->validation_.makeSeedTkToCMSSWTkMap(*m_event);
    }

    // this is a hack that allows us to associate seeds with cmssw tracks for the text dump plots
    if (Config::dumpForPlots && Config::readCmsswTracks)
    {
      for (size_t itrack = 0; itrack < m_event->cmsswTracks_.size(); itrack++)
      {
        const auto &cmsswtrack = m_event->cmsswTracks_[itrack];
        const auto cmsswlabel = cmsswtrack.label();
        auto &seedtrack = m_event->seedTracks_[cmsswlabel];
        seedtrack.setLabel(cmsswlabel);
      }
    }

    if (Config::seedCleaning == cleanSeedsN2)
    {
      m_event->clean_cms_seedtracks();

      // Select specific cmssw seed for detailed debug.
      // {
      //   Track xx = m_event->seedTracks_[6];
      //   m_event->seedTracks_.clear();
      //   m_event->seedTracks_.push_back(xx);
      // }
    }
    else if (Config::seedCleaning == cleanSeedsPure)
    {
      m_event->use_seeds_from_cmsswtracks();
    }
    else if (Config::seedCleaning == cleanSeedsBadLabel)
    {
      m_event->clean_cms_seedtracks_badlabel();
    }
    else if (Config::seedCleaning != noCleaning)
    {
      std::cerr << "Specified reading cmssw seeds, but an incorrect seed cleaning option! Exiting..." << std::endl;
      exit(1);
    }

    seed_post_cleaning(m_event->seedTracks_);

    // in rare corner cases, seed tracks could be fully cleaned out: skip mapping if so
    if (m_event->seedTracks_.empty()) return;
  }
  else if (Config::seedInput == findSeeds)
  {
    // MIMI - doesnotwork
    // find_seeds();
  }
  else
  {
    std::cerr << "No input seed collection option selected!! Exiting..." << std::endl;
    exit(1);
  }

  // Do not refit cmssw seeds (this if was nested in fit_one_seed_set() until now).
  // Eventually we can add force-refit option.
  if (Config::seedInput != cmsswSeeds)
  {
    // MIMI - doesnotwork
    // fit_seeds();
  }
}


// Overlap hit truth dumper that was part of MkBuilder::quality_store_tracks()

// #define DUMP_OVERLAP_RTTS

    void quality_store_tracks(const EventOfCombCandidates &eoccs, TrackVec &tracks) {

#ifdef DUMP_OVERLAP_RTTS

      // SIMTRACK DUMPERS

      static bool first = true;
      if (first) {
        // ./mkFit ... | perl -ne 'if (/^ZZZ_OVERLAP/) { s/^ZZZ_OVERLAP //og; print; }' > ovlp.rtt
        printf("SSS_OVERLAP label/I:prod_type/I:is_findable/I:layer/I:pt/F:eta/F:phi/F\n");

        printf(
            "SSS_TRACK "
            "label/I:prod_type/I:is_findable/I:pt/F:eta/F:phi/F:nhit_sim/I:nlay_sim/I:novlp/I:novlp_pix/I:novlp_strip/"
            "I:novlp_stereo/I\n");

        first = false;
      }

      for (int i = 0; i < (int)m_event->simTracks_.size(); ++i) {
        Track &bb = m_event->simTracks_[i];

        if (bb.prodType() == Track::ProdType::Signal) {
          bb.sortHitsByLayer();

          int no = 0, npix = 0, nstrip = 0, nstereo = 0, prev_lay = -1, last_ovlp = -1;

          for (int hi = 0; hi < bb.nTotalHits(); ++hi) {
            HitOnTrack hot = bb.getHitOnTrack(hi);

            if (hot.layer == prev_lay && hot.layer != last_ovlp) {
              last_ovlp = hot.layer;

              ++no;

              const LayerInfo &li = Config::TrkInfo.m_layers[hot.layer];

              if (li.is_pixb_lyr() || li.is_pixe_lyr()) {
                ++npix;
              } else {
                ++nstrip;
              }

              if (li.is_stereo_lyr())
                ++nstereo;

              printf("SSS_OVERLAP %d %d %d %d %f %f %f\n",
                     bb.label(),
                     (int)bb.prodType(),
                     bb.isFindable(),
                     hot.layer,
                     bb.pT(),
                     bb.posEta(),
                     bb.posPhi());
            }
            prev_lay = hot.layer;
          }

          printf("SSS_TRACK %d %d %d %f %f %f %d %d %d %d %d %d\n",
                 bb.label(),
                 (int)bb.prodType(),
                 bb.isFindable(),
                 bb.pT(),
                 bb.momEta(),
                 bb.momPhi(),
                 bb.nTotalHits(),
                 bb.nUniqueLayers(),
                 no,
                 npix,
                 nstrip,
                 nstereo);
        }
      }

#endif

      int chi2_500_cnt = 0, chi2_nan_cnt = 0;

      for (int i = 0; i < eoccs.m_size; i++) {
        // See MT-RATS comment below.
        assert(!eoccs.m_candidates[i].empty() && "BackwardFitBH requires output tracks to align with seeds.");

        // take the first one!
        if (!eoccs.m_candidates[i].empty()) {
          const TrackCand &bcand = eoccs.m_candidates[i].front();

          if (std::isnan(bcand.chi2()))
            ++chi2_nan_cnt;
          if (bcand.chi2() > 500)
            ++chi2_500_cnt;

#ifdef DUMP_OVERLAP_RTTS
          // DUMP overlap hits
          int no_good = 0;
          int no_bad = 0;
          int no = 0;  // total, esp for tracks that don't have good label
          const HoTNode *hnp = &bcand.refLastHoTNode();
          while (true) {
            if (hnp->m_index_ovlp >= 0) {
              static bool first = true;
              if (first) {
                // ./mkFit ... | perl -ne 'if (/^ZZZ_OVERLAP/) { s/^ZZZ_OVERLAP //og; print; }' > ovlp.rtt
                printf(
                    "ZZZ_OVERLAP label/I:prod_type/I:is_findable/I:layer/I:pt/F:eta/F:phi/F:"
                    "chi2/F:chi2_ovlp/F:module/I:module_ovlp/I:hit_label/I:hit_label_ovlp/I\n");
                first = false;
              }

              auto &LoH = m_event_of_hits.m_layers_of_hits[hnp->m_hot.layer];

              const Hit &h = LoH.GetHit(hnp->m_hot.index);
              const MCHitInfo &mchi = m_event->simHitsInfo_[h.mcHitID()];
              const Hit &o = LoH.GetHit(hnp->m_index_ovlp);
              const MCHitInfo &mcoi = m_event->simHitsInfo_[o.mcHitID()];

              const TrackBase &bb =
                  (bcand.label() >= 0) ? (const TrackBase &)m_event->simTracks_[bcand.label()] : bcand;

              if (bcand.label() >= 0) {
                if (bcand.label() == mcoi.mcTrackID())
                  ++no_good;
                else
                  ++no_bad;
              }
              ++no;

              // label/I:can_idx/I:layer/I:pt/F:eta/F:phi/F:chi2/F:chi2_ovlp/F:module/I:module_ovlp/I:hit_label/I:hit_label_ovlp/I
              printf("ZZZ_OVERLAP %d %d %d %d %f %f %f %f %f %u %u %d %d\n",
                     bb.label(),
                     (int)bb.prodType(),
                     bb.isFindable(),
                     hnp->m_hot.layer,
                     bb.pT(),
                     bb.posEta(),
                     bb.posPhi(),
                     hnp->m_chi2,
                     hnp->m_chi2_ovlp,
                     h.detIDinLayer(),
                     o.detIDinLayer(),
                     mchi.mcTrackID(),
                     mcoi.mcTrackID());
            }

            if (hnp->m_prev_idx >= 0)
              hnp = &eoccs.m_candidates[i].m_hots[hnp->m_prev_idx];
            else
              break;
          }

          if (bcand.label() >= 0) {
            static bool first = true;
            if (first) {
              // ./mkFit ... | perl -ne 'if (/^ZZZ_TRACK/) { s/^ZZZ_TRACK //og; print; }' > track.rtt
              printf(
                  "ZZZ_TRACK "
                  "label/I:prod_type/I:is_findable/I:pt/F:eta/F:phi/F:nhit_sim/I:nlay_sim/I:nhit_rec/I:nhit_miss_rec/"
                  "I:novlp/I:novlp_good/I:novlp_bad/I\n");
              first = false;
            }

            const Track &bb = m_event->simTracks_[bcand.label()];

            printf("ZZZ_TRACK %d %d %d %f %f %f %d %d %d %d %d %d %d\n",
                   bb.label(),
                   (int)bb.prodType(),
                   bb.isFindable(),
                   bb.pT(),
                   bb.momEta(),
                   bb.momPhi(),
                   bb.nTotalHits(),
                   bb.nUniqueLayers(),
                   bcand.nFoundHits(),
                   bcand.nMissingHits(),
                   no,
                   no_good,
                   no_bad);
          }
            // DUMP END
#endif

          tracks.emplace_back(bcand.exportTrack());

#ifdef DEBUG_BACKWARD_FIT_BH
          printf("CHITRK %d %g %g %g %g %g\n",
                 bcand.nFoundHits(),
                 bcand.chi2(),
                 bcand.chi2() / (bcand.nFoundHits() * 3 - 6),
                 bcand.pT(),
                 bcand.momPhi(),
                 bcand.theta());
#endif
        }
      }

      if (!Config::silent && (chi2_500_cnt > 0 || chi2_nan_cnt > 0)) {
        std::lock_guard<std::mutex> printlock(Event::printmutex);
        printf("MkBuilder::quality_store_tracks bad track chi2 (backward fit?). is-nan=%d, gt-500=%d.\n",
               chi2_nan_cnt,
               chi2_500_cnt);
      }
    }

*/
