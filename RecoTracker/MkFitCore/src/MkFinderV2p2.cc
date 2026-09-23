#include "MkFinderV2p2.h"

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
// #include "FindingFoos.h"

#include "MkBins.h"

#include <algorithm>

#if defined(MKFIT_STANDALONE)
#include "RecoTracker/MkFitCore/standalone/Event.h"
#endif

#ifdef MKFIT_TRACE
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/DataFormats/RntConversions.h"
#endif

//#define DEBUG
#include "Debug.h"

namespace mkfit {

  // Per-layer policy counters. These say whether a mechanism FIRED, which is a
  // different question from whether it changed the physics -- a cut that costs
  // nothing because it never fires and a cut that costs nothing because it fires
  // and the candidate was doomed anyway look identical in quality-val. Summed
  // over threads and events; mkFit.cc prints and resets them with the quality-val
  // summary.
  V2p2PolicyCounters g_v2p2_policy_counters;

  // Starting point: the two directions carry the SAME numbers, and the hit bonus
  // and miss penalty are phase2:LstIntoPix's 30 and 8. That is on purpose -- the
  // first A/B then measures the SELECTION alone, with the arithmetic unchanged.
  // The head/body asymmetry these exist to express (outward, trailing holes are
  // at large radius and cheap; inward, they are at small radius and are the most
  // expensive holes there are) is the next knob, and it now has somewhere to be
  // turned.
  V2p2ScoreParams g_v2p2_score_fwd;
  V2p2ScoreParams g_v2p2_score_bkw;
  int g_v2p2_score_mode = 0;

  // Reduction cap, PER SUB-LAYER. Was MkBins::NEW_MAX_HIT, a compile-time 6 for
  // the whole detector. It sits UPSTREAM of the in-layer combinatorial search, so
  // it bounds what that search can ever see, and one number cannot be right
  // everywhere: overlap availability runs from 2-3 % of pixel-barrel crossings to
  // 50-56 % of TFPX. Runtime so the question costs one build; per layer is where
  // it should end up.
  int g_v2p2_max_presel_hits = MkBins::NEW_MAX_HIT;

  // Most hits one in-layer path may take. See kMaxSecDepthMax below.
  int g_v2p2_max_sec_depth = 4;

  bool g_v2p2_force_mc = false;
  float g_v2p2_extra_dq = 3.0f;
  bool  g_v2p2_surface_q = true;


  void V2p2PolicyCounters::reset() {
    n_quadrant_skip = 0; n_stop_minpt = 0; n_stop_looper = 0;
    n_wsr_inside = 0; n_wsr_edge = 0; n_wsr_outside = 0; n_wsr_in_gap = 0;
    n_layer_skipped = 0;
    n_hole = 0; n_hot_edge = 0; n_hot_gap = 0; n_stop_holes = 0;
    n_ccand_retired = 0;
    n_sec_nodes = 0; n_sec_deep = 0; n_path_taken = 0; n_extra_hits = 0;
    n_sel_entries = 0; n_sel_kept = 0; n_selections = 0;
    n_same_module = 0; n_diff_module = 0; n_same_module_vetoed = 0;
    n_hole_slot_reserved = 0; n_best_short_offered = 0; n_best_short_taken = 0;
  }

  void V2p2PolicyCounters::print(const char *tag) const {
    const long n_wsr = n_wsr_inside + n_wsr_edge + n_wsr_outside;
    // Denominator is layer searches that got as far as a WSR verdict, i.e. after
    // the pull-in skips. Those are reported separately because they are a
    // different population: a candidate skipped at pull-in never became one.
    const double f = n_wsr > 0 ? 100.0 / n_wsr : 0.0;
    printf("MkFinderV2p2 layer policy (%s):\n"
           "  pull-in    : rz-quadrant skip %ld, stop-minPt %ld, stop-looper %ld\n"
           "  WSR of %ld : inside %ld (%.1f%%), edge %ld (%.1f%%), outside %ld (%.1f%%), in-gap %ld\n"
           "  no-hit HoT : hole %ld, edge %ld, gap %ld, stop-out-of-holes %ld; layers skipped %ld\n"
           "  retired    : %ld CombCandidates\n"
           "  in-layer   : %ld tree nodes (%ld at depth >= 2), %ld paths taken, %ld extra hits\n"
           "  selection  : %ld of them, %ld competitors -> %ld kept (%.2f -> %.2f per seed)\n"
           "  extra hits : %ld from another module (overlap), %ld from the SAME module (%.1f%%)"
           ", %ld same-module extensions vetoed\n"
           "  hole slots : %ld reserved for an outranked decliner\n"
           "  best-short : %ld stopped cands left the beam, %ld became the seed's best short\n",
           tag,
           n_quadrant_skip.load(), n_stop_minpt.load(), n_stop_looper.load(),
           n_wsr, n_wsr_inside.load(), f * n_wsr_inside, n_wsr_edge.load(), f * n_wsr_edge,
           n_wsr_outside.load(), f * n_wsr_outside, n_wsr_in_gap.load(),
           n_hole.load(), n_hot_edge.load(), n_hot_gap.load(), n_stop_holes.load(),
           n_layer_skipped.load(),
           n_ccand_retired.load(),
           n_sec_nodes.load(), n_sec_deep.load(), n_path_taken.load(), n_extra_hits.load(),
           n_selections.load(), n_sel_entries.load(), n_sel_kept.load(),
           n_selections > 0 ? (double) n_sel_entries / n_selections : 0.0,
           n_selections > 0 ? (double) n_sel_kept / n_selections : 0.0,
           n_diff_module.load(), n_same_module.load(),
           (n_same_module + n_diff_module) > 0 ?
             100.0 * n_same_module / (n_same_module + n_diff_module) : 0.0,
           n_same_module_vetoed.load(), n_hole_slot_reserved.load(),
           n_best_short_offered.load(), n_best_short_taken.load());
  }

  //------------------------------------------------------------------------------
  // Setup variables for full processing of a batch of CombCanditates

  void MkFinderV2p2::setup(const MkJob *job, EventOfCombCandidates &eoccs, int seed_begin, int seed_end,
                           SteeringParams::iterator &sp_it, const Event *ev) {
    mp_job = job;
    mp_steeringparams_iter = &sp_it;
    mp_event = ev;

    m_batch_mgr.setup(eoccs, seed_begin, seed_end);

    // Assure all ccands are dormant and have a single tcand (?)
    for (auto &ccand : m_batch_mgr) {
      if (ccand.state() != CombCandidate::Dormant)
        throw std::runtime_error("CombCandidate is expected to be dormant "
                                 "on entry into combinatorial search.");
      if (ccand.size() != 1)
        throw std::runtime_error("CombCandidate is expected to have a single "
                                 "TrackCand on entry into combinatorial search.");
    }
    m_batch_mgr.m_n_dormant = m_batch_mgr.n_total();

    // AAAA reserve based on iter-param max-cands and N-sub-layer / skipped hit combinatorials.
  }

  void MkFinderV2p2::release() {
    m_batch_mgr.release();
    mp_job = nullptr;
    mp_steeringparams_iter = nullptr;
    mp_event = nullptr;

    // XXXX this might need to be achieved through other means ... crashing when going to
    // XXXX multi-event setup as old ccands have stayed in.
    // See another XXXX about finalization below in end_layer()
    m_active_ccreps.clear();

    // Kept as a belt-and-braces reset across finder REUSE: g_exe_ctx.m_findersV2p2
    // recycles MkFinderV2p2 objects across seed blocks and events, and the queue
    // holds PrimTCandRep* into CCandRep::m_primTCs, which end_layer() clears -- so
    // an entry surviving into the next setup() is a dangling pointer, which is
    // what "this clear got us from 19 to 80 ttbar events" was. process_layer()
    // now asserts the queue is empty on exit, so if that assert never fires this
    // clear is provably redundant and can go.
    m_pre_select_queue.clear();
  }

  //------------------------------------------------------------------------------
  // Helpers.

  int MkFinderV2p2::awaken_candidates() {
    // Awaken dormant ccands with hits on current layer for processing on the next layer.
    int count = 0;
    const LayerControl &lc = mp_steeringparams_iter->layer_control();
    for (auto &ccand : m_batch_mgr) {

      // Answered (2026-09-09) -- picking up *at* the pair, as now, is right, and
      // all three questions below collapse into one mechanism once the two
      // sub-layers' hits are merged into a single path-length-ordered list:
      //
      //   A seed whose last hit already sits in one layer of the pair is just
      //   the cursor-initialisation case, i.e. the same machinery as hit
      //   skipping. And no cursor arithmetic is even needed: the candidate's
      //   state *is* at that hit, so propagating from there puts it at
      //   dalpha = 0 and everything further along at dalpha > 0. Extending
      //   forward-only therefore excludes it structurally -- the same canonical
      //   ordering that makes each hit subset reachable exactly once -- so
      //   "don't re-find the pre-existing hit" needs no check at all.
      //
      //   Two details: take dalpha > eps rather than > 0, since the hit itself
      //   sits at 0 +- eps while a genuine *overlap partner* in the same
      //   sub-layer sits at essentially the same path length and is exactly the
      //   "recuperate extra hits" case worth keeping (eps below the
      //   module-to-module path-length separation, above float noise). And
      //   "just move on" is the degenerate case of the same code path, not a
      //   decision: if the existing hit is already past every pre-selected hit
      //   the list after the cursor is empty and the pair is a no-op.
      //
      //   So pickupLayer() matching either sub-layer (as below) is what we want.
      //   If the goal is instead to *re-evaluate* the seed's hit rather than
      //   extend past it, that is a different question with an existing idiom:
      //   chop the hit off and let the search re-find it, as clear_out_pixel_hits
      //   does in RunLSTintoPix(), rather than special-casing the pair.
      //
      // What *does* differ for these candidates is the propagation setup, not the
      // hit bookkeeping: they sit INSIDE the pair's bounding shell, so their rin
      // crossing is at negative dalpha -- they are mid-layer, not at the entry
      // edge. That still needs no branch: propagate_to_r() takes the crossing with
      // the smaller |alpha|, so it finds the backward rin crossing, and since both
      // sp1 and sp2 dalphas are measured from the candidate's own state, dalpha = 0
      // *is* the existing hit. The only cost is that the window then also covers
      // the part of the shell behind the candidate -- more hits scanned, no wrong
      // answers.
      //
      // So the unclamped version is correct as-is, and batching these separately
      // is a pure optimisation. Its right shape is a *different MkRZLimits*, not a
      // branch in the kernel: clamp the shell to [r_hit, rout], give them their own
      // pre-select queue, run the same kernel on it -- partition the work by which
      // parameter set it needs, the same way the WSR near-miss / clear-miss split
      // wants to. Classification is free right here, since we already test both
      // sub-layers below.
      //
      // Worth measuring before building it: how many candidates per layer actually
      // pick up at m_layer_sec. A separate batch means partly-empty Matriplex
      // lanes, so a trickle is cheaper handled by the wider window, while a large
      // fraction (plausible for T5 seeds starting in TOB) makes the partition pay.
      //
      // See RecoTracker/CLAUDE.md, "A seed whose last hit is already in one
      // layer of the pair".

      if (ccand.state() == CombCandidate::Dormant &&
          (ccand.pickupLayer() == lc.m_layer || ccand.pickupLayer() == lc.m_layer_sec)) {
        ccand.setState(CombCandidate::Finding);
        // Start the in-flight score accumulator. Any constant seed term is
        // common to every candidate of this seed and so cancels in the per-seed
        // selection; the cross-seed comparison is done at the end, on the
        // official score.
        if (Config::v2p2InLayerComb)
          for (int ic = 0; ic < (int) ccand.size(); ++ic)
            ccand[ic].setScore(0.0f);
        // NB: auto& -- std::list::emplace_back returns a *reference*. With a plain
        // `auto` this copy-constructed the CCandRep (it is copyable: its members
        // are references), so the m_seed_mc_label / m_mc_layer_sequence stores
        // below landed on a temporary and the list element kept -1. Harmless
        // while those three MKFIT_STANDALONE fields are write-only, silently
        // wrong the moment they are wired into the trace.
        auto &ccrep = m_active_ccreps.emplace_back(ccand);
        dprintf("MkFinderV2p2::awaken_candidates dummy printout N_TrackCands=%d\n",
               (int) ccand.size());
        ++count;

        #if defined(MKFIT_STANDALONE)
        // XXXX hmmh, i could really just get it where i need it.
        auto sifh = mp_event->simInfoForCurrentSeed(ccand.seed_origin_index());
        ccrep.m_seed_mc_label = sifh.label;
        ccrep.m_mc_layer_sequence = 0;
        // ccrep.m_n_mc_hits_in_current_layer set later
        #endif
      }
    }
    m_batch_mgr.m_n_dormant -= count;
    m_batch_mgr.m_n_finding += count;

    dprintf("MkFinderV2p2::awaken_candidates woke up %d cands\n", count);

    return count;
  }

  //------------------------------------------------------------------------------
  // stop_cuts_at_pickup()
  //
  // The two reasons V1/V2 stop a candidate before a layer is even searched. Both
  // read only the candidate's own state, which is why they belong at pull-in.
  //
  // The looper stop is the older of the two and the physics is worth restating:
  // once the transverse angle between position and momentum reaches pi/2 - 0.2
  // (78.5 deg) the track is crossing modules at grazing incidence, the clusters
  // get wide, and further hits SPOIL the measurement rather than improve it. It
  // is a track-level stopping condition, not a per-step guard.
  //
  // The band test is on |posPhi - momPhi| with both wrapped to (-pi, pi], so an
  // angle past the limit appears either as dphi > kMaxAngPosMom or, when the pair
  // straddles the +-pi branch cut, as dphi < TwoPI - kMaxAngPosMom. The upper
  // bound is the wrap image of the lower and is DERIVED from it here rather than
  // written as a second literal -- V1 carried a hardcoded 4.512f, which is
  // pi + kMaxAngPosMom and lets the 78.5-101.5 deg band escape whenever the pair
  // straddles. As written the test is exactly equivalent to
  // cos(momPhi - posPhi) < sin(0.2).
  //
  // Both stops RECORD their reason on the TrackCand: "stopped" alone loses the
  // fact that a looper is a looper, and loopers are wanted afterwards for the
  // phase-2 timing detectors and HGCal, which sit past the radius at which the
  // tracker gives up on them.
  //----------------------------------------------------------------------------

  TrackCand::StopReason_e MkFinderV2p2::stop_cuts_at_pickup(const TrackCand &tc) const {
    const SteeringParams::iterator &spi = *mp_steeringparams_iter;
    const auto &iter_params = (spi.type() == SteeringParams::IT_BkwSearch) ? mp_job->params_bks()
                                                                          : mp_job->params();

    if (tc.pT() < iter_params.minPtCut)
      return TrackCand::SR_MinPt;

    // Forward search only: going inward the track is leaving the turning region,
    // so a large angle there is not evidence that it is about to curl up.
    if (spi.type() == SteeringParams::IT_FwdSearch && tc.pT() < 1.2f && tc.posRsq() > 625.0f) {
      constexpr float kMaxAngPosMom = Const::PIOver2 - 0.2f;
      const float dphi = std::abs(tc.posPhi() - tc.momPhi());
      if (dphi > kMaxAngPosMom && dphi < Const::TwoPI - kMaxAngPosMom)
        return TrackCand::SR_Looper;
    }

    return TrackCand::SR_NotStopped;
  }

  //----------------------------------------------------------------------------
  // fake_hit_index()
  //
  // What to record for a candidate that came out of a layer with no hit. V1's
  // order, and the order matters: the hole limits decide miss-vs-stop first, and
  // then the WSR overrides, because a layer the track only clipped must not count
  // against maxHolesPerCand at all.
  //----------------------------------------------------------------------------

  int MkFinderV2p2::fake_hit_index(const TrackCand &tc, const WSR_Result &wsr) const {
    const SteeringParams::iterator &spi = *mp_steeringparams_iter;
    const auto &iter_params = (spi.type() == SteeringParams::IT_BkwSearch) ? mp_job->params_bks()
                                                                          : mp_job->params();

    int fake_hit_idx = Hit::kHitMissIdx;

    if (Config::v2p2UseHoleLimits &&
        (tc.nAllMinusOneHits() >= iter_params.maxHolesPerCand ||
         tc.nTailMinusOneHits() >= iter_params.maxConsecHoles))
      fake_hit_idx = Hit::kHitStopIdx;

    if (Config::v2p2UseWsr) {
      if (wsr.m_wsr == WSR_Edge)
        fake_hit_idx = Hit::kHitEdgeIdx;
      else if (wsr.m_in_gap)
        fake_hit_idx = Hit::kHitInGapIdx;
    }

    return fake_hit_idx;
  }

  //------------------------------------------------------------------------------
  // Per layer initialization / cleanup tasks,

  void MkFinderV2p2::begin_layer() {
    // debug = true; // to be disabled at the end of end_layer()

    // Count number of awakend ccands and number of non-stopped tcands.
    // Well, I actually know n ccands.

    dprintf("MkFinderV2p2::begin_layer Expecting %d active ccands\n", (int) m_active_ccreps.size());

// #ifdef RNT_DUMP_MkF_SelHitIdcs
//     // clang-format off
//     const IterationConfig &IC = mp_job->m_iter_config;
//     SteeringParams::iterator &spi = *mp_steeringparams_iter;
//     const LayerOfHits &L = mp_job->m_event_of_hits[spi->m_layer];
//     const LayerInfo &LI = L.layer_info();
//     rnt_shi.ResetH();
//     rnt_shi.ResetF();
//     *rnt_shi.h = {mp_event->evtID(), IC.m_iteration_index, IC.m_track_algorithm,
//                   spi.region(), L.layer_id(),
//                   L.is_barrel() ? LI.rin() : LI.zmin(), LI.is_barrel() ? LI.rout() : LI.zmax(),
//                   L.is_barrel(), L.is_pixel(), L.is_stereo()};
//     *rnt_shi.f = *rnt_shi.h;

//     // Unlike in MkFinder, we process all active cands in one go.
//     // And actually do not use the inner indices as we assign CandInfo* into PrimTCand.
//     rnt_shi.InnerIdcsReset((int) m_active_ccreps.size());
//     // clang-format on
// #endif

#ifdef DEBUG
    { int i = 1;
      for (auto &ccrep : m_active_ccreps) {
        dprintf("  %2d. seed-idx=%d, n_tcands=%d\n", i,
               ccrep.m_ccand.seed_origin_index(), (int) ccrep.m_ccand.size());
        ++i;
      }
    }
#endif

    m_batch_mgr.reset_for_new_layer();
    m_active_ccreps_pos = m_active_ccreps.begin();

    { // Setup m_rz_limits.
      SteeringParams::iterator &spi = *mp_steeringparams_iter;

      const bool is_double_layer = spi->has_second_layer();
      const LayerInfo &LI_p = mp_job->m_trk_info[ spi->m_layer ];
      const LayerInfo &LI_s = mp_job->m_trk_info[ is_double_layer ? spi->m_layer_sec : 0 ];

      if (is_double_layer) {
        assert(LI_p.is_barrel() == LI_s.is_barrel() );
        m_rz_limits.setup(LI_p, LI_s, spi.is_outward());
      } else {
        m_rz_limits.setup(LI_p, spi.is_outward());
      }
    }

    // Hmmh, nothing to really do here, is it?

    // Initialize best short (?) - to what? How was it before? The seed itself?
    // Or let this happen after the first layer is processed?
    // Score is set to worst-possible elsewhere.
  }

  void MkFinderV2p2::begin_next_Ccrep_in_layer() {
    CCandRep &ccrep = * m_active_ccreps_pos;
    CombCandidate &ccand = ccrep.m_ccand;

    // QQQQ reserve to CombCand.capacity already done CCandRep ctor.
    // We will probably need a variable number per layer / layer pair.
    // But this will be first relevant / handled elsewhere.
    ccrep.m_primTCs.reserve(ccand.size());

    for (int ic = 0; ic < (int) ccand.size(); ++ic) {
      TrackCand &tcand = ccand[ic];

      if (tcand.getLastHitIdx() == Hit::kHitStopIdx)
        continue;

      // The two candidate-stopping cuts, at the earliest point that can take
      // them: both need only the candidate's own state, so they belong here,
      // alongside the rz_quadrant_check below, and not at end of layer. V1/V2
      // take them in MkBuilder::find_tracks_unroll_candidates(); v2p2 took
      // neither, so it ran with no minPtCut and no looper stop at all.
      if (Config::v2p2UseStopCuts) {
        const TrackCand::StopReason_e sr = stop_cuts_at_pickup(tcand);
        if (sr != TrackCand::SR_NotStopped) {
          tcand.setStopReason(sr);
          tcand.addHitIdx(Hit::kHitStopIdx, m_rz_limits.layer_info_1().layer_id(), 0.0f);
          ++(sr == TrackCand::SR_MinPt ? g_v2p2_policy_counters.n_stop_minpt
                                       : g_v2p2_policy_counters.n_stop_looper);
          continue;
        }
      }

      // XXXX Should one do rough "layer already passed" pre-check here?
      // Or in pre-select, where we engage MkBins ... but there I loose a vector slot.
      // Let's try.
      //
      // Note this skips the layer WITHOUT recording anything, which is the same
      // treatment WSR_Outside gets in process_kalman_results() -- it is the same
      // statement (the track does not reach this layer), made more cheaply and
      // with a much coarser test.

      if ( ! m_rz_limits.rz_quadrant_check(tcand.z(), tcand.pz())) {
        ++g_v2p2_policy_counters.n_quadrant_skip;
        continue;
      }

      // Create and Register PrimTCandRep for processing.
      // The CCandRep vector has the capacity for N_max_cands.
      {
        PrimTCandRep &ptc = ccrep.m_primTCs.emplace_back( &ccrep, ic );
        m_pre_select_queue.push_back(&ptc);
      }
    }
    ++m_active_ccreps_pos;

    #if defined(MKFIT_STANDALONE)
    {
      ccrep.m_n_mc_hits_in_layer = mp_event->countSimHitsInLayer(ccrep.m_seed_mc_label, mp_steeringparams_iter->layer());
      // ccrep.m_n_mc_hits_in_layer_sec = mp_event->countSimHitsInLayer(ccrep.m_seed_mc_label, mp_steeringparams_iter->layer_sec());
      if (ccrep.m_n_mc_hits_in_layer > 0)
        ++ccrep.m_mc_layer_sequence;
    }
    #endif
  }

  // void MkFinderV2p2::process_pre_select() -- below in the "complex stuff" section

  void MkFinderV2p2::end_layer() {
    // Stop tracks -- pT / apogee / missing layers.
    // Choose best-short.
    // Figure out what to copy back to EventOfCombCandidates.

    // clear out ccands -- well, might keep them -- just flush the tcands out of hot-tub
    int count = 0;
    auto ai = m_active_ccreps.begin();
    while (ai != m_active_ccreps.end()) {

      if (Config::v2p2InLayerComb)
        select_and_materialise(*ai);

      // QQQQ should attempt to reuse the PrimTCandReps
      ai->m_primTCs.clear();
      ai->m_sec_nodes.clear();

      // Retiring a CombCandidate is the one decision that genuinely needs the
      // layer's outcome, which is why it is here and the stopping cuts are at
      // pull-in. A CombCandidate is done when every TrackCand under it has been
      // stopped -- by minPtCut, by the looper cut, or by running out of hole
      // budget. Until the stopping cuts existed nothing could ever stop, so this
      // was hardcoded false and m_n_finished stayed 0 for the whole event.
      bool is_finished = false;
      if (Config::v2p2UseStopCuts || Config::v2p2UseHoleLimits) {
        CombCandidate &cc = ai->m_ccand;
        is_finished = true;
        for (int ic = 0; ic < (int) cc.size(); ++ic) {
          if (cc[ic].getLastHitIdx() != Hit::kHitStopIdx) {
            is_finished = false;
            break;
          }
        }
        if (is_finished) {
          cc.setState(CombCandidate::Finished);
          ++g_v2p2_policy_counters.n_ccand_retired;
        }
      }
      if (is_finished) {
        auto bi = ai++;
        m_active_ccreps.erase(bi);
        ++count;
        continue;
      }
      ++ai;
    }
    m_batch_mgr.m_n_finding -= count;
    m_batch_mgr.m_n_finished += count;

    // One rewind for the whole layer, keeping capacity: after a few layers the
    // arena is at high water and stops allocating.
    m_sec_arena.clear();

    m_rz_limits.reset();

    dprintf("MkFinderV2p2::end_layer %d cands finished\n", count);

    if (m_batch_mgr.has_dormant_ccands())
      awaken_candidates();

    // debug = false;
  }

  //------------------------------------------------------------------------------
  // The main processing function -- process_layer()

  void MkFinderV2p2::process_layer() {

    // The live pipeline is two stages: pull CombCandidates into the layer (one
    // PrimTCandRep per surviving TrackCand), then drain the pre-select queue in
    // NN-sized batches. begin_next_Ccrep_in_layer() is held back once the queue
    // has NN entries so the Matriplex batches run full rather than ragged.
    //
    // THE INTENDED PIPELINE HAS TWO MORE STAGES and neither exists yet. They were
    // written here as goto-labelled blocks guarded by flags that were never
    // assigned, i.e. as dead code; the sketch is kept as this comment instead so
    // it cannot be mistaken for behaviour:
    //
    //   pre_select      -- as now, but emitting SecTCandReps rather than
    //                      updating a single best hit per PrimTCandRep;
    //   prop_n_kalman   -- propagate + Kalman the NN batch of SecTCandReps and
    //                      process the results against their PrimTCandReps. That
    //                      can make later-stage SecTCandReps available (the
    //                      in-layer combinatorial expansion), finish or kill
    //                      PrimTCandReps, and thereby finish whole CCandReps;
    //   Ccs_finalize    -- for a CCandRep whose PrimTCandReps are all done,
    //                      select/merge its SecTCandRep leaves into the
    //                      CombCandidate and release its slots immediately,
    //                      rather than waiting for end_layer().
    //
    // The ordering constraint the goto version encoded, and which the eventual
    // loop still needs: finalize before starting new work, so slots are freed
    // before they are asked for.
    while (any_Ccreps_to_begin()) {
      while ( ! enough_work_for_pre_select() && any_Ccreps_to_begin()) {
        begin_next_Ccrep_in_layer();
      }
      while (enough_work_for_pre_select() ||
             ( ! any_Ccreps_to_begin() && any_work_for_pre_select())) {
        process_pre_select();
      }
    }

    // The queue is drained by construction: the inner loop's second clause runs
    // until empty once there is nothing left to pull in. release() used to clear
    // it unconditionally with a "something stays in (shouldn't)" note; assert
    // instead, so the claim is either true or fails loudly.
    assert(m_pre_select_queue.empty() && "pre-select queue not drained by process_layer()");
  }

  //============================================================================
  // More complex functions -- to separate them from the main "logic" flow
  //============================================================================

  //----------------------------------------------------------------------------
  // process_pre_select()
  // Propagate to layer edges, calculate layer-of-hits bin ranges and
  // determine candidate hits.
  //----------------------------------------------------------------------------

  //----------------------------------------------------------------------------
  // process_pre_select() -- the layer pass, in phases.
  //
  // One call handles up to NN PrimTCandReps together. It used to be a single
  // ~360-line function; the phases below are the ones the abandoned procedural
  // sketch at the bottom of this file already names, now made to work by giving
  // them an explicit carrier (LayerBatch) instead of a dozen arguments each.
  // Bodies were moved VERBATIM -- each phase binds local references with the
  // original names, so this is a re-grouping and not a rewrite.
  //
  // Two batch widths are in play, which is most of why the single function was
  // hard to follow: LayerBatch is NN CANDIDATES wide, HitBatch is NN
  // (candidate, hit) PAIRS wide.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::process_pre_select() {
    LayerBatch b;

    select_hits_prepare(b);       // pop the queue, propagate to the layer edges
    determine_search_windows(b);  // covariance -> dphi/dq windows -> bin ranges
    select_hits(b);               // walk the bins, pre-select, reduce in a pqueue
    prepare_kalman_workload(b);   // pqueue -> m_layer_hits, step-ordered
    if (Config::v2p2InLayerComb) {
      // No materialisation here: the paths are left in the arena and everything
      // competes at end of layer, in select_and_materialise().
      expand_in_layer(b);         // grow the SecTCandRep tree over the ordered hits
    } else {
      kalman_update(b);           // propagate to each module plane + update
      process_kalman_results(b);  // best-hit acceptance into the TrackCand
    }
  }

  //----------------------------------------------------------------------------
  // Phase 1 -- which candidates, and where do they meet the layer.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::select_hits_prepare(LayerBatch &b) {
    MkBins &B = b.B;
    MkBinTrackCovExtract &TCE = b.TCE;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;

    const int N_proc = b.N_proc = std::min(NN, (int) m_pre_select_queue.size());
    B.m_n_proc = N_proc;   // MkBins used to be constructed with it

    dprintf("MkFinderV2p2::process_pre_select work queue is %d, would process %d of them (NN=%d)\n",
            (int) m_pre_select_queue.size(), N_proc, NN);

    MPlexQF phi(0.0f);
    MPlexQI chg(0);

    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * m_pre_select_queue.front();
      prim_tcand_ptrs[i] = & ptc;
      TrackCand &tc = ptc.tcand();
      m_pre_select_queue.pop_front();

      // Copy in x, y,z, invpT, theta.
      B.m_isp.copyIn_partial_track_state(i, tc.state());
      // Extract track covariance at previous layer (for printouts only)
      TCE.m_cov_0_0[i] = tc.errors().At(0, 0);
      TCE.m_cov_0_1[i] = tc.errors().At(0, 1);
      TCE.m_cov_1_1[i] = tc.errors().At(1, 1);
      TCE.m_cov_2_2[i] = tc.errors().At(2, 2);
      phi[i] = tc.momPhi();
      chg[i] = tc.charge();
    }
    B.m_isp.init_momentum_vec_and_k(phi, chg);

    // Propagation ignoring the direction
    // B.prop_to_limits(m_rz_limits);

    // Propagation so point 1 is first edge hit, 2 the second
    B.prop_to_limits_in_order(m_rz_limits);

  }

  //----------------------------------------------------------------------------
  // Phase 2 -- the search windows.
  //
  // This is where the window covariance is built, i.e. where both of this
  // month's covariance fixes live (the dphi jacobian at min-r, and the surface
  // reference of dq). `pea` exists ONLY to produce the four position-block
  // elements MkBinTrackCovExtract reads; see the pea-removal note in CLAUDE.md.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::determine_search_windows(LayerBatch &b) {
    SteeringParams::iterator &spi = *mp_steeringparams_iter;
    const int N_proc = b.N_proc;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;
    MkBins &B = b.B;
    MkBinTrackCovExtract &TCE = b.TCE;
    MkBinLimits &BL_p = b.BL_p;
    MkBinLimits &BL_s = b.BL_s;
    mini_propagators::Hermite3D &H = b.H;
#ifdef MKFIT_TRACE
    int *tr_layersearch_ids = b.tr_layersearch_ids;
#endif

    PropErrsArgs pea;
    pea.prop_config = & mp_job->m_trk_info.prop_config();
    pea.tsXyz = mini_propagators::InitialStatePlex(B.m_sp2, B.m_isp);

    for (int i = 0; i < N_proc; ++i) {
      TrackCand &tc = prim_tcand_ptrs[i]->tcand();
      pea.item_begin();
      pea.load_state_err_chg(tc);
      pea.item_finished();
    }
    pea.compute_pars();
    pea.do_propagation_stuff();

    for (int i = 0; i < N_proc; ++i) {
      dprintf("%d: TCE %.4g %.4g %.4g %.4g  --   %.4g %.4g %.4g %.4g PROP\n", i,
        TCE.m_cov_0_0[i], TCE.m_cov_0_1[i], TCE.m_cov_1_1[i], TCE.m_cov_2_2[i],
        pea.propErr.At(i,0,0), pea.propErr.At(i,0,1), pea.propErr.At(i,1,1), pea.propErr.At(i,2,2));
    }

    TCE.init_from_track_errors( pea.propErr );

    // The final points are in B.m_sp2 ... sp.dalpha should be correct
    // Do full propagation + material.

    // Argh, do we really really need to do this?
    // Can't we just take the closest hit(s) regardless of preselection?
    // But we have no good measure of what good preselection would be.

    B.determine_bin_windows(TCE);

    // The WSR. It belongs to the sp1/sp2 question and could be answered right
    // after prop_to_limits_in_order(), except for one thing: the edge case needs
    // FUZZ, "does the track reach this layer" being a statement about a track
    // that has an error on it, and the only estimate of that error is dq_track,
    // which determine_bin_windows() has just computed. So it sits here, one line
    // later than the comment that used to ask for it.
    determine_wsr(b);

    B.find_bin_ranges(mp_job->m_event_of_hits[spi->m_layer], BL_p);

    if (m_rz_limits.m_is_double) {
      B.find_bin_ranges(mp_job->m_event_of_hits[spi->m_layer_sec], BL_s);
    }

    for (int i = 0; i < N_proc; ++i) {
      dprintf("%d: BinCheck Prim %c %+8.6f %+8.6f | %3d %3d || %+8.6f %+8.6f | %2d %2d\n",
              i, m_rz_limits.m_is_barrel ? 'B' : 'E',
              B.m_phi_center[i], B.m_phi_delta[i], BL_p.p1[i], BL_p.p2[i],
              B.m_q_min[i], B.m_q_max[i], BL_p.q1[i], BL_p.q2[i]);
      if (m_rz_limits.m_is_double) {
        dprintf("%d: BinCheck Sec  %c %+8.6f %+8.6f | %3d %3d || %+8.6f %+8.6f | %2d %2d\n",
                i, m_rz_limits.m_is_barrel ? 'B' : 'E',
                B.m_phi_center[i], B.m_phi_delta[i], BL_s.p1[i], BL_s.p2[i],
                B.m_q_min[i], B.m_q_max[i], BL_s.q1[i], BL_s.q2[i]);
      }
    }

    // The Hermite cubic through the two bounding-surface crossings. This is the
    // trajectory model the per-hit plane intersection below is solved on -- the
    // full propagation is only ever run once per candidate (pea, above) and then
    // again in the Kalman stage, never per scanned hit.
    // This might belong better somewhere else, TPrimCanRep? Calculation into minipropagators.

    namespace mp = mini_propagators;

    H.calculate_coeffs(B.m_sp1, B.m_sp2, B.m_isp.inv_k);

#ifdef MKFIT_TRACE
    for (int i = 0; i < N_proc; ++i) {
      TrLayerSearch ls;
      ls.state_id   = prim_tcand_ptrs[i]->tcand().m_trace_state_id;
      ls.layer      = spi->m_layer;
      ls.layer_sec  = m_rz_limits.m_is_double ? spi->m_layer_sec : -1;
      ls.is_barrel  = m_rz_limits.m_is_barrel;
      ls.is_outward = m_rz_limits.m_is_outward;

      ls.prop_entry = statep2propinfo(B.m_sp1, i);
      ls.prop_exit  = statep2propinfo(B.m_sp2, i);

      ls.phi_center = B.m_phi_center[i];
      ls.phi_delta  = B.m_phi_delta[i];
      ls.q_center   = B.m_q_center[i];
      ls.q_min      = B.m_q_min[i];
      ls.q_max      = B.m_q_max[i];
      ls.dphi_track = B.m_dphi_track[i];
      ls.dq_track   = B.m_dq_track[i];

      ls.cov_xx = TCE.m_cov_0_0[i];
      ls.cov_xy = TCE.m_cov_0_1[i];
      ls.cov_yy = TCE.m_cov_1_1[i];
      ls.cov_zz = TCE.m_cov_2_2[i];

      ls.p1 = BL_p.p1[i];  ls.p2 = BL_p.p2[i];
      ls.q1 = BL_p.q1[i];  ls.q2 = BL_p.q2[i];
      if (m_rz_limits.m_is_double) {
        ls.p1_sec = BL_s.p1[i];  ls.p2_sec = BL_s.p2[i];
        ls.q1_sec = BL_s.q1[i];  ls.q2_sec = BL_s.q2[i];
      }

      tr_layersearch_ids[i] = mp_event->trace_layersearch(std::move(ls)).id;
    }
#endif

#ifdef MKFIT_TRACE_PROP_COMPARE
    // Hermite vs helix in mid-layer, where the cubic is furthest from the two
    // endpoints it interpolates (at t = 0 and 1 it reproduces them by
    // construction, so those are roundoff checks only and not worth taking).
    // Evaluate the cubic at t = 0.5, then propagate exactly to the bounding
    // surface -- r in the barrel, z in the endcap -- that the cubic's midpoint
    // landed on, so both points sit on the same surface and the difference is a
    // pure in-surface deviation.
    {
      MPlexQF hx, hy, hz;
      H.evaluate(0.5f, hx, hy, hz);
      mp::StatePlex sp_mid;
      if (m_rz_limits.m_is_barrel) {
        MPlexQF r_mid = Matriplex::hypot(hx, hy);
        B.m_isp.propagate_to_r(mp::PA_Exact, r_mid, sp_mid, false, N_proc);
      } else {
        B.m_isp.propagate_to_z(mp::PA_Exact, hz, sp_mid, false, N_proc);
      }
      for (int i = 0; i < N_proc; ++i) {
        EVec3 dev(hx[i] - sp_mid.x[i], hy[i] - sp_mid.y[i], hz[i] - sp_mid.z[i]);
        dprintf("%d: HermiteMid dev %+9.6f %+9.6f %+9.6f  (|d| = %.6f, dalpha %.4f -> %.4f, derfac %.5f)\n",
                i, dev[0], dev[1], dev[2], std::sqrt(dev.Mag2()),
                B.m_sp1.dalpha[i], B.m_sp2.dalpha[i], H.m_Hderfac[i]);
#ifdef MKFIT_TRACE
        mp_event->tr_layersearch(tr_layersearch_ids[i]).hermite_mid_dev = dev;
#endif
      }
    }
#endif

  }

  //----------------------------------------------------------------------------
  // Phase 2b -- the within-sensitive-region verdict.
  //
  // This is not a port of V1's m_XWsrResult. V1 propagates to ONE surface (the
  // layer's propagate_to radius, or its z) and tests that single point's q
  // against the layer's q limits. v2p2 has both bounding-surface crossings, so
  // the question it can answer is the one that was actually wanted: layers are
  // finite SOLIDS, not infinite shells, and what matters is whether the SEGMENT
  // the track cuts through the shell lies in sensitive material.
  //
  // Two independent pieces of evidence, and they answer different halves:
  //
  //  - the two mini-propagator fail flags answer the RADIAL half for free. They
  //    already classify near miss from clear miss, and the classification was
  //    measured (10 events, inward T5 search, 135 flagged): 0 entry-only,
  //    36 exit-only, 99 both. Entry-only is not merely rare but geometrically
  //    impossible for a search that aims sp1 at the near surface -- missing the
  //    near surface while reaching the far one cannot happen. BOTH failing means
  //    the track turns around before the layer: a clear miss. EXIT-only means it
  //    gets in and turns around inside: a near miss, so at best an edge.
  //    propagate_to_z is closed form and cannot fail, so in the endcap this half
  //    is empty and the q test carries everything -- which is right, since for a
  //    disc "does the track reach it" IS the r question.
  //
  //  - the q extent answers the other half. m_q_min / m_q_max are the min and max
  //    of the two crossings' q with no fuzz applied, so they are the segment.
  //
  // The fuzz is the track's own dq at kWsrNSigma sigma. Being generous here is
  // the safe direction: the cost of calling a crossing Edge when it was Inside is
  // that a genuine hole is not counted, while the cost of calling it Inside when
  // it was Edge is a hole charged against a candidate that never crossed
  // anything, and the second is what the hole limits act on.
  //
  // "Inside" deliberately requires the WHOLE segment to be inside. A track that
  // spans the layer's entire q extent therefore reads as Edge, which is
  // conservative rather than exact -- it did cross sensitive material. Tightening
  // that needs the intersection of the segment with the layer rather than its
  // containment, and there is no measurement yet saying it matters.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::determine_wsr(LayerBatch &b) {
    // m_dq_track is 3 sigma (MkBins::determine_bin_windows), so this converts to
    // the 5-7 sigma band the edge test wants. 5 sigma is the low end of that;
    // it has not been scanned.
    constexpr float kWsrNSigma = 5.0f;
    constexpr float kDqTrackToNSigma = kWsrNSigma / 3.0f;

    const int N_proc = b.N_proc;
    MkBins &B = b.B;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;

    const bool is_barrel = m_rz_limits.m_is_barrel;
    const float q_lo = is_barrel ? m_rz_limits.m_zmin : m_rz_limits.m_rin;
    const float q_hi = is_barrel ? m_rz_limits.m_zmax : m_rz_limits.m_rout;
    const LayerInfo &LI = m_rz_limits.layer_info_1();

    for (int i = 0; i < N_proc; ++i) {
      const bool f_entry = B.m_sp1.fail_flag[i] != 0;
      const bool f_exit  = B.m_sp2.fail_flag[i] != 0;

      WSR_Result w;

      if (f_entry && f_exit) {
        // Clear miss: the track does not reach the layer at all.
        w = WSR_Result(WSR_Outside, false);
      } else {
        const float dq = kDqTrackToNSigma * B.m_dq_track[i];
        const float qa = B.m_q_min[i], qb = B.m_q_max[i];

        if (qb < q_lo - dq || qa > q_hi + dq)
          w = WSR_Result(WSR_Outside, false);
        else if (qa > q_lo + dq && qb < q_hi - dq)
          w = WSR_Result(WSR_Inside, false);
        else
          w = WSR_Result(WSR_Edge, false);

        // Near miss -- in the layer, but turning round inside it.
        if ((f_entry || f_exit) && w.m_wsr == WSR_Inside)
          w.m_wsr = WSR_Edge;

        // Endcap discs have a hole at small r. Sitting wholly inside it is a
        // miss with a reason; touching it is an edge, and either way m_in_gap
        // records that the absence of a hit is explained.
        if (!is_barrel && LI.has_r_range_hole() && w.m_wsr != WSR_Outside) {
          const float h_lo = LI.hole_r_min(), h_hi = LI.hole_r_max();
          if (qa > h_lo + dq && qb < h_hi - dq) {
            w = WSR_Result(WSR_Outside, true);
          } else if (qb > h_lo - dq && qa < h_hi + dq) {
            w.m_wsr = WSR_Edge;
            w.m_in_gap = true;
          }
        }
      }

      prim_tcand_ptrs[i]->m_wsr = w;

      switch (w.m_wsr) {
        case WSR_Inside:  ++g_v2p2_policy_counters.n_wsr_inside;  break;
        case WSR_Edge:    ++g_v2p2_policy_counters.n_wsr_edge;    break;
        case WSR_Outside: ++g_v2p2_policy_counters.n_wsr_outside; break;
        default: break;
      }
      if (w.m_in_gap)
        ++g_v2p2_policy_counters.n_wsr_in_gap;

      dprintf("%d: WSR %d in_gap %d  (q %.3f - %.3f vs layer %.3f - %.3f, dq3 %.4f, fail %d/%d)\n",
              i, w.m_wsr, (int) w.m_in_gap, B.m_q_min[i], B.m_q_max[i], q_lo, q_hi,
              B.m_dq_track[i], B.m_sp1.fail_flag[i], B.m_sp2.fail_flag[i]);
    }
  }

  //----------------------------------------------------------------------------
  // Phase 3 -- walk the bin ranges and pre-select hits.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::select_hits(LayerBatch &b) {
    SteeringParams::iterator &spi = *mp_steeringparams_iter;
    const int N_proc = b.N_proc;
    MkBins &B = b.B;
    MkBinLimits &BL_p = b.BL_p;
#ifdef MKFIT_TRACE
    int *tr_layersearch_ids = b.tr_layersearch_ids;
#endif

    // Prototype for extract hits

    HitBatch hb;
    int &fill_pos = hb.fill_pos;
    auto &is_plex = hb.is_plex;
    auto &prim_idcs = hb.prim_idcs;
    auto &hit_idcs = hb.hit_idcs;
    auto &hit_orig_idcs = hb.hit_orig_idcs;


    // Both sub-layers, primary then secondary, through the same block. Which one
    // a hit came from is a FILL-SIDE detail: the two feed separate reduction
    // queues, so each sensor keeps its own budget, and prepare_kalman_workload()
    // then merges them into one step-ordered list. Nothing downstream asks which
    // sensor a hit is from -- the per-hit q_half_length already carries what the
    // P/S distinction is worth, and it keeps working in a 2S stack where a stereo
    // bit would be meaningless.
    const int n_sub = m_rz_limits.m_is_double ? 2 : 1;
    for (int sub = 0; sub < n_sub; ++sub) {
      const bool is_sec_layer = (sub == 1);
      const int lay = is_sec_layer ? spi->m_layer_sec : spi->m_layer;
      const auto &L = mp_job->m_event_of_hits[lay];
      const auto &iteration_hit_mask = mp_job->get_mask_for_layer(lay);
      const auto &BL = is_sec_layer ? b.BL_s : BL_p;
      fill_pos = 0;

      for (int i = 0; i < N_proc; ++i) {

        using bidx_t = LayerOfHits::bin_index_t;
        using bcnt_t = LayerOfHits::bin_content_t;

        // The track does not reach this layer -- scanning its bins would be pure
        // waste, and anything it found would be a hit on a layer it never
        // crossed. Measured over 10 events on the inward search: the flagged
        // searches accounted for 27 % of all scanned hits and yielded zero
        // MC-matched accepted hits.
        if (Config::v2p2UseWsr && b.ptc[i]->m_wsr.m_wsr == WSR_Outside)
          continue;

        for (bidx_t qi = BL.q1[i]; qi != BL.q2[i]; ++qi) {
          for (bidx_t pi = BL.p1[i]; pi != BL.p2[i]; pi = L.phiMaskApply(pi + 1)) {

            // Dead regions -- Limit to central Q-bin ???
            // if (qi == qb && L.isBinDead(pi, qi) == true) {
            //   dprint("dead module for track in layer=" << L.layer_id() << " qb=" << qi << " pi=" << pi
            //                                            << " q=" << B.q_c[itrack] << " phi=" << B.phi_c[itrack]);
            //   m_XWsrResult[itrack].m_in_gap = true;
            // }

            auto pbi = L.phiQBinContent(pi, qi);
            for (bcnt_t hi = pbi.begin(); hi < pbi.end(); ++hi) {

              const unsigned int hi_orig = L.getOriginalHitIndex(hi);

              dprintf(" %d: P_HIT %3u %4u %5u [%5u]  %6.3f %6.3f %6.3f\n",
                i, pi, qi, hi, hi_orig, L.hit_phi(hi), L.hit_q(hi), L.hit_qbar(hi));

              ++b.n_scanned[i];
#ifdef MKFIT_TRACE
              ++mp_event->tr_layersearch(tr_layersearch_ids[i]).n_hits_scanned;
#endif

              if (iteration_hit_mask && (*iteration_hit_mask)[hi_orig]) {
                dprintf("Yay, denying masked hit on layer %u, hi %u, orig idx %u\n",
                        L.layer_info().layer_id(), hi, hi_orig);
#ifdef MKFIT_TRACE
                ++mp_event->tr_layersearch(tr_layersearch_ids[i]).n_hits_masked;
#endif
                continue;
              }

              // Try preloading Hits for the next step ... probably not really relevant.
              _mm_prefetch(&L.refHit(hi_orig), _MM_HINT_T0);

              prim_idcs[fill_pos] = i;
              hit_idcs[fill_pos] = hi;
              hit_orig_idcs[fill_pos] = hi_orig;
              is_plex.copyIn(fill_pos, B.m_isp, i);

              if (++fill_pos == NN) {
                preselect_hit_batch(b, hb, L, NN, is_sec_layer);
                fill_pos = 0;
              }
            }
          }
        }

        // Done with one PrimTCandRep. We might have enough hits to go into full KalmanProp.
        // Or wait for a change in ccand.
        // But there will be more work to be done, the overlaps, the other layer ...
        // ... so let's see.
      }
      if (fill_pos > 0) {
        preselect_hit_batch(b, hb, L, fill_pos, is_sec_layer);
      }
    }

    // Local hit density, hits per cm^2, over the window that was actually walked.
    // The window rather than the pre-selected hits, so the estimate does not
    // depend on the cut it is about to feed. The phi extent becomes a length at
    // the crossing radius.
    for (int i = 0; i < N_proc; ++i) {
      const float r = std::max(0.1f, std::hypot(B.m_sp2.x[i], B.m_sp2.y[i]));
      const float w_phi = (B.m_phi_max[i] - B.m_phi_min[i]) + 2.0f * B.m_dphi_track[i];
      const float w_q   = (B.m_q_max[i] - B.m_q_min[i]) + 2.0f * B.m_dq_track[i];
      const float area  = std::max(1e-4f, w_phi * r * w_q);
      b.log_rho[i] = std::log(std::max(1e-6f, (float) b.n_scanned[i] / area));
    }
  }

  //----------------------------------------------------------------------------
  // surface_referenced_dq()
  //
  // The predicted POINT is already on the module plane -- h3_state is the
  // Hermite solved onto it. The covariance behind MkBins::m_dq_track is NOT:
  // pea transported it to a fixed PATH LENGTH (errPropFromPathL_impl takes no
  // plane at all), so it describes the spread of where the track is after
  // travelling s -- a disc perpendicular to p^ -- and not the spread of where
  // the trajectory CROSSES the plane. Prediction and uncertainty were being
  // evaluated under two different conditions, and the gap is exactly the ds
  // degree of freedom: to put every member of the ensemble ON the plane, each
  // needs a different path length.
  //
  // Sliding along p^ until the surface is met is a linear map on the position
  // block,
  //      dx_s = (I - p^ n^T / (n^.p^)) dx ,
  // so the q variance is v^T C v with v = e_q - ((e_q.p^)/(n^.p^)) n^.
  //
  // n^ is the MODULE normal, not the layer cylinder's. That matters: TBPS
  // modules are tilted to face the interaction point, so n^.p^ ~ 1 and the
  // correction is ~1 there, while a cylinder normal would claim 1/sin^2(theta)
  // and over-widen by ~8x. Using the module normal makes tilted and flat layers
  // the same formula with no branching -- which is why this is done per hit
  // rather than in MkBins (MkBins::surface_reference_dq() is the cylinder form,
  // kept only because it is the route by which the corrected window reaches the
  // trace).
  //
  // Returns 3 sigma_q, matching the convention of MkBins::m_dq_track, which is
  // also what is passed in as the fallback.
  //----------------------------------------------------------------------------

  float MkFinderV2p2::surface_referenced_dq(float dq_track_fallback,
                                            const MkBinTrackCovExtract &TCE, int pi,
                                            const mini_propagators::StatePlex &h3_state, int h,
                                            const MPlex3V &module_norm, bool is_barrel) {
    const float px = h3_state.px[h], py = h3_state.py[h], pz = h3_state.pz[h];
    const float nx = module_norm(h, 0, 0), ny = module_norm(h, 1, 0), nz = module_norm(h, 2, 0);
    const float np = nx * px + ny * py + nz * pz;      // (n^.p) -- |p| cancels below
    // q direction: z in the barrel -- so e_q.p is just pz, with no dot product
    // and no hipo -- and r^ in the endcap.
    float ex = 0.0f, ey = 0.0f, ez = 1.0f, eqp = pz;
    bool ok = (np != 0.0f);
    if (ok && !is_barrel) {
      const float rr = hipo(h3_state.x[h], h3_state.y[h]);
      ok = (rr > 0.0f);
      if (ok) {
        ex = h3_state.x[h] / rr; ey = h3_state.y[h] / rr; ez = 0.0f;
        eqp = ex * px + ey * py;
      }
    }
    if (!ok)
      return dq_track_fallback;

    // f = (e_q.p^)/(n^.p^); |p| cancels, so no normalisation is needed. Clamped:
    // it diverges only at grazing incidence on the module, which is not an
    // operating point (the cluster is then too wide in phi to be a hit).
    const float f = std::clamp(eqp / np, -20.0f, 20.0f);
    const float v0 = ex - f * nx, v1 = ey - f * ny, v2 = ez - f * nz;
    const float var = v0 * v0 * TCE.m_cov_0_0[pi] + v1 * v1 * TCE.m_cov_1_1[pi] +
                      v2 * v2 * TCE.m_cov_2_2[pi] +
                      2.0f * (v0 * v1 * TCE.m_cov_0_1[pi] + v0 * v2 * TCE.m_cov_0_2[pi] +
                              v1 * v2 * TCE.m_cov_1_2[pi]);
    if (var <= 0.0f)
      return dq_track_fallback;
    return 3.0f * std::sqrt(var);
  }

  //----------------------------------------------------------------------------
  // Phase 3b -- pre-select one NN-wide batch of (candidate, hit) pairs.
  //
  // Solves the Hermite cubic onto each hit's own module plane, applies the
  // dq / dphi pre-selection cut, and pushes survivors into the candidate's
  // bounded priority queue. Was a lambda inside the monolith.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::preselect_hit_batch(LayerBatch &b, HitBatch &hb,
                                         const LayerOfHits &L, int N_proc_hits,
                                         bool is_sec_layer) {
    SteeringParams::iterator &spi = *mp_steeringparams_iter;
    MkBins &B = b.B;
    MkBinTrackCovExtract &TCE = b.TCE;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;
    mini_propagators::Hermite3D &H = b.H;
    namespace mp = mini_propagators;
    auto &is_plex = hb.is_plex;
    auto &prim_idcs = hb.prim_idcs;
    auto &hit_idcs = hb.hit_idcs;
    auto &hit_orig_idcs = hb.hit_orig_idcs;
#ifdef MKFIT_TRACE_PROP_COMPARE
    auto &h_plex = hb.h_plex;
#endif
#ifdef MKFIT_TRACE
    int *tr_layersearch_ids = b.tr_layersearch_ids;
#endif
    (void) spi; (void) TCE;

      MPlex3V module_pos;
      MPlex3V module_norm;
      mp::Hermite3D h3d;

#ifdef MKFIT_TRACE
      int tr_hitmatch_ids[NN];
      MPlex3V module_xdir, module_ydir;
#endif

      // Extract hit / target module data
      for (int h = 0; h < N_proc_hits; ++h) {
        const Hit &hit = L.refHit(hit_orig_idcs[h]);
        unsigned int mid = hit.detIDinLayer();
        const ModuleInfo &mi = L.layer_info().module_info(mid);
        module_pos.copyIn(h, mi.pos.Array());
        module_norm.copyIn(h, mi.zdir.Array());

        h3d.copyIn(h, H, prim_idcs[h]);

#ifdef MKFIT_TRACE
        module_xdir.copyIn(h, mi.xdir.Array());
        module_ydir.copyIn(h, mi.calc_ydir().Array());
        PrimTCandRep &ptc = * prim_tcand_ptrs[ prim_idcs[h] ];
        int sim_lbl = mp_event->simInfoForCurrentSeed(ptc.ccand().seed_origin_index()).label;
        const MCHitInfo &mchinfo = mp_event->simHitsInfo_[L.refHit(hit_orig_idcs[h]).mcHitID()];
        int hit_lbl = mchinfo.mcTrackID();

        // L.layer_id(), not spi->m_layer: with the secondary sub-layer scanned
        // these differ, and `layer` has to name the layer the hit is actually in.
        TrHitMatch hm { -1, ptc.tcand().m_trace_state_id, tr_layersearch_ids[ prim_idcs[h] ],
                        (int) L.layer_id(), (int) hit_orig_idcs[h], sim_lbl == hit_lbl };
        hm.is_sec_layer = is_sec_layer;
        tr_hitmatch_ids[h] = mp_event->trace_hitmatch(std::move(hm)).id;
#endif
      }

#ifdef MKFIT_TRACE_PROP_COMPARE
      is_plex.propagate_to_plane(mp::PA_Line, module_pos, module_norm, h_plex, true);
#endif

      mp::Hermite3DOnPlane h3dop;
      h3dop.init_coeffs(h3d, module_pos, module_norm);
#ifdef DEBUG
      // Distance-to-plane at the two layer bounding surfaces (d0, d1) and at the
      // Newton start point (t2, d2) -- debug-only, so keep them inside the ifdef
      // rather than trusting the optimiser to drop the calls.
      MPlexQF d0, d1, d2, t2;
      h3dop.evaluate(0.0f, d0);
      h3dop.evaluate(1.0f, d1);
      t2 = h3dop.m_T;
      h3dop.evaluate(h3dop.m_T, d2);
#endif
      // NOTE: solve() is a SINGLE Newton step, not an iteration to convergence.
      // d3 below is what says whether that one step was enough, so it goes into
      // the trace and not just into DEBUG.
      h3dop.solve();
#if defined(MKFIT_TRACE) || defined(DEBUG)
      MPlexQF d3;
      h3dop.evaluate(h3dop.m_T, d3); // residual distance to plane after the solve
#endif

      mp::StatePlex h3_state;
      h3d.evaluate(h3dop.m_T, h3_state);
      // h3_state.dalpha calculated below, as needed

      // Post-process hits into a heap in PrimTCandReps.
      //
      // Pre-selection runs on h3_state, the Hermite cubic solved onto the module
      // plane -- consistently with the state that goes into the pqueue and on to
      // the Kalman update.
      //
      // It used to run on h_plex, a PA_Line step, and that was not a modelling
      // choice: propagate_to_plane() implements ONLY PA_Line, PA_Quadratic and
      // PA_Exact both throw. (Nor is that unreasonable -- substituting the helix
      // into n.(r-p)=0 gives A sin(a) + B (1-cos(a)) + C a + D = 0 with
      // C = n_z k p_z, which is transcendental unless the module is untilted.
      // Hermite3DOnPlane IS the curved answer.)
      //
      // The line step was fine when it was written, because prop_to_limits()
      // left m_isp at the LAYER CENTRE -- see its "m_isp is now at the layer
      // center" comment. prop_to_limits_in_order(), which replaced it, leaves
      // m_isp at m_sp1, the entry edge, turning a short symmetric hop into a
      // one-sided extrapolation across the whole layer. Measured over 10 events
      // (52375 scanned hits, clean searches): |h3 - h_plex| median 32 um, 90th
      // pct 4 mm, 99th pct 110 cm -- for a tenth of hits, more than the entire
      // phi window.
      //
      // Switching gained: found tracks 2145 -> 2152, nH >= 80% 1500 -> 1505,
      // and unassociated 505 -> 498. More tracks, more good tracks, fewer fakes.
      //
      // NOTE: two other things still assume m_isp is the layer centre --
      // MkBins::determine_bin_windows() evaluates the dphi/dq jacobian there,
      // and m_q_center is set from it. See RecoTracker/CLAUDE.md.
      for (int h = 0; h < N_proc_hits; ++h) {
        PrimTCandRep &ptc = * prim_tcand_ptrs[ prim_idcs[h] ];
        float q, ddq, phi, ddphi;
        if (m_rz_limits.m_is_barrel) {
          q = h3_state.z[h];
        } else {
          q = hipo(h3_state.x[h], h3_state.y[h]);
        }
        ddq = std::abs(q - L.hit_q(hit_idcs[h]));
        phi = vdt::fast_atan2f(h3_state.y[h], h3_state.x[h]);
        ddphi = cdist(std::abs(phi - L.hit_phi(hit_idcs[h])));

        // Runtime so it can be scanned in one process. It was 3.0 because the
        // window it multiplies was missing its surface reference (see
        // MkBins::surface_reference_dq) and was therefore up to 9x too small
        // at |eta| > 2 -- the factor was compensating for that, not for
        // genuine tails. With the reference present it should be re-scanned;
        // note the dphi side carries no such factor and never needed one.
        // ---- Reference the q error to THIS MODULE'S PLANE ------------------
        //
        // The predicted POINT is already on the plane -- h3_state is the Hermite
        // solved onto it. The covariance behind B.m_dq_track is not: pea
        // transported it to a fixed PATH LENGTH (errPropFromPathL_impl takes no
        // plane at all), so it describes the spread of where the track is after
        // travelling s -- a disc perpendicular to p^ -- and NOT the spread of
        // where the trajectory crosses the plane. Prediction and uncertainty
        // were being evaluated under two different conditions, and the gap is
        // exactly the ds degree of freedom: to put every member of the ensemble
        // ON the plane, each needs a different path length.
        //
        // Sliding along p^ until the surface is met is a linear map on the
        // position block,
        //      dx_s = (I - p^ n^T / (n^.p^)) dx ,
        // so the q variance is v^T C v with v = e_q - ((e_q.p^)/(n^.p^)) n^.
        //
        // n^ is the MODULE normal, not the layer cylinder's. That matters: TBPS
        // modules are tilted to face the IP, so n^.p^ ~ 1 and the correction is
        // ~1 there, while a cylinder normal would claim 1/sin^2(theta) and
        // over-widen by ~8x. Using the module normal makes tilted and flat
        // layers the same formula with no branching -- which is the reason to do
        // this per hit rather than in MkBins.
        // Reference the q error to THIS MODULE'S plane -- see
        // surface_referenced_dq() above for the derivation.
        const float dq_trk = g_v2p2_surface_q
          ? surface_referenced_dq(B.m_dq_track[prim_idcs[h]], TCE, prim_idcs[h],
                                  h3_state, h, module_norm, m_rz_limits.m_is_barrel)
          : B.m_dq_track[prim_idcs[h]];

        const float EXTRA_DQ = g_v2p2_extra_dq;
        bool dqdphi_presel = ddq < EXTRA_DQ * dq_trk + EXTRA_DQ * MkBins::DDQ_PRESEL_FAC * L.hit_q_half_length(hit_idcs[h]) &&
                             ddphi < B.m_dphi_track[prim_idcs[h]] + MkBins::DDPHI_PRESEL_FAC * MkBins::HIT_PHI_HALF_EXTENT;

        // To be moved down, only for hits that pass pre-selection, needed here for printout.
        // Could be vectorized if we repack binnor stuff.
        h3_state.dalpha[h] = B.m_sp1.dalpha[prim_idcs[h]] + h3dop.m_T[h]*(B.m_sp2.dalpha[prim_idcs[h]] - B.m_sp1.dalpha[prim_idcs[h]]);

        // QQQQQQ testing, just keep phi cut
        // dqdphi_presel = ddphi < B.m_dphi_track[prim_idcs[h]] + MkBins::DDPHI_PRESEL_FAC * MkBins::HIT_PHI_HALF_EXTENT;

#ifdef DEBUG
        // clang-format off
        bool dq_presel = ddq < EXTRA_DQ * dq_trk + EXTRA_DQ * MkBins::DDQ_PRESEL_FAC * L.hit_q_half_length(hit_idcs[h]);
        bool dphi_presel = ddphi < B.m_dphi_track[prim_idcs[h]] + MkBins::DDPHI_PRESEL_FAC * MkBins::HIT_PHI_HALF_EXTENT;
        dprintf("     SelHit %6.3f %6.3f %6.4f %7.5f   %6.4f   %s [dq = %d, dphi = %d]\n",
                L.hit_q(hit_idcs[h]), L.hit_phi(hit_idcs[h]),
                ddq, ddphi, h_plex.dalpha[h], dqdphi_presel ? "PASS" : "REJECT", dq_presel, dphi_presel);
        dprintf("       ddq=%.3f, dq_track=%.4f, hit_q_half_len=%.4f, dq_expr=%.4f\n",
                ddq, B.m_dq_track[prim_idcs[h]], L.hit_q_half_length(hit_idcs[h]),
                EXTRA_DQ * dq_trk + EXTRA_DQ * MkBins::DDQ_PRESEL_FAC * L.hit_q_half_length(hit_idcs[h]))

        dprintf("      H3 d0=%.4f d1=%.4f -> d2=%e t2=%e -> d3=%e t3=%e ... dalpha=%6.4f\n",
               d0[h], d1[h], d2[h], t2[h], d3[h], h3dop.m_T[h],
               h3_state.dalpha[h]);
        // The two cheap propagations onto this module plane, side by side.
        // t3 outside [0,1] means the Hermite is extrapolating past the layer.
        dprintf("      H3   pos %8.4f %8.4f %8.4f | mom %8.4f %8.4f %8.4f\n",
                h3_state.x[h], h3_state.y[h], h3_state.z[h],
                h3_state.px[h], h3_state.py[h], h3_state.pz[h]);
        dprintf("      LINE pos %8.4f %8.4f %8.4f | mom %8.4f %8.4f %8.4f\n",
                h_plex.x[h], h_plex.y[h], h_plex.z[h],
                h_plex.px[h], h_plex.py[h], h_plex.pz[h]);
        dprintf("      H3 - LINE %+9.6f %+9.6f %+9.6f -- |d| = %.6f\n",
                h3_state.x[h] - h_plex.x[h], h3_state.y[h] - h_plex.y[h], h3_state.z[h] - h_plex.z[h],
                hipo(hipo(h3_state.x[h] - h_plex.x[h], h3_state.y[h] - h_plex.y[h]),
                     h3_state.z[h] - h_plex.z[h]));
        // clang-format on
#endif

#ifdef MKFIT_TRACE
        auto &tr_hitmatch = mp_event->tr_hitmatch(tr_hitmatch_ids[h]);
        // h3_state -- the state the cuts just above were taken on.
        tr_hitmatch.kine_on_plane = statep2bivec3(h3_state, h);
        tr_hitmatch.t_hermite = h3dop.m_T[h];
        tr_hitmatch.d_plane_h3 = d3[h];
        tr_hitmatch.dphi = ddphi;
        tr_hitmatch.dq = ddq;
        tr_hitmatch.hit_q_half_len = L.hit_q_half_length(hit_idcs[h]);
        tr_hitmatch.passed_preselect = dqdphi_presel;

        auto &tr_ls = mp_event->tr_layersearch(tr_layersearch_ids[ prim_idcs[h] ]);
        if (dqdphi_presel)
          ++tr_ls.n_hits_presel;

#ifdef MKFIT_TRACE_PROP_COMPARE
        tr_hitmatch.kine_on_plane_cmp = statep2bivec3(h_plex, h);
#endif

        // residuals
        EVec3 res = EVec3(h3_state.x[h], h3_state.y[h], h3_state.z[h]) - hit2pos(L.refHit(hit_orig_idcs[h]));
        tr_hitmatch.residual_x = res.Dot( EVec3(module_xdir(h, 0, 0), module_xdir(h, 1, 0), module_xdir(h, 2, 0)) );
        tr_hitmatch.residual_y = res.Dot( EVec3(module_ydir(h, 0, 0), module_ydir(h, 1, 0), module_ydir(h, 2, 0)) );
        tr_hitmatch.residual_z = res.Dot( EVec3(module_norm(h, 0, 0), module_norm(h, 1, 0), module_norm(h, 2, 0)) );
#endif

        if (/*prop_fail || */ !dqdphi_presel)
          continue;

        // float dalpha = h_plex.dalpha[h];
        // is_plex might come from somewhere else, through another index.

        // The reduction is PER SUB-LAYER, so each sensor keeps its own budget.
        const int sl = is_sec_layer ? 1 : 0;
        auto &pq = ptc.m_pqueue[sl];
        auto do_pqueue_push = [&]() {
#ifdef MKFIT_TRACE
          pq.push( { ddphi, hit_orig_idcs[h], hit_idcs[h], L.layer_id(), tr_hitmatch_ids[h], { h3_state, h, is_plex, h } } );
#else
          pq.push( { ddphi, hit_orig_idcs[h], hit_idcs[h], L.layer_id(), { h3_state, h, is_plex, h } } );
#endif
        };

        if (ptc.m_pqueue_size[sl] < g_v2p2_max_presel_hits) {
          do_pqueue_push();
          ++ptc.m_pqueue_size[sl];
        } else if (ddphi < pq.top().score) {
          pq.pop();
          do_pqueue_push();
        }
      }
  }

  //----------------------------------------------------------------------------
  // Phase 4 -- move the pqueue survivors into the per-candidate hit list.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::prepare_kalman_workload(LayerBatch &b) {
    const int N_proc = b.N_proc;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;
#ifdef MKFIT_TRACE
    int *tr_layersearch_ids = b.tr_layersearch_ids;
#endif

    // At this point PrimTCandReps have hits for the (first sub-) layer.
    // The open questions this block used to list are answered: the hits of both
    // sub-layers go into one list ordered by step distance (2 and 3), which is
    // also the plan the in-layer combinatorial search walks (8). What remains
    // from it is the per-layer reduction budget (9) -- g_v2p2_max_presel_hits is
    // one number for the whole detector, while overlap availability varies from
    // 2-3 % of pixel-barrel crossings to 50-56 % of TFPX.

    // Drain BOTH sub-layer queues into ONE list, then order it by step distance.
    //
    // sub_rank is the rank by score (ddphi) WITHIN one sub-layer and full_rank is
    // the rank across the merged layer; both are trace-only, and they are what
    // measures pre-selection quality against MC matching. The pqueue pops worst
    // first, hence the count-down.
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
#ifdef MKFIT_TRACE
      mp_event->tr_layersearch(tr_layersearch_ids[i]).n_hits_pqueue =
        ptc.m_pqueue_size[0] + ptc.m_pqueue_size[1];
#endif
      for (int sl = 0; sl < 2; ++sl) {
#ifdef MKFIT_TRACE
        int rank = ptc.m_pqueue_size[sl];
#endif
        while (ptc.m_pqueue_size[sl]) {
          --ptc.m_pqueue_size[sl];
          const auto &pqe = ptc.m_pqueue[sl].top();
          ptc.m_layer_hits.push_back( pqe );
#ifdef MKFIT_TRACE
          TrHitMatch &tr_hitmatch = mp_event->tr_hitmatch(pqe.tr_hitmatch_id);
          tr_hitmatch.sub_rank = rank--;
          tr_hitmatch.passed_pqueue = true;
#endif
          ptc.m_pqueue[sl].pop();
        }
      }

      // STEP-DISTANCE ORDERING. ddphi is the REDUCTION key -- it decides which
      // hits survive, and it is measured against MC matching through sub_rank.
      // It is not the TRAVERSAL key. The in-layer combinatorial search walks the
      // survivors forward along the trajectory, so the list has to be in path
      // order; forward-only over a totally ordered list then reaches every hit
      // SUBSET exactly once, so overlaps need no "take both" special case and
      // de-duplication is free. It also puts the Kalman updates in the order the
      // track meets the hits, without which the propagation between two of them
      // means nothing.
      //
      // dalpha is sufficient and cheaper than Hermite3D::path_length(): every hit
      // of one rep is reached from the same origin state with the same k, so
      // s = k |p| dalpha is monotone in dalpha. The sign factor makes it the PATH
      // order rather than the turn-angle order -- an inward search accumulates
      // negative dalpha, and a candidate woken up inside the layer has its
      // pre-existing hit at dalpha = 0 with overlap partners on either side.
      //
      // Once per rep is enough: a Kalman update perturbs the trajectory by about
      // the hit resolution, so it changes the dalphas but can only reorder pairs
      // already within that of each other.
      const float dir = m_rz_limits.is_outward() ? 1.0f : -1.0f;
      std::sort(ptc.m_layer_hits.begin(), ptc.m_layer_hits.end(),
                [dir](const PrimTCandRep::PQE &a, const PrimTCandRep::PQE &b) {
                  return dir * a.mixed_state.dalpha < dir * b.mixed_state.dalpha;
                });
    }

#ifdef MKFIT_TRACE
    // full_rank -- by score across the WHOLE layer, both sub-layers, against
    // sub_rank which is within one. Says whether the layer's best hit sits in the
    // primary or the secondary sensor. Counted rather than sorted: m_layer_hits
    // is already in step order and that order is what the search walks.
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      for (const auto &e : ptc.m_layer_hits) {
        int better = 0;
        for (const auto &o : ptc.m_layer_hits)
          better += (o.score < e.score);
        mp_event->tr_hitmatch(e.tr_hitmatch_id).full_rank = better + 1;
      }
    }
#endif
  }

  //----------------------------------------------------------------------------
  // Phase 5 -- propagate to each hit's module plane and run the Kalman update.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::kalman_update(LayerBatch &b) {
    const int N_proc = b.N_proc;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;

    auto do_kalman = [&](KalmanOpArgs& K) {
      K.compute_pars(); // Needed for conversion from mini_prop/bi-vec to std representation.
      K.do_kalman_stuff();
      K.reset();
    }; // end lambda do_kalman

    // KalmanProp, prim layer -- should also be done in-line as slots fill up.
    KalmanOpArgs koa;
    koa.prop_config = & mp_job->m_trk_info.prop_config();
#ifdef MKFIT_TRACE
    koa.mp_event = mp_event;
#endif

    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      int nlh = ptc.m_layer_hits.size();
      for (int lh = 0; lh < nlh; ++ lh) {
        const PrimTCandRep::PQE &pqe = ptc.m_layer_hits[lh];
        dprintf("scheduling %d %d %d %f\n", i, pqe.hit_index, pqe.hit_orig_index, pqe.mixed_state.dalpha);

        // Need to build all the Matriplexes for KalmanOperationPlane ... or some variant
        // There really needs to be an intermediate structure with packers so we can
        // set them up incrementally.

        TrackCand &tc = ptc.tcand();
        koa.item_begin(&ptc, { (int) pqe.hit_orig_index, pqe.layer });
        koa.load_state_err_chg(pqe.mixed_state, tc.state());
#ifdef MKFIT_TRACE
        koa.set_tr_hitmatch_id(pqe.tr_hitmatch_id);
#endif
        const auto &L = mp_job->m_event_of_hits[ pqe.layer ];
        const Hit &hit = L.refHit( pqe.hit_orig_index );
        unsigned int mid = hit.detIDinLayer();
        const ModuleInfo &mi = L.layer_info().module_info(mid);
        koa.load_hit_module(hit, mi);

        if (koa.item_finished()) {
          do_kalman(koa);
        }
      }
      // QQQQ hits stay in, somehow. Or it was leftover PrimTCands etc due to incomplete driver.
      ptc.m_layer_hits.clear();
    }
    if (koa.N_filled > 0) {
      do_kalman(koa);
    }

  }

  //----------------------------------------------------------------------------
  // Phase 6 -- acceptance. Still the best-hit hack: one hit per layer.
  //
  // This is the function the in-layer combinatorial search replaces. What it
  // does now is take the single lowest-chi2 hit if chi2 < 30, otherwise record
  // a hole; what it has to become is a selection over the SecTCandRep leaves of
  // the layer's expansion.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::process_kalman_results(LayerBatch &b) {
    const int N_proc = b.N_proc;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;

    // This, esp. the combinatorial part, should be done once prim-tcand is finished.
    // And, merging results, when ccand is finished.

    // XXXX MISSING: both candidate-stopping cuts that V1/V2 apply in
    // MkBuilder::find_tracks_unroll_candidates() -- pT < iter_params.minPtCut,
    // and the looper cut (fwd search, pT < 1.2, r > 25 cm, transverse angle
    // between position and momentum past pi/2 - 0.2). v2p2 applies NEITHER, and
    // no hole limits either (maxHolesPerCand / maxConsecHoles are read only in
    // MkFinder.cc). They do not all belong here: see the note at the bottom of
    // this file for where each one goes. When porting the looper cut, copy the
    // TwoPI - kMaxAngPosMom expression rather than writing a second literal, and
    // have the stop RECORD why (loopers are wanted later for the phase-2 timing
    // detectors and HGCal).
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      TrackCand &tc = ptc.tcand();

      // The track never reached this layer: leave the candidate exactly as it
      // was. Not a hole -- recording one would charge the candidate for a layer
      // the layer plan offered and its trajectory declined, and the plans are
      // deliberately inclusive (the transition plans are the union over tracks).
      // This is also the only branch that adds no HoT at all, which is what
      // "skip the layer" has to mean.
      if (Config::v2p2UseWsr && ptc.m_wsr.m_wsr == WSR_Outside) {
        dprintf("Outside to tcand %d, layer skipped\n", i);
        ++g_v2p2_policy_counters.n_layer_skipped;
        continue;
      }

#ifdef MKFIT_TRACE
      if (ptc.bChi2 < 30.0f || (g_v2p2_force_mc && ptc.bIsMc)) {
#else
      if (ptc.bChi2 < 30.0f) {
#endif
        // XXXX Extra missed layer -- to check stuff / maxgrowth / scores etc
        // This is somewhat impure :)
        // Add a copy of the held-back candidate before adding the hit.
        if (ptc.bChi2 > 5.0f && ! ptc.mp_ccrep->m_ccand.is_full()) {
          dprintf("ExtraMissed to tcand %d\n", i);
          ptc.mp_ccrep->m_ccand.push_back(tc).addHitIdx(
              fake_hit_index(tc, ptc.m_wsr), m_rz_limits.layer_info_1().layer_id(), 0.0f);

#ifdef MKFIT_TRACE
          // QQQQQ the parent extraction will be different
          int pid = ptc.tcand().m_trace_state_id;
          int id = mp_event->trace_new_cand_state(pid, (*mp_steeringparams_iter)->m_layer, track2bivec3(ptc.tcand()), ptc.tcand().state());
          ptc.mp_ccrep->m_ccand.back().m_trace_state_id = id;
#endif
        }

        dprintf("Output to tcand %d, idx=%d, layer=%d, chi2=%f\n", i, ptc.bHot.index, ptc.bHot.layer, ptc.bChi2);
        tc.addHitIdx(ptc.bHot.index, ptc.bHot.layer, ptc.bChi2);
        tc.setState(ptc.bState);

#ifdef MKFIT_TRACE
        // QQQQQ the parent extraction will be different; also fix: step, proper state (what is it)

        // This is also best-hit hack
        int pid = ptc.tcand().m_trace_state_id;
        int id = mp_event->trace_new_cand_state(pid, (*mp_steeringparams_iter)->m_layer, track2bivec3(ptc.tcand()), ptc.tcand().state());

        auto &ku = mp_event->tr_kalmanupdate( mp_event->tr_hitmatch(ptc.b_tr_hitmatch_id).kalman_id );
        ku.accepted = true;
        ku.state_id_out = id;
        // At this point could also set chi2_trk ... but this might get tricky with multiple hits added per (multi-)layer.
        // local / (multi-)layer score might be more relevant.

        ptc.tcand().m_trace_state_id = id;
#endif
      } else {
        // XXXX Here, we need to handle double layers correctly.
        // For now we have singles.
        //
        // No hit taken. What gets recorded is now a policy rather than an
        // unconditional -1: a real hole if the crossing was Inside and the
        // candidate is still under maxHolesPerCand / maxConsecHoles, a stop if it
        // is not, and kHitEdgeIdx or kHitInGapIdx if the WSR says the absence is
        // explained. v2p2 read neither hole limit before this -- they were
        // MkFinder.cc-only, i.e. V1/V2-only.
        const int fake_hit_idx = fake_hit_index(tc, ptc.m_wsr);
        dprintf("Missed to tcand %d, fake_hit_idx=%d\n", i, fake_hit_idx);
        switch (fake_hit_idx) {
          case Hit::kHitMissIdx:  ++g_v2p2_policy_counters.n_hole;      break;
          case Hit::kHitEdgeIdx:  ++g_v2p2_policy_counters.n_hot_edge;  break;
          case Hit::kHitInGapIdx: ++g_v2p2_policy_counters.n_hot_gap;   break;
          case Hit::kHitStopIdx:  ++g_v2p2_policy_counters.n_stop_holes; break;
          default: break;
        }
        if (fake_hit_idx == Hit::kHitStopIdx)
          tc.setStopReason(TrackCand::SR_TooManyHoles);
        tc.addHitIdx(fake_hit_idx, m_rz_limits.layer_info_1().layer_id(), 0.0f);

#ifdef MKFIT_TRACE
        // QQQQQ the parent extraction will be different; also fix: step, proper state (what is it)
        int pid = ptc.tcand().m_trace_state_id;
        int id = mp_event->trace_new_cand_state(pid, (*mp_steeringparams_iter)->m_layer, track2bivec3(ptc.tcand()), ptc.tcand().state());
        ptc.tcand().m_trace_state_id = id;
#endif
      }
    }

  }

  //----------------------------------------------------------------------------
  // THE IN-LAYER COMBINATORIAL SEARCH
  //
  // What it replaces: one Kalman update per pre-selected hit, all of them from
  // the candidate's incoming state, then take the single lowest-chi2 one. That
  // caps the search at ONE hit per layer, which is why disc efficiency tops out
  // around 75 % per hit -- a phase-2 track leaves 1.33 reconstructed hits per
  // disc layer, and 26-28 % of outer-tracker crossings carry three or more.
  //
  // What it does instead: m_layer_hits is in step order, so a path through the
  // layer is an increasing sequence of positions in it. The expansion is
  // breadth-first BY DEPTH -- every node at depth d is grown before any at
  // d+1 -- which is what lets one Matriplex batch draw lanes from many nodes,
  // many PrimTCandReps and many CCandReps at once, and which keeps
  // parent_idx < child_idx unconditionally true so a forward sweep of the arena
  // is always a valid topological order.
  //
  // Forward-only is a CANONICAL ORDERING, and that is the whole reason the hits
  // are sorted: each SUBSET of the pre-selected hits is reached exactly once
  // under any total order, so taking both members of an overlap needs no special
  // case and de-duplication is free. Skipping a hit is a cursor advance, and the
  // cursor is not stored -- it is m_hit_pos + 1.
  //
  // What path order buys beyond that is physical: the Kalman updates then happen
  // in the order the track meets the hits. Updating at s = 2 and then at s = 1
  // makes the propagation between them meaningless.
  //----------------------------------------------------------------------------

  // Depth cap, i.e. the most hits one path may take in one layer. Two is right
  // while a layer is a single sub-layer, where a second hit can only be a module
  // overlap. With the sub-layers paired it binds: a PS stack offers P and S, and
  // the neighbouring module is a stack too, so an overlap arrives as a PAIR and a
  // crossing can present four hits. Measured: 26-28 % of outer-tracker crossings
  // carry three or more, and the 4-hit bin is three times the 3-hit bin.
  // kMaxSecDepthMax only sizes the chain array; g_v2p2_max_sec_depth is the cap.
  static constexpr int kMaxSecDepthMax = 8;
  // Per-hit acceptance, the same cut the best-hit path applies.
  static constexpr float kSecChi2Cut = 30.0f;

  std::pair<int, int> MkFinderV2p2::harvest_sec_nodes(const LayerBatch &b) {
    const int begin = (int) m_sec_arena.size();
    for (const auto &o : m_sec_out) {
      if ( ! (o.chi2 < kSecChi2Cut))   // also rejects NaN
        continue;
      SecTCandRep n;
      n.m_ptc        = o.ptc;
      n.m_parent_idx = o.parent_idx;
      n.m_hit_pos    = o.hit_pos;
      n.m_hot        = o.hot;
      n.m_state      = o.state;
      n.m_chi2       = o.chi2;

      // Features accumulate parent -> child. The parent is always already in the
      // arena: harvest runs once per depth, after the whole depth has flushed.
      const LayerStepFeatures *pf =
        (o.parent_idx >= 0) ? &m_sec_arena[o.parent_idx].m_feat : nullptr;
      LayerStepFeatures &f = n.m_feat;
      if (pf)
        f = *pf;
      else {
        float lrho = 0.0f;
        for (int i = 0; i < b.N_proc; ++i)
          if (b.ptc[i] == o.ptc) { lrho = b.log_rho[i]; break; }
        fill_step_geometry(f, *o.ptc, lrho);
      }
      f.n_hits    = (pf ? pf->n_hits : 0) + 1;
      f.n_overlap = f.n_hits - 1;
      f.hole_kind = V2P2_NoHole;
      f.chi2_sum  = (pf ? pf->chi2_sum : 0.0f) + o.chi2;
      f.chi2_max  = std::max(pf ? pf->chi2_max : 0.0f, o.chi2);
      {
        const auto &L = mp_job->m_event_of_hits[o.hot.layer];
        const float qhl = L.hit_q_half_length(o.hit_in_layer);
        f.q_half_len_best = (pf && pf->n_hits > 0) ? std::min(pf->q_half_len_best, qhl) : qhl;
      }
      f.log_det_v_sum = (pf ? pf->log_det_v_sum : 0.0f) + std::log(std::max(1e-30f, o.det_v));
#ifdef MKFIT_TRACE
      n.m_tr_hitmatch_id = o.tr_hitmatch_id;
#endif
      o.ptc->m_has_sec_nodes = true;
      o.ptc->mp_ccrep->m_sec_nodes.push_back((int) m_sec_arena.size());
      m_sec_arena.push_back(n);
    }
    m_sec_out.clear();
    return {begin, (int) m_sec_arena.size()};
  }

  // The parts of a layer step that do not depend on which hits were taken: where
  // the step goes, which way, and what the candidate was when it arrived. Filled
  // once per path root and then carried down the tree.
  void MkFinderV2p2::fill_step_geometry(LayerStepFeatures &f, const PrimTCandRep &ptc,
                                        float log_rho) const {
    const TrackCand &tc = const_cast<PrimTCandRep &>(ptc).tcand();
    const LayerInfo &li = m_rz_limits.layer_info_1();
    // layer_from is the layer of the candidate's LAST FOUND HIT, so the pair
    // names the propagation that actually happens. A candidate that missed the
    // previous layer steps across two, and keying anything on plan adjacency
    // instead would attribute it to a step that never took place.
    f.layer_from = (short) tc.getLastFoundHitLyr();
    f.layer_to   = (short) li.layer_id();
    f.is_outward = m_rz_limits.is_outward();
    f.is_pixel   = li.is_pixel();
    f.is_barrel  = li.is_barrel();
    f.pt         = tc.pT();
    f.step       = (short) tc.nTotalHits();
    f.n_found_so_far = (short) tc.nFoundHits();
    f.n_holes_so_far = (short) tc.nAllMinusOneHits();
    f.log_rho = log_rho;
  }

  void MkFinderV2p2::expand_in_layer(LayerBatch &b) {
    const int N_proc = b.N_proc;
    PrimTCandRep **prim_tcand_ptrs = b.ptc;

    // NB: the arena is NOT rewound here. It has to live until end of layer, when
    // the whole CombCandidate's alternatives are selected between; end_layer()
    // rewinds it once for all of them.
    m_sec_out.clear();

    KalmanOpArgs koa;
    koa.prop_config = & mp_job->m_trk_info.prop_config();
    koa.mp_out = & m_sec_out;
#ifdef MKFIT_TRACE
    koa.mp_event = mp_event;
#endif

    auto load_module = [&](const PrimTCandRep::PQE &pqe) {
      const auto &L = mp_job->m_event_of_hits[ pqe.layer ];
      const Hit &hit = L.refHit( pqe.hit_orig_index );
      koa.load_hit_module(hit, L.layer_info().module_info(hit.detIDinLayer()));
    };
    auto flush = [&]() {
      if (koa.N_filled == 0)
        return;
      if ( ! koa.m_solve_plane)
        koa.compute_pars();   // propPar is an INPUT on the sPerp path, an output on the solve path
      koa.do_kalman_stuff();
      koa.reset();
    };

    // Depth 0. The Hermite has already solved the crossing for every
    // pre-selected hit, so this is the cheap sPerp path and is exactly what the
    // best-hit code does -- it just keeps every outcome instead of the winner.
    koa.m_solve_plane = false;
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      if (Config::v2p2UseWsr && ptc.m_wsr.m_wsr == WSR_Outside)
        continue;
      const int nlh = (int) ptc.m_layer_hits.size();
      for (int lh = 0; lh < nlh; ++lh) {
        const PrimTCandRep::PQE &pqe = ptc.m_layer_hits[lh];
        koa.item_begin(&ptc, { (int) pqe.hit_orig_index, pqe.layer }, -1, lh, pqe.hit_index);
        koa.load_state_err_chg(pqe.mixed_state, ptc.tcand().state());
#ifdef MKFIT_TRACE
        koa.set_tr_hitmatch_id(pqe.tr_hitmatch_id);
#endif
        load_module(pqe);
        if (koa.item_finished())
          flush();
      }
    }
    flush();
    const int arena_at_entry = (int) m_sec_arena.size() - 0;   // set below, after harvest
    (void) arena_at_entry;
    auto [f_beg, f_end] = harvest_sec_nodes(b);
    const int arena_batch_begin = f_beg;

    // Depths 1 and up. The starting state is now a node's UPDATED state, for
    // which no crossing has been solved, so propagate-to-plane solves it.
    for (int depth = 1; depth < g_v2p2_max_sec_depth && f_end > f_beg; ++depth) {
      koa.m_solve_plane = true;
      for (int ni = f_beg; ni < f_end; ++ni) {
        // By index, not by reference: the arena grows under us only at harvest,
        // but the discipline is what keeps the indices the handles.
        PrimTCandRep &ptc = * m_sec_arena[ni].m_ptc;
        const int nlh = (int) ptc.m_layer_hits.size();
        for (int lh = m_sec_arena[ni].m_hit_pos + 1; lh < nlh; ++lh) {
          const PrimTCandRep::PQE &pqe = ptc.m_layer_hits[lh];

          // A SECOND HIT FROM THE SAME MODULE IS NOT AN OVERLAP, AND IS FORBIDDEN.
          // An overlap is the track crossing two DIFFERENT modules. Two hits in
          // one module are a split cluster or two tracks' hits, and a split
          // cluster is one measurement seen twice: taking both feeds the same
          // information into the Kalman filter twice, which shrinks the
          // covariance without adding knowledge. Measured at 6.0 % of the extra
          // hits taken before this check.
          //
          // No truth is needed for the test -- the module id is on the hit -- and
          // the walk is cheap because it only ever looks back along one path,
          // which is at most g_v2p2_max_sec_depth long.
          bool same_module = false;
          for (int ci = ni; ci >= 0 && !same_module; ci = m_sec_arena[ci].m_parent_idx) {
            const SecTCandRep &an = m_sec_arena[ci];
            if (an.m_hot.layer != pqe.layer)
              continue;
            const auto &La = mp_job->m_event_of_hits[an.m_hot.layer];
            same_module = La.refHit(an.m_hot.index).detIDinLayer() ==
                          La.refHit(pqe.hit_orig_index).detIDinLayer();
          }
          if (same_module) {
            ++g_v2p2_policy_counters.n_same_module_vetoed;
            continue;
          }

          koa.item_begin(&ptc, { (int) pqe.hit_orig_index, pqe.layer }, ni, lh, pqe.hit_index);
          koa.load_state_err_chg(m_sec_arena[ni].m_state);
#ifdef MKFIT_TRACE
          koa.set_tr_hitmatch_id(pqe.tr_hitmatch_id);
#endif
          load_module(pqe);
          if (koa.item_finished())
            flush();
        }
      }
      flush();
      std::tie(f_beg, f_end) = harvest_sec_nodes(b);
      g_v2p2_policy_counters.n_sec_deep += f_end - f_beg;
    }

    // Only this batch's share -- the arena now spans the whole layer.
    g_v2p2_policy_counters.n_sec_nodes += (long) m_sec_arena.size() - arena_batch_begin;
  }

  //----------------------------------------------------------------------------
  // END-OF-LAYER SELECTION -- a single flat sort, and that is the design.
  //
  // Everything that could be a continuation of this CombCandidate competes in ONE
  // list on ONE number:
  //   - every in-layer path of every PrimTCandRep (the SecTCandRep arena);
  //   - every PrimTCandRep that produced no path, entered as the HOLE it would
  //     record -- so "take a well-fitting wrong hit" and "take a hole" are
  //     decided against each other rather than the hole being what is left when
  //     no hit passes. That is the third of divergences S16b measures as
  //     preferring a wrong hit to a hole;
  //   - every TrackCand that never became a PrimTCandRep -- already stopped, or
  //     skipped at pull-in, or with the layer outside its reach -- entered
  //     unchanged at its own score. Pulling those in is what gives the processing
  //     symmetry: nothing is special-cased, it just scores what it scores.
  //
  // One sort works because the score is additive (see V2p2Score.h), so a node's
  // global score is its parent's plus this step's delta, and paths from different
  // PrimTCandReps are directly comparable. A non-additive score would need a
  // two-level arbitration with an offset between the levels, which is the
  // question that dissolves here rather than being answered.
  //
  // This runs at END OF LAYER and not per NN batch because one CombCandidate's
  // PrimTCandReps can straddle a batch boundary.
  //----------------------------------------------------------------------------

  // Keep the best-scoring stopped candidate of this CombCandidate. The score is
  // the in-flight accumulator, which is on one scale within a seed -- the only
  // comparison made here.
  void MkFinderV2p2::offer_best_short(CombCandidate &ccand, const TrackCand &tc) const {
    ++g_v2p2_policy_counters.n_best_short_offered;
    if (ccand.refBestShortCand().combCandidate() == nullptr ||
        tc.score() > ccand.refBestShortCand().score()) {
      ccand.setBestShortCand(tc);
      ++g_v2p2_policy_counters.n_best_short_taken;
    }
  }

  void MkFinderV2p2::select_and_materialise(CCandRep &ccrep) {
    CombCandidate &ccand = ccrep.m_ccand;
    const int cap = ccand.capacity();

    m_sel.clear();

    // Which TrackCands got a PrimTCandRep this layer. The rest are entered
    // unchanged below.
    bool has_ptc[64] = {false};
    const int n_tc = (int) ccand.size();

    for (auto &ptc : ccrep.m_primTCs) {
      if (ptc.m_origin_tcand_index < 64)
        has_ptc[ptc.m_origin_tcand_index] = true;

      TrackCand &tc = ptc.tcand();

      // EVERY PrimTCandRep offers the hole, including the ones that DID find
      // paths. That is the whole point: S16b measures a third of divergences as
      // a candidate preferring a well-fitting wrong hit to a hole, and two
      // thirds as the true hit being present but the wrong one fitting better --
      // neither of which any single-hit score can fix, because the information
      // to decide is not there yet. Offering both and resolving at end of layer
      // is the answer, and it is only expressible once the hole is a COMPETITOR
      // rather than what is left when nothing passes.
      //
      // It also puts the cap under real pressure: without this the selection saw
      // ~4.5 competitors against a cap of 6 and therefore almost never bound,
      // which is measurably why the score parameters were inert.

      // Reached the layer but took nothing: it competes as a hole. WSR_Outside is
      // NOT a hole -- the track does not reach the layer, so there is nothing to
      // have missed, and it competes unchanged.
      if (Config::v2p2UseWsr && ptc.m_wsr.m_wsr == WSR_Outside) {
        ++g_v2p2_policy_counters.n_layer_skipped;
        m_sel.push_back({ptc.m_origin_tcand_index, -1, 0, false, tc.score()});
      } else {
        const int fake = fake_hit_index(tc, ptc.m_wsr);
        LayerStepFeatures f;
        fill_step_geometry(f, ptc, 0.0f);
        f.hole_kind = v2p2_hole_kind(fake);
        m_sel.push_back({ptc.m_origin_tcand_index, -1, fake, true,
                         tc.score() + v2p2_layer_step_score(f)});
      }
    }

    for (const int ni : ccrep.m_sec_nodes) {
      const SecTCandRep &n = m_sec_arena[ni];
      m_sel.push_back({n.m_ptc->m_origin_tcand_index, ni, 0, false,
                       n.m_ptc->tcand().score() + v2p2_layer_step_score(n.m_feat)});
    }

    // Stopped or pull-in-skipped TrackCands, unchanged, at their own score.
    //
    // BEST-SHORT. A stopped candidate cannot be extended again, so leaving it in
    // the beam costs a slot a live candidate could use -- v2p2 has been doing
    // exactly that, because end_layer() only retires a CombCandidate once ALL of
    // its TrackCands have stopped. V1 moves it out instead and keeps the best one
    // on the CombCandidate, which mergeCandsAndBestShortOne re-inserts at the end
    // if it still beats the worst survivor.
    //
    // "Short" misleads: what is kept is the best score SO FAR, and it is worth
    // keeping because a score can DEGRADE later, chi2 growing as a candidate
    // diverges or is over-compressed. Physically it is a hard hadronic scatter
    // for a pion, which about 30 % undergo to some extent, or hard bremsstrahlung
    // for an electron -- rare only because electrons are rare, and when it
    // happens the calorimeter recovers the energy while the track parameters at
    // the vertex are still wanted.
    //
    // Outward only. Going inward the trailing end of the hit sequence is the HEAD
    // of the track, so a truncated candidate is not a shorter track, it is one
    // that never reached the beamline.
    const bool best_short = Config::v2p2BestShort && m_rz_limits.is_outward();
    for (int ic = 0; ic < n_tc; ++ic) {
      if (ic < 64 && has_ptc[ic])
        continue;
      if (best_short && ccand[ic].getLastHitIdx() == Hit::kHitStopIdx) {
        offer_best_short(ccand, ccand[ic]);
        continue;
      }
      m_sel.push_back({ic, -1, 0, false, ccand[ic].score()});
    }

    if (m_sel.empty())
      return;

    // Partial sort: only the top cap matter, and cap is 6.
    const int n_keep = std::min((int) m_sel.size(), cap);
    std::partial_sort(m_sel.begin(), m_sel.begin() + n_keep, m_sel.end(),
                      [](const SelEntry &a, const SelEntry &b) { return a.score > b.score; });

    // BREATHING SPACE. Keep one continuation that declined the layer, even if it
    // is outranked by hit-taking ones.
    //
    // Taking a hit always shrinks the covariance; declining is the only move that
    // does not. So when every surviving hypothesis took a hit in this layer,
    // there is no branch left that is open to an earlier hit having been wrong,
    // and the windows keep narrowing around whatever the candidate has already
    // committed to. That matters most exactly where S16b measures the damage:
    // early, when the state is still seed-dominated and a candidate that takes a
    // wrong hit has an 11.6x higher kill rate thereafter, and after a run of
    // precise hits, when the covariance is smallest and a 1.4-2x under-estimate
    // of it does the most harm.
    //
    // This is a BEAM POLICY, not a score. The score ranks hypotheses given that
    // the error model is right; this hedges against its being wrong, which no
    // single scalar can express because the two are different questions.
    if (Config::v2p2ReserveHoleSlot && n_keep > 1) {
      bool kept_a_decliner = false;
      for (int k = 0; k < n_keep && !kept_a_decliner; ++k)
        kept_a_decliner = (m_sel[k].node_idx < 0);
      if (!kept_a_decliner) {
        int best_decliner = -1;
        for (int k = n_keep; k < (int) m_sel.size(); ++k)
          if (m_sel[k].node_idx < 0 &&
              (best_decliner < 0 || m_sel[k].score > m_sel[best_decliner].score))
            best_decliner = k;
        if (best_decliner >= 0) {
          std::swap(m_sel[n_keep - 1], m_sel[best_decliner]);
          ++g_v2p2_policy_counters.n_hole_slot_reserved;
        }
      }
    }

    // Build the survivors as copies BEFORE touching the CombCandidate: several
    // winners can descend from the same TrackCand -- that is what branching is --
    // so the sources have to stay readable while the copies are made.
    m_new_cands.clear();
    for (int k = 0; k < n_keep; ++k) {
      const SelEntry &e = m_sel[k];
      m_new_cands.push_back(ccand[e.tcand_idx]);
      TrackCand &nc = m_new_cands.back();

      if (e.node_idx >= 0) {
        int chain[kMaxSecDepthMax];
        int n_chain = 0;
        for (int ci = e.node_idx; ci >= 0; ci = m_sec_arena[ci].m_parent_idx)
          chain[n_chain++] = ci;
        for (int c = n_chain - 1; c >= 0; --c) {
          const SecTCandRep &n = m_sec_arena[chain[c]];
          nc.addHitIdx(n.m_hot.index, n.m_hot.layer, n.m_chi2);
          if (c != n_chain - 1) {
            nc.incOverlapCount();   // every hit past the first in one layer IS the overlap case
            // ... except when it is not. A genuine overlap is two hits in two
            // DIFFERENT modules. Two hits in the SAME module are a split cluster
            // or two tracks' hits, and a split cluster is ONE measurement seen
            // twice -- taking both treats one measurement as two independent
            // ones, which over-constrains the fit. Counted here before deciding
            // whether to forbid it.
            const SecTCandRep &pn = m_sec_arena[chain[c + 1]];
            const auto &Ln = mp_job->m_event_of_hits[n.m_hot.layer];
            const auto &Lp = mp_job->m_event_of_hits[pn.m_hot.layer];
            if (n.m_hot.layer == pn.m_hot.layer &&
                Ln.refHit(n.m_hot.index).detIDinLayer() == Lp.refHit(pn.m_hot.index).detIDinLayer())
              ++g_v2p2_policy_counters.n_same_module;
            else
              ++g_v2p2_policy_counters.n_diff_module;
          }
#ifdef MKFIT_TRACE
          int pid = nc.m_trace_state_id;
          int id = mp_event->trace_new_cand_state(pid, (*mp_steeringparams_iter)->m_layer,
                                                  track2bivec3(nc), n.m_state);
          if (n.m_tr_hitmatch_id >= 0) {
            auto &ku = mp_event->tr_kalmanupdate( mp_event->tr_hitmatch(n.m_tr_hitmatch_id).kalman_id );
            ku.accepted = true;
            ku.state_id_out = id;
          }
          nc.m_trace_state_id = id;
#endif
        }
        nc.setState(m_sec_arena[e.node_idx].m_state);
        ++g_v2p2_policy_counters.n_path_taken;
        if (n_chain > 1)
          g_v2p2_policy_counters.n_extra_hits += n_chain - 1;
      } else if (e.add_fake) {
        switch (e.fake_hit) {
          case Hit::kHitMissIdx:  ++g_v2p2_policy_counters.n_hole;       break;
          case Hit::kHitEdgeIdx:  ++g_v2p2_policy_counters.n_hot_edge;   break;
          case Hit::kHitInGapIdx: ++g_v2p2_policy_counters.n_hot_gap;    break;
          case Hit::kHitStopIdx:  ++g_v2p2_policy_counters.n_stop_holes; break;
          default: break;
        }
        if (e.fake_hit == Hit::kHitStopIdx)
          nc.setStopReason(TrackCand::SR_TooManyHoles);
        nc.addHitIdx(e.fake_hit, m_rz_limits.layer_info_1().layer_id(), 0.0f);
#ifdef MKFIT_TRACE
        int pid = nc.m_trace_state_id;
        nc.m_trace_state_id = mp_event->trace_new_cand_state(
            pid, (*mp_steeringparams_iter)->m_layer, track2bivec3(nc), nc.state());
#endif
      }

      // score_ is the in-flight accumulator. It is overwritten at the end of the
      // search by the official track_score_func, which is what output and
      // cross-seed duplicate removal compare -- so the two scoring systems are
      // separated by phase and never mixed within one comparison.
      nc.setScore(e.score);
    }

    // Candidates that stopped IN this layer leave the beam the same way.
    ccand.clear();
    for (auto &c : m_new_cands) {
      if (best_short && c.getLastHitIdx() == Hit::kHitStopIdx) {
        offer_best_short(ccand, c);
        continue;
      }
      ccand.push_back(c);
    }

    ++g_v2p2_policy_counters.n_selections;
    g_v2p2_policy_counters.n_sel_entries += (long) m_sel.size();
    g_v2p2_policy_counters.n_sel_kept += n_keep;
  }

  // Sketch of the SECONDARY sub-layer pass, left from before the decision to do
  // step-distance ordering over ONE merged list. Kept as the only in-tree record
  // of what the second find_bin_ranges() walk looks like; it only printed.
  /*
    // Let's try picking the secondary hit
    if (is_double_layer) {
      const auto &L = mp_job->m_event_of_hits[spi->m_layer_sec];
      const auto &iteration_hit_mask = mp_job->get_mask_for_layer(spi->m_layer_sec);
      const auto &BL = BL_s;

      for (int i = 0; i < N_proc; ++i) {

        using bidx_t = LayerOfHits::bin_index_t;
        using bcnt_t = LayerOfHits::bin_content_t;

        for (bidx_t qi = BL.q1[i]; qi != BL.q2[i]; ++qi) {
          for (bidx_t pi = BL.p1[i]; pi != BL.p2[i]; pi = L.phiMaskApply(pi + 1)) {
            auto pbi = L.phiQBinContent(pi, qi);
            for (bcnt_t hi = pbi.begin(); hi < pbi.end(); ++hi) {

              const unsigned int hi_orig = L.getOriginalHitIndex(hi);

              dprintf(" %d: S_HIT %3u %4u %5u   %6.3f %6.3f %6.3f\n",
                i, pi, qi, hi, L.hit_phi(hi), L.hit_q(hi), L.hit_qbar(hi));

              if (iteration_hit_mask && (*iteration_hit_mask)[hi_orig]) {
                dprintf("Yay, denying masked hit on layer %u, hi %u, orig idx %u\n",
                        L.layer_info().layer_id(), hi, hi_orig);
                continue;
              }
              PrimTCandRep &ptc = * prim_tcand_ptrs[i];
              int nh = ptc.m_layer_hits.size();
              for (int j = 0; j < nh; ++j) {
                auto &X = mp_job->m_event_of_hits[spi->m_layer];
                const PrimTCandRep::PQE &pqe = ptc.m_layer_hits[j];
                dprintf("           prim %d        %6.3f %6.3f %6.3f  \n", j,
                  X.hit_phi(pqe.hit_index), X.hit_q(pqe.hit_index), X.hit_qbar(pqe.hit_index));
              }
            }
          }
        }
      }
    }
  */

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------

    /*
    // previous -- procedural -- approach that can't quite work.

    prepare_select_hits_workload();

    select_hits();
    // make a bit of planning for what we want to try.
    // separate function for double layers with per-CCand steering structs.

    prepare_kalman_workload();

    kalman_update();

    process_kalman_results();
    // handle_combinatorials -- further plan

    */

  // NOTE on the two candidate-stopping cuts (minPtCut, and the pi/2 - 0.2 looper
  // stop): V1/V2 apply them in MkBuilder::find_tracks_unroll_candidates(), which
  // is live code -- read it there rather than from a stale copy. A verbatim
  // commented-out transcription used to sit here and has been removed.
  //
  // Where they belong in v2p2 is NOT one place (maintainer, 2026-09-21). The
  // decisions distribute over the layer pipeline, at the earliest point that can
  // make each one:
  //   - at pull-in, begin_next_Ccrep_in_layer(): minPtCut and the looper stop,
  //     which need only the candidate's own state, alongside the existing
  //     rz_quadrant_check();
  //   - after prop_to_limits_in_order(), where sp1/sp2 are known: the
  //     within-sensitive-region verdict. v2p2 sets NO WSR at all today -- there
  //     is only a "Set the WSR" comment in process_pre_select() -- while V1/V2
  //     carry MkFinder::m_XWsrResult and MkBuilder consumes WSR_Edge /
  //     WSR_Outside / WSR_Failed. That is also the near-miss vs clear-miss split
  //     the two mini-propagator fail flags already classify for free;
  //   - at end of layer: only what genuinely needs the layer's outcome, i.e. the
  //     hole counters and retiring a CombCandidate whose TrackCands have all
  //     stopped (end_layer()'s `is_finished` is hardcoded false today).
  // See RecoTracker/CLAUDE.md and SESSIONS.md S4.

} // namespace mkfit
