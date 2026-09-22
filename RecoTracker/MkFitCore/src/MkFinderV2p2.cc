#include "MkFinderV2p2.h"

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
// #include "FindingFoos.h"

#include "MkBins.h"

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

  bool g_v2p2_force_mc = false;
  float g_v2p2_extra_dq = 3.0f;
  bool  g_v2p2_surface_q = true;


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

      if (tcand.getLastHitIdx() == -2)
        continue;

      // XXXX V1 also did: min-pt-cut, apogee stop; and setting ccand.setState(CombCandidate::Finished)
      // That was after prop-to-layer-centroid.

      // XXXX Should one do rough "layer already passed" pre-check here?
      // Or in pre-select, where we engage MkBins ... but there I loose a vector slot.
      // Let's try.

      if ( ! m_rz_limits.rz_quadrant_check(tcand.z(), tcand.pz()))
        continue;

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

      // QQQQ should attempt to reuse the PrimTCandReps
      ai->m_primTCs.clear();

      // XXXX something is rotten here; well, just making them run to the end
      bool is_finished = false; // XXXX
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
    prepare_kalman_workload(b);   // pqueue -> m_layer_hits, stamping rank
    kalman_update(b);             // propagate to each module plane + update
    process_kalman_results(b);    // best-hit acceptance into the TrackCand
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

    // At this point we should check for prop-failures and/or if limits have
    // been reached.
    // Set the WSR.
    // There are also the apogee and minPt checks ... but those might be better done
    // elsewhere.

    B.determine_bin_windows(TCE);

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


    {
      // The PRIMARY sub-layer. The secondary pass is the same block against
      // spi->m_layer_sec / BL_s with is_sec_layer = true; it does not exist yet.
      const bool is_sec_layer = false;
      const auto &L = mp_job->m_event_of_hits[spi->m_layer];
      const auto &iteration_hit_mask = mp_job->get_mask_for_layer(spi->m_layer);
      const auto &BL = BL_p;

      for (int i = 0; i < N_proc; ++i) {

        using bidx_t = LayerOfHits::bin_index_t;
        using bcnt_t = LayerOfHits::bin_content_t;

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

        auto do_pqueue_push = [&]() {
#ifdef MKFIT_TRACE
          ptc.m_pqueue.push( { ddphi, hit_orig_idcs[h], hit_idcs[h], L.layer_id(), tr_hitmatch_ids[h], { h3_state, h, is_plex, h } } );
#else
          ptc.m_pqueue.push( { ddphi, hit_orig_idcs[h], hit_idcs[h], L.layer_id(), { h3_state, h, is_plex, h } } );
#endif
        };

        if (ptc.m_pqueue_size < MkBins::NEW_MAX_HIT) {
          do_pqueue_push();
          ++ptc.m_pqueue_size;
        } else if (ddphi < ptc.m_pqueue.top().score) {
          ptc.m_pqueue.pop();
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
    // QQQQ - We also need path lengths -- but let's postpone this.
    // We could:
    // 1 do full propagate-update for all of them.
    //   maybe improve the parameters? this somehow closes combinatorials
    // 2 look for the secondary sister hit in double layer.
    //   repeat selection for secondary layer (or delay)
    //   where do they go? another priority_queue, same one ...
    //   ... or extract current ones as in 3 below and then reuse.
    // 3 consider ordering the hits in s / t / z / r -- s would be ideal.
    //   t, really, s can go in negative direction, t is always 0 -> 1
    // 9 re-check the "extreme" overlap case in tilted layers -- increase
    //   max-hits there or what?
    // 8 knowing the s, the path ... can I make a proto combinatorial plan for each hit?

    // Move hits from priority-queue into vector for primary layer.
    // XXXX Should invert the order, pqueue has the worst at the top !!!!
    // Should really go into KalmanOpArgs directly, and processed as needed.
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
#ifdef MKFIT_TRACE
      int rank = ptc.m_pqueue_size;
      mp_event->tr_layersearch(tr_layersearch_ids[i]).n_hits_pqueue = ptc.m_pqueue_size;
#endif
      while (ptc.m_pqueue_size) {
        --ptc.m_pqueue_size;
        const auto &pqe = ptc.m_pqueue.top();
        ptc.m_layer_hits.push_back( pqe );

#ifdef MKFIT_TRACE
        TrHitMatch &tr_hitmatch = mp_event->tr_hitmatch(pqe.tr_hitmatch_id);
        tr_hitmatch.sub_rank = rank--;
        tr_hitmatch.passed_pqueue = true;
#endif

        // dprintf("pushing for %d  %f, %u %u\n", i, pqe.score, pqe.hit_orig_index, pqe.hit_index);
        ptc.m_pqueue.pop();
      }
    }

#ifdef MKFIT_TRACE
    // full_rank -- rank by score across the WHOLE layer, both sub-layers merged,
    // against sub_rank which is within one sub-layer. Says whether the layer's
    // best hit sits in the primary or the secondary sensor.
    //
    // Counted rather than sorted, deliberately: sorting would have to either
    // reorder m_layer_hits -- which is the order kalman_update() iterates, and
    // the best-hit comparison is strictly-less, so reordering can flip an exact
    // chi2 tie -- or build a parallel permutation. With at most
    // 2 * NEW_MAX_HIT entries an O(n^2) count is free and cannot perturb
    // anything. (The step-distance sort that IS coming does reorder
    // m_layer_hits; that is a deliberate change of execution order, not this.)
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      auto rank_against_both = [&](const std::vector<PrimTCandRep::PQE> &v) {
        for (const auto &e : v) {
          int better = 0;
          for (const auto &o : ptc.m_layer_hits)     better += (o.score < e.score);
          for (const auto &o : ptc.m_layer_sec_hits) better += (o.score < e.score);
          mp_event->tr_hitmatch(e.tr_hitmatch_id).full_rank = better + 1;
        }
      };
      rank_against_both(ptc.m_layer_hits);
      rank_against_both(ptc.m_layer_sec_hits);
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
          ptc.mp_ccrep->m_ccand.push_back(tc).addHitIdx(-1, m_rz_limits.layer_info_1().layer_id(), 0.0f);

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
        dprintf("Missed to tcand %d\n", i);
        tc.addHitIdx(-1, m_rz_limits.layer_info_1().layer_id(), 0.0f);

#ifdef MKFIT_TRACE
        // QQQQQ the parent extraction will be different; also fix: step, proper state (what is it)
        int pid = ptc.tcand().m_trace_state_id;
        int id = mp_event->trace_new_cand_state(pid, (*mp_steeringparams_iter)->m_layer, track2bivec3(ptc.tcand()), ptc.tcand().state());
        ptc.tcand().m_trace_state_id = id;
#endif
      }
    }

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
