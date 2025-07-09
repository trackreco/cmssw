#include "MkFinderV2p2.h"

#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "FindingFoos.h"
#include "KalmanUtilsMPlex.h"

#include "MatriplexPackers.h"
#include "MiniPropagators.h"
#include "MkBins.h"

#define DEBUG
#include "Debug.h"

namespace mkfit {

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

  }

  //------------------------------------------------------------------------------
  // Helpers.

  int MkFinderV2p2::awaken_candidates() {
    // Awaken dormant ccands with hits on current layer for processing on the next layer.
    int count = 0;
    const LayerControl &lc = mp_steeringparams_iter->layer_control();
    for (auto &ccand : m_batch_mgr) {
      if (ccand.state() == CombCandidate::Dormant &&
          (ccand.pickupLayer() == lc.m_layer || ccand.pickupLayer() == lc.m_layer_sec)) {
        ccand.setState(CombCandidate::Finding);
        auto ccrep = m_active_ccreps.emplace_back(m_hot_tub, ccand);
        dprintf("MkFinderV2p2::awaken_candidates dummy printout N_TrackCands=%d\n",
               (int) ccand.size());
        ++count;
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
    // Count number of awakend ccands and number of non-stopped tcands.
    // Well, I actually know n ccands.

    // debug = true; // to be disabled at the end of end_layer()

    dprintf("MkFinderV2p2::begin_layer Expecting %d active ccands\n", (int) m_active_ccreps.size());
    int i = 1;
    for (auto &ccrep : m_active_ccreps) {
      dprintf("  %2d. seed-idx=%d, n_tcands=%d\n", i,
             ccrep.m_ccand.seed_origin_index(), (int) ccrep.m_ccand.size());
      ++i;
    }

    m_batch_mgr.reset_for_new_layer();
    m_active_ccreps_pos = m_active_ccreps.begin();
    m_active_ccreps_tC_pos = 0;

    // Hmmh, nothing to really do here, is it?

    // Initialize best short (?) - to what? How was it before? The seed itself?
    // Or let this happen after the first layer is processed?
    // Score is set to worst-possible elsewhere.
  }

  void MkFinderV2p2::begin_next_Ccrep_in_layer() {
    CCandRep &ccrep = * m_active_ccreps_pos;
    CombCandidate &ccand = ccrep.m_ccand;
    ccrep.m_pTcs.reserve(ccand.size());
    for (int ic = 0; ic < (int) ccand.size(); ++ic) {
      // TrackCand &tcand = ccand[ic];
      // XXXX V1 also did: min-pt-cut, apogee stop; and setting ccand.setState(CombCandidate::Finished)
      if (ccand[ic].getLastHitIdx() != -2) {
        PrimTCandRep &ptc = ccrep.m_pTcs.emplace_back( &ccrep, ic );

        m_pre_select_queue.push_back(&ptc);
      }
    }
    ++m_active_ccreps_pos;
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
      // XXXX something is rotten here;
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

    dprintf("MkFinderV2p2::end_layer %d cands finished\n", count);

    if (m_batch_mgr.has_dormant_ccands())
      awaken_candidates();

    debug = false;
  }

  //------------------------------------------------------------------------------
  // The main processing function -- process_layer()

  void MkFinderV2p2::process_layer() {

    bool any_Ccs_to_finalize = false;
    bool enough_sTcs_to_prop_n_kalman = false;

    // BatchManager &BM = m_batch_mgr;

  do_Ccs_finalize:
    while (any_Ccs_to_finalize) {
      // process front Cc;
      // pop it off and release pTcs and their sTcs (probabl done as part of the above)
    }

  do_sTcs_prop_n_kalman:
    dprintf("BOO any-ccrepsto-begin=%d\n", any_Ccreps_to_begin());
    // AAAA should this be while ... what to do with the else below then?
    if (enough_sTcs_to_prop_n_kalman || ! any_Ccreps_to_begin()) {
      // prop & Kalman the NN batch
      // process reults in the context of corresponding pTcs and Ccs
      //
      // This can result in:
      // a) new (later-stage) sTcs becoming available
      // b) some pTcs being finished or unviable
      // c) some Ccs becoming fully processed (through all their pTcs being finished)
      // If c), finalize those Ccs right away to get them out of the hair / release slots.
      if (any_Ccs_to_finalize)
        goto do_Ccs_finalize;
    }

  // do_pTcs_pre_select:


  do_Ccs_initialize:
    if (any_Ccreps_to_begin()) {
      // pop one off, initialize Cc, populate with pTcs.
      begin_next_Ccrep_in_layer();

      // This if should be while? But, what about the else below ...?
      // Also think what happens in pre-select and if hit-matching is separate
      if (enough_work_for_pre_select() || ( ! any_Ccreps_to_begin() && any_work_for_pre_select())) {
        // do Binnor stuff, generate hit-lists / pre-selections / bi-layer planning
        // generate some amount of sTcs for each pTc, presumably to start prop-to-first hit
        process_pre_select();
      } else {
        if (any_Ccreps_to_begin())
          goto do_Ccs_initialize;
      }

      goto do_sTcs_prop_n_kalman;
    }
  }

  //============================================================================
  // More complex functions -- to separate then from "logic" flow
  //============================================================================

  //----------------------------------------------------------------------------
  // process_pre_select()
  // Propagate to layer edges, claclulate layer-of-hits bin ranges and
  // determine (some?) candidate hits.
  //----------------------------------------------------------------------------

  void MkFinderV2p2::process_pre_select() {

    SteeringParams::iterator &spi = *mp_steeringparams_iter;

    const bool is_dual_layer = spi->has_second_layer();
    const LayerInfo &LI_p = mp_job->m_trk_info[ spi->m_layer ];
    const LayerInfo &LI_s = mp_job->m_trk_info[ is_dual_layer ? spi->m_layer_sec : 0 ];

    const bool is_barrel = LI_p.is_barrel();
    if (is_dual_layer)
      assert(LI_s.is_barrel() == is_barrel);

    const int N_proc = std::min(NN, (int) m_pre_select_queue.size());
    dprintf("MkFinderV2p2::process_pre_select work queue is %d, would process %d of them (NN=%d)\n",
            (int) m_pre_select_queue.size(), N_proc, NN);

    MkBins B(is_barrel, N_proc);
    PrimTCandRep *prim_tcand_ptrs[NN];
    MPlexQF phi(0.0f);
    MkBinTrackCovExtract TCE;
    int i = 0;
    while (i < N_proc) {
      PrimTCandRep &ptc = * m_pre_select_queue.front();
      prim_tcand_ptrs[i] = & ptc;
      TrackCand &tc = ptc.tcand();
      m_pre_select_queue.pop_front();

      // Rewrite with dedicated mplex packer?
      B.m_isp.x[i] = tc.x();
      B.m_isp.y[i] = tc.y();
      B.m_isp.z[i] = tc.z();
      B.m_isp.inv_pt[i] = tc.invpT();
      B.m_isp.theta[i] = tc.theta();
      TCE.m_cov_0_0 = tc.errors().At(0, 0);
      TCE.m_cov_0_1 = tc.errors().At(0, 1);
      TCE.m_cov_1_1 = tc.errors().At(1, 1);
      TCE.m_cov_2_2 = tc.errors().At(2, 2);
      phi[i] = tc.momPhi();
      m_Chg[i] = tc.charge();

      ++i;
    }
    B.m_isp.init_momentum_vec_and_k(phi, m_Chg);

    MkRZLimits rz_lim;
    if (is_dual_layer) {
      rz_lim.setup(LI_p, LI_s);
    } else {
      rz_lim.setup(LI_p);
    }
    B.prop_to_limits(rz_lim);

    // At this point we should check for prop-failures and/or if limits have
    // been reached.
    // Set the WSR.
    // There are also the apogee and minPt checks ... but those might be better done
    // elsewhere.

    B.determine_bin_windows(TCE);

    MkBinLimits BL_p;
    B.find_bin_ranges(mp_job->m_event_of_hits[spi->m_layer], BL_p);

    // This might belong better somewhere else, TPrimCanRep? Calculation into minipropagators.
    // Anyway, here for now.

    namespace mp = mini_propagators;

    mp::Hermite3D H;
    H.calculate_coeffs(B.m_sp1, B.m_sp2, B.m_isp.inv_k);

    for (int i = 0; i < N_proc; ++i) {
        dprintf("%d: BinCheck Prim %c %+8.6f %+8.6f | %3d %3d || %+8.6f %+8.6f | %2d %2d\n",
                i, LI_p.is_barrel() ? 'B' : 'E',
                B.m_phi_center[i], B.m_phi_delta[i], BL_p.p1[i], BL_p.p2[i],
                B.m_q_min[i], B.m_q_max[i], BL_p.q1[i], BL_p.q2[i]);
    }

    MPlexQF hx, hy, hz;
    MPlexQF hdx, hdy, hdz;
    H.evaluate(0.0f, hx, hy, hz, hdx, hdy, hdz);
    for (int i = 0; i < N_proc; ++i) {
      dprintf("%d: sp1 %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | dalpha = %.4f |   AT 0\n"
              "   her %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f -- derfac %.5f\n", i,
             B.m_sp1.x[i], B.m_sp1.y[i], B.m_sp1.z[i],
             B.m_sp1.px[i], B.m_sp1.py[i], B.m_sp1.pz[i], B.m_sp1.dalpha[i],
             hx[i], hy[i], hz[i],
             hdx[i], hdy[i], hdz[i], H.m_Hderfac[i]);
    }
    H.evaluate(0.5f, hx, hy, hz, hdx, hdy, hdz);
    for (int i = 0; i < N_proc; ++i) {
      dprintf("%d: isp %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | dalpha = %.4f |   AT 0.5\n"
              "   her %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f -- derfac %.5f\n", i,
             B.m_isp.x[i], B.m_isp.y[i], B.m_isp.z[i],
             B.m_isp.px[i], B.m_isp.py[i], B.m_isp.pz[i], B.m_isp.dalpha[i],
             hx[i], hy[i], hz[i],
             hdx[i], hdy[i], hdz[i], H.m_Hderfac[i]);
    }
    H.evaluate(1.0f, hx, hy, hz, hdx, hdy, hdz);
    for (int i = 0; i < N_proc; ++i) {
      dprintf("%d: sp2 %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f | dalpha = %.4f |   AT 1\n"
              "   her %7.3f %7.3f %7.3f | %7.3f %7.3f %7.3f -- derfac %.5f\n", i,
             B.m_sp2.x[i], B.m_sp2.y[i], B.m_sp2.z[i],
             B.m_sp2.px[i], B.m_sp2.py[i], B.m_sp2.pz[i], B.m_sp2.dalpha[i],
             hx[i], hy[i], hz[i],
             hdx[i], hdy[i], hdz[i], H.m_Hderfac[i]);
    }

    MkBinLimits BL_s; // This should really be optional ... or somewhere else ... well, both.
    if (is_dual_layer) {
      B.find_bin_ranges(mp_job->m_event_of_hits[spi->m_layer_sec], BL_s);
      for (int i = 0; i < N_proc; ++i) {
        dprintf("%d: BinCheck Sec  %c %+8.6f %+8.6f | %3d %3d || %+8.6f %+8.6f | %2d %2d\n",
                i, LI_s.is_barrel() ? 'B' : 'E',
                B.m_phi_center[i], B.m_phi_delta[i], BL_s.p1[i], BL_s.p2[i],
                B.m_q_min[i], B.m_q_max[i], BL_s.q1[i], BL_s.q2[i]);
      }
    }

    // Prototype for extract hits

    int fill_pos = 0;
    namespace mp = mini_propagators;
    mp::InitialStatePlex is_plex; // initial state
    mp::StatePlex h_plex; // state on hit
    MPlexQI prim_idcs; // primary indices into input and MkBins
    MPlexQUI hit_idcs;
    MPlexQUI hit_orig_idcs;

    auto select_hits = [&](const LayerOfHits& L, int N_proc_hits) {
      MPlex3V module_pos;
      MPlex3V module_norm;
      mp::Hermite3D h3d;

      // Extract hit / target module data
      for (int h = 0; h < N_proc_hits; ++h) {
        const Hit &hit = L.refHit(hit_orig_idcs[h]);
        unsigned int mid = hit.detIDinLayer();
        const ModuleInfo &mi = L.layer_info().module_info(mid);
        module_pos.copyIn(h, mi.pos.Array());
        module_norm.copyIn(h, mi.zdir.Array());

        h3d.copyIn(h, H, prim_idcs[h]);
      }

      is_plex.propagate_to_plane(mp::PA_Line, module_pos, module_norm, h_plex, true);

      mp::Hermite3DOnPlane h3dop;
      h3dop.init_coeffs(h3d, module_pos, module_norm);
      MPlexQF t2, d2, d3;
      t2 = h3dop.m_T;
      h3dop.evaluate(h3dop.m_T, d2);
      h3dop.solve();
      h3dop.evaluate(h3dop.m_T, d3);

      mp::StatePlex h3_state;
      h3d.evaluate(h3dop.m_T, h3_state);
      // h3_state.dalpha calculated below, as needed

      // Just for printouts, internal to init_coeffs()
      MPlexQF d0, d1;
      h3dop.evaluate(0.0f, d0);
      h3dop.evaluate(1.0f, d1);

      // Post-process hits into a heap in PrimTCandReps
      for (int h = 0; h < N_proc_hits; ++h) {
        PrimTCandRep &ptc = * prim_tcand_ptrs[ prim_idcs[h] ];
        float q, ddq, phi, ddphi;
        if (is_barrel) {
          q = h_plex.z[h];
        } else {
          q = hipo(h_plex.x[h], h_plex.y[h]);
        }
        ddq = std::abs(q - L.hit_q(hit_idcs[h]));
        phi = vdt::fast_atan2f(h_plex.y[h], h_plex.x[h]);
        ddphi = cdist(std::abs(phi - L.hit_phi(hit_idcs[h])));


        bool dqdphi_presel = ddq < B.m_dq_track[prim_idcs[h]] + MkBins::DDQ_PRESEL_FAC * L.hit_q_half_length(hit_idcs[h]) &&
                             ddphi < B.m_dphi_track[prim_idcs[h]] + MkBins::DDPHI_PRESEL_FAC * 0.0123f;

        // To be moved down, only for hits that pass pre-selection, needed here for printout.
        // Could be vectorized if we repack binnor stuff.
        h3_state.dalpha[h] = B.m_sp1.dalpha[prim_idcs[h]] + h3dop.m_T[h]*(B.m_sp2.dalpha[prim_idcs[h]] - B.m_sp1.dalpha[prim_idcs[h]]);

        // clang-format off
        dprintf("     SelHit %6.3f %6.3f %6.4f %7.5f   %6.4f   %s\n",
                L.hit_q(hit_idcs[h]), L.hit_phi(hit_idcs[h]),
                ddq, ddphi, h_plex.dalpha[h], dqdphi_presel ? "PASS" : "REJECT");

        dprintf("      H3 d0=%.4f d1=%.4f -> d2=%e t2=%e -> d3=%e t3=%e ... dalpha=%6.4f\n",
               d0[h], d1[h], d2[h], t2[h], d3[h], h3dop.m_T[h],
               h3_state.dalpha[h]);
              //  B.m_sp1.dalpha[prim_idcs[h]] + h3dop.m_T[h]*(B.m_sp2.dalpha[prim_idcs[h]] - B.m_sp1.dalpha[prim_idcs[h]]));
        dprintf("      H3 PARS %f %f %f; %f %f %f\n", h3_state.x[h], h3_state.y[h], h3_state.z[h],
          h3_state.px[h], h3_state.py[h], h3_state.pz[h]);
        // clang-format on

        if (/*prop_fail || */ !dqdphi_presel)
          continue;

        // float dalpha = h_plex.dalpha[h];
        // is_plex might come from somewhere else, through another index.

        if (ptc.m_pqueue_size < MkBins::NEW_MAX_HIT) {
          ptc.m_pqueue.push( { ddphi, hit_orig_idcs[h], hit_idcs[h], L.layer_id(), { h3_state, h, is_plex, h } } );
          ++ptc.m_pqueue_size;
        } else if (ddphi < ptc.m_pqueue.top().score) {
          ptc.m_pqueue.pop();
          ptc.m_pqueue.push( { ddphi, hit_orig_idcs[h], hit_idcs[h], L.layer_id(), { h3_state, h, is_plex, h } } );
        }
      }
    }; // end lambda select_hits

    {
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

              dprintf(" %d: P_HIT %3u %4u %5u   %6.3f %6.3f %6.3f\n",
                i, pi, qi, hi, L.hit_phi(hi), L.hit_q(hi), L.hit_qbar(hi));

              if (iteration_hit_mask && (*iteration_hit_mask)[hi_orig]) {
                dprintf("Yay, denying masked hit on layer %u, hi %u, orig idx %u\n",
                        L.layer_info().layer_id(), hi, hi_orig);
                continue;
              }

              // Try preloading Hits for the next step ... probably not really relevant.
              _mm_prefetch(&L.refHit(hi_orig), _MM_HINT_T0);

              prim_idcs[fill_pos] = i;
              hit_idcs[fill_pos] = hi;
              hit_orig_idcs[fill_pos] = hi_orig;
              is_plex.copyIn(fill_pos, B.m_isp, i);

              if (++fill_pos == NN) {
                select_hits(L, NN);
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
        select_hits(L, fill_pos);
      }
    }


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
    // Should really go into KalmanOpArgs directly, and processed as needed.
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      while (ptc.m_pqueue_size) {
        --ptc.m_pqueue_size;
        const auto &pqe = ptc.m_pqueue.top();
        ptc.m_layer_hits.push_back( pqe );
        // dprintf("pushing for %d  %f, %u %u\n", i, pqe.score, pqe.hit_orig_index, pqe.hit_index);
        ptc.m_pqueue.pop();
      }
    }

    struct KalmanOpArgs {

      const PropagationConfig *prop_config = nullptr;

      PrimTCandRep *ptcp[NN];
      HitOnTrack    hot[NN];

      mp::InitialStatePlex tsXyz;
      MPlexLS tsErr; // input (on prev hit) and output (on current hit) [ts - track-state]
      MPlexLV tsPar; // ""
      MPlexQI tsChg; // "" Kalman update can flip it through curvature flip
      MPlexHS msErr; // input measurement / hit [ms - measurement state]
      MPlexHV msPar; // ""
      MPlexHV plNrm; // input detector plane [pl - plane]
      MPlexHV plDir; // ""
      MPlexHV plPnt; // ""
      MPlexQF sPerp; // path-length in transverse plane, calculated from alpha. p2plane really needs 3D path.

      MPlexLS propErr; // intermediate: propagated from tsErr and used as input to Kalman
      MPlexLV propPar; // input: pre-propagated as part of hit pre-selection

      MPlexQF tsChi2; // output
      MPlexQI outFailFlag; // dummy, can be detected in pre-propagation, no other errors detected / reported
      int N_filled = 0;

      void reset() { N_filled = 0; }

      // There will be some more of this state, also secondary or who knows what.
      void item_begin(PrimTCandRep *ptc, HitOnTrack ht) { ptcp[N_filled] = ptc; hot[N_filled] = ht; }
      bool item_finished() { return ++N_filled == NN; }

      void load_state_err_chg(const mp::InitialState &state_on_hit, const TrackBase &tb) {
        tsXyz.copyIn(N_filled, state_on_hit);
        tsPar.copyIn(N_filled, tb.posArray()); // propToPlane needs initial parameters, too
        tsErr.copyIn(N_filled, tb.errArray());
        tsChg[N_filled] = tb.charge();
      }

      void load_hit_module(const Hit &hit, const ModuleInfo & mi) {
        msErr.copyIn(N_filled, hit.errArray());
        msPar.copyIn(N_filled, hit.posArray());
        plNrm.copyIn(N_filled, mi.zdir.Array());
        plDir.copyIn(N_filled, mi.xdir.Array());
        plPnt.copyIn(N_filled, mi.pos.Array());
      }

      void compute_pars() {
        // Parameters are stored in StatePlex -- so we can vectorize translation to pt, phi, theta.
        // Some stuff could be passed over as it won't change before update: pt, theta, k_inv
        // They are passed in output-parameters as propagation also needs input pars.
        propPar.aij(0, 0) = tsXyz.x;
        propPar.aij(1, 0) = tsXyz.y;
        propPar.aij(2, 0) = tsXyz.z;
        propPar.aij(3, 0) = tsXyz.inv_pt;
        propPar.aij(4, 0) = Matriplex::fast_atan2(tsXyz.py, tsXyz.px);
        propPar.aij(5, 0) = tsXyz.theta;

        sPerp = tsXyz.dalpha / ( tsXyz.inv_pt * tsXyz.inv_k);
      }

      void do_kalman_stuff() {
        dprintf("do_kalman_stuff\n");
        for (int i = 0; i < N_filled; ++i) {
          dprintf("  %d: %f %f %f : %f %f %f : %f\n", i, tsXyz.x[i], tsXyz.y[i], tsXyz.z[i],
                 tsXyz.inv_pt[i], vdt::fast_atan2(tsXyz.py[i], tsXyz.px[i]), tsXyz.theta[i],
                 sPerp[i]);
        }
        propagateHelixToPlaneMPlex(tsErr, tsPar, tsChg, plPnt, plNrm, &sPerp,
                                   propErr, propPar, outFailFlag,
                                   N_filled, prop_config->finding_inter_layer_pflags, nullptr);
        kalmanOperationPlaneLocal(KFO_Calculate_Chi2 | KFO_Update_Params | KFO_Local_Cov,
                                  propErr, propPar, tsChg, msErr, msPar, plNrm, plDir, plPnt,
                                  tsErr, tsPar, tsChi2, N_filled);
        kalmanCheckChargeFlip(tsPar, tsChg, N_filled);

        // The original -- but Chi2 only.
        // kalmanPropagateAndComputeChi2Plane(tsErr, tsPar, tsChg, msErr, msPar, plNrm, plDir, plPnt,
        //                       nullptr,
        //                       tsChi2,
        //                       propPar,
        //                       outFailFlag,
        //                       N_filled,
        //                       prop_config->finding_intra_layer_pflags,
        //                       prop_config->finding_requires_propagation_to_hit_pos);

        // Update prim candidate state for best hit -- to be generalized
        dprintf("Kalman post-update check:\n");
        for (int i = 0; i < N_filled; ++i) {
          if (tsChi2[i] < ptcp[i]->bChi2) {
            dprintf("  Updating for i=%d, old-chi2 %f, new %f\n", i, ptcp[i]->bChi2, tsChi2[i]);
            tsErr.copyOut(i, ptcp[i]->bState.errors.Array());
            tsPar.copyOut(i, ptcp[i]->bState.parameters.Array());
            ptcp[i]->bState.charge = tsChg[i];
            ptcp[i]->bHot = hot[i];
            ptcp[i]->bChi2 = tsChi2[i];
          }
        }
      }
    };

    // auto call_kalman = [&](KalmanOpArgs& K) {
    // }; // end lambda call_kalman

    // KalmanProp, prim layer -- should also be done in-line as slots fill up.
    KalmanOpArgs koa;
    koa.prop_config = & mp_job->m_trk_info.prop_config();

    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      int nlh = ptc.m_layer_hits.size();
      for (int lh = 0; lh < nlh; ++ lh) {
        const PrimTCandRep::PQE &hie = ptc.m_layer_hits[lh];
        dprintf("scheduling %d %d %d %f\n", i, hie.hit_index, hie.hit_orig_index, hie.mixed_state.dalpha);

        // Need to build all the Matriplexes for KalmanOperationPlane ... or some variant
        // There really needs to be an intermediate structure with packers so we can
        // set them up incrementally.

        TrackCand &tc = ptc.tcand();
        koa.item_begin(&ptc, { (int) hie.hit_orig_index, hie.layer });
        koa.load_state_err_chg(hie.mixed_state, tc);

        const auto &L = mp_job->m_event_of_hits[ hie.layer ];
        const Hit &hit = L.refHit( hie.hit_orig_index );
        unsigned int mid = hit.detIDinLayer();
        const ModuleInfo &mi = L.layer_info().module_info(mid);
        koa.load_hit_module(hit, mi);

        if (koa.item_finished()) {
          koa.compute_pars();
          koa.do_kalman_stuff();
          koa.reset();
        }
      }
    }
    if (koa.N_filled > 0) {
      koa.compute_pars();
      koa.do_kalman_stuff();
      koa.reset();
    }

    // This, esp. the combinatorial part should be done once prim-tcand is finished.
    // And, merging results, when ccand is finished.
    for (int i = 0; i < N_proc; ++i) {
      PrimTCandRep &ptc = * prim_tcand_ptrs[i];
      TrackCand &tc = ptc.tcand();
      if (ptc.bChi2 < 20.0f) {
        dprintf("Output to tcand idx=%d, layer=%d, chi2=%f\n", ptc.bHot.index, ptc.bHot.layer, ptc.bChi2);
        tc.addHitIdx(ptc.bHot.index, ptc.bHot.layer, ptc.bChi2);
        tc.setState(ptc.bState);
      } else {
        tc.addHitIdx(-1, ptc.bHot.layer, 0.0f);
      }
    }

    // Let's try picking the secondary hit
    /*
    if (is_dual_layer) {
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
                const PrimTCandRep::HIE &hie = ptc.m_layer_hits[j];
                dprintf("           prim %d        %6.3f %6.3f %6.3f  \n", j,
                  X.hit_phi(hie.hit_index), X.hit_q(hie.hit_index), X.hit_qbar(hie.hit_index));
              }
            }
          }
        }
      }
    }
    */
  }

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

  /* //// int MkFinderV2p2::unroll_candidates()
  int MkBuilder::find_tracks_unroll_candidates_v2p2(std::vector<std::pair<int, int>> &seed_cand_vec,
                                                    int start_seed,
                                                    int end_seed,
                                                    int layer,
                                                    SteeringParams::IterationType_e iteration_dir) {
    int silly_count = 0;

    seed_cand_vec.clear();

    auto &iter_params = (iteration_dir == SteeringParams::IT_BkwSearch) ? m_job->m_iter_config.m_backward_params
                                                                        : m_job->m_iter_config.m_params;

    for (int iseed = start_seed; iseed < end_seed; ++iseed) {
      CombCandidate &ccand = m_event_of_comb_cands[iseed];

      if (ccand.state() == CombCandidate::Finding) {
        bool active = false;
        for (int ic = 0; ic < (int)ccand.size(); ++ic) {
          if (ccand[ic].getLastHitIdx() != -2) {
            // Stop candidates with pT<X GeV
            if (ccand[ic].pT() < iter_params.minPtCut) {
              ccand[ic].addHitIdx(-2, layer, 0.0f);
              continue;
            }
            // Check if the candidate is close to it's max_r, pi/2 - 0.2 rad (11.5 deg)
            if (iteration_dir == SteeringParams::IT_FwdSearch && ccand[ic].pT() < 1.2) {
              const float dphi = std::abs(ccand[ic].posPhi() - ccand[ic].momPhi());
              if (ccand[ic].posRsq() > 625.f && dphi > 1.371f && dphi < 4.512f) {
                // dprintf("Stopping cand at r=%f, posPhi=%.1f momPhi=%.2f pt=%.2f emomEta=%.2f\n",
                //        ccand[ic].posR(), ccand[ic].posPhi(), ccand[ic].momPhi(), ccand[ic].pT(), ccand[ic].momEta());
                ccand[ic].addHitIdx(-2, layer, 0.0f);
                continue;
              }
            }

            active = true;
            seed_cand_vec.push_back(std::pair<int, int>(iseed, ic));
            // ccand[ic].resetOverlaps();

            if constexpr (Const::nan_n_silly_check_cands_every_layer) {
              if (ccand[ic].hasSillyValues(Const::nan_n_silly_print_bad_cands_every_layer,
                                           Const::nan_n_silly_fixup_bad_cands_every_layer,
                                           "Per layer silly check"))
                ++silly_count;
            }
          }
        }
        if (!active) {
          ccand.setState(CombCandidate::Finished);
        }
      }
    }

    if constexpr (Const::nan_n_silly_check_cands_every_layer && silly_count > 0) {
      m_nan_n_silly_per_layer_count += silly_count;
    }

    return seed_cand_vec.size();
  }
  */
} // namespace mkfit
