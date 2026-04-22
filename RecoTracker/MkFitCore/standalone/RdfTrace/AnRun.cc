#include "AnRun.h"

#define EV CTX.ev

void AnRun::RunOldVecBased() {
  m_rdf_hitmatch = mkfit::RdfSources::MakeTrHitMatchDF(CTX.ev)
  .Define("C", [this](int cid) -> const TrCandState* { return & EV.trCandStates_[cid]; }, {"state_id"})
  .Define("cand_pt", [this](int cid) { return EV.trCandStates_[cid].kine.mom.R(); }, {"state_id"})
  .Define("cand_step", [](const TrCandState* C) { return C->step; }, {"C"})
  .Define("cand_layer", [](const TrCandState* C) { return C->layer; }, {"C"})
  .Define("meta_id", [](const TrCandState* C) { return C->meta_id; }, {"C"} )
  .Define("seed_gf", [this](int mid) { return EV.trSIFHforSeedByMeta_[mid].good_frac(); }, {"meta_id"})
  .Define("cand_gf", [this](int mid) { return EV.trSIFHforCandByMeta_[mid].good_frac(); }, {"meta_id"})
  ;

  { // Some ranges, could just go over a subset
    auto r = *m_rdf_hitmatch;
    std::vector<ROOT::RDF::RResultPtr<TStatistic>> stats =
      { r.Stats("cand_pt"), r.Stats("cand_step"), r.Stats("cand_layer") };

    printf("cand_pt: %f -> %f, cand_step: %f -> %f, cand_layer: %f - %f\n",
          stats[0]->GetMin(), stats[0]->GetMax(),
          stats[1]->GetMin(), stats[1]->GetMax(),
          stats[2]->GetMin(), stats[2]->GetMax());
  }
  { auto &C = NewCanvasGroup(2, 2, "cands", "Candidate properties");
    auto r = *m_rdf_hitmatch;
    C.Add(r.Histo1D("cand_pt"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("cand_step"));
    C.Add(r.Histo1D("cand_layer"));
    C.Add(r.Histo2D({"layer_v_step", "layer vs step", 60, 0, 60, 20, 0, 20}, "cand_layer", "cand_step"), "colz");
  }
  { auto &C = NewCanvasGroup(2, 2, "residuals_chi2", "Residuals and chi2");
    auto r = *m_rdf_hitmatch;
    C.Add(r.Histo1D("residual_x"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("residual_y"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("residual_z"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("kalman_chi2"), "s").add_pre(CGrp::logy);
  }

  { auto &C = NewCanvasGroup(2, 2, "chi2_by_step", "Chi2 by step");
    auto r = *m_rdf_hitmatch;
    C.Add(r.Histo2D({"chi2_vs_layer", "", 60, 0,60, 101, -1, 100}, "cand_layer", "kalman_chi2" ), "s").add_pre(CGrp::logy);
    C.Add(r.Filter("cand_step == 0").Histo1D({"chi2_step0", "chi2 on step 0", 101, -1, 100}, "kalman_chi2"), "s").add_pre(CGrp::logy);
    C.Add(r.Filter("cand_step == 1").Histo1D({"chi2_step1", "chi2 on step 1", 101, -1, 100}, "kalman_chi2"), "s").add_pre(CGrp::logy);
    C.Add(r.Filter("cand_step == 2").Histo1D({"chi2_step2", "chi2 on step 2", 101, -1, 100}, "kalman_chi2"), "s").add_pre(CGrp::logy);
  }

  // ==== large y residuals ====
  { auto &C = NewCanvasGroup(2, 2, "resy.gt.400", "Y Residuals greater than 400");

    auto r = (*m_rdf_hitmatch)
    .Filter("std::abs(residual_y) > 400")
    .Define("parent_layer", [this](const TrCandState* C) { return C->parent_id >=0 ? EV.trCandStates_[C->parent_id].layer : -1; }, {"C"} );

    C.Add(r.Histo1D("cand_layer"));
    C.Add(r.Histo1D("parent_layer"));
    C.Add(r.Histo2D({"parent_layer_v_cand_layer", "parent layer vs cand layer", 61, -1, 60, 61, -1, 60}, "parent_layer", "cand_layer"));
    C.Add(r.Histo1D("residual_x"), "s").add_pre(CGrp::logy);

    f_resdy_400 = r;
  }

  // ==== going into pixels layers ====
  // works for direct backward search only (cand_step == X)... to be tuned further
  auto define_kalman_stuff = [](RNode r) -> RNode {
    return r
    .Define("exx", "kalman_state.exx()")
    .Define("eyy", "kalman_state.eyy()")
    .Define("ezz", "kalman_state.ezz()")
    .Define("epT", "kalman_state.epT()")
    .Define("etheta", "kalman_state.etheta()")
    .Define("emomPhi", "kalman_state.emomPhi()");
  };
  auto plot_kalman_stuff = [](RNode r, CanvasGroup &C) -> void {
    C.Add(r.Histo1D("residual_x"));
    C.Add(r.Histo1D("residual_y"));
    C.Add(r.Histo1D("residual_z"));
    C.Add(r.Histo1D("dphi"));
    C.Add(r.Histo1D("dq"));
    C.Add(r.Histo1D("rank"));
    C.Add(r.Histo1D("kalman_accepted"));
    C.Add(r.Histo1D("kalman_chi2"));
    C.Add(r.Histo1D("exx"));
    C.Add(r.Histo1D("eyy"));
    C.Add(r.Histo1D("ezz"));
    C.Add(r.Histo1D("epT"));
    C.Add(r.Histo1D("etheta"));
    C.Add(r.Histo1D("emomPhi"));
  };
  { auto &C = NewCanvasGroup(5,3, "SeedIntoLay3", "Seed propagated into layer 3");
    auto r = m_rdf_hitmatch->Filter("cand_step == 0 && seed_gf > 0.9 && layer == 3 && mc_match");
    plot_kalman_stuff(define_kalman_stuff(r), C);
  }
  { auto &C = NewCanvasGroup(5,3, "SeedIntoLay2", "Seed propagated into layer 2");
    auto r = m_rdf_hitmatch->Filter("cand_step == 1 && seed_gf > 0.9 && layer == 2 && mc_match");
    plot_kalman_stuff(define_kalman_stuff(r), C);
  }

  // ==== hit goodness etc via meta ====
  { auto &C = NewCanvasGroup(2,2, "good_frac_by_meta", "good_fraction by meta");
    m_rdf_meta = mkfit::RdfSources::MakeTrCandMetaDF(CTX.ev)
    .Define("seed_gf", [this](int mid) { return EV.trSIFHforSeedByMeta_[mid].good_frac(); }, { "id" })
    .Define("cand_gf", [this](int mid) { return EV.trSIFHforCandByMeta_[mid].good_frac(); }, { "id" })
    .Define("cand_good_pix", [this](int mid) { return EV.trSIFHforCandByMeta_[mid].n_pix_match; }, { "id" })
    .Define("cand_bad_pix", [this](int mid) { return EV.trSIFHforCandByMeta_[mid].n_pix_bad(); }, { "id" })
    ;
    auto r = *m_rdf_meta;
    C.Add(r.Histo1D("seed_gf"));
    C.Add(r.Histo1D("cand_gf"));
    C.Add(r.Histo1D("cand_good_pix"));
    C.Add(r.Histo1D("cand_bad_pix"));
  }
}

//==============================================================================
// EventSource based stuff
//==============================================================================

/* Multi selection pre-proto
  struct SelectionInfo {
    std::string cut;
    std::string label;
    Color_t color;
  };

  std::vector<SelectionInfo> selections = {
    {"seed_gf > 0.9", "gf > 0.9", kRed},
    {"seed_gf > 0.7", "gf > 0.7", kBlue},
    {"seed_gf > 0.5", "gf > 0.5", kGreen},
  };
*/

namespace { // map, gather, compress

  // using RVecI = ROOT::RVec<int>;

  // ---------------------------------------------------------------------------
  // map_with_member:
  // ---------------------------------------------------------------------------
  template <typename VEC_T, typename F>
  auto map_with_member(const VEC_T& v, F f) {
    using T = std::decay_t<decltype(f(*v.begin()))>;
    ROOT::RVec<T> out;
    out.reserve(v.size());
    for (auto& x : v) out.push_back(f(x));
    return out;
  }
  // ---------------------------------------------------------------------------
  // map_with_func:
  // ---------------------------------------------------------------------------
  template <typename T, typename F>
  auto map_with_func(const std::vector<T>& v, const mkfit::Event* ev, F func) {
    using R = std::decay_t<decltype(func(ev, std::declval<T>()))>;
    ROOT::RVec<R> out;
    out.reserve(v.size());
    for (const auto& elem : v) out.push_back(func(ev, elem));
    return out;
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  template <typename VEC_T, typename F>
  auto gather_with_member(const VEC_T& v, F f, const ROOT::RVec<int>& idx_vec) {
    using T = std::decay_t<decltype(f(*v.begin()))>;
    ROOT::RVec<T> out;
    out.reserve(idx_vec.size());
    for (auto &i : idx_vec) out.push_back(f(v[i]));
    return out;
  }
  // ---------------------------------------------------------------------------
  // gather_with_func: Call function on selected indices only
  // ---------------------------------------------------------------------------
  template <typename T, typename F>
  auto gather_with_func(const std::vector<T>& v, const mkfit::Event* ev,
                        F func, const ROOT::RVec<int>& idx_vec) {
    using R = std::decay_t<decltype(func(ev, std::declval<T>()))>;
    ROOT::RVec<R> out;
    out.reserve(idx_vec.size());
    for (int i : idx_vec) out.push_back(func(ev, v[i]));
    return out;
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  template <typename VEC_T, typename F>
  auto compress_with_member(const VEC_T& v, F f, const ROOT::RVec<int>& mask) {
    assert(v.size() == mask.size());
    using T = std::decay_t<decltype(f(*v.begin()))>;
    ROOT::RVec<T> out;
    const size_t n = v.size();
    out.reserve(n); // This is aggressive, but it avoids multiple allocations.
    for (size_t i = 0; i < n; ++i) {
      if (mask[i]) out.push_back(f(v[i]));
    }
    return out;
  }
  // ---------------------------------------------------------------------------
  // compress_with_func: Call function on masked elements only
  // ---------------------------------------------------------------------------
  template <typename T, typename F>
  auto compress_with_func(const std::vector<T>& v, const mkfit::Event* ev,
                          F func, const ROOT::RVec<int>& mask) {
    assert(v.size() == mask.size());
    using R = std::decay_t<decltype(func(ev, std::declval<T>()))>;
    ROOT::RVec<R> out;
    const size_t n = v.size();
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (mask[i]) out.push_back(func(ev, v[i]));
    }
    return out;
  }

  // ---------------------------------------------------------------------------
  // mask_to_index_vec: Convert boolean mask (1s) to index list
  // e.g., {1, 0, 1, 0, 1} → {0, 2, 4}
  // ---------------------------------------------------------------------------
  inline ROOT::RVec<int> mask_to_index_vec(const ROOT::RVec<int>& mask) {
    ROOT::RVec<int> indices;
    indices.reserve(mask.size());
    for (size_t i = 0; i < mask.size(); ++i) {
      if (mask[i]) indices.push_back(i);
    }
    return indices;
  }
  // ---------------------------------------------------------------------------
  // neg_mask_to_index_vec: Convert boolean mask (0s) to index list
  // e.g., {1, 0, 1, 0, 1} → {1, 3}
  // ---------------------------------------------------------------------------
  inline ROOT::RVec<int> neg_mask_to_index_vec(const ROOT::RVec<int>& mask) {
    ROOT::RVec<int> indices;
    indices.reserve(mask.size());
    for (size_t i = 0; i < mask.size(); ++i) {
      if (!mask[i]) indices.push_back(i);
    }
    return indices;
  }
}

// Note: both EV_MEMBER and VEC_MEMBER can be data or function, m_id or pT()

#define EV_MAP(EV_MEMBER, VEC_MEMBER) \
  [](const mkfit::Event* ev) { \
    return map_with_member(ev->EV_MEMBER, [](auto& s){ return s.VEC_MEMBER; }); \
  }, { "event" }

#define EV_MAP_FUNC(EV_MEMBER, FUNC) \
  [](const mkfit::Event* ev) { \
    return map_with_func(ev->EV_MEMBER, ev, \
      [](const mkfit::Event* e, const auto& elem) { return e->FUNC(elem); }); \
  }, { "event" }

#define EV_GATHER(EV_MEMBER, VEC_MEMBER, IDX_COLUMN) \
  [](const mkfit::Event* ev, const ROOT::RVec<int> idx_vec) { \
    return gather_with_member(ev->EV_MEMBER, [](auto& s){ return s.VEC_MEMBER; }, idx_vec); \
  }, { "event", IDX_COLUMN }

#define EV_GATHER_FUNC(EV_MEMBER, FUNC, IDX_COLUMN) \
  [](const mkfit::Event* ev, const ROOT::RVec<int> indices) { \
    return gather_with_func(ev->EV_MEMBER, ev, \
      [](const mkfit::Event* e, const auto& elem) { return e->FUNC(elem); }, indices); \
  }, { "event", IDX_COLUMN }

#define EV_COMPRESS(EV_MEMBER, VEC_MEMBER, MASK_COLUMN) \
  [](const mkfit::Event* ev, const ROOT::RVec<int> mask) { \
    return compress_with_member(ev->EV_MEMBER, [](auto& s){ return s.VEC_MEMBER; }, mask); \
  }, { "event", MASK_COLUMN }

#define EV_COMPRESS_FUNC(EV_MEMBER, FUNC, MASK_COLUMN) \
  [](const mkfit::Event* ev, const ROOT::RVec<int> mask) { \
    return compress_with_func(ev->EV_MEMBER, ev, \
      [](const mkfit::Event* e, const auto& elem) { return e->FUNC(elem); }, mask); \
  }, { "event", MASK_COLUMN }

//====================================================================================

void AnRun::SetupRdfEvent(std::vector<const mkfit::Event*>& ev_vec) {
   m_rdf_event = mkfit::RdfSources::MakeEventDF(ev_vec)
  .Define("evtID", [](const mkfit::Event* ev) { return ev->evtID(); }, {"event"})
  .Define("meta_id", EV_MAP(trCandMetas_, id))
  ;
}

//====================================================================================

void AnRun::RunEventSourceTestAndDupCheck() {

  auto r = (*m_rdf_event)
  // seeds, directly from trSeeds_ -- to compare to stuff from trCandMetas_
  .Define("seed_pt",  EV_MAP(trSeeds_, pT()))
  .Define("seed_eta", EV_MAP(trSeeds_, momEta()))
  .Define("seed_n_hits", EV_MAP(trSeeds_, nFoundHits()))

  .Define("seed_gf", EV_GATHER(trSIFHforSeedByMeta_, good_frac(), "meta_id"))
  .Define("cand_gf", EV_GATHER(trSIFHforCandByMeta_, good_frac(), "meta_id" ))
  .Define("cand_good_pix", EV_GATHER(trSIFHforCandByMeta_, n_pix_match, "meta_id" ))
  .Define("cand_bad_pix", EV_GATHER(trSIFHforCandByMeta_, n_pix_bad(), "meta_id" ))
  ;


  { auto &C = NewCanvasGroup(4, 2, "seeds", "seed properties");
    C.Add(r.Histo1D("seed_pt"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("seed_eta"), "s");
    C.Add(r.Histo1D({"seed_n_hits", "", 7, 5.5, 12.5}, "seed_n_hits"), "s");
    C.Add(r.Histo2D({"seed_n_hits_vs_eta", "", 100, -4, 4, 7, 5.5, 12.5}, "seed_eta", "seed_n_hits" ), "colz");
    C.Add(r.Histo1D("seed_gf"));
    C.Add(r.Histo1D("cand_gf"));
    C.Add(r.Histo1D("cand_good_pix"));
    C.Add(r.Histo1D("cand_bad_pix"));
  }

  // Duplicate detection

  r = r
  .Define("meta_seed", EV_MAP(trCandMetas_, seed))
  .Define("n_metas", "(int) meta_id.size()")
  .Define("n_seeds", "(int) seed_pt.size()")
  .Define("n_unique_seeds",
        [](const ROOT::RVec<int>& v) -> int {
          std::set<int> s(v.begin(), v.end());
          return (int) s.size();
        },
        {"meta_seed"})
  .Define("has_duplicate_metas", "n_metas > n_unique_seeds")
  .Define("metas_per_seed_ratio", "(float)n_metas / n_seeds")

  .Define("min_seed", "Min(meta_seed)")
  .Define("max_seed", "Max(meta_seed)")
  .Define("seed_range", "max_seed - min_seed + 1")
  ;

  // === Find events with duplicates ===
  auto dup_events = r.Filter("has_duplicate_metas == true");

  printf("Events with duplicate metas: %llu / %llu\n",
         dup_events.Count().GetValue(),
         r.Count().GetValue());

  r.Foreach([](int evtID, int n_seeds, int n_metas, int seed_range,
                          int n_unique, bool has_dup, float ratio) {
    printf("%-10d ns=%-10d nm=%-10d ss_range=%-10d n_uniq_ss=%-10d has_dup=%-10s ratio=%-10.2f\n",
         evtID, n_seeds, n_metas, seed_range, n_unique, has_dup ? "YES" : "NO", ratio);
    }, {"evtID", "n_seeds", "n_metas", "seed_range",
        "n_unique_seeds", "has_duplicate_metas", "metas_per_seed_ratio"});

  // === Print details for duplicate events ===
  dup_events
    .Define("duplicate_seeds",
            [](const ROOT::RVec<int>& v) {
              std::map<int, int> counts;
              for (int x : v) counts[x]++;
              ROOT::RVec<int> dups;
              for (const auto& [val, cnt] : counts) {
                if (cnt > 1) dups.push_back(val);
              }
              return dups;
            },
            {"meta_seed"})
    .Foreach([](const ROOT::RVec<int>& dup_ss,
                const ROOT::RVec<int>& meta_id,
                const ROOT::RVec<int>& meta_seed,
                int evtID) {
      printf("Event %d: %lu duplicate seeds\n", evtID, dup_ss.size());
      for (int ss : dup_ss) {
        printf("  seed=%d: meta_ids={", ss);
        for (size_t i = 0; i < meta_seed.size(); ++i) {
          if (meta_seed[i] == ss) {
            printf("%d ", meta_id[i]);
          }
        }
        printf("}\n");
      }
    }, {"duplicate_seeds", "meta_id", "meta_seed", "evtID"});

  { auto &C = NewCanvasGroup(2, 2, "meta_dup", "search for duplicate metas");
    C.Add(r.Histo1D("n_metas"));
    C.Add(r.Histo1D("n_unique_seeds"));
    C.Add(r.Histo2D({"n_metas_vs_n_seeds", "n_metas vs n_seeds", 100, 0, 300, 100, 0, 300},
                    "n_seeds", "n_metas"), "colz");
    C.Add(r.Histo1D("has_duplicate_metas"));
  }
}

//==============================================================================

void AnRun::RunEventSourceSeedDive() {

  auto r = (*m_rdf_event)
  .Define("full_seed_gf", EV_GATHER(trSIFHforSeedByMeta_, good_frac(), "meta_id"))
  .Define("meta_mask", "full_seed_gf > 0.9")
  .Define("selected_metas", EV_COMPRESS(trCandMetas_, id, "meta_mask"))
  .Define("selected_seeds", EV_COMPRESS(trCandMetas_, seed, "meta_mask"))
  .Define("selected_sims", EV_COMPRESS(trCandMetas_, sim, "meta_mask"))

  .Define("seed_pt", EV_GATHER(trSeeds_, pT(), "selected_seeds"))
  .Define("seed_eta", EV_GATHER(trSeeds_, momEta(), "selected_seeds"))
  .Define("seed_n_hits", EV_GATHER(trSeeds_, nFoundHits(), "selected_seeds"))

  .Define("sim_pt", EV_GATHER(simTracks_, pT(), "selected_sims"))

  .Define("sim_n_pix_hits", [&](const mkfit::Event* ev, const ROOT::RVec<int> labels) {
    ROOT::RVec<int> out;
    out.reserve(labels.size());
    for (auto l : labels) {
      out.push_back(ev->countPixelHits( ev->simTracks_[l] ));
    }
    return out;
  },{"event", "selected_sims"})

  .Define("seed_n_dups", [](const mkfit::Event* ev, const ROOT::RVec<int> meta_idcs) {
    ROOT::RVec<int> out;
    std::map<int, int> lbl_to_count;
    for (auto idx : meta_idcs) {
      int label = ev->trSIFHforSeedByMeta_[idx].label;
      ++lbl_to_count[label];
    }
    for (auto [label, count] : lbl_to_count) {
      out.push_back(count - 1);
    }
    return out;
  }, {"event", "selected_metas"})

  .Define("seed_n_dups_re_pTs", [](const mkfit::Event* ev, const ROOT::RVec<int> meta_idcs) {
    ROOT::RVec<int> out;
    std::set<int> pT_lbls;
    for (int i = 0; i < ev->seedVecInsp_.n_pTNs; i++) {
      pT_lbls.insert(ev->simInfoForTrack(ev->seedTracks_[i]).label);
    }
    std::map<int, int> lbl_to_dupstate;
    for (auto idx : meta_idcs) {
      int label = ev->trSIFHforSeedByMeta_[idx].label;
      lbl_to_dupstate[label] = (pT_lbls.find(label) == pT_lbls.end()) ? 0 : 1;
    }
    for (auto [label, dupstate] : lbl_to_dupstate) {
      out.push_back(dupstate);
    }
    return out;
  }, {"event", "selected_metas"})
  ;

  r = r
  .Define("delta_pt", "(seed_pt - sim_pt)/sim_pt");
  { auto &C = NewCanvasGroup(4, 2, "seed_sim_stuff", "figuring out Txs with good_hit_frac > 0.9");
    C.Add(r.Histo1D({"delta_pt", "delta_pt/sim_pt", 100, -10, 10}, "delta_pt"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D({"seed_pt", "seed_pt", 100, 0, 30}, "seed_pt"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D({"sim_pt", "sim_pt", 100, 0, 30}, "sim_pt"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("sim_n_pix_hits"));
    C.Add(r.Histo1D({"sim_n_pix_hits", "sim_n_pix_hits", 13, -0.5, 12.5}, "sim_n_pix_hits"), "s");
    C.Add(r.Histo1D({"seed_n_dups", "seed_n_dups", 6, -0.5, 5.5}, "seed_n_dups"), "s");
    C.Add(r.Histo1D({"seed_n_dups_re_pTs", "seed_n_dups_re_pTs", 6, -0.5, 5.5}, "seed_n_dups_re_pTs"), "s");
  }
}

//==============================================================================

void AnRun::RunT5intoPix() {
  auto r = (*m_rdf_event)
  .Define("full_seed_gf", EV_GATHER(trSIFHforSeedByMeta_, good_frac(), "meta_id"))

  .Define("pre_meta_mask", "full_seed_gf >= 1.0f")
  .Define("pre_selected_metas", EV_COMPRESS(trCandMetas_, id, "pre_meta_mask"))
  .Define("pre_selected_seeds", EV_COMPRESS(trCandMetas_, seed, "pre_meta_mask"))
  .Define("pre_selected_sims", EV_COMPRESS(trCandMetas_, sim, "pre_meta_mask"))

  // .Define("pre_sim_n_pix_hits", EV_COMPRESS_FUNC(simTracks_, countPixelHits, "pre_selected_sims"))
  // .Define("pre_sim_n_pix_layers", EV_COMPRESS_FUNC(simTracks_, countPixelLayers, "pre_selected_sims"))
  .Define("pre_sim_n_pix_hits", EV_GATHER_FUNC(simTracks_, countPixelHits, "pre_selected_sims"))
  .Define("pre_sim_n_pix_layers", EV_GATHER_FUNC(simTracks_, countPixelLayers, "pre_selected_sims"))
  .Define("pre_sim_pT", EV_GATHER(simTracks_, pT(), "pre_selected_sims"))
  .Define("pre_sim_eta", EV_GATHER(simTracks_, momEta(), "pre_selected_sims"))
  .Define("pre_seed_first_layer", EV_GATHER(trSeeds_, getHitLyr(0), "pre_selected_seeds"))
  .Define("pre_sim_last_pixel_layer", EV_GATHER_FUNC(simTracks_, lastPixelLayer, "pre_selected_sims"))

  // "Final" filter for barrel, 4 sim pixel layers, first seed hit in barrel, last sim pixel in barrel
  .Define("pix_layer_mask", "    pre_sim_n_pix_layers >= 4"
                            "&& (pre_seed_first_layer == 4 || pre_seed_first_layer == 5)"
                            "&&  pre_sim_last_pixel_layer == 3")

  .Define("selected_sims", "pre_selected_sims[pix_layer_mask]")
  .Define("selected_metas", "pre_selected_metas[pix_layer_mask]")
  .Define("sim_n_pix_hits", "pre_sim_n_pix_hits[pix_layer_mask]")
  .Define("sim_n_pix_layers", "pre_sim_n_pix_layers[pix_layer_mask]")

  .Define("sim_pT", "pre_sim_pT[pix_layer_mask]")
  .Define("sim_eta", "pre_sim_eta[pix_layer_mask]")
  ;

  { auto &C = NewCanvasGroup(4, 2, "t5intoPix", "Tracing Tx into pixels -- selections");
    C.AddIntH1D(r, "pre_sim_n_pix_hits", 0, 20, "s").add_pre(CGrp::logy);
    C.AddIntH1D(r, "pre_sim_n_pix_layers", 0, 12, "s").add_pre(CGrp::logy);
    C.AddIntH1D(r, "sim_n_pix_hits", 0, 20, "s").add_pre(CGrp::logy);
    C.AddIntH1D(r, "sim_n_pix_layers", 0, 12, "s").add_pre(CGrp::logy);

    C.Add(r.Histo1D("sim_pT"), "s");
    C.Add(r.Histo1D("sim_eta"), "s");
  }
}

//==============================================================================

void AnRun::DrawCanvasGroups() {
  for (auto &cg : m_canvas_groups) {
    cg->Draw();
  }
}
