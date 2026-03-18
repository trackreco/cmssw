#include "AnRun.h"

#define EV CTX.ev
#define TI CTX.trk_info

void AnRun::RunBase() {
  m_rdf = std::make_unique<ROOT::RDataFrame>(mkfit::RdfSources::MakeTrHitMatchDF(CTX.ev));

  auto r = (*m_rdf)
  .Define("C", [this](int cid) -> const TrCandState* { return & EV.trCandStates_[cid]; }, {"state_id"})
  .Define("cand_pt", [this](int cid) { return EV.trCandStates_[cid].kine.mom.R(); }, {"state_id"})
  .Define("cand_step", [](const TrCandState* C) { return C->step; }, {"C"})
  .Define("cand_layer", [](const TrCandState* C) { return C->layer; }, {"C"})
  .Define("meta_id", [](const TrCandState* C) { return C->meta_id; }, {"C"} )
  .Define("seed_gf", [this](int mid) { return EV.trSIFHforSeedByMeta_[mid].good_frac(); }, {"meta_id"})
  .Define("cand_gf", [this](int mid) { return EV.trSIFHforCandByMeta_[mid].good_frac(); }, {"meta_id"})
  ;
  m_base = std::make_unique<ROOT::RDF::RNode>(r);

  { // Some ranges, could just go over a subset
    std::vector<ROOT::RDF::RResultPtr<TStatistic>> stats =
      { r.Stats("cand_pt"), r.Stats("cand_step"), r.Stats("cand_layer") };

    printf("cand_pt: %f -> %f, cand_step: %f -> %f, cand_layer: %f - %f\n",
          stats[0]->GetMin(), stats[0]->GetMax(),
          stats[1]->GetMin(), stats[1]->GetMax(),
          stats[2]->GetMin(), stats[2]->GetMax());
  }
  { auto &C = NewCanvasGroup(2, 2, "cands", "Candidate properties");
    C.Add(r.Histo1D("cand_pt"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("cand_step"));
    C.Add(r.Histo1D("cand_layer"));
    C.Add(r.Histo2D({"layer_v_step", "layer vs step", 60, 0, 60, 20, 0, 20}, "cand_layer", "cand_step"), "colz");
  }
  { auto &C = NewCanvasGroup(2, 2, "residuals_chi2", "Residuals and chi2"); 
    C.Add(r.Histo1D("residual_x"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("residual_y"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("residual_z"), "s").add_pre(CGrp::logy);
    C.Add(r.Histo1D("kalman_chi2"), "s").add_pre(CGrp::logy);
  }

  { auto &C = NewCanvasGroup(2, 2, "chi2_by_step", "Chi2 by step"); 
    C.Add(r.Histo2D({"chi2_vs_layer", "", 60, 0,60, 101, -1, 100}, "cand_layer", "kalman_chi2" ), "s").add_pre(CGrp::logy);
    C.Add(r.Filter("cand_step == 0").Histo1D({"chi2_step0", "chi2 on step 0", 101, -1, 100}, "kalman_chi2"), "s").add_pre(CGrp::logy);
    C.Add(r.Filter("cand_step == 1").Histo1D({"chi2_step1", "chi2 on step 1", 101, -1, 100}, "kalman_chi2"), "s").add_pre(CGrp::logy);
    C.Add(r.Filter("cand_step == 2").Histo1D({"chi2_step2", "chi2 on step 2", 101, -1, 100}, "kalman_chi2"), "s").add_pre(CGrp::logy);
  }

  // ==== large y residuals ====
  {
    auto r = m_base
      ->Filter("std::abs(residual_y) > 400")
      .Define("parent_layer", [this](const TrCandState* C) { return C->pid >=0 ? EV.trCandStates_[C->pid].layer : -1; }, {"C"} );

    auto &C = NewCanvasGroup(2, 2, "resy.gt.400", "Y Residuals greater than 400");
    C.Add(r.Histo1D("cand_layer"));
    C.Add(r.Histo1D("parent_layer"));
    C.Add(r.Histo2D({"parent_layer_v_cand_layer", "parent layer vs cand layer", 61, -1, 60, 61, -1, 60}, "parent_layer", "cand_layer"));
    C.Add(r.Histo1D("residual_x"), "s").add_pre(CGrp::logy);

    f_resdy_400 = std::make_unique<ROOT::RDF::RNode>(r);
  }

  DrawCanvasGroups();
}

void AnRun::DrawCanvasGroups() {
  for (auto &cg : m_canvas_groups) {
    cg->Draw();
  }
}
