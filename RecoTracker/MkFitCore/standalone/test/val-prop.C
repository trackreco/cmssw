#include "RecoTracker/MkFitCMS/standalone/Shell.h"
// Driver for the synthetic propagation / Kalman / fit validation.
//
// Same pattern as prop-kalman-validation.C: the work is in the compiled
// library, this file only gives cling a declaration to call through, so that
// no Matriplex header has to be parsed by the interpreter.
//
//   cd /foo/matevz/mic-dev/current/src/standalone
//   unset DISPLAY; export LD_LIBRARY_PATH=.
//   echo .q | ./mkFit --geom CMS-phase2 --input-file <any .bin> --num-events 1 \
//     --num-thr 1 --shell \
//     --shell-command 'gROOT->SetBatch(kTRUE)' \
//     --shell-command 'gROOT->ProcessLine(".L ../RecoTracker/MkFitCore/standalone/test/val-prop.C")' \
//     --shell-command 'val_gen("val-closure.root")'
//
// No geometry is used by the generator; the shell just wants a file to start.

#include "RecoTracker/MkFitCore/standalone/RdfTrace/ValProp.h"

void val_gen(const char *out = "val-closure.root") { mkfit::val_gen_closure(out); }

void val_dalpha_reset() { mkfit::val_dalpha_reset(); }
void val_dalpha_ev(const mkfit::Event *ev, int which = 0, float pt_min = 0.5) {
  mkfit::val_dalpha_add_event(ev, which, pt_min);
}
void val_dalpha_write(const char *out = "val-dalpha.root") { mkfit::val_dalpha_write(out); }

void val_info(const mkfit::Event *ev) { mkfit::val_sample_info(ev); }

void val_anchor(const mkfit::Event *ev, int n = 5) { mkfit::val_simtrack_anchor(ev, n); }

void val_seedcmp(const mkfit::Event *ev) { mkfit::val_seed_vs_cmssw(ev); }

void val_seedcov(const mkfit::Event *ev, int kind = 1) { mkfit::val_seed_cov(ev, kind); }

void val_covx(const char *f = "val-covxport.root", int n = 10000) { mkfit::val_cov_transport(f, n); }
void val_force_mc(bool on) { mkfit::val_force_mc(on); }
void val_max_cands(int n) { mkfit::val_max_cands(n); }
void val_sr_mat(bool on) { mkfit::val_search_material(on); }
void val_surf_q(bool on) { mkfit::val_mkbins_surface_q(on); }
void val_extra_dq(float f) { mkfit::val_extra_dq(f); }
void val_surf_q_hit(bool on) { mkfit::val_surf_q_hit(on); }
void val_in_layer_comb(bool on) { mkfit::val_in_layer_comb(on); }
void val_layer_policy(bool w, bool h, bool s) { mkfit::val_layer_policy(w, h, s); }
void val_reserve_hole_slot(bool on) { mkfit::val_reserve_hole_slot(on); }
void val_score_mode(int m, float e) { mkfit::val_score_mode(m, e); }
void val_score(float b, float c, float mf, float mb) { mkfit::val_score(b, c, mf, mb); }
void val_search_lite(int mode) { mkfit::val_search_lite(mode); }
void val_clsize(const mkfit::Event *e) { mkfit::val_cluster_sizes(e); }
void val_mat_profile(float z = 0.0f) { mkfit::val_material_profile(z); }
void val_kgain_reset() { mkfit::val_kgain_reset(); }
void val_kgain_ev(const mkfit::Event *e) { mkfit::val_kgain_event(e); }
void val_kgain_report() { mkfit::val_kgain_report(); }
void val_kinfo_reset() { mkfit::val_kinfo_reset(); }
void val_kinfo_ev(const mkfit::Event *e) { mkfit::val_kinfo_event(e); }
void val_kinfo_report() { mkfit::val_kinfo_report(); }
void val_mat_scale(float f) { mkfit::val_mat_scale(f); }
void val_eloss_var(float f) { mkfit::val_eloss_var_scale(f); }
void val_mat_fwd(float f) { mkfit::val_mat_fwdpix(f); }
// 0 = pT5, 1 = T5, 2 = pix.  chop = strip a pT5's pixel hits before searching.
void val_seeds(int kind, bool chop) {
  mkfit::Shell::s_hlt_seed_kind = (mkfit::Shell::HltSeedKind_e) kind;
  mkfit::Shell::s_hlt_chop_pixels = chop;
  printf("val_seeds: kind=%s chop_pixels=%d\n",
         mkfit::Shell::hlt_seed_kind_name(mkfit::Shell::s_hlt_seed_kind), (int) chop);
}
void val_chop_ev(const mkfit::Event *e) { mkfit::val_chop_recovery_event(e); }
void val_chop_report(const char *t) { mkfit::val_chop_recovery_report(t); }
void val_te_reset() { mkfit::val_track_eff_reset(); }
void val_te_ev(const mkfit::Event *e) { mkfit::val_track_eff_event(e); }
void val_te_report(const char *t) { mkfit::val_track_eff_report(t); }
// Sim-seeded forward search: seeds built from the sim tracks, starting at
// layer 0, so the outward pass covers the pixel barrel.
void val_simseed(int n_seed_hits = 1, float pt_min = 0.5f) {
  extern mkfit::Shell *g_shell_ptr;   // not used; call via s.ProcessEventSimSeeded()
}
void val_sr_reset() { mkfit::val_search_reset(); }
void val_sr_ev(const mkfit::Event *e, int i) { mkfit::val_search_event(e, i); }
void val_sr_write(const char *f = "val-search.root") { mkfit::val_search_write(f); }
void val_bt_reset() { mkfit::val_bkfit_trace_reset(); }
void val_bt_ev(const mkfit::Event *e) { mkfit::val_bkfit_trace_event(e); }
void val_bt_report(const char *p = "200-bkfit-trace") { mkfit::val_bkfit_trace_report(p); }
void val_bk_scale(float s) { mkfit::val_bkfit_err_scale(s); }
void val_bk_mat(bool on) { mkfit::val_bkfit_material(on); }

void val_geom_reset() { mkfit::val_geom_check_reset(); }
void val_geom_ev(const mkfit::Event *ev) { mkfit::val_geom_check_event(ev); }
void val_geom_report(const char *name = "") { mkfit::val_geom_check_report(name); }

void val_gen_displaced(const char *out = "val-closure-displaced.root",
                       float d0 = 10.0f, float z0 = 30.0f) {
  mkfit::val_gen_closure(out, 14, 32, 8, 20260911u, d0, z0);
}

void val_sister_reset() { mkfit::val_sister_reset(); }
void val_sister_ev(const mkfit::Event *ev, float pt_min = 0.5) { mkfit::val_sister_event(ev, pt_min); }
void val_sister_report() { mkfit::val_sister_report(); }
void val_sister_mcfilter(bool on) { mkfit::val_sister_mcfilter(on); }
void val_qbins_reset() { mkfit::val_qbins_reset(); }
void val_qbins_ev(const mkfit::Event *ev) { mkfit::val_qbins_event(ev); }
void val_qbins_report() { mkfit::val_qbins_report(); }

void val_eff_reset() { mkfit::val_eff_reset(); }
void val_eff_ref(const char *c) { mkfit::val_eff_ref(c); }
void val_eff_ev(const mkfit::Event *e, const char *c) { mkfit::val_eff_event(e, c); }
void val_eff_report(const char *p) { mkfit::val_eff_report(p); }
void val_eff_cmssw_ev(const mkfit::Event *e, const char *c) { mkfit::val_eff_cmssw_event(e, c); }

void val_chopres_reset() { mkfit::val_chopres_reset(); }
void val_chopres_ref(const char *c) { mkfit::val_chopres_ref(c); }
void val_chopres_ev(const mkfit::Event *e, const char *c) { mkfit::val_chopres_event(e, c); }
void val_chopres_report(const char *p) { mkfit::val_chopres_report(p); }

void val_score_terms(bool ur, float rc, bool ud, float dc) { mkfit::val_score_terms(ur, rc, ud, dc); }
void val_score_term_stats(bool on) { mkfit::val_score_term_stats(on); }
