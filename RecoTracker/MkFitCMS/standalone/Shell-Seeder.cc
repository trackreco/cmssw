// The mkFit seeder's quads as seeds for the track finding: the quads come from a
// `seedsurf --seeds` file (MkFitCMS/standalone/seeding/), the seed state from the
// quad's hits. See Shell.h, "The mkFit seeder's quads as seeds".

#include "RecoTracker/MkFitCMS/standalone/Shell.h"

#include "RecoTracker/MkFitCMS/interface/MkStdSeqs.h"
#include "RecoTracker/MkFitCMS/standalone/MkStandaloneSeqs.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/src/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitCore/src/Matrix.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace mkfit {

  // The prior at hit 0. sigma(1/pT) is RELATIVE, a fraction of the helix's own 1/pT: CMSSW's seed
  // creator starts from 1/pT_min, which in float leaves the update with variance ratios of 1e8 at
  // pT ~ 10 GeV and negative variances. The hits wash this prior out; SeederSeedCheck is how to
  // check that, by scaling it.
  float Shell::s_seeder_prior_sigma[6] = {1.0f, 1.0f, 1.0f, 1.0f, 0.03f, 0.03f};
  float Shell::s_seeder_prior_scale = 1.0f;
  bool Shell::s_seeder_pos_from_hit0 = false;
  int Shell::s_seeder_debug = 0;
  // mode 0 only: round values of the mode 1 fit's median sigmas at the last hit, D121 PU200, pT > 0.9
  // (q/pT 0.02-0.05 1/GeV, phi 0.7-1.5 mrad, theta 0.1-1.2 mrad)
  float Shell::s_seeder_fake_sigma[6] = {0.002f, 0.002f, 0.002f, 0.03f, 0.0015f, 0.0008f};

  //===========================================================================
  // Reading the quads
  //===========================================================================

  int Shell::LoadSeederQuads(const char *file) {
    m_seeder_quads.clear();
    FILE *f = fopen(file, "r");
    if (!f) {
      fprintf(stderr, "Shell::LoadSeederQuads: cannot open %s\n", file);
      return -1;
    }
    char line[512];
    int n = 0;
    while (fgets(line, sizeof(line), f)) {
      if (line[0] == '#')
        continue;
      int ev;
      SeederQuad q;
      // 10 columns, or 12 with the fake score and the ambiguity count (files from before that keep the defaults)
      const int nf = sscanf(line, "%d %d %d %d %d %d %d %d %d %f %f %d", &ev, &q.l[0], &q.l[1], &q.l[2], &q.l[3],
                            &q.h[0], &q.h[1], &q.h[2], &q.h[3], &q.score, &q.fake_score, &q.n_amb);
      if (nf != 10 && nf != 12)
        continue;
      if (nf == 10)
        q.fake_score = -1, q.n_amb = 0;
      // the file counts events from 0, GoToEvent() from 1
      m_seeder_quads[ev + 1].push_back(q);
      ++n;
    }
    fclose(f);
    printf("Shell::LoadSeederQuads: %d quads in %d events from %s\n", n, (int)m_seeder_quads.size(), file);
    return n;
  }

  //===========================================================================
  // The seed state
  //===========================================================================

  namespace {
    // The circle through three points in the transverse plane and the line in (arc length, z) through
    // all four hits. k: signed curvature, > 0 counter-clockwise from a to d.
    struct SeedHelix {
      float cx = 0, cy = 0, R = 0;  // centre and radius; R = 0 for a straight line
      int turn = 0;                 // +1 counter-clockwise, -1 clockwise, 0 straight
      float cot = 0;

      bool make(const Hit *H[4]) {
        const float ax = H[0]->x(), ay = H[0]->y(), dx = H[3]->x(), dy = H[3]->y();
        // the middle point: whichever of hits 1, 2 lies nearer the middle of a-d in r
        const float rm = 0.5f * (H[0]->r() + H[3]->r());
        const Hit &M = std::abs(H[1]->r() - rm) < std::abs(H[2]->r() - rm) ? *H[1] : *H[2];
        const float bx = M.x() - ax, by = M.y() - ay, ex = dx - ax, ey = dy - ay;
        const float dd = 2.f * (bx * ey - by * ex);
        const float b2 = bx * bx + by * by, e2 = ex * ex + ey * ey;
        if (std::abs(dd) < 1e-9f * e2)
          return false;
        const float ux = (ey * b2 - by * e2) / dd, uy = (bx * e2 - ex * b2) / dd;
        cx = ax + ux, cy = ay + uy, R = std::hypot(ux, uy);
        turn = dd > 0 ? 1 : -1;
        // z = z0 + cot * s, least squares over the four hits, s the arc length from hit 0
        float S = 0, Z = 0, SS = 0, SZ = 0;
        for (int k = 0; k < 4; ++k) {
          const float chord = std::hypot(H[k]->x() - ax, H[k]->y() - ay);
          const float s = 2.f * R * std::asin(std::min(1.f, chord / (2.f * R)));
          S += s, Z += H[k]->z(), SS += s * s, SZ += s * H[k]->z();
        }
        const float den = 4.f * SS - S * S;
        if (den <= 0)
          return false;
        cot = (4.f * SZ - S * Z) / den;
        return std::isfinite(cot) && std::isfinite(R);
      }

      // the state at hit h, moving away from hit 0. B along +z turns a positive charge clockwise.
      void state(const Hit &h, TrackState &ts) const {
        const float rx = h.x() - cx, ry = h.y() - cy;
        const float tx = turn > 0 ? -ry : ry, ty = turn > 0 ? rx : -rx;
        ts.charge = turn > 0 ? -1 : 1;
        const float pt = Const::sol_over_100 * Config::Bfield * R;
        ts.parameters = SVector6(h.x(), h.y(), h.z(), 1.f / pt, std::atan2(ty, tx), Const::PIOver2 - std::atan(cot));
      }
    };

    void diag_errors(SMatrixSym66 &e, const float *sig, float scale) {
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b <= a; ++b)
          e(a, b) = 0.f;
      for (int a = 0; a < 6; ++a)
        e(a, a) = (scale * sig[a]) * (scale * sig[a]);
    }
  }  // namespace

  int Shell::MakeSeederSeeds(EvCtx &ctx, int mode) {
    ctx.seeds.clear();
    m_seeder_seed_info.clear();
    const Event &ev = *ctx.ev;
    auto it = m_seeder_quads.find(ev.evtID());
    if (it == m_seeder_quads.end()) {
      printf("Shell::MakeSeederSeeds: no quads for event %d\n", ev.evtID());
      return 0;
    }
    const std::vector<SeederQuad> &Q = it->second;
    // the seed-quality field, by quad index: a seed's label is its quad's index (below)
    m_seeder_seed_info.resize(Q.size());
    for (size_t i = 0; i < Q.size(); ++i)
      m_seeder_seed_info[i] = {Q[i].score, Q[i].fake_score, Q[i].n_amb};
    const IterationConfig &itconf = Config::ItrInfo[m_it_index];
    const TrackerInfo &ti = Config::TrkInfo;
    const PropagationFlags &pf = ti.prop_config().backward_fit_pflags;

    int n_bad_helix = 0, n_fail = 0, n_neg_pos = 0;
    TrackVec &out = ctx.seeds;
    out.reserve(Q.size());

    for (int b0 = 0; b0 < (int)Q.size(); b0 += NN) {
      const int N = std::min(NN, (int)Q.size() - b0);
      TrackState ts[NN];
      bool ok[NN];
      for (int i = 0; i < N; ++i) {
        const SeederQuad &q = Q[b0 + i];
        const Hit *H[4];
        for (int k = 0; k < 4; ++k)
          H[k] = &ev.layerHits_[q.l[k]][q.h[k]];
        SeedHelix hx;
        ok[i] = hx.make(H);
        if (!ok[i]) {
          ++n_bad_helix;
          continue;
        }
        if (mode == 0) {
          hx.state(*H[3], ts[i]);
          diag_errors(ts[i].errors, s_seeder_fake_sigma, 1.f);
        } else {
          hx.state(*H[0], ts[i]);
          diag_errors(ts[i].errors, s_seeder_prior_sigma, s_seeder_prior_scale);
          ts[i].errors(3, 3) *= ts[i].parameters[3] * ts[i].parameters[3];
          if (s_seeder_pos_from_hit0) {
            // the position is hit 0's: its own covariance, and no update with it below
            const SMatrixSym33 &he = H[0]->error();
            for (int a = 0; a < 3; ++a)
              for (int b = 0; b <= a; ++b)
                ts[i].errors(a, b) = he(a, b);
          }
        }
      }
      float chi2[NN] = {0};
      if (mode == 1) {
        MPlexLS err_in, err_out;
        MPlexLV par_in, par_out;
        MPlexQI chg, fail;
        MPlexQF chi2_k;
        MPlexHS msErr;
        MPlexHV msPar, plNrm, plDir, plPnt;
        // lanes beyond N and lanes without a helix repeat the first good lane, so that every lane
        // computes on sane numbers
        int i_good = 0;
        while (i_good < N && !ok[i_good])
          ++i_good;
        if (i_good == N)
          continue;
        for (int i = 0; i < NN; ++i) {
          const int j = (i < N && ok[i]) ? i : i_good;
          err_in.copyIn(i, ts[j].errors.Array());
          par_in.copyIn(i, ts[j].parameters.Array());
          chg(i, 0, 0) = ts[j].charge;
        }
        int failed[NN] = {0};
        for (int k = s_seeder_pos_from_hit0 ? 1 : 0; k < 4; ++k) {
          for (int i = 0; i < NN; ++i) {
            const int j = (i < N && ok[i]) ? i : i_good;
            const SeederQuad &q = Q[b0 + j];
            const LayerInfo &li = ti[q.l[k]];
            const Hit &h = ev.layerHits_[q.l[k]][q.h[k]];
            const ModuleInfo &mi = li.module_info(h.detIDinLayer());
            msErr.copyIn(i, h.errArray());
            msPar.copyIn(i, h.posArray());
            plNrm.copyIn(i, mi.zdir.Array());
            plDir.copyIn(i, mi.xdir.Array());
            plPnt.copyIn(i, mi.pos.Array());
          }
          fail.setVal(0);
          kalmanPropagateAndUpdateAndChi2Plane(err_in, par_in, chg, msErr, msPar, plNrm, plDir, plPnt, err_out, par_out,
                                               fail, chi2_k, NN, pf, true);
          for (int i = 0; i < N; ++i) {
            bool bad = fail(i, 0, 0) != 0;
            int why = fail(i, 0, 0) != 0 ? 1 : 0;
            for (int d = 0; d < 6; ++d) {
              if (!isFinite(par_out(i, d, 0)))
                why |= 2;
              if (!isFinite(err_out(i, d, d)))
                why |= 4;
              else if (d >= 3 ? !(err_out(i, d, d) > 0.f) : err_out(i, d, d) < -1e-6f)
                why |= 8;
            }
            // the state lies on the module plane, so the position covariance has a zero direction along
            // its normal; float rounding leaves that diagonal at about -1e-9 cm^2, which is not a failure
            for (int d = 0; d < 3; ++d)
              n_neg_pos += k == 3 && err_out(i, d, d) <= 0.f && err_out(i, d, d) >= -1e-6f;
            bad = why != 0;
            if (bad && !failed[i] && s_seeder_debug > 0) {
              --s_seeder_debug;
              const SeederQuad &q = Q[b0 + i];
              printf("[seedfit] fail at k %d why %d layers %d %d %d %d  in: pt %.3f phi %.3f th %.3f | err diag",
                     k, why, q.l[0], q.l[1], q.l[2], q.l[3], 1.f / par_in(i, 3, 0), par_in(i, 4, 0), par_in(i, 5, 0));
              for (int d = 0; d < 6; ++d)
                printf(" %.3g", err_in(i, d, d));
              printf(" | out diag");
              for (int d = 0; d < 6; ++d)
                printf(" %.3g", err_out(i, d, d));
              printf("\n");
            }
            failed[i] |= bad;
            chi2[i] += chi2_k(i, 0, 0);
          }
          err_in = err_out;
          par_in = par_out;
        }
        for (int i = 0; i < N; ++i) {
          if (!ok[i])
            continue;
          if (failed[i]) {
            ok[i] = false;
            ++n_fail;
            continue;
          }
          err_in.copyOut(i, ts[i].errors.Array());
          par_in.copyOut(i, ts[i].parameters.Array());
          ts[i].charge = chg(i, 0, 0);
        }
      }
      for (int i = 0; i < N; ++i) {
        if (!ok[i])
          continue;
        const SeederQuad &q = Q[b0 + i];
        Track seed(ts[i], chi2[i], b0 + i, 0, nullptr);
        for (int k = 0; k < 4; ++k)
          seed.addHitIdx(q.h[k], q.l[k], 0.f);
        seed.setAlgorithm(TrackBase::TrackAlgorithm(itconf.m_track_algorithm));
        out.push_back(seed);
      }
    }
    printf("Shell::MakeSeederSeeds: event %d, mode %d: %d seeds from %d quads (%d without a helix, %d failed fits;"
           " %d position variances in [-1e-6, 0] cm^2 at the last hit)\n",
           ev.evtID(), mode, (int)out.size(), (int)Q.size(), n_bad_helix, n_fail, n_neg_pos);
    return (int)out.size();
  }

  void Shell::ProcessEventSeeder(EvCtx &ctx, int mode) {
    printf("\n##### BEG Event %d ##### forward search on the seeder's quads\n\n", ctx.ev->evtID());
    ctx.ev->filterOutMislabeledHitsInSimTracks();
    tm_mark();
    MakeSeederSeeds(ctx, mode);
    tm_add(TM_SeedFit);
    ProcessEvent(ctx, SS_PreSet);
    ctx.ev->candidateTracks_ = ctx.tracks;
    {
      StdSeq::Quality qval;
      qval.quality_val(ctx.ev);
    }
    printf("\n##### END Event %d ##### forward search on the seeder's quads\n\n", ctx.ev->evtID());
  }

  //===========================================================================
  // Seed states against truth
  //===========================================================================

  namespace {
    // [source 0 ours, 1 file][parameter 0 q/pT, 1 phi, 2 theta][|eta| bin][pT bin]
    constexpr int kNEta = 4, kNPt = 3;
    const char *const kEtaName[kNEta] = {"0-0.8", "0.8-1.6", "1.6-2.4", "2.4-4"};
    const char *const kPtName[kNPt] = {"0.9-2", "2-5", ">5"};
    const char *const kParName[3] = {"q/pT", "phi", "theta"};
    std::vector<float> g_pull[2][3][kNEta][kNPt];
    std::vector<float> g_sig[2][3][kNEta][kNPt];
    std::vector<float> g_ratio[3][kNEta][kNPt];
    long g_n_seed[2], g_n_true[2], g_n_binned[2], g_n_q_ok[2], g_n_matched;

    int eta_bin(float ae) { return ae < 0.8f ? 0 : ae < 1.6f ? 1 : ae < 2.4f ? 2 : ae < 4.f ? 3 : -1; }
    int pt_bin(float pt) { return pt < 0.9f ? -1 : pt < 2.f ? 0 : pt < 5.f ? 1 : 2; }

    float wrap_pi(float d) {
      while (d > Const::PI)
        d -= Const::TwoPI;
      while (d < -Const::PI)
        d += Const::TwoPI;
      return d;
    }

    float median(std::vector<float> v) {
      if (v.empty())
        return 0;
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    }
  }  // namespace

  void Shell::SeederSeedCheckReset() {
    for (auto &a : g_pull)
      for (auto &b : a)
        for (auto &c : b)
          for (auto &d : c)
            d.clear();
    for (auto &a : g_sig)
      for (auto &b : a)
        for (auto &c : b)
          for (auto &d : c)
            d.clear();
    for (auto &a : g_ratio)
      for (auto &b : a)
        for (auto &c : b)
          c.clear();
    g_n_seed[0] = g_n_seed[1] = g_n_true[0] = g_n_true[1] = g_n_binned[0] = g_n_binned[1] = 0;
    g_n_q_ok[0] = g_n_q_ok[1] = g_n_matched = 0;
  }

  void Shell::SeederSeedCheck(EvCtx &ctx) {
    const Event &ev = *ctx.ev;
    if (ev.simHitStates_.empty()) {
      printf("Shell::SeederSeedCheck: the sample has no SimHitStates\n");
      return;
    }
    const int algo = Config::ItrInfo[m_it_index].m_track_algorithm;

    // the sim track of a seed, if all its hits are on one; the state of its last hit
    auto truth = [&](const Track &s, int &lab, const SimHitState *&shs) -> bool {
      lab = -1;
      shs = nullptr;
      for (int k = 0; k < s.nTotalHits(); ++k) {
        const HitOnTrack hot = s.getHitOnTrack(k);
        if (hot.index < 0)
          return false;
        const int mcid = ev.layerHits_[hot.layer][hot.index].mcHitID();
        if (mcid < 0)
          return false;
        const int l = ev.simHitsInfo_[mcid].mcTrackID();
        if (l < 0 || (lab >= 0 && l != lab))
          return false;
        lab = l;
        if (k == s.nTotalHits() - 1 && mcid < (int)ev.simHitStates_.size() && ev.simHitStates_[mcid].is_valid())
          shs = &ev.simHitStates_[mcid];
      }
      return lab >= 0 && shs != nullptr;
    };

    auto book = [&](int src, const Track &s) {
      ++g_n_seed[src];
      int lab;
      const SimHitState *shs;
      if (!truth(s, lab, shs))
        return;
      ++g_n_true[src];
      const Track &st = ev.simTracks_[lab];
      const float pt = shs->pT();
      const float th = std::atan2(pt, shs->pz());
      const int be = eta_bin(std::abs(getEta(th))), bp = pt_bin(pt);
      if (be < 0 || bp < 0)
        return;
      ++g_n_binned[src];
      g_n_q_ok[src] += s.charge() == st.charge();
      const float tv[3] = {st.charge() / pt, std::atan2(shs->py(), shs->px()), th};
      const float sv[3] = {s.charge() * s.invpT(), s.momPhi(), s.theta()};
      const float se[3] = {std::sqrt(s.errors().At(3, 3)), std::sqrt(s.errors().At(4, 4)), std::sqrt(s.errors().At(5, 5))};
      for (int p = 0; p < 3; ++p) {
        const float d = p == 1 ? wrap_pi(sv[p] - tv[p]) : sv[p] - tv[p];
        g_pull[src][p][be][bp].push_back(d / se[p]);
        g_sig[src][p][be][bp].push_back(se[p]);
      }
    };

    // the file's seeds of this iteration, by their sorted hits
    std::map<std::vector<std::pair<int, int>>, const Track *> file_by_hits;
    int n_file_neg[6] = {0};
    for (const Track &s : ev.seedTracks_) {
      if (s.algoint() != algo)
        continue;
      for (int d = 0; d < 6; ++d)
        n_file_neg[d] += !(s.errors().At(d, d) > 0.f);
      book(1, s);
      std::vector<std::pair<int, int>> key;
      for (int k = 0; k < s.nTotalHits(); ++k) {
        const HitOnTrack hot = s.getHitOnTrack(k);
        key.push_back({hot.layer, hot.index});
      }
      std::sort(key.begin(), key.end());
      file_by_hits[key] = &s;
    }
    printf("Shell::SeederSeedCheck: file seeds with a diagonal <= 0, per parameter: %d %d %d %d %d %d\n", n_file_neg[0],
           n_file_neg[1], n_file_neg[2], n_file_neg[3], n_file_neg[4], n_file_neg[5]);
    for (const Track &s : ctx.seeds) {
      book(0, s);
      std::vector<std::pair<int, int>> key;
      for (int k = 0; k < s.nTotalHits(); ++k) {
        const HitOnTrack hot = s.getHitOnTrack(k);
        key.push_back({hot.layer, hot.index});
      }
      std::sort(key.begin(), key.end());
      auto f = file_by_hits.find(key);
      if (f == file_by_hits.end())
        continue;
      ++g_n_matched;
      const Track &fs = *f->second;
      const int be = eta_bin(std::abs(s.momEta())), bp = pt_bin(s.pT());
      if (be < 0 || bp < 0)
        continue;
      for (int p = 0; p < 3; ++p) {
        const int i = 3 + p;
        g_ratio[p][be][bp].push_back(std::sqrt(s.errors().At(i, i) / fs.errors().At(i, i)));
      }
    }
  }

  void Shell::SeederSeedCheckReport() {
    printf("\nShell::SeederSeedCheckReport\n");
    for (int src = 0; src < 2; ++src)
      printf("  %s: %ld seeds, %ld with all hits on one sim track and its last hit's state, %ld of them in the bins"
             " below (sim pT > 0.9, |eta| < 4); charge right among those %.4f\n",
             src == 0 ? "seeder" : "file  ", g_n_seed[src], g_n_true[src], g_n_binned[src],
             g_n_binned[src] ? double(g_n_q_ok[src]) / g_n_binned[src] : 0.);
    printf("  seeder seeds with a file seed of the same four hits: %ld\n", g_n_matched);
    printf("\n  pulls (residual to the last hit's sim state over the seed's sigma): median, robust sigma (1.4826 MAD),"
           " median sigma; seeder | file\n");
    for (int p = 0; p < 3; ++p) {
      printf("  %s\n  %-8s %-6s %7s | %8s %8s %10s | %8s %8s %10s | %7s\n", kParName[p], "|eta|", "pT", "n_seed",
             "med", "rsig", "sigma", "med", "rsig", "sigma", "n_file");
      for (int be = 0; be < kNEta; ++be)
        for (int bp = 0; bp < kNPt; ++bp) {
          float med[2], rsig[2], msig[2];
          for (int src = 0; src < 2; ++src) {
            const auto &v = g_pull[src][p][be][bp];
            med[src] = median(v);
            std::vector<float> ad(v.size());
            for (size_t i = 0; i < v.size(); ++i)
              ad[i] = std::abs(v[i] - med[src]);
            rsig[src] = 1.4826f * median(ad);
            msig[src] = median(g_sig[src][p][be][bp]);
          }
          printf("  %-8s %-6s %7zu | %8.3f %8.3f %10.3g | %8.3f %8.3f %10.3g | %7zu\n", kEtaName[be], kPtName[bp],
                 g_pull[0][p][be][bp].size(), med[0], rsig[0], msig[0], med[1], rsig[1], msig[1],
                 g_pull[1][p][be][bp].size());
        }
    }
    printf("\n  sigma, seeder over file, same four hits: median ratio (n)\n  %-8s %-6s", "|eta|", "pT");
    for (int p = 0; p < 3; ++p)
      printf(" %16s", kParName[p]);
    printf("\n");
    for (int be = 0; be < kNEta; ++be)
      for (int bp = 0; bp < kNPt; ++bp) {
        printf("  %-8s %-6s", kEtaName[be], kPtName[bp]);
        for (int p = 0; p < 3; ++p)
          printf("   %6.3f (%6zu)", median(g_ratio[p][be][bp]), g_ratio[p][be][bp].size());
        printf("\n");
      }
  }

}  // namespace mkfit

//=============================================================================
// Where the fakes come from; the sim tracks no seed is on
//=============================================================================

#include "RecoTracker/MkFitCore/standalone/TrackExtra.h"

namespace mkfit {

  namespace {
    constexpr int kNReg = 3;
    const char *const kRegName[kNReg] = {"barrel |eta|<0.9", "transition 0.9-1.7", "endcap >1.7"};
    const char *const kSeedClass[4] = {"true", "undecidable", "fake", "no seed found"};
    int reg_of(float ae) { return ae < 0.9f ? 0 : ae < 1.7f ? 1 : 2; }

    // [row][acceptance 0 all, 1 pT > 0.9 and |eta| < 2.5][region][seed class]
    long g_fo_trk[2][2][kNReg][4], g_fo_fake[2][2][kNReg][4];
    // fakes from a true seed whose sim track another track of the event found
    long g_fo_fake_true_found[2][2][kNReg];

    // [region]: selected sim tracks; not on a seeder seed; of those, found by production initialStep or
    // highPtTripletStep. Per distinct pixel layers 0..4 (4 = 4 or more), for both populations.
    long g_ms_sel[kNReg], g_ms_unseeded[kNReg], g_ms_unseeded_prod[kNReg];
    long g_ms_npix[2][kNReg][5];
    // pixel layers == 3: [population][region][has OT1-P (4)][has OT2-P (6)]
    long g_ms_ot[2][kNReg][2][2];
    // pixel layers == 3, barrel: which of layers 0-3 is missing, [population][layer]
    long g_ms_missing[2][4];

    // seed class from its hits: 0 true, 1 undecidable, 2 fake
    int seed_class(const Event &ev, const Track &s) {
      int lab = -1, n_unl = 0;
      bool two = false;
      for (int k = 0; k < s.nTotalHits(); ++k) {
        const HitOnTrack hot = s.getHitOnTrack(k);
        if (hot.index < 0)
          continue;
        const int mcid = ev.layerHits_[hot.layer][hot.index].mcHitID();
        const int l = mcid >= 0 ? ev.simHitsInfo_[mcid].mcTrackID() : -1;
        if (l < 0) {
          ++n_unl;
          continue;
        }
        if (lab >= 0 && l != lab)
          two = true;
        lab = l;
      }
      return two || lab < 0 ? 2 : n_unl > 0 ? 1 : 0;
    }
  }  // namespace

  void Shell::SeederDiagReset() {
    std::fill(&g_fo_trk[0][0][0][0], &g_fo_trk[0][0][0][0] + 2 * 2 * kNReg * 4, 0L);
    std::fill(&g_fo_fake[0][0][0][0], &g_fo_fake[0][0][0][0] + 2 * 2 * kNReg * 4, 0L);
    std::fill(&g_fo_fake_true_found[0][0][0], &g_fo_fake_true_found[0][0][0] + 2 * 2 * kNReg, 0L);
    std::fill(g_ms_sel, g_ms_sel + kNReg, 0L);
    std::fill(g_ms_unseeded, g_ms_unseeded + kNReg, 0L);
    std::fill(g_ms_unseeded_prod, g_ms_unseeded_prod + kNReg, 0L);
    std::fill(&g_ms_npix[0][0][0], &g_ms_npix[0][0][0] + 2 * kNReg * 5, 0L);
    std::fill(&g_ms_ot[0][0][0][0], &g_ms_ot[0][0][0][0] + 2 * kNReg * 4, 0L);
    std::fill(&g_ms_missing[0][0], &g_ms_missing[0][0] + 8, 0L);
  }

  void Shell::SeederFakeOrigin(EvCtx &ctx, int row) {
    const Event &ev = *ctx.ev;
    const TrackVec &seeds = ctx.seeds;
    const TrackVec &tracks = ev.candidateTracks_;
    std::map<std::pair<int, int>, std::vector<int>> hit2seed;
    for (int i = 0; i < (int)seeds.size(); ++i)
      for (int k = 0; k < seeds[i].nTotalHits(); ++k) {
        const HitOnTrack hot = seeds[i].getHitOnTrack(k);
        if (hot.index >= 0)
          hit2seed[{hot.layer, hot.index}].push_back(i);
      }
    struct Row {
      int reg, cls, mc, seed_lab;
      bool acc;
    };
    std::vector<Row> rows;
    std::set<int> found;
    for (const Track &c : tracks) {
      std::set<std::pair<int, int>> th;
      for (int k = 0; k < c.nTotalHits(); ++k) {
        const HitOnTrack hot = c.getHitOnTrack(k);
        if (hot.index >= 0)
          th.insert({hot.layer, hot.index});
      }
      // the seed: one whose every hit is on the track
      int si = -1;
      for (const auto &h : th) {
        auto it = hit2seed.find(h);
        if (it == hit2seed.end())
          continue;
        for (int i : it->second) {
          bool all = true;
          for (int k = 0; k < seeds[i].nTotalHits() && all; ++k) {
            const HitOnTrack hot = seeds[i].getHitOnTrack(k);
            all = hot.index < 0 || th.count({hot.layer, hot.index});
          }
          if (all) {
            si = i;
            break;
          }
        }
        if (si >= 0)
          break;
      }
      // the association, as val_eff makes it: over the non-seed hits
      TrackExtra extra(c.label());
      if (si >= 0)
        extra.findMatchingSeedHits(c, seeds[si], ev.layerHits_);
      extra.setMCTrackIDInfo(c, ev.layerHits_, ev.simHitsInfo_, ev.simTracks_, false, false);
      const int mc = extra.mcTrackID();
      const bool fake = mc < 0 || mc >= (int)ev.simTracks_.size();
      if (!fake)
        found.insert(mc);
      const int cls = si >= 0 ? seed_class(ev, seeds[si]) : 3;
      int seed_lab = -1;
      if (si >= 0 && cls == 0) {
        const HitOnTrack hot = seeds[si].getHitOnTrack(0);
        seed_lab = ev.simHitsInfo_[ev.layerHits_[hot.layer][hot.index].mcHitID()].mcTrackID();
      }
      const float ae = std::abs(c.momEta());
      rows.push_back({reg_of(ae), cls, fake ? -1 : mc, seed_lab, c.pT() > 0.9f && ae < 2.5f});
    }
    for (const Row &r : rows)
      for (int a = 0; a < 2; ++a) {
        if (a == 1 && !r.acc)
          continue;
        ++g_fo_trk[row][a][r.reg][r.cls];
        if (r.mc < 0) {
          ++g_fo_fake[row][a][r.reg][r.cls];
          if (r.cls == 0 && found.count(r.seed_lab))
            ++g_fo_fake_true_found[row][a][r.reg];
        }
      }
  }

  void Shell::SeederMissStudy(EvCtx &ctx) {
    const Event &ev = *ctx.ev;
    std::set<int> seeded;
    for (const Track &s : ctx.seeds) {
      const auto si = ev.simInfoForTrack(s);
      if (si.label >= 0)
        seeded.insert(si.label);
    }
    std::set<int> prod_found;
    for (const Track &t : ev.cmsswTracks_) {
      if (t.algorithm() != TrackBase::TrackAlgorithm::initialStep &&
          t.algorithm() != TrackBase::TrackAlgorithm::highPtTripletStep)
        continue;
      const auto si = ev.simInfoForTrack(t);
      if (si.label >= 0 && si.good_frac() > 0.75f)
        prod_found.insert(si.label);
    }
    for (int L = 0; L < (int)ev.simTracks_.size(); ++L) {
      const Track &st = ev.simTracks_[L];
      const float ae = std::abs(st.momEta()), pt = st.pT();
      // val_eff's MTV selection
      if (!st.isFindable() || std::hypot(st.x(), st.y()) > 3.5f || std::abs(st.z()) > 30.0f)
        continue;
      if (ae >= 2.5f || pt <= 0.9f || st.nUniqueLayers() < 4)
        continue;
      const int reg = reg_of(ae);
      ++g_ms_sel[reg];
      if (seeded.count(L))
        continue;
      ++g_ms_unseeded[reg];
      const bool pf = prod_found.count(L);
      g_ms_unseeded_prod[reg] += pf;
      std::set<int> pix;
      bool ot1p = false, ot2p = false;
      for (int i = 0; i < st.nTotalHits(); ++i) {
        const HitOnTrack hot = st.getHitOnTrack(i);
        if (hot.index < 0 || hot.layer < 0)
          continue;
        if (Config::TrkInfo[hot.layer].is_pixel())
          pix.insert(hot.layer);
        ot1p |= hot.layer == 4;
        ot2p |= hot.layer == 6;
      }
      const int np = std::min(4, (int)pix.size());
      for (int pop = 0; pop < 2; ++pop) {
        if (pop == 1 && !pf)
          continue;
        ++g_ms_npix[pop][reg][np];
        if (np == 3) {
          ++g_ms_ot[pop][reg][ot1p][ot2p];
          if (reg == 0)
            for (int l = 0; l < 4; ++l)
              g_ms_missing[pop][l] += !pix.count(l);
        }
      }
    }
  }

  void Shell::SeederDiagReport() {
    printf("\nShell::SeederDiagReport\n");
    printf("\n  found tracks and fakes (val_eff's association: no sim track by the non-seed hits), by the truth of"
           " their seed\n");
    for (int row = 0; row < 2; ++row)
      for (int a = 0; a < 2; ++a) {
        printf("  %s, %s\n  %-20s", row == 0 ? "file seeds" : "seeder seeds",
               a == 0 ? "all tracks" : "tracks with pT > 0.9, |eta| < 2.5", "region");
        for (int c = 0; c < 4; ++c)
          printf(" | %-24s", kSeedClass[c]);
        printf(" | fakes from a true seed whose sim track another track found\n");
        for (int r = 0; r < kNReg; ++r) {
          printf("  %-20s", kRegName[r]);
          for (int c = 0; c < 4; ++c)
            printf(" | %7ld trk %6ld fake  ", g_fo_trk[row][a][r][c], g_fo_fake[row][a][r][c]);
          printf(" | %ld\n", g_fo_fake_true_found[row][a][r]);
        }
      }
    printf("\n  selected sim tracks (MTV selection) no seeder seed is on, after the seed cleaning\n");
    printf("  %-20s %8s %9s %22s\n", "region", "selected", "unseeded", "of them prod found");
    for (int r = 0; r < kNReg; ++r)
      printf("  %-20s %8ld %9ld %22ld\n", kRegName[r], g_ms_sel[r], g_ms_unseeded[r], g_ms_unseeded_prod[r]);
    for (int pop = 0; pop < 2; ++pop) {
      printf("\n  %s: by distinct pixel layers with a sim hit\n  %-20s %7s %7s %7s %7s %7s | 3 pixel layers:"
             " OT1-P & OT2-P, OT1-P only, OT2-P only, neither\n",
             pop == 0 ? "unseeded" : "unseeded, found by prod initialStep or highPtTripletStep", "region", "0", "1",
             "2", "3", ">=4");
      for (int r = 0; r < kNReg; ++r)
        printf("  %-20s %7ld %7ld %7ld %7ld %7ld | %7ld %7ld %7ld %7ld\n", kRegName[r], g_ms_npix[pop][r][0],
               g_ms_npix[pop][r][1], g_ms_npix[pop][r][2], g_ms_npix[pop][r][3], g_ms_npix[pop][r][4],
               g_ms_ot[pop][r][1][1], g_ms_ot[pop][r][1][0], g_ms_ot[pop][r][0][1], g_ms_ot[pop][r][0][0]);
      printf("  barrel, 3 pixel layers, the missing one: L0 %ld, L1 %ld, L2 %ld, L3 %ld\n", g_ms_missing[pop][0],
             g_ms_missing[pop][1], g_ms_missing[pop][2], g_ms_missing[pop][3]);
    }
  }

  //===========================================================================
  // Displaced tracks
  //===========================================================================

  namespace {
    constexpr int kNDisp = 9;
    const float kDispEdge[kNDisp + 1] = {0.f, 0.01f, 0.02f, 0.05f, 0.1f, 0.2f, 0.5f, 1.f, 2.f, 1e9f};
    // production's iterations, in the order prod_iters.C adds them
    constexpr int kNIter = 6;
    const TrackBase::TrackAlgorithm kIterAlgo[kNIter] = {
        TrackBase::TrackAlgorithm::initialStep,      TrackBase::TrackAlgorithm::highPtTripletStep,
        TrackBase::TrackAlgorithm::lowPtQuadStep,    TrackBase::TrackAlgorithm::lowPtTripletStep,
        TrackBase::TrackAlgorithm::detachedQuadStep, TrackBase::TrackAlgorithm::pixelPairStep};
    const char *const kIterName[kNIter] = {"initial", "+highPtTr", "+lowPtQuad", "+lowPtTr", "+detQuad", "+pixPair"};
    // [region 0-2, 3 = all][d0 bin]
    long g_dp_sel[kNReg + 1][kNDisp], g_dp_seeded[kNReg + 1][kNDisp], g_dp_found[kNReg + 1][kNDisp];
    long g_dp_prod[kNReg + 1][kNDisp][kNIter];
    int g_dp_nev = 0;
  }  // namespace

  void Shell::SeederDisplacedReset() {
    memset(g_dp_sel, 0, sizeof(g_dp_sel));
    memset(g_dp_seeded, 0, sizeof(g_dp_seeded));
    memset(g_dp_found, 0, sizeof(g_dp_found));
    memset(g_dp_prod, 0, sizeof(g_dp_prod));
    g_dp_nev = 0;
  }

  void Shell::SeederDisplacedStudy(EvCtx &ctx) {
    const Event &ev = *ctx.ev;
    const BeamSpot &bs = ev.beamSpot_;
    std::set<int> seeded, found;
    for (const Track &s : ctx.seeds) {
      const auto si = ev.simInfoForTrack(s);
      if (si.label >= 0)
        seeded.insert(si.label);
    }
    // The association of val_eff (ValProp-Eff.cc ve_accumulate): the track's seed is the current seed sharing
    // the most hits with it, and TrackExtra::setMCTrackIDInfo over the non-seed hits gives the sim track.
    std::map<std::pair<int, int>, std::vector<int>> hit2seed;
    for (int i = 0; i < (int)ctx.seeds.size(); ++i)
      for (int h = 0; h < ctx.seeds[i].nTotalHits(); ++h) {
        const HitOnTrack hot = ctx.seeds[i].getHitOnTrack(h);
        if (hot.index >= 0 && hot.layer >= 0)
          hit2seed[{hot.layer, hot.index}].push_back(i);
      }
    auto sim_of = [&](const Track &t) {
      std::map<int, int> shared;
      for (int h = 0; h < t.nTotalHits(); ++h) {
        const HitOnTrack hot = t.getHitOnTrack(h);
        if (hot.index < 0 || hot.layer < 0)
          continue;
        auto it = hit2seed.find({hot.layer, hot.index});
        if (it != hit2seed.end())
          for (int i : it->second)
            ++shared[i];
      }
      int best = -1, bn = 0;
      for (const auto &[i, n] : shared)
        if (n > bn) {
          bn = n;
          best = i;
        }
      TrackExtra extra(t.label());
      if (best >= 0)
        extra.findMatchingSeedHits(t, ctx.seeds[best], ev.layerHits_);
      extra.setMCTrackIDInfo(t, ev.layerHits_, ev.simHitsInfo_, ev.simTracks_, false, false);
      return extra.mcTrackID();
    };
    for (const Track &t : ev.candidateTracks_) {
      const int mc = sim_of(t);
      if (mc >= 0)
        found.insert(mc);
    }
    // the first iteration (index into kIterAlgo) that found each sim track
    std::map<int, int> prod_first;
    for (const Track &t : ev.cmsswTracks_) {
      int it = -1;
      for (int k = 0; k < kNIter; ++k)
        if (t.algorithm() == kIterAlgo[k])
          it = k;
      if (it < 0)
        continue;
      const int mc = sim_of(t);
      if (mc < 0)
        continue;
      auto f = prod_first.find(mc);
      if (f == prod_first.end() || it < f->second)
        prod_first[mc] = it;
    }
    for (int L = 0; L < (int)ev.simTracks_.size(); ++L) {
      const Track &st = ev.simTracks_[L];
      const float ae = std::abs(st.momEta()), pt = st.pT();
      // val_eff's MTV selection
      if (!st.isFindable() || std::hypot(st.x(), st.y()) > 3.5f || std::abs(st.z()) > 30.0f)
        continue;
      if (ae >= 2.5f || pt <= 0.9f || st.nUniqueLayers() < 4)
        continue;
      // |d0| to the beam spot, from the production vertex along the momentum's transverse direction (a straight
      // line: at pT > 0.9 GeV the curvature changes it by < 1e-3 relative for d0 < 3.5 cm)
      const float phi = st.momPhi();
      const float d0 = std::abs((st.x() - bs.x) * std::sin(phi) - (st.y() - bs.y) * std::cos(phi));
      int b = 0;
      while (b < kNDisp - 1 && d0 >= kDispEdge[b + 1])
        ++b;
      const int reg = reg_of(ae);
      for (int r : {reg, kNReg}) {
        ++g_dp_sel[r][b];
        g_dp_seeded[r][b] += seeded.count(L);
        g_dp_found[r][b] += found.count(L);
        auto f = prod_first.find(L);
        if (f != prod_first.end())
          for (int k = f->second; k < kNIter; ++k)
            ++g_dp_prod[r][b][k];
      }
    }
    ++g_dp_nev;
  }

  void Shell::SeederDisplacedReport() {
    printf("\nShell::SeederDisplacedReport: %d events; selected sim tracks (MTV) by |d0| to the beam spot [cm]\n",
           g_dp_nev);
    printf("  seeded: a seeder seed is on it (after the seed cleaning); found: val_eff's association (2 mccount >="
           " nCandHits over the non-seed hits); production cumulative by iteration\n");
    for (int r = kNReg; r >= 0; --r) {
      printf("\n  %s\n  %-12s %7s | %7s %7s |", r == kNReg ? "all regions" : kRegName[r], "|d0| [cm]", "sel",
             "seeded", "found");
      for (int k = 0; k < kNIter; ++k)
        printf(" %9s", kIterName[k]);
      printf("\n");
      for (int b = 0; b < kNDisp; ++b) {
        if (b < kNDisp - 1)
          printf("  %5.2f-%-5.2f %7ld |", kDispEdge[b], kDispEdge[b + 1], g_dp_sel[r][b]);
        else
          printf("  %5.2f-      %7ld |", kDispEdge[b], g_dp_sel[r][b]);
        const double n = std::max(1L, g_dp_sel[r][b]);
        printf(" %6.1f%% %6.1f%% |", 100.0 * g_dp_seeded[r][b] / n, 100.0 * g_dp_found[r][b] / n);
        for (int k = 0; k < kNIter; ++k)
          printf(" %8.1f%%", 100.0 * g_dp_prod[r][b][k] / n);
        printf("\n");
      }
    }
  }

  //===========================================================================
  // Quad anatomy
  //===========================================================================

  namespace {
    // quad classes
    enum QaCls { QA_True = 0, QA_Wrong, QA_Unl, QA_Worse, QA_N };
    const char *const kQaName[QA_N] = {"true", "3+1 wrong", "3+1 unlinked", "2+2 and worse"};
    // the hit-density window: hits in the same layer within these of the hit (q: z in the barrel, r in the discs)
    constexpr float kQaDphi = 0.005f, kQaDq = 1.0f;
    constexpr int kQaNDens = 7;
    const int kQaDensEdge[kQaNDens] = {0, 1, 2, 3, 5, 9, 17};  // bins [0], [1], [2], [3,4], [5,8], [9,16], [17,..)
    const char *const kQaDensName[kQaNDens] = {"0", "1", "2", "3-4", "5-8", "9-16", ">=17"};
    int qa_dens_bin(int n) {
      int b = 0;
      while (b < kQaNDens - 1 && n >= kQaDensEdge[b + 1])
        ++b;
      return b;
    }
    constexpr int kQaNDphi = 6;
    const float kQaDphiEdge[kQaNDphi] = {0.f, 0.0005f, 0.001f, 0.002f, 0.005f, 0.01f};
    const char *const kQaDphiName[kQaNDphi] = {"<0.5", "0.5-1", "1-2", "2-5", "5-10", ">10"};
    constexpr int kQaNDq = 5;
    const float kQaDqEdge[kQaNDq] = {0.f, 0.05f, 0.2f, 1.f, 5.f};
    const char *const kQaDqName[kQaNDq] = {"<0.05", "0.05-0.2", "0.2-1", "1-5", ">5"};
    template <int N>
    int qa_bin(const float (&edge)[N], float x) {
      int b = 0;
      while (b < N - 1 && x >= edge[b + 1])
        ++b;
      return b;
    }

    long g_qa_cls[kNReg][QA_N];
    long g_qa_wrong_pos[kNReg][4], g_qa_wrong_comp[kNReg][4];  // 3+1 wrong: position; the majority track has a hit there
    long g_qa_wrong_layer[64];
    long g_qa_dens[2][kNReg][kQaNDens];  // [0: last hit of true quads, 1: the wrong hit of 3+1 wrong][region][bin]
    long g_qa_dens_comp[kNReg][kQaNDens];  // the majority track's own hit, where it has one
    long g_qa_dist_phi[kNReg][kQaNDphi], g_qa_dist_q[kNReg][kQaNDq];
    // tracks by the class of their quad: [0 found, 1 fake][region][class, QA_N = no quad found]
    long g_qa_trk[2][2][kNReg][QA_N + 1];  // [acceptance 0 all, 1 pT > 0.9 and |eta| < 2.5]
    long g_qa_fake_wrong_pos[2][kNReg][4];
    long g_qa_fake_wrong_mfound[2][kNReg];
    // tracks from TRUE quads, by the class of the seed the search actually started from (after the iteration's
    // seed cleaning, which merges the hits of the seeds it drops): [acc][found/fake][region][0 true, 1 undec, 2 fake, 3 none]
    long g_qa_true_seedcls[2][2][kNReg][4];
    // the quad's score from the --seeds file (seedsurf's cleaning score), in log bins:
    // per quad class [region][class][bin], and per track of the quad it grew from [acc][found/fake][region][bin]
    constexpr int kQaNSc = 9;
    const float kQaScEdge[kQaNSc] = {0.f, 0.01f, 0.03f, 0.1f, 0.3f, 1.f, 3.f, 10.f, 30.f};
    const char *const kQaScName[kQaNSc] = {"<0.01", "0.01-0.03", "0.03-0.1", "0.1-0.3", "0.3-1", "1-3", "3-10", "10-30", ">30"};
    long g_qa_sc[kNReg][QA_N][kQaNSc];
    long g_qa_trk_sc[2][2][kNReg][kQaNSc];
    // tracks in acceptance, [0 all quads, 1 flagged: quad score in [0.3, 1)][0 found, 1 fake]: hits beyond the quad's
    // four (histogram, last bin 30+), and over the track's hits against its own majority particle: foreign hits
    // among the quad's hits and among the added ones, unlinked hits, and whether the majority is one of the quad's
    constexpr int kQaNBeyond = 31;
    long g_qa_bey[2][2][kQaNBeyond], g_qa_bey_n[2][2];
    long g_qa_for_seed[2][2], g_qa_for_add[2][2], g_qa_unl[2][2], g_qa_maj_in_quad[2][2];
    long g_qa_for_only_seed[2][2], g_qa_for_none[2][2];
    // K scan: tracks from flagged quads (score in [0.3, 1)) with fewer than K hits beyond [0 the quad's four, 1 the
    // seed the search started from] produce no track. [def][K - kQaK0][region]: fake tracks in acceptance removed,
    // and selected sim tracks (MTV selection) no longer found by any track
    constexpr int kQaK0 = 2, kQaNK = 6;  // K = 2 .. 7
    long g_qa_k_fake[2][kQaNK][kNReg + 1], g_qa_k_lost[2][kQaNK][kNReg + 1], g_qa_k_found_tracks[2][kQaNK][kNReg + 1];
    long g_qa_k_fake_all[kNReg + 1], g_qa_k_found_all[kNReg + 1];
    // flag scan at K = 4 (hits beyond the seed the search started from): flagged = [var 0 cleaning score, 1 fake
    // score] >= threshold t; fake tracks in acceptance removed, selected sim tracks lost [var][t][region]
    constexpr int kQaNT = 10;
    const float kQaT[2][kQaNT] = {{0.05f, 0.1f, 0.15f, 0.2f, 0.3f, 0.4f, 0.5f, 0.7f, 1.0f, 2.0f},
                                  {0.05f, 0.1f, 0.15f, 0.2f, 0.25f, 0.3f, 0.35f, 0.4f, 0.5f, 0.6f}};
    long g_qa_t_fake[2][kQaNT][kNReg + 1], g_qa_t_lost[2][kQaNT][kNReg + 1];  // fakes from 3+1 wrong quads whose majority sim track another track found
    int g_qa_nev = 0;
  }  // namespace

  void Shell::SeederQuadAnatomyReset() {
    memset(g_qa_cls, 0, sizeof(g_qa_cls));
    memset(g_qa_wrong_pos, 0, sizeof(g_qa_wrong_pos));
    memset(g_qa_wrong_comp, 0, sizeof(g_qa_wrong_comp));
    memset(g_qa_wrong_layer, 0, sizeof(g_qa_wrong_layer));
    memset(g_qa_dens, 0, sizeof(g_qa_dens));
    memset(g_qa_dens_comp, 0, sizeof(g_qa_dens_comp));
    memset(g_qa_dist_phi, 0, sizeof(g_qa_dist_phi));
    memset(g_qa_dist_q, 0, sizeof(g_qa_dist_q));
    memset(g_qa_trk, 0, sizeof(g_qa_trk));
    memset(g_qa_fake_wrong_pos, 0, sizeof(g_qa_fake_wrong_pos));
    memset(g_qa_fake_wrong_mfound, 0, sizeof(g_qa_fake_wrong_mfound));
    memset(g_qa_true_seedcls, 0, sizeof(g_qa_true_seedcls));
    memset(g_qa_sc, 0, sizeof(g_qa_sc));
    memset(g_qa_trk_sc, 0, sizeof(g_qa_trk_sc));
    memset(g_qa_bey, 0, sizeof(g_qa_bey));
    memset(g_qa_bey_n, 0, sizeof(g_qa_bey_n));
    memset(g_qa_for_seed, 0, sizeof(g_qa_for_seed));
    memset(g_qa_for_add, 0, sizeof(g_qa_for_add));
    memset(g_qa_unl, 0, sizeof(g_qa_unl));
    memset(g_qa_maj_in_quad, 0, sizeof(g_qa_maj_in_quad));
    memset(g_qa_for_only_seed, 0, sizeof(g_qa_for_only_seed));
    memset(g_qa_for_none, 0, sizeof(g_qa_for_none));
    memset(g_qa_k_fake, 0, sizeof(g_qa_k_fake));
    memset(g_qa_k_lost, 0, sizeof(g_qa_k_lost));
    memset(g_qa_k_found_tracks, 0, sizeof(g_qa_k_found_tracks));
    memset(g_qa_k_fake_all, 0, sizeof(g_qa_k_fake_all));
    memset(g_qa_k_found_all, 0, sizeof(g_qa_k_found_all));
    memset(g_qa_t_fake, 0, sizeof(g_qa_t_fake));
    memset(g_qa_t_lost, 0, sizeof(g_qa_t_lost));
    g_qa_nev = 0;
  }

  void Shell::SeederQuadAnatomy(EvCtx &ctx) {
    const Event &ev = *ctx.ev;
    auto qit = m_seeder_quads.find(ev.evtID());
    if (qit == m_seeder_quads.end())
      return;
    const std::vector<SeederQuad> &Q = qit->second;
    auto lab_of = [&](int l, int h) {
      const int mcid = ev.layerHits_[l][h].mcHitID();
      return mcid >= 0 ? ev.simHitsInfo_[mcid].mcTrackID() : -1;
    };
    auto q_of = [&](int l, const Hit &h) { return Config::TrkInfo[l].is_barrel() ? h.z() : h.r(); };
    // per layer, the hits sorted in phi, built when first needed
    std::map<int, std::vector<std::pair<float, float>>> by_phi;
    auto density = [&](int l, int h) {
      auto &v = by_phi[l];
      if (v.empty()) {
        for (const Hit &x : ev.layerHits_[l])
          v.push_back({x.phi(), q_of(l, x)});
        std::sort(v.begin(), v.end());
      }
      const Hit &x = ev.layerHits_[l][h];
      const float phi = x.phi(), q = q_of(l, x);
      int n = 0;
      // the window, with the wrap at +-pi
      for (float off : {0.f, 2.f * float(M_PI), -2.f * float(M_PI)}) {
        auto lo = std::lower_bound(v.begin(), v.end(), std::make_pair(phi + off - kQaDphi, -1e9f));
        for (auto it = lo; it != v.end() && it->first <= phi + off + kQaDphi; ++it)
          n += std::abs(it->second - q) < kQaDq;
      }
      return n - 1;  // not counting the hit itself
    };
    // the majority track's own hit in a layer, if it has one
    auto own_hit = [&](int L, int layer) {
      const Track &st = ev.simTracks_[L];
      for (int i = 0; i < st.nTotalHits(); ++i) {
        const HitOnTrack hot = st.getHitOnTrack(i);
        if (hot.layer == layer && hot.index >= 0)
          return hot.index;
      }
      return -1;
    };

    struct QaRes {
      int cls, reg, pos, maj;
    };
    std::vector<QaRes> res(Q.size());
    std::map<std::pair<int, int>, std::vector<int>> hit2quad;  // keyed by the quad's first hit
    for (int qi = 0; qi < (int)Q.size(); ++qi) {
      const SeederQuad &q = Q[qi];
      hit2quad[{q.l[0], q.h[0]}].push_back(qi);
      int lab[4];
      std::map<int, int> cnt;
      int n_unl = 0;
      for (int k = 0; k < 4; ++k) {
        lab[k] = lab_of(q.l[k], q.h[k]);
        if (lab[k] < 0)
          ++n_unl;
        else
          ++cnt[lab[k]];
      }
      int maj = -1, nmaj = 0;
      for (const auto &[l, n] : cnt)
        if (n > nmaj) {
          nmaj = n;
          maj = l;
        }
      int cls = QA_Worse, pos = -1;
      if (nmaj == 4)
        cls = QA_True;
      else if (nmaj == 3) {
        for (int k = 0; k < 4; ++k)
          if (lab[k] != maj)
            pos = k;
        cls = lab[pos] < 0 ? QA_Unl : QA_Wrong;
      }
      const Hit &h0 = ev.layerHits_[q.l[0]][q.h[0]], &h3 = ev.layerHits_[q.l[3]][q.h[3]];
      const float ae = std::abs(std::asinh((h3.z() - h0.z()) / std::max(1e-3f, h3.r() - h0.r())));
      const int reg = reg_of(ae);
      res[qi] = {cls, reg, pos, maj};
      ++g_qa_cls[reg][cls];
      ++g_qa_sc[reg][cls][qa_bin(kQaScEdge, q.score)];
      if (cls == QA_True)
        ++g_qa_dens[0][reg][qa_dens_bin(density(q.l[3], q.h[3]))];
      if (cls == QA_Wrong) {
        ++g_qa_wrong_pos[reg][pos];
        ++g_qa_wrong_layer[q.l[pos] & 63];
        ++g_qa_dens[1][reg][qa_dens_bin(density(q.l[pos], q.h[pos]))];
        const int own = maj >= 0 && maj < (int)ev.simTracks_.size() ? own_hit(maj, q.l[pos]) : -1;
        if (own >= 0) {
          ++g_qa_wrong_comp[reg][pos];
          ++g_qa_dens_comp[reg][qa_dens_bin(density(q.l[pos], own))];
          const Hit &a = ev.layerHits_[q.l[pos]][q.h[pos]], &b = ev.layerHits_[q.l[pos]][own];
          float dphi = std::abs(a.phi() - b.phi());
          if (dphi > float(M_PI))
            dphi = 2.f * float(M_PI) - dphi;
          ++g_qa_dist_phi[reg][qa_bin(kQaDphiEdge, dphi)];
          ++g_qa_dist_q[reg][qa_bin(kQaDqEdge, std::abs(q_of(q.l[pos], a) - q_of(q.l[pos], b)))];
        }
      }
    }

    // the tracks: found or fake by val_eff's association, and the quad each grew from
    const TrackVec &seeds = ctx.seeds;
    std::map<std::pair<int, int>, std::vector<int>> hit2seed;
    for (int i = 0; i < (int)seeds.size(); ++i)
      for (int k = 0; k < seeds[i].nTotalHits(); ++k) {
        const HitOnTrack hot = seeds[i].getHitOnTrack(k);
        if (hot.index >= 0 && hot.layer >= 0)
          hit2seed[{hot.layer, hot.index}].push_back(i);
      }
    std::set<int> found;
    struct TrkRow {
      bool fake, acc;
      int qi, reg, scls;
      int n_bey, f_seed, f_add, n_unl;
      bool maj_in_quad;
      int mc, n_bey_seed;
    };
    std::vector<TrkRow> rows;
    for (const Track &c : ev.candidateTracks_) {
      std::set<std::pair<int, int>> th;
      std::map<int, int> shared;
      for (int k = 0; k < c.nTotalHits(); ++k) {
        const HitOnTrack hot = c.getHitOnTrack(k);
        if (hot.index < 0 || hot.layer < 0)
          continue;
        th.insert({hot.layer, hot.index});
        auto it = hit2seed.find({hot.layer, hot.index});
        if (it != hit2seed.end())
          for (int i : it->second)
            ++shared[i];
      }
      int si = -1, bn = 0;
      for (const auto &[i, n] : shared)
        if (n > bn) {
          bn = n;
          si = i;
        }
      TrackExtra extra(c.label());
      if (si >= 0)
        extra.findMatchingSeedHits(c, seeds[si], ev.layerHits_);
      extra.setMCTrackIDInfo(c, ev.layerHits_, ev.simHitsInfo_, ev.simTracks_, false, false);
      const int mc = extra.mcTrackID();
      const bool fake = mc < 0 || mc >= (int)ev.simTracks_.size();
      if (!fake)
        found.insert(mc);
      int qsel = -1;
      for (const auto &h : th) {
        auto it = hit2quad.find(h);
        if (it == hit2quad.end())
          continue;
        for (int qi : it->second) {
          const SeederQuad &q = Q[qi];
          bool all = true;
          for (int k = 1; k < 4 && all; ++k)
            all = th.count({q.l[k], q.h[k]});
          if (all) {
            qsel = qi;
            break;
          }
        }
        if (qsel >= 0)
          break;
      }
      const float ae = std::abs(c.momEta());
      // the track's hits against its own majority particle, split into the quad's four and the added ones
      int n_bey = 0, f_seed = 0, f_add = 0, n_unl = 0;
      bool maj_in_quad = false;
      if (qsel >= 0) {
        const SeederQuad &q = Q[qsel];
        std::set<std::pair<int, int>> qh;
        for (int k = 0; k < 4; ++k)
          qh.insert({q.l[k], q.h[k]});
        std::map<int, int> votes;
        std::vector<std::pair<int, bool>> hl;  // label, is a quad hit
        for (const auto &h : th) {
          const int lab = lab_of(h.first, h.second);
          const bool inq = qh.count(h);
          hl.push_back({lab, inq});
          n_bey += !inq;
          if (lab >= 0)
            ++votes[lab];
        }
        int maj = -1, nv = 0;
        for (const auto &[l, n] : votes)
          if (n > nv) {
            nv = n;
            maj = l;
          }
        for (const auto &[lab, inq] : hl) {
          if (lab < 0)
            ++n_unl;
          else if (lab != maj)
            ++(inq ? f_seed : f_add);
        }
        for (int k = 0; k < 4; ++k)
          maj_in_quad |= maj >= 0 && lab_of(q.l[k], q.h[k]) == maj;
      }
      // hits beyond the seed the search started from (after the iteration cleaner's merge)
      int n_bey_seed = 0;
      if (si >= 0) {
        std::set<std::pair<int, int>> sh;
        for (int k = 0; k < seeds[si].nTotalHits(); ++k) {
          const HitOnTrack hot = seeds[si].getHitOnTrack(k);
          if (hot.index >= 0 && hot.layer >= 0)
            sh.insert({hot.layer, hot.index});
        }
        for (const auto &h : th)
          n_bey_seed += !sh.count(h);
      }
      rows.push_back({fake, c.pT() > 0.9f && ae < 2.5f, qsel, reg_of(ae), si >= 0 ? seed_class(ev, seeds[si]) : 3,
                      n_bey, f_seed, f_add, n_unl, maj_in_quad, fake ? -1 : mc, n_bey_seed});
    }
    for (const TrkRow &r : rows) {
      if (r.acc && r.qi >= 0) {
        const float sc = Q[r.qi].score;
        for (int g = 0; g < 2; ++g) {
          if (g == 1 && !(sc >= 0.3f && sc < 1.f))
            continue;
          const int f = r.fake;
          ++g_qa_bey[g][f][std::min(r.n_bey, kQaNBeyond - 1)];
          ++g_qa_bey_n[g][f];
          g_qa_for_seed[g][f] += r.f_seed;
          g_qa_for_add[g][f] += r.f_add;
          g_qa_unl[g][f] += r.n_unl;
          g_qa_maj_in_quad[g][f] += r.maj_in_quad;
          g_qa_for_only_seed[g][f] += r.f_seed > 0 && r.f_add == 0;
          g_qa_for_none[g][f] += r.f_seed == 0 && r.f_add == 0;
        }
      }
      const int cls = r.qi >= 0 ? res[r.qi].cls : QA_N;
      for (int a = 0; a < 2; ++a) {
        if (a == 1 && !r.acc)
          continue;
        ++g_qa_trk[a][r.fake][r.reg][cls];
        if (r.qi >= 0)
          ++g_qa_trk_sc[a][r.fake][r.reg][qa_bin(kQaScEdge, Q[r.qi].score)];
        if (cls == QA_True)
          ++g_qa_true_seedcls[a][r.fake][r.reg][r.scls];
        if (r.fake && cls == QA_Wrong) {
          ++g_qa_fake_wrong_pos[a][r.reg][res[r.qi].pos];
          g_qa_fake_wrong_mfound[a][r.reg] += found.count(res[r.qi].maj);
        }
      }
    }
    // the K scan. The selected sim tracks (val_eff's MTV selection) and, per sim track, the tracks that find it.
    std::map<int, int> sim_reg;
    for (int L = 0; L < (int)ev.simTracks_.size(); ++L) {
      const Track &st = ev.simTracks_[L];
      const float ae = std::abs(st.momEta()), pt = st.pT();
      if (!st.isFindable() || std::hypot(st.x(), st.y()) > 3.5f || std::abs(st.z()) > 30.0f)
        continue;
      if (ae >= 2.5f || pt <= 0.9f || st.nUniqueLayers() < 4)
        continue;
      sim_reg[L] = reg_of(ae);
    }
    std::map<int, std::vector<int>> finders;  // sim label -> rows that find it
    for (int i = 0; i < (int)rows.size(); ++i)
      if (!rows[i].fake && sim_reg.count(rows[i].mc))
        finders[rows[i].mc].push_back(i);
    for (const TrkRow &r : rows)
      if (r.acc) {
        if (r.fake)
          ++g_qa_k_fake_all[r.reg], ++g_qa_k_fake_all[kNReg];
        else
          ++g_qa_k_found_all[r.reg], ++g_qa_k_found_all[kNReg];
      }
    for (int def = 0; def < 2; ++def)
      for (int kk = 0; kk < kQaNK; ++kk) {
        const int K = kQaK0 + kk;
        auto removed = [&](const TrkRow &r) {
          if (r.qi < 0)
            return false;
          const float sc = Q[r.qi].score;
          return sc >= 0.3f && sc < 1.f && (def == 0 ? r.n_bey : r.n_bey_seed) < K;
        };
        for (const TrkRow &r : rows)
          if (r.acc && removed(r)) {
            if (r.fake)
              ++g_qa_k_fake[def][kk][r.reg], ++g_qa_k_fake[def][kk][kNReg];
            else
              ++g_qa_k_found_tracks[def][kk][r.reg], ++g_qa_k_found_tracks[def][kk][kNReg];
          }
        for (const auto &[L, idx] : finders) {
          bool any = false;
          for (int i : idx)
            any |= !removed(rows[i]);
          if (!any)
            ++g_qa_k_lost[def][kk][sim_reg[L]], ++g_qa_k_lost[def][kk][kNReg];
        }
      }
    for (int var = 0; var < 2; ++var)
      for (int it = 0; it < kQaNT; ++it) {
        const float t = kQaT[var][it];
        auto removed = [&](const TrkRow &r) {
          if (r.qi < 0)
            return false;
          const float v = var == 0 ? Q[r.qi].score : Q[r.qi].fake_score;
          return v >= t && r.n_bey_seed < 4;
        };
        for (const TrkRow &r : rows)
          if (r.acc && r.fake && removed(r))
            ++g_qa_t_fake[var][it][r.reg], ++g_qa_t_fake[var][it][kNReg];
        for (const auto &[L, idx] : finders) {
          bool any = false;
          for (int i : idx)
            any |= !removed(rows[i]);
          if (!any)
            ++g_qa_t_lost[var][it][sim_reg[L]], ++g_qa_t_lost[var][it][kNReg];
        }
      }
    ++g_qa_nev;
  }

  void Shell::SeederQuadAnatomyReport() {
    printf("\nShell::SeederQuadAnatomyReport: %d events; region by the line from the quad's first to its last hit\n",
           g_qa_nev);
    auto row3 = [](const char *name, const long *v, int n) {
      long t = 0;
      for (int i = 0; i < n; ++i)
        t += v[i];
      printf("  %-26s %8ld |", name, t);
      for (int i = 0; i < n; ++i)
        printf(" %7ld %5.1f%% |", v[i], 100.0 * v[i] / std::max(1L, t));
      printf("\n");
    };
    printf("\n  QUADS by class\n  %-26s %8s |", "region", "quads");
    for (int c = 0; c < QA_N; ++c)
      printf(" %14s |", kQaName[c]);
    printf("\n");
    for (int r = 0; r < kNReg; ++r)
      row3(kRegName[r], g_qa_cls[r], QA_N);

    printf("\n  QUADS by their score (seedsurf's cleaning score), per class\n  %-26s %8s |", "", "quads");
    for (int b = 0; b < kQaNSc; ++b)
      printf(" %14s |", kQaScName[b]);
    printf("\n");
    for (int r = 0; r < kNReg; ++r) {
      printf("  %s\n", kRegName[r]);
      for (int c = 0; c < QA_N; ++c) {
        char name[64];
        snprintf(name, sizeof(name), "   %s", kQaName[c]);
        row3(name, g_qa_sc[r][c], kQaNSc);
      }
    }

    printf("\n  3+1 WRONG: the position of the wrong hit (0 = innermost), and how many of those have the majority track's"
           " own hit in that layer\n");
    for (int r = 0; r < kNReg; ++r) {
      printf("  %-26s", kRegName[r]);
      for (int p = 0; p < 4; ++p)
        printf(" | pos %d: %6ld (own hit there %5.1f%%)", p, g_qa_wrong_pos[r][p],
               100.0 * g_qa_wrong_comp[r][p] / std::max(1L, g_qa_wrong_pos[r][p]));
      printf("\n");
    }
    printf("  the wrong hit's layer:");
    for (int l = 0; l < 64; ++l)
      if (g_qa_wrong_layer[l] > 0)
        printf(" L%d %ld", l, g_qa_wrong_layer[l]);
    printf("\n");

    printf("\n  HIT DENSITY: other hits in the same layer within %.0f mrad and %.1f cm (z barrel, r disc) of the hit\n",
           1e3 * kQaDphi, kQaDq);
    printf("  %-26s %8s |", "", "hits");
    for (int b = 0; b < kQaNDens; ++b)
      printf(" %14s |", kQaDensName[b]);
    printf("\n");
    for (int r = 0; r < kNReg; ++r) {
      printf("  %s\n", kRegName[r]);
      row3("   last hit of true quads", g_qa_dens[0][r], kQaNDens);
      row3("   wrong hit of 3+1 wrong", g_qa_dens[1][r], kQaNDens);
      row3("   the majority's own hit", g_qa_dens_comp[r], kQaNDens);
    }

    printf("\n  3+1 WRONG with the majority track's own hit in the layer: the distance between the two hits\n");
    for (int r = 0; r < kNReg; ++r) {
      printf("  %s\n", kRegName[r]);
      printf("  %-26s %8s |", "   |dphi| [mrad]", "");
      for (int b = 0; b < kQaNDphi; ++b)
        printf(" %14s |", kQaDphiName[b]);
      printf("\n");
      row3("", g_qa_dist_phi[r], kQaNDphi);
      printf("  %-26s %8s |", "   |dq| [cm]", "");
      for (int b = 0; b < kQaNDq; ++b)
        printf(" %14s |", kQaDqName[b]);
      printf("\n");
      row3("", g_qa_dist_q[r], kQaNDq);
    }

    printf("\n  TRACKS in acceptance by their hits: beyond the quad's four, and against the track's majority particle\n");
    printf("  %-30s %7s %7s %8s %8s | %9s %9s %9s | %10s %10s %10s\n", "", "tracks", "median", "beyond>=5", "beyond<5",
           "for.seed", "for.added", "unlinked", "only seed", "no foreign", "maj in quad");
    for (int g = 0; g < 2; ++g)
      for (int f = 0; f < 2; ++f) {
        const long n = g_qa_bey_n[g][f];
        long acc = 0, ge5 = 0;
        int med = -1;
        for (int b = 0; b < kQaNBeyond; ++b) {
          acc += g_qa_bey[g][f][b];
          if (med < 0 && 2 * acc >= n)
            med = b;
          if (b >= 5)
            ge5 += g_qa_bey[g][f][b];
        }
        const double d = std::max(1L, n);
        printf("  %-30s %7ld %7d %7.1f%% %7.1f%% | %9.2f %9.2f %9.2f | %9.1f%% %9.1f%% %9.1f%%\n",
               g ? (f ? "flagged quads, fake" : "flagged quads, found") : (f ? "all quads, fake" : "all quads, found"), n,
               med, 100.0 * ge5 / d, 100.0 * (n - ge5) / d, g_qa_for_seed[g][f] / d, g_qa_for_add[g][f] / d,
               g_qa_unl[g][f] / d, 100.0 * g_qa_for_only_seed[g][f] / d, 100.0 * g_qa_for_none[g][f] / d,
               100.0 * g_qa_maj_in_quad[g][f] / d);
      }
    for (int def = 0; def < 2; ++def) {
      printf("\n  K SCAN: a track from a flagged quad (score in [0.3, 1)) with fewer than K hits beyond %s is removed:\n"
             "  fake tracks in acceptance removed / found tracks in acceptance removed / selected sim tracks no longer"
             " found by any track\n",
             def == 0 ? "the quad's four" : "the seed the search started from (after the iteration cleaner's merge)");
      printf("  %-6s", "K");
      for (int r = 0; r <= kNReg; ++r)
        printf(" | %-32s", r == kNReg ? "all regions" : kRegName[r]);
      printf("\n  %-6s", "total");
      for (int r = 0; r <= kNReg; ++r)
        printf(" | fakes %6ld found trk %6ld        ", g_qa_k_fake_all[r], g_qa_k_found_all[r]);
      printf("\n");
      for (int kk = 0; kk < kQaNK; ++kk) {
        printf("  K=%-4d", kQaK0 + kk);
        for (int r = 0; r <= kNReg; ++r)
          printf(" | -%5ld fake -%5ld trk -%4ld sim ", g_qa_k_fake[def][kk][r], g_qa_k_found_tracks[def][kk][r],
                 g_qa_k_lost[def][kk][r]);
        printf("\n");
      }
    }
    for (int var = 0; var < 2; ++var) {
      printf("\n  FLAG SCAN at K = 4 (hits beyond the seed after the merge): flagged = %s >= t; fake tracks in acceptance"
             " removed / selected sim tracks lost\n  %-8s", var ? "the FAKE score" : "the CLEANING score", "t");
      for (int r = 0; r <= kNReg; ++r)
        printf(" | %-24s", r == kNReg ? "all regions" : kRegName[r]);
      printf("\n");
      for (int it = 0; it < kQaNT; ++it) {
        printf("  %-8.2f", kQaT[var][it]);
        for (int r = 0; r <= kNReg; ++r)
          printf(" | -%5ld fake -%4ld sim       ", g_qa_t_fake[var][it][r], g_qa_t_lost[var][it][r]);
        printf("\n");
      }
    }
    printf("  (flagged: the quad's score in [0.3, 1); for.seed / for.added: mean number of the track's hits on another"
           " particle than its majority, among the quad's hits / the added hits; only seed: the foreign hits are all"
           " seed hits)\n");

    for (int a = 0; a < 2; ++a) {
      printf("\n  TRACKS by the class of the quad they grew from (the quad whose four hits are on the track), %s\n",
             a ? "pT > 0.9 and |eta| < 2.5" : "all");
      printf("  %-26s %8s |", "", "tracks");
      for (int c = 0; c < QA_N; ++c)
        printf(" %14s |", kQaName[c]);
      printf(" %14s |\n", "no quad");
      for (int f = 0; f < 2; ++f)
        for (int r = 0; r < kNReg; ++r) {
          char name[64];
          snprintf(name, sizeof(name), "%s %s", f ? "fake " : "found", kRegName[r]);
          row3(name, g_qa_trk[a][f][r], QA_N + 1);
        }
      printf("  FAKE tracks from 3+1 wrong quads: the wrong hit's position; and how many have their majority sim track"
             " found by another track\n");
      for (int r = 0; r < kNReg; ++r)
        printf("  %-26s pos 0 %5ld | pos 1 %5ld | pos 2 %5ld | pos 3 %5ld | majority found elsewhere %5ld\n",
               kRegName[r], g_qa_fake_wrong_pos[a][r][0], g_qa_fake_wrong_pos[a][r][1], g_qa_fake_wrong_pos[a][r][2],
               g_qa_fake_wrong_pos[a][r][3], g_qa_fake_wrong_mfound[a][r]);
      printf("  tracks by their quad's score (seedsurf's cleaning score)\n  %-26s %8s |", "", "tracks");
      for (int b = 0; b < kQaNSc; ++b)
        printf(" %14s |", kQaScName[b]);
      printf("\n");
      for (int f = 0; f < 2; ++f)
        for (int r = 0; r < kNReg; ++r) {
          char name[64];
          snprintf(name, sizeof(name), "%s %s", f ? "fake " : "found", kRegName[r]);
          row3(name, g_qa_trk_sc[a][f][r], kQaNSc);
        }
      printf("  tracks from TRUE quads, by the seed the search started from (after the iteration's seed cleaning):"
             " true / undecidable / fake / none\n");
      for (int f = 0; f < 2; ++f)
        for (int r = 0; r < kNReg; ++r) {
          char name[64];
          snprintf(name, sizeof(name), "%s %s", f ? "fake " : "found", kRegName[r]);
          row3(name, g_qa_true_seedcls[a][f][r], 4);
        }
    }
  }

}  // namespace mkfit

//=============================================================================
// CMSSW's MultiTrackSelector, phase-2 initialStepSelector
//=============================================================================

namespace mkfit {

  namespace {
    struct MtsPar {
      float chi2n_par, res_par[2], d0_par1, dz_par1, d0_par2, dz_par2;
      unsigned min_layers, min_3Dlayers, max_lost_layers;
      float min_eta, max_eta;
    };
    // RecoTracker/IterativeTracking/python/InitialStep_cff.py, trackingPhase2PU140 initialStepSelector;
    // the second element of every d0/dz pair is 4.0. The rest of looseMTS's defaults: max_d0 = max_z0 =
    // 100, nSigmaZ 4, minHitsToBypassChecks 20, applyAdaptedPVCuts true, no other cut.
    const MtsPar kMts[3] = {
        {2.0f, {0.003f, 0.002f}, 0.8f, 0.9f, 0.6f, 0.8f, 3, 3, 3, -9999.f, 9999.f},   // initialStepLoose
        {1.4f, {0.003f, 0.002f}, 0.7f, 0.8f, 0.5f, 0.7f, 3, 3, 2, -9999.f, 9999.f},   // initialStepTight
        {1.2f, {0.003f, 0.001f}, 0.6f, 0.7f, 0.45f, 0.55f, 3, 3, 2, -4.1f, 4.1f},     // initialStep (highPurity)
    };
    float pow4(float x) { return x * x * x * x; }
  }  // namespace

  int Shell::SelectTracksCMSSW(EvCtx &ctx, int level) {
    Event &ev = *ctx.ev;
    const BeamSpot &bs = ev.beamSpot_;
    // the primary vertices, from the sim tracks
    std::map<long, std::pair<SVector3, int>> pv;
    for (const Track &st : ev.simTracks_) {
      if (st.charge() == 0 || std::hypot(st.x() - bs.x, st.y() - bs.y) > 0.01f)
        continue;
      auto &e = pv[std::lround(st.z() * 1e4)];
      e.first = SVector3(st.x(), st.y(), st.z());
      ++e.second;
    }
    std::vector<SVector3> points;
    for (const auto &kv : pv)
      if (kv.second.second >= 2)
        points.push_back(kv.second.first);

    auto phys = [](int l) { return Config::TrkInfo[l].is_pixel() ? l : 100 + l / 2; };

    // the first cut a track fails: layers, 3D layers, lost layers, ndof, chi2, eta, d0, dz
    long why[8] = {0};
    double med_d0[2] = {0, 0}, med_r = 0;
    long n_med = 0;
    auto pass = [&](const Track &t, const MtsPar &P) -> bool {
      const int nh = t.nFoundHits();
      if (nh >= 20)  // minHitsToBypassChecks
        return true;
      std::map<int, int> valid;  // physical layer -> bit 1 first sub-layer, 2 second
      std::set<int> missed;
      int first = -1, last = -1;
      for (int i = 0; i < t.nTotalHits(); ++i)
        if (t.getHitOnTrack(i).index >= 0) {
          if (first < 0)
            first = i;
          last = i;
        }
      for (int i = 0; i < t.nTotalHits(); ++i) {
        const HitOnTrack h = t.getHitOnTrack(i);
        if (h.layer < 0)
          continue;
        if (h.index >= 0)
          valid[phys(h.layer)] |= Config::TrkInfo[h.layer].is_pixel() ? 3 : 1 << (h.layer & 1);
        else if (h.index == Hit::kHitMissIdx && i > first && i < last)
          missed.insert(phys(h.layer));
      }
      const unsigned nlayers = valid.size();
      unsigned n3d = 0, nlost = 0;
      for (const auto &kv : valid)
        n3d += kv.second == 3;
      for (int l : missed)
        nlost += !valid.count(l);
      if (nlayers < P.min_layers)
        return ++why[0], false;
      if (n3d < P.min_3Dlayers)
        return ++why[1], false;
      if (nlost > P.max_lost_layers)
        return ++why[2], false;
      const int ndof = 2 * nh - 5;
      if (ndof < 1)
        return ++why[3], false;
      if (t.chi2() / ndof > P.chi2n_par * nlayers)
        return ++why[4], false;
      const float pt = std::max(t.pT(), 1e-6f), eta = t.momEta();
      if (eta < P.min_eta || eta > P.max_eta)
        return ++why[5], false;
      // d0 and z0 at the point of closest approach to the beam line, from the helix through the state
      // (at the track's first hit); their errors by a linear transport of the state's covariance back
      // along the track (straight line, plus the sagitta's dependence on 1/pT for d0)
      const float k = (t.charge() < 0 ? 100.0f : -100.0f) / (Const::sol * Config::Bfield);
      const float R = std::abs(k) * pt, xc = t.x() - k * t.py(), yc = t.y() + k * t.px();
      const float ux = bs.x - xc, uy = bs.y - yc, ul = std::hypot(ux, uy);
      const float d0 = ul - R;
      const float rsx = t.x() - xc, rsy = t.y() - yc;
      const float ca = std::clamp((rsx * ux + rsy * uy) / (R * ul), -1.0f, 1.0f);
      const float s_arc = R * std::acos(ca);
      const float cot = 1.0f / std::tan(t.theta()), sth = std::sin(t.theta());
      const float z0 = t.z() - s_arc * cot;
      const float cphi = std::cos(t.momPhi()), sphi = std::sin(t.momPhi());
      const SMatrixSym66 &C = t.errors();
      // jacobians in (x, y, z, 1/pT, phi, theta)
      const float kq = 0.5f * s_arc * s_arc * Const::sol_over_100 * Config::Bfield * t.charge();
      const float Jd[6] = {-sphi, cphi, 0, kq, -s_arc, 0};
      const float Jz[6] = {-cphi * cot, -sphi * cot, 1, 0, 0, s_arc / (sth * sth)};
      float vd = 0, vz = 0;
      for (int a = 0; a < 6; ++a)
        for (int b = 0; b < 6; ++b)
          vd += Jd[a] * C(a, b) * Jd[b], vz += Jz[a] * C(a, b) * Jz[b];
      const float d0E = std::sqrt(std::max(0.f, vd)), dzE = std::sqrt(std::max(0.f, vz));
      const float nomd0E = std::sqrt(P.res_par[0] * P.res_par[0] + (P.res_par[1] / pt) * (P.res_par[1] / pt));
      const float nomdzE = nomd0E * std::cosh(eta);
      const float dzCut = std::min(pow4(P.dz_par1 * nlayers) * nomdzE, pow4(P.dz_par2 * nlayers) * dzE);
      const float d0Cut = std::min(pow4(P.d0_par1 * nlayers) * nomd0E, pow4(P.d0_par2 * nlayers) * d0E);
      const float dzbs = z0 - bs.z;
      bool zok = false, dok = false;
      if (points.empty()) {
        zok = std::abs(dzbs) < std::hypot(bs.sigmaZ * 4.f, dzCut);
        dok = std::abs(d0) < d0Cut;
      }
      for (const SVector3 &v : points) {
        if (zok && dok)
          break;
        // the vertices sit within 0.01 cm of the beam line: d0 to the beam line stands for d0 to each
        zok |= std::abs(z0 - v[2]) < dzCut;
        dok |= std::abs(d0) < d0Cut;
      }
      med_d0[0] += std::abs(d0), med_d0[1] += d0Cut, med_r += t.chi2() / ndof, ++n_med;
      if (std::abs(d0) > 100.f && !dok)
        return false;
      if (std::abs(dzbs) > 100.f && !zok)
        return false;
      if (!dok)
        return ++why[6], false;
      if (!zok) {
        if (s_seeder_debug > 0 && why[7] < 8) {
          float best = 1e9;
          for (const SVector3 &v : points)
            best = std::min(best, std::abs(z0 - v[2]));
          printf("[mts] dz fail: pt %.2f eta %.2f nl %u r %.3f z %.2f dzbs %.3f best |dzPV| %.4f dzCut %.4f (par1 %.4f par2 %.4f)"
                 " dzE %.3g nomdzE %.3g chi2n %.2f\n",
                 pt, eta, nlayers, std::hypot(t.x(), t.y()), t.z(), dzbs, best,
                 dzCut, pow4(P.dz_par1 * nlayers) * nomdzE, pow4(P.dz_par2 * nlayers) * dzE, dzE, nomdzE, t.chi2() / ndof);
        }
        return ++why[7], false;
      }
      return true;
    };

    TrackVec &tv = ev.candidateTracks_;
    const int n_in = tv.size();
    for (int lv = 1; lv <= level; ++lv)
      tv.erase(std::remove_if(tv.begin(), tv.end(), [&](const Track &t) { return !pass(t, kMts[lv - 1]); }), tv.end());
    printf("Shell::SelectTracksCMSSW: event %d, level %d: %d of %d tracks kept (%d sim primary vertices); first failed"
           " cut: layers %ld, 3D layers %ld, lost %ld, ndof %ld, chi2 %ld, eta %ld, d0 %ld, dz %ld; mean |d0| %.3g,"
           " d0 cut %.3g, chi2/ndof %.3g\n",
           ev.evtID(), level, (int)tv.size(), n_in, (int)points.size(), why[0], why[1], why[2], why[3], why[4], why[5],
           why[6], why[7], n_med ? med_d0[0] / n_med : 0., n_med ? med_d0[1] / n_med : 0., n_med ? med_r / n_med : 0.);
    return tv.size();
  }

}  // namespace mkfit
