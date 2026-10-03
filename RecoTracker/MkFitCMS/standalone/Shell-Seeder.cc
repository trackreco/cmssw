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
      if (sscanf(line, "%d %d %d %d %d %d %d %d %d %f", &ev, &q.l[0], &q.l[1], &q.l[2], &q.l[3], &q.h[0], &q.h[1],
                 &q.h[2], &q.h[3], &q.score) != 10)
        continue;
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
    const Event &ev = *ctx.ev;
    auto it = m_seeder_quads.find(ev.evtID());
    if (it == m_seeder_quads.end()) {
      printf("Shell::MakeSeederSeeds: no quads for event %d\n", ev.evtID());
      return 0;
    }
    const std::vector<SeederQuad> &Q = it->second;
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
