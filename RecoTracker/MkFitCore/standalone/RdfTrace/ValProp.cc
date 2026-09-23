#include "RecoTracker/MkFitCore/standalone/RdfTrace/ValProp.h"
#include "RecoTracker/MkFitCore/standalone/DataFormats/ValStructs.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/cms_common_macros.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"

#include "RecoTracker/MkFitCore/src/Matrix.h"
#include "RecoTracker/MkFitCore/src/PropagationMPlex.h"
#include "RecoTracker/MkFitCore/src/KalmanUtilsMPlex.h"
#include "RecoTracker/MkFitCore/src/MkFinder.h"
#include "RecoTracker/MkFitCore/src/MkFinderV2p2.h"
#include "RecoTracker/MkFitCore/src/V2p2Score.h"
#include "RecoTracker/MkFitCore/src/MkBins.h"

#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"
#include "RecoTracker/MkFitCore/standalone/TrackExtra.h"
#include "RecoTracker/MkFitCMS/standalone/Shell.h"
#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

#include <map>
#include <set>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <array>

#include "TFile.h"
#include "TTree.h"
#include "TH1D.h"
#include "TCanvas.h"
#include "TLegend.h"
#include "TVectorD.h"
#include "TMatrixD.h"
#include "TMatrixDSymEigen.h"
#include "TMatrixDSym.h"

#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace mkfit {

  // Lite trace: keep only pixel-barrel hits and skip the exact double-precision
  // helix-plane solve. That solve plus the sheer volume (~142k records/event) is
  // what makes val_search_event ~45 s/event; with this it is seconds, which is
  // what makes a covariance SCAN affordable.
  static int g_val_search_lite = 0;   // bit0 skip exact solve, bit1 pixel barrel only

  namespace {

    // ----------------------------------------------------------------------
    // Exact uniform-B helix from a production vertex, parametrized by signed
    // 3D path length s. This is the TRUTH: it defines where the planes are,
    // and nothing in it comes from the code under test.

    struct ValHelix {
      double x0 = 0, y0 = 0, z0 = 0;
      double px0 = 0, py0 = 0, pz0 = 0;
      double pt = 1, pmag = 1, theta = 1.57;
      double k = 0;  // cm/GeV, signed
      int chg = 1;

      void init(double a_pt, double eta, int charge, double phi0, double d0, double zv) {
        pt = a_pt;
        chg = charge;
        theta = 2.0 * std::atan(std::exp(-eta));
        px0 = pt * std::cos(phi0);
        py0 = pt * std::sin(phi0);
        pz0 = pt / std::tan(theta);
        pmag = pt / std::sin(theta);
        const double inv_k = ((charge < 0) ? 0.01 : -0.01) * (double)Const::sol * Config::Bfield;
        k = 1.0 / inv_k;
        // Displace the vertex PERPENDICULAR to the momentum in the transverse
        // plane, so d0 is a signed impact parameter and not just a shift along
        // the track (which would be no displacement at all).
        x0 = -d0 * std::sin(phi0);
        y0 = d0 * std::cos(phi0);
        z0 = zv;
      }

      double alpha_of_s(double s) const { return s / (k * pmag); }
      double s_of_alpha(double a) const { return a * k * pmag; }

      void at_alpha(double a, double p[3], double m[3]) const {
        const double sa = std::sin(a), ca = std::cos(a);
        p[0] = x0 + k * (px0 * sa - py0 * (1.0 - ca));
        p[1] = y0 + k * (py0 * sa + px0 * (1.0 - ca));
        p[2] = z0 + k * pz0 * a;
        m[0] = px0 * ca - py0 * sa;
        m[1] = px0 * sa + py0 * ca;
        m[2] = pz0;
      }
      void at_s(double s, double p[3], double m[3]) const { at_alpha(alpha_of_s(s), p, m); }

      double r_at_s(double s) const {
        double p[3], m[3];
        at_s(s, p, m);
        return std::hypot(p[0], p[1]);
      }

      // CCS parameter vector (x, y, z, 1/pT, phi, theta) at path length s.
      void ccs_at_s(double s, float par[6]) const {
        double p[3], m[3];
        at_s(s, p, m);
        par[0] = (float)p[0];
        par[1] = (float)p[1];
        par[2] = (float)p[2];
        par[3] = (float)(1.0 / pt);
        par[4] = (float)std::atan2(m[1], m[0]);
        par[5] = (float)theta;
      }

      // First crossing of radius R at s > 0, or false if the track never gets
      // there. Scan-then-bisect, capped at half a turn: beyond |alpha| = pi a
      // track from the origin is coming back in, and a closure test has no
      // business out there.
      bool first_crossing_r(double R, double &s_out) const {
        const double s_max = std::fabs(M_PI * k * pmag);
        const int N = 4000;
        double s_prev = 0.0, f_prev = r_at_s(0.0) - R;
        for (int i = 1; i <= N; ++i) {
          const double s = s_max * i / N;
          const double f = r_at_s(s) - R;
          if ((f_prev < 0.0) != (f < 0.0)) {
            double lo = s_prev, hi = s, flo = f_prev;
            for (int it = 0; it < 80; ++it) {
              const double sm = 0.5 * (lo + hi);
              const double fm = r_at_s(sm) - R;
              if ((flo < 0.0) == (fm < 0.0)) { lo = sm; flo = fm; } else { hi = sm; }
            }
            s_out = 0.5 * (lo + hi);
            return true;
          }
          s_prev = s;
          f_prev = f;
        }
        return false;
      }
    };

    // ----------------------------------------------------------------------
    // Plane construction.
    //
    // A plane is a point and a normal. The propagator solves n.(x(s) - p) = 0,
    // so the derivative of that equation along the track is n.dx/ds = |p^ . n|
    // -- the INCIDENCE. At incidence 1 the plane is normal to the track and the
    // solve is as well conditioned as it can be; as incidence -> 0 the track
    // runs parallel to the plane, touches rather than crosses, and the root
    // becomes double. That last case cannot occur in a detector (a module a
    // track grazes produces no hit), so it is excluded rather than measured.
    //
    // The tilt is applied in the plane spanned by the momentum and the radial
    // direction, which is where real module tilt lives.
    void make_plane(const double p[3], const double m[3], double cos_inc,
                    float pnt[3], float nrm[3]) {
      const double pm = std::sqrt(m[0]*m[0] + m[1]*m[1] + m[2]*m[2]);
      const double ph[3] = {m[0]/pm, m[1]/pm, m[2]/pm};
      double n[3] = {ph[0], ph[1], ph[2]};
      if (cos_inc < 0.999999) {
        // Rotation axis: perpendicular to both the momentum and the radial
        // direction, so the tilt happens in the (p^, r^) plane.
        const double rr = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        double rh[3] = {p[0]/rr, p[1]/rr, p[2]/rr};
        double u[3] = {ph[1]*rh[2] - ph[2]*rh[1],
                       ph[2]*rh[0] - ph[0]*rh[2],
                       ph[0]*rh[1] - ph[1]*rh[0]};
        double un = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
        if (un < 1e-6) {  // momentum parallel to the radius: any perpendicular will do
          const double a[3] = {0.0, 0.0, 1.0};
          u[0] = ph[1]*a[2] - ph[2]*a[1];
          u[1] = ph[2]*a[0] - ph[0]*a[2];
          u[2] = ph[0]*a[1] - ph[1]*a[0];
          un = std::sqrt(u[0]*u[0] + u[1]*u[1] + u[2]*u[2]);
        }
        for (int i = 0; i < 3; ++i) u[i] /= un;
        const double th = std::acos(cos_inc), c = std::cos(th), sn = std::sin(th);
        const double dot = u[0]*ph[0] + u[1]*ph[1] + u[2]*ph[2];
        const double cx[3] = {u[1]*ph[2] - u[2]*ph[1],
                              u[2]*ph[0] - u[0]*ph[2],
                              u[0]*ph[1] - u[1]*ph[0]};
        for (int i = 0; i < 3; ++i) n[i] = ph[i]*c + cx[i]*sn + u[i]*dot*(1.0 - c);
      }
      const double nn = std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
      for (int i = 0; i < 3; ++i) { pnt[i] = (float)p[i]; nrm[i] = (float)(n[i]/nn); }
    }

    // ----------------------------------------------------------------------
    // Matriplex helpers. Every lane is filled identically and lane 0 is read
    // back: wasteful, and it keeps the call identical to production.

    inline int ls_idx(int i, int j) {
      if (i < j) { const int t = i; i = j; j = t; }
      return i * (i + 1) / 2 + j;
    }

    void fill_lanes(MPlexLV &par, const float p[6]) {
      for (int n = 0; n < NN; ++n)
        for (int i = 0; i < 6; ++i) par(n, i, 0) = p[i];
    }
    void fill_lanes_sym(MPlexLS &err, const float c[21]) {
      for (int n = 0; n < NN; ++n)
        for (int i = 0; i < 21; ++i) err.fArray[i * NN + n] = c[i];
    }
    void fill_lanes_hv(MPlexHV &v, const float p[3]) {
      for (int n = 0; n < NN; ++n)
        for (int i = 0; i < 3; ++i) v(n, i, 0) = p[i];
    }
    void fill_lanes_q(MPlexQI &v, int x) {
      for (int n = 0; n < NN; ++n) v(n, 0, 0) = x;
    }
    void read_lane0(const MPlexLV &par, const MPlexLS &err, float p[6], float c[21]) {
      for (int i = 0; i < 6; ++i) p[i] = par.constAt(0, i, 0);
      for (int i = 0; i < 21; ++i) c[i] = err.fArray[i * NN];
    }

    // A plausible input covariance. Its exact value does not matter -- closure
    // is an internal-consistency test -- but it must be non-degenerate and
    // have off-diagonal terms, so that a jacobian bug cannot hide behind a
    // diagonal input.
    void nominal_cov(const float par[6], float c[21]) {
      std::memset(c, 0, 21 * sizeof(float));
      const float s_pos = 0.02f;             // 200 um
      const float s_ipt = 0.03f * par[3];    // 3% of 1/pT
      const float s_ang = 0.002f;            // 2 mrad
      c[ls_idx(0, 0)] = s_pos * s_pos;
      c[ls_idx(1, 1)] = s_pos * s_pos;
      c[ls_idx(2, 2)] = s_pos * s_pos;
      c[ls_idx(3, 3)] = s_ipt * s_ipt;
      c[ls_idx(4, 4)] = s_ang * s_ang;
      c[ls_idx(5, 5)] = s_ang * s_ang;
      c[ls_idx(0, 1)] = 0.2f * s_pos * s_pos;
      c[ls_idx(3, 4)] = 0.1f * s_ipt * s_ang;
      c[ls_idx(4, 5)] = 0.15f * s_ang * s_ang;
    }

    // ----------------------------------------------------------------------
    // The configurations. The two controls, scanned directly:
    //   dalpha  -- how far the outbound step turns  [rad]
    //   cos_inc -- how squarely it meets plane B
    // Nothing here is a radius. The radius reached is an output.
    const double kDalpha[]  = {0.02, 0.05, 0.10, 0.20, 0.40, 0.80};
    const double kCosInc[]  = {1.00, 0.70, 0.50, 0.30, 0.15};
    constexpr int kNDal = sizeof(kDalpha) / sizeof(kDalpha[0]);
    constexpr int kNInc = sizeof(kCosInc) / sizeof(kCosInc[0]);
    constexpr int kNCfg = kNDal * kNInc;

    // Tracker envelope. A configuration landing outside it is not measured:
    // it is not a propagation the detector ever asks for, and letting it in is
    // how plane "r = 30 cm" ended up at z = 8 m.
    constexpr double kMaxR = 120.0, kMaxZ = 300.0;

  }  // anonymous namespace

  //============================================================================

  void val_gen_closure(const char *out_file, int n_pt, int n_eta, int n_phi,
                       unsigned seed, float d0, float z0) {
    // Material OFF, uniform field: the propagation is then a diffeomorphism and
    // the exact closure residual is ZERO. Anything measured is float
    // arithmetic, or a defect.
    const PropagationFlags pf;  // all flags false

    const double kL0 = std::log10(0.3), kL1 = std::log10(200.0);
    const double dL = (kL1 - kL0) / n_pt;
    const double kE0 = -4.0, kE1 = 4.0;
    const double dE = (kE1 - kE0) / n_eta;
    const double dP = 2.0 * M_PI / n_phi;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> U(0.0, 1.0);

    TFile f(out_file, "RECREATE");
    TTree *t_cfg = new TTree("cfgs", "step configurations");
    ValCfgInfo ci;
    t_cfg->Branch("g", &ci);
    for (int ia = 0; ia < kNDal; ++ia)
      for (int ic = 0; ic < kNInc; ++ic) {
        ci.cfg = ia * kNInc + ic;
        ci.dalpha = (float)kDalpha[ia];
        ci.cos_inc = (float)kCosInc[ic];
        t_cfg->Fill();
      }

    TTree *t = new TTree("closure", "one-step-there-and-back propagation closure");
    ValClosure c;
    t->Branch("c", &c);

    int id = 0, track_id = 0;
    long n_try = 0, n_outside = 0;

    for (int ip = 0; ip < n_pt; ++ip)
      for (int ie = 0; ie < n_eta; ++ie)
        for (int iph = 0; iph < n_phi; ++iph)
          for (int iq = 0; iq < 2; ++iq, ++track_id) {
            // Jittered stratification: one draw inside each cell, so every cell
            // is populated but no two tracks share a pT, an eta or a phi.
            const double a_pt = std::pow(10.0, kL0 + (ip + U(rng)) * dL);
            const double eta = kE0 + (ie + U(rng)) * dE;
            const double phi0 = (iph + U(rng)) * dP;
            const int chg = iq ? +1 : -1;

            ValHelix h;
            h.init(a_pt, eta, chg, phi0, d0, z0);

            // PLANE A: at the production point, normal along the momentum there.
            // Incidence 1, so the return leg is as well conditioned as a plane
            // can be and adds no error of its own -- everything measured is a
            // property of the OUTBOUND step.
            double pa[3], ma[3];
            h.at_s(0.0, pa, ma);
            float par_o[6], cov_o[21];
            h.ccs_at_s(0.0, par_o);
            nominal_cov(par_o, cov_o);

            float pntA[3], nrmA[3];
            make_plane(pa, ma, 1.0, pntA, nrmA);

            for (int ia = 0; ia < kNDal; ++ia) {
              // B is placed at a chosen TURN ANGLE, and the radius it reaches
              // is an output. alpha runs with the sign of k, so fold it.
              const double alpha = (h.k < 0 ? -1.0 : 1.0) * kDalpha[ia];
              const double s_b = h.s_of_alpha(alpha);
              double pb[3], mb[3];
              h.at_s(s_b, pb, mb);
              const double r_b = std::hypot(pb[0], pb[1]);

              for (int ic = 0; ic < kNInc; ++ic) {
                ++n_try;
                // Outside the tracker envelope is not a propagation the
                // detector ever asks for.
                if (r_b > kMaxR || std::fabs(pb[2]) > kMaxZ) { ++n_outside; continue; }

                c = ValClosure{};
                c.id = id++;
                c.track_id = track_id;
                c.cfg = ia * kNInc + ic;
                c.pt = (float)a_pt; c.eta = (float)eta; c.phi0 = (float)phi0;
                c.d0 = d0; c.z0 = z0; c.charge = chg;
                c.dalpha_target = (float)kDalpha[ia];
                c.cosinc_target = (float)kCosInc[ic];
                c.r_a = (float)std::hypot(pa[0], pa[1]);  c.z_a = (float)pa[2];
                c.r_b = (float)r_b;                        c.z_b = (float)pb[2];
                c.s_ab = (float)s_b;
                c.dalpha_ab = (float)kDalpha[ia];

                for (int q = 0; q < 3; ++q) { c.pnt_a[q] = pntA[q]; c.nrm_a[q] = nrmA[q]; }
                make_plane(pb, mb, kCosInc[ic], c.pnt_b, c.nrm_b);
                // Keep the exact crossing in double -- see ValStructs.h.
                c.ref_pnt_b[0] = pb[0]; c.ref_pnt_b[1] = pb[1]; c.ref_pnt_b[2] = pb[2];
                c.ref_s_b = s_b;
                {
                  const double na = std::sqrt(ma[0]*ma[0] + ma[1]*ma[1] + ma[2]*ma[2]);
                  const double nb = std::sqrt(mb[0]*mb[0] + mb[1]*mb[1] + mb[2]*mb[2]);
                  c.cos_inc_a = (float)std::fabs((ma[0]*c.nrm_a[0] + ma[1]*c.nrm_a[1] + ma[2]*c.nrm_a[2]) / na);
                  c.cos_inc_b = (float)std::fabs((mb[0]*c.nrm_b[0] + mb[1]*c.nrm_b[1] + mb[2]*c.nrm_b[2]) / nb);
                }

                MPlexQI q_;  fill_lanes_q(q_, chg);
                MPlexHV mpA, mnA, mpB, mnB;
                fill_lanes_hv(mpA, c.pnt_a); fill_lanes_hv(mnA, c.nrm_a);
                fill_lanes_hv(mpB, c.pnt_b); fill_lanes_hv(mnB, c.nrm_b);

                MPlexLV p0;  fill_lanes(p0, par_o);
                MPlexLS e0;  fill_lanes_sym(e0, cov_o);

                // Leg O -> A. The state is already on plane A, so this is a
                // zero-length step; its job is to give the covariance the
                // rank-5 degeneracy the real code always carries. Needed for
                // the covariance closure, harmless for the position one.
                MPlexLS eA, eB, eC;  MPlexLV pA, pB_, pC;  MPlexQI fA{0}, fB{0}, fC{0};
                propagateHelixToPlaneMPlex(e0, p0, q_, mpA, mnA, nullptr, eA, pA, fA, NN, pf, nullptr);
                // Leg A -> B: the step under test.
                propagateHelixToPlaneMPlex(eA, pA, q_, mpB, mnB, nullptr, eB, pB_, fB, NN, pf, nullptr);
                // Leg B -> A': the return, onto the same well-conditioned plane A.
                propagateHelixToPlaneMPlex(eB, pB_, q_, mpA, mnA, nullptr, eC, pC, fC, NN, pf, nullptr);

                c.fail_oa = fA.constAt(0, 0, 0);
                c.fail_ab = fB.constAt(0, 0, 0);
                c.fail_ba = fC.constAt(0, 0, 0);
                read_lane0(pA, eA, c.par_a, c.err_a);
                read_lane0(pB_, eB, c.par_b, c.err_b);
                read_lane0(pC, eC, c.par_c, c.err_c);

                t->Fill();
              }
            }
          }

    t_cfg->Write();
    t->Write();
    f.Close();

    printf("val_gen_closure: %d tracks x %d cfgs = %ld trials, %ld outside the tracker\n"
           "                 envelope (r < %g, |z| < %g cm), %d written to '%s'\n",
           track_id, kNCfg, n_try, n_outside, kMaxR, kMaxZ, id, out_file);
    printf("  one step there and back: plane A at the production point with its normal ALONG\n"
           "  the momentum (incidence 1, so the return adds no error of its own); plane B at a\n"
           "  chosen TURN ANGLE with a chosen incidence. Radius is an output, not a control.\n");
    printf("  dalpha scan [rad] :");
    for (int i = 0; i < kNDal; ++i) printf(" %g", kDalpha[i]);
    printf("\n  incidence scan    :");
    for (int i = 0; i < kNInc; ++i) printf(" %g", kCosInc[i]);
    printf("\n  sample: log10 pT flat over [0.3, 200] GeV in %d cells, eta flat over [-4, +4] in\n"
           "          %d cells, phi in %d cells, both charges, one jittered draw per cell; seed %u\n",
           n_pt, n_eta, n_phi, seed);
    printf("  vertex: d0 = %g cm, z0 = %g cm;  material OFF, uniform B = %g T\n",
           d0, z0, Config::Bfield);
  }

  //============================================================================
  // What turn angles does a REAL track present between consecutive hits?
  //
  // A synthetic scan chooses the turn angle; a detector does not. Without this
  // distribution, a response measured against turn angle is being weighted
  // uniformly over a variable the experiment is not uniform in -- which is how
  // a scan can spend its statistics on a region that never occurs.
  //============================================================================

  namespace {
    std::vector<ValStep> g_val_steps;
    int g_val_evt = 0;
  }

  void val_dalpha_reset() { g_val_steps.clear(); g_val_evt = 0; }

  void val_dalpha_add_event(const Event *ev, int which, float pt_min) {
    if (ev == nullptr) { printf("val_dalpha_add_event: null event\n"); return; }
    const TrackVec &tv = (which == 1) ? ev->simTracks_
                       : (which == 2) ? ev->candidateTracks_
                                      : ev->cmsswTracks_;
    // R_c [cm] = pT / (0.01 * sol * B)
    const double inv_k = 0.01 * (double)Const::sol * Config::Bfield;
    const int evt = g_val_evt++;

    for (int it = 0; it < (int)tv.size(); ++it) {
      const Track &t = tv[it];
      if (t.pT() < pt_min) continue;
      const double Rc = t.pT() / inv_k;
      int prev_lay = -1, prev_idx = -1, step = 0;
      for (int ih = 0; ih < t.nTotalHits(); ++ih) {
        const int idx = t.getHitIdx(ih), lay = t.getHitLyr(ih);
        if (idx < 0 || lay < 0 || lay >= (int)ev->layerHits_.size()) continue;
        if (idx >= (int)ev->layerHits_[lay].size()) continue;
        if (prev_idx >= 0) {
          const Hit &a = ev->layerHits_[prev_lay][prev_idx];
          const Hit &b = ev->layerHits_[lay][idx];
          const double dx = b.x() - a.x(), dy = b.y() - a.y(), dz = b.z() - a.z();
          const double dperp = std::sqrt(dx * dx + dy * dy);
          ValStep vs;
          vs.event = evt; vs.track = it; vs.step = step++;
          vs.lay_a = prev_lay; vs.lay_b = lay;
          vs.pt = t.pT(); vs.eta = t.momEta();
          vs.r_a = (float)std::hypot(a.x(), a.y()); vs.z_a = a.z();
          vs.r_b = (float)std::hypot(b.x(), b.y()); vs.z_b = b.z();
          vs.d_perp = (float)dperp;
          vs.d_3d = (float)std::sqrt(dx * dx + dy * dy + dz * dz);
          // chord -> turn angle. Beyond 2*R_c the two hits cannot be on one
          // circle of this radius; clamp and let the analysis see it.
          vs.dalpha = (float)(2.0 * std::asin(std::min(1.0, dperp / (2.0 * Rc))));
          g_val_steps.push_back(vs);
        }
        prev_lay = lay; prev_idx = idx;
      }
    }
  }

  void val_dalpha_write(const char *out_file) {
    TFile f(out_file, "RECREATE");
    TTree *t = new TTree("steps", "turn angle between consecutive hits of real tracks");
    ValStep vs;
    t->Branch("s", &vs);
    for (const ValStep &v : g_val_steps) { vs = v; t->Fill(); }
    t->Write();
    f.Close();
    printf("val_dalpha_write: %zu steps from %d events -> '%s'\n",
           g_val_steps.size(), g_val_evt, out_file);
  }

  //============================================================================
  // Geometry compatibility of a sample -- see the header.
  //============================================================================

  namespace {
    std::vector<double> g_gc_d;
    long g_gc_bad_sid = 0, g_gc_hits = 0, g_gc_evt = 0;
  }

  void val_geom_check_reset() {
    g_gc_d.clear(); g_gc_bad_sid = 0; g_gc_hits = 0; g_gc_evt = 0;
  }

  void val_geom_check_event(const Event *ev) {
    if (ev == nullptr) { printf("val_geom_check_event: null event\n"); return; }
    const TrackerInfo &ti = Config::TrkInfo;
    ++g_gc_evt;
    for (int lay = 0; lay < (int)ev->layerHits_.size(); ++lay) {
      if (lay >= ti.n_layers()) break;
      const LayerInfo &li = ti.layer(lay);
      for (const Hit &h : ev->layerHits_[lay]) {
        ++g_gc_hits;
        const unsigned int sid = h.detIDinLayer();
        if ((int)sid >= li.n_modules()) { ++g_gc_bad_sid; continue; }
        const ModuleInfo &mi = li.module_info(sid);
        const double d = std::fabs((h.x() - mi.pos[0]) * mi.zdir[0] +
                                   (h.y() - mi.pos[1]) * mi.zdir[1] +
                                   (h.z() - mi.pos[2]) * mi.zdir[2]);
        g_gc_d.push_back(d);
      }
    }
  }

  void val_geom_check_report(const char *sample_name) {
    printf("\n================ geometry compatibility ================\n");
    printf("sample : %s\n", sample_name);
    printf("events : %ld,  hits : %ld\n", g_gc_evt, g_gc_hits);
    printf("module short_id out of range : %ld  (%.4f %%)\n",
           g_gc_bad_sid, g_gc_hits ? 100.0 * g_gc_bad_sid / g_gc_hits : 0.0);
    if (g_gc_d.empty()) { printf("no usable hits\n"); return; }
    std::sort(g_gc_d.begin(), g_gc_d.end());
    auto q = [&](double p) { return g_gc_d[(size_t)(0.01 * p * (g_gc_d.size() - 1))]; };
    printf("|n.(hit - module plane point)|, cm -- a hit is ON its module by construction\n");
    printf("  median %10.4g   p90 %10.4g   p99 %10.4g   max %10.4g\n",
           q(50), q(90), q(99), g_gc_d.back());
    long over = 0; for (double x : g_gc_d) if (x > 0.01) ++over;
    printf("  fraction over 100 um : %.4f %%\n", 100.0 * over / g_gc_d.size());
    printf("\nreference: 2.9e-9 cm median = COMPATIBLE;  0.26 cm median = NOT\n");
    printf("=======================================================\n");
  }

  void val_sample_info(const Event *ev) {
    if (ev == nullptr) { printf("val_sample_info: null event\n"); return; }
    auto describe = [](const char *nm, const TrackVec &tv) {
      if (tv.empty()) { printf("  %-18s empty\n", nm); return; }
      std::map<int, int> algo;
      double ptmin = 1e9, ptmax = -1e9;
      long nh = 0, nlab = 0;
      for (const Track &t : tv) {
        algo[t.algoint()]++;
        ptmin = std::min(ptmin, (double)t.pT());
        ptmax = std::max(ptmax, (double)t.pT());
        nh += t.nFoundHits();
        if (t.label() >= 0) ++nlab;
      }
      printf("  %-18s n=%-7zu  pT %.3g .. %.4g  <nhits> %.1f  labelled %.0f%%  algos:",
             nm, tv.size(), ptmin, ptmax, (double)nh / tv.size(), 100.0 * nlab / tv.size());
      for (auto &a : algo) printf(" %d:%d", a.first, a.second);
      printf("\n");
    };
    printf("\n---- sample contents, one event ----\n");
    printf("  %-18s %zu layers\n", "layerHits_", ev->layerHits_.size());
    long nh = 0; for (auto &v : ev->layerHits_) nh += v.size();
    printf("  %-18s %ld hits\n", "total hits", nh);
    printf("  %-18s %zu\n", "simHitsInfo_", ev->simHitsInfo_.size());
    printf("  %-18s %zu   <-- if this equals the number of SIM HITS there is a\n",
           "simTrackStates_", ev->simTrackStates_.size());
    printf("  %-18s     true state at every hit, and the bias measurement needs no fit\n", "");
    {  // how many hits actually carry usable MC info, and do labels line up?
      long with_mc = 0, foreign = 0, checked = 0;
      for (int lay = 0; lay < (int)ev->layerHits_.size(); ++lay)
        for (const Hit &h : ev->layerHits_[lay]) {
          const int mid = h.mcHitID();
          if (mid >= 0 && mid < (int)ev->simHitsInfo_.size()) ++with_mc;
        }
      for (const Track &t : ev->simTracks_) {
        for (int ih = 0; ih < t.nTotalHits(); ++ih) {
          const int idx = t.getHitIdx(ih), lay = t.getHitLyr(ih);
          if (idx < 0 || lay < 0 || lay >= (int)ev->layerHits_.size()) continue;
          if (idx >= (int)ev->layerHits_[lay].size()) continue;
          const Hit &h = ev->layerHits_[lay][idx];
          const int mid = h.mcHitID();
          if (mid < 0 || mid >= (int)ev->simHitsInfo_.size()) continue;
          ++checked;
          if (ev->simHitsInfo_[mid].mcTrackID() != t.label()) ++foreign;
        }
      }
      printf("  %-18s %ld of %ld hits carry MC info\n", "mc coverage", with_mc, nh);
      printf("  %-18s %ld of %ld hits ON a sim track belong to a DIFFERENT particle"
             " (%.2f%%)\n", "foreign hits", foreign, checked,
             checked ? 100.0 * foreign / checked : 0.0);
    }
    describe("simTracks_", ev->simTracks_);
    describe("seedTracks_", ev->seedTracks_);
    describe("cmsswTracks_", ev->cmsswTracks_);
  }

  void val_simtrack_anchor(const Event *ev, int n_print) {
    if (ev == nullptr) { printf("val_simtrack_anchor: null event\n"); return; }
    std::vector<double> d_first, d_orig;
    int printed = 0;
    for (const Track &t : ev->simTracks_) {
      if (t.nFoundHits() < 3) continue;
      // first hit in the stored order
      int lay = -1, idx = -1;
      for (int ih = 0; ih < t.nTotalHits(); ++ih) {
        if (t.getHitIdx(ih) >= 0) { lay = t.getHitLyr(ih); idx = t.getHitIdx(ih); break; }
      }
      if (idx < 0 || lay < 0 || lay >= (int)ev->layerHits_.size()) continue;
      if (idx >= (int)ev->layerHits_[lay].size()) continue;
      const Hit &h = ev->layerHits_[lay][idx];
      const double dx = t.x() - h.x(), dy = t.y() - h.y(), dz = t.z() - h.z();
      d_first.push_back(std::sqrt(dx*dx + dy*dy + dz*dz));
      d_orig.push_back(std::sqrt((double)t.x()*t.x() + (double)t.y()*t.y() + (double)t.z()*t.z()));
      if (printed < n_print) {
        printf("  trk %-6d pT %7.3g  state (%9.4f %9.4f %9.4f)  first hit lay %2d "
               "(%9.4f %9.4f %9.4f)  |d| %.4g\n",
               t.label(), t.pT(), t.x(), t.y(), t.z(), lay, h.x(), h.y(), h.z(),
               d_first.back());
        ++printed;
      }
    }
    auto rep = [](const char *nm, std::vector<double> &v) {
      if (v.empty()) { printf("  %-28s none\n", nm); return; }
      std::sort(v.begin(), v.end());
      auto q=[&](double p){ return v[(size_t)(0.01*p*(v.size()-1))]; };
      printf("  %-28s n=%-6zu med %10.4g  p90 %10.4g  p99 %10.4g  max %10.4g\n",
             nm, v.size(), q(50), q(90), q(99), v.back());
    };
    printf("\n---- where are sim-track parameters given? ----\n");
    rep("|state - FIRST HIT|  [cm]", d_first);
    rep("|state - ORIGIN|     [cm]", d_orig);
    printf("  (whichever is ~0 is the anchor)\n");
  }

  void val_seed_vs_cmssw(const Event *ev) {
    if (ev == nullptr) { printf("val_seed_vs_cmssw: null event\n"); return; }
    // Map every hit to the seeds that use it, then for each cmssw track find
    // the seed it shares most hits with. No labels involved: a label can mean
    // different things in two collections, a hit index cannot.
    std::map<long, std::vector<int>> hit_to_seed;
    auto key = [](int lay, int idx) { return (long)lay * 10000000L + idx; };
    std::vector<int> seed_nh(ev->seedTracks_.size(), 0);
    for (int si = 0; si < (int)ev->seedTracks_.size(); ++si) {
      const Track &s = ev->seedTracks_[si];
      for (int ih = 0; ih < s.nTotalHits(); ++ih) {
        const int idx = s.getHitIdx(ih), lay = s.getHitLyr(ih);
        if (idx >= 0 && lay >= 0) { hit_to_seed[key(lay, idx)].push_back(si); ++seed_nh[si]; }
      }
    }
    std::vector<double> best_frac;
    long n_any = 0;
    for (const Track &t : ev->cmsswTracks_) {
      std::map<int, int> common;
      int nh = 0;
      for (int ih = 0; ih < t.nTotalHits(); ++ih) {
        const int idx = t.getHitIdx(ih), lay = t.getHitLyr(ih);
        if (idx < 0 || lay < 0) continue;
        ++nh;
        auto it = hit_to_seed.find(key(lay, idx));
        if (it != hit_to_seed.end()) for (int si : it->second) common[si]++;
      }
      int bn = 0, bs = -1;
      for (auto &c : common) if (c.second > bn) { bn = c.second; bs = c.first; }
      if (bn > 0) ++n_any;
      best_frac.push_back(bs >= 0 && seed_nh[bs] > 0 ? (double)bn / seed_nh[bs] : 0.0);
    }
    printf("\n---- are cmsswTracks_ built from the seeds in this file? (by HITS) ----\n");
    printf("  seeds %zu (<hits> %.1f) , cmssw tracks %zu\n",
           ev->seedTracks_.size(),
           ev->seedTracks_.empty() ? 0.0 :
             (double)std::accumulate(seed_nh.begin(), seed_nh.end(), 0) / seed_nh.size(),
           ev->cmsswTracks_.size());
    printf("  cmssw tracks sharing ANY hit with ANY seed : %ld  (%.1f%%)\n",
           n_any, best_frac.empty() ? 0.0 : 100.0 * n_any / best_frac.size());
    if (!best_frac.empty()) {
      std::sort(best_frac.begin(), best_frac.end());
      auto q=[&](double p){ return best_frac[(size_t)(0.01*p*(best_frac.size()-1))]; };
      printf("  fraction of the best-matching SEED's hits present on the track:\n");
      printf("    p10 %.3f   median %.3f   p90 %.3f   max %.3f\n",
             q(10), q(50), q(90), best_frac.back());
      long full = 0; for (double f : best_frac) if (f > 0.999) ++full;
      printf("  tracks containing a WHOLE seed : %ld  (%.1f%%)\n",
             full, 100.0 * full / best_frac.size());
    }
    printf("  -> near-total containment means the collection is DOWNSTREAM of these\n"
           "     seeds and is not an independent reference.\n");
  }

  //============================================================================
  // Seed diagnostics, before any fit.
  //============================================================================

  void val_seed_cov(const Event *ev, int kind) {
    if (ev == nullptr) { printf("val_seed_cov: null event\n"); return; }
    const TrackerInfo &ti = Config::TrkInfo;

    auto first_last = [&](const Track &t, int &fl, int &fi, int &ll, int &li) {
      fl = ll = -1;
      for (int ih = 0; ih < t.nTotalHits(); ++ih) {
        if (t.getHitIdx(ih) < 0) continue;
        if (fl < 0) { fl = t.getHitLyr(ih); fi = t.getHitIdx(ih); }
        ll = t.getHitLyr(ih); li = t.getHitIdx(ih);
      }
    };
    // Eigenvalues of the symmetric 6x6 by cyclic Jacobi. Cholesky alone cannot
    // tell "rank-5 because the state lies on a surface" -- which is EXPECTED
    // for a propagate-to-surface output -- from a genuinely indefinite matrix.
    // The discriminator is the sign and size of the smallest eigenvalue
    // relative to the largest.
    auto eig6 = [](const SMatrixSym66 &C, double ev[6]) {
      double a[6][6];
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) a[i][j] = C(i, j);
      for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0;
        for (int i = 0; i < 6; ++i) for (int j = i + 1; j < 6; ++j) off += a[i][j] * a[i][j];
        if (off < 1e-30) break;
        for (int p2 = 0; p2 < 6; ++p2)
          for (int q2 = p2 + 1; q2 < 6; ++q2) {
            if (std::fabs(a[p2][q2]) < 1e-300) continue;
            const double theta = (a[q2][q2] - a[p2][p2]) / (2 * a[p2][q2]);
            const double t = (theta >= 0 ? 1.0 : -1.0) /
                             (std::fabs(theta) + std::sqrt(theta * theta + 1));
            const double c = 1 / std::sqrt(t * t + 1), sn = t * c;
            for (int k = 0; k < 6; ++k) {
              const double akp = a[k][p2], akq = a[k][q2];
              a[k][p2] = c * akp - sn * akq;
              a[k][q2] = sn * akp + c * akq;
            }
            for (int k = 0; k < 6; ++k) {
              const double apk = a[p2][k], aqk = a[q2][k];
              a[p2][k] = c * apk - sn * aqk;
              a[q2][k] = sn * apk + c * aqk;
            }
          }
      }
      for (int i = 0; i < 6; ++i) ev[i] = a[i][i];
      std::sort(ev, ev + 6);
    };
    // Cholesky: succeeds iff the matrix is positive definite.
    auto is_pd = [](const SMatrixSym66 &C) {
      double L[6][6] = {};
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) {
          double sum = C(i, j);
          for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
          if (i == j) { if (sum <= 0) return false; L[i][i] = std::sqrt(sum); }
          else L[i][j] = sum / L[j][j];
        }
      return true;
    };

    std::vector<double> sx, sz, sipt_rel, sphi, sth, d_first, d_last, nh, cond;
    std::vector<double> g_near_d, g_near_from_end, g_off_x, g_off_y, g_off_z, g_lastpair;
    long n_kind = 0, n_notpd = 0, n_neg = 0, n_rank5 = 0;
    int n_pT5 = 0, n_T5 = 0, n_pix = 0, n_other = 0;

    for (const Track &t : ev->seedTracks_) {
      int fl, fi, ll, li;
      first_last(t, fl, fi, ll, li);
      if (fl < 0 || ll < 0) continue;
      const bool f_pix = ti.layer(fl).is_pixel();
      const bool l_strp = !ti.layer(ll).is_pixel();
      int k = -1;
      if      ( f_pix &&  l_strp) { k = 0; ++n_pT5; }
      else if (!f_pix &&  l_strp) { k = 1; ++n_T5; }
      else if ( f_pix && !l_strp) { k = 2; ++n_pix; }
      else ++n_other;
      if (kind >= 0 && k != kind) continue;
      ++n_kind;

      const SMatrixSym66 &C = t.errors();
      if (!is_pd(C)) ++n_notpd;
      double ev6[6];
      eig6(C, ev6);
      const double lmax = ev6[5], lmin = ev6[0];
      cond.push_back(lmax > 0 ? lmin / lmax : 0.0);
      // The covariance is stored in float32, ~6e-8 relative precision, and a
      // covariance squares the scales -- so anything within ~1e-6 of the
      // largest eigenvalue is numerically ZERO and its sign is meaningless.
      // Only a clearly negative eigenvalue is a defect.
      if (lmin < -1e-6 * lmax) ++n_neg;             // genuinely indefinite
      else if (lmin < 1e-6 * lmax) ++n_rank5;       // degenerate: rank 5
      sx.push_back(std::sqrt(std::max(0.0f, C(0, 0))));
      sz.push_back(std::sqrt(std::max(0.0f, C(2, 2))));
      sipt_rel.push_back(std::sqrt(std::max(0.0f, C(3, 3))) * t.pT());
      sphi.push_back(std::sqrt(std::max(0.0f, C(4, 4))));
      sth.push_back(std::sqrt(std::max(0.0f, C(5, 5))));
      nh.push_back(t.nFoundHits());

      // Which hit is the state actually ON, and in what direction is the offset?
      // If the seed state sits on one sensor of a 2S pair and the LAST hit by
      // layer number is its sister, the offset should lie almost entirely along
      // the module's ydir -- the strip direction, which a parallel pair never
      // measures, so each hit is placed at its own strip centre.
      {
        int best_ih = -1, n_real = 0, best_from_end = -1;
        double best_d = 1e30;
        std::vector<std::pair<int,int>> real;   // (layer, idx) in stored order
        for (int ih = 0; ih < t.nTotalHits(); ++ih)
          if (t.getHitIdx(ih) >= 0) real.emplace_back(t.getHitLyr(ih), t.getHitIdx(ih));
        n_real = real.size();
        for (int k = 0; k < n_real; ++k) {
          const int lay = real[k].first, idx = real[k].second;
          if (lay >= (int)ev->layerHits_.size() || idx >= (int)ev->layerHits_[lay].size()) continue;
          const Hit &h = ev->layerHits_[lay][idx];
          const double dx = t.x()-h.x(), dy = t.y()-h.y(), dz = t.z()-h.z();
          const double d = std::sqrt(dx*dx+dy*dy+dz*dz);
          if (d < best_d) { best_d = d; best_ih = k; best_from_end = n_real-1-k; }
        }
        if (best_ih >= 0) {
          g_near_d.push_back(best_d);
          g_near_from_end.push_back(best_from_end);
          // decompose the offset to the LAST hit in the last hit's module frame
          if (ll < (int)ev->layerHits_.size() && li < (int)ev->layerHits_[ll].size()) {
            const Hit &h = ev->layerHits_[ll][li];
            const ModuleInfo &mi = ti.layer(ll).module_info(h.detIDinLayer());
            const double d3[3] = {t.x()-h.x(), t.y()-h.y(), t.z()-h.z()};
            const double ax = d3[0]*mi.xdir[0]+d3[1]*mi.xdir[1]+d3[2]*mi.xdir[2];
            const SVector3 yd = mi.calc_ydir();
            const double ay = d3[0]*yd[0]+d3[1]*yd[1]+d3[2]*yd[2];
            const double az = d3[0]*mi.zdir[0]+d3[1]*mi.zdir[1]+d3[2]*mi.zdir[2];
            g_off_x.push_back(std::fabs(ax));
            g_off_y.push_back(std::fabs(ay));
            g_off_z.push_back(std::fabs(az));
          }
          // are the last two hits in adjacent layers (a split-layer pair)?
          if (n_real >= 2) g_lastpair.push_back(real[n_real-1].first - real[n_real-2].first);
        }
      }
      auto dist = [&](int lay, int idx) {
        const Hit &h = ev->layerHits_[lay][idx];
        const double dx = t.x() - h.x(), dy = t.y() - h.y(), dz = t.z() - h.z();
        return std::sqrt(dx * dx + dy * dy + dz * dz);
      };
      if (fl < (int)ev->layerHits_.size() && fi < (int)ev->layerHits_[fl].size())
        d_first.push_back(dist(fl, fi));
      if (ll < (int)ev->layerHits_.size() && li < (int)ev->layerHits_[ll].size())
        d_last.push_back(dist(ll, li));
    }

    auto rep = [](const char *nm, std::vector<double> v, const char *unit) {
      if (v.empty()) { printf("  %-26s none\n", nm); return; }
      std::sort(v.begin(), v.end());
      auto q = [&](double p) { return v[(size_t)(0.01 * p * (v.size() - 1))]; };
      // spread/median: ~0 means every seed carries the SAME value, i.e. canned.
      const double spread = q(50) != 0 ? (q(84) - q(16)) / q(50) : 0;
      printf("  %-26s p16 %10.4g  med %10.4g  p84 %10.4g   spread/med %7.4f  %s\n",
             nm, q(16), q(50), q(84), spread, unit);
    };

    printf("\n---- seed diagnostics (kind %d: 0=pT5 1=T5 2=pix, -1=all) ----\n", kind);
    printf("  seeds in event: pT5 %d, T5 %d, pix %d, other %d ; selected %ld\n",
           n_pT5, n_T5, n_pix, n_other, n_kind);
    printf("  Cholesky fails        : %ld  (%.2f%%)\n",
           n_notpd, n_kind ? 100.0 * n_notpd / n_kind : 0.0);
    printf("  of which, by eigenvalue:\n");
    printf("    genuinely INDEFINITE (lmin < -1e-6 lmax) : %ld  (%.2f%%)  <-- broken\n",
           n_neg, n_kind ? 100.0 * n_neg / n_kind : 0.0);
    printf("    rank-5, lmin ~ 0 to float precision       : %ld  (%.2f%%)  <-- a surface constraint\n",
           n_rank5, n_kind ? 100.0 * n_rank5 / n_kind : 0.0);
    rep("lambda_min / lambda_max", cond, "  (negative = indefinite)");
    rep("n found hits", nh, "");
    rep("sigma_x", sx, "cm");
    rep("sigma_z", sz, "cm");
    rep("sigma(1/pT)/(1/pT)", sipt_rel, "");
    rep("sigma_phi", sphi, "rad");
    rep("sigma_theta", sth, "rad");
    printf("  --- where is the state anchored? (one of these should be ~0) ---\n");
    rep("|state - FIRST hit|", d_first, "cm");
    rep("|state - LAST hit|", d_last, "cm");
    rep("|state - NEAREST hit|", g_near_d, "cm");
    rep("nearest hit, index from end", g_near_from_end, " (0 = the last one)");
    printf("  --- offset to the LAST hit, in that hit's MODULE frame ---\n");
    rep("  along xdir (across strip)", g_off_x, "cm  <- the precise direction");
    rep("  along ydir (along strip)", g_off_y, "cm  <- never measured by a parallel pair");
    rep("  along zdir (module normal)", g_off_z, "cm");
    rep("layer(last) - layer(last-1)", g_lastpair, " (1 = adjacent, i.e. a split-layer pair)");
    printf("  a spread/med of ~0 means the covariance is CANNED, not a fit result.\n");
  }


  // ==========================================================================
  // S11 step 1 -- ENSEMBLE COVARIANCE TRANSPORT, ONE STEP.
  //
  // Draw N samples from C_A, push each through the production propagator, and
  // compare their empirical covariance S against the ONE covariance C_B that
  // the same propagator produces from C_A. No truth, no material, no MC: the
  // only thing under test is the jacobian.
  //
  // Two things this sees that nothing measured so far could:
  //  - a wrong CORRELATION. Six marginal ratios are blind to it; the
  //    eigenvalues of the whitened empirical covariance are not.
  //  - a wrong RANK. Propagating to a surface is a rank-5 operation in 6
  //    dimensions, and a missing surface-crossing term is exactly a rank/null
  //    space defect.
  //
  // The control is `cov_scale`: correct LINEAR transport is scale-invariant, so
  // a scale dependence separates "the jacobian is wrong" from "the step is too
  // nonlinear for any jacobian".
  void val_cov_transport(const char *out_file, int n_samp, unsigned seed) {
    const double kPt[]   = {1.0, 10.0};
    // Extended to 2.5 on 2026-09-17: the measured q under-estimate lives at
    // |eta| > 1.6 and this test stopped at 1.5, so it had never probed the
    // region where the covariance is claimed to be wrong.
    const double kEta[]  = {0.0, 1.5, 2.0, 2.5};
    const double kDal[]  = {0.02, 0.10, 0.40, 0.80, 1.50};
    const double kInc[]  = {1.00, 0.50};
    const float  kScale[] = {1.0f, 100.0f};

    PropagationFlags pf;   // no material, uniform B -- transport only
    std::vector<ValCovXport> out;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> gauss(0.0, 1.0);

    int cfg = 0;
    for (double pt : kPt) for (double eta : kEta)
    for (double dal : kDal) for (double inc : kInc) for (float scale : kScale) {
      ValCovXport v;
      v.cfg = cfg++;  v.pt = (float)pt;  v.eta = (float)eta;
      v.dalpha = (float)dal;  v.cos_inc = (float)inc;  v.cov_scale = scale;

      ValHelix h;  h.init(pt, eta, -1, 0.3, 0.0, 0.0);

      // State A: the helix's first crossing of r = 30 cm. Plane B: dalpha of
      // turn further along, tilted to the requested incidence.
      double sA = 0.0;
      if (!h.first_crossing_r(30.0, sA)) continue;
      const double sB = h.s_of_alpha(h.alpha_of_s(sA) + dal);
      double pB3[3], mB3[3];  h.at_s(sB, pB3, mB3);
      if (std::hypot(pB3[0], pB3[1]) > kMaxR || std::fabs(pB3[2]) > kMaxZ) continue;
      float plp[3], pln[3];   make_plane(pB3, mB3, inc, plp, pln);

      float parA[6], covA[21];
      h.ccs_at_s(sA, parA);
      nominal_cov(parA, covA);
      for (int i = 0; i < 21; ++i) covA[i] *= scale;

      // --- the reference: one propagation of the mean state with C_A.
      MPlexLV p0, pB;  MPlexLS e0, eB;  MPlexQI q_, fB;  MPlexHV mp, mn;
      fill_lanes(p0, parA);  fill_lanes_sym(e0, covA);
      fill_lanes_q(q_, h.chg);  fill_lanes_hv(mp, plp);  fill_lanes_hv(mn, pln);
      propagateHelixToPlaneMPlex(e0, p0, q_, mp, mn, nullptr, eB, pB, fB, NN, pf, nullptr);
      if (fB.constAt(0, 0, 0) != 0) { out.push_back(v); continue; }
      float parB[6], covB[21];
      read_lane0(pB, eB, parB, covB);
      {
        const double st = std::sin((double)parB[5]);
        const double ph[3] = {st*std::cos((double)parB[4]), st*std::sin((double)parB[4]),
                              std::cos((double)parB[5])};
        v.cos_inc_meas = (float)std::fabs(ph[0]*pln[0] + ph[1]*pln[1] + ph[2]*pln[2]);
        double cs = 0; for (int i = 0; i < 21; ++i) cs += std::fabs((double)covB[i]);
        v.covb_sum = (float)cs;
      }
      v.d_plane_ref = (float)std::fabs((double)(parB[0]-plp[0])*pln[0] +
                                       (double)(parB[1]-plp[1])*pln[1] +
                                       (double)(parB[2]-plp[2])*pln[2]);

      // --- Cholesky of C_A, in double. C_A is constructed PD, so a failure
      //     here is a bug in nominal_cov, not a result.
      double L[6][6] = {};
      {
        double A[6][6];
        for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) A[i][j] = covA[ls_idx(i, j)];
        bool ok = true;
        for (int i = 0; i < 6 && ok; ++i) {
          for (int j = 0; j <= i; ++j) {
            double sum = A[i][j];
            for (int k = 0; k < j; ++k) sum -= L[i][k] * L[j][k];
            if (i == j) { if (sum <= 0) { ok = false; break; } L[i][i] = std::sqrt(sum); }
            else L[i][j] = sum / L[j][j];
          }
        }
        if (!ok) { printf("val_cov_transport: cfg %d C_A not PD\n", v.cfg); out.push_back(v); continue; }
      }

      // --- the ensemble. NN samples per propagator call; the covariance
      //     argument is transported too and simply not read.
      std::vector<std::array<double,6>> acc, accv;
      std::vector<double> dpl;
      acc.reserve(n_samp);  accv.reserve(n_samp);  dpl.reserve(n_samp);
      double pref[3];
      { const double st = std::sin((double)parB[5]);
        pref[0] = st*std::cos((double)parB[4]); pref[1] = st*std::sin((double)parB[4]);
        pref[2] = std::cos((double)parB[5]); }
      int n_fail = 0;
      for (int base = 0; base < n_samp; base += NN) {
        const int nlane = std::min(NN, n_samp - base);
        MPlexLV ps;  MPlexLS es;  MPlexQI fs;  MPlexLV po;  MPlexLS eo;
        fill_lanes_sym(es, covA);
        float smp[NN][6];
        for (int n = 0; n < NN; ++n) {
          double z[6];
          for (int i = 0; i < 6; ++i) z[i] = gauss(rng);
          for (int i = 0; i < 6; ++i) {
            double d = 0.0;
            for (int k = 0; k <= i; ++k) d += L[i][k] * z[k];
            smp[n][i] = (float)(parA[i] + d);
          }
          for (int i = 0; i < 6; ++i) ps(n, i, 0) = smp[n][i];
        }
        propagateHelixToPlaneMPlex(es, ps, q_, mp, mn, nullptr, eo, po, fs, NN, pf, nullptr);
        for (int n = 0; n < nlane; ++n) {
          if (fs.constAt(n, 0, 0) != 0) { ++n_fail; continue; }
          std::array<double,6> a;
          for (int i = 0; i < 6; ++i) a[i] = po.constAt(n, i, 0);
          dpl.push_back(std::fabs((a[0]-plp[0])*pln[0] + (a[1]-plp[1])*pln[1] + (a[2]-plp[2])*pln[2]));
          // phi is squashed into (-pi, pi] by the propagator; unwrap it against
          // the reference before averaging, or a track near the branch cut
          // manufactures a variance of order pi^2.
          double dphi = a[4] - parB[4];
          while (dphi >  M_PI) dphi -= 2.0 * M_PI;
          while (dphi <= -M_PI) dphi += 2.0 * M_PI;
          a[4] = parB[4] + dphi;
          acc.push_back(a);
          // Curvilinear projection: slide the sample along ITS OWN momentum
          // until it reaches the plane through the reference point normal to
          // the REFERENCE momentum. Momentum is unchanged by the slide, so only
          // the three position components move.
          {
            const double st = std::sin(a[5]);
            const double ph[3] = {st*std::cos(a[4]), st*std::sin(a[4]), std::cos(a[5])};
            const double den = ph[0]*pref[0] + ph[1]*pref[1] + ph[2]*pref[2];
            std::array<double,6> b = a;
            if (std::fabs(den) > 1e-6) {
              const double tt = (((double)parB[0]-a[0])*pref[0] +
                                 ((double)parB[1]-a[1])*pref[1] +
                                 ((double)parB[2]-a[2])*pref[2]) / den;
              for (int i = 0; i < 3; ++i) b[i] = a[i] + tt*ph[i];
            }
            accv.push_back(b);
          }
        }
      }
      v.n_samp = (int)acc.size();
      v.n_fail = n_fail;
      if (!dpl.empty()) {
        std::sort(dpl.begin(), dpl.end());
        v.d_plane_med = (float)dpl[dpl.size()/2];
        v.d_plane_p90 = (float)dpl[(size_t)(0.90*(dpl.size()-1))];
        v.d_plane_max = (float)dpl.back();
      }
      if (v.n_samp < 100) { out.push_back(v); continue; }

      // --- empirical mean and covariance
      double mean[6] = {};
      for (auto &a : acc) for (int i = 0; i < 6; ++i) mean[i] += a[i];
      for (int i = 0; i < 6; ++i) mean[i] /= v.n_samp;
      double S[6][6] = {};
      for (auto &a : acc)
        for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
          S[i][j] += (a[i] - mean[i]) * (a[j] - mean[j]);
      for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) S[i][j] /= (v.n_samp - 1);

      for (int i = 0; i < 6; ++i) {
        const double cb = covB[ls_idx(i, i)];
        v.ratio[i]     = cb > 0 ? (float)std::sqrt(S[i][i] / cb) : -1.f;
        v.mean_pull[i] = cb > 0 ? (float)((mean[i] - parB[i]) / std::sqrt(cb)) : 0.f;
      }

      // --- whiten inside C_B's column space.
      //
      // NON-DIMENSIONALISE FIRST. C_B mixes cm^2 (position), rad^2 (two angles)
      // and GeV^-2 (1/pT), so its raw eigenvalues are not comparable to one
      // another and "lambda > eps * lambda_max" is not a rank test at all -- it
      // is a statement about the units the state happens to be written in.
      // Scaling each coordinate by its own sigma gives the CORRELATION matrix:
      // dimensionless, unit diagonal, eigenvalues summing to 6. The rank gap is
      // then a real gap. (Getting this wrong makes the whitening divide by
      // float32 dust and reports lam_max of 1e9 -- a measurement artefact, not
      // a propagator defect.)
      double sc[6];
      for (int i = 0; i < 6; ++i) {
        const double cb = covB[ls_idx(i, i)];
        sc[i] = cb > 0 ? 1.0 / std::sqrt(cb) : 0.0;
      }
      TMatrixDSym R(6), Sn(6);
      for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        R(i, j)  = covB[ls_idx(i, j)] * sc[i] * sc[j];
        Sn(i, j) = S[i][j]            * sc[i] * sc[j];
      }
      TMatrixDSymEigen eig(R);
      TVectorD d = eig.GetEigenValues();     // descending, dimensionless
      TMatrixD U = eig.GetEigenVectors();
      for (int i = 0; i < 6; ++i) v.cb_spec[i] = (float)d[i];
      // C_B is held in float32, so a structurally-zero eigenvalue comes back at
      // ~1e-7 of the unit scale. 1e-5 sits two decades above that dust and many
      // decades below any real direction.
      int rank = 0;
      while (rank < 6 && d[rank] > 1e-5) ++rank;
      v.rank_cb = rank;
      if (rank == 0) { out.push_back(v); continue; }

      TMatrixD W(rank, 6);
      for (int r = 0; r < rank; ++r) {
        const double f = 1.0 / std::sqrt(d[r]);
        for (int i = 0; i < 6; ++i) W(r, i) = f * U(i, r);
      }
      TMatrixD WS(W, TMatrixD::kMult, Sn);
      TMatrixD Wt(TMatrixD::kTransposed, W);
      TMatrixD Z(WS, TMatrixD::kMult, Wt);
      TMatrixDSym Zs(rank);
      for (int i = 0; i < rank; ++i) for (int j = 0; j < rank; ++j) Zs(i, j) = 0.5 * (Z(i, j) + Z(j, i));
      TMatrixDSymEigen ze(Zs);
      TVectorD zl = ze.GetEigenValues();
      v.lam_max = (float)zl[0];  v.lam_min = (float)zl[rank - 1];
      for (int i = 0; i < rank && i < 6; ++i) v.lam[i] = (float)zl[i];

      // --- the same comparison on the curvilinear surface. Off normal
      //     incidence this is the only fair one; at cos_inc = 1 it must agree
      //     with the block above, which is its own cross-check.
      {
        double mv[6] = {};
        for (auto &a : accv) for (int i = 0; i < 6; ++i) mv[i] += a[i];
        for (int i = 0; i < 6; ++i) mv[i] /= accv.size();
        TMatrixDSym Sv(6);
        for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) Sv(i, j) = 0.0;
        for (auto &a : accv)
          for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
            Sv(i, j) += (a[i]-mv[i])*(a[j]-mv[j]);
        for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j)
          Sv(i, j) *= sc[i]*sc[j] / (accv.size()-1);
        TMatrixD Zv(TMatrixD(W, TMatrixD::kMult, Sv), TMatrixD::kMult, Wt);
        TMatrixDSym Zvs(rank);
        for (int i = 0; i < rank; ++i) for (int j = 0; j < rank; ++j)
          Zvs(i, j) = 0.5*(Zv(i, j) + Zv(j, i));
        TVectorD vl = TMatrixDSymEigen(Zvs).GetEigenValues();
        v.rank_cv = rank;
        v.lam_max_cv = (float)vl[0];  v.lam_min_cv = (float)vl[rank-1];
      }

      // --- the null direction: the samples' spread along the direction C_B
      //     says has no width, in units of the smallest direction it does keep.
      if (rank < 6) {
        double nv = 0.0;
        for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) nv += U(i, rank) * Sn(i, j) * U(j, rank);
        v.null_sig = (float)(std::sqrt(std::max(nv, 0.0)) / std::sqrt(d[rank - 1]));
      }
      // --- FULL CHAIN. Offer each sample's landing point to the production
      //     chi2 as a hit with a negligible error, against the reference state
      //     and its transported covariance. This runs jacCurv2Loc and the local
      //     projection, so unlike the bare comparison above it is well posed at
      //     any incidence.
      {
        float pdir[3];
        {   // any in-plane direction will do: chi2 is invariant under rotation
            // about the normal (verified independently as the suite's finding 11)
          double a[3] = {0.0, 0.0, 1.0};
          double d[3] = {a[1]*pln[2]-a[2]*pln[1], a[2]*pln[0]-a[0]*pln[2], a[0]*pln[1]-a[1]*pln[0]};
          double dn = std::sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);
          if (dn < 1e-6) { d[0]=1; d[1]=0; d[2]=0; dn=1; }
          for (int i = 0; i < 3; ++i) pdir[i] = (float)(d[i]/dn);
        }
        MPlexLS pe;  MPlexLV pp;  MPlexHS me;  MPlexHV mv, nn, dd, pt3;  MPlexQF c2;
        fill_lanes(pp, parB);  fill_lanes_sym(pe, covB);
        fill_lanes_hv(nn, pln);  fill_lanes_hv(dd, pdir);  fill_lanes_hv(pt3, plp);
        const float tiny = 1e-10f;   // (0.1 um)^2 -- present only to keep the
                                     // 2-D matrix invertible, 4 orders below
                                     // any track covariance here
        for (int n = 0; n < NN; ++n)
          for (int i = 0; i < 6; ++i) me.fArray[i*NN + n] = (i==0||i==2||i==5) ? tiny : 0.f;
        std::vector<double> c2v;  c2v.reserve(acc.size());
        double c2sum = 0.0;
        for (size_t base = 0; base < acc.size(); base += NN) {
          const size_t nlane = std::min((size_t)NN, acc.size() - base);
          for (int n = 0; n < NN; ++n) {
            const auto &a = acc[std::min(base + n, acc.size() - 1)];
            for (int i = 0; i < 3; ++i) mv(n, i, 0) = (float)a[i];
          }
          kalmanComputeChi2Plane(pe, pp, q_, me, mv, nn, dd, pt3, c2, NN);
          for (size_t n = 0; n < nlane; ++n) {
            const float c = c2.constAt(n, 0, 0);
            // NOT std::isfinite: -Ofast implies -ffinite-math-only, under
            // which it returns true for an inf. isFinite is bit-based.
            if (!isFinite(c)) continue;
            c2v.push_back(c);  c2sum += c;
          }
        }
        if (c2v.size() > 100) {
          std::sort(c2v.begin(), c2v.end());
          v.chi2_med  = (float)c2v[c2v.size()/2];
          v.chi2_p90  = (float)c2v[(size_t)(0.90*(c2v.size()-1))];
          v.chi2_p99  = (float)c2v[(size_t)(0.99*(c2v.size()-1))];
          v.chi2_mean = (float)(c2sum / c2v.size());
          v.chi2_med_ratio = v.chi2_med / 1.38629436f;
        }
      }

      out.push_back(v);
    }

    TFile f(out_file, "RECREATE");
    TTree *t = new TTree("covxport", "ensemble covariance transport, one step");
    ValCovXport v;  t->Branch("v", &v);
    for (const auto &x : out) { v = x; t->Fill(); }
    t->Write();  f.Close();
    printf("val_cov_transport: %zu configs, %d samples each -> '%s'\n",
           out.size(), n_samp, out_file);
  }


  // ==========================================================================
  // Flatten the MKFIT_TRACE graph of ONE event's hit matches into a TTree.
  //
  // The trace is a graph of parent-linked records; everything below is a join,
  // not a measurement: TrHitMatch -> TrLayerSearch (for the track's own sigmas)
  // and TrHitMatch -> TrKalmanUpdate (for chi2 and the module-frame residual).
  // Both ids can legitimately be -1 on an untraced path, and every tr_*()
  // accessor is an UNCHECKED vector index, so each is guarded.
  namespace {
    std::vector<ValSearchHit> g_sh;
    std::vector<ValSearchMiss> g_sm;
    std::vector<ValCovStep>    g_cs;

    // Project a 6x6 CCS covariance onto the search's own coordinates. Reading
    // err(2,2) as "sigma_q" is only right in the barrel; in the endcap q is r
    // and needs the full transverse block.
    void project_sigma(const mkfit::TrackState &ts, bool barrel,
                       float &sig_q, float &sig_phi) {
      const double x = ts.parameters[0], y = ts.parameters[1];
      const double r2 = x*x + y*y;
      const double exx = ts.errors.At(0,0), eyy = ts.errors.At(1,1), exy = ts.errors.At(0,1);
      sig_q = barrel ? (float) std::sqrt(std::max(0.0, (double) ts.errors.At(2,2)))
                     : (r2 > 0 ? (float) std::sqrt(std::max(0.0,
                         (x*x*exx + 2*x*y*exy + y*y*eyy) / r2)) : -1.f);
      sig_phi = r2 > 0 ? (float) std::sqrt(std::max(0.0,
                  (y*y*exx - 2*x*y*exy + x*x*eyy) / (r2*r2))) : -1.f;
    }
    int g_sh_evt = -1;
  }


  namespace {
    // Exact crossing of a uniform-B helix with a plane, in double.
    // Substituting the helix into n.(x(a) - P) = 0 gives
    //   f(a) = A sin a + B (1 - cos a) + C a + D
    // which is transcendental unless n_z p_z = 0, hence scan-and-bisect rather
    // than a closed form. Returns the number of roots found in [-lim, lim].
    struct PlaneRoots {
      double a[8]; int n = 0;
    };
    PlaneRoots exact_plane_roots(const double x0[3], const double p0[3], double k,
                                 const double nrm[3], const double pnt[3], double lim) {
      const double A =  k * (nrm[0]*p0[0] + nrm[1]*p0[1]);
      const double B =  k * (nrm[1]*p0[0] - nrm[0]*p0[1]);
      const double C =  k * nrm[2]*p0[2];
      const double D =  nrm[0]*(x0[0]-pnt[0]) + nrm[1]*(x0[1]-pnt[1]) + nrm[2]*(x0[2]-pnt[2]);
      auto fv = [&](double a){ return A*std::sin(a) + B*(1.0-std::cos(a)) + C*a + D; };
      PlaneRoots R;
      const int N = 8000;
      double ap = -lim, fp = fv(-lim);
      for (int i = 1; i <= N && R.n < 8; ++i) {
        const double aa = -lim + 2.0*lim*i/N, ff = fv(aa);
        if ((fp < 0.0) != (ff < 0.0)) {
          double lo = ap, hi = aa, flo = fp;
          for (int it = 0; it < 100; ++it) {
            const double m = 0.5*(lo+hi), fm = fv(m);
            if ((flo < 0.0) == (fm < 0.0)) { lo = m; flo = fm; } else hi = m;
          }
          R.a[R.n++] = 0.5*(lo+hi);
        }
        ap = aa; fp = ff;
      }
      return R;
    }
  }

  namespace {
    // Exact crossing of the helix with a cylinder r = R (barrel) or a plane
    // z = Z (endcap), in double, by the same scan-and-bisect. Returns the root
    // of smallest |alpha|, or -999 if the surface is not reached.
    double exact_surface_alpha(const double x0[3], const double p0[3], double k,
                               double target, bool barrel, double lim) {
      auto fv = [&](double a) {
        const double sa = std::sin(a), ca = std::cos(a);
        if (!barrel) return x0[2] + k*p0[2]*a - target;
        const double X = x0[0] + k*(p0[0]*sa - p0[1]*(1.0-ca));
        const double Y = x0[1] + k*(p0[1]*sa + p0[0]*(1.0-ca));
        return std::hypot(X, Y) - target;
      };
      const int N = 4000;
      double best = -999.0, ap = -lim, fp = fv(-lim);
      for (int i = 1; i <= N; ++i) {
        const double aa = -lim + 2.0*lim*i/N, ff = fv(aa);
        if ((fp < 0.0) != (ff < 0.0)) {
          double lo = ap, hi = aa, flo = fp;
          for (int it = 0; it < 100; ++it) {
            const double m = 0.5*(lo+hi), fm = fv(m);
            if ((flo < 0.0) == (fm < 0.0)) { lo = m; flo = fm; } else hi = m;
          }
          const double r = 0.5*(lo+hi);
          if (best < -900 || std::fabs(r) < std::fabs(best)) best = r;
        }
        ap = aa; fp = ff;
      }
      return best;
    }
  }

  // Toggle material on the SEARCH's inter-layer propagation. Unlike pea -- whose
  // application is a measured no-op because it passes a zero plane normal, so
  // invCos = p/0 is masked to 0 and the radL < 1e-13 bailout fires -- the Kalman
  // step passes the real module plane, so this one is live.
  // Oracle: force the MC-matched hit to win its layer and bypass the chi2 cut.
  // Separates "the true hit was never available" from "ranking / pruning threw
  // it away". Never a production path.
  void val_force_mc(bool on) {
    g_v2p2_force_mc = on;
    printf("val_force_mc: g_v2p2_force_mc = %d\n", (int) on);
  }

  // Both parameter sets, not just the forward one: the beam width a candidate
  // actually gets is CombCandidate::capacity(), reserved from
  // MkJob::max_max_cands() = max(params(), params_bks()), so setting the forward
  // one alone leaves the capacity at whichever is larger and the knob does
  // nothing. That is not hypothetical -- it is what an earlier version of this
  // function did, and cap 3 and cap 6 then came out bit-identical.
  void val_max_cands(int n) {
    const int ni = Config::ItrInfo.size();
    for (int i = 0; i < ni; ++i) {
      Config::ItrInfo[i].m_params.maxCandsPerSeed = n;
      Config::ItrInfo[i].m_backward_params.maxCandsPerSeed = n;
    }
    printf("val_max_cands: maxCandsPerSeed = %d (fwd and bkw) for %d iteration configs\n", n, ni);
  }

  void val_search_material(bool on) {
    auto &pc = const_cast<PropagationConfig&>(Config::TrkInfo.prop_config());
    pc.finding_inter_layer_pflags.apply_material = on;
    printf("val_search_material: finding_inter_layer_pflags.apply_material = %d\n", (int) on);
  }

  // Reference the pre-selection q window to the layer SURFACE rather than to the
  // fixed path length errPropFromPathL_impl transports the covariance to. See
  // MkBins::surface_reference_dq(). Affects the WINDOW AND THE dq CUT ONLY --
  // the Kalman update already carries the term, via jacCurv2Loc's cosz.
  void val_mkbins_surface_q(bool on) {
    g_mkbins_surface_q = on;
    printf("val_mkbins_surface_q: g_mkbins_surface_q = %d\n", (int) on);
  }

  // The dq pre-selection allowance (MkFinderV2p2.cc). It was 3.0 to compensate
  // for a window that was up to 9x too small at |eta| > 2; with the surface
  // reference on, that compensation should no longer be needed. Scannable so
  // the question costs one build.
  void val_extra_dq(float f) {
    g_v2p2_extra_dq = f;
    printf("val_extra_dq: g_v2p2_extra_dq = %.3f\n", f);
  }

  // Per-hit surface reference, using the HIT'S OWN MODULE NORMAL. This is the
  // real fix; MkBins::surface_reference_dq (val_surf_q) is the layer-cylinder
  // scaffold that proved the mechanism and over-widens tilted TBPS ~8x.
  void val_layer_policy(bool wsr, bool hole_limits, bool stop_cuts) {
    Config::v2p2UseWsr = wsr;
    Config::v2p2UseHoleLimits = hole_limits;
    Config::v2p2UseStopCuts = stop_cuts;
    printf("val_layer_policy: wsr=%d hole_limits=%d stop_cuts=%d\n",
           (int) wsr, (int) hole_limits, (int) stop_cuts);
  }

  // Keep one beam slot for a continuation that declined the layer. A beam policy,
  // not a score: declining is the only move that does not shrink the covariance,
  // so it is the only branch left open to an earlier hit having been wrong.
  void val_reserve_hole_slot(bool on) {
    Config::v2p2ReserveHoleSlot = on;
    printf("val_reserve_hole_slot: Config::v2p2ReserveHoleSlot = %d\n", (int) on);
  }

  // Ablate one term of the log-likelihood at fixed eps. Passing the term's own
  // MEAN as the constant removes its variation and nothing else -- see the note
  // in V2p2Score.h for why that is the only comparison that can attribute a
  // regional effect to rho rather than to eps.
  void val_score_terms(bool use_rho, float rho_const, bool use_detv, float detv_const) {
    g_v2p2_score_use_rho = use_rho;      g_v2p2_score_rho_const = rho_const;
    g_v2p2_score_use_detv = use_detv;    g_v2p2_score_detv_const = detv_const;
    printf("val_score_terms: rho %s (const %.4f), det V %s (const %.4f)\n",
           use_rho ? "PER STEP" : "FLAT", rho_const,
           use_detv ? "PER HIT" : "FLAT", detv_const);
  }

  // Mean ln(rho) and mean ln(det V) per hit taken, so the ablation constants are
  // measured. Accumulation is not thread safe; a trace build serialises the
  // in-event loops, which is where this is meant to run.
  void val_score_term_stats(bool on) {
    if (on) {
      g_v2p2_score_n_hits = 0;
      g_v2p2_score_sum_log_rho = g_v2p2_score_sum_log_detv = 0.0;
      g_v2p2_score_accum = true;
      printf("val_score_term_stats: accumulating.\n");
      return;
    }
    g_v2p2_score_accum = false;
    const long n = g_v2p2_score_n_hits;
    printf("val_score_term_stats: %ld hits scored; mean ln(rho) = %.4f, "
           "mean ln(det V) = %.4f\n", n,
           n ? g_v2p2_score_sum_log_rho / n : 0.0,
           n ? g_v2p2_score_sum_log_detv / n : 0.0);
  }

  void val_score_mode(int mode, float hit_eff) {
    g_v2p2_score_mode = mode;
    g_v2p2_score_fwd.hit_eff = g_v2p2_score_bkw.hit_eff = hit_eff;
    printf("val_score_mode: g_v2p2_score_mode = %d (0 linear, 1 loglh), hit_eff = %g\n",
           mode, hit_eff);
  }

  void val_in_layer_comb(bool on) {
    Config::v2p2InLayerComb = on;
    printf("val_in_layer_comb: Config::v2p2InLayerComb = %d\n", (int) on);
  }

  // miss_fwd / miss_bkw are the head-body asymmetry: outward a trailing hole is
  // at large radius and cheap, inward it is at small radius and is the most
  // expensive hole there is.
  void val_score(float hit_bonus, float chi2_weight, float miss_fwd, float miss_bkw) {
    g_v2p2_score_fwd.hit_bonus = g_v2p2_score_bkw.hit_bonus = hit_bonus;
    g_v2p2_score_fwd.chi2_weight = g_v2p2_score_bkw.chi2_weight = chi2_weight;
    g_v2p2_score_fwd.miss_penalty = miss_fwd;
    g_v2p2_score_bkw.miss_penalty = miss_bkw;
    printf("val_score: hit_bonus=%.2f chi2_weight=%.2f miss_fwd=%.2f miss_bkw=%.2f "
           "(hole beats a hit above chi2 = %.1f inward)\n",
           hit_bonus, chi2_weight, miss_fwd, miss_bkw,
           chi2_weight > 0 ? (hit_bonus + miss_bkw) / chi2_weight : 0.0f);
  }

  void val_surf_q_hit(bool on) {
    g_v2p2_surface_q = on;
    printf("val_surf_q_hit: g_v2p2_surface_q = %d\n", (int) on);
  }

  // ==========================================================================
  // IS THE KALMAN GAIN RIGHT?
  //
  // Everything validated so far exercised kalmanComputeChi2Plane -- the
  // PREDICTION. kalmanUpdatePlane, i.e. C' = (I - KH) C, has never been tested
  // by anything, and a gain that is too large is exactly an ACCUMULATION
  // mechanism: each update over-shrinks, compounding over 4-18 layers, which is
  // the shape of the residual deficit (0.55 at handover -> 1.33 deep).
  //
  // This needs no new run: MKFIT_TRACE_KALMAN_DEBUG already stores the state
  // before and after each update, with full covariance. For a measurement of
  // variance R on a track of variance P, the posterior is
  //      1/P' = 1/P + 1/R.
  // Take the ratio (P' as the code computes it) / (P' as that says). Below 1
  // means the code shrinks TOO MUCH, i.e. the gain is too big.
  //
  // CAVEAT, stated because it bounds the conclusion: the real update is 2-D in
  // the module plane and carries position-angle correlations, so the 1-D law is
  // an approximation. It is a good one for the q direction of an untilted barrel
  // pixel, where the two local directions are near-independent -- so read a
  // LARGE departure as a finding and a few-percent one as the approximation.
  // ==========================================================================
  namespace {
    struct KGAcc {
      std::vector<double> ratio, sh_over_st, eta;
      std::vector<int> layer;
      long n_seen = 0, n_used = 0, n_nohit = 0, n_badcov = 0;
    };
    KGAcc g_kg;
  }

  void val_kgain_reset() { g_kg = KGAcc(); }

  void val_kgain_event(const Event *ev) {
    if (ev == nullptr) return;
#ifdef MKFIT_TRACE_KALMAN_DEBUG
    const int n_hm = (int) ev->trHitMatches_.size();
    for (const TrKalmanUpdate &ku : ev->trKalmanUpdates_) {
      ++g_kg.n_seen;
      if (ku.hit_match_id < 0 || ku.hit_match_id >= n_hm) continue;
      const TrHitMatch &hm = ev->trHitMatches_[ku.hit_match_id];
      if (!hm.mc_match) continue;                       // truth-matched only
      if (hm.layer < 0 || hm.layer >= (int) ev->layerHits_.size()) continue;
      if (hm.hit < 0 || hm.hit >= (int) ev->layerHits_[hm.layer].size()) { ++g_kg.n_nohit; continue; }
      const Hit &h = ev->layerHits_[hm.layer][hm.hit];
      const bool barrel = Config::TrkInfo[hm.layer].is_barrel();
      // q variance of the track, before and after; q = z in the barrel, r outside
      auto qvar = [&](const mkfit::TrackState &ts) {
        if (barrel) return (double) ts.errors.At(2, 2);
        const double x = ts.parameters[0], y = ts.parameters[1], r2 = x*x + y*y;
        if (r2 <= 0) return -1.0;
        return (x*x*ts.errors.At(0,0) + 2*x*y*ts.errors.At(0,1) + y*y*ts.errors.At(1,1)) / r2;
      };
      const double P = qvar(ku.propagated_state), Pp = qvar(ku.updated_state);
      double R;
      if (barrel) R = h.ezz();
      else {
        const double x = h.x(), y = h.y(), r2 = x*x + y*y;
        R = r2 > 0 ? (x*x*h.exx() + 2*x*y*h.exy() + y*y*h.eyy()) / r2 : -1.0;
      }
      if (!(P > 0 && Pp > 0 && R > 0)) { ++g_kg.n_badcov; continue; }
      const double Pp_pred = 1.0 / (1.0/P + 1.0/R);
      g_kg.ratio.push_back(std::sqrt(Pp / Pp_pred));     // in SIGMA
      g_kg.sh_over_st.push_back(std::sqrt(R / P));
      g_kg.eta.push_back(hm.kine_on_plane.pos.Eta());
      g_kg.layer.push_back(hm.layer);
      ++g_kg.n_used;
    }
#else
    printf("val_kgain_event: needs MKFIT_TRACE_KALMAN_DEBUG\n");
#endif
  }

  void val_kgain_report() {
    auto q = [](std::vector<double> u, double p) {
      if (u.empty()) return 0.0; std::sort(u.begin(), u.end());
      return u[(size_t)(p/100.*(u.size()-1))]; };
    printf("\n===== IS THE KALMAN GAIN RIGHT? =====\n");
    printf("MC-matched hits. ratio = sigma_q(after, as the code computes it)\n");
    printf("                      / sigma_q(after, from 1/P' = 1/P + 1/R).\n");
    printf("1.0 = correct. BELOW 1 = the update shrinks too much (gain too big).\n");
    printf("updates seen %ld, used %ld (no hit %ld, bad cov %ld)\n\n",
           g_kg.n_seen, g_kg.n_used, g_kg.n_nohit, g_kg.n_badcov);
    // by how much the hit constrains: sigma_hit/sigma_trk
    const double sb[] = {0, 0.5, 1, 2, 5, 1e9}; const int NS = 5;
    const char *sn[NS] = {"hit<<trk (<0.5)","0.5-1","1-2","2-5","hit>>trk (>5)"};
    printf("  %-18s %8s | %8s %8s %8s\n", "sigma_hit/sigma_trk", "n", "p25", "median", "p75");
    for (int b = 0; b < NS; ++b) {
      std::vector<double> v;
      for (size_t i = 0; i < g_kg.ratio.size(); ++i)
        if (g_kg.sh_over_st[i] >= sb[b] && g_kg.sh_over_st[i] < sb[b+1]) v.push_back(g_kg.ratio[i]);
      if (v.size() < 40) continue;
      printf("  %-18s %8zu | %8.3f %8.3f %8.3f\n", sn[b], v.size(), q(v,25), q(v,50), q(v,75));
    }
    printf("\n  %-18s %8s | %8s %8s\n", "pixel-barrel layer", "n", "median", "p75");
    for (int L = 0; L < 4; ++L) {
      std::vector<double> v;
      for (size_t i = 0; i < g_kg.ratio.size(); ++i) if (g_kg.layer[i] == L) v.push_back(g_kg.ratio[i]);
      if (v.size() < 40) continue;
      printf("  %-18d %8zu | %8.3f %8.3f\n", L, v.size(), q(v,50), q(v,75));
    }
    std::vector<double> all = g_kg.ratio;
    if (all.size() >= 40)
      printf("\n  ALL: n %zu  p25 %.3f  median %.3f  p75 %.3f\n",
             all.size(), q(all,25), q(all,50), q(all,75));
    printf("\n  NOTE: the real update is 2-D in the module plane with position-angle\n");
    printf("  correlations, so the 1-D law is approximate. Read a LARGE departure as\n");
    printf("  a finding and a few-percent one as the approximation.\n\n");
  }

  // ==========================================================================
  // THE EXACT GAIN TEST -- information gain, no jacobian reimplementation.
  //
  // For a true Kalman update with measurement matrix H and measurement
  // covariance R,
  //        P'^-1 - P^-1  =  H^T R^-1 H
  // exactly. The right-hand side is PSD and has RANK EQUAL TO THE NUMBER OF
  // MEASURED COORDINATES -- 2 here, the module plane. So from the two stored
  // covariances alone:
  //   * rank > 2  -> the update is injecting information it has no right to;
  //   * a negative eigenvalue -> it is REMOVING information, which no update can;
  //   * the two positive eigenvalues are 1/sigma^2 of the effective measurement,
  //     so they say directly whether the hit error used is too small.
  //
  // This needs no model of jacCurv2Loc and makes no 1-D approximation, which is
  // what the sigma-ratio version could not avoid.
  // ==========================================================================
  namespace {
    struct KIAcc {
      std::vector<double> l1, l2, l3, neg;   // top 3 eigenvalues, most negative
      std::vector<double> cond;              // condition number of whitened P
      std::vector<int> layer;
      long n = 0, n_singular = 0;
    };
    KIAcc g_ki;
    // WHITENED symmetric 6x6 inverse. The raw covariance mixes cm^2, rad^2 and
    // GeV^-2, so its condition number is ~1e10+ and inverting the float32-stored
    // matrix produces garbage in the small eigenvalues -- measured: 53 % of
    // updates came out with a NEGATIVE eigenvalue, which no update can produce.
    // Scaling every coordinate by its own sigma first (the correlation matrix:
    // dimensionless, unit diagonal) fixes that, and because S^-1 P S^-1 is a
    // CONGRUENCE, rank and inertia -- the only things this test reads -- are
    // unchanged. Both matrices are whitened by the SAME S, taken from P.
    bool inv6_white(const mkfit::TrackState &ts, const double *sc, TMatrixDSym &out) {
      TMatrixDSym M(6);
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) {
          const double v = (double) ts.errors.At(i,j) * sc[i] * sc[j];
          M(i,j) = v; M(j,i) = v;
        }
      double det = 0;
      M.Invert(&det);
      if (!(std::fabs(det) > 0) || !std::isfinite(det)) return false;
      out.ResizeTo(6,6); out = M; return true;
    }
    // condition number of the whitened P, so it can be said whether float32
    // storage can resolve this at all
    double cond_white(const mkfit::TrackState &ts, const double *sc) {
      TMatrixDSym M(6);
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) {
          const double v = (double) ts.errors.At(i,j) * sc[i] * sc[j];
          M(i,j) = v; M(j,i) = v;
        }
      TMatrixDSymEigen e(M); TVectorD ev = e.GetEigenValues();
      double mx = -1e300, mn = 1e300;
      for (int i = 0; i < 6; ++i) { mx = std::max(mx, ev[i]); mn = std::min(mn, ev[i]); }
      return (mn > 0) ? mx / mn : -1.0;
    }
  }

  void val_kinfo_reset() { g_ki = KIAcc(); }

  void val_kinfo_event(const Event *ev) {
    if (ev == nullptr) return;
#ifdef MKFIT_TRACE_KALMAN_DEBUG
    const int n_hm = (int) ev->trHitMatches_.size();
    for (const TrKalmanUpdate &ku : ev->trKalmanUpdates_) {
      if (ku.hit_match_id < 0 || ku.hit_match_id >= n_hm) continue;
      const TrHitMatch &hm = ev->trHitMatches_[ku.hit_match_id];
      if (!hm.mc_match || hm.layer < 0 || hm.layer >= 4) continue;   // pixel barrel
      // P is SINGULAR by construction -- the transported covariance is rank 5
      // with a null direction along the momentum (S11). So P^-1 does not exist
      // and the information form P'^-1 - P^-1 is ill-posed; measured, it gave a
      // negative eigenvalue in 54 % of updates, which is that null direction and
      // not a defect of the update. Use the form that needs NO inverse of P:
      //      P - P' = P H^T (H P H^T + R)^-1 H P
      // which is PSD of rank <= 2 for a 2-D measurement. Same content, no H, no
      // P^-1, and well conditioned. Whitened by P's own diagonal so the
      // eigenvalues are dimensionless -- a congruence, so rank and inertia,
      // which are the only things read here, are unchanged.
      double sc[6];
      bool okp = true;
      for (int i = 0; i < 6; ++i) {
        const double d = ku.propagated_state.errors.At(i,i);
        if (!(d > 0)) { okp = false; break; }
        sc[i] = 1.0 / std::sqrt(d);
      }
      if (!okp) { ++g_ki.n_singular; continue; }
      TMatrixDSym D(6);
      for (int i = 0; i < 6; ++i)
        for (int j = 0; j <= i; ++j) {
          const double v = ((double) ku.propagated_state.errors.At(i,j) -
                            (double) ku.updated_state.errors.At(i,j)) * sc[i] * sc[j];
          D(i,j) = v; D(j,i) = v;
        }
      g_ki.cond.push_back(1.0);   // not used in this form
      TMatrixDSymEigen eig(D);
      TVectorD ev6 = eig.GetEigenValues();              // descending
      std::vector<double> e(6);
      for (int i = 0; i < 6; ++i) e[i] = ev6[i];
      std::sort(e.begin(), e.end(), std::greater<double>());
      g_ki.l1.push_back(e[0]); g_ki.l2.push_back(e[1]); g_ki.l3.push_back(e[2]);
      g_ki.neg.push_back(e[5]);
      g_ki.layer.push_back(hm.layer);
      ++g_ki.n;
    }
#else
    printf("val_kinfo_event: needs MKFIT_TRACE_KALMAN_DEBUG\n");
#endif
  }

  void val_kinfo_report() {
    auto q = [](std::vector<double> u, double p) {
      if (u.empty()) return 0.0; std::sort(u.begin(), u.end());
      return u[(size_t)(p/100.*(u.size()-1))]; };
    printf("\n===== EXACT GAIN TEST: P - P' must be PSD of RANK <= 2 =====\n");
    printf("Pixel barrel, MC-matched. n = %ld (singular %ld)\n\n", g_ki.n, g_ki.n_singular);
    if (g_ki.n < 40) { printf("  too few\n"); return; }
    printf("  eigenvalues of the information gain (descending), quantiles:\n");
    printf("    %-14s %12s %12s %12s\n", "", "p25", "median", "p75");
    printf("    %-14s %12.4g %12.4g %12.4g\n", "lambda_1", q(g_ki.l1,25), q(g_ki.l1,50), q(g_ki.l1,75));
    printf("    %-14s %12.4g %12.4g %12.4g\n", "lambda_2", q(g_ki.l2,25), q(g_ki.l2,50), q(g_ki.l2,75));
    printf("    %-14s %12.4g %12.4g %12.4g   <- must be ~0 (rank<=2)\n", "lambda_3", q(g_ki.l3,25), q(g_ki.l3,50), q(g_ki.l3,75));
    printf("    %-14s %12.4g %12.4g %12.4g   <- must be >= 0 (PSD)\n", "lambda_min", q(g_ki.neg,25), q(g_ki.neg,50), q(g_ki.neg,75));
    // rank: how often is lambda_3 non-negligible against lambda_2
    long bad = 0, negs = 0;
    for (size_t i = 0; i < g_ki.l1.size(); ++i) {
      if (g_ki.l2[i] > 0 && g_ki.l3[i] > 1e-2 * g_ki.l2[i]) ++bad;
      if (g_ki.neg[i] < -1e-2 * std::fabs(g_ki.l1[i])) ++negs;
    }
    printf("\n    rank > 2 (lambda_3 > 1e-2 lambda_2): %ld of %ld = %.2f %%\n",
           bad, g_ki.n, 100.*bad/g_ki.n);
    printf("    a NEGATIVE eigenvalue:               %ld of %ld = %.2f %%\n",
           negs, g_ki.n, 100.*negs/g_ki.n);
    printf("\n  THRESHOLDS: lambda_3 and lambda_min sit at ~1e-3 against lambda_1 ~ 2.4,\n");
    printf("  a relative 6e-4 -- float32 round-off through the whitening and the\n");
    printf("  eigen-decomposition. They are noise, not rank or indefiniteness, so the\n");
    printf("  flags above are cut at 1e-2 relative. A real rank-3 component or a real\n");
    printf("  negative direction would be orders of magnitude larger.\n");
    printf("\n  lambda_1, lambda_2 are the variance REMOVED by the hit in the two\n");
    printf("  measured directions, in units of P's own sigma. They are NOT bounded by\n");
    printf("  1: whitened, P - P' is bounded by the CORRELATION matrix of P, whose\n");
    printf("  eigenvalues reach 6 when the parameters are correlated. So lambda_1 ~ 2.4\n");
    printf("  is allowed and says the hit constrains a strongly correlated combination.\n");
    printf("\n  WHAT THIS TESTS, AND WHAT IT DOES NOT: it proves the update is a valid\n");
    printf("  rank-2 Kalman update -- the gain injects no spurious information and\n");
    printf("  removes none outside the measured plane. It says NOTHING about whether R\n");
    printf("  (the hit covariance in the module frame) has the right MAGNITUDE; that\n");
    printf("  needs H, i.e. the jacCurv2Loc chain.\n");
    printf("\n");
  }

  // ==========================================================================
  // ARE THE MATERIAL VALUES RIGHT? -- our grid against an INDEPENDENT measurement.
  //
  // Everything so far tested how material is APPLIED. This tests the numbers
  // themselves. The geometry side measured the per-transition budget directly
  // from the real TGeo geometry (ANSWERS-for-mkfit.md section 10, "median
  // track" column, 40k helices from the IP per transition), so there is an
  // external reference for exactly this.
  //
  // Integrating our (z,r) grid radially between the same layer centroid radii
  // gives the same quantity. The comparison is only meaningful at central eta,
  // where a radial path IS the track path, which is why z = 0 is the row to
  // read.
  // ==========================================================================
  void val_material_profile(float z = 0.0f) {
    // layer centroid radii and the reference budgets, from
    // /baz/matevz/root-dev/geo-stuff/ANSWERS-for-mkfit.md sections 10 and 11.
    struct Ref { const char *name; float r_cent; float ref_median; };
    static const Ref L[] = {
      {"IT1",    3.06f, 0.0f},     {"IT2",    6.18f, 0.0310f},
      {"IT3",   10.52f, 0.0232f},  {"IT4",   14.71f, 0.0187f},
      {"OT1",   23.57f, 0.0664f},  {"OT2",   36.32f, 0.0611f},
      {"OT3",   51.45f, 0.0461f},  {"OT4",   68.80f, 0.0578f},
      {"OT5",   86.07f, 0.0308f},  {"OT6",  108.36f, 0.0299f},
    };
    const int NL = sizeof(L)/sizeof(Ref);
    const auto &ti = Config::TrkInfo;
    printf("\n===== OUR MATERIAL GRID vs THE GEOMETRY SIDE'S MEASUREMENT =====\n");
    printf("z = %.1f cm, radial path. 'ours' integrates the grid between the two\n", z);
    printf("layer centroid radii, 0.1 cm steps, treating radl as x/X0 PER BIN over a\n");
    printf("1 cm bin. 'theirs' is the median-track x/X0 for that transition, measured\n");
    printf("from TGeo with 40k helices (ANSWERS section 10).\n\n");
    printf("  %-12s %10s %10s %8s\n", "transition", "ours", "theirs", "ours/th");
    double tot_ours = 0, tot_theirs = 0;
    for (int i = 1; i < NL; ++i) {
      double sum = 0;
      const double r0 = L[i-1].r_cent, r1 = L[i].r_cent, dr = 0.1;
      for (double r = r0; r < r1; r += dr) {
        TrackerInfo::Material m = ti.material_checked(std::abs(z), (float) r);
        sum += m.radl * dr;            // radl is per-cm-of-bin; 1 cm bins
      }
      tot_ours += sum; tot_theirs += L[i].ref_median;
      printf("  %-4s -> %-4s %10.4f %10.4f %8.2f\n",
             L[i-1].name, L[i].name, sum, L[i].ref_median,
             L[i].ref_median > 0 ? sum / L[i].ref_median : 0.0);
    }
    printf("  %-12s %10.4f %10.4f %8.2f\n", "TOTAL", tot_ours, tot_theirs,
           tot_theirs > 0 ? tot_ours / tot_theirs : 0.0);
    printf("\n  And the raw grid, so the semantics of radl are visible -- if the\n");
    printf("  material sits in one bin per layer, radl is a per-BIN budget and the\n");
    printf("  integral above double counts; if it is smeared, it is a density.\n");
    printf("  %8s %12s %12s\n", "r [cm]", "radl", "bbxi");
    for (double r = 1.0; r < 120.0; r += 1.0) {
      TrackerInfo::Material m = ti.material_checked(std::abs(z), (float) r);
      if (m.radl > 1e-6f)
        printf("  %8.1f %12.5g %12.5g\n", r, m.radl, m.bbxi);
    }
    printf("\n");
  }

  // ==========================================================================
  // CLUSTER SIZES -- is the hit error Gaussian or uniform, per region?
  //
  // It decides how the residual may be read. A MULTI-cell cluster is
  // charge-interpolated and the error is Gaussian-ish; a SINGLE-cell cluster is
  // uniform over the pitch, and then IQR/1.349 (a Gaussian conversion)
  // over-estimates sigma by 1.28x. mkfit::Hit carries spanRows()/spanCols(), so
  // this is measurable rather than assumed.
  //
  // Convention: rows run along local x (the precise / phi direction), columns
  // along local y (the coarse / q direction) -- consistent with
  // ModuleInfo::xdir being commented "the precise / phi direction".
  // ==========================================================================
  void val_cluster_sizes(const Event *ev) {
    if (ev == nullptr) return;
    struct Reg { const char *name; int lo, hi; };
    static const Reg R[] = {
      {"PixB 0-3",       0,  3}, {"TBPS-P 4/6/8",   4,  9},
      {"TBPS-S 5/7/9",   4,  9}, {"TOB 2S 10-15",  10, 15},
      {"fwd pix 16-27", 16, 27}, {"TEC 28-37",     28, 37},
    };
    const int NR = sizeof(R)/sizeof(Reg);
    long n[NR] = {}, sr1[NR] = {}, sc1[NR] = {};
    double sr[NR] = {}, sc[NR] = {};
    for (int L = 0; L < (int) ev->layerHits_.size(); ++L) {
      int r = -1;
      for (int k = 0; k < NR; ++k) {
        if (L < R[k].lo || L > R[k].hi) continue;
        if (k == 1 && (L & 1) != 0) continue;      // TBPS-P = even
        if (k == 2 && (L & 1) != 1) continue;      // TBPS-S = odd
        r = k; break;
      }
      if (r < 0) continue;
      for (const Hit &h : ev->layerHits_[L]) {
        const unsigned int a = h.spanRows(), b = h.spanCols();
        ++n[r]; sr[r] += a; sc[r] += b;
        if (a == 1) ++sr1[r];
        if (b == 1) ++sc1[r];
      }
    }
    printf("\n===== CLUSTER SIZES, and what they imply for the hit error =====\n");
    printf("rows = local x = PRECISE / phi ; cols = local y = COARSE / q.\n");
    printf("A single-cell span means a UNIFORM error over the pitch, for which\n");
    printf("IQR/1.349 over-estimates sigma by 1.28x. Multi-cell means charge\n");
    printf("interpolation, i.e. Gaussian-ish, and the Gaussian conversion is right.\n\n");
    printf("  %-15s %9s | %8s %10s | %8s %10s\n", "region", "n hits",
           "<rows>", "rows==1", "<cols>", "cols==1");
    for (int k = 0; k < NR; ++k) {
      if (n[k] < 100) continue;
      printf("  %-15s %9ld | %8.2f %9.1f%% | %8.2f %9.1f%%\n", R[k].name, n[k],
             sr[k]/n[k], 100.0*sr1[k]/n[k], sc[k]/n[k], 100.0*sc1[k]/n[k]);
    }
    printf("\n");
  }

  // bit 0 = skip the exact double-precision helix-plane solve (the expensive
  // part); bit 1 = keep only pixel-barrel hits. 3 = both, which is what a
  // pixel-barrel covariance scan wants; 1 = whole detector but still fast,
  // which is what a phi-everywhere measurement wants.
  void val_search_lite(int mode) {
    g_val_search_lite = mode;
    printf("val_search_lite: mode %d (skip_solve=%d pixb_only=%d)\n",
           mode, mode & 1, (mode >> 1) & 1);
  }

  // Diagnostic scale on radL in applyMaterialEffects (scattering only; the
  // energy-loss terms use hitsXi and are untouched).
  void val_mat_scale(float f) {
    g_mat_scale = f;
    printf("val_mat_scale: g_mat_scale = %.3f\n", f);
  }

  // Scales ONLY the energy-loss straggling variance into err(3,3). Separate
  // from val_mat_scale, which touches radL and hence the angular terms: the two
  // probe different halves of "material", and only this one can let hits absorb
  // dE/dx mismodelling.
  // Extra radL factor applied ONLY in the forward pixel discs (|z|>22, r<26).
  // The prediction it tests: "outside the window" moves, chi2 does not.
  void val_mat_fwdpix(float f) {
    g_mat_scale_fwdpix = f;
    printf("val_mat_fwdpix: g_mat_scale_fwdpix = %.3f\n", f);
  }

  void val_eloss_var_scale(float f) {
    g_mat_eloss_var_scale = f;
    printf("val_eloss_var_scale: g_mat_eloss_var_scale = %.3f\n", f);
  }


  // ==========================================================================
  // Track-level true-hit efficiency of the inward search: of the sim track's
  // hits in the pixel barrel and the forward disks, how many did the FINAL
  // candidate actually collect? This is the quantity "the backward search gets
  // true hits down to the first pixel layers with 75% efficiency" refers to,
  // and unlike any per-layer-search measure it is not survivorship-biased: a
  // candidate that dies early simply has few matched hits.
  namespace {
    struct TEff { long n_trk = 0;
                  long sim_pix = 0, got_pix = 0, sim_dsk = 0, got_dsk = 0;
                  long deepest_ok = 0;
                  // "any true pixel-barrel hit" is only a fair question for a
                  // track that HAS one to find. High-|eta| tracks largely miss
                  // the pixel barrel (~1 hit each against ~3 for barrel-only),
                  // so counting them in the denominator understates the rate.
                  long n_trk_with_pix = 0;
                  // Distinct LAYERS with a sim hit, against the hit count. The
                  // search adds at most one hit per layer (the best-hit hack),
                  // so layers/hits is a hard ceiling on the efficiency above --
                  // and phase-2 disks carry module overlaps, so it is well below 1.
                  long sim_pix_lay = 0, sim_dsk_lay = 0; };
    TEff g_te[2];   // [0] barrel-only sim track, [1] touches the disks
    // Split by how many pixel-barrel hits the SIM track offers. A plain T5 is a
    // track whose pixel match FAILED upstream (pT5 = LST T5 + Patatrack pixels),
    // so the sharp question is: when the pixel hits are genuinely there, does
    // the inward search get them?
    TEff g_tp[3];   // spix 0 / 1-2 / >=3
    long g_te_ntrk = 0, g_te_nolbl = 0;
  }

  namespace { long g_chop_n = 0, g_chop_back = 0, g_chop_trk = 0, g_chop_all = 0; }

  void val_track_eff_reset() { g_te[0] = g_te[1] = TEff();
                               g_tp[0] = g_tp[1] = g_tp[2] = TEff();
                               g_te_ntrk = g_te_nolbl = 0;
                               g_chop_n = g_chop_back = g_chop_trk = g_chop_all = 0; }

  // The pT5-chopped control. A pT5's pixel hits were FOUND upstream, so after
  // chopping them off they are an exact, truth-free denominator for the inward
  // search: did it put the same hits back? Plain T5s are the population where
  // those hits were NOT found upstream, so they have no such denominator.
  void val_chop_recovery_event(const Event *ev) {
    if (ev == nullptr) return;
    if (Shell::s_chopped_hits.empty()) return;
    for (const Track &c : ev->candidateTracks_) {
      auto it = Shell::s_chopped_hits.find(c.label());
      if (it == Shell::s_chopped_hits.end() || it->second.empty()) continue;
      ++g_chop_trk;
      int back = 0;
      for (const HitOnTrack &ch : it->second) {
        for (int i = 0; i < c.nTotalHits(); ++i) {
          const HitOnTrack hot = c.getHitOnTrack(i);
          if (hot.layer == ch.layer && hot.index == ch.index) { ++back; break; }
        }
      }
      g_chop_n += (long) it->second.size();
      g_chop_back += back;
      if (back == (int) it->second.size()) ++g_chop_all;
    }
    Shell::s_chopped_hits.clear();
  }

  void val_chop_recovery_report(const char *tag) {
    printf("\n=== pT5 PIXEL-CHOP RECOVERY [%s] ===\n", tag);
    printf("Hits chopped off pT5 seeds and looked for again by the inward search.\n");
    printf("Exact comparison -- same (layer, index) -- no truth matching involved.\n");
    printf("  tracks with chopped hits : %ld\n", g_chop_trk);
    printf("  hits chopped             : %ld\n", g_chop_n);
    printf("  hits recovered           : %ld   (%.1f%%)\n", g_chop_back,
           g_chop_n ? 100.0*g_chop_back/g_chop_n : 0.0);
    printf("  tracks fully recovered   : %ld   (%.1f%%)\n", g_chop_all,
           g_chop_trk ? 100.0*g_chop_all/g_chop_trk : 0.0);
  }

  void val_track_eff_event(const Event *ev) {
    if (ev == nullptr) return;
    for (const Track &c : ev->candidateTracks_) {
      ++g_te_ntrk;
      auto si = ev->simInfoForTrack(c);
      if (!si.is_set()) { ++g_te_nolbl; continue; }
      const int L = si.label;
      const Track &st = ev->simTracks_[L];
      // the sim track's own hit content, by region
      long spix = 0, sdsk = 0;
      std::set<int> lpix, ldsk;
      for (int i = 0; i < st.nTotalHits(); ++i) {
        const HitOnTrack hot = st.getHitOnTrack(i);
        if (hot.index < 0) continue;
        if (hot.layer < 4) { ++spix; lpix.insert(hot.layer); }
        else if (hot.layer >= 16) { ++sdsk; ldsk.insert(hot.layer); }
      }
      if (spix == 0 && sdsk == 0) continue;
      const int cls = (sdsk > 0) ? 1 : 0;
      // what the candidate actually holds there, TRUTH-checked per hit
      long gpix = 0, gdsk = 0; int deepest = -1;
      for (int i = 0; i < c.nTotalHits(); ++i) {
        const HitOnTrack hot = c.getHitOnTrack(i);
        if (hot.index < 0 || hot.layer < 0) continue;
        const Hit &h = ev->layerHits_[hot.layer][hot.index];
        const int hl = ev->simHitsInfo_[h.mcHitID()].mcTrackID();
        if (hl != L) continue;
        if (hot.layer < 4)  { ++gpix; if (hot.layer > deepest) deepest = hot.layer; }
        else if (hot.layer >= 16) ++gdsk;
      }
      TEff &tp = g_tp[spix == 0 ? 0 : (spix < 3 ? 1 : 2)];
      ++tp.n_trk;  tp.sim_pix += spix;  tp.got_pix += gpix;
      tp.sim_dsk += sdsk;  tp.got_dsk += gdsk;
      if (spix > 0) { ++tp.n_trk_with_pix; if (gpix > 0) ++tp.deepest_ok; }
      tp.sim_pix_lay += (long) lpix.size();

      TEff &t = g_te[cls];
      ++t.n_trk;  t.sim_pix += spix;  t.got_pix += gpix;
      if (spix > 0) ++t.n_trk_with_pix;
      t.sim_pix_lay += (long) lpix.size();  t.sim_dsk_lay += (long) ldsk.size();
      t.sim_dsk += sdsk;  t.got_dsk += gdsk;
      if (gpix > 0) ++t.deepest_ok;
    }
  }

  void val_track_eff_report(const char *tag) {
    printf("\n=== TRACK-LEVEL TRUE-HIT EFFICIENCY [%s] ===\n", tag);
    printf("%ld candidate tracks, %ld with no sim label.\n", g_te_ntrk, g_te_nolbl);
    printf("  %-14s %7s | %9s %9s %7s | %9s %9s %7s | %8s %9s\n",
           "sim track","tracks","sim pixB","got","eff","sim disks","got","eff",
           "w/ pixB","any pixB*");
    printf("  (* fraction of tracks that HAVE a pixel-barrel hit and got at least one)\n");
    printf("  CEILING = distinct layers / hits. It was a hard ceiling while the search\n");
    printf("  added at most ONE hit per layer (the best-hit hack); with the in-layer\n");
    printf("  combinatorial (Config::v2p2InLayerComb) it is a soft one -- a path may take\n");
    printf("  several hits in a layer, so read these as a FRACTION OF CEILING, not against\n");
    printf("  100%%. Overlaps are what put the ceiling well below 100%% in the first place.\n");
    const char *nm[2] = {"barrel-only", "touches disks"};
    for (int c = 0; c < 2; ++c) {
      const TEff &t = g_te[c];
      if (t.n_trk == 0) continue;
      printf("  %-14s %7ld | %9ld %9ld %6.1f%% | %9ld %9ld %6.1f%% | %8ld %8.1f%%\n",
             nm[c], t.n_trk, t.sim_pix, t.got_pix,
             t.sim_pix ? 100.0*t.got_pix/t.sim_pix : 0.0,
             t.sim_dsk, t.got_dsk, t.sim_dsk ? 100.0*t.got_dsk/t.sim_dsk : 0.0,
             t.n_trk_with_pix,
             t.n_trk_with_pix ? 100.0*t.deepest_ok/t.n_trk_with_pix : 0.0);
      printf("  %-14s %7s | CEILING pixB %5.1f%% (of %ld) | CEILING disks %5.1f%% (of %ld)\n",
             "", "", t.sim_pix ? 100.0*t.sim_pix_lay/t.sim_pix : 0.0, t.sim_pix_lay,
             t.sim_dsk ? 100.0*t.sim_dsk_lay/t.sim_dsk : 0.0, t.sim_dsk_lay);
    }
    printf("\n  SPLIT BY HOW MANY PIXEL-BARREL HITS THE SIM TRACK OFFERS:\n");
    printf("  %-14s %7s | %8s %8s %7s %9s | %9s\n",
           "sim pixB hits","tracks","sim","got","eff","of ceiling","any pixB");
    const char *pn[3] = {"0 (none to find)", "1 - 2", ">= 3"};
    for (int c = 0; c < 3; ++c) {
      const TEff &t = g_tp[c];
      if (t.n_trk == 0) continue;
      const double eff = t.sim_pix ? 100.0*t.got_pix/t.sim_pix : 0.0;
      const double cei = t.sim_pix ? 100.0*t.sim_pix_lay/t.sim_pix : 0.0;
      printf("  %-14s %7ld | %8ld %8ld %6.1f%% %8.0f%% | %8.1f%%\n",
             pn[c], t.n_trk, t.sim_pix, t.got_pix, eff, cei > 0 ? 100.0*eff/cei : 0.0,
             t.n_trk_with_pix ? 100.0*t.deepest_ok/t.n_trk_with_pix : 0.0);
    }
  }

  void val_search_reset() { g_sh.clear(); g_sm.clear(); g_cs.clear(); g_sh_evt = -1; }

  void val_search_event(const Event *ev, int event_idx) {
    if (ev == nullptr) return;
    // seed index -> sim label. ProcessEventHlt() has already relabelled seeds
    // sequentially, so TrCandMeta::global_seed indexes seedTracks_ directly.
    std::vector<int> seed_sim(ev->seedTracks_.size(), -1);
    std::vector<float> seed_gf(ev->seedTracks_.size(), -1.f);
    std::vector<int> seed_nv(ev->seedTracks_.size(), -1), seed_nm(ev->seedTracks_.size(), -1);
    for (size_t i = 0; i < ev->seedTracks_.size(); ++i) {
      auto si = ev->simInfoForTrack(ev->seedTracks_[i]);
      if (si.is_set()) {
        seed_sim[i] = si.label;
        seed_gf[i]  = (float) si.good_frac();
        seed_nv[i]  = si.n_valid;
        seed_nm[i]  = si.n_match;
      }
    }
    g_sh_evt = event_idx;
    const int n_ls = (int) ev->trLayerSearches_.size();
    const int n_ku = (int) ev->trKalmanUpdates_.size();
    const int n_cs = (int) ev->trCandStates_.size();
    int n_add = 0;
    for (const TrHitMatch &hm : ev->trHitMatches_) {
      if ((g_val_search_lite & 2) && (hm.layer < 0 || hm.layer >= 4))
        continue;                                   // pixel barrel only
      ValSearchHit v;
      v.event = event_idx;
      v.layer = hm.layer;  v.hit = hm.hit;  v.mc_match = hm.mc_match;
      v.dphi = hm.dphi;  v.dq = hm.dq;  v.hit_q_half_len = hm.hit_q_half_len;
      v.passed_preselect = hm.passed_preselect;
      v.passed_pqueue = hm.passed_pqueue;  v.rank = hm.sub_rank;
      v.t_hermite = hm.t_hermite;  v.d_plane_h3 = hm.d_plane_h3;
      v.search_id = hm.search_id;
      // signed residuals, from the predicted point on the module plane
      if (hm.layer >= 0 && hm.layer < (int) ev->layerHits_.size() && hm.hit >= 0 &&
          hm.hit < (int) ev->layerHits_[hm.layer].size()) {
        const Hit &h = ev->layerHits_[hm.layer][hm.hit];
        const LayerInfo &li_t = Config::TrkInfo[hm.layer];
        const unsigned int sid_t = h.detIDinLayer();
        if ((int) sid_t < li_t.n_modules())
          v.mod_tilt = (float) std::fabs(li_t.module_info(sid_t).zdir[2]);
        // hit's own phi sigma, from its covariance: sigma_phi^2 =
        // (y^2 exx - 2xy exy + x^2 eyy) / r^4 -- same projection project_sigma()
        // uses for the track, applied to the hit.
        {
          const double hx = h.x(), hy = h.y(), hr2 = hx*hx + hy*hy;
          if (hr2 > 0) {
            const double pv = (hy*hy*h.exx() - 2*hx*hy*h.exy() + hx*hx*h.eyy()) / (hr2*hr2);
            v.hit_phi_sigma = (float) std::sqrt(std::max(0.0, pv));
          }
        }
        const auto &kp = hm.kine_on_plane.pos;
        const bool bar = Config::TrkInfo[hm.layer].is_barrel();
        const double pq = bar ? kp[2] : std::hypot(kp[0], kp[1]);
        const double hq = bar ? h.z()  : std::hypot(h.x(), h.y());
        v.dq_s = (float)(pq - hq);
        double dp = std::atan2((double)kp[1], (double)kp[0]) - std::atan2(h.y(), h.x());
        while (dp >  M_PI) dp -= 2.0*M_PI;
        while (dp <= -M_PI) dp += 2.0*M_PI;
        v.dphi_s = (float) dp;
      }

      if (hm.search_id >= 0 && hm.search_id < n_ls) {
        const TrLayerSearch &ls = ev->trLayerSearches_[hm.search_id];
        v.is_barrel = ls.is_barrel;  v.is_outward = ls.is_outward;
        // dphi_track / dq_track are stated as 3 sigma in the struct comment.
        v.sigma_q_trk   = ls.dq_track   / 3.0f;
        v.sigma_phi_trk = ls.dphi_track / 3.0f;
        v.cov_xx = ls.cov_xx;  v.cov_xy = ls.cov_xy;
        v.cov_yy = ls.cov_yy;  v.cov_zz = ls.cov_zz;
        if (ls.state_id >= 0 && ls.state_id < n_cs) {
          const TrCandState &cs = ev->trCandStates_[ls.state_id];
          v.pt = cs.state.pT();  v.eta = cs.state.momEta();  v.step = cs.step;
          v.theta = cs.state.theta();
          if (cs.meta_id >= 0 && cs.meta_id < (int) ev->trCandMetas_.size()) {
            const TrCandMeta &cm = ev->trCandMetas_[cs.meta_id];
            v.seed = cm.seed;  v.global_seed = cm.global_seed;  v.sim = cm.sim;
            if (cm.global_seed >= 0 && cm.global_seed < (int) seed_sim.size()) {
              v.sim_label = seed_sim[cm.global_seed];
              v.seed_good_frac = seed_gf[cm.global_seed];
              v.seed_n_valid   = seed_nv[cm.global_seed];
              v.seed_n_match   = seed_nm[cm.global_seed];
              if (v.sim_label >= 0 && hm.layer >= 0)
                v.n_sim_hits_in_layer = ev->countSimHitsInLayer(v.sim_label, hm.layer);
            }
          }
        }
      }
      if (hm.kalman_id >= 0 && hm.kalman_id < n_ku) {
        const TrKalmanUpdate &ku = ev->trKalmanUpdates_[hm.kalman_id];
        v.had_kalman = true;  v.accepted = ku.accepted;  v.chi2 = ku.chi2;
        v.residual_x = hm.residual_x;  v.residual_y = hm.residual_y;
        v.residual_z = hm.residual_z;
      }
      // --- exact helix-plane crossing from the candidate's own state
      if (!(g_val_search_lite & 1) &&
          hm.search_id >= 0 && hm.search_id < n_ls && hm.hit >= 0 && hm.layer >= 0) {
        const TrLayerSearch &ls = ev->trLayerSearches_[hm.search_id];
        if (ls.state_id >= 0 && ls.state_id < n_cs) {
          const mkfit::TrackState &st = ev->trCandStates_[ls.state_id].state;
          const auto &lhv = ev->layerHits_[hm.layer];
          if (hm.hit < (int) lhv.size()) {
            const Hit &hit = lhv[hm.hit];
            const LayerInfo &li = Config::TrkInfo[hm.layer];
            const unsigned int sid = hit.detIDinLayer();
            if ((int) sid < li.n_modules()) {
              const auto &mi = li.module_info(sid);
              const double nrm[3] = {mi.zdir[0], mi.zdir[1], mi.zdir[2]};
              const double pnt[3] = {mi.pos[0],  mi.pos[1],  mi.pos[2]};
              const double x0[3] = {st.parameters[0], st.parameters[1], st.parameters[2]};
              const double pt = 1.0 / st.invpT();
              const double p0[3] = {pt*std::cos(st.momPhi()), pt*std::sin(st.momPhi()),
                                    pt / std::tan((double) st.theta())};
              const double inv_k = ((st.charge < 0) ? 0.01 : -0.01) * (double)Const::sol * Config::Bfield;
              const double k = 1.0 / inv_k;
              // the Hermite's own turn angle, from the momentum it reports
              const double ph0 = std::atan2(p0[1], p0[0]);
              const double ph1 = std::atan2((double)hm.kine_on_plane.mom[1],
                                            (double)hm.kine_on_plane.mom[0]);
              double dah = ph1 - ph0;
              while (dah >  M_PI) dah -= 2.0*M_PI;
              while (dah <= -M_PI) dah += 2.0*M_PI;
              v.dalpha_hermite = (float) dah;
              // what sp1 / sp2 should have been
              v.alpha_in  = (float) exact_surface_alpha(x0, p0, k,
                              li.is_barrel() ? li.rin()  : li.zmin(), li.is_barrel(), 1.5);
              v.alpha_out = (float) exact_surface_alpha(x0, p0, k,
                              li.is_barrel() ? li.rout() : li.zmax(), li.is_barrel(), 1.5);
              PlaneRoots R = exact_plane_roots(x0, p0, k, nrm, pnt, 1.5);
              v.n_roots = R.n;
              if (R.n > 0) {
                int i_near = 0, i_small = 0;
                for (int i = 1; i < R.n; ++i) {
                  if (std::fabs(R.a[i] - dah) < std::fabs(R.a[i_near] - dah)) i_near = i;
                  if (std::fabs(R.a[i]) < std::fabs(R.a[i_small])) i_small = i;
                }
                const double ae = R.a[i_near];
                v.dalpha_exact = (float) ae;
                v.wrong_crossing = (i_near != i_small);
                v.alpha_small = (float) R.a[i_small];
                const double sa = std::sin(ae), ca = std::cos(ae);
                const double xe[3] = { x0[0] + k*(p0[0]*sa - p0[1]*(1.0-ca)),
                                       x0[1] + k*(p0[1]*sa + p0[0]*(1.0-ca)),
                                       x0[2] + k*p0[2]*ae };
                v.d_hermite_exact = (float) std::sqrt(
                    std::pow(xe[0]-(double)hm.kine_on_plane.pos[0], 2) +
                    std::pow(xe[1]-(double)hm.kine_on_plane.pos[1], 2) +
                    std::pow(xe[2]-(double)hm.kine_on_plane.pos[2], 2));
                if (std::fabs(ae) > 1e-9) v.rel_ds = (float)((dah - ae) / std::fabs(ae));
              }
            }
          }
        }
      }

      g_sh.push_back(v);
      ++n_add;
    }
    // ---- one row per KALMAN UPDATE: the covariance before and after it
    for (const TrKalmanUpdate &ku : ev->trKalmanUpdates_) {
      if (ku.hit_match_id < 0 || ku.hit_match_id >= (int) ev->trHitMatches_.size()) continue;
      const TrHitMatch &hm = ev->trHitMatches_[ku.hit_match_id];
      ValCovStep c;
      c.event = event_idx;  c.layer = hm.layer;  c.mc_match = hm.mc_match;
      c.accepted = ku.accepted;  c.chi2 = ku.chi2;
      c.state_in = ku.state_id_in;  c.state_out = ku.state_id_out;
      if (hm.search_id >= 0 && hm.search_id < n_ls) {
        const TrLayerSearch &ls = ev->trLayerSearches_[hm.search_id];
        c.is_barrel = ls.is_barrel;
        if (ls.state_id >= 0 && ls.state_id < n_cs) {
          const TrCandState &cst = ev->trCandStates_[ls.state_id];
          c.step = cst.step;  c.pt = cst.state.pT();  c.eta = cst.state.momEta();
        }
      }
      project_sigma(ku.propagated_state, c.is_barrel, c.sig_q_prop, c.sig_phi_prop);
      project_sigma(ku.updated_state,    c.is_barrel, c.sig_q_upd,  c.sig_phi_upd);
      g_cs.push_back(c);
    }

    // ---- one row per LAYER-SEARCH: where was the sim track's own hit?
    // Roll up what the search actually did, keyed by search id.
    struct Roll { int nsc = 0; bool mc_sc = false, mc_pre = false, mc_kal = false;
                  float mc_c2 = 1e30f, best_c2 = 1e30f; };
    std::vector<Roll> roll(n_ls);
    for (const TrHitMatch &hm : ev->trHitMatches_) {
      if (hm.search_id < 0 || hm.search_id >= n_ls) continue;
      Roll &r = roll[hm.search_id];
      ++r.nsc;
      float c2 = -999.f;
      if (hm.kalman_id >= 0 && hm.kalman_id < n_ku)
        c2 = ev->trKalmanUpdates_[hm.kalman_id].chi2;
      if (c2 > -900.f && c2 < r.best_c2) r.best_c2 = c2;
      if (hm.mc_match) { r.mc_sc = true;
        if (hm.passed_preselect) r.mc_pre = true;
        if (hm.kalman_id >= 0) { r.mc_kal = true;
          if (c2 > -900.f && c2 < r.mc_c2) r.mc_c2 = c2; } }
    }
    for (int is = 0; is < n_ls; ++is) {
      const TrLayerSearch &ls = ev->trLayerSearches_[is];
      if (ls.layer < 0 || ls.state_id < 0 || ls.state_id >= n_cs) continue;
      ValSearchMiss m;
      m.event = event_idx;  m.search_id = is;  m.layer = ls.layer;
      m.is_barrel = ls.is_barrel;
      const TrCandState &cs = ev->trCandStates_[ls.state_id];
      m.step = cs.step;  m.pt = cs.state.pT();  m.eta = cs.state.momEta();
      m.n_scanned = roll[is].nsc;  m.mc_scanned = roll[is].mc_sc;
      m.mc_preselect = roll[is].mc_pre;  m.mc_kalman = roll[is].mc_kal;
      if (roll[is].mc_c2   < 1e29f) m.mc_chi2   = roll[is].mc_c2;
      if (roll[is].best_c2 < 1e29f) m.best_chi2 = roll[is].best_c2;
      if (cs.meta_id >= 0 && cs.meta_id < (int) ev->trCandMetas_.size()) {
        const TrCandMeta &cm = ev->trCandMetas_[cs.meta_id];
        if (cm.global_seed >= 0 && cm.global_seed < (int) seed_sim.size())
          m.sim_label = seed_sim[cm.global_seed];
      }
      if (m.sim_label < 0) { g_sm.push_back(m); continue; }

      // Reconstruct the window centre in 3-D. q_center is z (barrel) or r
      // (endcap); the other coordinate is taken at the layer's mid-surface.
      const LayerInfo &li = Config::TrkInfo[ls.layer];
      const double rc = m.is_barrel ? 0.5*(li.rin() + li.rout()) : ls.q_center;
      const double zc = m.is_barrel ? ls.q_center : 0.5*(li.zmin() + li.zmax());
      const double cx = rc*std::cos(ls.phi_center), cy = rc*std::sin(ls.phi_center);

      const Track &st = ev->simTracks_[m.sim_label];
      double best_here = 1e30, best_any = 1e30;
      for (int i = 0; i < st.nTotalHits(); ++i) {
        const HitOnTrack hot = st.getHitOnTrack(i);
        if (hot.index < 0 || hot.layer < 0) continue;
        if (hot.layer >= (int) ev->layerHits_.size()) continue;
        const auto &lv = ev->layerHits_[hot.layer];
        if (hot.index >= (int) lv.size()) continue;
        const Hit &h = lv[hot.index];
        const double d3 = std::sqrt(std::pow(h.x()-cx,2) + std::pow(h.y()-cy,2)
                                  + std::pow(h.z()-zc,2));
        if (d3 < best_any) { best_any = d3; m.near_layer = hot.layer; m.near_d3d = (float) d3; }
        if (hot.layer != ls.layer) continue;
        ++m.n_sim_in_layer;
        double dphi = std::atan2(h.y(), h.x()) - ls.phi_center;
        while (dphi >  M_PI) dphi -= 2.0*M_PI;
        while (dphi <= -M_PI) dphi += 2.0*M_PI;
        if (std::fabs(dphi) < best_here) {
          best_here = std::fabs(dphi);
          m.has_sim_here = true;
          m.sim_dphi = (float) dphi;
          const double q = m.is_barrel ? h.z() : std::hypot(h.x(), h.y());
          m.sim_dq = (float) (q - ls.q_center);
          m.sim_d3d = (float) d3;
          m.sim_dphi_norm = ls.phi_delta > 0 ? (float)(std::fabs(dphi)/ls.phi_delta) : -999.f;
          const double half = 0.5*(ls.q_max - ls.q_min), mid = 0.5*(ls.q_max + ls.q_min);
          m.sim_dq_norm = half > 0 ? (float)((q - mid)/half) : -999.f;
        }
      }
      if (!m.has_sim_here)            m.verdict = 0;
      else if (m.mc_kalman) {
        if (m.mc_chi2 < -900.f)              m.verdict = 5;   // no usable chi2
        else if (m.mc_chi2 >= 30.f)          m.verdict = 5;   // killed by the cut
        else if (m.best_chi2 > -900.f &&
                 m.best_chi2 < m.mc_chi2)    m.verdict = 6;   // outranked
        else                                 m.verdict = 7;   // won
      }
      else if (m.mc_preselect)        m.verdict = 4;
      else if (m.mc_scanned)          m.verdict = 3;
      else if (m.sim_dphi_norm > 1.f || std::fabs(m.sim_dq_norm) > 1.f) m.verdict = 1;
      else                            m.verdict = 2;
      g_sm.push_back(m);
    }

    printf("val_search_event: ev %d -- %d hit matches, %d searches, %d kalman updates\n",
           event_idx, n_add, n_ls, n_ku);
  }

  void val_search_write(const char *out_file) {
    TFile f(out_file, "RECREATE");
    TTree *t = new TTree("search", "backward-search hit matches");
    ValSearchHit v;  t->Branch("h", &v);
    for (const auto &x : g_sh) { v = x; t->Fill(); }
    TTree *t3 = new TTree("cov", "one row per Kalman update: covariance before/after");
    ValCovStep c;  t3->Branch("c", &c);
    for (const auto &x : g_cs) { c = x; t3->Fill(); }

    TTree *t2 = new TTree("miss", "one row per layer-search: where was the sim hit");
    ValSearchMiss m;  t2->Branch("m", &m);
    for (const auto &x : g_sm) { m = x; t2->Fill(); }
    t->Write();  t2->Write();  t3->Write();  f.Close();
    printf("val_search_write: %zu hit matches, %zu searches, %zu kalman updates -> '%s'\n",
           g_sh.size(), g_sm.size(), g_cs.size(), out_file);
  }

  // ==========================================================================
  // chi2 through the backward fit, hit by hit -- the TRACE port of
  // test/an-val-bkfit.C. Reads trBkFitUpdates_ instead of a ValFitHit TTree.
  //
  // Two things come for free that the TTree version had to build by hand:
  // the per-track grouping (records carry state_id_in -> TrCandState.meta_id,
  // so a track IS a meta) and the seed truth (TrCandMeta::global_seed indexes
  // seedTracks_ directly, so no label->truth map and no relabel trap).
  //
  // Step 0 is EXCLUDED throughout: the seed state already sits on the outermost
  // hit's plane, so that update re-uses the hit that produced the state and its
  // chi2 measures nothing. That is also the zero-length propagation flagged as
  // TrBkFitUpdate::degenerate_step.
  namespace {
    struct BTAcc {
      std::vector<double> red;                 // chi2/(2n) per track, all cats
      std::vector<double> red_cat[3], per_hit[3], per_hit_mc[3][2];
      long nfail[3] = {0,0,0}, ntot[3] = {0,0,0}, ntrk[3] = {0,0,0};
      long n_nogf = 0, n_short = 0, n_hits = 0, n_trk = 0, n_ev = 0;
      long n_degen = 0, n_fail_nondegen = 0;
      void clear() { *this = BTAcc(); }
    };
    BTAcc g_bt;
  }

  void val_bkfit_trace_reset() { g_bt.clear(); }

  void val_bkfit_trace_event(const Event *ev) {
    if (ev == nullptr) return;
    ++g_bt.n_ev;
    // group records by meta -- i.e. by track
    std::map<int, std::vector<const TrBkFitUpdate*>> by_meta;
    for (const auto &t : ev->trBkFitUpdates_) {
      ++g_bt.n_hits;
      if (t.step == 0) ++g_bt.n_degen;     // zero-length by construction
      else if (t.fail) ++g_bt.n_fail_nondegen;
      if (t.state_id_in < 0 || t.state_id_in >= (int) ev->trCandStates_.size()) continue;
      by_meta[ev->trCandStates_[t.state_id_in].meta_id].push_back(&t);
    }
    for (auto &kv : by_meta) {
      const int meta_id = kv.first;
      if (meta_id < 0 || meta_id >= (int) ev->trCandMetas_.size()) continue;
      auto &v = kv.second;
      ++g_bt.n_trk;
      std::sort(v.begin(), v.end(),
                [](const TrBkFitUpdate *a, const TrBkFitUpdate *b){ return a->step < b->step; });

      double c2 = 0.0; int n = 0;
      for (const auto *x : v)
        if (x->step > 0 && isFinite(x->chi2)) { c2 += x->chi2; ++n; }
      if (n < 2) { ++g_bt.n_short; continue; }
      const double r = c2 / (2.0 * n);
      g_bt.red.push_back(r);

      // Seed truth, straight off the meta. No label->truth map, no relabel.
      const TrCandMeta &cm = ev->trCandMetas_[meta_id];
      if (cm.global_seed < 0 || cm.global_seed >= (int) ev->seedTracks_.size()) {
        ++g_bt.n_nogf; continue;
      }
      auto si = ev->simInfoForTrack(ev->seedTracks_[cm.global_seed]);
      if (!si.is_set()) { ++g_bt.n_nogf; continue; }
      const float gf = si.good_frac();
      const int sim_label = si.label;   // THE namespace for mc_track_id
      const int cat = gf >= 0.9999f ? 0 : (gf >= 0.8f ? 1 : 2);

      g_bt.red_cat[cat].push_back(r);
      ++g_bt.ntrk[cat];
      for (const auto *x : v) {
        if (x->step == 0 || !isFinite(x->chi2)) continue;
        g_bt.per_hit[cat].push_back(x->chi2);
        ++g_bt.ntot[cat];
        if (x->fail) ++g_bt.nfail[cat];
        g_bt.per_hit_mc[cat][(x->mc_track_id == sim_label) ? 1 : 0].push_back(x->chi2);
      }
    }
  }

  void val_bkfit_trace_report(const char *prefix) {
    FILE *log = fopen(Form("%s.txt", prefix), "w");
    auto P = [&](const char *fmt, ...) {
      va_list a; va_start(a, fmt); vprintf(fmt, a); va_end(a);
      if (log) { va_start(a, fmt); vfprintf(log, fmt, a); va_end(a); }
    };
    auto q = [](std::vector<double> &u, double p) {
      return u.empty() ? 0.0 : u[(size_t)(0.01 * p * (u.size() - 1))];
    };

    P("### %s -- chi2 through the backward fit, FROM THE TRACE GRAPH\n###\n", prefix);
    P("### source     : Event::trBkFitUpdates_, grouped by TrCandState.meta_id.\n");
    P("### definition : per-hit chi2 from kalmanOperationPlaneLocal. Each hit is\n");
    P("###              a 2-D measurement in the module plane, so chi2/hit has\n");
    P("###              expectation 2, median of chi2_2 = 1.386, and the track\n");
    P("###              total ~2*n_hits.\n");
    P("### population : %ld hit records, %ld tracks, %ld events.\n",
      g_bt.n_hits, g_bt.n_trk, g_bt.n_ev);
    P("### step 0     : EXCLUDED -- the seed state already lies on the outermost\n");
    P("###              hit's plane, so that update re-uses the hit that made it.\n");
    P("### step-0 hits : %ld (one per track, the zero-length propagation).\n", g_bt.n_degen);
    P("### prop fails : %ld beyond step 0. NOTE Leonardo's getS carries no\n", g_bt.n_fail_nondegen);
    P("###              bracket, so it DETECTS no failures -- 0 here does not\n");
    P("###              mean the fit did not fail.\n###\n");

    std::sort(g_bt.red.begin(), g_bt.red.end());
    P("tracks with >=2 usable hits: %zu  (%ld too short)\n", g_bt.red.size(), g_bt.n_short);
    P("chi2 / (2 * n_hits), per track:\n");
    P("  p10 %.3g   p25 %.3g   median %.3g   p75 %.3g   p90 %.3g   p99 %.3g   max %.3g\n",
      q(g_bt.red,10), q(g_bt.red,25), q(g_bt.red,50), q(g_bt.red,75),
      q(g_bt.red,90), q(g_bt.red,99), g_bt.red.empty() ? 0.0 : g_bt.red.back());
    P("  expectation ~1.0 (median of chi2_k/k is 1 - 2/(3k), so ~0.96 at 15 dof)\n\n");

    // Categories by the SEED's truth purity, not by chi2 itself: splitting on
    // chi2 would slice the symptom by itself, purity tests a cause.
    const char *cname[3] = {"pure  (gf = 1.0)", "mostly (gf >= 0.8)", "dirty (gf < 0.8)"};
    P("tracks with no truth match at all: %ld\n\n", g_bt.n_nogf);
    P("PER TRACK, chi2/(2*n_hits) -- expectation 1.0\n");
    P("  %-20s %7s %9s %9s %9s %9s %9s\n", "category", "tracks", "p25", "median", "p75", "p90", "max");
    for (int c = 0; c < 3; ++c) {
      auto &u = g_bt.red_cat[c]; std::sort(u.begin(), u.end());
      P("  %-20s %7ld %9.3g %9.3g %9.3g %9.3g %9.3g\n", cname[c], g_bt.ntrk[c],
        q(u,25), q(u,50), q(u,75), q(u,90), u.empty() ? 0.0 : u.back());
    }
    P("\nPER HIT, chi2 -- expectation 2, median of chi2_2 = 1.386\n");
    P("  %-20s %7s %9s %9s %9s %9s %9s\n", "category", "hits", "p25", "median", "p75", "p90", "max");
    for (int c = 0; c < 3; ++c) {
      auto &u = g_bt.per_hit[c]; std::sort(u.begin(), u.end());
      P("  %-20s %7zu %9.3g %9.3g %9.3g %9.3g %9.3g\n", cname[c], u.size(),
        q(u,25), q(u,50), q(u,75), q(u,90), u.empty() ? 0.0 : u.back());
    }
    P("\nPER HIT, split by whether THAT HIT's mcTrackID matches the track's sim label\n");
    P("  %-20s %8s %9s %9s | %8s %9s %9s\n", "category",
      "n match", "med", "p90", "n other", "med", "p90");
    for (int c = 0; c < 3; ++c) {
      auto &m = g_bt.per_hit_mc[c][1]; std::sort(m.begin(), m.end());
      auto &o = g_bt.per_hit_mc[c][0]; std::sort(o.begin(), o.end());
      P("  %-20s %8zu %9.3g %9.3g | %8zu %9.3g %9.3g\n", cname[c],
        m.size(), q(m,50), q(m,90), o.size(), q(o,50), q(o,90));
    }
    P("\n  propagation failures: pure %ld/%ld, mostly %ld/%ld, dirty %ld/%ld\n",
      g_bt.nfail[0], g_bt.ntot[0], g_bt.nfail[1], g_bt.ntot[1], g_bt.nfail[2], g_bt.ntot[2]);
    if (log) fclose(log);
  }



  void val_bkfit_err_scale(float s) {
    g_bkfit_err_scale = s;
    printf("val_bkfit_err_scale: %g  (variance scale on the input covariance; %g in sigma)\n",
           s, std::sqrt(s));
  }

  // Toggle PF_apply_material on the backward-fit propagation only. Returns the
  // previous state. Diagnostic: material inflates the propagated covariance, so
  // turning it off must RAISE chi2; how much says how big the material term is.
  void val_bkfit_material(bool on) {
    auto &pc = const_cast<PropagationConfig&>(Config::TrkInfo.prop_config());
    pc.backward_fit_pflags.apply_material = on;
    printf("val_bkfit_material: backward_fit_pflags.apply_material = %d\n", (int) on);
  }



  //============================================================================
  // val_sister -- how often does a hit have a SISTER HIT in the partner
  // sub-layer of the same physical CMS layer, and how far away is it?
  //
  // The question this answers (maintainer, 2026-09-21): "not all PS or 2S hits
  // will have sister hits ... but most probably should? I don't know what the
  // efficiency implications would be." It decides whether in-layer processing
  // may ANCHOR on one sensor (e.g. pre-select the pair on the finer-q P hit) or
  // must treat the two symmetrically: anchoring is only safe if the anchor
  // sensor is almost always present.
  //
  // Truth direction is sim -> rec: the sim track's OWN hit list, which is built
  // from rhIdxs at ntuple-writing time and is NOT gated by the bestTkIdx
  // arbitration. So this does not inherit the percent-level truth-link loss
  // measured for the rec -> sim direction.
  //
  // "Sister" here means only "a hit in the partner sub-layer". Whether it sits
  // on the BONDED partner module or on a different (phi-overlapping) module of
  // the same layer is not knowable from the data today -- ModuleInfo carries no
  // partner link -- so it is separated geometrically instead, by the 3-D
  // distance between the two hits. A bonded stack is ~1.6-4 mm thick; an overlap
  // partner is a module width away.
  //============================================================================

  namespace {
    struct SisterGroup {
      const char *name;
      long n_cross = 0;      // sim-track crossings with >= 1 hit in the pair
      long n_a_only = 0, n_b_only = 0, n_both = 0;
      long n_mult[9] = {0,0,0,0,0,0,0,0,0};   // hits in the pair: 1 .. 8, >=9
      std::vector<float> dist;        // 3-D distance, closest A-B pair
      std::vector<float> hl_a, hl_b;  // q half-lengths, to show the asymmetry
    };
    std::vector<SisterGroup> g_sis;
    long g_sis_ntrk = 0;
    bool g_sis_mcfilter = true;

    int sister_group_of(int lay) {
      if (lay >=  4 && lay <=  9) return 0;   // TBPS  (P/S)
      if (lay >= 10 && lay <= 15) return 1;   // TB2S  (2S/2S)
      if (lay >= 28 && lay <= 37) return 2;   // TEDD+
      if (lay >= 50 && lay <= 59) return 3;   // TEDD-
      return -1;
    }
  }

  void val_sister_reset() {
    g_sis.assign(5, SisterGroup());
    g_sis[0].name = "TBPS  4-9   (P + S)";
    g_sis[1].name = "TB2S  10-15 (2S + 2S)";
    g_sis[2].name = "TEDD+ 28-37";
    g_sis[3].name = "TEDD- 50-59";
    g_sis[4].name = "TEDD inner r<65 (PS region)";
    g_sis_ntrk = 0;
  }

  void val_sister_mcfilter(bool on) {
    g_sis_mcfilter = on;
    printf("val_sister_mcfilter: require hit mcTrackID == sim label: %d\n", (int) on);
  }

  void val_sister_event(const Event *ev, float pt_min) {
    if (g_sis.empty()) val_sister_reset();

    for (const auto &st : ev->simTracks_) {
      if (st.pT() < pt_min) continue;
      ++g_sis_ntrk;

      // Bucket this track's rec hits by mkFit layer.
      std::map<int, std::vector<int>> by_layer;   // layer -> hit indices
      const int nh = st.nTotalHits();
      for (int i = 0; i < nh; ++i) {
        int hi = st.getHitIdx(i), hl = st.getHitLyr(i);
        if (hi < 0 || hl < 0) continue;
        if (hl >= (int) ev->layerHits_.size()) continue;
        if (hi >= (int) ev->layerHits_[hl].size()) continue;
        // Truth BINDING, not truth similarity: 27 % of the hits sitting on a
        // sim track belong to a different particle (delta rays, neighbours).
        // Without this a delta-ray hit in sub-layer B makes "both" true when the
        // track itself only crossed A, and it is what puts 120 cm entries in the
        // sister-distance tail. This is a truth decision independent of any
        // residual or chi2, so it does not drag the survivors toward zero the
        // way a chi2 cut would.
        if (g_sis_mcfilter) {
          unsigned int mch = ev->layerHits_[hl][hi].mcHitID();
          if (mch >= ev->simHitsInfo_.size()) continue;
          if (ev->simHitsInfo_[mch].mcTrackID() != st.label()) continue;
        }
        by_layer[hl].push_back(hi);
      }

      // Walk the pairs. Even layer = A, odd = B; post-10781bd48eb the EVEN one
      // is P in a PS stack (verified from the data: median sqrt(3 ezz) is
      // 0.037/0.042/0.053 cm on L4/6/8 against 0.589/0.676/0.847 on L5/7/9).
      for (int la = 4; la <= 58; la += 2) {
        int g = sister_group_of(la);
        if (g < 0 || sister_group_of(la + 1) != g) continue;
        auto ia = by_layer.find(la), ib = by_layer.find(la + 1);
        const bool ha = ia != by_layer.end(), hb = ib != by_layer.end();
        if (!ha && !hb) continue;

        const int na = ha ? (int) ia->second.size() : 0;
        const int nb = hb ? (int) ib->second.size() : 0;

        auto tally = [&](SisterGroup &G) {
          ++G.n_cross;
          if (ha && hb) ++G.n_both; else if (ha) ++G.n_a_only; else ++G.n_b_only;
          int m = na + nb;
          ++G.n_mult[std::min(m, 9) - 1];
          // Same branch LayerOfHits::registerHit() uses: ezz in the barrel, the
          // transverse trace in the endcap. Using ezz everywhere reports the
          // module THICKNESS for a disc, which is not the q extent at all.
          const bool brl = (la < 16);
          auto qhl = [&](const Hit &H) {
            return std::sqrt(3.0f * (brl ? H.ezz() : H.exx() + H.eyy()));
          };
          if (ha) for (int x : ia->second) G.hl_a.push_back(qhl(ev->layerHits_[la][x]));
          if (hb) for (int x : ib->second) G.hl_b.push_back(qhl(ev->layerHits_[la+1][x]));
          if (ha && hb) {
            float best = 1e9f;
            for (int x : ia->second) for (int y : ib->second) {
              const Hit &A = ev->layerHits_[la][x], &B = ev->layerHits_[la+1][y];
              float dx = A.x()-B.x(), dy = A.y()-B.y(), dz = A.z()-B.z();
              best = std::min(best, std::sqrt(dx*dx + dy*dy + dz*dz));
            }
            G.dist.push_back(best);
          }
        };

        tally(g_sis[g]);
        // TEDD is radially mixed: PS inside (r out to ~65 cm), 2S outside. Split
        // it on the radius of the first hit found, so the PS region of the discs
        // can be compared against TBPS.
        if (g == 2 || g == 3) {
          float r = ha ? ev->layerHits_[la][ia->second[0]].r()
                       : ev->layerHits_[la+1][ib->second[0]].r();
          if (r < 65.0f) tally(g_sis[4]);
        }
      }
    }
  }

  void val_sister_report() {
    auto q = [](std::vector<float> &v, float f) {
      if (v.empty()) return -1.0f;
      std::sort(v.begin(), v.end());
      return v[std::min(v.size() - 1, (size_t)(f * v.size()))];
    };
    printf("\n=== val_sister: sister hits in the partner sub-layer ===\n");
    printf("sim tracks scanned: %ld\n", g_sis_ntrk);
    printf("%-28s %8s | %7s %7s %7s | %s\n", "group", "crossings",
           "A only", "B only", "both", "hits in the pair: 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 / 9+");
    for (auto &G : g_sis) {
      if (!G.name || G.n_cross == 0) continue;
      double n = G.n_cross;
      printf("%-28s %8ld | %6.1f%% %6.1f%% %6.1f%% |", G.name, G.n_cross,
             100.0 * G.n_a_only / n, 100.0 * G.n_b_only / n, 100.0 * G.n_both / n);
      for (int m = 0; m < 9; ++m)
        printf(" %5.2f", 100.0 * G.n_mult[m] / n);
      printf("\n");
    }
    printf("\n%-28s %8s %8s %8s %8s | %10s %10s\n", "group", "d p10", "d p50",
           "d p90", "d max", "q_hl A", "q_hl B");
    for (auto &G : g_sis) {
      if (!G.name || G.n_cross == 0) continue;
      // q half-length as the code computes it: sqrt(3 * ezz) for a strip layer.
      std::vector<float> a(G.hl_a), b(G.hl_b);
      printf("%-28s %8.4f %8.4f %8.4f %8.4f | %10.4f %10.4f\n", G.name,
             q(G.dist, 0.10f), q(G.dist, 0.50f), q(G.dist, 0.90f),
             G.dist.empty() ? -1.0f : *std::max_element(G.dist.begin(), G.dist.end()),
             q(a, 0.50f), q(b, 0.50f));
    }
    printf("(distances in cm, closest A-B pair; q_hl = median sqrt(3*ezz) in cm)\n\n");
  }


  //============================================================================
  // val_qbins -- what the q BINNING actually costs, per layer.
  //
  // phase2QBins gives q_bin = 6.0 cm to EVERY outer-tracker barrel layer, P and
  // S alike (MkFitGeometryESProducer.cc, carrying its own "TODO: Review these
  // numbers"). The two sensors' q extents differ by ~16x, so one number cannot
  // be right for both. This reports, per layer: the q-bin span actually opened
  // (q2 - q1 from MkBinLimits), the q window that span is covering, and how many
  // hits were scanned against how many survived pre-selection.
  //
  // The ratio that matters is (bin span in cm) / (window width in cm): it is the
  // over-scan factor, i.e. how much of the q range pulled out of the binnor the
  // cut was never going to accept.
  //============================================================================

  namespace {
    struct QBinRow {
      long n = 0;
      double span_bins = 0, win_cm = 0, scanned = 0, presel = 0;
    };
    std::map<int, QBinRow> g_qb;
  }

  void val_qbins_reset() { g_qb.clear(); }

  void val_qbins_event(const Event *ev) {
    for (const auto &ls : ev->trLayerSearches_) {
      if (ls.layer < 0) continue;
      auto &R = g_qb[ls.layer];
      ++R.n;
      // q1/q2 are binnor bin indices; the walk is `for (qi = q1; qi != q2; ++qi)`.
      int nb = (int) ls.q2 - (int) ls.q1;
      if (nb < 0) nb += 1 << 16;          // wrap, defensive
      R.span_bins += nb;
      // The q window the cut will actually use, full width. dq_track is 3 sigma.
      R.win_cm  += (double) (ls.q_max - ls.q_min);
      R.scanned += ls.n_hits_scanned;
      R.presel  += ls.n_hits_presel;
    }
  }

  void val_qbins_report() {
    printf("\n=== val_qbins: q binning vs the q window actually used ===\n");
    printf("%5s %8s %10s %10s %10s %10s %9s %9s\n", "layer", "searches", "q_bin[cm]",
           "bins", "binspan cm", "window cm", "over-scan", "scan/presel");
    for (auto &[lay, R] : g_qb) {
      if (R.n < 100) continue;
      const float qb = Config::TrkInfo[lay].q_bin();
      double bins = R.span_bins / R.n;
      double bspan = bins * qb;
      double win = R.win_cm / R.n;
      printf("%5d %8ld %10.2f %10.2f %10.2f %10.3f %9.1f %9.2f\n",
             lay, R.n, qb, bins, bspan, win,
             win > 0 ? bspan / win : -1.0,
             R.presel > 0 ? R.scanned / R.presel : -1.0);
    }
    printf("over-scan = (q range pulled from the binnor) / (q window the cut uses)\n\n");
  }

  // ==========================================================================
  // val_eff -- per-SIM-TRACK efficiency, resolved in |eta|, pT and hits-per-layer.
  //
  // WHY THIS AND NOT THE STANDARD COUNTERS. Everything measured for the in-layer
  // combinatorial search so far is a scalar summed over a run, so none of it says
  // WHERE the gain lands. The two decisions waiting on that -- whether the search
  // should default ON and at which maxCandsPerSeed, and whether the likelihood
  // score's -ln(rho) occupancy term behaves per region -- are both regional.
  //
  // THE METRIC, and why each piece is safe:
  //  - Association is `TrackExtra::setMCTrackIDInfo`, i.e. `2*mccount >= nCandHits`
  //    over the non-seed hits -- exactly what quality-val's "found tracks" counts.
  //    A WRONG extra hit raises the denominator only, so a gain here cannot be
  //    bought by taking more hits. `nH >= 80 %` is NOT used anywhere: it tests raw
  //    reco HITS against sim LAYERS and therefore rewards the thing under test.
  //  - The denominator is SIM tracks, not reco tracks, and it is restricted to sim
  //    tracks that a seed actually points at (`Event::simInfoForCurrentSeed`). That
  //    removes the seeding efficiency, which is common to both configurations and
  //    would otherwise dilute the regional shape without changing the difference.
  //  - Duplicates and fakes come off the same pass, so the three move together and
  //    an efficiency gain paid for in fakes cannot hide.
  //
  // PAIRING. Same events, same seeds, same denominator in every configuration, so
  // the error bar quoted on a difference is the spread of the per-EVENT difference
  // in the numerator, not Poisson on the total. The denominator is checked to be
  // identical across configurations and a mismatch is reported.
  namespace {
    // Axis 0: |eta| of the sim track.  Axis 1: its pT.  Axis 2: hits per layer,
    // i.e. how much overlap the sim track's own hit content offers -- the axis
    // along which the in-layer combinatorial is supposed to pay.
    constexpr int VE_NAX = 3;
    constexpr int VE_NB  = 12;
    const int   ve_nbin[VE_NAX] = {12, 10, 6};
    const char *ve_axname[VE_NAX] = {"|eta|", "pT [GeV]", "sim hits / layer"};

    // THE SELECTION IS CMSSW's MTV CONVENTION: |eta| < 2.5 and pT > 0.9, with the
    // cut on the plotted variable RELEASED for that variable's own plot. So the
    // eta axis carries every selected track of any pT above 0.2, the pT axis
    // carries every selected track of any |eta| below 3, and everything else --
    // the totals, the regions, the hits-per-layer axis, the resolution -- uses
    // both cuts. Quoting one efficiency for "the sample" and another for a bin of
    // its own plot is the convention, not an inconsistency.
    constexpr float VE_ETA_CUT = 2.5f;
    constexpr float VE_PT_CUT  = 0.9f;

    const float ve_pt_edge[11] = {0.2f, 0.3f, 0.5f, 0.7f, 0.9f, 1.2f,
                                  1.6f, 2.5f, 4.0f, 10.0f, 1e9f};
    const char *ve_hpl_lab[6]  = {"= 1.00", "1.0-1.1", "1.1-1.2", "1.2-1.35", "1.35-1.5", "> 1.5"};

    int ve_bin_eta(float ae) { int b = (int)(ae / 0.25f); return (b < 0 || b > 11) ? -1 : b; }
    int ve_bin_pt(float pt) {
      if (pt < ve_pt_edge[0]) return -1;
      for (int b = 0; b < 10; ++b) if (pt < ve_pt_edge[b+1]) return b;
      return 9;
    }
    int ve_bin_hpl(float r) {
      if (r < 1.0001f) return 0;
      if (r <= 1.1f)  return 1;
      if (r <= 1.2f)  return 2;
      if (r <= 1.35f) return 3;
      if (r <= 1.5f)  return 4;
      return 5;
    }
    // Regions, for the per-region read the likelihood score needs. Barrel /
    // transition / endcap by |eta| of the sim track, matching the eta ranges the
    // phase-2 seed partitioner uses closely enough to name them.
    const char *ve_regname[3] = {"barrel   |eta|<0.9", "transition 0.9-1.7", "endcap    >1.7"};
    int ve_region(float ae) { return ae < 0.9f ? 0 : (ae < 1.7f ? 1 : 2); }

    struct VeBins { long b[VE_NAX][VE_NB] = {}; long reg[3] = {}; long tot = 0; };
    struct VeRes { int evt; int lbl; int reg; float r; };
    // Same shape as VeBins but summing a weight, for mean track length. Kept
    // separate rather than templating VeBins, which is counted in longs and is
    // the thing every paired statistic runs on.
    struct VeSums { double b[VE_NAX][VE_NB] = {}; double reg[3] = {}; double tot = 0.0; };

    struct VeCfg {
      std::string name;
      VeBins den, num, dup;        // sim-binned; den is the MTV denominator
      // TRACK LENGTH, over the found sim tracks (so the denominator is num).
      // n_found is every hit on the best-matched reco track, seed hits included;
      // n_match is the subset whose mcTrackID is that sim track, EXCLUDING the
      // seed hits, since setMCTrackIDInfo skips them. n_sim is the sim track's
      // own valid-hit count, which is what both should be read against. The
      // matched one is the safe counter: a wrong extra hit cannot raise it.
      VeSums n_found, n_match, n_sim;
      VeBins dens;                 // ... of which a seed points at them
      VeBins reco, fake;           // reco-binned (axes 0,1 and region only)
      // Seed quality, per region. The search cannot find what it is not seeded
      // for, and MTV's "central" requirement is in practice imposed by the seeds
      // rather than by us -- so the seeding efficiency and the seed purity are
      // the ceiling every number below sits under, and belong in the same report.
      long n_seed[3] = {}, n_seed_pure[3] = {}, n_seed_on_sel[3] = {};
      double sum_seed_gf[3] = {};
      // What the seeds are MADE OF, which is as close as the .bin gets to naming
      // the seeding algorithm: the file carries the track algorithm and the hits,
      // not the producer. Four pixel hits is a pixel quadruplet either way --
      // Patatrack and the standard chain differ in the fit, not in the hit
      // content -- so this bounds the question rather than settling it.
      long sum_seed_hits[3] = {}, sum_seed_pix[3] = {};
      long n_ev = 0;
      std::vector<VeBins> ev_num, ev_den, ev_fake, ev_reco, ev_dup;
      // d(pT)/pT of the best-matched reco track of each found sim track, one row
      // per found sim track and keyed by (event, sim label) so the report can
      // restrict every configuration to the tracks they ALL found. That
      // restriction is not a refinement -- unrestricted, a configuration that
      // finds more tracks is measured on a harder population, and the difference
      // that produces is larger than the effect being looked for.
      std::vector<VeRes> res;
    };

    std::vector<VeCfg> g_ve;
    std::string g_ve_ref;
    FILE *g_ve_log = nullptr;

    int ve_printf(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
    int ve_printf(const char *fmt, ...) {
      va_list ap;  va_start(ap, fmt);  int n = vprintf(fmt, ap);  va_end(ap);
      if (g_ve_log) { va_start(ap, fmt); vfprintf(g_ve_log, fmt, ap); va_end(ap); }
      return n;
    }

    VeCfg &ve_cfg(const char *name) {
      for (auto &c : g_ve) if (c.name == name) return c;
      g_ve.push_back(VeCfg());  g_ve.back().name = name;  return g_ve.back();
    }

    // ae, pt gate which axes this track enters, per the MTV convention above:
    // the eta axis wants the pT cut only, the pT axis the eta cut only, and
    // everything else both.
    void ve_fill(VeBins &v, int be, int bp, int bh, int reg, float ae, float pt) {
      const bool ok_eta = ae < VE_ETA_CUT, ok_pt = pt > VE_PT_CUT;
      if (be >= 0 && ok_pt) ++v.b[0][be];
      if (bp >= 0 && ok_eta) ++v.b[1][bp];
      if (!(ok_eta && ok_pt)) return;
      if (bh >= 0) ++v.b[2][bh];
      if (reg >= 0) ++v.reg[reg];
      ++v.tot;
    }
    void ve_fill_w(VeSums &v, int be, int bp, int bh, int reg, float ae, float pt, double w) {
      const bool ok_eta = ae < VE_ETA_CUT, ok_pt = pt > VE_PT_CUT;
      if (be >= 0 && ok_pt) v.b[0][be] += w;
      if (bp >= 0 && ok_eta) v.b[1][bp] += w;
      if (!(ok_eta && ok_pt)) return;
      if (bh >= 0) v.b[2][bh] += w;
      if (reg >= 0) v.reg[reg] += w;
      v.tot += w;
    }
    void ve_add(VeBins &a, const VeBins &b) {
      for (int x = 0; x < VE_NAX; ++x) for (int i = 0; i < VE_NB; ++i) a.b[x][i] += b.b[x][i];
      for (int r = 0; r < 3; ++r) a.reg[r] += b.reg[r];
      a.tot += b.tot;
    }
    // Paired: mean and sigma-on-the-sum of the per-event difference num(A) - num(B).
    void ve_paired(const std::vector<long> &a, const std::vector<long> &b, double &sum, double &sig) {
      const size_t n = std::min(a.size(), b.size());
      sum = 0.0;  sig = 0.0;
      if (n < 2) return;
      double s = 0.0, s2 = 0.0;
      for (size_t i = 0; i < n; ++i) { const double d = (double)a[i] - (double)b[i]; s += d; s2 += d*d; }
      sum = s;
      const double mean = s / n;
      const double var = (s2 - n*mean*mean) / (n - 1);
      sig = std::sqrt(std::max(0.0, var) * n);   // sigma on the SUM of the n differences
    }
  }

  void val_eff_reset() { g_ve.clear(); g_ve_ref.clear(); }
  void val_eff_ref(const char *cfg) { g_ve_ref = cfg; }

  void val_eff_event(const Event *ev, const char *cfg) {
    if (ev == nullptr) return;
    VeCfg &C = ve_cfg(cfg);
    VeBins e_den, e_num, e_dup, e_reco, e_fake, e_dens;

    // ---- seeds: which sim tracks did the search actually get a chance at, and
    // where is each seed, so its hits can be excluded from the association count.
    std::map<int, int> seed_by_label;      // seed label -> index in currentSeedTracks()
    std::set<int> seeded_sim;
    std::map<int, int> n_seed_for_sim;     // sim label -> how many seeds point at it
    const TrackVec *seeds = nullptr;
    try { seeds = &ev->currentSeedTracks(); } catch (...) { seeds = nullptr; }
    if (seeds) {
      for (int i = 0; i < (int) seeds->size(); ++i) {
        seed_by_label.emplace((*seeds)[i].label(), i);
        const auto sifh = ev->simInfoForCurrentSeed(i);
        const int sl = sifh.label;
        if (sl >= 0 && sl < (int) ev->simTracks_.size()) {
          seeded_sim.insert(sl);
          ++n_seed_for_sim[sl];
        }
        const int sr = ve_region(std::abs((*seeds)[i].momEta()));
        if (sr >= 0 && sr < 3) {
          ++C.n_seed[sr];
          C.sum_seed_gf[sr] += sifh.good_frac();
          if (sifh.good_frac() > 0.999f) ++C.n_seed_pure[sr];
          const Track &sd = (*seeds)[i];
          for (int h = 0; h < sd.nTotalHits(); ++h) {
            const HitOnTrack hot = sd.getHitOnTrack(h);
            if (hot.index < 0 || hot.layer < 0) continue;
            ++C.sum_seed_hits[sr];
            if (Config::TrkInfo[hot.layer].is_pixel()) ++C.sum_seed_pix[sr];
          }
        }
      }
    }

    // ---- reco side: association exactly as quality-val defines it.
    std::map<int, int> n_assoc;            // sim label -> number of reco tracks on it
    struct VeBest { float pt = 0.0f; int n_match = -1; int n_found = 0; };
    std::map<int, VeBest> best;            // sim label -> its best-matched reco track
    for (const Track &c : ev->candidateTracks_) {
      TrackExtra extra(c.label());
      auto si = seed_by_label.find(c.label());
      if (seeds && si != seed_by_label.end())
        extra.findMatchingSeedHits(c, (*seeds)[si->second], ev->layerHits_);
      extra.setMCTrackIDInfo(c, ev->layerHits_, ev->simHitsInfo_, ev->simTracks_, false, false);
      const int mc = extra.mcTrackID();

      const float rae = std::abs(c.momEta());
      const int rbe = ve_bin_eta(rae), rbp = ve_bin_pt(c.pT()), rreg = ve_region(rae);
      ve_fill(e_reco, rbe, rbp, -1, rreg, rae, c.pT());
      if (mc < 0 || mc >= (int) ev->simTracks_.size())
        ve_fill(e_fake, rbe, rbp, -1, rreg, rae, c.pT());
      else {
        ++n_assoc[mc];
        // Keep the best-matched reco track per sim track, so a duplicate does not
        // get to vote twice on the resolution.
        auto &b = best[mc];
        if (extra.nHitsMatched() > b.n_match)
          b = {c.pT(), extra.nHitsMatched(), c.nFoundHits()};
      }
    }

    // ---- sim side. The denominator is EVERY selected sim track, not only the
    // seeded ones, which is what MTV means by efficiency. The seeded subset is
    // counted alongside so the seeding ceiling is visible rather than assumed.
    //
    // Selection, CMSSW TrackingParticleSelector's shape: findable, the production
    // vertex central (tip < 3.5 cm, lip < 30 cm), and at least 4 distinct layers,
    // since a seed needs four and a sim track with fewer is not reconstructible
    // by this algorithm at all. The eta and pT cuts are applied per axis inside
    // ve_fill(), so each plot releases the cut on its own variable.
    const int nsim = (int) ev->simTracks_.size();
    for (int L = 0; L < nsim; ++L) {
      const Track &st = ev->simTracks_[L];
      const float ae = std::abs(st.momEta()), pt = st.pT();
      if (!st.isFindable()) continue;
      if (std::hypot(st.x(), st.y()) > 3.5f || std::abs(st.z()) > 30.0f) continue;
      if (ae >= 3.0f || pt < ve_pt_edge[0]) continue;
      const int nlay = st.nUniqueLayers();
      if (nlay < 4) continue;
      int nval = 0;
      for (int i = 0; i < st.nTotalHits(); ++i)
        if (st.getHitOnTrack(i).index >= 0) ++nval;
      const int be = ve_bin_eta(ae), bp = ve_bin_pt(pt), reg = ve_region(ae);
      const int bh = ve_bin_hpl((float) nval / (float) nlay);
      ve_fill(e_den, be, bp, bh, reg, ae, pt);
      if (seeded_sim.count(L)) {
        ve_fill(e_dens, be, bp, bh, reg, ae, pt);
        if (reg >= 0 && ae < VE_ETA_CUT && pt > VE_PT_CUT)
          C.n_seed_on_sel[reg] += n_seed_for_sim[L];
      }
      auto it = n_assoc.find(L);
      if (it != n_assoc.end()) {
        ve_fill(e_num, be, bp, bh, reg, ae, pt);
        for (int d = 1; d < it->second; ++d) ve_fill(e_dup, be, bp, bh, reg, ae, pt);
        auto bi = best.find(L);
        if (bi != best.end()) {
          ve_fill_w(C.n_found, be, bp, bh, reg, ae, pt, bi->second.n_found);
          ve_fill_w(C.n_match, be, bp, bh, reg, ae, pt, std::max(0, bi->second.n_match));
          ve_fill_w(C.n_sim,   be, bp, bh, reg, ae, pt, nval);
          if (pt > 0.0f && ae < VE_ETA_CUT && pt > VE_PT_CUT)
            C.res.push_back({ev->evtID(), L, reg, (bi->second.pt - pt) / pt});
        }
      }
    }

    ve_add(C.den, e_den);  ve_add(C.num, e_num);  ve_add(C.dup, e_dup);
    ve_add(C.dens, e_dens);
    ve_add(C.reco, e_reco); ve_add(C.fake, e_fake);
    C.ev_den.push_back(e_den);  C.ev_num.push_back(e_num);  C.ev_dup.push_back(e_dup);
    C.ev_reco.push_back(e_reco); C.ev_fake.push_back(e_fake);
    ++C.n_ev;
  }

  namespace {
    // Pull one bin's per-event series out of a config, for the paired statistics.
    std::vector<long> ve_series(const std::vector<VeBins> &ev, int ax, int bin) {
      std::vector<long> v;  v.reserve(ev.size());
      for (const auto &e : ev) v.push_back(ax < 0 ? e.tot : (ax == 3 ? e.reg[bin] : e.b[ax][bin]));
      return v;
    }
    std::string ve_binlabel(int ax, int b) {
      char s[32];
      if (ax == 0) snprintf(s, sizeof(s), "%.2f-%.2f", 0.25*b, 0.25*(b+1));
      else if (ax == 1) {
        if (b == 9) snprintf(s, sizeof(s), "> 10");
        else snprintf(s, sizeof(s), "%.1f-%.1f", ve_pt_edge[b], ve_pt_edge[b+1]);
      }
      else snprintf(s, sizeof(s), "%s", ve_hpl_lab[b]);
      return s;
    }

    // Mean track length per bin. which = 0 reco hits (seed included), 1 matched
    // hits (seed excluded, the safe counter), 2 the sim track's own hits. The
    // denominator is the FOUND sim tracks, so it is a property of the tracks a
    // configuration reconstructed and moves with the population -- read the
    // matched row, and read it against the sim row in the same column.
    void ve_len_table(int ax, const VeCfg *ref, int which) {
      const int nb = (ax == 3) ? 3 : ve_nbin[ax];
      const char *wn[3] = {"reco hits (incl. seed)", "MATCHED hits (excl. seed)",
                           "sim track's own hits"};
      ve_printf("\n--- mean %s vs %s ---\n", wn[which],
                ax == 3 ? "region" : ve_axname[ax]);
      ve_printf("%-20s %9s", ax == 3 ? "region" : "bin", "found");
      for (const auto &c : g_ve) ve_printf(" | %-10.10s", c.name.c_str());
      ve_printf("\n");
      for (int b = 0; b < nb; ++b) {
        const long dn = (ax == 3) ? ref->num.reg[b] : ref->num.b[ax][b];
        if (dn < 20) continue;
        std::string lab = (ax == 3) ? ve_regname[b] : ve_binlabel(ax, b);
        ve_printf("%-20s %9ld", lab.c_str(), dn);
        for (const auto &c : g_ve) {
          const VeSums &S = (which == 0) ? c.n_found : (which == 1 ? c.n_match : c.n_sim);
          const long n = (ax == 3) ? c.num.reg[b] : c.num.b[ax][b];
          const double v = (ax == 3) ? S.reg[b] : S.b[ax][b];
          ve_printf(" |   %8.3f ", n ? v/n : 0.0);
        }
        ve_printf("\n");
      }
    }

    // One resolved table: efficiency per bin for every configuration, and the
    // paired difference of each against the reference.
    void ve_table(int ax, const VeCfg *ref) {
      const int nb = (ax == 3) ? 3 : ve_nbin[ax];
      ve_printf("\n--- efficiency vs %s ---\n", ax == 3 ? "region" : ve_axname[ax]);
      ve_printf("%-20s %9s", ax == 3 ? "region" : "bin", "sim trks");
      for (const auto &c : g_ve) ve_printf(" | %-10.10s", c.name.c_str());
      ve_printf("\n");
      for (int b = 0; b < nb; ++b) {
        const long den = (ax == 3) ? ref->den.reg[b] : ref->den.b[ax][b];
        if (den < 20) continue;
        std::string lab = (ax == 3) ? ve_regname[b] : ve_binlabel(ax, b);
        ve_printf("%-20s %9ld", lab.c_str(), den);
        for (const auto &c : g_ve) {
          const long d = (ax == 3) ? c.den.reg[b] : c.den.b[ax][b];
          const long n = (ax == 3) ? c.num.reg[b] : c.num.b[ax][b];
          if (&c == ref) ve_printf(" |   %6.2f%% ", d ? 100.0*n/d : 0.0);
          else {
            double sum, sig;
            ve_paired(ve_series(c.ev_num, ax == 3 ? 3 : ax, b),
                      ve_series(ref->ev_num, ax == 3 ? 3 : ax, b), sum, sig);
            const double dp = d ? 100.0*sum/d : 0.0;          // difference in points
            const double sp = d ? 100.0*sig/d : 0.0;
            ve_printf(" | %+6.2f%s%-3.3s", dp,
                      sp > 0 && std::abs(dp) > 3*sp ? "*" : " ",
                      sp > 0 ? (std::abs(dp) > 3*sp ? "sig" : "") : "");
          }
        }
        ve_printf("\n");
      }
      ve_printf("  reference column is the absolute efficiency; the others are the\n"
                "  PAIRED difference in points, * = |delta| > 3 sigma of the per-event spread.\n");
    }

    void ve_ratio_table(int ax, const VeCfg *ref, bool fakes) {
      const int nb = (ax == 3) ? 3 : ve_nbin[ax];
      ve_printf("\n--- %s vs %s ---\n",
                fakes ? "FAKE fraction, per RECO track" : "EXTRA reco tracks per found SIM track",
                ax == 3 ? "region" : ve_axname[ax]);
      ve_printf("%-20s", ax == 3 ? "region" : "bin");
      for (const auto &c : g_ve) ve_printf(" | %-10.10s", c.name.c_str());
      ve_printf("\n");
      for (int b = 0; b < nb; ++b) {
        const long dref = (ax == 3) ? (fakes ? ref->reco.reg[b] : ref->den.reg[b])
                                    : (fakes ? ref->reco.b[ax][b] : ref->den.b[ax][b]);
        if (dref < 20) continue;
        std::string lab = (ax == 3) ? ve_regname[b] : ve_binlabel(ax, b);
        ve_printf("%-20s", lab.c_str());
        for (const auto &c : g_ve) {
          const long d = (ax == 3) ? (fakes ? c.reco.reg[b] : c.den.reg[b])
                                   : (fakes ? c.reco.b[ax][b] : c.den.b[ax][b]);
          const long n = (ax == 3) ? (fakes ? c.fake.reg[b] : c.dup.reg[b])
                                   : (fakes ? c.fake.b[ax][b] : c.dup.b[ax][b]);
          ve_printf(" |   %6.2f%% ", d ? 100.0*n/d : 0.0);
        }
        ve_printf("\n");
      }
    }
  }

  void val_eff_report(const char *prefix) {
    if (g_ve.empty()) { printf("val_eff_report: nothing accumulated.\n"); return; }
    const VeCfg *ref = &g_ve[0];
    if (!g_ve_ref.empty())
      for (const auto &c : g_ve) if (c.name == g_ve_ref) ref = &c;

    const std::string txt = std::string(prefix) + ".txt";
    g_ve_log = fopen(txt.c_str(), "w");

    ve_printf("\n================================================================\n");
    ve_printf("  val_eff -- per-sim-track efficiency, resolved\n");
    ve_printf("================================================================\n");
    ve_printf("DENOMINATOR -- CMSSW MTV's convention. Sim tracks that are findable, have a\n");
    ve_printf("  CENTRAL production vertex (tip < 3.5 cm, lip < 30 cm, as\n");
    ve_printf("  TrackingParticleSelector has it), at least 4 distinct layers, |eta| < 2.5\n");
    ve_printf("  and pT > 0.9 -- with the cut on the PLOTTED variable released for that\n");
    ve_printf("  variable's own plot, so the eta table carries every pT above 0.2 and the pT\n");
    ve_printf("  table every |eta| below 3. Efficiency below is against ALL of them, not\n");
    ve_printf("  only the seeded ones; the seeded subset is in the next block, because the\n");
    ve_printf("  search cannot find what it was not seeded for and MTV's 'central' is in\n");
    ve_printf("  practice imposed by the seeds rather than by us.\n");
    ve_printf("NUMERATOR: >= 1 reco track associated to it by 2*mccount >= nCandHits over the\n");
    ve_printf("  non-seed hits (TrackExtra::setMCTrackIDInfo), which is what quality-val's\n");
    ve_printf("  'found tracks' counts. A wrong extra hit raises the denominator of that\n");
    ve_printf("  rule only, so nothing here can be bought by taking more hits.\n");
    ve_printf("  nH >= 80%% is NOT used anywhere.\n");
    ve_printf("Reference configuration: %s\n", ref->name.c_str());

    ve_printf("\n--- the seeding ceiling (identical in every configuration by construction) ---\n");
    ve_printf("%-22s %10s %10s %9s %10s %8s %7s %7s %7s\n", "region", "sim sel", "seeded",
              "seed eff", "all seeds", "per sel", "pure", "hits", "pixel");
    for (int r = 0; r < 3; ++r) {
      const long ds = ref->den.reg[r], ss = ref->dens.reg[r], ns = ref->n_seed[r];
      ve_printf("%-22s %10ld %10ld %8.2f%% %10ld %8.2f %6.1f%% %7.2f %6.1f%%\n",
                ve_regname[r], ds, ss, ds ? 100.0*ss/ds : 0.0, ns,
                ss ? (double) ref->n_seed_on_sel[r]/ss : 0.0,
                ns ? 100.0*ref->n_seed_pure[r]/ns : 0.0,
                ns ? (double) ref->sum_seed_hits[r]/ns : 0.0,
                ref->sum_seed_hits[r] ? 100.0*ref->sum_seed_pix[r]/ref->sum_seed_hits[r] : 0.0);
    }
    ve_printf("  'seed eff' is the ceiling on every efficiency below: a track with no seed\n");
    ve_printf("  cannot be found. 'all seeds' is every seed of that region, most of which\n");
    ve_printf("  are on sim tracks OUTSIDE the selection (below 0.9 GeV, mostly), so it is\n");
    ve_printf("  not a duplicate rate; 'per sel' is, being seeds per SELECTED seeded sim\n");
    ve_printf("  track. 'pure' = seeds all of whose valid hits come from one sim track\n");
    ve_printf("  (Event::SimInfoFromHits::good_frac() == 1). 'hits' and 'pixel' say what the\n");
    ve_printf("  seeds are made of; the .bin carries the track algorithm and the hits, not\n");
    ve_printf("  the producer, so this bounds the seeding-algorithm question without\n");
    ve_printf("  settling it -- Patatrack and the standard chain differ in the FIT.\n");
    for (const auto &c : g_ve)
      for (int r = 0; r < 3; ++r)
        if (c.n_seed[r] != ref->n_seed[r])
          ve_printf("  !! %s has %ld seeds in region %d against the reference's %ld\n",
                    c.name.c_str(), c.n_seed[r], r, ref->n_seed[r]);

    ve_printf("\n--- totals ---\n");
    ve_printf("%-22s %7s %9s %9s %8s %8s %9s %9s %8s\n", "configuration", "events",
              "sim sel", "found", "eff", "of seed", "reco trks", "fakes", "dup/sim");
    for (const auto &c : g_ve) {
      ve_printf("%-22s %7ld %9ld %9ld %7.2f%% %7.2f%% %9ld %9ld %7.2f%%\n",
                c.name.c_str(), c.n_ev, c.den.tot, c.num.tot,
                c.den.tot ? 100.0*c.num.tot/c.den.tot : 0.0,
                c.dens.tot ? 100.0*c.num.tot/c.dens.tot : 0.0,
                c.reco.tot, c.fake.tot,
                c.den.tot ? 100.0*c.dup.tot/c.den.tot : 0.0);
      if (c.den.tot != ref->den.tot)
        ve_printf("   !! denominator differs from the reference by %ld -- pairing is NOT exact\n",
                  c.den.tot - ref->den.tot);
    }
    ve_printf("  'eff' is against every selected sim track (MTV); 'of seed' is against the\n");
    ve_printf("  seeded subset, i.e. what the SEARCH alone is responsible for.\n");
    ve_printf("\n--- paired differences against %s, whole sample ---\n", ref->name.c_str());
    ve_printf("%-22s %14s %14s %14s\n", "configuration", "d found", "d fakes", "d duplicates");
    for (const auto &c : g_ve) {
      if (&c == ref) continue;
      double s1, e1, s2, e2, s3, e3;
      ve_paired(ve_series(c.ev_num, -1, 0),  ve_series(ref->ev_num, -1, 0),  s1, e1);
      ve_paired(ve_series(c.ev_fake, -1, 0), ve_series(ref->ev_fake, -1, 0), s2, e2);
      ve_paired(ve_series(c.ev_dup, -1, 0),  ve_series(ref->ev_dup, -1, 0),  s3, e3);
      ve_printf("%-22s %+8.0f %4.1fs %+8.0f %4.1fs %+8.0f %4.1fs\n", c.name.c_str(),
                s1, e1 > 0 ? s1/e1 : 0.0, s2, e2 > 0 ? s2/e2 : 0.0, s3, e3 > 0 ? s3/e3 : 0.0);
    }
    ve_printf("  's' is the paired significance: the sum of the per-event difference over\n"
              "  the sigma of that sum, so it is the spread of the DIFFERENCE, not Poisson.\n");

    // Resolution, per region. NOT paired -- the population differs between
    // configurations by construction, since a configuration that finds more
    // tracks finds harder ones. Read it as the price of the extra tracks, and
    // read the "common" rows, which restrict every configuration to the sim
    // tracks ALL of them found, as the like-for-like comparison.
    // The COMMON subset: sim tracks every configuration found. Built as the
    // intersection over configurations of the (event, sim label) keys.
    std::map<std::pair<int,int>, int> seen;
    for (const auto &c : g_ve)
      for (const auto &x : c.res) ++seen[{x.evt, x.lbl}];
    const int ncfg = (int) g_ve.size();

    ve_printf("\n--- d(pT)/pT of the best-matched reco track, per region ---\n");
    ve_printf("ALL = every track that configuration found; COMMON = only the sim tracks\n"
              "every configuration found, which is the like-for-like comparison.\n");
    ve_printf("%-22s %-20s %-7s %8s %9s %9s %8s\n", "configuration", "region",
              "sample", "n", "median", "width", "|d|>20%");
    for (const auto &c : g_ve) {
      for (int pass = 0; pass < 2; ++pass) {
        for (int r = 0; r < 4; ++r) {
          std::vector<float> v;
          for (const auto &x : c.res) {
            if (r < 3 && x.reg != r) continue;
            if (pass == 1 && seen[{x.evt, x.lbl}] != ncfg) continue;
            v.push_back(x.r);
          }
          if (v.size() < 50) continue;
          std::sort(v.begin(), v.end());
          const size_t n = v.size();
          const double med = v[n/2];
          const double wid = 0.5 * (v[(size_t)(0.84*n)] - v[(size_t)(0.16*n)]);
          long tail = 0;
          for (float x : v) if (std::abs(x) > 0.2f) ++tail;
          ve_printf("%-22s %-20s %-7s %8zu %+9.5f %9.5f %7.2f%%\n", c.name.c_str(),
                    r == 3 ? "ALL REGIONS" : ve_regname[r], pass ? "COMMON" : "all",
                    n, med, wid, 100.0*tail/n);
        }
      }
    }

    ve_table(3, ref);
    for (int ax = 0; ax < VE_NAX; ++ax) ve_table(ax, ref);
    ve_ratio_table(3, ref, true);
    ve_ratio_table(0, ref, true);
    ve_ratio_table(3, ref, false);
    ve_ratio_table(2, ref, false);

    // TRACK LENGTH. Not paired: the denominator is each configuration's own
    // found tracks, so a configuration that finds more finds shorter ones and
    // the mean moves by composition. The sim row is the same population's truth
    // content and is what the other two should be read against.
    for (int w = 0; w < 3; ++w) ve_len_table(3, ref, w);
    for (int w = 0; w < 2; ++w) { ve_len_table(0, ref, w); ve_len_table(1, ref, w); }
    ve_len_table(0, ref, 2);  ve_len_table(1, ref, 2);

    // ---- the .root file: one efficiency TH1 per configuration per axis, with
    // binomial errors, plus an overlay canvas per axis so show-anrun's TBrowser
    // opens on something readable.
    // ---- the .root file. Two canvases per axis: the efficiency overlay, and the
    // PAIRED DIFFERENCE against the reference. The overlay alone is unreadable
    // once several configurations are in -- they sit within a few points of each
    // other on a 0-1 axis -- and the difference is the quantity with the small
    // error bar, since the denominator is shared and only the numerator moves.
    const std::string rootf = std::string(prefix) + ".root";
    TFile f(rootf.c_str(), "RECREATE");
    static const int kCol[8] = {kBlack, kRed + 1, kBlue + 1, kGreen + 2,
                                kMagenta + 1, kOrange + 7, kCyan + 2, kGray + 2};
    for (int ax = 0; ax < VE_NAX; ++ax) {
      const int nb = ve_nbin[ax];
      TCanvas *cv = new TCanvas(Form("c_eff_ax%d", ax),
                                Form("efficiency vs %s", ve_axname[ax]), 900, 600);
      TCanvas *cd = new TCanvas(Form("c_deff_ax%d", ax),
                                Form("efficiency difference vs %s", ve_axname[ax]), 900, 600);
      TLegend *lg = new TLegend(0.60, 0.13, 0.98, 0.13 + 0.05*g_ve.size());
      TLegend *ld = new TLegend(0.60, 0.13, 0.98, 0.13 + 0.05*g_ve.size());
      int ic = 0, id = 0;
      for (const auto &c : g_ve) {
        TH1D *hn = new TH1D(Form("num_ax%d_%d", ax, ic), "", nb, -0.5, nb - 0.5);
        TH1D *hd = new TH1D(Form("den_ax%d_%d", ax, ic), "", nb, -0.5, nb - 0.5);
        for (int b = 0; b < nb; ++b) {
          hn->SetBinContent(b+1, (double) c.num.b[ax][b]);
          hd->SetBinContent(b+1, (double) c.den.b[ax][b]);
        }
        TH1D *he = (TH1D*) hn->Clone(Form("eff_ax%d_%s", ax, c.name.c_str()));
        he->SetTitle(Form("efficiency vs %s;%s;efficiency", ve_axname[ax], ve_axname[ax]));
        he->Divide(hn, hd, 1.0, 1.0, "B");
        for (int b = 0; b < nb; ++b)
          he->GetXaxis()->SetBinLabel(b+1, ve_binlabel(ax, b).c_str());
        he->SetLineColor(kCol[ic % 8]);  he->SetMarkerColor(kCol[ic % 8]);
        he->SetMarkerStyle(20 + (ic % 8));  he->SetLineWidth(2);
        he->SetMinimum(0.0);  he->SetMaximum(1.05);  he->SetStats(0);
        he->Write();
        cv->cd();  he->Draw(ic == 0 ? "E1" : "E1 SAME");
        lg->AddEntry(he, c.name.c_str(), "lp");

        // the paired difference, in points, error = sigma of the per-event sum
        if (&c != ref) {
          TH1D *hdd = new TH1D(Form("d_eff_ax%d_%s", ax, c.name.c_str()),
                               Form("efficiency difference vs %s, paired;%s;points",
                                    ve_axname[ax], ve_axname[ax]), nb, -0.5, nb - 0.5);
          for (int b = 0; b < nb; ++b) {
            const long d = ref->den.b[ax][b];
            hdd->GetXaxis()->SetBinLabel(b+1, ve_binlabel(ax, b).c_str());
            if (d < 20) continue;
            double sum, sig;
            ve_paired(ve_series(c.ev_num, ax, b), ve_series(ref->ev_num, ax, b), sum, sig);
            hdd->SetBinContent(b+1, 100.0*sum/d);
            hdd->SetBinError(b+1, 100.0*sig/d);
          }
          hdd->SetLineColor(kCol[(id+1) % 8]);  hdd->SetMarkerColor(kCol[(id+1) % 8]);
          hdd->SetMarkerStyle(20 + ((id+1) % 8));  hdd->SetLineWidth(2);  hdd->SetStats(0);
          hdd->Write();
          cd->cd();  hdd->Draw(id == 0 ? "E1" : "E1 SAME");
          ld->AddEntry(hdd, c.name.c_str(), "lp");
          ++id;
        }
        delete hn;  delete hd;
        ++ic;
      }
      cv->cd();  lg->Draw();  cv->Write();
      cd->cd();  ld->Draw();  cd->Write();
    }
    f.Close();

    ve_printf("\nval_eff_report: wrote %s and %s (%zu configurations)\n",
              rootf.c_str(), txt.c_str(), g_ve.size());
    if (g_ve_log) { fclose(g_ve_log); g_ve_log = nullptr; }
  }

  // ==========================================================================
  // val_chopres -- the pT5 pixel-chop recovery, RESOLVED.
  //
  // Same measurement as val_chop_recovery_* and the same reason for preferring
  // it on this sample: the chopped hits were found by the upstream
  // reconstruction, so the comparison is an exact (layer, index) match and NO
  // TRUTH IS INVOLVED. That matters here beyond the usual mc_match caution --
  // the HLT March sample predates the split-cluster arbitration fix, so its
  // rec->sim links are stale and any truth-matched efficiency on it is biased.
  // This metric is immune to that.
  //
  // Resolved in |eta| and pT of the candidate itself, and in chopped hits per
  // chopped LAYER, which is the axis the one-hit-per-layer ceiling lives on.
  namespace {
    struct ChopCfg {
      std::string name;
      VeBins hden, hnum;             // chopped hits, recovered hits
      VeBins tden, tnum;             // tracks with chopped hits, fully recovered
      std::vector<VeBins> ev_hnum, ev_tnum;
      long n_ev = 0;
    };
    std::vector<ChopCfg> g_cr;
    std::string g_cr_ref;

    ChopCfg &cr_cfg(const char *name) {
      for (auto &c : g_cr) if (c.name == name) return c;
      g_cr.push_back(ChopCfg());  g_cr.back().name = name;  return g_cr.back();
    }
    void ve_fill_n(VeBins &v, int be, int bp, int bh, int reg, long n) {  // chopres: no MTV gate
      if (be >= 0) v.b[0][be] += n;
      if (bp >= 0) v.b[1][bp] += n;
      if (bh >= 0) v.b[2][bh] += n;
      if (reg >= 0) v.reg[reg] += n;
      v.tot += n;
    }
  }

  void val_chopres_reset() { g_cr.clear(); g_cr_ref.clear(); }
  void val_chopres_ref(const char *cfg) { g_cr_ref = cfg; }

  void val_chopres_event(const Event *ev, const char *cfg) {
    if (ev == nullptr) return;
    ChopCfg &C = cr_cfg(cfg);
    VeBins e_hnum, e_tnum;
    for (const Track &c : ev->candidateTracks_) {
      auto it = Shell::s_chopped_hits.find(c.label());
      if (it == Shell::s_chopped_hits.end() || it->second.empty()) continue;
      const auto &chopped = it->second;
      std::set<int> clay;
      for (const HitOnTrack &ch : chopped) clay.insert(ch.layer);
      const float ae = std::abs(c.momEta());
      const int be = ve_bin_eta(ae), bp = ve_bin_pt(c.pT()), reg = ve_region(ae);
      const int bh = ve_bin_hpl((float) chopped.size() / (float) std::max<size_t>(1, clay.size()));
      int back = 0;
      for (const HitOnTrack &ch : chopped) {
        for (int i = 0; i < c.nTotalHits(); ++i) {
          const HitOnTrack hot = c.getHitOnTrack(i);
          if (hot.layer == ch.layer && hot.index == ch.index) { ++back; break; }
        }
      }
      ve_fill_n(C.hden, be, bp, bh, reg, (long) chopped.size());
      ve_fill_n(e_hnum, be, bp, bh, reg, back);
      ve_fill(C.tden, be, bp, bh, reg, 0.0f, 1e9f);   // chopres is not MTV-gated
      if (back == (int) chopped.size()) ve_fill(e_tnum, be, bp, bh, reg, 0.0f, 1e9f);
    }
    ve_add(C.hnum, e_hnum);  ve_add(C.tnum, e_tnum);
    C.ev_hnum.push_back(e_hnum);  C.ev_tnum.push_back(e_tnum);
    ++C.n_ev;
    Shell::s_chopped_hits.clear();
  }

  namespace {
    void cr_table(int ax, const ChopCfg *ref, bool track_level) {
      const int nb = (ax == 3) ? 3 : ve_nbin[ax];
      ve_printf("\n--- %s vs %s ---\n",
                track_level ? "tracks FULLY recovered" : "chopped hits recovered",
                ax == 3 ? "region" : (ax == 2 ? "chopped hits / chopped layer" : ve_axname[ax]));
      ve_printf("%-20s %9s", ax == 3 ? "region" : "bin", track_level ? "tracks" : "hits");
      for (const auto &c : g_cr) ve_printf(" | %-10.10s", c.name.c_str());
      ve_printf("\n");
      for (int b = 0; b < nb; ++b) {
        const VeBins &D = track_level ? ref->tden : ref->hden;
        const long den = (ax == 3) ? D.reg[b] : D.b[ax][b];
        if (den < 20) continue;
        std::string lab = (ax == 3) ? ve_regname[b] : ve_binlabel(ax, b);
        ve_printf("%-20s %9ld", lab.c_str(), den);
        for (const auto &c : g_cr) {
          const VeBins &Dc = track_level ? c.tden : c.hden;
          const VeBins &Nc = track_level ? c.tnum : c.hnum;
          const long d = (ax == 3) ? Dc.reg[b] : Dc.b[ax][b];
          const long n = (ax == 3) ? Nc.reg[b] : Nc.b[ax][b];
          if (&c == ref) ve_printf(" |   %6.2f%% ", d ? 100.0*n/d : 0.0);
          else {
            double sum, sig;
            ve_paired(ve_series(track_level ? c.ev_tnum : c.ev_hnum, ax == 3 ? 3 : ax, b),
                      ve_series(track_level ? ref->ev_tnum : ref->ev_hnum, ax == 3 ? 3 : ax, b),
                      sum, sig);
            const double dp = d ? 100.0*sum/d : 0.0;
            const double sp = d ? 100.0*sig/d : 0.0;
            ve_printf(" | %+6.2f%s%-3.3s", dp,
                      sp > 0 && std::abs(dp) > 3*sp ? "*" : " ",
                      sp > 0 && std::abs(dp) > 3*sp ? "sig" : "");
          }
        }
        ve_printf("\n");
      }
    }
  }

  void val_chopres_report(const char *prefix) {
    if (g_cr.empty()) { printf("val_chopres_report: nothing accumulated.\n"); return; }
    const ChopCfg *ref = &g_cr[0];
    if (!g_cr_ref.empty())
      for (const auto &c : g_cr) if (c.name == g_cr_ref) ref = &c;

    const std::string txt = std::string(prefix) + ".txt";
    g_ve_log = fopen(txt.c_str(), "w");

    ve_printf("\n================================================================\n");
    ve_printf("  val_chopres -- pT5 pixel-chop recovery, resolved\n");
    ve_printf("================================================================\n");
    ve_printf("Truth-FREE: exact (layer, index) match against the hits the chop removed.\n");
    ve_printf("Reference configuration: %s\n", ref->name.c_str());
    ve_printf("\n--- totals ---\n");
    ve_printf("%-22s %7s %9s %9s %8s %9s %9s %8s\n", "configuration", "events",
              "hits", "recovered", "frac", "tracks", "full", "frac");
    for (const auto &c : g_cr)
      ve_printf("%-22s %7ld %9ld %9ld %7.2f%% %9ld %9ld %7.2f%%\n",
                c.name.c_str(), c.n_ev, c.hden.tot, c.hnum.tot,
                c.hden.tot ? 100.0*c.hnum.tot/c.hden.tot : 0.0,
                c.tden.tot, c.tnum.tot, c.tden.tot ? 100.0*c.tnum.tot/c.tden.tot : 0.0);
    for (const auto &c : g_cr)
      if (c.hden.tot != ref->hden.tot)
        ve_printf("  !! %s: chopped-hit denominator differs from the reference by %ld\n",
                  c.name.c_str(), c.hden.tot - ref->hden.tot);

    cr_table(3, ref, false);  cr_table(0, ref, false);
    cr_table(1, ref, false);  cr_table(2, ref, false);
    cr_table(3, ref, true);   cr_table(2, ref, true);
    ve_printf("  reference column is absolute; the others are the PAIRED difference in\n"
              "  points, * = |delta| > 3 sigma of the per-event spread.\n");

    const std::string rootf = std::string(prefix) + ".root";
    TFile f(rootf.c_str(), "RECREATE");
    for (int ax = 0; ax < VE_NAX; ++ax) {
      const int nb = ve_nbin[ax];
      TCanvas *cv = new TCanvas(Form("c_chop_ax%d", ax), Form("chop recovery vs axis %d", ax), 900, 600);
      TLegend *lg = new TLegend(0.60, 0.15, 0.98, 0.15 + 0.05*g_cr.size());
      int ic = 0;
      for (const auto &c : g_cr) {
        TH1D *hn = new TH1D(Form("cnum_ax%d_%d", ax, ic), "", nb, -0.5, nb - 0.5);
        TH1D *hd = new TH1D(Form("cden_ax%d_%d", ax, ic), "", nb, -0.5, nb - 0.5);
        for (int b = 0; b < nb; ++b) {
          hn->SetBinContent(b+1, (double) c.hnum.b[ax][b]);
          hd->SetBinContent(b+1, (double) c.hden.b[ax][b]);
        }
        TH1D *he = (TH1D*) hn->Clone(Form("chop_ax%d_%s", ax, c.name.c_str()));
        he->SetTitle(Form("chopped hits recovered vs %s;%s;recovered",
                          ax == 2 ? "chopped hits / layer" : ve_axname[ax],
                          ax == 2 ? "chopped hits / layer" : ve_axname[ax]));
        he->Divide(hn, hd, 1.0, 1.0, "B");
        for (int b = 0; b < nb; ++b) he->GetXaxis()->SetBinLabel(b+1, ve_binlabel(ax, b).c_str());
        he->SetLineColor(1 + ic);  he->SetMarkerColor(1 + ic);  he->SetMarkerStyle(20 + ic);
        he->SetMinimum(0.0);  he->SetMaximum(1.05);
        he->Write();
        cv->cd();  he->Draw(ic == 0 ? "E1" : "E1 SAME");
        lg->AddEntry(he, c.name.c_str(), "lp");
        delete hn;  delete hd;  ++ic;
      }
      lg->Draw();  cv->Write();
    }
    f.Close();
    ve_printf("\nval_chopres_report: wrote %s and %s (%zu configurations)\n",
              rootf.c_str(), txt.c_str(), g_cr.size());
    if (g_ve_log) { fclose(g_ve_log); g_ve_log = nullptr; }
  }

}  // namespace mkfit
