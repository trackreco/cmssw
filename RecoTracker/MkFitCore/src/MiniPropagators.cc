#include "RecoTracker/MkFitCore/src/MiniPropagators.h"

namespace mkfit::mini_propagators {

  State::State(const MPlexLV& par, int ti) {
    x = par.constAt(ti, 0, 0);
    y = par.constAt(ti, 1, 0);
    z = par.constAt(ti, 2, 0);
    const float pt = 1.0f / par.constAt(ti, 3, 0);
    px = pt * std::cos(par.constAt(ti, 4, 0));
    py = pt * std::sin(par.constAt(ti, 4, 0));
    pz = pt / std::tan(par.constAt(ti, 5, 0));
  }

  bool InitialState::propagate_to_r(PropAlgo_e algo, float R, State& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        // Momentum is always updated -- used as temporary for stepping.
        const float k = 1.0f / inv_k;

        const float curv = 0.5f * inv_k * inv_pt;
        const float oo_curv = 1.0f / curv;  // 2 * radius of curvature
        const float lambda = pz * inv_pt;

        float D = 0;

        c = *this;
        for (int i = 0; i < Config::Niter; ++i) {
          // compute tangental and ideal distance for the current iteration.
          // 3-rd order asin for symmetric incidence (shortest arc lenght).
          float r0 = hipo(c.x, c.y);
          float td = (R - r0) * curv;
          float id = oo_curv * td * (1.0f + 0.16666666f * td * td);
          // This would be for line approximation:
          // float id = R - r0;
          D += id;

          //printf("%-3d r0=%f R-r0=%f td=%f id=%f id_line=%f delta_id=%g\n",
          //       i, r0, R-r0, td, id, R - r0, id - (R-r0));

          float alpha = id * inv_pt * inv_k;
          float sina, cosa;
          vdt::fast_sincosf(alpha, sina, cosa);

          // update parameters
          c.dalpha += alpha;
          c.x += k * (c.px * sina - c.py * (1.0f - cosa));
          c.y += k * (c.py * sina + c.px * (1.0f - cosa));

          const float o_px = c.px;  // copy before overwriting
          c.px = c.px * cosa - c.py * sina;
          c.py = c.py * cosa + o_px * sina;
        }

        c.z += lambda * D;
      }
    }
    // should have some epsilon constant / member? relative vs. abs?
    c.fail_flag = std::abs(hipo(c.x, c.y) - R) < 0.1f ? 0 : 1;
    return c.fail_flag;
  }

  bool InitialState::propagate_to_z(PropAlgo_e algo, float Z, State& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        const float k = 1.0f / inv_k;

        const float dz = Z - z;
        const float alpha = dz * inv_k / pz;

        float sina, cosa;
        vdt::fast_sincosf(alpha, sina, cosa);

        c.dalpha = dalpha + alpha;
        c.x = x + k * (px * sina - py * (1.0f - cosa));
        c.y = y + k * (py * sina + px * (1.0f - cosa));
        c.z = Z;

        if (update_momentum) {
          c.px = px * cosa - py * sina;
          c.py = py * cosa + px * sina;
          c.pz = pz;
        }
      } break;
    }
    c.fail_flag = 0;
    return c.fail_flag;
  }

  bool InitialState::propagate_to_plane(PropAlgo_e algo, const SVector3& pos, const SVector3& zdir, State& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
        // Momentum is never changed ... we simply step along its direction
        // to hit the plane.
        // const float k = 1.0f / inv_k;

        // const float curv = 0.5f * inv_k * inv_pt;
        // const float oo_curv = 1.0f / curv;  // 2 * radius of curvature
        // const float lambda = pz * inv_pt;

        //X float dist = (x - pos(0)) * zdir(0) +
        //X              (y - pos(1)) * zdir(1) +
        //X              (z - pos(2)) * zdir(2);

        // t * p_vec intersects the plane:
        float t = (pos(0) * zdir(0) + pos(1) * zdir(1) + pos(2) * zdir(2) - x * zdir(0) -
                   y * zdir(1) - z * zdir(2)) /
                  (px * zdir(0) + py * zdir(1) + pz * zdir(2));

        //X printf("  module-center: %.2f,%.2f,%.2f  pos: %.2f,%.2f,%.2f  normal: %.2f,%.2f,%.2f\n",
        //X        pos(0),pos(1),pos(2), x, y, z, zdir(0),zdir(1),zdir(2));

        c = *this;
        c.x += t * c.px;
        c.y += t * c.py;
        c.z += t * c.pz;

        c.dalpha += std::hypot(t * c.px, t * c.py) * inv_k * inv_pt; // correct for curvature?

        //X re-check ditance to plane
        //X float dist2 = (c.x - pos(0)) * zdir(0) +
        //X               (c.y - pos(1)) * zdir(1) +
        //X               (c.z - pos(2)) * zdir(2);
        //X printf("  dist = %.3f, t = %.4f ..... dist2 = %.4f\n", dist, t, dist2);
        break;
      }

      case PA_Quadratic: {
        throw std::runtime_error("Quadratic prop_to_plane not implemented");
      }

      case PA_Exact: {
        throw std::runtime_error("Exact prop_to_plane not implemented");
      }
    }
    return false;
  }

  //===========================================================================
  // Vectorized version
  //===========================================================================

  StatePlex::StatePlex(const MPlexLV& par) {
    x = par.ReduceFixedIJ(0, 0);
    y = par.ReduceFixedIJ(1, 0);
    z = par.ReduceFixedIJ(2, 0);
    const MPF pt = 1.0f / par.ReduceFixedIJ(3, 0);
    Matriplex::fast_sincos(par.ReduceFixedIJ(4, 0), py, px);
    px *= pt;
    py *= pt;
    pz = pt / Matriplex::fast_tan(par.ReduceFixedIJ(5, 0));
  }

  void InitialStatePlex::init_momentum_vec_and_k(const MPF& phi, const MPI& chg, float bf) {
    const MPF pt = 1.0f / inv_pt;
    phi.fast_sincos(py, px);
    px *= pt;
    py *= pt;
    pz = pt / Matriplex::fast_tan(theta);
    for (int i = 0; i < inv_k.kTotSize; ++i) {
      inv_k[i] = ((chg[i] < 0) ? 0.01f : -0.01f) * Const::sol * bf;
    }
  }

  // propagate to radius; returns number of failed propagations
  int InitialStatePlex::propagate_to_r(
      PropAlgo_e algo, const MPF& R, StatePlex& c, bool update_momentum, int N_proc) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        // Momentum is always updated -- used as temporary for stepping.
        const MPF k = 1.0f / inv_k;

        const MPF curv = 0.5f * inv_k * inv_pt;
        const MPF oo_curv = 1.0f / curv;  // 2 * radius of curvature
        const MPF lambda = pz * inv_pt;

        MPF D = 0;

        c = *this;
        for (int i = 0; i < Config::Niter; ++i) {
          // compute tangental and ideal distance for the current iteration.
          // 3-rd order asin for symmetric incidence (shortest arc lenght).
          MPF r0 = Matriplex::hypot(c.x, c.y);
          MPF td = (R - r0) * curv;
          MPF id = oo_curv * td * (1.0f + 0.16666666f * td * td);
          // This would be for line approximation:
          // float id = R - r0;
          D += id;

          //printf("%-3d r0=%f R-r0=%f td=%f id=%f id_line=%f delta_id=%g\n",
          //       i, r0, R-r0, td, id, R - r0, id - (R-r0));

          MPF alpha = id * inv_pt * inv_k;

          MPF sina, cosa;
          Matriplex::fast_sincos(alpha, sina, cosa);

          // update parameters
          c.dalpha += alpha;
          c.x += k * (c.px * sina - c.py * (1.0f - cosa));
          c.y += k * (c.py * sina + c.px * (1.0f - cosa));

          MPF o_px = c.px;  // copy before overwriting
          c.px = c.px * cosa - c.py * sina;
          c.py = c.py * cosa + o_px * sina;
        }

        c.z += lambda * D;
      }
    }

    // should have some epsilon constant / member? relative vs. abs?
    MPF r = Matriplex::hypot(c.x, c.y);
    c.fail_flag = 0;
    int n_fail = 0;
    for (int i = 0; i < N_proc; ++i) {
      if (std::abs(R[i] - r[i]) > 0.1f) {
        c.fail_flag[i] = 1;
        ++n_fail;
      }
    }
    return n_fail;
  }

  int InitialStatePlex::propagate_to_z(
      PropAlgo_e algo, const MPF& Z, StatePlex& c, bool update_momentum, int N_proc) const {
    switch (algo) {
      case PA_Line: {
      }
      case PA_Quadratic: {
      }

      case PA_Exact: {
        MPF k = 1.0f / inv_k;

        MPF dz = Z - z;
        MPF alpha = dz * inv_k / pz;

        MPF sina, cosa;
        Matriplex::fast_sincos(alpha, sina, cosa);

        c.dalpha = dalpha + alpha;
        c.x = x + k * (px * sina - py * (1.0f - cosa));
        c.y = y + k * (py * sina + px * (1.0f - cosa));
        c.z = Z;

        if (update_momentum) {
          c.px = px * cosa - py * sina;
          c.py = py * cosa + px * sina;
          c.pz = pz;
        }
      } break;
    }
    c.fail_flag = 0;
    return 0;
  }

  int InitialStatePlex::propagate_to_plane(PropAlgo_e algo, const MP3V &pos, const MP3V &zdir, StatePlex& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {
        // Momentum is never changed ... we simply step along its direction
        // to hit the plane.
        // const float k = 1.0f / inv_k;

        // const float curv = 0.5f * inv_k * inv_pt;
        // const float oo_curv = 1.0f / curv;  // 2 * radius of curvature
        // const float lambda = pz * inv_pt;

        // MPF dist = (x - pos(0,0)) * zdir(0,0) +
        //            (y - pos(1,0)) * zdir(1,0) +
        //            (z - pos(2,0)) * zdir(2,0);

        // t * p_vec intersects the plane:
        MPF t = ((pos(0,0) - x) * zdir(0,0) + (pos(1,0) - y) * zdir(1,0) + (pos(2,0) - z) * zdir(2,0)) /
                  (px * zdir(0,0) + py * zdir(1,0) + pz * zdir(2,0));

        //X printf("  module-center: %.2f,%.2f,%.2f  pos: %.2f,%.2f,%.2f  normal: %.2f,%.2f,%.2f\n",
        //X        pos(0),pos(1),pos(2), x, y, z, zdir(0),zdir(1),zdir(2));

        c = *this;
        c.x += t * c.px;
        c.y += t * c.py;
        c.z += t * c.pz;

        c.dalpha += Matriplex::hypot(t * c.px, t * c.py) * inv_k * inv_pt; // correct for curvature?

        //X re-check ditance to plane
        // MPF dist2 = (c.x - pos(0,0)) * zdir(0,0) +
        //             (c.y - pos(1,0)) * zdir(1,0) +
        //             (c.z - pos(2,0)) * zdir(2,0);
        // for (int i=0; i < NN; ++i)
        //   printf("    i:%d  dist = %.3f, t = %.4f ..... dist2 = %.3f\n", i, dist[i], t[i], dist2[i]);

        break;
      }

      case PA_Quadratic: {
        throw std::runtime_error("Quadratic prop_to_plane not implemented");
      }

      case PA_Exact: {
        throw std::runtime_error("Exact prop_to_plane not implemented");
      }
    }
    c.fail_flag = 0;
    return 0;
  }

  //===========================================================================
  // Hermite3D interpolation
  //===========================================================================

  // MP4V m_Hx, m_Hy, m_Hz;
  // MPF  m_Hderfac; // derivative scaling factor
  // do we need delta-alpha to transition for t -> alpha -> path-length

  void Hermite3D::hermite_1d(const MPF &x1, const MPF &p1, const MPF &x2, const MPF &p2,
                             const MPF &derfac, MPlex4V &h)
  {
    MPF P = x2 - x1;
    MPF Q = p1 * derfac;
    MPF R = p2 * derfac - 2.0f * P + Q;
    h.aij(0, 0) = x1;
    h.aij(1, 0) = Q;
    h.aij(2, 0) = P - Q - R;
    h.aij(3, 0) = R;

  }

  void Hermite3D::calculate_coeffs(const StatePlex &sp1, const StatePlex &sp2, const MPF &inv_k)
  {
    // mini_propagators::InitialStatePlex m_isp;
    // mini_propagators::StatePlex m_sp1, m_sp2;

    // // Hermite p(3) for trajectory approimation, could be Matriplex<float, 4, 3, NN>
    // MPlex3V m_H3, m_H2, m_H1, m_H0; // Hermite p(3) coefficients for x, y, z
    // MPlex4V m_Hx, m_Hy, m_Hz;
    m_Hderfac = (sp2.dalpha - sp1.dalpha) / inv_k;
    hermite_1d(sp1.x, sp1.px, sp2.x, sp2.px, m_Hderfac, m_Hx);
    hermite_1d(sp1.y, sp1.py, sp2.y, sp2.py, m_Hderfac, m_Hy);
    hermite_1d(sp1.z, sp1.pz, sp2.z, sp2.pz, m_Hderfac, m_Hz);
    m_Hderfac = 1.0f / m_Hderfac;

  }

  void Hermite3D::evaluate(const MPF &t, MPF &x, MPF &y, MPF &z) const
  {
    MPF t2 = t * t;
    MPF t3 = t2 * t;
    x = m_Hx(0,0) + t * m_Hx(1,0) + t2 * m_Hx(2,0) + t3 * m_Hx(3,0);
    y = m_Hy(0,0) + t * m_Hy(1,0) + t2 * m_Hy(2,0) + t3 * m_Hy(3,0);
    z = m_Hz(0,0) + t * m_Hz(1,0) + t2 * m_Hz(2,0) + t3 * m_Hz(3,0);
  }

  void Hermite3D::evaluate(const MPF &t, MPF &x, MPF &y, MPF &z, MPF &dx, MPF &dy, MPF &dz) const
  {
    MPF t2 = t * t;
    MPF t3 = t2 * t;
    x = m_Hx(0,0) + t * m_Hx(1,0) + t2 * m_Hx(2,0) + t3 * m_Hx(3,0);
    y = m_Hy(0,0) + t * m_Hy(1,0) + t2 * m_Hy(2,0) + t3 * m_Hy(3,0);
    z = m_Hz(0,0) + t * m_Hz(1,0) + t2 * m_Hz(2,0) + t3 * m_Hz(3,0);
    dx = (m_Hx(1,0) + 2.0f * t * m_Hx(2,0) + 3.0f * t2 * m_Hx(3,0)) * m_Hderfac;
    dy = (m_Hy(1,0) + 2.0f * t * m_Hy(2,0) + 3.0f * t2 * m_Hy(3,0)) * m_Hderfac;
    dz = (m_Hz(1,0) + 2.0f * t * m_Hz(2,0) + 3.0f * t2 * m_Hz(3,0)) * m_Hderfac;
  }

  void Hermite3D::evaluate(const MPF &t, StatePlex &sp) const
  {
    evaluate(t, sp.x, sp.y, sp.z, sp.px, sp.py, sp.pz);
  }

  //===========================================================================
  // Hermite3DOnPlane intersection
  //===========================================================================

    void Hermite3DOnPlane::init_coeffs(const Hermite3D &h, const MP3V& pos, const MP3V& zdir)
    {
      m_D = 0.0f;
      m_D.aij(0, 0) = (pos(0,0) - h.m_Hx(0,0)) * zdir(0,0) +
                      (pos(1,0) - h.m_Hy(0,0)) * zdir(1,0) +
                      (pos(2,0) - h.m_Hz(0,0)) * zdir(2,0);
      for (int i = 1; i < 4; ++i) {
        m_D.aij(i, 0) = (h.m_Hx(i,0) * zdir(0,0) +
                         h.m_Hy(i,0) * zdir(1,0) +
                         h.m_Hz(i,0) * zdir(2,0)).negate();
      }
      // // QQQQ do i even need this, it's trivial, can multiply in place, each time.
      // m_dDdt = 0.0f;
      // for (int i = 1; i < 4; ++i) {
      //   m_dDdt.aij(i-1, 0) = float(i) * m_D(i, 0);
      // }

      MPF d0, d1;
      evaluate(0.0f, d0);
      evaluate(1.0f, d1);
      m_T = d0 / (d0 - d1);
    }

    void Hermite3DOnPlane::evaluate(const MPF &t, MPF &d)
    {
      MPF t2 = t * t;
      d = m_D(0,0) + m_D(1,0) * t + m_D(2,0) * t2 + m_D(3,0) * t2 * t;
    }

    void Hermite3DOnPlane::evaluate(const MPF &t, MPF &d, MPF &dddt)
    {
      MPF t2 = t * t;
      d = m_D(0,0) + m_D(1,0) * t + m_D(2,0) * t2 + m_D(3,0) * t2 * t;
      dddt = m_D(1,0) + 2.0f * m_D(2,0) * t + 3.0f * m_D(3,0) * t2;
    }

    void Hermite3DOnPlane::solve()
    {
      MPF f, df;
      evaluate(m_T, f, df);
      m_T -= f / df;
    }

}  // namespace mkfit::mini_propagators
