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
      case PA_Line: {}
      case PA_Quadratic: {}

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
          float r0 = std::hypot(c.x, c.y);
          float td = (R - r0) * curv;
          float id = oo_curv * td * (1.0f + 0.16666666f * td * td);
          // This would be for line approximation:
          // float id = R - r0;
          D += id;

          //printf("%-3d r0=%f R-r0=%f td=%f id=%f id_line=%f delta_id=%g\n",
          //       i, r0, R-r0, td, id, R - r0, id - (R-r0));

          float cosa = std::cos(id * inv_pt * inv_k);
          float sina = std::sin(id * inv_pt * inv_k);

          // update parameters
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
    return std::abs(std::hypot(c.x, c.y) - R) < 0.1f;
  }

  bool InitialState::propagate_to_z(PropAlgo_e algo, float Z, State& c, bool update_momentum) const {
    switch (algo) {
      case PA_Line: {}
      case PA_Quadratic: {}

      case PA_Exact: {
        const float k = 1.0f / inv_k;

        const float dz = Z - z;
        const float alpha = dz * inv_k / pz;

        const float cosa = std::cos(alpha);
        const float sina = std::sin(alpha);

        c.x = x + k * (px * sina - py * (1.0f - cosa));
        c.y = y + k * (py * sina + px * (1.0f - cosa));
        c.z = Z;

        if (update_momentum) {
          c.px = px * cosa - py * sina;
          c.py = py * cosa + px * sina;
          c.pz = pz;
        }
      }
      break;
    }

    return true;
  }

}
