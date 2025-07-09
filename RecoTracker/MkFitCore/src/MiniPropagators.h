#ifndef RecoTracker_MkFitCore_src_MiniPropagators_h
#define RecoTracker_MkFitCore_src_MiniPropagators_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"
#include "Matrix.h"

namespace mkfit::mini_propagators {

  enum PropAlgo_e { PA_Line, PA_Quadratic, PA_Exact };

  struct StatePlex;
  struct InitialStatePlex;

  struct State {
    float x, y, z;
    float px, py, pz;
    float dalpha;
    int fail_flag;

    State(float x=0, float y=0, float z=0, float px=0, float py=0, float pz=0, float dalpha=0, int fail_flag=0) :
      x(x), y(y), z(z), px(px), py(py), pz(pz), dalpha(dalpha), fail_flag(fail_flag) {}
    State(const MPlexLV& par, int ti);
    State(const StatePlex &sp, int i);
  };

  struct InitialState : public State {
    float inv_pt, inv_k;
    float theta;

    InitialState(const MPlexLV& par, const MPlexQI& chg, int ti)
        : InitialState(State(par, ti), chg.constAt(ti, 0, 0), par.constAt(ti, 3, 0), par.constAt(ti, 5, 0)) {}

    InitialState(const State &s, short charge, float ipt, float tht, float bf = Config::Bfield)
        : State(s), inv_pt(ipt), theta(tht) {
      inv_k = ((charge < 0) ? 0.01f : -0.01f) * Const::sol * bf;
    }

    InitialState(const StatePlex &sp, int i, const InitialStatePlex &isp, int j);

    bool propagate_to_r(PropAlgo_e algo, float R, State& c, bool update_momentum) const;
    bool propagate_to_z(PropAlgo_e algo, float Z, State& c, bool update_momentum) const;

    bool propagate_to_plane(PropAlgo_e algo, const SVector3& pos, const SVector3& zdir, State& c, bool update_momentum) const;
  };

  //-----------------------------------------------------------
  // Vectorized version
  //-----------------------------------------------------------

  using MPF = MPlexQF;
  using MPI = MPlexQI;
  using MP3V = MPlex3V;
  using MP4V = MPlex4V;

  struct StatePlex {
    MPF x, y, z;
    MPF px, py, pz;
    MPF dalpha{0};
    MPI fail_flag{0};

    StatePlex() = default;
    StatePlex(const MPlexLV& par);

    void copyIn(int dst_slot, const State &src) {
      x[dst_slot] = src.x;
      y[dst_slot] = src.y;
      z[dst_slot] = src.z;
      px[dst_slot] = src.px;
      py[dst_slot] = src.py;
      pz[dst_slot] = src.pz;
      dalpha[dst_slot] = src.dalpha;
      fail_flag[dst_slot] = src.fail_flag;
    }

    void copyIn(int dst_slot, const StatePlex &src, int src_slot) {
      x[dst_slot] = src.x[src_slot];
      y[dst_slot] = src.y[src_slot];
      z[dst_slot] = src.z[src_slot];
      px[dst_slot] = src.px[src_slot];
      py[dst_slot] = src.py[src_slot];
      pz[dst_slot] = src.pz[src_slot];
      dalpha[dst_slot] = src.dalpha[src_slot];
      fail_flag[dst_slot] = src.fail_flag[src_slot];
    }

    State state(int i) const {
      return { x[i], y[i], z[i], px[i], py[i], pz[i], dalpha[i], fail_flag[i]};
    }
  };

  struct InitialStatePlex : public StatePlex {
    MPF inv_pt, inv_k;
    MPF theta;

    InitialStatePlex() = default;

    InitialStatePlex(const MPlexLV& par, const MPI& chg)
        : InitialStatePlex(StatePlex(par), chg, par.ReduceFixedIJ(3, 0), par.ReduceFixedIJ(5, 0)) {}

    InitialStatePlex(const StatePlex &sp, MPI charge, MPF ipt, MPF tht, float bf = Config::Bfield)
        : StatePlex(sp), inv_pt(ipt), theta(tht) {
      for (int i = 0; i < inv_k.kTotSize; ++i) {
        inv_k[i] = ((charge[i] < 0) ? 0.01f : -0.01f) * Const::sol * bf;
      }
    }

    InitialStatePlex(const StatePlex &sp, const InitialStatePlex &isp)
        : StatePlex(sp), inv_pt(isp.inv_pt), inv_k(isp.inv_k), theta(isp.theta)
    {}

    void copyIn(int dst_slot, const InitialState &src) {
      StatePlex::copyIn(dst_slot, src);
      inv_pt[dst_slot] = src.inv_pt;
      inv_k[dst_slot] = src.inv_k;
      theta[dst_slot] = src.theta;
    }

    void copyIn(int dst_slot, const InitialStatePlex &src, int src_slot) {
      StatePlex::copyIn(dst_slot, src, src_slot);
      inv_pt[dst_slot] = src.inv_pt[src_slot];
      inv_k[dst_slot] = src.inv_k[src_slot];
      theta[dst_slot] = src.theta[src_slot];
    }

    using StatePlex::operator=;

    void init_momentum_vec_and_k(const MPF& phi, const MPI& chg, float bf = Config::Bfield);

    int propagate_to_r(PropAlgo_e algo, const MPF& R, StatePlex& c, bool update_momentum, int N_proc = NN) const;
    int propagate_to_z(PropAlgo_e algo, const MPF& Z, StatePlex& c, bool update_momentum, int N_proc = NN) const;

    int propagate_to_plane(PropAlgo_e algo, const MP3V& pos, const MP3V& zdir, StatePlex& c, bool update_momentum) const;
  };

  // Projecting constructors from plexes to regular state objects

  inline State::State(const StatePlex &sp, int i) : State(sp.state(i)) {}

  inline InitialState::InitialState(const StatePlex &sp, int i, const InitialStatePlex &isp, int j)
    : State(sp.state(i)), inv_pt(isp.inv_pt[j]), inv_k(isp.inv_k[j]), theta(isp.theta[j]) {}

  //-----------------------------------------------------------
  // Hermite interpolation
  //-----------------------------------------------------------

  struct Hermite3D {
    // Hermite p(3) for trajectory approximation, could be Matriplex<float, 4, 3, NN>
    MP4V m_Hx, m_Hy, m_Hz; // Hermite p(3) coefficients for x, y, z
    MPF  m_Hderfac; // derivative scaling factor
    // do we need delta-alpha to transition for t -> alpha -> path-length

    void copyIn(int dst_slot, const Hermite3D &src, int src_slot) {
      m_Hx.copyIn(dst_slot, src.m_Hx, src_slot);
      m_Hy.copyIn(dst_slot, src.m_Hy, src_slot);
      m_Hz.copyIn(dst_slot, src.m_Hz, src_slot);
      m_Hderfac.copyIn(dst_slot, src.m_Hderfac, src_slot);
    }

    // For distance-to-plane -- to go into a new class, NewtonIntersector
    // MPlex4V m_Cx, m_Cy, m_Cz; // p(3) for distance to plane
    // MPlex3V m_Dx, m_Dy, m_Dz; // Derivative of the above

    static void hermite_1d(const MPF &x1, const MPF &p1,
                           const MPF &x2, const MPF &p2,
                           const MPF &derfac,
                           MP4V &h);

    void calculate_coeffs(const StatePlex &sp1, const StatePlex &sp2, const MPF &inv_k);

    void evaluate(const MPF &t, MPF &x, MPF &y, MPF &z) const;
    void evaluate(const MPF &t, MPF &x, MPF &y, MPF &z, MPF &dx, MPF &dy, MPF &dz) const;
    void evaluate(const MPF &t, StatePlex &sp) const;
  };

  struct Hermite3DOnPlane {
    MPlex4V m_D;    // p(3) for distance to plane
    //MPlex3V m_dDdt; // derivative of the above
    MPF     m_T;    // t at solution

    void init_coeffs(const Hermite3D &h, const MP3V& pos, const MP3V& zdir);

    void evaluate(const MPF &t, MPF &d);
    void evaluate(const MPF &t, MPF &d, MPF &dddt);

    void solve();
  };

}  // namespace mkfit::mini_propagators

#endif
