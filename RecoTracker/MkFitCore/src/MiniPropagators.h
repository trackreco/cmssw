#ifndef RecoTracker_MkFitCore_src_MiniPropagators_h
#define RecoTracker_MkFitCore_src_MiniPropagators_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"
#include "Matrix.h"

namespace mkfit::mini_propagators {

  enum PropAlgo_e { PA_Line, PA_Quadratic, PA_Exact };

  struct State {
    float x, y, z;
    float px, py, pz;

    State() = default;
    State(const MPlexLV& par, int ti);
  };

  struct InitialState : public State {
    float inv_pt, inv_k;
    float theta;

    InitialState(const MPlexLV& par, const MPlexQI& chg, int ti) :
      InitialState(State(par, ti), chg.constAt(ti, 0, 0), par.constAt(ti, 3, 0), par.constAt(ti, 5, 0))
    {}

    InitialState(State s, short charge, float ipt, float tht, float bf=Config::Bfield) :
      State(s), inv_pt(ipt), theta(tht) {
        inv_k = ((charge < 0) ? 0.01f : -0.01f) * Const::sol * bf;
    }

    bool propagate_to_r(PropAlgo_e algo, float R, State& c, bool update_momentum) const;
    bool propagate_to_z(PropAlgo_e algo, float Z, State& c, bool update_momentum) const;

  };
}

#endif
