#ifndef RecoTracker_MkFitCore_standalone_RntDumper_RntStructs_h
#define RecoTracker_MkFitCore_standalone_RntDumper_RntStructs_h

// Avoid MkFit includes for now to simpligy pure ROOT builds.
// #include "RecoTracker/MkFitCore/interface/

#include "ROOT/REveVector.hxx"
#include "Math/Point3D.h"
#include "Math/Vector3D.h"

// From CMSSW data formats
/// point in space with cartesian internal representation
typedef ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<float> > XYZPointF;
/// spatial vector with cartesian internal representation
typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<float> > XYZVectorF;
/// spatial vector with cylindrical internal representation using pseudorapidity
typedef ROOT::Math::DisplacementVector3D<ROOT::Math::CylindricalEta3D<float> > RhoEtaPhiVectorF;
/// spatial vector with polar internal representation
/// WARNING: ROOT dictionary not provided for the type below
// typedef ROOT::Math::DisplacementVector3D<ROOT::Math::Polar3D<float> > RThetaPhiVectorF;


/*
struct XXX {

  XXX() = default;
  XXX& operator=(const XXX&) = default;
};
*/

using RVec = ROOT::Experimental::REveVector;

struct HeaderLayer {
  int event, iter_idx, iter_algo, eta_region, layer;
  float qb_min, qb_max; // qbar layer limits, r for barrel, z for endcap
  bool is_barrel, is_pix, is_stereo;

  HeaderLayer() = default;
  HeaderLayer& operator=(const HeaderLayer&) = default;
};

struct State {
  RVec pos, mom;

  State() = default;
  State& operator=(const State&) = default;
};

struct PropState : public State {
  float dalpha; // helix angle during propagation
  int fail_flag;

  PropState() = default;
  PropState& operator=(const PropState&) = default;
};

struct SimSeedInfo {
  State s_sim;
  State s_seed;
  int   sim_lbl, seed_lbl, seed_idx;
  int   n_hits, n_match;
  bool  has_sim = false;

  float good_frac() const { return (float)n_match/n_hits; }

  SimSeedInfo() = default;
  SimSeedInfo& operator=(const SimSeedInfo&) = default;
};

struct BinSearch {
  float phi, dphi, q, dq;
  short unsigned int p1, p2, q1, q2;
  short int wsr;
  bool wsr_in_gap;
  bool has_nans = false;

  bool nan_check();

  BinSearch() = default;
  BinSearch& operator=(const BinSearch&) = default;
};

struct HitMatch {
  RVec p_hit, m_hit;
  bool prop_ok;

  /*
    "c_x/F:c_y:c_z:c_px:c_py:c_pz:" // c_ for center layer (i.e., origin)
    "h_x/F:h_y:h_z:h_px:h_py:h_pz:" // h_ for at hit (i.e., new)
    "dq/F:dphi:"
    "c_ddq/F:c_ddphi:c_accept/I:" // (1 dq, 2 dphi, 3 both)
    "h_ddq/F:h_ddphi:h_accept/I:" // (1 dq, 2 dphi, 3 both)
    "hit_q/F:hit_qhalflen:hit_qbar:hit_phi:"
    "c_prop_ok/I:h_prop_ok/I:chi2/F"
  */
};

struct CandMatch {
// header
// siminfo
/*
  "all_hits/I:matched_hits:"
  "acc_old/I:acc_new:acc_matched_old:acc_matched_new:"
  "pos0/I:pos1:pos2:pos3:"
  "dphi0/F:dphi1:dphi2:dphi3:"
  "chi20/F:chi21:chi22:chi23:"
  "idx0/I:idx1:idx2:idx3"

  int pcc_all_hits = 0, pcc_matced_hits = 0;
  int pcc_acc_old = 0, pcc_acc_matched_old = 0, pcc_acc_new = 0, pcc_acc_matched_new = 0;

  pcc_pos[0], pcc_pos[1], pcc_pos[2], pcc_pos[3],
  pmr0.dphi, pmr1.dphi, pmr2.dphi, pmr3.dphi,
  pmr0.chi2, pmr1.chi2, pmr2.chi2, pmr3.chi2,
  pmr0.idx, pmr1.idx, pmr2.idx, pmr3.idx
*/
};

struct CandInfo {
  SimSeedInfo ssi;
  State s_ctr;
  PropState ps_min, ps_max;
  BinSearch bso;
  BinSearch bsn;
  bool has_nans = false;

  CandInfo(const SimSeedInfo &s, const State &c) : ssi(s), s_ctr(c) {}

  void nan_check();

  CandInfo() = default;
  CandInfo& operator=(const CandInfo&) = default;
};

struct FailedPropInfo {
  SimSeedInfo ssi;
  State s_prev;
  State s_final;
  bool has_nans = false;

  FailedPropInfo(const SimSeedInfo &s, const State &p, const State &f) :
    ssi(s), s_prev(p), s_final(f) {}

  void nan_check();

  FailedPropInfo() = default;
  FailedPropInfo& operator=(const FailedPropInfo&) = default;
};

#endif
