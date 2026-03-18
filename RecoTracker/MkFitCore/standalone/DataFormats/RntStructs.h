#ifndef RecoTracker_MkFitCore_standalone_DataFormats_RntStructs_h
#define RecoTracker_MkFitCore_standalone_DataFormats_RntStructs_h

#include "RecoTracker/MkFitCore/interface/IdxChi2List.h"
#include "RecoTracker/MkFitCore/interface/TrackState.h"

#include "ROOT/REveVector.hxx"

/*
#include "Math/Point3D.h"
#include "Math/Vector3D.h"
#include "Math/SMatrix.h"

typedef ROOT::Math::SMatrix<float, 6, 6, ROOT::Math::MatRepSym<float, 6> > SMatrixSym66;

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
*/

//==============================================================
// Basic kine structs
//==============================================================

using EVec3 = ROOT::Experimental::REveVector;

struct EBiVec3 {
  EVec3 pos, mom;
};

//==============================================================
// Oldish RntDumper into TTree / RNTuple, from selectHitIndices
//==============================================================

struct PropInfo : public EBiVec3 {
  float dalpha;  // helix angle during propagation
  int fail_flag;
};

struct HeaderLayer {
  int event, iter_idx, iter_algo, eta_region, layer;
  float qb_min, qb_max;  // qbar layer limits, r for barrel, z for endcap
  bool is_barrel, is_pix, is_stereo;
};

struct SimSeedInfo {
  EBiVec3 s_sim;
  EBiVec3 s_seed;
  int sim_lbl, seed_lbl, seed_idx;
  int n_hits, n_match;
  bool has_sim = false;

  float good_frac() const { return (float)n_match / n_hits; }
};

struct BinSearch {
  float phi, dphi, q, dq;
  short unsigned int p1, p2, q1, q2;
  short int wsr;
  bool wsr_in_gap;
  bool has_nans = false;

  bool nan_check();
};

struct HitInfo {
  EVec3 hit_pos;
  float hit_q, hit_qhalflen, hit_qbar, hit_phi;
  int hit_lbl;
};

struct HitMatchInfo : public HitInfo {
  EVec3 trk_pos, trk_mom;
  float ddq, ddphi;
  float chi2_true;
  int hit_index;
  bool match;
  bool presel;
  bool prop_ok;
  bool has_ic2list{false};
  mkfit::IdxChi2List ic2list;

  bool accept() const { return presel && prop_ok; }
};

struct CandInfo {
  SimSeedInfo ssi;
  EBiVec3 s_ctr;
  PropInfo ps_min, ps_max;
  BinSearch bso;
  BinSearch bsn;
  std::vector<HitMatchInfo> hmi;
  int n_all_hits = 0, n_hits_pass = 0, n_hits_match = 0, n_hits_pass_match = 0;
  int ord_first_match = -1;
  float dphi_first_match = -9999.0f, dq_first_match = -9999.0f;
  bool has_nans = false;

  CandInfo() = default;

  CandInfo(const SimSeedInfo& s, const EBiVec3& c) : ssi(s), s_ctr(c) {}

  void nan_check();
  void reset_hits_match() {
    n_all_hits = n_hits_pass = n_hits_match = n_hits_pass_match = 0;
    ord_first_match = -1;
    dphi_first_match = dq_first_match = -9999.0f;
  }

  bool assignIdxChi2List(const mkfit::IdxChi2List& ic2l) {
    for (auto& hm : hmi) {
      if (hm.hit_index == ic2l.hitIdx) {
        hm.has_ic2list = true;
        hm.ic2list = ic2l;
        return true;
      }
    }
    return false;
  }
};

struct FailedPropInfo {
  SimSeedInfo ssi;
  EBiVec3 s_prev;
  EBiVec3 s_final;
  bool has_nans = false;

  FailedPropInfo() = default;

  FailedPropInfo(const SimSeedInfo& s, const EBiVec3& p, const EBiVec3& f) : ssi(s), s_prev(p), s_final(f) {}

  void nan_check();
};

//==============================================================
// Trace and RDF stuff
//==============================================================

struct TrCandMeta {
  int id;
  int event;
  int sub_seed = -1; // index of seed in event currentSeedTracks_ (filled from builder in standalone)
  int seed = -1; // index of seed in event seedTracks_
  int sim = -1; // index of sim track in event simTracks_
  int cand = -1; // index of candidate in event candidateTracks_
  int root_state_id = -1;
  int final_state_id = -1;
};

struct TrCandState {
  int id = -1;
  int pid = -1;
  int meta_id = -1;

  int layer;
  int step; // not sure -- for now just depth number, incresed for every new state

  // missing direction, forward / backwards, now only backward
  // n_hits, n_missed_hits

  EBiVec3 kine;
  // covariance ? mkfit::TrackState ?

  bool has_children = false; // true if it has children
  bool on_final_path = false; // true if it is on the final selected candidate path (final candidate and its ancestors)
};

struct TrHitMatch {
  int id;
  int state_id;
  int layer; // or HitOnTrack?
  int hit;
  bool mc_match; // is hit mc-matching

  // hit info -- can get from event->layer-of-hits ....
  // EVec3 hit_pos;
  // float hit_phi;
  // float hit_q;
  // float hit_qbar;
  // int   hit_lbl;

  // pre-selection quantities
  EBiVec3 kine_on_plane { EVec3(), EVec3() }; // Hermite-propagated track state
  float dphi = -999.99;
  float dq = -999.99;
  float score = -999.99; // usually dphi, could add dq checks
  bool passed_preselect = false;

  float residual_x = -999.99; // distance from hit to track in precise / phi direction, for accepted hits
  float residual_y = -999.99; // distance from hit to track in coarse direction, for accepted hits
  float residual_z = -999.99; // distance from detector plane, should be about 0

  // after pre-selection via priority-queue
  int  rank = -1; // for those that pass pqueue selection (would be nice to have it for others, std::vec instead of pqueue for MKFIT_TRACE?)
  bool passed_pqueue = false;

  // if passed, add int kalman-id ?
  // yes, for now just cram in kalman state
  mkfit::TrackState kalman_state {};
  float kalman_chi2 = -1.0f;
  bool kalman_accepted = false;
};

struct TrKalmanUpdate {
  int id;
  int state_id_in;       // state before update
  int state_id_out;      // state after update (new CandID)
  int layer;  // or HitOnTrack? better, match_id !
  int hit;

  mkfit::TrackState trk_state {};
  // propagated track state, parameters are same as in TrHitMatch (as we pass it to Kalman update)
  // Need covariance matrix, sigh
  // Also need dx, dy, distance from point to hit in detector plane coordinate system.
  // Again, this should be the same as for hit match.

  // Link back to HitMmatch?

  float   chi2;
  float   chi2_trk;

  bool    accepted;    // this hit advanced the state
};

// Another struct for missed layer, for some reason

#endif
