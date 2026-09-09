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
  int seed = -1;        // index of seed in event trSeeds_
  int global_seed = -1; // index of seed in event seedTracks_ (from label of seed)
  int sim = -1;         // index of sim track in event simTracks_
  int cand = -1;        // candidateTracks_ index (valid for final stage)
  int stage_ids[3] {-1, -1, -1}; // indices of stages
};

struct TrCandStage {
  int id = -1;
  int meta_id = -1;         // link to global meta
  int parent_stage_id = -1; // previous stage (for multi-stage linkage)

  int stage = 0;            // 0=FwdSearch, 1=BkwFit, 2=BkwSearch, as in SteeringParams::IterationType_e
  int iteration_idx = 0;    // which iteration

  int root_state_id = -1;   // first state in this stage
  int final_state_id = -1;  // best final state in this stage
};

struct TrCandState {
  int id = -1;
  int parent_id = -1;
  int meta_id = -1;
  int stage_id = -1;

  int layer;
  int step; // not sure -- for now just depth number, incresed for every new state

  // missing: n_hits, n_missed_hits

  EBiVec3 kine;
  mkfit::TrackState state; // TrackState at this point

  int  search_id = -1; // set in post processing (might need multiple searches for thick layers)
  bool has_children = false; // true if it has children
  bool on_final_path = false; // true if it is on the final selected candidate path (final candidate and its ancestors)
};

// TrLayerSearch -- one hit-search window, for one candidate state, on one layer.
//
// Filled by MkFinderV2p2::process_pre_select(), stage 1 ("initial propagation"):
// the mini-propagator is run to the two layer bounding surfaces, the full
// propagation supplies the covariance, and the two together give the phi/q
// window and the bin ranges that the stage-2 hit loop then scans.
//
// This is the v2p2 counterpart of the old RntDumper BinSearch plus the two
// PropInfos in CandInfo (ps_min / ps_max) -- keep them comparable.
struct TrLayerSearch {
  int id = -1;
  int state_id = -1;
  int layer = -1;
  int layer_sec = -1;      // second sub-layer of a double layer, -1 when not double
  bool is_barrel = false;
  bool is_outward = false; // false for the inward search into the pixels

  // Stage 1a -- mini-propagator (PA_Exact) onto the layer bounding surfaces,
  // in propagation order: entry is crossed first, exit second. NOT sorted by
  // q or phi. dalpha is the helix angle turned, fail_flag the propagation status.
  // The Hermite cubic used for the per-hit propagation is built from these two,
  // so its interpolation error scales as (dalpha_exit - dalpha_entry)^4.
  PropInfo prop_entry;     // MkBins::m_sp1
  PropInfo prop_exit;      // MkBins::m_sp2

  // Stage 1b -- search window from MkBins::determine_bin_windows().
  // phi_delta, dphi_track and dq_track are HALF-widths; q_min/q_max a full range.
  float phi_center = -999.99f, phi_delta = -999.99f;
  float q_center = -999.99f, q_min = -999.99f, q_max = -999.99f;
  float dphi_track = -999.99f, dq_track = -999.99f; // 3 sigma track errors

  // Covariance the window was built from (MkBinTrackCovExtract), i.e. err(0,0),
  // err(0,1), err(1,1), err(2,2) after the full propagation to prop_exit.
  // NOTE: the phi jacobian is evaluated at min(r_entry, r_exit) -- the smallest
  // radius the track crosses in the layer, where sigma_phi is largest, so one
  // window covers the whole layer. It used to be taken at m_isp (the entry edge)
  // while the covariance came from m_sp2 (the exit), which scaled dphi_track by
  // rout/rin: 1.33 at pixel layer 0. The ENDCAP dq jacobian is still at m_isp,
  // but |grad r| = 1 there so there is no scale factor. Recorded
  // here so the size of that mismatch can actually be measured.
  float cov_xx = -999.99f, cov_xy = -999.99f, cov_yy = -999.99f, cov_zz = -999.99f;

  // Stage 1c -- bin ranges. p2/q2 are exclusive; p wraps around, q does not.
  unsigned short p1 = 0, p2 = 0, q1 = 0, q2 = 0;             // primary layer
  unsigned short p1_sec = 0, p2_sec = 0, q1_sec = 0, q2_sec = 0; // secondary sub-layer

  // Stage 2 tallies over the hit loop. n_hits_pqueue is capped at
  // MkBins::NEW_MAX_HIT and equals the number of TrHitMatch with passed_pqueue.
  int n_hits_scanned = 0;  // hits visited inside the bin ranges
  int n_hits_masked = 0;   // of those, rejected by the iteration hit mask
  int n_hits_presel = 0;   // of those, passed the dq/dphi pre-selection
  int n_hits_pqueue = 0;   // of those, survived the priority queue -> Kalman

  // Only computed with MKFIT_TRACE_PROP_COMPARE, otherwise left at zero. The
  // member itself is unconditional: unlike the per-hit compare state this is one
  // vector per search, not per scanned hit, and keeping it out of the #ifdef
  // means one less layout the ROOT dictionary has to agree with.
  //
  // How far the Hermite cubic strays from the helix in the middle of the layer,
  // where it is furthest from the two endpoints it was built to interpolate:
  // evaluate the cubic at t = 0.5, then propagate the mini-propagator exactly to
  // the bounding surface (r or z) that point lands on, and difference the two.
  // Both are on the same surface, so this is a pure in-surface deviation.
  EVec3 hermite_mid_dev { 0, 0, 0 };
};

struct TrHitMatch {
  int id;
  int state_id;
  int search_id = -1; // TrLayerSearch this hit was scanned for
  int layer; // NOTE: TrCandState.layer we are pointing to is PREVIUOUS layer
  int hit;
  bool mc_match; // is hit mc-matching

  // hit info -- can get from event->layer-of-hits ....
  // EVec3 hit_pos;
  // float hit_phi;
  // float hit_q;
  // float hit_qbar;
  // int   hit_lbl;

  // pre-selection quantities.
  // kine_on_plane is the state the dphi / dq / residual_* below were computed
  // from -- currently the mini-propagator PA_Line step onto the module plane.
  EBiVec3 kine_on_plane { EVec3(), EVec3() };
  // Hermite parameter at the module-plane crossing. Inside [0,1] the crossing is
  // between the layer bounding surfaces; outside it the cubic is extrapolating
  // and both kine_on_plane and the window it was pre-selected against are suspect.
  float t_hermite = -999.99;
  // Distance from the Hermite point at t_hermite to the module plane, in cm.
  // Hermite3DOnPlane::solve() takes a SINGLE Newton step, so this is not
  // machine-zero by construction -- it is the measure of whether one step
  // sufficed for this hit.
  float d_plane_h3 = -999.99;
  float dphi = -999.99;
  float dq = -999.99;
  // Half-length of the hit in the q direction, i.e. half the strip length in the
  // barrel. Needed to make dq interpretable: for a long strip the residual is
  // dominated by this, not by the track error, so dq / sigma_track is NOT a
  // covariance pull there. It is also the second term of the dq cut.
  float hit_q_half_len = -999.99;
  float score = -999.99; // usually dphi, could add dq checks
  bool passed_preselect = false;

  float residual_x = -999.99; // distance from hit to track in precise / phi direction, for accepted hits
  float residual_y = -999.99; // distance from hit to track in coarse direction, for accepted hits
  float residual_z = -999.99; // distance from detector plane, should be about 0

  // after pre-selection via priority-queue
  int  rank = -1; // for those that pass pqueue selection (would be nice to have it for others, std::vec instead of pqueue for MKFIT_TRACE?)
  bool passed_pqueue = false;

  int kalman_id = -1;

  // Optional -- with MKFIT_TRACE_PROP_COMPARE.
  // The OTHER of the two cheap propagations onto the module plane: the code runs
  // both a mini-propagator PA_Line step and a Hermite-cubic plane solve for every
  // scanned hit, and uses one of them (see kine_on_plane) for the cuts. This is
  // the one it did not use, so the two can be differenced for ALL scanned hits,
  // including rejected ones.
  // For the pre-selected hits the exact reference is already in the trace and
  // needs no define: TrKalmanUpdate::propagated_state, via kalman_id.
  // Set the macro in Makefile.config -- it must be in CPPFLAGS so that the code
  // and the ROOT dictionary agree on the layout.
#ifdef MKFIT_TRACE_PROP_COMPARE
  EBiVec3 kine_on_plane_cmp { EVec3(), EVec3() };
#endif
};

struct TrKalmanUpdate {
  int id;
  int hit_match_id;
  int state_id_in;      // state before update; could get it from hitmatch.state_id
  int state_id_out = -1; // state after update, -1 if not accepted

  float   chi2 = -999.99f;
  float   chi2_trk = -999.99f;
  bool    accepted = false;    // this hit advanced the state

  // Propagated track parameters are same as in TrHitMatch (as we pass it to Kalman update),
  // The local dx, dy, dz distance from point to hit is the same as for hit match.

  // Optional -- with MKFIT_TRACE_KALMAN_DEBUG.
  // For an accepted update the post-update state is also reachable as
  // trCandStates_[state_id_out].state; this member is what gives it for the
  // REJECTED ones too, which is why it is worth the space when studying chi2
  // and error-matrix behaviour. Set the macro in Makefile.config -- it must be
  // in CPPFLAGS so that the code and the ROOT dictionary agree on the layout.
#ifdef MKFIT_TRACE_KALMAN_DEBUG
  mkfit::TrackState propagated_state {}; // pre-update, propagated onto the hit plane
  mkfit::TrackState updated_state {};    // post-update
#endif
};

// Another struct for missed layer, for some reason?

// Inspection structs

struct SeedVecInsp {
  int n_pTNs;
  int n_TNs;
  int n_ps;
  int n_total() const { return n_pTNs + n_TNs + n_ps; }
};



#endif
