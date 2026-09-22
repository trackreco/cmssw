#ifndef RecoTracker_MkFitCore_interface_TrackState_h
#define RecoTracker_MkFitCore_interface_TrackState_h

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/MatrixSTypes.h"
#include "RecoTracker/MkFitCore/interface/MathInlineFunctions.h"

#include <cmath>
#include <vector>

namespace mkfit {

  //==============================================================================
  // TrackState
  //==============================================================================

  struct TrackState  //  possible to add same accessors as track?
  {
  public:
    TrackState() : charge(0), valid(true) {}
    TrackState(int charge, const SVector3& pos, const SVector3& mom, const SMatrixSym66& err)
        : parameters(SVector6(pos.At(0), pos.At(1), pos.At(2), mom.At(0), mom.At(1), mom.At(2))),
          errors(err),
          charge(charge),
          valid(true) {}
    SVector3 position() const { return SVector3(parameters[0], parameters[1], parameters[2]); }
    SVector6 parameters;
    SMatrixSym66 errors;
    short charge;
    bool valid;

    // Access for packing into Matriplexes
    const float* parArray() const { return parameters.Array(); }
    const float* errArray() const { return errors.Array(); }
    // And non-const for copying back out.
    float* parArray_nc() { return parameters.Array(); }
    float* errArray_nc() { return errors.Array(); }

    // track state position
    float x() const { return parameters.At(0); }
    float y() const { return parameters.At(1); }
    float z() const { return parameters.At(2); }
    float posR() const { return getHypot(x(), y()); }
    float posRsq() const { return x() * x() + y() * y(); }
    float posPhi() const { return getPhi(x(), y()); }
    float posEta() const { return getEta(posR(), z()); }

    // track state position errors
    float exx() const { return std::sqrt(errors.At(0, 0)); }
    float eyy() const { return std::sqrt(errors.At(1, 1)); }
    float ezz() const { return std::sqrt(errors.At(2, 2)); }
    float exy() const { return std::sqrt(errors.At(0, 1)); }
    float exz() const { return std::sqrt(errors.At(0, 2)); }
    float eyz() const { return std::sqrt(errors.At(1, 2)); }

    float eposR() const { return std::sqrt(getRadErr2(x(), y(), errors.At(0, 0), errors.At(1, 1), errors.At(0, 1))); }
    float eposPhi() const { return std::sqrt(getPhiErr2(x(), y(), errors.At(0, 0), errors.At(1, 1), errors.At(0, 1))); }
    float eposEta() const {
      return std::sqrt(getEtaErr2(x(),
                                  y(),
                                  z(),
                                  errors.At(0, 0),
                                  errors.At(1, 1),
                                  errors.At(2, 2),
                                  errors.At(0, 1),
                                  errors.At(0, 2),
                                  errors.At(1, 2)));
    }

    // track state momentum
    float invpT() const { return parameters.At(3); }
    float momPhi() const { return parameters.At(4); }
    float theta() const { return parameters.At(5); }
    float pT() const { return std::abs(1.f / parameters.At(3)); }
    float px() const { return pT() * std::cos(parameters.At(4)); }
    float py() const { return pT() * std::sin(parameters.At(4)); }
    float pz() const { return pT() / std::tan(parameters.At(5)); }
    float momEta() const { return getEta(theta()); }
    float p() const { return pT() / std::sin(parameters.At(5)); }

    float einvpT() const { return std::sqrt(errors.At(3, 3)); }
    float emomPhi() const { return std::sqrt(errors.At(4, 4)); }
    float etheta() const { return std::sqrt(errors.At(5, 5)); }
    float epT() const { return std::sqrt(errors.At(3, 3)) / (parameters.At(3) * parameters.At(3)); }
    float emomEta() const { return std::sqrt(errors.At(5, 5)) / std::sin(parameters.At(5)); }
    float epxpx() const { return std::sqrt(getPxPxErr2(invpT(), momPhi(), errors.At(3, 3), errors.At(4, 4))); }
    float epypy() const { return std::sqrt(getPyPyErr2(invpT(), momPhi(), errors.At(3, 3), errors.At(4, 4))); }
    float epzpz() const { return std::sqrt(getPyPyErr2(invpT(), theta(), errors.At(3, 3), errors.At(5, 5))); }

    void convertFromCartesianToCCS();
    void convertFromCCSToCartesian();
    SMatrix66 jacobianCCSToCartesian(float invpt, float phi, float theta) const;
    SMatrix66 jacobianCartesianToCCS(float px, float py, float pz) const;

    void convertFromGlbCurvilinearToCCS();
    void convertFromCCSToGlbCurvilinear();
    //last row/column are zeros
    SMatrix66 jacobianCCSToCurvilinear(float invpt, float cosP, float sinP, float cosT, float sinT, short charge) const;
    SMatrix66 jacobianCurvilinearToCCS(float px, float py, float pz, short charge) const;

    bool hasNanNSillyValues() const;
  };

  //==============================================================================
  // SimHitState
  //==============================================================================

  // Truth state at a SIM HIT: position and momentum, and nothing else.
  //
  // Indexed by mcHitID, i.e. parallel to Event::simHitsInfo_ -- the same
  // convention the legacy Event::simTrackStates_ uses.
  //
  // Deliberately NOT a TrackState, which is 112 B:
  //  - no covariance. A Geant truth state has none, and the sim TRACK's own
  //    covariance is a documented placeholder (err(i,i) = value^2, a 100 %
  //    relative error, singular at the origin). Writing it would be 84 bytes of
  //    zeros per sim hit.
  //  - no charge. It is a per-TRACK property, reachable from the same index as
  //      simTracks_[ simHitsInfo_[mcHitID].mcTrackID() ].charge()
  //
  // That is 24 B against 112, which is what makes the section affordable: the
  // April PU sample carries 383k-471k sim hits per event, so the full TrackState
  // form would be 49 MB/event and more than double the file, against 10.5
  // MB/event (+30 %) for this.
  //
  // INVALID IS ZERO MOMENTUM. There is no separate valid flag -- a real sim hit
  // never has |p| = 0, so `mom` all-zero means "no truth state for this hit",
  // which is the case for a rec hit whose sim link was not established. Test it
  // with is_valid() rather than by reading mcTrackID, because the two are not
  // equivalent: bestTkIdx() can clear the track link while the sim hit itself is
  // perfectly well defined (see the arbitration defect in RecoTracker/CLAUDE.md).
  struct SimHitState {
    SVector3 pos;
    SVector3 mom;

    SimHitState() : pos(0.f, 0.f, 0.f), mom(0.f, 0.f, 0.f) {}
    SimHitState(const SVector3 &p, const SVector3 &m) : pos(p), mom(m) {}
    SimHitState(float x, float y, float z, float px, float py, float pz)
        : pos(x, y, z), mom(px, py, pz) {}

    bool is_valid() const { return mom[0] != 0.f || mom[1] != 0.f || mom[2] != 0.f; }

    float x() const { return pos[0]; }
    float y() const { return pos[1]; }
    float z() const { return pos[2]; }
    float px() const { return mom[0]; }
    float py() const { return mom[1]; }
    float pz() const { return mom[2]; }

    float r() const { return std::hypot(pos[0], pos[1]); }
    float pT() const { return std::hypot(mom[0], mom[1]); }
    float p() const { return std::sqrt(mom[0] * mom[0] + mom[1] * mom[1] + mom[2] * mom[2]); }
    float momPhi() const { return std::atan2(mom[1], mom[0]); }
    float momEta() const {
      const float pt = pT();
      return std::log((p() + mom[2]) / (pt > 0.f ? pt : 1e-9f));
    }
  };

  typedef std::vector<SimHitState> SHSVec;

}  // namespace mkfit

#endif // RecoTracker_MkFitCore_interface_TrackState_h
