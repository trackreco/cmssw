#ifndef RecoTracker_MkFitCore_src_PropagationMPlex_h
#define RecoTracker_MkFitCore_src_PropagationMPlex_h

#include "Matrix.h"

namespace mkfit {

  class PropagationFlags;

  inline void squashPhiMPlex(MPlexLV& par, const int N_proc) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      if (n < N_proc) {
        if (par(n, 4, 0) >= Const::PI)
          par(n, 4, 0) -= Const::TwoPI;
        if (par(n, 4, 0) < -Const::PI)
          par(n, 4, 0) += Const::TwoPI;
      }
    }
  }

  inline void squashPhiMPlexGeneral(MPlexLV& par, const int N_proc) {
#pragma omp simd
    for (int n = 0; n < NN; ++n) {
      par(n, 4, 0) -= std::floor(0.5f * Const::InvPI * (par(n, 4, 0) + Const::PI)) * Const::TwoPI;
    }
  }

  // Barrel / R: PropagationMPlex.cc

  void propagateLineToRMPlex(const MPlexLS& psErr,
                             const MPlexLV& psPar,
                             const MPlexHS& msErr,
                             const MPlexHV& msPar,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             const int N_proc);

  void propagateHelixToRMPlex(const MPlexLS& inErr,
                              const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexQF& msRad,
                              MPlexLS& outErr,
                              MPlexLV& outPar,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags& pflags,
                              const MPlexQI* noMatEffPtr = nullptr);

  void helixAtRFromIterativeCCSFullJac(const MPlexLV& inPar,
                                       const MPlexQI& inChg,
                                       const MPlexQF& msRad,
                                       MPlexLV& outPar,
                                       MPlexLL& errorProp,
                                       MPlexQI& outFailFlag,
                                       const int N_proc);

  void helixAtRFromIterativeCCS(const MPlexLV& inPar,
                                const MPlexQI& inChg,
                                const MPlexQF& msRad,
                                MPlexLV& outPar,
                                MPlexLL& errorProp,
                                MPlexQI& outFailFlag,
                                const int N_proc,
                                const PropagationFlags& pflags);

  // Parameters only, bitwise identical to helixAtRFromIterativeCCS(); pathD = transverse path length.
  void helixAtRFromIterativeCCSParOnly(const MPlexLV& inPar,
                                       const MPlexQI& inChg,
                                       const MPlexQF& msRad,
                                       MPlexLV& outPar,
                                       MPlexQF& pathD,
                                       MPlexQI& outFailFlag,
                                       const int N_proc,
                                       const PropagationFlags& pflags);

  // Endcap / Z: PropagationMPlexEndcap.cc

  void propagateHelixToZMPlex(const MPlexLS& inErr,
                              const MPlexLV& inPar,
                              const MPlexQI& inChg,
                              const MPlexQF& msZ,
                              MPlexLS& outErr,
                              MPlexLV& outPar,
                              MPlexQI& outFailFlag,
                              const int N_proc,
                              const PropagationFlags& pflags,
                              const MPlexQI* noMatEffPtr = nullptr);

  void helixAtZ(const MPlexLV& inPar,
                const MPlexQI& inChg,
                const MPlexQF& msZ,
                MPlexLV& outPar,
                MPlexLL& errorProp,
                MPlexQI& outFailFlag,
                const int N_proc,
                const PropagationFlags& pflags);

  // Plane: PropagationMPlexPlane.cc

  void helixAtPlane(const MPlexLV& inPar,
                    const MPlexQI& inChg,
                    const MPlexHV& plPnt,
                    const MPlexHV& plNrm,
                    MPlexQF& pathL,
                    MPlexLV& outPar,
                    MPlexLL& errorProp,
                    MPlexQI& outFailFlag,
                    const int N_proc,
                    const PropagationFlags& pflags);

  void propagateHelixToPlaneMPlex(const MPlexLS& inErr,
                                  const MPlexLV& inPar,
                                  const MPlexQI& inChg,
                                  const MPlexHV& plPnt,
                                  const MPlexHV& plNrm,
                                  MPlexLS& outErr,
                                  MPlexLV& outPar,
                                  MPlexQI& outFailFlag,
                                  const int N_proc,
                                  const PropagationFlags& pflags,
                                  const MPlexQI* noMatEffPtr = nullptr);

  // The same with the covariance in curvilinear coordinates, transported with the curvilinear Jacobian.
  void propagateHelixToPlaneMPlex(const MPlex5S& inErr,
                                  const MPlexLV& inPar,
                                  const MPlexQI& inChg,
                                  const MPlexHV& plPnt,
                                  const MPlexHV& plNrm,
                                  MPlex5S& outErr,
                                  MPlexLV& outPar,
                                  MPlexQI& outFailFlag,
                                  const int N_proc,
                                  const PropagationFlags& pflags,
                                  const MPlexQI* noMatEffPtr = nullptr);

  // Propagation to the layer (R for barrel, Z for endcap) for the hit selection, with curvilinear input errors.
  // The parameters are exactly those of propagateHelixToRMPlex() / propagateHelixToZMPlex(). Of outErr only the
  // position block (x, y, z) is filled: the curvilinear error transported along the path, projected on the layer
  // surface along the track direction (what the CCS propagation to R / Z gives). The rest of outErr is not set.
  void propagateHelixToLayerPosErrCurv(const MPlex5S& inErr,
                                       const MPlexLV& inPar,
                                       const MPlexQI& inChg,
                                       const MPlexQF& msRZ,
                                       const bool isBarrel,
                                       MPlexLS& outErr,
                                       MPlexLV& outPar,
                                       MPlexQI& outFailFlag,
                                       const int N_proc,
                                       const PropagationFlags& pflags);

  // Common functions: PropagationMPlexCommon.cc

  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            const MPlexHV& plNrm,
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            const int N_proc,
                            // per-lane |p| [GeV] at which the multiple-scattering noise is evaluated, in place
                            // of the running estimate; nullptr = the running estimate
                            const float* msRefP = nullptr);

  // Curvilinear error (q/p, lambda, phi, x_perp, y_perp); parameters stay in CCS.

  // Material effects on a curvilinear error, computed in curvilinear coordinates.
  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            const MPlexHV& plNrm,
                            MPlex5S& outErr,
                            MPlexLV& outPar,
                            const int N_proc,
                            const float* msRefP = nullptr);

  // C = A * B * A^T for the transport, curvilinear->local and local->curvilinear jacobians; C may alias B
  void similarityCurvProp(const MPlex55& A, const MPlex5S& B, MPlex5S& C);
  void similarityCurv2Loc(const MPlex55& A, const MPlex5S& B, MPlex5S& C);
  void similarityLoc2Curv(const MPlex55& A, const MPlex5S& B, MPlex5S& C);
  void jacobianCCSToCurv(const MPlexLV& par, const MPlexQI& chg, MPlex56& jac);
  void jacobianCurvToCCS(const MPlexLV& par, const MPlexQI& chg, MPlex65& jac);
  void convertErrCCSToCurv(const MPlexLS& inErr, const MPlexLV& par, const MPlexQI& chg, MPlex5S& outErr);
  void convertErrCurvToCCS(const MPlex5S& inErr, const MPlexLV& par, const MPlexQI& chg, MPlexLS& outErr);

  // Curvilinear track errors are stored in the first 15 elements of the 6x6 storage, the rest zero
  // (Config::useCurvTrackErrors). Packed into an MPlexLS, the 5x5 is its first 15 elements.
  inline void copyErrCurvFromLS(const MPlexLS& in, MPlex5S& out) {
    std::copy(in.fArray, in.fArray + MPlex5S::kTotSize, out.fArray);
  }
  inline void copyOutErrCurv(const MPlex5S& err, const int i, float* out) {
    err.copyOut(i, out);
    std::fill(out + MPlex5S::kSize, out + MPlexLS::kSize, 0.f);
  }

  void MultHelixPropFull(const MPlexLL& A, const MPlexLS& B, MPlexLL& C);
  void MultHelixPropTranspFull(const MPlexLL& A, const MPlexLL& B, MPlexLS& C);

}  // end namespace mkfit
#endif
