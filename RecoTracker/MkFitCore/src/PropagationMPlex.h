#ifndef RecoTracker_MkFitCore_src_PropagationMPlex_h
#define RecoTracker_MkFitCore_src_PropagationMPlex_h

#include "Matrix.h"
#include "RecoTracker/MkFitCore/interface/MathInlineCore.h"

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
                    const MPlexQF* sPerp,
                    MPlexQF& pathL,
                    MPlexLV& outPar,
                    MPlexLL& errorProp,
                    MPlexQI& outFailFlag,
                    const int N_proc,
                    const PropagationFlags& pflags);

  // Selects the root form used by getS() in the plane solve: true (default) the
  // numerically stable one, false the historical (-B +- sqrt)/(2C). A runtime
  // switch so the two can be compared in one binary. The A/B that settled it
  // is atticked: mkFit-external/mkfit-standalone-attic/prop-kalman-validation.
  extern bool g_getS_stable_root;
  // Diagnostic: multiplies radL in applyMaterialEffects. Default 1.
  extern float g_mat_scale;
  // Diagnostic: multiplies the energy-loss straggling variance into err(3,3).
  extern float g_mat_eloss_var_scale;
  // Diagnostic: extra radL factor for the FORWARD PIXEL DISCS only.
  extern float g_mat_scale_fwdpix;

  void propagateHelixToPlaneMPlex(const MPlexLS& inErr,
                                  const MPlexLV& inPar,
                                  const MPlexQI& inChg,
                                  const MPlexHV& plPnt,
                                  const MPlexHV& plNrm,
                                  const MPlexQF* sPerp,
                                  MPlexLS& outErr,
                                  MPlexLV& outPar,
                                  MPlexQI& outFailFlag,
                                  const int N_proc,
                                  const PropagationFlags& pflags,
                                  const MPlexQI* noMatEffPtr = nullptr);

  // Common functions: PropagationMPlexCommon.cc

  void applyMaterialEffects(const MPlexQF& hitsRl,
                            const MPlexQF& hitsXi,
                            const MPlexQF& propSign,
                            const MPlexHV& plNrm,
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            const int N_proc);

  void MultHelixPropFull(const MPlexLL& A, const MPlexLS& B, MPlexLL& C);
  void MultHelixPropTranspFull(const MPlexLL& A, const MPlexLL& B, MPlexLS& C);

}  // end namespace mkfit
#endif
