#ifndef RecoTracker_MkFitCore_src_KalmanUtilsMPlex_h
#define RecoTracker_MkFitCore_src_KalmanUtilsMPlex_h

#include "RecoTracker/MkFitCore/interface/Track.h"
#include "Matrix.h"

namespace mkfit {

  //------------------------------------------------------------------------------

  enum KalmanFilterOperation { KFO_Calculate_Chi2 = 1, KFO_Update_Params = 2, KFO_Local_Cov = 4 };

  inline void kalmanCheckChargeFlip(MPlexLV& outPar, MPlexQI& Chg, int N_proc) {
    for (int n = 0; n < NN; ++n) {
      if (n < N_proc && outPar.At(n, 3, 0) < 0) {
        Chg.At(n, 0, 0) = -Chg.At(n, 0, 0);
        outPar.At(n, 3, 0) = -outPar.At(n, 3, 0);
      }
    }
  }

  //------------------------------------------------------------------------------

  void kalmanUpdate(const MPlexLS& psErr,
                    const MPlexLV& psPar,
                    const MPlexHS& msErr,
                    const MPlexHV& msPar,
                    MPlexLS& outErr,
                    MPlexLV& outPar,
                    const int N_proc);

  void kalmanPropagateAndUpdate(const MPlexLS& psErr,
                                const MPlexLV& psPar,
                                MPlexQI& Chg,
                                const MPlexHS& msErr,
                                const MPlexHV& msPar,
                                MPlexLS& outErr,
                                MPlexLV& outPar,
                                MPlexQI& outFailFlag,
                                const int N_proc,
                                const PropagationFlags& propFlags,
                                const bool propToHit);

  void kalmanComputeChi2(const MPlexLS& psErr,
                         const MPlexLV& psPar,
                         const MPlexQI& inChg,
                         const MPlexHS& msErr,
                         const MPlexHV& msPar,
                         MPlexQF& outChi2,
                         const int N_proc);

  void kalmanPropagateAndComputeChi2(const MPlexLS& psErr,
                                     const MPlexLV& psPar,
                                     const MPlexQI& inChg,
                                     const MPlexHS& msErr,
                                     const MPlexHV& msPar,
                                     MPlexQF& outChi2,
                                     MPlexLV& propPar,
                                     MPlexQI& outFailFlag,
                                     const int N_proc,
                                     const PropagationFlags& propFlags,
                                     const bool propToHit);

  void kalmanOperation(const int kfOp,
                       const MPlexLS& psErr,
                       const MPlexLV& psPar,
                       const MPlexHS& msErr,
                       const MPlexHV& msPar,
                       MPlexLS& outErr,
                       MPlexLV& outPar,
                       MPlexQF& outChi2,
                       const int N_proc);

  //------------------------------------------------------------------------------

  void kalmanUpdateEndcap(const MPlexLS& psErr,
                          const MPlexLV& psPar,
                          const MPlexHS& msErr,
                          const MPlexHV& msPar,
                          MPlexLS& outErr,
                          MPlexLV& outPar,
                          const int N_proc);

  void kalmanPropagateAndUpdateEndcap(const MPlexLS& psErr,
                                      const MPlexLV& psPar,
                                      MPlexQI& Chg,
                                      const MPlexHS& msErr,
                                      const MPlexHV& msPar,
                                      MPlexLS& outErr,
                                      MPlexLV& outPar,
                                      MPlexQI& outFailFlag,
                                      const int N_proc,
                                      const PropagationFlags& propFlags,
                                      const bool propToHit);

  void kalmanComputeChi2Endcap(const MPlexLS& psErr,
                               const MPlexLV& psPar,
                               const MPlexQI& inChg,
                               const MPlexHS& msErr,
                               const MPlexHV& msPar,
                               MPlexQF& outChi2,
                               const int N_proc);

  void kalmanPropagateAndComputeChi2Endcap(const MPlexLS& psErr,
                                           const MPlexLV& psPar,
                                           const MPlexQI& inChg,
                                           const MPlexHS& msErr,
                                           const MPlexHV& msPar,
                                           MPlexQF& outChi2,
                                           MPlexLV& propPar,
                                           MPlexQI& outFailFlag,
                                           const int N_proc,
                                           const PropagationFlags& propFlags,
                                           const bool propToHit);

  void kalmanOperationEndcap(const int kfOp,
                             const MPlexLS& psErr,
                             const MPlexLV& psPar,
                             const MPlexHS& msErr,
                             const MPlexHV& msPar,
                             MPlexLS& outErr,
                             MPlexLV& outPar,
                             MPlexQF& outChi2,
                             const int N_proc);

  //------------------------------------------------------------------------------

  void kalmanUpdatePlane(const MPlexLS& psErr,
                         const MPlexLV& psPar,
                         const MPlexQI& Chg,
                         const MPlexHS& msErr,
                         const MPlexHV& msPar,
                         const MPlexHV& plNrm,
                         const MPlexHV& plDir,
                         const MPlexHV& plPnt,
                         MPlexLS& outErr,
                         MPlexLV& outPar,
                         const int N_proc);

  void kalmanPropagateAndUpdatePlane(const MPlexLS& psErr,
                                     const MPlexLV& psPar,
                                     MPlexQI& Chg,
                                     const MPlexHS& msErr,
                                     const MPlexHV& msPar,
                                     const MPlexHV& plNrm,
                                     const MPlexHV& plDir,
                                     const MPlexHV& plPnt,
                                     MPlexLS& outErr,
                                     MPlexLV& outPar,
                                     MPlexQI& outFailFlag,
                                     const int N_proc,
                                     const PropagationFlags& propFlags,
                                     const bool propToHit);

  void kalmanPropagateAndUpdateAndChi2Plane(const MPlexLS& psErr,
                                            const MPlexLV& psPar,
                                            MPlexQI& Chg,
                                            const MPlexHS& msErr,
                                            const MPlexHV& msPar,
                                            const MPlexHV& plNrm,
                                            const MPlexHV& plDir,
                                            const MPlexHV& plPnt,
                                            MPlexLS& outErr,
                                            MPlexLV& outPar,
                                            MPlexQI& outFailFlag,
                                            MPlexQF& outChi2,
                                            const int N_proc,
                                            const PropagationFlags& propFlags,
                                            const bool propToHit,
                                            const MPlexQI* noMatEffPtr = nullptr,
                                            const MPlexQI* doCPE = nullptr,
                                            cpe_func cpe_corr_func = nullptr);

  void kalmanComputeChi2Plane(const MPlexLS& psErr,
                              const MPlexLV& psPar,
                              const MPlexQI& inChg,
                              const MPlexHS& msErr,
                              const MPlexHV& msPar,
                              const MPlexHV& plNrm,
                              const MPlexHV& plDir,
                              const MPlexHV& plPnt,
                              MPlexQF& outChi2,
                              const int N_proc);

  void kalmanPropagateAndComputeChi2Plane(const MPlexLS& psErr,
                                          const MPlexLV& psPar,
                                          const MPlexQI& inChg,
                                          const MPlexHS& msErr,
                                          const MPlexHV& msPar,
                                          const MPlexHV& plNrm,
                                          const MPlexHV& plDir,
                                          const MPlexHV& plPnt,
                                          MPlexQF& outChi2,
                                          MPlexLV& propPar,
                                          MPlexQI& outFailFlag,
                                          const int N_proc,
                                          const PropagationFlags& propFlags,
                                          const bool propToHit);

  void kalmanOperationPlane(const int kfOp,
                            const MPlexLS& psErr,
                            const MPlexLV& psPar,
                            const MPlexQI& Chg,
                            const MPlexHS& msErr,
                            const MPlexHV& msPar,
                            const MPlexHV& plNrm,
                            const MPlexHV& plDir,
                            const MPlexHV& plPnt,
                            MPlexLS& outErr,
                            MPlexLV& outPar,
                            MPlexQF& outChi2,
                            const int N_proc);

  void kalmanOperationPlaneLocal(const int kfOp,
                                 const MPlexLS& psErr,
                                 const MPlexLV& psPar,
                                 const MPlexQI& Chg,
                                 const MPlexHS& msErr,
                                 const MPlexHV& msPar,
                                 const MPlexHV& plNrm,
                                 const MPlexHV& plDir,
                                 const MPlexHV& plPnt,
                                 MPlexLS& outErr,
                                 MPlexLV& outPar,
                                 MPlexQF& outChi2,
                                 const int N_proc,
                                 const MPlexQI* doCPE = nullptr,
                                 cpe_func cpe_corr_func = nullptr,
                                 bool use_param_b_field = false,
                                 // Determinant of the 2x2 residual covariance in the
                                 // module's local frame, the matrix chi2 is formed
                                 // with. Cramer already computes it in double and
                                 // throws it away. chi2 = r^T V^-1 r is dimensionless,
                                 // so it has divided the precision out; -0.5*ln(det V)
                                 // is what puts it back, and it is what tells a
                                 // macro-pixel hit from a 2S strip hit at equal chi2.
                                 MPlexQF* outDetV = nullptr);

}  // end namespace mkfit
#endif
