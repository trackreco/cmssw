#ifndef RecoTracker_MkFitCore_interface_MatrixSTypes_h
#define RecoTracker_MkFitCore_interface_MatrixSTypes_h

#include "Math/SMatrix.h"

namespace mkfit {

  typedef ROOT::Math::SMatrix<float, 6, 6, ROOT::Math::MatRepSym<float, 6> > SMatrixSym66;
  typedef ROOT::Math::SMatrix<float, 6> SMatrix66;
  typedef ROOT::Math::SVector<float, 6> SVector6;

  typedef ROOT::Math::SMatrix<float, 3> SMatrix33;
  typedef ROOT::Math::SMatrix<float, 3, 3, ROOT::Math::MatRepSym<float, 3> > SMatrixSym33;
  typedef ROOT::Math::SVector<float, 3> SVector3;

  typedef ROOT::Math::SMatrix<float, 2> SMatrix22;
  typedef ROOT::Math::SMatrix<float, 2, 2, ROOT::Math::MatRepSym<float, 2> > SMatrixSym22;
  typedef ROOT::Math::SVector<float, 2> SVector2;

  typedef ROOT::Math::SMatrix<float, 3, 6> SMatrix36;
  typedef ROOT::Math::SMatrix<float, 6, 3> SMatrix63;

  typedef ROOT::Math::SMatrix<float, 2, 6> SMatrix26;
  typedef ROOT::Math::SMatrix<float, 6, 2> SMatrix62;

  template <typename Matrix>
  inline void diagonalOnly(Matrix& m) {
    for (int r = 0; r < m.kRows; r++) {
      for (int c = 0; c < m.kCols; c++) {
        if (r != c)
          m[r][c] = 0.f;
      }
    }
  }

  template <typename Matrix>
  void dumpMatrix(Matrix m, const char *pfx0= "", const char *pfxN="") {
    for (int r = 0; r < m.kRows; ++r) {
      std::cout << (r == 0 ? pfx0 : pfxN);
      for (int c = 0; c < m.kCols; ++c) {
        std::cout << std::setw(12) << m.At(r, c) << " ";
      }
      std::cout << std::endl;
    }
  }

  inline float hipo(float x, float y) { return std::sqrt(x * x + y * y); }

  inline float hipo_sqr(float x, float y) { return x * x + y * y; }

  inline void sincos4(const float x, float& sin, float& cos) {
    // Had this writen with explicit division by factorial.
    // The *whole* fitting test ran like 2.5% slower on MIC, sigh.

    const float x2 = x * x;
    cos = 1.f - 0.5f * x2 + 0.04166667f * x2 * x2;
    sin = x - 0.16666667f * x * x2;
  }

}  // namespace mkfit

#endif
