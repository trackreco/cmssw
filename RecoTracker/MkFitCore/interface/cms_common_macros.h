#ifndef RecoTracker_MkFitCore_interface_cms_common_macros_h
#define RecoTracker_MkFitCore_interface_cms_common_macros_h

#ifdef MKFIT_STANDALONE
#include <cmath>
#define CMS_SA_ALLOW
#else
#include "FWCore/Utilities/interface/isFinite.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"
#endif

namespace mkfit {
  inline bool isFinite(float x) {
#ifdef MKFIT_STANDALONE
    return std::isfinite(x);
#else
    return edm::isFinite(x);
#endif
  }
} // namespace mkfit

#endif
