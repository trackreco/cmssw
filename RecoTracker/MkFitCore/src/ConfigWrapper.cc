#include "RecoTracker/MkFitCore/interface/ConfigWrapper.h"
#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"

namespace mkfit {
  namespace ConfigWrapper {
    void initializeForCMSSW(bool silent) {
      Config::silent = silent;

      // to do backward fit to the first layer, not point of closest approach
      Config::includePCA = false;
    }

    void setNTotalLayers(int nTotalLayers) { Config::nTotalLayers = nTotalLayers; }
  }  // namespace ConfigWrapper
}  // namespace mkfit
