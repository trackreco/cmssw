#ifndef RecoTracker_MkFit_MkFitEventWrapper_h
#define RecoTracker_MkFit_MkFitEventWrapper_h

#include "RecoTracker/MkFit/interface/MkFitIndexLayer.h"

namespace mkfit {
  class Event;
  class TrackVec;
}

class MkFitEventWrapper {
public:
  MkFitEventWrapper();
  MkFitEventWrapper(MkFitIndexLayer&& indexLayers);
  ~MkFitEventWrapper();

  MkFitEventWrapper(MkFitEventWrapper&&) = default;
  MkFitEventWrapper& operator=(MkFitEventWrapper&&) = default;

  MkFitIndexLayer const& indexLayers() const { return indexLayer_; }

private:
  MkFitIndexLayuer indexLayers_; //!
  Event event_; //!
  TrackVec seeds_; //!
};

#endif
