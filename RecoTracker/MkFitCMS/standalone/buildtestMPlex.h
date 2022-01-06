#ifndef RecoTracker_MkFitCMS_interface_buildtestMPlex_h
#define RecoTracker_MkFitCMS_interface_buildtestMPlex_h

#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"

namespace mkfit {

  class IterationConfig;
  class MkBuilder;

  void runBuildingTestPlexDumbCMSSW(Event& ev, const EventOfHits& eoh, MkBuilder& builder);

  double runBuildingTestPlexBestHit(Event& ev, const EventOfHits& eoh, MkBuilder& builder);
  double runBuildingTestPlexStandard(Event& ev, const EventOfHits& eoh, MkBuilder& builder);
  double runBuildingTestPlexCloneEngine(Event& ev, const EventOfHits& eoh, MkBuilder& builder);

  std::vector<double> runBtpCe_MultiIter(Event& ev, const EventOfHits& eoh, MkBuilder& builder, int n);

}  // end namespace mkfit
#endif
