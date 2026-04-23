// filepath: /foo/matevz/mic-dev/current/src/RecoTracker/MkFitCore/standalone/RntDumper/RntSources.h
#ifndef RecoTracker_MkFitCore_standalone_RdfTrace_RdfSources_h
#define RecoTracker_MkFitCore_standalone_RdfTrace_RdfSources_h

#include "RecoTracker/MkFitCore/standalone/DataFormats/RntStructs.h"
#include "RecoTracker/MkFitCore/standalone/Event.h"
#include "ROOT/RDataFrame.hxx"

namespace mkfit {
  
  // RdfSources, static factory for creating RDataFrames from the Event,
  // with different sources (HitMatch, SeedData, etc.)
  
  class RdfSources {
    public:
    static ROOT::RDataFrame MakeTrCandMetaDF(const Event &ev);
    static ROOT::RDataFrame MakeTrCandStateDF(const Event &ev);
    static ROOT::RDataFrame MakeTrHitMatchDF(const Event &ev);

    static ROOT::RDataFrame MakeTrackDF(const TrackVec &tvec);
    static ROOT::RDataFrame MakeSeedDF(const Event &ev);

    static ROOT::RDataFrame MakeEventDF(std::vector<const Event*>& events);
  };

  // RdfCtx -- to access event and potentially other things in lambda column readers, without making them capture-heavy.
  // Use as:
  // auto C = mkfit::MakeCtx(s.event());
  // r.Define("kalmanIdx", [C](int id) { return C.ev->kalmanByState_[id]; }, {"id"});
  //
  // Also, can do this, in C++ and TRint, and use EV-> instead of C.ev-> in the lambda bodies:
  // #define EV C.ev

  struct RdfCtx {
    const Event& ev;
    const TrackerInfo& trk_info;
  };

  inline RdfCtx MakeCtx(const Event* ev, const TrackerInfo& ti) { return {*ev, ti}; }

}  // end namespace mkfit

#endif
