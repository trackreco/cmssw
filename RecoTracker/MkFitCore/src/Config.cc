#include "RecoTracker/MkFitCore/interface/Config.h"

namespace mkfit {

  namespace Config {

    bool usePropToPlane = false;
    bool usePtMultScat = false;

    bool v2p2UseWsr = true;
    bool v2p2UseHoleLimits = true;
    bool v2p2UseStopCuts = true;
    // ON by default since 2026-09-23: the in-layer combinatorial search is the
    // production configuration, paired with maxCandsPerSeed = 3 in the CMS-phase2
    // geometry plugin. Measured on 30 events of ttbar-PU200-D121-C22 against the
    // best-hit path: +391 found sim tracks (63.33 % against 62.01 % of the MTV
    // denominator) and +1.9 to +3.0 truth-matched hits per found track, largest
    // in the endcap. Turn it off with --v2p2-in-layer-comb 0.
    bool v2p2InLayerComb = true;
    bool v2p2ReserveHoleSlot = false;
    bool v2p2BestShort = false;

    // Multi threading configuration
#if defined(MKFIT_STANDALONE)
    int numThreadsFinder = 1;
    int numThreadsEvents = 1;
    int numSeedsPerTask = 32;
#endif

#if defined(MKFIT_STANDALONE)
    bool removeDuplicates = false;
    bool useHitsForDuplicates = true;
#endif
    const float maxdPt = 0.5;
    const float maxdPhi = 0.25;
    const float maxdEta = 0.05;
    const float maxdR = 0.0025;
    const float minFracHitsShared = 0.75;

    const float maxd1pt = 1.8;     //windows for hit
    const float maxdphi = 0.37;    //and/or dr
    const float maxdcth = 0.37;    //comparisons
    const float maxcth_ob = 1.99;  //eta 1.44
    const float maxcth_fw = 6.05;  //eta 2.5

#ifdef CONFIG_PhiQArrays
    bool usePhiQArrays = true;
#endif
  }  // namespace Config

}  // end namespace mkfit
