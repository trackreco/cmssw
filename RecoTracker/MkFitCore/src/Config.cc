#include "RecoTracker/MkFitCore/interface/Config.h"

#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

namespace mkfit {

  namespace Config {

    int nTotalLayers = -1;

    std::string geomPlugin = "CylCowWLids";

    // Multi threading and Clone engine configuration
    int numThreadsFinder = 1;
    int numThreadsEvents = 1;
    int numSeedsPerTask = 32;

    seedOpts seedInput = simSeeds;

    bool removeDuplicates = false;
    bool useHitsForDuplicates = true;
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

    // material effects
    float RlgridME[Config::nBinsZME][Config::nBinsRME];
    float XigridME[Config::nBinsZME][Config::nBinsRME];

    bool finding_requires_propagation_to_hit_pos;
    PropagationFlags finding_inter_layer_pflags;
    PropagationFlags finding_intra_layer_pflags;
    PropagationFlags backward_fit_pflags;
    PropagationFlags forward_fit_pflags;
    PropagationFlags seed_fit_pflags;
    PropagationFlags pca_prop_pflags;

#ifdef CONFIG_PhiQArrays
    bool usePhiQArrays = true;
#endif

    bool useCMSGeom = false;
    bool includePCA = false;

    bool silent = false;
    bool json_verbose = false;

  }  // namespace Config

}  // end namespace mkfit
