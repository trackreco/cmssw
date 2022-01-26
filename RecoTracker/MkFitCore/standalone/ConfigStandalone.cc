#include "RecoTracker/MkFitCore/standalone/ConfigStandalone.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

namespace mkfit {

  namespace Config {

    TrackerInfo TrkInfo;
    IterationsInfo ItrInfo;

    std::string geomPlugin = "CylCowWLids";

    int nTracks = 10000;
    int nEvents = 20;
    int nItersCMSSW = 0;
    bool loopOverFile = false;

    seedOpts seedInput = simSeeds;
    cleanOpts seedCleaning = noCleaning;

    bool readCmsswTracks = false;

    bool dumpForPlots = false;

    bool cf_seeding = false;
    bool cf_fitting = false;

    bool quality_val = false;
    bool sim_val_for_cmssw = false;
    bool sim_val = false;
    bool cmssw_val = false;
    bool fit_val = false;
    bool readSimTrackStates = false;
    bool inclusiveShorts = false;
    bool keepHitInfo = false;
    bool tryToSaveSimInfo = false;
    matchOpts cmsswMatchingFW = hitBased;
    matchOpts cmsswMatchingBK = trkParamBased;

    bool useDeadModules = false;

    // number of hits per task for finding seeds
    int numHitsPerTask = 32;

    bool mtvLikeValidation = false;
    bool mtvRequireSeeds = false;
    int cmsSelMinLayers = 12;
    int nMinFoundHits = 10;

    bool kludgeCmsHitErrors = false;
    bool backwardFit = false;
    bool backwardSearch = true;

    int numThreadsSimulation = 12;

    int finderReportBestOutOfN = 1;

    // ================================================================

    bool json_dump_before = false;
    bool json_dump_after = false;
    std::vector<std::string> json_patch_filenames;
    std::vector<std::string> json_load_filenames;
    std::string json_save_iters_fname_fmt;
    bool json_save_iters_include_iter_info_preamble = false;

    // ================================================================

    void recalculateDependentConstants() {}

  }  // namespace Config

}  // namespace mkfit
