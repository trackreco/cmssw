#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

// mkFit includes
#include "RecoTracker/MkFitCore/interface/IterationConfig.h"

class MkFitIterationConfigESProducer : public edm::ESProducer {
public:
  MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<mkfit::IterationConfig> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> geomToken_;
  const std::string configFile_;
  const float minPtCut_;
  const unsigned int maxClusterSize_;
  const float validHitBonus_;
  const float validHitSlope_;
  const float overlapHitBonus_;  // set to negative for penalty
  const float missingHitPenalty_;
  const float tailMissingHitPenalty_;
  const float validHitBonusBkw_;
  const float validHitSlopeBkw_;
  const float overlapHitBonusBkw_;  // set to negative for penalty
  const float missingHitPenaltyBkw_;
  const float tailMissingHitPenaltyBkw_;
};

MkFitIterationConfigESProducer::MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig)
    : geomToken_{setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName")).consumes()},
      configFile_{iConfig.getParameter<edm::FileInPath>("config").fullPath()},
      minPtCut_{(float)iConfig.getParameter<double>("minPt")},
      maxClusterSize_{iConfig.getParameter<unsigned int>("maxClusterSize")},
      validHitBonus_{(float)iConfig.getParameter<double>("validHitBonus")},
      validHitSlope_{(float)iConfig.getParameter<double>("validHitSlope")},
      overlapHitBonus_{(float)iConfig.getParameter<double>("overlapHitBonus")},
      missingHitPenalty_{(float)iConfig.getParameter<double>("missingHitPenalty")},
      tailMissingHitPenalty_{(float)iConfig.getParameter<double>("tailMissingHitPenalty")},
      validHitBonusBkw_{(float)iConfig.getParameter<double>("validHitBonusBkw")},
      validHitSlopeBkw_{(float)iConfig.getParameter<double>("validHitSlopeBkw")},
      overlapHitBonusBkw_{(float)iConfig.getParameter<double>("overlapHitBonusBkw")},
      missingHitPenaltyBkw_{(float)iConfig.getParameter<double>("missingHitPenaltyBkw")},
      tailMissingHitPenaltyBkw_{(float)iConfig.getParameter<double>("tailMissingHitPenaltyBkw")} {}

void MkFitIterationConfigESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName", "")->setComment("Product label");
  desc.add<edm::FileInPath>("config", edm::FileInPath())
      ->setComment("Path to the JSON file for the mkFit configuration parameters");
  desc.add<double>("minPt", 0.0)->setComment("min pT cut applied during track building");
  desc.add<unsigned int>("maxClusterSize", 8)->setComment("Max cluster size of SiStrip hits");
  desc.add<double>("validHitBonus", 4)->setComment("xx");
  desc.add<double>("validHitSlope", 0.2)->setComment("yy");
  desc.add<double>("overlapHitBonus", 0)->setComment("zz");
  desc.add<double>("missingHitPenalty", 8)->setComment("zzz");
  desc.add<double>("tailMissingHitPenalty", 3)->setComment("tttt");
  desc.add<double>("validHitBonusBkw", 4)->setComment("xx");
  desc.add<double>("validHitSlopeBkw", 0.2)->setComment("yy");
  desc.add<double>("overlapHitBonusBkw", 0)->setComment("zz");
  desc.add<double>("missingHitPenaltyBkw", 8)->setComment("zzz");
  desc.add<double>("tailMissingHitPenaltyBkw", 3)->setComment("tttt");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<mkfit::IterationConfig> MkFitIterationConfigESProducer::produce(
    const TrackerRecoGeometryRecord &iRecord) {
  mkfit::ConfigJson cj;
  auto it_conf = cj.load_File(configFile_);
  it_conf->m_params.minPtCut = minPtCut_;
  it_conf->m_backward_params.minPtCut = minPtCut_;
  it_conf->m_params.maxClusterSize = maxClusterSize_;
  it_conf->m_backward_params.maxClusterSize = maxClusterSize_;
  it_conf->m_params.validHitBonus = validHitBonus_;
  it_conf->m_backward_params.validHitBonus = validHitBonusBkw_;
  it_conf->m_params.validHitSlope = validHitSlope_;
  it_conf->m_backward_params.validHitSlope = validHitSlopeBkw_;
  it_conf->m_params.overlapHitBonus = overlapHitBonus_;
  it_conf->m_backward_params.overlapHitBonus = overlapHitBonusBkw_;
  it_conf->m_params.missingHitPenalty = missingHitPenalty_;
  it_conf->m_backward_params.missingHitPenalty = missingHitPenaltyBkw_;
  it_conf->m_params.tailMissingHitPenalty = tailMissingHitPenalty_;
  it_conf->m_backward_params.tailMissingHitPenalty = tailMissingHitPenaltyBkw_;
  it_conf->setupStandardFunctionsFromNames();
  return it_conf;
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitIterationConfigESProducer);
