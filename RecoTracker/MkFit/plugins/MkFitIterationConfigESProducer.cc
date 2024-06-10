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
  const float thetasymmin_;
  const float thetasymmin_l_;
  const float thetasymmin_h_;
  const float invptmin_;
  const float d0_max_;
  const float d0_maxf_;
  const int f1_;
  const int f2_;
  const int f3_;
  const int h3_;
  const int h4_;
  const int h5_;
  const int ly3_;
  const int ly4_;
  const int ly5_;
  const float thetasymminB_;
  const float thetasymmin_lB_;
  const float thetasymmin_hB_;
  const float invptminB_;
  const float d0_maxB_;
  const float d0_maxfB_;
  const int f1B_;
  const int f2B_;
  const int f3B_;
  const int h3B_;
  const int h4B_;
  const int h5B_;
  const int ly3B_;
  const int ly4B_;
  const int ly5B_;
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
      tailMissingHitPenaltyBkw_{(float)iConfig.getParameter<double>("tailMissingHitPenaltyBkw")},
      thetasymmin_{(float)iConfig.getParameter<double>("thetasymmin")},
      thetasymmin_l_{(float)iConfig.getParameter<double>("thetasymmin_l")},
      thetasymmin_h_{(float)iConfig.getParameter<double>("thetasymmin_h")},
      invptmin_{(float)iConfig.getParameter<double>("invptmin")},
      d0_max_{(float)iConfig.getParameter<double>("d0_max")},
      d0_maxf_{(float)iConfig.getParameter<double>("d0_maxf")},
      f1_{(int)iConfig.getParameter<double>("f1")},
      f2_{(int)iConfig.getParameter<double>("f2")},
      f3_{(int)iConfig.getParameter<double>("f3")},
      h3_{(int)iConfig.getParameter<double>("h3")},
      h4_{(int)iConfig.getParameter<double>("h4")},
      h5_{(int)iConfig.getParameter<double>("h5")},
      ly3_{(int)iConfig.getParameter<double>("ly3")},
      ly4_{(int)iConfig.getParameter<double>("ly4")},
      ly5_{(int)iConfig.getParameter<double>("ly5")},
      thetasymminB_{(float)iConfig.getParameter<double>("thetasymminB")},
      thetasymmin_lB_{(float)iConfig.getParameter<double>("thetasymmin_lB")},
      thetasymmin_hB_{(float)iConfig.getParameter<double>("thetasymmin_hB")},
      invptminB_{(float)iConfig.getParameter<double>("invptminB")},
      d0_maxB_{(float)iConfig.getParameter<double>("d0_maxB")},
      d0_maxfB_{(float)iConfig.getParameter<double>("d0_maxfB")},
      f1B_{(int)iConfig.getParameter<double>("f1B")},
      f2B_{(int)iConfig.getParameter<double>("f2B")},
      f3B_{(int)iConfig.getParameter<double>("f3B")},
      h3B_{(int)iConfig.getParameter<double>("h3B")},
      h4B_{(int)iConfig.getParameter<double>("h4B")},
      h5B_{(int)iConfig.getParameter<double>("h5B")},
      ly3B_{(int)iConfig.getParameter<double>("ly3B")},
      ly4B_{(int)iConfig.getParameter<double>("ly4B")},
      ly5B_{(int)iConfig.getParameter<double>("ly5B")} {}

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

  desc.add<double>("thetasymminB", 1.11)->setComment("ddd");
  desc.add<double>("thetasymmin_lB", 0.80)->setComment("ddd");
  desc.add<double>("thetasymmin_hB", 1.11)->setComment("ddd");
  desc.add<double>("invptminB", 1.11)->setComment("ddd");
  desc.add<double>("d0_maxB", 0.1)->setComment("ddd");
  desc.add<double>("d0_maxfB", 0.05)->setComment("ddd");
  desc.add<double>("f1B", 4)->setComment("ddd");
  desc.add<double>("f2B", 3)->setComment("ddd");
  desc.add<double>("f3B", 4)->setComment("ddd");
  desc.add<double>("h3B", 3)->setComment("ddd");
  desc.add<double>("h4B", 4)->setComment("ddd");
  desc.add<double>("h5B", 5)->setComment("ddd");
  desc.add<double>("ly3B", 3)->setComment("ddd");
  desc.add<double>("ly4B", 4)->setComment("ddd");
  desc.add<double>("ly5B", 5)->setComment("ddd");
  desc.add<double>("thetasymmin", 1.11)->setComment("ddd");
  desc.add<double>("thetasymmin_l", 0.80)->setComment("ddd");
  desc.add<double>("thetasymmin_h", 1.11)->setComment("ddd");
  desc.add<double>("invptmin", 1.11)->setComment("ddd");
  desc.add<double>("d0_max", 0.1)->setComment("ddd");
  desc.add<double>("d0_maxf", 0.05)->setComment("ddd");
  desc.add<double>("f1", 4)->setComment("ddd");
  desc.add<double>("f2", 3)->setComment("ddd");
  desc.add<double>("f3", 4)->setComment("ddd");
  desc.add<double>("h3", 3)->setComment("ddd");
  desc.add<double>("h4", 4)->setComment("ddd");
  desc.add<double>("h5", 5)->setComment("ddd");
  desc.add<double>("ly3", 3)->setComment("ddd");
  desc.add<double>("ly4", 4)->setComment("ddd");
  desc.add<double>("ly5", 5)->setComment("ddd");

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

  it_conf->m_params.thetasymmin = thetasymmin_;
  it_conf->m_backward_params.thetasymmin = thetasymminB_;
  it_conf->m_params.thetasymmin_l = thetasymmin_l_;
  it_conf->m_backward_params.thetasymmin_l = thetasymmin_lB_;
  it_conf->m_params.thetasymmin_h = thetasymmin_h_;
  it_conf->m_backward_params.thetasymmin_h = thetasymmin_hB_;
  it_conf->m_params.invptmin = invptmin_;
  it_conf->m_backward_params.invptmin = invptminB_;
  it_conf->m_params.d0_max = d0_max_;
  it_conf->m_backward_params.d0_max = d0_maxB_;
  it_conf->m_params.d0_maxf = d0_maxf_;
  it_conf->m_backward_params.d0_maxf = d0_maxfB_;
  it_conf->m_params.f1 = f1_;
  it_conf->m_backward_params.f2 = f1B_;
  it_conf->m_params.f2 = f2_;
  it_conf->m_backward_params.f2 = f2B_;
  it_conf->m_params.f3 = f3_;
  it_conf->m_backward_params.f3 = f1B_;
  it_conf->m_params.h3 = h3_;
  it_conf->m_backward_params.h3 = h3B_;
  it_conf->m_params.h4 = h4_;
  it_conf->m_backward_params.h4 = h4B_;
  it_conf->m_params.h5 = h5_;
  it_conf->m_backward_params.h5 = h5B_;
  it_conf->m_params.ly3 = ly3_;
  it_conf->m_backward_params.ly3 = ly3B_;
  it_conf->m_params.ly4 = ly4_;
  it_conf->m_backward_params.ly4 = ly4B_;
  it_conf->m_params.ly5 = ly5_;
  it_conf->m_backward_params.ly5 = ly5B_;

  it_conf->setupStandardFunctionsFromNames();
  return it_conf;
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitIterationConfigESProducer);
