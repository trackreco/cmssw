#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeedCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

// MkFit includes
#include "ConfigWrapper.h"
#include "Event.h"
#include "mkFit/buildtestMPlex.h"
#include "mkFit/MkBuilderWrapper.h"

// TBB includes
#include "tbb/task_arena.h"

// std includes
#include <functional>
#include <mutex>

class MkFitProducer: public edm::global::EDProducer<edm::StreamCache<mkfit::MkBuilderWrapper> > {
public:
  explicit MkFitProducer(edm::ParameterSet const& iConfig);
  ~MkFitProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<mkfit::MkBuilderWrapper> beginStream(edm::StreamID) const;
private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  std::vector<mkfit::HitVec> convertHits(const SiPixelRecHitCollection& pixelHits,
                                         const SiStripRecHit2DCollection& stripRphiHits,
                                         const SiStripRecHit2DCollection& stripStereoHits) const;

  mkfit::TrackVec convertSeeds(const TrajectorySeedCollection& seeds) const;

  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::EDGetTokenT<TrajectorySeedCollection> seedToken_;
  std::function<double(mkfit::Event&, mkfit::MkBuilder&)> buildFunction_;
};

MkFitProducer::MkFitProducer(edm::ParameterSet const& iConfig):
  pixelRecHitToken_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHits"))),
  stripRphiRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripRphiRecHits"))),
  stripStereoRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripStereoRecHits"))),
  seedToken_(consumes<TrajectorySeedCollection>(iConfig.getParameter<edm::InputTag>("seeds")))
{
  const auto build = iConfig.getParameter<std::string>("buildingRoutine");
  bool isFV = false;
  if(build == "bestHit") {
    buildFunction_ = mkfit::runBuildingTestPlexBestHit;
  }
  else if(build == "standard") {
    buildFunction_ = mkfit::runBuildingTestPlexStandard;
  }
  else if(build == "cloneEngine") {
    buildFunction_ = mkfit::runBuildingTestPlexCloneEngine;
  }
  else if(build == "fullVector") {
    isFV = true;
    buildFunction_ = mkfit::runBuildingTestPlexFV;
  }
  else {
    throw cms::Exception("Configuration") << "Invalid value for parameter 'buildingRoutine' " << build << ", allowed are bestHit, standard, cloneEngine, fullVector";
  }
  // TODO: what to do when we have multiple instances of MkFitProducer in a job?
  mkfit::MkBuilderWrapper::populate(isFV);
  mkfit::ConfigWrapper::setInputToCMSSW();

  produces<TrackCandidateCollection>();
  produces<std::vector<SeedStopInfo> >();
}

void MkFitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.add("stripRphiRecHits", edm::InputTag("siStripMatchedRecHits", "rphiRecHit"));
  desc.add("stripStereoRecHits", edm::InputTag("siStripMatchedRecHits", "stereoRecHit"));
  desc.add("seeds", edm::InputTag("initialStepSeeds"));
  desc.add("buildingRoutine", std::string("what should be the default?"));

  descriptions.add("mkFitProducer", desc);
}

std::unique_ptr<mkfit::MkBuilderWrapper> MkFitProducer::beginStream(edm::StreamID iID) const {
  return std::make_unique<mkfit::MkBuilderWrapper>();
}

namespace {
  std::once_flag geometryFlag;
}
void MkFitProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  // First do some initialization
  // TODO: the mechanism needs to be improved...
  std::call_once(geometryFlag, [&iSetup]() {
      edm::ESHandle<TrackerGeometry> trackerGeometry;
      iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
      const auto& geom = *trackerGeometry;
      const auto barrelLayers = geom.numberOfLayers(PixelSubdetector::PixelBarrel) + geom.numberOfLayers(StripSubdetector::TIB) + geom.numberOfLayers(StripSubdetector::TOB);
      const auto endcapLayers = geom.numberOfLayers(PixelSubdetector::PixelEndcap) + geom.numberOfLayers(StripSubdetector::TID) + geom.numberOfLayers(StripSubdetector::TEC);
      // TODO: Number of stereo layers is hardcoded for now, so this won't work for phase2 tracker
      const auto barrelStereo = 2 + 2;
      const auto endcapStereo = geom.numberOfLayers(StripSubdetector::TID) + geom.numberOfLayers(StripSubdetector::TEC);
      const auto nlayers = barrelLayers + barrelStereo + 2*(endcapLayers + endcapStereo);
      LogDebug("MkFitProducer") << "Total number of tracker layers (stereo counted separately) " << nlayers;
      mkfit::ConfigWrapper::setNTotalLayers(nlayers);
    });

  const auto& pixelHits = edm::get(iEvent, pixelRecHitToken_);
  const auto& stripRphiHits = edm::get(iEvent, stripRphiRecHitToken_);
  const auto& stripStereoHits = edm::get(iEvent, stripStereoRecHitToken_);
  const auto& seeds = edm::get(iEvent, seedToken_);

  auto mkfitHits = convertHits(pixelHits, stripRphiHits, stripStereoHits);
  auto mkfitSeeds = convertSeeds(seeds);

  // CMSSW event ID (64-bit unsigned) does not fit in int
  // In addition, unique ID requires also lumi and run
  // But does the event ID really matter within mkfit?
  mkfit::Event ev(iEvent.id().event());

  ev.setInputFromCMSSW(std::move(mkfitHits), std::move(mkfitSeeds));

  tbb::this_task_arena::isolate([&](){
      buildFunction_(ev, streamCache(iID)->get());
    });

  // For starters let's put empty collections
  iEvent.put(std::make_unique<TrackCandidateCollection>());
  iEvent.put(std::make_unique<std::vector<SeedStopInfo> >(seeds.size()));
}

std::vector<mkfit::HitVec> MkFitProducer::convertHits(const SiPixelRecHitCollection& pixelHits,
                                                      const SiStripRecHit2DCollection& stripRphiHits,
                                                      const SiStripRecHit2DCollection& stripStereoHits) const {
  std::vector<mkfit::HitVec> ret;
  return ret;
}

mkfit::TrackVec MkFitProducer::convertSeeds(const TrajectorySeedCollection& seeds) const {
  mkfit::TrackVec ret;
  return ret;
}

DEFINE_FWK_MODULE(MkFitProducer);
