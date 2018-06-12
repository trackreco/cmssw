#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/KalmanUpdators/interface/KFUpdator.h"
#include "TrackingTools/KalmanUpdators/interface/Chi2MeasurementEstimator.h"
#include "TrackingTools/TrackFitters/interface/KFTrajectoryFitter.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkClonerImpl.h"
#include "RecoTracker/TransientTrackingRecHit/interface/TkTransientTrackingRecHitBuilder.h"
#include "TrackingTools/MaterialEffects/src/PropagatorWithMaterial.cc"

#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

// MkFit includes
#include "ConfigWrapper.h"
#include "Event.h"
#include "LayerNumberConverter.h"
#include "mkFit/buildtestMPlex.h"
#include "mkFit/MkBuilderWrapper.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// TBB includes
#include "tbb/task_arena.h"

// std includes
#include <functional>
#include <mutex>

namespace {
  class IndexLayer {
  public:
    struct HitInfo {
      HitInfo(): index(-1), layer(-1) {}
      HitInfo(int i, int l): index(i), layer(l) {}
      int index;
      int layer;
    };

    struct Coll {
      explicit Coll(edm::ProductID id): productID(id) {}
      edm::ProductID productID;
      std::vector<HitInfo> infos; // indexed by cluster index
    };

    IndexLayer() {}
    ~IndexLayer() = default;

    void insert(edm::ProductID id, size_t clusterIndex, int hit, int layer, const TrackingRecHit *hitPtr) {
      // mapping CMSSW->mkfit
      auto found = std::find_if(colls_.begin(), colls_.end(), [&](const auto& item) {
          return item.productID == id;
        });
      if(found == colls_.end()) {
        found = colls_.emplace(colls_.end(), id);
      }
      if(found->infos.size() <= clusterIndex) {
        found->infos.resize(clusterIndex+1);
      }
      found->infos[clusterIndex] = HitInfo(hit, layer);

      // mapping mkfit->CMSSW
      if(layer >= static_cast<int>(hits_.size())) {
        hits_.resize(layer+1);
      }
      if(hit >= static_cast<int>(hits_[layer].size())) {
        hits_[layer].resize(hit+1);
      }
      //edm::LogPrint("Foo") << "Putting hitPtr on layer " << layer << " hit " << hit;
      hits_[layer][hit].ptr = hitPtr;
      hits_[layer][hit].clusterIndex = clusterIndex;
    }

    const HitInfo& get(edm::ProductID id, size_t clusterIndex) const {
      auto found = std::find_if(colls_.begin(), colls_.end(), [&](const auto& item) {
          return item.productID == id;
        });
      if(found == colls_.end()) {
        auto exp = cms::Exception("Assert");
        exp << "Encountered a seed with a hit having productID " << id << " which is not any of the input hit collections: ";
        for(const auto& elem: colls_) {
          exp << elem.productID << " ";
        }
        throw exp;
      }
      const HitInfo& ret = found->infos[clusterIndex];
      if(ret.index < 0) {
        throw cms::Exception("Assert") << "No hit index for cluster " << clusterIndex << " of collection " << id;
      }
      return ret;
    }

    const TrackingRecHit *getHitPtr(int layer, int hit) const {
      //edm::LogPrint("Foo") << "Getting hitPtr from layer " << layer << " hit " << hit;
      return hits_.at(layer).at(hit).ptr;
    }

    size_t getClusterIndex(int layer, int hit) const {
      return hits_.at(layer).at(hit).clusterIndex;
    }

  private:

    struct CMSSWHit {
      const TrackingRecHit *ptr = nullptr;
      size_t clusterIndex = 0;
    };

    std::vector<Coll> colls_; // mapping from CMSSW(ProductID, index) -> mkfit(index, layer)
    std::vector<std::vector<CMSSWHit> > hits_; // reverse mapping, mkfit(layer, index) -> CMSSW hit
  };
}

class MkFitProducer: public edm::global::EDProducer<edm::StreamCache<mkfit::MkBuilderWrapper> > {
public:
  explicit MkFitProducer(edm::ParameterSet const& iConfig);
  ~MkFitProducer() = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<mkfit::MkBuilderWrapper> beginStream(edm::StreamID) const;
private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  std::vector<const DetLayer *> createDetLayers(const mkfit::LayerNumberConverter& lnc,
                                                const GeometricSearchTracker& tracker,
                                                const TrackerTopology& ttopo) const;

  template <typename HitCollection>
  void convertHits(const HitCollection& hits,
                   std::vector<mkfit::HitVec>& mkfitHits,
                   IndexLayer& indexLayers,
                   int& totalHits,
                   const TrackerTopology& ttopo,
                   const TransientTrackingRecHitBuilder& ttrhBuilder,
                   const mkfit::LayerNumberConverter& lnc) const;

  mkfit::TrackVec convertSeeds(const edm::View<TrajectorySeed>& seeds,
                               const IndexLayer& indexLayers,
                               const TransientTrackingRecHitBuilder& ttrhBuilder,
                               const MagneticField& mf) const;

  std::unique_ptr<TrackCandidateCollection> convertCandidates(const mkfit::Event& ev,
                                                              const IndexLayer& indexLayers,
                                                              const edm::View<TrajectorySeed>& seeds,
                                                              const MagneticField& mf,
                                                              const Propagator& propagatorAlong,
                                                              const Propagator& propagatorOpposite,
                                                              const TkClonerImpl& hitCloner,
                                                              const std::vector<const DetLayer *>& detLayers,
                                                              const mkfit::TrackVec& mkfitSeeds) const;
  std::pair<TrajectoryStateOnSurface, const GeomDet *> convertToInitialState(const FreeTrajectoryState& fts,
                                                                             const edm::OwnVector<TrackingRecHit>& hits,
                                                                             const Propagator& propagatorAlong,
                                                                             const Propagator& propagatorOpposite,
                                                                             const TkClonerImpl& hitCloner) const;

  using SVector3 = ROOT::Math::SVector<float, 3>;
  using SMatrixSym33 = ROOT::Math::SMatrix<float,3,3,ROOT::Math::MatRepSym<float,3> >;
  using SMatrixSym66 = ROOT::Math::SMatrix<float,6,6,ROOT::Math::MatRepSym<float,6> >;

  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken_;
  edm::EDGetTokenT<MeasurementTrackerEvent> mteToken_;
  std::string ttrhBuilderName_;
  std::string propagatorAlongName_;
  std::string propagatorOppositeName_;
  std::function<double(mkfit::Event&, mkfit::MkBuilder&)> buildFunction_;
};

MkFitProducer::MkFitProducer(edm::ParameterSet const& iConfig):
  pixelRecHitToken_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHits"))),
  stripRphiRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripRphiRecHits"))),
  stripStereoRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripStereoRecHits"))),
  seedToken_(consumes<edm::View<TrajectorySeed> >(iConfig.getParameter<edm::InputTag>("seeds"))),
  mteToken_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("measurementTrackerEvent"))),
  ttrhBuilderName_(iConfig.getParameter<std::string>("ttrhBuilder")),
  propagatorAlongName_(iConfig.getParameter<std::string>("propagatorAlong")),
  propagatorOppositeName_(iConfig.getParameter<std::string>("propagatorOpposite"))
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

  const auto seedClean = iConfig.getParameter<std::string>("seedCleaning");
  auto seedCleanOpt = mkfit::ConfigWrapper::SeedCleaningOpts::noCleaning;
  if(seedClean == "none") {
    seedCleanOpt = mkfit::ConfigWrapper::SeedCleaningOpts::noCleaning;
  }
  else if(seedClean == "N2") {
    seedCleanOpt = mkfit::ConfigWrapper::SeedCleaningOpts::cleanSeedsN2;
  }
  else {
    throw cms::Exception("Configuration") << "Invalida value for parameter 'seedCleaning' " << seedClean << ", allowed are none, N2";
  }

  // TODO: what to do when we have multiple instances of MkFitProducer in a job?
  mkfit::MkBuilderWrapper::populate(isFV);
  mkfit::ConfigWrapper::initializeForCMSSW(seedCleanOpt, mkfit::ConfigWrapper::BackwardFit::noFit);

  produces<TrackCandidateCollection>();
  produces<std::vector<SeedStopInfo> >();
}

void MkFitProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.add("stripRphiRecHits", edm::InputTag("siStripMatchedRecHits", "rphiRecHit"));
  desc.add("stripStereoRecHits", edm::InputTag("siStripMatchedRecHits", "stereoRecHit"));
  desc.add("seeds", edm::InputTag("initialStepSeeds"));
  desc.add("measurementTrackerEvent", edm::InputTag("MeasurementTrackerEvent"));
  desc.add<std::string>("ttrhBuilder", "WithTrackAngle");
  desc.add<std::string>("propagatorAlong", "PropagatorWithMaterial");
  desc.add<std::string>("propagatorOpposite", "PropagatorWithMaterialOpposite");
  desc.add("buildingRoutine", std::string("what should be the default?"));
  desc.add<std::string>("seedCleaning", "none")->setComment("Valid values are: 'none', 'N2'");

  descriptions.add("mkFitProducer", desc);
}

std::unique_ptr<mkfit::MkBuilderWrapper> MkFitProducer::beginStream(edm::StreamID iID) const {
  return std::make_unique<mkfit::MkBuilderWrapper>();
}

namespace {
  std::once_flag geometryFlag;
}
void MkFitProducer::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  mkfit::LayerNumberConverter lnc{mkfit::TkLayout::phase1};

  // First do some initialization
  // TODO: the mechanism needs to be improved...
  std::call_once(geometryFlag, [&iSetup, &lnc]() {
      edm::ESHandle<TrackerGeometry> trackerGeometry;
      iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
      const auto& geom = *trackerGeometry;
      // TODO: eventually automatize fully
      // For now it is easier to use purely the infrastructure from mkfit
      /*
      const auto barrelLayers = geom.numberOfLayers(PixelSubdetector::PixelBarrel) + geom.numberOfLayers(StripSubdetector::TIB) + geom.numberOfLayers(StripSubdetector::TOB);
      const auto endcapLayers = geom.numberOfLayers(PixelSubdetector::PixelEndcap) + geom.numberOfLayers(StripSubdetector::TID) + geom.numberOfLayers(StripSubdetector::TEC);
      // TODO: Number of stereo layers is hardcoded for now, so this won't work for phase2 tracker
      const auto barrelStereo = 2 + 2;
      const auto endcapStereo = geom.numberOfLayers(StripSubdetector::TID) + geom.numberOfLayers(StripSubdetector::TEC);
      const auto nlayers = barrelLayers + barrelStereo + 2*(endcapLayers + endcapStereo);
      LogDebug("MkFitProducer") << "Total number of tracker layers (stereo counted separately) " << nlayers;
      */
      if(geom.numberOfLayers(PixelSubdetector::PixelBarrel) != 4 || geom.numberOfLayers(PixelSubdetector::PixelEndcap) != 3) {
        throw cms::Exception("Assert") << "For now this code works only with phase1 tracker, you have something else";
      }
      mkfit::ConfigWrapper::setNTotalLayers(lnc.nLayers());
    });

  // Then import hits
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  iEvent.getByToken(pixelRecHitToken_, pixelHits);

  edm::Handle<SiStripRecHit2DCollection> stripRphiHits;
  iEvent.getByToken(stripRphiRecHitToken_, stripRphiHits);

  edm::Handle<SiStripRecHit2DCollection> stripStereoHits;
  iEvent.getByToken(stripStereoRecHitToken_, stripStereoHits);

  edm::Handle<MeasurementTrackerEvent> mte;
  iEvent.getByToken(mteToken_, mte);

  edm::ESHandle<TrackerTopology> ttopo;
  iSetup.get<TrackerTopologyRcd>().get(ttopo);

  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhBuilder;
  iSetup.get<TransientRecHitRecord>().get(ttrhBuilderName_, ttrhBuilder);
  const auto *tkBuilder = dynamic_cast<TkTransientTrackingRecHitBuilder const *>(ttrhBuilder.product());
  if(!tkBuilder) {
    throw cms::Exception("LogicError") << "TTRHBuilder must be of type TkTransientTrackingRecHitBuilder";
  }
  TkClonerImpl hc = tkBuilder->cloner();

  edm::ESHandle<Propagator> propagatorAlong;
  iSetup.get<TrackingComponentsRecord>().get(propagatorAlongName_, propagatorAlong);
  edm::ESHandle<Propagator> propagatorOpposite;
  iSetup.get<TrackingComponentsRecord>().get(propagatorOppositeName_, propagatorOpposite);

  auto detLayers = createDetLayers(lnc, *(mte->geometricSearchTracker()), *ttopo);

  std::vector<mkfit::HitVec> mkfitHits(lnc.nLayers());
  IndexLayer indexLayers;
  int totalHits = 0; // I need to have a global hit index in order to have the hit remapping working?
  convertHits(*pixelHits, mkfitHits, indexLayers, totalHits, *ttopo, *ttrhBuilder, lnc);
  convertHits(*stripRphiHits, mkfitHits, indexLayers, totalHits, *ttopo, *ttrhBuilder, lnc);
  convertHits(*stripStereoHits, mkfitHits, indexLayers, totalHits, *ttopo, *ttrhBuilder, lnc);


  // Then import seeds
  edm::Handle<edm::View<TrajectorySeed>> seeds;
  iEvent.getByToken(seedToken_, seeds);

  edm::ESHandle<MagneticField> mf;
  iSetup.get<IdealMagneticFieldRecord>().get(mf);

  auto mkfitSeeds = convertSeeds(*seeds, indexLayers, *ttrhBuilder, *mf);

  // CMSSW event ID (64-bit unsigned) does not fit in int
  // In addition, unique ID requires also lumi and run
  // But does the event ID really matter within mkfit?
  mkfit::Event ev(iEvent.id().event());

  auto tmpSeeds = mkfitSeeds; // temporary copy to ensure that we got the candidate->seed mapping correctly, to be removed in the future
  ev.setInputFromCMSSW(std::move(mkfitHits), std::move(mkfitSeeds));

  tbb::this_task_arena::isolate([&](){
      buildFunction_(ev, streamCache(iID)->get());
    });

  // Convert mkfit presentation back to CMSSW
  auto cands = convertCandidates(ev, indexLayers, *seeds, *mf, *propagatorAlong, *propagatorOpposite, hc, detLayers, tmpSeeds);

  // For starters let's put empty collections
  iEvent.put(std::move(cands));
  iEvent.put(std::make_unique<std::vector<SeedStopInfo> >(seeds->size()));
}

std::vector<const DetLayer *> MkFitProducer::createDetLayers(const mkfit::LayerNumberConverter& lnc,
                                                             const GeometricSearchTracker& tracker,
                                                             const TrackerTopology& ttopo) const {
  std::vector<const DetLayer *> dets(lnc.nLayers(), nullptr);

  auto set = [&](unsigned int index, DetId id) {
    auto layer = tracker.idToLayer(id);
    if(layer == nullptr) {
      throw cms::Exception("LogicError") << "No layer for DetId " << id.rawId();
    }
    LogTrace("MkFitProducer") << "Setting DetLayer for index " << index
                              << " subdet " << id.subdetId()
                              << " layer " << ttopo.layer(id)
                              << " ptr " << layer;
    
    dets[index] = layer;
  };

  // TODO: currently hardcoded...
  // Logic copied from mkfit::LayerNumberConverter
  unsigned int off = 1;
  // BPix
  set(0, ttopo.pxbDetId(1, 0, 0));
  set(1, ttopo.pxbDetId(2, 0, 0));
  set(2, ttopo.pxbDetId(3, 0, 0));
  set(3, ttopo.pxbDetId(4, 0, 0));
  // TIB
  set(off+3, ttopo.tibDetId(1, 0, 0, 0, 0, 0));
  set(off+4, ttopo.tibDetId(1, 0, 0, 0, 0, 1));
  set(off+5, ttopo.tibDetId(2, 0, 0, 0, 0, 0));
  set(off+6, ttopo.tibDetId(2, 0, 0, 0, 0, 1));
  set(off+7, ttopo.tibDetId(3, 0, 0, 0, 0, 0));
  set(off+8, ttopo.tibDetId(4, 0, 0, 0, 0, 0));
  // TOB
  set(off+9, ttopo.tobDetId(1, 0, 0, 0, 0));
  set(off+10, ttopo.tobDetId(1, 0, 0, 0, 1));
  set(off+11, ttopo.tobDetId(2, 0, 0, 0, 0));
  set(off+12, ttopo.tobDetId(2, 0, 0, 0, 1));
  set(off+13, ttopo.tobDetId(3, 0, 0, 0, 0));
  set(off+14, ttopo.tobDetId(4, 0, 0, 0, 0));
  set(off+15, ttopo.tobDetId(5, 0, 0, 0, 0));
  set(off+16, ttopo.tobDetId(6, 0, 0, 0, 0));

  auto setForward = [&](unsigned int side) {
    // FPix
    set(off+0, ttopo.pxfDetId(side, 1, 0, 0, 0));
    set(off+1, ttopo.pxfDetId(side, 2, 0, 0, 0));
    set(off+2, ttopo.pxfDetId(side, 3, 0, 0, 0));
    // TID+
    off += 1;
    set(off+2, ttopo.tidDetId(side, 1, 0, 0, 0, 0));
    set(off+3, ttopo.tidDetId(side, 1, 0, 0, 0, 1));
    set(off+4, ttopo.tidDetId(side, 2, 0, 0, 0, 0));
    set(off+5, ttopo.tidDetId(side, 2, 0, 0, 0, 1));
    set(off+6, ttopo.tidDetId(side, 3, 0, 0, 0, 0));
    set(off+7, ttopo.tidDetId(side, 3, 0, 0, 0, 1));
    // TEC
    set(off+8, ttopo.tecDetId(side, 1, 0, 0, 0, 0, 0));
    set(off+9, ttopo.tecDetId(side, 1, 0, 0, 0, 0, 1));
    set(off+10, ttopo.tecDetId(side, 2, 0, 0, 0, 0, 0));
    set(off+11, ttopo.tecDetId(side, 2, 0, 0, 0, 0, 1));
    set(off+12, ttopo.tecDetId(side, 3, 0, 0, 0, 0, 0));
    set(off+13, ttopo.tecDetId(side, 3, 0, 0, 0, 0, 1));
    set(off+14, ttopo.tecDetId(side, 4, 0, 0, 0, 0, 0));
    set(off+15, ttopo.tecDetId(side, 4, 0, 0, 0, 0, 1));
    set(off+16, ttopo.tecDetId(side, 5, 0, 0, 0, 0, 0));
    set(off+17, ttopo.tecDetId(side, 5, 0, 0, 0, 0, 1));
    set(off+18, ttopo.tecDetId(side, 6, 0, 0, 0, 0, 0));
    set(off+19, ttopo.tecDetId(side, 6, 0, 0, 0, 0, 1));
    set(off+20, ttopo.tecDetId(side, 7, 0, 0, 0, 0, 0));
    set(off+21, ttopo.tecDetId(side, 7, 0, 0, 0, 0, 1));
    set(off+22, ttopo.tecDetId(side, 8, 0, 0, 0, 0, 0));
    set(off+23, ttopo.tecDetId(side, 8, 0, 0, 0, 0, 1));
    set(off+24, ttopo.tecDetId(side, 9, 0, 0, 0, 0, 0));
    set(off+25, ttopo.tecDetId(side, 9, 0, 0, 0, 0, 1));
  };

  // plus
  off = 17+1;
  setForward(2);

  // minus
  off = 17+1+25+2;
  setForward(1);

  return dets;
}

template <typename HitCollection>
void MkFitProducer::convertHits(const HitCollection& hits,
                                std::vector<mkfit::HitVec>& mkfitHits,
                                IndexLayer& indexLayers,
                                int& totalHits,
                                const TrackerTopology& ttopo,
                                const TransientTrackingRecHitBuilder& ttrhBuilder,
                                const mkfit::LayerNumberConverter& lnc) const {
  for(const auto& detset: hits) {
    const DetId detid = detset.detId();
    const auto subdet = detid.subdetId();
    const auto layer = ttopo.layer(detid);
    const auto isStereo = ttopo.isStereo(detid);

    for(const auto& hit: detset) {
      TransientTrackingRecHit::RecHitPointer ttrh = ttrhBuilder.build(&hit);

      SVector3 pos(ttrh->globalPosition().x(),
                   ttrh->globalPosition().y(),
                   ttrh->globalPosition().z());
      SMatrixSym33 err;
      err.At(0,0) = ttrh->globalPositionError().cxx();
      err.At(1,1) = ttrh->globalPositionError().cyy();
      err.At(2,2) = ttrh->globalPositionError().czz();
      err.At(0,1) = ttrh->globalPositionError().cyx();
      err.At(0,2) = ttrh->globalPositionError().czx();
      err.At(1,2) = ttrh->globalPositionError().czy();

      const auto ilay = lnc.convertLayerNumber(subdet, layer, false, isStereo, ttrh->globalPosition().z()>0);
      LogTrace("MkFitProducer") << "Adding hit detid " << detid.rawId()
                                << " subdet " << subdet
                                << " layer " << layer
                                << " isStereo " << isStereo
                                << " zplus " << (ttrh->globalPosition().z()>0)
                                << " ilay " << ilay;


      indexLayers.insert(hit.firstClusterRef().id(), hit.firstClusterRef().index(), mkfitHits[ilay].size(), ilay, &hit);
      mkfitHits[ilay].emplace_back(pos, err, totalHits);
      ++totalHits;
    }
  }
}

mkfit::TrackVec MkFitProducer::convertSeeds(const edm::View<TrajectorySeed>& seeds,
                                            const IndexLayer& indexLayers,
                                            const TransientTrackingRecHitBuilder& ttrhBuilder,
                                            const MagneticField& mf) const {
  mkfit::TrackVec ret;
  ret.reserve(seeds.size());
  int index = 0;
  for(const auto& seed: seeds) {
    const auto hitRange = seed.recHits();
    const auto lastRecHit = ttrhBuilder.build(&*(hitRange.second-1));
    const auto tsos = trajectoryStateTransform::transientState( seed.startingState(), lastRecHit->surface(), &mf);
    const auto& stateGlobal = tsos.globalParameters();
    SVector3 pos(stateGlobal.position().x(),
                 stateGlobal.position().y(),
                 stateGlobal.position().z());
    SVector3 mom(stateGlobal.momentum().x(),
                 stateGlobal.momentum().y(),
                 stateGlobal.momentum().z());

    const auto& cov = tsos.cartesianError().matrix();
    SMatrixSym66 err;
    for(int i=0; i<6; ++i) {
      for(int j=i; j<6; ++j) {
        err.At(i, j) = cov[i][j];
      }
    }

    mkfit::TrackState state(tsos.charge(), pos, mom, err);
    state.convertFromCartesianToCCS();
    ret.emplace_back(state, 0, index, 0, nullptr);

    // Add hits
    for(auto iHit = hitRange.first; iHit != hitRange.second; ++iHit) {
      const auto *hit = dynamic_cast<const BaseTrackerRecHit *>(&*iHit);
      if(hit == nullptr) {
        throw cms::Exception("Assert") << "Encountered a seed with a hit which is not BaseTrackerRecHit";
      }

      const auto& info = indexLayers.get(hit->firstClusterRef().id(), hit->firstClusterRef().index());
      ret.back().addHitIdx(info.index, info.layer, 0); // per-hit chi2 is not known
    }
    ++index;
  }
  return ret;
}

std::unique_ptr<TrackCandidateCollection> MkFitProducer::convertCandidates(const mkfit::Event& ev,
                                                                           const IndexLayer& indexLayers,
                                                                           const edm::View<TrajectorySeed>& seeds,
                                                                           const MagneticField& mf,
                                                                           const Propagator& propagatorAlong,
                                                                           const Propagator& propagatorOpposite,
                                                                           const TkClonerImpl& hitCloner,
                                                                           const std::vector<const DetLayer *>& detLayers,
                                                                           const mkfit::TrackVec& mkfitSeeds) const {
  auto output = std::make_unique<TrackCandidateCollection>();
  output->reserve(ev.candidateTracks_.size());

  LogTrace("MkFitProducer") << "Number of candidates " << ev.candidateTracks_.size()
                            << " extras " << ev.candidateTracksExtra_.size()
                            << "  seeds " << ev.seedTracks_.size();

  int candIndex = -1;
  for(const auto& cand: ev.candidateTracks_) {
    ++candIndex;
    LogTrace("MkFitProducer") << "Candidate " << candIndex << " pT " << cand.pT() << " eta " << cand.momEta() << " phi " << cand.momPhi() << " chi2 " << cand.chi2();

    // hits
    edm::OwnVector<TrackingRecHit> recHits;
    const int nhits = cand.nTotalHits(); // what exactly is the difference between nTotalHits() and nFoundHits()?
    for(int i=0; i<nhits; ++i) {
      const auto& hitOnTrack = cand.getHitOnTrack(i);
      LogTrace("MkFitProducer") << " hit on layer " << hitOnTrack.layer << " index " << hitOnTrack.index;
      if(hitOnTrack.index < 0 ) {
        // What is the exact meaning of -1, -2, -3?
        // In order to use the regular InvalidTrackingRecHit I'd need
        // a GeomDet (and "unfortunately" that is needed in
        // TrackProducer).
        //
        // I guess we could take the track state and propagate it to
        // each layer to find the actual module the track crosses, and
        // check whether it is active or not to be able to mark
        // inactive hits
        const auto *detLayer = detLayers.at(hitOnTrack.layer);
        if(detLayer == nullptr) {
          throw cms::Exception("LogicError") << "DetLayer for layer index " << hitOnTrack.layer << " is null!";
        }
        // Actually it is necessary to leave dealing with invalid hits to the TrackProducer?
        //recHits.push_back(new InvalidTrackingRecHitNoDet(detLayer->surface(), TrackingRecHit::missing)); // let's put them all as missing for now
      }
      else {
        recHits.push_back(indexLayers.getHitPtr(hitOnTrack.layer, hitOnTrack.index)->clone());
        LogTrace("MkFitProducer") << "  pos " << recHits.back().globalPosition().x()
                                  << " " << recHits.back().globalPosition().y()
                                  << " " << recHits.back().globalPosition().z()
                                  << " mag2 " << recHits.back().globalPosition().mag2()
                                  << " detid " << recHits.back().geographicalId().rawId()
                                  << " cluster " << indexLayers.getClusterIndex(hitOnTrack.layer, hitOnTrack.index);
      }
    }

    // seed
    const auto seedIndex = cand.label();
    LogTrace("MkFitProducer") << " from seed " << seedIndex << " seed hits";
    const auto& mkseed = mkfitSeeds.at(cand.label());
    for(int i=0; i<mkseed.nTotalHits(); ++i) {
      const auto& hitOnTrack = mkseed.getHitOnTrack(i);
      LogTrace("MkFitProducer") << "  hit on layer " << hitOnTrack.layer << " index " << hitOnTrack.index;
      // sanity check for now
      const auto& candHitOnTrack = cand.getHitOnTrack(i);
      if(hitOnTrack.layer != candHitOnTrack.layer) {
        //throw cms::Exception("LogicError")
        edm::LogError("MkFitProducer") << "Candidate " << candIndex << " from seed " << seedIndex << " hit " << i
                                       << " has different layer in candidate (" << candHitOnTrack.layer << ") and seed (" << hitOnTrack.layer << ")."
                                       << " Hit indices are " << candHitOnTrack.index << " and " << hitOnTrack.index << ", respectively";
      }
      if(hitOnTrack.index != candHitOnTrack.index) {
        //throw cms::Exception("LogicError")
        edm::LogError("MkFitProducer")
          << "Candidate " << candIndex << " from seed " << seedIndex << " hit " << i
          << " has different hit index in candidate (" << candHitOnTrack.index << ") and seed (" << hitOnTrack.index << ") on layer " << hitOnTrack.layer;
      }
    }

    // state
    auto state = cand.state(); // copy because have to modify
    state.convertFromCCSToCartesian();
    const auto& param = state.parameters;
    const auto& err = state.errors;
    AlgebraicSymMatrix66 cov;
    for(int i=0; i<6; ++i) {
      for(int j=i; j<6; ++j) {
        cov[i][j] = err.At(i, j);
      }
    }

    auto fts = FreeTrajectoryState(GlobalTrajectoryParameters(GlobalPoint(param[0], param[1], param[2]),
                                                              GlobalVector(param[3], param[4], param[5]),
                                                              state.charge,
                                                              &mf),
                                   CartesianTrajectoryError(cov));
    if(!fts.curvilinearError().posDef()) {
      edm::LogWarning("MkFitProducer") << "Curvilinear error not pos-def\n" << fts.curvilinearError().matrix()
                                       << "\noriginal 6x6 covariance matrix\n" << cov
                                       << "\ncandidate ignored";
      continue;
    }

    auto tsosDet = convertToInitialState(fts, recHits, propagatorAlong, propagatorOpposite, hitCloner);
    if(!tsosDet.first.isValid()) {
      edm::LogWarning("MkFitProducer") << "Backward fit of candidate " << candIndex << " failed, ignoring the candidate";
      continue;
    }
    
    // convert to persistent, from CkfTrackCandidateMakerBase
    auto pstate = trajectoryStateTransform::persistentState(tsosDet.first, tsosDet.second->geographicalId().rawId());

    output->emplace_back(recHits,
                         seeds.at(seedIndex),
                         pstate,
                         seeds.refAt(seedIndex),
                         0, // nloops, let's ignore for now
                         static_cast<uint8_t>(StopReason::UNINITIALIZED) // let's ignore the details of stopping reason as well for now
                         );
  }
  return output;
}

std::pair<TrajectoryStateOnSurface, const GeomDet *> MkFitProducer::convertToInitialState(const FreeTrajectoryState& fts,
                                                                                          const edm::OwnVector<TrackingRecHit>& hits,
                                                                                          const Propagator& propagatorAlong,
                                                                                          const Propagator& propagatorOpposite,
                                                                                          const TkClonerImpl& hitCloner) const {
  // First filter valid hits as in TransientInitialStateEstimator
  TransientTrackingRecHit::ConstRecHitContainer firstHits;

  for(int i=hits.size()-1; i >= 0; --i) {
    if(hits[i].det()) {
      // TransientTrackingRecHit::ConstRecHitContainer has shared_ptr,
      // and it is passed to backFitter below so it is really needed
      // to keep the interface. Since we keep the ownership in hits,
      // let's disable the deleter.
      firstHits.emplace_back(&(hits[i]), edm::do_nothing_deleter{});
    }
  }

  // Then propagate along to the surface of the last hit to get a TSOS
  const auto& lastHitSurface = firstHits.front()->det()->surface();
  /*
  const Propagator *propagator = &propagatorAlong;
  if(const auto *prop = dynamic_cast<const PropagatorWithMaterial *>(propagator)) {
    propagator = prop;
  }
  */
  auto tsosDouble = propagatorAlong.propagateWithPath(fts, lastHitSurface);
  if(!tsosDouble.first.isValid()) {
    LogDebug("MkFitProducer") << "Propagating to startingState along momentum failed, trying opposite next";
    tsosDouble = propagatorOpposite.propagateWithPath(fts, lastHitSurface);
  }
  auto& startingState = tsosDouble.first;

  if(!startingState.isValid()) {
    edm::LogWarning("MkFitProducer") << "startingState is not valid, FTS was\n"
                                     << fts
                                     << " last hit surface surface:"
                                     << "\n position " << lastHitSurface.position()
                                     << "\n phiSpan " << lastHitSurface.phiSpan().first << "," << lastHitSurface.phiSpan().first
                                     << "\n rSpan " << lastHitSurface.rSpan().first << "," << lastHitSurface.rSpan().first
                                     << "\n zSpan " << lastHitSurface.zSpan().first << "," << lastHitSurface.zSpan().first;
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>();
  }

  // Then return back to the logic from TransientInitialStateEstimator
  startingState.rescaleError(100.);

  // avoid cloning
  KFUpdator const aKFUpdator;
  Chi2MeasurementEstimator const aChi2MeasurementEstimator( 100., 3);
  KFTrajectoryFitter backFitter( &propagatorAlong,
                                 &aKFUpdator,
                                 &aChi2MeasurementEstimator,
                                 firstHits.size(), nullptr, &hitCloner);

  PropagationDirection backFitDirection = oppositeToMomentum; // assume for now that the propagation in mkfit always alongMomentum

  // only direction matters in this contest
  TrajectorySeed fakeSeed(PTrajectoryStateOnDet(),
                          edm::OwnVector<TrackingRecHit>(),
                          backFitDirection);

  Trajectory && fitres = backFitter.fitOne( fakeSeed, firstHits, startingState, TrajectoryFitter::standard); // ignore loopers for now

  LogDebug("MkFitProducer")
    <<"using a backward fit of :"<<firstHits.size()<<" hits, starting from:\n"<<startingState
    <<" to get the estimate of the initial state of the track.";

  if(!fitres.isValid()) {
    edm::LogWarning("MkFitProducer") << "FitTester: first hits fit failed";
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>();
  }

  TrajectoryMeasurement const & firstMeas = fitres.lastMeasurement();

  // magnetic field can be different!
  TrajectoryStateOnSurface firstState(firstMeas.updatedState().localParameters(),
                                      firstMeas.updatedState().localError(),
                                      firstMeas.updatedState().surface(),
                                      propagatorAlong.magneticField());

  firstState.rescaleError(100.);

  LogDebug("MkFitProducer")
    <<"the initial state is found to be:\n:"<<firstState
    <<"\n it's field pointer is: "<<firstState.magneticField()
    <<"\n the pointer from the state of the back fit was: "<<firstMeas.updatedState().magneticField();


  return std::make_pair(firstState, firstMeas.recHit()->det());
}


DEFINE_FWK_MODULE(MkFitProducer);
