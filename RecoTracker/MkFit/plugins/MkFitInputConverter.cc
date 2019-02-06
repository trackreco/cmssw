#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"

#include "RecoTracker/MkFit/interface/MkFitInputWrapper.h"

// ROOT
#include "Math/SVector.h"
#include "Math/SMatrix.h"

// MkFit includes
#include "Hit.h"
#include "Track.h"
#include "LayerNumberConverter.h"

class MkFitInputConverter: public edm::global::EDProducer<> {
public:
  explicit MkFitInputConverter(edm::ParameterSet const& iConfig);
  ~MkFitInputConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  template <typename HitCollection>
  void convertHits(const HitCollection& hits,
                   std::vector<mkfit::HitVec>& mkfitHits,
                   MkFitIndexLayer& indexLayers,
                   int& totalHits,
                   const TrackerTopology& ttopo,
                   const TransientTrackingRecHitBuilder& ttrhBuilder,
                   const mkfit::LayerNumberConverter& lnc) const;

  bool passCCC(const SiStripRecHit2D& hit, const DetId hitId) const;
  bool passCCC(const SiPixelRecHit& hit, const DetId hitId) const;

  mkfit::TrackVec convertSeeds(const edm::View<TrajectorySeed>& seeds,
                               const MkFitIndexLayer& indexLayers,
                               const TransientTrackingRecHitBuilder& ttrhBuilder,
                               const MagneticField& mf) const;

  using SVector3 = ROOT::Math::SVector<float, 3>;
  using SMatrixSym33 = ROOT::Math::SMatrix<float,3,3,ROOT::Math::MatRepSym<float,3> >;
  using SMatrixSym66 = ROOT::Math::SMatrix<float,6,6,ROOT::Math::MatRepSym<float,6> >;

  edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken_;
  edm::EDPutTokenT<MkFitInputWrapper> putToken_;
  std::string ttrhBuilderName_;
};

MkFitInputConverter::MkFitInputConverter(edm::ParameterSet const& iConfig):
  pixelRecHitToken_(consumes<SiPixelRecHitCollection>(iConfig.getParameter<edm::InputTag>("pixelRecHits"))),
  stripRphiRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripRphiRecHits"))),
  stripStereoRecHitToken_(consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stripStereoRecHits"))),
  seedToken_(consumes<edm::View<TrajectorySeed> >(iConfig.getParameter<edm::InputTag>("seeds"))),
  putToken_(produces<MkFitInputWrapper>()),
  ttrhBuilderName_(iConfig.getParameter<std::string>("ttrhBuilder"))
{}

void MkFitInputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("pixelRecHits", edm::InputTag("siPixelRecHits"));
  desc.add("stripRphiRecHits", edm::InputTag("siStripMatchedRecHits", "rphiRecHit"));
  desc.add("stripStereoRecHits", edm::InputTag("siStripMatchedRecHits", "stereoRecHit"));
  desc.add("seeds", edm::InputTag("initialStepSeeds"));
  desc.add<std::string>("ttrhBuilder", "WithTrackAngle");

  descriptions.addWithDefaultLabel(desc);
}

void MkFitInputConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  mkfit::LayerNumberConverter lnc{mkfit::TkLayout::phase1};

  // Then import hits
  edm::Handle<SiPixelRecHitCollection> pixelHits;
  iEvent.getByToken(pixelRecHitToken_, pixelHits);

  edm::Handle<SiStripRecHit2DCollection> stripRphiHits;
  iEvent.getByToken(stripRphiRecHitToken_, stripRphiHits);

  edm::Handle<SiStripRecHit2DCollection> stripStereoHits;
  iEvent.getByToken(stripStereoRecHitToken_, stripStereoHits);

  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhBuilder;
  iSetup.get<TransientRecHitRecord>().get(ttrhBuilderName_, ttrhBuilder);

  edm::ESHandle<TrackerTopology> ttopo;
  iSetup.get<TrackerTopologyRcd>().get(ttopo);

  std::vector<mkfit::HitVec> mkfitHits(lnc.nLayers());
  MkFitIndexLayer indexLayers;
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

  iEvent.emplace(putToken_, std::move(indexLayers), std::move(mkfitHits), std::move(mkfitSeeds), std::move(lnc));
}

bool MkFitInputConverter::passCCC(const SiStripRecHit2D& hit, const DetId hitId) const {
  return (siStripClusterTools::chargePerCM(hitId,hit.firstClusterRef().stripCluster()) < 1620 );
}

bool MkFitInputConverter::passCCC(const SiPixelRecHit& hit, const DetId hitId) const {
  return true;
}

template <typename HitCollection>
void MkFitInputConverter::convertHits(const HitCollection& hits,
                                std::vector<mkfit::HitVec>& mkfitHits,
                                MkFitIndexLayer& indexLayers,
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
      if(!passCCC(hit, detid)) continue;

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
      LogTrace("MkFitInputConverter") << "Adding hit detid " << detid.rawId()
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

mkfit::TrackVec MkFitInputConverter::convertSeeds(const edm::View<TrajectorySeed>& seeds,
                                            const MkFitIndexLayer& indexLayers,
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

DEFINE_FWK_MODULE(MkFitInputConverter);
