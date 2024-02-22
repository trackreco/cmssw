#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"
#include "DataFormats/TrackerRecHit2D/interface/SiStripRecHit2DCollection.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"

#include "Geometry/Records/interface/TrackerTopologyRcd.h"

#include "TrackingTools/Records/interface/TransientRecHitRecord.h"

#include "RecoTracker/MkFit/interface/MkFitHitWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "convertHits.h"

namespace {
  class ConvertHitTraits {
  public:
    ConvertHitTraits(float minCharge) : minGoodStripCharge_(minCharge) {}

    static constexpr bool applyCCC() { return true; }
    static float clusterCharge(const SiStripRecHit2D& hit, DetId hitId) {
      return siStripClusterTools::chargePerCM(hitId, hit.firstClusterRef().stripCluster());
    }
    bool passCCC(float charge) const { return charge > minGoodStripCharge_; }
    static void setDetails(mkfit::Hit& mhit, const SiStripCluster& cluster, int shortId, float charge) {
      mhit.setupAsStrip(shortId, charge, cluster.amplitudes().size());
    }

  private:
    const float minGoodStripCharge_;
  };

  struct ConvertHitTraitsPhase2 {
    static constexpr bool applyCCC() { return false; }
    static std::nullptr_t clusterCharge(const Phase2TrackerRecHit1D& hit, DetId hitId) { return nullptr; }
    static bool passCCC(std::nullptr_t) { return true; }
    static void setDetails(mkfit::Hit& mhit, const Phase2TrackerCluster1D& cluster, int shortId, std::nullptr_t) {
      mhit.setupAsStrip(shortId, (1 << 8) - 1, cluster.size());
    }
  };

}  // namespace

class MkFitSiStripHitConverter : public edm::global::EDProducer<> {
public:
  explicit MkFitSiStripHitConverter(edm::ParameterSet const& iConfig);
  ~MkFitSiStripHitConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::EDGetTokenT<SiStripRecHit2DCollection> stripRphiRecHitToken_;
  const edm::EDGetTokenT<SiStripRecHit2DCollection> stripStereoRecHitToken_;
  const edm::EDGetTokenT<Phase2TrackerRecHit1DCollectionNew> siPhase2RecHitToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhBuilderToken_;
  const edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
  const edm::EDPutTokenT<MkFitHitWrapper> wrapperPutToken_;
  const edm::EDPutTokenT<MkFitClusterIndexToHit> clusterIndexPutToken_;
  const edm::EDPutTokenT<std::vector<float>> clusterChargePutToken_;
  const ConvertHitTraits convertTraits_;
  const bool isPhase2_;
};

MkFitSiStripHitConverter::MkFitSiStripHitConverter(edm::ParameterSet const& iConfig)
    : stripRphiRecHitToken_{consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("rphiHits"))},
      stripStereoRecHitToken_{consumes<SiStripRecHit2DCollection>(iConfig.getParameter<edm::InputTag>("stereoHits"))},
      siPhase2RecHitToken_{
          consumes<Phase2TrackerRecHit1DCollectionNew>(iConfig.getParameter<edm::InputTag>("siPhase2Hits"))},
      ttrhBuilderToken_{esConsumes<TransientTrackingRecHitBuilder, TransientRecHitRecord>(
          iConfig.getParameter<edm::ESInputTag>("ttrhBuilder"))},
      ttopoToken_{esConsumes<TrackerTopology, TrackerTopologyRcd>()},
      mkFitGeomToken_{esConsumes<MkFitGeometry, TrackerRecoGeometryRecord>()},
      wrapperPutToken_{produces<MkFitHitWrapper>()},
      clusterIndexPutToken_{produces<MkFitClusterIndexToHit>()},
      clusterChargePutToken_{produces<std::vector<float>>()},
      convertTraits_{static_cast<float>(
          iConfig.getParameter<edm::ParameterSet>("minGoodStripCharge").getParameter<double>("value"))},
      isPhase2_{iConfig.getParameter<bool>("isPhase2")} {}

void MkFitSiStripHitConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("rphiHits", edm::InputTag{"siStripMatchedRecHits", "rphiRecHit"});
  desc.add("stereoHits", edm::InputTag{"siStripMatchedRecHits", "stereoRecHit"});
  desc.add("siPhase2Hits", edm::InputTag{"siPhase2RecHits"});
  desc.add("ttrhBuilder", edm::ESInputTag{"", "WithTrackAngle"});

  edm::ParameterSetDescription descCCC;
  descCCC.add<double>("value");
  desc.add("minGoodStripCharge", descCCC);

  desc.add("isPhase2", false);

  descriptions.add("mkFitSiStripHitConverterDefault", desc);
}

void MkFitSiStripHitConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& ttrhBuilder = iSetup.getData(ttrhBuilderToken_);
  const auto& ttopo = iSetup.getData(ttopoToken_);
  const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

  MkFitHitWrapper hitWrapper;
  MkFitClusterIndexToHit clusterIndexToHit;
  std::vector<float> clusterCharge;

  auto convert = [&](auto& hits) {
    return mkfit::convertHits(
        convertTraits_, hits, hitWrapper.hits(), clusterIndexToHit.hits(), clusterCharge, ttopo, ttrhBuilder, mkFitGeom);
  };

  edm::ProductID stripClusterID;
  if (!(isPhase2_)) {
    const auto& stripRphiHits = iEvent.get(stripRphiRecHitToken_);
    const auto& stripStereoHits = iEvent.get(stripStereoRecHitToken_);
    if (not stripRphiHits.empty()) {
      stripClusterID = convert(stripRphiHits);
    }
    if (not stripStereoHits.empty()) {
      auto stripStereoClusterID = convert(stripStereoHits);
      if (stripRphiHits.empty()) {
        stripClusterID = stripStereoClusterID;
      } else if (stripClusterID != stripStereoClusterID) {
        throw cms::Exception("LogicError") << "Encountered different cluster ProductIDs for strip RPhi hits ("
                                           << stripClusterID << ") and stereo (" << stripStereoClusterID << ")";
      }
    }
  } else {
    const auto& phase2Hits = iEvent.get(siPhase2RecHitToken_);
    std::vector<float> dummy;
    if (not phase2Hits.empty()) {
      stripClusterID = mkfit::convertHits(ConvertHitTraitsPhase2{},
                                          phase2Hits,
                                          hitWrapper.hits(),
                                          clusterIndexToHit.hits(),
                                          dummy,
                                          ttopo,
                                          ttrhBuilder,
                                          mkFitGeom);
    }
  }
  hitWrapper.setClustersID(stripClusterID);

  iEvent.emplace(wrapperPutToken_, std::move(hitWrapper));
  iEvent.emplace(clusterIndexPutToken_, std::move(clusterIndexToHit));
  iEvent.emplace(clusterChargePutToken_, std::move(clusterCharge));
}

DEFINE_FWK_MODULE(MkFitSiStripHitConverter);
