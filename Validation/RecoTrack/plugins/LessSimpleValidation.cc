// system include files
#include <memory>

#include "TTree.h"
#include "TFile.h"


// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "SimTracker/TrackerHitAssociation/interface/ClusterTPAssociation.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"
#include "SimDataFormats/Associations/interface/TrackToTrackingParticleAssociator.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"
#include "CommonTools/RecoAlgos/interface/RecoTrackSelectorBase.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "Validation/RecoTrack/interface/MTVHistoProducerAlgoForTracker.h"

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.

using reco::TrackCollection;

struct RegionInfo {
  int rt = 0;
  int at = 0;
  int st = 0;
  int dt = 0;
  int ast = 0;
};

template <typename T>
struct ExtraSelector {
  double minPt = 0;
  double maxPt = 1.e100;
  double minTip = 0;
  double maxTip = 1.e100;

  bool operator()(const T& t) {
    if (t.pt() < minPt) {
      // if constexpr (std::is_same_v<T, TrackingParticle>) std::clog << "Discarding track because the pt " << t.pt() << " is smaller than min " << minPt << std::endl;
      return false;
    }
    if (t.pt() > maxPt) {
      // if constexpr (std::is_same_v<T, TrackingParticle>) std::clog << "Discarding track because the pt " << t.pt() << " is larger than max " << maxPt << std::endl;
      return false;
    }
    if (not tipOk(t)) {
      return false;
    }

    return true;
  }
private:
  bool tipOk(const T& t);
};

template <>
bool ExtraSelector<reco::Track>::tipOk(const reco::Track& t) {
  reco::Track::Point vertex;
  auto tip = abs(t.dxy(vertex));
  if (tip < minTip) {
    return false;
  }
  if (tip > maxTip) {
    return false;
  }
  return true;
}

template <>
bool ExtraSelector<TrackingParticle>::tipOk(const TrackingParticle& tp) {
  auto tip2 = tp.vertex().perp2();
  if (tip2 < minTip*minTip) {
    // std::clog << "Discarding track because the tip " << sqrt(tip2) << " is smaller than " << minTip << std::endl;
    return false;
  }
  if (tip2 > maxTip*maxTip) {
    // std::clog << "Discarding track because the tip " << sqrt(tip2) << " is larger than " << maxTip << std::endl;
    return false;
  }
  return true;
}

class LessSimpleValidation : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit LessSimpleValidation(const edm::ParameterSet&);
  ~LessSimpleValidation() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  std::vector<RegionInfo> regions;
  std::vector<std::string> region_labels;

  TrackingParticleSelector tpSelector;
  std::vector<ExtraSelector<TrackingParticle>> tpSelectors;
  std::vector<ExtraSelector<reco::Track>> trackSelectors;
  TTree* output_tree_;
  std::vector<edm::InputTag> trackLabels_;
  edm::EDGetTokenT<ClusterTPAssociation> tpMap_;
//   edm::EDGetTokenT<std::vector<PileupSummaryInfo>>  infoPileUp_;
  std::vector<edm::EDGetTokenT<edm::View<reco::Track>>> trackTokens_;
  edm::EDGetTokenT<reco::TrackToTrackingParticleAssociator> trackAssociatorToken_;
  edm::EDGetTokenT<TrackingParticleCollection> trackingParticleToken_;

//   const double sharingFraction_;
//   const double sharingFractionForTriplets_;

  // static RecoTrackSelectorBase makeTrackSelector(const edm::ParameterSet& pset);
};

// template <typename T>
// T pset_fallback(const edm::ParameterSet& pset, const edm::ParameterSet& fallback, const char* label) {
//   if (pset.existsAs<T>(label)) return pset.getParameter<T>(label);
//   else return fallback.getParameter<T>(label);
// }

// TrackingParticleSelector tpSelectorWithFallback(const edm::ParameterSet& pset, const edm::ParameterSet& fallback) {
//
//   return TrackingParticleSelector{
//     pset_fallback<double>(pset, fallback, "ptMinTP"),
//     pset_fallback<double>(pset, fallback, "ptMaxTP"),
//     pset_fallback<double>(pset, fallback, "minRapidityTP"),
//     pset_fallback<double>(pset, fallback, "maxRapidityTP"),
//     pset_fallback<double>(pset, fallback, "tipTP"),
//     pset_fallback<double>(pset, fallback, "lipTP"),
//     pset_fallback<int>(pset, fallback, "minHitTP"),
//     pset_fallback<bool>(pset, fallback, "signalOnlyTP"),
//     pset_fallback<bool>(pset, fallback, "intimeOnlyTP"),
//     pset_fallback<bool>(pset, fallback, "chargedOnlyTP"),
//     pset_fallback<bool>(pset, fallback, "stableOnlyTP"),
//     pset_fallback<std::vector<int>>(pset, fallback, "pdgIdTP"),
//     pset_fallback<bool>(pset, fallback, "invertRapidityCutTP")
//   };
// }

template <typename T>
ExtraSelector<T> makeSelector(const edm::ParameterSet& pset) {
  ExtraSelector<T> sel;
  if (pset.existsAs<double>("minPt")) {
    sel.minPt = pset.getParameter<double>("minPt");
    //std::clog << "Setting pt min to " << sel.minPt << " in region " << pset.getParameter<std::string>("label") << std::endl;
  }
  if (pset.existsAs<double>("maxPt")) {
    sel.maxPt = pset.getParameter<double>("maxPt");
    //std::clog << "Setting pt max to " << sel.maxPt << " in region " << pset.getParameter<std::string>("label") << std::endl;
  }
  if (pset.existsAs<double>("minTip")) {
    sel.minTip = pset.getParameter<double>("minTip");
    //std::clog << "Setting tip min to " << sel.minTip << " in region " << pset.getParameter<std::string>("label") << std::endl;
  }
  if (pset.existsAs<double>("maxTip")) {
    sel.maxTip = pset.getParameter<double>("maxTip");
    //std::clog << "Setting tip max to " << sel.maxTip << " in region " << pset.getParameter<std::string>("label") << std::endl;
  }

  return sel;
}

LessSimpleValidation::LessSimpleValidation(const edm::ParameterSet& iConfig)
    : trackLabels_(iConfig.getParameter<std::vector<edm::InputTag>>("trackLabels")),
      // tpMap_(consumes(iConfig.getParameter<edm::InputTag>("tpMap"))),
    //   infoPileUp_(consumes(iConfig.getParameter< edm::InputTag >("infoPileUp"))),
      trackAssociatorToken_(consumes<reco::TrackToTrackingParticleAssociator>(iConfig.getUntrackedParameter<edm::InputTag>("trackAssociator"))),
      trackingParticleToken_(consumes<TrackingParticleCollection>(iConfig.getParameter< edm::InputTag >("trackingParticles")))
    //   sharingFraction_(iConfig.getUntrackedParameter<double>("sharingFraction")),
    //   sharingFractionForTriplets_(iConfig.getUntrackedParameter<double>("sharingFractionForTriplets"))
{
  for (auto& itag : trackLabels_) {
    trackTokens_.push_back(consumes<edm::View<reco::Track>>(itag));
    // edm::LogPrint("TrackValidator") << itag.label() << "\n";
  }
  tpSelector = TrackingParticleSelector(iConfig.getParameter<double>("ptMinTP"),
                                        iConfig.getParameter<double>("ptMaxTP"),
                                        iConfig.getParameter<double>("minRapidityTP"),
                                        iConfig.getParameter<double>("maxRapidityTP"),
                                        iConfig.getParameter<double>("tipTP"),
                                        iConfig.getParameter<double>("lipTP"),
                                        iConfig.getParameter<int>("minHitTP"),
                                        iConfig.getParameter<bool>("signalOnlyTP"),
                                        iConfig.getParameter<bool>("intimeOnlyTP"),
                                        iConfig.getParameter<bool>("chargedOnlyTP"),
                                        iConfig.getParameter<bool>("stableOnlyTP"),
                                        iConfig.getParameter<std::vector<int>>("pdgIdTP"),
                                        iConfig.getParameter<bool>("invertRapidityCutTP"));

  auto&& regionSets = iConfig.getParameter<std::vector<edm::ParameterSet>>("regions");
  tpSelectors.reserve(regionSets.size());

  for (auto& regionPSet: regionSets) {
    tpSelectors.push_back(makeSelector<TrackingParticle>(regionPSet));
    trackSelectors.push_back(makeSelector<reco::Track>(regionPSet));
    region_labels.push_back(regionPSet.getParameter<std::string>("label"));
  }

  regions.resize(regionSets.size());

  //now do what ever initialization is needed

  //trackSelector = makeTrackSelector(iConfig);
}

// RecoTrackSelectorBase LessSimpleValidation::makeTrackSelector(const edm::ParameterSet& pset) {
//   edm::ParameterSet psetTrack;
//   psetTrack.copyForModify(pset);
//
//   auto add_if_needed = [&psetTrack](const char* label, auto value) {
//     if (not psetTrack.existsAs<decltype(value)>(label)) psetTrack.addParameter(label, value);
//   };
//   add_if_needed("minHit", 0);
//   add_if_needed("maxChi2", 1e10);
//   add_if_needed("minPixelHit", 0);
//   add_if_needed("minLayer", 0);
//   add_if_needed("min3DLayer", 0);
//   add_if_needed("quality", std::vector<std::string>{});
//   add_if_needed("algorithm", std::vector<std::string>{});
//   add_if_needed("originalAlgorithm", std::vector<std::string>{});
//   add_if_needed("algorithmMaskContains", std::vector<std::string>{});
//   add_if_needed("minPhi", -3.2);
//   add_if_needed("maxPhi", 3.2);
//   add_if_needed("minRapidity", -1.e100);
//   add_if_needed("maxRapidity", 1.e100);
//   add_if_needed("ptMin", 0.);
//   add_if_needed("ptMax", 1.e100);
//   add_if_needed("tip", 1.e100);
//   add_if_needed("lip", 1.e100);
//
//   return psetTrack;
// }

LessSimpleValidation::~LessSimpleValidation() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
  //
  // please remove this method altogether if it would be left empty
}

//
// member functions
//

// ------------ method called for each event  ------------
void LessSimpleValidation::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
   

//   auto const& tpClust = iEvent.get(tpMap_);
  auto const& associatorByHits = iEvent.get(trackAssociatorToken_);
  
  TrackingParticleRefVector tpCollection;
  edm::Handle<TrackingParticleCollection> TPCollectionH;
  iEvent.getByToken(trackingParticleToken_, TPCollectionH);
//   auto const& tp = iEvent.get(trackingParticleToken_);

  for (size_t i = 0, size = TPCollectionH->size(); i < size; ++i) {
    auto tp = TrackingParticleRef(TPCollectionH, i);
    tpCollection.push_back(tp);
  }

  std::vector<RegionInfo> local_regions(regions.size());
  for (const auto& trackToken : trackTokens_) 
  {

    edm::Handle<edm::View<reco::Track>> tracksHandle;
    iEvent.getByToken(trackToken, tracksHandle);
    const edm::View<reco::Track>& tracks = *tracksHandle;
    
    edm::RefToBaseVector<reco::Track> trackRefs;
    for (edm::View<reco::Track>::size_type i = 0; i < tracks.size(); ++i) {
      auto trackRef = tracks.refAt(i);
      trackRefs.push_back(trackRef);
    }

    reco::RecoToSimCollection recSimColl = associatorByHits.associateRecoToSim(trackRefs, tpCollection);
    reco::SimToRecoCollection simRecColl = associatorByHits.associateSimToReco(trackRefs, tpCollection);

    local_regions.clear();
    local_regions.resize(regions.size());

    for (const auto& track : trackRefs) {
      for (int iReg = 0; iReg < (int)regions.size(); iReg++) if (trackSelectors[iReg](*track)) regions[iReg].rt++;
      auto foundTP = recSimColl.find(track);
      if (foundTP != recSimColl.end()) {
        const auto& tp = foundTP->val;
        if (!tp.empty()) {
          for (int iReg = 0; iReg < (int)regions.size(); iReg++) if (trackSelectors[iReg](*track)) regions[iReg].at++;
        }
        if (simRecColl.find(tp[0].first) != simRecColl.end()) {
          if (simRecColl[tp[0].first].size() > 1) {
            for (int iReg = 0; iReg < (int)regions.size(); iReg++) if (trackSelectors[iReg](*track)) regions[iReg].dt++;
          }
        }
      }
    }
    for (const TrackingParticleRef& tpr : tpCollection) {
      if (not tpSelector(*tpr)) continue;
      for (int iReg = 0; iReg < (int)regions.size(); iReg++) if (tpSelectors[iReg](*tpr)) regions[iReg].st++;
      auto foundTrack = simRecColl.find(tpr);
      if (foundTrack != simRecColl.end()) {
        for (int iReg = 0; iReg < (int)regions.size(); iReg++) if (tpSelectors[iReg](*tpr)) regions[iReg].ast++;
      }
    }

    // LogPrint("TrackValidator") << "Tag " << trackLabels_[0].label() << " Total in collection " << tracks.size() << " Total simulated "<< st << " Associated tracks " << at << " Total reconstructed " << rt;

    for (int iReg = 0; iReg < (int)regions.size(); iReg++) {
      regions[iReg].rt += local_regions[iReg].rt;
      regions[iReg].at += local_regions[iReg].at;
      regions[iReg].dt += local_regions[iReg].dt;
      regions[iReg].st += local_regions[iReg].st;
      regions[iReg].ast += local_regions[iReg].ast;
    }
  }
}

// ------------ method called once each job just before starting event loop  ------------
void LessSimpleValidation::beginJob() {
  // please remove this method if not needed
  edm::Service<TFileService> fs;
  output_tree_ = fs->make<TTree>("output", "putput params");

  for (int iReg = 0; iReg < (int)region_labels.size(); iReg++) {
    auto&& name = region_labels[iReg];
    auto&& region = regions[iReg];

    output_tree_->Branch((name + "_rt").c_str(), &region.rt);
    output_tree_->Branch((name + "_at").c_str(), &region.at);
    output_tree_->Branch((name + "_st").c_str(), &region.st);
    output_tree_->Branch((name + "_dt").c_str(), &region.dt);
    output_tree_->Branch((name + "_ast").c_str(), &region.ast);
  }
}

// ------------ method called once each job just after ending the event loop  ------------
void LessSimpleValidation::endJob() {
  // please remove this method if not needed
  output_tree_->Fill();
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void LessSimpleValidation::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(LessSimpleValidation);
