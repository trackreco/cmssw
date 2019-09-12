#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Utilities/interface/do_nothing_deleter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/TrajectorySeed/interface/TrajectorySeed.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"
#include "DataFormats/TrackReco/interface/SeedStopInfo.h"
#include "DataFormats/TrackingRecHit/interface/InvalidTrackingRecHit.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
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

#include "RecoTracker/MkFit/interface/MkFitInputWrapper.h"
#include "RecoTracker/MkFit/interface/MkFitOutputWrapper.h"

// mkFit indludes
#include "LayerNumberConverter.h"
#include "Track.h"

namespace {
  template <typename T>
  bool isBarrel(T subdet) {
    return subdet == PixelSubdetector::PixelBarrel || subdet == StripSubdetector::TIB || subdet == StripSubdetector::TOB;
  }

  template <typename T>
  bool isEndcap(T subdet) {
    return subdet == PixelSubdetector::PixelEndcap || subdet == StripSubdetector::TID || subdet == StripSubdetector::TEC;
  }
}


class MkFitOutputConverter: public edm::global::EDProducer<> {
public:
  explicit MkFitOutputConverter(edm::ParameterSet const& iConfig);
  ~MkFitOutputConverter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  std::vector<const DetLayer*> createDetLayers(const mkfit::LayerNumberConverter& lnc,
                                               const GeometricSearchTracker& tracker,
                                               const TrackerTopology& ttopo) const;

  std::unique_ptr<TrackCandidateCollection> convertCandidates(const MkFitOutputWrapper& mkFitOutput,
                                                              const MkFitHitIndexMap& hitIndexMap,
                                                              const edm::View<TrajectorySeed>& seeds,
                                                              const TrackerGeometry& geom,
                                                              const MagneticField& mf,
                                                              const Propagator& propagatorAlong,
                                                              const Propagator& propagatorOpposite,
                                                              const TkClonerImpl& hitCloner,
                                                              const std::vector<const DetLayer*>& detLayers,
                                                              const mkfit::TrackVec& mkFitSeeds) const;

  std::pair<TrajectoryStateOnSurface, const GeomDet*> backwardFit(const FreeTrajectoryState& fts,
                                                                  const edm::OwnVector<TrackingRecHit>& hits,
                                                                  const Propagator& propagatorAlong,
                                                                  const Propagator& propagatorOpposite,
                                                                  const TkClonerImpl& hitCloner,
                                                                  bool lastHitWasInvalid,
                                                                  bool lastHitWasChanged) const;

  std::pair<TrajectoryStateOnSurface, const GeomDet*> convertInnermostState(const FreeTrajectoryState& fts,
                                                                            const edm::OwnVector<TrackingRecHit>& hits,
                                                                            const Propagator& propagatorAlong,
                                                                            const Propagator& propagatorOpposite) const;

  edm::EDGetTokenT<MkFitInputWrapper> hitsSeedsToken_;
  edm::EDGetTokenT<MkFitOutputWrapper> tracksToken_;
  edm::EDGetTokenT<edm::View<TrajectorySeed> > seedToken_;
  edm::EDGetTokenT<MeasurementTrackerEvent> mteToken_;
  std::string ttrhBuilderName_;
  std::string propagatorAlongName_;
  std::string propagatorOppositeName_;
  bool backwardFitInCMSSW_;
};

MkFitOutputConverter::MkFitOutputConverter(edm::ParameterSet const& iConfig):
  hitsSeedsToken_(consumes<MkFitInputWrapper>(iConfig.getParameter<edm::InputTag>("hitsSeeds"))),
  tracksToken_(consumes<MkFitOutputWrapper>(iConfig.getParameter<edm::InputTag>("tracks"))),
  seedToken_(consumes<edm::View<TrajectorySeed> >(iConfig.getParameter<edm::InputTag>("seeds"))),
  mteToken_(consumes<MeasurementTrackerEvent>(iConfig.getParameter<edm::InputTag>("measurementTrackerEvent"))),
  ttrhBuilderName_(iConfig.getParameter<std::string>("ttrhBuilder")),
  propagatorAlongName_(iConfig.getParameter<std::string>("propagatorAlong")),
  propagatorOppositeName_(iConfig.getParameter<std::string>("propagatorOpposite")),
  backwardFitInCMSSW_(iConfig.getParameter<bool>("backwardFitInCMSSW"))
{
  produces<TrackCandidateCollection>();
  produces<std::vector<SeedStopInfo> >();
}

void MkFitOutputConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add("hitsSeeds", edm::InputTag("mkFitInputConverter"));
  desc.add("tracks", edm::InputTag("mkfitProducer"));
  desc.add("seeds", edm::InputTag("initialStepSeeds"));
  desc.add("measurementTrackerEvent", edm::InputTag("MeasurementTrackerEvent"));
  desc.add<std::string>("ttrhBuilder", "WithTrackAngle");
  desc.add<std::string>("propagatorAlong", "PropagatorWithMaterial");
  desc.add<std::string>("propagatorOpposite", "PropagatorWithMaterialOpposite");
  desc.add("backwardFitInCMSSW", false)->setComment("Do backward fit (to innermost hit) in CMSSW (true) or mkFit (false)");

  descriptions.addWithDefaultLabel(desc);
}

void MkFitOutputConverter::produce(edm::StreamID iID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<MkFitInputWrapper> hhitsSeeds;
  iEvent.getByToken(hitsSeedsToken_, hhitsSeeds);
  const auto& hitsSeeds = *hhitsSeeds;

  edm::Handle<MkFitOutputWrapper> tracks;
  iEvent.getByToken(tracksToken_, tracks);

  edm::Handle<edm::View<TrajectorySeed>> seeds;
  iEvent.getByToken(seedToken_, seeds);

  edm::Handle<MeasurementTrackerEvent> mte;
  iEvent.getByToken(mteToken_, mte);

  edm::ESHandle<TrackerGeometry> trackerGeometry;
  iSetup.get<TrackerDigiGeometryRecord>().get(trackerGeometry);
  const auto& geom = *trackerGeometry;

  edm::ESHandle<Propagator> propagatorAlong;
  iSetup.get<TrackingComponentsRecord>().get(propagatorAlongName_, propagatorAlong);
  edm::ESHandle<Propagator> propagatorOpposite;
  iSetup.get<TrackingComponentsRecord>().get(propagatorOppositeName_, propagatorOpposite);

  edm::ESHandle<TrackerTopology> ttopo;
  iSetup.get<TrackerTopologyRcd>().get(ttopo);

  edm::ESHandle<MagneticField> mf;
  iSetup.get<IdealMagneticFieldRecord>().get(mf);

  edm::ESHandle<TransientTrackingRecHitBuilder> ttrhBuilder;
  iSetup.get<TransientRecHitRecord>().get(ttrhBuilderName_, ttrhBuilder);
  const auto *tkBuilder = dynamic_cast<TkTransientTrackingRecHitBuilder const *>(ttrhBuilder.product());
  if(!tkBuilder) {
    throw cms::Exception("LogicError") << "TTRHBuilder must be of type TkTransientTrackingRecHitBuilder";
  }
  TkClonerImpl hc = tkBuilder->cloner();

  auto detLayers = createDetLayers(hitsSeeds.layerNumberConverter(), *(mte->geometricSearchTracker()), *ttopo);

  // Convert mkfit presentation back to CMSSW
  auto cands = convertCandidates(*tracks, hitsSeeds.hitIndexMap(), *seeds, geom, *mf, *propagatorAlong, *propagatorOpposite, hc, detLayers, hitsSeeds.seeds());

  iEvent.put(std::move(cands));
  // For starters let's put empty collections
  iEvent.put(std::make_unique<std::vector<SeedStopInfo> >(seeds->size()));
}


std::vector<const DetLayer *> MkFitOutputConverter::createDetLayers(const mkfit::LayerNumberConverter& lnc,
                                                             const GeometricSearchTracker& tracker,
                                                             const TrackerTopology& ttopo) const {
  std::vector<const DetLayer *> dets(lnc.nLayers(), nullptr);

  auto isPlusSide = [&ttopo](const DetId& detid) {
    return ttopo.side(detid) == static_cast<unsigned>(TrackerDetSide::PosEndcap);
  };
  constexpr int isMono = 0;
  constexpr int isStereo = 1;
  for (const DetLayer* lay : tracker.allLayers()) {
    const auto& comp = lay->basicComponents();
    if (UNLIKELY(comp.empty())) {
      throw cms::Exception("LogicError") << "Got a tracker layer (subdet " << lay->subDetector()
                                         << ") with empty basicComponents.";
    }
    // First component is enough for layer and side information
    const auto& detId = comp.front()->geographicalId();
    const auto subdet = detId.subdetId();
    const auto layer = ttopo.layer(detId);

    // TODO: mono/stereo structure is still hardcoded for phase0/1 strip tracker
    dets[lnc.convertLayerNumber(subdet, layer, false, isMono, isPlusSide(detId))] = lay;
    if (((subdet == StripSubdetector::TIB or subdet == StripSubdetector::TOB) and (layer == 1 or layer == 2)) or
        subdet == StripSubdetector::TID or subdet == StripSubdetector::TEC) {
      dets[lnc.convertLayerNumber(subdet, layer, false, isStereo, isPlusSide(detId))] = lay;
    }
  }

  return dets;
}

std::unique_ptr<TrackCandidateCollection> MkFitOutputConverter::convertCandidates(const MkFitOutputWrapper& mkFitOutput,
                                                                                  const MkFitHitIndexMap& hitIndexMap,
                                                                                  const edm::View<TrajectorySeed>& seeds,
                                                                                  const TrackerGeometry& geom,
                                                                                  const MagneticField& mf,
                                                                                  const Propagator& propagatorAlong,
                                                                                  const Propagator& propagatorOpposite,
                                                                                  const TkClonerImpl& hitCloner,
                                                                                  const std::vector<const DetLayer *>& detLayers,
                                                                                  const mkfit::TrackVec& mkFitSeeds) const {
  auto output = std::make_unique<TrackCandidateCollection>();
  const auto& candidates = backwardFitInCMSSW_ ? mkFitOutput.candidateTracks() : mkFitOutput.fitTracks();
  output->reserve(candidates.size());

  LogTrace("MkFitOutputConverter") << "Number of candidates " << mkFitOutput.candidateTracks().size();

  int candIndex = -1;
  for(const auto& cand: candidates) {
    ++candIndex;
    LogTrace("MkFitOutputConverter") << "Candidate " << candIndex << " pT " << cand.pT() << " eta " << cand.momEta() << " phi " << cand.momPhi() << " chi2 " << cand.chi2();

    // hits
    edm::OwnVector<TrackingRecHit> recHits;
    // nTotalHits() gives sum of valid hits (nFoundHits()) and
    // invalid/missing hits (up to a maximum of 32 inside mkFit,
    // restriction to be lifted in the future)
    const int nhits = cand.nTotalHits();
    bool lastHitInvalid = false;
    for(int i=0; i<nhits; ++i) {
      const auto& hitOnTrack = cand.getHitOnTrack(i);
      LogTrace("MkFitOutputConverter") << " hit on layer " << hitOnTrack.layer << " index " << hitOnTrack.index;
      if (hitOnTrack.index < 0) {
        // See index-desc.txt file in mkFit for description of negative values
        //
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
        // In principle an InvalidTrackingRecHitNoDet could be
        // inserted here, but it seems that it is best to deal with
        // them in the TrackProducer.
        lastHitInvalid = true;
      } else {
        recHits.push_back(hitIndexMap.hitPtr(MkFitHitIndexMap::MkFitHit{hitOnTrack.index, hitOnTrack.layer})->clone());
        LogTrace("MkFitOutputConverter") << "  pos " << recHits.back().globalPosition().x() << " "
                                         << recHits.back().globalPosition().y() << " "
                                         << recHits.back().globalPosition().z() << " mag2 "
                                         << recHits.back().globalPosition().mag2() << " detid "
                                         << recHits.back().geographicalId().rawId() << " cluster "
                                         << hitIndexMap.clusterIndex(
                                                MkFitHitIndexMap::MkFitHit{hitOnTrack.index, hitOnTrack.layer});
        lastHitInvalid = false;
      }
    }

    const auto lastHitId = recHits.back().geographicalId();

    // MkFit hits are *not* in the order of propagation, sort by 3D radius for now (as we don't have loopers)
    // TODO: Improve the sorting (extract keys? maybe even bubble sort would work well as the hits are almost in the correct order)
    recHits.sort([&geom](const auto& a, const auto& b) {
        const auto aid = a.geographicalId();
        const auto bid = b.geographicalId();

        const auto asub = aid.subdetId();
        const auto bsub = bid.subdetId();
        if(asub != bsub) {
          // Subdetector order (BPix, FPix, TIB, TID, TOB, TEC) corresponds also the navigation
          return asub < bsub;
        }

        /*
        const auto *adet = geom.idToDet(aid);
        const auto *bdet = geom.idToDet(bid);

        const auto& apos = adet->position();
        const auto& bpos = bdet->position();
        */

        const auto& apos = a.globalPosition();
        const auto& bpos = b.globalPosition();

        if(isBarrel(asub)) {
          return apos.perp2() < bpos.perp2();
        }
        return std::abs(apos.z()) < std::abs(bpos.z());
      });

    const bool lastHitChanged = (recHits.back().geographicalId() != lastHitId); // TODO: make use of the bools

    // seed
    const auto seedIndex = cand.label();
    LogTrace("MkFitOutputConverter") << " from seed " << seedIndex << " seed hits";
    const auto& mkseed = mkFitSeeds.at(cand.label());
    for (int i = 0; i < mkseed.nTotalHits(); ++i) {
      const auto& hitOnTrack = mkseed.getHitOnTrack(i);
      LogTrace("MkFitOutputConverter") << "  hit on layer " << hitOnTrack.layer << " index " << hitOnTrack.index;
      // sanity check for now
      const auto& candHitOnTrack = cand.getHitOnTrack(i);
      if(hitOnTrack.layer != candHitOnTrack.layer) {
        //throw cms::Exception("LogicError")
        edm::LogError("MkFitOutputConverter") << "Candidate " << candIndex << " from seed " << seedIndex << " hit " << i
                                       << " has different layer in candidate (" << candHitOnTrack.layer << ") and seed (" << hitOnTrack.layer << ")."
                                       << " Hit indices are " << candHitOnTrack.index << " and " << hitOnTrack.index << ", respectively";
      }
      if(hitOnTrack.index != candHitOnTrack.index) {
        //throw cms::Exception("LogicError")
        edm::LogError("MkFitOutputConverter")
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
      edm::LogWarning("MkFitOutputConverter") << "Curvilinear error not pos-def\n" << fts.curvilinearError().matrix()
                                       << "\noriginal 6x6 covariance matrix\n" << cov
                                       << "\ncandidate ignored";
      continue;
    }

    auto tsosDet = backwardFitInCMSSW_ ? backwardFit(fts, recHits, propagatorAlong, propagatorOpposite, hitCloner, lastHitInvalid, lastHitChanged):
                                         convertInnermostState(fts, recHits, propagatorAlong, propagatorOpposite);
    if(!tsosDet.first.isValid()) {
      edm::LogWarning("MkFitOutputConverter") << "Backward fit of candidate " << candIndex << " failed, ignoring the candidate";
      continue;
    }
    
    // convert to persistent, from CkfTrackCandidateMakerBase
    auto pstate = trajectoryStateTransform::persistentState(tsosDet.first, tsosDet.second->geographicalId().rawId());

    output->emplace_back(
        recHits,
        seeds.at(seedIndex),
        pstate,
        seeds.refAt(seedIndex),
        0,                                               // mkFit does not produce loopers, so set nLoops=0
        static_cast<uint8_t>(StopReason::UNINITIALIZED)  // TODO: ignore details of stopping reason as well for now
    );
  }
  return output;
}

std::pair<TrajectoryStateOnSurface, const GeomDet *> MkFitOutputConverter::backwardFit(const FreeTrajectoryState& fts,
                                                                                const edm::OwnVector<TrackingRecHit>& hits,
                                                                                const Propagator& propagatorAlong,
                                                                                const Propagator& propagatorOpposite,
                                                                                const TkClonerImpl& hitCloner,
                                                                                bool lastHitWasInvalid,
                                                                                bool lastHitWasChanged) const {
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

  const Propagator *tryFirst = &propagatorAlong;
  const Propagator *trySecond = &propagatorOpposite;
  if(lastHitWasInvalid || lastHitWasChanged) {
    LogTrace("MkFitOutputConverter") << "Propagating first opposite, then along, because lastHitWasInvalid? " << lastHitWasInvalid
                              << " or lastHitWasChanged? " << lastHitWasChanged;
    std::swap(tryFirst, trySecond);
  }
  else {
    const auto lastHitSubdet = firstHits.front()->geographicalId().subdetId();
    const auto& surfacePos = lastHitSurface.position();
    const auto& lastHitPos = firstHits.front()->globalPosition();
    bool doSwitch = false;
    if(isBarrel(lastHitSubdet)) {
      doSwitch = (surfacePos.perp2() < lastHitPos.perp2());
    }
    else {
      doSwitch = (surfacePos.z() < lastHitPos.z());
    }
    if(doSwitch) {
      LogTrace("MkFitOutputConverter") << "Propagating first opposite, then along, because surface is inner than the hit; surface perp2 " << surfacePos.perp()
                                << " hit " << lastHitPos.perp2()
                                << " surface z " << surfacePos.z()
                                << " hit " << lastHitPos.z();

      std::swap(tryFirst, trySecond);
    }
  }

  /*
  const Propagator *propagator = &propagatorAlong;
  if(const auto *prop = dynamic_cast<const PropagatorWithMaterial *>(propagator)) {
    propagator = prop;
  }
  */
  auto tsosDouble = tryFirst->propagateWithPath(fts, lastHitSurface);
  if(!tsosDouble.first.isValid()) {
    LogDebug("MkFitOutputConverter") << "Propagating to startingState failed, trying in another direction next";
    tsosDouble = trySecond->propagateWithPath(fts, lastHitSurface);
  }
  auto& startingState = tsosDouble.first;

  if(!startingState.isValid()) {
    edm::LogWarning("MkFitOutputConverter") << "startingState is not valid, FTS was\n"
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

  // assume for now that the propagation in mkfit always alongMomentum
  PropagationDirection backFitDirection = oppositeToMomentum;

  // only direction matters in this context
  TrajectorySeed fakeSeed(PTrajectoryStateOnDet(), edm::OwnVector<TrackingRecHit>(), backFitDirection);

  // ignore loopers for now
  Trajectory fitres = backFitter.fitOne(fakeSeed, firstHits, startingState, TrajectoryFitter::standard);

  LogDebug("MkFitOutputConverter")
    <<"using a backward fit of :"<<firstHits.size()<<" hits, starting from:\n"<<startingState
    <<" to get the estimate of the initial state of the track.";

  if(!fitres.isValid()) {
    edm::LogWarning("MkFitOutputConverter") << "FitTester: first hits fit failed";
    return std::pair<TrajectoryStateOnSurface, const GeomDet*>();
  }

  TrajectoryMeasurement const & firstMeas = fitres.lastMeasurement();

  // magnetic field can be different!
  TrajectoryStateOnSurface firstState(firstMeas.updatedState().localParameters(),
                                      firstMeas.updatedState().localError(),
                                      firstMeas.updatedState().surface(),
                                      propagatorAlong.magneticField());

  firstState.rescaleError(100.);

  LogDebug("MkFitOutputConverter")
    <<"the initial state is found to be:\n:"<<firstState
    <<"\n it's field pointer is: "<<firstState.magneticField()
    <<"\n the pointer from the state of the back fit was: "<<firstMeas.updatedState().magneticField();


  return std::make_pair(firstState, firstMeas.recHit()->det());
}

std::pair<TrajectoryStateOnSurface, const GeomDet *> MkFitOutputConverter::convertInnermostState(const FreeTrajectoryState& fts,
                                                                                          const edm::OwnVector<TrackingRecHit>& hits,
                                                                                          const Propagator& propagatorAlong,
                                                                                          const Propagator& propagatorOpposite) const {
  auto det = hits[0].det();
  if(det == nullptr) {
    throw cms::Exception("LogicError") << "Got nullptr from the first hit det()";
  }

  const auto& firstHitSurface = det->surface();

  auto tsosDouble = propagatorAlong.propagateWithPath(fts, firstHitSurface);
  if(!tsosDouble.first.isValid()) {
    LogDebug("MkFitOutputConverter") << "Propagating to startingState along momentum failed, trying opposite next";
    tsosDouble = propagatorOpposite.propagateWithPath(fts, firstHitSurface);
  }

  return std::make_pair(tsosDouble.first, det);
}

DEFINE_FWK_MODULE(MkFitOutputConverter);
