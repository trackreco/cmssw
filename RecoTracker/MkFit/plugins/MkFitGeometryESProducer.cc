#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

// mkFit includes
#include "ConfigWrapper.h"
// TODO: the need for this define MUST be removed
// e.g. make it a build configure time choice instead of build time shoice
#define USE_MATRIPLEX
// TODO: this include causes "warning: ignoring #pragma omp simd" from Matriplex.h, can we try to hide the Matriplex from CMSSW?
#include "mkFit/SteeringParams.h"
#include "TrackerInfo.h"

#include <atomic>

class MkFitGeometryESProducer : public edm::ESProducer {
public:
  MkFitGeometryESProducer(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<MkFitGeometry> produce(const TrackerRecoGeometryRecord& iRecord);

private:
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  edm::ESGetToken<GeometricSearchTracker, TrackerRecoGeometryRecord> trackerToken_;
  edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> ttopoToken_;
};

MkFitGeometryESProducer::MkFitGeometryESProducer(const edm::ParameterSet& iConfig) {
  auto cc = setWhatProduced(this);
  geomToken_ = cc.consumes();
  trackerToken_ = cc.consumes();
  ttopoToken_ = cc.consumes();
}

void MkFitGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<MkFitGeometry> MkFitGeometryESProducer::produce(const TrackerRecoGeometryRecord& iRecord) {
  auto trackerInfo = std::make_unique<mkfit::TrackerInfo>();
  auto iterationsInfo = std::make_unique<mkfit::IterationsInfo>();
  // TODO: absorb the functionality to CMSSW
  mkfit::TrackerInfo::ExecTrackerInfoCreatorPlugin("CMS-2017", *trackerInfo, *iterationsInfo);
  return std::make_unique<MkFitGeometry>(iRecord.get(geomToken_),
                                         iRecord.get(trackerToken_),
                                         iRecord.get(ttopoToken_),
                                         std::move(trackerInfo),
                                         std::move(iterationsInfo));
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitGeometryESProducer);
