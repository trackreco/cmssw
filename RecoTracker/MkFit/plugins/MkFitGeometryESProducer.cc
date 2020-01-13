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

#include <atomic>

class MkFitGeometryESProducer : public edm::ESProducer {
public:
  MkFitGeometryESProducer(const edm::ParameterSet& iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<MkFitGeometry> produce(const TrackerRecoGeometryRecord& iRecord);
};

MkFitGeometryESProducer::MkFitGeometryESProducer(const edm::ParameterSet& iConfig) {
  setWhatProduced(this);
}

void MkFitGeometryESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<MkFitGeometry> MkFitGeometryESProducer::produce(const TrackerRecoGeometryRecord& iRecord) {
  edm::ESHandle<TrackerGeometry> geom;
  iRecord.getRecord<TrackerDigiGeometryRecord>().get(geom);

  edm::ESHandle<GeometricSearchTracker> tracker;
  iRecord.get(tracker);

  edm::ESHandle<TrackerTopology> ttopo;
  iRecord.getRecord<TrackerTopologyRcd>().get(ttopo);

  auto ret = std::make_unique<MkFitGeometry>(*geom, *tracker, *ttopo);

  return ret;
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitGeometryESProducer);
