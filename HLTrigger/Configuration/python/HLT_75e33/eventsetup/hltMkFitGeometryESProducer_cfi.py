import FWCore.ParameterSet.Config as cms

def _addProcessMkFitGeometry(process):
    process.mkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
        appendToDataLabel = cms.string('')
    )

from Configuration.ProcessModifiers.trackingMkFitCommon_cff import trackingMkFitCommon
modifyConfigurationForTrackingMkFitGeometryMkfit_ = trackingMkFitCommon.makeProcessModifier(_addProcessMkFitGeometry)
