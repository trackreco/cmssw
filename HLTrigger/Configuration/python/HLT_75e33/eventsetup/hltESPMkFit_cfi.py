import FWCore.ParameterSet.Config as cms

def _addProcessMkFitGeometry(process):
    process.mkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
        appendToDataLabel = cms.string('')
    )

from Configuration.ProcessModifiers.trackingMkFitCommon_cff import trackingMkFitCommon
modifyConfigurationForTrackingMkFitGeometryMkfit_ = trackingMkFitCommon.makeProcessModifier(_addProcessMkFitGeometry)

def _addProcesshltInitialStepMkFitConfig(process):
    process.hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-initialStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.9)
    )
def _addProcesshltHighPtTripletStepMkFitConfig(process):
    process.hltHighPtTripletStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltHighPtTripletStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-highPtTripletStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.9)
    )
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
from Configuration.ProcessModifiers.hltTrackingMkFitInitialStep_cff import hltTrackingMkFitInitialStep
from Configuration.ProcessModifiers.hltTrackingMkFitHighPtTripletStep_cff import hltTrackingMkFitHighPtTripletStep
modifyConfigurationForTrackingMkFithltInitialStepMkFitConfig_ = hltTrackingMkFitInitialStep.makeProcessModifier(_addProcesshltInitialStepMkFitConfig)
modifyConfigurationForTrackingMkFithltHighPtTripletStepMkFitConfig_ = hltTrackingMkFitHighPtTripletStep.makeProcessModifier(_addProcesshltHighPtTripletStepMkFitConfig)


