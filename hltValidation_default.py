import FWCore.ParameterSet.Config as cms
#from Validation.RecoTrack.HLTmultiTrackValidator_cff import hltMultiTrackValidation
#from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *

process = cms.Process("HLTRACKVALIDATOR")
process.load("Validation.RecoTrack.HLTmultiTrackValidator_cff")
process.load("SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load("Configuration.StandardSequences.RawToDigi_cff")
process.load("Configuration.EventContent.EventContent_cff")
process.load("Configuration.StandardSequences.Reconstruction_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load('Configuration.StandardSequences.EndOfProcess_cff')

# source
readFiles = cms.untracked.vstring()
source = cms.Source ("PoolSource",fileNames = readFiles)
readFiles.extend([
    'file:output_default.root',
])
process.source = source
process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

### conditions
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2024_realistic', '')

process.hltTrackValidator.cores = cms.InputTag("")
process.hltTrackValidator.label = ["hltPixelTracks", "hltIter0PFlowCtfWithMaterialTracks", "hltIter0PFlowTrackSelectionHighPurity", "hltDoubletRecoveryPFlowTrackSelectionHighPurity", "hltMergedTracks"]

process.hltTracksValidationTruth = cms.Sequence(
    process.hltTPClusterProducer
    + process.hltTrackAssociatorByHits
    + process.trackingParticleNumberOfLayersProducer
)
process.hltMultiTrackValidation = cms.Sequence(
    process.hltTracksValidationTruth
    + process.hltTrackValidator
)

# paths
process.validation = cms.Path(process.hltMultiTrackValidation)

# Output definition
process.load( "DQMServices.Core.DQMStore_cfi" )
process.dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string('file:DQMOutput_default.root'),
)

process.endjob_step = cms.EndPath(process.endOfProcess)
process.DQMoutput_step = cms.EndPath(process.dqmOutput)

process.schedule = cms.Schedule(
      process.validation,process.endjob_step,process.DQMoutput_step
)

process.options = cms.untracked.PSet(
    numberOfThreads = cms.untracked.uint32(16),
    numberOfStreams = cms.untracked.uint32(0),
    wantSummary = cms.untracked.bool(True)
)
>
