import FWCore.ParameterSet.Config as cms
import FWCore.Utilities.FileUtils as FileUtils

process = cms.Process('HARVESTING')

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_AutoFromDBCurrent_cff')
process.load('Configuration.StandardSequences.EDMtoMEAtRunEnd_cff')
process.load('Configuration.StandardSequences.Harvesting_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.configurationMetadata = cms.untracked.PSet(
    version = cms.untracked.string(': 1.1 $'),
    annotation = cms.untracked.string('harvest nevts:100'),
    name = cms.untracked.string('PyReleaseValidation')
)
process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)
process.options = cms.untracked.PSet(
    Rethrow = cms.untracked.vstring('ProductNotFound'),
    fileMode = cms.untracked.string('FULLMERGE')
)

# Input source
process.source = cms.Source("DQMRootSource",
    fileNames = cms.untracked.vstring('file:dummyfile.root')
)

# Other statements
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:run2_mc_50ns', '')

# Path and EndPath definitions
process.edmtome_step = cms.Path(process.EDMtoME)
process.validationpreprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_preprod)
process.validationprodHarvesting = cms.Path(process.postValidation*process.hltpostvalidation_prod)
process.dqmHarvesting = cms.Path(process.DQMOffline_SecondStep*process.DQMOffline_Certification)
process.validationHarvesting = cms.Path(process.postValidation*process.hltpostvalidation)
#process.validationHarvestingFS = cms.Path(process.HarvestingFastSim)
process.dqmHarvestingPOG = cms.Path(process.DQMOffline_SecondStep_PrePOG)
process.dqmsave_step = cms.Path(process.DQMSaver)

# Schedule definition                                                                                                                                        
process.schedule = cms.Schedule(process.edmtome_step,process.dqmHarvesting,process.dqmsave_step) 
 #-----------------------------------------------------------------------------------                                                                         
# Mark's changes start (everything above this point is the output from cmsDriver)                                                                            
#                                                                                                                                                            

# For some reason a seed harvester isn't included in the standard sequences. If this next processor isn't                                                    
# run then things like efficiencies are just added together instead of recalculated.                                                                         
process.dqmSaver.saveAtJobEnd = cms.untracked.bool(True)
#process.dqmSaver.workflow = "/G4e/RelVal/Validation"
#process.dqmSaver.forceRunNumber = cms.untracked.int32(1)

import FWCore.ParameterSet.Config as cms
from Validation.RecoVertex.HLTpostProcessorVertex_cfi import *

process.load('Validation.RecoTrack.HLTpostProcessorTracker_cfi')
process.postProcessorHLTtrackingSequence = cms.Sequence(process.postProcessorHLTtracking+process.postProcessorHLTtrackingSummary)

process.postProcessorHLTtracking.subDirs.append("HLT/Tracking/ValidationWRTtpOnlineCuts/*")
process.postProcessorHLTtrackingSummary.subDirs.append("HLT/Tracking/ValidationWRTtpOnlineCuts")

from DQM.TrackingMonitorClient.TrackingEffFromHitPatternClientConfig_cff import *
process.trackingEffFromHitPatternHLT = trackingEffFromHitPattern.clone()
process.trackingEffFromHitPatternHLT.subDirs = cms.untracked.vstring(
    "HLT/Tracking/pixelTracks/HitEffFromHitPattern",
    "HLT/Tracking/iter0/HitEffFromHitPattern",
    "HLT/Tracking/iter0HP/HitEffFromHitPattern",
    "HLT/Tracking/iter1/HitEffFromHitPattern",
    "HLT/Tracking/iter1HP/HitEffFromHitPattern",
    "HLT/Tracking/iter2/HitEffFromHitPattern",
    "HLT/Tracking/iter2HP/HitEffFromHitPattern",
    "HLT/Tracking/iter2Merged/HitEffFromHitPattern",
    "HLT/Tracking/iter3HP/HitEffFromHitPattern",
    "HLT/Tracking/iter3Merged/HitEffFromHitPattern",
    "HLT/Tracking/iter4/HitEffFromHitPattern",
    "HLT/Tracking/iter4HP/HitEffFromHitPattern",
    "HLT/Tracking/iter4Merged/HitEffFromHitPattern"
)
# Remove the HLT harvesting from the validation harvesting step                                                                                              
process.validationHarvesting = cms.Path(process.postValidation)
process.trackingOnlyHarvesting = cms.Path(process.postProcessorHLTtrackingSequence)
process.trackingEffFromHitHarvesting = cms.Path(process.trackingEffFromHitPatternHLT)
process.vertexingHarvesting = cms.Path(process.postProcessorHLTvertexing)
process.schedule = cms.Schedule(process.edmtome_step,
                                process.trackingOnlyHarvesting,
                                process.trackingEffFromHitHarvesting,
                                process.vertexingHarvesting,
                                process.dqmsave_step)

files = cms.untracked.vstring() 
files = [ 
'file:DQMIO.root'
] 
process.source.fileNames = files 
