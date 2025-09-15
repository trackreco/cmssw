import FWCore.ParameterSet.Config as cms

# MkFitSeedConverter options
hltHighPtTripletStepMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        mightGet = cms.optional.untracked.vstring,
        seeds = cms.InputTag("hltHighPtTripletStepSeeds"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)

from Configuration.ProcessModifiers.hltTrackingMkFitHighPtTripletStep_cff import hltTrackingMkFitHighPtTripletStep
from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
(seedingLST & trackingLST & hltTrackingMkFitHighPtTripletStep).toModify(hltHighPtTripletStepMkFitSeeds,seeds = cms.InputTag("hltInitialStepTrackCandidates:pLSTSsLST"))
