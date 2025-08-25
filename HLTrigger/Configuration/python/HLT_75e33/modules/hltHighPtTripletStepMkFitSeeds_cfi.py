import FWCore.ParameterSet.Config as cms

# MkFitSeedConverter options
hltHighPtTripletStepMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        mightGet = cms.optional.untracked.vstring,
        seeds = cms.InputTag("hltHighPtTripletStepSeeds"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)
