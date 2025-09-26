import FWCore.ParameterSet.Config as cms

# hltlStepTrackCandidatesMkFit options

hltHighPtTripletStepTrackCandidatesMkFit = cms.EDProducer("MkFitProducer",
        backwardFitInCMSSW = cms.bool(False),
        buildingRoutine = cms.string('cloneEngine'),
        clustersToSkip = cms.InputTag("hltHighPtTripletStepClusters"),
        config = cms.ESInputTag("","hltHighPtTripletStepTrackCandidatesMkFitConfig"),
        eventOfHits = cms.InputTag("hltMkFitEventOfHits"),
        limitConcurrency = cms.untracked.bool(False),
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(),
        mkFitSilent = cms.untracked.bool(True),
        pixelHits = cms.InputTag("hltMkFitSiPixelHits"),
        removeDuplicates = cms.bool(True),
        seedCleaning = cms.bool(True),
        seeds = cms.InputTag("hltHighPtTripletStepMkFitSeeds"),
        stripHits = cms.InputTag("hltMkFitSiPhase2Hits")
)
