import FWCore.ParameterSet.Config as cms

# mkFit seeder: mkFit seeds from all pixel and outer-tracker hits (phase2MkSeeder)
hltMkFitSeederSeedsMkFit = cms.EDProducer("MkFitSeederProducer",
    pixelHits = cms.InputTag("hltMkFitSiPixelHits"),
    stripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
    beamSpot = cms.InputTag("hltOnlineBeamSpot"),
    config = cms.ESInputTag("", "hltMkFitSeederConfig"),
    algorithm = cms.string("initialStep"),
    maxNSeeds = cms.uint32(500000),
    # also the same seeds with the state of the inward fit ('beamline'), for the pixel tracks and vertices
    beamlineSeeds = cms.bool(True),
    mightGet = cms.optional.untracked.vstring
)
