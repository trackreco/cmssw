import FWCore.ParameterSet.Config as cms

hltHighPtTripletStepTrackCandidatespLSTCLST = cms.EDProducer("CkfTrackCandidateMaker",
    MeasurementTrackerEvent = cms.InputTag("hltMeasurementTrackerEvent"),
    NavigationSchool = cms.string('SimpleNavigationSchool'),
    RedundantSeedCleaner = cms.string('CachingSeedCleanerBySharedInput'),
    TrajectoryBuilderPSet = cms.PSet(
        refToPSet_ = cms.string('highPtTripletStepTrajectoryBuilder')
    ),
    TrajectoryCleaner = cms.string('highPtTripletStepTrajectoryCleanerBySharedHits'),
    TransientInitialStateEstimatorParameters = cms.PSet(
        numberMeasurementsForFit = cms.int32(4),
        propagatorAlongTISE = cms.string('PropagatorWithMaterialParabolicMf'),
        propagatorOppositeTISE = cms.string('PropagatorWithMaterialParabolicMfOpposite')
    ),
    cleanTrajectoryAfterInOut = cms.bool(True),
    doSeedingRegionRebuilding = cms.bool(True),
    maxNSeeds = cms.uint32(100000),
    maxSeedsBeforeCleaning = cms.uint32(1000),
    numHitsForSeedCleaner = cms.int32(50),
    onlyPixelHitsForSeedCleaner = cms.bool(True),
    phase2clustersToSkip = cms.InputTag("hltHighPtTripletStepClusters"),
    reverseTrajectories = cms.bool(False),
    src = cms.InputTag("hltInitialStepTrackCandidates:pLSTSsLST"),
    useHitsSplitting = cms.bool(False)
)

_hltHighPtTripletStepTrackCandidatespLSTCLSTMkFit = cms.EDProducer("MkFitOutputConverter",
        batchSize = cms.int32(16),
        candCutSel = cms.bool(True),
        candMinNHitsCut = cms.int32(4),
        candMinPtCut = cms.double(0.9),
        candMVASel = cms.bool(False),
        candWP = cms.double(0),
        doErrorRescale = cms.bool(True),
        mightGet = cms.optional.untracked.vstring,
        mkFitEventOfHits = cms.InputTag("hltMkFitEventOfHits"),
        mkFitPixelHits = cms.InputTag("hltMkFitSiPixelHits"),
        mkFitSeeds = cms.InputTag("hltHighPtTripletStepMkFitSeeds"),
        mkFitStripHits = cms.InputTag("hltMkFitSiPhase2Hits"),
        propagatorAlong = cms.ESInputTag("","PropagatorWithMaterial"),
        propagatorOpposite = cms.ESInputTag("","PropagatorWithMaterialOpposite"),
        qualityMaxInvPt = cms.double(100),
        qualityMaxPosErr = cms.double(100),
        qualityMaxR = cms.double(120),
        qualityMaxZ = cms.double(280),
        qualityMinTheta = cms.double(0.01),
        qualitySignPt = cms.bool(True),
        seeds = cms.InputTag("hltInitialStepTrackCandidates:pLSTSsLST"),
        tfDnnLabel = cms.string('trackSelectionTf'),
        tracks = cms.InputTag("hltHighPtTripletStepTrackCandidatesMkFit"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
)

from Configuration.ProcessModifiers.hltTrackingMkFitHighPtTripletStep_cff import hltTrackingMkFitHighPtTripletStep
hltTrackingMkFitHighPtTripletStep.toReplaceWith(hltHighPtTripletStepTrackCandidatespLSTCLST,_hltHighPtTripletStepTrackCandidatespLSTCLSTMkFit)
