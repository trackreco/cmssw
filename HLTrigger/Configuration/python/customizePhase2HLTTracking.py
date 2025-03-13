import FWCore.ParameterSet.Config as cms

def customizePhase2HLTMkFitCommon(process):

    process.HLTItLocalRecoSequence = cms.Sequence(process.hltSiPhase2Clusters+process.hltSiPhase2RecHits+process.hltSiPixelClusters+process.hltSiPixelClusterShapeCache+process.hltSiPixelRecHits)

    ### Attempt to use duplicate track merger in offline-like fashion
    from RecoTracker.FinalTrackSelectors.TrackCollectionMerger_cfi import TrackCollectionMerger
    process.hltPreGeneralTracks = TrackCollectionMerger.clone(
        trackProducers   = [
            "hltInitialStepTracks",
            "hltHighPtTripletStepTracks"
        ],
        inputClassifiers = [
            "hltInitialStepTrackCutClassifier",
            "hltHighPtTripletStepTrackCutClassifier"
        ],
        minQuality = "highPurity"
    )

    from RecoTracker.FinalTrackSelectors.DuplicateTrackMerger_cfi import DuplicateTrackMerger
    from RecoTracker.FinalTrackSelectors.DuplicateListMerger_cfi import DuplicateListMerger
    from TrackingTools.KalmanUpdators.Chi2MeasurementEstimator_cfi import Chi2MeasurementEstimator as _Chi2MeasurementEstimator
    process.duplicateTrackCandidatesChi2Est = _Chi2MeasurementEstimator.clone(
        ComponentName = "duplicateTrackCandidatesChi2Est",
        MaxChi2 = 100,
    )
    process.hltDuplicateTrackCandidates = DuplicateTrackMerger.clone(
        source = "hltPreGeneralTracks",
        useInnermostState  = True,
        ttrhBuilderName   = "WithTrackAngle",
        chi2EstimatorName = "duplicateTrackCandidatesChi2Est"
    )

    import RecoTracker.TrackProducer.TrackProducer_cfi
    process.hltMergedDuplicateTracks = RecoTracker.TrackProducer.TrackProducer_cfi.TrackProducer.clone(
        src = "hltDuplicateTrackCandidates:candidates",
        beamSpot = "hltOnlineBeamSpot",
        TTRHBuilder = "WithTrackAngle",
        MeasurementTrackerEvent = "hltMeasurementTrackerEvent",
        Fitter='RKFittingSmoother' # no outlier rejection!
    )

    from RecoTracker.FinalTrackSelectors.TrackCutClassifier_cff import TrackCutClassifier
    process.hltDuplicateTrackClassifier = TrackCutClassifier.clone(
        src='hltMergedDuplicateTracks',
        beamspot = cms.InputTag("hltOnlineBeamSpot"),
        vertices = cms.InputTag("hltPhase2PixelVertices"),
        mva = dict(
	    minPixelHits = [0,0,4],
	    maxChi2 = [9999.,9999.,9999.],
	    maxChi2n = [10.,1.0,0.4],  # [9999.,9999.,9999.]
	    minLayers = [0,0,4],
	    min3DLayers = [0,0,4],
	    maxLostLayers = [99,99,99])
    )

    process.hltGeneralTracks = DuplicateListMerger.clone(
        originalSource      = "hltPreGeneralTracks",
        originalMVAVals     = "hltPreGeneralTracks:MVAValues",
        mergedSource        = "hltMergedDuplicateTracks",
        mergedMVAVals       = "hltDuplicateTrackClassifier:MVAValues",
        candidateSource     = "hltDuplicateTrackCandidates:candidates",
        candidateComponents = "hltDuplicateTrackCandidates:candidateMap"
    )

    process.hltGeneralTracksTask = cms.Task(
        process.hltPreGeneralTracks,
        process.hltDuplicateTrackCandidates,
        process.hltMergedDuplicateTracks,
        process.hltDuplicateTrackClassifier,
        process.hltGeneralTracks
    )
    process.hltGeneralTracksSequence = cms.Sequence(process.hltGeneralTracksTask)
    process.HLTTrackingSequence.replace(process.hltGeneralTracks, process.hltGeneralTracksSequence)
    ###

    return process

def customizePhase2HLTMkFitInitialStepTracks(process):

    process.hltInitialStepSeeds.includeFourthHit = True

    process.mkFitSiPixelHits = cms.EDProducer("MkFitSiPixelHitConverter",
        hits = cms.InputTag("hltSiPixelRecHits"),
        mightGet = cms.optional.untracked.vstring,
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.mkFitSiStripHits = cms.EDProducer("MkFitSiStripHitConverter",
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
        ),
        rphiHits = cms.InputTag("siStripMatchedRecHits","rphiRecHit"),
        stereoHits = cms.InputTag("siStripMatchedRecHits","stereoRecHit"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.mkFitSiPhase2Hits = cms.EDProducer("MkFitPhase2HitConverter",
        mightGet = cms.optional.untracked.vstring,
        hits = cms.InputTag("hltSiPhase2RecHits"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.mkFitEventOfHits = cms.EDProducer("MkFitEventOfHitsProducer",
        beamSpot = cms.InputTag("offlineBeamSpot"),
        mightGet = cms.optional.untracked.vstring,
        pixelHits = cms.InputTag("mkFitSiPixelHits"),
        stripHits = cms.InputTag("mkFitSiPhase2Hits"),
        usePixelQualityDB = cms.bool(True),
        useStripStripQualityDB = cms.bool(False)
    )

    process.mkFitGeometryESProducer = cms.ESProducer("MkFitGeometryESProducer",
        appendToDataLabel = cms.string('')
    )

    process.hltInitialStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltInitialStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-initialStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.9)
    )

    process.hltInitialStepTrackCandidatesMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        mightGet = cms.optional.untracked.vstring,
        seeds = cms.InputTag("hltInitialStepSeeds"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.hltInitialStepTrackCandidatesMkFit = cms.EDProducer("MkFitProducer",
        backwardFitInCMSSW = cms.bool(False),
        buildingRoutine = cms.string('cloneEngine'),
        clustersToSkip = cms.InputTag(""),
        config = cms.ESInputTag("","hltInitialStepTrackCandidatesMkFitConfig"),
        eventOfHits = cms.InputTag("mkFitEventOfHits"),
        limitConcurrency = cms.untracked.bool(False),
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
        ),
        mkFitSilent = cms.untracked.bool(True),
        pixelHits = cms.InputTag("mkFitSiPixelHits"),
        removeDuplicates = cms.bool(True),
        seedCleaning = cms.bool(True),
        seeds = cms.InputTag("hltInitialStepTrackCandidatesMkFitSeeds"),
        stripHits = cms.InputTag("mkFitSiPhase2Hits")
    )

    process.hltInitialStepTrackCandidates = cms.EDProducer("MkFitOutputConverter",
        batchSize = cms.int32(16),
        candMVASel = cms.bool(False),
        candWP = cms.double(0),
        doErrorRescale = cms.bool(True),
        mightGet = cms.optional.untracked.vstring,
        mkFitEventOfHits = cms.InputTag("mkFitEventOfHits"),
        mkFitPixelHits = cms.InputTag("mkFitSiPixelHits"),
        mkFitSeeds = cms.InputTag("hltInitialStepTrackCandidatesMkFitSeeds"),
        mkFitStripHits = cms.InputTag("mkFitSiPhase2Hits"),
        propagatorAlong = cms.ESInputTag("","PropagatorWithMaterial"),
        propagatorOpposite = cms.ESInputTag("","PropagatorWithMaterialOpposite"),
        qualityMaxInvPt = cms.double(100),
        qualityMaxPosErr = cms.double(100),
        qualityMaxR = cms.double(120),
        qualityMaxZ = cms.double(280),
        qualityMinTheta = cms.double(0.01),
        qualitySignPt = cms.bool(True),
        seeds = cms.InputTag("hltInitialStepSeeds"),
        tfDnnLabel = cms.string('trackSelectionTf'),
        tracks = cms.InputTag("hltInitialStepTrackCandidatesMkFit"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.HLTInitialStepSequence= cms.Sequence(process.hltInitialStepSeeds+process.mkFitSiPixelHits+process.mkFitSiPhase2Hits+process.mkFitEventOfHits+process.hltInitialStepTrackCandidatesMkFitSeeds+process.hltInitialStepTrackCandidatesMkFit+process.hltInitialStepTrackCandidates+process.hltInitialStepTracks+process.hltInitialStepTrackCutClassifier+process.hltInitialStepTrackSelectionHighPurity)

    return process

def customizePhase2HLTMkFitHighPtTripletStepTracks(process):

    process.hltHighPtTripletStepTrackCandidatesMkFitConfig = cms.ESProducer("MkFitIterationConfigESProducer",
        ComponentName = cms.string('hltHighPtTripletStepTrackCandidatesMkFitConfig'),
        appendToDataLabel = cms.string(''),
        config = cms.FileInPath('RecoTracker/MkFit/data/mkfit-phase2-highPtTripletStep.json'),
        maxClusterSize = cms.uint32(8),
        minPt = cms.double(0.9)
    )

    process.hltHighPtTripletStepTrackCandidatesMkFitSeeds = cms.EDProducer("MkFitSeedConverter",
        maxNSeeds = cms.uint32(500000),
        mightGet = cms.optional.untracked.vstring,
        seeds = cms.InputTag("hltHighPtTripletStepSeeds"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.hltHighPtTripletStepTrackCandidatesMkFit = cms.EDProducer("MkFitProducer",
        backwardFitInCMSSW = cms.bool(False),
        buildingRoutine = cms.string('cloneEngine'),
        clustersToSkip = cms.InputTag("hltHighPtTripletStepClusters"),
        config = cms.ESInputTag("","hltHighPtTripletStepTrackCandidatesMkFitConfig"),
        eventOfHits = cms.InputTag("mkFitEventOfHits"),
        limitConcurrency = cms.untracked.bool(False),
        mightGet = cms.optional.untracked.vstring,
        minGoodStripCharge = cms.PSet(
            refToPSet_ = cms.string('SiStripClusterChargeCutLoose')
        ),
        mkFitSilent = cms.untracked.bool(True),
        pixelHits = cms.InputTag("mkFitSiPixelHits"),
        removeDuplicates = cms.bool(True),
        seedCleaning = cms.bool(True),
        seeds = cms.InputTag("hltHighPtTripletStepTrackCandidatesMkFitSeeds"),
        stripHits = cms.InputTag("mkFitSiPhase2Hits")
    )

    process.hltHighPtTripletStepTrackCandidates = cms.EDProducer("MkFitOutputConverter",
        batchSize = cms.int32(16),
        candCutSel = cms.bool(True),
        candMinPtCut = cms.double(0.9),
        candMinNHitsCut = cms.int32(4),
        candMVASel = cms.bool(False),
        candWP = cms.double(0),
        doErrorRescale = cms.bool(True),
        mightGet = cms.optional.untracked.vstring,
        mkFitEventOfHits = cms.InputTag("mkFitEventOfHits"),
        mkFitPixelHits = cms.InputTag("mkFitSiPixelHits"),
        mkFitSeeds = cms.InputTag("hltHighPtTripletStepTrackCandidatesMkFitSeeds"),
        mkFitStripHits = cms.InputTag("mkFitSiPhase2Hits"),
        propagatorAlong = cms.ESInputTag("","PropagatorWithMaterial"),
        propagatorOpposite = cms.ESInputTag("","PropagatorWithMaterialOpposite"),
        qualityMaxInvPt = cms.double(100),
        qualityMaxPosErr = cms.double(100),
        qualityMaxR = cms.double(120),
        qualityMaxZ = cms.double(280),
        qualityMinTheta = cms.double(0.01),
        qualitySignPt = cms.bool(True),
        seeds = cms.InputTag("hltHighPtTripletStepSeeds"),
        tfDnnLabel = cms.string('trackSelectionTf'),
        tracks = cms.InputTag("hltHighPtTripletStepTrackCandidatesMkFit"),
        ttrhBuilder = cms.ESInputTag("","WithTrackAngle")
    )

    process.HLTHitPtTripletStepSequenceMkFit = cms.Sequence(
        process.hltHighPtTripletStepTrackCandidatesMkFitSeeds+
        process.hltHighPtTripletStepTrackCandidatesMkFit+
        process.hltHighPtTripletStepTrackCandidates+
        process.hltHighPtTripletStepTracks+
        process.hltHighPtTripletStepTrackCutClassifier+
        process.hltHighPtTripletStepTrackSelectionHighPurity
    )
    process.HLTHighPtTripletStepSequence= cms.Sequence(process.HLTHighPtTripletStepSeedingSequence+process.HLTHitPtTripletStepSequenceMkFit)

    return process
