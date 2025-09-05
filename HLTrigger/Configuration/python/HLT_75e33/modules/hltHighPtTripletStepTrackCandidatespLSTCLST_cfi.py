import FWCore.ParameterSet.Config as cms

from ..modules.hltHighPtTripletStepTrackCandidates_cfi import hltHighPtTripletStepTrackCandidates as _hltHighPtTripletStepTrackCandidates
hltHighPtTripletStepTrackCandidatespLSTCLST = _hltHighPtTripletStepTrackCandidates.clone()
from Configuration.ProcessModifiers.hltTrackingMkFitHighPtTripletStep_cff import hltTrackingMkFitHighPtTripletStep
(~hltTrackingMkFitHighPtTripletStep).toModify(hltHighPtTripletStepTrackCandidatespLSTCLST, src = "hltInitialStepTrackCandidates:pLSTSsLST")

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

hltTrackingMkFitHighPtTripletStep.toReplaceWith(hltHighPtTripletStepTrackCandidatespLSTCLST,_hltHighPtTripletStepTrackCandidatespLSTCLSTMkFit)
