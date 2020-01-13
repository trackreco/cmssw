
import FWCore.ParameterSet.Config as cms

from Validation.RecoTrack.HLTmultiTrackValidator_cff import *

doPixel = True
doTrack02 = True

hltTrackValidator.label = []
if doPixel:
	hltTrackValidator.label.extend([
    		"hltPixelTracks",
    		"hltPixelTracksFromTriplets",
    		"hltPixelTracksMerged"
	])
if doTrack02:
	hltTrackValidator.label.extend([
                "hltIter0PFlowCtfWithMaterialTracks",
    		"hltIter0PFlowTrackSelectionHighPurity",
#    		"hltIter1PFlowTrackSelectionHighPurity",
#    		"hltIter1Merged",
#     		"hltIter2PFlowTrackSelectionHighPurity",
#    		"hltIter2Merged",
#    		"hltTripletRecoveryPFlowTrackSelectionHighPurity",
#    		"hltTripletRecoveryMerged",
#  		"hltDoubletRecoveryPFlowTrackSelectionHighPurity",
#    		"hltMergedTracks"
	])
hltTrackValidatorOnlineCuts = hltTrackValidator.clone(
        dirName = "HLT/Tracking/ValidationWRTtpOnlineCuts/"
)

# reset to offline cuts
hltTrackValidator.ptMinTP = multiTrackValidator.ptMinTP.value()
hltTrackValidator.lipTP = multiTrackValidator.lipTP.value()
hltTrackValidator.tipTP = multiTrackValidator.tipTP.value()
hltTrackValidator.histoProducerAlgoBlock = multiTrackValidator.histoProducerAlgoBlock.clone()

from SimGeneral.TrackingAnalysis.trackingParticleNumberOfLayersProducer_cfi import *
#hltTracksValidationTruth = cms.Sequence(hltTPClusterProducer+hltTrackAssociatorByHits+trackingParticleRecoTrackAsssociation+VertexAssociatorByPositionAndTracks+trackingParticleNumberOfLayersProducer)
hltTracksValidationTruth = cms.Sequence(hltTPClusterProducer+hltTrackAssociatorByHits+trackingParticleNumberOfLayersProducer)


hltMultiTrackValidation = cms.Sequence(
    hltTracksValidationTruth
    + hltTrackValidator
    + hltTrackValidatorOnlineCuts
)
#from Validation.RecoTrack.associators_cff import *


from SimTracker.VertexAssociation.VertexAssociatorByPositionAndTracks_cfi import *
hltVertexAssociatorByPositionAndTracks = VertexAssociatorByPositionAndTracks.clone()
hltVertexAssociatorByPositionAndTracks.trackAssociation = "tpToHLTpixelTrackAssociation"

from Validation.RecoVertex.HLTmultiPVvalidator_cff import *
hltMultiPVanalysis.verbose = False
#hltMultiPVanalysis.trackAssociatorMap = "tpToHLTpixelTrackAssociation"
#hltMultiPVanalysis.vertexAssociator   = "vertexAssociatorByPositionAndTracks4pixelTracks"
#tpToHLTpixelTrackAssociation.ignoremissingtrackcollection = False
hltPixelPVanalysis.trackAssociatorMap = "tpToHLTpixelTrackAssociation"
hltPixelPVanalysis.vertexAssociator = "vertexAssociatorByPositionAndTracks4pixelTracks" 

tpToHLTpixelTrackAssociation.label_tr = "hltPixelTracksMerged"

validation = cms.EndPath(
    hltMultiTrackValidation
#    + hltTrackAssociatorByHits
#    + tpToHLTpixelTrackAssociation
 #    + hltVertexAssociatorByPositionAndTracks
#    + hltMultiPVanalysis
#    + hltMultiPVValidation
)


dqmOutput = cms.OutputModule("DQMRootOutputModule",
    fileName = cms.untracked.string("DQMIO.root")
)
DQMOutput = cms.EndPath( dqmOutput )

