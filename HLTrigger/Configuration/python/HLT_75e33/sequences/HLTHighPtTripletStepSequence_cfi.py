import FWCore.ParameterSet.Config as cms

from ..modules.hltHighPtTripletStepTrackCandidates_cfi import *
from ..modules.hltHighPtTripletStepTrackCutClassifier_cfi import *
from ..modules.hltHighPtTripletStepTracks_cfi import *
from ..modules.hltHighPtTripletStepTrackSelectionHighPurity_cfi import *
from ..sequences.HLTHighPtTripletStepSeedingSequence_cfi import *

HLTHighPtTripletStepSequence = cms.Sequence(
     HLTHighPtTripletStepSeedingSequence
    +hltHighPtTripletStepTrackCandidates
    +hltHighPtTripletStepTracks
    +hltHighPtTripletStepTrackCutClassifier
    +hltHighPtTripletStepTrackSelectionHighPurity
)

from Configuration.ProcessModifiers.trackingLST_cff import trackingLST
trackingLST.toReplaceWith(HLTHighPtTripletStepSequence, HLTHighPtTripletStepSequence.copyAndExclude([HLTHighPtTripletStepSeedingSequence]))

from ..modules.hltHighPtTripletStepTrackCandidatespLSTCLST_cfi import *
from ..modules.hltHighPtTripletStepTrackspLSTCLST_cfi import *
from ..modules.hltHighPtTripletStepTrackCutClassifierpLSTCLST_cfi import *
from ..modules.hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST_cfi import *
_HLTHighPtTripletStepSequenceLSTSeeding = cms.Sequence(
     hltHighPtTripletStepTrackCandidatespLSTCLST
    +hltHighPtTripletStepTrackspLSTCLST
    +hltHighPtTripletStepTrackCutClassifierpLSTCLST
    +hltHighPtTripletStepTrackSelectionHighPuritypLSTCLST
)

from Configuration.ProcessModifiers.seedingLST_cff import seedingLST
(seedingLST & trackingLST).toReplaceWith(HLTHighPtTripletStepSequence, _HLTHighPtTripletStepSequenceLSTSeeding)

from ..modules.hltMkFitSiPixelHits_cfi import *
from ..modules.hltMkFitSiStripHits_cfi import *
from ..modules.hltMkFitSiPhase2Hits_cfi import *
from ..modules.hltMkFitEventOfHits_cfi import *
from ..modules.hltHighPtTripletStepTrackCandidatesMkFitSeeds_cfi import *
from ..modules.hltHighPtTripletStepTrackCandidatesMkFit_cfi import *

_HLTHighPtTripletStepSequenceMkFit = cms.Sequence(
    hltMkFitSiPixelHits
    +hltMkFitSiPhase2Hits
    +hltMkFitEventOfHits
    +hltHighPtTripletStepTrackCandidatesMkFitSeeds
    +hltHighPtTripletStepTrackCandidatesMkFit
    +hltHighPtTripletStepTrackCandidates
    +hltHighPtTripletStepTracks
    +hltHighPtTripletStepTrackCutClassifier
    +hltHighPtTripletStepTrackSelectionHighPurity
)
from Configuration.ProcessModifiers.trackingMkFitHighPtTripletStep_cff import hltTrackingMkFitHighPtTripletStep
(hltTrackingMkFitHighPtTripletStep & trackingLST).toReplaceWith(HLTHighPtTripletStepSequence,_HLTHighPtTripletStepSequenceMkFit)
