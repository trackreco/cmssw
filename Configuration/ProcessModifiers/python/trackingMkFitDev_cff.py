import FWCore.ParameterSet.Config as cms

from Configuration.ProcessModifiers.trackingMkFitCommon_cff import *
from Configuration.ProcessModifiers.trackingMkFitLowPtQuadStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitLowPtTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitDetachedTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitPixelPairStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitMixedTripletStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitPixelLessStep_cff import *
from Configuration.ProcessModifiers.trackingMkFitTobTecStep_cff import *

# Use mkFit in selected iterations
trackingMkFitDev = cms.ModifierChain(
    trackingMkFitCommon,
#    trackingMkFitLowPtQuadStep,       # to be enabled later
#    trackingMkFitLowPtTripletStep,    # to be enabled later
#    trackingMkFitDetachedTripletStep, # to be enabled later
#    trackingMkFitPixelPairStep,       # to be enabled later
#    trackingMkFitMixedTripletStep,    # to be enabled later
#    trackingMkFitPixelLessStep,       # to be enabled later
#    trackingMkFitTobTecStep,          # to be enabled later
)
