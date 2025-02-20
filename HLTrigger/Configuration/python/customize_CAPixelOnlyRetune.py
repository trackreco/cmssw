import FWCore.ParameterSet.Config as cms

# same eff
def customize_CAPixelOnlyRetuneSameEff(process):
    # Target the specific module type
    target_module_name = "hltPixelTracksSoA"
    # Check if the target module exists in the process
    if hasattr(process, target_module_name):
        module = getattr(process, target_module_name)
        # Update parameters directly

    module.phiCuts = cms.vint32([965, 1241, 395, 698, 1058, 1211, 348, 782, 1016, 810, 463, 755, 694, 531, 770, 471, 592, 750, 348])
    module.dcaCutInnerTriplet = cms.double(0.09181130994905196)
    module.dcaCutOuterTriplet = cms.double(0.4207246178345847)
    module.CAThetaCutBarrel = cms.double(0.001233027054994468)
    module.CAThetaCutForward = cms.double(0.003556913217741844)
    module.hardCurvCut = cms.double(0.5031696900017477)
    return process

def customize_CAPixelOnlyRetuneLowerEff(process):
    # Target the specific module type
    target_module_name = "hltPixelTracksSoA"
    # Check if the target module exists in the process
    if hasattr(process, target_module_name):
        module = getattr(process, target_module_name)
        # Update parameters directly

    module.phiCuts = cms.vint32([617, 767, 579, 496, 900, 1252, 435, 832, 1051, 913, 515, 604, 763, 706, 678, 560, 597, 574, 532])
    module.dcaCutInnerTriplet = cms.double(0.07268965383396808)
    module.dcaCutOuterTriplet = cms.double(0.35106213112457163)
    module.CAThetaCutBarrel = cms.double(0.001033994253338825)
    module.CAThetaCutForward = cms.double(0.003640941685013238)
    module.hardCurvCut = cms.double(0.6592029738506096)
    return process
