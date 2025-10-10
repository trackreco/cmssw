import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run3_2024_cff import Run3_2024
from Configuration.ProcessModifiers.egamma_lowPt_exclusive_cff import egamma_lowPt_exclusive
from Configuration.Eras.Modifier_highBetaStar_cff import highBetaStar
from Configuration.Eras.Modifier_dedx_lfit_cff import dedx_lfit
from Configuration.Eras.Modifier_run3_upc_cff import run3_upc
from Configuration.Eras.Modifier_run3_upc_2024_cff import run3_upc_2024

Run3_2024_UPC = cms.ModifierChain(Run3_2024, egamma_lowPt_exclusive, highBetaStar, dedx_lfit, run3_upc, run3_upc_2024)
