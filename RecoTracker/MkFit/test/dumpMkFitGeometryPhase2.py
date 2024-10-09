import FWCore.ParameterSet.Config as cms

# from Configuration.ProcessModifiers.trackingMkFit_cff import trackingMkFit
from Configuration.ProcessModifiers.trackingMkFitCommon_cff import trackingMkFitCommon
trackingMkFit = cms.ModifierChain(trackingMkFitCommon)

# No era in Fireworks/Geom reco dumper
process = cms.Process('DUMP',Phase2C17I13M9,trackingMkFit)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D110Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Reconstruction_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, "auto:phase2_realistic_T33", '')

# In Fireworks/Geom reco dumper:
# from Configuration.AlCa.autoCond import autoCond
# process.GlobalTag.globaltag = autoCond['phase2_realistic']

process.MessageLogger.cerr.threshold = "INFO"
process.MessageLogger.cerr.MkFitGeometryESProducer = dict(limit=-1)

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1


process.add_(cms.ESProducer("MkFitGeometryESProducer"))

defaultOutputFileName="phase2-trackerinfo.bin"

# level: 0 - no printout; 1 - print layers, 2 - print shapes and modules
# outputFileName: binary dump file; no dump if empty string
process.dump = cms.EDAnalyzer("DumpMkFitGeometry",
                              level = cms.untracked.int32(1),
                              outputFileName = cms.untracked.string(defaultOutputFileName)
                              )

print("Requesting MkFit geometry dump into file:", defaultOutputFileName, "\n");
process.p = cms.Path(process.dump)
