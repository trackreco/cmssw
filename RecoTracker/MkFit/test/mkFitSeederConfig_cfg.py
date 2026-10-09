import FWCore.ParameterSet.Config as cms

# Makes the mkFit seeder's configuration (mkfit::SeederConfig) with MkFitSeederConfigESProducer on one empty
# event: the default JSON of customizeInitialStepMkFitSeeder is found through FileInPath and read. The geometry
# and the global tag are the release's phase-2 default; the seeder configuration does not depend on them.

import Configuration.Geometry.defaultPhase2ConditionsEra_cff as _settings
_PH2_GLOBAL_TAG, _PH2_ERA = _settings.get_era_and_conditions(_settings.DEFAULT_VERSION)

process = cms.Process('SEEDERCONFIG', _PH2_ERA)

process.load('Configuration.StandardSequences.Services_cff')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.Geometry.GeometryExtended%sReco_cff' % _settings.DEFAULT_VERSION)
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, _PH2_GLOBAL_TAG, '')

process.source = cms.Source("EmptySource")
process.maxEvents.input = 1

from RecoTracker.MkFit.mkFitSeederConfigESProducer_cfi import mkFitSeederConfigESProducer as _config
process.mkFitSeederConfig = _config.clone(
    ComponentName = 'mkFitSeederConfig',
    config = 'RecoTracker/MkFit/data/mkfit-phase2-seeder.json',
)

process.get = cms.EDAnalyzer("EventSetupRecordDataGetter",
    toGet = cms.VPSet(cms.PSet(
        record = cms.string('TrackerRecoGeometryRecord'),
        data = cms.vstring('mkfit::SeederConfig/mkFitSeederConfig'),
    )),
    verbose = cms.untracked.bool(True),
)
process.p = cms.Path(process.get)
