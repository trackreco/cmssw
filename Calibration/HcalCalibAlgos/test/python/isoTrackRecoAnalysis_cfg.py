import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Run2_2017_cff import Run2_2017
process = cms.Process("ANALYSIS",Run2_2017)
#process = cms.Process("ANALYSIS")

process.load("FWCore.MessageService.MessageLogger_cfi")
process.load('Configuration.StandardSequences.Services_cff')
process.load("Configuration.StandardSequences.GeometryRecoDB_cff")
process.load("Configuration.StandardSequences.MagneticField_cff")
process.load("RecoLocalCalo.EcalRecAlgos.EcalSeverityLevelESProducer_cfi")
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

from Configuration.AlCa.autoCond import autoCond
#process.GlobalTag.globaltag='101X_dataRun2_Prompt_v10'
process.GlobalTag.globaltag='106X_dataRun2_v20'
#106X_mcRun3_2021_realistic_v3
if 'MessageLogger' in process.__dict__:
    process.MessageLogger.HcalIsoTrack=dict()

process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.options = cms.untracked.PSet( wantSummary = cms.untracked.bool(True) )

process.load('RecoLocalCalo.CaloTowersCreator.calotowermaker_cfi')
process.towerMakerAll = process.calotowermaker.clone()
process.towerMakerAll.hbheInput = cms.InputTag("hbhereco")
process.towerMakerAll.hoInput = cms.InputTag("none")
process.towerMakerAll.hfInput = cms.InputTag("none")
process.towerMakerAll.ecalInputs = cms.VInputTag(cms.InputTag("ecalRecHit","EcalRecHitsEB"), cms.InputTag("ecalRecHit","EcalRecHitsEE"))
process.towerMakerAll.AllowMissingInputs = True

process.load('Calibration.HcalCalibAlgos.HcalIsoTrkAnalyzer_cff')
process.source = cms.Source("PoolSource", 
                            fileNames = cms.untracked.vstring(
        #'/store/data/Run2018B/JetHT/ALCARECO/HcalCalIsoTrkFilter-PromptReco-v1/000/317/696/00000/D60EC93B-9870-E811-BAF3-FA163E8DA20D.root',
#        '/store/mc/Run3Summer19DR/DoublePion_E-50/GEN-SIM-RECO/2021ScenarioNZSRECONoPU_106X_mcRun3_2021_realistic_v3-v2/270000/22481A14-0F65-E046-809A-C03709C76325.root'                        
#       'file:/afs/cern.ch/work/s/sdey/public/forsunandada/C2F61205-A366-E711-9AFA-02163E01A2B0.root',
#        '/store/mc/Run3Summer19DRPremix/QCD_Pt-15to7000_TuneCP5_Flat_13TeV_pythia8/GEN-SIM-RECO/2021ScenarioNZSRECO_106X_mcRun3_2021_realistic_v3-v2/210000/0049D816-1F5D-C649-9D36-12A55EF44FE6.root'
#        '/store/mc/Run3Summer19DRPremix/QCD_Pt-15_IsoTrkFilter_Pt-30_TuneCP5_14TeV-pythia8/GEN-SIM-RECO/2021ScenarioRECO_106X_mcRun3_2021_realistic_v3-v2/30000/9CF8F3E3-B8C7-F84A-BAF2-17AD46769E1E.root'
                                'root://xrootd.ba.infn.it///store/data/Run2017E/JetHT/ALCARECO/HcalCalIsoTrkFilter-09Aug2019_UL2017-v1/50000/FF19B3B8-39D3-D941-8162-1AA7FB482D48.root'

    )
)

process.maxEvents = cms.untracked.PSet( input = cms.untracked.int32(-1) )

process.TFileService = cms.Service("TFileService",
   fileName = cms.string('output.root')
)

#process.HcalIsoTrkAnalyzer.maxDzPV = 1.0
#process.HcalIsoTrkAnalyzer.minOuterHit =  0
#process.HcalIsoTrkAnalyzer.minLayerCrossed =  0

process.HcalIsoTrkAnalyzer.triggers = []
process.HcalIsoTrkAnalyzer.oldID = [21701, 21603]
process.HcalIsoTrkAnalyzer.newDepth = [2, 4]
process.HcalIsoTrkAnalyzer.hep17 = True
process.HcalIsoTrkAnalyzer.dataType = 0 #0 for jetHT else 1
#process.HcalIsoTrkAnalyzer.maximumEcalEnergy = 100 # set MIP cut  
#process.HcalIsoTrkAnalyzer.useRaw = 2
process.HcalIsoTrkAnalyzer.unCorrect = True

process.p = cms.Path(process.HcalIsoTrkAnalyzer)

