#!/usr/bin/env python3
# Checks the wiring of customizeInitialStepMkFitSeeder in a step3 configuration (the file given as the argument):
# the seeder runs in initialStep, its seeds reach MkFitProducer and the output converter, the flagged-seed cut is
# on, and initialStep's high-purity selection has no |eta| cut.
import sys

exec(open(sys.argv[1]).read())
p = process

errors = []
def check(ok, what):
    print(('ok    ' if ok else 'FAIL  ') + what)
    if not ok:
        errors.append(what)

check(p.mkFitSeederConfig.type_() == 'MkFitSeederConfigESProducer', 'mkFitSeederConfig is MkFitSeederConfigESProducer')
check(p.mkFitSeederConfig.config.value() == 'RecoTracker/MkFit/data/mkfit-phase2-seeder.json',
      'the seeder configuration is the default JSON')
check(p.initialStepMkFitSeeder.type_() == 'MkFitSeederProducer', 'initialStepMkFitSeeder is MkFitSeederProducer')
check(p.initialStepSeeds.type_() == 'MkFitTrajectorySeedConverter', 'initialStepSeeds is MkFitTrajectorySeedConverter')
check(p.initialStepSeeds.mkFitSeeds.value() == 'initialStepMkFitSeeder', 'the converter takes the seeder\'s seeds')
check(p.initialStepTrackCandidatesMkFit.seeds.value() == 'initialStepSeeds', 'MkFitProducer takes initialStepSeeds')
check(p.initialStepTrackCandidates.mkFitSeeds.value() == 'initialStepSeeds',
      'the output converter takes the mkFit seeds of initialStepSeeds')
check(p.initialStepTrackCandidatesMkFit.flaggedSeedCut.minAddedHits.value() == 4, 'the flagged-seed cut is on (K = 4)')
names = p.InitialStepTask.moduleNames()
check('initialStepMkFitSeeder' in names and 'initialStepSeeds' in names, 'the seeder and the converter are in InitialStepTask')
for sel in p.initialStepSelector.trackSelectors:
    if hasattr(sel, 'max_eta'):
        check(sel.max_eta.value() >= 9999 and sel.min_eta.value() <= -9999,
              'initialStepSelector %s has no |eta| cut' % sel.name.value())

if errors:
    print('%d check(s) failed' % len(errors))
    sys.exit(1)
