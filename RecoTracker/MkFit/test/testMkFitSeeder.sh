#!/bin/bash

function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

# The release's phase-2 default geometry, its era and its global tag, for cmsDriver
read GEOM ERA GT < <(python3 -c "
from Configuration.PyReleaseValidation.upgradeWorkflowComponents import upgradeProperties as p
import Configuration.Geometry.defaultPhase2ConditionsEra_cff as s
v = s.DEFAULT_VERSION
print(v, p['Run4'][v]['Era'], p['Run4'][v]['GT'])") || die "failed to read the phase-2 defaults" $?

# step3 with customizeInitialStepMkFitSeeder; the input file is never opened
(cmsDriver.py step3 -s RAW2DIGI,RECO:reconstruction_trackingOnly --conditions ${GT} --geometry Extended${GEOM} \
   --era ${ERA} --filein file:step2.root -n 1 \
   --customise RecoTracker/MkFit/customizeInitialStepMkFitSeeder.customizeInitialStepMkFitSeeder \
   --python_filename step3_seeder_cfg.py --no_exec) || die "failed to make step3 with customizeInitialStepMkFitSeeder" $?
(python3 ${SCRAM_TEST_PATH}/checkMkFitSeederCustomise.py step3_seeder_cfg.py) || die "the customise's wiring is wrong" $?

(cmsRun ${SCRAM_TEST_PATH}/mkFitSeederConfig_cfg.py) || die "failed to run mkFitSeederConfig_cfg.py" $?
