import FWCore.ParameterSet.Config as cms

def customizeInitialStepMkFitSeeder(process, config = 'RecoTracker/MkFit/data/mkfit-phase2-seeder.json'):
    """The mkFit seeder in place of initialStep's CMSSW seeding, for phase 2 with mkFit in initialStep
    (process modifier trackingMkFitInitialStep).

    MkFitSeederProducer makes mkFit seeds from the mkFit hits; MkFitTrajectorySeedConverter, under the label
    initialStepSeeds, turns them into TrajectorySeeds and puts out the matching mkFit seeds, which
    initialStepTrackCandidatesMkFit (MkFitProducer) and initialStepTrackCandidates (MkFitOutputConverter) then
    take. The tracks keep the algorithm initialStep, from the seeds' label. The CMSSW seeding modules of
    initialStep and the MkFitSeedConverter are no longer consumed, so they do not run.

    config: the seeder's configuration (mkfit::SeederConfig JSON, `seedsurf --write-config`)."""
    from RecoTracker.MkFit.mkFitSeederConfigESProducer_cfi import mkFitSeederConfigESProducer as _config
    from RecoTracker.MkFit.mkFitSeederProducer_cfi import mkFitSeederProducer as _seeder
    from RecoTracker.MkFit.mkFitTrajectorySeedConverter_cfi import mkFitTrajectorySeedConverter as _converter

    for m in ('initialStepTrackCandidatesMkFit', 'initialStepTrackCandidates', 'initialStepSeeds', 'InitialStepTask'):
        if not hasattr(process, m):
            raise RuntimeError('customizeInitialStepMkFitSeeder: the process has no ' + m +
                               '; it needs phase 2 with trackingMkFitInitialStep')

    process.mkFitSeederConfig = _config.clone(
        ComponentName = 'mkFitSeederConfig',
        config = config,
    )
    process.initialStepMkFitSeeder = _seeder.clone(
        config = ('', 'mkFitSeederConfig'),
    )
    process.initialStepSeeds = _converter.clone(
        mkFitSeeds = 'initialStepMkFitSeeder',
    )
    process.InitialStepTask.add(process.initialStepMkFitSeeder)

    process.initialStepTrackCandidatesMkFit.seeds = 'initialStepSeeds'
    process.initialStepTrackCandidates.mkFitSeeds = 'initialStepSeeds'
    return process
