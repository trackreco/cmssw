import RecoTracker.MkFit.mkFitProducer_cfi as mkFitProducer_cfi

def customizeInitialStepToMkFit(process):
    process.initialStepTrackCandidates = mkFitProducer_cfi.mkFitProducer.clone(
        seeds = "initialStepSeeds",
    )
    return process
