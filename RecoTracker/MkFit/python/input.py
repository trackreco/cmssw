from FWCore.ParameterSet.VarParsing import VarParsing

def parseArguments():
    options = VarParsing()
    options.register("sample",
                     "10mu",
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.string,
                     "Sample: 10mu, ttbarnopu, ttbarpu35, ttbarpu50, ttbarpu70")
    options.register("mkfit",
                     "all",
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.string,
                     "Run MkFit for all iterations ('all'), specific iteration(s) (e.g. 'InitialStep', can be comma separated list), or CKF ('') tracking")
    options.register("timing",
                     "",
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.string,
                     "Timing method to use: '', framework, FastTimerService")
#    options.register("hltOnDemand",
#                     0,
#                     VarParsing.multiplicity.singleton,
#                     VarParsing.varType.int,
#                     "Run HLT strip local reco fully (0) or on-demand (1) (on-demand won't work for mkfit)")
#    options.register("hltIncludeFourthHit",
#                     0,
#                     VarParsing.multiplicity.singleton,
#                     VarParsing.varType.int,
#                     "Include fourth hit of pixel track in iter0 seeds (default 0)")
    options.register("maxEvents",
                     0,
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.int,
                     "Maximum number of events to be procfessed. 0 means a sample-specific, relatively small default, N=>0 means to process at most N events, and -1 means to process all events of the sample")
    options.register("nthreads",
                     1,
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.int,
                     "Number of threads (default 1)")
    options.register("nstreams",
                     0,
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.int,
                     "Number of streams (or events in flight, default 1)")
    options.register("trackingNtuple",
                     "",
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.string,
                     "Produce trackingNtuple instead of DQM. Possible values are '', 'generalTracks', 'InitialStep' etc (default '' to disable)")
    options.parseArguments()
    return options

def apply(process, options):
    mkfit = (options.mkfit != "")

#    if mkfit and options.hltOnDemand != 0:
#        raise Exception("hltOnDemand does not work with mkfit")

    process.options.numberOfThreads = options.nthreads
    process.options.numberOfStreams = options.nstreams

    prefix = "file:/data2/slava77/samples/CMSSW_11_2_0-orig/2021"
    fname = "step2_sorted.root"
    if options.sample == "10mu":
        process.source.fileNames = [prefix+"/10muPt0p2to1000HS/"+fname]
        process.maxEvents.input = 1000
    elif options.sample == "ttbarnopu":
        process.source.fileNames = [prefix+"/11834.0_TTbar_14TeV+2021/AVE_0_BX01_25ns/"+fname]
        process.maxEvents.input = 100
    elif options.sample == "ttbarpu35":
        process.source.fileNames = [prefix+"/11834.0_TTbar_14TeV+2021/AVE_35_BX01_25ns/"+fname]
        process.maxEvents.input = 100
    elif options.sample == "ttbarpu50":
        process.source.fileNames = [prefix+"/11834.0_TTbar_14TeV+2021/AVE_50_BX01_25ns/"+fname]
        process.maxEvents.input = 100
    elif options.sample == "ttbarpu70":
        process.source.fileNames = [prefix+"/11834.0_TTbar_14TeV+2021/AVE_70_BX01_25ns/"+fname]
        process.maxEvents.input = 10
    else:
        raise Exception("Incorrect value of sample=%s, supported ones are 10mu, ttbarnopu, ttbarpu35, ttbarpu50, ttbarpu70" % options.sample)

    if options.maxEvents != 0:
        process.maxEvents.input = options.maxEvents

    if process.maxEvents.input.value() > 100 or process.maxEvents.input.value()  == -1:
        process.MessageLogger.cerr.FwkReport.reportEvery = 100

    if options.timing != "":
        if options.timing == "framework":
            process.options.wantSummary = True
        elif options.timing == "FastTimerService":
            process.FastTimerService.enableDQMbyPath = True
        else:
            raise Exception("Incorrect value of timing={}, supported are '', framework, FastTimerService".format(options.timing))
        for it in ["initialStepPreSplitting", "initialStep", "lowPtQuadStep", "highPtTripletStep", "lowPtTripletStep",
                   "detachedQuadStep", "detachedTripletStep", "pixelPairStep", "mixedTripletStep",
                   "pixelLessStep", "tobTecStep"]:
            try:
                getattr(process, it+"TrackCandidatesMkFit").limitConcurrency = True
            except AttributeError:
                pass

    return options
