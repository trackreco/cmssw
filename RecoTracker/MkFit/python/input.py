from FWCore.ParameterSet.VarParsing import VarParsing

def apply(process):
    options = VarParsing()
    options.register("sample",
                     "10mu",
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.string,
                     "Sample: 10mu, ttbarnopu, ttbar_pu50, ttbar_pu70")
    options.register("mkfit",
                     1,
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.int,
                     "Run MkFit (1) or CMSSW (0) tracking")
    options.register("timing",
                     0,
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.int,
                     "Run validation configuration (0) or a timing configuration (1)")
    options.register("hltOnDemand",
                     0,
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.int,
                     "Run HLT strip local reco fully (0) or on-demand (1) (on-demand won't work for mkfit)")
    options.register("hltIncludeFourthHit",
                     0,
                     VarParsing.multiplicity.singleton,
                     VarParsing.varType.int,
                     "Include fourth hit of pixel track in iter0 seeds (default 0)")
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
    options.parseArguments()

    timing = (options.timing != 0)
    mkfit = (options.mkfit != 0)

    if mkfit and options.hltOnDemand != 0:
        raise Exception("hltOnDemand does not work with mkfit")

    process.options.numberOfThreads = options.nthreads
    process.options.numberOfStreams = options.nstreams

    if timing:
        prefix = "file:/data2/mkortela/cmssw_samples"; fname = "step2_raw.root"
    else:
        prefix = "root://redirector.t2.ucsd.edu//store/user"; fname = "step2.root"
    if options.sample == "10mu":
        process.source.fileNames = [prefix+"/slava77/CMSSW_10_4_0_patch1-orig/10muPt0p2to10HS/"+fname]
        process.maxEvents.input = 1000
    elif options.sample == "ttbarnopu":
        process.source.fileNames = [prefix+"/slava77/CMSSW_10_4_0_patch1-orig/11024.0_TTbar_13/AVE_0_BX01_25ns/"+fname]
        process.maxEvents.input = 100
    elif options.sample == "ttbarpu50":
        process.source.fileNames = [prefix+"/slava77/CMSSW_10_4_0_patch1-orig/11024.0_TTbar_13/AVE_50_BX01_25ns/"+fname]
        process.maxEvents.input = 100
    elif options.sample == "ttbarpu70":
        process.source.fileNames = [prefix+"/slava77/CMSSW_10_4_0_patch1-orig/11024.0_TTbar_13/AVE_70_BX01_25ns/"+fname]
        process.maxEvents.input = 10
    else:
        raise Exception("Incorrect value of sample=%s, supported ones are 10mu, ttbarnopu, ttbarpu50, ttbarpu70" % options.sample)

    if options.maxEvents != 0:
        process.maxEvents.input = options.maxEvents

    if process.maxEvents.input.value() > 100 or process.maxEvents.input.value()  == -1:
        process.MessageLogger.cerr.FwkReport.reportEvery = 100

    return options
