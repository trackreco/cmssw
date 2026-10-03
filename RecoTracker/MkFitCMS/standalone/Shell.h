#ifndef RecoTracker_MkFitCMS_standalone_Shell_h
#define RecoTracker_MkFitCMS_standalone_Shell_h

#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/Track.h"
#include "RecoTracker/MkFitCore/interface/DeadRegion.h"

#include <map>
#include <string>

class TTree;

#ifdef WITH_REVE
namespace ROOT::Experimental {
  class REveManager;
  class REveTrackPropagator;
}
#endif

namespace mkfit {

  class DataFile;
  class Event;
  class EventOfHits;
  class MkBuilder;
  class TrackerInfo;

  // Per-event processing context: everything the track-finding pipeline needs
  // that cannot be shared between concurrently processed events.
  //
  // Shared, and deliberately NOT in here: the MkFinder / CandCloner / MkFitter
  // pools in g_exe_ctx (thread-safe, create-on-demand), the DataFile (its event
  // cursor is mutex-guarded) and Config.
  //
  // Shell holds one of these for interactive use; a multi-threaded driver holds
  // one per event in flight.
  struct EvCtx {
    Event       *ev  = nullptr;
    EventOfHits *eoh = nullptr;
    MkBuilder   *bld = nullptr;
    FILE        *fp  = nullptr;  // own read handle, only needed when running MT

    TrackVec seeds;   // working copies; published into ev during processing
    TrackVec tracks;
  };

  class Shell {
  public:
    enum SeedSelect_e { SS_UseAll = 0, SS_Label, SS_IndexPreCleaning, SS_IndexPostCleaning, SS_PreSet };

    // HLT seeds arrive as one iteration, three kinds back to back, told apart
    // only by hit pattern: does the seed start in the pixels, does it end in
    // the strips. In file order: pT5 (T5 matched to pixel tracks), T5 (the
    // leftover, unmatched ones), pix (leftover pixel-only).
    enum HltSeedKind_e { HSK_pT5 = 0, HSK_T5, HSK_pix };

    struct HltSeedBlocks {
      int first = -1;   // first seed index of the wanted kind, -1 if there are none
      int count = 0;    // how many of them
      int n_pT5 = 0, n_T5 = 0, n_pix = 0;   // totals, all three kinds
      bool contiguous = true;  // false if the wanted kind was not one solid block
    };

    static const char* hlt_seed_kind_name(HltSeedKind_e k);

    // Runtime selectors for ProcessEventHlt(), so the pT5-chopped control can
    // be run against plain T5 in one binary. A pT5's pixel hits were FOUND by
    // the upstream reconstruction, so chopping them off gives a denominator of
    // hits known to be findable -- which a plain T5 does not.
    static HltSeedKind_e s_hlt_seed_kind;
    static bool          s_hlt_chop_pixels;

    // What the chop actually removed, keyed by the seed's (sequential) label.
    // This is the exact denominator for "did the inward search put them back":
    // these hits were found by the upstream reconstruction, so they are known
    // findable, and no truth matching is involved in the comparison.
    static std::map<int, std::vector<HitOnTrack>> s_chopped_hits;

    // Copy the seeds of one algo into `out`, honouring the SS_* selector.
    // Seeds are taken to be grouped by algo, so the scan stops once it leaves
    // that block. SS_PreSet leaves `out` untouched.
    static void select_seeds(const Event &ev, int algo, SeedSelect_e seed_select,
                             int selected_seed, int count, TrackVec &out);

    // Classify the seeds of one algo by hit pattern and return the block of the
    // wanted kind. Everything downstream takes [first, first+count), so a false
    // `contiguous` means the file layout changed and that range is wrong.
    static HltSeedBlocks hlt_seed_preselect(const Event &ev, const TrackerInfo &ti,
                                            int algo, HltSeedKind_e want);

    Shell();
    Shell(std::vector<DeadVec> &dv, const std::string &in_file, int start_ev);
    ~Shell();
    void RunShell(const std::vector<std::string> &commands);

    void Status();

    void GoToEvent(int eid);
    void NextEvent(int skip = 1);

    // Event range used by the Loop / Test drivers below.
    // 1-based and inclusive: event 1 is the first event in the file, as on the
    // command line. Defaults come from --start-event and --num-events; a
    // positive count argument to a driver overrides the length for that call.
    void SetEventRangeFirstLast(int first, int last);
    void SetEventRangeBegCnt(int beg, int count);
    int EventRangeFirst() const { return m_ev_first; }
    int EventRangeLast() const { return m_ev_last; }
    int EventsInFile() const { return m_evs_in_file; }
    // Pipeline functions take the context explicitly so several events can be
    // in flight at once. The no-context overloads run on the shell's own
    // context -- that is what you want from the ROOT prompt.
    void ProcessEvent(EvCtx &ctx, SeedSelect_e seed_select = SS_UseAll, int selected_seed = -1, int count = 1);
    void ProcessEvent(SeedSelect_e seed_select = SS_UseAll, int selected_seed = -1, int count = 1)
      { ProcessEvent(m_ctx, seed_select, selected_seed, count); }
    Event* RelinquishEvent();

    void SelectIterationIndex(int itidx);
    void SelectIterationAlgo(int algo);
    void PrintIterations();

    // ROOT implicit MT for the RDF analysis. n_thr <= 0 disables it.
    // NOTE: mkFit.cc caps total TBB parallelism at Config::numThreadsFinder via
    // tbb::global_control, so run with --num-thr >= n_thr or this starves.
    void EnableRdfMT(int n_thr);

    bool GetDebug() const;
    void SetDebug(bool b);
    void SetCleanSeeds(bool b);
    void SetBackwardFit(bool b);
    void SetBackwardSearch(bool b);
    void SetRemoveDuplicates(bool b);
    void SetUseDeadModules(bool b);
    void SetUseV2p2(bool b);
    // Backward search with findTracksStandardv2p2 when v2p2 is on, as CMSSW's
    // run_OneIteration() does; the default keeps findTracksCloneEngine.
    void SetBkwSearchV2p2(bool b) { m_bkw_search_v2p2 = b; }

    Event *event() { return m_ctx.ev; }
    EventOfHits *eoh() { return m_ctx.eoh; }
    MkBuilder *builder() { return m_ctx.bld; }
    TrackerInfo *tracker_info();

    EvCtx &ctx() { return m_ctx; }

    const TrackVec &seeds() const { return m_ctx.seeds; }
    const TrackVec &tracks() const { return m_ctx.tracks; }

    void SetSeedsFromIdcs(std::vector<int> idcs);

    // Build "fake but true" seeds from the SIM tracks: the seed state is the
    // sim track's own state and its hits are that track's innermost ones. Run
    // with SS_PreSet, this lets the FORWARD search start at layer 0 and go
    // outward through the pixel barrel -- the one region the normal forward
    // search never scans, because that is where its real seeds already are.
    // n_seed_hits: how many innermost hits to hand the seed (>=1).
    // Returns the number of seeds built.
    int MakeSimSeeds(EvCtx &ctx, int n_seed_hits = 1, float pt_min = 0.5f);
    int MakeSimSeeds(int n_seed_hits = 1, float pt_min = 0.5f)
      { return MakeSimSeeds(m_ctx, n_seed_hits, pt_min); }
    void ProcessEventSimSeeded(EvCtx &ctx, int n_seed_hits = 1, float pt_min = 0.5f);
    void ProcessEventSimSeeded(int n_seed_hits = 1, float pt_min = 0.5f)
      { ProcessEventSimSeeded(m_ctx, n_seed_hits, pt_min); }

    // ---- The mkFit seeder's quads as seeds, in Shell-Seeder.cc -------------
    // LoadSeederQuads() reads a `seedsurf --seeds` file (MkFitCMS/standalone/seeding/). Per event,
    // MakeSeederSeeds() turns the current event's quads into seed tracks, and ProcessEventSeeder()
    // runs the selected iteration on them, as ProcessEventStd() does on the file's own seeds.
    // mode 0: the helix through the first, a middle and the last hit, at the last hit, with the
    //         diagonal covariance s_seeder_fake_sigma;
    // mode 1: the same helix at the first hit with the broad prior s_seeder_prior_sigma (times
    //         s_seeder_prior_scale), then the Kalman update with each of the four hits, propagating
    //         onto each hit's module plane with material (backward_fit_pflags); the state ends at the
    //         last hit, as a CMSSW seed's does.
    struct SeederQuad {
      int l[4];
      int h[4];
      float score;
    };
    int LoadSeederQuads(const char *file);
    int MakeSeederSeeds(EvCtx &ctx, int mode = 1);
    int MakeSeederSeeds(int mode = 1) { return MakeSeederSeeds(m_ctx, mode); }
    void ProcessEventSeeder(EvCtx &ctx, int mode = 1);
    void ProcessEventSeeder(int mode = 1) { ProcessEventSeeder(m_ctx, mode); }
    // sigma of x, y, z [cm], 1/pT (relative to the helix's 1/pT), phi, theta [rad]
    static float s_seeder_prior_sigma[6];
    static float s_seeder_prior_scale;
    // mode 1: start with hit 0's position covariance and update with hits 1-3 only
    static bool s_seeder_pos_from_hit0;
    // print the first this many failed fits
    static int s_seeder_debug;
    static float s_seeder_fake_sigma[6];

    // Seed states against truth: for each seed whose four hits share one sim track, the residual of
    // q/pT, phi and theta to the SimHitState of its last hit, over the seed's own sigma; the same for
    // the file's seeds of the selected iteration; and, for our seeds whose four hits are a file
    // seed's, the ratio of the sigmas. Needs a sample with SimHitStates.
    void SeederSeedCheckReset();
    void SeederSeedCheck(EvCtx &ctx);
    void SeederSeedCheck() { SeederSeedCheck(m_ctx); }
    void SeederSeedCheckReport();

    // Where the fakes come from: per found track, its seed (by hits) and the seed's truth: true (all hits
    // on one sim track), undecidable (the linked hits agree, some hits unlinked), fake, or no seed found.
    // row 0: the tracks of the file's seeds, 1: of the seeder's. Call right after the processing.
    // SeederMissStudy(): the selected sim tracks (val_eff's MTV selection) that no seeder seed is on,
    // by their pixel layers and OT1-P / OT2-P hits, and whether production's initialStep or
    // highPtTripletStep found them. Call after ProcessEventSeeder().
    void SeederDiagReset();
    void SeederFakeOrigin(EvCtx &ctx, int row);
    void SeederFakeOrigin(int row) { SeederFakeOrigin(m_ctx, row); }
    void SeederMissStudy(EvCtx &ctx);
    void SeederMissStudy() { SeederMissStudy(m_ctx); }
    void SeederDiagReport();

    // Displaced tracks: the selected sim tracks (val_eff's MTV selection) by the transverse impact parameter of
    // their production vertex and momentum to the beam spot, |d0|, and per bin how many the seeder's seeds are
    // on, how many the event's candidateTracks_ found (call after ProcessEventSeeder() and, for the row like
    // production's, SelectTracksCMSSW(1)), and how many production found, iteration by iteration
    // (cumulative; needs a sample with every iteration's tracks). "Found" is val_eff's association (the track's
    // seed by shared hits, TrackExtra::setMCTrackIDInfo over the non-seed hits).
    void SeederDisplacedReset();
    void SeederDisplacedStudy(EvCtx &ctx);
    void SeederDisplacedStudy() { SeederDisplacedStudy(m_ctx); }
    void SeederDisplacedReport();

    // The anatomy of the seeder's quads (the --seeds file, before the iteration's seed cleaning): per quad the sim
    // track of each hit, the class (true; 3 + 1 wrong: one hit on another sim track; 3 + 1 unlinked; 2 + 2 and
    // worse), for 3 + 1 wrong the position and layer of the wrong hit, whether the majority sim track has its own
    // hit in that layer (a competitor lost), how far apart the two are, and the hit density around the wrong hit
    // against that around the last hit of true quads. Then, per candidate track (call after ProcessEventSeeder()
    // and SelectTracksCMSSW(1)), the class of the quad it grew from (the quad whose four hits are on it), for fake
    // and for found tracks; found and fake by val_eff's association.
    void SeederQuadAnatomyReset();
    void SeederQuadAnatomy(EvCtx &ctx);
    void SeederQuadAnatomy() { SeederQuadAnatomy(m_ctx); }
    void SeederQuadAnatomyReport();

    // CMSSW's MultiTrackSelector::select() with the phase-2 initialStepSelector parameter sets, on the
    // event's candidateTracks_, in place: level 1 loose (what the track merger keeps, so what
    // generalTracks holds), 2 tight, 3 highPurity, each applied after the ones below it as the
    // selector's prefilters are. d0 and z0 come from the helix through the track's state (its first hit);
    // their errors from a linear transport of its covariance (--backward-fit-pca exports tracks with a
    // zero covariance and chi2 on this branch, so it is not used). The primary vertices are the SIM ones (production vertices
    // within 0.01 cm of the beam line with >= 2 charged sim tracks), in place of
    // firstStepPrimaryVertices. Returns the number of tracks kept.
    int SelectTracksCMSSW(EvCtx &ctx, int level);
    int SelectTracksCMSSW(int level) { return SelectTracksCMSSW(m_ctx, level); }

    // ---- Stage timing, in Shell.cc ----------------------------------------
    // Process CPU and wall time of the stages of ProcessEvent(), of MakeSeederSeeds() and of the hit
    // loading in GoToEvent(), summed per row: TimingRow() names the row the following calls add to
    // (an empty name stops the timing), TimingReport() prints ms per event, an event being one pass
    // through the find stage of that row. The hit loading goes to its own row, "load hits".
    enum TmStage_e { TM_SeedFit = 0, TM_SeedClean, TM_Find, TM_BkFit, TM_Final, TM_Load, TM_N };
    void TimingReset();
    void TimingRow(const char *row);
    void TimingReport(const char *file = nullptr);

    // --------------------------------------------------------
    // Analysis helpers

    int LabelFromHits(Track &t, bool replace, float good_frac);
    void FillByLabelMaps_CkfBase();

    bool CheckMkFitLayerPlanVsReferenceHits(const Track &mkft, const Track &reft, const std::string &name);

    // --------------------------------------------------------
    // Analysis drivers / main functions / Comparators

    void Compare();

    // --------------------------------------------------------
    // Seed study prototype
    using seed_selector_cf = bool(const Track &);
    using seed_selector_func = std::function<seed_selector_cf>;

    void StudySimAndSeeds(bool report_lost_seeds=true);
    void PreSelectSeeds(int iter_idx, seed_selector_func selector = [](const Track&) {return true;});

    void FindInterestingSimTracks();

    void WriteSimTree();
    void ReadSimTree();

    // --------------------------------------------------------
    // Low-level checks
    TTree* CheckHitVsModulePosition();

    // --------------------------------------------------------
    // Visualization stuff
#ifdef WITH_REVE
    void ReveInit();
    void ShowTracker(int lay_first, int lay_last);
    void ShowSimTrack(int sim_idx);

    ROOT::Experimental::REveManager& EveMgr() { return *m_reve_mgr; }
#endif

    // --------------------------------------------------------
    // Experimental phase2 / LST stuff, in Shell-LST.cc
    void RunLSTintoPix(EvCtx &ctx, SeedSelect_e seed_select = SS_UseAll, int selected_seed = -1, int count = 1);
    void RunLSTintoPix(SeedSelect_e seed_select = SS_UseAll, int selected_seed = -1, int count = 1)
      { RunLSTintoPix(m_ctx, seed_select, selected_seed, count); }

    void LoopNEvents(int N_events);

    // One event: seed classification + RunLSTintoPix + quality-val + reports.
    // Assumes ctx is already loaded with the wanted event.
    void ProcessEventHlt(EvCtx &ctx, const int wanted_algo = 4);
    void ProcessEventHlt(const int wanted_algo = 4) { ProcessEventHlt(m_ctx, wanted_algo); }
    void LoopNEventsHlt(int N_events = -1, const int wanted_algo = 4);

    // ---- Standard (non-HLT) forward search, traced ------------------------
    // The Hlt drivers above run the inward T5-into-pixels search on LST seeds.
    // These run the ordinary outward forward search on the iteration's own
    // seeds. ProcessEvent() already produces a full trace on the v2p2 path, so
    // all that was missing was a driver. Written as a deliberate parallel of
    // the *Hlt versions rather than factored with them -- commonalities to be
    // pulled out once we know which parts really are common.
    // Motivation: pre-selection pulls are only the visible end. Kalman, chi2
    // and scoring are where a bad covariance actually costs tracks, and that
    // wants the outward direction as well as the inward one.
    void ProcessEventStd(EvCtx &ctx);
    void ProcessEventStd() { ProcessEventStd(m_ctx); }
    void TraceFwdSearch(int Nevents = -1, const char *prefix = "mkfit-fwd", int n_thr = 1);

    // Current default processing. N_events <= 0 means "use the configured
    // event range", see SetEventRangeFirstLast() / SetEventRangeBegCnt().
    // NOTE on naming: Test() really is a test; TestVectorSource() and
    // TestEventSource() are tracing drivers and want to be Trace*.
    void Test(int Nevents = -1);
    void TestVectorSource();
    // n_thr > 1 runs that many events in flight, each in its own EvCtx; the RDF
    // analysis afterwards is unchanged. n_thr < 0 takes --num-thr-ev.
    void TestEventSource(int Nevents = -1, const char *prefix = "mkfit", int n_thr = 1);

  protected:
    int select_seeds_for_algo(int algo, TrackVec &seeds);

    // Find tracks over [ev_first, ev_last] and hand the processed Events out,
    // in event order. Serial for n_thr <= 1.
    void collect_events_hlt(int ev_first, int ev_last, int n_thr, int wanted_algo,
                            std::vector<const Event *> &out);
    void collect_events_std(int ev_first, int ev_last, int n_thr,
                            std::vector<const Event *> &out);

    // Effective range for one driver call: count > 0 overrides the length of
    // the configured range, starting from its first event.
    void resolve_event_range(int count, int &first, int &last) const;

  private:
    std::vector<DeadVec> &m_deadvectors;
    std::string m_in_file;             // kept so MT slots can open their own handles
    DataFile *m_data_file = nullptr;   // shared: event cursor is mutex-guarded
    EvCtx m_ctx;                       // the interactive / single-threaded context
    int m_evs_in_file = -1;
    int m_ev_first = 1;
    int m_ev_last = 1;
    int m_it_index = 0;
    bool m_clean_seeds = true;
    bool m_backward_fit = true;
    // Gates the backward SEARCH only; the backward FIT is m_backward_fit above.
    // Initialised from Config::backwardSearch (--no-backward-search), which this
    // path used to ignore -- see ProcessEvent().
    bool m_backward_search = true;
    bool m_bkw_search_v2p2 = false;
    bool m_remove_duplicates = true;

    using map_t = std::map<int, Track *>;
    using map_i = map_t::iterator;

    std::map<int, Track *> m_ckf_map, m_sim_map, m_seed_map, m_mkf_map;

    // Stage timing: per row, CPU and wall seconds per stage and the number of events
    struct TmRow {
      double cpu[TM_N] = {0}, wall[TM_N] = {0};
      int n_ev = 0;
    };
    std::map<std::string, TmRow> m_tm_rows;
    std::vector<std::string> m_tm_order;
    std::string m_tm_row;
    double m_tm_t0[2] = {0, 0};
    void tm_mark();
    void tm_add(TmStage_e st, const std::string &row);
    void tm_add(TmStage_e st) { tm_add(st, m_tm_row); }

    // LoadSeederQuads(): by event id (1-based, as GoToEvent())
    std::map<int, std::vector<SeederQuad>> m_seeder_quads;

#ifdef WITH_REVE
    ROOT::Experimental::REveManager *m_reve_mgr = nullptr;
    ROOT::Experimental::REveTrackPropagator *m_reve_track_prop = nullptr;
#endif

  };

}  // namespace mkfit

#endif
