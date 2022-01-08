#ifndef RecoTracker_MkFitCore_src_MkFinder_h
#define RecoTracker_MkFitCore_src_MkFinder_h

#include "MkBase.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

#include "RecoTracker/MkFitCore/interface/HitStructures.h"

// Define to get printouts about track and hit chi2.
// See also MkBuilder::BackwardFit().

//#define DEBUG_BACKWARD_FIT_BH
//#define DEBUG_BACKWARD_FIT

namespace mkfit {

  class CandCloner;
  class CombCandidate;
  class LayerOfHits;
  class FindingFoos;
  class IterationParams;
  class IterationLayerConfig;
  class SteeringParams;

  //#ifdef DEBUG_BACKWARD_FIT
  //class Event;
  //#endif

#ifdef DUMPHITWINDOW
  class Event;
#endif

  // NOTES from MkFitter ... where things were getting super messy.
  //
  // Would like to get rid of msErr, msPar arrays ... single one should do.
  // Would also like to get rid of Nhits ... but this is a tougher one.
  // Fitting is somewhat depending on this (not the new inter-slurp version).
  // No need to have hit array as matriplex. Ideally, don't even want the full
  // array if we could copy / clone it from the original candidate.
  // Copy as needed or copy full size (compile time constant)? memcpy, std::copy
  // Do need per track size.
  //
  // Can also have nFoundHits ... then do not need countValid/InvalidHits().
  //
  // For now will prefix the new variables with 'tf' for track finding.
  // Changes will go into all finding functions + import-with-hit-idcs.
  //
  // Actually ... am tempted to make MkFinder :)

  class MkFinder : public MkBase {
  public:
    static constexpr int MPlexHitIdxMax = 16;

    using MPlexHitIdx = Matriplex::Matriplex<int, MPlexHitIdxMax, 1, NN>;
    using MPlexQHoT = Matriplex::Matriplex<HitOnTrack, 1, 1, NN>;

    //----------------------------------------------------------------------------

    MPlexQF Chi2;
    MPlexQI Label;  // seed index in global seed vector (for MC truth match)

    MPlexQI NHits;
    MPlexQI NFoundHits;

    HitOnTrack HoTArrs[NN][Config::nMaxTrkHits];

#ifdef DUMPHITWINDOW
    MPlexQI SeedAlgo;   // seed algorithm
    MPlexQI SeedLabel;  // seed label
#endif

    MPlexQI SeedIdx;  // seed index in local thread (for bookkeeping at thread level)
    MPlexQI CandIdx;  // candidate index for the given seed (for bookkeeping of clone engine)

    MPlexQI Stopped;  // Flag for BestHit that a track has been stopped (and copied out already)

    // Additions / substitutions for TrackCand copy_in/out()
    // One could really access the original TrackCand for all of those, especially the ones that
    // are STD only. This then requires access back to that TrackCand memory.
    // So maybe one should just have flags for CopyIn methods (or several versions). Yay, etc.
    MPlexQI NMissingHits;         // sub: NHits, sort of, STD only
    MPlexQI NOverlapHits;         // add: num of overlaps registered in HitOnTrack, STD only
    MPlexQI NInsideMinusOneHits;  // sub: before we copied all hit idcs and had a loop counting them only
    MPlexQI NTailMinusOneHits;    // sub: before we copied all hit idcs and had a loop counting them only
    MPlexQI LastHitCcIndex;       // add: index of last hit in CombCand hit tree, STD only
    MPlexQUI TrkStatus;           // STD only, status bits
    HitOnTrack LastHoT[NN];
    CombCandidate *CombCand[NN];
    // const TrackCand *TrkCand[NN]; // hmmh, could get all data through this guy ... but scattered
    // storing it in now for bkfit debug printouts
    TrackCand *TrkCand[NN];
    // XXXX - for bk-fit debug

    //#ifdef DEBUG_BACKWARD_FIT
    //  Event     *m_event;
    //#endif

    // Hit indices into LayerOfHits to explore.
    WSR_Result XWsrResult[NN];  // Could also merge it with XHitSize. Or use smaller arrays.
    MPlexQI XHitSize;
    MPlexHitIdx XHitArr;

    // Hit errors / parameters for hit matching, update.
    MPlexHS msErr;
    MPlexHV msPar;

    // An idea: Do propagation to hit in FindTracksXYZZ functions.
    // Have some state / functions here that make this short to write.
    // This would simplify KalmanUtils (remove the propagate functions).
    // Track errors / parameters propagated to current hit.
    // MPlexLS    candErrAtCurrHit;
    // MPlexLV    candParAtCurrHit;

    const IterationParams *m_iteration_params = nullptr;
    const IterationLayerConfig *m_iteration_layer_config = nullptr;
    const std::vector<bool> *m_iteration_hit_mask = nullptr;

    //============================================================================

    MkFinder() {}

    void Setup(const IterationParams &ip, const IterationLayerConfig &ilc, const std::vector<bool> *ihm);
    void Release();

    //----------------------------------------------------------------------------

    void InputTracksAndHitIdx(const std::vector<Track> &tracks, int beg, int end, bool inputProp);

    void InputTracksAndHitIdx(const std::vector<Track> &tracks,
                              const std::vector<int> &idxs,
                              int beg,
                              int end,
                              bool inputProp,
                              int mp_offset);

    void InputTracksAndHitIdx(const std::vector<CombCandidate> &tracks,
                              const std::vector<std::pair<int, int>> &idxs,
                              int beg,
                              int end,
                              bool inputProp);

    void InputTracksAndHitIdx(const std::vector<CombCandidate> &tracks,
                              const std::vector<std::pair<int, IdxChi2List>> &idxs,
                              int beg,
                              int end,
                              bool inputProp);

    void OutputTracksAndHitIdx(std::vector<Track> &tracks, int beg, int end, bool outputProp) const;

    void OutputTracksAndHitIdx(
        std::vector<Track> &tracks, const std::vector<int> &idxs, int beg, int end, bool outputProp) const;

    void OutputTrackAndHitIdx(Track &track, int itrack, bool outputProp) const {
      const int iO = outputProp ? iP : iC;
      copy_out(track, itrack, iO);
    }

    void OutputNonStoppedTracksAndHitIdx(
        std::vector<Track> &tracks, const std::vector<int> &idxs, int beg, int end, bool outputProp) const {
      const int iO = outputProp ? iP : iC;
      for (int i = beg, imp = 0; i < end; ++i, ++imp) {
        if (!Stopped[imp])
          copy_out(tracks[idxs[i]], imp, iO);
      }
    }

    HitOnTrack BestHitLastHoT(int itrack) const { return HoTArrs[itrack][NHits(itrack, 0, 0) - 1]; }

    //----------------------------------------------------------------------------

    void getHitSelDynamicWindows(
        const float invpt, const float theta, float &min_dq, float &max_dq, float &min_dphi, float &max_dphi);

    float getHitSelDynamicChi2Cut(const int itrk, const int ipar);

    void SelectHitIndices(const LayerOfHits &layer_of_hits, const int N_proc);

    void AddBestHit(const LayerOfHits &layer_of_hits, const int N_proc, const FindingFoos &fnd_foos);

    //----------------------------------------------------------------------------

    void FindCandidates(const LayerOfHits &layer_of_hits,
                        std::vector<std::vector<TrackCand>> &tmp_candidates,
                        const int offset,
                        const int N_proc,
                        const FindingFoos &fnd_foos);

    //----------------------------------------------------------------------------

    void FindCandidatesCloneEngine(const LayerOfHits &layer_of_hits,
                                   CandCloner &cloner,
                                   const int offset,
                                   const int N_proc,
                                   const FindingFoos &fnd_foos);

    void UpdateWithLastHit(const LayerOfHits &layer_of_hits, int N_proc, const FindingFoos &fnd_foos);

    void CopyOutParErr(std::vector<CombCandidate> &seed_cand_vec, int N_proc, bool outputProp) const;

    //----------------------------------------------------------------------------
    // Backward fit hack

    int CurHit[NN];
    const HitOnTrack *HoTArr[NN];

    int CurNode[NN];
    HoTNode *HoTNodeArr[NN];  // Not const as we can modify it!

    void BkFitInputTracks(TrackVec &cands, int beg, int end);
    void BkFitOutputTracks(TrackVec &cands, int beg, int end, bool outputProp);

    void BkFitInputTracks(EventOfCombCandidates &eocss, int beg, int end);
    void BkFitOutputTracks(EventOfCombCandidates &eocss, int beg, int end, bool outputProp);

    void BkFitFitTracksBH(const EventOfHits &eventofhits,
                          const SteeringParams &st_par,
                          const int N_proc,
                          bool chiDebug = false);

    void BkFitFitTracks(const EventOfHits &eventofhits,
                        const SteeringParams &st_par,
                        const int N_proc,
                        bool chiDebug = false);

    void BkFitPropTracksToPCA(const int N_proc);

    //----------------------------------------------------------------------------

#ifdef DUMPHITWINDOW
    Event *m_event;
#endif

  private:
    void copy_in(const Track &trk, const int mslot, const int tslot) {
      Err[tslot].CopyIn(mslot, trk.errors().Array());
      Par[tslot].CopyIn(mslot, trk.parameters().Array());

      Chg(mslot, 0, 0) = trk.charge();
      Chi2(mslot, 0, 0) = trk.chi2();
      Label(mslot, 0, 0) = trk.label();

      NHits(mslot, 0, 0) = trk.nTotalHits();
      NFoundHits(mslot, 0, 0) = trk.nFoundHits();

      NInsideMinusOneHits(mslot, 0, 0) = trk.nInsideMinusOneHits();
      NTailMinusOneHits(mslot, 0, 0) = trk.nTailMinusOneHits();

      std::copy(trk.BeginHitsOnTrack(), trk.EndHitsOnTrack(), HoTArrs[mslot]);
    }

    void copy_out(Track &trk, const int mslot, const int tslot) const {
      Err[tslot].CopyOut(mslot, trk.errors_nc().Array());
      Par[tslot].CopyOut(mslot, trk.parameters_nc().Array());

      trk.setCharge(Chg(mslot, 0, 0));
      trk.setChi2(Chi2(mslot, 0, 0));
      trk.setLabel(Label(mslot, 0, 0));

      trk.resizeHits(NHits(mslot, 0, 0), NFoundHits(mslot, 0, 0));
      std::copy(HoTArrs[mslot], &HoTArrs[mslot][NHits(mslot, 0, 0)], trk.BeginHitsOnTrack_nc());
    }

    void copy_in(const TrackCand &trk, const int mslot, const int tslot) {
      Err[tslot].CopyIn(mslot, trk.errors().Array());
      Par[tslot].CopyIn(mslot, trk.parameters().Array());

      Chg(mslot, 0, 0) = trk.charge();
      Chi2(mslot, 0, 0) = trk.chi2();
      Label(mslot, 0, 0) = trk.label();

      LastHitCcIndex(mslot, 0, 0) = trk.lastCcIndex();
      NFoundHits(mslot, 0, 0) = trk.nFoundHits();
      NMissingHits(mslot, 0, 0) = trk.nMissingHits();
      NOverlapHits(mslot, 0, 0) = trk.nOverlapHits();

      NInsideMinusOneHits(mslot, 0, 0) = trk.nInsideMinusOneHits();
      NTailMinusOneHits(mslot, 0, 0) = trk.nTailMinusOneHits();

      LastHoT[mslot] = trk.getLastHitOnTrack();
      CombCand[mslot] = trk.combCandidate();
      TrkStatus[mslot] = trk.getRawStatus();
    }

    void copy_out(TrackCand &trk, const int mslot, const int tslot) const {
      Err[tslot].CopyOut(mslot, trk.errors_nc().Array());
      Par[tslot].CopyOut(mslot, trk.parameters_nc().Array());

      trk.setCharge(Chg(mslot, 0, 0));
      trk.setChi2(Chi2(mslot, 0, 0));
      trk.setLabel(Label(mslot, 0, 0));

      trk.setLastCcIndex(LastHitCcIndex(mslot, 0, 0));
      trk.setNFoundHits(NFoundHits(mslot, 0, 0));
      trk.setNMissingHits(NMissingHits(mslot, 0, 0));
      trk.setNOverlapHits(NOverlapHits(mslot, 0, 0));

      trk.setNInsideMinusOneHits(NInsideMinusOneHits(mslot, 0, 0));
      trk.setNTailMinusOneHits(NTailMinusOneHits(mslot, 0, 0));

      trk.setCombCandidate(CombCand[mslot]);
      trk.setRawStatus(TrkStatus(mslot, 0, 0));
    }

    void add_hit(const int mslot, int index, int layer) {
      // Only used by BestHit.
      // NInsideMinusOneHits and NTailMinusOneHits are maintained here but are
      // not used and are not copied out (as Track does not have these members).

      int &n_tot_hits = NHits(mslot, 0, 0);
      int &n_fnd_hits = NFoundHits(mslot, 0, 0);

      if (n_tot_hits < Config::nMaxTrkHits) {
        HoTArrs[mslot][n_tot_hits++] = {index, layer};
        if (index >= 0) {
          ++n_fnd_hits;
          NInsideMinusOneHits(mslot, 0, 0) += NTailMinusOneHits(mslot, 0, 0);
          NTailMinusOneHits(mslot, 0, 0) = 0;
        } else if (index == -1) {
          ++NTailMinusOneHits(mslot, 0, 0);
        }
      } else {
        // printf("WARNING MkFinder::add_hit hit-on-track limit reached for label=%d\n", label_);

        const int LH = Config::nMaxTrkHits - 1;

        if (index >= 0) {
          if (HoTArrs[mslot][LH].index < 0)
            ++n_fnd_hits;
          HoTArrs[mslot][LH] = {index, layer};
        } else if (index == -2) {
          if (HoTArrs[mslot][LH].index >= 0)
            --n_fnd_hits;
          HoTArrs[mslot][LH] = {index, layer};
        }
      }
    }

    int num_all_minus_one_hits(const int mslot) const {
      return NInsideMinusOneHits(mslot, 0, 0) + NTailMinusOneHits(mslot, 0, 0);
    }

    int num_inside_minus_one_hits(const int mslot) const { return NInsideMinusOneHits(mslot, 0, 0); }
  };

}  // end namespace mkfit
#endif
