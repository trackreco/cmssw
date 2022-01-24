#include "MkFitter.h"

#include "KalmanUtilsMPlex.h"
#include "MatriplexPackers.h"

//#define DEBUG
#include "Debug.h"

#include <sstream>

namespace mkfit {

  void MkFitter::checkAlignment() {
    printf("MkFitter alignment check:\n");
    Matriplex::align_check("  Err[0]   =", &Err[0].fArray[0]);
    Matriplex::align_check("  Err[1]   =", &Err[1].fArray[0]);
    Matriplex::align_check("  Par[0]   =", &Par[0].fArray[0]);
    Matriplex::align_check("  Par[1]   =", &Par[1].fArray[0]);
    Matriplex::align_check("  msErr[0] =", &msErr[0].fArray[0]);
    Matriplex::align_check("  msPar[0] =", &msPar[0].fArray[0]);
  }

  void MkFitter::printPt(int idx) {
    for (int i = 0; i < NN; ++i) {
      printf("%5.2f  ", std::hypot(Par[idx].At(i, 3, 0), Par[idx].At(i, 4, 0)));
    }
  }

  int MkFitter::countValidHits(int itrack, int end_hit) const {
    int result = 0;
    for (int hi = 0; hi < end_hit; ++hi) {
      if (HoTArr[hi](itrack, 0, 0).index >= 0)
        result++;
    }
    return result;
  }

  int MkFitter::countInvalidHits(int itrack, int end_hit) const {
    int result = 0;
    for (int hi = 0; hi < end_hit; ++hi) {
      // XXXX MT: Should also count -2 hits as invalid?
      if (HoTArr[hi](itrack, 0, 0).index == -1)
        result++;
    }
    return result;
  }

  //==============================================================================

  void MkFitter::inputTracksAndHits(const std::vector<Track>& tracks,
                                    const std::vector<HitVec>& layerHits,
                                    int beg,
                                    int end) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    int itrack = 0;

    for (int i = beg; i < end; ++i, ++itrack) {
      const Track& trk = tracks[i];

      Err[iC].copyIn(itrack, trk.errors().Array());
      Par[iC].copyIn(itrack, trk.parameters().Array());

      Chg(itrack, 0, 0) = trk.charge();
      Chi2(itrack, 0, 0) = trk.chi2();
      Label(itrack, 0, 0) = trk.label();

      // CopyIn seems fast enough, but indirections are quite slow.
      for (int hi = 0; hi < Nhits; ++hi) {
        HoTArr[hi](itrack, 0, 0) = trk.getHitOnTrack(hi);

        const int hidx = trk.getHitIdx(hi);
        if (hidx < 0)
          continue;

        const Hit& hit = layerHits[hi][hidx];
        msErr[hi].copyIn(itrack, hit.errArray());
        msPar[hi].copyIn(itrack, hit.posArray());
      }
    }
  }

  void MkFitter::inputTracksAndHits(const std::vector<Track>& tracks,
                                    const std::vector<LayerOfHits>& layerHits,
                                    int beg,
                                    int end) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    int itrack;

    for (int i = beg; i < end; ++i) {
      itrack = i - beg;
      const Track& trk = tracks[i];

      Label(itrack, 0, 0) = trk.label();

      Err[iC].copyIn(itrack, trk.errors().Array());
      Par[iC].copyIn(itrack, trk.parameters().Array());

      Chg(itrack, 0, 0) = trk.charge();
      Chi2(itrack, 0, 0) = trk.chi2();

      // CopyIn seems fast enough, but indirections are quite slow.
      for (int hi = 0; hi < Nhits; ++hi) {
        const int hidx = trk.getHitIdx(hi);
        const int hlyr = trk.getHitLyr(hi);
        const Hit& hit = layerHits[hlyr].refHit(hidx);

        msErr[hi].copyIn(itrack, hit.errArray());
        msPar[hi].copyIn(itrack, hit.posArray());

        HoTArr[hi](itrack, 0, 0) = trk.getHitOnTrack(hi);
      }
    }
  }

  void MkFitter::slurpInTracksAndHits(const std::vector<Track>& tracks,
                                      const std::vector<HitVec>& layerHits,
                                      int beg,
                                      int end) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    MatriplexTrackPacker mtp(tracks[beg]);

    for (int i = beg; i < end; ++i) {
      int itrack = i - beg;
      const Track& trk = tracks[i];

      Label(itrack, 0, 0) = trk.label();

      mtp.addInput(trk);

      Chg(itrack, 0, 0) = trk.charge();
      Chi2(itrack, 0, 0) = trk.chi2();
    }

    mtp.pack(Err[iC], Par[iC]);

    // CopyIn seems fast enough, but indirections are quite slow.
    for (int hi = 0; hi < Nhits; ++hi) {
      MatriplexHitPacker mhp(layerHits[hi][0]);

      for (int i = beg; i < end; ++i) {
        const int hidx = tracks[i].getHitIdx(hi);
        const Hit& hit = layerHits[hi][hidx];

        HoTArr[hi](i - beg, 0, 0) = tracks[i].getHitOnTrack(hi);

        mhp.addInput(hit);
      }

      mhp.pack(msErr[hi], msPar[hi]);
    }
  }

  void MkFitter::inputTracksAndHitIdx(const std::vector<Track>& tracks, int beg, int end, bool inputProp) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    const int iI = inputProp ? iP : iC;

    int itrack = 0;
    for (int i = beg; i < end; ++i, ++itrack) {
      const Track& trk = tracks[i];

      Err[iI].copyIn(itrack, trk.errors().Array());
      Par[iI].copyIn(itrack, trk.parameters().Array());

      Chg(itrack, 0, 0) = trk.charge();
      Chi2(itrack, 0, 0) = trk.chi2();
      Label(itrack, 0, 0) = trk.label();

      for (int hi = 0; hi < Nhits; ++hi) {
        HoTArr[hi](itrack, 0, 0) = trk.getHitOnTrack(hi);
      }
    }
  }

  void MkFitter::inputTracksAndHitIdx(const std::vector<std::vector<Track> >& tracks,
                                      const std::vector<std::pair<int, int> >& idxs,
                                      int beg,
                                      int end,
                                      bool inputProp) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    const int iI = inputProp ? iP : iC;

    int itrack = 0;
    for (int i = beg; i < end; ++i, ++itrack) {
      const Track& trk = tracks[idxs[i].first][idxs[i].second];

      Label(itrack, 0, 0) = trk.label();
      SeedIdx(itrack, 0, 0) = idxs[i].first;
      CandIdx(itrack, 0, 0) = idxs[i].second;

      Err[iI].copyIn(itrack, trk.errors().Array());
      Par[iI].copyIn(itrack, trk.parameters().Array());

      Chg(itrack, 0, 0) = trk.charge();
      Chi2(itrack, 0, 0) = trk.chi2();

      for (int hi = 0; hi < Nhits; ++hi) {
        HoTArr[hi](itrack, 0, 0) = trk.getHitOnTrack(hi);
      }
    }
  }

  void MkFitter::inputSeedsTracksAndHits(const std::vector<Track>& seeds,
                                         const std::vector<Track>& tracks,
                                         const std::vector<HitVec>& layerHits,
                                         int beg,
                                         int end) {
    // Assign track parameters to initial state and copy hit values in.

    // This might not be true for the last chunk!
    // assert(end - beg == NN);

    int itrack;
    for (int i = beg; i < end; ++i) {
      itrack = i - beg;

      const Track& see = seeds[i];

      Label(itrack, 0, 0) = see.label();
      if (see.label() < 0)
        continue;

      Err[iC].copyIn(itrack, see.errors().Array());
      Par[iC].copyIn(itrack, see.parameters().Array());

      Chg(itrack, 0, 0) = see.charge();
      Chi2(itrack, 0, 0) = see.chi2();

      const Track& trk = tracks[see.label()];

      // CopyIn seems fast enough, but indirections are quite slow.
      for (int hi = 0; hi < Nhits; ++hi) {
        HoTArr[hi](itrack, 0, 0) = trk.getHitOnTrack(hi);

        const int hidx = trk.getHitIdx(hi);
        if (hidx < 0)
          continue;  //fixme, check if this is harmless

        const Hit& hit = layerHits[hi][hidx];
        msErr[hi].copyIn(itrack, hit.errArray());
        msPar[hi].copyIn(itrack, hit.posArray());
      }
    }
  }

  //------------------------------------------------------------------------------
  // Fitting with interleaved hit loading
  //------------------------------------------------------------------------------

  void MkFitter::inputTracksForFit(const std::vector<Track>& tracks, int beg, int end) {
    // Loads track parameters and hit indices.

    // XXXXMT4K has Config::nLayers: How many hits do we read in?
    // Check for max? Expect an argument?
    // What to do with invalid hits? Skip?

    // XXXX MT Here the same idx array WAS used for slurping in of tracks and
    // hots. With this, two index arrays are built, one within each packer.

    MatriplexTrackPacker mtp(tracks[beg]);
    MatriplexHoTPacker mhotp(*tracks[beg].getHitsOnTrackArray());

    int itrack = 0;

    for (int i = beg; i < end; ++i, ++itrack) {
      const Track& trk = tracks[i];

      Chg(itrack, 0, 0) = trk.charge();
      Chi2(itrack, 0, 0) = trk.chi2();
      Label(itrack, 0, 0) = trk.label();

      mtp.addInput(trk);

      mhotp.addInput(*trk.getHitsOnTrackArray());
    }

    mtp.pack(Err[iC], Par[iC]);
    for (int ll = 0; ll < Config::nLayers; ++ll) {
      mhotp.pack(HoTArr[ll], ll);
    }
  }

  void MkFitter::fitTracksWithInterSlurp(const std::vector<HitVec>& layersohits, const int N_proc) {
    // XXXX This has potential issues hits coming from different layers!
    // Expected to only work reliably with barrel (consecutive layers from 0 -> Nhits)
    // and with hits present on every layer for every track.

    // Loops over layers and:
    // a) slurps in hit parameters;
    // b) propagates and updates tracks

    for (int ii = 0; ii < Nhits; ++ii) {
      // XXXX Assuming hit index corresponds to layer!
      MatriplexHitPacker mhp(layersohits[ii][0]);

      for (int i = 0; i < N_proc; ++i) {
        const int hidx = HoTArr[ii](i, 0, 0).index;
        const int hlyr = HoTArr[ii](i, 0, 0).layer;

        // XXXXMT4K What to do with hidx < 0 ????
        // This could solve the unbalanced fit.
        // Or, if the hidx is the "universal" missing hit, it could just work.
        // Say, hidx = 0 ... grr ... but then we don't know it is missing.

        if (hidx < 0 || hlyr < 0) {
          mhp.addNullInput();
        } else {
          mhp.addInput(layersohits[hlyr][hidx]);
        }
      }

      mhp.pack(msErr[0], msPar[0]);

      propagateTracksToHitR(msPar[0], N_proc, Config::forward_fit_pflags);

      kalmanUpdate(Err[iP], Par[iP], msErr[0], msPar[0], Err[iC], Par[iC], N_proc);
    }
  }

  //==============================================================================
  // Fitting functions
  //==============================================================================

  void MkFitter::outputTracks(std::vector<Track>& tracks, int beg, int end, int iCP) const {
    // Copies last track parameters (updated) into Track objects.
    // The tracks vector should be resized to allow direct copying.

    int itrack = 0;
    for (int i = beg; i < end; ++i, ++itrack) {
      Err[iCP].copyOut(itrack, tracks[i].errors_nc().Array());
      Par[iCP].copyOut(itrack, tracks[i].parameters_nc().Array());

      tracks[i].setCharge(Chg(itrack, 0, 0));

      // XXXXX chi2 is not set (also not in SMatrix fit, it seems)
      tracks[i].setChi2(Chi2(itrack, 0, 0));
      tracks[i].setLabel(Label(itrack, 0, 0));
    }
  }

  void MkFitter::outputFittedTracksAndHitIdx(std::vector<Track>& tracks, int beg, int end, bool outputProp) const {
    // Copies last track parameters (updated) into Track objects and up to Nhits.
    // The tracks vector should be resized to allow direct copying.

    const int iO = outputProp ? iP : iC;

    int itrack = 0;
    for (int i = beg; i < end; ++i, ++itrack) {
      Err[iO].copyOut(itrack, tracks[i].errors_nc().Array());
      Par[iO].copyOut(itrack, tracks[i].parameters_nc().Array());

      tracks[i].setCharge(Chg(itrack, 0, 0));
      tracks[i].setChi2(Chi2(itrack, 0, 0));
      tracks[i].setLabel(Label(itrack, 0, 0));

      // QQQQ Could do resize and std::copy, as in MkFinder::copy_out(), but
      // we do not know the correct N_found_hits.
      tracks[i].resetHits();
      tracks[i].reserveHits(Nhits);
      for (int hi = 0; hi < Nhits; ++hi) {
        tracks[i].addHitIdx(HoTArr[hi](itrack, 0, 0), 0.);
      }
    }
  }

}  // end namespace mkfit
