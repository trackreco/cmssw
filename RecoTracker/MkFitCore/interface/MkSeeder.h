#ifndef RecoTracker_MkFitCore_interface_MkSeeder_h
#define RecoTracker_MkFitCore_interface_MkSeeder_h

// MkSeeder: the mkFit seeder's driver, as MkBuilder is the track finding's.
// Per event it fills the seeder's layers (fill), runs the feed-forward chain on
// both z sides (find) and removes quads that share hits with a better one
// (clean).
//
// The configuration of the two passes, one SeedChain per z side, is set up by
// the caller and handed to setup(); so are the finders' fake cuts, through
// finder(). The layers are added by the caller through hits().

#include "RecoTracker/MkFitCore/interface/SeedStructures.h"
#include "RecoTracker/MkFitCore/interface/radix_sort.h"

#include <array>
#include <memory>
#include <utility>
#include <vector>

namespace mkfit {

  struct SeedChain;
  class SeedChainFinder;

  class MkSeeder {
  public:
    MkSeeder();
    ~MkSeeder();

    // the two passes: plus for the +z side, minus for the -z side
    void setup(SeedChain &plus, SeedChain &minus);

    SeedEventOfHits &hits() { return m_hits; }
    const SeedEventOfHits &hits() const { return m_hits; }
    // side 0: +z, 1: -z
    SeedChainFinder &finder(int side) { return *m_finder[side]; }

    // layer_hits: the event's HitVecs, indexed by mkFit layer id; bs: the event's beam spot, the origin
    // of the seeder's transverse coordinates
    void fill(const std::vector<HitVec> &layer_hits, const BeamSpot &bs);

    // out: (layer ids, original hit indices) per quad, +z side first; scores, if given: the cleaning
    // score of each quad, parallel to out; fake_scores, if given: the fake score (the sum the fake cut
    // is applied to), parallel to out
    void find(std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out,
              SeedCounters &cnt,
              std::vector<float> *scores = nullptr,
              std::vector<float> *fake_scores = nullptr);

    // Over all quads, keep a quad only if it shares fewer than min_shared hits with every better kept
    // quad; better = fewer outer-tracker layers, then the smaller score. The quads are given in the
    // caller's order, which breaks ties: layers[i], quads[i] and scores[i] describe quad i, and
    // keep[i] is set to 0 for a dropped one, 1 otherwise. n_dropped, if given: per quad, how many
    // quads were dropped for sharing hits with it (each dropped quad is charged to the first kept
    // quad found sharing enough hits).
    void clean(const std::vector<HitVec> &layer_hits,
               const std::vector<std::array<int, 4>> &layers,
               const std::vector<SeedQuad> &quads,
               const std::vector<float> &scores,
               int min_shared,
               std::vector<char> &keep,
               std::vector<int> *n_dropped = nullptr);

  private:
    SeedEventOfHits m_hits;
    std::unique_ptr<SeedChainFinder> m_finder[2];

    // the cleaning's buffers, kept across events: allocated afresh, their first touch cost as much as
    // the cleaning itself. m_cl_hl: per global hit (layer offset + hit), the list of kept quads using it
    // and its length; all entries are back at {-1, 0} between events.
    struct CleanHL {
      int head = -1, len = 0;
    };
    std::vector<CleanHL> m_cl_hl;
    std::vector<std::pair<int, int>> m_cl_link;  // (kept quad, next entry)
    std::vector<unsigned int> m_cl_key, m_cl_rank;
    radix_sort<unsigned int, unsigned int> m_cl_sort;
    std::vector<std::array<unsigned int, 4>> m_cl_gh;
  };

}  // namespace mkfit

#endif
