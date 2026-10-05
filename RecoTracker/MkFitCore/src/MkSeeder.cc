#include "RecoTracker/MkFitCore/interface/MkSeeder.h"
#include "RecoTracker/MkFitCore/interface/SeedChain.h"
#include "RecoTracker/MkFitCore/src/SeedChainFinder.h"

#include <algorithm>

namespace mkfit {

  MkSeeder::MkSeeder() {
    m_finder[0] = std::make_unique<SeedChainFinder>();
    m_finder[1] = std::make_unique<SeedChainFinder>();
  }

  MkSeeder::~MkSeeder() = default;

  void MkSeeder::setup(SeedChain &plus, SeedChain &minus) {
    m_finder[0]->setup(plus);
    m_finder[1]->setup(minus);
  }

  void MkSeeder::fill(const std::vector<HitVec> &layer_hits, const BeamSpot &bs) { m_hits.fill(layer_hits, bs); }

  void MkSeeder::fill(const SeedHitSource &src, const BeamSpot &bs) { m_hits.fill(src, bs); }

  void MkSeeder::find(std::vector<std::pair<std::array<int, 4>, SeedQuad>> &out,
                      SeedCounters &cnt,
                      std::vector<float> *scores,
                      std::vector<float> *fake_scores) {
    SeedChainFinder::run_both(*m_finder[0], *m_finder[1], m_hits.layer_map(), out, cnt, scores, fake_scores);
  }

  void MkSeeder::clean(const std::vector<HitVec> &layer_hits,
                       const std::vector<std::array<int, 4>> &layers,
                       const std::vector<SeedQuad> &quads,
                       const std::vector<float> &scores,
                       int min_shared,
                       std::vector<char> &keep,
                       std::vector<int> *n_dropped) {
    clean(seed_hit_source(layer_hits), layers, quads, scores, min_shared, keep, n_dropped);
  }

  void MkSeeder::clean(const SeedHitSource &src,
                       const std::vector<std::array<int, 4>> &layers,
                       const std::vector<SeedQuad> &quads,
                       const std::vector<float> &scores,
                       int min_shared,
                       std::vector<char> &keep,
                       std::vector<int> *n_dropped) {
    const int nc = quads.size();
    keep.assign(nc, 1);
    if (n_dropped)
      n_dropped->assign(nc, 0);
    // the order: tier first (a pattern with an outer-tracker layer after every pure-pixel one: its
    // windows are several times wider, so its scores are not comparable), then the score, packed in
    // one 32-bit key for the (stable) radix sort: the tier in the top 3 bits, then the score's float
    // bits without their 3 lowest (the score is >= 0, so its bits sort as the value; relative 1e-6)
    std::vector<unsigned int> &key = m_cl_key, &rank = m_cl_rank;
    key.resize(nc);
    for (int i = 0; i < nc; ++i) {
      int t = 0;
      for (int l : layers[i])
        t += !SeedLayerEnvelopes::is_pix(l);
      const float sf = scores[i];
      unsigned int sb;
      __builtin_memcpy(&sb, &sf, 4);
      key[i] = (unsigned int)t << 29 | sb >> 3;
    }
    m_cl_sort.sort(key, rank);
    // each quad's hits as sorted global indices: the layer's offset + the hit's position within the layer.
    // Which quads are kept does not depend on the numbering, but which kept quad a dropped one is charged
    // to (n_dropped) does, through ties; so a layer given by index (SeedLayerHits::idx) is numbered by
    // position too, as if it had a HitVec of its own. m_cl_pos: the position of each external index.
    const int nl = src.size();
    std::vector<unsigned int> off(nl + 1, 0);
    for (int l = 0; l < nl; ++l)
      off[l + 1] = off[l] + src[l].n;
    std::vector<const unsigned int *> pos_of(nl, nullptr);
    {
      std::vector<std::pair<const HitVec *, size_t>> base;  // per shared HitVec: its start in m_cl_pos
      size_t total = 0;
      for (int l = 0; l < nl; ++l)
        if (src[l].idx) {
          bool found = false;
          for (const auto &b : base)
            found |= b.first == src[l].hits;
          if (!found)
            base.push_back({src[l].hits, total}), total += src[l].hits->size();
        }
      m_cl_pos.resize(total);
      for (int l = 0; l < nl; ++l)
        if (src[l].idx) {
          size_t b0 = 0;
          for (const auto &b : base)
            if (b.first == src[l].hits)
              b0 = b.second;
          unsigned int *p = m_cl_pos.data() + b0;
          for (unsigned int k = 0; k < src[l].n; ++k)
            p[src[l].idx[k]] = k;
          pos_of[l] = p;
        }
    }
    std::vector<std::array<unsigned int, 4>> &gh = m_cl_gh;
    gh.resize(nc);
    for (int i = 0; i < nc; ++i) {
      const auto &ll = layers[i];
      for (int k = 0; k < 4; ++k)
        gh[i][k] = off[ll[k]] + (pos_of[ll[k]] ? pos_of[ll[k]][quads[i][k]] : quads[i][k]);
      std::sort(gh[i].begin(), gh[i].end());
    }
    std::vector<CleanHL> &hl = m_cl_hl;
    if (hl.size() < off[nl])
      hl.resize(off[nl]);
    std::vector<std::pair<int, int>> &link = m_cl_link;
    link.clear();
    // a kept quad sharing >= N of the 4 hits holds at least one of any 5 - N of them: walk the
    // 5 - N shortest lists only, and count the shared hits of each quad found there directly
    const int nwalk = std::clamp(5 - min_shared, 1, 4);
    for (int ik = 0; ik < nc; ++ik) {
      // the order is by score, so every access below is random: prefetch the hit lists 8 quads ahead
      // and their hit indices 16 ahead (the loop is latency-bound, ~135 ns per quad without)
      if (ik + 16 < nc)
        __builtin_prefetch(&gh[rank[ik + 16]]);
      if (ik + 8 < nc)
        for (unsigned int gk : gh[rank[ik + 8]])
          __builtin_prefetch(&hl[gk]);
      const int i = rank[ik];
      const auto &g = gh[i];
      // the positions by list length: rank each (ties by position), no branches
      const int ln[4] = {hl[g[0]].len, hl[g[1]].len, hl[g[2]].len, hl[g[3]].len};
      int pos[4];
      for (int a = 0; a < 4; ++a) {
        int r = 0;
        for (int b = 0; b < 4; ++b)
          r += ln[b] < ln[a] || (ln[b] == ln[a] && b < a);
        pos[r] = a;
      }
      bool drop = false;
      for (int w = 0; w < nwalk && !drop; ++w)
        for (int e = hl[g[pos[w]]].head; e >= 0; e = link[e].second) {
          const auto &h = gh[link[e].first];
          // shared hits: all 16 pairs, no branches (the hits of one quad are distinct)
          int ns = 0;
          for (int a = 0; a < 4; ++a)
            for (int b = 0; b < 4; ++b)
              ns += g[a] == h[b];
          if (ns >= min_shared) {
            drop = true;
            if (n_dropped)
              ++(*n_dropped)[link[e].first];
            break;
          }
        }
      if (drop) {
        keep[i] = 0;
        continue;
      }
      for (int k = 0; k < 4; ++k) {
        CleanHL &h = hl[g[k]];
        link.push_back({i, h.head});
        h.head = link.size() - 1;
        ++h.len;
      }
    }
    // back to empty, for the next event
    for (const auto &e : link)
      for (unsigned int gk : gh[e.first])
        hl[gk] = CleanHL();
  }

}  // namespace mkfit
