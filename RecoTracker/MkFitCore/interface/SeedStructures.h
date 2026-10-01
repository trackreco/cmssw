#ifndef RecoTracker_MkFitCore_interface_SeedStructures_h
#define RecoTracker_MkFitCore_interface_SeedStructures_h

// Hit storage of the mkFit seeder (MkSeeder): one SeedLayerOfHits per layer
// the seeder visits, held by a SeedEventOfHits.
//
// The hits are the same per-layer HitVecs that EventOfHits takes; a seed
// refers to them by layer and ORIGINAL index in that HitVec (SeedQuad), so a
// seed's hits are its HitOnTrack entries as they stand.
//
// Like LayerOfHits, a layer is binned by the binnor in (phi, q), q = z in the
// barrel and r on the discs, and every hit cached in BIN ORDER. It differs in
// the binning (2^11 phi bins, against 2^8; the q bin width is the caller's)
// and in the cache: struct-of-arrays of phi, z, r, 1/r, x, y, the cluster
// length along z, and the r and phi in double for the double-precision
// reference finder. A range of phi bins within one q bin is one contiguous
// run of the arrays, and start_[] is a CSR over the bins in that order.

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/Hit.h"
#include "RecoTracker/MkFitCore/interface/TrackerInfo.h"
#include "RecoTracker/MkFitCore/interface/binnor.h"

#include <array>
#include <map>
#include <memory>
#include <vector>

namespace mkfit {

  // Per-stage counts of one seeding run.
  struct SeedCounters {
    long doublets = 0, c_touched = 0, triplets = 0, d_touched = 0, quads = 0;
    double t_stage[7] = {0, 0, 0, 0, 0, 0, 0};  // s, staged finder only
    void add(const SeedCounters &o) {
      for (int i = 0; i < 7; ++i)
        t_stage[i] += o.t_stage[i];
      doublets += o.doublets;
      c_touched += o.c_touched;
      triplets += o.triplets;
      d_touched += o.d_touched;
      quads += o.quads;
    }
  };

  // (ia, ib, ic, id): ORIGINAL hit indices within each layer's HitVec.
  using SeedQuad = std::array<unsigned int, 4>;

  //==============================================================================
  // SeedLayerOfHits
  //==============================================================================

  class SeedLayerOfHits {
  public:
    // phi N-bins: 2^11 = 2048, 3.07 mrad (measured: 11 and 12 bits equal, 10 slower)
    static constexpr int c_phi_nbits = 11;
    using AxPhi = axis_pow2_u1<float, unsigned short, 16, c_phi_nbits>;
    using AxQ = axis<float, unsigned short, 16, 8>;
    using binnor_t = binnor<unsigned int, AxPhi, AxQ, 18, 14>;

    // qbin: the q bin width in cm
    SeedLayerOfHits(int id_, const LayerInfo &li, double qbin);

    void fill(const HitVec &hits);

    // the hit's own qbar and q
    double qbar(unsigned int k) const { return disc ? z_[k] : r_[k]; }
    double q(unsigned int k) const { return disc ? r_[k] : z_[k]; }

    // [begin, end) of the hits in q N-bin qi and phi N-bins [p1, p2), NO wrap.
    unsigned int run_begin(unsigned int qi, unsigned int p1) const { return start_[qi * n_phi_bins() + p1]; }
    unsigned int run_end(unsigned int qi, unsigned int p2) const { return start_[qi * n_phi_bins() + p2]; }

    // Calls f(i) for every hit in phi N-bins [p.begin, p.end) (half-open, may
    // wrap) and q N-bins [q.begin, q.end).  A range with begin == end on the
    // circle is EMPTY: callers clamp their half-width below pi first.
    template <typename F>
    void for_each_in(typename AxPhi::I_pair p, typename AxQ::I_pair q, F &&f) const {
      const unsigned int np = n_phi_bins();
      for (unsigned int qi = q.begin; qi < q.end; ++qi) {
        if (p.begin < p.end || p.end == 0) {
          const unsigned int e = p.end == 0 ? np : p.end;
          for (unsigned int i = run_begin(qi, p.begin); i < run_end(qi, e); ++i)
            f(i);
        } else if (p.begin > p.end) {
          for (unsigned int i = run_begin(qi, p.begin); i < run_end(qi, np); ++i)
            f(i);
          for (unsigned int i = run_begin(qi, 0); i < run_end(qi, p.end); ++i)
            f(i);
        }
      }
    }

    // As for_each_in(), but hands over whole contiguous runs, f(begin, end).
    template <typename F>
    void for_each_run(typename AxPhi::I_pair p, typename AxQ::I_pair q, F &&f) const {
      const unsigned int np = n_phi_bins();
      for (unsigned int qi = q.begin; qi < q.end; ++qi) {
        if (p.begin < p.end || p.end == 0) {
          f(run_begin(qi, p.begin), run_end(qi, p.end == 0 ? np : p.end));
        } else if (p.begin > p.end) {
          f(run_begin(qi, p.begin), run_end(qi, np));
          f(run_begin(qi, 0), run_end(qi, p.end));
        }
      }
    }

    // lo, hi may lie outside (-pi, pi]; wrap them first, since the axis floors
    // (r - R_min) * fac straight into an unsigned bin index.
    typename AxPhi::I_pair phi_range(float lo, float hi) const {
      return ax_phi_.from_R_minmax_to_N_bins(wrap(lo), wrap(hi));
    }
    static float wrap(float p) {
      if (p > Const::PI)
        p -= 2 * Const::PI;
      else if (p < -Const::PI)
        p += 2 * Const::PI;
      return p;
    }
    typename AxQ::I_pair q_range(float lo, float hi) const { return ax_q_.from_R_minmax_to_N_bins(lo, hi); }
    typename AxQ::I_pair q_all() const { return typename AxQ::I_pair(0, n_q_bins()); }

    unsigned int n_phi_bins() const { return ax_phi_.size_of_N(); }
    unsigned int n_q_bins() const { return ax_q_.size_of_N(); }
    float q_min() const { return ax_q_.m_R_min; }
    float q_max() const { return ax_q_.m_R_max; }
    unsigned int n() const { return n_; }

    const binnor_t &binnor_ref() const { return binnor_; }

    // the layer
    int id = -1;
    bool disc = false;
    // r range (barrel) or z range (disc), and z range (barrel) or r range (disc): the LayerInfo extent,
    // widened in fill() to hold every hit of the event. The pixel barrel's LayerInfo::rin() lay inside
    // its inner shell before the rin fix (layer 0: 2.868 cm, hits from 2.750), so ~half its hits had r
    // below it, and a fetch over the nominal extent could miss them.
    double qbar_lo = 0, qbar_hi = 0;
    double q_lo = 0, q_hi = 0;
    double qbar_lo_nom = 0, qbar_hi_nom = 0, q_lo_nom = 0, q_hi_nom = 0;

    // struct-of-arrays, bin order
    std::vector<float> phi_, z_, r_, invr_, x_, y_;
    std::vector<unsigned int> orig_;
    // the cluster's length in columns (Hit::spanCols()): along z in the barrel pixels
    std::vector<int> span_;
    // r and phi of each hit as the double-precision finders compute them from the float x, y: only
    // with with_double; without it the finders take the float r and phi
    std::vector<double> pr_, pphi_;
    bool with_double = true;

  private:
    static unsigned int nq(double lo, double hi, double bin);

    AxPhi ax_phi_;
    AxQ ax_q_;
    binnor_t binnor_;
    std::vector<unsigned int> start_;
    unsigned int n_ = 0;
  };

  //==============================================================================
  // SeedEventOfHits
  //==============================================================================

  // The seeder's layers, by mkFit layer id; a layer is added once and filled per event.
  class SeedEventOfHits {
  public:
    // a no-op if the layer is there already
    void add_layer(int l, const LayerInfo &li, double qbin);
    bool has(int l) const { return layers_.count(l); }
    const SeedLayerOfHits *layer(int l) const {
      auto it = layers_.find(l);
      return it == layers_.end() ? nullptr : it->second.get();
    }
    // every layer, by id
    const std::map<int, const SeedLayerOfHits *> &layer_map() const { return map_; }

    void set_with_double(bool wd);
    // layer_hits: the event's HitVecs, indexed by mkFit layer id
    void fill(const std::vector<HitVec> &layer_hits);

  private:
    std::map<int, std::unique_ptr<SeedLayerOfHits>> layers_;
    std::map<int, const SeedLayerOfHits *> map_;
  };

}  // namespace mkfit

#endif
