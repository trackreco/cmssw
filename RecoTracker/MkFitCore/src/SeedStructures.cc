#include "RecoTracker/MkFitCore/interface/SeedStructures.h"

#include <algorithm>
#include <cmath>

namespace mkfit {

  //==============================================================================
  // SeedLayerOfHits
  //==============================================================================

  unsigned int SeedLayerOfHits::nq(double lo, double hi, double bin) {
    return std::max(1u, (unsigned int)std::ceil((hi - lo) / bin));
  }

  SeedLayerOfHits::SeedLayerOfHits(int id_, const LayerInfo &li, double qbin)
      : id(id_),
        disc(!li.is_barrel()),
        qbar_lo(li.is_barrel() ? li.rin() : li.zmin()),
        qbar_hi(li.is_barrel() ? li.rout() : li.zmax()),
        q_lo(li.is_barrel() ? li.zmin() : li.rin()),
        q_hi(li.is_barrel() ? li.zmax() : li.rout()),
        qbar_lo_nom(qbar_lo),
        qbar_hi_nom(qbar_hi),
        q_lo_nom(q_lo),
        q_hi_nom(q_hi),
        ax_phi_(-Const::PI, Const::PI),
        ax_q_((float)q_lo, (float)q_hi, nq(q_lo, q_hi, qbin)),
        binnor_(ax_phi_, ax_q_, true, false) {}

  void SeedLayerOfHits::fill(const HitVec &hits) {
    // the registration is LayerOfHits::suckInHits()'s
    n_ = hits.size();
    binnor_.reset_contents();
    binnor_.begin_registration(n_);
    for (unsigned int i = 0; i < n_; ++i)
      binnor_.register_entry_safe(hits[i].phi(), disc ? hits[i].r() : hits[i].z());
    binnor_.finalize_registration();

    phi_.resize(n_);
    z_.resize(n_);
    r_.resize(n_);
    invr_.resize(n_);
    x_.resize(n_);
    y_.resize(n_);
    orig_.resize(n_);
    for (unsigned int i = 0; i < n_; ++i) {
      const unsigned int j = binnor_.m_ranks[i];
      const Hit &h = hits[j];
      phi_[i] = h.phi();
      z_[i] = h.z();
      r_[i] = h.r();
      invr_[i] = 1.0f / r_[i];
      // float r times float cos(phi), so a finder reads the same value it would compute
      x_[i] = r_[i] * std::cos(phi_[i]);
      y_[i] = r_[i] * std::sin(phi_[i]);
      orig_[i] = j;
    }

    const unsigned int nb = n_phi_bins() * n_q_bins();
    start_.assign(nb + 1, 0);
    for (unsigned int k = 0; k < nb; ++k)
      start_[k + 1] = start_[k] + binnor_.m_bins[k].count;

    pr_.resize(with_double ? n_ : 0);
    pphi_.resize(with_double ? n_ : 0);
    span_.resize(n_);
    qbar_lo = qbar_lo_nom, qbar_hi = qbar_hi_nom, q_lo = q_lo_nom, q_hi = q_hi_nom;
    if (with_double)
      for (unsigned int k = 0; k < n_; ++k) {
        pr_[k] = std::hypot((double)x_[k], (double)y_[k]);
        pphi_[k] = std::atan2((double)y_[k], (double)x_[k]);
      }
    for (unsigned int k = 0; k < n_; ++k) {
      span_[k] = hits[orig_[k]].spanCols();
      const double u = qbar(k), v = q(k);
      qbar_lo = std::min(qbar_lo, u), qbar_hi = std::max(qbar_hi, u);
      q_lo = std::min(q_lo, v), q_hi = std::max(q_hi, v);
    }
  }

  //==============================================================================
  // SeedEventOfHits
  //==============================================================================

  void SeedEventOfHits::add_layer(int l, const LayerInfo &li, double qbin) {
    if (layers_.count(l))
      return;
    layers_[l] = std::make_unique<SeedLayerOfHits>(l, li, qbin);
    map_[l] = layers_[l].get();
  }

  void SeedEventOfHits::set_with_double(bool wd) {
    for (auto &kv : layers_)
      kv.second->with_double = wd;
  }

  void SeedEventOfHits::fill(const std::vector<HitVec> &layer_hits) {
    for (auto &kv : layers_)
      kv.second->fill(layer_hits[kv.first]);
  }

}  // namespace mkfit
