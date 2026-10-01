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
      : m_id(id_),
        m_disc(!li.is_barrel()),
        m_qbar_lo(li.is_barrel() ? li.rin() : li.zmin()),
        m_qbar_hi(li.is_barrel() ? li.rout() : li.zmax()),
        m_q_lo(li.is_barrel() ? li.zmin() : li.rin()),
        m_q_hi(li.is_barrel() ? li.zmax() : li.rout()),
        m_qbar_lo_nom(m_qbar_lo),
        m_qbar_hi_nom(m_qbar_hi),
        m_q_lo_nom(m_q_lo),
        m_q_hi_nom(m_q_hi),
        m_ax_phi(-Const::PI, Const::PI),
        m_ax_q((float)m_q_lo, (float)m_q_hi, nq(m_q_lo, m_q_hi, qbin)),
        m_binnor(m_ax_phi, m_ax_q, true, false) {}

  void SeedLayerOfHits::fill(const HitVec &hits) {
    // the registration is LayerOfHits::suckInHits()'s
    m_n = hits.size();
    m_binnor.reset_contents();
    m_binnor.begin_registration(m_n);
    for (unsigned int i = 0; i < m_n; ++i)
      m_binnor.register_entry_safe(hits[i].phi(), m_disc ? hits[i].r() : hits[i].z());
    m_binnor.finalize_registration();

    m_phi.resize(m_n);
    m_z.resize(m_n);
    m_r.resize(m_n);
    m_invr.resize(m_n);
    m_x.resize(m_n);
    m_y.resize(m_n);
    m_orig.resize(m_n);
    for (unsigned int i = 0; i < m_n; ++i) {
      const unsigned int j = m_binnor.m_ranks[i];
      const Hit &h = hits[j];
      m_phi[i] = h.phi();
      m_z[i] = h.z();
      m_r[i] = h.r();
      m_invr[i] = 1.0f / m_r[i];
      // float r times float cos(phi), so a finder reads the same value it would compute
      m_x[i] = m_r[i] * std::cos(m_phi[i]);
      m_y[i] = m_r[i] * std::sin(m_phi[i]);
      m_orig[i] = j;
    }

    const unsigned int nb = n_phi_bins() * n_q_bins();
    m_start.assign(nb + 1, 0);
    for (unsigned int k = 0; k < nb; ++k)
      m_start[k + 1] = m_start[k] + m_binnor.m_bins[k].count;

    m_pr.resize(m_with_double ? m_n : 0);
    m_pphi.resize(m_with_double ? m_n : 0);
    m_span.resize(m_n);
    m_qbar_lo = m_qbar_lo_nom, m_qbar_hi = m_qbar_hi_nom, m_q_lo = m_q_lo_nom, m_q_hi = m_q_hi_nom;
    if (m_with_double)
      for (unsigned int k = 0; k < m_n; ++k) {
        m_pr[k] = std::hypot((double)m_x[k], (double)m_y[k]);
        m_pphi[k] = std::atan2((double)m_y[k], (double)m_x[k]);
      }
    for (unsigned int k = 0; k < m_n; ++k) {
      m_span[k] = hits[m_orig[k]].spanCols();
      const double u = qbar(k), v = q(k);
      m_qbar_lo = std::min(m_qbar_lo, u), m_qbar_hi = std::max(m_qbar_hi, u);
      m_q_lo = std::min(m_q_lo, v), m_q_hi = std::max(m_q_hi, v);
    }
  }

  //==============================================================================
  // SeedEventOfHits
  //==============================================================================

  void SeedEventOfHits::add_layer(int l, const LayerInfo &li, double qbin) {
    if (m_layers.count(l))
      return;
    m_layers[l] = std::make_unique<SeedLayerOfHits>(l, li, qbin);
    m_map[l] = m_layers[l].get();
  }

  void SeedEventOfHits::set_with_double(bool wd) {
    for (auto &kv : m_layers)
      kv.second->m_with_double = wd;
  }

  void SeedEventOfHits::fill(const std::vector<HitVec> &layer_hits) {
    for (auto &kv : m_layers)
      kv.second->fill(layer_hits[kv.first]);
  }

}  // namespace mkfit
