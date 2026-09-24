#include "RecoTracker/MkFitCore/interface/HitStructures.h"

#include "RecoTracker/MkFitCore/interface/IterationConfig.h"
#include "Matriplex/Memory.h"

#include "Debug.h"
#include <algorithm>

namespace mkfit {

  void LayerOfHits::Initializator::setup(float qmin, float qmax, float dq) {
    assert(qmax > qmin);
    float extent = qmax - qmin;
    m_nq = std::ceil(extent / dq);
    float extra = 0.5f * (m_nq * dq - extent);
    m_qmin = qmin - extra;
    m_qmax = qmax + extra;
  }

  LayerOfHits::Initializator::Initializator(const LayerInfo &li, float qmin, float qmax, unsigned int nq)
      : m_linfo(li), m_qmin(qmin), m_qmax(qmax), m_nq(nq) {}

  LayerOfHits::Initializator::Initializator(const LayerInfo &li, float qmin, float qmax, float dq) : m_linfo(li) {
    setup(qmin, qmax, dq);
  }

  LayerOfHits::Initializator::Initializator(const LayerInfo &li) : m_linfo(li) {
    if (li.is_barrel())
      setup(li.zmin(), li.zmax(), li.q_bin());
    else
      setup(li.rin(), li.rout(), li.q_bin());
  }

  LayerOfHits::LayerOfHits(const LayerOfHits::Initializator &i)
      : m_ax_phi(-Const::PI, Const::PI),
        m_ax_eta(i.m_qmin, i.m_qmax, i.m_nq),
        m_binnor(m_ax_phi, m_ax_eta, true, false)  // yes-radix, no-keep-cons
  {
    m_layer_info = &i.m_linfo;
    m_is_barrel = m_layer_info->is_barrel();

    m_dead_bins.resize(m_ax_eta.size_of_N() * m_ax_phi.size_of_N());
  }

  LayerOfHits::~LayerOfHits() {
#ifdef COPY_SORTED_HITS
    free_hits();
#endif
  }

#ifdef COPY_SORTED_HITS
  void LayerOfHits::alloc_hits(int size) {
    m_hits = (Hit *)Matriplex::aligned_alloc64(sizeof(Hit) * size);
    m_capacity = size;
    for (int ihit = 0; ihit < m_capacity; ihit++) {
      m_hits[ihit] = Hit();
    }
  }

  void LayerOfHits::free_hits() { std::free(m_hits); }
#endif

  //==============================================================================

  void LayerOfHits::reset() {
    m_hit_infos.clear();
    m_ext_idcs.clear();
    m_min_ext_idx = std::numeric_limits<unsigned int>::max();
    m_max_ext_idx = std::numeric_limits<unsigned int>::min();
    m_n_hits = 0;
    m_binnor.reset_contents();
  }

  //==============================================================================

  void LayerOfHits::suckInHits(const HitVec &hitv) {
    m_ext_hits = &hitv;
    m_n_hits = hitv.size();

    m_binnor.begin_registration(m_n_hits);

#ifdef COPY_SORTED_HITS
    if (m_capacity < m_n_hits) {
      free_hits();
      alloc_hits(m_n_hits);
    }
#endif

    std::vector<HitInfo> hinfos;
    if (Config::usePhiQArrays) {
      hinfos.reserve(m_n_hits);
      m_hit_infos.reserve(m_n_hits);
    }

    // Factor to get from hit sigma to half-length in q direction.
    const float hl_fac = is_pixel() ? 3.0f : std::sqrt(3.0f);

    for (unsigned int i = 0; i < m_n_hits; ++i) {
      const Hit &h = hitv[i];

      float phi = h.phi();
      float q = m_is_barrel ? h.z() : h.r();

      m_binnor.register_entry_safe(phi, q);

      if (Config::usePhiQArrays) {
        float half_length, qbar;
        if (m_is_barrel) {
          half_length = hl_fac * std::sqrt(h.ezz());
          qbar = h.r();
        } else {
          half_length = hl_fac * std::sqrt(h.exx() + h.eyy());
          qbar = h.z();
        }
        m_max_q_half_length = std::max(m_max_q_half_length, half_length);
        hinfos.emplace_back(HitInfo({phi, q, half_length, qbar}));
      }
    }

    m_binnor.finalize_registration();

    for (unsigned int i = 0; i < m_n_hits; ++i) {
      unsigned int j = m_binnor.m_ranks[i];
#ifdef COPY_SORTED_HITS
      memcpy(&m_hits[i], &hitv[j], sizeof(Hit));
#endif
      if (Config::usePhiQArrays) {
        m_hit_infos.emplace_back(hinfos[j]);
      }
    }
  }

  //==============================================================================

  void LayerOfHits::suckInDeads(const DeadVec &deadv) {
    m_dead_bins.assign(m_dead_bins.size(), false);

    for (const auto &d : deadv) {
      bin_index_t q_bin_1 = qBinChecked(d.q1);
      bin_index_t q_bin_2 = qBinChecked(d.q2) + 1;
      bin_index_t phi_bin_1 = phiBin(d.phi1);
      bin_index_t phi_bin_2 = phiMaskApply(phiBin(d.phi2) + 1);

      for (bin_index_t q_bin = q_bin_1; q_bin != q_bin_2; q_bin++) {
        const unsigned int qoff = q_bin * m_ax_phi.size_of_N();
        for (bin_index_t pb = phi_bin_1; pb != phi_bin_2; pb = phiMaskApply(pb + 1)) {
          m_dead_bins[qoff + pb] = true;
        }
      }
    }
  }

  //==============================================================================

  void LayerOfHits::beginRegistrationOfHits(const HitVec &hitv) {
    m_ext_hits = &hitv;
    m_n_hits = 0;

    m_binnor.begin_registration(128);  // initial reserve for cons vectors
  }

  void LayerOfHits::registerHit(unsigned int idx) {
    const Hit &h = (*m_ext_hits)[idx];

    m_ext_idcs.push_back(idx);
    m_min_ext_idx = std::min(m_min_ext_idx, idx);
    m_max_ext_idx = std::max(m_max_ext_idx, idx);

    float phi = h.phi();
    float q = m_is_barrel ? h.z() : h.r();

    m_binnor.register_entry_safe(phi, q);

    if (Config::usePhiQArrays) {
      // Factor to get from hit sigma to half-length in q direction.
      //
      // Why sqrt(3) for strips: for a segment of half-length L along unit vector u
      // with the crossing point uniform along it, the covariance is (L^2/3) u u^T.
      // So ezz = L^2 u_z^2 / 3 and sqrt(3 * ezz) = L * |u_z| -- exactly the strip's
      // projection onto global z. The module tilt is therefore ALREADY folded in
      // here, via the covariance; this is why the measured TBPS half-lengths come
      // out 0.36 / 0.44 / 0.59 cm against a nominal 1.2 (|zdir_z| = 0.87 / 0.82 /
      // 0.69, i.e. tilt ~60 deg at the inner pair), and why TOB 2S -- flat -- comes
      // out 2.5125 against a nominal 2.5. Both are correct, not discrepancies.
      //
      // Module frame convention: xdir is perpendicular to the strips (essentially
      // azimuthal), ydir runs along them, zdir is the normal -- so ydir and zdir
      // both lie in the (r,z) plane. Hence strip length feeds z and r but NOT phi,
      // and the across-strip pitch feeds phi only. The barrel (phi, q=z) window
      // axes are therefore well aligned with the module frame in both directions.
      //
      // Endcap caveat: exx + eyy below is the TRACE of the transverse block, which
      // is rotation-invariant and equals sigma_r^2 + r^2 sigma_phi^2 -- i.e. the
      // radial variance PLUS the azimuthal one, where the barrel branch uses the
      // single element ezz (a true marginal). The strict analogue would be
      // err = (x^2 exx + 2 xy exy + y^2 eyy) / r^2. Harmless for radial 2S strips
      // (sigma_r^2 = L^2/3 ~ 2.08 cm^2 swamps p^2/12 ~ 7e-6), but for an endcap
      // PIXEL sigma_x ~ sigma_y, so sqrt(exx+eyy) = sigma*sqrt(2) and with
      // hl_fac = 3 this yields 4.24 sigma where 3 sigma was intended -- a sqrt(2)
      // over-estimate. The measured endcap-pixel dq/sigma_tot of 0.89-1.25,
      // dipping below 1, is exactly that signature. See RecoTracker/CLAUDE.md.
      const float hl_fac = is_pixel() ? 3.0f : std::sqrt(3.0f);
      float half_length, qbar;
      if (m_is_barrel) {
        half_length = hl_fac * std::sqrt(h.ezz());
        qbar = h.r();
      } else {
        half_length = hl_fac * std::sqrt(h.exx() + h.eyy());
        qbar = h.z();
      }
      m_max_q_half_length = std::max(m_max_q_half_length, half_length);
      m_hit_infos.emplace_back(HitInfo({phi, q, half_length, qbar}));
    }
  }

  void LayerOfHits::endRegistrationOfHits(bool build_original_to_internal_map) {
    m_n_hits = m_ext_idcs.size();
    if (m_n_hits == 0)
      return;

    m_binnor.finalize_registration();

    // copy q/phi

#ifdef COPY_SORTED_HITS
    if (m_capacity < m_n_hits) {
      free_hits();
      alloc_hits(m_n_hits);
    }
#endif

    std::vector<HitInfo> hinfos;
    if (Config::usePhiQArrays) {
      hinfos.swap(m_hit_infos);
      m_hit_infos.reserve(m_n_hits);
    }

    for (unsigned int i = 0; i < m_n_hits; ++i) {
      unsigned int j = m_binnor.m_ranks[i];  // index in intermediate
      unsigned int k = m_ext_idcs[j];        // index in external hit_vec

#ifdef COPY_SORTED_HITS
      memcpy(&m_hits[i], &hitv[k], sizeof(Hit));
#endif

      if (Config::usePhiQArrays) {
        m_hit_infos.emplace_back(hinfos[j]);
      }

      // Redirect m_binnor.m_ranks[i] to point to external/original index.
      m_binnor.m_ranks[i] = k;
    }

    if (build_original_to_internal_map) {
      if (m_max_ext_idx - m_min_ext_idx + 1 > 8 * m_n_hits) {
        // If this happens we might:
        // a) Use external indices for everything. -- *** We are now. ***
        // b) Build these maps for seeding layers only.
        // c) Have a flag in hit-on-track that tells us if the hit index has been remapped,
        //    essentially, if it is a seed hit. This might be smart anyway.
        //    One could use index < -256 or something similar.

        printf(
            "LayerOfHits::endRegistrationOfHits() original_to_internal index map vector is largish: m_n_hits=%d, "
            "map_vector_size=%d\n",
            m_n_hits,
            m_max_ext_idx - m_min_ext_idx + 1);
      }

      m_ext_idcs.resize(m_max_ext_idx - m_min_ext_idx + 1);
      for (unsigned int i = 0; i < m_n_hits; ++i) {
        m_ext_idcs[m_hit_ranks[i] - m_min_ext_idx] = i;
      }
    }

    // We can release m_hit_infos and, if not used, also m_ext_idcs -- and realloc them
    // on next beginRegistration().
    // If binnor had keep_cons on we could use it for pre-selection in selectHitIndices()
    // instead of q and phi arrays -- assuming sufficient precision can be achieved..
  }

  void LayerOfHits::printBins() {
    for (bin_index_t qb = 0; qb <= m_ax_eta.m_last_N_bin; ++qb) {
      printf("%c bin %d\n", is_barrel() ? 'Z' : 'R', qb);
      for (bin_index_t pb = 0; pb <= m_ax_phi.m_last_N_bin; ++pb) {
        if (pb % 8 == 0)
          printf(" Phi %4d: ", pb);
        auto content = m_binnor.get_content(pb, qb);
        printf("%5d,%4d   %s", content.first, content.count, ((pb + 1) % 8 == 0) ? "\n" : "");
      }
    }
  }

  //==============================================================================
  // EventOfHits
  //==============================================================================

  EventOfHits::EventOfHits(const TrackerInfo &trk_inf) : m_n_layers(trk_inf.n_layers()) {
    m_layers_of_hits.reserve(trk_inf.n_layers());
    for (int ii = 0; ii < trk_inf.n_layers(); ++ii) {
      m_layers_of_hits.emplace_back(LayerOfHits::Initializator(trk_inf.layer(ii)));
    }
  }

}  // end namespace mkfit
