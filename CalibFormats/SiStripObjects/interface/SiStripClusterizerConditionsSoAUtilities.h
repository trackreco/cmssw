#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAUtilities_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAUtilities_h

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

struct SiStripClusterizerConditionsSoA {
  const static auto stripsPerFedCh = sistrip::STRIPS_PER_FEDCH;
  const static auto apvsPerFedCh = sistrip::APVS_PER_FEDCH;
  using arraySTRIPS_PER_FEDCH = std::array<std::uint16_t, stripsPerFedCh>;
  using arrayAPVS_PER_FEDCH = std::array<float, apvsPerFedCh>;
  GENERATE_SOA_LAYOUT(SiStripClusterizerConditionsSoALayout,
                      SOA_COLUMN(arraySTRIPS_PER_FEDCH, noise),
                      SOA_COLUMN(float, invthick),
                      SOA_COLUMN(stripgpu::detId_t, detID),
                      SOA_COLUMN(stripgpu::apvPair_t, iPair),
                      SOA_COLUMN(arrayAPVS_PER_FEDCH, gain));
};

using SiStripClusterizerConditionsLayout =
    typename SiStripClusterizerConditionsSoA::SiStripClusterizerConditionsSoALayout<>;
using SiStripClusterizerConditionsView =
    typename SiStripClusterizerConditionsSoA::SiStripClusterizerConditionsSoALayout<>::View;
using SiStripClusterizerConditionsConstView =
    typename SiStripClusterizerConditionsSoA::SiStripClusterizerConditionsSoALayout<>::ConstView;

class SiStripQuality;
class SiStripGain;
class SiStripNoises;

namespace stripgpu {
  static constexpr std::uint16_t badBit = 1 << 15;
  __host__ __device__ inline fedId_t fedIndexHD(fedId_t fed) { return fed - sistrip::FED_ID_MIN; }
  __host__ __device__ inline std::uint32_t stripIndexHD(stripId_t strip) { return (strip % sistrip::STRIPS_PER_FEDCH); }
  __host__ __device__ inline std::uint32_t apvIndexHD(stripId_t strip) {
    return (strip % sistrip::STRIPS_PER_FEDCH) / sistrip::STRIPS_PER_APV;
  }
  __host__ __device__ inline std::uint32_t channelIndexHD(fedId_t fed, fedCh_t channel) {
    return fedIndexHD(fed) * sistrip::FEDCH_PER_FED + channel;
  }

}  // namespace stripgpu

#endif
