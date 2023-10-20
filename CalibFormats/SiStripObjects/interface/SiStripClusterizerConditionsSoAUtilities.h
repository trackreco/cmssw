#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAUtilities_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAUtilities_h

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"
#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

//struct DetToFedSoA {
//  GENERATE_SOA_LAYOUT(DetToFedSoALayout,
//		                  SOA_COLUMN(stripgpu::detId_t, detID_),
//                      SOA_COLUMN(stripgpu::apvPair_t, pair_),
//                      SOA_COLUMN(stripgpu::fedId_t, fedID_),
//                      SOA_COLUMN(stripgpu::fedCh_t, fedCh_));
//};
//
//using DetToFedLayout = typename DetToFedSoA::DetToFedSoALayout<>;
//using DetToFedView = typename DetToFedSoA::DetToFedSoALayout<>::View;
//using DetToFedConstView = typename DetToFedSoA::DetToFedSoALayout<>::ConstView;

struct DataSoA {
  const static auto stripsPerFedCh = sistrip::STRIPS_PER_FEDCH;
  const static auto apvsPerFedCh = sistrip::APVS_PER_FEDCH;
  using arraySTRIPS_PER_FEDCH = std::array<std::uint16_t, stripsPerFedCh>;
  using arrayAPVS_PER_FEDCH = std::array<float, apvsPerFedCh>;
  GENERATE_SOA_LAYOUT(DataSoALayout,
                      SOA_COLUMN(arraySTRIPS_PER_FEDCH, noise),
                      SOA_COLUMN(float, invthick),
                      SOA_COLUMN(stripgpu::detId_t, detID),
                      SOA_COLUMN(stripgpu::apvPair_t, iPair),
                      SOA_COLUMN(arrayAPVS_PER_FEDCH, gain));
};

using DataLayout = typename DataSoA::DataSoALayout<>;
using DataView = typename DataSoA::DataSoALayout<>::View;
using DataConstView = typename DataSoA::DataSoALayout<>::ConstView;

class SiStripQuality;
class SiStripGain;
class SiStripNoises;

namespace stripgpu {
  static constexpr std::uint16_t badBit = 1 << 15;

  __host__ __device__ inline fedId_t fedIndexHD(fedId_t fed) { return fed - sistrip::FED_ID_MIN; }
  __host__ __device__ inline std::uint32_t stripIndexHD(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return (strip % sistrip::STRIPS_PER_FEDCH);
  }
  __host__ __device__ inline std::uint32_t apvIndexHD(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return (strip % sistrip::STRIPS_PER_FEDCH) / sistrip::STRIPS_PER_APV;
  }
  __host__ __device__ inline std::uint32_t channelIndexHD(fedId_t fed, fedCh_t channel) {
    return fedIndexHD(fed) * sistrip::FEDCH_PER_FED + channel;
  }

}  // namespace stripgpu

#endif
