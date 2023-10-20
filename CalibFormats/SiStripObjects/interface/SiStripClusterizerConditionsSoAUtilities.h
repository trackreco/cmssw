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
  using arraySTRIPS_PER_FEDCH = std::array<std::uint16_t,sistrip::STRIPS_PER_FEDCH>;
  using arrayAPVS_PER_FEDCH = std::array<float,sistrip::APVS_PER_FEDCH>;
  GENERATE_SOA_LAYOUT(DataSoALayout,
		                  //SOA_COLUMN(arraySTRIPS_PER_FEDCH, noise_),
		                  SOA_COLUMN(std::uint16_t, noise_),
                      SOA_COLUMN(float, invthick_),
                      SOA_COLUMN(stripgpu::detId_t, detID_),
                      SOA_COLUMN(stripgpu::apvPair_t, iPair_),
                      //SOA_COLUMN(arrayAPVS_PER_FEDCH, gain_));
                      SOA_COLUMN(float, gain_));
};

using DataLayout = typename DataSoA::DataSoALayout<>;
using DataView = typename DataSoA::DataSoALayout<>::View;
using DataConstView = typename DataSoA::DataSoALayout<>::ConstView;

class SiStripQuality;
class SiStripGain;
class SiStripNoises;

namespace stripgpu {
  static constexpr std::uint16_t badBit = 1 << 15;
//
//  __host__ __device__ inline fedId_t fedIndex(fedId_t fed) { return fed - sistrip::FED_ID_MIN; }
//  __host__ __device__ inline std::uint32_t stripIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
//    return fedIndex(fed) * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH + channel * sistrip::STRIPS_PER_FEDCH +
//           (strip % sistrip::STRIPS_PER_FEDCH);
//  }
//  __host__ __device__ inline std::uint32_t apvIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
//    return fedIndex(fed) * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED + sistrip::APVS_PER_CHAN * channel +
//           (strip % sistrip::STRIPS_PER_FEDCH) / sistrip::STRIPS_PER_APV;
//  }
//  __host__ __device__ inline std::uint32_t channelIndex(fedId_t fed, fedCh_t channel) {
//    return fedIndex(fed) * sistrip::FEDCH_PER_FED + channel;
//  }
//
}  // namespace stripgpu

#endif
