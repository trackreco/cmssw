#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAUtilities_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAUtilities_h

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "DataFormats/SiStripCommon/interface/ConstantsForHardwareSystems.h"

#include "HeterogeneousCore/CUDACore/interface/ESProduct.h"
//#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/HostAllocator.h"

struct DetToFedSoA {
  GENERATE_SOA_LAYOUT(DetToFedSoALayout,
		      SOA_COLUMN(stripgpu::detId_t, detid_),
                      SOA_COLUMN(stripgpu::avpPair_t, ipair_),
                      SOA_COLUMN(stripgpu::fedId_t, fedid_),
                      SOA_COLUMN(stripgpu::fedCh_t, fedch_));
  
};

using DetToFedLayout = typename DetToFedSoA::DetToFedSoALayout<>;
using DetToFedView = typename DetToFedSoA::DetToFedSoALayout<>::View;
using DetToFedConstView = typename DetToFedSoA::DetToFedSoALayout<>::ConstView;

struct Data {
  GENERATE_SOA_LAYOUT(DataSoALayout,
		      SOA_COLUMN(std::uint16_t, noise_),
                      SOA_COLUMN(float, invthick_),
                      SOA_COLUMN(stripgpu::apvPair_t, iPair_),
                      SOA_COLUMN(float, gain_));

};
/*
const std::uint16_t* noise_;  //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH];
const float* invthick_;       //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
const detId_t* detID_;        //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
const apvPair_t* iPair_;      //[sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED];
const float* gain_;           //[sistrip::NUMBER_OF_FEDS*sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED];
*/

using DataLayout = typename DataSoA::DataSoALayout<>;
using DataView = typename DataSoA::DataSoALayout<>::View;
using DataConstView = typename DataSoA::DataSoALayout<>::ConstView;

class SiStripQuality;
class SiStripGain;
class SiStripNoises;

namespace stripgpu {
  __host__ __device__ inline fedId_t fedIndex(fedId_t fed) { return fed - sistrip::FED_ID_MIN; }
  __host__ __device__ inline std::uint32_t stripIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return fedIndex(fed) * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH + channel * sistrip::STRIPS_PER_FEDCH +
           (strip % sistrip::STRIPS_PER_FEDCH);
  }
  __host__ __device__ inline std::uint32_t apvIndex(fedId_t fed, fedCh_t channel, stripId_t strip) {
    return fedIndex(fed) * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED + sistrip::APVS_PER_CHAN * channel +
           (strip % sistrip::STRIPS_PER_FEDCH) / sistrip::STRIPS_PER_APV;
  }
  __host__ __device__ inline std::uint32_t channelIndex(fedId_t fed, fedCh_t channel) {
    return fedIndex(fed) * sistrip::FEDCH_PER_FED + channel;
  }

  static constexpr std::uint16_t badBit = 1 << 15;
  __device__ inline detId_t detID(fedId_t fed, fedCh_t channel) const {
    return detID_[channelIndex(fed, channel)];
  }
  
  __device__ inline apvPair_t iPair(fedId_t fed, fedCh_t channel) const {
    return iPair_[channelIndex(fed, channel)];
  }
  
  __device__ inline float invthick(fedId_t fed, fedCh_t channel) const {
    return invthick_[channelIndex(fed, channel)];
  }
  
  __device__ inline float noise(fedId_t fed, fedCh_t channel, stripId_t strip) const {
    // noise is stored as 9 bits with a fixed point scale factor of 0.1
    return 0.1f * (noise_[stripIndex(fed, channel, strip)] & ~badBit);
  }
  
  __device__ inline float gain(fedId_t fed, fedCh_t channel, stripId_t strip) const {
    return gain_[apvIndex(fed, channel, strip)];
  }
  
  __device__ inline bool bad(fedId_t fed, fedCh_t channel, stripId_t strip) const {
    return badBit == (noise_[stripIndex(fed, channel, strip)] & badBit);
  }
  
}  // namespace stripgpu

#endif
