#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoADevice_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoADevice_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoAUtilities.h"

namespace stripgpu {

  class SiStripClusterizerConditionsSoADevice
      : public cms::cuda::PortableDeviceCollection<SiStripClusterizerConditionsLayout> {
  public:
    using cms::cuda::PortableDeviceCollection<SiStripClusterizerConditionsLayout>::view;
    using cms::cuda::PortableDeviceCollection<SiStripClusterizerConditionsLayout>::const_view;
    using cms::cuda::PortableDeviceCollection<SiStripClusterizerConditionsLayout>::buffer;
    using cms::cuda::PortableDeviceCollection<SiStripClusterizerConditionsLayout>::bufferSize;

    SiStripClusterizerConditionsSoADevice() = default;
    ~SiStripClusterizerConditionsSoADevice() = default;

    SiStripClusterizerConditionsSoADevice(const SiStripClusterizerConditionsSoADevice &&) = delete;
    SiStripClusterizerConditionsSoADevice &operator=(const SiStripClusterizerConditionsSoADevice &&) = delete;
    SiStripClusterizerConditionsSoADevice(SiStripClusterizerConditionsSoADevice &&) = default;
    SiStripClusterizerConditionsSoADevice &operator=(SiStripClusterizerConditionsSoADevice &&) = default;

    explicit SiStripClusterizerConditionsSoADevice(
        SiStripClusterizerConditionsSoAHost cpuSiStrClCond,
        cudaStream_t stream)  // sistrip::NUMBER_OF_FEDS*sistrip::FEDCH_PER_FED
        : cms::cuda::PortableDeviceCollection<SiStripClusterizerConditionsLayout>(sizeof(cpuSiStrClCond->detID()),
                                                                                  stream) {
      const auto noise = cpuSiStrClCond->noise();
      cudaCheck(
          cudaMemcpyAsync(view().noise(), noise, sizeof(std::uint16_t) * sizeof(noise), cudaMemcpyDefault, stream));
      const auto invthick = cpuSiStrClCond->invthick();
      cudaCheck(
          cudaMemcpyAsync(view().invthick(), invthick, sizeof(float) * sizeof(invthick), cudaMemcpyDefault, stream));
      const auto detID = cpuSiStrClCond->detID();
      cudaCheck(cudaMemcpyAsync(view().detID(), detID, sizeof(detId_t) * sizeof(detID), cudaMemcpyDefault, stream));
      const auto iPair = cpuSiStrClCond->iPair();
      cudaCheck(cudaMemcpyAsync(view().iPair(), iPair, sizeof(apvPair_t) * sizeof(iPair), cudaMemcpyDefault, stream));
      const auto gain = cpuSiStrClCond->gain();
      cudaCheck(cudaMemcpyAsync(view().gain(), gain, sizeof(float) * sizeof(gain), cudaMemcpyDefault, stream));
    };

    __device__ inline detId_t detID(fedId_t fed, fedCh_t channel) const {
      return view().detID()[channelIndexHD(fed, channel)];
    }

    __device__ inline apvPair_t iPair(fedId_t fed, fedCh_t channel) const {
      return view().iPair()[channelIndexHD(fed, channel)];
    }

    __device__ inline float invthick(fedId_t fed, fedCh_t channel) const {
      return view().invthick()[channelIndexHD(fed, channel)];
    }

    __device__ inline float noise(fedId_t fed, fedCh_t channel, stripId_t strip) const {
      // noise is stored as 9 bits with a fixed point scale factor of 0.1
      return 0.1f * (view().noise()[channelIndexHD(fed, channel)][stripIndexHD(strip)] & ~badBit);
    }

    __device__ inline float gain(fedId_t fed, fedCh_t channel, stripId_t strip) const {
      return view().gain()[channelIndexHD(fed, channel)][apvIndexHD(strip)];
    }

    __device__ inline bool bad(fedId_t fed, fedCh_t channel, stripId_t strip) const {
      return badBit == (view().noise()[channelIndexHD(fed, channel)][stripIndexHD(strip)] & badBit);
    }
  };
  //  // Helper that takes care of complexity of transferring the data to
  //  // multiple devices
  //  cms::cuda::ESProduct<Data> gpuData_;
  //};

}  // namespace stripgpu

#endif
