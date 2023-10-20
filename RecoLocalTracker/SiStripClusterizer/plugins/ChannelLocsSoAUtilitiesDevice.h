#ifndef CalibFormats_SiStripObjects_ChannelLocsSoADevice_h
#define CalibFormats_SiStripObjects_ChannelLocsSoADevice_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

#include "ChannelLocsSoAUtilities.h"
#include "ChannelLocsSoAUtilitiesHost.h"

class ChannelLocsSoADevice : public cms::cuda::PortableDeviceCollection<ChannelLocsLayout> {
public:
  using cms::cuda::PortableDeviceCollection<ChannelLocsLayout>::view;
  using cms::cuda::PortableDeviceCollection<ChannelLocsLayout>::const_view;
  using cms::cuda::PortableDeviceCollection<ChannelLocsLayout>::buffer;
  using cms::cuda::PortableDeviceCollection<ChannelLocsLayout>::bufferSize;

  ChannelLocsSoADevice() = default;
  ~ChannelLocsSoADevice() = default;

  ChannelLocsSoADevice(ChannelLocsSoADevice&) = delete;
  ChannelLocsSoADevice(const ChannelLocsSoADevice&) = delete;
  ChannelLocsSoADevice& operator=(const ChannelLocsSoADevice&) = delete;
  ChannelLocsSoADevice& operator=(ChannelLocsSoADevice&&) = delete;

  explicit ChannelLocsSoADevice(const ChannelLocsSoAHost cpuChLocs, std::unique_ptr<const uint8_t*[]> inputGPU, cudaStream_t stream)
      : cms::cuda::PortableDeviceCollection<ChannelLocsLayout>(sizeof(cpuChLocs->detID()), stream) {
    size_t size_ = sizeof(cpuChLocs->input());
    cudaCheck(cudaMemcpyAsync(view().input(), &inputGPU, size_, cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(view().inoff(), cpuChLocs->inoff(), size_, cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(view().offset(), cpuChLocs->offset(), size_, cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(view().length(), cpuChLocs->length(), size_, cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(view().fedID(), cpuChLocs->fedID(), size_, cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(view().fedCh(), cpuChLocs->fedCh(), size_, cudaMemcpyDefault, stream));
    cudaCheck(cudaMemcpyAsync(view().detID(), cpuChLocs->detID(), size_, cudaMemcpyDefault, stream));
  }

};

#endif
