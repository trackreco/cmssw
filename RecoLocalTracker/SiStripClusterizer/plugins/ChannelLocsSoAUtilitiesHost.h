#ifndef CalibFormats_SiStripObjects_ChannelLocsSoAHost_h
#define CalibFormats_SiStripObjects_ChannelLocsSoAHost_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "ChannelLocsSoAUtilities.h"

class ChannelLocsSoAHost : public cms::cuda::PortableHostCollection<ChannelLocsLayout> {
public:
  using cms::cuda::PortableHostCollection<ChannelLocsLayout>::view;
  using cms::cuda::PortableHostCollection<ChannelLocsLayout>::const_view;
  using cms::cuda::PortableHostCollection<ChannelLocsLayout>::buffer;
  using cms::cuda::PortableHostCollection<ChannelLocsLayout>::bufferSize;

  ChannelLocsSoAHost() = default;
  ~ChannelLocsSoAHost() = default;

  ChannelLocsSoAHost(ChannelLocsSoAHost&) = delete;
  ChannelLocsSoAHost(const ChannelLocsSoAHost&) = delete;
  ChannelLocsSoAHost& operator=(const ChannelLocsSoAHost&) = delete;
  ChannelLocsSoAHost& operator=(ChannelLocsSoAHost&&) = delete;

  explicit ChannelLocsSoAHost(size_t size, cudaStream_t stream)
      : cms::cuda::PortableHostCollection<ChannelLocsLayout>(size, stream) {}

  ChannelLocsSoAHost(ChannelLocsSoAHost&& arg) {
    *view().input() = *arg->input();
    *view().inoff() = *arg->inoff();
    *view().offset() = *arg->offset();
    *view().length() = *arg->length();
    *view().fedID() = *arg->fedID();
    *view().fedCh() = *arg->fedCh();
    *view().detID() = *arg->detID();
  }

  void setChannelLoc(uint32_t index,
                     const uint8_t* input,
                     size_t inoff,
                     size_t offset,
                     uint16_t length,
                     stripgpu::fedId_t fedID,
                     stripgpu::fedCh_t fedCh,
                     stripgpu::detId_t detID) {
    view().input()[index] = input;
    view().inoff()[index] = inoff;
    view().offset()[index] = offset;
    view().length()[index] = length;
    view().fedID()[index] = fedID;
    view().fedCh()[index] = fedCh;
    view().detID()[index] = detID;
  }

};

#endif
