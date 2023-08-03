#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripUtilities.h"

class SiStripClustersCUDADevice : public cms::cuda::PortableDeviceCollection<SiStripAlpaka::SiStripClustersSoA> {
public:
  SiStripClustersCUDADevice() = default;
  ~SiStripClustersCUDADevice() = default;

  explicit SiStripClustersCUDADevice(uint32_t maxClusters, cudaStream_t stream)
      : cms::cuda::PortableDeviceCollection<SiStripAlpaka::SiStripClustersSoA>(maxClusters, stream){};

  SiStripClustersCUDADevice(SiStripClustersCUDADevice &&) = default;
  SiStripClustersCUDADevice &operator=(SiStripClustersCUDADevice &&) = default;

private:
};

#endif
