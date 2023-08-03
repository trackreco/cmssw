#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDAHost_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDAHost_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripUtilities.h"

class SiStripClustersCUDAHost : public cms::cuda::PortableHostCollection<SiStripAlpaka::SiStripClustersSoA> {
public:
  SiStripClustersCUDAHost() = default;
  ~SiStripClustersCUDAHost() = default;

  explicit SiStripClustersCUDAHost(uint32_t maxClusters, cudaStream_t stream)
      : PortableHostCollection<SiStripAlpaka::SiStripClustersSoA>(maxClusters, stream){};

  SiStripClustersCUDAHost(SiStripClustersCUDAHost &&) = default;
  SiStripClustersCUDAHost &operator=(SiStripClustersCUDAHost &&) = default;

private:
};

#endif
