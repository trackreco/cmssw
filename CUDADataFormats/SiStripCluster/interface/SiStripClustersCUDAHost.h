#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDAHost_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDAHost_h

#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripUtilities.h"


#include <cuda_runtime.h>

class SiStripClustersCUDAHost : public cms::cuda::PortableHostCollection<SiStripClustersSoA> {
public:
  SiStripClustersCUDAHost() = default;
  ~SiStripClustersCUDAHost() = default;

  using cms::cuda::PortableHostCollection<SiStripClustersSoA>::view;
  using cms::cuda::PortableHostCollection<SiStripClustersSoA>::const_view;
  using cms::cuda::PortableHostCollection<SiStripClustersSoA>::buffer;
  using cms::cuda::PortableHostCollection<SiStripClustersSoA>::bufferSize;

  explicit SiStripClustersCUDAHost(uint32_t maxClusters, cudaStream_t stream)
      : PortableHostCollection<SiStripClustersSoA>(maxClusters, stream){};

  SiStripClustersCUDAHost(SiStripClustersCUDAHost &&) = default;
  SiStripClustersCUDAHost &operator=(SiStripClustersCUDAHost &&) = default;

private:
  
};


#endif
