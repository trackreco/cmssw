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

  explicit SiStripClustersCUDAHost(uint32_t maxClusters, uint32_t maxStripsPerCluster, cudaStream_t stream)
      : PortableHostCollection<SiStripClustersSoA>(maxClusters, stream){};

  SiStripClustersCUDAHost(SiStripClustersCUDAHost &&) = default;
  SiStripClustersCUDAHost &operator=(SiStripClustersCUDAHost &&) = default;

  //uint32_t nClusters() const { return nClusters_; }
  //uint32_t *nClustersPtr() { return &nClusters_; }
  //int32_t offsetBPIX2() const { return offsetBPIX2_h; }
  //DeviceView = SiPixelClustersCUDALayout<>::View;

  //uint32_t nClusters() const { return nClusters_; }
  //uint32_t *nClustersPtr() { return &nClusters_; }
  /*  uint32_t maxClusterSize() const { return maxClusterSize_; }
  uint32_t *maxClusterSizePtr() { return &maxClusterSize_; }
*/
private:
  //uint32_t nClusters_;
  //uint32_t maxClusterSize_;
};
/*
class SiStripClustersCUDAHost : public SiStripClustersSOABase<cms::cuda::host::unique_ptr> {
public:
  SiStripClustersCUDAHost() = default;
  explicit SiStripClustersCUDAHost(const SiStripClustersCUDADevice &clusters_d, cudaStream_t stream);
  ~SiStripClustersCUDAHost() override = default;

  SiStripClustersCUDAHost(const SiStripClustersCUDAHost &) = delete;
  SiStripClustersCUDAHost &operator=(const SiStripClustersCUDAHost &) = delete;
  SiStripClustersCUDAHost(SiStripClustersCUDAHost &&) = default;
  SiStripClustersCUDAHost &operator=(SiStripClustersCUDAHost &&) = default;
};*/

#endif
