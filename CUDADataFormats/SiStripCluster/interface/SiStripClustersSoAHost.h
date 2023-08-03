#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersSoAHost_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersSoAHost_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersSoAUtilities.h"

class SiStripClustersSoAHost : public cms::cuda::PortableHostCollection<SiStripClustersLayout> {
public:
  using cms::cuda::PortableHostCollection<SiStripClustersLayout>::view;
  using cms::cuda::PortableHostCollection<SiStripClustersLayout>::const_view;
  using cms::cuda::PortableHostCollection<SiStripClustersLayout>::buffer;
  using cms::cuda::PortableHostCollection<SiStripClustersLayout>::bufferSize;

  SiStripClustersSoAHost() = default;
  ~SiStripClustersSoAHost() = default;

  explicit SiStripClustersSoAHost(uint32_t maxClusters, cudaStream_t stream)
      : PortableHostCollection<SiStripClustersLayout>(maxClusters, stream){};

  SiStripClustersSoAHost(const SiStripClustersSoAHost &&) = delete;
  SiStripClustersSoAHost &operator=(const SiStripClustersSoAHost &&) = delete;
  SiStripClustersSoAHost(SiStripClustersSoAHost &&) = default;
  SiStripClustersSoAHost &operator=(SiStripClustersSoAHost &&) = default;

private:
};

#endif
