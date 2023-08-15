#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersSoAHost_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersSoAHost_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersSoAUtilities.h"

class SiStripClustersSoAHost : public cms::cuda::PortableHostCollection<SiStripClustersLayout> {
public:
  SiStripClustersSoAHost() = default;
  ~SiStripClustersSoAHost() = default;

  explicit SiStripClustersSoAHost(uint32_t maxClusters, cudaStream_t stream)
      : PortableHostCollection<SiStripClustersLayout>(maxClusters, stream) {
        const uint32_t maxStripsPerCluster = 768;
        view().nClusters() = maxStripsPerCluster;
      };

  SiStripClustersSoAHost(const SiStripClustersSoAHost &&) = delete;
  SiStripClustersSoAHost &operator=(const SiStripClustersSoAHost &&) = delete;
  SiStripClustersSoAHost(SiStripClustersSoAHost &&) = default;
  SiStripClustersSoAHost &operator=(SiStripClustersSoAHost &&) = default;

private:
};

#endif
