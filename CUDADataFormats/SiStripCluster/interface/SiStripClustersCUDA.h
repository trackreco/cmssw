#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h

#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"

#include <cuda_runtime.h>

GENERATE_SOA_LAYOUT(SiStripClustersSoALayout,
                    SOA_COLUMN(uint32_t, clusterIndex),
                    SOA_COLUMN(uint32_t, clusterSize),
                    SOA_COLUMN(uint8_t, clusterADCs),
                    SOA_COLUMN(stripgpu::detId_t, clusterDetId),
                    SOA_COLUMN(stripgpu::stripId_t, firstStrip),
                    SOA_COLUMN(bool, trueCluster),
                    SOA_COLUMN(float, barycenter),
                    SOA_COLUMN(float, charge),
                    SOA_SCALAR(uint32_t, nClusters),
                    SOA_COLUMN(uint32_t, maxClusterSize)

)
using SiStripClustersSoA = SiStripClustersSoALayout<>;
using SiStripClustersSoAView = SiStripClustersSoALayout<>::View;
using SiStripClustersSOAConstView = SiStripClustersSoALayout<>::ConstView;

class SiStripClustersCUDADevice : public cms::cuda::PortableDeviceCollection<SiStripClustersSoALayout<>> {
public:
  SiStripClustersCUDADevice() = default;
  ~SiStripClustersCUDADevice() = default;

  explicit SiStripClustersCUDADevice(uint32_t maxClusters, uint32_t maxStripsPerCluster, cudaStream_t stream)
      : PortableDeviceCollection<SiStripClustersSoALayout<>>(maxClusters, stream){};

  SiStripClustersCUDADevice(SiStripClustersCUDADevice &&) = default;
  SiStripClustersCUDADevice &operator=(SiStripClustersCUDADevice &&) = default;

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
