#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersSoADevice_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersSoADevice_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersSoAUtilities.h"

class SiStripClustersSoADevice : public cms::cuda::PortableDeviceCollection<SiStripClustersLayout> {
public:
  using cms::cuda::PortableDeviceCollection<SiStripClustersLayout>::view;
  using cms::cuda::PortableDeviceCollection<SiStripClustersLayout>::const_view;
  using cms::cuda::PortableDeviceCollection<SiStripClustersLayout>::buffer;
  using cms::cuda::PortableDeviceCollection<SiStripClustersLayout>::bufferSize;

  SiStripClustersSoADevice() = default;
  ~SiStripClustersSoADevice() = default;

  explicit SiStripClustersSoADevice(uint32_t maxClusters, cudaStream_t stream)
      : cms::cuda::PortableDeviceCollection<SiStripClustersLayout>(maxClusters, stream), maxClusters_{maxClusters} {
    const uint32_t maxStripsPerCluster = SiStripClustersSoA::maxStripsPerCluster;  //768
    cudaCheck(
        cudaMemcpyAsync(&(view().maxClusterSize()), &maxStripsPerCluster, sizeof(uint32_t), cudaMemcpyDefault, stream));
  };

  SiStripClustersSoADevice(const SiStripClustersSoADevice &&) = delete;
  SiStripClustersSoADevice &operator=(const SiStripClustersSoADevice &&) = delete;
  SiStripClustersSoADevice(SiStripClustersSoADevice &&) = default;
  SiStripClustersSoADevice &operator=(SiStripClustersSoADevice &&) = default;

  uint32_t maxClusters() const { return maxClusters_; }

private:
  uint32_t maxClusters_;
};

#endif
