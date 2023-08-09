#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersCUDA_h

#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripUtilities.h"

#include <cuda_runtime.h>

class SiStripClustersCUDADevice : public cms::cuda::PortableDeviceCollection<SiStripClustersSoA> {

public:
  SiStripClustersCUDADevice() = default;
  ~SiStripClustersCUDADevice() = default;
  
  using cms::cuda::PortableDeviceCollection<SiStripClustersSoA>::view;
  using cms::cuda::PortableDeviceCollection<SiStripClustersSoA>::const_view;
  using cms::cuda::PortableDeviceCollection<SiStripClustersSoA>::buffer;
  using cms::cuda::PortableDeviceCollection<SiStripClustersSoA>::bufferSize;
  
 
  explicit SiStripClustersCUDADevice(uint32_t maxClusters, cudaStream_t stream) 
  : cms::cuda::PortableDeviceCollection<SiStripClustersSoA>(maxClusters, stream) {};
	 

 
  SiStripClustersCUDADevice(SiStripClustersCUDADevice &&) = default;
  SiStripClustersCUDADevice &operator=(SiStripClustersCUDADevice &&) = default;

private:
  
};

#endif
