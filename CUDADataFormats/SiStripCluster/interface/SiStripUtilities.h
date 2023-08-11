#ifndef CUDADataFormats_SiStripCluster_interface_SiStripUtilities_h
#define CUDADataFormats_SiStripCluster_interface_SiStripUtilities_h

#include <Eigen/Core>
#include <Eigen/Dense>

#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"


#include <cuda_runtime.h>

//using clusterADCStartArray = std::array<hindex_type, TrackerTraits::numberOfModules + 1>;
namespace stripTest {
struct SiStripSoA {


		//using clusterADCsColumn = std::array<uint8_t, 768>;

		GENERATE_SOA_LAYOUT(SiStripClustersSoALayout,
							SOA_COLUMN(uint32_t, clusterIndex),
							SOA_COLUMN(uint32_t, clusterSize),
							//SOA_COLUMN(clusterADCsColumn, clusterADCs),
							//SOA_COLUMN(uint8_t, clusterADCs),
							SOA_COLUMN(stripgpu::detId_t, clusterDetId),
							SOA_COLUMN(stripgpu::stripId_t, firstStrip),
							SOA_COLUMN(bool, trueCluster),
							SOA_COLUMN(float, barycenter),
							SOA_COLUMN(float, charge),
							SOA_SCALAR(uint32_t, nClusters),
							SOA_SCALAR(uint32_t, maxClusterSize)

		)
		
};


using SiStripClustersSoA = typename SiStripSoA::SiStripClustersSoALayout<>;
using SiStripClustersSoAView = typename SiStripSoA::SiStripClustersSoALayout<>::View;
using SiStripClustersSoAConstView = typename SiStripSoA::SiStripClustersSoALayout<>::ConstView;
}

#endif