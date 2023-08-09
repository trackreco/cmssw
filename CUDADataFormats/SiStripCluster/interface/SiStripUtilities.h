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

struct SiStripSoA {




//std::array<uint8_t, clusterADCStartArray_>
	//SiStripSoA(uint8_t maxStripsPerCluster) {
		//using Matrix = Eigen::Matrix <uint8_t , 768 , 2>;
		using clusterADCsColumn = std::array<uint8_t, 768>;
		//clusterADCStartArray = const_cast<uint8_t&>(clusterADCStartArray_);
		//uint8_t clusterADCStartArray_ = clusterADCStartArray;
		//std::vector<uint8_t> HitModuleStartArraya(clusterADCStartArray);
		GENERATE_SOA_LAYOUT(SiStripClustersSoALayout,
							SOA_COLUMN(uint32_t, clusterIndex),
							SOA_COLUMN(uint32_t, clusterSize),
							SOA_COLUMN(clusterADCsColumn, clusterADCs),
							SOA_COLUMN(stripgpu::detId_t, clusterDetId),
							SOA_COLUMN(stripgpu::stripId_t, firstStrip),
							SOA_COLUMN(bool, trueCluster),
							SOA_COLUMN(float, barycenter),
							SOA_COLUMN(float, charge),
							SOA_SCALAR(uint32_t, nClusters),
							SOA_SCALAR(uint32_t, maxClusterSize)

		)
		
		//return SiStripClustersSoALayout;

	//}

};

/*
struct TracksUtilities {
	using SiStripClustersSoA = SiStripSoA::SiStripClustersSoALayout<>;
	using SiStripClustersSoAView = SiStripSoA::SiStripClustersSoALayout<>::View;
	using SiStripClustersSoAConstView = SiStripSoA::SiStripClustersSoALayout<>::ConstView;
	
  //using hindex_type = typename SiStripSoA::hindex_type;
};
*/
using SiStripClustersSoA = typename SiStripSoA::SiStripClustersSoALayout<>;
using SiStripClustersSoAView = typename SiStripSoA::SiStripClustersSoALayout<>::View;
using SiStripClustersSoAConstView = typename SiStripSoA::SiStripClustersSoALayout<>::ConstView;


//using TrackLayout = TracksUtilities.SiStripClustersSoA;
/*using TrackLayout = SiStripSoA::template SiStripSoALayout<>;

using SiStripSoAView = typename SiStripSoA::template SiStripSoALayout<>::View;

using SiStripSoAConstView = typename SiStripSoA::template SiStripSoALayout<>::ConstView;*/

#endif