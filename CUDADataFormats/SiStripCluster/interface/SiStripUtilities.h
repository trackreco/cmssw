#ifndef CUDADataFormats_SiStripCluster_interface_SiStripUtilities_h
#define CUDADataFormats_SiStripCluster_interface_SiStripUtilities_h

#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"
#include "HeterogeneousCore/CUDAUtilities/interface/device_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_unique_ptr.h"
#include "HeterogeneousCore/CUDAUtilities/interface/cudaCompat.h"

#include "DataFormats/SoATemplate/interface/SoALayout.h"


#include <cuda_runtime.h>


struct SiStripSoA {

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
                    SOA_SCALAR(uint32_t, maxClusterSize)

)

};

/*
struct TracksUtilities {
	using SiStripClustersSoA = SiStripSoA::SiStripClustersSoALayout<>;
	using SiStripClustersSoAView = SiStripSoA::SiStripClustersSoALayout<>::View;
	using SiStripClustersSoAConstView = SiStripSoA::SiStripClustersSoALayout<>::ConstView;
	
  //using hindex_type = typename SiStripSoA::hindex_type;
};
*/
using SiStripClustersSoA = SiStripSoA::SiStripClustersSoALayout<>;
using SiStripClustersSoAView = SiStripSoA::SiStripClustersSoALayout<>::View;
using SiStripClustersSoAConstView = SiStripSoA::SiStripClustersSoALayout<>::ConstView;


//using TrackLayout = TracksUtilities.SiStripClustersSoA;
/*using TrackLayout = SiStripSoA::template SiStripSoALayout<>;

using SiStripSoAView = typename SiStripSoA::template SiStripSoALayout<>::View;

using SiStripSoAConstView = typename SiStripSoA::template SiStripSoALayout<>::ConstView;*/

#endif