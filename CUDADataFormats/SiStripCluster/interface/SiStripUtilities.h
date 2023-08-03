#ifndef CUDADataFormats_SiStripCluster_interface_SiStripUtilities_h
#define CUDADataFormats_SiStripCluster_interface_SiStripUtilities_h

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SiStripCluster/interface/SiStripClustersSOABase.h"

namespace SiStripAlpaka {
  struct SiStripSoA {
    using clusterADCsColumn = std::array<uint8_t, 768>;
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
  };

  using SiStripClustersSoA = typename SiStripSoA::SiStripClustersSoALayout<>;
  using SiStripClustersSoAView = typename SiStripSoA::SiStripClustersSoALayout<>::View;
  using SiStripClustersSoAConstView = typename SiStripSoA::SiStripClustersSoALayout<>::ConstView;
}  // namespace SiStripAlpaka

#endif
