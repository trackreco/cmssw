#ifndef CUDADataFormats_SiStripCluster_interface_SiStripClustersSoAUtilities_h
#define CUDADataFormats_SiStripCluster_interface_SiStripClustersSoAUtilities_h

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

struct SiStripClustersSoA {
  const static auto maxStripsPerCluster = 768;
  using clusterADCsColumn = std::array<uint8_t, maxStripsPerCluster /*768*/>;
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
                      SOA_SCALAR(uint32_t, maxClusterSize));
};

using SiStripClustersLayout = typename SiStripClustersSoA::SiStripClustersSoALayout<>;
using SiStripClustersView = typename SiStripClustersSoA::SiStripClustersSoALayout<>::View;
using SiStripClustersConstView = typename SiStripClustersSoA::SiStripClustersSoALayout<>::ConstView;

#endif
