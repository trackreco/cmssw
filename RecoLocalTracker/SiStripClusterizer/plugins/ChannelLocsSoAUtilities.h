#ifndef CalibFormats_SiStripObjects_ChannelLocsSoAUtilities_h
#define CalibFormats_SiStripObjects_ChannelLocsSoAUtilities_h

#include "DataFormats/SiStripCluster/interface/SiStripTypes.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

struct ChannelLocsSoA {
  GENERATE_SOA_LAYOUT(ChannelLocsSoALayout,
		                  SOA_COLUMN(const uint8_t*, input),
		                  SOA_COLUMN(size_t, inoff),
		                  SOA_COLUMN(size_t, offset),
                      SOA_COLUMN(uint16_t, length),
                      SOA_COLUMN(stripgpu::fedId_t, fedID),
                      SOA_COLUMN(stripgpu::fedCh_t, fedCh),
                      SOA_COLUMN(stripgpu::detId_t, detID));
};

using ChannelLocsLayout = typename ChannelLocsSoA::ChannelLocsSoALayout<>;
using ChannelLocsView = typename ChannelLocsSoA::ChannelLocsSoALayout<>::View;
using ChannelLocsConstView = typename ChannelLocsSoA::ChannelLocsSoALayout<>::ConstView;

#endif
