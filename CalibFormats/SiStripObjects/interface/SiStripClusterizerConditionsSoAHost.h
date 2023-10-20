#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAHost_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoAHost_h

#include "CUDADataFormats/Common/interface/PortableHostCollection.h"
#include "CalibFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoAUtilities.h"
#include "DataFormats/SiStripCluster/interface/SiStripClusterTools.h"

namespace stripgpu {

  //class SiStripClusterizerConditionsSoAHost {
  //public:
    class DetToFed {
    public:
      DetToFed(detId_t detid, apvPair_t ipair, fedId_t fedid, fedCh_t fedch)
          : detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch) {}
      detId_t detID() const { return detid_; }
      apvPair_t pair() const { return ipair_; }
      fedId_t fedID() const { return fedid_; }
      fedCh_t fedCh() const { return fedch_; }

    private:
      detId_t detid_;
      apvPair_t ipair_;
      fedId_t fedid_;
      fedCh_t fedch_;
    };
    //class DetToFedSoAHost : public cms::cuda::PortableHostCollection<DetToFedLayout> {
    //public:
    //  using cms::cuda::PortableHostCollection<DetToFedLayout>::view;
    //  using cms::cuda::PortableHostCollection<DetToFedLayout>::const_view;
    //  using cms::cuda::PortableHostCollection<DetToFedLayout>::buffer;
    //  using cms::cuda::PortableHostCollection<DetToFedLayout>::bufferSize;
    //  
    //  DetToFedSoAHost() = default;
    //  ~DetToFedSoAHost() = default;
    //  
    //  explicit DetToFedSoAHost(detId_t detid, apvPair_t ipair, fedId_t fedid, fedCh_t fedch, cudaStream_t stream)
    //      : PortableHostCollection<DetToFedLayout>(
    //          detID_(detid), pair_(ipair), fedID_(fedid), fedCh_(fedch)){};
    //  
    //  DetToFedSoAHost(const DetToFedSoAHost &&) = delete;
    //  DetToFedSoAHost &operator=(const DetToFedSoAHost &&) = delete;
    //  DetToFedSoAHost(DetToFedSoAHost &&) = default;
    //  DetToFedSoAHost &operator=(DetToFedSoAHost &&) = default;
    //};    
    using DetToFeds = std::vector<DetToFed>;    

    //const DetToFeds& detToFeds() const { return detToFeds_; }

    class DataSoAHost : public cms::cuda::PortableHostCollection<DataLayout> {
    public:
      using cms::cuda::PortableHostCollection<DataLayout>::view;
      using cms::cuda::PortableHostCollection<DataLayout>::const_view;
      using cms::cuda::PortableHostCollection<DataLayout>::buffer;
      using cms::cuda::PortableHostCollection<DataLayout>::bufferSize;
      
      DataSoAHost() = default;
      ~DataSoAHost() = default;

      DataSoAHost(const DataSoAHost &&) = delete;
      DataSoAHost &operator=(const DataSoAHost &&) = delete;
      DataSoAHost(DataSoAHost &&) = default;
      DataSoAHost &operator=(DataSoAHost &&) = default;

      explicit DataSoAHost(const SiStripQuality& quality, const SiStripGain* gains, const SiStripNoises& noises, DetToFeds& detToFeds, cudaStream_t stream)
          : cms::cuda::PortableHostCollection<DataLayout>(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED, stream) {
        // connected: map<DetID, std::vector<int>>
        // map of KEY=detid DATA=vector of apvs, maximum 6 APVs per detector module :
        const auto& connected = quality.cabling()->connected();
        // detCabling: map<DetID, std::vector<const FedChannelConnection *>
        // map of KEY=detid DATA=vector<FedChannelConnection>
        const auto& detCabling = quality.cabling()->getDetCabling();
        
        for (const auto& conn : connected) {
          const auto det = conn.first;
          if (!quality.IsModuleBad(det)) {
            const auto detConn_it = detCabling.find(det);
      
            if (detCabling.end() != detConn_it) {
              for (const auto& chan : (*detConn_it).second) {
                if (chan && chan->fedId() && chan->isConnected()) {
                  const auto detID = chan->detId();
                  const auto fedID = chan->fedId();
                  const auto fedCh = chan->fedCh();
                  const auto iPair = chan->apvPairNumber();
      
                  detToFeds.emplace_back(detID, iPair, fedID, fedCh);
      
                  view().detID_()[channelIndex(fedID, fedCh)] = detID;
                  view().iPair_()[channelIndex(fedID, fedCh)] = iPair;
                  setInvThickness(fedID, fedCh, siStripClusterTools::sensorThicknessInverse(detID));
      
                  auto offset = 256 * iPair;
                  
                  for (auto strip = 0; strip < 256; ++strip) {
                    const auto gainRange = gains->getRange(det);
                    
                    const auto detstrip = strip + offset;
                    const std::uint16_t noise = SiStripNoises::getRawNoise(detstrip, noises.getRange(det));
                    const auto gain = SiStripGain::getStripGain(detstrip, gainRange);
                    const auto bad = quality.IsStripBad(quality.getRange(det), detstrip);
                    
                    // gain is actually stored per-APV, not per-strip
                    setStrip(fedID, fedCh, detstrip, noise, gain, bad);
                  }
                }
              }
            }
          }
        }
        
        std::sort(detToFeds.begin(), detToFeds.end(), [](const DetToFed& a, const DetToFed& b) {
          return a.detID() < b.detID() || (a.detID() == b.detID() && a.pair() < b.pair());
        });
      }
      
    private:
      void setStrip(fedId_t fed, fedCh_t channel, stripId_t strip, std::uint16_t noise, float gain, bool bad) {
        view().gain_()[apvIndex(fed, channel, strip)] = gain;
        view().noise_()[stripIndex(fed, channel, strip)] = noise;
        if (bad) {
          view().noise_()[stripIndex(fed, channel, strip)] |= badBit;
        }
      }

      void setInvThickness(fedId_t fed, fedCh_t channel, float invthick) {
        view().invthick_()[channelIndex(fed, channel)] = invthick;
      }
    };

  //  explicit SiStripClusterizerConditionsSoAHost(const SiStripQuality& quality,
  //                                               const SiStripGain* gains,
  //                                               const SiStripNoises& noises,
  //                                               cudaStream_t stream) {};
    //    : data_(quality, gains, noises, stream) {};

  //  ~SiStripClusterizerConditionsSoAHost() = default;

    //DataSoAHost data() const { return data_; }

  //private:
  //  DetToFeds detToFeds_;
  //  //DataSoAHost data_;

  //};

}  // namespace stripgpu

#endif
