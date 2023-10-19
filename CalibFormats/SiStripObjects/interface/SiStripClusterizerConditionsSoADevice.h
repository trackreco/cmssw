#ifndef CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoADevice_h
#define CalibFormats_SiStripObjects_SiStripClusterizerConditionsSoADevice_h

#include "CUDADataFormats/Common/interface/PortableDeviceCollection.h"
#include "CalibFormats/SiStripObjects/SiStripClusterizerConditionsSoAUtilities.h"

namespace stripgpu {

  class SiStripClusterizerConditionsSoADevice {
  public:
    class DetToFedSoADevice : public cms::cuda::PortableDeviceCollection<DetToFedLayout> {
    public:
      using cms::cuda::PortableDeviceCollection<DetToFedLayout>::view;
      using cms::cuda::PortableDeviceCollection<DetToFedLayout>::const_view;
      using cms::cuda::PortableDeviceCollection<DetToFedLayout>::buffer;
      using cms::cuda::PortableDeviceCollection<DetToFedLayout>::bufferSize;
      
      DetToFedSoADevice() = default;
      ~DetToFedSoADevice() = default;
      
      explicit DetToFedSoADevice(detId_t detid, apvPair_t ipair, fedId_t fedid, fedCh_t fedch)
	: PortableDeviceCollection<DetToFedLayout>(detid_(detid), ipair_(ipair), fedid_(fedid), fedch_(fedch)){};
      
      DetToFedSoADevice(const DetToFedSoADevice &&) = delete;
      DetToFedSoADevice &operator=(const DetToFedSoADevice &&) = delete;
      DetToFedSoADevice(DetToFedSoADevice &&) = default;
      DetToFedSoADevice &operator=(DetToFedSoADevice &&) = default;
      
    private:
    };    
    using DetToFeds = std::vector<DetToFed>;    

    class DataSoADevice : public cms::cuda::PortableDeviceCollection<DataLayout> {
    public:
      using cms::cuda::PortableDeviceCollection<DataLayout>::view;
      using cms::cuda::PortableDeviceCollection<DataLayout>::const_view;
      using cms::cuda::PortableDeviceCollection<DataLayout>::buffer;
      using cms::cuda::PortableDeviceCollection<DataLayout>::bufferSize;
      
      DataSoADevice() = default;
      ~DataSoADevice() = default;

      DataSoADevice(const DataSoADevice &&) = delete;
      DataSoADevice &operator=(const DataSoADevice &&) = delete;
      DataSoADevice(DataSoADevice &&) = default;
      DataSoADevice &operator=(DataSoADevice &&) = default;
      
    private:
    };    

    explicit SiStripClusterizerConditionsSoADevice(const SiStripQuality& quality,
					     const SiStripGain* gains,
					     const SiStripNoises& noises)
      : noise_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED * sistrip::STRIPS_PER_FEDCH),
        invthick_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED),
        detID_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED),
        iPair_(sistrip::NUMBER_OF_FEDS * sistrip::FEDCH_PER_FED),
        gain_(sistrip::NUMBER_OF_FEDS * sistrip::APVS_PER_FEDCH * sistrip::FEDCH_PER_FED) {
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
		
		detToFeds_.emplace_back(detID, iPair, fedID, fedCh);
		
		detID_[channelIndex(fedID, fedCh)] = detID;
		iPair_[channelIndex(fedID, fedCh)] = iPair;
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
      
      std::sort(detToFeds_.begin(), detToFeds_.end(), [](const DetToFed& a, const DetToFed& b) {
							return a.detID() < b.detID() || (a.detID() == b.detID() && a.pair() < b.pair());
						      });
    }
    ~SiStripClusterizerConditionsSoADevice() = default;

    // Function to return the actual payload on the memory of the current device
    DataSoADevice const& getGPUProductAsync(cudaStream_t stream) const {
      auto const& data = gpuData_.dataForCurrentDeviceAsync(stream, [this](DataSoADevice& data, cudaStream_t stream) {
								      data.noise_ = cms::cuda::make_device_unique<std::uint16_t[]>(noise_.size(), stream);
								      data.invthick_ = cms::cuda::make_device_unique<float[]>(invthick_.size(), stream);
								      data.detID_ = cms::cuda::make_device_unique<detId_t[]>(detID_.size(), stream);
								      data.iPair_ = cms::cuda::make_device_unique<apvPair_t[]>(iPair_.size(), stream);
								      data.gain_ = cms::cuda::make_device_unique<float[]>(gain_.size(), stream);
								      
								      cms::cuda::copyAsync(data.noise_, noise_, stream);
								      cms::cuda::copyAsync(data.invthick_, invthick_, stream);
								      cms::cuda::copyAsync(data.detID_, detID_, stream);
								      cms::cuda::copyAsync(data.iPair_, iPair_, stream);
								      cms::cuda::copyAsync(data.gain_, gain_, stream);
								    });
      
      return data;
    }

    const DetToFeds& detToFeds() const { return detToFeds_; }

  private:
    void setStrip(fedId_t fed, fedCh_t channel, stripId_t strip, std::uint16_t noise, float gain, bool bad) {
      gain_[apvIndex(fed, channel, strip)] = gain;
      noise_[stripIndex(fed, channel, strip)] = noise;
      if (bad) {
        noise_[stripIndex(fed, channel, strip)] |= badBit;
      }
    }

    void setInvThickness(fedId_t fed, fedCh_t channel, float invthick) {
      invthick_[channelIndex(fed, channel)] = invthick;
    }

    // Helper that takes care of complexity of transferring the data to
    // multiple devices
    cms::cuda::ESProduct<Data> gpuData_;
    DetToFeds detToFeds_;
  };

}  // namespace stripgpu

#endif
