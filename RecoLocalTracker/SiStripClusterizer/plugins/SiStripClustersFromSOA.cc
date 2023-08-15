/*
 */
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersSoAHost.h"

#include "CUDADataFormats/Common/interface/Product.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include <iostream>
#include <memory>

class SiStripClustersFromSOA final : public edm::stream::EDProducer<> {
public:
  explicit SiStripClustersFromSOA(const edm::ParameterSet& conf)
      : inputToken_(consumes(conf.getParameter<edm::InputTag>("ProductLabel"))),
        outputToken_(produces<edmNew::DetSetVector<SiStripCluster>>()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add("ProductLabel", edm::InputTag("siStripClustersSOAtoHost"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void produce(edm::Event& ev, const edm::EventSetup& es) override {
    const auto& clust_data = ev.get(inputToken_);

    const int nSeedStripsNC = clust_data->nClusters();
    const auto clusterSize = clust_data->clusterSize();
    const auto clusterADCs = clust_data->clusterADCs();
    const auto detIDs = clust_data->clusterDetId();
    const auto stripIDs = clust_data->firstStrip();
    const auto trueCluster = clust_data->trueCluster();

    const unsigned int initSeedStripsSize = 15000;

    using out_t = edmNew::DetSetVector<SiStripCluster>;
    auto output{std::make_unique<out_t>(edmNew::DetSetVector<SiStripCluster>())};
    output->reserve(initSeedStripsSize, nSeedStripsNC);

    std::vector<uint8_t> adcs;

    for (int i = 0; i < nSeedStripsNC;) {
      const auto detid = detIDs[i];
      out_t::FastFiller record(*output, detid);

      while (i < nSeedStripsNC && detIDs[i] == detid) {
        if (trueCluster[i]) {
          const auto size = clusterSize[i];
          const auto firstStrip = stripIDs[i];

          adcs.clear();
          adcs.reserve(size);

          for (uint32_t j = 0; j < size; ++j) {
            adcs.push_back(clusterADCs[i][j]);
          }
          record.push_back(SiStripCluster(firstStrip, std::move(adcs)));
        }
        i++;
      }
    }

    output->shrink_to_fit();
    ev.put(std::move(output));
  }

private:
  edm::EDGetTokenT<SiStripClustersSoAHost> inputToken_;
  edm::EDPutTokenT<edmNew::DetSetVector<SiStripCluster>> outputToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClustersFromSOA);
