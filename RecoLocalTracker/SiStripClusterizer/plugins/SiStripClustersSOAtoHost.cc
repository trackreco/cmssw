/*
 */
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersSoADevice.h"
#include "CUDADataFormats/SiStripCluster/interface/SiStripClustersSoAHost.h"

#include <memory>

class SiStripSOAtoHost {
public:
  SiStripSOAtoHost() = default;
  void makeAsync(const SiStripClustersSoADevice& clusters_d, cudaStream_t stream) {
    maxClusters_ = clusters_d.maxClusters();
    clusters_h_ = SiStripClustersSoAHost(maxClusters_, stream);
    cudaCheck(cudaMemcpyAsync(clusters_h_.buffer().get(),
                              clusters_d.const_buffer().get(),
                              clusters_d.bufferSize(),
                              cudaMemcpyDeviceToHost,
                              stream));  // Copy data from Device to Host
  }

  SiStripClustersSoAHost getResults() { return std::move(clusters_h_); }

private:
  SiStripClustersSoAHost clusters_h_;
  uint32_t maxClusters_;
};

class SiStripClustersSOAtoHost final : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit SiStripClustersSOAtoHost(const edm::ParameterSet& conf)
      : inputToken_(
            consumes<cms::cuda::Product<SiStripClustersSoADevice>>(conf.getParameter<edm::InputTag>("ProductLabel"))),
        outputToken_(produces<SiStripClustersSoAHost>()) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add("ProductLabel", edm::InputTag("siStripClusterizerFromRawGPU"));
    descriptions.addWithDefaultLabel(desc);
  }

private:
  void acquire(edm::Event const& ev,
               edm::EventSetup const& es,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override {
    const auto& wrapper = ev.get(inputToken_);

    // Sets the current device and creates a CUDA stream
    cms::cuda::ScopedContextAcquire ctx{wrapper, std::move(waitingTaskHolder)};

    const auto& input = ctx.get(wrapper);

    // Queues asynchronous data transfers and kernels to the CUDA stream
    // returned by cms::cuda::ScopedContextAcquire::stream()
    gpuAlgo_.makeAsync(input, ctx.stream());

    // Destructor of ctx queues a callback to the CUDA stream notifying
    // waitingTaskHolder when the queued asynchronous work has finished
  }

  void produce(edm::Event& ev, const edm::EventSetup& es) override { ev.emplace(outputToken_, gpuAlgo_.getResults()); }

private:
  SiStripSOAtoHost gpuAlgo_;

  edm::EDGetTokenT<cms::cuda::Product<SiStripClustersSoADevice>> inputToken_;
  edm::EDPutTokenT<SiStripClustersSoAHost> outputToken_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripClustersSOAtoHost);
