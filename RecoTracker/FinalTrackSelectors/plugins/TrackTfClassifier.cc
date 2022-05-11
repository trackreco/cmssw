#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTracker/FinalTrackSelectors/interface/getBestVertex.h"

#include "TrackingTools/Records/interface/TfGraphRecord.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "RecoTracker/FinalTrackSelectors/interface/TfGraphDefWrapper.h"

namespace {
  class TfDnn {
  public:
    TfDnn(const edm::ParameterSet& cfg, edm::ConsumesCollector iC)
        : tfDnnLabel_(cfg.getParameter<std::string>("tfDnnLabel")),
          tfDnnToken_(iC.esConsumes(edm::ESInputTag("", tfDnnLabel_))),
          session_(nullptr)

    {}

    static const char* name() { return "trackTfClassifierDefault"; }

    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<std::string>("tfDnnLabel", "trackSelectionTf");
    }
    void beginStream() {}

    void initEvent(const edm::EventSetup& es) {
      if (session_ == nullptr) {
        session_ = es.getData(tfDnnToken_).getSession();
      }
    }

    float operator()(reco::Track const& trk,
                     reco::BeamSpot const& beamSpot,
                     reco::VertexCollection const& vertices) const {
      const auto& bestVertex = getBestVertex(trk, vertices);

      tensorflow::Tensor input1(tensorflow::DT_FLOAT, {1, 29});
      tensorflow::Tensor input2(tensorflow::DT_FLOAT, {1, 1});

      input1.matrix<float>()(0, 0) = trk.pt();
      input1.matrix<float>()(0, 1) = trk.innerMomentum().x();
      input1.matrix<float>()(0, 2) = trk.innerMomentum().y();
      input1.matrix<float>()(0, 3) = trk.innerMomentum().z();
      input1.matrix<float>()(0, 4) = trk.innerMomentum().rho();
      input1.matrix<float>()(0, 5) = trk.outerMomentum().x();
      input1.matrix<float>()(0, 6) = trk.outerMomentum().y();
      input1.matrix<float>()(0, 7) = trk.outerMomentum().z();
      input1.matrix<float>()(0, 8) = trk.outerMomentum().rho();
      input1.matrix<float>()(0, 9) = trk.ptError();
      input1.matrix<float>()(0, 10) = trk.dxy(bestVertex);
      input1.matrix<float>()(0, 11) = trk.dz(bestVertex);
      input1.matrix<float>()(0, 12) = trk.dxy(beamSpot.position());
      input1.matrix<float>()(0, 13) = trk.dz(beamSpot.position());
      input1.matrix<float>()(0, 14) = trk.dxyError();
      input1.matrix<float>()(0, 15) = trk.dzError();
      input1.matrix<float>()(0, 16) = trk.normalizedChi2();
      input1.matrix<float>()(0, 17) = trk.eta();
      input1.matrix<float>()(0, 18) = trk.phi();
      input1.matrix<float>()(0, 19) = trk.etaError();
      input1.matrix<float>()(0, 20) = trk.phiError();
      input1.matrix<float>()(0, 21) = trk.hitPattern().numberOfValidPixelHits();
      input1.matrix<float>()(0, 22) = trk.hitPattern().numberOfValidStripHits();
      input1.matrix<float>()(0, 23) = trk.ndof();
      input1.matrix<float>()(0, 24) = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
      input1.matrix<float>()(0, 25) = trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
      input1.matrix<float>()(0, 26) =
          trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_INNER_HITS);
      input1.matrix<float>()(0, 27) =
          trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_OUTER_HITS);
      input1.matrix<float>()(0, 28) = trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);

      //Original algo as its own input, it will enter the graph so that it gets one-hot encoded, as is the preferred
      //format for categorical inputs, where the labels do not have any metric amongst them
      input2.matrix<float>()(0, 0) = trk.originalAlgo();

      //The names for the input tensors get locked when freezing the trained tensorflow model. The NamedTensors must
      //match those names
      tensorflow::NamedTensorList inputs;
      inputs.resize(2);
      inputs[0] = tensorflow::NamedTensor("x", input1);
      inputs[1] = tensorflow::NamedTensor("y", input2);
      std::vector<tensorflow::Tensor> outputs;

      //evaluate the input
      tensorflow::run(const_cast<tensorflow::Session*>(session_), inputs, {"Identity"}, &outputs);
      //scale output to be [-1, 1] due to convention
      float output = 2.0 * outputs[0].matrix<float>()(0, 0) - 1.0;
      return output;
    }

    const std::string tfDnnLabel_;
    const edm::ESGetToken<TfGraphDefWrapper, TfGraphRecord> tfDnnToken_;
    const tensorflow::Session* session_;
  };

  class TfDnnBatch {
  public:
    TfDnnBatch(const edm::ParameterSet& cfg, edm::ConsumesCollector iC)
        : tfDnnLabel_(cfg.getParameter<std::string>("tfDnnLabel")),
          tfDnnToken_(iC.esConsumes(edm::ESInputTag("", tfDnnLabel_))),
          session_(nullptr),
          bsize_(cfg.getParameter<int>("batchSize"))

    {}

    static const char* name() { return "trackTfClassifierDefaultBatch"; }

    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<std::string>("tfDnnLabel", "trackSelectionTf");
      desc.add<int>("batchSize", 16);
    }
    void beginStream() {}

    void initEvent(const edm::EventSetup& es) {
      if (session_ == nullptr) {
        session_ = es.getData(tfDnnToken_).getSession();
      }
    }

    std::vector<float> operator()(reco::TrackCollection const& tracks,
                                  reco::BeamSpot const& beamSpot,
                                  reco::VertexCollection const& vertices) const {
      //std::cout <<"in batch eval part"<< std::endl;

      int size_in = (int)tracks.size();
      int bsize = bsize_;
      int nbatches = size_in / bsize;

      std::vector<float> output;
      output.resize(size_in);

      tensorflow::Tensor input1(tensorflow::DT_FLOAT, {bsize, 29});
      tensorflow::Tensor input2(tensorflow::DT_FLOAT, {bsize, 1});

      for (auto nb = 0; nb < nbatches + 1; nb++) {
        for (auto nt = 0; nt < bsize; nt++) {
          int itrack = nt + bsize * nb;
          if (itrack >= size_in)
            continue;
          const auto& trk = tracks.at(itrack);

          const auto& bestVertex = getBestVertex(trk, vertices);

          input1.matrix<float>()(nt, 0) = trk.pt();
          input1.matrix<float>()(nt, 1) = trk.innerMomentum().x();
          input1.matrix<float>()(nt, 2) = trk.innerMomentum().y();
          input1.matrix<float>()(nt, 3) = trk.innerMomentum().z();
          input1.matrix<float>()(nt, 4) = trk.innerMomentum().rho();
          input1.matrix<float>()(nt, 5) = trk.outerMomentum().x();
          input1.matrix<float>()(nt, 6) = trk.outerMomentum().y();
          input1.matrix<float>()(nt, 7) = trk.outerMomentum().z();
          input1.matrix<float>()(nt, 8) = trk.outerMomentum().rho();
          input1.matrix<float>()(nt, 9) = trk.ptError();
          input1.matrix<float>()(nt, 10) = trk.dxy(bestVertex);
          input1.matrix<float>()(nt, 11) = trk.dz(bestVertex);
          input1.matrix<float>()(nt, 12) = trk.dxy(beamSpot.position());
          input1.matrix<float>()(nt, 13) = trk.dz(beamSpot.position());
          input1.matrix<float>()(nt, 14) = trk.dxyError();
          input1.matrix<float>()(nt, 15) = trk.dzError();
          input1.matrix<float>()(nt, 16) = trk.normalizedChi2();
          input1.matrix<float>()(nt, 17) = trk.eta();
          input1.matrix<float>()(nt, 18) = trk.phi();
          input1.matrix<float>()(nt, 19) = trk.etaError();
          input1.matrix<float>()(nt, 20) = trk.phiError();
          input1.matrix<float>()(nt, 21) = trk.hitPattern().numberOfValidPixelHits();
          input1.matrix<float>()(nt, 22) = trk.hitPattern().numberOfValidStripHits();
          input1.matrix<float>()(nt, 23) = trk.ndof();
          input1.matrix<float>()(nt, 24) =
              trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
          input1.matrix<float>()(nt, 25) =
              trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
          input1.matrix<float>()(nt, 26) =
              trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_INNER_HITS);
          input1.matrix<float>()(nt, 27) =
              trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_OUTER_HITS);
          input1.matrix<float>()(nt, 28) =
              trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);

          //Original algo as its own input, it will enter the graph so that it gets one-hot encoded, as is the preferred
          //format for categorical inputs, where the labels do not have any metric amongst them
          input2.matrix<float>()(nt, 0) = trk.originalAlgo();
        }

        //The names for the input tensors get locked when freezing the trained tensorflow model. The NamedTensors must
        //match those names
        tensorflow::NamedTensorList inputs;
        inputs.resize(2);
        inputs[0] = tensorflow::NamedTensor("x", input1);
        inputs[1] = tensorflow::NamedTensor("y", input2);
        std::vector<tensorflow::Tensor> outputs;

        //evaluate the input
        tensorflow::run(const_cast<tensorflow::Session*>(session_), inputs, {"Identity"}, &outputs);

        for (auto nt = 0; nt < bsize; nt++) {
          int itrack = nt + bsize * nb;
          if (itrack >= size_in)
            continue;
          float out0 = 2.0 * outputs[0].matrix<float>()(nt, 0) - 1.0;
          output[itrack] = out0;
        }
      }
      return output;
    }

    const std::string tfDnnLabel_;
    const edm::ESGetToken<TfGraphDefWrapper, TfGraphRecord> tfDnnToken_;
    const tensorflow::Session* session_;
    const int bsize_;
  };

  //   using TrackTfClassifier = TrackMVAClassifier<TfDnn>;
  //   using TrackTfClassifierBatch = TrackMVAClassifier<TfDnnBatch>;

}  // namespace

template <>
void trackMVAClassifierImpl::ComputeMVA<void>::operator()(::TfDnnBatch const& mva,
                                                          reco::TrackCollection const& tracks,
                                                          reco::BeamSpot const& beamSpot,
                                                          reco::VertexCollection const& vertices,
                                                          TrackMVAClassifierBase::MVAPairCollection& mvas) {
  std::vector<float> scores = mva(tracks, beamSpot, vertices);
  size_t current = 0;

  for (auto score : scores) {
    std::pair<float, bool> output(score, true);
    mvas[current++] = output;
  }
}

namespace {
  using TrackTfClassifier = TrackMVAClassifier<TfDnn>;
  using TrackTfClassifierBatch = TrackMVAClassifier<TfDnnBatch>;

}  // namespace

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackTfClassifier);
DEFINE_FWK_MODULE(TrackTfClassifierBatch);

