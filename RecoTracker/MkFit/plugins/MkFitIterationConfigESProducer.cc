#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "RecoTracker/MkFit/interface/MkFitGeometry.h"

// mkFit includes
#include "Track.h"
#include "TrackerInfo.h"
#include "mkFit/HitStructures.h"
#include "mkFit/IterationConfig.h"

namespace {
  using namespace mkfit;

  void partitionSeeds0(const TrackerInfo &trk_info,
                       const TrackVec &in_seeds,
                       const EventOfHits &eoh,
                       IterationSeedPartition &part) {
    // Seeds are placed into eta regions and sorted on region + eta.

    const size_t size = in_seeds.size();

    for (size_t i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      const bool z_dir_pos = S.pz() > 0;

      HitOnTrack hot = S.getLastHitOnTrack();
      // MIMI ACHTUNG -- here we assume seed hits have already been remapped.
      // This was true at that time :)
      const float eta = eoh[hot.layer].GetHit(hot.index).eta();
      // float  eta = S.momEta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      // Max eta used for region sorting
      constexpr float maxEta_regSort = 7.0;

      // Hardcoded for cms ... needs some lists of layers (hit/miss) for brl / ecp tests.
      // MM: Check lambda functions/std::function
      const LayerInfo &outer_brl = trk_info.outer_barrel_layer();

      const LayerInfo &tib1 = trk_info.m_layers[4];
      const LayerInfo &tob1 = trk_info.m_layers[10];

      const LayerInfo &tecp1 = trk_info.m_layers[27];
      const LayerInfo &tecn1 = trk_info.m_layers[54];

      const LayerInfo &tec_first = z_dir_pos ? tecp1 : tecn1;

      // If a track hits outer barrel ... it is in the barrel (for central, "outgoing" tracks).
      // This is also true for cyl-cow.
      // Better check is: hits outer TIB, misses inner TEC (but is +-z dependant).
      // XXXX Calculate z ... then check is inside or less that first EC z.
      // There are a lot of tracks that go through that crack.

      // XXXX trying a fix for low pT tracks that are in barrel after half circle
      const float maxR = S.maxReachRadius();
      float z_at_maxr;

      bool can_reach_outer_brl = S.canReachRadius(outer_brl.m_rout);
      float z_at_outer_brl;
      bool misses_first_tec;
      if (can_reach_outer_brl) {
        z_at_outer_brl = S.zAtR(outer_brl.m_rout);
        if (z_dir_pos)
          misses_first_tec = z_at_outer_brl < tec_first.m_zmin;
        else
          misses_first_tec = z_at_outer_brl > tec_first.m_zmax;
      } else {
        z_at_maxr = S.zAtR(maxR);
        if (z_dir_pos)
          misses_first_tec = z_at_maxr < tec_first.m_zmin;
        else
          misses_first_tec = z_at_maxr > tec_first.m_zmax;
      }

      if (/*can_reach_outer_brl &&*/ misses_first_tec)
      // outer_brl.is_within_z_limits(S.zAtR(outer_brl.r_mean())))
      {
        reg = TrackerInfo::Reg_Barrel;
      } else {
        // This should be a list of layers
        // CMS, first tib, tob: 4, 10

        if ((S.canReachRadius(tib1.m_rin) && tib1.is_within_z_limits(S.zAtR(tib1.m_rin))) ||
            (S.canReachRadius(tob1.m_rin) && tob1.is_within_z_limits(S.zAtR(tob1.m_rin)))) {
          // transition region ... we are still hitting barrel layers

          reg = z_dir_pos ? TrackerInfo::Reg_Transition_Pos : TrackerInfo::Reg_Transition_Neg;
        } else {
          // endcap ... no barrel layers will be hit anymore.

          reg = z_dir_pos ? TrackerInfo::Reg_Endcap_Pos : TrackerInfo::Reg_Endcap_Neg;
        }
      }

      part.m_region[i] = reg;

      // TrackerInfo::EtaRegion is enum from 0 to 5 (Reg_Endcap_Neg,Reg_Transition_Neg,Reg_Barrel,Reg_Transition_Pos,Reg_Endcap_Pos)
      // Symmetrization around TrackerInfo::Reg_Barrel for sorting is required
      part.m_sort_score[i] = maxEta_regSort * (reg - TrackerInfo::Reg_Barrel) + eta;
    }
  }

  void partitionSeeds1(const TrackerInfo &trk_info,
                       const TrackVec &in_seeds,
                       const EventOfHits &eoh,
                       IterationSeedPartition &part) {
    // Seeds are placed into eta regions and sorted on region + eta.

    const LayerInfo &tib1 = trk_info.m_layers[4];
    //const LayerInfo &tib6 = trk_info.m_layers[9];
    const LayerInfo &tob1 = trk_info.m_layers[10];
    //const LayerInfo &tob8 = trk_info.m_layers[17];

    const LayerInfo &tidp1 = trk_info.m_layers[21];
    const LayerInfo &tidn1 = trk_info.m_layers[48];

    const LayerInfo &tecp1 = trk_info.m_layers[27];
    const LayerInfo &tecn1 = trk_info.m_layers[54];

    // Merge first two layers to account for mono/stereo coverage.
    // TrackerInfo could hold joint limits for sub-detectors.
    const auto &L = trk_info.m_layers;
    const float tidp_rin = std::min(L[21].m_rin, L[22].m_rin);
    const float tidp_rout = std::max(L[21].m_rout, L[22].m_rout);
    const float tecp_rin = std::min(L[27].m_rin, L[28].m_rin);
    const float tecp_rout = std::max(L[27].m_rout, L[28].m_rout);
    const float tidn_rin = std::min(L[48].m_rin, L[49].m_rin);
    const float tidn_rout = std::max(L[48].m_rout, L[49].m_rout);
    const float tecn_rin = std::min(L[54].m_rin, L[55].m_rin);
    const float tecn_rout = std::max(L[54].m_rout, L[55].m_rout);

    // Bias towards more aggressive transition-region assignemnts.
    // With current tunning it seems to make things a bit worse.
    const float tid_z_extra = 0.0f;  // 5.0f;
    const float tec_z_extra = 0.0f;  // 10.0f;

    const int size = in_seeds.size();

    auto barrel_pos_check = [](const Track &S, float maxR, float rin, float zmax) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) < zmax;
      return inside;
    };

    auto barrel_neg_check = [](const Track &S, float maxR, float rin, float zmin) -> bool {
      bool inside = maxR > rin && S.zAtR(rin) > zmin;
      return inside;
    };

    auto endcap_pos_check = [](const Track &S, float maxR, float rout, float rin, float zmin) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) > zmin : (maxR > rin && S.zAtR(maxR) > zmin);
      return inside;
    };

    auto endcap_neg_check = [](const Track &S, float maxR, float rout, float rin, float zmax) -> bool {
      bool inside = maxR > rout ? S.zAtR(rout) < zmax : (maxR > rin && S.zAtR(maxR) < zmax);
      return inside;
    };

    for (int i = 0; i < size; ++i) {
      const Track &S = in_seeds[i];

      HitOnTrack hot = S.getLastHitOnTrack();
      float eta = eoh[hot.layer].GetHit(hot.index).eta();
      // float  eta = S.momEta();

      // Region to be defined by propagation / intersection tests
      TrackerInfo::EtaRegion reg;

      const bool z_dir_pos = S.pz() > 0;
      const float maxR = S.maxReachRadius();

      if (z_dir_pos) {
        bool in_tib = barrel_pos_check(S, maxR, tib1.m_rin, tib1.m_zmax);
        bool in_tob = barrel_pos_check(S, maxR, tob1.m_rin, tob1.m_zmax);

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Pos;
        } else {
          bool in_tid = endcap_pos_check(S, maxR, tidp_rout, tidp_rin, tidp1.m_zmin - tid_z_extra);
          bool in_tec = endcap_pos_check(S, maxR, tecp_rout, tecp_rin, tecp1.m_zmin - tec_z_extra);

          if (!in_tid && !in_tec) {
            reg = TrackerInfo::Reg_Barrel;
          } else {
            reg = TrackerInfo::Reg_Transition_Pos;
          }
        }
      } else {
        bool in_tib = barrel_neg_check(S, maxR, tib1.m_rin, tib1.m_zmin);
        bool in_tob = barrel_neg_check(S, maxR, tob1.m_rin, tob1.m_zmin);

        if (!in_tib && !in_tob) {
          reg = TrackerInfo::Reg_Endcap_Neg;
        } else {
          bool in_tid = endcap_neg_check(S, maxR, tidn_rout, tidn_rin, tidn1.m_zmax + tid_z_extra);
          bool in_tec = endcap_neg_check(S, maxR, tecn_rout, tecn_rin, tecn1.m_zmax + tec_z_extra);

          if (!in_tid && !in_tec) {
            reg = TrackerInfo::Reg_Barrel;
          } else {
            reg = TrackerInfo::Reg_Transition_Neg;
          }
        }
      }

      part.m_region[i] = reg;
      part.m_sort_score[i] = 7.0f * (reg - 2) + eta;
    }
  }
}  // namespace

class MkFitIterationConfigESProducer : public edm::ESProducer {
public:
  MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  std::unique_ptr<mkfit::IterationConfig> produce(const TrackerRecoGeometryRecord &iRecord);

private:
  const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> geomToken_;
  const std::string configFile_;
};

MkFitIterationConfigESProducer::MkFitIterationConfigESProducer(const edm::ParameterSet &iConfig)
    : geomToken_{setWhatProduced(this, iConfig.getParameter<std::string>("ComponentName")).consumes()},
      configFile_{iConfig.getParameter<edm::FileInPath>("config").fullPath()} {}

void MkFitIterationConfigESProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("ComponentName")->setComment("Product label");
  desc.add<edm::FileInPath>("config")->setComment("Path to the JSON file for the mkFit configuration parameters");
  descriptions.addWithDefaultLabel(desc);
}

std::unique_ptr<mkfit::IterationConfig> MkFitIterationConfigESProducer::produce(
    const TrackerRecoGeometryRecord &iRecord) {
  // Avoid unused variable warnings.
  (void)partitionSeeds0;
  (void)partitionSeeds1;
  auto it_conf = mkfit::ConfigJson_Load_File(configFile_);
  it_conf->m_partition_seeds = partitionSeeds1;
  return it_conf;
}

DEFINE_FWK_EVENTSETUP_MODULE(MkFitIterationConfigESProducer);
