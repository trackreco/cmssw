#ifndef RecoTracker_MkFitCore_standalone_RdfTrace_AnRun_h
#define RecoTracker_MkFitCore_standalone_RdfTrace_AnRun_h

#include "RecoTracker/MkFitCore/standalone/RdfTrace/RdfSources.h"
#include "RecoTracker/MkFitCore/standalone/RdfTrace/CanvasGroup.h"

#include "ROOT/RDataFrame.hxx"

#include <optional>

class CanvasGroup;

struct AnRun {
  using ANode = std::optional<ROOT::RDF::RNode>;
  using RNode = ROOT::RDF::RNode;
  using RVecI = ROOT::RVec<int>;
  using RVecF = ROOT::RVec<float>;

  std::vector<std::unique_ptr<CanvasGroup>> m_canvas_groups;

  mkfit::RdfCtx CTX;

  AnRun(const mkfit::TrackerInfo& ti) : CTX(mkfit::MakeCtx(nullptr, ti))
  {}

  AnRun(const mkfit::Event* ev, const mkfit::TrackerInfo& ti) :
    CTX(mkfit::MakeCtx(ev, ti))
  {}

  ~AnRun();

  // -----

  ANode m_rdf_hitmatch;
  ANode m_rdf_meta;
  ANode f_resdy_400;

  void RunOldVecBased();

  // -----

  ANode m_rdf_event;
  std::vector<const mkfit::Event*> m_ev_vec;
  const mkfit::Event* get_event_ptr(int event_id) const;

  // Swaps out ev_vec and owns it.
  void SetupRdfEvent(std::vector<const mkfit::Event*>& ev_vec);

  void RunBasicSeedCandCheck();

  void RunMetaVsSeedDuplicateCheck();

  void Run_T5_vs_pT5_AsSeeds_DuplicateCount();

  void Run_T5s_into_Pix();

  ANode m_T5;

  // CanvasGroup management

  CanvasGroup& NewCanvasGroup(const std::string &n="", const std::string &t="", const std::string &pfx="") {
    m_canvas_groups.emplace_back( std::make_unique<CanvasGroup>(n,t,pfx) );
    return *m_canvas_groups.back();
  }
  CanvasGroup& NewCanvasGroup(int dx=1, int dy=1, const std::string &n="", const std::string &t="", const std::string &pfx="") {
    m_canvas_groups.emplace_back( std::make_unique<CanvasGroup>(dx, dy, n, t, pfx) );
    return *m_canvas_groups.back();
  }

  void DrawCanvasGroups();
  void WriteCanvasGroupsToFile(const std::string &fname) const;
};

#endif
