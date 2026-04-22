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

  std::vector<std::unique_ptr<CanvasGroup>> m_canvas_groups;

  mkfit::RdfCtx CTX;

  AnRun(const mkfit::TrackerInfo& ti) : CTX(mkfit::MakeCtx(nullptr, ti))
  {}

  AnRun(const mkfit::Event* ev, const mkfit::TrackerInfo& ti) : 
    CTX(mkfit::MakeCtx(ev, ti))
  {}

  // -----

  ANode m_rdf_hitmatch;
  ANode m_rdf_meta;
  ANode f_resdy_400;

  void RunOldVecBased();

  // -----

  ANode m_rdf_event;

  void SetupRdfEvent(std::vector<const mkfit::Event*>& ev_vec);

  void RunEventSourceTestAndDupCheck();

  void RunEventSourceSeedDive();

  void RunT5intoPix();

  // CanvasGroup management

  CanvasGroup& NewCanvasGroup(const char *n=0, const char *t=0, const char *pfx=0) {
    m_canvas_groups.emplace_back( std::make_unique<CanvasGroup>(n,t,pfx) );
    return *m_canvas_groups.back();
  }
  CanvasGroup& NewCanvasGroup(int dx=1, int dy=1, const char *n=0, const char *t=0, const char *pfx=0) {
    m_canvas_groups.emplace_back( std::make_unique<CanvasGroup>(dx, dy, n, t, pfx) );
    return *m_canvas_groups.back();
  }

  void DrawCanvasGroups();
};

#endif
