#ifndef RecoTracker_MkFitCore_standalone_RdfTrace_AnRun_h
#define RecoTracker_MkFitCore_standalone_RdfTrace_AnRun_h

#include "RecoTracker/MkFitCore/standalone/RdfTrace/RdfSources.h"
#include "RecoTracker/MkFitCore/standalone/RdfTrace/CanvasGroup.h"

#include "ROOT/RDataFrame.hxx"

class CanvasGroup;

struct AnRun {

  std::unique_ptr<ROOT::RDataFrame> m_rdf;
  std::unique_ptr<ROOT::RDF::RNode> m_base;

  std::unique_ptr<ROOT::RDF::RNode> f_first_layer_brl;
  std::unique_ptr<ROOT::RDF::RNode> f_first_layer_ec;

  std::unique_ptr<ROOT::RDF::RNode> f_resdy_400;

  std::vector<std::unique_ptr<CanvasGroup>> m_canvas_groups;

  mkfit::RdfCtx CTX;

  AnRun(const mkfit::Event* ev, const mkfit::TrackerInfo& ti) : 
    CTX(mkfit::MakeCtx(ev, ti))
  {}

  void RunBase();
  
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
