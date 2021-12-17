#include "RecoTracker/MkFitCore/interface/MkBuilderWrapper.h"
#include "RecoTracker/MkFitCore/interface/MkBuilder.h"

namespace mkfit {
  MkBuilderWrapper::MkBuilderWrapper() : builder_(MkBuilder::make_builder()) {}

  MkBuilderWrapper::~MkBuilderWrapper() {}

  void MkBuilderWrapper::populate() { MkBuilder::populate(); }
}  // namespace mkfit
