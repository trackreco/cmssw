#include "RecoTracker/MkFit/plugins/MkFitXGBoostEvaluator.h"

#include "FWCore/Utilities/interface/Exception.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <limits>
#include <utility>

namespace {
  using json = nlohmann::json;

  // xgboost writes the learner parameters as strings ("5E-1", "41"); accept numbers too
  double asDouble(const json& j) { return j.is_string() ? std::stod(j.get<std::string>()) : j.get<double>(); }
  long asLong(const json& j) {
    if (j.is_string())
      return std::stol(j.get<std::string>());
    if (j.is_boolean())
      return j.get<bool>();
    return j.get<long>();
  }

  const json& at(const json& j, const char* key, const char* where) {
    const auto it = j.find(key);
    if (it == j.end())
      throw cms::Exception("XGBoostEvaluator") << "no '" << key << "' in " << where;
    return *it;
  }
}  // namespace

namespace mkfitseedclf {

  // The trees padded to the depth of the deepest one, in heap order (the header). Called by load() on the nodes as
  // read and checked there; children come after their parent, so every walk ends at a leaf.
  void XGBoostEvaluator::build(const std::vector<Node>& nodes, const std::vector<int>& roots, const std::string& path) {
    nTrees_ = roots.size();
    depth_ = 0;
    std::vector<std::pair<int, int>> stack;  // (node, depth)
    for (const int root : roots) {
      stack.assign(1, {root, 0});
      while (!stack.empty()) {
        const auto [k, d] = stack.back();
        stack.pop_back();
        depth_ = std::max(depth_, d);
        if (nodes[k].left >= 0) {
          stack.push_back({nodes[k].left, d + 1});
          stack.push_back({nodes[k].right, d + 1});
        }
      }
    }
    if (depth_ > kMaxDepth)
      throw cms::Exception("XGBoostEvaluator")
          << path << ": trees of depth " << depth_ << ", at most " << kMaxDepth << " are supported";
    nSplits_ = (1 << depth_) - 1;
    nLeaves_ = 1 << depth_;
    threshold_.assign(size_t(nTrees_) * nSplits_, 0.f);
    feature_.assign(size_t(nTrees_) * nSplits_, 0);
    nanRight_.assign(size_t(nTrees_) * nSplits_, 0);
    leaf_.assign(size_t(nTrees_) * nLeaves_, 0.f);
    // (tree, model node, heap position, level)
    struct Item {
      int t, k, h, level;
    };
    std::vector<Item> todo;
    for (int t = 0; t < nTrees_; ++t) {
      todo.assign(1, {t, roots[t], 0, 0});
      while (!todo.empty()) {
        const Item it = todo.back();
        todo.pop_back();
        const Node& nd = nodes[it.k];
        if (it.level == depth_) {  // a leaf of the model, at full depth or below a padding split
          leaf_[size_t(it.t) * nLeaves_ + it.h - nSplits_] = nd.value;
          continue;
        }
        const size_t s = size_t(it.t) * nSplits_ + it.h;
        const bool leaf = nd.left < 0;
        // a padding split goes left for every value but +inf, and both of its subtrees end in the same leaf
        threshold_[s] = leaf ? std::numeric_limits<float>::infinity() : nd.value;
        feature_[s] = leaf ? 0 : nd.feature;
        nanRight_[s] = leaf ? 0 : !nd.defaultLeft;
        todo.push_back({it.t, leaf ? it.k : nd.left, 2 * it.h + 1, it.level + 1});
        todo.push_back({it.t, leaf ? it.k : nd.right, 2 * it.h + 2, it.level + 1});
      }
    }
  }

  void XGBoostEvaluator::load(const std::string& path) {
    std::ifstream in(path);
    if (!in)
      throw cms::Exception("XGBoostEvaluator") << "cannot open " << path;
    json doc;
    try {
      doc = json::parse(in);
    } catch (const json::exception& e) {
      throw cms::Exception("XGBoostEvaluator") << "cannot parse " << path << ": " << e.what();
    }

    try {
      const json& learner = at(doc, "learner", path.c_str());
      const json& mp = at(learner, "learner_model_param", "learner");
      numFeature_ = asLong(at(mp, "num_feature", "learner_model_param"));
      if (mp.contains("num_class") && asLong(mp["num_class"]) > 1)
        throw cms::Exception("XGBoostEvaluator") << path << ": multi-class models are not supported";
      const double baseScore = asDouble(at(mp, "base_score", "learner_model_param"));

      objective_ = at(at(learner, "objective", "learner"), "name", "objective").get<std::string>();
      if (objective_ != "binary:logistic" && objective_ != "reg:logistic")
        throw cms::Exception("XGBoostEvaluator")
            << path << ": objective " << objective_ << ", only binary:logistic and reg:logistic are supported";
      if (!(baseScore > 0 && baseScore < 1))
        throw cms::Exception("XGBoostEvaluator") << path << ": base_score " << baseScore << " is not in (0, 1)";
      baseMargin_ = -std::log(1. / baseScore - 1.);

      const json& gb = at(learner, "gradient_booster", "learner");
      const std::string gbName = at(gb, "name", "gradient_booster").get<std::string>();
      if (gbName != "gbtree")
        throw cms::Exception("XGBoostEvaluator") << path << ": booster " << gbName << ", only gbtree is supported";
      const json& trees = at(at(gb, "model", "gradient_booster"), "trees", "model");

      std::vector<Node> nodes;
      std::vector<int> roots;
      for (const json& t : trees) {
        const json& si = at(t, "split_indices", "tree");
        const json& sc = at(t, "split_conditions", "tree");
        const json& lc = at(t, "left_children", "tree");
        const json& rc = at(t, "right_children", "tree");
        const json& dl = at(t, "default_left", "tree");
        const size_t n = lc.size();
        if (si.size() != n || sc.size() != n || rc.size() != n || dl.size() != n || n == 0)
          throw cms::Exception("XGBoostEvaluator") << path << ": inconsistent node arrays in tree " << roots.size();
        if (t.contains("split_type"))
          for (const json& s : t["split_type"])
            if (asLong(s) != 0)
              throw cms::Exception("XGBoostEvaluator") << path << ": categorical splits are not supported";
        const int base = nodes.size();
        roots.push_back(base);
        for (size_t k = 0; k < n; ++k) {
          Node nd;
          nd.value = static_cast<float>(sc[k].get<double>());
          const long l = asLong(lc[k]), r = asLong(rc[k]);
          const bool leaf = l < 0;
          nd.left = leaf ? -1 : base + l;
          nd.right = leaf ? -1 : base + r;
          nd.feature = leaf ? 0 : asLong(si[k]);
          nd.defaultLeft = asLong(dl[k]) != 0;
          // children come after their parent in xgboost's trees, which also rules out a cycle
          if (!leaf && (l <= long(k) || l >= long(n) || r <= long(k) || r >= long(n) || nd.feature < 0 ||
                        nd.feature >= numFeature_))
            throw cms::Exception("XGBoostEvaluator") << path << ": bad node " << k << " in tree " << roots.size() - 1;
          nodes.push_back(nd);
        }
      }
      build(nodes, roots, path);
    } catch (const json::exception& e) {
      throw cms::Exception("XGBoostEvaluator") << "unexpected content in " << path << ": " << e.what();
    }
  }

}  // namespace mkfitseedclf
