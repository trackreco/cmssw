#ifndef RecoTracker_MkFit_plugins_MkFitXGBoostEvaluator_h
#define RecoTracker_MkFit_plugins_MkFitXGBoostEvaluator_h

// XGBoostEvaluator: a plain C++ evaluator of a gradient-boosted tree model saved by xgboost's
// Booster.save_model("x.json") (the JSON format of xgboost 1.7), for the objectives binary:logistic and
// reg:logistic. No dependency on the xgboost library.
//
// What is read: learner.learner_model_param.{base_score, num_feature, num_class}, learner.objective.name and,
// for each of learner.gradient_booster.model.trees, split_indices, split_conditions, left_children,
// right_children and default_left. As in xgboost, a node goes left when x[feature] < split_condition, compared
// in float; a NaN feature (xgboost's missing value) follows default_left; a leaf's value is its split_condition.
// The margin is the sum of the leaves plus logit(base_score); the probability is the sigmoid of the margin.
// Categorical splits, multi-class models and dart boosters are rejected.
//
// The trees are stored padded to the depth of the deepest one, in heap order: per tree its 2^depth - 1 splits
// (node h's children are 2h + 1 and 2h + 2) and its 2^depth leaves, tree after tree. A leaf of the model above
// that depth becomes a padding split that always goes left, with the leaf's value in every leaf below it. A seed
// then takes exactly depth steps per tree, each without a branch on the data, and kBatch seeds go through each
// tree together; the trees as xgboost saves them, walked one seed at a time, cost 4.7 times as much on phi3
// (2026-11-13). The leaves are summed in tree order, in double, from the base margin, as before: the same leaf of
// every tree and the same sum, so the scores are unchanged bit for bit.

#include <cstdint>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>

namespace mkfitseedclf {

  class XGBoostEvaluator {
  public:
    // the seeds that go through each tree together
    static constexpr int kBatch = 8;

    XGBoostEvaluator() = default;
    explicit XGBoostEvaluator(const std::string& jsonPath) { load(jsonPath); }

    // Throws cms::Exception on a file it cannot read or a model it does not support.
    void load(const std::string& jsonPath);

    int numFeatures() const { return numFeature_; }
    int numTrees() const { return nTrees_; }
    int depth() const { return depth_; }
    const std::string& objective() const { return objective_; }

    // x: numFeatures() floats; NaN means missing
    double margin(const float* x) const {
      double sum = baseMargin_;
      const float* th = threshold_.data();
      const int32_t* fe = feature_.data();
      const int32_t* nr = nanRight_.data();
      const float* lv = leaf_.data();
      for (int t = 0; t < nTrees_; ++t, th += nSplits_, fe += nSplits_, nr += nSplits_, lv += nLeaves_) {
        int32_t h = 0;
        for (int l = 0; l < depth_; ++l)
          h = child(h, x[fe[h]], th[h], nr[h]);
        sum += lv[h - nSplits_];
      }
      return sum;
    }

    // kBatch seeds, numFeatures() floats each, one after the other from x; their margins to out
    void margins(const float* x, double* out) const {
      double sum[kBatch];
      for (int b = 0; b < kBatch; ++b)
        sum[b] = baseMargin_;
      const float* th = threshold_.data();
      const int32_t* fe = feature_.data();
      const int32_t* nr = nanRight_.data();
      const float* lv = leaf_.data();
      for (int t = 0; t < nTrees_; ++t, th += nSplits_, fe += nSplits_, nr += nSplits_, lv += nLeaves_) {
        int32_t h[kBatch];
        for (int b = 0; b < kBatch; ++b)
          h[b] = 0;
        for (int l = 0; l < depth_; ++l)
          for (int b = 0; b < kBatch; ++b)
            h[b] = child(h[b], x[b * numFeature_ + fe[h[b]]], th[h[b]], nr[h[b]]);
        for (int b = 0; b < kBatch; ++b)
          sum[b] += lv[h[b] - nSplits_];
      }
      for (int b = 0; b < kBatch; ++b)
        out[b] = sum[b];
    }

    // The probability of class 1, as xgboost's predict_proba(x)[:, 1].
    double predict(const float* x) const { return sigmoid(margin(x)); }

    // n seeds, numFeatures() floats each, one after the other from x; their probabilities to p
    void predict(const float* x, unsigned int n, double* p) const {
      unsigned int i = 0;
      for (; i + kBatch <= n; i += kBatch) {
        margins(x + static_cast<size_t>(i) * numFeature_, p + i);
        for (int b = 0; b < kBatch; ++b)
          p[i + b] = sigmoid(p[i + b]);
      }
      for (; i < n; ++i)
        p[i] = predict(x + static_cast<size_t>(i) * numFeature_);
    }

  private:
    // a node as xgboost saves it, as load() reads it: children by index over all trees, -1 for a leaf
    struct Node {
      float value;  // the split condition, or the leaf value
      int left;
      int right;
      int feature;
      bool defaultLeft;
    };
    static constexpr int kMaxDepth = 16;
    // the padded trees from the nodes and the trees' roots
    void build(const std::vector<Node>& nodes, const std::vector<int>& roots, const std::string& path);

    static double sigmoid(double m) { return 1. / (1. + std::exp(-m)); }

    // the child of split h for the value v: right when !(v < threshold), a NaN to the right when nanRight
    static int32_t child(int32_t h, float v, float threshold, int32_t nanRight) {
      const int32_t right = isNaN(v) ? nanRight : int32_t(!(v < threshold));
      return 2 * h + 1 + right;
    }

    // a NaN test by the bit pattern, so that it survives -ffast-math
    static bool isNaN(float v) {
      uint32_t u;
      std::memcpy(&u, &v, sizeof u);
      return (u & 0x7fffffffu) > 0x7f800000u;
    }

    // per tree nSplits_ splits and nLeaves_ leaves, in heap order
    std::vector<float> threshold_;
    std::vector<int32_t> feature_;
    std::vector<int32_t> nanRight_;
    std::vector<float> leaf_;
    int nTrees_ = 0;
    int depth_ = 0;
    int nSplits_ = 0;
    int nLeaves_ = 1;
    double baseMargin_ = 0;
    int numFeature_ = 0;
    std::string objective_;
  };

}  // namespace mkfitseedclf

#endif
