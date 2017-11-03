#ifndef __DYNET_LAYER_H__
#define __DYNET_LAYER_H__

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include "dynet/dynet.h"
#include "dynet/expr.h"
#include "dynet/rnn.h"
#include "dynet/param-init.h"

struct LayerI {
  enum ACTIVATION_TYPE { kRelu, kTanh };
  bool trainable;
  LayerI(bool trainable) : trainable(trainable) {}
  void active_training() { trainable = true; }
  void inactive_training() { trainable = false; }
  // Initialize parameter
  virtual void new_graph(dynet::ComputationGraph& cg) = 0;
  virtual std::vector<dynet::Expression> get_params() = 0;
};

struct SymbolEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;

  SymbolEmbedding(dynet::ParameterCollection& m,
                  unsigned n,
                  unsigned dim,
                  bool trainable = true);
  void new_graph(dynet::ComputationGraph& cg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression embed(unsigned label_id);
};

struct BinnedDistanceEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;
  unsigned max_bin;

  BinnedDistanceEmbedding(dynet::ParameterCollection& m,
                          unsigned hidden,
                          unsigned n_bin = 8,
                          bool trainable = true);
  void new_graph(dynet::ComputationGraph& cg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression embed(int distance);
};

struct BinnedDurationEmbedding : public LayerI {
  dynet::ComputationGraph* cg;
  dynet::LookupParameter p_e;
  unsigned max_bin;

  BinnedDurationEmbedding(dynet::ParameterCollection& m,
                          unsigned hidden,
                          unsigned n_bin = 8,
                          bool trainable = true);
  void new_graph(dynet::ComputationGraph& cg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression embed(unsigned dur);
};

typedef std::pair<dynet::Expression, dynet::Expression> BiRNNOutput;

template<typename RNNBuilderType>
struct RNNLayer : public LayerI {
  unsigned n_items;
  RNNBuilderType rnn;
  dynet::Parameter p_guard;
  dynet::Expression guard;
  bool has_guard;
  bool reversed;

  RNNLayer(dynet::ParameterCollection& model,
           unsigned n_layers,
           unsigned dim_input,
           unsigned dim_hidden,
           bool rev = false,
           bool has_guard = true,
           bool trainable = true) :
    LayerI(trainable),
    n_items(0),
    rnn(n_layers, dim_input, dim_hidden, model),
    p_guard(model.add_parameters({ dim_input, 1 })),
    has_guard(has_guard),
    reversed(rev) {
  }

  void add_inputs(const std::vector<dynet::Expression>& inputs) {
    n_items = inputs.size();
    rnn.start_new_sequence();
    if (has_guard) { rnn.add_input(guard); }
    if (reversed) {
      for (int i = n_items - 1; i >= 0; --i) { rnn.add_input(inputs[i]); }
    } else {
      for (unsigned i = 0; i < n_items; ++i) { rnn.add_input(inputs[i]); }
    }
  }

  dynet::Expression get_output(dynet::ComputationGraph* hg, int index) {
    if (reversed) { return rnn.get_h(dynet::RNNPointer(n_items - index)).back(); }
    return rnn.get_h(dynet::RNNPointer(index + 1)).back();
  }

  void get_outputs(dynet::ComputationGraph* hg,
                   std::vector<dynet::Expression>& outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) { outputs[i] = get_output(hg, i); }
  }

  dynet::Expression get_final() {
    return rnn.back();
  }

  void new_graph(dynet::ComputationGraph& hg) {
    if (!trainable) {
      std::cerr << "WARN: not-trainable RNN is not implemented." << std::endl;
    }
    rnn.new_graph(hg);
    if (has_guard) {
      guard = dynet::parameter(hg, p_guard);
    }
  }

  std::vector<dynet::Expression> get_params() override {
    std::vector<dynet::Expression> ret;
    for (auto & layer : rnn.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
    if (has_guard) { ret.push_back(guard); }
    return ret;
  }

  void set_dropout(float& rate) { rnn.set_dropout(rate); }
  void disable_dropout() { rnn.disable_dropout(); }
};

template<typename RNNBuilderType>
struct BiRNNLayer : public LayerI {
  unsigned n_items;
  bool has_guard;
  RNNBuilderType fw_rnn;
  RNNBuilderType bw_rnn;
  dynet::Parameter p_fw_guard;
  dynet::Parameter p_bw_guard;
  dynet::Expression fw_guard;
  dynet::Expression bw_guard;
  std::vector<dynet::Expression> fw_hidden;
  std::vector<dynet::Expression> bw_hidden;

  BiRNNLayer(dynet::ParameterCollection& model,
             unsigned n_layers,
             unsigned dim_input,
             unsigned dim_hidden,
             bool has_guard = true,
             bool trainable = true) :
    LayerI(trainable),
    n_items(0),
    has_guard(has_guard),
    fw_rnn(n_layers, dim_input, dim_hidden, model),
    bw_rnn(n_layers, dim_input, dim_hidden, model),
    p_fw_guard(model.add_parameters({ dim_input, 1 })),
    p_bw_guard(model.add_parameters({ dim_input, 1 })) {
  }

  void add_inputs(const std::vector<dynet::Expression>& inputs) {
    n_items = inputs.size();
    fw_rnn.start_new_sequence();
    bw_rnn.start_new_sequence();
    fw_hidden.resize(n_items);
    bw_hidden.resize(n_items);

    if (has_guard) {
      fw_rnn.add_input(fw_guard);
      bw_rnn.add_input(bw_guard);
    }
    for (unsigned i = 0; i < n_items; ++i) {
      fw_hidden[i] = fw_rnn.add_input(inputs[i]);
      bw_hidden[n_items - i - 1] = bw_rnn.add_input(inputs[n_items - i - 1]);
    }

  }

  BiRNNOutput get_output(int index) {
    return std::make_pair(fw_hidden[index], bw_hidden[index]);
  }

  BiRNNOutput get_final() {
    return std::make_pair(fw_hidden[n_items - 1], bw_hidden[0]);
  }

  void get_outputs(std::vector<BiRNNOutput>& outputs) {
    outputs.resize(n_items);
    for (unsigned i = 0; i < n_items; ++i) {
      outputs[i] = get_output(i);
    }
  }

  void new_graph(dynet::ComputationGraph& hg) override {
    if (!trainable) {
      std::cerr << "WARN: not-trainable RNN is not implemented." << std::endl;
    }
    fw_rnn.new_graph(hg);
    bw_rnn.new_graph(hg);
    if (has_guard) {
      fw_guard = dynet::parameter(hg, p_fw_guard);
      bw_guard = dynet::parameter(hg, p_bw_guard);
    }
  }

  std::vector<dynet::Expression> get_params() override {
    std::vector<dynet::Expression> ret;
    for (auto & layer : fw_rnn.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
    for (auto & layer : bw_rnn.param_vars) { for (auto & e : layer) { ret.push_back(e); } }
    if (has_guard) {
      ret.push_back(fw_guard);
      ret.push_back(bw_guard);
    }
    return ret;
  }

  void set_dropout(float& rate) {
    fw_rnn.set_dropout(rate);
    bw_rnn.set_dropout(rate);
  }

  void disable_dropout() {
    fw_rnn.disable_dropout();
    bw_rnn.disable_dropout();
  }
};

struct Conv1dLayer : public LayerI {
  std::vector<dynet::Parameter> p_filters;
  std::vector<dynet::Parameter> p_biases;
  std::vector<dynet::Expression> filters;
  std::vector<dynet::Expression> biases;
  dynet::Expression padding;
  std::vector<std::pair<unsigned, unsigned>> filters_info;
  ACTIVATION_TYPE activation;
  unsigned n_filter_types;
  unsigned dim;
  bool has_bias;

  Conv1dLayer(dynet::ParameterCollection & m,
              unsigned dim,
              const std::vector<std::pair<unsigned, unsigned>>& filter_info,
              ACTIVATION_TYPE actionation = kTanh,
              bool has_bias = false,
              bool trainable = true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const std::vector<dynet::Expression>& exprs);
};

struct InputLayer : public LayerI {
  dynet::ComputationGraph * _cg;
  unsigned dim;
  InputLayer(unsigned dim) : LayerI(false), dim(dim) {}

  void new_graph(dynet::ComputationGraph& cg) override {
    _cg = &cg;
  }

  std::vector<dynet::Expression> get_params() override {
    return std::vector<dynet::Expression>();
  }

  dynet::Expression get_output(const std::vector<float> & data) {
    return dynet::input(*_cg, { dim }, data);
  }
};

struct SoftmaxLayer : public LayerI {
  dynet::Parameter p_B, p_W;
  dynet::Expression B, W;

  SoftmaxLayer(dynet::ParameterCollection& model,
               unsigned dim_input,
               unsigned dim_output,
               bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression & expr);
};

struct DenseLayer : public LayerI {
  dynet::Parameter p_W, p_B;
  dynet::Expression W, B;

  DenseLayer(dynet::ParameterCollection& model,
             unsigned dim_input,
             unsigned dim_output,
             bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression & expr);
};

struct BiLinearLayer : public LayerI {
  dynet::Parameter p_W;
  dynet::Expression W;

  BiLinearLayer(dynet::ParameterCollection & model,
                unsigned dim,
                bool trainable = true);

  void new_graph(dynet::ComputationGraph & cg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression & expr1,
                               const dynet::Expression & expr2);
};

struct BiAffineLayer : public LayerI {
  dynet::Parameter p_W;
  dynet::Expression W;

  BiAffineLayer(dynet::ParameterCollection & model,
                unsigned dim,
                bool trainable = true);

  void new_graph(dynet::ComputationGraph & cg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression & expr1,
                               const dynet::Expression & expr2);

};

struct Merge2Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2;
  dynet::Expression B, W1, W2;

  Merge2Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression & expr1,
                               const dynet::Expression & expr2);
};

struct Merge3Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3;
  dynet::Expression B, W1, W2, W3;

  Merge3Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3);
};

struct Merge4Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4;
  dynet::Expression B, W1, W2, W3, W4;

  Merge4Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3,
                               const dynet::Expression& expr4);
};

struct Merge5Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4, p_W5;
  dynet::Expression B, W1, W2, W3, W4, W5;

  Merge5Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_input5,
              unsigned dim_output,
              bool trainable = true);

  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3,
                               const dynet::Expression& expr4,
                               const dynet::Expression& expr5);
};

struct Merge6Layer : public LayerI {
  dynet::Parameter p_B, p_W1, p_W2, p_W3, p_W4, p_W5, p_W6;
  dynet::Expression B, W1, W2, W3, W4, W5, W6;

  Merge6Layer(dynet::ParameterCollection& model,
              unsigned dim_input1,
              unsigned dim_input2,
              unsigned dim_input3,
              unsigned dim_input4,
              unsigned dim_input5,
              unsigned dim_input6,
              unsigned dim_output,
              bool trainable = true);
  void new_graph(dynet::ComputationGraph& hg) override;
  std::vector<dynet::Expression> get_params() override;
  dynet::Expression get_output(const dynet::Expression& expr1,
                               const dynet::Expression& expr2,
                               const dynet::Expression& expr3,
                               const dynet::Expression& expr4,
                               const dynet::Expression& expr5,
                               const dynet::Expression& expr6);
};

template <class RNNBuilderType>
struct SegRNN : public LayerI {
  // uni-directional segment embedding.
  dynet::Parameter p_h0;
  dynet::Expression h0;
  RNNBuilderType builder;
  std::vector<std::vector<dynet::Expression>> h;
  unsigned input_dim;
  unsigned output_dim;
  unsigned len;
  unsigned max_seg_len;
  bool has_guard;

  // Single directional Segment RNN
  explicit SegRNN(dynet::ParameterCollection& m,
                  unsigned n_layers,
                  unsigned input_dim,
                  unsigned output_dim,
                  unsigned max_seg_len,
                  bool has_guard = false,
                  bool trainable = true) :
    LayerI(trainable),
    p_h0(m.add_parameters({ input_dim }, dynet::ParameterInitConst(0.f))),
    builder(n_layers, input_dim, output_dim, m),
    input_dim(input_dim),
    output_dim(output_dim),
    max_seg_len(max_seg_len),
    has_guard(has_guard) {
    assert(max_seg_len > 0);
  }

  void new_graph(dynet::ComputationGraph & cg) override {
    builder.new_graph(cg);
    if (has_guard) {
      h0 = trainable ? dynet::parameter(cg, p_h0) : dynet::const_parameter(cg, p_h0);
    }
  }
  
  std::vector<dynet::Expression> get_params() override {
    std::vector<dynet::Expression> ret;
    for (auto & layer : builder.param_vars) {
      for (auto & e : layer) { ret.push_back(e); } 
    }
    if (has_guard) { ret.push_back(h0); }
    return ret;
  }
  
  void construct_chart(const std::vector<dynet::Expression>& c) {
    len = c.size();
    h.clear(); // The first dimension for h is the starting point, the second is length.
    h.resize(len);
    for (unsigned i = 0; i < len; ++i) {
      unsigned max_j = i + max_seg_len;
      if (max_j > len) { max_j = len; }
      unsigned seg_len = max_j - i;
      auto& hi = h[i];
      hi.resize(seg_len);

      builder.start_new_sequence();
      if (has_guard) { builder.add_input(h0); }
      // Put one span in h[i][j]
      for (unsigned k = 0; k < seg_len; ++k) {
        hi[k] = builder.add_input(c[i + k]);
      }
    }
  }

  const dynet::Expression& operator()(unsigned i, unsigned j) const {
    assert(j <= len);
    assert(j >= i);
    return h[i][j - i];
  }

  void set_dropout(float& rate) {
    builder.set_dropout(rate);
  }

  void disable_dropout() {
    builder.disable_dropout();
  }
};

template <class RNNBuilderType>
struct SegBiRNN : public LayerI {
  typedef std::pair<dynet::Expression, dynet::Expression> ExpressionPair;
  SegRNN<RNNBuilderType> fwd;
  SegRNN<RNNBuilderType> bwd;
  std::vector<std::vector<ExpressionPair>> h;
  unsigned len;
  unsigned input_dim;
  unsigned output_dim;
  unsigned max_seg_len;

  explicit SegBiRNN(dynet::ParameterCollection& m,
                    unsigned n_layers,
                    unsigned input_dim,
                    unsigned output_dim,
                    unsigned max_seg_len,
                    bool has_guard = false,
                    bool trainable = true) :
    LayerI(trainable),
    fwd(m, n_layers, input_dim, output_dim, max_seg_len, has_guard),
    bwd(m, n_layers, input_dim, output_dim, max_seg_len, has_guard),
    input_dim(input_dim),
    output_dim(output_dim),
    max_seg_len(max_seg_len) {
  }

  void new_graph(dynet::ComputationGraph & cg) override {
    fwd.new_graph(cg);
    bwd.new_graph(cg);
  }

  std::vector<dynet::Expression> get_params() override {
    std::vector<dynet::Expression> ret;
    for (auto & e : fwd.get_params()) { ret.push_back(e); }
    for (auto & e : bwd.get_params()) { ret.push_back(e); }
    return ret;
  }

  void construct_chart(const std::vector<dynet::Expression>& c) {
    len = c.size();
    fwd.construct_chart(c);

    std::vector<dynet::Expression> rc(len);
    for (unsigned i = 0; i < len; ++i) { rc[i] = c[len - i - 1]; }
    bwd.construct_chart(rc);

    h.clear();
    h.resize(len);
    for (unsigned i = 0; i < len; ++i) {
      unsigned max_j = i + max_seg_len;
      if (max_j > len) { max_j = len; }
      auto& hi = h[i];
      unsigned seg_len = max_j - i;
      hi.resize(seg_len);
      for (unsigned k = 0; k < seg_len; ++k) {
        unsigned j = i + k;
        const dynet::Expression& fe = fwd(i, j);
        const dynet::Expression& be = bwd(len - 1 - j, len - 1 - i);
        hi[k] = std::make_pair(fe, be);
      }
    }
  }

  const ExpressionPair& operator()(unsigned i, unsigned j) const {
    assert(j <= len);
    assert(j >= i);
    return h[i][j - i];
  }

  void set_dropout(float& rate) {
    fwd.set_dropout(rate);
    bwd.set_dropout(rate);
  }

  void disable_dropout() {
    fwd.disable_dropout();
    bwd.disable_dropout();
  }
};

struct SegConcate : public LayerI {
  DenseLayer dense;
  dynet::Expression z;
  std::vector<std::vector<dynet::Expression>> h;
  unsigned len;
  unsigned input_dim;
  unsigned output_dim;
  unsigned max_seg_len;

  explicit SegConcate(dynet::ParameterCollection & m,
                      unsigned input_dim,
                      unsigned output_dim,
                      unsigned max_seg_len,
                      bool trainable=true);

  void new_graph(dynet::ComputationGraph & cg) override;

  std::vector<dynet::Expression> get_params() override;

  void construct_chart(const std::vector<dynet::Expression>& c);

  const dynet::Expression& operator()(unsigned i, unsigned j) const;
};

struct SegConv : public LayerI {
  Conv1dLayer conv;
  std::vector<std::vector<dynet::Expression>> h;
  unsigned len;
  unsigned input_dim;
  unsigned max_seg_len;

  SegConv(dynet::ParameterCollection & m,
          unsigned input_dim,
          unsigned max_seg_len,
          const std::vector<std::pair<unsigned, unsigned>>& filter_info,
          bool trainable=true);

  void new_graph(dynet::ComputationGraph & cg) override;

  std::vector<dynet::Expression> get_params() override;

  void construct_chart(const std::vector<dynet::Expression> & c);

  const dynet::Expression & operator()(unsigned i, unsigned j) const;
};

struct SegDiff : public LayerI {
  std::vector<std::vector<dynet::Expression>> h;
  unsigned len;
  unsigned input_dim;
  unsigned output_dim;
  unsigned max_seg_len;

  // use the diff of first repr with the last repr to represent
  // the segmentation. The dimension of input equals to that of
  // the output.
  SegDiff(dynet::ParameterCollection & m,
          unsigned input_dim,
          unsigned max_seg_len,
          bool trainable = true);

  void new_graph(dynet::ComputationGraph & cg) override;

  std::vector<dynet::Expression> get_params() override;

  void construct_chart(const std::vector<dynet::Expression> & c);
  
  const dynet::Expression & operator()(unsigned i, unsigned j) const;
};

#endif  //  end for LAYER_H
