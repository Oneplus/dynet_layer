#include "layer.h"
#include "dynet/expr.h"
#include "dynet/param-init.h"

SymbolEmbedding::SymbolEmbedding(dynet::ParameterCollection& m,
                                 unsigned n,
                                 unsigned dim,
                                 bool trainable) :
  LayerI(trainable),
  p_e(m.add_lookup_parameters(n, { dim, 1 })) {
}

void SymbolEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

std::vector<dynet::Expression> SymbolEmbedding::get_params() {
  std::vector<dynet::Expression> ret;
  return ret;
}

dynet::Expression SymbolEmbedding::embed(unsigned index) {
  return (trainable ?
          dynet::lookup((*cg), p_e, index) :
          dynet::const_lookup((*cg), p_e, index));
}

BinnedDistanceEmbedding::BinnedDistanceEmbedding(dynet::ParameterCollection& m,
                                                 unsigned dim,
                                                 unsigned n_bins,
                                                 bool trainable) :
  LayerI(trainable),
  p_e(m.add_lookup_parameters(n_bins * 2, { dim, 1 })),
  max_bin(n_bins - 1) {
}

void BinnedDistanceEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

dynet::Expression BinnedDistanceEmbedding::embed(int dist) {
  unsigned base = (dist < 0 ? max_bin : 0);
  unsigned dist_std = 0;
  if (dist) {
    dist_std = static_cast<unsigned>(log(dist < 0 ? -dist : dist) / log(1.6f)) + 1;
  }
  if (dist_std > max_bin) {
    dist_std = max_bin;
  }
  return (trainable ?
          dynet::lookup(*cg, p_e, dist_std + base) :
          dynet::const_lookup(*cg, p_e, dist_std + base));
}

BinnedDurationEmbedding::BinnedDurationEmbedding(dynet::ParameterCollection& m,
                                                 unsigned dim,
                                                 unsigned n_bins,
                                                 bool trainable) :
  LayerI(trainable),
  p_e(m.add_lookup_parameters(n_bins, { dim, 1 })),
  max_bin(n_bins - 1) {
}

void BinnedDurationEmbedding::new_graph(dynet::ComputationGraph& cg_) {
  cg = &cg_;
}

std::vector<dynet::Expression> BinnedDistanceEmbedding::get_params() {
  std::vector<dynet::Expression> ret;
  return ret;
}

dynet::Expression BinnedDurationEmbedding::embed(unsigned dur) {
  if (dur) {
    dur = static_cast<unsigned>(log(dur) / log(1.6f)) + 1;
  }
  if (dur > max_bin) {
    dur = max_bin;
  }
  return (trainable ?
          dynet::lookup((*cg), p_e, dur) :
          dynet::const_lookup((*cg), p_e, dur));
}

std::vector<dynet::Expression> BinnedDurationEmbedding::get_params() {
  std::vector<dynet::Expression> ret;
  return ret;
}

SoftmaxLayer::SoftmaxLayer(dynet::ParameterCollection& m,
                           unsigned dim_input,
                           unsigned dim_output,
                           bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W(m.add_parameters({ dim_output, dim_input })) {
}

void SoftmaxLayer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W = dynet::parameter(hg, p_W);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W = dynet::const_parameter(hg, p_W);
  }
}

std::vector<dynet::Expression> SoftmaxLayer::get_params() {
  std::vector<dynet::Expression> ret = { B, W };
  return ret;
}

dynet::Expression SoftmaxLayer::get_output(const dynet::Expression& expr) {
  return dynet::log_softmax(dynet::affine_transform({ B, W, expr }));
}

DenseLayer::DenseLayer(dynet::ParameterCollection& m,
                       unsigned dim_input,
                       unsigned dim_output,
                       bool trainable) :
  LayerI(trainable),
  p_W(m.add_parameters({ dim_output, dim_input })),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))) {
}

void DenseLayer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    W = dynet::parameter(hg, p_W);
    B = dynet::parameter(hg, p_B);
  } else {
    W = dynet::const_parameter(hg, p_W);
    B = dynet::const_parameter(hg, p_B);
  }
}

std::vector<dynet::Expression> DenseLayer::get_params() {
  std::vector<dynet::Expression> ret = { B, W };
  return ret;
}

dynet::Expression DenseLayer::get_output(const dynet::Expression& expr) {
  return dynet::affine_transform({ B, W, expr });
}

Merge2Layer::Merge2Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })) {
}

void Merge2Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
  }
}

std::vector<dynet::Expression> Merge2Layer::get_params() {
  std::vector<dynet::Expression> ret = { B, W1, W2 };
  return ret;
}

dynet::Expression Merge2Layer::get_output(const dynet::Expression& expr1,
                                          const dynet::Expression& expr2) {
  return dynet::affine_transform({ B, W1, expr1, W2, expr2 });
}

Merge3Layer::Merge3Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })) {
}

void Merge3Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
  }
}

std::vector<dynet::Expression> Merge3Layer::get_params() {
  std::vector<dynet::Expression> ret = { B, W1, W2, W3 };
  return ret;
}

dynet::Expression Merge3Layer::get_output(const dynet::Expression& expr1,
                                          const dynet::Expression& expr2,
                                          const dynet::Expression& expr3) {
  return dynet::affine_transform({ B, W1, expr1, W2, expr2, W3, expr3 });
}

Merge4Layer::Merge4Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })) {
}

void Merge4Layer::new_graph(dynet::ComputationGraph& hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
    W4 = dynet::parameter(hg, p_W4);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
    W4 = dynet::const_parameter(hg, p_W4);
  }
}

std::vector<dynet::Expression> Merge4Layer::get_params() {
  std::vector<dynet::Expression> ret = { B, W1, W2, W3, W4 };
  return ret;
}

dynet::Expression Merge4Layer::get_output(const dynet::Expression& expr1,
                                          const dynet::Expression& expr2,
                                          const dynet::Expression& expr3,
                                          const dynet::Expression& expr4) {
  return dynet::affine_transform({ B, W1, expr1, W2, expr2, W3, expr3, W4, expr4 });
}

Merge5Layer::Merge5Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })),
  p_W5(m.add_parameters({ dim_output, dim_input5 })) {
}

void Merge5Layer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
    W4 = dynet::parameter(hg, p_W4);
    W5 = dynet::parameter(hg, p_W5);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
    W4 = dynet::const_parameter(hg, p_W4);
    W5 = dynet::const_parameter(hg, p_W5);
  }
}

std::vector<dynet::Expression> Merge5Layer::get_params() {
  std::vector<dynet::Expression> ret = { B, W1, W2, W3, W4, W5 };
  return ret;
}
dynet::Expression Merge5Layer::get_output(const dynet::Expression& expr1,
                                          const dynet::Expression& expr2,
                                          const dynet::Expression& expr3,
                                          const dynet::Expression& expr4,
                                          const dynet::Expression& expr5) {
  return dynet::affine_transform({
    B, W1, expr1, W2, expr2, W3, expr3, W4, expr4, W5, expr5
  });
}

Merge6Layer::Merge6Layer(dynet::ParameterCollection& m,
                         unsigned dim_input1,
                         unsigned dim_input2,
                         unsigned dim_input3,
                         unsigned dim_input4,
                         unsigned dim_input5,
                         unsigned dim_input6,
                         unsigned dim_output,
                         bool trainable) :
  LayerI(trainable),
  p_B(m.add_parameters({ dim_output, 1 }, dynet::ParameterInitConst(0.f))),
  p_W1(m.add_parameters({ dim_output, dim_input1 })),
  p_W2(m.add_parameters({ dim_output, dim_input2 })),
  p_W3(m.add_parameters({ dim_output, dim_input3 })),
  p_W4(m.add_parameters({ dim_output, dim_input4 })),
  p_W5(m.add_parameters({ dim_output, dim_input5 })),
  p_W6(m.add_parameters({ dim_output, dim_input6 })) {
}

void Merge6Layer::new_graph(dynet::ComputationGraph & hg) {
  if (trainable) {
    B = dynet::parameter(hg, p_B);
    W1 = dynet::parameter(hg, p_W1);
    W2 = dynet::parameter(hg, p_W2);
    W3 = dynet::parameter(hg, p_W3);
    W4 = dynet::parameter(hg, p_W4);
    W5 = dynet::parameter(hg, p_W5);
    W6 = dynet::parameter(hg, p_W6);
  } else {
    B = dynet::const_parameter(hg, p_B);
    W1 = dynet::const_parameter(hg, p_W1);
    W2 = dynet::const_parameter(hg, p_W2);
    W3 = dynet::const_parameter(hg, p_W3);
    W4 = dynet::const_parameter(hg, p_W4);
    W5 = dynet::const_parameter(hg, p_W5);
    W6 = dynet::const_parameter(hg, p_W6);
  }
}

std::vector<dynet::Expression> Merge6Layer::get_params() {
  std::vector<dynet::Expression> ret = { B, W1, W2, W3, W4, W5, W6 };
  return ret;
}

dynet::Expression Merge6Layer::get_output(
  const dynet::Expression& expr1,
  const dynet::Expression& expr2,
  const dynet::Expression& expr3,
  const dynet::Expression& expr4,
  const dynet::Expression& expr5,
  const dynet::Expression& expr6) {
  return dynet::affine_transform({
    B, W1, expr1, W2, expr2, W3, expr3, W4, expr4, W5, expr5, W6, expr6
  });
}

Conv1dLayer::Conv1dLayer(dynet::ParameterCollection & m,
                         unsigned dim,
                         const std::vector<std::pair<unsigned, unsigned>>& filters_info,
                         ACTIVATION_TYPE activation,
                         bool has_bias,
                         bool trainable) :
  LayerI(trainable),
  filters_info(filters_info),
  activation(activation),
  dim(dim),
  has_bias(has_bias) {
  unsigned n_filter_types = filters_info.size();
  unsigned combined_dim = 0;
  p_filters.resize(n_filter_types);
  if (has_bias) {
    p_biases.resize(n_filter_types);
  }
  for (unsigned i = 0; i < n_filter_types; ++i) {
    const auto& filter_width = filters_info[i].first;
    const auto& nb_filters = filters_info[i].second;
    p_filters[i] = m.add_parameters({ dim, filter_width, nb_filters });
    if (has_bias) {
      p_biases[i] = m.add_parameters({ dim, nb_filters }, dynet::ParameterInitConst(0.f));
    }
  }
}

void Conv1dLayer::new_graph(dynet::ComputationGraph & hg) {
  unsigned n_filter_types = filters_info.size();
  filters.resize(n_filter_types);
  if (has_bias) {
    biases.resize(n_filter_types);
  }
  for (unsigned i = 0; i < n_filter_types; ++i) {
    filters[i] = (trainable ?
                  dynet::parameter(hg, p_filters[i]) :
                  dynet::const_parameter(hg, p_filters[i]));

    if (has_bias) {
      biases[i] = (trainable ?
                   dynet::parameter(hg, p_biases[i]) :
                   dynet::const_parameter(hg, p_biases[i]));
    }
  }
  padding = dynet::zeroes(hg, { dim });
}

std::vector<dynet::Expression> Conv1dLayer::get_params() {
  std::vector<dynet::Expression> ret;
  for (auto & e : filters) { ret.push_back(e); }
  if (has_bias) {
    for (auto & e : biases) { ret.push_back(e); }
  }
  return ret;
}

dynet::Expression Conv1dLayer::get_output(const std::vector<dynet::Expression>& exprs) {
  std::vector<dynet::Expression> tmp;
  unsigned n_filter_types = filters_info.size();
  for (unsigned ii = 0; ii < n_filter_types; ++ii) {
    const auto& filter_width = filters_info[ii].first;
    unsigned n_cols = exprs.size() + (filter_width - 1) * 2;
    std::vector<dynet::Expression> s(n_cols);
    for (unsigned p = 0; p < filter_width - 1; ++p) {
      s[p] = padding;
      s[n_cols - 1 - p] = padding;
    }
    for (unsigned i = 0; i < exprs.size(); ++i) {
      s[filter_width - 1 + i] = exprs[i];
    }
     
    auto& filter = filters[ii];
    auto t = dynet::filter1d_narrow(dynet::concatenate_cols(s), filter);
    if (activation == kRelu) {
      t = dynet::rectify(t);
    } else if (activation == kTanh) {
      t = dynet::tanh(t);
    }
    if (has_bias) {
      auto& bias = biases[ii];
      t = dynet::colwise_add(t, bias);
    }
    tmp.push_back(dynet::kmax_pooling(t, 1));
  }
  return dynet::concatenate(tmp);
}

BiLinearLayer::BiLinearLayer(dynet::ParameterCollection & model,
                             unsigned dim,
                             bool trainable) :
  LayerI(trainable),
  p_W(model.add_parameters({dim}))
{
}

void BiLinearLayer::new_graph(dynet::ComputationGraph & cg) {
  W = dynet::parameter(cg, p_W);
}

std::vector<dynet::Expression> BiLinearLayer::get_params() {
  std::vector<dynet::Expression> ret = { W };
  return ret;
}

dynet::Expression BiLinearLayer::get_output(const dynet::Expression & expr1,
                                            const dynet::Expression & expr2) {
  dynet::Expression ret = dynet::transpose(expr1) * W * expr2;
  return ret;
}

BiAffineLayer::BiAffineLayer(dynet::ParameterCollection & model,
                             unsigned dim, 
                             bool trainable) :
  LayerI(trainable),
  p_W(model.add_parameters({dim}))
{
}

void BiAffineLayer::new_graph(dynet::ComputationGraph & cg) {
  W = dynet::parameter(cg, p_W);
}

std::vector<dynet::Expression> BiAffineLayer::get_params() {
  std::vector<dynet::Expression> ret = { W };
  return ret;
}

dynet::Expression BiAffineLayer::get_output(const dynet::Expression & expr1, 
                                            const dynet::Expression & expr2) {
  return dynet::dot_product(dynet::cmult(expr1, W), expr2);
}


SegConcateEmbedding::SegConcateEmbedding(dynet::ParameterCollection& m,
                                         unsigned input_dim, 
                                         unsigned output_dim,
                                         unsigned max_seg_len,
                                         bool trainable) :
  LayerI(trainable),
  dense(m, input_dim * max_seg_len, output_dim),
  input_dim(input_dim),
  max_seg_len(max_seg_len) {
}

void SegConcateEmbedding::new_graph(dynet::ComputationGraph & cg) {
  dense.new_graph(cg);
  z = dynet::zeroes(cg, { input_dim });
}

std::vector<dynet::Expression> SegConcateEmbedding::get_params() {
  return dense.get_params();
}

void SegConcateEmbedding::construct_chart(const std::vector<dynet::Expression>& c,
                                          int max_seg_len) {
  len = c.size();
  h.clear(); // The first dimension for h is the starting point, the second is length.
  h.resize(len);

  for (unsigned i = 0; i < len; ++i) {
    unsigned max_j = i + len;
    if (max_seg_len) { max_j = i + max_seg_len; }
    if (max_j > len) { max_j = len; }
    unsigned seg_len = max_j - i;
    auto& hi = h[i];
    hi.resize(seg_len);

    std::vector<dynet::Expression> s(max_seg_len, z);
    for (unsigned k = 0; k < seg_len; ++k) {
      s[k] = c[i + k];
      hi[k] = dynet::rectify(dense.get_output(dynet::concatenate(s)));
    }
  }
}

const dynet::Expression& SegConcateEmbedding::operator()(unsigned i, unsigned j) const {
  return h[i][j - i];
}
