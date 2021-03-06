#include "opt_builder.h"
#include <iostream>
#include <fstream>
#include <sstream>


po::options_description OptimizerBuilder::get_options() {
  po::options_description cmd("Optimizer options");
  cmd.add_options()
    ("optimizer", po::value<std::string>()->default_value("simple_sgd"), 
     "The choice of optimizer [simple_sgd, momentum_sgd, rmsprop, adagrad, adadelta, adam].")
    ("optimizer-eta", po::value<float>(), "The initial value of learning rate (eta).")
    ("optimizer-mom", po::value<float>()->default_value(0.9f), "the momentum used in Momentum trainer.")
    ("optimizer-adam-beta1", po::value<float>()->default_value(0.9f), "The beta1 hyper-parameter of adam")
    ("optimizer-adam-beta2", po::value<float>()->default_value(0.999f), "The beta2 hyper-parameter of adam.")
    ("optimizer-enable-clipping", po::value<bool>()->default_value(false), "enable clipping.")
    ;

  return cmd;
}

std::string OptimizerBuilder::get_optimizer_signature(const po::variables_map & conf) {
  std::ostringstream os;
  if (conf["optimizer"].as<std::string>() == "simple_sgd") {
    os << "sgd";
  } else if (conf["optimizer"].as<std::string>() == "momentum_sgd") {
    os << "momsgd";
  } else {
    os << conf["optimizer"].as<std::string>();
  }
  if (conf["optimizer-enable-clipping"].as<bool>()) {
    os << "_clip";
  } else {
    os << "_noclip";
  }
  return os.str();
}

OptimizerBuilder::OptimizerBuilder(const po::variables_map & conf) : enable_clipping(false) {
  eta0 = 0.1f;
  if (conf.count("optimizer-eta0")) {
    eta0 = conf["optimizer-eta0"].as<float>();
  }

  if (!conf.count("optimizer") || conf["optimizer"].as<std::string>() == "simple_sgd") {
    optimizer_type = kSimpleSGD;
  } else if (conf["optimizer"].as<std::string>() == "momentum_sgd") {
    optimizer_type = kMomentumSGD;
  } else if (conf["optimizer"].as<std::string>() == "adagrad") {
    optimizer_type = kAdaGrad;
  } else if (conf["optimizer"].as<std::string>() == "adadelta") {
    optimizer_type = kAdaDelta;
  } else if (conf["optimizer"].as<std::string>() == "rmsprop") {
    optimizer_type = kRMSProp;
  } else if (conf["optimizer"].as<std::string>() == "adam") {
    optimizer_type = kAdam;
    eta0 = 0.001f;
    adam_beta1 = conf["optimizer-adam-beta1"].as<float>();
    adam_beta2 = conf["optimizer-adam-beta2"].as<float>();
  } else {
    std::cerr << "dynet_layer|optimizer_builder unknown optimizer: " << conf["optimizer"].as<std::string>();
    exit(1);
  }

  enable_clipping = conf["optimizer-enable-clipping"].as<bool>();
}

dynet::Trainer* OptimizerBuilder::build(dynet::ParameterCollection & model) {
  dynet::Trainer* ret = nullptr;
  if (optimizer_type == kSimpleSGD) {
    ret = new dynet::SimpleSGDTrainer(model, eta0);
  } else if (optimizer_type == kMomentumSGD) {
    ret = new dynet::MomentumSGDTrainer(model, eta0);
  } else if (optimizer_type == kAdaGrad) {
    ret = new dynet::AdagradTrainer(model, eta0);
  } else if (optimizer_type == kAdaDelta) {
    ret = new dynet::AdadeltaTrainer(model, eta0);
  } else if (optimizer_type == kAdam) {
    ret = new dynet::AdamTrainer(model, eta0, adam_beta1, adam_beta2);
  } else {
    std::cerr << "dynet_layer|optimizer_builder internal error." << std::endl;
    exit(1);
  }

  ret->clipping_enabled = enable_clipping;
  return ret;
}

void OptimizerBuilder::update(dynet::Trainer *trainer, unsigned iter) {
  trainer->learning_rate = eta0 / (1.f + static_cast<float>(iter) * .08f);
}
