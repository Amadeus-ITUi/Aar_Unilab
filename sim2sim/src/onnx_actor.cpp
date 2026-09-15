#include "aar/onnx_actor.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace aar {
namespace {

std::size_t tensor_size(const TensorContract& tensor) {
  const auto result = std::accumulate(tensor.shape.begin(), tensor.shape.end(), std::int64_t{1},
                                      std::multiplies<>());
  if (result <= 0) throw std::runtime_error("dynamic or empty runtime tensor: " + tensor.name);
  return static_cast<std::size_t>(result);
}

}  // namespace

struct OnnxActor::Impl {
  explicit Impl(const std::filesystem::path& policy_path, const DeploymentContract& selected)
      : environment(ORT_LOGGING_LEVEL_WARNING, "aar_sim2sim"),
        session_options(),
        session(environment, policy_path.c_str(), session_options),
        contract(selected) {
    if (session.GetInputCount() != contract.inputs.size() ||
        session.GetOutputCount() != contract.outputs.size()) {
      throw std::runtime_error("ONNX tensor count does not match deployment manifest");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    for (std::size_t index = 0; index < contract.inputs.size(); ++index) {
      const auto name = session.GetInputNameAllocated(index, allocator);
      if (std::string(name.get()) != contract.inputs[index].name) {
        throw std::runtime_error("ONNX input order/name differs from manifest");
      }
      input_names.push_back(contract.inputs[index].name.c_str());
    }
    for (std::size_t index = 0; index < contract.outputs.size(); ++index) {
      const auto name = session.GetOutputNameAllocated(index, allocator);
      if (std::string(name.get()) != contract.outputs[index].name) {
        throw std::runtime_error("ONNX output order/name differs from manifest");
      }
      output_names.push_back(contract.outputs[index].name.c_str());
    }
  }

  Ort::Env environment;
  Ort::SessionOptions session_options;
  Ort::Session session;
  DeploymentContract contract;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;
};

OnnxActor::OnnxActor(const std::filesystem::path& policy_path,
                     const DeploymentContract& contract)
    : impl_(std::make_unique<Impl>(policy_path, contract)) {}
OnnxActor::~OnnxActor() = default;
OnnxActor::OnnxActor(OnnxActor&&) noexcept = default;
OnnxActor& OnnxActor::operator=(OnnxActor&&) noexcept = default;

std::vector<std::vector<float>> OnnxActor::run(
    const std::vector<std::vector<float>>& inputs) {
  if (inputs.size() != impl_->contract.inputs.size()) {
    throw std::runtime_error("runtime input count differs from manifest");
  }
  const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<Ort::Value> input_values;
  input_values.reserve(inputs.size());
  for (std::size_t index = 0; index < inputs.size(); ++index) {
    const auto& spec = impl_->contract.inputs[index];
    if (inputs[index].size() != tensor_size(spec)) {
      throw std::runtime_error("runtime input size differs for " + spec.name);
    }
    input_values.push_back(Ort::Value::CreateTensor<float>(
        memory, const_cast<float*>(inputs[index].data()), inputs[index].size(), spec.shape.data(),
        spec.shape.size()));
  }
  auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, impl_->input_names.data(),
                                    input_values.data(), input_values.size(),
                                    impl_->output_names.data(), impl_->output_names.size());
  std::vector<std::vector<float>> result;
  result.reserve(outputs.size());
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const std::size_t count = tensor_size(impl_->contract.outputs[index]);
    const float* values = outputs[index].GetTensorData<float>();
    result.emplace_back(values, values + count);
  }
  return result;
}

}  // namespace aar
