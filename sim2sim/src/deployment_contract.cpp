#include "aar/deployment_contract.hpp"

#include <fstream>
#include <stdexcept>

#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

namespace aar {
namespace {

std::filesystem::path relative_path(const std::string& value, const std::string& field) {
  std::filesystem::path path(value);
  if (path.is_absolute()) throw std::runtime_error(field + " must be release-relative");
  for (const auto& component : path) {
    if (component == "..") throw std::runtime_error(field + " escapes release directory");
  }
  return path;
}

std::vector<TensorContract> tensors(const boost::property_tree::ptree& root,
                                    const std::string& path) {
  std::vector<TensorContract> result;
  for (const auto& item : root.get_child(path)) {
    TensorContract tensor;
    tensor.name = item.second.get<std::string>("name");
    tensor.dtype = item.second.get<std::string>("dtype");
    if (tensor.dtype != "float32") {
      throw std::runtime_error("only float32 tensors are supported: " + tensor.name);
    }
    for (const auto& dim : item.second.get_child("shape")) {
      tensor.shape.push_back(dim.second.get_value<std::int64_t>());
    }
    if (tensor.name.empty() || tensor.shape.empty()) {
      throw std::runtime_error("tensor name and shape are required");
    }
    result.push_back(std::move(tensor));
  }
  if (result.empty()) throw std::runtime_error(path + " must not be empty");
  return result;
}

std::vector<std::string> strings(const boost::property_tree::ptree& root,
                                 const std::string& path) {
  std::vector<std::string> result;
  for (const auto& item : root.get_child(path)) result.push_back(item.second.get_value<std::string>());
  return result;
}

}  // namespace

DeploymentContract load_contract(const std::filesystem::path& manifest_path) {
  boost::property_tree::ptree document;
  try {
    boost::property_tree::read_json(manifest_path.string(), document);
  } catch (const std::exception& error) {
    throw std::runtime_error("cannot parse manifest " + manifest_path.string() + ": " + error.what());
  }
  const std::string schema = document.get<std::string>("schema");
  if (schema != "aar-unilab.actor.v1") {
    throw std::runtime_error("unsupported deployment contract: " + schema);
  }
  return DeploymentContract{
      schema,
      document.get<std::string>("robot.id"),
      document.get<std::string>("task.id"),
      relative_path(document.get<std::string>("artifacts.policy_path"), "policy_path"),
      relative_path(document.get<std::string>("artifacts.runtime_config_path"),
                    "runtime_config_path"),
      relative_path(document.get<std::string>("artifacts.scene_path"), "scene_path"),
      tensors(document, "policy.inputs"),
      tensors(document, "policy.outputs"),
      document.get<std::string>("policy.observation_builder", "golden_inputs"),
      strings(document, "robot.joint_order"),
      document.get<double>("control.physics_hz"),
      document.get<double>("control.motor_hz"),
      document.get<double>("control.policy_hz"),
      document.get<double>("control.action_scale"),
      document.get<double>("control.action_clip"),
  };
}

}  // namespace aar
