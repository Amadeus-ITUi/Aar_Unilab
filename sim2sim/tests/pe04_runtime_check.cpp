// Exercise PE04 observation/history/control independently of ONNX inference.
#include "aar/deployment_contract.hpp"
#include "aar/pe04_runtime.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const std::filesystem::path release(argv[1]);
  auto contract = aar::load_contract(release / "deployment_manifest.json");
  char error[1024]{};
  auto* model = mj_loadXML((release / contract.scene_path).c_str(), nullptr, error, sizeof(error));
  if (!model) throw std::runtime_error(error);
  auto* data = mj_makeData(model);
  mj_resetDataKeyframe(model, data, mj_name2id(model, mjOBJ_KEY, contract.reset_keyframe.c_str()));
  mj_forward(model, data);
  aar::PE04Runtime runtime((release / contract.scene_path).parent_path() / "pe04_runtime.json", model, contract);
  std::vector<std::vector<float>> inputs{{}, {}, {}};
  for (std::size_t i=0; i<contract.inputs.size(); ++i) {
    std::size_t size=1;
    for (auto dimension : contract.inputs[i].shape) size*=dimension;
    inputs[i].resize(size);
  }
  std::vector<float> history(300);
  std::ofstream out(argv[2]);
  out << std::setprecision(17);
  for (int step=0; step<60; ++step) {
    runtime.update_inputs(data, contract, inputs, history);
    for (const auto& name : {"observation_history", "observation"}) {
      for (std::size_t i=0; i<contract.inputs.size(); ++i)
        if (contract.inputs[i].name == name) for (float value : inputs[i]) out << value << ' ';
    }
    std::vector<float> action(6);
    for (int i=0;i<6;++i) action[i]= step==3 ? 1000.f : static_cast<float>(.2*std::sin(step+i));
    runtime.prepare_action(data, action, contract);
    for (int substep=0;substep<8;++substep) {
      runtime.before_step(model,data);
      mj_step(model,data);
      runtime.after_step(data);
    }
    runtime.end_policy_step();
    for (int i=0;i<6;++i) out << data->ctrl[i] << ' ';
    out << '\n';
  }
  mj_deleteData(data); mj_deleteModel(model);
}
