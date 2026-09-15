#include <mujoco/mujoco.h>

#include <array>
#include <iostream>
#include <memory>

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: aar_mujoco_smoke scene.xml\n";
    return 2;
  }
  std::array<char, 2048> error{};
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
      mj_loadXML(argv[1], nullptr, error.data(), error.size()), mj_deleteModel);
  if (!model) {
    std::cerr << error.data() << '\n';
    return 1;
  }
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data(mj_makeData(model.get()), mj_deleteData);
  if (!data) return 1;
  mj_step(model.get(), data.get());
  std::cout << "MuJoCo " << mj_versionString() << " nq=" << model->nq << " nv=" << model->nv
            << " nu=" << model->nu << " time=" << data->time << '\n';
  return std::string(mj_versionString()) == "3.8.0" ? 0 : 1;
}
