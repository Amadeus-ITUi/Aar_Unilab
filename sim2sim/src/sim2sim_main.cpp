#include "aar/deployment_contract.hpp"
#include "aar/npy.hpp"
#include "aar/onnx_actor.hpp"
#include "aar/telemetry.hpp"

#include <mujoco/mujoco.h>
#ifdef AAR_WITH_GLFW
#include <GLFW/glfw3.h>
#endif

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::filesystem::path release;
  std::filesystem::path telemetry;
  int steps{1};
  bool interactive{false};
};

Options parse_options(int argc, char** argv) {
  if (argc < 2) {
    throw std::runtime_error("usage: aar_sim2sim RELEASE [--steps N] [--telemetry DIR]");
  }
  Options options;
  options.release = argv[1];
  options.telemetry = options.release / "telemetry";
  for (int index = 2; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--steps" && index + 1 < argc) {
      options.steps = std::stoi(argv[++index]);
    } else if (argument == "--telemetry" && index + 1 < argc) {
      options.telemetry = argv[++index];
    } else if (argument == "--interactive") {
      options.interactive = true;
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + argument);
    }
  }
  if (options.steps <= 0) throw std::runtime_error("--steps must be positive");
  return options;
}

#ifdef AAR_WITH_GLFW
struct Viewer {
  GLFWwindow* window{nullptr};
  mjModel* model{nullptr};
  mjData* data{nullptr};
  mjvCamera camera{};
  mjvOption option{};
  mjvScene scene{};
  mjrContext context{};
  mjvFigure figure{};
  bool left{false};
  bool middle{false};
  bool right{false};
  double last_x{0.0};
  double last_y{0.0};
  int force_body{0};

  Viewer(mjModel* selected_model, mjData* selected_data)
      : model(selected_model), data(selected_data) {
    if (!glfwInit()) throw std::runtime_error("GLFW initialization failed");
    window = glfwCreateWindow(1280, 800, "Aar_Unilab C++ Sim2Sim", nullptr, nullptr);
    if (!window) {
      glfwTerminate();
      throw std::runtime_error("GLFW window creation failed");
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetScrollCallback(window, scroll);
    mjv_defaultCamera(&camera);
    mjv_defaultOption(&option);
    mjv_defaultScene(&scene);
    mjr_defaultContext(&context);
    mjv_makeScene(model, &scene, 3000);
    mjr_makeContext(model, &context, mjFONTSCALE_150);
    camera.distance = 1.4;
    camera.azimuth = 135.0;
    camera.elevation = -20.0;
    force_body = model->njnt > 0 ? model->jnt_bodyid[0] : 0;
    mjv_defaultFigure(&figure);
    figure.flg_legend = 1;
    figure.flg_extend = 0;
    figure.range[0][0] = -199.0F;
    figure.range[0][1] = 0.0F;
    figure.range[1][0] = -1.0F;
    figure.range[1][1] = 1.0F;
    mju_strncpy(figure.title, "actor action[0]", sizeof(figure.title));
    mju_strncpy(figure.linename[0], "action_0", sizeof(figure.linename[0]));
    figure.linepnt[0] = 200;
    for (int point = 0; point < 200; ++point) {
      figure.linedata[0][2 * point] = static_cast<float>(point - 199);
      figure.linedata[0][2 * point + 1] = 0.0F;
    }
  }

  ~Viewer() {
    mjr_freeContext(&context);
    mjv_freeScene(&scene);
    if (window) glfwDestroyWindow(window);
    glfwTerminate();
  }

  static Viewer* self(GLFWwindow* window) {
    return static_cast<Viewer*>(glfwGetWindowUserPointer(window));
  }

  static void mouse_button(GLFWwindow* window, int button, int action, int) {
    auto* viewer = self(window);
    viewer->left = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    viewer->middle = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    viewer->right = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    glfwGetCursorPos(window, &viewer->last_x, &viewer->last_y);
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE && viewer->force_body > 0) {
      mju_zero(viewer->data->xfrc_applied + 6 * viewer->force_body, 6);
    }
  }

  static void mouse_move(GLFWwindow* window, double x, double y) {
    auto* viewer = self(window);
    if (!viewer->left && !viewer->middle && !viewer->right) return;
    int width = 1;
    int height = 1;
    glfwGetWindowSize(window, &width, &height);
    const double dx = (x - viewer->last_x) / static_cast<double>(height);
    const double dy = (y - viewer->last_y) / static_cast<double>(height);
    viewer->last_x = x;
    viewer->last_y = y;
    if (viewer->right && viewer->force_body > 0) {
      auto* force = viewer->data->xfrc_applied + 6 * viewer->force_body;
      force[0] = -dy * 250.0;
      force[1] = dx * 250.0;
      return;
    }
    const bool shift = glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
                       glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;
    const mjtMouse action = viewer->middle
                                ? (shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V)
                                : (shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V);
    mjv_moveCamera(viewer->model, action, dx, dy, &viewer->scene, &viewer->camera);
  }

  static void scroll(GLFWwindow* window, double, double offset) {
    auto* viewer = self(window);
    mjv_moveCamera(viewer->model, mjMOUSE_ZOOM, 0.0, -0.05 * offset, &viewer->scene,
                   &viewer->camera);
  }

  void gamepad(const aar::DeploymentContract& contract,
               std::vector<std::vector<float>>& inputs,
               std::vector<float>& command) {
    if (!glfwJoystickPresent(GLFW_JOYSTICK_1)) return;
    int count = 0;
    const float* axes = glfwGetJoystickAxes(GLFW_JOYSTICK_1, &count);
    if (command.size() >= 3 && count >= 2) {
      command[0] = -axes[1];
      command[1] = axes[0];
      command[2] = count >= 4 ? axes[3] : 0.0F;
    }
    for (std::size_t index = 0; index < contract.inputs.size(); ++index) {
      if ((contract.inputs[index].name == "command" ||
           contract.inputs[index].name == "commands") &&
          inputs[index].size() >= 3 && count >= 2) {
        inputs[index][0] = -axes[1];
        inputs[index][1] = axes[0];
        inputs[index][2] = count >= 4 ? axes[3] : 0.0F;
      }
    }
  }

  void draw(float action) {
    if (force_body > 0) {
      camera.lookat[0] = data->xpos[3 * force_body];
      camera.lookat[1] = data->xpos[3 * force_body + 1];
      camera.lookat[2] = data->xpos[3 * force_body + 2];
    }
    for (int point = 0; point < 199; ++point) {
      figure.linedata[0][2 * point + 1] = figure.linedata[0][2 * (point + 1) + 1];
    }
    figure.linedata[0][399] = action;
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    const mjrRect viewport{0, 0, width, height};
    mjv_updateScene(model, data, &option, nullptr, &camera, mjCAT_ALL, &scene);
    mjr_render(viewport, &scene, &context);
    const mjrRect plot{0, 0, std::max(width / 3, 1), std::max(height / 3, 1)};
    mjr_figure(plot, &figure, &context);
    mjr_overlay(mjFONT_NORMAL, mjGRID_TOPLEFT, viewport,
                "LMB rotate | Shift+LMB pan | wheel zoom | RMB drag force",
                "Gamepad left stick: command | camera follows robot", &context);
    glfwSwapBuffers(window);
    glfwPollEvents();
  }
};
#endif

std::vector<std::vector<float>> initial_inputs(const std::filesystem::path& release,
                                                const aar::DeploymentContract& contract) {
  std::vector<std::vector<float>> inputs;
  for (const auto& tensor : contract.inputs) {
    const auto file = release / "golden_inputs" / (tensor.name + ".npy");
    auto loaded = aar::load_float32_npy(file);
    if (loaded.shape != tensor.shape) {
      throw std::runtime_error("golden input shape differs from manifest: " + tensor.name);
    }
    inputs.push_back(std::move(loaded.values));
  }
  return inputs;
}

std::size_t input_index(const aar::DeploymentContract& contract, const std::string& name);

int sensor_address(const mjModel* model, const std::string& name) {
  const int id = mj_name2id(model, mjOBJ_SENSOR, name.c_str());
  if (id < 0) throw std::runtime_error("missing MuJoCo sensor: " + name);
  return model->sensor_adr[id];
}

struct We11State {
  std::vector<std::vector<float>> term_history;
  std::vector<float> previous_action = std::vector<float>(6, 0.0F);
  std::vector<float> command = std::vector<float>(3, 0.0F);
};

void append_term_history(std::vector<float>& result, std::vector<float>& history,
                         const std::vector<float>& current) {
  if (history.empty()) history.assign(current.size() * 5, 0.0F);
  std::move(history.begin() + static_cast<std::ptrdiff_t>(current.size()), history.end(),
            history.begin());
  std::copy(current.begin(), current.end(), history.end() - current.size());
  result.insert(result.end(), history.begin(), history.end());
}

void update_we11_inputs(const mjModel* model, const mjData* data,
                        const aar::DeploymentContract& contract,
                        std::vector<std::vector<float>>& inputs, We11State& state) {
  const auto obs_index = input_index(contract, "obs");
  const int gyro_adr = sensor_address(model, "imu_gyro");
  const int x_adr = sensor_address(model, "xvector");
  const int y_adr = sensor_address(model, "yvector");
  const int z_adr = sensor_address(model, "upvector");
  const int left_wing_adr = sensor_address(model, "left_wing_pos");
  const int right_wing_adr = sensor_address(model, "right_wing_pos");
  const int left_wing_vel_adr = sensor_address(model, "left_wing_vel_sensor");
  const int right_wing_vel_adr = sensor_address(model, "right_wing_vel_sensor");
  const std::vector<float> gyro{
      static_cast<float>(data->sensordata[gyro_adr]),
      static_cast<float>(data->sensordata[gyro_adr + 1]),
      static_cast<float>(data->sensordata[gyro_adr + 2])};
  const std::vector<float> gravity{
      static_cast<float>(-data->sensordata[x_adr + 2]),
      static_cast<float>(-data->sensordata[y_adr + 2]),
      static_cast<float>(-data->sensordata[z_adr + 2])};
  constexpr double defaults[6] = {0.8, -1.6, 0.0, 0.8, -1.6, 0.0};
  constexpr int leg_indices[4] = {0, 1, 3, 4};
  constexpr int velocity_indices[6] = {0, 1, 3, 4, 2, 5};
  std::vector<float> leg_position;
  for (const int index : leg_indices) {
    const int joint = model->actuator_trnid[2 * index];
    leg_position.push_back(static_cast<float>(data->qpos[model->jnt_qposadr[joint]] - defaults[index]));
  }
  std::vector<float> velocity;
  for (const int index : velocity_indices) {
    const int joint = model->actuator_trnid[2 * index];
    velocity.push_back(static_cast<float>(0.1 * data->qvel[model->jnt_dofadr[joint]]));
  }
  const std::vector<float> wing_angle{
      static_cast<float>(data->sensordata[left_wing_adr] / mjPI),
      static_cast<float>(data->sensordata[right_wing_adr] / mjPI)};
  const std::vector<float> wing_velocity{
      static_cast<float>(0.1 * data->sensordata[left_wing_vel_adr]),
      static_cast<float>(0.1 * data->sensordata[right_wing_vel_adr])};
  std::vector<std::vector<float>> terms{gyro, gravity, leg_position, velocity,
                                        state.previous_action, wing_angle};
  terms.push_back(wing_velocity);
  terms.push_back(state.command);
  if (state.term_history.empty()) state.term_history.resize(terms.size());
  std::vector<float> observation;
  for (std::size_t index = 0; index < terms.size(); ++index) {
    append_term_history(observation, state.term_history[index], terms[index]);
  }
  if (observation.size() != inputs[obs_index].size()) {
    throw std::runtime_error("WE11 observation dimension mismatch: built " +
                             std::to_string(observation.size()));
  }
  inputs[obs_index] = std::move(observation);
}

void apply_we11_control(const mjModel* model, mjData* data,
                        const std::vector<float>& action, int substep) {
  constexpr double defaults[6] = {0.8, -1.6, 0.0, 0.8, -1.6, 0.0};
  constexpr double kp[6] = {2.0, 7.59, 0.0, 2.0, 7.59, 0.0};
  constexpr double kd[6] = {0.08, 0.682, 0.05, 0.08, 0.682, 0.05};
  constexpr double scale[6] = {0.5, 0.5, 10.0, 0.5, 0.5, 10.0};
  constexpr double limit[6] = {5.5, 14.0, 5.5, 5.5, 14.0, 5.5};
  constexpr bool wheel[6] = {false, false, true, false, false, true};
  if (substep % 2 == 0) {
    for (int index = 0; index < 6; ++index) {
      const int joint = model->actuator_trnid[2 * index];
      const double position = data->qpos[model->jnt_qposadr[joint]];
      const double velocity = data->qvel[model->jnt_dofadr[joint]];
      const double selected = wheel[index]
                                  ? std::clamp(static_cast<double>(action[index]), -3.5, 3.5)
                                  : std::clamp(static_cast<double>(action[index]), -100.0, 100.0);
      const double torque = wheel[index]
                                ? kd[index] * (selected * scale[index] - velocity)
                                : kp[index] * (defaults[index] + selected * scale[index] - position) -
                                      kd[index] * velocity;
      data->ctrl[index] = std::clamp(torque, -limit[index], limit[index]);
    }
    for (int index = 6; index < std::min(static_cast<int>(model->nu), 8); ++index) {
      const int joint = model->actuator_trnid[2 * index];
      data->ctrl[index] = std::clamp(-10.0 * data->qpos[model->jnt_qposadr[joint]] -
                                        0.5 * data->qvel[model->jnt_dofadr[joint]],
                                    -5.5, 5.5);
    }
  }
}

std::size_t input_index(const aar::DeploymentContract& contract, const std::string& name) {
  for (std::size_t index = 0; index < contract.inputs.size(); ++index) {
    if (contract.inputs[index].name == name) return index;
  }
  throw std::runtime_error("observation builder requires input: " + name);
}

void update_pe01_inputs(const mjData* data, const aar::DeploymentContract& contract,
                        std::vector<std::vector<float>>& inputs,
                        std::vector<float>& history) {
  const auto history_index = input_index(contract, "observation_history");
  const auto observation_index = input_index(contract, "observation");
  std::vector<float> frame(inputs[observation_index].size(), 0.0F);
  if (frame.size() != 30 || history.size() != 300) {
    throw std::runtime_error("pe01_v1 requires 30D frame and 300D history");
  }
  for (int index = 0; index < 6; ++index) {
    frame[static_cast<std::size_t>(index)] = static_cast<float>(data->qpos[7 + index]);
    frame[static_cast<std::size_t>(6 + index)] = static_cast<float>(data->qvel[6 + index]);
    frame[static_cast<std::size_t>(18 + index)] = static_cast<float>(data->ctrl[index]);
  }
  for (int index = 0; index < 3; ++index) {
    frame[static_cast<std::size_t>(12 + index)] = static_cast<float>(data->qvel[3 + index]);
    frame[static_cast<std::size_t>(15 + index)] = static_cast<float>(data->qvel[index]);
  }
  std::move(history.begin() + 30, history.end(), history.begin());
  std::copy(frame.begin(), frame.end(), history.end() - 30);
  inputs[history_index] = history;
  inputs[observation_index] = std::move(frame);
}

void validate_golden_output(const std::filesystem::path& release,
                            const aar::DeploymentContract& contract,
                            const std::vector<std::vector<float>>& outputs) {
  for (std::size_t index = 0; index < contract.outputs.size(); ++index) {
    const auto expected = aar::load_float32_npy(
        release / "golden_outputs" / (contract.outputs[index].name + ".npy"));
    if (expected.values.size() != outputs[index].size()) {
      throw std::runtime_error("golden output size differs");
    }
    float maximum_error = 0.0F;
    for (std::size_t value = 0; value < expected.values.size(); ++value) {
      maximum_error = std::max(
          maximum_error, std::abs(expected.values[value] - outputs[index][value]));
    }
    if (maximum_error > 1.0e-5F) {
      throw std::runtime_error("ONNX golden output mismatch: max error=" +
                               std::to_string(maximum_error));
    }
  }
}

void validate_model_contract(const mjModel* model, const aar::DeploymentContract& contract) {
  if (contract.joint_order.size() > static_cast<std::size_t>(model->nu)) {
    throw std::runtime_error("joint_order exceeds MuJoCo actuator count");
  }
  for (std::size_t actuator = 0; actuator < contract.joint_order.size(); ++actuator) {
    const int joint_id = model->actuator_trnid[2 * static_cast<int>(actuator)];
    const char* joint_name = mj_id2name(model, mjOBJ_JOINT, joint_id);
    if (!joint_name || contract.joint_order[static_cast<std::size_t>(actuator)] != joint_name) {
      throw std::runtime_error("joint_order differs from MuJoCo actuator mapping at index " +
                               std::to_string(actuator));
    }
  }
  if (contract.physics_hz <= 0.0 || contract.policy_hz <= 0.0 ||
      contract.motor_hz <= 0.0) {
    throw std::runtime_error("control frequencies must be positive");
  }
  const double model_physics_hz = 1.0 / model->opt.timestep;
  if (std::abs(model_physics_hz - contract.physics_hz) > 1.0e-9) {
    throw std::runtime_error("manifest physics_hz differs from MuJoCo timestep");
  }
  const double substeps = contract.physics_hz / contract.policy_hz;
  if (std::abs(substeps - std::round(substeps)) > 1.0e-9) {
    throw std::runtime_error("physics_hz must be an integer multiple of policy_hz");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = parse_options(argc, argv);
    const auto contract = aar::load_contract(options.release / "deployment_manifest.json");
    const auto scene_path = options.release / contract.scene_path;
    const auto policy_path = options.release / contract.policy_path;
    if (!std::filesystem::is_regular_file(scene_path) ||
        !std::filesystem::is_regular_file(policy_path)) {
      throw std::runtime_error("release policy or scene is missing");
    }
    char error[1024]{};
    mjModel* model = mj_loadXML(scene_path.c_str(), nullptr, error, sizeof(error));
    if (!model) throw std::runtime_error("MuJoCo load failed: " + std::string(error));
    mjData* data = mj_makeData(model);
    if (!data) {
      mj_deleteModel(model);
      throw std::runtime_error("MuJoCo data allocation failed");
    }
    validate_model_contract(model, contract);
    if (contract.observation_builder.rfind("we11_", 0) == 0) {
      const int home = mj_name2id(model, mjOBJ_KEY, "home");
      if (home >= 0) mj_resetDataKeyframe(model, data, home);
      mj_forward(model, data);
    }
    auto inputs = initial_inputs(options.release, contract);
    aar::OnnxActor actor(policy_path, contract);
    const auto initial_outputs = actor.run(inputs);
    validate_golden_output(options.release, contract, initial_outputs);
    std::filesystem::create_directories(options.telemetry);
    aar::TelemetryWriter telemetry(options.telemetry / "sim2sim.csv",
                                   options.telemetry / "sim2sim.jsonl");
    const int substeps = std::max(
        1, static_cast<int>(std::llround(contract.physics_hz / contract.policy_hz)));
    std::vector<float> history;
    We11State we11;
    if (contract.observation_builder == "pe01_v1") {
      history = inputs[input_index(contract, "observation_history")];
    } else if (contract.observation_builder != "we11_v2_145" &&
               contract.observation_builder != "golden_inputs") {
      throw std::runtime_error("unknown observation builder: " + contract.observation_builder);
    }
#ifdef AAR_WITH_GLFW
    std::unique_ptr<Viewer> viewer;
    if (options.interactive) viewer = std::make_unique<Viewer>(model, data);
#else
    if (options.interactive) throw std::runtime_error("sim2sim was built without GLFW");
#endif
    for (int step = 0; step < options.steps; ++step) {
#ifdef AAR_WITH_GLFW
      if (viewer && glfwWindowShouldClose(viewer->window)) break;
      if (viewer) viewer->gamepad(contract, inputs, we11.command);
#endif
      if (contract.observation_builder == "pe01_v1") {
        update_pe01_inputs(data, contract, inputs, history);
      } else if (contract.observation_builder.rfind("we11_", 0) == 0) {
        update_we11_inputs(model, data, contract, inputs, we11);
      }
      const auto outputs = actor.run(inputs);
      const auto& action = outputs.front();
      if (action.size() != contract.joint_order.size()) {
        throw std::runtime_error("actor output size differs from policy joint order");
      }
      if (contract.observation_builder.rfind("we11_", 0) != 0) {
        for (std::size_t actuator = 0; actuator < action.size(); ++actuator) {
          const double clipped = std::clamp(static_cast<double>(action[actuator]),
                                            -contract.action_clip, contract.action_clip);
          data->ctrl[static_cast<int>(actuator)] = clipped * contract.action_scale;
        }
      }
      for (int substep = 0; substep < substeps; ++substep) {
        if (contract.observation_builder.rfind("we11_", 0) == 0) {
          apply_we11_control(model, data, action, substep);
        }
        mj_step(model, data);
      }
      if (contract.observation_builder.rfind("we11_", 0) == 0) {
        we11.previous_action = action;
      }
      std::map<std::string, double> values{{"base_height", data->qpos[2]}};
      for (std::size_t index = 0; index < action.size(); ++index) {
        values["action_" + std::to_string(index)] = action[index];
        values["ctrl_" + std::to_string(index)] = data->ctrl[static_cast<int>(index)];
      }
      telemetry.write(data->time, values);
#ifdef AAR_WITH_GLFW
      if (viewer) viewer->draw(action.empty() ? 0.0F : action.front());
#endif
    }
    std::cout << "sim2sim ok robot=" << contract.robot_id << " task=" << contract.task_id
              << " steps=" << options.steps << " mujoco=" << mj_versionString() << '\n';
    mj_deleteData(data);
    mj_deleteModel(model);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "sim2sim error: " << error.what() << '\n';
    return 2;
  }
}
