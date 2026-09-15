/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#include "rl_sim_mujoco.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <limits>
#include <sstream>

RL_Sim* RL_Sim::instance = nullptr;

namespace
{
constexpr int kDr002RewardPlotPoints = 500;

void InitDr002RewardFigure(
    mjvFigure &figure,
    const char *title,
    const char *line0_name,
    const char *line1_name,
    float y_min,
    float y_max)
{
    mjv_defaultFigure(&figure);
    figure.figurergba[3] = 0.80f;
    figure.flg_extend = 0;
    figure.flg_legend = 1;
    figure.gridsize[0] = 5;
    figure.gridsize[1] = 5;
    figure.range[0][0] = -static_cast<float>(kDr002RewardPlotPoints - 1);
    figure.range[0][1] = 0.0f;
    figure.range[1][0] = y_min;
    figure.range[1][1] = y_max;
    mju::strcpy_arr(figure.title, title);
    mju::strcpy_arr(figure.xlabel, "history (50 Hz samples)");
    mju::strcpy_arr(figure.linename[0], line0_name);
    mju::strcpy_arr(figure.linename[1], line1_name);
    figure.linergb[0][0] = 0.2f;
    figure.linergb[0][1] = 0.8f;
    figure.linergb[0][2] = 1.0f;
    figure.linergb[1][0] = 1.0f;
    figure.linergb[1][1] = 0.25f;
    figure.linergb[1][2] = 0.2f;
    for (int line = 0; line < 2; ++line)
    {
        figure.linepnt[line] = kDr002RewardPlotPoints;
        for (int point = 0; point < kDr002RewardPlotPoints; ++point)
        {
            figure.linedata[line][2 * point] =
                static_cast<float>(point - (kDr002RewardPlotPoints - 1));
            figure.linedata[line][2 * point + 1] = 0.0f;
        }
    }
}

void PushDr002RewardFigureSample(mjvFigure &figure, int line, float value)
{
    for (int point = 0; point < kDr002RewardPlotPoints - 1; ++point)
    {
        figure.linedata[line][2 * point + 1] =
            figure.linedata[line][2 * (point + 1) + 1];
    }
    figure.linedata[line][2 * (kDr002RewardPlotPoints - 1) + 1] = value;
}
}  // namespace

static std::string ResolveDr002ConfigName()
{
    const char *config_env = std::getenv("RL_SAR_DR002_CONFIG");
    if (config_env && std::string(config_env).size() > 0)
    {
        return config_env;
    }
    return "robot_lab";
}

static int Dr002ActionTraceLimit()
{
    const char *trace_env = std::getenv("RL_SAR_TRACE_ACTIONS");
    if (!trace_env)
    {
        return 0;
    }

    try
    {
        return std::max(0, std::stoi(trace_env));
    }
    catch (const std::exception&)
    {
        return 20;
    }
}

static void PrintTraceVector(const std::string &label, const std::vector<float> &values)
{
    std::cout << label << "=[";
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << values[i];
    }
    std::cout << "]";
}

static float InterpolateNxbxTnTorque(
    float speed_rad_s,
    const std::vector<float> &speed_rpm,
    const std::vector<float> &torque_nm)
{
    if (speed_rpm.size() < 2 || speed_rpm.size() != torque_nm.size())
    {
        throw std::runtime_error(
            "NXBx T-N curve requires matching speed/torque arrays with at least two points");
    }
    for (size_t i = 0; i < speed_rpm.size(); ++i)
    {
        if (!std::isfinite(speed_rpm[i]) || !std::isfinite(torque_nm[i]) ||
            speed_rpm[i] < 0.0f || torque_nm[i] < 0.0f ||
            (i > 0 && (speed_rpm[i] <= speed_rpm[i - 1] ||
                       torque_nm[i] > torque_nm[i - 1])))
        {
            throw std::runtime_error(
                "NXBx T-N curve requires ascending non-negative RPM and "
                "non-increasing non-negative torque");
        }
    }

    constexpr float rad_s_to_rpm = 60.0f / (2.0f * 3.14159265358979323846f);
    const float query_rpm = std::fabs(speed_rad_s) * rad_s_to_rpm;
    if (query_rpm <= speed_rpm.front())
    {
        return torque_nm.front();
    }
    if (query_rpm >= speed_rpm.back())
    {
        return torque_nm.back();
    }

    const auto upper = std::upper_bound(speed_rpm.begin(), speed_rpm.end(), query_rpm);
    const size_t upper_index = static_cast<size_t>(upper - speed_rpm.begin());
    const size_t lower_index = upper_index - 1;
    const float span = speed_rpm[upper_index] - speed_rpm[lower_index];
    const float alpha = (query_rpm - speed_rpm[lower_index]) / span;
    return torque_nm[lower_index] +
           alpha * (torque_nm[upper_index] - torque_nm[lower_index]);
}

RL_Sim::RL_Sim(int argc, char **argv)
{
    // Set static instance pointer early for signal handler
    instance = this;

    if (argc < 3)
    {
        std::cout << LOGGER::ERROR << "Usage: " << argv[0] << " robot_name scene_name" << std::endl;
        throw std::runtime_error("Invalid arguments");
    }
    else
    {
        this->robot_name = argv[1];
        this->scene_name = argv[2];
    }
    if (this->robot_name == "dr002" && this->scene_name == "scene")
    {
        std::cout << LOGGER::WARNING << "[MuJoCo] DR002 uses latest V4 assets; remapping scene -> scene_unilab_latest" << std::endl;
        this->scene_name = "scene_unilab_latest";
    }

    this->ang_vel_axis = "body";

    // now launch mujoco
    std::cout << LOGGER::INFO << "[MuJoCo] Launching..." << std::endl;

    // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
    if (rosetta_error_msg)
    {
        DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
        std::exit(1);
    }
#endif

    // print version, check compatibility
    std::cout << LOGGER::INFO << "[MuJoCo] Version: " << mj_versionString() << std::endl;
    if (mjVERSION_HEADER != mj_version())
    {
        mju_error("Headers and library have different versions");
    }

    // scan for libraries in the plugin directory to load additional plugins
    scanPluginLibraries();

    mjvCamera cam;
    mjv_defaultCamera(&cam);

    mjvOption opt;
    mjv_defaultOption(&opt);

    mjvPerturb pert;
    mjv_defaultPerturb(&pert);

    // simulate object encapsulates the UI
    sim = std::make_unique<mj::Simulate>(
        std::make_unique<mj::GlfwAdapter>(),
        &cam, &opt, &pert, /* is_passive = */ false);

    std::string filename = std::string(CMAKE_CURRENT_SOURCE_DIR) + "/../rl_sar_zoo/" + this->robot_name + "_description/mjcf/" + this->scene_name + ".xml";

    // start physics thread
    std::thread physicsthreadhandle(&PhysicsThread, sim.get(), filename.c_str());
    physicsthreadhandle.detach();

    while (1)
    {
        if (d)
        {
            std::cout << LOGGER::INFO << "[MuJoCo] Data prepared" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    this->mj_model = m;
    this->mj_data = d;
    std::string joystick_device = "/dev/input/js0";
    if (const char* env_joystick = std::getenv("RL_SAR_JOYSTICK"))
    {
        joystick_device = env_joystick;
    }
    else
    {
        for (int i = 0; i < 8; ++i)
        {
            std::string candidate = "/dev/input/js" + std::to_string(i);
            if (access(candidate.c_str(), R_OK) == 0)
            {
                joystick_device = candidate;
                break;
            }
        }
    }
    this->SetupSysJoystick(joystick_device, 16); // 16 bits joystick

    // read params from yaml
    this->ReadYaml(this->robot_name, "base.yaml");
    if (this->robot_name == "dr002")
    {
        // DR002 standby/reset must use the same fixed gains and initial pose as
        // the selected policy. Loading only base.yaml here left startup on the
        // base gains until the first locomotion transition.
        this->config_name = ResolveDr002ConfigName();
        this->ReadYaml(
            this->robot_name + "/" + this->config_name,
            "config.yaml");
        if (!this->params.Has("model_name"))
        {
            throw std::runtime_error(
                "DR002 policy config is missing or invalid: " +
                this->robot_name + "/" + this->config_name + "/config.yaml");
        }
        std::cout << LOGGER::INFO << "[DR002] Preloaded policy config: "
                  << this->config_name << std::endl;
    }
    this->InitDr002DeployLimits();

    // auto load FSM by robot_name
    if (FSMManager::GetInstance().IsTypeSupported(this->robot_name))
    {
        auto fsm_ptr = FSMManager::GetInstance().CreateFSM(this->robot_name, this);
        if (fsm_ptr)
        {
            this->fsm = *fsm_ptr;
        }
    }
    else
    {
        std::cout << LOGGER::ERROR << "[FSM] No FSM registered for robot: " << this->robot_name << std::endl;
    }

    // init robot
    this->InitJointNum(this->params.Get<int>("num_of_dofs"));
    this->InitOutputs();
    this->InitControl();
    this->CacheMujocoAddresses();
    this->InitWingCommand();
    if (this->robot_name == "dr002")
    {
        this->InitDr002RewardFigures();
        this->simulation_running = false;
        this->sim->run = 0;
        this->sim->manual_step_mode = 1;
        this->InitDr002BaseComOffset();
        std::cout << LOGGER::NOTE
                  << "MuJoCo drag perturbation: double-click a robot body to select it; "
                  << "hold Ctrl and right-drag to apply force, or Ctrl and left-drag to apply torque."
                  << std::endl;
        this->ResetMujocoToInitialPose(true, false, "DR002 two-wheel startup");
        if (const char *autostart_env = std::getenv("RL_SAR_DR002_AUTOSTART"))
        {
            const std::string value(autostart_env);
            if (!(value == "0" || value == "false" || value == "False" || value == "off"))
            {
                this->sim->run = 1;
                std::cout << LOGGER::INFO << "[DR002] Autostart requested by RL_SAR_DR002_AUTOSTART." << std::endl;
            }
        }
        std::cout << LOGGER::NOTE
                  << "DR002 controls: 1/RB+DPadUp start, P/LB+X stop, "
                  << "R/RB+Y reset, LY=vx, RX=yaw, RY=height velocity, "
                  << "LB/RB=wing down/up."
                  << std::endl;
    }
    else if (this->robot_name == "nxbx")
    {
        this->simulation_running = false;
        this->sim->run = 0;
        this->sim->manual_step_mode = 1;
        this->SampleNxbxTargetDelaySteps();
        this->ResetNxbxTargetDelayBuffer();
        this->ResetMujocoToInitialPose(true, false, "NXBx two-wheel startup");
        std::cout << LOGGER::NOTE
                  << "NXBx controls: 1/RB+DPadUp start, P/LB+X stop, "
                  << "R/RB+Y reset, LY=vx, RX=yaw." << std::endl;
    }

    // loop
    this->loop_control = std::make_shared<LoopFunc>("loop_control", this->params.Get<float>("dt"), std::bind(&RL_Sim::RobotControl, this));
    this->loop_control->start();
    if (this->robot_name != "dr002" && this->robot_name != "nxbx")
    {
        this->loop_rl = std::make_shared<LoopFunc>("loop_rl", this->params.Get<float>("dt") * this->params.Get<int>("decimation"), std::bind(&RL_Sim::RunModel, this));
        this->loop_rl->start();
    }

    // keyboard
    this->loop_keyboard = std::make_shared<LoopFunc>("loop_keyboard", 0.05, std::bind(&RL_Sim::KeyboardInterface, this));
    this->loop_keyboard->start();

    // joystick
    this->loop_joystick = std::make_shared<LoopFunc>("loop_joystick", 0.01, std::bind(&RL_Sim::GetSysJoystick, this));
    this->loop_joystick->start();

#ifdef PLOT
    this->plot_t = std::vector<int>(this->plot_size, 0);
    this->plot_real_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    this->plot_target_joint_pos.resize(this->params.Get<int>("num_of_dofs"));
    for (auto &vector : this->plot_real_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    for (auto &vector : this->plot_target_joint_pos) { vector = std::vector<float>(this->plot_size, 0); }
    this->loop_plot = std::make_shared<LoopFunc>("loop_plot", 0.001, std::bind(&RL_Sim::Plot, this));
    this->loop_plot->start();
#endif
#ifdef CSV_LOGGER
    this->CSVInit(this->robot_name);
#endif

    std::cout << LOGGER::INFO << "RL_Sim start" << std::endl;

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
}

RL_Sim::~RL_Sim()
{
    // Clear static instance pointer
    instance = nullptr;

    this->loop_keyboard->shutdown();
    this->loop_joystick->shutdown();
    this->loop_control->shutdown();
    if (this->loop_rl)
    {
        this->loop_rl->shutdown();
    }
#ifdef PLOT
    this->loop_plot->shutdown();
#endif
    std::cout << LOGGER::INFO << "RL_Sim exit" << std::endl;
}

void RL_Sim::CacheMujocoAddresses()
{
    if (!this->mj_model || !this->params.Has("num_of_dofs"))
    {
        return;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    this->joint_qpos_addr.assign(num_dofs, -1);
    this->joint_qvel_addr.assign(num_dofs, -1);
    this->mujoco_joint_side_torques.assign(num_dofs, 0.0f);
    this->root_qpos_addr = -1;
    this->root_qvel_addr = -1;
    this->mujoco_gyro_sensor_adr = -1;
    this->mujoco_gyro_sensor_dim = 0;
    this->mujoco_imu_site_id = mj_name2id(this->mj_model, mjOBJ_SITE, "imu");
    this->mujoco_base_body_id = mj_name2id(this->mj_model, mjOBJ_BODY, "base_link");

    for (int jid = 0; jid < this->mj_model->njnt; ++jid)
    {
        if (this->mj_model->jnt_type[jid] == mjJNT_FREE)
        {
            this->root_qpos_addr = this->mj_model->jnt_qposadr[jid];
            this->root_qvel_addr = this->mj_model->jnt_dofadr[jid];
            break;
        }
    }

    const auto joint_names = this->params.Get<std::vector<std::string>>("joint_names");
    for (int i = 0; i < num_dofs && i < static_cast<int>(joint_names.size()); ++i)
    {
        const int joint_id = mj_name2id(this->mj_model, mjOBJ_JOINT, joint_names[i].c_str());
        if (joint_id < 0)
        {
            continue;
        }
        this->joint_qpos_addr[i] = this->mj_model->jnt_qposadr[joint_id];
        this->joint_qvel_addr[i] = this->mj_model->jnt_dofadr[joint_id];
    }

    if (this->params.Has("joint_controller_names"))
    {
        const auto joint_controller_names = this->params.Get<std::vector<std::string>>("joint_controller_names");
        for (int i = 0; i < num_dofs && i < static_cast<int>(joint_controller_names.size()); ++i)
        {
            this->joint_efforts[joint_controller_names[i]] = 0.0f;
        }
    }

    auto cache_sensor = [this](const char *name, int &adr, int &dim) {
        const int sensor_id = mj_name2id(this->mj_model, mjOBJ_SENSOR, name);
        if (sensor_id < 0)
        {
            return false;
        }
        adr = this->mj_model->sensor_adr[sensor_id];
        dim = this->mj_model->sensor_dim[sensor_id];
        return dim >= 3;
    };

    if (!cache_sensor("gyro", this->mujoco_gyro_sensor_adr, this->mujoco_gyro_sensor_dim))
    {
        cache_sensor("imu_gyro", this->mujoco_gyro_sensor_adr, this->mujoco_gyro_sensor_dim);
    }

    if (this->robot_name == "dr002")
    {
        std::cout << LOGGER::INFO << "[MuJoCo IMU] gyro sensor "
                  << (this->mujoco_gyro_sensor_adr >= 0 ? "enabled" : "missing")
                  << ", projected gravity source "
                  << (this->mujoco_imu_site_id >= 0 ? "imu site xmat" : "base_link xmat")
                  << std::endl;
    }
}

void RL_Sim::InitDr002BaseComOffset()
{
    const char *offset_env = std::getenv("RL_SAR_DR002_BASE_COM_OFFSET_BODY");
    if (!offset_env || std::string(offset_env).empty())
    {
        return;
    }
    if (!this->mj_model || !this->mj_data || !this->sim)
    {
        std::cout << LOGGER::WARNING << "[DR002 COM] Model data unavailable; ignoring offset." << std::endl;
        return;
    }

    std::istringstream input(offset_env);
    std::array<double, 3> offset = {0.0, 0.0, 0.0};
    std::string trailing;
    if (!(input >> offset[0] >> offset[1] >> offset[2]) || (input >> trailing) ||
        !std::isfinite(offset[0]) || !std::isfinite(offset[1]) || !std::isfinite(offset[2]))
    {
        std::cout << LOGGER::WARNING
                  << "[DR002 COM] RL_SAR_DR002_BASE_COM_OFFSET_BODY must contain exactly "
                  << "three finite values in meters; ignoring '" << offset_env << "'."
                  << std::endl;
        return;
    }

    const int base_body_id = mj_name2id(this->mj_model, mjOBJ_BODY, "base_link");
    if (base_body_id < 0)
    {
        std::cout << LOGGER::WARNING << "[DR002 COM] base_link body not found; ignoring offset." << std::endl;
        return;
    }

    const std::unique_lock<std::recursive_mutex> lock(this->sim->mtx);
    mj_forward(this->mj_model, this->mj_data);

    std::array<double, 3> total_com_before = {
        this->mj_data->subtree_com[3 * base_body_id + 0],
        this->mj_data->subtree_com[3 * base_body_id + 1],
        this->mj_data->subtree_com[3 * base_body_id + 2],
    };
    double total_mass = 0.0;
    for (int body_id = 0; body_id < this->mj_model->nbody; ++body_id)
    {
        total_mass += this->mj_model->body_mass[body_id];
    }

    for (int axis = 0; axis < 3; ++axis)
    {
        this->mj_model->body_ipos[3 * base_body_id + axis] += offset[axis];
    }
    mj_setConst(this->mj_model, this->mj_data);
    mj_forward(this->mj_model, this->mj_data);

    std::array<double, 3> total_com_delta = {
        this->mj_data->subtree_com[3 * base_body_id + 0] - total_com_before[0],
        this->mj_data->subtree_com[3 * base_body_id + 1] - total_com_before[1],
        this->mj_data->subtree_com[3 * base_body_id + 2] - total_com_before[2],
    };
    std::cout << LOGGER::INFO << std::fixed << std::setprecision(6)
              << "[DR002 COM] base_link body-frame offset=["
              << offset[0] << ", " << offset[1] << ", " << offset[2] << "] m"
              << ", whole-robot COM world delta=["
              << total_com_delta[0] << ", " << total_com_delta[1] << ", " << total_com_delta[2] << "] m"
              << ", total_mass=" << total_mass << " kg"
              << std::defaultfloat << std::endl;
}

void RL_Sim::UpdateMujocoJointTorques()
{
    if (!this->mj_model || !this->mj_data || !this->params.Has("num_of_dofs"))
    {
        return;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    if (static_cast<int>(this->mujoco_joint_side_torques.size()) != num_dofs)
    {
        this->mujoco_joint_side_torques.assign(num_dofs, 0.0f);
    }

    std::vector<std::string> joint_controller_names;
    if (this->params.Has("joint_controller_names"))
    {
        joint_controller_names = this->params.Get<std::vector<std::string>>("joint_controller_names");
    }

    for (int i = 0; i < num_dofs; ++i)
    {
        float torque = 0.0f;
        if (i < static_cast<int>(this->joint_qvel_addr.size()))
        {
            const int dof_adr = this->joint_qvel_addr[i];
            if (dof_adr >= 0 && dof_adr < this->mj_model->nv && this->mj_data->qfrc_actuator)
            {
                torque = static_cast<float>(this->mj_data->qfrc_actuator[dof_adr]);
            }
        }
        this->mujoco_joint_side_torques[i] = torque;
        if (i < static_cast<int>(joint_controller_names.size()))
        {
            this->joint_efforts[joint_controller_names[i]] = torque;
        }
    }
}

void RL_Sim::UpdateJointTorqueOverlay()
{
    if (!this->sim ||
        (this->robot_name != "dr002" && this->robot_name != "nxbx") ||
        this->mujoco_joint_side_torques.empty())
    {
        return;
    }

    std::ostringstream content;
    content << std::fixed << std::setprecision(2);
    if (this->robot_name == "nxbx" && this->mujoco_joint_side_torques.size() >= 2)
    {
        content << "L wheel: " << this->mujoco_joint_side_torques[0] << "\n";
        content << "R wheel: " << this->mujoco_joint_side_torques[1];
    }
    else if (this->mujoco_joint_side_torques.size() >= 6)
    {
        content << "L thigh/calf/foot: "
                << this->mujoco_joint_side_torques[0] << "  "
                << this->mujoco_joint_side_torques[1] << "  "
                << this->mujoco_joint_side_torques[2] << "\n";
        content << "R thigh/calf/foot: "
                << this->mujoco_joint_side_torques[3] << "  "
                << this->mujoco_joint_side_torques[4] << "  "
                << this->mujoco_joint_side_torques[5];
    }
    else
    {
        for (size_t i = 0; i < this->mujoco_joint_side_torques.size(); ++i)
        {
            if (i > 0)
            {
                content << "  ";
            }
            content << "j" << i << "=" << this->mujoco_joint_side_torques[i];
        }
    }

    this->sim->SetUserOverlay("joint-side torque [Nm]\nsource: qfrc_actuator", content.str().c_str());
}

// ----------------------------------------------------------------------------
// Height and wing drivers (sim-side joystick integration)
// ----------------------------------------------------------------------------
// The RL policy does not control the wings. Instead LB/RB produce a signed
// rad/s target that we integrate into a position setpoint at
// every control tick, clamp to the joint's mechanical range, and turn into a
// motor torque via a plain position PD:
//     tau = kp * (pos_target - pos) - kd * vel
// This means stick-neutral (rad/s = 0) freezes the setpoint and the PD holds
// the wing at its current angle instead of letting it fall under gravity,
// matching the hardware wing_motor_node behavior in RC_CONTROL mode. Wing
// observations are read back from the actual qpos/qvel of the wing joints, so
// any range clamp shows up naturally in the observation.
void RL_Sim::InitWingCommand()
{
    // Only DR002 has wing actuators/joints in its MJCF.
    if (this->robot_name != "dr002" || !this->mj_model)
    {
        this->wing_cmd_enabled_ = false;
        return;
    }

    if (this->params.Has("height_velocity_command"))
    {
        const YAML::Node hc = this->params.config_node["height_velocity_command"];
        if (hc && hc.IsMap())
        {
            if (hc["min_height"])
                this->dr002_height_command_min_ = hc["min_height"].as<float>();
            if (hc["max_height"])
                this->dr002_height_command_max_ = hc["max_height"].as<float>();
            if (hc["initial_height"])
            {
                this->dr002_height_command_initial_ = hc["initial_height"].as<float>();
                this->dr002_height_command_ = this->dr002_height_command_initial_;
            }
            if (hc["max_m_per_s"])
                this->dr002_height_velocity_max_m_s_ = hc["max_m_per_s"].as<float>();
            if (hc["joystick_axis"])
                this->dr002_height_joystick_axis_ = hc["joystick_axis"].as<int>();
            if (hc["joystick_invert_sign"])
                this->dr002_height_joystick_invert_sign_ =
                    hc["joystick_invert_sign"].as<bool>();
        }
    }
    if (this->dr002_height_command_min_ >= this->dr002_height_command_max_ ||
        this->dr002_height_velocity_max_m_s_ <= 0.0f)
    {
        throw std::runtime_error("Invalid DR002 height_velocity_command configuration");
    }
    this->dr002_height_command_ = std::clamp(
        this->dr002_height_command_,
        this->dr002_height_command_min_,
        this->dr002_height_command_max_);

    // Config defaults are documented in config.yaml. Each key is optional so a
    // missing config.yaml block still yields a working sim.
    // The `wing_velocity_command` node is nested in YAML — read the sub-Node
    // directly rather than going through the flat dotted-key lookup in
    // YamlParams::Get (which only supports top-level keys).
    if (this->params.Has("wing_velocity_command"))
    {
        const YAML::Node wc = this->params.config_node["wing_velocity_command"];
        if (wc && wc.IsMap())
        {
            if (wc["enabled"])
            {
                this->wing_cmd_enabled_ = wc["enabled"].as<bool>();
            }
            if (wc["max_rad_per_s"])
            {
                this->wing_max_rad_per_s_ =
                    static_cast<double>(wc["max_rad_per_s"].as<float>());
            }
            if (wc["kp"])
            {
                this->wing_kp_ = static_cast<double>(wc["kp"].as<float>());
            }
            if (wc["kd"])
            {
                this->wing_kd_ = static_cast<double>(wc["kd"].as<float>());
            }
            if (wc["decrease_button"])
            {
                this->wing_decrease_button_ = wc["decrease_button"].as<int>();
            }
            if (wc["increase_button"])
            {
                this->wing_increase_button_ = wc["increase_button"].as<int>();
            }
        }
    }

    // Actuator ctrl indices default to the last two slots (nu=8, indices 6,7).
    // Config override supports models with a different actuator order.
    if (this->params.Has("wing_actuator_ctrl_indices"))
    {
        const auto ctrl = this->params.Get<std::vector<int>>(
            "wing_actuator_ctrl_indices");
        if (ctrl.size() >= 2)
        {
            this->wing_actuator_ctrl_idx_[0] = ctrl[0];
            this->wing_actuator_ctrl_idx_[1] = ctrl[1];
        }
    }

    // Resolve wing joint qpos/qvel addresses via mj_name2id. Joint names
    // default to the plan's canonical [left_wing_joint, right_wing_joint];
    // config `wing_joint_names` overrides this.
    std::array<std::string, 2> joint_names = {"left_wing_joint", "right_wing_joint"};
    if (this->params.Has("wing_joint_names"))
    {
        const auto names = this->params.Get<std::vector<std::string>>(
            "wing_joint_names");
        if (names.size() >= 2)
        {
            joint_names[0] = names[0];
            joint_names[1] = names[1];
        }
    }

    for (int i = 0; i < 2; ++i)
    {
        const int jid = mj_name2id(
            this->mj_model, mjOBJ_JOINT, joint_names[i].c_str());
        if (jid < 0)
        {
            std::cout << LOGGER::WARNING
                      << "[WingCmd] Wing joint '" << joint_names[i]
                      << "' not found in MuJoCo model; wing command disabled."
                      << std::endl;
            this->wing_cmd_enabled_ = false;
            this->wing_qpos_addr_[i] = -1;
            this->wing_qvel_addr_[i] = -1;
            continue;
        }
        this->wing_qpos_addr_[i] = this->mj_model->jnt_qposadr[jid];
        this->wing_qvel_addr_[i] = this->mj_model->jnt_dofadr[jid];
        // Read the joint range from the model so we clamp the setpoint to
        // the same bounds MuJoCo enforces via <joint range="...">.
        if (this->mj_model->jnt_limited[jid])
        {
            this->wing_pos_lower_[i] = this->mj_model->jnt_range[jid * 2 + 0];
            this->wing_pos_upper_[i] = this->mj_model->jnt_range[jid * 2 + 1];
        }
    }

    // Validate ctrl indices against the loaded model nu.
    for (int i = 0; i < 2; ++i)
    {
        const int ctrl_idx = this->wing_actuator_ctrl_idx_[i];
        if (ctrl_idx < 0 || ctrl_idx >= this->mj_model->nu)
        {
            std::cout << LOGGER::WARNING
                      << "[WingCmd] wing_actuator_ctrl_indices[" << i << "]="
                      << ctrl_idx << " is out of range for nu=" << this->mj_model->nu
                      << "; wing command disabled."
                      << std::endl;
            this->wing_cmd_enabled_ = false;
        }
    }

    // Prime the command so pre-policy resets have a sensible zero.
    this->wing_velocity_command_rad_s_ = {0.0, 0.0};
    // Position setpoint tracks the current qpos on reset; see ResetWingCommand.
    this->wing_position_command_rad_ = {0.0, 0.0};

    if (this->wing_cmd_enabled_)
    {
        std::cout << LOGGER::INFO
                  << "[WingCmd] Enabled: max_rad_per_s=" << this->wing_max_rad_per_s_
                  << ", kp=" << this->wing_kp_ << ", kd=" << this->wing_kd_
                  << ", range=[" << this->wing_pos_lower_[0] << "," << this->wing_pos_upper_[0]
                  << "],[" << this->wing_pos_lower_[1] << "," << this->wing_pos_upper_[1] << "]"
                  << ", buttons=[" << this->wing_decrease_button_
                  << ", " << this->wing_increase_button_ << "]"
                  << ", ctrl_idx=[" << this->wing_actuator_ctrl_idx_[0]
                  << ", " << this->wing_actuator_ctrl_idx_[1] << "]"
                  << ", joints=[" << joint_names[0] << ", " << joint_names[1] << "]"
                  << std::endl;
    }
}

void RL_Sim::ResetWingCommand()
{
    // Called from reset paths: zero the latched velocity target and re-align
    // the position setpoint to the actual joint qpos so the PD does not fight
    // the physical state. The joystick callback re-latches a non-zero value
    // on the next stick sample.
    this->wing_velocity_command_rad_s_ = {0.0, 0.0};
    if (this->mj_model && this->mj_data && this->mj_data->qpos)
    {
        for (int i = 0; i < 2; ++i)
        {
            const int qa = this->wing_qpos_addr_[i];
            if (qa >= 0 && qa < this->mj_model->nq)
            {
                this->wing_position_command_rad_[i] =
                    static_cast<double>(this->mj_data->qpos[qa]);
            }
            else
            {
                this->wing_position_command_rad_[i] = 0.0;
            }
        }
    }
    else
    {
        this->wing_position_command_rad_ = {0.0, 0.0};
    }
}

void RL_Sim::TickDr002HeightCommand()
{
    if (this->robot_name != "dr002" || !this->mj_model)
    {
        return;
    }
    double stick = 0.0;
    const int axis_idx = this->dr002_height_joystick_axis_;
    if (axis_idx >= 0 && axis_idx < static_cast<int>(std::size(this->sys_js_axis)))
    {
        stick = -static_cast<double>(this->sys_js_axis[axis_idx]) /
                static_cast<double>(this->sys_js_max_value);
    }
    if (this->dr002_height_joystick_invert_sign_)
    {
        stick = -stick;
    }
    stick = std::clamp(stick, -1.0, 1.0);
    const double dt = static_cast<double>(this->mj_model->opt.timestep);
    this->dr002_height_command_ = std::clamp(
        this->dr002_height_command_ +
            static_cast<float>(stick * this->dr002_height_velocity_max_m_s_ * dt),
        this->dr002_height_command_min_,
        this->dr002_height_command_max_);
}

void RL_Sim::TickWingCommand()
{
    if (!this->wing_cmd_enabled_ || !this->mj_model || !this->mj_data)
    {
        return;
    }

    // LB and RB drive the shared wing velocity in opposite directions. Both
    // wing hinges use axis vectors that cancel the
    // sign, so a shared command produces a symmetric left/right flap on the
    // WE11 model.
    //
    // Concurrency note: TickWingCommand is only called from RobotControl,
    // which already holds control_input_mutex for the full cycle. Re-locking
    // the same non-recursive mutex would deadlock, so read sys_js_button
    // directly. GetSysJoystick (the writer on loop_joystick) does its own
    // locking, and RobotControl serializes readers via that outer lock.
    const bool decrease = this->wing_decrease_button_ >= 0 &&
        this->wing_decrease_button_ < static_cast<int>(std::size(this->sys_js_button)) &&
        this->sys_js_button[this->wing_decrease_button_].pressed;
    const bool increase = this->wing_increase_button_ >= 0 &&
        this->wing_increase_button_ < static_cast<int>(std::size(this->sys_js_button)) &&
        this->sys_js_button[this->wing_increase_button_].pressed;
    const double stick = static_cast<double>(increase) - static_cast<double>(decrease);
    const double cmd_rad_s = stick * this->wing_max_rad_per_s_;
    this->wing_velocity_command_rad_s_[0] = cmd_rad_s;
    this->wing_velocity_command_rad_s_[1] = cmd_rad_s;

    // Integrate rad/s into the position setpoint at the physics timestep and
    // clamp to the mechanical range. When the stick is at rest the setpoint
    // is frozen and the PD below actively holds the wing.
    const double dt = static_cast<double>(this->mj_model->opt.timestep);
    for (int i = 0; i < 2; ++i)
    {
        double target = this->wing_position_command_rad_[i] +
                        this->wing_velocity_command_rad_s_[i] * dt;
        target = std::clamp(
            target, this->wing_pos_lower_[i], this->wing_pos_upper_[i]);
        this->wing_position_command_rad_[i] = target;
    }

    // Position-PD torque: tau = kp * (pos_target - pos) - kd * vel.
    // Writes directly to the wing motor ctrl slots; MuJoCo <motor> actuator
    // interprets ctrl as torque and clamps it by ctrlrange.
    if (!this->mj_data->ctrl || !this->mj_data->qpos || !this->mj_data->qvel)
    {
        return;
    }
    for (int i = 0; i < 2; ++i)
    {
        const int ctrl_idx = this->wing_actuator_ctrl_idx_[i];
        const int qa = this->wing_qpos_addr_[i];
        const int va = this->wing_qvel_addr_[i];
        if (ctrl_idx < 0 || ctrl_idx >= this->mj_model->nu) continue;
        if (qa < 0 || qa >= this->mj_model->nq) continue;
        if (va < 0 || va >= this->mj_model->nv) continue;
        const double pos = static_cast<double>(this->mj_data->qpos[qa]);
        const double vel = static_cast<double>(this->mj_data->qvel[va]);
        const double tau = this->wing_kp_ *
                               (this->wing_position_command_rad_[i] - pos)
                           - this->wing_kd_ * vel;
        this->mj_data->ctrl[ctrl_idx] = static_cast<mjtNum>(tau);
    }
}

void RL_Sim::UpdateWingObservation()
{
    // Read the actual wing joint pos/vel from the MuJoCo state. Normalize
    // position by pi (matches training's rad/pi convention) and velocity by
    // 0.1 (matches training's dof_vel_scale convention).
    this->obs.wing_angle.assign(2, 0.0f);
    this->obs.wing_vel.assign(2, 0.0f);
    if (!this->wing_cmd_enabled_ || !this->mj_model || !this->mj_data)
    {
        return;
    }

    static constexpr double kPi = 3.141592653589793;
    for (int i = 0; i < 2; ++i)
    {
        const int qpos_addr = this->wing_qpos_addr_[i];
        const int qvel_addr = this->wing_qvel_addr_[i];
        if (qpos_addr >= 0 && qpos_addr < this->mj_model->nq &&
            this->mj_data->qpos)
        {
            const double pos_rad = static_cast<double>(this->mj_data->qpos[qpos_addr]);
            this->obs.wing_angle[i] = static_cast<float>(pos_rad / kPi);
        }
        if (qvel_addr >= 0 && qvel_addr < this->mj_model->nv &&
            this->mj_data->qvel)
        {
            const double vel_rad_s = static_cast<double>(this->mj_data->qvel[qvel_addr]);
            this->obs.wing_vel[i] = static_cast<float>(vel_rad_s * 0.1);
        }
    }
}


void RL_Sim::GetState(RobotState<float> *state)
{
    if (mj_data)
    {
        const int num_dofs = this->params.Get<int>("num_of_dofs");
        if (this->joint_qpos_addr.empty())
        {
            this->CacheMujocoAddresses();
        }

        if (this->root_qpos_addr >= 0 && this->root_qvel_addr >= 0)
        {
            state->imu.quaternion[0] = mj_data->qpos[this->root_qpos_addr + 3];
            state->imu.quaternion[1] = mj_data->qpos[this->root_qpos_addr + 4];
            state->imu.quaternion[2] = mj_data->qpos[this->root_qpos_addr + 5];
            state->imu.quaternion[3] = mj_data->qpos[this->root_qpos_addr + 6];

            // MuJoCo free-joint angular velocity is already in the child/body frame.
            state->imu.gyroscope[0] = mj_data->qvel[this->root_qvel_addr + 3];
            state->imu.gyroscope[1] = mj_data->qvel[this->root_qvel_addr + 4];
            state->imu.gyroscope[2] = mj_data->qvel[this->root_qvel_addr + 5];
        }

        const std::vector<float> sensor_gyro = this->ReadMujocoSensor3(
            this->mujoco_gyro_sensor_adr,
            this->mujoco_gyro_sensor_dim);
        if (sensor_gyro.size() == 3)
        {
            state->imu.gyroscope[0] = sensor_gyro[0];
            state->imu.gyroscope[1] = sensor_gyro[1];
            state->imu.gyroscope[2] = sensor_gyro[2];
        }

        this->UpdateMujocoJointTorques();
        for (int i = 0; i < num_dofs; ++i)
        {
            if (i < static_cast<int>(this->joint_qpos_addr.size()) && this->joint_qpos_addr[i] >= 0)
            {
                state->motor_state.q[i] = mj_data->qpos[this->joint_qpos_addr[i]];
            }
            if (i < static_cast<int>(this->joint_qvel_addr.size()) && this->joint_qvel_addr[i] >= 0)
            {
                state->motor_state.dq[i] = mj_data->qvel[this->joint_qvel_addr[i]];
            }
            if (i < static_cast<int>(this->mujoco_joint_side_torques.size()))
            {
                state->motor_state.tau_est[i] = this->mujoco_joint_side_torques[i];
            }
            else if (this->mj_model && mj_data->sensordata && i + 2 * num_dofs < this->mj_model->nsensordata)
            {
                state->motor_state.tau_est[i] = mj_data->sensordata[i + 2 * num_dofs];
            }
        }
    }
}

std::vector<float> RL_Sim::ReadMujocoSensor3(int sensor_adr, int sensor_dim) const
{
    if (!this->mj_model || !this->mj_data || !this->mj_data->sensordata ||
        sensor_adr < 0 || sensor_dim < 3 || sensor_adr + 2 >= this->mj_model->nsensordata)
    {
        return {};
    }

    return {
        static_cast<float>(this->mj_data->sensordata[sensor_adr + 0]),
        static_cast<float>(this->mj_data->sensordata[sensor_adr + 1]),
        static_cast<float>(this->mj_data->sensordata[sensor_adr + 2]),
    };
}

std::vector<float> RL_Sim::GetMujocoProjectedGravity() const
{
    if (!this->mj_model || !this->mj_data)
    {
        return {0.0f, 0.0f, -1.0f};
    }

    const mjtNum *rot = nullptr;
    if (this->mujoco_imu_site_id >= 0)
    {
        rot = this->mj_data->site_xmat + 9 * this->mujoco_imu_site_id;
    }
    else if (this->mujoco_base_body_id >= 0)
    {
        rot = this->mj_data->xmat + 9 * this->mujoco_base_body_id;
    }
    if (!rot)
    {
        return {0.0f, 0.0f, -1.0f};
    }

    return {
        static_cast<float>(-rot[6]),
        static_cast<float>(-rot[7]),
        static_cast<float>(-rot[8]),
    };
}

void RL_Sim::SetCommand(const RobotCommand<float> *command)
{
    if (mj_data)
    {
        auto joint_mapping = this->params.Get<std::vector<int>>("joint_mapping");
        auto torque_limits = this->params.Get<std::vector<float>>("torque_limits");
        int num_dofs = this->params.Get<int>("num_of_dofs");
        const bool nxbx_torque_speed_envelope =
            this->robot_name == "nxbx" &&
            this->params.Has("torque_speed_envelope") &&
            this->params.Get<bool>("torque_speed_envelope");
        const std::vector<float> nxbx_tn_speed_rpm = nxbx_torque_speed_envelope
            ? this->params.Get<std::vector<float>>("motor_tn_speed_rpm")
            : std::vector<float>{};
        const std::vector<float> nxbx_tn_torque_nm = nxbx_torque_speed_envelope
            ? this->params.Get<std::vector<float>>("motor_tn_torque_nm")
            : std::vector<float>{};
        const float nxbx_braking_fraction = nxbx_torque_speed_envelope
            ? this->params.Get<float>("motor_braking_torque_fraction")
            : 1.0f;
        if (this->joint_qpos_addr.empty())
        {
            this->CacheMujocoAddresses();
        }
        std::vector<float> clipped_torques(static_cast<size_t>(num_dofs), 0.0f);
        for (int i = 0; i < num_dofs; ++i)
        {
            const float q = (i < static_cast<int>(this->joint_qpos_addr.size()) && this->joint_qpos_addr[i] >= 0)
                ? static_cast<float>(mj_data->qpos[this->joint_qpos_addr[i]])
                : static_cast<float>(mj_data->sensordata[joint_mapping[i]]);
            const float dq = (i < static_cast<int>(this->joint_qvel_addr.size()) && this->joint_qvel_addr[i] >= 0)
                ? static_cast<float>(mj_data->qvel[this->joint_qvel_addr[i]])
                : static_cast<float>(mj_data->sensordata[joint_mapping[i] + num_dofs]);
            float torque =
                command->motor_command.tau[i] +
                command->motor_command.kp[i] * (command->motor_command.q[i] - q) +
                command->motor_command.kd[i] * (command->motor_command.dq[i] - dq);
            if (i < static_cast<int>(torque_limits.size()))
            {
                float torque_limit = torque_limits[i];
                if (nxbx_torque_speed_envelope)
                {
                    const bool motoring = torque * dq > 0.0f;
                    if (motoring)
                    {
                        const float tn_limit = InterpolateNxbxTnTorque(
                            dq, nxbx_tn_speed_rpm, nxbx_tn_torque_nm);
                        torque_limit = std::min(torque_limit, std::max(0.0f, tn_limit));
                    }
                    else
                    {
                        torque_limit *= nxbx_braking_fraction;
                    }
                }
                torque = clamp(torque, -torque_limit, torque_limit);
            }
            clipped_torques[static_cast<size_t>(i)] = torque;
        }

        // The identified WE6 latency is downstream of the PD controller.  Keep
        // the FIFO here, after PD evaluation and software torque clipping, so
        // delayed commands cannot be re-evaluated against a newer joint state.
        const std::vector<float> torques_to_apply =
            (this->robot_name == "dr002" && this->simulation_running)
                ? this->GetDr002DelayedTorques(clipped_torques)
                : clipped_torques;
        for (int i = 0; i < num_dofs; ++i)
        {
            mj_data->ctrl[joint_mapping[i]] = torques_to_apply[static_cast<size_t>(i)];
        }
    }
}

void RL_Sim::ResetMujocoToInitialPose(bool clear_command, bool request_passive, const std::string& reason)
{
    if (!this->mj_model || !this->mj_data)
    {
        return;
    }

    bool use_getup_pose = false;
    bool use_getup_difficulty = false;
    double getup_difficulty = 0.0;
    bool use_balance_difficulty = false;
    double balance_difficulty = 0.0;
    double balance_direction = 1.0;
    double balance_pitch_progress = 0.0;
    double balance_rate_progress = 0.0;
    bool use_getup_stage = false;
    std::string getup_stage;
    std::string getup_reset_category;
    if (this->robot_name == "dr002")
    {
        if (const char *start_pose_env = std::getenv("RL_SAR_DR002_START_POSE"))
        {
            const std::string start_pose(start_pose_env);
            use_getup_pose = start_pose == "getup";
            if (!(start_pose == "upright" || start_pose == "getup"))
            {
                std::cout << LOGGER::WARNING
                          << "[DR002] Unknown RL_SAR_DR002_START_POSE='" << start_pose
                          << "'; falling back to upright." << std::endl;
            }
        }
        if (const char *difficulty_env = std::getenv("RL_SAR_DR002_GETUP_DIFFICULTY"))
        {
            const std::string value(difficulty_env);
            size_t parsed = 0;
            try
            {
                getup_difficulty = std::stod(value, &parsed);
            }
            catch (const std::exception&)
            {
                throw std::runtime_error(
                    "RL_SAR_DR002_GETUP_DIFFICULTY must be a number in [0, 1]");
            }
            if (parsed != value.size() || !std::isfinite(getup_difficulty) ||
                getup_difficulty < 0.0 || getup_difficulty > 1.0)
            {
                throw std::runtime_error(
                    "RL_SAR_DR002_GETUP_DIFFICULTY must be a number in [0, 1]");
            }
            use_getup_difficulty = true;
        }
        if (const char *difficulty_env = std::getenv("RL_SAR_DR002_BALANCE_DIFFICULTY"))
        {
            const std::string value(difficulty_env);
            size_t parsed = 0;
            try
            {
                balance_difficulty = std::stod(value, &parsed);
            }
            catch (const std::exception&)
            {
                throw std::runtime_error(
                    "RL_SAR_DR002_BALANCE_DIFFICULTY must be a number in [0, 1]");
            }
            if (parsed != value.size() || !std::isfinite(balance_difficulty) ||
                balance_difficulty < 0.0 || balance_difficulty > 1.0)
            {
                throw std::runtime_error(
                    "RL_SAR_DR002_BALANCE_DIFFICULTY must be a number in [0, 1]");
            }
            if (use_getup_difficulty)
            {
                throw std::runtime_error(
                    "Getup and balance difficulty cannot be enabled together");
            }
            use_balance_difficulty = true;
            balance_direction = (std::rand() % 2 == 0) ? -1.0 : 1.0;
            use_getup_pose = false;
        }
        if (const char *stage_env = std::getenv("RL_SAR_DR002_GETUP_STAGE"))
        {
            getup_stage = stage_env;
            if (!(getup_stage == "home_to_getup" || getup_stage == "exact_getup" ||
                  getup_stage == "getup_with_home" || getup_stage == "balance" ||
                  getup_stage == "mixed"))
            {
                throw std::runtime_error(
                    "RL_SAR_DR002_GETUP_STAGE must be home_to_getup, exact_getup, "
                    "getup_with_home, balance, or mixed");
            }
            if (use_balance_difficulty)
            {
                throw std::runtime_error(
                    "Getup stage and standalone balance difficulty cannot be enabled together");
            }

            double stage_difficulty = 1.0;
            if (const char *stage_difficulty_env =
                    std::getenv("RL_SAR_DR002_GETUP_STAGE_DIFFICULTY"))
            {
                const std::string value(stage_difficulty_env);
                size_t parsed = 0;
                try
                {
                    stage_difficulty = std::stod(value, &parsed);
                }
                catch (const std::exception&)
                {
                    throw std::runtime_error(
                        "RL_SAR_DR002_GETUP_STAGE_DIFFICULTY must be a number in [0, 1]");
                }
                if (parsed != value.size() || !std::isfinite(stage_difficulty) ||
                    stage_difficulty < 0.0 || stage_difficulty > 1.0)
                {
                    throw std::runtime_error(
                        "RL_SAR_DR002_GETUP_STAGE_DIFFICULTY must be a number in [0, 1]");
                }
            }

            std::uniform_real_distribution<double> uniform01(0.0, 1.0);
            const auto draw01 = [&]() { return uniform01(this->dr002_getup_replay_rng); };
            const auto sample_path_progress = [&](double difficulty) {
                if (difficulty <= 0.0)
                {
                    return 0.0;
                }
                const bool replay = draw01() < 0.20;
                double progress = replay
                    ? draw01() * difficulty
                    : std::max(0.0, difficulty - 0.05) +
                          draw01() * std::min(0.05, difficulty);
                if (difficulty >= 1.0 && draw01() < 0.10)
                {
                    progress = 1.0;
                }
                return progress;
            };
            // Training selects the nearest entry from the 0.01-spaced
            // Home-to-Getup pose bank rather than interpolating between rows.
            const auto nearest_path_pose = [](double progress) {
                return std::clamp(std::round(progress * 100.0) / 100.0, 0.0, 1.0);
            };

            use_getup_stage = true;
            use_getup_pose = false;
            use_getup_difficulty = false;
            if (getup_stage == "home_to_getup")
            {
                getup_reset_category = "home_to_getup";
                getup_difficulty = nearest_path_pose(sample_path_progress(stage_difficulty));
                use_getup_difficulty = true;
            }
            else
            {
                const double draw = draw01();
                double exact_cutoff = 0.80;
                double home_cutoff = 0.80;
                double balance_cutoff = 0.80;
                if (getup_stage == "getup_with_home")
                {
                    exact_cutoff = 0.60;
                    home_cutoff = 0.90;
                    balance_cutoff = 0.90;
                }
                else if (getup_stage == "balance")
                {
                    exact_cutoff = 0.40;
                    home_cutoff = 0.60;
                    balance_cutoff = 0.90;
                }
                else if (getup_stage == "mixed")
                {
                    exact_cutoff = 0.50;
                    home_cutoff = 0.70;
                    balance_cutoff = 0.90;
                }

                if (draw < exact_cutoff)
                {
                    getup_reset_category = "exact_getup";
                    use_getup_pose = true;
                }
                else if (draw < home_cutoff)
                {
                    getup_reset_category = "home";
                }
                else if (draw < balance_cutoff)
                {
                    getup_reset_category = "balance_recovery";
                    balance_difficulty = sample_path_progress(stage_difficulty);
                    const double low = std::max(0.0, balance_difficulty - 0.05);
                    balance_pitch_progress = low + draw01() * (balance_difficulty - low);
                    balance_rate_progress = low + draw01() * (balance_difficulty - low);
                    balance_direction = draw01() < 0.5 ? -1.0 : 1.0;
                    use_balance_difficulty = true;
                }
                else
                {
                    getup_reset_category = "home_to_getup";
                    getup_difficulty = nearest_path_pose(draw01());
                    use_getup_difficulty = true;
                }
            }
        }
    }
    const std::string reset_dof_key =
        use_getup_pose && this->params.Has("getup_initial_dof_pos")
            ? "getup_initial_dof_pos"
            : (this->params.Has("initial_dof_pos") ? "initial_dof_pos" : "default_dof_pos");
    auto reset_pos = this->params.Get<std::vector<float>>(reset_dof_key);
    if (use_getup_difficulty)
    {
        const auto easy_pos = this->params.Get<std::vector<float>>(
            this->params.Has("initial_dof_pos") ? "initial_dof_pos" : "default_dof_pos");
        const auto hard_pos = this->params.Get<std::vector<float>>("getup_initial_dof_pos");
        if (easy_pos.size() != hard_pos.size())
        {
            throw std::runtime_error("WE11 upright/getup DOF vectors have different sizes");
        }
        reset_pos.resize(easy_pos.size());
        for (size_t i = 0; i < easy_pos.size(); ++i)
        {
            reset_pos[i] = static_cast<float>(
                easy_pos[i] + getup_difficulty * (hard_pos[i] - easy_pos[i]));
        }
    }
    if (this->robot_name == "dr002")
    {
        if (use_getup_stage)
        {
            std::cout << LOGGER::INFO << "[DR002] Training stage=" << getup_stage
                      << ", sampled reset=" << getup_reset_category;
            if (use_getup_difficulty)
            {
                std::cout << ", path progress=" << getup_difficulty;
            }
            else if (use_balance_difficulty)
            {
                std::cout << ", disturbance progress=" << balance_difficulty
                          << ", direction="
                          << (balance_direction > 0.0 ? "forward" : "backward");
            }
            std::cout << std::endl;
        }
        else if (use_getup_difficulty)
        {
            std::cout << LOGGER::INFO << "[DR002] Reset start pose: curriculum difficulty="
                      << getup_difficulty << std::endl;
        }
        else if (use_balance_difficulty)
        {
            std::cout << LOGGER::INFO << "[DR002] Reset start pose: balance difficulty="
                      << balance_difficulty << ", direction="
                      << (balance_direction > 0.0 ? "forward" : "backward") << std::endl;
        }
        else
        {
            std::cout << LOGGER::INFO << "[DR002] Reset start pose: "
                      << (use_getup_pose ? "getup" : "upright") << std::endl;
        }
    }
    auto joint_names = this->params.Get<std::vector<std::string>>("joint_names");

    mj_resetData(this->mj_model, this->mj_data);
    if (this->root_qpos_addr >= 0)
    {
        const std::string base_z_key =
            use_getup_pose && this->params.Has("getup_initial_base_z")
                ? "getup_initial_base_z"
                : "initial_base_z";
        if (use_getup_difficulty)
        {
            const float easy_z = this->params.Get<float>("initial_base_z");
            const float hard_z = this->params.Get<float>("getup_initial_base_z");
            if (getup_difficulty <= 1.0e-12)
            {
                this->mj_data->qpos[this->root_qpos_addr + 2] = easy_z;
            }
            else if (getup_difficulty >= 1.0 - 1.0e-12)
            {
                this->mj_data->qpos[this->root_qpos_addr + 2] = hard_z;
            }
            else
            {
                // Recomputed from wheel collision geometry after joints are set.
                this->mj_data->qpos[this->root_qpos_addr + 2] = 0.0;
            }
        }
        else if (this->params.Has(base_z_key))
        {
            std::string scene_base_z_key = "initial_base_z_" + this->scene_name;
            float initial_base_z = !use_getup_pose && this->params.Has(scene_base_z_key)
                ? this->params.Get<float>(scene_base_z_key)
                : this->params.Get<float>(base_z_key);
            this->mj_data->qpos[this->root_qpos_addr + 2] = initial_base_z;
        }

        const std::string scene_base_quat_key = "initial_base_quat_" + this->scene_name;
        const std::string base_quat_key =
            use_getup_pose && this->params.Has("getup_initial_base_quat")
                ? "getup_initial_base_quat"
                : "initial_base_quat";
        if (use_getup_difficulty)
        {
            auto qa = this->params.Get<std::vector<float>>("initial_base_quat");
            auto qb = this->params.Get<std::vector<float>>("getup_initial_base_quat");
            if (qa.size() < 4 || qb.size() < 4)
            {
                throw std::runtime_error("WE11 upright/getup base quaternions must have 4 values");
            }
            double norm_a = 0.0;
            double norm_b = 0.0;
            double dot = 0.0;
            for (int i = 0; i < 4; ++i)
            {
                norm_a += static_cast<double>(qa[i]) * qa[i];
                norm_b += static_cast<double>(qb[i]) * qb[i];
            }
            norm_a = std::sqrt(norm_a);
            norm_b = std::sqrt(norm_b);
            if (norm_a <= 1.0e-12 || norm_b <= 1.0e-12)
            {
                throw std::runtime_error("WE11 upright/getup base quaternion is zero length");
            }
            for (int i = 0; i < 4; ++i)
            {
                qa[i] = static_cast<float>(qa[i] / norm_a);
                qb[i] = static_cast<float>(qb[i] / norm_b);
                dot += static_cast<double>(qa[i]) * qb[i];
            }
            if (dot < 0.0)
            {
                dot = -dot;
                for (int i = 0; i < 4; ++i)
                {
                    qb[i] = -qb[i];
                }
            }
            dot = std::clamp(dot, -1.0, 1.0);
            double scale_a = 1.0 - getup_difficulty;
            double scale_b = getup_difficulty;
            if (dot <= 0.9995)
            {
                const double theta = std::acos(dot);
                const double denominator = std::sin(theta);
                scale_a = std::sin((1.0 - getup_difficulty) * theta) / denominator;
                scale_b = std::sin(getup_difficulty * theta) / denominator;
            }
            double interpolated[4] = {};
            double norm = 0.0;
            for (int i = 0; i < 4; ++i)
            {
                interpolated[i] = scale_a * qa[i] + scale_b * qb[i];
                norm += interpolated[i] * interpolated[i];
            }
            norm = std::sqrt(norm);
            for (int i = 0; i < 4; ++i)
            {
                this->mj_data->qpos[this->root_qpos_addr + 3 + i] =
                    static_cast<mjtNum>(interpolated[i] / norm);
            }
        }
        else if (this->params.Has(base_quat_key) ||
            (!use_getup_pose && this->params.Has(scene_base_quat_key)))
        {
            auto initial_base_quat = !use_getup_pose && this->params.Has(scene_base_quat_key)
                ? this->params.Get<std::vector<float>>(scene_base_quat_key)
                : this->params.Get<std::vector<float>>(base_quat_key);
            if (initial_base_quat.size() >= 4)
            {
                const double norm = std::sqrt(
                    static_cast<double>(initial_base_quat[0]) * initial_base_quat[0] +
                    static_cast<double>(initial_base_quat[1]) * initial_base_quat[1] +
                    static_cast<double>(initial_base_quat[2]) * initial_base_quat[2] +
                    static_cast<double>(initial_base_quat[3]) * initial_base_quat[3]);
                if (norm > 1e-6)
                {
                    for (int i = 0; i < 4; ++i)
                    {
                        this->mj_data->qpos[this->root_qpos_addr + 3 + i] =
                            static_cast<mjtNum>(initial_base_quat[i] / norm);
                    }
                }
                else
                {
                    std::cout << LOGGER::WARNING << "[MuJoCo] Ignoring zero-length initial_base_quat" << std::endl;
                }
            }
        }
    }
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        if (i >= static_cast<int>(joint_names.size()) || i >= static_cast<int>(reset_pos.size()))
        {
            continue;
        }
        int joint_id = mj_name2id(this->mj_model, mjOBJ_JOINT, joint_names[i].c_str());
        if (joint_id < 0)
        {
            continue;
        }
        int qpos_adr = this->mj_model->jnt_qposadr[joint_id];
        int qvel_adr = this->mj_model->jnt_dofadr[joint_id];
        this->mj_data->qpos[qpos_adr] = reset_pos[i];
        this->mj_data->qvel[qvel_adr] = 0.0;
    }
    if (use_balance_difficulty && this->root_qpos_addr >= 0 && this->root_qvel_addr >= 0)
    {
        constexpr double pi = 3.14159265358979323846;
        const double max_pitch = (use_getup_stage ? 16.25 : 25.0) * pi / 180.0;
        const double max_pitch_rate = use_getup_stage ? 0.13 : 1.2;
        constexpr double pivot_x = 0.02543588;
        constexpr double pivot_z = -0.24231853;
        const double pitch_progress = use_getup_stage
            ? balance_pitch_progress
            : balance_difficulty;
        const double rate_progress = use_getup_stage
            ? balance_rate_progress
            : balance_difficulty;
        const double pitch = balance_direction * pitch_progress * max_pitch;
        const double half = 0.5 * pitch;
        const double pitch_quat[4] = {std::cos(half), 0.0, std::sin(half), 0.0};
        const double current_quat[4] = {
            this->mj_data->qpos[this->root_qpos_addr + 3],
            this->mj_data->qpos[this->root_qpos_addr + 4],
            this->mj_data->qpos[this->root_qpos_addr + 5],
            this->mj_data->qpos[this->root_qpos_addr + 6],
        };
        this->mj_data->qpos[this->root_qpos_addr + 3] =
            pitch_quat[0] * current_quat[0] - pitch_quat[2] * current_quat[2];
        this->mj_data->qpos[this->root_qpos_addr + 4] =
            pitch_quat[0] * current_quat[1] + pitch_quat[2] * current_quat[3];
        this->mj_data->qpos[this->root_qpos_addr + 5] =
            pitch_quat[0] * current_quat[2] + pitch_quat[2] * current_quat[0];
        this->mj_data->qpos[this->root_qpos_addr + 6] =
            pitch_quat[0] * current_quat[3] - pitch_quat[2] * current_quat[1];

        // Rotate the base about the home wheel-axis midpoint so wheel height
        // and the configured 15 mm home clearance remain unchanged.
        const double c = std::cos(pitch);
        const double s = std::sin(pitch);
        const double root_from_pivot_x = -pivot_x;
        const double root_from_pivot_z = -pivot_z;
        this->mj_data->qpos[this->root_qpos_addr + 0] +=
            pivot_x + c * root_from_pivot_x + s * root_from_pivot_z;
        this->mj_data->qpos[this->root_qpos_addr + 2] +=
            pivot_z - s * root_from_pivot_x + c * root_from_pivot_z;
        this->mj_data->qvel[this->root_qvel_addr + 4] =
            balance_direction * rate_progress * max_pitch_rate;

        std::cout << LOGGER::INFO << "[DR002] Balance reset pitch="
                  << pitch * 180.0 / pi << " deg, pitch_rate="
                  << this->mj_data->qvel[this->root_qvel_addr + 4] << " rad/s" << std::endl;
    }
    if (use_getup_difficulty && getup_difficulty > 1.0e-12 &&
        getup_difficulty < 1.0 - 1.0e-12 && this->root_qpos_addr >= 0)
    {
        // Match the Getup training reset: place the lowest wheel collision
        // surface on the floor, with clearance fading from 15 mm at D=0.
        this->mj_data->qpos[this->root_qpos_addr + 2] = 0.0;
        mj_kinematics(this->mj_model, this->mj_data);
        double lowest_bottom = std::numeric_limits<double>::infinity();
        for (const char *geom_name : {"left_foot_collision", "right_foot_collision"})
        {
            const int geom_id = mj_name2id(this->mj_model, mjOBJ_GEOM, geom_name);
            if (geom_id < 0)
            {
                throw std::runtime_error(std::string("Missing WE11 wheel geom: ") + geom_name);
            }
            const double radius = this->mj_model->geom_size[3 * geom_id];
            const double half_length = this->mj_model->geom_size[3 * geom_id + 1];
            const double axis_z = this->mj_data->geom_xmat[9 * geom_id + 8];
            const double radial_z = radius * std::sqrt(std::max(0.0, 1.0 - axis_z * axis_z));
            const double axial_z = half_length * std::abs(axis_z);
            const double bottom = this->mj_data->geom_xpos[3 * geom_id + 2] - radial_z - axial_z;
            lowest_bottom = std::min(lowest_bottom, bottom);
        }
        const double easy_clearance = this->params.Has("getup_ground_clearance_easy")
            ? this->params.Get<float>("getup_ground_clearance_easy")
            : 0.015;
        this->mj_data->qpos[this->root_qpos_addr + 2] =
            -lowest_bottom + easy_clearance * (1.0 - getup_difficulty);
    }
    if (use_getup_stage)
    {
        // The Getup task independently randomizes both wings on every reset.
        std::uniform_real_distribution<double> wing_angle(-0.5 * 3.14159265358979323846, 0.0);
        for (int i = 0; i < 2; ++i)
        {
            const int qpos_addr = this->wing_qpos_addr_[i];
            if (qpos_addr >= 0)
            {
                this->mj_data->qpos[qpos_addr] = wing_angle(this->dr002_getup_replay_rng);
            }
        }
    }
    this->dr002_sync_step_count = 0;
    this->nxbx_sync_step_count = 0;
    this->nxbx_history_initialized = false;
    this->nxbx_command_x_limiter_initialized = true;
    this->nxbx_limited_command_x = 0.0f;
    // Zero wing motor ctrl slots so the reset pose does not carry a stale
    // velocity command until the next TickWingCommand() write.
    if (this->robot_name == "dr002")
    {
        for (int i = 0; i < 2; ++i)
        {
            const int ctrl_idx = this->wing_actuator_ctrl_idx_[i];
            if (this->mj_data->ctrl &&
                ctrl_idx >= 0 && ctrl_idx < this->mj_model->nu)
            {
                this->mj_data->ctrl[ctrl_idx] = 0.0;
            }
        }
    }
    mj_forward(this->mj_model, this->mj_data);

    std::vector<float> stale_output;
    while (this->output_dof_pos_queue.try_pop(stale_output)) {}
    while (this->output_dof_vel_queue.try_pop(stale_output)) {}
    while (this->output_dof_tau_queue.try_pop(stale_output)) {}

    this->rl_init_done = false;
    this->episode_length_buf = 0;
    this->dr002_startup_hold_active = false;
    this->dr002_startup_hold_steps = 0;
    this->dr002_command_x_limiter_initialized = false;
    this->dr002_limited_command_x = 0.0f;
    this->dr002_next_command_x_sample_step = 0;
    this->dr002_motor_control_substep_index = 0;
    this->InitOutputs();
    if (clear_command)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
    }
    if (this->robot_name == "dr002")
    {
        this->dr002_height_command_ = std::clamp(
            this->dr002_height_command_initial_,
            this->dr002_height_command_min_,
            this->dr002_height_command_max_);
    }

    auto kp = this->params.Get<std::vector<float>>("fixed_kp");
    auto kd = this->params.Get<std::vector<float>>("fixed_kd");
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->robot_command.motor_command.q[i] = reset_pos[i];
        this->robot_command.motor_command.dq[i] = 0.0f;
        this->robot_command.motor_command.kp[i] = kp[i];
        this->robot_command.motor_command.kd[i] = kd[i];
        this->robot_command.motor_command.tau[i] = 0.0f;
    }
    // The selected policy YAML is already preloaded for DR002, but delay values
    // are sampled/reloaded at policy start. Pose reset only clears the buffers.
    this->ResetDr002ActionDelayBuffer(false);
    this->ResetDr002TorqueDelayBuffer(false);
    this->ResetNxbxTargetDelayBuffer();
    this->SetCommand(&this->robot_command);
    this->GetState(&this->robot_state);

    if (request_passive)
    {
        this->fsm.RequestStateChange("RLFSMStatePassive");
    }
    std::cout << std::endl << LOGGER::INFO << reason << ": reset MuJoCo to initial pose" << std::endl;
}

void RL_Sim::InitDr002MotorControl()
{
    this->dr002_motor_control_decimation = 1;
    this->dr002_motor_control_substep_index = 0;
    if (this->robot_name != "dr002" || !this->mj_model)
    {
        return;
    }

    const double physics_dt = this->mj_model->opt.timestep;
    if (!std::isfinite(physics_dt) || physics_dt <= 0.0)
    {
        throw std::runtime_error("DR002 motor control requires a finite positive MuJoCo timestep");
    }

    const double physics_hz = 1.0 / physics_dt;
    const double motor_control_hz = this->params.Has("motor_control_hz")
        ? static_cast<double>(this->params.Get<float>("motor_control_hz"))
        : physics_hz;
    if (!std::isfinite(motor_control_hz) || motor_control_hz <= 0.0)
    {
        throw std::runtime_error("DR002 motor_control_hz must be finite and positive");
    }

    const double exact_decimation = physics_hz / motor_control_hz;
    const int rounded_decimation = static_cast<int>(std::lround(exact_decimation));
    if (rounded_decimation < 1 ||
        std::fabs(exact_decimation - static_cast<double>(rounded_decimation)) > 1.0e-9)
    {
        throw std::runtime_error(
            "DR002 physics frequency must be an integer multiple of motor_control_hz");
    }

    if (this->params.Has("motor_control_hz") && this->params.Has("dt"))
    {
        const double configured_dt = static_cast<double>(this->params.Get<float>("dt"));
        if (!std::isfinite(configured_dt) ||
            std::fabs(configured_dt - physics_dt) > 1.0e-9)
        {
            throw std::runtime_error(
                "DR002 configured dt must match the MuJoCo timestep when motor_control_hz is set");
        }
    }

    this->dr002_motor_control_decimation = rounded_decimation;
    const int policy_decimation = this->params.Get<int>("decimation");
    if (policy_decimation <= 0 || policy_decimation % rounded_decimation != 0)
    {
        throw std::runtime_error(
            "DR002 policy decimation must be a positive multiple of motor-control decimation");
    }
    std::cout << LOGGER::INFO
              << "[DR002] motor control: " << motor_control_hz << " Hz, every "
              << rounded_decimation << " physics step(s), torque ZOH="
              << physics_dt * static_cast<double>(rounded_decimation) * 1000.0
              << " ms" << std::endl;
}

bool RL_Sim::IsDr002MotorControlUpdateDue() const
{
    return this->dr002_motor_control_substep_index == 0;
}

void RL_Sim::AdvanceDr002MotorControl()
{
    const int decimation = std::max(this->dr002_motor_control_decimation, 1);
    this->dr002_motor_control_substep_index =
        (this->dr002_motor_control_substep_index + 1) % decimation;
}

int RL_Sim::SampleDr002ActionDelaySteps()
{
    if (this->robot_name != "dr002")
    {
        this->dr002_action_delay_steps = 0;
        this->dr002_action_delay_steps_by_joint.clear();
        return 0;
    }

    const int num_dofs = this->params.Has("num_of_dofs")
        ? this->params.Get<int>("num_of_dofs")
        : 0;
    if (this->params.Has("action_delay_steps_by_joint"))
    {
        const auto delay_steps =
            this->params.Get<std::vector<int>>("action_delay_steps_by_joint");
        if (num_dofs <= 0 || static_cast<int>(delay_steps.size()) != num_dofs)
        {
            throw std::runtime_error(
                "DR002 action_delay_steps_by_joint must contain one entry per DOF");
        }
        if (std::any_of(delay_steps.begin(), delay_steps.end(),
                        [](int value) { return value < 0; }))
        {
            throw std::runtime_error(
                "DR002 action_delay_steps_by_joint must contain non-negative integers");
        }
        this->dr002_action_delay_steps_by_joint = delay_steps;
        this->dr002_action_delay_steps =
            *std::max_element(delay_steps.begin(), delay_steps.end());
        return this->dr002_action_delay_steps;
    }

    if (this->params.Has("action_delay_min_steps") ||
        this->params.Has("action_delay_max_steps"))
    {
        const int min_steps = this->params.Has("action_delay_min_steps")
            ? this->params.Get<int>("action_delay_min_steps")
            : 0;
        const int max_steps = this->params.Has("action_delay_max_steps")
            ? this->params.Get<int>("action_delay_max_steps")
            : min_steps;
        if (min_steps < 0 || max_steps < min_steps)
        {
            throw std::runtime_error(
                "DR002 action delay requires 0 <= action_delay_min_steps <= action_delay_max_steps");
        }
        std::uniform_int_distribution<int> distribution(min_steps, max_steps);
        this->dr002_action_delay_steps = distribution(this->dr002_command_rng);
        this->dr002_action_delay_steps_by_joint.assign(
            static_cast<size_t>(std::max(num_dofs, 0)),
            this->dr002_action_delay_steps);
        return this->dr002_action_delay_steps;
    }

    const int delay_steps = this->params.Has("action_delay_steps")
        ? this->params.Get<int>("action_delay_steps")
        : 0;
    this->dr002_action_delay_steps = std::max(delay_steps, 0);
    this->dr002_action_delay_steps_by_joint.assign(
        static_cast<size_t>(std::max(num_dofs, 0)),
        this->dr002_action_delay_steps);
    return this->dr002_action_delay_steps;
}

int RL_Sim::GetDr002ActionDelaySteps()
{
    return this->robot_name == "dr002" ? this->dr002_action_delay_steps : 0;
}

void RL_Sim::ResetDr002ActionDelayBuffer(bool resample_delay)
{
    this->dr002_delay_q.clear();
    this->dr002_delay_dq.clear();
    this->dr002_delay_tau.clear();

    const int delay_steps = resample_delay
        ? this->SampleDr002ActionDelaySteps()
        : this->GetDr002ActionDelaySteps();
    if (delay_steps <= 0 || !this->params.Has("num_of_dofs"))
    {
        return;
    }

    if (resample_delay)
    {
        const double physics_dt = this->mj_model ? this->mj_model->opt.timestep : 0.0;
        const double motor_dt =
            physics_dt * static_cast<double>(std::max(this->dr002_motor_control_decimation, 1));
        std::cout << LOGGER::INFO << "[DR002] command delay by joint: [";
        for (size_t i = 0; i < this->dr002_action_delay_steps_by_joint.size(); ++i)
        {
            if (i > 0) std::cout << ", ";
            const int joint_delay = this->dr002_action_delay_steps_by_joint[i];
            std::cout << joint_delay << " ("
                      << joint_delay * motor_dt * 1000.0 << " ms)";
        }
        std::cout << "]" << std::endl;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    std::vector<float> hold_q = this->robot_command.motor_command.q;
    if (static_cast<int>(hold_q.size()) != num_dofs)
    {
        hold_q = this->params.Has("initial_dof_pos")
            ? this->params.Get<std::vector<float>>("initial_dof_pos")
            : this->params.Get<std::vector<float>>("default_dof_pos");
    }
    hold_q.resize(num_dofs, 0.0f);

    const std::vector<float> hold_dq(num_dofs, 0.0f);
    const std::vector<float> hold_tau(num_dofs, 0.0f);
    for (int i = 0; i <= delay_steps; ++i)
    {
        this->dr002_delay_q.push_back(hold_q);
        this->dr002_delay_dq.push_back(hold_dq);
        this->dr002_delay_tau.push_back(hold_tau);
    }
}

const RobotCommand<float>* RL_Sim::GetDr002DelayedCommand(const RobotCommand<float> *command)
{
    const int delay_steps = this->GetDr002ActionDelaySteps();
    if (this->robot_name != "dr002" || delay_steps <= 0 || command == nullptr)
    {
        return command;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    if (static_cast<int>(this->dr002_action_delay_steps_by_joint.size()) != num_dofs)
    {
        this->dr002_action_delay_steps_by_joint.assign(
            static_cast<size_t>(num_dofs), delay_steps);
    }
    const size_t target_size = static_cast<size_t>(delay_steps + 1);
    if (this->dr002_delay_q.size() != target_size ||
        this->dr002_delay_dq.size() != target_size ||
        this->dr002_delay_tau.size() != target_size)
    {
        this->ResetDr002ActionDelayBuffer(false);
    }

    std::vector<float> q = command->motor_command.q;
    std::vector<float> dq = command->motor_command.dq;
    std::vector<float> tau = command->motor_command.tau;
    q.resize(num_dofs, 0.0f);
    dq.resize(num_dofs, 0.0f);
    tau.resize(num_dofs, 0.0f);

    this->dr002_delay_q.push_front(q);
    this->dr002_delay_dq.push_front(dq);
    this->dr002_delay_tau.push_front(tau);
    while (this->dr002_delay_q.size() > target_size) this->dr002_delay_q.pop_back();
    while (this->dr002_delay_dq.size() > target_size) this->dr002_delay_dq.pop_back();
    while (this->dr002_delay_tau.size() > target_size) this->dr002_delay_tau.pop_back();

    this->dr002_delayed_robot_command = *command;
    this->dr002_delayed_robot_command.motor_command.q = q;
    this->dr002_delayed_robot_command.motor_command.dq = dq;
    this->dr002_delayed_robot_command.motor_command.tau = tau;
    for (int i = 0; i < num_dofs; ++i)
    {
        const size_t joint_delay = static_cast<size_t>(std::clamp(
            this->dr002_action_delay_steps_by_joint[static_cast<size_t>(i)],
            0,
            delay_steps));
        this->dr002_delayed_robot_command.motor_command.q[static_cast<size_t>(i)] =
            this->dr002_delay_q[joint_delay][static_cast<size_t>(i)];
        this->dr002_delayed_robot_command.motor_command.dq[static_cast<size_t>(i)] =
            this->dr002_delay_dq[joint_delay][static_cast<size_t>(i)];
        this->dr002_delayed_robot_command.motor_command.tau[static_cast<size_t>(i)] =
            this->dr002_delay_tau[joint_delay][static_cast<size_t>(i)];
    }
    return &this->dr002_delayed_robot_command;
}

int RL_Sim::LoadDr002TorqueDelaySteps()
{
    if (this->robot_name != "dr002")
    {
        this->dr002_torque_delay_steps = 0;
        this->dr002_torque_delay_steps_by_joint.clear();
        return 0;
    }

    const int num_dofs = this->params.Has("num_of_dofs")
        ? this->params.Get<int>("num_of_dofs")
        : 0;
    if (this->params.Has("torque_delay_steps_by_joint"))
    {
        const auto delay_steps =
            this->params.Get<std::vector<int>>("torque_delay_steps_by_joint");
        if (num_dofs <= 0 || static_cast<int>(delay_steps.size()) != num_dofs)
        {
            throw std::runtime_error(
                "DR002 torque_delay_steps_by_joint must contain one entry per DOF");
        }
        if (std::any_of(delay_steps.begin(), delay_steps.end(),
                        [](int value) { return value < 0; }))
        {
            throw std::runtime_error(
                "DR002 torque_delay_steps_by_joint must contain non-negative integers");
        }
        this->dr002_torque_delay_steps_by_joint = delay_steps;
        this->dr002_torque_delay_steps =
            *std::max_element(delay_steps.begin(), delay_steps.end());
        return this->dr002_torque_delay_steps;
    }

    const int delay_steps = this->params.Has("torque_delay_steps")
        ? this->params.Get<int>("torque_delay_steps")
        : 0;
    if (delay_steps < 0)
    {
        throw std::runtime_error("DR002 torque_delay_steps must be non-negative");
    }
    this->dr002_torque_delay_steps = delay_steps;
    this->dr002_torque_delay_steps_by_joint.assign(
        static_cast<size_t>(std::max(num_dofs, 0)), delay_steps);
    return this->dr002_torque_delay_steps;
}

void RL_Sim::ResetDr002TorqueDelayBuffer(bool reload_config)
{
    this->dr002_torque_delay_buffer.clear();
    const int delay_steps = reload_config
        ? this->LoadDr002TorqueDelaySteps()
        : this->dr002_torque_delay_steps;
    if (delay_steps <= 0 || !this->params.Has("num_of_dofs"))
    {
        return;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    if (static_cast<int>(this->dr002_torque_delay_steps_by_joint.size()) != num_dofs)
    {
        throw std::runtime_error(
            "DR002 torque delay state does not match the configured number of DOFs");
    }

    if (reload_config)
    {
        const double physics_dt = this->mj_model ? this->mj_model->opt.timestep : 0.0;
        const double motor_dt =
            physics_dt * static_cast<double>(std::max(this->dr002_motor_control_decimation, 1));
        std::cout << LOGGER::INFO
                  << "[DR002] post-controller, post-clip torque delay by joint: [";
        for (size_t i = 0; i < this->dr002_torque_delay_steps_by_joint.size(); ++i)
        {
            if (i > 0) std::cout << ", ";
            const int joint_delay = this->dr002_torque_delay_steps_by_joint[i];
            std::cout << joint_delay << " ("
                      << joint_delay * motor_dt * 1000.0 << " ms)";
        }
        std::cout << "]" << std::endl;
    }

    const std::vector<float> zero_torque(static_cast<size_t>(num_dofs), 0.0f);
    for (int i = 0; i <= delay_steps; ++i)
    {
        this->dr002_torque_delay_buffer.push_back(zero_torque);
    }
}

std::vector<float> RL_Sim::GetDr002DelayedTorques(const std::vector<float> &torques)
{
    if (this->robot_name != "dr002" || this->dr002_torque_delay_steps <= 0)
    {
        return torques;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    if (static_cast<int>(torques.size()) != num_dofs ||
        static_cast<int>(this->dr002_torque_delay_steps_by_joint.size()) != num_dofs)
    {
        throw std::runtime_error("DR002 torque delay input does not match the configured DOFs");
    }

    const size_t target_size = static_cast<size_t>(this->dr002_torque_delay_steps + 1);
    if (this->dr002_torque_delay_buffer.size() != target_size)
    {
        this->ResetDr002TorqueDelayBuffer(false);
    }

    this->dr002_torque_delay_buffer.push_front(torques);
    while (this->dr002_torque_delay_buffer.size() > target_size)
    {
        this->dr002_torque_delay_buffer.pop_back();
    }

    std::vector<float> delayed_torques = torques;
    for (int i = 0; i < num_dofs; ++i)
    {
        const size_t joint_delay = static_cast<size_t>(std::clamp(
            this->dr002_torque_delay_steps_by_joint[static_cast<size_t>(i)],
            0,
            this->dr002_torque_delay_steps));
        delayed_torques[static_cast<size_t>(i)] =
            this->dr002_torque_delay_buffer[joint_delay][static_cast<size_t>(i)];
    }
    return delayed_torques;
}

float RL_Sim::ShapeNxbxCommand(float axis, float min_nonzero, float max_value) const
{
    const float deadzone = std::clamp(
        this->params.Get<float>("command_axis_deadzone", 0.05f), 0.0f, 0.99f);
    const float clamped_axis = std::clamp(axis, -1.0f, 1.0f);
    const float magnitude = std::fabs(clamped_axis);
    if (magnitude <= deadzone)
    {
        return 0.0f;
    }

    min_nonzero = std::clamp(std::fabs(min_nonzero), 0.0f, std::fabs(max_value));
    max_value = std::fabs(max_value);
    const float normalized = (magnitude - deadzone) / (1.0f - deadzone);
    const float shaped = min_nonzero + normalized * (max_value - min_nonzero);
    return std::copysign(shaped, clamped_axis);
}

void RL_Sim::AdvanceNxbxCommandX()
{
    if (this->robot_name != "nxbx")
    {
        return;
    }

    const float target = this->ShapeNxbxCommand(
        this->control.x,
        this->params.Get<float>("command_x_min_nonzero", 0.15f),
        this->params.Get<float>("command_x_max", 5.0f));
    const float maximum_acceleration = std::fabs(
        this->params.Get<float>("command_x_accel_limit", 1.0f));
    float acceleration_limit = maximum_acceleration;
    const bool growing_same_direction =
        target * this->nxbx_limited_command_x >= 0.0f &&
        std::fabs(target) > std::fabs(this->nxbx_limited_command_x);
    if (growing_same_direction && this->params.Get<bool>("torque_speed_envelope"))
    {
        const auto torque_limits =
            this->params.Get<std::vector<float>>("torque_limits");
        if (torque_limits.empty())
        {
            throw std::runtime_error("NXBx requires at least one torque limit");
        }
        const auto minimum_torque = std::min_element(
            torque_limits.begin(),
            torque_limits.end(),
            [](float lhs, float rhs) { return std::fabs(lhs) < std::fabs(rhs); });
        const float motor_torque = std::fabs(*minimum_torque);
        const float wheel_radius = this->params.Get<float>("wheel_radius");
        if (!(motor_torque > 0.0f) || !(wheel_radius > 0.0f))
        {
            throw std::runtime_error(
                "NXBx torque limits and wheel_radius must be positive");
        }
        const auto tn_speed_rpm =
            this->params.Get<std::vector<float>>("motor_tn_speed_rpm");
        const auto tn_torque_nm =
            this->params.Get<std::vector<float>>("motor_tn_torque_nm");
        const float wheel_speed =
            std::fabs(this->nxbx_limited_command_x) / wheel_radius;
        const float tn_limit = InterpolateNxbxTnTorque(
            wheel_speed, tn_speed_rpm, tn_torque_nm);
        const float motoring_fraction = std::clamp(
            std::min(motor_torque, tn_limit) / motor_torque, 0.0f, 1.0f);
        acceleration_limit *= motoring_fraction;
    }
    const float physics_dt = this->params.Get<float>("dt", 0.0025f);
    const int decimation = std::max(1, this->params.Get<int>("decimation", 1));
    const float max_delta = acceleration_limit * physics_dt * decimation;

    if (!this->nxbx_command_x_limiter_initialized)
    {
        this->nxbx_limited_command_x = 0.0f;
        this->nxbx_command_x_limiter_initialized = true;
    }
    const float delta = std::clamp(
        target - this->nxbx_limited_command_x,
        -max_delta,
        max_delta);
    this->nxbx_limited_command_x += delta;
}

void RL_Sim::UpdateNxbxObservationState()
{
    if (this->robot_name != "nxbx")
    {
        return;
    }

    this->GetState(&this->robot_state);
    this->obs.base_quat = this->robot_state.imu.quaternion;
    this->obs.dof_pos = this->robot_state.motor_state.q;
    this->obs.dof_vel = this->robot_state.motor_state.dq;
    this->obs.gravity_vec = this->GetMujocoProjectedGravity();

    if (this->mj_model && this->mj_data && this->root_qvel_addr >= 0 &&
        this->root_qvel_addr + 5 < this->mj_model->nv)
    {
        const std::vector<float> raw_linear_velocity = {
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 0]),
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 1]),
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 2]),
        };
        this->obs.lin_vel = QuatRotateInverse(this->obs.base_quat, raw_linear_velocity);
        // GetState reads the body-aligned MJCF gyro sensor (and only falls back
        // to body-frame free-joint qvel if an external scene lacks that sensor).
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
    }
    else
    {
        this->obs.lin_vel.assign(3, 0.0f);
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
    }

    const float vx = this->nxbx_command_x_limiter_initialized
        ? this->nxbx_limited_command_x
        : 0.0f;
    const float yaw = this->ShapeNxbxCommand(
        this->control.yaw,
        this->params.Get<float>("command_yaw_min_nonzero", 0.30f),
        this->params.Get<float>("command_yaw_max", 1.0f));
    this->obs.commands = {vx, yaw};
}

void RL_Sim::ResetNxbxHistoryToCurrentState()
{
    if (this->robot_name != "nxbx" || !this->rl_init_done)
    {
        return;
    }

    this->obs.actions.assign(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->UpdateNxbxObservationState();
    const std::vector<float> frame = this->ComputeObservation();
    const int expected_frame_dim = this->params.Get<int>("observation_frame_dim", 13);
    if (static_cast<int>(frame.size()) != expected_frame_dim)
    {
        throw std::runtime_error(
            "NXBx observation frame dimension mismatch: expected " +
            std::to_string(expected_frame_dim) + ", got " + std::to_string(frame.size()));
    }

    this->history_obs_buf.reset({0}, frame);
    this->history_obs = this->history_obs_buf.get_obs_vec(
        this->params.Get<std::vector<int>>("observations_history"));
    this->nxbx_history_initialized = true;
}

bool RL_Sim::PrimeNxbxPolicyCommand()
{
    if (this->robot_name != "nxbx" || !this->rl_init_done || !this->model)
    {
        return false;
    }

    if (!this->nxbx_history_initialized)
    {
        this->ResetNxbxHistoryToCurrentState();
    }
    this->UpdateNxbxObservationState();
    this->obs.actions = this->Forward();
    this->ComputeOutput(
        this->obs.actions,
        this->output_dof_pos,
        this->output_dof_vel,
        this->output_dof_tau);

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    if (static_cast<int>(this->output_dof_pos.size()) != num_dofs ||
        static_cast<int>(this->output_dof_vel.size()) != num_dofs)
    {
        throw std::runtime_error("NXBx policy output does not match the two-wheel command contract");
    }

    const auto kp = this->params.Get<std::vector<float>>("rl_kp");
    const auto kd = this->params.Get<std::vector<float>>("rl_kd");
    for (int i = 0; i < num_dofs; ++i)
    {
        this->robot_command.motor_command.q[i] = this->output_dof_pos[i];
        this->robot_command.motor_command.dq[i] = this->output_dof_vel[i];
        this->robot_command.motor_command.kp[i] = kp[i];
        this->robot_command.motor_command.kd[i] = kd[i];
        this->robot_command.motor_command.tau[i] = 0.0f;
    }
    return true;
}

int RL_Sim::SampleNxbxTargetDelaySteps()
{
    if (this->robot_name != "nxbx")
    {
        this->nxbx_target_delay_steps = 0;
        return 0;
    }

    const int min_steps = this->params.Get<int>("target_delay_min_steps", 0);
    const int max_steps = this->params.Get<int>("target_delay_max_steps", min_steps);
    if (min_steps < 0 || max_steps < min_steps)
    {
        throw std::runtime_error(
            "NXBx target delay requires 0 <= target_delay_min_steps <= target_delay_max_steps");
    }

    std::uniform_int_distribution<int> distribution(min_steps, max_steps);
    this->nxbx_target_delay_steps = distribution(this->nxbx_delay_rng);
    std::cout << LOGGER::INFO << "[NXBx] sampled velocity-target delay: "
              << this->nxbx_target_delay_steps << " physics steps ("
              << this->nxbx_target_delay_steps * this->params.Get<float>("dt") * 1000.0f
              << " ms); retained across resets" << std::endl;
    return this->nxbx_target_delay_steps;
}

void RL_Sim::ResetNxbxTargetDelayBuffer()
{
    this->nxbx_delayed_velocity_targets.clear();
    if (this->robot_name != "nxbx")
    {
        return;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    const std::vector<float> zero_target(num_dofs, 0.0f);
    for (int i = 0; i <= this->nxbx_target_delay_steps; ++i)
    {
        this->nxbx_delayed_velocity_targets.push_back(zero_target);
    }
}

const RobotCommand<float>* RL_Sim::GetNxbxDelayedCommand(
    const RobotCommand<float> *command)
{
    if (this->robot_name != "nxbx" || this->nxbx_target_delay_steps <= 0 || !command)
    {
        return command;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    const size_t target_size = static_cast<size_t>(this->nxbx_target_delay_steps + 1);
    if (this->nxbx_delayed_velocity_targets.size() != target_size)
    {
        this->ResetNxbxTargetDelayBuffer();
    }

    std::vector<float> velocity_target = command->motor_command.dq;
    velocity_target.resize(num_dofs, 0.0f);
    this->nxbx_delayed_velocity_targets.push_front(velocity_target);
    while (this->nxbx_delayed_velocity_targets.size() > target_size)
    {
        this->nxbx_delayed_velocity_targets.pop_back();
    }

    this->nxbx_delayed_robot_command = *command;
    this->nxbx_delayed_robot_command.motor_command.dq =
        this->nxbx_delayed_velocity_targets.back();
    return &this->nxbx_delayed_robot_command;
}

void RL_Sim::ResetRLHistoryToCurrentState()
{
    if (!this->params.Has("observations") || !this->params.Has("observations_history"))
    {
        return;
    }

    this->obs.ang_vel = this->robot_state.imu.gyroscope;
    if (this->robot_name == "dr002")
    {
        const float command_x = this->dr002_command_x_limiter_initialized
            ? this->dr002_limited_command_x
            : this->control.x;
        this->obs.commands = {command_x, this->control.yaw, this->dr002_height_command_};
    }
    else
    {
        this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
    }
    this->obs.base_quat = this->robot_state.imu.quaternion;
    if (this->robot_name == "dr002")
    {
        this->obs.gravity_vec = this->GetMujocoProjectedGravity();
    }
    this->obs.dof_pos = this->robot_state.motor_state.q;
    this->obs.dof_vel = this->robot_state.motor_state.dq;
    this->obs.actions.assign(this->params.Get<int>("num_of_dofs"), 0.0f);
    this->UpdateWingObservation();

    std::vector<float> current_obs = this->ComputeObservation();
    auto observations_history = this->params.Get<std::vector<int>>("observations_history");
    if (!observations_history.empty())
    {
        this->history_obs_buf.reset({0}, current_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(observations_history);
    }
}

void RL_Sim::InitDr002RewardFigures()
{
    if (this->robot_name != "dr002" || !this->sim)
    {
        return;
    }
    std::lock_guard<std::mutex> lock(this->sim->user_figure_mutex);
    InitDr002RewardFigure(
        this->sim->user_figures[0],
        "Tracking error",
        "vx cmd - actual",
        "yaw cmd - actual",
        -2.2f,
        2.2f);
    InitDr002RewardFigure(
        this->sim->user_figures[1],
        "Stationary reward/s",
        "vx penalty",
        "yaw penalty",
        -2.1f,
        0.1f);
    this->sim->user_figure_count = 2;
}

void RL_Sim::UpdateDr002RewardFigures()
{
    if (this->robot_name != "dr002" || !this->sim || this->obs.commands.size() < 2 ||
        this->obs.ang_vel.size() < 3)
    {
        return;
    }

    float actual_vx = 0.0f;
    if (this->mj_model && this->mj_data && this->mj_data->qvel &&
        this->root_qvel_addr >= 0 && this->root_qvel_addr + 2 < this->mj_model->nv)
    {
        const std::vector<float> world_linear_velocity = {
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 0]),
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 1]),
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 2]),
        };
        const std::vector<float> local_linear_velocity =
            QuatRotateInverse(this->obs.base_quat, world_linear_velocity);
        if (!local_linear_velocity.empty())
        {
            actual_vx = local_linear_velocity[0];
        }
    }

    const float command_vx = this->obs.commands[0];
    const float command_yaw = this->obs.commands[1];
    const float actual_yaw = this->obs.ang_vel[2];
    const float vx_error = command_vx - actual_vx;
    const float yaw_error = command_yaw - actual_yaw;
    const float gate_vx = this->params.Get<float>("zero_cmd_stationary_gate_vx", 0.05f);
    const float gate_wz = this->params.Get<float>("zero_cmd_stationary_gate_wz", 0.05f);
    const bool stationary_gate =
        std::fabs(command_vx) < gate_vx && std::fabs(command_yaw) < gate_wz;

    const float vx_scale = std::fabs(
        this->params.Get<float>("zero_cmd_stationary_vx_scale", -4.0f));
    const float yaw_scale = std::fabs(
        this->params.Get<float>("zero_cmd_stationary_yaw_scale", -12.0f));
    const float vx_clip = std::fabs(
        this->params.Get<float>("zero_cmd_stationary_vx_term_clip", 2.0f));
    const float yaw_clip = std::fabs(
        this->params.Get<float>("zero_cmd_stationary_yaw_term_clip", 2.0f));
    const float vx_reward = stationary_gate
        ? -std::min(vx_scale * std::fabs(actual_vx), vx_clip)
        : 0.0f;
    const float yaw_reward = stationary_gate
        ? -std::min(yaw_scale * std::fabs(actual_yaw), yaw_clip)
        : 0.0f;

    std::lock_guard<std::mutex> lock(this->sim->user_figure_mutex);
    mjvFigure &error_figure = this->sim->user_figures[0];
    mjvFigure &reward_figure = this->sim->user_figures[1];
    PushDr002RewardFigureSample(error_figure, 0, vx_error);
    PushDr002RewardFigureSample(error_figure, 1, yaw_error);
    PushDr002RewardFigureSample(reward_figure, 0, vx_reward);
    PushDr002RewardFigureSample(reward_figure, 1, yaw_reward);
    std::snprintf(error_figure.title, sizeof(error_figure.title),
                  "Error  vx %.3f  yaw %.3f", vx_error, yaw_error);
    std::snprintf(reward_figure.title, sizeof(reward_figure.title),
                  "Reward/s  vx %.3f  yaw %.3f", vx_reward, yaw_reward);
}

void RL_Sim::PrintDr002Status()
{
    if (this->robot_name != "dr002" || !this->mj_model || !this->mj_data ||
        this->obs.commands.size() < 3 || this->obs.ang_vel.size() < 3 ||
        this->obs.wing_angle.size() < 2)
    {
        return;
    }
    const int interval = std::max(
        1, this->params.Get<int>("status_print_interval_policy_steps", 10));
    if (this->episode_length_buf % interval != 0)
    {
        return;
    }

    float actual_vx = 0.0f;
    if (this->mj_data->qvel && this->root_qvel_addr >= 0 &&
        this->root_qvel_addr + 2 < this->mj_model->nv)
    {
        const std::vector<float> world_linear_velocity = {
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 0]),
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 1]),
            static_cast<float>(this->mj_data->qvel[this->root_qvel_addr + 2]),
        };
        const auto local_linear_velocity =
            QuatRotateInverse(this->obs.base_quat, world_linear_velocity);
        if (!local_linear_velocity.empty())
        {
            actual_vx = local_linear_velocity[0];
        }
    }
    const float actual_height =
        this->mj_data->qpos && this->root_qpos_addr >= 0 &&
            this->root_qpos_addr + 2 < this->mj_model->nq
        ? static_cast<float>(this->mj_data->qpos[this->root_qpos_addr + 2])
        : 0.0f;
    constexpr double kRadToDeg = 57.2957795130823208768;

    std::ostringstream status;
    status << std::fixed << std::setprecision(3)
           << "[WE11 status] cmd{vx=" << this->obs.commands[0]
           << ", vyaw=" << this->obs.commands[1]
           << ", height=" << this->obs.commands[2] << "m} "
           << "actual{vx=" << actual_vx
           << ", vyaw=" << this->obs.ang_vel[2]
           << ", height=" << actual_height << "m} "
           << std::setprecision(1)
           << "wing_deg{actual=[" << this->obs.wing_angle[0] * 180.0f
           << ", " << this->obs.wing_angle[1] * 180.0f
           << "], target=[" << this->wing_position_command_rad_[0] * kRadToDeg
           << ", " << this->wing_position_command_rad_[1] * kRadToDeg << "]}";
    std::cout << LOGGER::INFO << status.str() << std::endl;
}

bool RL_Sim::PrimeDr002PolicyCommand()
{
    if (this->robot_name != "dr002" || !this->rl_init_done || !this->model)
    {
        return false;
    }

    this->GetState(&this->robot_state);

    this->obs.ang_vel = this->robot_state.imu.gyroscope;
    const float command_x = this->LimitDr002CommandX(this->control.x);
    this->obs.commands = {command_x, this->control.yaw, this->dr002_height_command_};
    this->obs.base_quat = this->robot_state.imu.quaternion;
    this->obs.gravity_vec = this->GetMujocoProjectedGravity();
    this->obs.dof_pos = this->robot_state.motor_state.q;
    this->obs.dof_vel = this->robot_state.motor_state.dq;
    this->UpdateWingObservation();
    this->UpdateDr002RewardFigures();
    this->PrintDr002Status();

    this->obs.actions = this->Forward();
    this->ComputeDr002OutputWithDeployLimits(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

    static int trace_output_count = 0;
    const int trace_output_limit = Dr002ActionTraceLimit();
    if (trace_output_count < trace_output_limit)
    {
        ++trace_output_count;
        std::cout << LOGGER::INFO << "[DR002 policy trace output #" << trace_output_count << "] ";
        PrintTraceVector("command", this->obs.commands);
        std::cout << " ";
        PrintTraceVector("gravity", this->obs.gravity_vec);
        std::cout << " ";
        PrintTraceVector("dof_pos", this->obs.dof_pos);
        std::cout << " ";
        PrintTraceVector("dof_vel", this->obs.dof_vel);
        std::cout << " ";
        PrintTraceVector("wing_angle", this->obs.wing_angle);
        std::cout << " ";
        PrintTraceVector("wing_vel", this->obs.wing_vel);
        std::cout << " ";
        PrintTraceVector("raw_action", this->obs.actions);
        if (this->params.Get<bool>("wing_angle_force_zero_output", false))
        {
            const auto observation_names =
                this->params.Get<std::vector<std::string>>("observations");
            const auto wing_it = std::find(
                observation_names.begin(), observation_names.end(), "wing_angle");
            const auto history_indices =
                this->params.Get<std::vector<int>>("observations_history");
            if (wing_it == observation_names.end() ||
                this->params.Get<std::string>("observations_history_priority") != "term" ||
                this->obs_dims.size() != observation_names.size())
            {
                throw std::runtime_error(
                    "Masked wing-angle trace requires the configured term-major observation schema");
            }

            const size_t wing_term =
                static_cast<size_t>(std::distance(observation_names.begin(), wing_it));
            size_t wing_start = 0;
            for (size_t i = 0; i < wing_term; ++i)
            {
                wing_start +=
                    static_cast<size_t>(this->obs_dims[i]) * history_indices.size();
            }
            const size_t wing_count =
                static_cast<size_t>(this->obs_dims[wing_term]) * history_indices.size();
            if (wing_start + wing_count > this->history_obs.size())
            {
                throw std::runtime_error(
                    "Masked wing-angle history slice exceeds policy observation size");
            }
            const std::vector<float> policy_wing_history(
                this->history_obs.begin() + static_cast<std::ptrdiff_t>(wing_start),
                this->history_obs.begin() +
                    static_cast<std::ptrdiff_t>(wing_start + wing_count));
            if (std::any_of(
                    policy_wing_history.begin(),
                    policy_wing_history.end(),
                    [](float value) { return value != 0.0f; }))
            {
                throw std::runtime_error(
                    "Masked wing-angle history contains a non-zero policy input");
            }
            std::cout << " ";
            PrintTraceVector("policy_wing_history", policy_wing_history);
        }
        std::cout << " ";
        PrintTraceVector("motor_q", this->output_dof_pos);
        std::cout << " ";
        PrintTraceVector("motor_dq", this->output_dof_vel);
        std::cout << std::endl;
    }

    const int num_dofs = this->params.Get<int>("num_of_dofs");
    if (static_cast<int>(this->output_dof_pos.size()) < num_dofs ||
        static_cast<int>(this->output_dof_vel.size()) < num_dofs)
    {
        return false;
    }

    const auto kp = this->params.Get<std::vector<float>>("rl_kp");
    const auto kd = this->params.Get<std::vector<float>>("rl_kd");
    for (int i = 0; i < num_dofs; ++i)
    {
        this->robot_command.motor_command.q[i] = this->output_dof_pos[i];
        this->robot_command.motor_command.dq[i] = this->output_dof_vel[i];
        this->robot_command.motor_command.kp[i] = kp[i];
        this->robot_command.motor_command.kd[i] = kd[i];
        this->robot_command.motor_command.tau[i] = 0.0f;
    }

    return true;
}

void RL_Sim::InitDr002DeployLimits()
{
    if (this->robot_name != "dr002")
    {
        return;
    }

    bool enabled = this->params.Get<bool>("deploy_joint_position_limits_enabled", false);
    const char *mode_env = std::getenv("RL_SAR_DEPLOY_LIMITS");
    if (mode_env)
    {
        const std::string mode(mode_env);
        if (mode.empty() || mode == "0" || mode == "false" || mode == "False" || mode == "off")
        {
            enabled = false;
        }
        else
        {
            enabled = true;
            if (mode != "legacy" && mode != "1" && mode != "true" && mode != "True")
            {
                std::cout << LOGGER::WARNING << "[DeployLimit] Unknown RL_SAR_DEPLOY_LIMITS='"
                          << mode << "', enabling configured deploy limits." << std::endl;
            }
        }
    }

    if (!enabled)
    {
        return;
    }

    this->dr002_deploy_limit_lower = {-1.23f, -1.07f, -6.28f, -1.23f, -1.07f, -6.28f};
    this->dr002_deploy_limit_upper = { 0.77f,  1.43f,  6.28f,  0.77f,  1.43f,  6.28f};

    if (this->params.Has("joint_position_limits"))
    {
        const auto flat_limits = this->params.Get<std::vector<float>>("joint_position_limits");
        const int num_dofs = this->params.Get<int>("num_of_dofs");
        if (static_cast<int>(flat_limits.size()) >= num_dofs * 2)
        {
            this->dr002_deploy_limit_lower.assign(num_dofs, 0.0f);
            this->dr002_deploy_limit_upper.assign(num_dofs, 0.0f);
            for (int i = 0; i < num_dofs; ++i)
            {
                this->dr002_deploy_limit_lower[i] = flat_limits[2 * i];
                this->dr002_deploy_limit_upper[i] = flat_limits[2 * i + 1];
            }
        }
        else
        {
            std::cout << LOGGER::WARNING
                      << "[DeployLimit] joint_position_limits has " << flat_limits.size()
                      << " values; expected at least " << (num_dofs * 2)
                      << ". Using DR002 deploy defaults." << std::endl;
        }
    }
    this->dr002_deploy_limits_enabled = true;

    std::cout << LOGGER::WARNING
              << "[DeployLimit] DR002 deploy relative position limits enabled. "
              << "This simulates motors_node clipping before default_dof_pos is added. "
              << "limits=[";
    for (size_t i = 0; i < this->dr002_deploy_limit_lower.size(); ++i)
    {
        if (i > 0)
        {
            std::cout << ", ";
        }
        std::cout << this->dr002_deploy_limit_lower[i] << ", " << this->dr002_deploy_limit_upper[i];
    }
    std::cout << "]." << std::endl;
}

void RL_Sim::ComputeDr002OutputWithDeployLimits(const std::vector<float> &actions,
                                                std::vector<float> &output_dof_pos,
                                                std::vector<float> &output_dof_vel,
                                                std::vector<float> &output_dof_tau)
{
    if (!this->dr002_deploy_limits_enabled)
    {
        this->ComputeOutput(actions, output_dof_pos, output_dof_vel, output_dof_tau);
        return;
    }

    std::vector<float> actions_scaled = actions * this->params.Get<std::vector<float>>("action_scale");
    std::vector<float> pos_actions_scaled = actions_scaled;
    std::vector<float> vel_actions_scaled(actions.size(), 0.0f);
    std::vector<bool> is_wheel(actions.size(), false);
    const bool reverse_wheel_action_output = this->params.Get<bool>("reverse_wheel_action_output", false);
    for (int i : this->params.Get<std::vector<int>>("wheel_indices"))
    {
        if (i < 0 || i >= static_cast<int>(actions.size()))
        {
            continue;
        }
        is_wheel[i] = true;
        pos_actions_scaled[i] = 0.0f;
        vel_actions_scaled[i] = actions_scaled[i];
        if (reverse_wheel_action_output)
        {
            vel_actions_scaled[i] = -vel_actions_scaled[i];
        }
    }

    static int clip_warning_count = 0;
    const int n = std::min({
        static_cast<int>(pos_actions_scaled.size()),
        static_cast<int>(this->dr002_deploy_limit_lower.size()),
        static_cast<int>(this->dr002_deploy_limit_upper.size())
    });
    for (int i = 0; i < n; ++i)
    {
        if (is_wheel[i])
        {
            continue;
        }
        const float original = pos_actions_scaled[i];
        pos_actions_scaled[i] = std::max(
            this->dr002_deploy_limit_lower[i],
            std::min(this->dr002_deploy_limit_upper[i], pos_actions_scaled[i]));
        if (pos_actions_scaled[i] != original)
        {
            clip_warning_count += 1;
            if (clip_warning_count <= 10 || clip_warning_count % 100 == 0)
            {
                std::cout << LOGGER::WARNING << "[DeployLimit] motor" << (i + 1)
                          << " relative target clipped ["
                          << this->dr002_deploy_limit_lower[i] << ", "
                          << this->dr002_deploy_limit_upper[i] << "]: raw="
                          << original << " clipped=" << pos_actions_scaled[i]
                          << std::endl;
            }
        }
    }

    std::vector<float> all_actions_scaled = pos_actions_scaled + vel_actions_scaled;
    output_dof_pos = pos_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos");
    output_dof_vel = vel_actions_scaled;
    output_dof_tau = this->params.Get<std::vector<float>>("rl_kp") *
        (all_actions_scaled + this->params.Get<std::vector<float>>("default_dof_pos") - this->obs.dof_pos) -
        this->params.Get<std::vector<float>>("rl_kd") * this->obs.dof_vel;
    output_dof_tau = clamp(
        output_dof_tau,
        -this->params.Get<std::vector<float>>("torque_limits"),
        this->params.Get<std::vector<float>>("torque_limits"));
}

float RL_Sim::LimitDr002CommandX(float target_x)
{
    const float accel_limit = this->params.Has("command_x_accel_limit")
        ? std::fabs(this->params.Get<float>("command_x_accel_limit"))
        : 0.0f;
    if (accel_limit <= 0.0f)
    {
        this->dr002_command_x_limiter_initialized = true;
        this->dr002_limited_command_x = target_x;
        return target_x;
    }

    const float sim_dt = this->params.Has("dt") ? this->params.Get<float>("dt") : 0.0f;
    const int decimation = this->params.Has("decimation") ? this->params.Get<int>("decimation") : 1;
    const float policy_dt = sim_dt > 0.0f ? sim_dt * static_cast<float>(std::max(decimation, 1)) : 0.02f;
    const float max_delta = accel_limit * policy_dt;

    if (!this->dr002_command_x_limiter_initialized)
    {
        this->dr002_limited_command_x = target_x;
        this->dr002_command_x_limiter_initialized = true;
        return this->dr002_limited_command_x;
    }

    const float delta = target_x - this->dr002_limited_command_x;
    const float clipped_delta = std::max(-max_delta, std::min(max_delta, delta));
    this->dr002_limited_command_x += clipped_delta;
    return this->dr002_limited_command_x;
}

float RL_Sim::SampleDr002CommandX()
{
    const float low = this->params.Has("command_x_random_min")
        ? this->params.Get<float>("command_x_random_min")
        : 0.0f;
    const float high = this->params.Has("command_x_random_max")
        ? this->params.Get<float>("command_x_random_max")
        : this->params.Get<float>("default_command_x");
    const float sample_low = std::min(low, high);
    const float sample_high = std::max(low, high);
    std::uniform_real_distribution<float> dist(sample_low, sample_high);
    return dist(this->dr002_command_rng);
}

void RL_Sim::UpdateDr002RandomCommandXTarget(bool force_resample)
{
    if (!this->params.Has("command_x_random_enabled") ||
        !this->params.Get<bool>("command_x_random_enabled"))
    {
        return;
    }
    if (!force_resample && this->dr002_sync_step_count < this->dr002_next_command_x_sample_step)
    {
        return;
    }

    this->control.x = this->SampleDr002CommandX();

    const float resampling_time = this->params.Has("command_x_resampling_time")
        ? this->params.Get<float>("command_x_resampling_time")
        : 5.0f;
    const float dt = this->params.Has("dt") ? this->params.Get<float>("dt") : 0.0f;
    const int resampling_steps = (resampling_time > 0.0f && dt > 0.0f)
        ? std::max(1, static_cast<int>(resampling_time / dt + 0.5f))
        : 1;
    this->dr002_next_command_x_sample_step = this->dr002_sync_step_count + resampling_steps;

    std::cout << std::endl << LOGGER::INFO << "DR002 sampled command x target: "
              << this->control.x << " next_resample_step="
              << this->dr002_next_command_x_sample_step << std::endl;
}

void RL_Sim::RobotControl()
{
    // Lock the sim mutex once for the entire control cycle to prevent race conditions
    const std::lock_guard<std::recursive_mutex> lock(sim->mtx);
    // Keyboard and joystick callbacks run on separate threads. Consume their
    // shared Control state as one packet so combo keys and axes cannot race the
    // 400 Hz controller's ClearInput()/state-transition logic.
    const std::lock_guard<std::mutex> input_lock(this->control_input_mutex);

    this->GetState(&this->robot_state);
    this->UpdateJointTorqueOverlay();

    const bool dr002_enter_toggle =
        this->robot_name == "dr002" &&
        (this->control.current_keyboard == Input::Keyboard::Enter ||
         this->control.current_gamepad == Input::Gamepad::RB_X);
    const bool dr002_ui_run_requested =
        this->robot_name == "dr002" &&
        this->sim->run &&
        !this->simulation_running &&
        !dr002_enter_toggle;
    const bool dr002_policy_start =
        this->robot_name == "dr002" &&
        (this->control.current_keyboard == Input::Keyboard::Num1 ||
         this->control.current_keyboard == Input::Keyboard::Num2 ||
         this->control.current_gamepad == Input::Gamepad::RB_DPadUp ||
         (dr002_enter_toggle && !this->simulation_running) ||
         dr002_ui_run_requested);
    const bool dr002_reset_to_start =
        this->robot_name == "dr002" &&
        (this->control.current_keyboard == Input::Keyboard::Num0 ||
         this->control.current_gamepad == Input::Gamepad::A);
    const bool dr002_pause =
        this->robot_name == "dr002" &&
        (this->control.current_keyboard == Input::Keyboard::P ||
         this->control.current_gamepad == Input::Gamepad::LB_X ||
         (dr002_enter_toggle && this->simulation_running));
    bool dr002_mode_key_consumed = false;
    if (dr002_reset_to_start)
    {
        this->ResetMujocoToInitialPose(true, true, "DR002 reset-to-start");
        this->simulation_running = false;
        this->sim->run = 0;
        dr002_mode_key_consumed = true;
    }
    if (dr002_policy_start)
    {
        this->ResetMujocoToInitialPose(false, false, "DR002 policy start");
        this->fsm.RequestStateChange("RLFSMStateRLLocomotion");
        this->simulation_running = false;
        this->sim->run = 0;
        dr002_mode_key_consumed = true;
    }
    if (dr002_pause)
    {
        this->fsm.RequestStateChange("RLFSMStatePassive");
        this->simulation_running = false;
        this->sim->run = 0;
        dr002_mode_key_consumed = true;
    }
    if (dr002_mode_key_consumed)
    {
        this->control.current_keyboard = Input::Keyboard::None;
        this->control.last_keyboard = Input::Keyboard::None;
        this->control.current_gamepad = Input::Gamepad::None;
        this->control.last_gamepad = Input::Gamepad::None;
    }

    const bool nxbx_enter_toggle =
        this->robot_name == "nxbx" &&
        (this->control.current_keyboard == Input::Keyboard::Enter ||
         this->control.current_gamepad == Input::Gamepad::RB_X);
    const bool nxbx_ui_run_requested =
        this->robot_name == "nxbx" && this->sim->run &&
        !this->simulation_running && !nxbx_enter_toggle;
    const bool nxbx_policy_start =
        this->robot_name == "nxbx" &&
        (this->control.current_keyboard == Input::Keyboard::Num1 ||
         this->control.current_keyboard == Input::Keyboard::Num2 ||
         this->control.current_gamepad == Input::Gamepad::RB_DPadUp ||
         (nxbx_enter_toggle && !this->simulation_running) ||
         nxbx_ui_run_requested);
    const bool nxbx_reset_to_start =
        this->robot_name == "nxbx" &&
        (this->control.current_keyboard == Input::Keyboard::Num0 ||
         this->control.current_keyboard == Input::Keyboard::R ||
         this->control.current_gamepad == Input::Gamepad::A ||
         this->control.current_gamepad == Input::Gamepad::RB_Y);
    const bool nxbx_pause =
        this->robot_name == "nxbx" &&
        (this->control.current_keyboard == Input::Keyboard::P ||
         this->control.current_gamepad == Input::Gamepad::LB_X ||
         (nxbx_enter_toggle && this->simulation_running));
    bool nxbx_mode_key_consumed = false;
    if (nxbx_reset_to_start)
    {
        this->ResetMujocoToInitialPose(true, true, "NXBx reset-to-start");
        this->simulation_running = false;
        this->sim->run = 0;
        nxbx_mode_key_consumed = true;
    }
    if (nxbx_policy_start)
    {
        this->ResetMujocoToInitialPose(false, false, "NXBx policy start");
        this->fsm.RequestStateChange("RLFSMStateRLLocomotion");
        this->simulation_running = false;
        this->sim->run = 0;
        nxbx_mode_key_consumed = true;
    }
    if (nxbx_pause)
    {
        this->fsm.RequestStateChange("RLFSMStatePassive");
        this->simulation_running = false;
        this->sim->run = 0;
        nxbx_mode_key_consumed = true;
    }
    if (nxbx_mode_key_consumed)
    {
        this->control.current_keyboard = Input::Keyboard::None;
        this->control.last_keyboard = Input::Keyboard::None;
        this->control.current_gamepad = Input::Gamepad::None;
        this->control.last_gamepad = Input::Gamepad::None;
    }

    this->StateController(&this->robot_state, &this->robot_command);

    // StateController performs the Passive -> RLLocomotion transition and
    // RLFSMStateRLLocomotion::Enter() loads the selected model via InitRL().
    // Reinitialize safety/control contracts after InitRL reloads the same YAML.
    // Delay steps are counted in motor-control ticks.
    if (dr002_policy_start && this->robot_name == "dr002" && this->rl_init_done)
    {
        this->InitDr002DeployLimits();
        this->InitDr002MotorControl();
        this->ResetDr002ActionDelayBuffer(true);
        this->ResetDr002TorqueDelayBuffer(true);
    }

    if (nxbx_policy_start && this->rl_init_done)
    {
        // The sampled target delay is an environment property. Reset only its
        // FIFO here so repeated starts match training's resample=false contract.
        this->ResetNxbxTargetDelayBuffer();
        this->ResetNxbxHistoryToCurrentState();
        const bool primed = this->PrimeNxbxPolicyCommand();
        std::cout << std::endl << LOGGER::INFO << "NXBx policy command prime: "
                  << (primed ? "ok" : "failed")
                  << " vx=" << (this->obs.commands.size() > 0 ? this->obs.commands[0] : 0.0f)
                  << " yaw=" << (this->obs.commands.size() > 1 ? this->obs.commands[1] : 0.0f)
                  << " delay=" << this->nxbx_target_delay_steps << " physics steps"
                  << std::endl;
        this->simulation_running = primed;
        this->sim->run = 0;
    }

    if (dr002_policy_start)
    {
        const float startup_stand_seconds = this->params.Has("startup_stand_seconds")
            ? this->params.Get<float>("startup_stand_seconds")
            : 0.0f;
        const float dt = this->params.Has("dt") ? this->params.Get<float>("dt") : 0.0f;
        this->dr002_startup_hold_steps = (startup_stand_seconds > 0.0f && dt > 0.0f)
            ? static_cast<int>(startup_stand_seconds / dt + 0.5f)
            : 0;
        this->dr002_startup_hold_active = this->dr002_startup_hold_steps > 0;

        if (this->dr002_startup_hold_active)
        {
            this->control.x = 0.0f;
            this->control.yaw = 0.0f;
        }
        else
        {
            if (this->params.Has("command_x_random_enabled") &&
                this->params.Get<bool>("command_x_random_enabled"))
            {
                this->UpdateDr002RandomCommandXTarget(true);
            }
            else if (this->params.Has("default_command_x"))
            {
                this->control.x = this->params.Get<float>("default_command_x");
            }
            if (this->params.Has("default_command_yaw"))
            {
                this->control.yaw = this->params.Get<float>("default_command_yaw");
            }
        }
        this->dr002_limited_command_x = 0.0f;
        this->dr002_command_x_limiter_initialized = true;
        this->ResetWingCommand();
        this->ResetRLHistoryToCurrentState();
        const bool primed = this->PrimeDr002PolicyCommand();
        std::cout << std::endl << LOGGER::INFO << "DR002 policy command prime: "
                  << (primed ? "ok" : "failed")
                  << " x=" << this->control.x
                  << " yaw=" << this->control.yaw;
        if (this->dr002_startup_hold_active)
        {
            std::cout << " startup_hold=" << startup_stand_seconds << "s";
        }
        std::cout << std::endl;
        this->simulation_running = primed && this->rl_init_done;
        if (!this->simulation_running)
        {
            this->dr002_startup_hold_active = false;
            this->fsm.RequestStateChange("RLFSMStatePassive");
            std::cout << LOGGER::ERROR
                      << "DR002 policy start aborted; simulation remains stopped."
                      << std::endl;
        }
        this->sim->run = 0;
    }

    if (this->control.current_keyboard == Input::Keyboard::R || this->control.current_gamepad == Input::Gamepad::RB_Y)
    {
        if (this->mj_model && this->mj_data)
        {
            this->ResetMujocoToInitialPose(true, true, "MuJoCo reset");
            if (this->robot_name == "dr002" || this->robot_name == "nxbx")
            {
                this->simulation_running = false;
                this->sim->run = 0;
            }
        }
    }
    if (this->robot_name != "dr002" && this->robot_name != "nxbx" &&
        (this->control.current_keyboard == Input::Keyboard::Enter || this->control.current_gamepad == Input::Gamepad::RB_X))
    {
        if (simulation_running)
        {
            sim->run = 0;
            std::cout << std::endl << LOGGER::INFO << "Simulation Stop" << std::endl;
        }
        else
        {
            sim->run = 1;
            std::cout << std::endl << LOGGER::INFO << "Simulation Start" << std::endl;
        }
        simulation_running = !simulation_running;
    }

    if (this->robot_name == "dr002" && this->simulation_running && !dr002_policy_start)
    {
        this->sim->run = 0;
        if (this->dr002_startup_hold_active)
        {
            this->control.x = 0.0f;
            this->control.yaw = 0.0f;
            if (this->dr002_sync_step_count >= this->dr002_startup_hold_steps)
            {
                if (this->params.Has("command_x_random_enabled") &&
                    this->params.Get<bool>("command_x_random_enabled"))
                {
                    this->UpdateDr002RandomCommandXTarget(true);
                }
                else if (this->params.Has("default_command_x"))
                {
                    this->control.x = this->params.Get<float>("default_command_x");
                }
                if (this->params.Has("default_command_yaw"))
                {
                    this->control.yaw = this->params.Get<float>("default_command_yaw");
                }
                this->dr002_startup_hold_active = false;
                // Training changes the command at the startup boundary without
                // resetting observation history or previous actions. Preserve
                // that continuity here and let new command frames roll in.
                std::cout << std::endl << LOGGER::INFO << "DR002 startup hold done: x="
                          << this->control.x << " yaw=" << this->control.yaw << std::endl;
            }
        }
        if (!this->dr002_startup_hold_active)
        {
            this->UpdateDr002RandomCommandXTarget(false);
        }
        const int decimation = this->params.Get<int>("decimation");
        if (decimation > 0 &&
            this->dr002_sync_step_count > 0 &&
            this->dr002_sync_step_count % decimation == 0)
        {
            this->episode_length_buf += 1;
            this->PrimeDr002PolicyCommand();
        }
    }

    if (this->robot_name == "nxbx" && this->simulation_running && !nxbx_policy_start)
    {
        this->sim->run = 0;
        const int decimation = this->params.Get<int>("decimation");
        if (decimation > 0 && this->nxbx_sync_step_count > 0 &&
            this->nxbx_sync_step_count % decimation == 0)
        {
            this->episode_length_buf += 1;
            this->AdvanceNxbxCommandX();
            this->PrimeNxbxPolicyCommand();
        }
    }

    this->control.ClearInput();

    const RobotCommand<float> *command_to_apply = &this->robot_command;
    bool apply_command_this_step = true;
    if (this->robot_name == "dr002" && this->simulation_running)
    {
        apply_command_this_step = this->IsDr002MotorControlUpdateDue();
        if (apply_command_this_step)
        {
            command_to_apply = this->GetDr002DelayedCommand(&this->robot_command);
        }
    }
    else if (this->robot_name == "nxbx" && this->simulation_running)
    {
        command_to_apply = this->GetNxbxDelayedCommand(&this->robot_command);
    }
    if (apply_command_this_step)
    {
        this->SetCommand(command_to_apply);
    }

    if (this->robot_name == "dr002" && this->simulation_running && this->mj_data)
    {
        this->sim->run = 0;
        this->TickDr002HeightCommand();
        // TickWingCommand advances the RC-like wing velocity target once per
        // policy step boundary and writes ctrl[wing_*] every physics tick so
        // the MuJoCo motor actuator maintains the command through the step.
        this->TickWingCommand();
        mj_step(this->mj_model, this->mj_data);
        this->AdvanceDr002MotorControl();
        this->dr002_sync_step_count += 1;
        this->GetState(&this->robot_state);
        this->UpdateJointTorqueOverlay();
    }
    else if (this->robot_name == "nxbx" && this->simulation_running && this->mj_data)
    {
        this->sim->run = 0;
        mj_step(this->mj_model, this->mj_data);
        this->nxbx_sync_step_count += 1;
        this->GetState(&this->robot_state);
        this->UpdateJointTorqueOverlay();
    }
}

void RL_Sim::SetupSysJoystick(const std::string& device, int bits)
{
    this->sys_js = std::make_unique<Joystick>(device);
    if (!this->sys_js->isFound())
    {
        std::cout << LOGGER::ERROR << "Joystick [" << device << "] open failed." << std::endl;
        // exit(1);
    }

    this->sys_js_max_value = (1 << (bits - 1));
}

void RL_Sim::KeyboardInterface()
{
    const std::lock_guard<std::mutex> lock(this->control_input_mutex);
    RL::KeyboardInterface();
}

void RL_Sim::GetSysJoystick()
{
    const std::lock_guard<std::mutex> lock(this->control_input_mutex);

    // Clear all button event states
    for (int i = 0; i < 20; ++i)
    {
        this->sys_js_button[i].on_press = false;
        this->sys_js_button[i].on_release = false;
    }

    // Check if joystick is valid before using
    if (!this->sys_js)
    {
        return;
    }

    while (this->sys_js->sample(&this->sys_js_event))
    {
        if (this->sys_js_event.isButton())
        {
            const size_t button_index = static_cast<size_t>(this->sys_js_event.number);
            if (button_index < std::size(this->sys_js_button))
            {
                this->sys_js_button[button_index].update(this->sys_js_event.value);
            }
        }
        else if (this->sys_js_event.isAxis())
        {
            const size_t axis_index = static_cast<size_t>(this->sys_js_event.number);
            if (axis_index >= std::size(this->sys_js_axis))
            {
                continue;
            }
            double normalized = double(this->sys_js_event.value) / this->sys_js_max_value;
            if (std::abs(normalized) < this->axis_deadzone)
            {
                this->sys_js_axis[axis_index] = 0;
            }
            else
            {
                this->sys_js_axis[axis_index] = this->sys_js_event.value;
            }
        }
    }

    // Button indices measured with scripts/joystick_calibrate.py on the active
    // Xbox pad. LStick / RStick / DPad buttons retain standard SDL fallbacks.
    constexpr int kBtnA = 0;
    constexpr int kBtnB = 1;
    constexpr int kBtnX = 2;
    constexpr int kBtnY = 3;
    constexpr int kBtnLB = 4;
    constexpr int kBtnRB = 5;
    constexpr int kBtnLStick = 9;
    constexpr int kBtnRStick = 10;
    // DPad on many pads is exposed as axes 6/7. Left untouched for the same
    // reason; harmless if the axis does not exist on this device.
    constexpr int kAxisDPadX = 6;
    constexpr int kAxisDPadY = 7;

    if (this->sys_js_button[kBtnA].on_press) this->control.SetGamepad(Input::Gamepad::A);
    if (this->sys_js_button[kBtnB].on_press) this->control.SetGamepad(Input::Gamepad::B);
    if (this->sys_js_button[kBtnX].on_press) this->control.SetGamepad(Input::Gamepad::X);
    if (this->sys_js_button[kBtnY].on_press) this->control.SetGamepad(Input::Gamepad::Y);
    if (this->sys_js_button[kBtnLB].on_press) this->control.SetGamepad(Input::Gamepad::LB);
    if (this->sys_js_button[kBtnRB].on_press) this->control.SetGamepad(Input::Gamepad::RB);
    if (this->sys_js_button[kBtnLStick].on_press) this->control.SetGamepad(Input::Gamepad::LStick);
    if (this->sys_js_button[kBtnRStick].on_press) this->control.SetGamepad(Input::Gamepad::RStick);
    if (this->sys_js_axis[kAxisDPadY] < 0) this->control.SetGamepad(Input::Gamepad::DPadUp);
    if (this->sys_js_axis[kAxisDPadY] > 0) this->control.SetGamepad(Input::Gamepad::DPadDown);
    if (this->sys_js_axis[kAxisDPadX] > 0) this->control.SetGamepad(Input::Gamepad::DPadLeft);
    if (this->sys_js_axis[kAxisDPadX] < 0) this->control.SetGamepad(Input::Gamepad::DPadRight);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_button[kBtnA].on_press) this->control.SetGamepad(Input::Gamepad::LB_A);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_button[kBtnB].on_press) this->control.SetGamepad(Input::Gamepad::LB_B);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_button[kBtnX].on_press) this->control.SetGamepad(Input::Gamepad::LB_X);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_button[kBtnY].on_press) this->control.SetGamepad(Input::Gamepad::LB_Y);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_button[kBtnLStick].on_press) this->control.SetGamepad(Input::Gamepad::LB_LStick);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_button[kBtnRStick].on_press) this->control.SetGamepad(Input::Gamepad::LB_RStick);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_axis[kAxisDPadY] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadUp);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_axis[kAxisDPadY] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadDown);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_axis[kAxisDPadX] > 0) this->control.SetGamepad(Input::Gamepad::LB_DPadLeft);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_axis[kAxisDPadX] < 0) this->control.SetGamepad(Input::Gamepad::LB_DPadRight);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_button[kBtnA].on_press) this->control.SetGamepad(Input::Gamepad::RB_A);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_button[kBtnB].on_press) this->control.SetGamepad(Input::Gamepad::RB_B);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_button[kBtnX].on_press) this->control.SetGamepad(Input::Gamepad::RB_X);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_button[kBtnY].on_press) this->control.SetGamepad(Input::Gamepad::RB_Y);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_button[kBtnLStick].on_press) this->control.SetGamepad(Input::Gamepad::RB_LStick);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_button[kBtnRStick].on_press) this->control.SetGamepad(Input::Gamepad::RB_RStick);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_axis[kAxisDPadY] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadUp);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_axis[kAxisDPadY] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadDown);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_axis[kAxisDPadX] > 0) this->control.SetGamepad(Input::Gamepad::RB_DPadLeft);
    if (this->sys_js_button[kBtnRB].pressed && this->sys_js_axis[kAxisDPadX] < 0) this->control.SetGamepad(Input::Gamepad::RB_DPadRight);
    if (this->sys_js_button[kBtnLB].pressed && this->sys_js_button[kBtnRB].on_press) this->control.SetGamepad(Input::Gamepad::LB_RB);

    // Left stick vertical (axis 1) -> forward/backward x. Right stick
    // horizontal (axis 3) -> yaw, while its vertical axis is consumed by the
    // height integrator. LB/RB are consumed by TickWingCommand. y-strafe is not
    // used by the DR002 policy so it stays zero.
    float ly = -float(this->sys_js_axis[1]) / float(this->sys_js_max_value);
    float rx = -float(this->sys_js_axis[3]) / float(this->sys_js_max_value);

    bool has_input = (ly != 0.0f || rx != 0.0f);

    if (has_input)
    {
        this->control.x = ly;
        this->control.yaw = rx;
        this->control.y = 0.0f;
        this->sys_js_active = true;
    }
    else if (this->sys_js_active)
    {
        this->control.x = 0.0f;
        this->control.y = 0.0f;
        this->control.yaw = 0.0f;
        this->sys_js_active = false;
    }
}

void RL_Sim::RunModel()
{
    if (this->robot_name == "dr002" || this->robot_name == "nxbx")
    {
        return;
    }

    if (this->rl_init_done && simulation_running)
    {
        {
            const std::lock_guard<std::recursive_mutex> lock(sim->mtx);
            this->GetState(&this->robot_state);
        }

        this->episode_length_buf += 1;
        this->obs.ang_vel = this->robot_state.imu.gyroscope;
        if (this->robot_name == "dr002")
        {
            this->obs.commands = {this->control.x, this->control.yaw, this->dr002_height_command_};
        }
        else
        {
            this->obs.commands = {this->control.x, this->control.y, this->control.yaw};
        }
        //not currently available for non-ros mujoco version
        // if (this->control.navigation_mode)
        // {
        //     this->obs.commands = {(float)this->cmd_vel.linear.x, (float)this->cmd_vel.linear.y, (float)this->cmd_vel.angular.z};
        // }
        this->obs.base_quat = this->robot_state.imu.quaternion;
        if (this->robot_name == "dr002")
        {
            this->obs.gravity_vec = this->GetMujocoProjectedGravity();
        }
        this->obs.dof_pos = this->robot_state.motor_state.q;
        this->obs.dof_vel = this->robot_state.motor_state.dq;

        this->obs.actions = this->Forward();
        this->ComputeOutput(this->obs.actions, this->output_dof_pos, this->output_dof_vel, this->output_dof_tau);

        // RLControl() pops pos first and vel second. Push pos last so a control tick
        // cannot consume half of a freshly generated action packet.
        if (!this->output_dof_vel.empty())
        {
            output_dof_vel_queue.push(this->output_dof_vel);
        }
        if (!this->output_dof_tau.empty())
        {
            output_dof_tau_queue.push(this->output_dof_tau);
        }
        if (!this->output_dof_pos.empty())
        {
            output_dof_pos_queue.push(this->output_dof_pos);
        }

        // this->TorqueProtect(this->output_dof_tau);
        // this->AttitudeProtect(this->robot_state.imu.quaternion, 75.0f, 75.0f);

#ifdef CSV_LOGGER
        std::vector<float> tau_est(this->params.Get<int>("num_of_dofs"), 0.0f);
        for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
        {
            tau_est[i] = this->joint_efforts[this->params.Get<std::vector<std::string>>("joint_controller_names")[i]];
        }
        this->CSVLogger(this->output_dof_tau, tau_est, this->obs.dof_pos, this->output_dof_pos, this->obs.dof_vel);
#endif
    }
}

std::vector<float> RL_Sim::Forward()
{
    std::unique_lock<std::mutex> lock(this->model_mutex, std::try_to_lock);

    // If model is being reinitialized, return previous actions to avoid blocking
    if (!lock.owns_lock())
    {
        std::cout << LOGGER::WARNING << "Model is being reinitialized, using previous actions" << std::endl;
        return this->obs.actions;
    }

    std::vector<float> clamped_obs = this->ComputeObservation();

    std::vector<float> actions;
    if (this->params.Get<std::vector<int>>("observations_history").size() != 0)
    {
        this->history_obs_buf.insert(clamped_obs);
        this->history_obs = this->history_obs_buf.get_obs_vec(this->params.Get<std::vector<int>>("observations_history"));
        if (this->robot_name == "nxbx")
        {
            if (this->obs.commands.size() != 2)
            {
                throw std::runtime_error("NXBx command observation must contain exactly [vx, yaw]");
            }
            this->history_obs.insert(
                this->history_obs.end(), this->obs.commands.begin(), this->obs.commands.end());
            const int expected_obs = this->params.Get<int>("num_observations", 41);
            if (static_cast<int>(this->history_obs.size()) != expected_obs)
            {
                throw std::runtime_error(
                    "NXBx policy observation dimension mismatch: expected " +
                    std::to_string(expected_obs) + ", got " +
                    std::to_string(this->history_obs.size()));
            }
        }
        actions = this->model->forward({this->history_obs});
    }
    else
    {
        actions = this->model->forward({clamped_obs});
    }

    if (this->robot_name == "nxbx" &&
        static_cast<int>(actions.size()) != this->params.Get<int>("num_of_dofs"))
    {
        throw std::runtime_error(
            "NXBx policy output dimension mismatch: expected 2, got " +
            std::to_string(actions.size()));
    }

    std::vector<float> clipped_actions = actions;
    if (!this->params.Get<std::vector<float>>("clip_actions_upper").empty() && !this->params.Get<std::vector<float>>("clip_actions_lower").empty())
    {
        clipped_actions = clamp(actions, this->params.Get<std::vector<float>>("clip_actions_lower"), this->params.Get<std::vector<float>>("clip_actions_upper"));
    }

    static int trace_forward_count = 0;
    const int trace_forward_limit = Dr002ActionTraceLimit();
    if (this->robot_name == "dr002" && trace_forward_count < trace_forward_limit)
    {
        ++trace_forward_count;
        std::cout << LOGGER::INFO << "[DR002 policy trace forward #" << trace_forward_count << "] ";
        PrintTraceVector("raw", actions);
        std::cout << " ";
        PrintTraceVector("clipped", clipped_actions);
        std::cout << std::endl;
    }

    return clipped_actions;
}

void RL_Sim::Plot()
{
    this->plot_t.erase(this->plot_t.begin());
    this->plot_t.push_back(this->motiontime);
    plt::cla();
    plt::clf();
    for (int i = 0; i < this->params.Get<int>("num_of_dofs"); ++i)
    {
        this->plot_real_joint_pos[i].erase(this->plot_real_joint_pos[i].begin());
        this->plot_target_joint_pos[i].erase(this->plot_target_joint_pos[i].begin());
        this->plot_real_joint_pos[i].push_back(mj_data->sensordata[i]);
        // this->plot_target_joint_pos[i].push_back();  // TODO
        plt::subplot(this->params.Get<int>("num_of_dofs"), 1, i + 1);
        plt::named_plot("_real_joint_pos", this->plot_t, this->plot_real_joint_pos[i], "r");
        plt::named_plot("_target_joint_pos", this->plot_t, this->plot_target_joint_pos[i], "b");
        plt::xlim(this->plot_t.front(), this->plot_t.back());
    }
    // plt::legend();
    plt::pause(0.01);
}

// Signal handler for Ctrl+C
void signalHandler(int signum)
{
    std::cout << LOGGER::INFO << "Received signal " << signum << ", exiting..." << std::endl;
    if (RL_Sim::instance && RL_Sim::instance->sim)
    {
        RL_Sim::instance->sim->exitrequest.store(1);
    }
}

int main(int argc, char **argv)
{
    signal(SIGINT, signalHandler);
    RL_Sim rl_sar(argc, argv);
    return 0;
}
