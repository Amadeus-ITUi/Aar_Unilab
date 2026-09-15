/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RL_SIM_HPP
#define RL_SIM_HPP

// #define PLOT
// #define CSV_LOGGER

#include "rl_sdk.hpp"
#include "observation_buffer.hpp"
#include "inference_runtime.hpp"
#include "loop.hpp"
#include "fsm_all.hpp"

#include <csignal>
#include <array>
#include <vector>
#include <deque>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <sys/wait.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <random>

#include <mujoco/mujoco.h>
#include "joystick.hh"
#include "mujoco_utils.hpp"

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

class Button
{
public:
    Button() {}

    void update(bool state)
    {
        on_press = state ? state != pressed : false;
        on_release = state ? false : state != pressed;
        pressed = state;
    }

    bool pressed = false;
    bool on_press = false;
    bool on_release = false;
};

class RL_Sim : public RL
{
public:
    RL_Sim(int argc, char **argv);
    ~RL_Sim();

    std::unique_ptr<mj::Simulate> sim;
    static RL_Sim* instance;

private:
    // rl functions
    std::vector<float> Forward() override;
    void GetState(RobotState<float> *state) override;
    void SetCommand(const RobotCommand<float> *command) override;
    void RunModel();
    void RobotControl();
    void KeyboardInterface();

    // loop
    std::shared_ptr<LoopFunc> loop_keyboard;
    std::shared_ptr<LoopFunc> loop_joystick;
    std::shared_ptr<LoopFunc> loop_control;
    std::shared_ptr<LoopFunc> loop_rl;
    std::shared_ptr<LoopFunc> loop_plot;

    // plot
    const int plot_size = 100;
    std::vector<int> plot_t;
    std::vector<std::vector<float>> plot_real_joint_pos, plot_target_joint_pos;
    void Plot();

    // mujoco
    mjData *mj_data;
    mjModel *mj_model;
    std::string scene_name;
    void CacheMujocoAddresses();
    void InitDr002BaseComOffset();
    void UpdateMujocoJointTorques();
    void UpdateJointTorqueOverlay();
    std::vector<float> ReadMujocoSensor3(int sensor_adr, int sensor_dim) const;
    std::vector<float> GetMujocoProjectedGravity() const;
    void ResetMujocoToInitialPose(bool clear_command, bool request_passive, const std::string& reason);
    void ResetRLHistoryToCurrentState();
    void InitDr002RewardFigures();
    void UpdateDr002RewardFigures();
    void PrintDr002Status();
    bool PrimeDr002PolicyCommand();
    void InitDr002DeployLimits();
    void ComputeDr002OutputWithDeployLimits(const std::vector<float> &actions,
                                            std::vector<float> &output_dof_pos,
                                            std::vector<float> &output_dof_vel,
                                            std::vector<float> &output_dof_tau);
    float LimitDr002CommandX(float target_x);
    float SampleDr002CommandX();
    void UpdateDr002RandomCommandXTarget(bool force_resample = false);
    void InitWingCommand();
    void ResetWingCommand();
    void TickDr002HeightCommand();
    void TickWingCommand();
    void UpdateWingObservation();
    void InitDr002MotorControl();
    bool IsDr002MotorControlUpdateDue() const;
    void AdvanceDr002MotorControl();
    int SampleDr002ActionDelaySteps();
    int GetDr002ActionDelaySteps();
    void ResetDr002ActionDelayBuffer(bool resample_delay = true);
    const RobotCommand<float>* GetDr002DelayedCommand(const RobotCommand<float> *command);
    int LoadDr002TorqueDelaySteps();
    void ResetDr002TorqueDelayBuffer(bool reload_config = true);
    std::vector<float> GetDr002DelayedTorques(const std::vector<float> &torques);
    void UpdateNxbxObservationState();
    void AdvanceNxbxCommandX();
    void ResetNxbxHistoryToCurrentState();
    bool PrimeNxbxPolicyCommand();
    float ShapeNxbxCommand(float axis, float min_nonzero, float max_value) const;
    int SampleNxbxTargetDelaySteps();
    void ResetNxbxTargetDelayBuffer();
    const RobotCommand<float>* GetNxbxDelayedCommand(const RobotCommand<float> *command);
    std::vector<int> joint_qpos_addr;
    std::vector<int> joint_qvel_addr;
    std::vector<float> mujoco_joint_side_torques;
    int root_qpos_addr = -1;
    int root_qvel_addr = -1;
    int mujoco_gyro_sensor_adr = -1;
    int mujoco_gyro_sensor_dim = 0;
    int mujoco_imu_site_id = -1;
    int mujoco_base_body_id = -1;
    int dr002_sync_step_count = 0;
    bool dr002_startup_hold_active = false;
    int dr002_startup_hold_steps = 0;
    bool dr002_command_x_limiter_initialized = false;
    float dr002_limited_command_x = 0.0f;
    int dr002_next_command_x_sample_step = 0;
    bool dr002_deploy_limits_enabled = false;
    std::vector<float> dr002_deploy_limit_lower;
    std::vector<float> dr002_deploy_limit_upper;
    std::mt19937 dr002_command_rng{std::random_device{}()};
    std::mt19937 dr002_getup_replay_rng{std::random_device{}()};
    int dr002_motor_control_decimation = 1;
    int dr002_motor_control_substep_index = 0;
    int dr002_action_delay_steps = 0;
    std::vector<int> dr002_action_delay_steps_by_joint;
    std::deque<std::vector<float>> dr002_delay_q;
    std::deque<std::vector<float>> dr002_delay_dq;
    std::deque<std::vector<float>> dr002_delay_tau;
    RobotCommand<float> dr002_delayed_robot_command;
    int dr002_torque_delay_steps = 0;
    std::vector<int> dr002_torque_delay_steps_by_joint;
    std::deque<std::vector<float>> dr002_torque_delay_buffer;
    int nxbx_sync_step_count = 0;
    bool nxbx_command_x_limiter_initialized = false;
    float nxbx_limited_command_x = 0.0f;
    int nxbx_target_delay_steps = 0;
    bool nxbx_history_initialized = false;
    std::deque<std::vector<float>> nxbx_delayed_velocity_targets;
    RobotCommand<float> nxbx_delayed_robot_command;
    std::mt19937 nxbx_delay_rng{std::random_device{}()};

    // Right-stick vertical is a velocity input integrated into the absolute
    // height target supplied to policy command[2].
    float dr002_height_command_ = 0.25f;
    float dr002_height_command_initial_ = 0.25f;
    float dr002_height_command_min_ = 0.20f;
    float dr002_height_command_max_ = 0.30f;
    float dr002_height_velocity_max_m_s_ = 0.03f;
    int dr002_height_joystick_axis_ = 4;
    bool dr002_height_joystick_invert_sign_ = false;

    // Wing driver (joystick-driven position PD). The RL policy does NOT
    // control the wings; LB/RB produce a signed rad/s target that the driver
    // integrates into a position setpoint every tick. The setpoint is clamped to the wing
    // joint's mechanical range. A position PD (kp/kd) then produces the
    // torque command written to the wing motor ctrl slots, so stick-neutral
    // (rad/s = 0) freezes the setpoint and the PD actively holds the wing
    // at its current angle instead of letting it drop under gravity.
    // Observations are read back from the real MuJoCo joint state so any
    // hardware limit hit shows up naturally.
    std::array<double, 2> wing_velocity_command_rad_s_ = {0.0, 0.0};
    std::array<double, 2> wing_position_command_rad_ = {0.0, 0.0};
    double wing_max_rad_per_s_ = 0.7853981633974483;  // pi/4 rad/s
    double wing_kp_ = 10.0;
    double wing_kd_ = 0.5;
    // Wing joint mechanical range in radians; read once from the model at init.
    std::array<double, 2> wing_pos_lower_ = {-1.5707963267948966, -1.5707963267948966};
    std::array<double, 2> wing_pos_upper_ = {0.0, 0.0};
    bool wing_cmd_enabled_ = true;
    int wing_actuator_ctrl_idx_[2] = {6, 7};
    int wing_qpos_addr_[2] = {-1, -1};
    int wing_qvel_addr_[2] = {-1, -1};
    // Raw js0 button indices on the calibrated workstation Xbox pad.
    int wing_decrease_button_ = 4;
    int wing_increase_button_ = 5;

    // joystick
    std::unique_ptr<Joystick> sys_js;
    JoystickEvent sys_js_event;

    Button sys_js_button[20];
    int sys_js_axis[10] = {0};
    bool sys_js_active = false;
    float axis_deadzone = 0.05f;
    int sys_js_max_value = (1 << (16 - 1));
    std::mutex control_input_mutex;
    void SetupSysJoystick(const std::string& device, int bits);
    void GetSysJoystick();

    // others
    std::string gazebo_model_name;
    std::map<std::string, float> joint_positions;
    std::map<std::string, float> joint_velocities;
    std::map<std::string, float> joint_efforts;
    void StartJointController(const std::string& ros_namespace, const std::vector<std::string>& names);
};

#endif // RL_SIM_HPP
