/*
 * Copyright (c) 2024-2025 Ziqi Fan
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DR002_FSM_HPP
#define DR002_FSM_HPP

#include "fsm.hpp"
#include "rl_sdk.hpp"

namespace dr002_fsm
{

class RLFSMStatePassive : public RLFSMState
{
public:
    RLFSMStatePassive(RL *rl) : RLFSMState(*rl, "RLFSMStatePassive") {}

    void Enter() override
    {
        std::cout << LOGGER::NOTE << "Entered DR002 two-wheel standby. Press '2' to start from the ground pose and walk forward." << std::endl;
    }

    void Run() override
    {
        auto hold_pos = rl.params.Has("initial_dof_pos")
            ? rl.params.Get<std::vector<float>>("initial_dof_pos")
            : rl.params.Get<std::vector<float>>("default_dof_pos");
        auto kp = rl.params.Get<std::vector<float>>("fixed_kp");
        auto kd = rl.params.Get<std::vector<float>>("fixed_kd");
        for (int i = 0; i < rl.params.Get<int>("num_of_dofs"); ++i)
        {
            fsm_command->motor_command.q[i] = hold_pos[i];
            fsm_command->motor_command.dq[i] = 0;
            fsm_command->motor_command.kp[i] = kp[i];
            fsm_command->motor_command.kd[i] = kd[i];
            fsm_command->motor_command.tau[i] = 0;
        }
    }

    void Exit() override {}

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::Num1 ||
            rl.control.current_keyboard == Input::Keyboard::Num2 ||
            rl.control.current_gamepad == Input::Gamepad::RB_DPadUp)
        {
            return "RLFSMStateRLLocomotion";
        }
        return state_name_;
    }
};

class RLFSMStateRLLocomotion : public RLFSMState
{
public:
    RLFSMStateRLLocomotion(RL *rl) : RLFSMState(*rl, "RLFSMStateRLLocomotion") {}

    void Enter() override
    {
        rl.episode_length_buf = 0;
        // MuJoCo preloads this selector before standby/reset so startup and
        // policy control use one configuration. Keep the fallback for other
        // DR002 frontends that have not selected a config yet.
        if (rl.config_name.empty())
        {
            rl.config_name = "robot_lab";
            if (const char* config_env = std::getenv("RL_SAR_DR002_CONFIG"))
            {
                if (std::string(config_env).size() > 0)
                {
                    rl.config_name = config_env;
                }
            }
        }
        try
        {
            rl.InitRL(rl.robot_name + "/" + rl.config_name);
            rl.now_state = *fsm_state;
            rl.rl_init_done = true;
        }
        catch (const std::exception& e)
        {
            std::cout << LOGGER::ERROR << "InitRL() failed: " << e.what() << std::endl;
            rl.rl_init_done = false;
            rl.fsm.RequestStateChange("RLFSMStatePassive");
        }
    }

    void Run() override
    {
        // Enter() owns initialization success. Never turn a failed model load
        // back into a runnable state on the same FSM tick.
        if (!rl.rl_init_done)
        {
            return;
        }
        if (rl.motiontime % 250 == 0)
        {
            std::cout << std::endl << LOGGER::INFO << "RL Controller [" << rl.config_name << "] x:" << rl.control.x << " y:" << rl.control.y << " yaw:" << rl.control.yaw << std::endl;
        }
        RLControl();
    }

    void Exit() override
    {
        rl.rl_init_done = false;
    }

    std::string CheckChange() override
    {
        if (rl.control.current_keyboard == Input::Keyboard::P || rl.control.current_gamepad == Input::Gamepad::LB_X)
        {
            return "RLFSMStatePassive";
        }
        return state_name_;
    }
};

} // namespace dr002_fsm

class DR002FSMFactory : public FSMFactory
{
public:
    DR002FSMFactory(const std::string& initial) : initial_state_(initial) {}

    std::shared_ptr<FSMState> CreateState(void *context, const std::string &state_name) override
    {
        auto *rl = static_cast<RL*>(context);
        if (state_name == "RLFSMStatePassive")
            return std::make_shared<dr002_fsm::RLFSMStatePassive>(rl);
        else if (state_name == "RLFSMStateRLLocomotion")
            return std::make_shared<dr002_fsm::RLFSMStateRLLocomotion>(rl);
        return nullptr;
    }

    std::string GetType() const override { return "dr002"; }

    std::vector<std::string> GetSupportedStates() const override
    {
        return {
            "RLFSMStatePassive",
            "RLFSMStateRLLocomotion"
        };
    }

    std::string GetInitialState() const override { return initial_state_; }

private:
    std::string initial_state_;
};

REGISTER_FSM_FACTORY(DR002FSMFactory, "RLFSMStatePassive")

#endif // DR002_FSM_HPP
