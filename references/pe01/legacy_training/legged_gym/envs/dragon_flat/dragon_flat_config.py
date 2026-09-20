# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
from pickle import TRUE
from legged_gym.envs.base.base_config import BaseConfig
from legged_gym import LEGGED_GYM_ROOT_DIR

import os
robot_type = os.getenv("ROBOT_TYPE")

class DragonCfgFlat(BaseConfig):
    class env:
        num_envs = 8192
        num_observations = 30
        num_critic_observations = 3 + num_observations
        num_height_samples = 117
        num_actions = 6
        env_spacing = 3.0  # not used with heightfields/trimeshes
        send_timeouts = True  # send time out information to the algorithm
        episode_length_s = 20  # episode length in seconds
        obs_history_length = 10  # number of observations stacked together
        dof_vel_use_pos_diff = True
        fail_to_terminal_time_s = 0.5

    class terrain:
        mesh_type = "plane"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 0.4
        dynamic_friction = 0.4
        restitution = 0.8
        # rough terrain only:
        measure_heights = False
        critic_measure_heights = True
        measured_points_x = [
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
        ]  # 1mx1.6m rectangle (without center line)
        measured_points_y = [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
        selected = False  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 5 + 4  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        # trimesh only:
        slope_treshold = (
            0.75  # slopes above this threshold will be corrected to vertical surfaces
        )

    class commands:
        curriculum = True
        smooth_max_lin_vel_x = 1.0
        smooth_max_lin_vel_y = 0.5
        non_smooth_max_lin_vel_x = 0.5
        non_smooth_max_lin_vel_y = 0.5
        max_ang_vel_yaw = 1.0
        curriculum_threshold = 0.75
        num_commands = 3  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error, only work on adaptive group
        min_norm = 0.1
        zero_command_prob = 0.1

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-0.6, 0.6]  # min max [m/s]
            # lin_vel_x = [-1.7, 1.7]  # min max [m/s]
            # lin_vel_y = [-1.7, 1.7]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14159, 3.14159]

    class gait:
        num_gait_params = 4
        resampling_time = 5  # time before command are changed[s]

        class ranges:
            frequencies = [2.0, 2.0]
            offsets = [0, 1]  # offset is hard to learn
            # durations = [0.3, 0.8]  # small durations(<0.4) is hard to learn
            # frequencies = [2, 2]
            # offsets = [0.5, 0.5]
            durations = [0.5, 0.5]
            swing_height = [0.06, 0.06]  # 调整为DRAGON尺寸 (原0.1)

    class init_state:
        pos = [0.0, 0.0, 0.32]  # x,y,z [m]
        rot = [0.0, 0.0, 0.0, 1.0]  # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # target angles when action = 0.0
            'left_hip_joint': -0.1,   # [rad]
            'right_hip_joint': 0.1,  # [rad]

            'left_thigh_joint': 0.6,     # [rad]
            'right_thigh_joint': 0.6,     # [rad]

            'left_calf_joint': -1.2,   # [rad]
            'right_calf_joint': -1.2,  # [rad]
        }

    class control:
        action_scale = 0.25

        control_type = "P"
        stiffness = {
            'hip_joint': 4.3,
            'thigh_joint': 4.3,
            'calf_joint': 4.9,
        }  # [N*m/rad]
        damping = {
            'hip_joint': 0.34,
            'thigh_joint': 0.34,
            'calf_joint': 0.24,
        }  # [N*m*s/rad]
        torque_limits = {
            'hip_joint': 5.5,
            'thigh_joint': 5.5,
            'calf_joint': 14,
        }     # [N*m]
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 8
        user_torque_limit = 14.0  # 匹配URDF中calf的最大力矩 (原值20.0)
        max_power = 400.0  # 更小的机器人功率限制应更低 (PF原值1000.0)

    class asset:
        file = "{}/resources/robots/DRAGON_3/urdf/v_end.urdf".format(LEGGED_GYM_ROOT_DIR)
        # file = "{}/resources/robots/DRAGON_2/urdf/V19.urdf".format(LEGGED_GYM_ROOT_DIR)

        name = "pointfoot_flat"
        foot_name = "foot"
        foot_radius = 0.025
        penalize_contacts_on = ["trunk", "thigh", "calf"]
        terminate_after_contacts_on = ["trunk"]
        disable_gravity = False
        collapse_fixed_joints = True  # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        fix_base_link = False # fixe the base of the robot
        default_dof_drive_mode = 3  # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = True  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = (
            False  # Some .obj meshes must be flipped from y-up to z-up
        )

        density = 0.001
        angular_damping = 0.0
        linear_damping = 0.0
        max_angular_velocity = 1000.0
        max_linear_velocity = 1000.0
        armature = 0.0
        thickness = 0.01

    class domain_rand:
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-0.2, 0.2]
        randomize_base_com = True
        rand_com_vec = [0.01, 0.01, 0.01]
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 0.5
        rand_force = False
        force_resampling_time_s = 15
        max_force = 50.0
        rand_force_curriculum_level = 0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2] 
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.02, 0.02]
        randomize_action_delay = True
        randomize_imu_offset = True
        randomize_imu_offset_range = [-1.2, 1.2]
        delay_ms_range = [25, 50]

        randmize_joint_friction = False
        joint_friction_range = [0.0, 0.05]

    class rewards:
        class scales:
            # termination related rewards
            keep_balance = 1.0

            # tracking related rewards
            tracking_lin_vel = 1
            tracking_ang_vel = 0.5
            # tracking_lin_vel_zero = -0.5
            # tracking_ang_vel_zero = -0.5

            # regulation related rewards
            base_height = -3
            lin_vel_z = -0.5
            ang_vel_xy = -0.01
            torques = -0.0002  # 力矩更小，惩罚需更敏感 (PF原值-0.00008)
            dof_acc = -2.5e-7  # 质量更轻，加速度惩罚需调整 (PF原值-2.5e-7)
            action_rate = -0.01
            dof_pos_limits = -2.0
            collision = -3.0  # 更小的机器人更易碰撞，加大惩罚 (PF原值-1)
            action_smooth = -0.005
            orientation = -5.0
            feet_distance = -150  # 机器人更窄，需更严格控制 (PF原值-100)
            feet_regulation = -0.08  # 调整比例 (PF原值-0.05)
            foot_landing_vel = -0.15  # 更轻的机器人需要更柔和着地 (PF原值-0.15)
            tracking_contacts_shaped_force = -2
            tracking_contacts_shaped_vel = -2

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        clip_reward = 100
        clip_single_reward = 5
        tracking_sigma = 0.15  # 更小的机器人，速度控制需更精确 (PF原值0.2)
        ang_tracking_sigma = 0.20  # 角速度控制更精确 (PF原值0.25)
        height_tracking_sigma = 0.01
        soft_dof_pos_limit = (
            0.95  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 0.8
        base_height_target = 0.25  # 根据DRAGON腿长调整 (PF原值0.68)
        feet_height_target = 0.04  # 比例缩小 (PF原值0.10)
        min_feet_distance = 0.08  # 基座更窄 (PF原值0.115)
        about_landing_threshold = 0.05  # 比例缩小 (PF原值0.08)
        max_contact_force = 20.0  # 质量更轻，接触力更小 (PF原值100.0)
        kappa_gait_probs = 0.05
        gait_force_sigma = 5.0  # 接触力更小，sigma需调整 (PF原值25.0)
        gait_vel_sigma = 0.20  # 速度控制更严格 (PF原值0.25)
        gait_height_sigma = 0.005

    class normalization:
        class obs_scales:
            lin_vel = 1.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.1
            dof_acc = 0.0025
            height_measurements = 1.0
            contact_forces = 0.01
            torque = 0.05

        clip_observations = 100.0
        clip_actions = 100.0

    class noise:
        add_noise = True
        noise_level = 1.5  # scales other values

        class noise_scales:
            dof_pos = 0.08
            dof_vel = 1.5
            lin_vel = 0.2
            ang_vel = 0.2
            gravity = 0.2
            height_measurements = 0.1

    # viewer camera:
    class viewer:
        ref_env = 0
        pos = [5, -5, 3]  # [m]
        # lookat = [11.0, 5, 3.0]  # [m]
        lookat = [0, 0, 0]  # [m]
        realtime_plot = True

    class sim:
        dt = 0.0025
        substeps = 1
        gravity = [0.0, 0.0, -9.81]  # [m/s^2]
        up_axis = 1  # 0 is y, 1 is z

        class physx:
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.5  # 0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            contact_collection = (
                2  # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            )


class DragonCfgPPOD(BaseConfig):
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class MLP_Encoder:
        output_detach = True
        num_input_dim = DragonCfgFlat.env.num_observations * DragonCfgFlat.env.obs_history_length
        num_output_dim = 3
        hidden_dims = [256, 128]
        activation = "elu"
        orthogonal_init = False

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        orthogonal_init = False

    class algorithm:
        # PPO training params
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0

        # Extra training params
        est_learning_rate = 1.0e-3
        ts_learning_rate = 1.0e-4
        critic_take_latent = True

    class runner:
        encoder_class_name = "MLP_Encoder"
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        num_steps_per_env = 24  # per iteration
        max_iterations = 15000  # number of policy updates

        # logging
        logger = "tensorboard"
        exptid = ""
        wandb_project = "legged_gym_Dragon"
        save_interval = 400  # check for potential saves every this many iterations
        experiment_name = robot_type
        run_name = ""
        # load and resume
        resume = False
        load_run = "-1"  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = "None"  # updated from load_run and chkpt
