# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2021 ETH Zurich, Nikita Rudin
# Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES
# See LICENSE.pe01; PE05 sole-kinematics adaptation is maintained independently.
"""PE01-shaped gait costs using PE05 sole kinematics (unweighted, before dt)."""

import numpy as np
from scipy.special import ndtr


def desired_contacts(phase, gaits, kappa):
    foot_phase = (phase[:, None] + np.column_stack((np.zeros(len(phase)), gaits[:, 1]))) % 1
    duration = gaits[:, 2:3]
    mapped = np.where(
        foot_phase < duration,
        foot_phase * 0.5 / duration,
        0.5 + (foot_phase - duration) * 0.5 / (1 - duration),
    )
    return ndtr(mapped / kappa) * (1 - ndtr((mapped - 0.5) / kappa)) + ndtr(
        (mapped - 1) / kappa
    ) * (1 - ndtr((mapped - 1.5) / kappa))


def foot_costs(feet, desired, reward):
    force = np.linalg.norm(feet["ground_force"], axis=-1)
    velocity = feet["reference_velocity"]
    height = feet["clearance"]
    landing = (
        (height < reward["about_landing_threshold"])
        & (force <= reward["landing_force_threshold"])
        & (velocity[..., 2] < 0)
    )
    # Each array is a foot's contribution to the raw whole-robot cost.
    return {
        "tracking_contacts_shaped_force": (1 - desired)
        * (-np.expm1(-(force**2) / reward["gait_force_sigma"]))
        / 2,
        "tracking_contacts_shaped_vel": desired
        * (-np.expm1(-np.square(velocity).sum(2) / reward["gait_vel_sigma"]))
        / 2,
        "feet_regulation": np.exp(-np.maximum(height, 0) / reward["feet_regulation_height_scale"])
        * np.square(velocity[..., :2]).sum(2),
        "foot_landing_vel": np.square(np.where(landing, velocity[..., 2], 0)),
    }
