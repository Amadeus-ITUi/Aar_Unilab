"""Typed, independent selection contracts for robots, tasks and algorithms.

The catalog is deliberately metadata-only. Existing WE11 classes remain the
owners of their behavior; adapters resolve these IDs to the established Hydra
configuration rather than duplicating rewards or observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True)
class RobotSpec:
    id: str
    asset: str
    joints: tuple[str, ...]
    actuators: tuple[str, ...]
    capabilities: frozenset[str] = frozenset()
    metadata: Mapping[str, object] = field(default_factory=dict)
    scene: str = ""
    asset_version: str = ""
    runtime_assets: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    id: str
    robot_ids: frozenset[str]
    owner_config: str
    runtime_name: str
    capabilities: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ObservationSpec:
    id: str
    actor_groups: tuple[str, ...]
    critic_groups: tuple[str, ...]
    history: int
    actor_dim: int | None = None
    critic_dim: int | None = None
    adapter_overrides: tuple[str, ...] = ()
    robot_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class PolicySpec:
    id: str
    architecture: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    stateful: bool = False
    robot_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    adapter: str
    runner: str
    policy_ids: frozenset[str]
    entrypoint: str = ""


@dataclass(frozen=True)
class SimulatorSpec:
    id: str
    backend: str
    capabilities: frozenset[str]


@dataclass(frozen=True)
class Selection:
    robot: RobotSpec
    task: TaskSpec
    observation: ObservationSpec
    policy: PolicySpec
    algorithm: AlgorithmSpec
    simulator: SimulatorSpec


class Catalog:
    def __init__(self) -> None:
        self.robots: dict[str, RobotSpec] = {}
        self.tasks: dict[str, TaskSpec] = {}
        self.observations: dict[str, ObservationSpec] = {}
        self.policies: dict[str, PolicySpec] = {}
        self.algorithms: dict[str, AlgorithmSpec] = {}
        self.simulators: dict[str, SimulatorSpec] = {}

    def resolve(self, values: Mapping[str, str]) -> Selection:
        selected = Selection(
            robot=self._get(self.robots, "robot", values["robot"]),
            task=self._get(self.tasks, "task", values["task"]),
            observation=self._get(self.observations, "observation", values["observation"]),
            policy=self._get(self.policies, "policy", values["policy"]),
            algorithm=self._get(self.algorithms, "algorithm", values["algorithm"]),
            simulator=self._get(self.simulators, "simulator", values["simulator"]),
        )
        if selected.robot.id not in selected.task.robot_ids:
            raise ValueError(
                f"task={selected.task.id!r} does not support robot={selected.robot.id!r}"
            )
        if (
            selected.observation.robot_ids
            and selected.robot.id not in selected.observation.robot_ids
        ):
            raise ValueError(
                f"observation={selected.observation.id!r} does not support "
                f"robot={selected.robot.id!r}"
            )
        if selected.policy.id not in selected.algorithm.policy_ids:
            raise ValueError(
                f"algorithm={selected.algorithm.id!r} does not support policy={selected.policy.id!r}"
            )
        if selected.policy.robot_ids and selected.robot.id not in selected.policy.robot_ids:
            raise ValueError(
                f"policy={selected.policy.id!r} does not support robot={selected.robot.id!r}"
            )
        return selected

    @staticmethod
    def _get(items: Mapping[str, object], kind: str, key: str):
        try:
            return items[key]
        except KeyError as exc:
            choices = ", ".join(sorted(items))
            raise ValueError(f"unknown {kind}={key!r}; available: {choices}") from exc

    def view(self) -> Mapping[str, Mapping[str, object]]:
        return MappingProxyType(
            {
                "robot": MappingProxyType(self.robots),
                "task": MappingProxyType(self.tasks),
                "observation": MappingProxyType(self.observations),
                "policy": MappingProxyType(self.policies),
                "algorithm": MappingProxyType(self.algorithms),
                "simulator": MappingProxyType(self.simulators),
            }
        )


catalog = Catalog()

_we11_joints = (
    "left_hip_joint",
    "left_thigh_joint",
    "left_calf_joint",
    "right_hip_joint",
    "right_thigh_joint",
    "right_calf_joint",
)
catalog.robots["we11"] = RobotSpec(
    id="we11",
    asset="src/unilab/assets/robots/dr002/we11/we11.xml",
    joints=_we11_joints,
    actuators=_we11_joints,
    capabilities=frozenset({"wheels", "legs", "getup"}),
)
catalog.robots["pe01"] = RobotSpec(
    id="pe01",
    asset="src/unilab/assets/robots/pe01/dragon_3/urdf/v_end.urdf",
    joints=_we11_joints,
    actuators=("left_hip", "left_thigh", "left_calf", "right_hip", "right_thigh", "right_calf"),
    capabilities=frozenset({"legs", "legacy-custom-ppo"}),
    metadata={"source_asset_name": "DRAGON_3", "status": "mechanical-validation"},
    scene="src/unilab/assets/robots/pe01/scene.xml",
    asset_version="dragon-3-final-20260915",
    runtime_assets=("pe01.xml", "scene.xml"),
)
catalog.robots["pe02"] = RobotSpec(
    id="pe02",
    asset="src/unilab/assets/robots/pe02/urdf/pe02.urdf",
    joints=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    actuators=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    capabilities=frozenset({"legs", "legacy-custom-ppo"}),
    metadata={"source_asset_name": "dianzu23", "training_profile": "pe02"},
    scene="src/unilab/assets/robots/pe02/scene.xml",
    asset_version="dianzu23-collision-v2-20260916",
    runtime_assets=("pe02.xml", "scene.xml", "runtime_meshes", "collision_meshes"),
)
for task_id, owner, runtime in (
    ("flat", "dr002_joystick_flat_we11", "DR002JoystickFlatWE11"),
    ("rough", "dr002_joystick_rough_we11", "DR002JoystickRoughWE11"),
    ("getup", "dr002_joystick_getup_we11", "DR002JoystickGetupWE11"),
):
    catalog.tasks[task_id] = TaskSpec(task_id, frozenset({"we11"}), owner, runtime)
catalog.tasks["pe01_flat"] = TaskSpec(
    "pe01_flat", frozenset({"pe01"}), "pe01/task/pe01_flat", "PE01Flat"
)
catalog.tasks["pe02_flat"] = TaskSpec(
    "pe02_flat", frozenset({"pe02"}), "pe02/task/pe02_flat", "PE02Flat"
)
catalog.observations["we11_v2_145"] = ObservationSpec(
    "we11_v2_145",
    ("policy",),
    ("critic",),
    history=5,
    actor_dim=145,
    robot_ids=frozenset({"we11"}),
)
# Backward-compatible selector spelling.  Both names resolve to the current
# training contract and do not alter the tuned WE11 defaults.
catalog.observations["we11_default"] = ObservationSpec(
    "we11_default",
    ("policy",),
    ("critic",),
    history=5,
    actor_dim=145,
    robot_ids=frozenset({"we11"}),
)
catalog.observations["pe01_legacy"] = ObservationSpec(
    "pe01_legacy",
    ("proprioception",),
    ("privileged",),
    history=10,
    actor_dim=300,
    critic_dim=33,
    robot_ids=frozenset({"pe01"}),
)
catalog.observations["pe02_v1"] = ObservationSpec(
    "pe02_v1",
    ("proprioception",),
    ("privileged",),
    history=10,
    actor_dim=300,
    critic_dim=33,
    robot_ids=frozenset({"pe02"}),
)
catalog.observations["pe01_v2"] = ObservationSpec(
    "pe01_v2",
    ("actor", "frame", "command"),
    ("critic",),
    history=10,
    actor_dim=300,
    critic_dim=33,
    robot_ids=frozenset({"pe01"}),
)
catalog.observations["pe02_v2"] = ObservationSpec(
    "pe02_v2",
    ("actor", "frame", "command"),
    ("critic",),
    history=10,
    actor_dim=300,
    critic_dim=33,
    robot_ids=frozenset({"pe02"}),
)
catalog.observations["pe02_v3"] = ObservationSpec(
    "pe02_v3",
    ("actor", "frame", "command"),
    ("critic",),
    history=10,
    actor_dim=240,
    critic_dim=27,
    robot_ids=frozenset({"pe02"}),
)
catalog.policies["we11_mlp"] = PolicySpec(
    "we11_mlp", "mlp", ("obs",), ("act",), robot_ids=frozenset({"we11"})
)
catalog.policies["pe01_encoder_mlp"] = PolicySpec(
    "pe01_encoder_mlp",
    "mlp-encoder",
    ("observation_history", "observation", "command"),
    ("action",),
    robot_ids=frozenset({"pe01"}),
)
catalog.policies["pe02_encoder_mlp"] = PolicySpec(
    "pe02_encoder_mlp",
    "mlp-encoder",
    ("observation_history", "observation", "command"),
    ("action",),
    robot_ids=frozenset({"pe02"}),
)
catalog.algorithms["rsl_rl_ppo"] = AlgorithmSpec(
    "rsl_rl_ppo", "unilab.adapters.rsl_rl", "OnPolicyRunner", frozenset({"we11_mlp"})
)
catalog.algorithms["pe01_custom_ppo"] = AlgorithmSpec(
    "pe01_custom_ppo",
    "unilab.adapters.pe01_ppo",
    "PE01PPOAdapter",
    frozenset({"pe01_encoder_mlp"}),
    entrypoint="scripts/train_pe01.py",
)
catalog.algorithms["pe02_custom_ppo"] = AlgorithmSpec(
    "pe02_custom_ppo",
    "unilab.adapters.pe02_ppo",
    "PE02PPOAdapter",
    frozenset({"pe02_encoder_mlp"}),
    entrypoint="scripts/train_pe02.py",
)
# Independently owned CNC prototype training contracts.
catalog.robots["pe03"] = RobotSpec(
    id="pe03",
    asset="src/unilab/assets/robots/pe03/urdf/pe03.urdf",
    joints=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    actuators=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    capabilities=frozenset({"legs", "legacy-custom-ppo"}),
    metadata={"source_asset_name": "点足CNC", "training_profile": "pe03"},
    scene="src/unilab/assets/robots/pe03/scene.xml",
    asset_version="cnc-collision-v1-20260918",
    runtime_assets=(
        "pe03.xml",
        "pe03_joint_limits.xml",
        "scene_joint_limits.xml",
        "scene.xml",
        "play_visual.xml",
        "runtime_meshes",
        "collision_meshes",
    ),
)
catalog.tasks["pe03_flat"] = TaskSpec(
    "pe03_flat", frozenset({"pe03"}), "pe03/task/pe03_flat", "PE03Flat"
)
catalog.tasks["pe03_gait_flat"] = TaskSpec(
    "pe03_gait_flat", frozenset({"pe03"}), "pe03/task/pe03_gait_flat", "PE03GaitFlat"
)
catalog.observations["pe03_v4"] = ObservationSpec(
    "pe03_v4",
    ("actor", "frame", "command"),
    ("critic",),
    history=30,
    actor_dim=1140,
    critic_dim=14,
    robot_ids=frozenset({"pe03"}),
)
catalog.policies["pe03_history_velocity_mlp"] = PolicySpec(
    "pe03_history_velocity_mlp",
    "mlp-history-velocity",
    ("observation_history", "observation", "command"),
    ("action",),
    robot_ids=frozenset({"pe03"}),
)
catalog.observations["pe03_v2"] = ObservationSpec(
    "pe03_v2",
    ("actor", "frame", "command"),
    ("critic",),
    history=10,
    actor_dim=300,
    critic_dim=33,
    robot_ids=frozenset({"pe03"}),
)
catalog.observations["pe03_v3"] = ObservationSpec(
    "pe03_v3",
    ("actor", "frame", "command"),
    ("critic",),
    history=10,
    actor_dim=240,
    critic_dim=27,
    robot_ids=frozenset({"pe03"}),
)
catalog.policies["pe03_encoder_mlp"] = PolicySpec(
    "pe03_encoder_mlp",
    "mlp-encoder",
    ("observation_history", "observation", "command"),
    ("action",),
    robot_ids=frozenset({"pe03"}),
)
catalog.algorithms["pe03_custom_ppo"] = AlgorithmSpec(
    "pe03_custom_ppo",
    "unilab.adapters.pe03_ppo",
    "PE03PPOAdapter",
    frozenset({"pe03_encoder_mlp", "pe03_history_velocity_mlp"}),
    entrypoint="scripts/train_pe03.py",
)

# PE04 owns its complete training and runtime implementation.
catalog.robots["pe04"] = RobotSpec(
    id="pe04",
    asset="src/unilab/assets/robots/pe04/urdf/pe04.urdf",
    joints=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    actuators=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    capabilities=frozenset({"legs"}),
    scene="src/unilab/assets/robots/pe04/scene.xml",
    asset_version="pe04-pe03-cnc-joint-limits-v3",
    runtime_assets=(
        "pe04.xml",
        "scene.xml",
        "play_visual.xml",
        "runtime_meshes",
        "collision_meshes",
        "provenance.json",
    ),
)
catalog.tasks["pe04_flat"] = TaskSpec(
    "pe04_flat", frozenset({"pe04"}), "pe04/task/pe04_flat", "PE04Flat"
)
catalog.observations["pe04_tron1_v1"] = ObservationSpec(
    "pe04_tron1_v1",
    ("actor", "frame", "command"),
    ("critic",),
    history=10,
    actor_dim=300,
    critic_dim=267,
    robot_ids=frozenset({"pe04"}),
)
catalog.policies["pe04_encoder_mlp"] = PolicySpec(
    "pe04_encoder_mlp",
    "mlp-encoder",
    ("observation_history", "observation", "command"),
    ("action",),
    robot_ids=frozenset({"pe04"}),
)
catalog.algorithms["pe04_custom_ppo"] = AlgorithmSpec(
    "pe04_custom_ppo",
    "unilab.adapters.pe04_ppo",
    "PE04PPOAdapter",
    frozenset({"pe04_encoder_mlp"}),
    entrypoint="scripts/train_pe04.py",
)

# PE05 owns its complete training and runtime implementation.
catalog.robots["pe05"] = RobotSpec(
    id="pe05",
    asset="src/unilab/assets/robots/pe05/urdf/pe05.urdf",
    joints=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    actuators=("L_hip_", "L_thigh_", "L_calf_", "R_hip_", "R_thigh_", "R_calf_"),
    capabilities=frozenset({"legs"}),
    scene="src/unilab/assets/robots/pe05/scene.xml",
    asset_version="pe05-pe03-cnc-joint-limits-v3",
    runtime_assets=(
        "pe05.xml",
        "scene.xml",
        "play_visual.xml",
        "runtime_meshes",
        "collision_meshes",
        "provenance.json",
    ),
)
catalog.tasks["pe05_flat"] = TaskSpec(
    "pe05_flat", frozenset({"pe05"}), "pe05/task/pe05_flat", "PE05Flat"
)
catalog.observations["pe05_v1"] = ObservationSpec(
    "pe05_v1",
    ("actor", "frame", "command"),
    ("critic",),
    history=10,
    actor_dim=300,
    critic_dim=33,
    robot_ids=frozenset({"pe05"}),
)
catalog.policies["pe05_encoder_mlp"] = PolicySpec(
    "pe05_encoder_mlp",
    "mlp-encoder",
    ("observation_history", "observation", "command"),
    ("action",),
    robot_ids=frozenset({"pe05"}),
)
catalog.algorithms["pe05_custom_ppo"] = AlgorithmSpec(
    "pe05_custom_ppo",
    "unilab.adapters.pe05_ppo",
    "PE05PPOAdapter",
    frozenset({"pe05_encoder_mlp"}),
    entrypoint="scripts/train_pe05.py",
)

catalog.simulators["mujoco"] = SimulatorSpec(
    "mujoco", "mujoco", frozenset({"headless", "render", "mouse-force", "camera"})
)


def repository_path(relative: str, root: Path) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"catalog path escapes repository: {relative}")
    return path
