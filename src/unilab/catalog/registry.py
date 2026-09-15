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


@dataclass(frozen=True)
class PolicySpec:
    id: str
    architecture: str
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    stateful: bool = False


@dataclass(frozen=True)
class AlgorithmSpec:
    id: str
    adapter: str
    runner: str
    policy_ids: frozenset[str]


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
            raise ValueError(f"task={selected.task.id!r} does not support robot={selected.robot.id!r}")
        if selected.policy.id not in selected.algorithm.policy_ids:
            raise ValueError(
                f"algorithm={selected.algorithm.id!r} does not support policy={selected.policy.id!r}"
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
    "left_hip_joint", "left_thigh_joint", "left_calf_joint",
    "right_hip_joint", "right_thigh_joint", "right_calf_joint",
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
    joints=(),
    actuators=(),
    capabilities=frozenset({"legs", "legacy-custom-ppo"}),
    metadata={"source_asset_name": "DRAGON_3", "status": "mechanical-validation"},
)
for task_id, owner, runtime in (
    ("flat", "dr002_joystick_flat_we11", "DR002JoystickFlatWE11"),
    ("rough", "dr002_joystick_rough_we11", "DR002JoystickRoughWE11"),
    ("getup", "dr002_joystick_getup_we11", "DR002JoystickGetupWE11"),
):
    catalog.tasks[task_id] = TaskSpec(task_id, frozenset({"we11"}), owner, runtime)
catalog.tasks["pe01_flat"] = TaskSpec(
    "pe01_flat", frozenset({"pe01"}), "pe01_flat", "PE01Flat", frozenset({"legacy-semantics"})
)
catalog.observations["we11_default"] = ObservationSpec(
    "we11_default", ("policy",), ("critic",), history=5, actor_dim=145
)
catalog.observations["pe01_legacy"] = ObservationSpec(
    "pe01_legacy", ("proprioception",), ("privileged",), history=10, actor_dim=300,
    critic_dim=33
)
catalog.policies["we11_mlp"] = PolicySpec("we11_mlp", "mlp", ("obs",), ("act",))
catalog.policies["pe01_encoder_mlp"] = PolicySpec(
    "pe01_encoder_mlp", "mlp-encoder", ("observation_history",), ("action",)
)
catalog.algorithms["rsl_rl_ppo"] = AlgorithmSpec(
    "rsl_rl_ppo", "unilab.adapters.rsl_rl", "OnPolicyRunner", frozenset({"we11_mlp"})
)
catalog.algorithms["pe01_custom_ppo"] = AlgorithmSpec(
    "pe01_custom_ppo",
    "unilab.adapters.pe01_legacy",
    "LegacyPPOAdapter",
    frozenset({"pe01_encoder_mlp"}),
)
catalog.simulators["mujoco"] = SimulatorSpec(
    "mujoco", "mujoco", frozenset({"headless", "render", "mouse-force", "camera"})
)


def repository_path(relative: str, root: Path) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents and path != root.resolve():
        raise ValueError(f"catalog path escapes repository: {relative}")
    return path
