"""Compare PE01/PE02 v1 compatibility CPU paths and a PE02 collision-only control.

Run without rendering or concurrent benchmarks. No production asset is changed.
The old-collision control uses one visual-mesh convex hull per robot body while
keeping the current PE02 mechanics, scene, home pose and training configuration.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import statistics
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from unilab.adapters.pe01_legacy import PE01EncoderPolicy
from unilab.adapters.pe01_legacy import train_minimal as train_pe01
from unilab.adapters.pe02_ppo import train_minimal as train_pe02
from unilab.algos.torch.pe02 import PE02EncoderPolicy
from unilab.envs.locomotion.pe01 import PE01Env
from unilab.envs.locomotion.pe02 import PE02Env
from unilab.envs.locomotion.pe02.config import load_legacy_config as load_config

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/unilab/assets/robots"


def collision_control(destination: Path) -> None:
    asset = ASSETS / "pe02"
    root = ET.parse(asset / "pe02.xml").getroot()
    root.find("compiler").set("meshdir", str(asset / "runtime_meshes"))
    for body in root.findall(".//body"):
        visual = next(geom for geom in body.findall("geom") if geom.get("group") == "2")
        for geom in list(body.findall("geom")):
            if geom.get("group") == "3":
                body.remove(geom)
        collision = copy.deepcopy(visual)
        collision.attrib.update(
            name=f"{body.get('name')}_legacy_collision",
            group="3",
            contype="1",
            conaffinity="1",
            density="0",
        )
        body.append(collision)
    used_meshes = {geom.get("mesh") for geom in root.findall(".//geom")}
    for mesh in list(root.find("asset")):
        if mesh.get("name") not in used_meshes:
            root.find("asset").remove(mesh)
    for element in ET.parse(asset / "scene.xml").getroot():
        if element.tag == "worldbody":
            root.find("worldbody").extend(copy.deepcopy(list(element)))
        elif element.tag != "include":
            root.append(copy.deepcopy(element))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(destination, encoding="unicode")


def summarize(samples: list[float], count: int) -> dict:
    median = statistics.median(samples)
    return {
        "seconds": samples,
        "median_seconds": median,
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "count": count,
        "units_per_second": count / median,
        "microseconds_per_unit": median * 1e6 / count,
    }


def time_call(function) -> float:
    gc.collect()
    start = time.perf_counter()
    function()
    return time.perf_counter() - start


def physics_rollout(env, actions: np.ndarray, *, telemetry: bool = False) -> dict:
    env.reset()
    model, data = env.model, env.data
    contacts, resets = [], 0
    for index, action in enumerate(actions):
        data.ctrl[:] = np.clip(action, -1, 1)
        for _ in range(8):
            mujoco.mj_step(model, data)
        if telemetry:
            contacts.append(int(data.ncon))
        if data.qpos[2] < 0.08 or (index + 1) % 128 == 0:
            env.reset()
            resets += 1
    if not np.isfinite(data.qpos).all() or sum(w.number for w in data.warning):
        raise RuntimeError("Non-finite state or MuJoCo warning in benchmark")
    return {
        "resets": resets,
        "mean_contacts_at_action_boundary": float(np.mean(contacts)) if contacts else None,
        "max_contacts_at_action_boundary": max(contacts) if contacts else None,
    }


def environment_rollout(env, actions: np.ndarray) -> None:
    env.reset()
    for index, action in enumerate(actions):
        observation, reward, done, _ = env.step(action)
        if done or (index + 1) % 128 == 0:
            env.reset()
    if not np.isfinite(observation.actor).all() or not np.isfinite(reward):
        raise RuntimeError("Non-finite environment result")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory inside repository"
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--env-steps", type=int, default=4096)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("--output must be inside the repository for PE02 model-path validation")
    if args.repeats < 1 or args.env_steps < 1:
        parser.error("--repeats and --env-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    old_path = args.output.resolve() / "pe02_old_collision_control.xml"
    collision_control(old_path)
    paths = {
        "pe01": ASSETS / "pe01/scene.xml",
        "pe02": ASSETS / "pe02/scene.xml",
        "pe02_old_collision": old_path,
    }
    configs = {
        name: load_config([f"env.model_path={path}"])
        for name, path in paths.items()
        if name != "pe01"
    }
    envs = {
        name: PE01Env(path) if name == "pe01" else PE02Env(config=configs[name])
        for name, path in paths.items()
    }
    for field in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_iquat",
        "body_pos",
        "body_quat",
        "jnt_range",
        "jnt_axis",
        "actuator_gear",
        "dof_armature",
        "dof_damping",
        "key_qpos",
    ):
        np.testing.assert_array_equal(
            getattr(envs["pe02"].model, field),
            getattr(envs["pe02_old_collision"].model, field),
        )
    cpu_info = Path("/proc/cpuinfo").read_text()
    report = {
        "metadata": {
            "utc_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "cpu": next(
                line.split(":", 1)[1].strip()
                for line in cpu_info.splitlines()
                if line.startswith("model name")
            ),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "torch": torch.__version__,
            "default_torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "start_load_average": os.getloadavg(),
            "rendering": False,
            "device": "cpu",
            "physics_hz": 400,
            "policy_hz": 50,
            "repeats": args.repeats,
            "env_steps": args.env_steps,
            "action_seed": 20260916,
            "training_seed": 1,
            "raw_and_env_horizon": 128,
        },
        "models": {},
        "pe02_config": OmegaConf.to_container(load_config(), resolve=True),
        "measurements": {},
    }
    policies = {"pe01": PE01EncoderPolicy(), "pe02": PE02EncoderPolicy(configs["pe02"])}
    report["network_parameter_counts"] = {
        name: sum(parameter.numel() for parameter in policy.parameters())
        for name, policy in policies.items()
    }
    source_paths = [Path(__file__), ROOT / "conf/pe02/config.yaml"]
    for name, env in envs.items():
        model = env.model
        active = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
        robot = model.geom_bodyid != 0
        report["models"][name] = {
            "ngeom": model.ngeom,
            "robot_collision_geoms": int(np.count_nonzero(active & robot)),
            "nmesh": model.nmesh,
            "mesh_vertices": model.nmeshvert,
            "mass_kg": float(model.body_mass.sum()),
            "nq": model.nq,
            "nv": model.nv,
            "nu": model.nu,
        }
    for folder in (
        "src/unilab/assets/robots/pe01",
        "src/unilab/assets/robots/pe02",
        "src/unilab/envs/locomotion/pe01",
        "src/unilab/envs/locomotion/pe02",
        "src/unilab/algos/torch/pe02",
        "conf/pe02",
    ):
        source_paths.extend(
            p
            for p in (ROOT / folder).rglob("*")
            if p.suffix.lower() in {".py", ".xml", ".stl", ".yaml"}
        )
    source_paths.extend(
        ROOT / name
        for name in (
            "src/unilab/adapters/pe01_legacy.py",
            "src/unilab/adapters/pe02_ppo.py",
            "src/unilab/base/backend/mujoco/single_robot.py",
        )
    )
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(source_paths))
    }

    def record(key, samples, count):
        report["measurements"][key] = {
            name: summarize(values, count) for name, values in samples.items()
        }
        (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            key,
            {name: round(statistics.median(values), 6) for name, values in samples.items()},
            flush=True,
        )

    actions = np.random.default_rng(20260916).normal(size=(args.env_steps, 6)).astype(np.float32)
    report["rollout_telemetry"] = {
        name: physics_rollout(env, actions, telemetry=True) for name, env in envs.items()
    }
    for key, function, count in (
        ("physics", physics_rollout, args.env_steps * 8),
        ("environment", environment_rollout, args.env_steps),
    ):
        for env in envs.values():
            function(env, actions[:128])
        samples = {name: [] for name in envs}
        for repeat in range(args.repeats):
            names = list(envs)
            names = names[repeat % 3 :] + names[: repeat % 3]
            for name in names:
                samples[name].append(time_call(lambda: function(envs[name], actions)))
        record(key, samples, count)

    # Imports, Hydra composition and ONNX export are excluded. Model/network
    # construction, rollout, backward, Adam and checkpoint/config writes included.
    with tempfile.TemporaryDirectory(prefix="pe-training-benchmark-") as temporary:
        for threads in (report["metadata"]["default_torch_threads"], 1):
            torch.set_num_threads(threads)
            for steps in (128, 1024):
                for config in configs.values():
                    config.training.steps = steps

                def train(name):
                    output = Path(temporary) / name
                    if name == "pe01":
                        result = train_pe01(output, steps=steps)
                    else:
                        result = train_pe02(output, config=configs[name])
                    if not np.isfinite(result.mean_reward):
                        raise RuntimeError("Non-finite training reward")

                for name in envs:
                    train(name)
                samples = {name: [] for name in envs}
                for repeat in range(args.repeats):
                    names = list(envs)
                    names = names[repeat % 3 :] + names[: repeat % 3]
                    for name in names:
                        samples[name].append(time_call(lambda: train(name)))
                record(f"training_{steps}_steps_{threads}_threads", samples, steps)
    report["metadata"]["end_load_average"] = os.getloadavg()
    (args.output / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
