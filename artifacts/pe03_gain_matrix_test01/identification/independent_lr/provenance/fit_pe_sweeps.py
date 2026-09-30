"""Offline fixed-base PE03 sweep identification; never connects to hardware.

Uses A/B for estimation and reserves C for final validation. Outputs are
experimental artifacts, not automatically applied training parameters.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import subprocess
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
JOINTS = [f"{side}_{group}_" for side in ("L", "R") for group in ("hip", "thigh", "calf")]
PORTS = [4, 5, 6, 1, 2, 3]
GROUPS = ["hip", "thigh", "calf"]
BASE = np.array([0.01, 0.01, 0.0193715, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0, 0])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(source, out):
    tree = ET.parse(source)
    root = tree.getroot()
    comp = root.find("compiler")
    meshdir = source.parent / comp.get("meshdir", "")
    comp.set("meshdir", str(meshdir.resolve()))
    for parent in root.iter():
        for item in list(parent):
            if item.tag == "freejoint":
                parent.remove(item)
    root.remove(root.find("actuator"))  # apply Nm directly; no duplicate gear factor
    # Fixing the root welds it to world. MuJoCo's automatic parent filter no
    # longer filters world-child pairs; explicitly preserve the original
    # root/hip exclusions rather than introducing artificial bracket contacts.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for side in ("L", "R"):
        ET.SubElement(contact, "exclude", body1="body_link", body2=f"{side}_hip_Link")
    root.find("option").set("timestep", "0.002")
    target = out / "fixture.xml"
    tree.write(target, encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(target))
    assert model.nq == model.nv == 6
    assert [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(6)] == JOINTS
    mujoco.mj_saveModel(model, str(out / "fixture.mjb"), None)
    assets = [source] + [(meshdir / x.get("file")).resolve() for x in root.findall("asset/mesh")]
    return model, {str(p): digest(p) for p in assets}


def native_library(out):
    package = Path(mujoco.__file__).parent
    lib = next(package.glob("libmujoco.so.*"))
    binary = out / "pe_replay.so"
    subprocess.run(
        [
            "g++",
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(Path(__file__).with_name("pe_replay.cpp")),
            "-I" + str(package / "include"),
            str(lib),
            "-Wl,-rpath," + str(package),
            "-o",
            str(binary),
        ],
        check=True,
    )
    dll = ctypes.CDLL(str(binary))
    ptr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    dll.pe_create.argtypes = [ctypes.c_char_p]
    dll.pe_create.restype = ctypes.c_void_p
    dll.pe_destroy.argtypes = [ctypes.c_void_p]
    dll.pe_run.argtypes = [
        ctypes.c_void_p,
        ptr,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ptr,
        ptr,
        ptr,
        ptr,
        ctypes.c_int,
        ptr,
        ptr,
        ctypes.c_int,
        ctypes.c_int,
    ]
    dll.pe_run.restype = ctypes.c_int
    dll.pe_run_independent.argtypes = dll.pe_run.argtypes
    dll.pe_run_independent.restype = ctypes.c_int
    return dll


class Trial:
    def __init__(self, path, quality, dll, out):
        self.name = path.stem
        self.config = yaml.safe_load(path.with_suffix(".yaml").read_text())
        if not self.config["completed"] or self.config["mode"] != "position":
            raise ValueError(f"Incomplete or unsupported sweep: {path}")
        self.group = GROUPS.index(self.config["group_id"])
        self.selected = [self.group, self.group + 3]
        raw = np.genfromtxt(path, delimiter=",", names=True)
        assert digest(path) == quality["csv_sha256"]
        assert digest(path.with_suffix(".yaml")) == quality["yaml_sha256"]
        t = (raw["host_monotonic_ns"] - raw["host_monotonic_ns"][0]) * 1e-9
        start, end = quality["sweep_start_host_s"], quality["sweep_end_host_s"]
        formal = (t >= start) & (t < end)
        assert np.all(raw["command_status_flags"][formal] == 1)
        keep = (raw["command_status_flags"] == 1) & (t <= end)
        initial = t[keep][0]
        self.times = np.ascontiguousarray(t[keep] - initial)
        if not np.all(np.diff(self.times) > 0):
            raise ValueError(f"Non-monotonic host timestamps: {path}")
        self.start = start - initial
        self.end = end - initial
        self.raw = raw[keep]
        self.commands = np.ascontiguousarray(
            np.column_stack(
                [self.channel(field) for field in ["cmd_q", "cmd_dq", "kp", "kd", "cmd_tau"]]
            )
        )
        self.q = self.channel("q")
        self.dq = self.channel("dq")
        if not all(np.isfinite(a).all() for a in (self.commands, self.q, self.dq)):
            raise ValueError(f"Non-finite command or joint state: {path}")
        self.q0, self.dq0 = self.q[0].copy(), self.dq[0].copy()
        self.samples = np.ascontiguousarray(np.arange(0, self.end, 0.02))
        self.measured = np.column_stack(
            [np.interp(self.samples, self.times, self.q[:, j]) for j in range(6)]
        )
        self.measured_dq = np.column_stack(
            [np.interp(self.samples, self.times, self.dq[:, j]) for j in range(6)]
        )
        self.mask = (self.samples >= self.start + 0.2) & (self.samples < self.end)
        self.dll = dll
        self.handle = dll.pe_create(str(out / "fixture.mjb").encode())
        if not self.handle:
            raise RuntimeError("Cannot create six-joint fixture")
        self.amplitude = self.config["targets"][0]["amplitude"]
        p = self.channel("kp") * (self.channel("cmd_q") - self.q)
        d = self.channel("kd") * (self.channel("cmd_dq") - self.dq)
        demand = abs(p) + abs(d) + abs(self.channel("cmd_tau"))
        limit = np.array([5.5, 5.5, 14] * 2)
        active = self.times >= self.start
        self.audit = {
            "name": self.name,
            "csv_sha256": digest(path),
            "yaml_sha256": digest(path.with_suffix(".yaml")),
            "rows_formal": int(formal.sum()),
            "start_host_s": start,
            "end_host_s": end,
            "kp": self.config["kp"],
            "kd": self.config["kd"],
            "indicative_scale_fraction_by_joint": np.mean(demand[active] > limit, axis=0).tolist(),
            "indicative_max_demand_nm": np.max(demand[active], axis=0).tolist(),
            "record_interval_ms_percentiles": (
                np.percentile(np.diff(t[formal]), [0, 50, 95, 100]) * 1000
            ).tolist(),
            "actual_min_q": self.q[active].min(axis=0).tolist(),
            "actual_max_q": self.q[active].max(axis=0).tolist(),
            "formal_kp_min": self.channel("kp")[active].min(axis=0).tolist(),
            "formal_kp_max": self.channel("kp")[active].max(axis=0).tolist(),
            "formal_kd_min": self.channel("kd")[active].min(axis=0).tolist(),
            "formal_kd_max": self.channel("kd")[active].max(axis=0).tolist(),
            "max_abs_command_dq": float(abs(self.channel("cmd_dq")[active]).max()),
            "max_abs_feedforward_tau": float(abs(self.channel("cmd_tau")[active]).max()),
            "source_seq_equals_current_state_fraction": float(
                np.mean(raw["source_state_sample_seq"][formal] == raw["state_sample_seq"][formal])
            ),
        }

    def channel(self, field):
        return np.column_stack(
            [self.raw[f"p{p}_{j}_{field}"] for p, j in zip(PORTS, JOINTS, strict=True)]
        )

    def replay(self, par, dt=0.002, control_dt=0.002, legacy=False, contacts=True):
        par = np.ascontiguousarray(par, dtype=np.float64)
        if par.shape not in ((12,), (24,)) or not np.isfinite(par).all():
            raise ValueError("Expected 12 grouped or 24 independent finite parameters")
        run = self.dll.pe_run if len(par) == 12 else self.dll.pe_run_independent
        result = np.zeros((len(self.samples), 25))
        status = run(
            self.handle,
            par,
            dt,
            control_dt,
            len(self.times),
            self.times,
            self.commands,
            self.q0,
            self.dq0,
            len(self.samples),
            self.samples,
            result,
            int(legacy),
            int(contacts),
        )
        if status:
            raise RuntimeError(f"Replay diverged {self.name}: {status}")
        return result

    def close(self):
        self.dll.pe_destroy(self.handle)


def decode(x, base, group):
    p = base.copy()
    p[group] = 10 ** np.clip(x[0], -5, -0.7)
    p[3 + group] = np.clip(x[1], 0, 0.8)
    p[6 + group] = np.clip(x[2], 0, 0.6)
    p[9 + group] = np.clip(x[3], 0, 0.06)
    return p


def metrics(trial, result):
    error = (result[:, :6] - trial.measured)[trial.mask]
    velocity_error = (result[:, 6:12] - trial.measured_dq)[trial.mask]
    return {
        "q_rmse_rad": np.sqrt(np.mean(error**2, axis=0)).tolist(),
        "q_bias_rad": np.mean(error, axis=0).tolist(),
        "dq_rmse_rad_s": np.sqrt(np.mean(velocity_error**2, axis=0)).tolist(),
        "selected_q_nrmse_amplitude": float(
            np.sqrt(np.mean(error[:, trial.selected] ** 2)) / trial.amplitude
        ),
        "contact_sample_fraction": float(np.mean(result[trial.mask, 24] > 0)),
        "max_contacts": float(result[trial.mask, 24].max()),
        "scale_fraction": np.mean(result[trial.mask, 18:24] < 1 - 1e-8, axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/pe03_gain_matrix_test01")
    parser.add_argument("--maxiter", type=int, default=24)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    out = args.data / "identification"
    out.mkdir(parents=True, exist_ok=True)
    model, hashes = fixture(ROOT / "src/unilab/assets/robots/pe03/pe03_joint_limits.xml", out)
    dll = native_library(out)
    quality = json.loads((args.data / "visualizations/data_quality.json").read_text())
    expected = [
        f"{i * 3 + j + 1:02d}_{gain}_{group}"
        for i, gain in enumerate(["A_baseline", "B_position", "C_damping"])
        for j, group in enumerate(GROUPS)
    ]
    if [q["run"] for q in quality] != expected:
        raise ValueError("Expected the complete ordered nine-run gain matrix")
    trials = [Trial(args.data / (q["run"] + ".csv"), q, dll, out) for q in quality]
    audit = []
    for trial in trials:
        data = mujoco.MjData(model)
        ncon = []
        for row in trial.q[::20]:
            data.qpos[:] = row
            mujoco.mj_forward(model, data)
            ncon.append(data.ncon)
        trial.audit["measured_pose_contact_fraction"] = float(np.mean(np.array(ncon) > 0))
        audit.append(trial.audit)
    provenance = {
        "mujoco": mujoco.__version__,
        "model_and_mesh_sha256": hashes,
        "runner_sha256": digest(__file__),
        "native_sha256": digest(Path(__file__).with_name("pe_replay.cpp")),
        "joint_order": JOINTS,
        "invocation": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "ports": PORTS,
        "trials": audit,
        "fixture": "fixed upright body; suspended feet; original self collisions and limits retained",
        "timing": "host receive/publish timestamps; ZOH reconstructed commands; not wire execution times",
    }
    (out / "audit.json").write_text(json.dumps(provenance, indent=2))
    par = BASE.copy()
    if args.resume and (out / "checkpoint.json").exists():
        par = np.maximum(
            0, np.array(json.loads((out / "checkpoint.json").read_text())["parameters"])
        )
    pool = ThreadPoolExecutor(max_workers=2)
    baseline = {}
    for trial in trials:
        t0 = time.monotonic()
        result = trial.replay(BASE)
        baseline[trial.name] = metrics(trial, result)
        np.savez_compressed(
            out / (trial.name + "_baseline.npz"),
            time=trial.samples - trial.start,
            replay=result,
            q=trial.measured,
            dq=trial.measured_dq,
        )
        print(
            "BASE",
            trial.name,
            baseline[trial.name]["selected_q_nrmse_amplitude"],
            "seconds",
            round(time.monotonic() - t0, 3),
            flush=True,
        )
    (out / "baseline_metrics.json").write_text(json.dumps(baseline, indent=2))
    if args.audit_only:
        for trial in trials:
            trial.close()
        pool.shutdown()
        return
    history = []
    for cycle in range(args.passes):
        for group in range(3):
            train = [t for t in trials[:6] if t.group == group]
            calls = 0
            best = [float("inf"), None]
            t0 = time.monotonic()

            def objective(x):
                nonlocal calls
                candidate = decode(x, par, group)
                results = list(
                    pool.map(lambda trial: trial.replay(candidate, contacts=False), train)
                )
                value = float(
                    np.mean(
                        [
                            np.mean(
                                (
                                    (res[:, :6] - trial.measured)[trial.mask][:, trial.selected]
                                    / trial.amplitude
                                )
                                ** 2
                            )
                            for trial, res in zip(train, results, strict=True)
                        ]
                    )
                )
                calls += 1
                if value < best[0]:
                    best[:] = [value, candidate.tolist()]
                if calls % 50 == 0:
                    print(
                        "FIT",
                        cycle,
                        GROUPS[group],
                        calls,
                        "best",
                        round(best[0], 6),
                        "elapsed",
                        round(time.monotonic() - t0),
                        flush=True,
                    )
                return value

            initial = np.array(
                [np.log10(par[group]), par[3 + group], par[6 + group], par[9 + group]]
            )
            initial_loss = objective(initial)
            fitted = minimize(
                objective,
                initial,
                method="Powell",
                bounds=[(-5, -0.7), (0, 0.8), (0, 0.6), (0, 0.06)],
                options={"maxiter": args.maxiter, "xtol": 2e-4, "ftol": 2e-4},
            )
            par = np.array(best[1])
            history.append(
                {
                    "pass": cycle,
                    "group": GROUPS[group],
                    "initial_loss": initial_loss,
                    "loss": best[0],
                    "calls": calls,
                    "success": bool(fitted.success),
                    "message": str(fitted.message),
                    "parameters": par.tolist(),
                }
            )
            (out / "checkpoint.json").write_text(
                json.dumps({"parameters": par.tolist(), "history": history}, indent=2)
            )
            print("DONE", history[-1], flush=True)
    scores = {}
    for trial in trials:
        result = trial.replay(par)
        np.savez_compressed(
            out / (trial.name + "_fit.npz"),
            time=trial.samples - trial.start,
            replay=result,
            q=trial.measured,
            dq=trial.measured_dq,
        )
        scores[trial.name] = {"baseline": baseline[trial.name], "fit": metrics(trial, result)}
        print(
            "VALIDATION" if trial in trials[6:] else "TRAIN",
            trial.name,
            scores[trial.name]["fit"],
            flush=True,
        )
    candidate = {
        "status": "experimental_not_promoted",
        "fit_runs": [t.name for t in trials[:6]],
        "held_out_runs": [t.name for t in trials[6:]],
        "joint_type_order": GROUPS,
        "armature_total_kg_m2": par[:3].tolist(),
        "passive_damping_nm_s_rad": par[3:6].tolist(),
        "coulomb_frictionloss_nm": par[6:9].tolist(),
        "effective_delay_s": par[9:12].tolist(),
        "fixed_torque_scale": 1,
        "fixed_encoder_offset_rad": 0,
        "search_bounds_not_confidence_intervals": {
            "armature_total_kg_m2": [1e-5, float(10**-0.7)],
            "passive_damping_nm_s_rad": [0, 0.8],
            "coulomb_frictionloss_nm": [0, 0.6],
            "effective_delay_s": [0, 0.06],
        },
        "firmware_gain_scaling": "min(1, effort_limit / (abs(P)+abs(D)+abs(FF)))",
        "physics_dt_s": 0.002,
        "gain_scaling_update_s": 0.002,
        "limitations": [
            "No execution timestamps or motor sample age",
            "Motor internal servo approximated by instantaneous PD",
            "Uncalibrated feedback torque is diagnostic only",
            "Paired synchronous excitation cannot identify all coupling parameters",
        ],
    }
    (out / "candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False))
    (out / "metrics.json").write_text(json.dumps(scores, indent=2))
    for trial in trials:
        trial.close()
    pool.shutdown()


if __name__ == "__main__":
    main()
