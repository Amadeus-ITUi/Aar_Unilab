"""Fit six independent PE03 actuator parameter sets at the frozen shared delay.

Offline only. A/B estimate parameters; C is scored only after parameters freeze.
Two sides have distinct native handles; no handle is replayed concurrently.
The fixed-base, contact-free fitting model has mechanically independent legs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scripts.identification.fit_pe_sweeps import (
    JOINTS,
    ROOT,
    Trial,
    digest,
    metrics,
    native_library,
)


def expand(grouped):
    return np.concatenate([np.tile(block, 2) for block in np.split(grouped, 4)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/pe03_gain_matrix_test01")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--maxfev", type=int, default=220)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = args.data / "identification"
    out = source / "independent_lr"
    out.mkdir(exist_ok=True)
    started = time.time()
    comparison_path = source / "shared_delay_comparison/comparison.json"
    grouped = np.array(json.loads(comparison_path.read_text())["shared_parameters"])
    initial = expand(grouped)
    assert np.all(initial[18:] == initial[18])
    dll = native_library(out)
    shutil.copyfile(source / "fixture.mjb", out / "fixture.mjb")
    quality = json.loads((args.data / "visualizations/data_quality.json").read_text())
    assert [q["run"][3] for q in quality] == list("AAABBBCCC")

    def make_trials(rows):
        return [Trial(args.data / (q["run"] + ".csv"), q, dll, out) for q in rows]

    def fit_side(side):
        trials = make_trials(quality[:6])
        par = initial.copy()
        history = []
        checkpoint = out / f"checkpoint_{side}.json"
        first_pass = 0
        if args.resume and checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            par = np.array(saved["parameters"])
            history = saved["history"]
            first_pass = max(h["pass_index"] for h in history) + 1
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:

                def loss(candidate, selected, joint):
                    predictions = pool.map(lambda t: t.replay(candidate, contacts=False), selected)
                    return float(
                        np.mean(
                            [
                                np.mean(
                                    ((r[:, joint] - t.measured[:, joint])[t.mask] / t.amplitude)
                                    ** 2
                                )
                                for t, r in zip(selected, predictions, strict=True)
                            ]
                        )
                    )

                for cycle in range(first_pass, args.passes):
                    for group in range(3):
                        joint = group + 3 * side
                        selected = [t for t in trials if t.group == group]
                        best = [loss(par, selected, joint), par.copy()]
                        before = best[0]
                        calls = 0

                        def objective(x):
                            nonlocal calls
                            candidate = par.copy()
                            candidate[joint] = 10 ** x[0]
                            candidate[6 + joint], candidate[12 + joint] = x[1:]
                            value = loss(candidate, selected, joint)
                            calls += 1
                            if value < best[0]:
                                best[:] = [value, candidate.copy()]
                            return value

                        result = minimize(
                            objective,
                            [np.log10(par[joint]), par[6 + joint], par[12 + joint]],
                            method="Powell",
                            bounds=[(-5, -0.7), (0, 0.8), (0, 0.6)],
                            options=dict(maxfev=args.maxfev, xtol=2e-4, ftol=2e-4),
                        )
                        # Explicitly test zero boundaries; bounded line searches otherwise
                        # report tiny positive values that should not be physical claims.
                        for block in (6, 12):
                            candidate = best[1].copy()
                            candidate[block + joint] = 0.0
                            value = loss(candidate, selected, joint)
                            if value < best[0]:
                                best[:] = [value, candidate]
                        par = best[1]
                        entry = dict(
                            pass_index=cycle,
                            joint=JOINTS[joint],
                            calls=calls,
                            before_loss=before,
                            after_loss=best[0],
                            converged=bool(result.success),
                            message=str(result.message),
                            parameters=[float(par[joint + k]) for k in (0, 6, 12)],
                        )
                        history.append(entry)
                        print("REFIT", entry, flush=True)
                    checkpoint.write_text(
                        json.dumps(dict(parameters=par.tolist(), history=history), indent=2)
                    )
        finally:
            for t in trials:
                t.close()
        return par, history

    with ThreadPoolExecutor(max_workers=2) as sides:
        fitted = list(sides.map(fit_side, (0, 1)))
    par = initial.copy()
    history = []
    for side, (side_par, entries) in enumerate(fitted):
        for offset in (0, 6, 12):
            par[offset + side * 3 : offset + side * 3 + 3] = side_par[
                offset + side * 3 : offset + side * 3 + 3
            ]
        history.extend(entries)
    # Freeze before opening any C trial. No C-based optimization or selection.
    frozen = dict(parameters=par.tolist(), history=history, delay_fixed_s=float(par[18]))
    (out / "checkpoint.json").write_text(json.dumps(frozen, indent=2))
    print("FROZEN", par.tolist(), flush=True)
    trials = make_trials(quality)
    scores, invariance = {}, {}
    try:
        for t in trials:
            tied = t.replay(grouped)
            expanded = t.replay(initial)
            np.testing.assert_array_equal(tied, expanded)
            fit = t.replay(par)
            scores[t.name] = dict(tied=metrics(t, tied), independent=metrics(t, fit))
            np.savez_compressed(
                out / (t.name + ".npz"),
                time=t.samples - t.start,
                mask=t.mask,
                measured=t.measured,
                measured_dq=t.measured_dq,
                tied=tied,
                independent=fit,
            )
            # Verify parallel side fits did not silently rely on cross-leg coupling.
            if t.name[3] != "C":
                for side in (0, 1):
                    own = t.replay(fitted[side][0], contacts=False)
                    combined = t.replay(par, contacts=False)
                    columns = slice(3 * side, 3 * side + 3)
                    error = float(np.max(abs(own[:, columns] - combined[:, columns])))
                    invariance[f"{t.name}_{side}"] = error
                    if error > 1e-9:
                        raise RuntimeError("Parallel fit invalid: legs are coupled")

        def train_loss(parameters):
            with ThreadPoolExecutor(max_workers=6) as pool:
                replays = pool.map(lambda t: t.replay(parameters, contacts=False), trials[:6])
                return float(
                    np.mean(
                        [
                            np.mean(
                                ((r[:, :6] - t.measured)[t.mask][:, t.selected] / t.amplitude) ** 2
                            )
                            for t, r in zip(trials[:6], replays, strict=True)
                        ]
                    )
                )

        profile = []
        for delay in np.arange(0, 0.060001, 0.002):
            candidate = par.copy()
            candidate[18:] = delay
            profile.append([float(delay), train_loss(candidate)])
        sensitivity = []
        baseline_loss = train_loss(par)
        for joint in range(6):
            for offset, delta in (
                (0, -0.1 * par[joint]),
                (0, 0.1 * par[joint]),
                (6, 0.01),
                (12, 0.02),
            ):
                candidate = par.copy()
                candidate[offset + joint] += delta
                sensitivity.append(
                    dict(
                        joint=JOINTS[joint],
                        block=offset // 6,
                        delta=float(delta),
                        loss=train_loss(candidate),
                    )
                )
        report = dict(
            status="experimental_not_promoted",
            parameter_order="armature(6), damping(6), frictionloss(6), delay(6)",
            joint_order=JOINTS,
            tied_parameters=initial.tolist(),
            independent_parameters=par.tolist(),
            fixed_shared_delay_s=float(par[18]),
            tied_train_loss=train_loss(initial),
            independent_train_loss=baseline_loss,
            fit_runs=[t.name for t in trials[:6]],
            held_out_runs=[t.name for t in trials[6:]],
            history=history,
            metrics=scores,
            delay_profile_conditional_on_frozen_physical_parameters=profile,
            perturbation_sensitivity_not_confidence_intervals=sensitivity,
            cross_leg_invariance_max_abs_rad=invariance,
            elapsed_s=time.time() - started,
            provenance={
                "shared_comparison_sha256": digest(comparison_path),
                "fixture_sha256": digest(source / "fixture.mjb"),
                "script_sha256": digest(__file__),
                "native_source_sha256": digest(Path(__file__).with_name("pe_replay.cpp")),
                "runs": [t.audit for t in trials],
            },
        )
        (out / "comparison.json").write_text(json.dumps(report, indent=2))
        candidate = dict(
            status="experimental_not_promoted",
            joint_order=JOINTS,
            delay_model="fixed_shared_effective_delay",
            fixed_shared_delay_s=float(par[18]),
            fit_runs=report["fit_runs"],
            held_out_runs=report["held_out_runs"],
            armature_total_kg_m2=par[:6].tolist(),
            passive_damping_nm_s_rad=par[6:12].tolist(),
            coulomb_frictionloss_nm=par[12:18].tolist(),
            effective_delay_s=par[18:].tolist(),
            fixed_torque_scale=1,
            fixed_encoder_offset_rad=0,
            physics_dt_s=0.002,
            gain_scaling_update_s=0.002,
            firmware_gain_scaling="min(1, effort_limit / (abs(P)+abs(D)+abs(FF)))",
            search_bounds_not_confidence_intervals=dict(
                armature_total_kg_m2=[1e-5, 10**-0.7],
                passive_damping_nm_s_rad=[0, 0.8],
                coulomb_frictionloss_nm=[0, 0.6],
            ),
            final_pass_search_converged=all(
                h["converged"] for h in history if h["pass_index"] == args.passes - 1
            ),
            limitations=[
                "Effective parameters conditional on a fixed 24 ms delay and the CAD rigid-body model.",
                "Zero boundary estimates do not establish zero physical damping or friction.",
                "Paired excitation and log timing cannot separate all physical and servo effects.",
                "Local bounded coordinate search, not proof of a global optimum.",
                "C was used only for validation; training configurations were not modified.",
            ],
        )
        (out / "candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False))
        print("COMPLETE", report["tied_train_loss"], baseline_loss, flush=True)
    finally:
        for t in trials:
            t.close()
    # Rebuilt from source on rerun; avoid duplicating a 190 MB fixture artifact.
    (out / "fixture.mjb").unlink()
    (out / "pe_replay.so").unlink()


if __name__ == "__main__":
    main()
