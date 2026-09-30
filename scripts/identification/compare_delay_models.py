"""Compare shared versus grouped effective delays on existing sweeps only.

Fit A/B, freeze the shared model, then score C. Physical parameters are refitted
under the shared-delay constraint; merely averaging the old delays is not a fit.
This is a bounded coordinate search, not a proof of a global optimum.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import minimize
from scripts.identification.fit_pe_sweeps import (
    GROUPS,
    ROOT,
    Trial,
    digest,
    metrics,
    native_library,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "artifacts/pe03_gain_matrix_test01")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--maxfev", type=int, default=160)
    args = parser.parse_args()
    source = args.data / "identification"
    out = source / "shared_delay_comparison"
    out.mkdir(exist_ok=True)
    dll = native_library(out)
    # Reuse the exact audited fixture; the original fit artifacts remain unchanged.
    (out / "fixture.mjb").write_bytes((source / "fixture.mjb").read_bytes())
    quality = json.loads((args.data / "visualizations/data_quality.json").read_text())
    trials = [Trial(args.data / (q["run"] + ".csv"), q, dll, out) for q in quality]
    grouped = np.array(json.loads((source / "checkpoint.json").read_text())["parameters"])
    par = grouped.copy()
    history = []
    with ThreadPoolExecutor(max_workers=6) as pool:

        def loss(parameters, selected):
            results = list(pool.map(lambda t: t.replay(parameters, contacts=False), selected))
            return float(
                np.mean(
                    [
                        np.mean(((r[:, :6] - t.measured)[t.mask][:, t.selected] / t.amplitude) ** 2)
                        for t, r in zip(selected, results, strict=True)
                    ]
                )
            )

        grouped_loss = loss(grouped, trials[:6])
        for cycle in range(args.passes):
            profile = []
            # 2 ms resolution matches the replay controller's command hold.
            for delay in np.arange(0, 0.060001, 0.002):
                candidate = par.copy()
                candidate[9:] = delay
                profile.append([float(delay), loss(candidate, trials[:6])])
            delay, value = min(profile, key=lambda entry: entry[1])
            par[9:] = delay
            history.append(dict(pass_index=cycle, delay_profile=profile))
            print("PROFILE", cycle, delay, value, flush=True)
            for group in range(3):
                selected = [t for t in trials[:6] if t.group == group]
                best = [loss(par, selected), par.copy()]
                calls = 0

                def objective(x):
                    nonlocal calls
                    candidate = par.copy()
                    candidate[group] = 10 ** x[0]
                    candidate[3 + group], candidate[6 + group] = x[1:]
                    value = loss(candidate, selected)
                    calls += 1
                    if value < best[0]:
                        best[:] = [value, candidate.copy()]
                    return value

                result = minimize(
                    objective,
                    [np.log10(par[group]), par[3 + group], par[6 + group]],
                    method="Powell",
                    bounds=[(-5, -0.7), (0, 0.8), (0, 0.6)],
                    options=dict(maxfev=args.maxfev, xtol=2e-4, ftol=2e-4),
                )
                par = best[1]
                history.append(
                    dict(
                        pass_index=cycle,
                        group=GROUPS[group],
                        calls=calls,
                        loss=best[0],
                        converged=bool(result.success),
                        message=str(result.message),
                    )
                )
                (out / "checkpoint.json").write_text(
                    json.dumps(dict(parameters=par.tolist(), history=history), indent=2)
                )
                print("REFIT", history[-1], flush=True)
        frozen_loss = loss(par, trials[:6])
        scores = {}
        for t in trials:
            scores[t.name] = {}
            for name, params in (("grouped", grouped), ("shared", par)):
                replay = t.replay(params)
                scores[t.name][name] = metrics(t, replay)
                if name == "shared":
                    np.savez_compressed(
                        out / (t.name + ".npz"), replay=replay, time=t.samples - t.start
                    )
        report = dict(
            source_candidate_sha256=digest(source / "candidate.yaml"),
            script_sha256=digest(__file__),
            grouped_parameters=grouped.tolist(),
            shared_parameters=par.tolist(),
            grouped_train_loss=grouped_loss,
            shared_train_loss=frozen_loss,
            fit_runs=[t.name for t in trials[:6]],
            held_out_runs=[t.name for t in trials[6:]],
            history=history,
            metrics=scores,
            interpretation="Effective delay, not measured transport latency. Bounded coordinate search; C never used for fitting.",
        )
        (out / "comparison.json").write_text(json.dumps(report, indent=2))
        candidate = yaml.safe_load((source / "candidate.yaml").read_text())
        for name, values in zip(
            (
                "armature_total_kg_m2",
                "passive_damping_nm_s_rad",
                "coulomb_frictionloss_nm",
                "effective_delay_s",
            ),
            np.split(par, 4),
            strict=True,
        ):
            candidate[name] = values.tolist()
        candidate.pop("boundary_estimates", None)
        candidate.pop("sweep_kinematic_gate_passed", None)
        candidate["delay_model"] = "shared_effective_delay"
        candidate["search_converged_all_blocks"] = all(h.get("converged", True) for h in history)
        (out / "candidate.yaml").write_text(yaml.safe_dump(candidate, sort_keys=False))
        print("COMPLETE", grouped_loss, frozen_loss, par.tolist(), flush=True)
    for t in trials:
        t.close()


if __name__ == "__main__":
    main()
