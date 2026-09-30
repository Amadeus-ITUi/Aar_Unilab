"""Finish the running PE03 experiment: evaluate, export, and stage a candidate.

Never selects a robot policy, restarts a service, or sends motor commands.
The status file distinguishes completed computation from behavioral acceptance.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", required=True)
    args = parser.parse_args()
    root = args.artifacts.resolve()
    launch = json.loads((root / "launch.json").read_text())
    run = Path(launch["run"])
    count = int(launch["max_iterations"])
    checkpoint = run / f"model_{count}.pt"
    status_path = root / "completion_status.json"
    lock = root / "completion.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()) + "\n")

    def status(stage, **fields):
        value = dict(
            stage=stage,
            updated_at=datetime.now(timezone.utc).isoformat(),
            training_pid=args.training_pid,
            worker_pid=os.getpid(),
            run=str(run),
            hardware_activated=False,
            **fields,
        )
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(status_path)
        print(json.dumps(value), flush=True)

    def execute(argv):
        print("RUN", shlex.join(map(str, argv)), flush=True)
        subprocess.run(list(map(str, argv)), check=True)

    try:
        process = Path(f"/proc/{args.training_pid}/cmdline")
        while True:
            try:
                running = b"scripts/train_pe03.py" in process.read_bytes().split(b"\0")
            except FileNotFoundError:
                running = False
            if not running:
                break
            lines = (run / "metrics.jsonl").read_text().splitlines()
            try:
                metrics = json.loads(lines[-1]) if lines else {}
            except json.JSONDecodeError:
                metrics = {}
            status(
                "training",
                iteration=metrics.get("iteration"),
                episode_seconds=metrics.get("episode/seconds"),
            )
            time.sleep(30)
        if not checkpoint.is_file():
            raise RuntimeError(f"training exited without the requested checkpoint: {checkpoint}")
        status("evaluating", checkpoint=str(checkpoint))
        evaluation = root / f"evaluation_{count}"
        execute(
            [
                sys.executable,
                "-m",
                "tools.evaluate_pe03_identified",
                checkpoint,
                "--output",
                evaluation,
                "--render",
            ]
        )
        report = json.loads((evaluation / "evaluation.json").read_text())
        accepted = bool(report["full_range"]["acceptance"]["passed"])
        status("exporting", simulation_accepted=accepted)
        import torch
        from scripts.train_pe03 import export_release

        torch.set_num_threads(1)
        # A distinct immutable release avoids replacing the trainer's own export.
        release = export_release(checkpoint, run.name + f"_identified_{count}", device="cpu")
        execute(
            [
                sys.executable,
                "-m",
                "tools.compare_pe03_sim2sim",
                "--binary",
                "sim2sim/build/aar_sim2sim",
                "--release",
                release,
            ]
        )
        candidate = root / f"candidate_{count}"
        execute(
            [
                sys.executable,
                "-m",
                "tools.package_pe03_candidate",
                release,
                "--converter",
                args.converter,
                "--output",
                candidate,
            ]
        )
        status("staging", simulation_accepted=accepted, candidate=str(candidate))
        ssh = [
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "StrictHostKeyChecking=yes",
        ]
        remote = args.remote_root.rstrip("/")
        remote_candidate = remote + "/" + candidate.name
        execute(["ssh", *ssh, args.host, "mkdir -p " + shlex.quote(remote)])
        execute(
            [
                "scp",
                "-q",
                *ssh,
                "-r",
                candidate,
                "tools/check_pe03_mnn_parity.py",
                args.host + ":" + remote + "/",
            ]
        )
        command = (
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 "
            + shlex.quote(remote + "/check_pe03_mnn_parity.py")
            + " "
            + shlex.quote(remote_candidate)
        )
        execute(["ssh", *ssh, args.host, command])
        execute(
            [
                "scp",
                "-q",
                *ssh,
                args.host + ":" + remote_candidate + "/mnn_parity.json",
                candidate / "mnn_parity.json",
            ]
        )
        console = Path("/tmp/pe03-identified-training.log")
        if console.is_file():
            shutil.copyfile(console, root / "training_console.log")
        status(
            "completed_candidate_staged" if accepted else "completed_candidate_not_accepted",
            simulation_accepted=accepted,
            checkpoint=str(checkpoint),
            release=str(release.resolve()),
            evaluation=str(evaluation),
            candidate=str(candidate),
            remote_candidate=args.host + ":" + remote_candidate,
            note="Candidate uploaded and numerically checked only. Matching policy profile and behavioral review are required before activation.",
        )
    except Exception as exc:
        status("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
