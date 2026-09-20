#!/usr/bin/env python3
"""Validate the WE11 Play policy and publish it to the board-side Deploy tree.

The default invocation is a dry run.  Nothing on the board is changed unless
--apply is supplied.  This tool deliberately does not update a local Deploy
copy: Play is the release source and the real Deploy workspace lives on the
board.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4


EXPECTED_INPUT = [1, 145]
EXPECTED_OUTPUT = [1, 6]
PUBLISHED_FILES = ("policy.onnx", "lab_policy.mnn", "lab_policy_manifest.json")


class PublishError(RuntimeError):
    pass


def find_default_deploy_tools_root(play_root: Path) -> Path:
    """Locate the host-side MNN tools across supported workspace layouts."""
    workspace_root = play_root.parent
    candidates = (
        workspace_root / "Deploy",
        workspace_root / "DeployHostOnly/Deploy",
        workspace_root / "DeployHostDevice/Deploy",
    )
    relative_converter = Path(
        "src/inference/thirdparty/MNNConverter/x64/MNNConvert"
    )
    for candidate in candidates:
        if (candidate / relative_converter).is_file():
            return candidate
    return candidates[0]


def run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"+ {printable}")
    try:
        return subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            input=input_text,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        if detail:
            raise PublishError(f"command failed: {printable}\n{detail}") from exc
        raise PublishError(f"command failed: {printable}") from exc
    except OSError as exc:
        raise PublishError(f"cannot run command: {printable}\n{exc}") from exc


def print_result(success: bool, mode: str, detail: str) -> None:
    label = "成功" if success else "失败"
    print("\n" + "=" * 72, file=sys.stdout if success else sys.stderr)
    print(f"[{label}] {mode}: {detail}", file=sys.stdout if success else sys.stderr)
    print("=" * 72, file=sys.stdout if success else sys.stderr)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_play_manifest(path: Path, onnx_path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PublishError(f"cannot read Play manifest {path}: {exc}") from exc

    try:
        artifact = manifest["artifacts"]["onnx"]
        contract = manifest["policy_contract"]
        input_shape = contract["input"]["shape"]
        output_shape = contract["output"]["shape"]
    except (KeyError, TypeError) as exc:
        raise PublishError(f"incomplete Play deployment manifest: missing {exc}") from exc

    actual_hash = sha256(onnx_path)
    actual_size = onnx_path.stat().st_size
    if artifact.get("sha256") != actual_hash or artifact.get("size_bytes") != actual_size:
        raise PublishError(
            "Play ONNX does not match deployment_manifest.json "
            f"(actual sha256={actual_hash}, size={actual_size})"
        )
    if input_shape != EXPECTED_INPUT or output_shape != EXPECTED_OUTPUT:
        raise PublishError(
            f"unexpected policy contract: input={input_shape}, output={output_shape}; "
            f"expected {EXPECTED_INPUT} -> {EXPECTED_OUTPUT}"
        )
    return manifest


def copy_converter_bundle(source_dir: Path, destination: Path) -> Path:
    required = (
        "MNNConvert",
        "MNNConvert.sh",
        "libMNN.so",
        "libMNNConvertDeps.so",
        "libMNN_Express.so",
    )
    destination.mkdir(parents=True)
    for name in required:
        source = source_dir / name
        if not source.is_file():
            raise PublishError(f"missing x64 converter component: {source}")
        shutil.copy2(source, destination / name)
    (destination / "MNNConvert").chmod(0o755)
    (destination / "MNNConvert.sh").chmod(0o755)
    return destination / "MNNConvert.sh"


def convert_and_inspect(
    *,
    onnx_path: Path,
    output_mnn: Path,
    deploy_tools_root: Path,
    scratch: Path,
    inspector_source: Path,
) -> dict[str, Any]:
    converter_source = (
        deploy_tools_root / "src/inference/thirdparty/MNNConverter/x64"
    )
    converter_dir = scratch / "converter"
    converter = copy_converter_bundle(converter_source, converter_dir)
    run(
        [
            str(converter),
            "-f",
            "ONNX",
            "--modelFile",
            str(onnx_path),
            "--MNNModel",
            str(output_mnn),
            "--bizCode",
            "PheonixWE11",
        ]
    )
    if not output_mnn.is_file() or output_mnn.stat().st_size == 0:
        raise PublishError("MNN converter did not create a non-empty model")

    mnn_root = deploy_tools_root / "src/inference/thirdparty/mnn-linux-x64"
    include_dir = mnn_root / "include"
    library_dir = mnn_root / "lib"
    library = library_dir / "libMNN.so"
    if not inspector_source.is_file() or not library.is_file():
        raise PublishError("missing WE11 inspector source or x64 MNN runtime")

    inspector = scratch / "we11_policy_mnn_inspect"
    run(
        [
            "g++",
            "-std=c++17",
            "-O2",
            str(inspector_source),
            f"-I{include_dir}",
            f"-L{library_dir}",
            f"-Wl,-rpath,{library_dir}",
            "-lMNN",
            "-o",
            str(inspector),
        ]
    )
    inspect = run([str(inspector), str(output_mnn)], capture=True)
    print(inspect.stdout.strip())
    return {
        "command": [
            "MNNConvert.sh",
            "-f",
            "ONNX",
            "--modelFile",
            "{source_onnx}",
            "--MNNModel",
            "{output_mnn}",
            "--bizCode",
            "PheonixWE11",
        ],
        "converter_sha256": sha256(converter_source / "MNNConvert"),
        "wrapper_sha256": sha256(converter_source / "MNNConvert.sh"),
        "inspection": {
            "input_dim": 145,
            "output_dim": 6,
            "finite_output": True,
        },
    }


def create_deploy_manifest(
    play_manifest: dict[str, Any], onnx_path: Path, mnn_path: Path, conversion: dict[str, Any]
) -> dict[str, Any]:
    source = {
        "task": play_manifest.get("source_task"),
        "run": play_manifest.get("source_run"),
        "checkpoint": play_manifest.get("source_checkpoint"),
        "checkpoint_iteration": play_manifest.get("source_checkpoint_iteration"),
        "checkpoint_sha256": play_manifest.get("source_checkpoint_sha256"),
        "run_config_sha256": play_manifest.get("source_run_config_sha256"),
        "play_onnx_sha256": sha256(onnx_path),
    }
    return {
        "schema_version": 1,
        "kind": "we11_deploy_policy",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_model": "lab_policy.mnn",
        "artifacts": {
            "onnx": {
                "file": "policy.onnx",
                "sha256": sha256(onnx_path),
                "size_bytes": onnx_path.stat().st_size,
            },
            "mnn": {
                "file": "lab_policy.mnn",
                "sha256": sha256(mnn_path),
                "size_bytes": mnn_path.stat().st_size,
            },
        },
        "contract": play_manifest["policy_contract"],
        "conversion": conversion,
        "source": {key: value for key, value in source.items() if value is not None},
    }


def remote_preflight(host: str, remote_root: str) -> None:
    # Bracketed patterns do not match this ssh command's own argument string.
    script = r'''
set -euo pipefail
root="$1"
models="$root/src/inference/models"
test -d "$root"
test -d "$models"
test -w "$models"
available_kb=$(df -Pk "$models" | awk 'NR==2 {print $4}')
test "${available_kb:-0}" -ge 10240
active=""
for pattern in '[s]tart_robot.sh' '[w]e11_supervisor' '[l]ab_inference_node' '[m]otors_node' '[w]ing_motor_node'; do
  found=$(pgrep -af "$pattern" || true)
  if [ -n "$found" ]; then active="${active}${found}\n"; fi
done
if [ -n "$active" ]; then
  printf 'refusing to publish while WE11 control processes are active:\n%b' "$active" >&2
  exit 23
fi
printf 'remote preflight OK: %s (%s KB free)\n' "$models" "$available_kb"
'''
    result = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            host,
            "bash -s -- " + shlex.quote(remote_root),
        ],
        capture=True,
        input_text=script,
    )
    print(result.stdout.strip())


def publish_remote(host: str, remote_root: str, release_dir: Path) -> str:
    release_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    remote_models = f"{remote_root.rstrip('/')}/src/inference/models"
    remote_stage = f"{remote_models}/.we11-publish-{release_id}"
    remote_backup = (
        f"{str(Path(remote_root).parent)}/artifacts/we11-policy-backups/"
        f"{release_id}/Deploy/src/inference/models"
    )

    run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "mkdir -p -- " + shlex.quote(remote_stage),
        ]
    )
    try:
        run(
            [
                "scp",
                "-q",
                *(str(release_dir / name) for name in PUBLISHED_FILES),
                f"{host}:{remote_stage}/",
            ]
        )
        expected = {name: sha256(release_dir / name) for name in PUBLISHED_FILES}
        apply_script = r'''
set -euo pipefail
models="$1"; stage="$2"; backup="$3"
shift 3
active=""
for pattern in '[s]tart_robot.sh' '[w]e11_supervisor' '[l]ab_inference_node' '[m]otors_node' '[w]ing_motor_node'; do
  found=$(pgrep -af "$pattern" || true)
  if [ -n "$found" ]; then active="${active}${found}\n"; fi
done
if [ -n "$active" ]; then
  printf 'refusing to publish while WE11 control processes are active:\n%b' "$active" >&2
  exit 23
fi
mkdir -p "$backup"
for name in policy.onnx lab_policy.mnn lab_policy_manifest.json; do
  expected="$1"; shift
  actual=$(sha256sum "$stage/$name" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "uploaded SHA-256 mismatch for $name" >&2
    exit 31
  fi
done
for name in policy.onnx lab_policy.mnn lab_policy_manifest.json; do
  if [ -e "$models/$name" ]; then cp -a "$models/$name" "$backup/$name"; fi
done
rollback() {
  for name in policy.onnx lab_policy.mnn lab_policy_manifest.json; do
    if [ -e "$backup/$name" ]; then
      cp -a "$backup/$name" "$models/$name"
    else
      rm -f -- "$models/$name"
    fi
  done
}
trap rollback ERR
# Publish the manifest last so readers never accept a partially updated pair.
mv -f "$stage/policy.onnx" "$models/policy.onnx"
mv -f "$stage/lab_policy.mnn" "$models/lab_policy.mnn"
mv -f "$stage/lab_policy_manifest.json" "$models/lab_policy_manifest.json"
rmdir "$stage"
trap - ERR
for name in policy.onnx lab_policy.mnn lab_policy_manifest.json; do
  sha256sum "$models/$name"
done
'''
        result = run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "bash -s -- "
                + " ".join(
                    shlex.quote(value)
                    for value in (
                        remote_models,
                        remote_stage,
                        remote_backup,
                        *(expected[name] for name in PUBLISHED_FILES),
                    )
                ),
            ],
            capture=True,
            input_text=apply_script,
        )
        print(result.stdout.strip())
    except Exception:
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                host,
                "bash -s -- " + shlex.quote(remote_stage),
            ],
            check=False,
            text=True,
            input='set -euo pipefail\ncase "$1" in */.we11-publish-*) rm -rf -- "$1" ;; *) exit 2 ;; esac\n',
        )
        raise
    return remote_backup


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    play_root = script_dir.parent
    default_deploy = find_default_deploy_tools_root(play_root)
    parser = argparse.ArgumentParser(
        description="Convert and publish the current Play WE11 policy to the board Deploy tree."
    )
    parser.add_argument("--apply", action="store_true", help="actually replace the board-side model files")
    parser.add_argument("--prepare-only", action="store_true", help="only validate/convert locally; do not contact the board")
    parser.add_argument("--host", default="esd@192.168.8.202")
    parser.add_argument("--remote-root", default="/home/esd/Pheonix/Deploy")
    parser.add_argument("--play-root", type=Path, default=play_root)
    parser.add_argument("--deploy-tools-root", type=Path, default=default_deploy)
    args = parser.parse_args()
    if args.apply and args.prepare_only:
        parser.error("--apply and --prepare-only cannot be used together")
    return args


def main() -> int:
    args = parse_args()
    play_root = args.play_root.resolve()
    deploy_tools_root = args.deploy_tools_root.resolve()
    policy_dir = play_root / "policy/dr002/we11"
    onnx_path = policy_dir / "policy.onnx"
    play_manifest_path = policy_dir / "deployment_manifest.json"
    inspector_source = play_root / "scripts/we11_policy_mnn_inspect.cpp"

    if not onnx_path.is_file():
        raise PublishError(f"missing Play policy: {onnx_path}")
    play_manifest = load_play_manifest(play_manifest_path, onnx_path)
    print(
        f"Play source OK: task={play_manifest.get('source_task', 'unknown')} "
        f"run={play_manifest.get('source_run', 'unknown')} sha256={sha256(onnx_path)}"
    )

    with tempfile.TemporaryDirectory(prefix="we11-board-publish-") as temp_name:
        scratch = Path(temp_name)
        release_dir = scratch / "release"
        release_dir.mkdir()
        release_onnx = release_dir / "policy.onnx"
        release_mnn = release_dir / "lab_policy.mnn"
        shutil.copy2(onnx_path, release_onnx)
        conversion = convert_and_inspect(
            onnx_path=release_onnx,
            output_mnn=release_mnn,
            deploy_tools_root=deploy_tools_root,
            scratch=scratch,
            inspector_source=inspector_source,
        )
        deploy_manifest = create_deploy_manifest(
            play_manifest, release_onnx, release_mnn, conversion
        )
        manifest_path = release_dir / "lab_policy_manifest.json"
        manifest_path.write_text(
            json.dumps(deploy_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Release prepared: ONNX={sha256(release_onnx)} "
            f"MNN={sha256(release_mnn)}"
        )

        if args.prepare_only:
            print_result(
                True,
                "本地准备检查",
                "ONNX/MNN 合同与推理检查通过；未连接板端，未修改任何板端文件。",
            )
            return 0

        remote_preflight(args.host, args.remote_root)
        if not args.apply:
            print_result(
                True,
                "板端只读预检",
                "SSH、目录、磁盘空间和控制进程检查通过；未修改板端文件。",
            )
            return 0

        backup = publish_remote(args.host, args.remote_root, release_dir)
        print(f"PUBLISHED: {args.host}:{args.remote_root}")
        print(f"Board backup: {args.host}:{backup}")
        print("No control process was started or stopped; motors were not enabled.")
        print_result(
            True,
            "板端策略发布",
            f"三个模型文件已更新；旧文件已备份到 {args.host}:{backup}",
        )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except PublishError as exc:
        print_result(False, "WE11 策略发布", str(exc))
        sys.exit(1)
