#!/usr/bin/env python3
"""Switch motors.yaml gains for single-motor sweep tests.

The motors node reads kp/kd once at startup. Use this helper before launching
motors_node so only the motor under test uses the requested sweep gains, while
the other joints are held with higher damping/stiffness.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


MOTOR_NAMES = [
    "left_thigh_joint",
    "left_calf_joint",
    "left_foot_joint",
    "right_thigh_joint",
    "right_calf_joint",
    "right_foot_joint",
]


def default_config_path() -> Path:
    deploy_root = Path(__file__).resolve().parents[3]
    return deploy_root / "src/motors/config/motors.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set motors.yaml kp/kd for one-motor sweep tests.")
    parser.add_argument("--config", type=Path, default=default_config_path(), help="Path to motors.yaml.")
    parser.add_argument("--motor-id", type=int, choices=range(1, 7), help="Physical motor id to sweep, 1..6.")
    parser.add_argument("--motor-index", type=int, choices=range(0, 6), help="0-based motor index to sweep.")
    parser.add_argument("--sweep-kp", type=float, default=None, help="Kp for the swept motor. Default: keep current.")
    parser.add_argument("--sweep-kd", type=float, default=None, help="Kd for the swept motor. Default: keep current.")
    parser.add_argument("--hold-kp", type=float, default=8.0, help="Kp for non-swept position joints.")
    parser.add_argument("--hold-kd", type=float, default=0.8, help="Kd for non-swept position joints.")
    parser.add_argument("--wheel-hold-kd", type=float, default=0.8, help="Kd for non-swept wheel velocity joints.")
    parser.add_argument("--dry-run", action="store_true", help="Print the new kp/kd without writing motors.yaml.")
    parser.add_argument("--restore", action="store_true", help="Restore motors.yaml from the backup made by this script.")
    return parser.parse_args()


def find_array(text: str, key: str) -> tuple[list[float | bool], re.Match[str]]:
    pattern = re.compile(rf"^(\s*{re.escape(key)}:\s*)\[([^\]]*)\](.*)$", re.MULTILINE)
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"cannot find '{key}: [...]' in config")

    raw_values = [value.strip() for value in match.group(2).split(",") if value.strip()]
    if key == "vel_only_mode":
        values: list[float | bool] = [value.lower() == "true" for value in raw_values]
    else:
        values = [float(value) for value in raw_values]
    if len(values) != 6:
        raise ValueError(f"{key} must contain 6 values, got {len(values)}")
    return values, match


def format_float(value: float) -> str:
    text = f"{value:.6g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text


def replace_array(text: str, match: re.Match[str], values: list[float]) -> str:
    replacement = f"{match.group(1)}[{', '.join(format_float(value) for value in values)}]{match.group(3)}"
    return text[: match.start()] + replacement + text[match.end() :]


def main() -> None:
    args = parse_args()
    config = args.config.resolve()
    backup = config.with_suffix(config.suffix + ".before_sweep_gains")

    if args.restore:
        if not backup.exists():
            raise FileNotFoundError(f"backup not found: {backup}")
        if args.dry_run:
            print(f"[DRY] would restore {config} from {backup}")
            return
        shutil.copy2(backup, config)
        print(f"[DONE] restored {config} from {backup}")
        return

    if args.motor_id is None and args.motor_index is None:
        raise ValueError("provide --motor-id 1..6 or --motor-index 0..5")
    motor_index = args.motor_index if args.motor_index is not None else args.motor_id - 1

    text = config.read_text(encoding="utf-8")
    current_kp, kp_match = find_array(text, "kp")
    current_kd, kd_match = find_array(text, "kd")
    vel_only, _ = find_array(text, "vel_only_mode")

    kp = [float(value) for value in current_kp]
    kd = [float(value) for value in current_kd]

    for index in range(6):
        if index == motor_index:
            kp[index] = 0.0 if vel_only[index] else (args.sweep_kp if args.sweep_kp is not None else kp[index])
            kd[index] = args.sweep_kd if args.sweep_kd is not None else kd[index]
        elif vel_only[index]:
            kp[index] = 0.0
            kd[index] = args.wheel_hold_kd
        else:
            kp[index] = args.hold_kp
            kd[index] = args.hold_kd

    # Replace kd first so kp_match offsets remain valid only if kp is before kd.
    new_text = replace_array(text, kd_match, kd)
    kp_values, kp_match_new = find_array(new_text, "kp")
    del kp_values
    new_text = replace_array(new_text, kp_match_new, kp)

    print(f"[INFO] sweep motor {motor_index + 1}: {MOTOR_NAMES[motor_index]}")
    print(f"[INFO] new kp = [{', '.join(format_float(value) for value in kp)}]")
    print(f"[INFO] new kd = [{', '.join(format_float(value) for value in kd)}]")

    if args.dry_run:
        print("[DRY] motors.yaml not modified")
        return

    if not backup.exists():
        shutil.copy2(config, backup)
        print(f"[INFO] wrote backup: {backup}")
    config.write_text(new_text, encoding="utf-8")
    print(f"[DONE] updated {config}")
    print("[NEXT] restart motors_node for the new gains to take effect")


if __name__ == "__main__":
    main()
