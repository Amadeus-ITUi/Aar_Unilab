from __future__ import annotations

import json
import multiprocessing
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PACE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "pace"
sys.path.insert(0, str(PACE_DIR))

from fit_mujoco_pace_params import (  # noqa: E402
    DEFAULT_CMA_INITIAL_ARMATURE,
    DEFAULT_FIXTURE_HOLD_KD,
    DEFAULT_FIXTURE_HOLD_KP,
    CandidateWorkerContext,
    CmaSearchSpace,
    MujocoPaceReplay,
    SourceData,
    canonicalize_mirrored_fit_joints,
    causal_previous_sample_zoh,
    default_cma_mean,
    delay_artifact_fields,
    evaluate_candidate_with_context,
    model_native_params,
    parse_args,
    parse_fit_joints,
    parse_int_list,
    raw_mse,
    resolve_fixture_mode,
    simulation_schedule,
    source_active_joint_ids,
    source_artifact_token,
    source_bode_time_window,
    source_controller_gain,
    with_candidate_params,
    write_generation_progress,
)
from mujoco_dr002_common import normalize_params  # noqa: E402


def _base_params() -> dict[str, object]:
    return {
        "armature": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "viscous_friction": [0.11, 0.12, 0.13, 0.14, 0.15, 0.16],
        "coulomb_friction": [0.21, 0.22, 0.23, 0.24, 0.25, 0.26],
        "encoder_bias": [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06],
        "motor_strength": [1.0] * 6,
        "command_delay_steps": 0,
    }


class PaceParameterAssemblyTests(unittest.TestCase):
    def test_repeated_joint_sources_get_unique_artifact_tokens(self) -> None:
        source = {"joint_id": 2, "kp": 0.0, "kd": 0.05}

        self.assertEqual(source_artifact_token(0, source), "source00_kd0p05")
        self.assertEqual(source_artifact_token(1, source), "source01_kd0p05")
        self.assertEqual(
            source_artifact_token(2, {"joint_id": 2, "kd": [0.05] * 6}),
            "source02",
        )

    def test_source_controller_gain_overrides_only_active_joint(self) -> None:
        base = np.asarray([1.0, 2.0, 0.05, 4.0, 5.0, 0.05])

        actual = source_controller_gain(
            {"active_joint_ids": [2], "kd": 0.2},
            "kd",
            base,
        )

        np.testing.assert_allclose(actual, [1.0, 2.0, 0.2, 4.0, 5.0, 0.05])
        np.testing.assert_allclose(
            source_controller_gain({"active_joint_ids": [2]}, "kd", base),
            base,
        )

    def test_delay_artifact_fields_are_mutually_exclusive(self) -> None:
        self.assertEqual(
            delay_artifact_fields(6, "command"),
            {"command_delay_steps": 6, "torque_delay_steps": 0},
        )
        self.assertEqual(
            delay_artifact_fields(6, "torque"),
            {"command_delay_steps": 0, "torque_delay_steps": 6},
        )

    def test_we_fitter_defaults_to_pre_pd_command_delay(self) -> None:
        with patch.object(
            sys,
            "argv",
            ["fit_mujoco_pace_params.py", "--truth-run-dir", "unused"],
        ):
            args = parse_args()
            self.assertEqual(args.delay_semantics, "command")
            self.assertEqual(args.bode_freq_range, [0.5, 4.5])
            self.assertEqual(args.time_score_freq_range, [0.5, 4.5])
            self.assertEqual(args.chirp_source_freq_range, [0.1, 5.0])
            self.assertTrue(args.lock_wheel_positions)
            self.assertEqual(args.bode_command_psd_threshold_db, -20.0)
            self.assertEqual(args.population_size, 64)
            self.assertEqual(args.max_generations, 40)
            self.assertEqual(args.sigma, 0.35)
            self.assertEqual(args.epsilon, 0.01)
            self.assertIn(args.mp_start_method, multiprocessing.get_all_start_methods())
            np.testing.assert_allclose(
                args.initial_armature,
                [0.0076577, 0.01738699, 0.0008, 0.0076577, 0.01738699, 0.0008],
            )
            self.assertFalse(hasattr(args, "score_metric"))
            self.assertFalse(hasattr(args, "maxiter"))

    def test_cma_search_space_excludes_fixed_bounds_and_decodes_shared_delay(self) -> None:
        space = CmaSearchSpace(
            np.asarray([[0.0, 10.0], [2.0, 2.0]], dtype=np.float64),
            (4, 6, 9),
        )

        self.assertEqual(space.dimension, 2)
        np.testing.assert_allclose(space.normalized_bounds, [[-1.0, 1.0], [-1.0, 1.0]])
        continuous, delay = space.decode(np.asarray([0.0, 0.0]))
        np.testing.assert_allclose(continuous, [5.0, 2.0])
        self.assertEqual(delay, 6)
        np.testing.assert_allclose(space.encode(np.asarray([10.0, 2.0]), 4), [1.0, -1.0])

    def test_pace_time_domain_loss_uses_only_the_requested_mask(self) -> None:
        response = np.asarray([100.0, 1.0, 3.0, -100.0], dtype=np.float64)
        truth = np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        mask = np.asarray([False, True, True, False])

        self.assertEqual(raw_mse(response, truth, mask), 2.5)
        self.assertGreater(raw_mse(response, truth), 1000.0)

    def test_default_cma_mean_uses_requested_armature_for_active_joints(self) -> None:
        bounds = np.asarray(
            [(1.0e-5, 0.08)] * 2 + [(0.0, 1.5)] * 2 + [(0.0, 0.35)] * 2 + [(0.0, 0.0)] * 2,
            dtype=np.float64,
        )
        space = CmaSearchSpace(bounds, (4, 5, 6))

        continuous, delay = space.decode(
            default_cma_mean(space, [3, 4], DEFAULT_CMA_INITIAL_ARMATURE)
        )

        np.testing.assert_allclose(continuous[:2], [0.0076577, 0.01738699])
        np.testing.assert_allclose(continuous[2:6], [0.75, 0.75, 0.175, 0.175])
        np.testing.assert_allclose(continuous[6:], [0.0, 0.0])
        self.assertEqual(delay, 5)

    def test_pace_source_uses_strict_linear_chirp_scoring_window(self) -> None:
        time = np.arange(0.0, 40.0 + 0.005, 0.005)
        item = SourceData(
            source={"joint_id": 3, "duration_s": 40.0},
            payload={},
            target_kind="position",
        )

        window = source_bode_time_window(
            item,
            time,
            chirp_source_frequency_range_hz=(0.1, 5.0),
            fmin=0.5,
            fmax=4.5,
            fit_start=0.5,
            fit_end_margin=0.25,
        )

        self.assertEqual(window.sampled_time_range_s, (3.27, 35.915))
        self.assertEqual(window.sample_count, 6530)

    def test_paired_source_ids_take_precedence_over_legacy_representative(self) -> None:
        paired = SourceData(
            source={
                "joint_id": 0,
                "joint": "left_thigh_joint",
                "active_joint_ids": [0, 3],
                "active_joints": ["left_thigh_joint", "right_thigh_joint"],
            },
            payload={},
            target_kind="position",
        )
        legacy = SourceData(
            source={"joint_id": 4, "joint": "right_calf_joint"},
            payload={},
            target_kind="position",
        )

        self.assertEqual(source_active_joint_ids(paired.source), (0, 3))
        self.assertEqual(source_active_joint_ids(legacy.source), (4,))
        self.assertEqual(parse_fit_joints("active", [paired, legacy]), [0, 3, 4])
        with self.assertRaisesRegex(ValueError, "must name one"):
            source_active_joint_ids({"joint_id": 1, "active_joint_ids": [0, 3]})

    def test_mirrored_paired_fit_uses_one_canonical_parameter_set(self) -> None:
        self.assertEqual(
            canonicalize_mirrored_fit_joints(
                [0, 1, 3, 4],
                mirror_side_dynamics=True,
            ),
            [0, 1],
        )
        self.assertEqual(
            canonicalize_mirrored_fit_joints(
                [3, 4],
                mirror_side_dynamics=True,
            ),
            [3, 4],
        )
        self.assertEqual(
            canonicalize_mirrored_fit_joints(
                [0, 1, 3, 4],
                mirror_side_dynamics=False,
            ),
            [0, 1, 3, 4],
        )

    def test_paired_candidate_scores_both_sides_equally_with_one_replay(self) -> None:
        class FakeReplay:
            def __init__(self) -> None:
                self.calls = 0

            def replay(
                self,
                item: SourceData,
                params: dict[str, object],
                *,
                log_full: bool = False,
            ) -> dict[str, np.ndarray]:
                del item, params, log_full
                self.calls += 1
                return {
                    "response": np.asarray([[1.0, 10.0], [1.0, 10.0]]),
                    "truth": np.zeros((2, 2), dtype=np.float64),
                }

        source = SourceData(
            source={"joint_id": 0, "active_joint_ids": [0, 3]},
            payload={},
            target_kind="position",
        )
        source.score_time_window = type(
            "Window",
            (),
            {"mask": np.asarray([True, True])},
        )()
        replay = FakeReplay()
        context = CandidateWorkerContext(
            replay=replay,  # type: ignore[arg-type]
            search_space=CmaSearchSpace(
                np.zeros((4, 2), dtype=np.float64),
                (0,),
            ),
            base_params=_base_params(),
            fit_joints=(0,),
            sources=(source,),
            mirror_side_dynamics=True,
            mirror_encoder_bias=False,
        )

        result = evaluate_candidate_with_context(np.zeros(0), context)

        self.assertEqual(replay.calls, 1)
        self.assertEqual(result["source_mse"], [1.0, 100.0])
        self.assertEqual(result["score"], 50.5)

    def test_normalize_params_rejects_per_joint_delay_array(self) -> None:
        params = _base_params()
        params["command_delay_steps"] = [4, 6, 0, 4, 6, 0]
        with self.assertRaisesRegex(ValueError, "one scalar shared by all joints"):
            normalize_params(params)

    def test_fixture_mode_preserves_legacy_equality_flag(self) -> None:
        self.assertEqual(resolve_fixture_mode(None, True), "equality")
        self.assertEqual(resolve_fixture_mode(None, False), "free")
        self.assertEqual(
            resolve_fixture_mode(None, None, default="high-impedance"),
            "high-impedance",
        )
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_fixture_mode("high-impedance", True)

    def test_command_resampling_is_causal_previous_sample_zoh(self) -> None:
        sample_time = np.asarray([0.0, 0.006, 0.011], dtype=np.float64)
        values = np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        query_time = np.asarray([0.0, 0.005, 0.006, 0.010, 0.011, 0.020])

        sampled = causal_previous_sample_zoh(sample_time, values, query_time)

        np.testing.assert_array_equal(
            sampled,
            [[1.0, 10.0], [1.0, 10.0], [2.0, 20.0], [2.0, 20.0], [3.0, 30.0], [3.0, 30.0]],
        )

    def test_explicit_initial_qpos_only_overrides_non_source_joints(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/dr002/we11/we11.xml"
        )
        if not model_path.exists():
            self.skipTest(f"WE11 model is not available: {model_path}")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")

        initial_qpos = np.asarray([0.8, -1.6, 0.0, 1.15, -2.17, 0.0])
        initial_qvel = np.zeros(6)
        initial_measured = np.asarray([9.0, 8.0, 7.0, 1.23, 6.0, 5.0])
        initial_measured_velocity = np.asarray([9.0, 8.0, 7.0, -0.42, 6.0, 5.0])
        params = model_native_params(model_path, fixed_base=True)
        params["encoder_bias"] = [0.1] * 6
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=1000.0,
            control_hz=200.0,
            delay_semantics="command",
            control_mode="position",
            kp=np.asarray([4.0, 4.0, 0.0, 4.0, 4.0, 0.0]),
            kd=np.asarray([0.2] * 6),
            effort_limit=np.asarray([5.5] * 6),
            torque_clip=True,
            free_base=False,
            root_z=0.35,
            enable_gravity=True,
            lock_non_source_joints=True,
            initial_joint_qpos=initial_qpos,
            initial_joint_qvel=initial_qvel,
        )

        replay.reset(
            initial_measured,
            params,
            active_joint_id=3,
            initial_measured_velocity=initial_measured_velocity,
        )

        expected_qpos = initial_qpos.copy()
        expected_qpos[3] = initial_measured[3] + params["encoder_bias"][3]
        np.testing.assert_allclose(replay.data.qpos[replay.qpos_ids], expected_qpos, atol=0.0)
        expected_qvel = initial_qvel.copy()
        expected_qvel[3] = initial_measured_velocity[3]
        np.testing.assert_allclose(replay.data.qvel[replay.qvel_ids], expected_qvel, atol=0.0)
        np.testing.assert_allclose(
            replay.model.eq_data[replay.lock_equality_ids, 0],
            expected_qpos,
            atol=0.0,
        )

    def test_explicit_integer_delay_and_dynamics_only_mirroring(self) -> None:
        candidate = np.asarray([0.07, 0.08, 0.31, 0.32, 0.11, 0.12, 0.03, 0.04])
        params = with_candidate_params(
            _base_params(),
            [3, 4],
            candidate,
            delay_steps=11,
            mirror_side_dynamics=True,
            mirror_encoder_bias=False,
        )

        self.assertEqual(params["command_delay_steps"], 11)
        for key in ("armature", "viscous_friction", "coulomb_friction"):
            self.assertEqual(params[key][0], params[key][3])
            self.assertEqual(params[key][1], params[key][4])
        self.assertEqual(params["encoder_bias"][0:2], [-0.01, -0.02])
        self.assertEqual(params["encoder_bias"][3:5], [0.03, 0.04])
        self.assertEqual(params["armature"][2], 0.03)
        self.assertEqual(params["coulomb_friction"][5], 0.26)

    def test_legacy_embedded_delay_remains_supported(self) -> None:
        candidate = np.asarray([0.07, 0.08, 0.31, 0.32, 0.11, 0.12, 0.03, 0.04, 10.6])
        params = with_candidate_params(_base_params(), [3, 4], candidate)

        self.assertEqual(params["command_delay_steps"], 11)
        self.assertEqual(params["armature"][0], 0.01)
        self.assertEqual(params["armature"][3], 0.07)

    def test_parse_int_list_is_sorted_and_unique(self) -> None:
        self.assertEqual(parse_int_list("11, 9,10,11"), [9, 10, 11])

    def test_generation_progress_is_atomically_flushed(self) -> None:
        params = _base_params()
        candidate = {
            "score": 0.125,
            "mean_rmse": 0.25,
            "source_mse": [0.1, 0.15],
            "delay_steps": 5,
            "normalized_x": np.asarray([0.1, -0.2]),
            "x": np.asarray([0.01, 0.02]),
            "params": params,
            "seed": 0,
            "generation": 1,
            "member": 3,
        }
        eval_rows = [
            {
                "eval": 1,
                "seed": 0,
                "generation": 1,
                "member": 3,
                "time_mse": 0.125,
                "mean_rmse": 0.25,
                "source_mse": "[0.1, 0.15]",
                "delay_steps": 5,
                "normalized_x": "[0.1, -0.2]",
                "x": "[0.01, 0.02]",
                "params": json.dumps(params),
            }
        ]
        optimizer_rows = [
            {
                "seed": 0,
                "generation": 1,
                "best_time_mse": 0.125,
                "mean_time_mse": 0.2,
                "worst_time_mse": 0.3,
                "relative_score_spread": 1.4,
                "sigma": 0.35,
                "cma_should_stop": False,
                "epsilon_converged": False,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_csv = root / "candidates.csv"
            optimizer_csv = root / "optimizer.csv"
            progress_json = root / "progress.json"
            write_generation_progress(
                candidate_csv=candidate_csv,
                optimizer_csv=optimizer_csv,
                progress_json=progress_json,
                eval_rows=eval_rows,
                optimizer_rows=optimizer_rows,
                best_candidate=candidate,
                status="running",
                seed=0,
                generation=1,
            )

            self.assertIn("time_mse", candidate_csv.read_text(encoding="utf-8"))
            self.assertIn("relative_score_spread", optimizer_csv.read_text(encoding="utf-8"))
            progress = json.loads(progress_json.read_text(encoding="utf-8"))
            self.assertEqual(progress["completed_objective_evals"], 1)
            self.assertEqual(progress["best_candidate"]["member"], 3)
            self.assertEqual(progress["best_candidate"]["time_mse"], 0.125)
            self.assertFalse(any(path.name.startswith(".") for path in root.iterdir()))

    def test_non_integer_1000_400_schedule_alternates_three_and_two_steps(self) -> None:
        sim_time, update_indices, control_ticks = simulation_schedule(
            np.arange(9, dtype=np.float64) / 400.0,
            sim_hz=1000.0,
            control_hz=400.0,
        )

        np.testing.assert_array_equal(update_indices[:9], [0, 3, 5, 8, 10, 13, 15, 18, 20])
        np.testing.assert_array_equal(control_ticks[:9], np.arange(9))
        self.assertGreaterEqual(sim_time[-1], 8.0 / 400.0)

    def test_1000_500_schedule_updates_every_two_physics_steps(self) -> None:
        sim_time, update_indices, control_ticks = simulation_schedule(
            np.arange(9, dtype=np.float64) / 500.0,
            sim_hz=1000.0,
            control_hz=500.0,
        )

        np.testing.assert_array_equal(update_indices[:9], [0, 2, 4, 6, 8, 10, 12, 14, 16])
        np.testing.assert_array_equal(control_ticks[:9], np.arange(9))
        self.assertGreaterEqual(sim_time[-1], 8.0 / 500.0)

    def test_400_200_schedule_updates_every_two_physics_steps(self) -> None:
        sim_time, update_indices, control_ticks = simulation_schedule(
            np.arange(9, dtype=np.float64) / 200.0,
            sim_hz=400.0,
            control_hz=200.0,
        )

        np.testing.assert_array_equal(update_indices[:9], [0, 2, 4, 6, 8, 10, 12, 14, 16])
        np.testing.assert_array_equal(control_ticks[:9], np.arange(9))
        self.assertGreaterEqual(sim_time[-1], 8.0 / 200.0)

    def test_high_impedance_fixture_uses_reset_pose_and_independent_hold_gains(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/dr002/we11/we11.xml"
        )
        if not model_path.exists():
            self.skipTest(f"WE11 model is not available: {model_path}")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")

        initial_qpos = np.asarray([0.8, -1.6, 0.0, 1.15, -2.17, 0.0])
        params = model_native_params(model_path, fixed_base=True)
        params["encoder_bias"] = [0.0] * 6
        active_kp = np.asarray([4.0, 4.0, 0.0, 4.0, 4.0, 0.0])
        active_kd = np.asarray([0.2] * 6)
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=400.0,
            control_hz=200.0,
            delay_semantics="command",
            control_mode="position",
            kp=active_kp,
            kd=active_kd,
            effort_limit=np.asarray([100.0] * 6),
            torque_clip=True,
            free_base=False,
            root_z=0.35,
            enable_gravity=True,
            fixture_mode="high-impedance",
            fixture_hold_kp=DEFAULT_FIXTURE_HOLD_KP,
            fixture_hold_kd=DEFAULT_FIXTURE_HOLD_KD,
            initial_joint_qpos=initial_qpos,
            initial_joint_qvel=np.zeros(6),
        )
        initial_measured = initial_qpos.copy()
        initial_measured[3] = 1.05
        replay.reset(initial_measured, params, active_joint_id=3)
        q = initial_qpos + np.asarray([0.10, -0.20, 0.30, 0.05, 0.25, -0.40])
        qd = np.asarray([0.2, -0.3, 0.4, -0.5, 0.6, -0.7])
        desired = np.full(6, 100.0)
        desired[3] = q[3] + 0.25

        tau = replay.torque(
            q,
            qd,
            desired,
            np.zeros(6),
            np.zeros(6),
            active_joint_id=3,
        )

        self.assertEqual(replay.fixture_mode, "high-impedance")
        expected_fixture_target = initial_qpos.copy()
        expected_fixture_target[3] = initial_measured[3]
        np.testing.assert_allclose(
            replay.fixture_target_qpos,
            expected_fixture_target,
            atol=0.0,
        )
        self.assertAlmostEqual(tau[3], active_kp[3] * 0.25 - active_kd[3] * qd[3])
        non_source_legs = np.asarray([True, True, False, False, True, False])
        expected_hold = DEFAULT_FIXTURE_HOLD_KP * (initial_qpos - q) - DEFAULT_FIXTURE_HOLD_KD * qd
        np.testing.assert_allclose(tau[non_source_legs], expected_hold[non_source_legs])
        np.testing.assert_array_equal(tau[[2, 5]], 0.0)
        self.assertGreater(np.max(np.abs(tau[non_source_legs])), 0.0)

        time = np.arange(0.0, 0.1, 1.0 / 200.0)
        desired_position = np.tile(initial_qpos, (time.size, 1))
        desired_position[:, 3] += 0.1 * np.sin(2.0 * np.pi * 2.0 * time)
        source = SourceData(
            source={"joint_id": 3, "joint": "right_thigh_joint"},
            payload={
                "time": time,
                "des_dof_pos": desired_position,
                "des_dof_vel": np.zeros_like(desired_position),
                "dof_pos": np.tile(initial_qpos, (time.size, 1)),
                "dof_vel": np.zeros_like(desired_position),
            },
            target_kind="position",
        )
        result = replay.replay(source, params, log_full=True)
        self.assertGreater(np.max(np.abs(result["tau"][:, non_source_legs])), 1.0e-6)
        np.testing.assert_array_equal(
            result["q"][:, [2, 5]],
            np.tile(initial_qpos[[2, 5]], (time.size, 1)),
        )
        np.testing.assert_array_equal(result["qd"][:, [2, 5]], 0.0)
        np.testing.assert_array_equal(result["tau"][:, [2, 5]], 0.0)

    def test_replay_delays_one_complete_frame_before_pd_with_current_state(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/dr002/we11/we11.xml"
        )
        if not model_path.exists():
            self.skipTest(f"WE11 model is not available: {model_path}")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")

        sample_hz = 400.0
        time = np.arange(5, dtype=np.float64) / sample_hz
        initial_qpos = np.asarray([0.8, -1.6, 0.0, 1.15, -2.17, 0.0])
        joint_offsets = np.asarray([0.01, -0.02, 0.03, -0.04, 0.05, -0.06])
        joint_slopes = np.asarray([0.02, 0.03, 0.04, 0.05, 0.06, 0.07])
        desired_position = (
            initial_qpos[None, :]
            + joint_offsets[None, :]
            + np.arange(time.size, dtype=np.float64)[:, None] * joint_slopes[None, :]
        )
        desired_velocity = (
            10.0 * np.arange(time.size, dtype=np.float64)[:, None]
            + np.arange(6, dtype=np.float64)[None, :]
        )
        source = SourceData(
            source={"joint_id": 3, "joint": "right_thigh_joint"},
            payload={
                "time": time,
                "des_dof_pos": desired_position,
                "des_dof_vel": desired_velocity,
                "dof_pos": np.tile(initial_qpos, (time.size, 1)),
                "dof_vel": np.zeros_like(desired_position),
            },
            target_kind="position",
        )
        params = model_native_params(model_path, fixed_base=True)
        params["encoder_bias"] = [0.0] * 6
        params["command_delay_steps"] = 2
        kp = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
        kd = np.asarray([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=sample_hz,
            control_hz=sample_hz,
            delay_semantics="command",
            control_mode="position",
            kp=kp,
            kd=kd,
            effort_limit=np.asarray([1000.0] * 6),
            torque_clip=False,
            free_base=False,
            root_z=0.35,
            enable_gravity=False,
            fixture_mode="free",
            lock_wheel_positions=False,
            initial_joint_qpos=initial_qpos,
            initial_joint_qvel=np.zeros(6),
        )

        result = replay.replay(source, params, log_full=True)
        delayed_indices = np.asarray([0, 0, 0, 1, 2])
        np.testing.assert_allclose(
            result["applied_des"],
            desired_position[delayed_indices],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            result["applied_desvel"],
            desired_velocity[delayed_indices],
            rtol=0.0,
            atol=0.0,
        )
        expected_tau = (
            kp[None, :] * (result["applied_des"] - result["q"]) - kd[None, :] * result["qd"]
        )
        np.testing.assert_allclose(result["tau"], expected_tau, rtol=0.0, atol=1.0e-12)
        self.assertGreater(float(np.max(np.abs(result["tau"][0]))), 0.0)

    def test_free_fixture_still_locks_both_wheel_positions_exactly(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/dr002/we11/we11.xml"
        )
        if not model_path.exists():
            self.skipTest(f"WE11 model is not available: {model_path}")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")

        sample_hz = 400.0
        time = np.arange(0.0, 0.1, 1.0 / sample_hz)
        initial_qpos = np.asarray([0.8, -1.6, 0.31, 1.15, -2.17, -0.27])
        desired = np.tile(initial_qpos, (time.size, 1))
        desired[:, 3] += 0.1 * np.sin(2.0 * np.pi * 2.0 * time)
        desired[:, 2] += 10.0
        desired[:, 5] -= 10.0
        source = SourceData(
            source={"joint_id": 3, "joint": "right_thigh_joint"},
            payload={
                "time": time,
                "des_dof_pos": desired,
                "des_dof_vel": np.zeros_like(desired),
                "dof_pos": np.tile(initial_qpos, (time.size, 1)),
                "dof_vel": np.zeros_like(desired),
            },
            target_kind="position",
        )
        params = model_native_params(model_path, fixed_base=True)
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=sample_hz,
            control_hz=200.0,
            delay_semantics="command",
            control_mode="position",
            kp=np.asarray([4.0, 4.0, 20.0, 4.0, 4.0, 20.0]),
            kd=np.asarray([0.2] * 6),
            effort_limit=np.asarray([100.0] * 6),
            torque_clip=True,
            free_base=False,
            root_z=0.35,
            enable_gravity=False,
            fixture_mode="free",
            initial_joint_qpos=initial_qpos,
            initial_joint_qvel=np.zeros(6),
        )

        result = replay.replay(source, params, log_full=True)

        np.testing.assert_array_equal(
            result["q"][:, [2, 5]],
            np.tile(initial_qpos[[2, 5]], (time.size, 1)),
        )
        np.testing.assert_array_equal(result["qd"][:, [2, 5]], 0.0)
        np.testing.assert_array_equal(result["tau"][:, [2, 5]], 0.0)
        self.assertGreater(np.ptp(result["q"][:, 3]), 1.0e-4)

    def test_schedule_rejects_control_rate_above_physics_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            simulation_schedule(np.asarray([0.0, 0.01]), sim_hz=400.0, control_hz=1000.0)

    def test_non_source_joint_equalities_hold_five_joints_fixed(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/dr002/we11/we11.xml"
        )
        if not model_path.exists():
            self.skipTest(f"WE11 model is not available: {model_path}")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")

        sample_hz = 500.0
        time = np.arange(0.0, 0.5, 1.0 / sample_hz)
        desired = np.zeros((time.size, 6), dtype=np.float64)
        desired[:, 3] = 0.2 * np.sin(2.0 * np.pi * 2.0 * time)
        measured = np.zeros_like(desired)
        source = SourceData(
            source={"joint_id": 3, "joint": "right_thigh_joint"},
            payload={
                "time": time,
                "des_dof_pos": desired,
                "des_dof_vel": np.zeros_like(desired),
                "dof_pos": measured,
                "dof_vel": np.zeros_like(measured),
            },
            target_kind="position",
        )
        params = model_native_params(model_path, fixed_base=True)
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=1000.0,
            control_hz=500.0,
            delay_semantics="command",
            control_mode="position",
            kp=np.asarray([4.0, 4.0, 0.0, 4.0, 4.0, 0.0]),
            kd=np.asarray([0.2] * 6),
            effort_limit=np.asarray([5.5] * 6),
            torque_clip=True,
            free_base=False,
            root_z=0.35,
            enable_gravity=False,
            lock_non_source_joints=True,
        )

        result = replay.replay(source, params, log_full=True)
        locked = np.asarray([True, True, True, False, True, True])
        locked_drift_by_joint = np.max(
            np.abs(result["q"][:, locked] - result["q"][0, locked]), axis=0
        )
        locked_velocity_by_joint = np.max(np.abs(result["qd"][:, locked]), axis=0)
        locked_drift = np.max(locked_drift_by_joint)

        self.assertGreater(np.ptp(result["q"][:, 3]), 1.0e-3)
        self.assertLess(locked_drift, 1.0e-5, msg=str(locked_drift_by_joint.tolist()))
        self.assertLess(
            np.max(locked_velocity_by_joint),
            1.0e-3,
            msg=str(locked_velocity_by_joint.tolist()),
        )
        np.testing.assert_allclose(result["tau"][:, locked], 0.0, atol=0.0)
        np.testing.assert_array_equal(
            result["q"][:, [2, 5]],
            np.tile(result["q"][0, [2, 5]], (time.size, 1)),
        )
        np.testing.assert_array_equal(result["qd"][:, [2, 5]], 0.0)

    def test_paired_replay_drives_both_legs_and_locks_all_other_joints(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/dr002/we11/we11.xml"
        )
        if not model_path.exists():
            self.skipTest(f"WE11 model is not available: {model_path}")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")

        sample_hz = 400.0
        time = np.arange(int(0.5 * sample_hz), dtype=np.float64) / sample_hz
        initial_qpos = np.asarray([0.8, -1.6, 0.31, 1.15, -2.17, -0.27])
        desired = np.tile(initial_qpos, (time.size, 1))
        desired[:, 0] += 0.15 * np.sin(2.0 * np.pi * 2.0 * time)
        desired[:, 3] -= 0.10 * np.sin(2.0 * np.pi * 2.0 * time)
        measured = np.tile(initial_qpos, (time.size, 1))
        source = SourceData(
            source={
                "joint_id": 0,
                "joint": "left_thigh_joint",
                "active_joint_ids": [0, 3],
                "active_joints": ["left_thigh_joint", "right_thigh_joint"],
            },
            payload={
                "time": time,
                "des_dof_pos": desired,
                "des_dof_vel": np.zeros_like(desired),
                "dof_pos": measured,
                "dof_vel": np.zeros_like(measured),
            },
            target_kind="position",
        )
        params = model_native_params(model_path, fixed_base=True)
        params["encoder_bias"] = [0.0] * 6
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=sample_hz,
            control_hz=sample_hz,
            delay_semantics="command",
            control_mode="position",
            kp=np.asarray([4.0, 4.0, 0.0, 4.0, 4.0, 0.0]),
            kd=np.asarray([0.2] * 6),
            effort_limit=np.asarray([100.0] * 6),
            torque_clip=True,
            free_base=False,
            root_z=0.35,
            enable_gravity=False,
            fixture_mode="equality",
            initial_joint_qpos=initial_qpos,
            initial_joint_qvel=np.zeros(6),
        )

        result = replay.replay(source, params, log_full=True)

        self.assertEqual(result["response"].shape, (time.size, 2))
        np.testing.assert_array_equal(result["active_joint_ids"], [0, 3])
        np.testing.assert_allclose(result["applied_des"][:, 0], desired[:, 0])
        np.testing.assert_allclose(result["applied_des"][:, 3], desired[:, 3])
        self.assertGreater(np.ptp(result["q"][:, 0]), 1.0e-3)
        self.assertGreater(np.ptp(result["q"][:, 3]), 1.0e-3)
        locked = np.asarray([False, True, True, False, True, True])
        np.testing.assert_allclose(
            result["q"][:, [1, 4]],
            np.tile(initial_qpos[[1, 4]], (time.size, 1)),
            atol=1.0e-5,
        )
        self.assertLess(np.max(np.abs(result["qd"][:, [1, 4]])), 1.0e-3)
        np.testing.assert_array_equal(
            result["q"][:, [2, 5]],
            np.tile(initial_qpos[[2, 5]], (time.size, 1)),
        )
        np.testing.assert_array_equal(result["qd"][:, [2, 5]], 0.0)
        np.testing.assert_array_equal(result["tau"][:, locked], 0.0)

    def test_wheel_velocity_source_releases_only_target_wheel(self) -> None:
        model_path = (
            Path(__file__).resolve().parents[2] / "src/unilab/assets/robots/dr002/we11/we11.xml"
        )
        if not model_path.exists():
            self.skipTest(f"WE11 model is not available: {model_path}")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            self.skipTest("mujoco is not installed")

        sample_hz = 400.0
        time = np.arange(int(0.5 * sample_hz), dtype=np.float64) / sample_hz
        initial_qpos = np.asarray([0.8, -1.6, 0.0, 0.8, -1.6, 0.0])
        desired_position = np.tile(initial_qpos, (time.size, 1))
        desired_velocity = np.zeros_like(desired_position)
        desired_velocity[:, 2] = 5.0 * np.sin(2.0 * np.pi * 2.0 * time)
        measured_velocity = np.zeros_like(desired_position)
        source = SourceData(
            source={
                "joint_id": 2,
                "joint": "left_foot_joint",
                "active_joint_ids": [2],
                "target_type": "velocity",
            },
            payload={
                "time": time,
                "des_dof_pos": desired_position,
                "des_dof_vel": desired_velocity,
                "dof_pos": desired_position.copy(),
                "dof_vel": measured_velocity,
            },
            target_kind="velocity",
        )
        params = model_native_params(model_path, fixed_base=True)
        params["encoder_bias"] = [0.0] * 6
        replay = MujocoPaceReplay(
            model_path,
            sim_hz=sample_hz,
            control_hz=200.0,
            delay_semantics="command",
            control_mode="mixed",
            kp=np.asarray([2.0, 8.0, 0.0, 2.0, 8.0, 0.0]),
            kd=np.asarray([0.1, 0.8, 0.2, 0.1, 0.8, 0.2]),
            effort_limit=np.asarray([10.0] * 6),
            torque_clip=True,
            free_base=False,
            root_z=0.35,
            enable_gravity=False,
            fixture_mode="equality",
            lock_wheel_positions=True,
            initial_joint_qpos=initial_qpos,
            initial_joint_qvel=np.zeros(6),
        )

        result = replay.replay(source, params, log_full=True)

        self.assertGreater(np.max(np.abs(result["qd"][:, 2])), 1.0e-3)
        locked = np.asarray([True, True, False, True, True, True])
        np.testing.assert_allclose(
            result["q"][:, locked],
            np.tile(initial_qpos[locked], (time.size, 1)),
            atol=1.0e-5,
        )
        self.assertLess(np.max(np.abs(result["qd"][:, locked])), 1.0e-3)
        np.testing.assert_array_equal(result["qd"][:, 5], 0.0)
        np.testing.assert_array_equal(result["tau"][:, locked], 0.0)


if __name__ == "__main__":
    unittest.main()
