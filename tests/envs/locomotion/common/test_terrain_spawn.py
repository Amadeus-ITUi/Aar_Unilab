from __future__ import annotations

import numpy as np

from unilab.envs.locomotion.common.terrain_spawn import (
    TerrainCurriculumCfg,
    TerrainSpawnManager,
)


def _origins(num_rows: int = 4, num_cols: int = 2) -> np.ndarray:
    origins = np.zeros((num_rows, num_cols, 3), dtype=np.float64)
    for row in range(num_rows):
        for col in range(num_cols):
            origins[row, col] = [row * 8.0, col * 8.0, 0.0]
    return origins


def test_path_length_promotes_while_command_gate_keeps_standing_neutral() -> None:
    manager = TerrainSpawnManager(
        num_envs=2,
        terrain_origins=_origins(),
        cell_size=8.0,
        cfg=TerrainCurriculumCfg(enabled=True, use_path_length=True, seed=7),
    )
    manager.levels[:] = 1
    env_ids = np.asarray([0, 1], dtype=np.int32)
    starts = np.zeros((2, 3), dtype=np.float64)
    manager.record_episode_start(env_ids, starts)

    manager.record_step(
        env_ids,
        np.asarray([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float64),
    )
    manager.record_step(
        np.asarray([0], dtype=np.int32),
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
    )
    stats = manager.update_on_done(
        env_ids,
        starts,
        demote_eligible=np.asarray([True, False]),
    )

    np.testing.assert_array_equal(manager.levels, [2, 1])
    assert stats["num_promoted"] == 1
    assert stats["num_demoted"] == 0
    assert stats["mean_walked"] == 3.0


def test_legacy_net_displacement_and_unmasked_demotion_remain_default() -> None:
    manager = TerrainSpawnManager(
        num_envs=1,
        terrain_origins=_origins(),
        cell_size=8.0,
        cfg=TerrainCurriculumCfg(enabled=True, seed=3),
    )
    manager.levels[:] = 2
    env_ids = np.asarray([0], dtype=np.int32)
    manager.record_episode_start(env_ids, np.zeros((1, 3), dtype=np.float64))
    manager.record_step(env_ids, np.asarray([[3.0, 0.0, 0.0]], dtype=np.float64))
    stats = manager.update_on_done(
        env_ids,
        np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
    )

    np.testing.assert_array_equal(manager.levels, [1])
    assert stats["num_promoted"] == 0
    assert stats["num_demoted"] == 1
    assert stats["mean_walked"] == 0.0


def test_terrain_curriculum_state_round_trip_preserves_rng_levels_and_types() -> None:
    cfg = TerrainCurriculumCfg(enabled=True, use_path_length=True, seed=11)
    source = TerrainSpawnManager(
        num_envs=4,
        terrain_origins=_origins(),
        cell_size=8.0,
        cfg=cfg,
    )
    source.levels[:] = [0, 1, 2, 3]
    source.type_cols[:] = [0, 1, 0, 1]
    saved = source.state_dict()

    target = TerrainSpawnManager(
        num_envs=4,
        terrain_origins=_origins(),
        cell_size=8.0,
        cfg=cfg,
    )
    target.load_state_dict(saved)

    np.testing.assert_array_equal(target.levels, source.levels)
    np.testing.assert_array_equal(target.type_cols, source.type_cols)

    env_ids = np.asarray([3], dtype=np.int32)
    start = np.zeros((1, 3), dtype=np.float64)
    end = np.asarray([[5.0, 0.0, 0.0]], dtype=np.float64)
    source.record_episode_start(env_ids, start)
    target.record_episode_start(env_ids, start)
    source.update_on_done(env_ids, end)
    target.update_on_done(env_ids, end)
    np.testing.assert_array_equal(target.levels, source.levels)


def test_locked_terrain_curriculum_finishes_episode_without_changing_level() -> None:
    manager = TerrainSpawnManager(
        num_envs=1,
        terrain_origins=_origins(num_rows=10),
        cell_size=8.0,
        cfg=TerrainCurriculumCfg(enabled=True, use_path_length=True, seed=13),
    )
    manager.levels[:] = 1
    env_ids = np.asarray([0], dtype=np.int32)
    start = np.zeros((1, 3), dtype=np.float64)
    end = np.asarray([[5.0, 0.0, 0.0]], dtype=np.float64)

    manager.record_episode_start(env_ids, start)
    locked_stats = manager.update_on_done(env_ids, end, allow_progression=False)
    np.testing.assert_array_equal(manager.levels, [1])
    assert locked_stats["num_promoted"] == 0
    assert locked_stats["num_demoted"] == 0

    manager.record_episode_start(env_ids, start)
    unlocked_stats = manager.update_on_done(env_ids, end, allow_progression=True)
    np.testing.assert_array_equal(manager.levels, [2])
    assert unlocked_stats["num_promoted"] == 1


def test_initial_terrain_level_selects_deterministic_playback_row() -> None:
    manager = TerrainSpawnManager(
        num_envs=4,
        terrain_origins=_origins(num_rows=10),
        cell_size=8.0,
        cfg=TerrainCurriculumCfg(enabled=True, initial_level=9, seed=17),
    )

    np.testing.assert_array_equal(manager.levels, [9, 9, 9, 9])
