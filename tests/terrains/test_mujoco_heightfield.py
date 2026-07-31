from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from unilab.terrains import GeneratedTerrain


def _generated_terrain() -> GeneratedTerrain:
    heights_yx = np.asarray(
        [
            [-0.20, -0.05, 0.10, 0.25],
            [-0.10, 0.20, 0.55, 0.70],
            [0.00, 0.35, 0.65, 0.80],
        ],
        dtype=np.float64,
    )
    return GeneratedTerrain(
        heights_yx=heights_yx,
        horizontal_scale=0.5,
        z_min=float(heights_yx.min()),
        z_max=float(heights_yx.max()),
        base_thickness=0.1,
        terrain_origins=np.zeros((1, 1, 3), dtype=np.float64),
    )


def _load_hfield_model(tmp_path: Path, terrain: GeneratedTerrain) -> mujoco.MjModel:
    hfield_path = tmp_path / "terrain.hfield"
    terrain.write_mujoco_hfield(hfield_path)
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text(
        (
            "<mujoco><asset>"
            f'<hfield name="terrain" file="{hfield_path}" size="{terrain.hfield_size_xml()}"/>'
            "</asset><worldbody>"
            f'<geom name="floor" type="hfield" hfield="terrain" pos="{terrain.geom_pos_xml()}"/>'
            "</worldbody></mujoco>"
        ),
        encoding="utf-8",
    )
    return mujoco.MjModel.from_xml_path(str(xml_path))


def test_float_hfield_binary_preserves_rows_and_precision(tmp_path: Path) -> None:
    terrain = _generated_terrain()
    hfield_path = tmp_path / "terrain.hfield"
    terrain.write_mujoco_hfield(hfield_path)

    payload = hfield_path.read_bytes()
    shape = np.frombuffer(payload, dtype="<i4", count=2)
    data = np.frombuffer(payload, dtype="<f4", offset=8).reshape(tuple(shape))

    np.testing.assert_array_equal(shape, terrain.heights_yx.shape)
    np.testing.assert_array_equal(data, terrain.to_mujoco_hfield_data())
    normalized_image = (terrain.heights_yx - terrain.z_min) / (terrain.z_max - terrain.z_min)
    np.testing.assert_allclose(data[0], normalized_image[-1], rtol=0.0, atol=1e-7)


def test_sampler_matches_mujoco_collision_triangles(tmp_path: Path) -> None:
    terrain = _generated_terrain()
    model = _load_hfield_model(tmp_path, terrain)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    expected = terrain.to_mujoco_hfield_data()
    np.testing.assert_array_equal(model.hfield_data.reshape(expected.shape), expected)

    rng = np.random.default_rng(17)
    radius_x, radius_y = terrain.hfield_size[:2]
    points_xy = np.column_stack(
        [
            rng.uniform(-0.9 * radius_x, 0.9 * radius_x, 500),
            rng.uniform(-0.9 * radius_y, 0.9 * radius_y, 500),
        ]
    )
    geom_group = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.uint8)
    geom_id = np.empty((1,), dtype=np.int32)
    collision_heights = np.empty((points_xy.shape[0],), dtype=np.float64)
    for index, (x_pos, y_pos) in enumerate(points_xy):
        distance = mujoco.mj_ray(
            model,
            data,
            np.asarray([x_pos, y_pos, 2.0]),
            np.asarray([0.0, 0.0, -1.0]),
            geom_group,
            1,
            -1,
            geom_id,
        )
        assert distance >= 0.0
        collision_heights[index] = 2.0 - distance

    sampled_heights = terrain.surface_sampler().sample_height(points_xy)
    np.testing.assert_allclose(sampled_heights, collision_heights, rtol=0.0, atol=1e-6)
