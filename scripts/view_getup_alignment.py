"""Open the standalone WE11 get-up alignment scene at its saved start pose."""

from pathlib import Path

import mujoco
import mujoco.viewer

ROOT_DIR = Path(__file__).resolve().parent.parent
SCENE_PATH = (
    ROOT_DIR
    / "src"
    / "unilab"
    / "assets"
    / "robots"
    / "dr002"
    / "we11"
    / "scene_getup_alignment_we11.xml"
)
KEYFRAME_NAME = "getup_start_v2"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, KEYFRAME_NAME)
    if keyframe_id < 0:
        raise ValueError(f"Keyframe {KEYFRAME_NAME!r} is missing from {SCENE_PATH}")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model=model, data=data)


if __name__ == "__main__":
    main()
