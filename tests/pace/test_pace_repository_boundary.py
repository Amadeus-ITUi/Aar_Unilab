from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PACE_DIR = REPO_ROOT / "scripts" / "pace"


class PaceRepositoryBoundaryTests(unittest.TestCase):
    def test_pace_scripts_are_we11_only_and_portable(self) -> None:
        for path in sorted(PACE_DIR.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("robots/dr002/u9", text, path)
            self.assertNotIn("sweep_u9", text, path)
            self.assertIsNone(re.search(r"/home/[^/]+/", text), path)

    def test_fitter_does_not_import_deploy_source(self) -> None:
        for path in sorted(PACE_DIR.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("Walking_Eagle-Deploy_final/scripts", text, path)
            self.assertNotIn("Walking_Eagle-Deploy-final-publish", text, path)

    def test_default_identification_model_is_repository_local_we11(self) -> None:
        common = (PACE_DIR / "mujoco_dr002_common.py").read_text(encoding="utf-8")
        self.assertIn(
            'DEFAULT_IDENTIFICATION_MODEL = "src/unilab/assets/robots/dr002/we11/we11.xml"',
            common,
        )
        self.assertTrue((REPO_ROOT / "src/unilab/assets/robots/dr002/we11/we11.xml").is_file())
