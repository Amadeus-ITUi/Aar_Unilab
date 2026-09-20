"""PE03 playback observation record."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PE03Observation:
    actor: np.ndarray
    critic: np.ndarray
