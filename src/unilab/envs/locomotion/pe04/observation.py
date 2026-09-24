"""PE04 playback observation record."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PE04Observation:
    actor: np.ndarray
    critic: np.ndarray
