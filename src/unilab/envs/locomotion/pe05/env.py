"""Single-robot playback observation container for the PE05 contract."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PE05Observation:
    actor: np.ndarray
    critic: np.ndarray
