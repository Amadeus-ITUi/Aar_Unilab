"""PE04 vector math; quaternions use wxyz throughout the MuJoCo contract."""

import numpy as np


def _cross3(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # All vectors here are three-dimensional, so no axis normalization or
    # two-dimensional compatibility handling is needed in this hot path.
    return np.stack(
        (
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ),
        axis=-1,
    )


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    t = 2 * _cross3(q[..., 1:], v)
    return v + q[..., :1] * t + _cross3(q[..., 1:], t)


def inverse_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    inverse = q.copy()
    inverse[..., 1:] *= -1
    return rotate(inverse, v)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            a[..., :1] * b[..., :1] - (a[..., 1:] * b[..., 1:]).sum(-1, keepdims=True),
            a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:]),
        ),
        axis=-1,
    )


def wrap(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2 * np.pi) - np.pi
