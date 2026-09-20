"""PE01 vector math; quaternions use wxyz throughout the MuJoCo contract."""

import numpy as np


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    t = 2 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


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
