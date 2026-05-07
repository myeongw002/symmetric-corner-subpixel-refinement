from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError as exc:
    raise ImportError("This module requires scipy. Install with: pip install scipy") from exc


@dataclass(frozen=True)
class SymmetricRefineConfig:
    half_window: float = 7.0
    num_samples: int = 256
    min_radius: float = 0.75
    rng_seed: int = 0
    max_iter: int = 80
    eps: float = 1e-5
    max_displacement: float = 1.0
    use_smoothed_gradient: bool = True
    min_valid_pairs: int = 8


@dataclass
class SymmetricRefineInfo:
    cost: float
    iterations: int
    converged: bool
    valid_pairs: int
    displacement_from_initial: float
    rejected: bool


def make_symmetric_offsets(
    half_window: float,
    num_samples: int,
    min_radius: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=num_samples)

    # Uniform sampling over disk area, excluding the immediate center.
    r2_min = min_radius * min_radius
    r2_max = half_window * half_window
    radii = np.sqrt(rng.uniform(r2_min, r2_max, size=num_samples))

    dx = radii * np.cos(angles)
    dy = radii * np.sin(angles)
    return np.column_stack([dx, dy]).astype(np.float64)


def bilinear_sample(image: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    ia = image[y0c, x0c].astype(np.float64)
    ib = image[y0c, x1c].astype(np.float64)
    ic = image[y1c, x0c].astype(np.float64)
    id_ = image[y1c, x1c].astype(np.float64)

    # Use unclipped x0/y0 for correct fractional weights inside valid region.
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    return wa * ia + wb * ib + wc * ic + wd * id_


def valid_symmetric_pairs_mask(
    q: np.ndarray,
    offsets: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    h, w = image_shape[:2]
    p_plus = q[None, :] + offsets
    p_minus = q[None, :] - offsets
    margin = 2.0

    valid_plus = (
        (p_plus[:, 0] >= margin)
        & (p_plus[:, 0] < w - 1 - margin)
        & (p_plus[:, 1] >= margin)
        & (p_plus[:, 1] < h - 1 - margin)
    )
    valid_minus = (
        (p_minus[:, 0] >= margin)
        & (p_minus[:, 0] < w - 1 - margin)
        & (p_minus[:, 1] >= margin)
        & (p_minus[:, 1] < h - 1 - margin)
    )
    return valid_plus & valid_minus


def symmetric_residual_and_jacobian(
    gray_float: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    q: np.ndarray,
    offsets: np.ndarray,
    min_valid_pairs: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    mask = valid_symmetric_pairs_mask(q, offsets, gray_float.shape)
    valid_offsets = offsets[mask]
    valid_pairs = len(valid_offsets)

    if valid_pairs < min_valid_pairs:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0, 2), dtype=np.float64),
            valid_pairs,
        )

    p_plus = q[None, :] + valid_offsets
    p_minus = q[None, :] - valid_offsets

    i_plus = bilinear_sample(gray_float, p_plus)
    i_minus = bilinear_sample(gray_float, p_minus)
    residual = i_plus - i_minus

    gx_plus = bilinear_sample(grad_x, p_plus)
    gy_plus = bilinear_sample(grad_y, p_plus)
    gx_minus = bilinear_sample(grad_x, p_minus)
    gy_minus = bilinear_sample(grad_y, p_minus)

    # r(q) = I(q + delta) - I(q - delta)
    # dr/dq = grad I(q + delta) - grad I(q - delta)
    jacobian = np.column_stack([gx_plus - gx_minus, gy_plus - gy_minus])

    return residual, jacobian, valid_pairs


def refine_one_corner_symmetric(
    gray_float: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    initial_q: np.ndarray,
    offsets: np.ndarray,
    config: SymmetricRefineConfig,
) -> tuple[np.ndarray, SymmetricRefineInfo]:
    q0 = initial_q.astype(np.float64).copy()

    residual0, _, valid_pairs0 = symmetric_residual_and_jacobian(
        gray_float,
        grad_x,
        grad_y,
        q0,
        offsets,
        config.min_valid_pairs,
    )
    if len(residual0) < config.min_valid_pairs:
        return q0, SymmetricRefineInfo(
            cost=float("inf"),
            iterations=0,
            converged=False,
            valid_pairs=valid_pairs0,
            displacement_from_initial=0.0,
            rejected=True,
        )

    def fun(q: np.ndarray) -> np.ndarray:
        residual, _, _ = symmetric_residual_and_jacobian(
            gray_float,
            grad_x,
            grad_y,
            q,
            offsets,
            config.min_valid_pairs,
        )
        return residual

    def jac(q: np.ndarray) -> np.ndarray:
        _, jacobian, _ = symmetric_residual_and_jacobian(
            gray_float,
            grad_x,
            grad_y,
            q,
            offsets,
            config.min_valid_pairs,
        )
        return jacobian

    try:
        result = least_squares(
            fun=fun,
            x0=q0,
            jac=jac,
            method="lm",
            max_nfev=config.max_iter,
            xtol=config.eps,
            ftol=1e-10,
            gtol=1e-10,
        )
    except Exception:
        return q0, SymmetricRefineInfo(
            cost=0.5 * float(np.dot(residual0, residual0)),
            iterations=0,
            converged=False,
            valid_pairs=valid_pairs0,
            displacement_from_initial=0.0,
            rejected=True,
        )

    q_refined = result.x.astype(np.float64)
    displacement = float(np.linalg.norm(q_refined - q0))

    if displacement > config.max_displacement:
        return q0, SymmetricRefineInfo(
            cost=0.5 * float(np.dot(residual0, residual0)),
            iterations=int(result.nfev),
            converged=False,
            valid_pairs=valid_pairs0,
            displacement_from_initial=displacement,
            rejected=True,
        )

    residual_final, _, valid_pairs_final = symmetric_residual_and_jacobian(
        gray_float,
        grad_x,
        grad_y,
        q_refined,
        offsets,
        config.min_valid_pairs,
    )
    final_cost = 0.5 * float(np.dot(residual_final, residual_final))

    return q_refined, SymmetricRefineInfo(
        cost=final_cost,
        iterations=int(result.nfev),
        converged=bool(result.success),
        valid_pairs=valid_pairs_final,
        displacement_from_initial=displacement,
        rejected=False,
    )


def refine_symmetric_subpixel(
    gray: np.ndarray,
    initial_corners: np.ndarray,
    config: SymmetricRefineConfig | None = None,
) -> tuple[np.ndarray, list[SymmetricRefineInfo]]:
    if config is None:
        config = SymmetricRefineConfig()

    gray_float = gray.astype(np.float64)

    if config.use_smoothed_gradient:
        grad_source = cv2.GaussianBlur(gray_float, (3, 3), 0)
    else:
        grad_source = gray_float

    grad_x = cv2.Sobel(grad_source, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grad_source, cv2.CV_64F, 0, 1, ksize=3)

    offsets = make_symmetric_offsets(
        half_window=config.half_window,
        num_samples=config.num_samples,
        min_radius=config.min_radius,
        seed=config.rng_seed,
    )

    refined = np.zeros_like(initial_corners, dtype=np.float64)
    infos: list[SymmetricRefineInfo] = []

    for idx, q0 in enumerate(initial_corners):
        q_refined, info = refine_one_corner_symmetric(
            gray_float=gray_float,
            grad_x=grad_x,
            grad_y=grad_y,
            initial_q=q0,
            offsets=offsets,
            config=config,
        )
        refined[idx] = q_refined
        infos.append(info)

    return refined, infos
