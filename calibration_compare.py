from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from symmetric_subpixel import (
    SymmetricRefineConfig,
    SymmetricRefineInfo,
    refine_symmetric_subpixel,
)


# ============================================================
# User settings
# ============================================================
IMAGE_DIR = Path("data") / "imgs" / "leftcamera"
IMAGE_GLOB = "*.png"

BOARD_COLS = 11  # Number of inner corners along chessboard columns.
BOARD_ROWS = 7  # Number of inner corners along chessboard rows.
SQUARE_SIZE = 1.0  # Real square size. Use mm, cm, m, or 1.0 if scale is not important.

# If the original image fails detection, set DETECT_SCALE to 0.5.
# The detected corners are then scaled back to original resolution and refined there.
DETECT_SCALE = 1.0

OUTPUT_DIR = Path("calibration_compare_output")

# Initial detection method:
#   "classic" : findChessboardCorners + cornerSubPix
#   "sb"      : findChessboardCornersSB for initial detection
INITIAL_DETECTOR = "classic"

# OpenCV cornerSubPix parameters
OPENCV_WIN_SIZE = (11, 11)
OPENCV_ZERO_ZONE = (-1, -1)
OPENCV_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    50,
    1e-4,
)

# Symmetric refinement parameters
SYM_HALF_WINDOW = 7.0
SYM_NUM_SAMPLES = 256
SYM_MIN_RADIUS = 0.75
SYM_RNG_SEED = 0
SYM_MAX_ITER = 80
SYM_EPS = 1e-5
SYM_MAX_DISPLACEMENT = 1.0  # reject if symmetric result moves too far from OpenCV result
SYM_USE_SMOOTHED_GRADIENT = True

SYM_CONFIG = SymmetricRefineConfig(
    half_window=SYM_HALF_WINDOW,
    num_samples=SYM_NUM_SAMPLES,
    min_radius=SYM_MIN_RADIUS,
    rng_seed=SYM_RNG_SEED,
    max_iter=SYM_MAX_ITER,
    eps=SYM_EPS,
    max_displacement=SYM_MAX_DISPLACEMENT,
    use_smoothed_gradient=SYM_USE_SMOOTHED_GRADIENT,
)

# Calibration flags.
# Start with 0. Add flags only when needed, e.g. cv2.CALIB_FIX_K3.
CALIB_FLAGS = 0


@dataclass
class ViewCorners:
    image_path: Path
    image_size: tuple[int, int]
    initial_corners: np.ndarray
    opencv_corners: np.ndarray
    symmetric_corners: np.ndarray
    symmetric_infos: list[SymmetricRefineInfo]


@dataclass
class CalibrationResult:
    method: str
    rms: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]
    per_point_errors: np.ndarray
    per_view_mean_errors: np.ndarray
    mean_error: float
    median_error: float
    max_error: float
    std_error: float


# ============================================================
# Basic IO and chessboard detection
# ============================================================
def list_images(image_dir: Path, image_glob: str) -> list[Path]:
    image_paths = sorted(image_dir.glob(image_glob))
    if not image_paths:
        raise FileNotFoundError(f"No images found: {image_dir / image_glob}")
    return image_paths


def load_gray(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def resize_for_detection(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    if scale <= 0.0:
        raise ValueError("DETECT_SCALE must be positive.")

    width = int(round(gray.shape[1] * scale))
    height = int(round(gray.shape[0] * scale))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid DETECT_SCALE.")

    return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)


def detect_initial_chessboard_corners(
    gray_original: np.ndarray,
    board_size: tuple[int, int],
    detector: str,
    detect_scale: float,
) -> np.ndarray | None:
    gray_detect = resize_for_detection(gray_original, detect_scale)

    if detector == "classic":
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
            # FILTER_QUADS can reject difficult original-resolution images.
            # | cv2.CALIB_CB_FILTER_QUADS
        )
        found, corners = cv2.findChessboardCorners(gray_detect, board_size, flags)
    elif detector == "sb":
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray_detect, board_size, flags)
    else:
        raise ValueError("INITIAL_DETECTOR must be 'classic' or 'sb'.")

    if not found or corners is None:
        return None

    corners = corners.astype(np.float64).reshape(-1, 2)

    if detect_scale != 1.0:
        corners /= detect_scale

    return corners


def refine_opencv_subpixel(gray: np.ndarray, initial_corners: np.ndarray) -> np.ndarray:
    corners = initial_corners.astype(np.float32).reshape(-1, 1, 2).copy()
    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=OPENCV_WIN_SIZE,
        zeroZone=OPENCV_ZERO_ZONE,
        criteria=OPENCV_CRITERIA,
    )
    return refined.reshape(-1, 2).astype(np.float64)


# Calibration and reprojection evaluation
# ============================================================
def build_object_points(board_size: tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = board_size
    objp = np.zeros((rows * cols, 3), np.float32)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, 0] = grid_x.reshape(-1) * square_size
    objp[:, 1] = grid_y.reshape(-1) * square_size
    return objp


def compute_reprojection_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: Iterable[np.ndarray],
    tvecs: Iterable[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    all_errors: list[np.ndarray] = []
    per_view_means: list[float] = []

    for objp, imgp, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        projected = projected.reshape(-1, 2)
        imgp2 = imgp.reshape(-1, 2)
        errors = np.linalg.norm(imgp2 - projected, axis=1)
        all_errors.append(errors)
        per_view_means.append(float(np.mean(errors)))

    return np.concatenate(all_errors), np.asarray(per_view_means, dtype=np.float64)


def calibrate_and_evaluate(
    method: str,
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> CalibrationResult:
    image_points_cv = [pts.astype(np.float32).reshape(-1, 1, 2) for pts in image_points]
    object_points_cv = [pts.astype(np.float32).reshape(-1, 1, 3) for pts in object_points]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points_cv,
        image_points_cv,
        image_size,
        None,
        None,
        flags=CALIB_FLAGS,
    )

    per_point_errors, per_view_mean_errors = compute_reprojection_errors(
        object_points=object_points,
        image_points=image_points,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
    )

    return CalibrationResult(
        method=method,
        rms=float(rms),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=tuple(rvecs),
        tvecs=tuple(tvecs),
        per_point_errors=per_point_errors,
        per_view_mean_errors=per_view_mean_errors,
        mean_error=float(np.mean(per_point_errors)),
        median_error=float(np.median(per_point_errors)),
        max_error=float(np.max(per_point_errors)),
        std_error=float(np.std(per_point_errors)),
    )


# ============================================================
# Save outputs
# ============================================================
def save_result_summary(results: list[CalibrationResult], output_path: Path) -> None:
    rows = []
    for res in results:
        dist = res.dist_coeffs.reshape(-1)
        padded_dist = np.full(8, np.nan, dtype=np.float64)
        padded_dist[: min(len(dist), len(padded_dist))] = dist[: min(len(dist), len(padded_dist))]

        rows.append(
            [
                res.method,
                res.rms,
                res.mean_error,
                res.median_error,
                res.max_error,
                res.std_error,
                res.camera_matrix[0, 0],
                res.camera_matrix[1, 1],
                res.camera_matrix[0, 2],
                res.camera_matrix[1, 2],
                *padded_dist.tolist(),
            ]
        )

    header = [
        "method",
        "rms",
        "mean_error_px",
        "median_error_px",
        "max_error_px",
        "std_error_px",
        "fx",
        "fy",
        "cx",
        "cy",
        "dist_0",
        "dist_1",
        "dist_2",
        "dist_3",
        "dist_4",
        "dist_5",
        "dist_6",
        "dist_7",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def save_per_view_errors(
    image_paths: list[Path],
    opencv_result: CalibrationResult,
    symmetric_result: CalibrationResult,
    output_path: Path,
) -> None:
    rows = []
    for idx, image_path in enumerate(image_paths):
        e_cv = opencv_result.per_view_mean_errors[idx]
        e_sym = symmetric_result.per_view_mean_errors[idx]
        improvement = (e_cv - e_sym) / max(e_cv, 1e-12) * 100.0
        rows.append([idx, image_path.name, e_cv, e_sym, improvement])

    header = "view_id,image,opencv_mean_error_px,symmetric_mean_error_px,improvement_percent"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        np.asarray(rows, dtype=object),
        delimiter=",",
        header=header,
        comments="",
        fmt="%s",
    )


def save_corner_displacements(
    views: list[ViewCorners],
    output_path: Path,
) -> None:
    rows = []
    for view_id, view in enumerate(views):
        diff = view.symmetric_corners - view.opencv_corners
        dist = np.linalg.norm(diff, axis=1)
        for corner_id in range(len(view.opencv_corners)):
            info = view.symmetric_infos[corner_id]
            rows.append(
                [
                    view_id,
                    view.image_path.name,
                    corner_id,
                    view.opencv_corners[corner_id, 0],
                    view.opencv_corners[corner_id, 1],
                    view.symmetric_corners[corner_id, 0],
                    view.symmetric_corners[corner_id, 1],
                    diff[corner_id, 0],
                    diff[corner_id, 1],
                    dist[corner_id],
                    info.cost,
                    info.iterations,
                    int(info.converged),
                    info.valid_pairs,
                    info.displacement_from_initial,
                    int(info.rejected),
                ]
            )

    header = (
        "view_id,image,corner_id,"
        "opencv_x,opencv_y,symmetric_x,symmetric_y,"
        "sym_minus_opencv_dx,sym_minus_opencv_dy,sym_minus_opencv_dist_px,"
        "symmetric_cost,symmetric_iterations,symmetric_converged,"
        "symmetric_valid_pairs,symmetric_displacement_from_initial,symmetric_rejected"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        np.asarray(rows, dtype=object),
        delimiter=",",
        header=header,
        comments="",
        fmt="%s",
    )


def save_visual_check_images(views: list[ViewCorners], output_dir: Path) -> None:
    vis_dir = output_dir / "visual_check"
    vis_dir.mkdir(parents=True, exist_ok=True)

    for view in views:
        image = cv2.imread(str(view.image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        for cv_pt, sym_pt in zip(view.opencv_corners, view.symmetric_corners):
            cv_xy = tuple(np.round(cv_pt).astype(int))
            sym_xy = tuple(np.round(sym_pt).astype(int))
            cv2.circle(image, cv_xy, 4, (0, 255, 0), -1)  # OpenCV: green
            cv2.circle(image, sym_xy, 3, (0, 0, 255), -1)  # Symmetric: red
            cv2.line(image, cv_xy, sym_xy, (0, 255, 255), 1)  # Difference: yellow

        cv2.imwrite(str(vis_dir / f"{view.image_path.stem}_compare.jpg"), image)


# ============================================================
# Main pipeline
# ============================================================
def collect_corners(image_paths: list[Path]) -> tuple[list[ViewCorners], list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int]]:
    board_size = (BOARD_COLS, BOARD_ROWS)
    objp = build_object_points(board_size, SQUARE_SIZE)

    views: list[ViewCorners] = []
    object_points: list[np.ndarray] = []
    opencv_image_points: list[np.ndarray] = []
    symmetric_image_points: list[np.ndarray] = []
    used_image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        image, gray = load_gray(image_path)
        image_size = (gray.shape[1], gray.shape[0])

        if used_image_size is None:
            used_image_size = image_size
        elif image_size != used_image_size:
            print(f"[SKIP] {image_path.name}: image size mismatch {image_size} != {used_image_size}")
            continue

        initial = detect_initial_chessboard_corners(
            gray_original=gray,
            board_size=board_size,
            detector=INITIAL_DETECTOR,
            detect_scale=DETECT_SCALE,
        )

        if initial is None:
            print(f"[SKIP] {image_path.name}: chessboard not found")
            continue

        if len(initial) != BOARD_COLS * BOARD_ROWS:
            print(f"[SKIP] {image_path.name}: invalid corner count {len(initial)}")
            continue

        opencv_corners = refine_opencv_subpixel(gray, initial)

        # Important: symmetric refinement starts from OpenCV subpixel corners.
        symmetric_corners, symmetric_infos = refine_symmetric_subpixel(
            gray,
            opencv_corners,
            config=SYM_CONFIG,
        )

        views.append(
            ViewCorners(
                image_path=image_path,
                image_size=image_size,
                initial_corners=initial,
                opencv_corners=opencv_corners,
                symmetric_corners=symmetric_corners,
                symmetric_infos=symmetric_infos,
            )
        )
        object_points.append(objp.copy())
        opencv_image_points.append(opencv_corners.copy())
        symmetric_image_points.append(symmetric_corners.copy())

        rejected_count = sum(info.rejected for info in symmetric_infos)
        converged_count = sum(info.converged for info in symmetric_infos)
        print(
            f"[OK] {image_path.name}: corners={len(initial)}, "
            f"sym_converged={converged_count}/{len(symmetric_infos)}, "
            f"sym_rejected={rejected_count}/{len(symmetric_infos)}"
        )

    if used_image_size is None:
        raise RuntimeError("No valid image was loaded.")

    if len(views) < 3:
        raise RuntimeError(
            f"Only {len(views)} valid views found. Calibration needs multiple views."
        )

    return views, object_points, opencv_image_points, symmetric_image_points, used_image_size


def print_comparison(opencv_result: CalibrationResult, symmetric_result: CalibrationResult) -> None:
    def improvement(cv_value: float, sym_value: float) -> float:
        return (cv_value - sym_value) / max(cv_value, 1e-12) * 100.0

    print("\n================ Calibration Comparison ================")
    print(f"Used views: {len(opencv_result.per_view_mean_errors)}")
    print("\nMetric                         OpenCV        Symmetric     Improvement")
    print("---------------------------------------------------------------")
    print(
        f"RMS reprojection error      {opencv_result.rms:10.6f}  "
        f"{symmetric_result.rms:10.6f}  "
        f"{improvement(opencv_result.rms, symmetric_result.rms):10.3f}%"
    )
    print(
        f"Mean reprojection error     {opencv_result.mean_error:10.6f}  "
        f"{symmetric_result.mean_error:10.6f}  "
        f"{improvement(opencv_result.mean_error, symmetric_result.mean_error):10.3f}%"
    )
    print(
        f"Median reprojection error   {opencv_result.median_error:10.6f}  "
        f"{symmetric_result.median_error:10.6f}  "
        f"{improvement(opencv_result.median_error, symmetric_result.median_error):10.3f}%"
    )
    print(
        f"Max reprojection error      {opencv_result.max_error:10.6f}  "
        f"{symmetric_result.max_error:10.6f}  "
        f"{improvement(opencv_result.max_error, symmetric_result.max_error):10.3f}%"
    )

    print("\nOpenCV camera matrix:")
    print(opencv_result.camera_matrix)
    print("OpenCV distortion:")
    print(opencv_result.dist_coeffs.ravel())

    print("\nSymmetric camera matrix:")
    print(symmetric_result.camera_matrix)
    print("Symmetric distortion:")
    print(symmetric_result.dist_coeffs.ravel())

    if symmetric_result.mean_error < opencv_result.mean_error:
        print("\nResult: symmetric refinement improved mean reprojection error.")
    else:
        print("\nResult: symmetric refinement did NOT improve mean reprojection error.")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(IMAGE_DIR, IMAGE_GLOB)
    print(f"Found {len(image_paths)} candidate images.")

    views, object_points, opencv_points, symmetric_points, image_size = collect_corners(image_paths)

    opencv_result = calibrate_and_evaluate(
        method="opencv_cornerSubPix",
        object_points=object_points,
        image_points=opencv_points,
        image_size=image_size,
    )

    symmetric_result = calibrate_and_evaluate(
        method="symmetric_refinement",
        object_points=object_points,
        image_points=symmetric_points,
        image_size=image_size,
    )

    print_comparison(opencv_result, symmetric_result)

    save_result_summary(
        [opencv_result, symmetric_result],
        OUTPUT_DIR / "calibration_summary.csv",
    )
    save_per_view_errors(
        [v.image_path for v in views],
        opencv_result,
        symmetric_result,
        OUTPUT_DIR / "per_view_errors.csv",
    )
    save_corner_displacements(
        views,
        OUTPUT_DIR / "corner_displacements.csv",
    )
    save_visual_check_images(views, OUTPUT_DIR)

    print("\nSaved outputs:")
    print(f"  {OUTPUT_DIR / 'calibration_summary.csv'}")
    print(f"  {OUTPUT_DIR / 'per_view_errors.csv'}")
    print(f"  {OUTPUT_DIR / 'corner_displacements.csv'}")
    print(f"  {OUTPUT_DIR / 'visual_check'}")


if __name__ == "__main__":
    main()
