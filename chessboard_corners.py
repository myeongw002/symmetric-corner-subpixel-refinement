from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from symmetric_subpixel import (
    SymmetricRefineConfig,
    SymmetricRefineInfo,
    refine_symmetric_subpixel,
)


# =========================
# User settings
# =========================
IMAGE_PATH = Path("data") / "imgs" / "leftcamera" / "Im_L_1.png"

BOARD_COLS = 11  # Number of inner corners along the chessboard columns.
BOARD_ROWS = 7  # Number of inner corners along the chessboard rows.

RESIZE_SCALE = 1.0  # Set to 1.0 if you do not want resizing.

OUTPUT_IMAGE_PATH = Path("chessboard_corners_compare_result.jpg")
OUTPUT_CSV_PATH = Path("chessboard_corners_compare.csv")


# OpenCV cornerSubPix parameters
OPENCV_WIN_SIZE = (11, 11)
OPENCV_ZERO_ZONE = (-1, -1)
OPENCV_CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

# Symmetric refinement parameters
SYM_CONFIG = SymmetricRefineConfig(
    half_window=7.0,
    num_samples=256,
    min_radius=0.75,
    rng_seed=0,
    max_iter=50,
    eps=1e-4,
    max_displacement=1.0,
    use_smoothed_gradient=True,
)


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")
    return image


def resize_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return image

    if scale <= 0.0:
        raise ValueError("RESIZE_SCALE must be positive.")

    width = int(round(image.shape[1] * scale))
    height = int(round(image.shape[0] * scale))

    if width <= 0 or height <= 0:
        raise ValueError("Invalid resize scale.")

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def detect_initial_chessboard_corners(
    gray: np.ndarray,
    board_size: tuple[int, int],
) -> np.ndarray:
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    found, corners = cv2.findChessboardCorners(gray, board_size, flags)
    if not found or corners is None:
        raise RuntimeError(
            f"Chessboard corners not found. Check board size: {board_size}"
        )

    return corners.astype(np.float64).reshape(-1, 2)


def refine_opencv_subpixel(
    gray: np.ndarray,
    initial_corners: np.ndarray,
) -> np.ndarray:
    corners = initial_corners.astype(np.float32).reshape(-1, 1, 2).copy()

    refined = cv2.cornerSubPix(
        gray,
        corners,
        winSize=OPENCV_WIN_SIZE,
        zeroZone=OPENCV_ZERO_ZONE,
        criteria=OPENCV_CRITERIA,
    )

    return refined.reshape(-1, 2).astype(np.float64)


def save_comparison_image(
    image: np.ndarray,
    opencv_corners: np.ndarray,
    symmetric_corners: np.ndarray,
    output_path: Path,
) -> None:
    result = image.copy()

    for idx, (cv_pt, sym_pt) in enumerate(zip(opencv_corners, symmetric_corners)):
        cv_xy = tuple(np.round(cv_pt).astype(int))
        sym_xy = tuple(np.round(sym_pt).astype(int))

        cv2.circle(result, cv_xy, 4, (0, 255, 0), -1)
        cv2.circle(result, sym_xy, 3, (0, 0, 255), -1)
        cv2.line(result, cv_xy, sym_xy, (0, 255, 255), 1)
        cv2.putText(
            result,
            str(idx),
            (cv_xy[0] + 4, cv_xy[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)


def save_comparison_csv(
    output_path: Path,
    initial_corners: np.ndarray,
    opencv_corners: np.ndarray,
    symmetric_corners: np.ndarray,
    symmetric_infos: list[SymmetricRefineInfo],
    resize_scale: float,
) -> None:
    diff = symmetric_corners - opencv_corners
    dist = np.linalg.norm(diff, axis=1)
    inv_scale = 1.0 / resize_scale

    rows = []
    for idx, info in enumerate(symmetric_infos):
        init_x, init_y = initial_corners[idx]
        cv_x, cv_y = opencv_corners[idx]
        sym_x, sym_y = symmetric_corners[idx]
        dx, dy = diff[idx]

        rows.append(
            [
                idx,
                init_x,
                init_y,
                cv_x,
                cv_y,
                sym_x,
                sym_y,
                dx,
                dy,
                dist[idx],
                init_x * inv_scale,
                init_y * inv_scale,
                cv_x * inv_scale,
                cv_y * inv_scale,
                sym_x * inv_scale,
                sym_y * inv_scale,
                info.cost,
                info.iterations,
                int(info.converged),
                info.valid_pairs,
                info.displacement_from_initial,
                int(info.rejected),
            ]
        )

    header = (
        "corner_id,"
        "initial_x,initial_y,"
        "opencv_x,opencv_y,"
        "symmetric_x,symmetric_y,"
        "sym_minus_opencv_dx,sym_minus_opencv_dy,"
        "sym_minus_opencv_dist_px,"
        "initial_x_original,initial_y_original,"
        "opencv_x_original,opencv_y_original,"
        "symmetric_x_original,symmetric_y_original,"
        "symmetric_cost,symmetric_iterations,symmetric_converged,"
        "symmetric_valid_pairs,symmetric_displacement_from_initial,symmetric_rejected"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output_path,
        np.asarray(rows, dtype=np.float64),
        delimiter=",",
        header=header,
        comments="",
        fmt="%.10f",
    )


def print_summary(
    opencv_corners: np.ndarray,
    symmetric_corners: np.ndarray,
    symmetric_infos: list[SymmetricRefineInfo],
) -> None:
    diff = symmetric_corners - opencv_corners
    dist = np.linalg.norm(diff, axis=1)

    converged_count = sum(info.converged for info in symmetric_infos)
    rejected_count = sum(info.rejected for info in symmetric_infos)

    print(f"Found {len(opencv_corners)} corners.")
    print(f"Symmetric converged: {converged_count}/{len(symmetric_infos)}")
    print(f"Symmetric rejected: {rejected_count}/{len(symmetric_infos)}")
    print("")
    print("Difference: symmetric - OpenCV")
    print(f"  mean dx       : {np.mean(diff[:, 0]):.6f} px")
    print(f"  mean dy       : {np.mean(diff[:, 1]):.6f} px")
    print(f"  mean distance : {np.mean(dist):.6f} px")
    print(f"  median dist   : {np.median(dist):.6f} px")
    print(f"  max distance  : {np.max(dist):.6f} px")


def main() -> None:
    board_size = (BOARD_COLS, BOARD_ROWS)

    image = load_image(IMAGE_PATH)
    image_proc = resize_image(image, RESIZE_SCALE)
    gray = cv2.cvtColor(image_proc, cv2.COLOR_BGR2GRAY)

    initial_corners = detect_initial_chessboard_corners(gray, board_size)
    opencv_corners = refine_opencv_subpixel(
        gray=gray,
        initial_corners=initial_corners,
    )
    symmetric_corners, symmetric_infos = refine_symmetric_subpixel(
        gray=gray,
        initial_corners=opencv_corners,
        config=SYM_CONFIG,
    )

    save_comparison_image(
        image=image_proc,
        opencv_corners=opencv_corners,
        symmetric_corners=symmetric_corners,
        output_path=OUTPUT_IMAGE_PATH,
    )
    save_comparison_csv(
        output_path=OUTPUT_CSV_PATH,
        initial_corners=initial_corners,
        opencv_corners=opencv_corners,
        symmetric_corners=symmetric_corners,
        symmetric_infos=symmetric_infos,
        resize_scale=RESIZE_SCALE,
    )

    print_summary(opencv_corners, symmetric_corners, symmetric_infos)
    print("")
    print(f"Saved result image: {OUTPUT_IMAGE_PATH}")
    print(f"Saved comparison CSV: {OUTPUT_CSV_PATH}")
    print("")
    print("Legend:")
    print("  Green dot : OpenCV cornerSubPix")
    print("  Red dot   : Symmetric refinement")
    print("  Yellow line: Difference vector")


if __name__ == "__main__":
    main()
