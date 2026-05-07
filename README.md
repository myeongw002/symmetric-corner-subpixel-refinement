# Symmetric Corner Subpixel Refinement

Utilities for converting calibration images, detecting chessboard corners, and
comparing OpenCV `cornerSubPix` results with symmetric subpixel refinement.

## Requirements

Install the Python packages used by the scripts:

```powershell
pip install opencv-python numpy scipy pillow
```

## Files

- `tif_to_jpg.py`: Converts `.tif` / `.tiff` images to `.jpg`.
- `chessboard_corners.py`: Loads one image, detects chessboard corners, refines
  them with OpenCV and symmetric subpixel refinement, then saves a visual check
  image and CSV.
- `calibration_compare.py`: Runs calibration on multiple images and compares
  OpenCV subpixel corners against symmetric refined corners.
- `symmetric_subpixel.py`: Reusable module for symmetric subpixel refinement.

## Convert TIFF To JPG

Convert a folder recursively:

```powershell
python tif_to_jpg.py calib_example -o jpg_output -r
```

Convert one file:

```powershell
python tif_to_jpg.py calib_example\Image1.tif -o jpg_output
```

Use `--overwrite` to replace existing JPG files.

## Single Image Corner Check

Edit the user settings at the top of `chessboard_corners.py`:

```python
IMAGE_PATH = Path("data") / "imgs" / "leftcamera" / "Im_L_1.png"

BOARD_COLS = 11
BOARD_ROWS = 7

RESIZE_SCALE = 1.0

OUTPUT_IMAGE_PATH = Path("chessboard_corners_compare_result.jpg")
OUTPUT_CSV_PATH = Path("chessboard_corners_compare.csv")
```

`BOARD_COLS` and `BOARD_ROWS` are the number of inner chessboard corners, not
the number of squares.

Run:

```powershell
python chessboard_corners.py
```

Outputs:

- `chessboard_corners_compare_result.jpg`
- `chessboard_corners_compare.csv`

Visualization legend:

- Green dot: OpenCV `cornerSubPix`
- Red dot: symmetric refinement
- Yellow line: displacement between the two points

## Calibration Comparison

Edit the user settings at the top of `calibration_compare.py`:

```python
IMAGE_DIR = Path("data") / "imgs" / "leftcamera"
IMAGE_GLOB = "*.png"

BOARD_COLS = 11
BOARD_ROWS = 7
SQUARE_SIZE = 1.0

DETECT_SCALE = 1.0
OUTPUT_DIR = Path("calibration_compare_output")
INITIAL_DETECTOR = "classic"
```

`CALIB_FLAGS` controls the OpenCV `cv2.calibrateCamera` model and constraints.
Start with:

```python
CALIB_FLAGS = 0
```

Common examples:

```python
# Use the default 5-coefficient distortion model:
# k1, k2, p1, p2, k3
CALIB_FLAGS = 0

# Use the rational distortion model:
# k1, k2, p1, p2, k3, k4, k5, k6
CALIB_FLAGS = cv2.CALIB_RATIONAL_MODEL

# Ignore tangential distortion:
# p1 = p2 = 0
CALIB_FLAGS = cv2.CALIB_ZERO_TANGENT_DIST

# Fix k3 to zero during optimization:
CALIB_FLAGS = cv2.CALIB_FIX_K3

# Combine multiple flags:
CALIB_FLAGS = (
    cv2.CALIB_RATIONAL_MODEL
    | cv2.CALIB_ZERO_TANGENT_DIST
    | cv2.CALIB_FIX_K3
)
```

Useful `calibrateCamera` flags:

- `cv2.CALIB_USE_INTRINSIC_GUESS`: start from a provided camera matrix and
  distortion coefficients.
- `cv2.CALIB_FIX_PRINCIPAL_POINT`: keep `cx`, `cy` fixed.
- `cv2.CALIB_FIX_FOCAL_LENGTH`: keep `fx`, `fy` fixed.
- `cv2.CALIB_FIX_ASPECT_RATIO`: keep the `fx / fy` ratio fixed.
- `cv2.CALIB_ZERO_TANGENT_DIST`: set `p1`, `p2` to zero and keep them fixed.
- `cv2.CALIB_FIX_K1` ... `cv2.CALIB_FIX_K6`: keep selected radial distortion
  coefficients fixed.
- `cv2.CALIB_RATIONAL_MODEL`: enable `k4`, `k5`, `k6`.
- `cv2.CALIB_THIN_PRISM_MODEL`: enable `s1`, `s2`, `s3`, `s4`.
- `cv2.CALIB_FIX_S1_S2_S3_S4`: keep thin prism coefficients fixed.
- `cv2.CALIB_TILTED_MODEL`: enable tilted sensor coefficients `tauX`, `tauY`.
- `cv2.CALIB_FIX_TAUX_TAUY`: keep tilted sensor coefficients fixed.
- `cv2.CALIB_USE_QR`: use QR decomposition instead of SVD. Faster, but
  potentially less precise.
- `cv2.CALIB_USE_LU`: use LU decomposition instead of SVD. Faster, but
  potentially less stable.

Run:

```powershell
python calibration_compare.py
```

Outputs are saved under `calibration_compare_output`:

- `calibration_summary.csv`
- `per_view_errors.csv`
- `corner_displacements.csv`
- `visual_check\*_compare.jpg`

## Symmetric Subpixel Module

The symmetric refinement logic is implemented in `symmetric_subpixel.py`.
Configure it with `SymmetricRefineConfig`:

```python
from symmetric_subpixel import SymmetricRefineConfig, refine_symmetric_subpixel

config = SymmetricRefineConfig(
    half_window=7.0,
    num_samples=256,
    min_radius=0.75,
    rng_seed=0,
    max_iter=80,
    eps=1e-5,
    max_displacement=1.0,
    use_smoothed_gradient=True,
)

symmetric_corners, infos = refine_symmetric_subpixel(
    gray=gray,
    initial_corners=opencv_corners,
    config=config,
)
```

The refinement minimizes the intensity symmetry residual around each corner:

```text
I(q + delta) - I(q - delta)
```

If a refined point moves farther than `max_displacement`, the result is rejected
and the initial point is kept.

## Reference

The symmetric subpixel refinement implementation was written with reference to:

- Zezhun Shi, **Accurate Checkerboard Corner Detection under Defoucs**,
  arXiv:2410.13371v1.
  https://arxiv.org/html/2410.13371v1

In particular, the module follows the paper's symmetry-based refinement idea:
sample subpixel offsets around a checkerboard corner and optimize the corner
position so that symmetric intensity pairs have minimal difference.

## Notes

- Start symmetric refinement from OpenCV `cornerSubPix` results for stability.
- If chessboard detection fails, check `BOARD_COLS`, `BOARD_ROWS`, image path,
  and `DETECT_SCALE`.
- `DETECT_SCALE` can help when full-resolution detection is difficult. Detected
  corners are scaled back before final refinement in `calibration_compare.py`.
