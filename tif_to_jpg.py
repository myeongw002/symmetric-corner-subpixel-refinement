from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageSequence


TIFF_EXTENSIONS = {".tif", ".tiff"}


def prepare_for_jpeg(image: Image.Image) -> Image.Image:
    """Return an RGB image that can be saved as JPEG."""
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        background.alpha_composite(rgba)
        return background.convert("RGB")

    if image.mode == "RGB":
        return image

    return image.convert("RGB")


def output_path_for(
    source: Path,
    input_root: Path,
    output_root: Path,
    page_index: int | None = None,
) -> Path:
    relative = source.relative_to(input_root) if source != input_root else source.name

    if isinstance(relative, str):
        relative_path = Path(relative)
    else:
        relative_path = relative

    stem = relative_path.stem
    if page_index is not None:
        stem = f"{stem}_p{page_index + 1:03d}"

    return output_root / relative_path.with_name(f"{stem}.jpg")


def convert_tiff_file(
    source: Path,
    input_root: Path,
    output_root: Path,
    quality: int,
    overwrite: bool,
) -> list[Path]:
    saved_paths: list[Path] = []

    with Image.open(source) as image:
        frames = list(ImageSequence.Iterator(image))
        multiple_frames = len(frames) > 1

        for page_index, frame in enumerate(frames):
            destination = output_path_for(
                source,
                input_root,
                output_root,
                page_index if multiple_frames else None,
            )
            destination.parent.mkdir(parents=True, exist_ok=True)

            if destination.exists() and not overwrite:
                print(f"skip: {destination} already exists")
                continue

            jpeg_image = prepare_for_jpeg(frame)
            jpeg_image.save(destination, "JPEG", quality=quality, optimize=True)
            saved_paths.append(destination)
            print(f"saved: {destination}")

    return saved_paths


def find_tiff_files(path: Path, recursive: bool) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in TIFF_EXTENSIONS:
            raise ValueError(f"Not a TIFF file: {path}")
        return [path]

    pattern = "**/*" if recursive else "*"
    return sorted(
        file
        for file in path.glob(pattern)
        if file.is_file() and file.suffix.lower() in TIFF_EXTENSIONS
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert TIFF images to JPG.")
    parser.add_argument("input", type=Path, help="TIFF file or folder containing TIFF files")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("jpg_output"),
        help="Output folder. Default: jpg_output",
    )
    parser.add_argument(
        "-q",
        "--quality",
        type=int,
        default=95,
        help="JPEG quality from 1 to 100. Default: 95",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Convert TIFF files in subfolders too.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JPG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_root = args.output.resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if not 1 <= args.quality <= 100:
        raise ValueError("Quality must be between 1 and 100.")

    input_root = input_path if input_path.is_dir() else input_path.parent
    tiff_files = find_tiff_files(input_path, args.recursive)

    if not tiff_files:
        print("No TIFF files found.")
        return

    total_saved = 0
    for tiff_file in tiff_files:
        total_saved += len(
            convert_tiff_file(
                source=tiff_file,
                input_root=input_root,
                output_root=output_root,
                quality=args.quality,
                overwrite=args.overwrite,
            )
        )

    print(f"Done. Converted {total_saved} JPG file(s).")


if __name__ == "__main__":
    main()
