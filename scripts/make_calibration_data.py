"""Turn a folder of images into an .npz of calibration samples for INT8 quantization.

Usage:
    python scripts/make_calibration_data.py <image_folder> [--out calibration/samples.npz]
                                            [--max 40] [--input-name input]

Use 20-100 real images similar to what the model will see in production.
Needs Pillow:  pip install pillow
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess(path: Path) -> np.ndarray:
    """Standard ImageNet preprocessing: resize to 256, centre-crop 224, normalise, CHW."""
    img = Image.open(path).convert("RGB")
    width, height = img.size
    scale = 256 / min(width, height)
    img = img.resize(
        (max(224, round(width * scale)), max(224, round(height * scale))),
        Image.Resampling.BILINEAR,
    )
    width, height = img.size
    left, top = (width - 224) // 2, (height - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    array = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
    return array.transpose(2, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--out", type=Path, default=Path("calibration/samples.npz"))
    parser.add_argument("--max", type=int, default=40)
    parser.add_argument("--input-name", default="input")
    args = parser.parse_args()

    files = sorted(p for p in args.folder.rglob("*") if p.suffix.lower() in EXTENSIONS)[: args.max]
    if len(files) < 4:
        raise SystemExit(f"Need at least 4 images, found {len(files)} in {args.folder}")

    batch = np.stack([preprocess(p) for p in files]).astype(np.float32)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **{args.input_name: batch})
    print(f"Saved {len(files)} samples, shape {batch.shape}, to {args.out}")


if __name__ == "__main__":
    main()