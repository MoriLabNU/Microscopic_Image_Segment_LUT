"""Single-image LUT segmentation for Fiji / command-line use."""

import argparse
import os

import numpy as np
from PIL import Image

from lut_core import apply_lut_segmentation, load_lut

UPSCALE = 1
SAMPLING_INTERVAL = 4


def _default_lut_path():
    here = os.path.dirname(os.path.abspath(__file__))
    bundled = os.path.join(here, "..", "lut", "Seg2_LUT_HV_2_{}.npy".format(SAMPLING_INTERVAL))
    if os.path.isfile(bundled):
        return os.path.normpath(bundled)
    repo_lut = os.path.join(
        os.path.dirname(__file__), "..", "..", "3_Test_using_LUT",
        "Seg2_LUT_HV_2_{}.npy".format(SAMPLING_INTERVAL),
    )
    return os.path.normpath(repo_lut)


def segment_file(input_path, output_path, lut_path, sampling_interval, threshold=0.5):
    img_lr = Image.open(input_path).convert("L")
    img_lr = np.asarray(img_lr, dtype="float32")[:, :, np.newaxis]
    lut = load_lut(lut_path, upscale=UPSCALE)
    prob = apply_lut_segmentation(
        img_lr, lut, sampling_interval, upscale=UPSCALE
    )
    mask = (prob >= threshold).astype(np.uint8) * 255
    Image.fromarray(mask).save(output_path)


def main():
    parser = argparse.ArgumentParser(description="SEG-LUT single-image inference")
    parser.add_argument("--input", required=True, help="Input grayscale image (PNG)")
    parser.add_argument("--output", required=True, help="Output binary mask (PNG)")
    parser.add_argument(
        "--lut",
        default=_default_lut_path(),
        help="Path to .npy LUT (default: bundled Seg2_LUT_HV_2_4.npy)",
    )
    parser.add_argument(
        "--sampling-interval",
        type=int,
        default=SAMPLING_INTERVAL,
        help="Must match LUT export (bundled LUT uses 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for foreground",
    )
    args = parser.parse_args()
    if not os.path.isfile(args.input):
        raise SystemExit("Input not found: {}".format(args.input))
    if not os.path.isfile(args.lut):
        raise SystemExit("LUT not found: {}".format(args.lut))
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    segment_file(
        args.input,
        args.output,
        args.lut,
        args.sampling_interval,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
