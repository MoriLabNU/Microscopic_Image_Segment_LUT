# Seg-LUT_HV

Lookup-table (LUT) accelerated image segmentation: train a small patch CNN, distill it into a 4D neighborhood LUT, and run fast inference in Python or Fiji/ImageJ.

## Citation

If you use this code, please cite our paper:

```bibtex
@article{seglut_hv_tbd,
  title   = {TBD},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD},
  note    = {Replace this placeholder with the official BibTeX entry.}
}
```

## Overview

The repository is organized as a four-stage pipeline:

| Stage | Directory | Purpose |
|-------|-----------|---------|
| 1 | [`1_Train_deep_model/`](1_Train_deep_model/) | Train the segmentation CNN |
| 2 | [`2_Transfer_to_LUT/`](2_Transfer_to_LUT/) | Enumerate 4D inputs and export a `.npy` LUT |
| 3 | [`3_Test_using_LUT/`](3_Test_using_LUT/) | Batch evaluation with LUT + simplex interpolation |
| 4 | [`4_toImageJ/`](4_toImageJ/) | Fiji plugin (Jython) calling a Python inference script |

Bundled LUT (for testing and the plugin):

- **File:** `3_Test_using_LUT/Seg2_LUT_HV_2_4.npy` (duplicate under `4_toImageJ/lut/`)
- **`SAMPLING_INTERVAL`:** `4` (must match when exporting or loading the LUT)

Classical baselines: `otsu.py` and `triangle.py` in stage 1.

## Requirements

- Python 3.8+ (3.10 recommended)
- PyTorch with CUDA optional
- [Fiji](https://fiji.sc/) for the ImageJ plugin (Jython + a separate CPython with NumPy)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Training without Weights & Biases:

```bash
set WANDB_MODE=disabled   # Windows
export WANDB_MODE=disabled  # Linux/macOS
```

Optional W&B settings: `WANDB_PROJECT`, `WANDB_ENTITY`.

## Dataset layout

Images are **not** included in this repository. Place data as follows (each stage uses paths relative to its folder):

```
train/
  image/*.png
  mask/*.png
val/
  image/*.png
  mask/*.png
test/
  image/*.png
  mask/*.png
```

## Usage

### 1. Train

```bash
cd 1_Train_deep_model
python Train_Model_Seg.py
```

Checkpoints are written under `checkpoint/` (gitignored). Adjust `VERSION`, paths, and hyperparameters at the top of the script.

### 2. Export LUT

Place your trained weights (e.g. `model_G_i105000_best.pth`) and set `MODEL_PATH` in `2_Transfer_to_LUT/Transfer_Model_Seg.py`. Use the same `SAMPLING_INTERVAL` as the LUT you intend to ship (default **4**).

```bash
cd 2_Transfer_to_LUT
python Transfer_Model_Seg.py
```

Output: `Seg2_LUT_HV_2_<interval>.npy`.

### 3. Test with LUT

```bash
cd 3_Test_using_LUT
python test_lut.py
```

Settings: [`config.py`](3_Test_using_LUT/config.py). Predictions go to `output_Seg2_LUT_HV_2_test_<interval>bit/` (gitignored).

### 4. Single-image CLI (also used by Fiji)

```bash
python 4_toImageJ/py_lut/segment_lut.py \
  --input path/to/image.png \
  --output path/to/mask.png \
  --lut 3_Test_using_LUT/Seg2_LUT_HV_2_4.npy \
  --sampling-interval 4
```

## Fiji / ImageJ plugin

Fiji’s built-in Jython cannot load NumPy. The plugin saves the current image, runs **CPython** [`segment_lut.py`](4_toImageJ/py_lut/segment_lut.py), and opens the result.

1. Install Fiji and a CPython environment with `numpy` and `Pillow`.
2. Copy into `Fiji.app/plugins/Seg-LUT/`:
   - [`4_toImageJ/fiji/Seg_LUT.py`](4_toImageJ/fiji/Seg_LUT.py) → `Seg_LUT.py`
   - [`4_toImageJ/fiji/config.example.txt`](4_toImageJ/fiji/config.example.txt) → `config.txt` (edit paths)
   - Folder [`4_toImageJ/py_lut/`](4_toImageJ/py_lut/) → `py_lut/`
   - Folder [`4_toImageJ/lut/`](4_toImageJ/lut/) → `lut/`
3. In Fiji: open a grayscale image, run **Seg_LUT** (under Plugins or Scripts, depending on install).
4. Configure `PYTHON_EXE`, `SEG_LUT_SCRIPT`, and `SEG_LUT_NPY` in `config.txt`, or set environment variables with the same names.

Default LUT path in the plugin: `plugins/Seg-LUT/lut/Seg2_LUT_HV_2_4.npy`.

## Paper figures

Scripts and plots for runtime vs. Dice comparisons live in [`docs/figures/`](docs/figures/).

## License

MIT — see [LICENSE](LICENSE).
