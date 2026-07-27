# PA-DCNet

Official implementation of **PA-DCNet**, a propagation-aware dual-coordinate network for radio-map reconstruction.

PA-DCNet models transmitter-centered global propagation in a polar coordinate branch and environmental obstruction correction in a Cartesian branch. Polar features spatially modulate Cartesian encoder features, and a gated dual-coordinate fusion produces the bottleneck representation for decoding.

## Repository Contents

- `models/padcnet.py`: PA-DCNet model implementation.
- `datasetloader/DataLoader.py`: RadioMapSeer data loader.
- `configs/srm.yaml`: static radio-map reconstruction configuration.
- `configs/drm.yaml`: dynamic radio-map reconstruction configuration with vehicle input.
- `src/train.py`, `src/eval.py`: training and prediction entry points.
- `meterics/`: evaluation scripts for SRM, DRM, and thresholded outputs.

Training logs, checkpoints, prediction images, datasets, and historical experimental models are intentionally excluded from this repository.

## Environment

The released configuration was developed with Python 3.10, PyTorch 2.5.1 and CUDA 12.1.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`mamba-ssm` requires a CUDA-enabled PyTorch installation compatible with the local CUDA toolchain. Install the matching PyTorch wheel first if your platform differs from CUDA 12.1.

## Dataset Layout

Place RadioMapSeer under `data/RadioMapSeer`, or change `data.dataset_root_dir` in the selected YAML file.

```text
data/RadioMapSeer/
├── gain/
│   ├── DPM/
│   └── carsDPM/
└── png/
    ├── antennas/
    ├── buildings_complete/
    └── cars/
```

The static task (SRM) consumes building map and Tx one-hot inputs. The loader still emits three channels for compatibility, while the model ignores the duplicated third building channel. The dynamic task (DRM) uses building map, Tx one-hot map, and vehicle map; its target is `gain/carsDPM`.

## Training

Run commands from the repository root.

```bash
python src/train.py --config configs/srm.yaml
python src/train.py --config configs/drm.yaml
```

The SRM configuration currently retains the 512-bin setting used for polar-resolution ablation. Set `polar_radial_bins` and `polar_theta_bins` to the desired final setting before a reproduction run. Both outputs and newly generated TensorBoard logs are written locally under `results/` and `tb_logs/`, which are ignored by Git.

Generate a custom configuration when needed:

```bash
python src/make_config.py --task srm --version custom --polar-radial-bins 256 --polar-theta-bins 256
```

## Evaluation

```bash
python src/eval.py --config configs/srm.yaml --checkpoint /path/to/checkpoint.ckpt --device cuda:0
python meterics/evaluate_nocars.py --pred-dir results/PADCNet_v1_512/test_predictions
python meterics/evaluate_withcars.py --pred-dir results/PADCNet_drm/test_predictions
python meterics/evaluate_threshold.py --pred-dir results/PADCNet_v1_512/test_predictions --threshold 0.2
```

The metric scripts default to the RadioMapSeer dataset path above. Use `--dataset-root`, `--gt-dir`, `--limit`, and `--output` to override paths or save a metric summary.

## Checkpoint Compatibility

The public Python class is named `PADCNet`; the former internal class name was `PGCNet`. The rename does not alter module attribute names or tensor shapes, so existing `PGCNet` model weights can be loaded into `PADCNet` without retraining when the model configuration is unchanged.

## License

Copyright (c) 2026 <Your Name or Institution>.

This project is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details.
