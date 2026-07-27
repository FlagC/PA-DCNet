# PA-DCNet

Official implementation of **PA-DCNet**, a propagation-aware dual-coordinate network for radio-map reconstruction.

PA-DCNet models transmitter-centered global propagation in a polar coordinate branch and environmental obstruction correction in a Cartesian branch. Polar features spatially modulate Cartesian encoder features, and a gated dual-coordinate fusion produces the bottleneck representation for decoding.

## Environment

The released configuration was developed with Python 3.10, PyTorch 2.5.1 and CUDA 12.1.

```bash
python -m venv .venv
source .venv/bin/activate
# Install matching causal-conv1d and mamba-ssm wheels first; see below.
pip install -r requirements.txt
```

### Mamba-SSM CUDA Extension

PA-DCNet uses `Mamba2` from `mamba-ssm`. `mamba-ssm` and `causal-conv1d` are CUDA extensions, so their prebuilt wheels must match the installed PyTorch, PyTorch CUDA runtime, Python version, and CXX11 ABI. Install a matching `causal-conv1d` wheel first, then a matching `mamba-ssm` wheel, before running `pip install -r requirements.txt`.

Detailed installation instructions are available in [Chinese](Mamba_turtle.md) and [English](Mamba_turtle_en.md). The released setup was tested with Python 3.10, PyTorch 2.5.1+cu121, `causal-conv1d==1.6.0`, and `mamba-ssm==2.3.0`.

## Dataset Layout

Change `data.dataset_root_dir` in the selected YAML file.

```text
RadioMapSeer/
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
python src/train.py --config configs/PADCNet_srm.yaml
python src/train.py --config configs/PADCNet_drm.yaml
```

The SRM configuration currently retains the 512-bin setting used for polar-resolution ablation. Set `polar_radial_bins` and `polar_theta_bins` to the desired final setting before a reproduction run. Both outputs and newly generated TensorBoard logs are written locally under `results/` and `tb_logs/`, which are ignored by Git.

Generate a custom configuration when needed:

```bash
python src/make_config.py --task srm --version custom --polar-radial-bins 256 --polar-theta-bins 256
```

## Evaluation

```bash
python src/eval.py --config configs/PADCNet_srm.yaml --checkpoint /path/to/checkpoint.ckpt --device cuda:0
python meterics/evaluate_nocars.py --pred-dir results/PADCNet_v1/test_predictions
python meterics/evaluate_withcars.py --pred-dir results/PADCNet_drm/test_predictions
```

The metric scripts default to the RadioMapSeer dataset path above. Use `--dataset-root`, `--gt-dir`, `--limit`, and `--output` to override paths or save a metric summary.


## License

Copyright (c) 2026 <Your Name or Institution>.

This project is licensed under the Apache License, Version 2.0.
See [LICENSE](LICENSE) for details.
