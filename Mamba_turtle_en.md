# Mamba-SSM Installation Guide: Matching Prebuilt Wheels

> Chinese version: [Mamba_turtle.md](Mamba_turtle.md)
>
> Tested PA-DCNet environment: Python 3.10, PyTorch 2.5.1+cu121,
> `mamba-ssm==2.3.0`, and `causal-conv1d==1.6.0`.

PA-DCNet imports `Mamba2` from `mamba-ssm`. Its CUDA extensions are compiled
against a specific combination of PyTorch, CUDA runtime, Python, and CXX11 ABI.
A generic `pip install mamba-ssm` can therefore trigger an incompatible source
build. Prefer matched prebuilt wheels whenever they are available.

## 1. Install Order

1. Install the required PyTorch CUDA build.
2. Inspect the local Python, PyTorch, CUDA runtime, and ABI values.
3. Install the matching `causal-conv1d` wheel first.
4. Install the matching `mamba-ssm` wheel.
5. Install the remaining project dependencies with `pip install -r requirements.txt`.

The pins in `requirements.txt` record the tested package versions. They cannot
select a single portable Mamba wheel for every CUDA/PyTorch/Python combination.

## 2. Inspect the Environment

Run the following before selecting a wheel:

```bash
python - <<'PY'
import platform
import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CXX11 ABI: {torch._C._GLIBCXX_USE_CXX11_ABI}")
print(f"Platform: {platform.system()} {platform.machine()}")
PY
```

Use `torch.version.cuda` to match the CUDA tag in a wheel name. `nvidia-smi`
reports the CUDA capability of the GPU driver, not necessarily the CUDA runtime
used by the installed PyTorch wheel. `nvcc -V` is primarily relevant when
building from source.

## 3. Match Wheel Tags

For a wheel named:

```text
mamba_ssm-2.2.0+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

verify every tag:

| Wheel tag | Must match |
| --- | --- |
| `cu118` | `torch.version.cuda` (CUDA 11.8) |
| `torch2.2` | Installed PyTorch major/minor version |
| `cxx11abiFALSE` | `torch._C._GLIBCXX_USE_CXX11_ABI` |
| `cp310` | Python 3.10 |
| `linux_x86_64` | Operating system and CPU architecture |

Choose a `causal-conv1d` release compatible with the selected `mamba-ssm`
release. The verified PA-DCNet pair is `causal-conv1d==1.6.0` and
`mamba-ssm==2.3.0`.

Official project pages:

- [state-spaces/mamba releases](https://github.com/state-spaces/mamba/releases)
- [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)

## 4. Install and Verify

After downloading compatible wheels, install them in this order:

```bash
pip install /path/to/causal_conv1d-<matched>.whl
pip install /path/to/mamba_ssm-<matched>.whl
pip install -r requirements.txt
```

Verify that the CUDA extensions import successfully:

```bash
python -c "import causal_conv1d; import mamba_ssm; from mamba_ssm import Mamba2; print('Mamba-SSM installation succeeded.')"
```

## 5. Troubleshooting

- A CXX11 ABI mismatch commonly causes an `ImportError` at import time.
- A Python minor-version mismatch, such as `cp310` versus `cp311`, cannot use
  the same wheel.
- Windows does not generally have compatible official wheels; use Linux or WSL2.
- If no compatible wheel exists, build from source only after installing a CUDA
  toolkit compatible with the PyTorch build and the packages listed in
  `requirements.txt`.
