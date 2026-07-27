# Mamba-SSM 安装指南：精准匹配预编译 Wheel

> English version: [Mamba_turtle_en.md](Mamba_turtle_en.md)
>
> 本项目验证环境：Python 3.10、PyTorch 2.5.1+cu121、
> `mamba-ssm==2.3.0`、`causal-conv1d==1.6.0`。

PA-DCNet 使用 `mamba-ssm` 提供的 `Mamba2`。`mamba-ssm` 与
`causal-conv1d` 包含 CUDA 扩展，Wheel 必须与本地的 PyTorch、PyTorch CUDA
runtime、Python 版本和 CXX11 ABI 同时匹配。直接执行通用的
`pip install mamba-ssm` 可能触发不兼容的源码编译，因此优先使用匹配的预编译
Wheel。

## 1. 推荐安装顺序

1. 安装目标 PyTorch CUDA 版本。
2. 查询本地 Python、PyTorch、CUDA runtime 和 ABI 信息。
3. 安装匹配的 `causal-conv1d` Wheel。
4. 安装匹配的 `mamba-ssm` Wheel。
5. 执行 `pip install -r requirements.txt` 安装其余项目依赖。

`requirements.txt` 记录了本项目的验证版本，但无法为所有 CUDA、PyTorch、
Python 和 ABI 组合自动选择唯一正确的 Mamba Wheel。

## 2. 查询环境信息

选择 Wheel 前执行：

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

应以 `torch.version.cuda` 匹配 Wheel 名中的 CUDA 标识。`nvidia-smi` 显示的是
GPU 驱动支持的 CUDA 上限，并不一定等于当前 PyTorch 使用的 CUDA runtime；
`nvcc -V` 主要用于源码编译场景。

## 3. 匹配 Wheel 标识

例如，下面的 Wheel：

```text
mamba_ssm-2.2.0+cu118torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
```

需要逐项核对：

| Wheel 标识 | 应匹配的本地环境 |
| --- | --- |
| `cu118` | `torch.version.cuda` 为 CUDA 11.8 |
| `torch2.2` | PyTorch 主/次版本为 2.2 |
| `cxx11abiFALSE` | `torch._C._GLIBCXX_USE_CXX11_ABI` |
| `cp310` | Python 3.10 |
| `linux_x86_64` | Linux x86_64 系统架构 |

应选择与 `mamba-ssm` 发布版本兼容的 `causal-conv1d`。PA-DCNet 已验证的组合为
`causal-conv1d==1.6.0` 与 `mamba-ssm==2.3.0`。

官方项目页：

- [state-spaces/mamba releases](https://github.com/state-spaces/mamba/releases)
- [Dao-AILab/causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)

## 4. 安装与验证

下载匹配 Wheel 后，按以下顺序安装：

```bash
pip install /path/to/causal_conv1d-<matched>.whl
pip install /path/to/mamba_ssm-<matched>.whl
pip install -r requirements.txt
```

安装完成后验证导入：

```bash
python -c "import causal_conv1d; import mamba_ssm; from mamba_ssm import Mamba2; print('Mamba-SSM 安装成功')"
```

## 5. 常见问题

- CXX11 ABI 不匹配通常会在导入时触发 `ImportError`。
- Python 小版本标签不匹配，例如 `cp310` 与 `cp311`，不能共用 Wheel。
- Windows 通常没有兼容的官方 Wheel，建议使用 Linux 或 WSL2。
- 若不存在匹配 Wheel，仅应在本地 CUDA toolkit 与 PyTorch 构建版本兼容、且已安装
  `requirements.txt` 中依赖时再尝试源码编译。
