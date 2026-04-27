# Hands-on-EEG

## Graduation project

EEG signal classification using Artificial Neural Networks (ANN).  
（这是本项目的公开分支，在使用 CUDA 上的一些实践可切换至其他分支）

---

## 🚀 Quick start

### Prerequisites

```bash
pip install numpy pandas pyyaml
# Optional – required for the animated visualiser:
pip install vispy
# Optional – required for model training:
pip install torch
```

### Configuration

Copy and edit the example configuration file:

```bash
cp config.yaml my_config.yaml
# Edit my_config.yaml → set data.data_dir to your data folder
```

### Load and visualise EEG data

```python
from src.config import load_config
from src.visualization import EEGVisualizer

cfg = load_config("my_config.yaml")
viz = EEGVisualizer("data/raw/sample.csv", config=cfg)
viz.run()
```

### Load data and preprocess

```python
from src.data import load_eeg_csv, sliding_window, normalize_signal

data = load_eeg_csv("data/raw/sample.csv")          # (32, T)
data = normalize_signal(data, method="zscore")
windows = sliding_window(data, window_size=128)     # (32, 128, N)
```

### Train a CNN model

```python
import torch
from src.config import load_config
from src.models import CNNModel

cfg = load_config("my_config.yaml")
model = CNNModel.from_config(cfg)
# … standard PyTorch training loop …
```

---

## 📂 Project structure

```
Hands-on-EEG/
├── config.yaml               # Example configuration file
├── src/                      # ✨ New modular package
│   ├── config/               #   Configuration management (YAML + env vars)
│   ├── data/                 #   Data loading & preprocessing
│   ├── visualization/        #   EEG waveform visualiser (replaces polt.py)
│   ├── utils/                #   Logging & validation helpers
│   └── models/               #   CNN and Transformer models
├── legacy/                   # Original notebooks (preserved, not maintained)
│   ├── open/                 #   Original app scripts & notebooks
│   ├── new_implement/        #   Early CNN experiments
│   └── normal/               #   Miscellaneous notebooks
└── pic/                      # Figures
```

### `src/` module summary

| Module | Description |
|--------|-------------|
| `src.config` | Load `config.yaml`; override via environment variables (`EEG_DATA_DIR`, etc.) |
| `src.data.loader` | Portable CSV loading with error handling – no hardcoded paths |
| `src.data.preprocessor` | Sliding-window segmentation, z-score / min-max normalisation |
| `src.visualization.eeg_visualizer` | Object-oriented VisPy visualiser (replaces `open/app/polt.py`) |
| `src.utils.logger` | Unified logging: console + optional file output |
| `src.utils.validators` | DataFrame and NumPy array shape/type checks |
| `src.models.cnn_model` | Parameterised CNN model with `from_config` / `load` helpers |
| `src.models.transformer_model` | Transformer encoder model with `from_config` / `load` helpers |

---

## 📋 Background & original pipeline

本项目使用 EMOTIV Plex 采集脑电信号，使用 EMOTIV PRO 软件导出 CSV 格式数据，
设计神经网络模型对脑电信号进行分类。

原始处理流程（见 `legacy/` 目录中的笔记本）：

```
raw CSV → process.ipynb (切片、数据集分割)
        → sliding_window (128 样本窗口)
        → CNN / Transformer 训练
```

六类任务标签：`lefthand` / `read` / `rest` / `walkbase` / `walkl` / `walkfocus`

出于对实际受试人的保护，数据集不公开。可向相关人求取。

