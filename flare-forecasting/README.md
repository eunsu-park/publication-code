# Solar Flare Forecasting

Deep Convolutional Neural Network for solar flare occurrence prediction using full-disk solar magnetograms.

## Publication

**Title:** Application of the Deep Convolutional Neural Network to the Forecast of Solar Flare Occurrence Using Full-disk Solar Magnetograms

**Authors:** Eunsu Park, Yong-Jae Moon, Seulki Shin, Kangwoo Yi, Daye Lim, Harim Lee, Gyungin Shin

**Journal:** The Astrophysical Journal, 869:91 (6pp), 2018 December 20

**DOI:** [10.3847/1538-4357/aaed40](https://doi.org/10.3847/1538-4357/aaed40)

## Overview

This study applies CNN to forecast solar flare occurrence using full-disk magnetograms without any preprocessing or feature extraction. Three models are compared: AlexNet, GoogLeNet, and a proposed model combining GoogLeNet and DenseNet architectures.

### Input/Output

| Type | Description |
|------|-------------|
| Input | Full-disk LOS magnetogram (1024 × 1024, 1 channel) |
| Output | Binary classification: Flare (1) or No-flare (0) |

### Flare Definition

- **Flare (1):** C, M, or X class (≥ C1.0)
- **No-flare (0):** Weaker than C1.0

## Network Architecture

### Proposed Model (Model 3)

The proposed model combines GoogLeNet's inception-style architecture with DenseNet's dense connectivity.

```
Input: 1024 × 1024 × 1 (magnetogram)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODULE (×4)                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              CONVOLUTION BLOCK (×6)                      │    │
│  ├─────────────────────────────────────────────────────────┤    │
│  │                                                          │    │
│  │  Input (k)                                               │    │
│  │      │                                                   │    │
│  │      ├──→ Conv 1×1 (k+r) ──┐                            │    │
│  │      ├──→ Conv 1×3 (k+r) ──┤                            │    │
│  │      ├──→ Conv 3×1 (k+r) ──┼──→ Concat (4k+4r)          │    │
│  │      └──→ Conv 3×3 (k+r) ──┘         │                  │    │
│  │                                       │                  │    │
│  │                          ┌────────────┘                  │    │
│  │                          │                               │    │
│  │                          ├──→ Conv 1×1 (4k+4r) ──┐      │    │
│  │                          └──→ Conv 3×3 (k+r) ────┼──→   │    │
│  │                                                  │       │    │
│  │                                        Concat (2k+r)     │    │
│  │                                             │            │    │
│  │  + Dense connections to all previous blocks │            │    │
│  │                                             ▼            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                              │                                   │
│                    Transition Layer                              │
│              (Conv 1×1 + AvgPool 2×2)                           │
│                              │                                   │
└──────────────────────────────┼───────────────────────────────────┘
                               │
                               ▼
                      Global Average Pool
                               │
                               ▼
                      Fully Connected
                               │
                               ▼
                         Softmax (2)
                               │
                               ▼
                    Output: [P(No-flare), P(Flare)]
```

### Key Components

| Component | Description |
|-----------|-------------|
| Modules | 4 modules, each with 6 convolution blocks |
| Growth Rate (r) | 16 (feature maps increase by r per block) |
| Initial Features | 16 |
| Conv Filters | 1×1, 1×3, 3×1, 3×3 (inception-style) |
| Dense Connections | Each block connects to all previous blocks |
| Normalization | Batch Normalization after all conv layers |
| Activation | ReLU |
| Classifier | Softmax |

### Three Models Comparison

| Model | Architecture | Parameters |
|-------|--------------|------------|
| Model 1 | AlexNet (pretrained) | 5 conv + 3 maxpool + 2 FC |
| Model 2 | GoogLeNet (pretrained) | 9 inception modules, 64 conv, 16 pool |
| Model 3 | Proposed (GoogLeNet + DenseNet) | 4 modules × 6 blocks |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | Cross Entropy |
| Optimizer | Adam |
| Learning Rate | 2 × 10⁻⁴ |
| Input Size | 1024 × 1024 |
| Output Classes | 2 (Flare / No-flare) |

## Data

### Sources
- **SOHO/MDI:** 1996 May - 2010 December
- **SDO/HMI:** 2011 January - 2017 June
- **Time:** 00:00 UT daily
- **Labels:** GOES X-ray flare observations

### Data Split (Chronological)

| Set | Period | Images | No-flare | Flare |
|-----|--------|--------|----------|-------|
| Training | 1996-2008 (Cycle 23) | 4,298 | 1,895 | 2,403 |
| Test | 2009-2017 (Cycle 24) | 3,043 | 1,374 | 1,669 |
| **Total** | | **7,341** | **3,269** | **4,072** |

## Results

### Statistical Scores

| Metric | Model 1 (AlexNet) | Model 2 (GoogLeNet) | Model 3 (Proposed) |
|--------|-------------------|---------------------|-------------------|
| ACC | 0.78 | 0.79 | **0.82** |
| POD | 0.72 | 0.84 | **0.85** |
| CSI | 0.64 | 0.68 | **0.73** |
| FAR | **0.14** | 0.21 | 0.17 |
| HSS | 0.57 | 0.57 | **0.63** |
| TSS | 0.57 | 0.56 | **0.63** |

### Metrics Definition

| Metric | Formula | Best |
|--------|---------|------|
| ACC (Accuracy) | (H+N)/(H+F+M+N) | 1 |
| POD (Probability of Detection) | H/(H+M) | 1 |
| CSI (Critical Success Index) | H/(H+F+M) | 1 |
| FAR (False Alarm Ratio) | F/(H+F) | 0 |
| HSS (Heidke Skill Score) | 2[(H×N)-(M×F)]/[(H+M)(M+N)+(H+F)(F+N)] | 1 |
| TSS (True Skill Statistics) | H/(H+M) - F/(F+N) | 1 |

Where: H=Hit, M=Miss, F=False Alarm, N=Null

### 10-Fold Cross-Validation (Proposed Model)

| Metric | Mean | StdDev |
|--------|------|--------|
| ACC | 0.84 | 0.014 |
| POD | 0.83 | 0.034 |
| CSI | 0.75 | 0.023 |
| FAR | 0.11 | 0.019 |
| HSS | 0.69 | 0.027 |
| TSS | 0.69 | 0.025 |

## Requirements

- Python 3.6+
- PyTorch 1.0+
- NumPy
- SunPy

## Usage

```python
import torch
from networks import FlarePredictor

# Initialize model
model = FlarePredictor(
    in_channels=1,
    num_classes=2,
    growth_rate=16,
    num_modules=4,
    blocks_per_module=6,
)

# Input: (batch, 1, 1024, 1024) - magnetogram
magnetogram = torch.randn(1, 1, 1024, 1024)

# Output: (batch, 2) - [P(No-flare), P(Flare)]
output = model(magnetogram)
prediction = torch.argmax(output, dim=1)  # 0: No-flare, 1: Flare
```

## Citation

```bibtex
@article{Park_2018,
    title={Application of the Deep Convolutional Neural Network to the Forecast of Solar Flare Occurrence Using Full-disk Solar Magnetograms},
    author={Park, Eunsu and Moon, Yong-Jae and Shin, Seulki and Yi, Kangwoo and Lim, Daye and Lee, Harim and Shin, Gyungin},
    journal={The Astrophysical Journal},
    volume={869},
    number={2},
    pages={91},
    year={2018},
    month={dec},
    publisher={The American Astronomical Society},
    doi={10.3847/1538-4357/aaed40}
}
```

## License

MIT License
