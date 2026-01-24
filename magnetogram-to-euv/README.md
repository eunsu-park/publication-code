# Magnetogram to UV/EUV Image Generation

Deep learning-based image-to-image translation from SDO/HMI magnetograms to SDO/AIA UV and EUV images.

## Publication

**Title:** Generation of Solar UV and EUV Images from SDO/HMI Magnetograms by Deep Learning

**Authors:** Eunsu Park, Yong-Jae Moon, Jin-Yi Lee, Rok-Soon Kim, Harim Lee, Daye Lim, Gyungin Shin, Taeyoung Kim

**Journal:** The Astrophysical Journal Letters, 884:L23 (8pp), 2019 October 10

**DOI:** [10.3847/2041-8213/ab46bb](https://doi.org/10.3847/2041-8213/ab46bb)

**Code:** [GitHub Repository](https://github.com/eunsu-park/solar_euv_generation)

## Overview

This method applies deep learning to translate solar magnetograms into UV and EUV images. Two CNN models are compared: one using only L1 loss (Model A), and another using L1 + cGAN loss (Model B, Pix2Pix style).

### Input/Output

| Type | Description |
|------|-------------|
| Input | SDO/HMI LOS magnetogram (1 channel) |
| Output | SDO/AIA UV/EUV image (1 channel per passband) |

### Target Passbands

| Passband (Å) | Temperature (K) | Region |
|--------------|-----------------|--------|
| 1700 | 4,500 | Temperature minimum, photosphere |
| 1600 | 10,000 | Transition region, upper photosphere |
| 304 | 50,000 | Chromosphere, transition region |
| 171 | 600,000 | Quiet corona, upper transition region |
| 193 | 1,000,000 | Corona, hot flare plasma |
| 211 | 2,000,000 | Active region corona |
| 335 | 2,500,000 | Active region corona |
| 94 | 6,000,000 | Flaring corona |
| 131 | 10,000,000 | Transition region, flaring corona |

## Network Architecture

The model is based on Pix2Pix (Isola et al., 2016) architecture with a U-Net Generator and PatchGAN Discriminator.

### Generator (U-Net)

```
Input: 1024 × 1024 × 1 (magnetogram)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                  │
├─────────────────────────────────────────────────────────────────┤
│  Conv(1→64, k4s2) ─────────────────────────────────────┐        │
│      │                                                  │ skip   │
│      ▼                                                  │        │
│  Conv(64→128, k4s2) → BN → LeakyReLU ──────────┐       │        │
│      │                                          │ skip  │        │
│      ▼                                          │       │        │
│  Conv(128→256, k4s2) → BN → LeakyReLU ─────┐   │       │        │
│      │                                      │   │       │        │
│      ▼                                      │   │       │        │
│  Conv(256→512, k4s2) → BN → LeakyReLU ──┐  │   │       │        │
│      │                                   │  │   │       │        │
│      ▼                                   │  │   │       │        │
│  Conv(512→512, k4s2) → BN → LeakyReLU ┐ │  │   │       │        │
│      │                                 │ │  │   │       │        │
│      ▼                                 │ │  │   │       │        │
│  Conv(512→512, k4s2) → BN → LeakyReLU │ │  │   │       │        │
│      │                                 │ │  │   │       │        │
│      ▼                                 │ │  │   │       │        │
│  Conv(512→512, k4s2) → BN → LeakyReLU │ │  │   │       │        │
│      │                                 │ │  │   │       │        │
│      ▼                                 │ │  │   │       │        │
│  Conv(512→512, k4s2) → BN → ReLU      │ │  │   │       │        │
│      │ (Bottleneck)                   │ │  │   │       │        │
└──────┼────────────────────────────────┼─┼──┼───┼───────┼────────┘
       │                                │ │  │   │       │
       ▼                                ▼ ▼  ▼   ▼       ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER                                  │
├─────────────────────────────────────────────────────────────────┤
│  ConvT(512→512, k4s2) → BN → Dropout → ReLU + Skip              │
│      │                                                           │
│      ▼                                                           │
│  ConvT(1024→512, k4s2) → BN → Dropout → ReLU + Skip             │
│      │                                                           │
│      ▼                                                           │
│  ConvT(1024→512, k4s2) → BN → Dropout → ReLU + Skip             │
│      │                                                           │
│      ▼                                                           │
│  ConvT(1024→512, k4s2) → BN → ReLU + Skip                       │
│      │                                                           │
│      ▼                                                           │
│  ConvT(1024→256, k4s2) → BN → ReLU + Skip                       │
│      │                                                           │
│      ▼                                                           │
│  ConvT(512→128, k4s2) → BN → ReLU + Skip                        │
│      │                                                           │
│      ▼                                                           │
│  ConvT(256→64, k4s2) → BN → ReLU + Skip                         │
│      │                                                           │
│      ▼                                                           │
│  ConvT(128→1, k4s2) → Tanh                                       │
│      │                                                           │
└──────┼───────────────────────────────────────────────────────────┘
       │
       ▼
Output: 1024 × 1024 × 1 (UV/EUV image)
```

### Discriminator (PatchGAN)

```
Input: 1024 × 1024 × 2 (magnetogram + image, concatenated)
    │
    ▼
  Conv(2→64, k4s2) → LeakyReLU
    │
    ▼
  Conv(64→128, k4s2) → BN → LeakyReLU
    │
    ▼
  Conv(128→256, k4s2) → BN → LeakyReLU
    │
    ▼
  Conv(256→512, k4s1) → BN → LeakyReLU
    │
    ▼
  Conv(512→1, k4s1) → Sigmoid
    │
    ▼
Output: 62 × 62 × 1 (patch probabilities)
```

### Two Models

| Model | Loss Function | Characteristics |
|-------|---------------|-----------------|
| Model A | L1 only | Better metrics (CC, RE, PPE10), more blurred |
| Model B | L1 + cGAN | More realistic/sharper images, slightly lower metrics |

### Loss Functions

**L1 Loss:**
```
L1 = E[||y - G(x)||₁]
```

**cGAN Loss:**
```
L_cGAN = E[log D(x, y)] + E[log(1 - D(x, G(x)))]
```

**Combined Loss (Model B):**
```
G* = arg min_G max_D L_cGAN(G, D) + λ·L1(G)
```
where λ = 100

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 2 × 10⁻⁴ |
| β₁ | 0.5 |
| β₂ | 0.999 |
| Iterations | 500,000 |
| Image Size | 1024 × 1024 |
| Conv Weight Init | Normal(0.0, 0.02) |
| BatchNorm Weight Init | Normal(1.0, 0.02) |

## Data

- **Source:** SDO/HMI LOS magnetograms and SDO/AIA 9-passband images
- **Period:** 2011-2017
- **Cadence:** 6 hours (4 pairs per day)
- **Training:** 8,544 pairs (2011-2016)
- **Validation:** 714 pairs (2017 Jan-Jun)
- **Test:** 727 pairs (2017 Jul-Dec)
- **Preprocessing:** Level 1.5 calibration, exposure time normalization, degradation correction

### Data Normalization

- AIA images: DN/s → log scale (0-14) → rescale to (-1, 1)
- HMI magnetograms: Rescale to (-1, 1)

## Results

| Passband | CC (Model A) | CC (Model B) | RE (Model A) | RE (Model B) |
|----------|--------------|--------------|--------------|--------------|
| 1700 Å | 0.97 | 0.95 | 0.01 | 0.02 |
| 1600 Å | 0.94 | 0.92 | 0.11 | 0.11 |
| 304 Å | 0.84 | 0.83 | -0.18 | -0.17 |
| 171 Å | 0.69 | 0.66 | -0.05 | -0.04 |
| 193 Å | 0.78 | 0.74 | 0.07 | 0.07 |
| 211 Å | 0.86 | 0.78 | 0.08 | 0.08 |
| 335 Å | 0.86 | 0.85 | -0.03 | 0.04 |
| 94 Å | 0.79 | 0.75 | -0.03 | -0.02 |
| 131 Å | 0.81 | 0.78 | -0.04 | -0.03 |
| **Average** | **0.84** | **0.83** | **0.07** | **0.06** |

## Requirements

- Python 3.6+
- PyTorch 1.0+ (or TensorFlow/Keras as in original)
- NumPy
- SunPy
- SolarSoft (for data preprocessing)

## Usage

```python
import torch
from networks import Generator, Discriminator

# Initialize models
generator = Generator(in_channels=1, out_channels=1)
discriminator = Discriminator(in_channels=2)  # magnetogram + image

# Input: (batch, 1, 1024, 1024) - HMI magnetogram normalized to [-1, 1]
magnetogram = torch.randn(1, 1, 1024, 1024)

# Output: (batch, 1, 1024, 1024) - Generated AIA image in [-1, 1]
generated_euv = generator(magnetogram)
```

## Citation

```bibtex
@article{Park_2019,
    title={Generation of Solar UV and EUV Images from SDO/HMI Magnetograms by Deep Learning},
    author={Park, Eunsu and Moon, Yong-Jae and Lee, Jin-Yi and Kim, Rok-Soon and Lee, Harim and Lim, Daye and Shin, Gyungin and Kim, Taeyoung},
    journal={The Astrophysical Journal Letters},
    volume={884},
    number={2},
    pages={L23},
    year={2019},
    month={oct},
    publisher={The American Astronomical Society},
    doi={10.3847/2041-8213/ab46bb}
}
```

## License

MIT License
