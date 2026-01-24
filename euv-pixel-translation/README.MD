# EUV Pixel-to-Pixel Translation

Pixel-to-pixel translation of solar extreme-ultraviolet (EUV) images using Fully Connected Networks (FCN) for Differential Emission Measure (DEM) analysis.

## Publication

**Title:** Pixel-to-pixel Translation of Solar Extreme-ultraviolet Images for DEMs by Fully Connected Networks

**Authors:** Eunsu Park, Harim Lee, Yong-Jae Moon, Jin-Yi Lee, Il-Hyun Cho, Kyoung-Sun Lee, Daye Lim, Hyun-Jin Jeong, Jae-Ok Lee

**Journal:** The Astrophysical Journal Supplement Series, 264:33 (11pp), 2023 February

**DOI:** [10.3847/1538-4365/aca902](https://doi.org/10.3847/1538-4365/aca902)

## Overview

This method performs pixel-to-pixel image translation among solar EUV images from SDO/AIA. Unlike CNN-based approaches that use neighboring pixel information, this FCN model treats each pixel independently, which is physically motivated by the DEM equation where each pixel's intensity depends only on local temperature and density.

### Input/Output Channels

| Type | Channels | Wavelengths |
|------|----------|-------------|
| Input | 3 | 17.1 nm, 19.3 nm, 21.1 nm |
| Output | 3 | 9.4 nm, 13.1 nm, 33.5 nm |

## Network Architecture

The model uses a U-Net style encoder-decoder architecture with skip connections, but replaces all convolution layers with fully connected layers for pixel-independent processing.

### Architecture Diagram

```
Input (3)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         ENCODER                                  │
├─────────────────────────────────────────────────────────────────┤
│  FC(3→64) → SiLU ─────────────────────────────────────┐        │
│      │                                                 │ skip   │
│      ▼                                                 │        │
│  FC(64→128) → SiLU ──────────────────────────┐        │        │
│      │                                        │ skip   │        │
│      ▼                                        │        │        │
│  FC(128→256) → SiLU ─────────────────┐       │        │        │
│      │                                │ skip  │        │        │
│      ▼                                │       │        │        │
│  FC(256→512) → SiLU ──────────┐      │       │        │        │
│      │                         │ skip │       │        │        │
│      ▼                         │      │       │        │        │
│  FC(512→512) → SiLU ─────┐    │      │       │        │        │
│      │                    │    │      │       │        │        │
│      ▼                    │    │      │       │        │        │
│  FC(512→512) → SiLU ──┐  │    │      │       │        │        │
│      │                 │  │    │      │       │        │        │
│      ▼                 │  │    │      │       │        │        │
│  FC(512→512) → SiLU ┐ │  │    │      │       │        │        │
│      │              │ │  │    │      │       │        │        │
│      ▼              │ │  │    │      │       │        │        │
│  FC(512→512)        │ │  │    │      │       │        │        │
│      │              │ │  │    │      │       │        │        │
└──────┼──────────────┼─┼──┼────┼──────┼───────┼────────┼────────┘
       │              │ │  │    │      │       │        │
       ▼              ▼ ▼  ▼    ▼      ▼       ▼        ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DECODER                                  │
├─────────────────────────────────────────────────────────────────┤
│  Concat(512+512) → FC(1024→512) → SiLU                          │
│      │                                                           │
│      ▼                                                           │
│  Concat(512+512) → FC(1024→512) → SiLU                          │
│      │                                                           │
│      ▼                                                           │
│  Concat(512+512) → FC(1024→512) → SiLU                          │
│      │                                                           │
│      ▼                                                           │
│  Concat(512+512) → FC(1024→256) → SiLU                          │
│      │                                                           │
│      ▼                                                           │
│  Concat(256+256) → FC(512→128) → SiLU                           │
│      │                                                           │
│      ▼                                                           │
│  Concat(128+128) → FC(256→64) → SiLU                            │
│      │                                                           │
│      ▼                                                           │
│  Concat(64+64) → FC(128→64) → SiLU                              │
│      │                                                           │
│      ▼                                                           │
│  FC(64→3)  ← No activation                                       │
│      │                                                           │
└──────┼───────────────────────────────────────────────────────────┘
       │
       ▼
Output (3)
```

### Key Features

- **Encoder:** 8 fully connected layers (3 → 64 → 128 → 256 → 512 → 512 → 512 → 512 → 512)
- **Decoder:** 8 fully connected layers with skip connections via concatenation
- **Activation:** SiLU (Sigmoid Linear Unit / Swish) between all layers except final output
- **Skip Connections:** U-Net style concatenation from encoder to decoder
- **No Regularization:** The FCN model does not include batch normalization or dropout

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | L2 (MSE) |
| Optimizer | Adam |
| Initial Learning Rate | 2 × 10⁻⁴ |
| LR Schedule | Constant for 50 epochs, then linear decay to 0 for 50 epochs |
| Total Epochs | 100 |

## Data

- **Training:** 358 pairs of SDO/AIA EUV data at 00:00 UT in 2011
- **Test:** 80 pairs at 00:00 UT on the first day of every month from 2012 to 2018
- **Preprocessing:** Centering, rotating, normalizing, and correcting degradation using SunPy and AIAPy
- **ROI:** Pixels inside 1.2 solar radii

## Results

| Channel | Correlation Coefficient | RMSE | S/N (AIA) | S/N (FCN) |
|---------|------------------------|------|-----------|-----------|
| 9.4 nm  | 0.78 ± 0.11 | 1.18 ± 1.04 | 1.37 ± 0.26 | 23.20 ± 17.50 |
| 13.1 nm | 0.89 ± 0.08 | 4.14 ± 4.85 | 3.03 ± 0.57 | 16.32 ± 3.37 |
| 33.5 nm | 0.85 ± 0.09 | 3.58 ± 1.24 | 2.35 ± 0.98 | 14.31 ± 5.49 |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- SunPy
- AIAPy
- NumPy
- Astropy

## Usage

```python
import torch
from networks import FCNPixelTranslator

# Initialize model
model = FCNPixelTranslator(in_channels=3, out_channels=3)

# Input: (batch, num_pixels, 3) or (batch, height, width, 3)
# Values are normalized DN/s from AIA 17.1, 19.3, 21.1 nm channels
input_pixels = torch.randn(1, 1000, 3)

# Output: (batch, num_pixels, 3)
# Generated DN/s for AIA 9.4, 13.1, 33.5 nm channels
output_pixels = model(input_pixels)
```

## Citation

```bibtex
@article{Park_2023,
    title={Pixel-to-pixel Translation of Solar Extreme-ultraviolet Images for DEMs by Fully Connected Networks},
    author={Park, Eunsu and Lee, Harim and Moon, Yong-Jae and Lee, Jin-Yi and Cho, Il-Hyun and Lee, Kyoung-Sun and Lim, Daye and Jeong, Hyun-Jin and Lee, Jae-Ok},
    journal={The Astrophysical Journal Supplement Series},
    volume={264},
    number={2},
    pages={33},
    year={2023},
    month={feb},
    publisher={The American Astronomical Society},
    doi={10.3847/1538-4365/aca902}
}
```

## License

MIT License
