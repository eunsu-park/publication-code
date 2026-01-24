# Solar Magnetogram Denoising

Deep learning-based noise reduction for SDO/HMI solar magnetograms using DCGAN.

## Publication

**Title:** De-noising SDO/HMI Solar Magnetograms by Image Translation Method Based on Deep Learning

**Authors:** Eunsu Park, Yong-Jae Moon, Daye Lim, Harim Lee

**Journal:** The Astrophysical Journal Letters, 891:L4 (9pp), 2020 March 1

**DOI:** [10.3847/2041-8213/ab74d2](https://doi.org/10.3847/2041-8213/ab74d2)

## Overview

This study applies DCGAN (Deep Convolutional Generative Adversarial Network) to de-noise solar magnetograms. The model performs image-to-image translation from a single noisy magnetogram to a stacked (low-noise) magnetogram equivalent.

### Input/Output

| Type | Description |
|------|-------------|
| Input | Single SDO/HMI LOS magnetogram (256 × 256, 1 channel) |
| Target | 21-frame stacked magnetogram (256 × 256, 1 channel) |
| Output | De-noised magnetogram |

### Key Innovation

- De-noises magnetograms without requiring long exposure times
- Reduces noise level from ~8.66 G to ~3.21 G (equivalent to 21-frame stacking)
- Applicable to full-disk magnetograms from solar center to limb

## Network Architecture

### DCGAN (Pix2Pix-based)

The architecture follows Pix2Pix with U-Net Generator and PatchGAN Discriminator.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            GENERATOR (U-Net)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: 256 × 256 × 1 (Noisy magnetogram)                               │
│      │                                                                   │
│      ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    ENCODER (Downsampling)                       │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │                                                                 │     │
│  │  e1: Conv 4×4, s2 → 64   (128×128×64)     ──────────────┐      │     │
│  │  e2: Conv-BN-LReLU → 128 (64×64×128)      ─────────────┐│      │     │
│  │  e3: Conv-BN-LReLU → 256 (32×32×256)      ────────────┐││      │     │
│  │  e4: Conv-BN-LReLU → 512 (16×16×512)      ───────────┐│││      │     │
│  │  e5: Conv-BN-LReLU → 512 (8×8×512)        ──────────┐││││      │     │
│  │  e6: Conv-BN-LReLU → 512 (4×4×512)        ─────────┐│││││      │     │
│  │  e7: Conv-BN-LReLU → 512 (2×2×512)        ────────┐││││││      │     │
│  │  e8: Conv-ReLU → 512     (1×1×512)        ───────┐│││││││      │     │
│  │                                                   ││││││││      │     │
│  └───────────────────────────────────────────────────┼┼┼┼┼┼┼┼──────┘     │
│                                                      ││││││││            │
│  ┌───────────────────────────────────────────────────┼┼┼┼┼┼┼┼──────┐     │
│  │                    DECODER (Upsampling)           ││││││││      │     │
│  ├───────────────────────────────────────────────────┼┼┼┼┼┼┼┼──────┤     │
│  │                                                   ││││││││      │     │
│  │  d1: ConvT-BN-Drop-ReLU → 512 + e7 ◄──────────────┘│││││││      │     │
│  │  d2: ConvT-BN-Drop-ReLU → 512 + e6 ◄───────────────┘││││││      │     │
│  │  d3: ConvT-BN-Drop-ReLU → 512 + e5 ◄────────────────┘│││││      │     │
│  │  d4: ConvT-BN-ReLU → 512 + e4      ◄─────────────────┘││││      │     │
│  │  d5: ConvT-BN-ReLU → 256 + e3      ◄──────────────────┘│││      │     │
│  │  d6: ConvT-BN-ReLU → 128 + e2      ◄───────────────────┘││      │     │
│  │  d7: ConvT-BN-ReLU → 64 + e1       ◄────────────────────┘│      │     │
│  │  d8: ConvT-Tanh → 1                ◄─────────────────────┘      │     │
│  │                                                                 │     │
│  └─────────────────────────────────────────────────────────────────┘     │
│      │                                                                   │
│      ▼                                                                   │
│  Output: 256 × 256 × 1 (De-noised magnetogram)                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       DISCRIMINATOR (PatchGAN)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Concatenated [Input, Output] (256 × 256 × 2)                    │
│      │                                                                   │
│      ▼                                                                   │
│  Layer 1: Conv 4×4, s2 → 64, LReLU          (128×128×64)                │
│  Layer 2: Conv 4×4, s2 → 128, BN, LReLU     (64×64×128)                 │
│  Layer 3: Conv 4×4, s2 → 256, BN, LReLU     (32×32×256)                 │
│  Layer 4: Conv 4×4, s1 → 512, BN, LReLU     (31×31×512)                 │
│  Layer 5: Conv 4×4, s1 → 1, Sigmoid         (30×30×1)                   │
│      │                                                                   │
│      ▼                                                                   │
│  Output: Patch probability map (real/fake)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Loss Functions

**L1 Loss (Mean Absolute Error):**
```
L1(G) = (1/N) * Σ|M_target - M_denoised|
```

**cGAN Loss:**
```
L_cGAN(G, D) = log(D(M_input, M_target)) + log(1 - D(M_input, M_denoised))
```

**Total Loss:**
```
G* = argmin_G max_D L_cGAN(G, D) + λ * L1(G)
```

### Key Components

| Component | Description |
|-----------|-------------|
| Generator | U-Net with 8 encoder + 8 decoder layers |
| Discriminator | PatchGAN (70×70 receptive field) |
| Skip Connections | Encoder features concatenated to decoder |
| Dropout | 50% on first 3 decoder layers |
| Normalization | Batch Normalization |
| Activation | LeakyReLU (encoder), ReLU (decoder) |

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | L1 + cGAN (BCE) |
| Lambda (L1 weight) | 100 |
| Optimizer | Adam |
| Iterations | 500,000 |
| Input Size | 256 × 256 |
| Data Range | -100 G to +100 G |

## Data

### Data Preparation

| Item | Description |
|------|-------------|
| Source | SDO/HMI LOS 45s magnetograms |
| Period | 2013 January - 2013 December |
| Region | Solar disk center (256 × 256 patch) |
| Cadence | 1 hour |

### Target Generation

- 21-frame stacked magnetograms (10 before + center + 10 after)
- Solar rotation compensation applied
- Equivalent to ~15 minute effective exposure

### Data Split (Chronological)

| Set | Period | Pairs |
|-----|--------|-------|
| Training | 2013 Jan - Oct | 7,004 |
| Validation | 2013 Nov | 707 |
| Test | 2013 Dec | 736 |
| **Total** | | **8,447** |

## Results

### Noise Level Reduction

| Magnetogram | Noise Level (G) |
|-------------|-----------------|
| Input (single frame) | 8.66 |
| Target (21-frame stacked) | 3.21 |
| De-noised (model output) | 3.21 |
| SDO/HMI 720s (reference) | 6.3 |

### Comparison with Smoothing Methods (Test Set)

| Metric | Input | De-noised (Ours) | Median | Gaussian | Bilateral |
|--------|-------|------------------|--------|----------|-----------|
| Noise Level (G) | 8.66 | **3.21** | 4.57 | 4.27 | 4.36 |
| Pixel CC | 0.88 | **0.94** | 0.93 | 0.95 | 0.94 |
| RE (TUMF) | 0.529 | **0.001** | 0.012 | 0.043 | 0.053 |
| NMSE | 0.31 | **0.12** | 0.13 | 0.09 | 0.12 |
| Peak S/N (dB) | 28.53 | 32.62 | 32.17 | **33.53** | 32.72 |

### Metrics Definition

| Metric | Description | Best |
|--------|-------------|------|
| Pixel CC | Pixel-to-pixel correlation coefficient | 1 |
| RE | Relative error of total unsigned magnetic flux | 0 |
| NMSE | Normalized mean squared error | 0 |
| Peak S/N | Peak signal-to-noise ratio | Higher |

## Requirements

- Python 3.6+
- PyTorch 1.0+
- NumPy
- SunPy

## Usage

```python
import torch
from networks import Generator, Pix2Pix

# Initialize generator for inference
generator = Generator(in_channels=1, out_channels=1)

# Load trained weights
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()

# Input: (batch, 1, 256, 256) - noisy magnetogram
noisy_mag = torch.randn(1, 1, 256, 256)

# Output: (batch, 1, 256, 256) - de-noised magnetogram
with torch.no_grad():
    denoised_mag = generator(noisy_mag)
```

## Reference Code

Original implementation: [https://github.com/eunsu-park/solar_magnetogram_denoising](https://github.com/eunsu-park/solar_magnetogram_denoising)

## Citation

```bibtex
@article{Park_2020,
    title={De-noising SDO/HMI Solar Magnetograms by Image Translation Method Based on Deep Learning},
    author={Park, Eunsu and Moon, Yong-Jae and Lim, Daye and Lee, Harim},
    journal={The Astrophysical Journal Letters},
    volume={891},
    number={1},
    pages={L4},
    year={2020},
    month={mar},
    publisher={The American Astronomical Society},
    doi={10.3847/2041-8213/ab74d2}
}
```

## License

MIT License
