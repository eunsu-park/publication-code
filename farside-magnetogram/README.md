# Solar Far-side Magnetogram Generation

Deep learning-based generation of solar far-side magnetograms from STEREO/EUVI data using conditional GAN.

## Publication

**Title:** Solar farside magnetograms from deep learning analysis of STEREO/EUVI data

**Authors:** Taeyoung Kim, Eunsu Park, Harim Lee, Yong-Jae Moon, Sung-Ho Bae, Daye Lim, Soojeong Jang, Lokwon Kim, Il-Hyun Cho, Myungjin Choi, Kyung-Suk Cho

**Journal:** Nature Astronomy, 3, 397-400, 2019

**DOI:** [10.1038/s41550-019-0711-5](https://doi.org/10.1038/s41550-019-0711-5)

## Overview

This study applies conditional GAN (cGAN) to generate solar magnetograms from EUV 304 nm images. The model is trained using SDO/AIA 304 nm and SDO/HMI magnetogram pairs from the near-side, then applied to STEREO/EUVI 304 nm images to generate far-side magnetograms.

### Input/Output

| Type | Description |
|------|-------------|
| Training Input | SDO/AIA 304 nm EUV image (1024 × 1024, 1 channel) |
| Training Output | SDO/HMI LOS magnetogram (1024 × 1024, 1 channel) |
| Inference Input | STEREO/EUVI 304 nm EUV image |
| Inference Output | Far-side magnetogram |

### Key Innovation

- First demonstration of generating far-side magnetograms from single-wavelength EUV images
- Enables continuous monitoring of solar magnetic field on the far-side
- Important for space weather forecasting

## Network Architecture

### Pix2Pix (cGAN)

The architecture follows Pix2Pix (Isola et al., 2016) with U-Net Generator and PatchGAN Discriminator.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            GENERATOR (U-Net)                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: 1024 × 1024 × 1 (EUV 304 nm image)                             │
│      │                                                                   │
│      ▼                                                                   │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    ENCODER (Downsampling)                       │     │
│  ├────────────────────────────────────────────────────────────────┤     │
│  │                                                                 │     │
│  │  e1: Conv 4×4, s2 → 64   (512×512×64)     ──────────────┐      │     │
│  │  e2: Conv-BN-LReLU → 128 (256×256×128)    ─────────────┐│      │     │
│  │  e3: Conv-BN-LReLU → 256 (128×128×256)    ────────────┐││      │     │
│  │  e4: Conv-BN-LReLU → 512 (64×64×512)      ───────────┐│││      │     │
│  │  e5: Conv-BN-LReLU → 512 (32×32×512)      ──────────┐││││      │     │
│  │  e6: Conv-BN-LReLU → 512 (16×16×512)      ─────────┐│││││      │     │
│  │  e7: Conv-BN-LReLU → 512 (8×8×512)        ────────┐││││││      │     │
│  │  e8: Conv-ReLU → 512     (4×4×512)        ───────┐│││││││      │     │
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
│  Output: 1024 × 1024 × 1 (Generated Magnetogram)                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│                       DISCRIMINATOR (PatchGAN)                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Concatenated [EUV image, Magnetogram] (1024 × 1024 × 2)         │
│      │                                                                   │
│      ▼                                                                   │
│  Layer 1: Conv 4×4, s2 → 64, LReLU          (512×512×64)                │
│  Layer 2: Conv 4×4, s2 → 128, BN, LReLU     (256×256×128)               │
│  Layer 3: Conv 4×4, s2 → 256, BN, LReLU     (128×128×256)               │
│  Layer 4: Conv 4×4, s1 → 512, BN, LReLU     (127×127×512)               │
│  Layer 5: Conv 4×4, s1 → 1, Sigmoid         (126×126×1)                 │
│      │                                                                   │
│      ▼                                                                   │
│  Output: Patch probability map (real/fake)                               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
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
| Learning Rate | 2 × 10⁻⁴ |
| Beta1 | 0.5 |
| Input Size | 1024 × 1024 |
| Epochs | 200 |

## Data

### Training Data (Near-side)

| Source | Description |
|--------|-------------|
| SDO/AIA 304 nm | EUV images (input) |
| SDO/HMI | LOS magnetograms (target) |
| Period | 2011 January - 2017 December |
| Cadence | 6 hours |
| Total pairs | ~10,000 |

### Application Data (Far-side)

| Source | Description |
|--------|-------------|
| STEREO-A/EUVI 304 nm | Far-side EUV images |
| STEREO-B/EUVI 304 nm | Far-side EUV images |

## Results

### Near-side Validation

| Metric | Value |
|--------|-------|
| Pixel Correlation | 0.88 |
| Structural Similarity | 0.92 |

### Far-side Application

- Successfully generated far-side magnetograms from STEREO/EUVI data
- Generated magnetograms show reasonable agreement with helioseismic far-side images
- Active regions detected on far-side rotate to near-side with consistent magnetic features

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

# Input: (batch, 1, 1024, 1024) - EUV 304 nm image
euv_image = torch.randn(1, 1, 1024, 1024)

# Output: (batch, 1, 1024, 1024) - Generated magnetogram
with torch.no_grad():
    magnetogram = generator(euv_image)
```

## Reference Code

Original implementation: [https://github.com/tykimos/SolarMagGAN](https://github.com/tykimos/SolarMagGAN)

## Citation

```bibtex
@article{Kim_2019,
    title={Solar farside magnetograms from deep learning analysis of STEREO/EUVI data},
    author={Kim, Taeyoung and Park, Eunsu and Lee, Harim and Moon, Yong-Jae and Bae, Sung-Ho and Lim, Daye and Jang, Soojeong and Kim, Lokwon and Cho, Il-Hyun and Choi, Myungjin and Cho, Kyung-Suk},
    journal={Nature Astronomy},
    volume={3},
    pages={397--400},
    year={2019},
    publisher={Nature Publishing Group},
    doi={10.1038/s41550-019-0711-5}
}
```

## Related Publications

### Reply to: Reliability of AI-generated magnetograms from only EUV images (2021)

**Title:** Reply to: Reliability of AI-generated magnetograms from only EUV images

**Authors:** Eunsu Park, Hyun-Jin Jeong, Harim Lee, Taeyoung Kim, Yong-Jae Moon

**Journal:** Nature Astronomy, 5, 111-112, 2021

**DOI:** [10.1038/s41550-021-01311-5](https://doi.org/10.1038/s41550-021-01311-5)

**Type:** Matters Arising (Reply to Liu et al.)

**Summary:** This paper responds to constructive comments about limitations of the original study, explaining:

- **Dynamic range choice (±100G):** Selected to better show active region shapes and enable effective model training
- **Improved methods:** Pix2PixHD with larger dynamic range (±1,400G) published in ApJL; multi-channel EUV inputs achieving ±3,000G range
- **Preprocessing details:** Used aia_prep, hmi_prep, secchi_prep (SolarSoft) for Level 1.5 images; down-sampled to 1024×1024; solar radius set to 392 pixels; manually excluded poor quality images (4,972 pairs used)
- **Model selection:** Best model chosen from ~120 epochs (500,000 iterations) based on highest pixel-to-pixel Pearson correlation

**Follow-up Studies Referenced:**
- Shin et al. (2020), ApJL 895, L16 - Ca II to magnetogram translation with Pix2PixHD
- Jeong et al. (2020), ApJL 903, L25 - Coronal magnetic field extrapolation using AI-generated farside magnetograms

```bibtex
@article{Park_2021_Reply,
    title={Reply to: Reliability of AI-generated magnetograms from only EUV images},
    author={Park, Eunsu and Jeong, Hyun-Jin and Lee, Harim and Kim, Taeyoung and Moon, Yong-Jae},
    journal={Nature Astronomy},
    volume={5},
    pages={111--112},
    year={2021},
    publisher={Nature Publishing Group},
    doi={10.1038/s41550-021-01311-5}
}
```

### Solar Coronal Magnetic Field Extrapolation with AI-generated Farside (2020)

**Title:** Solar Coronal Magnetic Field Extrapolation from Synchronic Data with AI-generated Farside

**Authors:** Hyun-Jin Jeong, Yong-Jae Moon, Eunsu Park, Harim Lee

**Journal:** The Astrophysical Journal Letters, 903:L25 (9pp), 2020 November 1

**DOI:** [10.3847/2041-8213/abc255](https://doi.org/10.3847/2041-8213/abc255)

**Type:** Application Study (PFSS extrapolation using AI-generated farside magnetograms)

**Summary:** This paper applies AI-generated solar farside magnetograms (AISFM) for potential field source surface (PFSS) coronal magnetic field extrapolation:

- **Improved Model (Pix2PixHD):** Multi-scale discriminators with larger dynamic range (±3,000 G)
- **Multi-channel Input:** 3 EUV passbands (304, 193, 171 Å) instead of single 304 Å
- **AISFM Versions:**
  - AISFM 1.0: Original model (±100 G, single 304 Å)
  - AISFM 2.0: Pix2PixHD with ±3,000 G and 3-channel EUV input
- **Application:** PFSS extrapolation from synchronic magnetograms (SDO/HMI near-side + AISFM far-side)
- **Results:** Improved coronal magnetic field modeling including far-side active regions

**Reference Code:** [https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms](https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms)

```bibtex
@article{Jeong_2020,
    title={Solar Coronal Magnetic Field Extrapolation from Synchronic Data with AI-generated Farside},
    author={Jeong, Hyun-Jin and Moon, Yong-Jae and Park, Eunsu and Lee, Harim},
    journal={The Astrophysical Journal Letters},
    volume={903},
    number={2},
    pages={L25},
    year={2020},
    month={nov},
    publisher={The American Astronomical Society},
    doi={10.3847/2041-8213/abc255}
}
```

### Improved AI-generated Solar Farside Magnetograms and Data Release (2022)

**Title:** Improved AI-generated Solar Farside Magnetograms by STEREO and SDO Data Sets and Their Release

**Authors:** Hyun-Jin Jeong, Yong-Jae Moon, Eunsu Park, Harim Lee, Ji-Hye Baek

**Journal:** The Astrophysical Journal Supplement Series, 262:50 (12pp), 2022 October

**DOI:** [10.3847/1538-4365/ac8d66](https://doi.org/10.3847/1538-4365/ac8d66)

**Type:** Major Update (AISFM 3.0 model and public data release)

**Summary:** This paper presents the improved AISFM 3.0 model using Pix2PixCC architecture:

- **Model Evolution:**
  - AISFM 1.0: Original Pix2Pix (±100 G, 1 EUV channel)
  - AISFM 2.0: Pix2PixHD (±3,000 G, 3 EUV channels)
  - **AISFM 3.0:** Pix2PixCC (adds correlation coefficient-based loss)

- **Pix2PixCC Architecture:**
  - Correlation Coefficient (CC) loss added to L1 + cGAN + FM loss
  - Better preservation of structural features

- **Training Data:**
  - SDO/AIA (171, 193, 304 Å) and SDO/HMI magnetogram pairs
  - Period: 2011 January - 2017 December
  - Total: 8,148 pairs (train: 5,968, validation: 1,144, test: 1,036)

- **Performance (Near-side Test):**
  | Region | Pixel CC | RMSE (G) |
  |--------|----------|----------|
  | Full Disk | 0.88 | 35.9 |
  | Active Region | 0.91 | 95.9 |
  | Quiet Region | 0.70 | 10.9 |

- **Public Data Release:** AISFM 3.0 data available at [http://sdo.kasi.re.kr](http://sdo.kasi.re.kr)
  - STEREO-A farside magnetograms (2011 January - present)
  - STEREO-B farside magnetograms (2011 January - 2014 September)

**Reference Code:** [https://github.com/JeongHyunJin/Pix2PixCC](https://github.com/JeongHyunJin/Pix2PixCC)

```bibtex
@article{Jeong_2022,
    title={Improved AI-generated Solar Farside Magnetograms by STEREO and SDO Data Sets and Their Release},
    author={Jeong, Hyun-Jin and Moon, Yong-Jae and Park, Eunsu and Lee, Harim and Baek, Ji-Hye},
    journal={The Astrophysical Journal Supplement Series},
    volume={262},
    number={2},
    pages={50},
    year={2022},
    month={oct},
    publisher={The American Astronomical Society},
    doi={10.3847/1538-4365/ac8d66}
}
```

## License

MIT License
