# -Corruption-aware-image-restoration-for-SR-Inmapinting-
🛰️  Corruption‑aware image restoration for SR &amp; Inmapinting   for spaceborne optical navigation. Transformer based SR &amp; inpainting on SPEED+.

# 🛰️ Corruption-Aware Image Restoration for Spaceborne Optical Navigation

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mayhammad227/space-image-restoration)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides a complete **Python pipeline** for restoring corrupted spacecraft images, as described in the paper:

> *"A Corruption-Aware Image Restoration Framework for Spaceborne Optical Navigation Using the SPEED+ Dataset"*

We address two common degradation types encountered in on‑orbit optical navigation:
- **Super‑Resolution (SR):** Recovering high‑resolution details from low‑resolution sensor inputs using **SwinIR**.
- **Inpainting:** Reconstructing missing regions caused by sensor dropout or debris occlusion using **LaMa** (Large Mask Inpainting).

The entire workflow—from synthetic degradation to paper‑ready evaluation—is implemented in four self‑contained Colab notebooks.

---

## 📁 Repository Structure

├── 01_Setup_and_Degradation.ipynb # Creates LR/HR pairs & masked images
├── 02_SuperResolution_SwinIR.ipynb # Fine‑tunes SwinIR on SPEED+
├── 03_Inpainting_LaMa.ipynb # Fine‑tunes LaMa with spacecraft‑aware masks
├── 04_Evaluation_and_Summary.ipynb # Generates tables, plots & LaTeX output
├── README.md
└── .gitignore



---

## ⚡ Quick Start

### 1. Open in Google Colab
 Open in "Colab",  manually upload the notebooks to [colab.research.google.com] or run as Python, after installing the required packages  

### 2. Prepare Your Dataset
Place the **SPEED+** synthetic images in your Google Drive:

My Drive/
└── speedplus/
└── synthetic/
└── images/
├── img000001.jpg
├── img000002.jpg
└── ...



### 3. Run Notebooks in Order
1. `01_Setup_and_Degradation.ipynb` – Mount Drive, install dependencies, generate degraded pairs.
2. `02_SuperResolution_SwinIR.ipynb` – Train SwinIR for 4× SR.
3. `03_Inpainting_LaMa.ipynb` – Train LaMa on rectangular/irregular masks.
4. `04_Evaluation_and_Summary.ipynb` – Compute metrics, produce publication‑quality figures.

> **GPU Required:** Notebooks 2 & 3 need a T4 GPU (Runtime → Change runtime type).

---

## 📊 Expected Outputs

All results are saved to `My Drive/space_restoration/`:

| Output | Description |
|--------|-------------|
| `sr/hr/` & `sr/lr/` | High‑ and low‑resolution image pairs |
| `inpaint/gt/`, `masks/`, `masked/` | Inpainting ground truth, masks, and corrupted inputs |
| `checkpoints/swinir/` | Trained SwinIR model weights |
| `checkpoints/lama/` | Trained LaMa model weights |
| `paper_figures/` | Bar charts, distribution plots, and LaTeX tables |
| `sr_results.csv` | Per‑image PSNR/SSIM for SR |
| `inpaint_results.csv` | Per‑image PSNR/SSIM for inpainting |

---

## 🧠 Models

| Task | Model | Key Features |
|------|-------|--------------|
| Super‑Resolution | **SwinIR** | Shifted window transformer, pretrained on DIV2K, fine‑tuned with L1 loss |
| Inpainting | **LaMa** | Fast Fourier Convolutions (FFC), large‑mask robustness, mixed rect/irregular training |

---

## 📈 Sample Results

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Bicubic Upsampling | 36.15 | 0.932 |
| **SwinIR (Fine‑Tuned)** | **39.00** | **0.961** |
| Masked Input | 22.40 | 0.784 |
| **LaMa (Fine‑Tuned)** | **43.17** | **0.986** |

*(Results from a 100‑epoch run on SPEED+; actual values depend on training duration.)*

<img width="1700" height="1621" alt="download" src="https://github.com/user-attachments/assets/8f00906d-995e-4947-bf02-e6fba475942f" />
<img width="1273" height="415" alt="r1" src="https://github.com/user-attachments/assets/ddeca566-ed82-4cca-8828-61c6781b0aa6" />




---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{space_restoration_2026,
  author       = {May Hammad},
  title        = {A Corruption-Aware Image Restoration Framework for Spaceborne Optical Navigation},
  year         = {2026},
  
}


