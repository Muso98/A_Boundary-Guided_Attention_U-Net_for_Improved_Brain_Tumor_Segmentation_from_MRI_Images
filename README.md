# Boundary-Guided Attention U-Net for Brain Tumor Segmentation from MRI Images

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)]()

## Overview

This repository provides the full implementation of a **Boundary-Guided Attention U-Net (BGA-UNet)** for improved brain tumor segmentation from MRI images. The project extends the classical Attention U-Net architecture by incorporating explicit boundary-awareness mechanisms, enabling the model to better delineate tumor edges — a critical challenge in medical image segmentation where accurate boundary detection directly impacts clinical decision-making.

Three model variants are implemented and compared:

- **Standard U-Net** (`model.py`) — baseline encoder-decoder segmentation network.
- **Attention U-Net** (`attention_unet.py`) — U-Net augmented with soft attention gates in skip connections.
- **Boundary-Guided Attention U-Net v1** (`boundary_attention_unet.py`) — introduces boundary supervision as an auxiliary training signal.
- **Boundary-Guided Attention U-Net v2** (`boundary_attention_unet_v2.py`) — refined architecture with improved boundary feature integration.
- **Hybrid Model v1 & v2** (`hybrid_model.py`, `hybrid_model_v2.py`) — combines boundary guidance with additional architectural components for further performance gains.

---

## Repository Structure

```
.
├── model.py                                         # Baseline U-Net
├── attention_unet.py                                # Attention U-Net
├── boundary_attention_unet.py                       # BGA-UNet v1
├── boundary_attention_unet_v2.py                    # BGA-UNet v2
├── hybrid_model.py                                  # Hybrid model v1
├── hybrid_model_v2.py                               # Hybrid model v2
├── dataset.py                                       # Dataset loading and preprocessing
├── main.py                                          # Main training entry point
├── train.py                                         # Training loop (baseline)
├── train_attention_unet.py                          # Training loop for Attention U-Net
├── train_boundary_attention_unet.py                 # Training loop for BGA-UNet v1
├── train_boundary_attention_unet_v2.py              # Training loop for BGA-UNet v2
├── train_hybrid.py                                  # Training loop for Hybrid v1
├── train_hybrid_v2.py                               # Training loop for Hybrid v2
├── infer_sample.py                                  # Single sample inference
├── infer_all.py                                     # Batch inference across all models
├── infer_attention_unet.py                          # Inference for Attention U-Net
├── infer_boundary_attention_unet.py                 # Inference for BGA-UNet v1
├── infer_boundary_attention_unet_v2.py              # Inference for BGA-UNet v2
├── infer_hybrid.py                                  # Inference for Hybrid v1
├── infer_hybrid_v2.py                               # Inference for Hybrid v2
├── eval_boundary_attention_unet.py                  # Evaluation for BGA-UNet v1
├── evaluate_boundary_attention_unet_v2.py           # Evaluation for BGA-UNet v2
├── confusion_matrix_eval.py                         # Confusion matrix computation
├── confusion_matrix_plot.py                         # Confusion matrix visualisation
├── plot_confusion_matrix_boundary_attention_unet_v2.py
├── plot_confusion_matrix_normalized.py
├── visual_results_boundary_attention_unet_v2.py     # Visual output generation
├── notes/                                           # Experimental notes
├── outputs/                                         # Saved predictions and figures
└── .gitignore
```

---

## Installation

```bash
git clone https://github.com/Muso98/A_Boundary-Guided_Attention_U-Net_for_Improved_Brain_Tumor_Segmentation_from_MRI_Images.git
cd A_Boundary-Guided_Attention_U-Net_for_Improved_Brain_Tumor_Segmentation_from_MRI_Images
pip install -r requirements.txt
```

**Main dependencies:**

- Python 3.8+
- PyTorch >= 1.12
- torchvision
- NumPy
- OpenCV
- scikit-learn
- matplotlib

---

## Dataset

This project is designed to work with the **BraTS (Brain Tumor Segmentation)** benchmark dataset or any MRI dataset structured with paired image and mask files. Preprocessing and loading logic is handled in `dataset.py`.

Place your dataset in a directory with the following structure:

```
data/
├── images/
│   ├── sample_001.png
│   └── ...
└── masks/
    ├── sample_001.png
    └── ...
```

Update the data paths in the relevant training script before launching.

---

## Training

Each model variant has a dedicated training script. To train the Boundary-Guided Attention U-Net v2 (recommended):

```bash
python train_boundary_attention_unet_v2.py
```

To train the baseline U-Net for comparison:

```bash
python train.py
```

All training scripts log loss curves and save model checkpoints to the `outputs/` directory.

---

## Inference

To run inference on a single sample:

```bash
python infer_sample.py
```

To run batch inference across all trained models for comparison:

```bash
python infer_all.py
```

Model-specific inference scripts are also available (e.g., `infer_boundary_attention_unet_v2.py`).

---

## Evaluation

To evaluate the BGA-UNet v2 and generate quantitative metrics (Dice Score, IoU, precision, recall):

```bash
python evaluate_boundary_attention_unet_v2.py
```

To compute and visualise confusion matrices:

```bash
python confusion_matrix_eval.py
python plot_confusion_matrix_normalized.py
```

Visual segmentation outputs are generated via:

```bash
python visual_results_boundary_attention_unet_v2.py
```

---

## Key Contributions

- Explicit **boundary supervision** as an auxiliary loss to improve delineation of tumor edges.
- Integration of **attention gates** in skip connections to focus the decoder on diagnostically relevant regions.
- Systematic comparison across five model variants under identical training conditions.
- Modular codebase allowing straightforward replacement or extension of any architectural component.

---

## Authors

Musabek Musaev, Mirafzal Mirholikov, Akmalbek Abdusalomov, Renat Abdirayimov, Lutfulla Murodjonov, Gulkhayo Urinova, Kyandoghere Kyamakya, Selain Kasereka

---

## Citation

If you use this code in your research, please cite this repository:

```bibtex
@misc{muso98_bga_unet_2024,
  author       = {Musabek Musaev and Mirafzal Mirholikov and Akmalbek Abdusalomov and Renat Abdirayimov and Lutfulla Murodjonov and Gulkhayo Urinova and Kyandoghere Kyamakya and Selain Kasereka},
  title        = {A Boundary-Guided Attention U-Net for Improved Brain Tumor Segmentation from MRI Images},
  year         = {2026},
  howpublished = {\url{https://github.com/Muso98/A_Boundary-Guided_Attention_U-Net_for_Improved_Brain_Tumor_Segmentation_from_MRI_Images}}
}
```

---

## License

This project is released under the MIT License. See `LICENSE` for details.
