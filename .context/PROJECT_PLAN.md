# Project Plan

## Goals
- Provide a reliable super-resolution workflow for historical photos.
- Offer GUI and CLI entry points with optional face enhancement and texture refinement.
- Support evaluation outputs (PSNR/SSIM/LPIPS/FID) and exportable assets.

## Tech Stack
- Python 3.10 (venv with system torch)
- torch 2.1.0 (system), basicsr 1.4.2, realesrgan 0.3.0
- gfpgan 1.3.8 (optional), diffusers/transformers/accelerate (optional texture)
- numpy 1.26.4, opencv-python-headless 4.8.1.78, Pillow, customtkinter
- scikit-image, lpips, pytorch-fid, matplotlib, scipy

## Module Breakdown
- GUI: `esrgan_gui.py`, `(gui)super-resolution processing.py`
- CLI: `super-resolution processing.py`
- Evaluation: `Quantitative assessment and frequency domain analysis.py`
- Model: `RRDBNet_arch.py`
- Utilities: `download.py`, config and output helpers
- Data: `LR/`, `SR/`, `HR/`, `outputs/`, `evaluation_results/`
