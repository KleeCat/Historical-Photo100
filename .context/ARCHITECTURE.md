# Architecture

## Runtime Flow
- Load input image (LR) with OpenCV, normalize channels to BGR.
- Run Real-ESRGAN upsampler (RRDBNet + RealESRGANer).
- Optional: GFPGAN face enhancement, blended with SR base.
- Optional: texture generation via diffusers img2img pipeline.
- Post-process: blend with LR, unsharp mask, film grain, feature capture.
- Save outputs under `outputs/` (compare/features/result) and metrics tables.

## Entry Points
- `(gui)super-resolution processing.py`: main GUI workflow and exports.
- `esrgan_gui.py`: batch GUI with metrics/CSV export.
- `super-resolution processing.py`: CLI batch SR with optional GFPGAN.
- `Quantitative assessment and frequency domain analysis.py`: evaluation + plots.

## Key Dependencies
- torch + basicsr + realesrgan for SR inference.
- gfpgan for face enhancement (optional).
- diffusers/transformers/accelerate for texture generation (optional).
- scikit-image, lpips, pytorch-fid for evaluation.

## Paths and Storage
- Data roots: `LR/`, `SR/`, optional `HR/`.
- Outputs: `outputs/compare`, `outputs/features`, `evaluation_results/`.
- Real-ESRGAN weights: `~/.cache/realesrgan/RealESRGAN_x4plus.pth`.
- Texture model: `TEXTURE_MODEL_ID` points to local diffusers repo.
