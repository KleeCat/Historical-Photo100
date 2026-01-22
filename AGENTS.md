# Codex Working Rules (Must Follow)

## Language
- Chat with me in Chinese.
- Comments inside code must be in English (//, #, /* */, docstring).

## Install Location
- When installing tools/dependencies, install to D: drive by default (e.g., D:\Tools\ or D:\Programs\).
- Prefer reproducible install commands (show commands and paths).

## File Editing Policy
- When modifying files, only use patch/diff style edits (minimal changes).
- Do NOT replace entire files with full rewrites.
- Always show unified diff for modifications.

## Output Style (Save Context)
- For long outputs (>40 lines or >500 chars), write to a file instead of printing in terminal.
- Terminal output should be short: 1) What was done (1-2 sentences) 2) Which files changed (with paths) 3) Next steps (max 3)

## Workflow
- Default: Plan (3 bullets) -> Execute -> Verify.
- If deleting/overwriting files, list targets first; if context might be insufficient, suggest starting a new session.

## Repository Overview
- This repo is a set of Python scripts for image super-resolution and evaluation.
- Key scripts (entry points): `esrgan_gui.py` (Tk GUI for batch SR, metrics table, CSV export); `(gui)super-resolution processing.py` (CustomTk GUI with comparison + feature export); `super-resolution processing.py` (CLI batch SR with optional GFPGAN); `Quantitative assessment and frequency domain analysis.py` (PSNR/SSIM/LPIPS/FID + plots); `RRDBNet_arch.py` (RRDBNet architecture definition); `download.py` (DIV2K LR sample download).

## Environment Setup (Windows)
- Prefer a virtual environment on the D: drive.
- Example (adjust CUDA wheel as needed):
```
python -m venv D:\Tools\historical-photo100-venv
D:\Tools\historical-photo100-venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install basicsr realesrgan opencv-python numpy pillow
python -m pip install gfpgan customtkinter scikit-image lpips pytorch-fid matplotlib scipy requests
```
- If GPU is not available, install the CPU torch wheel instead.

## Run Commands (Scripts)
- GUI (recommended): `python esrgan_gui.py`
- GUI (CustomTk): `python "(gui)super-resolution processing.py"`
- CLI batch SR: `python "super-resolution processing.py"`
- Evaluation + reports: `python "Quantitative assessment and frequency domain analysis.py"`
- Download sample data: `python download.py`
- Note: several scripts have hard-coded default paths; update config blocks when needed.

## Build / Lint / Test
- No build step; scripts run directly with Python.
- No automated test suite is included by default.
- If you add tests under `tests/`, use pytest:
  - Run all tests: `python -m pytest`
  - Run a single file: `python -m pytest tests/test_file.py`
  - Run a single test: `python -m pytest tests/test_file.py::TestClass::test_name`
  - Run a single test by pattern: `python -m pytest -k "pattern"`
  - Run a single test in a file by pattern: `python -m pytest tests/test_file.py -k "pattern"`
- Optional lint/format (only if tools are installed):
  - `python -m ruff check .`
  - `python -m black .`
  - `python -m isort .`

## Code Style Guidelines
- Imports: standard library, third-party, then local; keep groups separated by blank lines.
- Imports: prefer one import per line; avoid unused imports.
- Formatting: 4-space indent; line length target around 100; prefer f-strings.
- Naming: snake_case for functions/vars; CamelCase for classes; UPPER_SNAKE for constants.
- Types: add type hints for new/modified public functions; use `Optional`, `List`, `Tuple`.
- Types: avoid `Any` unless necessary; prefer precise container types.
- Docstrings: short English docstrings for non-obvious functions and public APIs.
- Entrypoints: wrap CLI execution in `if __name__ == "__main__":` and keep top-level code minimal.
- Error handling: guard file IO, model loading, and GPU ops with try/except and clear messages.
- Error handling: preserve tracebacks for unexpected errors; do not silently ignore failures.
- Logging: keep CLI logs concise; avoid noisy per-pixel logging.
- Logging: for GUI, post updates via the queue; update UI on the main thread only.
- Image handling: normalize channels (BGR/GRAY/BGRA), clip to uint8 before saving.
- Config: keep default paths in one place; allow user override via variables/GUI fields.

## Paths, Data, and Outputs
- Default data dirs: `LR/`, `SR/`, optional `HR/`, and `evaluation_results/`.
- Generated outputs may also appear under `outputs/` or `output/`.
- Do not commit large datasets, outputs, or weights; respect `.gitignore`.
- Default Real-ESRGAN weight path:
  - `C:\Users\ihggk\.cache\realesrgan\RealESRGAN_x4plus.pth`
- GFPGAN weights are optional and should live in user cache locations.

## Model/GUI Notes
- Use `tile_size` for memory-heavy images; 0 means no tiling.
- Prefer worker threads for SR processing; keep UI responsive; use queue-based messaging to avoid cross-thread UI access.

## Dependency Notes
- `torch`, `basicsr`, `realesrgan` are required for SR.
- `gfpgan` is optional; only enable face enhancement when installed and weights exist.
- GUI preview uses `Pillow`; CustomTk GUI needs `customtkinter`.
- Evaluation uses `scikit-image`, `lpips`, `pytorch-fid`, `matplotlib`, `scipy`.
- If a dependency is missing, print a clear install hint and exit gracefully.
- Use `torch.cuda.is_available()` to decide CPU vs GPU execution.

## Configuration and Defaults
- Keep default paths in one place (top of file or `CONFIG` dict).
- Use raw strings for Windows paths.
- Prefer `os.path.join` over string concatenation.
- Ensure output directories exist before writing.
- Avoid writing to repo root unless explicitly requested.
- Avoid hardcoding user-specific paths in new scripts.
- Keep config dicts uppercase for clarity in CLI scripts.

## Image IO and Metrics
- Read images with `cv2.imread(..., cv2.IMREAD_UNCHANGED)` and normalize channels.
- Convert GRAY or BGRA to BGR before processing.
- Clamp/convert outputs to uint8 before saving.
- Use consistent resize interpolation (`cv2.INTER_CUBIC`) for baselines.
- For evaluation, match HR to LR/SR by filename conventions (e.g., `x4` suffix).
- Log PSNR/SSIM/LPIPS/FID with clear units and ranges.
- Prefer BGR for cv2; convert to RGB only for UI previews.

## Performance and GPU
- Wrap inference in `torch.no_grad()` to reduce memory.
- Use `tile_size` when GPU memory is limited; keep `tile_size=0` for full image.
- Prefer CPU-safe defaults (`half=False`) unless performance is validated.
- Avoid per-pixel logging; log per-image progress only.
- Release temporary tensors promptly to reduce VRAM pressure.

## Execution Tips
- Use absolute paths when running from other directories; quote file paths with spaces or parentheses.
- Start with a small image set; keep a backup of outputs when overwriting; validate model weight paths before long batch runs.

## Reporting and Artifacts
- `evaluation_results/` stores CSVs, plots, and reports; `outputs/compare` and `outputs/features` are used by the CustomTk GUI.
- Use timestamps in filenames to avoid overwrites; confirm write permissions before long batch runs; for long logs, save to a text file and link it in messages.

## Version Control Hygiene
- Do not commit datasets, outputs, or model weights.
- Keep generated artifacts under `LR/`, `SR/`, `outputs/`, `evaluation_results/`.
- Clean up large local files before committing.
- Respect existing `.gitignore` entries.

## Git Push Proxy
- If GitHub push fails on this machine, retry with Clash proxy: `git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push origin main`.

## Troubleshooting
- If a script fails to find images, verify the input directory and extensions.
- If CUDA OOM occurs, lower `tile_size` or switch to CPU.
- If evaluation fails, confirm `LR/`, `HR/`, and `SR/` filenames align.
- When in doubt, run on a single small image first.

## Cursor/Copilot Rules
- No `.cursor` rules or Copilot instructions were found in this repo.
