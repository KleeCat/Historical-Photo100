# Current Task

Last update: 2026-02-12

## Recently Completed (2026-02-12)
- 完成两版 GUI 脚本差异分析并输出论文式中文文档：
  - `generated_documents/gui_super_resolution_comparative_study_cn_1770886461751.docx`
  - 对比对象：`(gui)super-resolution processing.py` 与 `(gui)super-resolution processing_server.py`

## In Progress
- Drag-and-drop disabled (windnd/Win32 incompatible with customtkinter); revisit if CTk adds native DnD.

## Known Optimization Opportunities
- **单图完成弹窗**: 每次单图处理完弹 `messagebox.showinfo`，可改为状态栏提示以减少打断。

## Recently Completed (2026-02-10)
- Fixed `actual_widget` identity check bug in `render_zoomed_image`, `show_image_file_ctk`, `show_image_ctk`:
  - Captured `is_output_panel` flag before `_set_label_image` call to prevent wrong panel being lifted after label recreation.
- Fixed batch output panel stale display:
  - `start_next_batch_item` now clears `self.img_output = None` and resets output filename label when advancing to next image.
  - `start_processing` now shows "Processing" overlay uniformly for both single and batch modes.
- Scratch repair thread safety: `process_image` now copies `self.img_input` into local `input_img` under `_state_lock` before processing, never mutates the shared reference.
- Moved `from gfpgan import GFPGANer` to file-level `try/except`; `process_image` checks `GFPGANer is None` instead of re-importing.
- `render_main_images` clear paths now use `_recreate_input_label()` / `_recreate_output_label()` instead of `configure(image="")` to avoid pyimageX errors.
- Unified `_state_lock` usage: added lock snapshots in `render_main_images`, `_render_output_frame_once`, `calculate_metrics`, `save_comparison`, `start_next_batch_item`.

## Recently Completed (2026-02-09)
- Run-scoped callback guard, thread-safe UI queue, persistent CTkImage refs, label recreation for pyimageX fix.
- See CHANGELOG.md for full details.

## Recently Completed (2026-02-08)
- **Thread safety**: `_state_lock` (img_input/img_output), `_model_lock` (model loading).
- **Model load mutex**: ComboBox disabled during load; `_model_lock.acquire(blocking=False)`.
- **Batch retry fix**: 500ms delay + status text before retry; prevents tight recursion.
- **GFPGANer cache**: `face_enhancer` / `face_enhancer_scale` instance vars; only rebuilt on scale change.
- **GPU VRAM cleanup**: `torch.cuda.empty_cache()` in process_image finally block.
- **Auto tile**: `auto_tile_size(h, w, scale)` picks tile based on VRAM; set on `upsampler.tile`.
- **Scratch Repair UI**: switch in sidebar row 9; calls `apply_scratch_repair` before upscale.
- **Output formats**: save_image supports TIFF/WebP with quality params; save dialog updated.
- **Progress bar fix**: creep animation (+0.001/tick) when bar catches target; stage targets rebalanced (upscale 10%→65%).
- **Logging**: `logging` module replaces `print`; `logger = logging.getLogger("super_resolution_gui")`.
- **Type hints**: added to `ensure_dir`, `write_json_file`, `save_image`, `blend_images`, `apply_unsharp_mask`, `apply_film_grain`, `blend_with_lr`, `estimate_image_metrics`, `auto_tile_size`.
- **UI constants**: `UI_WINDOW_SIZE`, `UI_SIDEBAR_WIDTH`, `UI_COLOR_*` centralized.
- **Image display fix**: `render_zoomed_image` / `show_image_ctk` use parent frame size (not label); fit mode at zoom<=1.0.
- **Image panel layout fix**: filenames moved below headers (row 1), image panels row 2, resolution labels row 3 (no overlap).
- **Filename UX**: show `Input: <basename>` on load and `Output: <basename>` after processing.
- **Image sizing refinement**: `_get_image_display_size(label_widget)` now reads panel frame dimensions directly to avoid clipping.
- **Progress creep refinement**: long-stage creep increased to `max(0.0015, remaining * 0.012)` toward 0.98.
- **Output render regression fix**: added `prepare_display_image()` normalization to contiguous `uint8` BGR before GUI render.
- **Output repaint hardening**: added `after(20, show_image_ctk)` after completion to force final output frame refresh.
- **Output render pipeline hardening**: added `refresh_output_after_success()` and consolidated final success repaint on UI thread.
- **Overlay safety**: `render_main_images()` now auto-hides output overlay when processing is finished and output exists.
- **Output stacking fix**: `hide_output_overlay()` now calls `lower()`, and output labels call `lift()` after each render.
- **Stable output buffer**: after saving `run_output_path`, GUI reloads that snapshot for display to avoid transient memory-layout render failures.
- **Agent workflow hardening**: updated `AGENTS.md` with `apply_patch` stale-context guidance for
  `Failed to find expected lines` (re-read region, use minimal hunks, regenerate after partial apply).
- **Batch output visibility fix**: keep previous output visible during batch item transitions; do not force overlay on output panel in batch mode.
- **Batch labeling fix**: update `Input:` filename for each batch item in `start_next_batch_item()`.
- **Batch repaint pacing**: add 120ms delay before launching next batch item to let just-finished output render.
- **Single repaint hardening**: added `_render_output_frame_once()` + multi-pass retries in `refresh_output_after_success()`.
- **Single final fallback**: added delayed `render_output_from_file(run_output_path)` after success for guaranteed output panel refresh.
- **View reset on single runs**: `run_processing_thread()` now calls `reset_view_state()` before processing to avoid stale zoom crop display.
- **Direct file render fallback**: added `show_image_file_ctk(path, label_widget)` and updated `render_output_from_file()` to render from disk first, then sync `img_output` for metrics.
- **GIL crash fixes**: `load_model` Tkinter calls wrapped in `self.after(0, ...)`; windnd removed.
- **Drag-and-drop disabled**: windnd and Win32 SetWindowLongPtr both crash CTk; methods retained for future.

## Handoff Notes (2026-01-24)
- Server uses `/root/rivermind-data/venv-hp` with system-site-packages; do not reinstall torch.
- Required pins: numpy 1.26.4, opencv-python-headless 4.8.1.78, basicsr 1.4.2 (`--no-deps --no-build-isolation`), gfpgan 1.3.8 (`--no-deps`), realesrgan 0.3.0 (`--no-deps`).
- X11 GUI via VcXsrv + `ssh -X`; set `DISPLAY=localhost:0.0` on Windows.
- If diffusers error `torch.distributed.device_mesh` appears, downgrade: diffusers 0.25.1, transformers 4.36.2, accelerate 0.25.0.
- Server should pull latest commit (336a190) or upload files before retest.

## Handoff Notes (2026-01-25)
- Local repo at `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100`; server repo at `/root/rivermind-data/Historical-Photo100`.
- RDP works via tunnel: `ssh -L 3390:localhost:3389 root@sh01-ssh.gpuhome.cc -p 30011`, then RDP to `localhost:3390` (Xorg). Restart xrdp as root (no sudo).
- Terminal paste/encoding fixed; custom motd error silenced in `~/.bashrc` on server.
- Backup GUI texture updates: filename labels (basename), error handler captures exception string, texture logs (loading/ready/start/done/failed), fp16 variant support, CPU offload + VAE slicing/tiling, `TEXTURE_MAX_DIM` downscale + upsample, and `upsampler` guard in `run_processing_thread`.
- Diffusers pins on server: diffusers 0.25.1, transformers 4.36.2, accelerate 0.25.0, huggingface_hub 0.20.2, safetensors.
- Model path: `/root/rivermind-data/models/stable-diffusion-v1-5` (fp16 only). `TEXTURE_MAX_DIM=1024` tested successfully.
- Local commit `c570464` pushed to `main` (texture refinement stability under VRAM limits).

## Done
- Added gitignore entries for datasets, outputs, and new log folders.
- Show `Batch i/N` prefix in status updates during batch runs.
- Ignore `outputs/batch/` run artifacts in `.gitignore`.
- Added additional cancel checks to shorten batch cancel latency in `(gui)super-resolution processing.py`.
- Compacted sidebar spacing and improved disabled button label contrast in `(gui)super-resolution processing.py`.
- Added batch queue count pop-up and `batch_queue.json` output for batch runs.
- Switched Codespaces devcontainer image to `mcr.microsoft.com/devcontainers/python:3.10` and removed python feature (yarn repo error fix).
- Configured Git LFS tracking and migrated data folders into Git.
- Added `LR/`, `HR/`, `SR/`, `outputs/` directories with sample assets committed.
- Implemented `web_sr_server.py` web UI with desktop-like layout and feature parity (GT metrics, comparison, feature export).
- Added GUI guards for missing input/model before processing to improve stability.
- Added per-run output snapshots and JSON logs under `outputs/<timestamp>_<name>_<id>/` for reproducibility.
- Added per-stage timing and metrics logging (PSNR/SSIM, per-step durations) to `run_log.json`.
- Added "Open Last Run Folder" button for quick access to outputs.
- Added scratch repair and colorization flow to the server GUI pipeline.
- Added non-systemd xrdp startup steps to `.context/remote_access.md`.
- Normalized remaining SSH port references across the repo to 30011.
- Updated SSH port references to 30011 for the new instance.
- Set `TEXTURE_MAX_DIM` default to 1536 for the server GUI.
- Renamed the backup GUI to `(gui)super-resolution processing_server.py` and updated runbook references.
- Updated `AGENTS.md` response suffix from "喵" to "喵~".
- Moved `docs/remote_access.md` to `.context/remote_access.md` and added the response suffix rule to `AGENTS.md`.
- Added `docs/remote_access.md` with RDP tunnel and texture pipeline notes.
- Logged 2026-01-25 RDP and texture pipeline handoff notes.
- Updated `AGENTS.md` with test placement and file IO/optional dependency guidance.
- Clone instance created and data disk verified.
- Real-ESRGAN weights present under `/root/.cache/realesrgan`.
- GUI output refresh fix applied locally.
- Backup GUI has texture generation enabled locally.
- Backup GUI includes torch.xpu compatibility stub for diffusers.
- Stable Diffusion v1.5 diffusers repo trimmed to ~9.4G at `/root/rivermind-data/models/stable-diffusion-v1-5`.
- GT is cleared when loading a new input image in both GUI files.
- Added output refresh fallback to force render via `show_image_ctk` if zoom render fails.
- Updated `AGENTS.md` with build/test commands and code-style guidance.
- Refined `AGENTS.md` with build/lint/test commands and agent style guidance (2026-01-26).

## Command Steps
1) Upload GUI files
```
scp -P 30011 "D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(gui)super-resolution processing.py" root@sh01-ssh.gpuhome.cc:/root/rivermind-data/Historical-Photo100/
scp -P 30011 "D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(gui)super-resolution processing_server.py" root@sh01-ssh.gpuhome.cc:/root/rivermind-data/Historical-Photo100/
```

2) Connect with X11
```
set DISPLAY=localhost:0.0
ssh -X -o ForwardX11Trusted=yes -o ExitOnForwardFailure=yes root@sh01-ssh.gpuhome.cc -p 30011
```

3) Run texture GUI
```
cd /root/rivermind-data/Historical-Photo100
source /root/rivermind-data/venv-hp/bin/activate
TEXTURE_MODEL_ID="/root/rivermind-data/models/stable-diffusion-v1-5" \
  python "(gui)super-resolution processing_server.py"
```

4) If diffusers is missing
```
python -m pip install diffusers transformers accelerate safetensors huggingface_hub
```

## Notes
- Prefer running GUI via X11 forwarding (VcXsrv + `ssh -X`).
- Keep large models on `/root/rivermind-data` to avoid filling system disk.
