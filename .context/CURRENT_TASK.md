# Current Task

Last update: 2026-01-25

## In Progress
- Validate GT reset behavior on both GUI variants.
- Validate output render stability after processing completes.
- Switch server interaction to RDP (xrdp).
- xrdp is listening on 3389 (IPv6).
- Direct RDP connection failed; plan to use SSH tunnel (3389 -> localhost).
- RDP login shows blank desktop; adjust xrdp session startup to Xfce.
- Applied startxfce4 in ~/.xsession and /etc/xrdp/startwm.sh; xrdp-sesman init script missing.
- Provider notice: system disk small; /root/rivermind-data not persisted on release; need to locate data disk mount for large files.
- df shows /root/rivermind-data is 49G; no other mounts listed yet.
- User confirmed data already stored in /root/rivermind-data; no migration needed now.
- Next fix: install xorgxrdp/dbus-x11 and restart xrdp to resolve RDP blank screen.
- xorgxrdp/dbus-x11 already installed; xrdp restarted, xrdp-sesman reported already running.
- SSH tunnel attempt failed due to password rejection; need current root password or reset via provider console.
- User reset root password via provider console; still failing SSH auth. Recommend setting password via `passwd` in active session.
- SSH port changed to 30011; need to update tunnel/known_hosts and re-auth.
- SSH login succeeded on port 30011.
- RDP connection still failing; need to verify SSH tunnel and local port availability.
- Windows localhost:3389 already occupied; use alternate local port (e.g., 3390) for tunnel.
- SSH tunnel reports connection refused; server port 3389 likely not listening (xrdp stopped).
- RDP session cannot launch terminal emulator; need to install/configure xfce4-terminal or xterm.
- Restarted xrdp/xrdp-sesman; xrdp listening on 3389. Installed xterm; set xfce4-terminal as default.
- RDP terminal shows garbled output; set LANG/LC_ALL in session and relogin.
- RDP clipboard paste not working; verify RDP clipboard setting and xrdp-chansrv.
- Update xfce4 terminalrc to set UTF-8 encoding and disable unsafe paste dialog.
- Fix texture generation error handler to avoid NameError in backup GUI.
- Server backup GUI patched; rerun GUI to confirm no NameError.
- Backup GUI relaunched; NameError resolved (only torchvision deprecation warning remains).
- Add filename labels above input/output images in backup GUI.
- Add explicit texture generation console logs in backup GUI.
- Adjust filename labels to show raw name without prefix.
- Texture log shows start without done; reset TEXTURE_STEPS to default and verify completion.
- Add pipeline load/failed logs to identify texture generation stall.
- Diffusers missing on server; install pinned diffusers/transformers/accelerate without reinstalling torch.
- Texture generation still reports diffusers missing; verify install inside venv-hp.
- diffusers import fails due to huggingface_hub cached_download removal; pin huggingface_hub to compatible version.
- diffusers now fails on OfflineModeIsEnabled; pin huggingface_hub to 0.20.x to satisfy both APIs.
- Texture generation fails: missing diffusion_pytorch_model.bin in model dir; need to download diffusers weights into stable-diffusion-v1-5.
- Use fp16 weights when available (diffusion_pytorch_model.fp16.bin) to avoid needing fp32 bin.
- Guard run_processing_thread to reload model if upsampler is None.
- Enable CPU offload/vae slicing/tiling for texture pipeline to avoid CUDA OOM.
- Add TEXTURE_MAX_DIM downscale to reduce texture generation VRAM use.
- TEXTURE_MAX_DIM patch failed on server; use targeted replacements for apply_texture_generation.
- Texture generation succeeds with downscale + fp16; verify TEXTURE_MAX_DIM=1024 run for quality/VRAM balance.

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
