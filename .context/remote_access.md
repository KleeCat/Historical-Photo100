# Remote Access + Texture Pipeline Notes

## Repo Locations
- Local: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100`
- Server: `/root/rivermind-data/Historical-Photo100`

## RDP via SSH Tunnel (Xorg)
1) Open tunnel from local machine:
```bash
ssh -L 3390:localhost:3389 root@sh01-ssh.gpuhome.cc -p 30011
```
2) Connect RDP client to `localhost:3390`.
3) If RDP fails to connect, restart xrdp as root (no sudo required):
```bash
systemctl restart xrdp xrdp-sesman
```

## Server Environment
- Virtual environment: `/root/rivermind-data/venv-hp` (system-site-packages enabled).
- Do not reinstall torch in this venv.
- Diffusers pins (avoid torch reinstall):
  - `diffusers==0.25.1`
  - `transformers==4.36.2`
  - `accelerate==0.25.0`
  - `huggingface_hub==0.20.2`
  - `safetensors`

## Texture Pipeline Usage
Run the server GUI with the model path set:
```bash
TEXTURE_MODEL_ID="/root/rivermind-data/models/stable-diffusion-v1-5" \
  python3 "(gui)super-resolution processing_server.py"
```

Optional environment overrides:
- `TEXTURE_MAX_DIM` (default: 1536). Set to 1024 if you need lower VRAM.

## Model Files
- Diffusers model dir: `/root/rivermind-data/models/stable-diffusion-v1-5`.
- This repo currently has fp16 weights only.
- The GUI will use `variant="fp16"` automatically when
  `diffusion_pytorch_model.fp16.bin` is present.

## Texture Stability Notes
- CPU offload + VAE slicing/tiling reduce VRAM pressure.
- `TEXTURE_MAX_DIM` downsizes before texture generation and upscales after.
- Logs show: `Texture pipeline: loading/ready` and
  `Texture generation: start/downscale/upscale/done`.

## Terminal Encoding and MOTD
- If terminal paste/encoding is broken, ensure UTF-8 locale is set for the session.
- If you see a custom MOTD error on login, comment the
  `/etc/custom-motdexport` reference in `~/.bashrc`.
