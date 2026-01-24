# Current Task

Last update: 2026-01-24

## In Progress
- Validate GT reset behavior on both GUI variants.
- Validate output render stability after processing completes.

## Done
- Clone instance created and data disk verified.
- Real-ESRGAN weights present under `/root/.cache/realesrgan`.
- GUI output refresh fix applied locally.
- Backup GUI has texture generation enabled locally.
- Backup GUI includes torch.xpu compatibility stub for diffusers.
- Stable Diffusion v1.5 diffusers repo trimmed to ~9.4G at `/root/rivermind-data/models/stable-diffusion-v1-5`.
- GT is cleared when loading a new input image in both GUI files.
- Added output refresh fallback to force render via `show_image_ctk` if zoom render fails.

## Command Steps
1) Upload GUI files
```
scp -P 30017 "D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(gui)super-resolution processing.py" root@sh01-ssh.gpuhome.cc:/root/rivermind-data/Historical-Photo100/
scp -P 30017 "D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(gui)super-resolution processing_backup.py" root@sh01-ssh.gpuhome.cc:/root/rivermind-data/Historical-Photo100/
```

2) Connect with X11
```
set DISPLAY=localhost:0.0
ssh -X -o ForwardX11Trusted=yes -o ExitOnForwardFailure=yes root@sh01-ssh.gpuhome.cc -p 30017
```

3) Run texture GUI
```
cd /root/rivermind-data/Historical-Photo100
source /root/rivermind-data/venv-hp/bin/activate
TEXTURE_MODEL_ID="/root/rivermind-data/models/stable-diffusion-v1-5" \
  python "(gui)super-resolution processing_backup.py"
```

4) If diffusers is missing
```
python -m pip install diffusers transformers accelerate safetensors huggingface_hub
```

## Notes
- Prefer running GUI via X11 forwarding (VcXsrv + `ssh -X`).
- Keep large models on `/root/rivermind-data` to avoid filling system disk.
