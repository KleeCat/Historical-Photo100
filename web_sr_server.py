import argparse
import html
import importlib.util
import json
import mimetypes
import os
import sys
import tempfile
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlparse

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    psnr = None
    ssim = None


BASE_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = BASE_DIR / "super-resolution processing.py"
JOBS: Dict[str, Dict[str, object]] = {}
CLI_MODULE = None


def load_cli_module():
    global CLI_MODULE
    if CLI_MODULE is not None:
        return CLI_MODULE
    spec = importlib.util.spec_from_file_location("sr_cli", CLI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load CLI module from {CLI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sr_cli"] = module
    spec.loader.exec_module(module)
    CLI_MODULE = module
    return module


def default_model_path(scale_factor: int) -> str:
    env_path = os.environ.get("REALESRGAN_MODEL_PATH")
    if env_path:
        return env_path
    model_name = "RealESRGAN_x2plus.pth" if scale_factor == 2 else "RealESRGAN_x4plus.pth"
    return str(Path.home() / ".cache" / "realesrgan" / model_name)


def guess_content_type(path: str) -> str:
    content_type, _ = mimetypes.guess_type(path)
    return content_type or "application/octet-stream"


def normalize_input_image(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def save_image(path: str, bgr_img: np.ndarray) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp"]:
        ext = ".png"
        path = path + ext
    success, buf = cv2.imencode(ext, bgr_img)
    if not success:
        raise RuntimeError("Failed to encode image")
    buf.tofile(path)
    return path


def make_comparison_images(lr_bgr: np.ndarray, sr_bgr: np.ndarray, scale: int, base_name: str, out_dir: str) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    h, w = sr_bgr.shape[:2]
    lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    pair = np.hstack([lr_up, sr_bgr])
    pair_path = os.path.join(out_dir, f"{base_name}_x{scale}_lr_sr.png")
    save_image(pair_path, pair)

    diff = cv2.absdiff(sr_bgr, lr_up)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_norm = cv2.normalize(diff_gray, np.zeros_like(diff_gray), 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)

    crop_size = max(32, min(h, w) // 3)
    cx, cy = w // 2, h // 2
    x1 = max(0, min(cx - crop_size // 2, w - crop_size))
    y1 = max(0, min(cy - crop_size // 2, h - crop_size))

    lr_crop = lr_up[y1:y1 + crop_size, x1:x1 + crop_size]
    sr_crop = sr_bgr[y1:y1 + crop_size, x1:x1 + crop_size]
    zoom = np.hstack([lr_crop, sr_crop])
    zoom_vis = cv2.resize(zoom, (w, h), interpolation=cv2.INTER_NEAREST)

    grid_top = np.hstack([lr_up, sr_bgr])
    grid_bottom = np.hstack([heat, zoom_vis])
    grid = np.vstack([grid_top, grid_bottom])
    grid_path = os.path.join(out_dir, f"{base_name}_x{scale}_grid.png")
    save_image(grid_path, grid)
    return pair_path, grid_path


def tensor_to_grid_image(tensor: torch.Tensor, grid: int = 4, max_channels: int = 16) -> Optional[np.ndarray]:
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] == 0:
        return None
    channels = min(tensor.shape[0], max_channels)
    images: List[np.ndarray] = []
    for idx in range(channels):
        feature = tensor[idx].cpu().numpy()
        f_min = float(feature.min())
        f_max = float(feature.max())
        if f_max - f_min < 1e-6:
            feature_norm = np.zeros_like(feature, dtype=np.uint8)
        else:
            feature_norm = ((feature - f_min) / (f_max - f_min) * 255.0).astype(np.uint8)
        images.append(cv2.cvtColor(feature_norm, cv2.COLOR_GRAY2BGR))
    if not images:
        return None
    while len(images) < grid * grid:
        images.append(np.zeros_like(images[0]))
    rows = []
    for r in range(grid):
        rows.append(np.hstack(images[r * grid:(r + 1) * grid]))
    return np.vstack(rows)


def save_feature_grids(feature_maps: List[Tuple[str, torch.Tensor]], base_name: str, scale: int, out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    saved: List[str] = []
    for idx, (name, tensor) in enumerate(feature_maps):
        grid_img = tensor_to_grid_image(tensor)
        if grid_img is None:
            continue
        path = os.path.join(out_dir, f"{idx:02d}_{base_name}_x{scale}_{name}.png")
        save_image(path, grid_img)
        saved.append(path)
    return saved


def capture_feature_maps(model: torch.nn.Module, bgr_img: np.ndarray, device: torch.device, max_maps: int = 6) -> List[Tuple[str, torch.Tensor]]:
    feature_maps: List[Tuple[str, torch.Tensor]] = []
    hooks = []

    def make_hook(name: str):
        def hook(_module, _input, output):
            if len(feature_maps) >= max_maps:
                return
            if isinstance(output, (tuple, list)):
                if not output:
                    return
                output_tensor = output[0]
            else:
                output_tensor = output
            if not torch.is_tensor(output_tensor) or output_tensor.ndim != 4:
                return
            _, _, h, w = output_tensor.shape
            if h < 16 or w < 16 or h > 1024 or w > 1024:
                return
            feature_maps.append((name, output_tensor.detach().cpu()))
        return hook

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb_img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model(tensor)

    for handle in hooks:
        handle.remove()
    return feature_maps


def compute_metrics(gt_bgr: Optional[np.ndarray], output_bgr: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if psnr is None or ssim is None or gt_bgr is None:
        return None, None
    h, w = output_bgr.shape[:2]
    gt_resized = cv2.resize(gt_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    psnr_val = float(cast(Any, psnr)(gt_resized, output_bgr, data_range=255))
    ssim_val = float(cast(Any, ssim)(gt_resized, output_bgr, data_range=255, channel_axis=2))
    return psnr_val, ssim_val


def process_image(
    input_path: Path,
    gt_path: Optional[Path],
    output_dir: Path,
    scale: int,
    tile: int,
    use_face: bool,
    use_scratch: bool,
    scratch_model_path: str,
    scratch_threshold: float,
    inpaint_radius: int,
) -> Dict[str, object]:
    module = load_cli_module()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    model_path = default_model_path(scale)
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )

    face_enhancer = None
    if use_face:
        try:
            from gfpgan import GFPGANer

            gfpgan_model_path = os.environ.get(
                "GFPGAN_MODEL_PATH",
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            )
            face_enhancer = GFPGANer(
                model_path=gfpgan_model_path,
                upscale=scale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
        except Exception:
            face_enhancer = None

    scratch_model = None
    if use_scratch:
        scratch_model = module.load_scratch_model(scratch_model_path, device)

    img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError("Failed to read input image")
    img = normalize_input_image(img)

    if use_scratch and scratch_model is not None:
        img = module.apply_scratch_repair(img, scratch_model, device, scratch_threshold, inpaint_radius)

    with torch.no_grad():
        if use_face and face_enhancer is not None:
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=scale)

    if output is None:
        raise RuntimeError("No output generated")

    output = np.clip(output.astype(np.float32), 0, 255).astype(np.uint8)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / input_path.name
    save_image(str(output_path), output)

    feature_maps = capture_feature_maps(model, img, device)
    features_dir = output_dir / "features"
    feature_files = save_feature_grids(feature_maps, input_path.stem, scale, str(features_dir))

    compare_dir = output_dir / "compare"
    pair_path, grid_path = make_comparison_images(img, output, scale, input_path.stem, str(compare_dir))

    gt_img = None
    if gt_path is not None and gt_path.exists():
        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        if gt_img is not None:
            gt_img = normalize_input_image(gt_img)

    psnr_val, ssim_val = compute_metrics(gt_img, output)
    return {
        "input": str(input_path),
        "output": str(output_path),
        "pair": pair_path,
        "grid": grid_path,
        "features": feature_files,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "scale": scale,
        "input_size": img.shape,
        "output_size": output.shape,
    }


def render_page(body: str, subtitle: str, status_text: str, badge: str) -> bytes:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Image Super-Resolution System (ESRGAN)</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700&display=swap');
    :root {{
      --bg: #f1f1f1;
      --panel: #f7f7f7;
      --panel-alt: #e6e6e6;
      --stroke: #cfcfcf;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #3b82f6;
      --accent-strong: #2563eb;
      --success: #22c55e;
      --warning: #facc15;
      --shadow: rgba(0, 0, 0, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Barlow", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    a {{ color: var(--accent-strong); text-decoration: none; }}
    .app-shell {{
      max-width: 1200px;
      margin: 24px auto;
      background: var(--panel);
      border-radius: 14px;
      box-shadow: 0 10px 30px var(--shadow);
      overflow: hidden;
      border: 1px solid var(--stroke);
    }}
    .app-header {{
      padding: 18px 24px;
      background: #ededed;
      border-bottom: 1px solid var(--stroke);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .app-header h1 {{ margin: 0; font-size: 20px; }}
    .app-header p {{ margin: 6px 0 0; color: var(--muted); font-size: 14px; }}
    .badge {{
      background: #e2f8ec;
      color: #166534;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 600;
    }}
    .app-body {{ display: grid; grid-template-columns: 260px 1fr; min-height: 560px; }}
    .sidebar {{
      background: var(--panel-alt);
      padding: 20px;
      border-right: 1px solid var(--stroke);
    }}
    .sidebar h2 {{ font-size: 18px; margin: 0 0 16px; }}
    .divider {{ height: 1px; background: var(--stroke); margin: 16px 0; }}
    .field-label {{ display: block; margin-top: 10px; font-weight: 600; font-size: 13px; }}
    .hint {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
    input[type="file"], select, input[type="number"] {{
      width: 100%;
      margin-top: 6px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--stroke);
      background: #fff;
    }}
    input[type="file"] {{ display: none; }}
    .file-name {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
    .btn {{
      display: block;
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: none;
      margin-top: 10px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      text-decoration: none;
    }}
    .btn.primary {{ background: var(--accent); color: #fff; }}
    .btn.primary:hover {{ background: var(--accent-strong); }}
    .btn.secondary {{ background: #dbeafe; color: #1e3a8a; }}
    .btn.success {{ background: var(--success); color: #fff; }}
    .btn.warning {{ background: var(--warning); color: #78350f; }}
    .btn.ghost {{ background: transparent; border: 1px solid var(--stroke); color: var(--text); }}
    .btn.disabled {{ opacity: 0.6; cursor: default; }}
    .toggle {{ display: flex; align-items: center; gap: 10px; margin-top: 10px; font-size: 13px; }}
    .toggle input {{ width: auto; }}
    .workspace {{ padding: 20px; background: #f9f9f9; }}
    .workspace-header {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-weight: 600; font-size: 14px; }}
    .panel-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; }}
    .panel {{
      background: #e3e3e3;
      border: 1px solid var(--stroke);
      border-radius: 10px;
      min-height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #555;
      font-weight: 600;
      overflow: hidden;
    }}
    .panel img {{
      max-width: 100%;
      max-height: 100%;
      border-radius: 8px;
    }}
    .panel-footer {{ display: grid; grid-template-columns: 1fr 1fr; margin-top: 12px; font-size: 12px; color: var(--muted); }}
    .metrics {{ margin-top: 16px; }}
    .metrics h3 {{ margin: 0 0 8px; font-size: 13px; }}
    .metrics p {{ margin: 4px 0; font-size: 12px; color: var(--muted); }}
    .status-bar {{
      padding: 10px 20px;
      background: #ededed;
      border-top: 1px solid var(--stroke);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      color: var(--muted);
    }}
    .compare-wrap {{
      position: relative;
      width: 100%;
      height: 320px;
      margin-top: 12px;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--stroke);
      background: #e3e3e3;
    }}
    .compare-wrap img {{
      position: absolute;
      top: 0;
      left: 0;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    .compare-overlay {{
      position: absolute;
      top: 0;
      left: 0;
      height: 100%;
      overflow: hidden;
    }}
    .compare-controls {{ margin-top: 8px; }}
    .compare-controls input {{ width: 100%; }}
    @media (max-width: 980px) {{
      .app-body {{ grid-template-columns: 1fr; }}
      .workspace-header, .panel-grid, .panel-footer {{ grid-template-columns: 1fr; }}
      .compare-wrap {{ height: 260px; }}
    }}
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <h1>Image Super-Resolution System (ESRGAN)</h1>
        <p>{html.escape(subtitle)}</p>
      </div>
      <div class="badge">{html.escape(badge)}</div>
    </header>
    {body}
    <footer class="status-bar">
      <div>{html.escape(status_text)}</div>
      <div>Elapsed: --</div>
    </footer>
  </div>
  <script>
    const fileInput = document.getElementById("image-input");
    const fileName = document.getElementById("file-name");
    if (fileInput && fileName) {{
      fileInput.addEventListener("change", () => {{
        const name = fileInput.files.length ? fileInput.files[0].name : "No file selected";
        fileName.textContent = name;
      }});
    }}
    const gtInput = document.getElementById("gt-input");
    const gtName = document.getElementById("gt-name");
    if (gtInput && gtName) {{
      gtInput.addEventListener("change", () => {{
        const name = gtInput.files.length ? gtInput.files[0].name : "No file selected";
        gtName.textContent = name;
      }});
    }}
    const slider = document.getElementById("compare-slider");
    const overlay = document.getElementById("compare-overlay");
    if (slider && overlay) {{
      slider.addEventListener("input", () => {{
        overlay.style.width = `${{slider.value}}%`;
      }});
    }}
    const compareToggle = document.getElementById("compare-toggle");
    const compareSection = document.getElementById("compare-section");
    if (compareToggle && compareSection) {{
      compareToggle.addEventListener("change", () => {{
        compareSection.style.display = compareToggle.checked ? "block" : "none";
      }});
    }}
  </script>
</body>
</html>"""
    return html_doc.encode("utf-8")


def render_form(message: str = "") -> bytes:
    status_text = "Model x4 loaded | Device: cpu"
    message_html = f"<div class=\"hint\">{html.escape(message)}</div>" if message else ""
    body = f"""
<form method="POST" enctype="multipart/form-data" action="/process">
  <div class="app-body">
    <aside class="sidebar">
      <h2>Super Resolution</h2>
      <label class="btn primary" for="image-input">Open Image</label>
      <input id="image-input" name="image" type="file" accept=".png,.jpg,.jpeg,.bmp" required />
      <div class="file-name" id="file-name">No file selected</div>
      <label class="btn ghost" for="gt-input">Load Ground Truth</label>
      <input id="gt-input" name="gt" type="file" accept=".png,.jpg,.jpeg,.bmp" />
      <div class="file-name" id="gt-name">No file selected</div>
      <div class="divider"></div>
      <label class="field-label" for="scale">Upscale Factor</label>
      <select id="scale" name="scale">
        <option value="4">x4</option>
        <option value="2">x2</option>
      </select>
      <label class="field-label" for="tile">Tile size (0 = no tiling)</label>
      <input id="tile" name="tile" type="number" value="0" min="0" />
      <div class="btn secondary disabled">Set Default Output Dir</div>
      <div class="hint">Default output: ./outputs</div>
      <label class="toggle"><input type="checkbox" name="quick" checked />Fast mode (x2, tile 256)</label>
      <label class="toggle"><input type="checkbox" name="face" />Face Enhancement</label>
      <label class="toggle"><input type="checkbox" name="scratch" />Scratch repair</label>
      <button type="submit" class="btn success">Start Restoration</button>
      <div class="btn secondary disabled">Save Comparison</div>
      <div class="btn warning disabled">Export Features</div>
      <div class="btn primary disabled">Save Result</div>
      <label class="toggle"><input type="checkbox" disabled />Compare Slider</label>
      <div class="metrics">
        <h3>Output vs GT</h3>
        <p>PSNR: --</p>
        <p>SSIM: --</p>
        <p>Load Ground Truth to calculate metrics.</p>
      </div>
      {message_html}
    </aside>
    <section class="workspace">
      <div class="workspace-header">
        <div>Original Input</div>
        <div>Super-Resolution Output</div>
      </div>
      <div class="panel-grid">
        <div class="panel">Waiting for input...</div>
        <div class="panel">Waiting for processing...</div>
      </div>
      <div class="panel-footer">
        <div>Input: -- x --</div>
        <div>Output: -- x --</div>
      </div>
    </section>
  </div>
</form>
"""
    return render_page(body, "Upload one image and run ESRGAN with the desktop pipeline.", status_text, "CPU READY")


def render_processing(job_id: str, message: str = "Processing image...") -> bytes:
    status_text = "Processing | Device: cpu"
    body = f"""
<div class="app-body">
  <aside class="sidebar">
    <h2>Super Resolution</h2>
    <div class="hint">Job ID: {job_id}</div>
    <div class="divider"></div>
    <div class="hint" id="status-msg">{html.escape(message)}</div>
    <div class="metrics">
      <h3>Status</h3>
      <p>Processing…</p>
      <p>This page will refresh when done.</p>
    </div>
  </aside>
  <section class="workspace">
    <div class="workspace-header">
      <div>Original Input</div>
      <div>Super-Resolution Output</div>
    </div>
    <div class="panel-grid">
      <div class="panel">Processing…</div>
      <div class="panel">Waiting for output…</div>
    </div>
    <div class="panel-footer">
      <div>Input: -- x --</div>
      <div>Output: -- x --</div>
    </div>
  </section>
</div>
<script>
  async function pollStatus() {{
    try {{
      const res = await fetch(`/status/{job_id}`);
      if (!res.ok) throw new Error("status failed");
      const data = await res.json();
      if (data.status === "done") {{
        window.location.href = `/result/{job_id}`;
        return;
      }}
      if (data.status === "error") {{
        const msg = document.getElementById("status-msg");
        if (msg) msg.textContent = data.error || "Processing failed.";
        return;
      }}
    }} catch (err) {{
      console.warn(err);
    }}
    setTimeout(pollStatus, 2000);
  }}
  pollStatus();
</script>
"""
    return render_page(body, "Processing in the background. This page will redirect when complete.", status_text, "WORKING")


def format_metric(value: Optional[float], digits: int) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def render_result(job: Dict[str, object]) -> bytes:
    status_text = f"Model x{job['scale']} loaded | Device: cpu"
    input_url = f"/asset/{job['id']}/input"
    output_url = f"/asset/{job['id']}/output"
    pair_url = f"/asset/{job['id']}/pair"
    grid_url = f"/asset/{job['id']}/grid"
    feature_url = f"/features/{job['id']}"
    psnr_val = format_metric(cast(Optional[float], job.get("psnr")), 2)
    ssim_val = format_metric(cast(Optional[float], job.get("ssim")), 4)
    input_size = job.get("input_size")
    output_size = job.get("output_size")
    input_text = "Input: -- x --"
    output_text = "Output: -- x --"
    if isinstance(input_size, tuple):
        input_text = f"Input: {input_size[1]} x {input_size[0]}"
    if isinstance(output_size, tuple):
        output_text = f"Output: {output_size[1]} x {output_size[0]}"

    body = f"""
<div class="app-body">
  <aside class="sidebar">
    <h2>Super Resolution</h2>
    <a class="btn primary" href="/">Open Image</a>
    <div class="divider"></div>
    <div class="field-label">Last input</div>
    <div class="hint">{html.escape(str(job['name']))}</div>
    <div class="field-label" style="margin-top:12px;">Scale</div>
    <div class="hint">x{job['scale']}</div>
    <div class="field-label" style="margin-top:12px;">Processing</div>
    <div class="hint">{html.escape(str(job['elapsed']))}</div>
    <a class="btn secondary" href="{pair_url}" download>Save Comparison</a>
    <a class="btn warning" href="{feature_url}">Export Features</a>
    <a class="btn primary" href="{output_url}" download>Save Result</a>
    <label class="toggle"><input type="checkbox" id="compare-toggle" />Compare Slider</label>
    <div class="metrics">
      <h3>Output vs GT</h3>
      <p>PSNR: {psnr_val}</p>
      <p>SSIM: {ssim_val}</p>
      <p>Grid preview: <a href="{grid_url}" download>Download grid</a></p>
    </div>
  </aside>
  <section class="workspace">
    <div class="workspace-header">
      <div>Original Input</div>
      <div>Super-Resolution Output</div>
    </div>
    <div class="panel-grid">
      <div class="panel"><img src="{input_url}" alt="Input" /></div>
      <div class="panel"><img src="{output_url}" alt="Output" /></div>
    </div>
    <div class="panel-footer">
      <div>{input_text}</div>
      <div>{output_text}</div>
    </div>
    <div id="compare-section" style="display:none;">
      <div class="compare-wrap">
        <img src="{output_url}" alt="Output" />
        <div class="compare-overlay" id="compare-overlay" style="width:50%;">
          <img src="{input_url}" alt="Input" />
        </div>
      </div>
      <div class="compare-controls">
        <input id="compare-slider" type="range" min="0" max="100" value="50" />
      </div>
    </div>
  </section>
</div>
"""
    return render_page(body, "Processing complete. Download the output or run another image.", status_text, "DONE")


def render_features(job: Dict[str, object]) -> bytes:
    files = cast(List[str], job.get("features") or [])
    items = []
    for idx, path in enumerate(files):
        items.append(f"<li><a href=\"/asset/{job['id']}/feature/{idx}\">Feature grid {idx + 1}</a></li>")
    if not items:
        items.append("<li>No feature maps captured.</li>")
    list_html = "".join(items)
    body = f"""
<div class="app-body">
  <aside class="sidebar">
    <h2>Export Features</h2>
    <a class="btn primary" href="/result/{job['id']}">Back to Result</a>
    <div class="divider"></div>
    <div class="hint">Feature grids are generated from RRDBNet conv layers.</div>
  </aside>
  <section class="workspace">
    <div class="workspace-header">
      <div>Feature Grids</div>
      <div></div>
    </div>
    <div style="margin-top:12px;">
      <ul>{list_html}</ul>
    </div>
  </section>
</div>
"""
    return render_page(body, "Download feature grids generated during processing.", "Ready", "FEATURES")


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_form())
            return
        if parsed.path.startswith("/processing/"):
            job_id = parsed.path.split("/", 2)[2]
            if job_id not in JOBS:
                self.send_error(HTTPStatus.NOT_FOUND, "Result not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_processing(job_id))
            return
        if parsed.path.startswith("/status/"):
            job_id = parsed.path.split("/", 2)[2]
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Result not found")
                return
            payload = {
                "status": job.get("status", "running"),
                "error": job.get("error"),
            }
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode("utf-8"))
            return
        if parsed.path.startswith("/result/"):
            job_id = parsed.path.split("/", 2)[2]
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Result not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_result(job))
            return
        if parsed.path.startswith("/features/"):
            job_id = parsed.path.split("/", 2)[2]
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Result not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_features(job))
            return
        if parsed.path.startswith("/asset/"):
            parts = parsed.path.split("/")
            if len(parts) < 4:
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            job_id = parts[2]
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            if parts[3] == "feature":
                if len(parts) < 5:
                    self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                    return
                try:
                    idx = int(parts[4])
                except ValueError:
                    self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                    return
                features = cast(List[str], job.get("features") or [])
                if idx < 0 or idx >= len(features):
                    self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                    return
                file_path = features[idx]
            else:
                file_path = cast(Optional[str], job.get(parts[3]))
            if not file_path or not os.path.exists(str(file_path)):
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            file_path = str(file_path)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", guess_content_type(file_path))
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        if self.path != "/process":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        form = self.parse_form()
        if form is None:
            return
        upload = form.get("image")
        if upload is None or not getattr(upload, "filename", None):
            self.send_error(HTTPStatus.BAD_REQUEST, "No file provided")
            return
        gt_upload = form.get("gt")
        scale = int(form.get("scale", "4"))
        tile = int(form.get("tile", "0"))
        use_face = "face" in form
        use_scratch = "scratch" in form
        if "quick" in form:
            scale = 2
            tile = 256
            use_face = False
            use_scratch = False
        job_id = uuid.uuid4().hex
        work_dir = Path(tempfile.mkdtemp(prefix="sr_web_"))
        lr_dir = work_dir / "lr"
        sr_dir = work_dir / "sr"
        lr_dir.mkdir(parents=True, exist_ok=True)
        sr_dir.mkdir(parents=True, exist_ok=True)
        input_path = lr_dir / Path(upload.filename).name
        with open(input_path, "wb") as f:
            f.write(upload.file.read())
        gt_path = None
        if gt_upload is not None and getattr(gt_upload, "filename", None):
            gt_path = lr_dir / f"gt_{Path(gt_upload.filename).name}"
            with open(gt_path, "wb") as f:
                f.write(gt_upload.file.read())

        scratch_model_path = os.environ.get("SCRATCH_MODEL_PATH", "")
        JOBS[job_id] = {
            "id": job_id,
            "name": input_path.name,
            "scale": scale,
            "status": "running",
        }

        def run_job():
            start_time = time.perf_counter()
            try:
                result = process_image(
                    input_path=input_path,
                    gt_path=gt_path,
                    output_dir=sr_dir,
                    scale=scale,
                    tile=tile,
                    use_face=use_face,
                    use_scratch=use_scratch,
                    scratch_model_path=scratch_model_path,
                    scratch_threshold=float(os.environ.get("SCRATCH_MASK_THRESHOLD", "0.5")),
                    inpaint_radius=int(os.environ.get("SCRATCH_INPAINT_RADIUS", "3")),
                )
            except Exception as exc:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(exc)
                return

            elapsed = time.perf_counter() - start_time
            JOBS[job_id].update({
                "elapsed": f"{elapsed:.1f}s",
                "input": result["input"],
                "output": result["output"],
                "pair": result["pair"],
                "grid": result["grid"],
                "features": result["features"],
                "psnr": result["psnr"],
                "ssim": result["ssim"],
                "input_size": result["input_size"],
                "output_size": result["output_size"],
                "status": "done",
            })

        threading.Thread(target=run_job, daemon=True).start()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(render_processing(job_id))

    def parse_form(self):
        import cgi

        try:
            form = cgi.FieldStorage(
                fp=cast(Any, self.rfile),
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type") or "",
                },
            )
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Failed to parse form: {exc}")
            return None
        data = {}
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                item = item[0]
            if item.filename:
                data[key] = item
            else:
                data[key] = item.value
        return data


def main():
    parser = argparse.ArgumentParser(description="Run ESRGAN web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()

    if not CLI_SCRIPT.exists():
        raise RuntimeError(f"Missing CLI script at {CLI_SCRIPT}")

    server = HTTPServer((args.host, args.port), WebHandler)
    print(f"Web UI running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
