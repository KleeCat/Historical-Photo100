import contextlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
import uuid
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("super_resolution_gui")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


@contextlib.contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to devnull.

    Used sparingly to silence noisy C-level warnings from libraries like
    customtkinter and OpenCV that cannot be filtered via Python's warnings
    module.  Should NOT be used around code where errors need to be visible.
    """
    if sys.stderr is None:
        yield
        return
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = os.dup(fd)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


# Silence noisy third-party warnings on import.
warnings.filterwarnings("ignore", category=UserWarning, module=r".*_distutils_hack")
try:
    from diffusers import StableDiffusionImg2ImgPipeline
except ImportError:
    StableDiffusionImg2ImgPipeline = None
with suppress_stderr():
    import customtkinter as ctk
from customtkinter.windows.widgets.appearance_mode import AppearanceModeTracker
from tkinter import filedialog, messagebox, TclError
from PIL import Image
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

# Try importing metrics libraries
try:
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    psnr = None
    ssim = None
    logger.warning("skimage not installed, metrics unavailable.")

# Try importing GFPGAN
try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None
    logger.warning("gfpgan not installed, face enhancement unavailable.")

# --- Global Theme Settings ---
with suppress_stderr():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("green")

TEXTURE_MODEL_ID = os.environ.get("TEXTURE_MODEL_ID", "").strip()
TEXTURE_PROMPT = os.environ.get(
    "TEXTURE_PROMPT",
    "restored vintage photo, realistic skin texture, fabric detail, subtle film grain",
)


def _safe_float(env_key: str, default: float) -> float:
    try:
        return float(os.environ.get(env_key, str(default)))
    except (ValueError, TypeError):
        return default


def _safe_int(env_key: str, default: int) -> int:
    try:
        return int(os.environ.get(env_key, str(default)))
    except (ValueError, TypeError):
        return default

TEXTURE_STRENGTH = _safe_float("TEXTURE_STRENGTH", 0.35)
TEXTURE_GUIDANCE = _safe_float("TEXTURE_GUIDANCE", 5.0)
TEXTURE_STEPS = _safe_int("TEXTURE_STEPS", 2)
TEXTURE_ENABLED = False
SCRATCH_MODEL_PATH = os.environ.get("SCRATCH_MODEL_PATH", "").strip()
SCRATCH_MASK_THRESHOLD = _safe_float("SCRATCH_MASK_THRESHOLD", 0.5)
SCRATCH_INPAINT_RADIUS = _safe_int("SCRATCH_INPAINT_RADIUS", 3)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_BATCH_RETRIES = 1

# --- UI Constants ---
UI_WINDOW_SIZE = "1300x900"
UI_SIDEBAR_WIDTH = 240
UI_FONT_TITLE = ("", 24, "bold")
UI_FONT_SECTION = ("", 14, "bold")
UI_FONT_LABEL = ("", 12)
UI_FONT_BUTTON_LARGE = ("", 16, "bold")
UI_FONT_BUTTON_MEDIUM = ("", 14, "bold")
UI_FONT_SECTION_HEADER = ("", 12, "bold")
UI_COLOR_PRIMARY = "#10B981"
UI_COLOR_PRIMARY_HOVER = "#059669"
UI_COLOR_DANGER = "#EF4444"
UI_COLOR_DANGER_HOVER = "#DC2626"
UI_COLOR_DANGER_MUTED = "#6B4C4A"
UI_COLOR_DANGER_MUTED_HOVER = "#7D5553"
UI_COLOR_SECTION_TEXT = ("#737373", "#737373")
UI_COLOR_CARD_BG = ("#FFFFFF", "#141414")
UI_COLOR_CARD_BORDER = ("#E5E5E5", "#262626")
UI_COLOR_BG = ("#FAFAFA", "#0A0A0A")
UI_COLOR_SECONDARY_BG = ("#F5F5F5", "#1A1A1A")
UI_COLOR_SECONDARY_HOVER = ("#EBEBEB", "#262626")
UI_COLOR_SECONDARY_TEXT = ("#404040", "#D4D4D4")
UI_COLOR_TEXT_PRIMARY = ("#171717", "#EDEDED")
UI_COLOR_TEXT_MUTED = ("#737373", "#6B6B6B")
UI_COLOR_IMAGE_BG = ("#F0F0F0", "#0F0F0F")
UI_COLOR_SWITCH_OFF = ("#D4D4D4", "#404040")
UI_COLOR_SWITCH_ON = "#10B981"


class ToolTip:
    """Simple hover tooltip for any widget."""

    def __init__(self, widget: Any, text: str) -> None:
        self.widget = widget
        self.text = text
        self._tw: Any = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def update_text(self, text: str) -> None:
        self.text = text

    def _show(self, _event: Any = None) -> None:
        if self._tw or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify="left",
            background="#1A1A1A", foreground="#EDEDED",
            relief="solid", borderwidth=1,
            font=("", 10), padx=6, pady=3,
        )
        label.pack()

    def _hide(self, _event: Any = None) -> None:
        if self._tw:
            self._tw.destroy()
            self._tw = None


class UserCancelledError(Exception):
    pass


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def write_json_file(path: str, payload: Dict[str, Any]) -> None:
    """Write a dictionary as a JSON file, creating parent directories as needed."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def timestamp_str() -> str:
    """Return a timestamp string suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_basename(path: Optional[str], fallback: str = "image") -> str:
    """Return the stem of a file path, or *fallback* if path is empty."""
    if not path:
        return fallback
    return os.path.splitext(os.path.basename(path))[0]


_SUPPORTED_SAVE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"})


def save_image(path: str, bgr_img: np.ndarray) -> str:
    """Encode and save a BGR image to disk."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_SAVE_EXTS:
        ext = ".png"
        path = path + ext
    with suppress_stderr():
        if ext in (".tiff", ".tif"):
            params = []
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 95]
        elif ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        else:
            params = []
        success, buf = cv2.imencode(ext, bgr_img, params)
    if not success:
        raise RuntimeError("Failed to encode image")
    buf.tofile(path)
    return path


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ScratchUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(128, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        d3 = self.down3(self.pool2(d2))
        b = self.bottleneck(self.pool3(d3))

        u3 = self.up3(b)
        u3 = self.dec3(torch.cat([u3, d3], dim=1))
        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, d1], dim=1))
        return self.out(u1)


def clean_state_dict(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Remove 'module.' prefix from state dict keys for DataParallel compatibility."""
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def load_scratch_model(model_path: str, device: torch.device) -> Optional[nn.Module]:
    if not model_path:
        return None
    if not os.path.exists(model_path):
        return None
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e:
        logger.warning("Failed to load scratch model from %s: %s", model_path, e)
        return None
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model = ScratchUNet()
    model.load_state_dict(clean_state_dict(state_dict), strict=False)
    model.to(device)
    model.eval()
    return model


def predict_scratch_mask(
    bgr_img: np.ndarray,
    model: Optional[nn.Module],
    device: torch.device,
    threshold: float,
) -> Optional[np.ndarray]:
    """Predict a binary scratch mask using the UNet model."""
    if model is None:
        return None
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    inp = gray.astype(np.float32) / 255.0
    tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))
    mask = pred.squeeze().detach().cpu().numpy()
    mask = (mask >= threshold).astype(np.uint8) * 255
    if mask.shape[:2] != gray.shape[:2]:
        mask = cv2.resize(
            mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return mask


def apply_scratch_repair(
    bgr_img: np.ndarray,
    model: Optional[nn.Module],
    device: torch.device,
    threshold: float,
    inpaint_radius: int,
) -> np.ndarray:
    """Detect scratches and inpaint them. Returns original if no model or no scratches."""
    if model is None:
        return bgr_img
    mask = predict_scratch_mask(bgr_img, model, device, threshold)
    if mask is None or not np.any(mask):
        return bgr_img
    return cv2.inpaint(bgr_img, mask, inpaint_radius, cv2.INPAINT_TELEA)


def blend_images(
    img_a: Optional[np.ndarray], img_b: Optional[np.ndarray], alpha: float
) -> Optional[np.ndarray]:
    """Alpha-blend two images. Returns the non-None image if one is None."""
    if img_a is None:
        return img_b
    if img_b is None:
        return img_a
    weight = float(np.clip(alpha, 0.0, 1.0))
    if weight <= 0.0:
        return img_b
    if weight >= 1.0:
        return img_a
    return cv2.addWeighted(img_a, weight, img_b, 1.0 - weight, 0)


def apply_unsharp_mask(
    bgr_img: np.ndarray,
    strength: float,
    radius: float = 1.5,
    blend_weight: float = 0.0,
) -> np.ndarray:
    """Apply edge-aware unsharp mask to the Y channel in YCrCb space."""
    weight = float(np.clip(strength, 0.0, 1.0))
    if weight <= 0.0:
        return bgr_img
    blend_amount = float(np.clip(blend_weight, 0.0, 1.0))
    if blend_amount > 0.0:
        if blend_amount >= 0.03:
            return bgr_img
        weight *= max(0.2, 1.0 - 3.0 * blend_amount)
        radius = min(radius, 0.9)
    if weight <= 0.0:
        return bgr_img
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = cv2.magnitude(grad_x, grad_y)
    max_edge = float(np.max(edge_mag))
    if max_edge > 1e-6:
        edge_zone = np.clip(edge_mag / max_edge, 0.0, 1.0)
        edge_zone = cv2.GaussianBlur(edge_zone, (0, 0), sigmaX=2.2, sigmaY=2.2)
        detail_gate = 1.0 - np.clip(edge_zone * 3.4, 0.0, 1.0)
        detail_gate = (detail_gate * detail_gate).astype(np.float32)
    else:
        detail_gate = np.ones((bgr_img.shape[0], bgr_img.shape[1]), dtype=np.float32)

    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y_channel = ycrcb[:, :, 0]
    blurred_y = cv2.GaussianBlur(y_channel, (0, 0), radius)
    detail_y = y_channel - blurred_y
    sharpened_y = y_channel + weight * detail_y * detail_gate
    ycrcb[:, :, 0] = np.clip(sharpened_y, 0, 255)
    return cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def apply_film_grain(bgr_img: np.ndarray, strength: float) -> np.ndarray:
    """Add synthetic film grain noise to an image."""
    weight = float(np.clip(strength, 0.0, 1.0))
    if weight <= 0.0:
        return bgr_img
    h, w = bgr_img.shape[:2]
    sigma = 12.0 * weight
    noise = np.random.default_rng().normal(0.0, sigma, (h, w, 1)).astype(np.float32)
    grain = bgr_img.astype(np.float32) + noise
    return np.clip(grain, 0, 255).astype(np.uint8)


def blend_with_lr(
    sr_bgr: np.ndarray, lr_bgr: np.ndarray, strength: float
) -> np.ndarray:
    """Blend SR output with upscaled LR using phase-correlation alignment."""
    weight = float(np.clip(strength, 0.0, 1.0))
    if weight <= 0.0:
        return sr_bgr
    h, w = sr_bgr.shape[:2]
    lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    sr_f = sr_bgr.astype(np.float32)

    sigma = 3.0 + 4.0 * weight
    low_sr = cv2.GaussianBlur(sr_f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    lr_aligned = lr_up
    low_lr = cv2.GaussianBlur(
        lr_up.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
    )
    if low_sr is None or low_lr is None:
        fallback = blend_images(lr_up, sr_bgr, min(weight, 0.08))
        return sr_bgr if fallback is None else fallback

    gated_weight = weight
    try:
        sr_shift_gray = cv2.cvtColor(low_sr, cv2.COLOR_BGR2GRAY)
        lr_shift_gray = cv2.cvtColor(low_lr, cv2.COLOR_BGR2GRAY)
        (shift_x, shift_y), response = cv2.phaseCorrelate(sr_shift_gray, lr_shift_gray)
        shift_norm = float(np.hypot(shift_x, shift_y))

        if response < 0.05 or shift_norm > 0.35:
            return sr_bgr
        if shift_norm > 0.20:
            gated_weight *= 0.5

        if shift_norm > 1e-3:
            transform = np.float32([[1.0, 0.0, -shift_x], [0.0, 1.0, -shift_y]])
            lr_aligned = cv2.warpAffine(
                lr_up,
                transform,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REFLECT101,
            )
            low_lr = cv2.GaussianBlur(
                lr_aligned.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
            )
    except Exception as e:
        logger.debug("LR alignment failed, skipping blend: %s", e)
        return sr_bgr
    gated_weight = float(np.clip(gated_weight, 0.0, 0.05))

    sr_gray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Scharr(sr_gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(sr_gray, cv2.CV_32F, 0, 1)
    edge_mag = cv2.magnitude(grad_x, grad_y)
    max_edge = float(np.max(edge_mag))
    if max_edge > 1e-6:
        edge_norm = (edge_mag / max_edge).astype(np.float32)
    else:
        edge_norm = np.zeros((h, w), dtype=np.float32)

    edge_ratio = float(np.mean(edge_norm > 0.18))
    if edge_ratio > 0.02:
        return sr_bgr

    edge_soft = cv2.GaussianBlur(edge_norm, (0, 0), sigmaX=4.5, sigmaY=4.5)
    if edge_soft is None or edge_soft.shape[:2] != (h, w):
        fallback = blend_images(lr_aligned, sr_bgr, min(gated_weight, 0.04))
        return sr_bgr if fallback is None else fallback

    flat_gate = 1.0 - np.clip(edge_soft * 3.2, 0.0, 1.0)
    flat_gate = flat_gate * flat_gate * flat_gate
    blend_map = (gated_weight * flat_gate)[:, :, np.newaxis]
    fused = sr_f + (low_lr - low_sr) * blend_map
    return np.clip(fused, 0, 255).astype(np.uint8)


def suppress_edge_ringing(
    sr_bgr: np.ndarray,
    lr_bgr: Optional[np.ndarray],
    strength: float = 0.5,
) -> np.ndarray:
    """Reduce halo/ringing artifacts near edges in the SR output."""
    amount = float(np.clip(strength, 0.0, 1.0))
    if amount <= 0.0:
        return sr_bgr

    h, w = sr_bgr.shape[:2]
    sr_gray = cv2.cvtColor(sr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sr_edge = np.abs(cv2.Laplacian(sr_gray, cv2.CV_32F, ksize=3))

    lr_edge = None
    if lr_bgr is not None:
        try:
            lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
            lr_gray = cv2.cvtColor(lr_up, cv2.COLOR_BGR2GRAY).astype(np.float32)
            lr_edge = np.abs(cv2.Laplacian(lr_gray, cv2.CV_32F, ksize=3))
        except Exception:
            lr_edge = None

    if lr_edge is None:
        ref_gray = cv2.GaussianBlur(sr_gray, (0, 0), sigmaX=1.4, sigmaY=1.4)
        ref_edge = np.abs(cv2.Laplacian(ref_gray, cv2.CV_32F, ksize=3))
        edge_ref = ref_edge
        overshoot = sr_edge - (1.20 * ref_edge + 2.5)
    else:
        edge_ref = lr_edge
        overshoot = sr_edge - (1.14 * lr_edge + 2.5)

    halo_mask = np.clip(overshoot / 35.0, 0.0, 1.0)
    edge_ref_norm = edge_ref / (float(np.max(edge_ref)) + 1e-6)
    edge_focus = np.clip((edge_ref_norm - 0.12) / 0.32, 0.0, 1.0)
    halo_mask = halo_mask * edge_focus
    if float(np.max(halo_mask)) <= 0.01:
        return sr_bgr

    halo_mask = cv2.GaussianBlur(halo_mask, (0, 0), sigmaX=2.0, sigmaY=2.0)
    halo_mask = (halo_mask * amount)[:, :, np.newaxis].astype(np.float32)

    smoothed = cv2.bilateralFilter(sr_bgr, d=5, sigmaColor=12, sigmaSpace=20).astype(
        np.float32
    )
    src_f = sr_bgr.astype(np.float32)
    corrected = src_f * (1.0 - halo_mask) + smoothed * halo_mask
    return np.clip(corrected, 0, 255).astype(np.uint8)


def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """Clamp a float value between min_value and max_value."""
    return max(min_value, min(float(value), max_value))


def auto_tile_size(img_h: int, img_w: int, scale: int) -> int:
    """Return a tile size for RealESRGAN based on image size and available VRAM.

    Returns 0 (no tiling) when the image is small enough, otherwise picks
    a tile size that keeps peak VRAM usage reasonable.
    """
    pixels = img_h * img_w * scale * scale
    if not torch.cuda.is_available():
        # CPU: always tile for large images to limit RAM
        if pixels > 1024 * 1024:
            return 256
        return 0
    try:
        free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
    except Exception:
        free_vram_mb = 2048  # conservative fallback
    # Rough heuristic: ~12 bytes per output pixel during inference
    estimated_mb = pixels * 12 / (1024 * 1024)
    if estimated_mb < free_vram_mb * 0.6:
        return 0  # fits comfortably
    if free_vram_mb >= 6000:
        return 400
    if free_vram_mb >= 3000:
        return 256
    return 192


def estimate_image_metrics(bgr_img: np.ndarray) -> Dict[str, float]:
    """Compute sharpness, noise, contrast, and edge density metrics."""
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
    noise_sigma = float(np.std(gray.astype(np.float32) - blur.astype(np.float32)))
    contrast = float(np.std(gray))
    edges = cv2.Canny(gray, 60, 120)
    edge_density = float(np.mean(edges > 0))
    return {
        "lap_var": lap_var,
        "noise_sigma": noise_sigma,
        "contrast": contrast,
        "edge_density": edge_density,
    }


def make_comparison_images(
    lr_bgr: np.ndarray,
    sr_bgr: np.ndarray,
    scale: int,
    base_name: str,
    out_dir: str,
) -> Tuple[str, str]:
    """Generate side-by-side and grid comparison images."""
    ensure_dir(out_dir)
    ts = timestamp_str()
    h, w = sr_bgr.shape[:2]
    lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)

    pair = np.hstack([lr_up, sr_bgr])
    pair_path = os.path.join(out_dir, f"{base_name}_x{scale}_{ts}_lr_sr.png")
    save_image(pair_path, pair)

    diff = cv2.absdiff(sr_bgr, lr_up)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_norm = cv2.normalize(diff_gray, None, 0, 255, cv2.NORM_MINMAX)
    heat = cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET)

    crop_size = max(32, min(h, w) // 3)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - crop_size // 2)
    y1 = max(0, cy - crop_size // 2)
    x1 = min(x1, w - crop_size)
    y1 = min(y1, h - crop_size)

    lr_crop = lr_up[y1 : y1 + crop_size, x1 : x1 + crop_size]
    sr_crop = sr_bgr[y1 : y1 + crop_size, x1 : x1 + crop_size]
    zoom = np.hstack([lr_crop, sr_crop])
    zoom_vis = cv2.resize(zoom, (w, h), interpolation=cv2.INTER_NEAREST)

    grid_top = np.hstack([lr_up, sr_bgr])
    grid_bottom = np.hstack([heat, zoom_vis])
    grid = np.vstack([grid_top, grid_bottom])
    grid_path = os.path.join(out_dir, f"{base_name}_x{scale}_{ts}_grid.png")
    save_image(grid_path, grid)
    return pair_path, grid_path


def tensor_to_grid_image(
    tensor: torch.Tensor, grid: int = 4, max_channels: int = 16,
) -> Optional[np.ndarray]:
    """Convert a feature tensor to a grid visualization image."""
    if not torch.is_tensor(tensor):
        return None
    t = tensor
    if t.ndim == 4:
        t = t[0]
    if t.ndim != 3 or t.shape[0] == 0:
        return None
    c = min(t.shape[0], max_channels)
    imgs = []
    for i in range(c):
        f = t[i].cpu().numpy()
        f_min = float(f.min())
        f_max = float(f.max())
        if f_max - f_min < 1e-6:
            f_norm = np.zeros_like(f, dtype=np.uint8)
        else:
            f_norm = ((f - f_min) / (f_max - f_min) * 255.0).astype(np.uint8)
        f_bgr = cv2.cvtColor(f_norm, cv2.COLOR_GRAY2BGR)
        imgs.append(f_bgr)
    if not imgs:
        return None
    while len(imgs) < grid * grid:
        imgs.append(np.zeros_like(imgs[0]))
    rows = []
    for r in range(grid):
        row = np.hstack(imgs[r * grid : (r + 1) * grid])
        rows.append(row)
    return np.vstack(rows)


def save_feature_grids(
    feature_maps: List[Tuple[str, torch.Tensor]],
    base_name: str,
    scale: int,
    out_dir: str,
) -> List[str]:
    """Save feature map grid visualizations to disk."""
    ensure_dir(out_dir)
    ts = timestamp_str()
    saved = []
    for idx, (name, tensor) in enumerate(feature_maps):
        grid_img = tensor_to_grid_image(tensor)
        if grid_img is None:
            continue
        path = os.path.join(out_dir, f"{idx:02d}_{base_name}_x{scale}_{ts}.png")
        save_image(path, grid_img)
        saved.append(path)
    return saved


class ModernApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()

        # Window setup
        self.title("Image Super-Resolution System (ESRGAN)")
        self.geometry(UI_WINDOW_SIZE)  # Increased height for new controls

        # Thread safety locks
        self._state_lock = (
            threading.Lock()
        )  # protects img_input/img_output/feature_maps
        self._model_lock = threading.Lock()  # protects model loading (upsampler/model)

        # Run-scoped callback guard: incremented at each run start so that
        # delayed after() callbacks from a previous run are silently ignored.
        self._current_run_id: int = 0
        # Persistent CTkImage reference for the output panel.  Prevents the
        # Tk image from being garbage-collected before the label finishes
        # rendering it.
        self._output_ctk_image: Optional[ctk.CTkImage] = None
        self._input_ctk_image: Optional[ctk.CTkImage] = None
        self._ui_queue: "queue.Queue[Tuple[Optional[int], int, Callable[[], None]]]" = (
            queue.Queue()
        )

        # Core variables
        self.model_folder = os.environ.get(
            "REALESRGAN_MODEL_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "realesrgan"),
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.upsampler = None
        self.model = None
        self.scale_factor = 4  # Default
        self.img_input = None
        self.img_output = None
        self.texture_pipe = None
        self.scratch_model = None
        self.face_enhancer = None  # cached GFPGANer instance
        self.face_enhancer_scale = None  # scale used when face_enhancer was created
        self._face_cascade = None  # cached CascadeClassifier for detect_faces
        self.img_gt = None
        self.gt_path = None
        self.input_path = None
        self.is_processing = False
        self.is_batch_processing = False
        self.batch_queue = []
        self.batch_index = 0
        self.batch_total = 0
        self.batch_cancelled = False
        self.batch_output_dir = None
        self.batch_run_id = None
        self.batch_errors = []
        self.batch_folder = None
        self.batch_retry_counts = {}
        self.batch_retry_limit = DEFAULT_BATCH_RETRIES
        self.batch_retry_max = ctk.IntVar(value=DEFAULT_BATCH_RETRIES)
        self.cancel_requested = False
        self.last_run_dir = None
        self.last_run_id = None
        self.face_blend = ctk.DoubleVar(value=0.65)
        self.natural_blend = ctk.DoubleVar(value=0.0)
        self.texture_boost = ctk.DoubleVar(value=0.08)
        self.film_grain = ctk.DoubleVar(value=0.0)
        self.compare_mode = ctk.BooleanVar(value=False)
        self.compare_split = ctk.DoubleVar(value=0.5)
        self.feature_maps = []
        self.hook_handles = []
        self.max_feature_maps = 6
        self.zoom_factor = 1.0
        self.view_center = [0.5, 0.5]
        self.pan_start = None
        self.compare_hold_active = False
        self.progress_target = 0.0
        self.processing_start_time = None
        self.overlay_base_text = "Waiting for processing..."
        self.resize_job = None
        self.overlay_animation_job = None
        self.success_render_jobs = []
        self._resize_seq = 0
        self._action_button_state_cache = None
        self._last_resize_sizes = {}  # Track last resize per widget to avoid loops
        self._rendering_in_progress = False  # Suppress resize events during render
        self.last_processing_durations = []  # history of elapsed times for adaptive progress
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.project_dir, "output_config.json")
        self.default_output_dir = None
        self.load_config()

        # Layout configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_ui()
        self._start_ui_dispatch_loop()

        # Start background model loading (Default load x4)
        self.status_label.configure(
            text=f"Initializing core components ({self.device})..."
        )
        self.progress_bar.set(0.5)
        threading.Thread(target=self.load_model, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self._last_win_state = "normal"
        self.after(200, self._poll_window_state)

    def _poll_window_state(self) -> None:
        try:
            current = self.state()
        except Exception:
            return
        if self._last_win_state == "iconic" and current == "normal":
            self.after(150, self._force_ctk_redraw)
        self._last_win_state = current
        self.after(200, self._poll_window_state)

    def _force_ctk_redraw(self) -> None:
        """强制重绘所有 CTk 控件和 Canvas，解决最小化恢复后黑块问题。"""
        start_time = time.time()
        try:
            AppearanceModeTracker.update_callbacks()
            self._update_all_scrollable_frames()
            self.update_idletasks()
            self.after(50, self._update_all_scrollable_frames)

            elapsed = (time.time() - start_time) * 1000
            if elapsed > 100:
                logger.warning("_force_ctk_redraw took %.1fms", elapsed)

        except Exception as e:
            logger.warning("Failed to force CTk redraw: %s", e)

    def _update_all_scrollable_frames(self) -> None:
        """递归更新所有 CTkScrollableFrame 的 Canvas 背景色和 scrollregion。"""
        def update_widget(widget, _depth=0):
            if _depth > 20:
                logger.debug("_update_all_scrollable_frames: max depth exceeded for %s", widget)
                return
            if hasattr(widget, '_parent_canvas') and hasattr(widget, '_parent_frame'):
                try:
                    fg = widget._parent_frame.cget("fg_color")
                    if fg == "transparent":
                        color_tuple = widget._parent_frame.cget("bg_color")
                    else:
                        color_tuple = fg
                    if hasattr(widget, '_apply_appearance_mode'):
                        bg_color = widget._apply_appearance_mode(color_tuple)
                    else:
                        bg_color = color_tuple[0] if isinstance(color_tuple, (list, tuple)) else color_tuple
                    widget._parent_canvas.configure(bg=bg_color)

                    bbox = widget._parent_canvas.bbox("all")
                    if bbox is not None:
                        widget._parent_canvas.configure(scrollregion=bbox)
                except (TclError, ValueError):
                    pass

            try:
                for child in widget.winfo_children():
                    update_widget(child, _depth + 1)
            except TclError:
                pass

        update_widget(self)

    def _make_card(self, parent: Any, row: int, title: str, **grid_kw: Any) -> Tuple[Any, int]:
        """Create a card frame with title inside the sidebar. Returns (card_frame, first_content_row)."""
        card = ctk.CTkFrame(
            parent, corner_radius=8,
            fg_color=UI_COLOR_CARD_BG,
            border_width=1, border_color=UI_COLOR_CARD_BORDER,
        )
        card.grid(row=row, column=0, padx=8, pady=(4, 2), sticky="ew", **grid_kw)
        card.grid_columnconfigure(0, weight=1)
        lbl = ctk.CTkLabel(
            card, text=title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=UI_COLOR_SECTION_TEXT,
        )
        lbl.grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")
        return card, 1

    def setup_ui(self) -> None:
        # === 1. Sidebar (Left) ===
        self._sidebar_outer = ctk.CTkFrame(
            self, width=UI_SIDEBAR_WIDTH + 16, corner_radius=0,
            fg_color=UI_COLOR_BG,
        )
        self._sidebar_outer.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self._sidebar_outer.grid_rowconfigure(0, weight=1)
        self._sidebar_outer.grid_columnconfigure(0, weight=1)
        self._sidebar_outer.grid_propagate(False)
        self.sidebar = ctk.CTkFrame(
            self._sidebar_outer,
            width=UI_SIDEBAR_WIDTH,
            corner_radius=0,
            fg_color=UI_COLOR_BG,
        )
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Logo / Title
        self.logo_label = ctk.CTkLabel(
            self.sidebar,
            text="Super Resolution",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=UI_COLOR_TEXT_PRIMARY,
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(16, 4))

        # === Card: Input ===
        self._card_input, cr = self._make_card(self.sidebar, 1, "Input")

        self.btn_load = ctk.CTkButton(
            self._card_input, text="Open Image", command=self.load_input_image, height=36,
            fg_color=UI_COLOR_SECONDARY_BG, hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT,
        )
        self.btn_load.grid(row=cr, column=0, padx=12, pady=(4, 6), sticky="ew")

        self.btn_gt = ctk.CTkButton(
            self._card_input,
            text="Load Ground Truth",
            command=self.load_gt_image,
            fg_color="transparent",
            hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_TEXT_PRIMARY,
            border_width=1,
            border_color=UI_COLOR_CARD_BORDER,
            height=36,
        )
        self.btn_gt.grid(row=cr + 1, column=0, padx=12, pady=(0, 10), sticky="ew")

        # === Card: Settings ===
        self._card_settings, cr = self._make_card(self.sidebar, 2, "Settings")
        ctk.CTkLabel(
            self._card_settings,
            text="Upscale Factor:",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).grid(row=cr, column=0, padx=12, pady=(6, 0), sticky="w")
        self.scale_var = ctk.StringVar(value="x4")
        self.scale_combo = ctk.CTkComboBox(
            self._card_settings,
            values=["x2", "x4"],
            variable=self.scale_var,
            command=self.change_model_scale,
        )
        self.scale_combo.grid(row=cr + 1, column=0, padx=12, pady=(6, 8))

        # Output dir: Entry + folder button in a row frame
        self._output_dir_row = ctk.CTkFrame(self._card_settings, fg_color="transparent")
        self._output_dir_row.grid(row=cr + 2, column=0, padx=12, pady=(2, 8), sticky="ew")
        self._output_dir_row.grid_columnconfigure(0, weight=1)
        self.entry_output_dir = ctk.CTkEntry(
            self._output_dir_row,
            placeholder_text="Output directory...",
            state="readonly",
            height=30,
            font=ctk.CTkFont(size=11),
        )
        self.entry_output_dir.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        if self.default_output_dir:
            self.entry_output_dir.configure(state="normal")
            self.entry_output_dir.insert(0, self.default_output_dir)
            self.entry_output_dir.configure(state="readonly")
        self.btn_output_dir = ctk.CTkButton(
            self._output_dir_row,
            text="\U0001F4C1",
            command=self.set_default_output_dir,
            width=32, height=30,
            fg_color=UI_COLOR_SECONDARY_BG,
            hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT,
            border_width=0,
        )
        self.btn_output_dir.grid(row=0, column=1)

        # Face Enhance Switch
        self.use_face_enhance = ctk.BooleanVar(value=False)
        self.switch_face = ctk.CTkSwitch(
            self._card_settings, text="Face Enhancement", variable=self.use_face_enhance,
            progress_color=UI_COLOR_SWITCH_ON,
            button_color=UI_COLOR_SWITCH_OFF,
        )
        self.switch_face.grid(row=cr + 3, column=0, padx=12, pady=4, sticky="w")

        # Scratch Repair Switch
        self.use_scratch_repair = ctk.BooleanVar(value=False)
        self.switch_scratch = ctk.CTkSwitch(
            self._card_settings, text="Scratch Repair", variable=self.use_scratch_repair,
            progress_color=UI_COLOR_SWITCH_ON,
            button_color=UI_COLOR_SWITCH_OFF,
        )
        self.switch_scratch.grid(row=cr + 4, column=0, padx=12, pady=4, sticky="w")

        self.lbl_face_blend = ctk.CTkLabel(
            self._card_settings, text=f"Face Blend: {self.face_blend.get():.2f}",
            fg_color=self._card_settings.cget("fg_color"),
        )
        self.lbl_face_blend.grid(row=cr + 5, column=0, padx=12, pady=(0, 4), sticky="w")
        self.lbl_face_blend.grid_remove()
        self.slider_face_blend = ctk.CTkSlider(
            self._card_settings,
            from_=0.0, to=1.0, number_of_steps=20,
            variable=self.face_blend,
            command=self.on_face_blend_change,
        )
        self.slider_face_blend.grid(row=cr + 6, column=0, padx=12, pady=(0, 10), sticky="ew")
        self.slider_face_blend.grid_remove()

        self.lbl_natural_blend = ctk.CTkLabel(
            self._card_settings, text=f"Natural Blend: {self.natural_blend.get():.2f}",
            fg_color=self._card_settings.cget("fg_color"),
        )
        self.lbl_natural_blend.grid(row=cr + 7, column=0, padx=12, pady=(0, 4), sticky="w")
        self.lbl_natural_blend.grid_remove()
        self.slider_natural_blend = ctk.CTkSlider(
            self._card_settings,
            from_=0.0, to=0.20, number_of_steps=10,
            variable=self.natural_blend,
            command=self.on_natural_blend_change,
        )
        self.slider_natural_blend.grid(row=cr + 8, column=0, padx=12, pady=(0, 10), sticky="ew")
        self.slider_natural_blend.grid_remove()

        self.lbl_texture_boost = ctk.CTkLabel(
            self._card_settings, text=f"Texture Boost: {self.texture_boost.get():.2f}",
            fg_color=self._card_settings.cget("fg_color"),
        )
        self.lbl_texture_boost.grid(row=cr + 9, column=0, padx=12, pady=(0, 4), sticky="w")
        self.lbl_texture_boost.grid_remove()
        self.slider_texture_boost = ctk.CTkSlider(
            self._card_settings,
            from_=0.0, to=0.35, number_of_steps=7,
            variable=self.texture_boost,
            command=self.on_texture_boost_change,
        )
        self.slider_texture_boost.grid(row=cr + 10, column=0, padx=12, pady=(0, 10), sticky="ew")
        self.slider_texture_boost.grid_remove()

        self.lbl_film_grain = ctk.CTkLabel(
            self._card_settings, text=f"Film Grain: {self.film_grain.get():.2f}",
            fg_color=self._card_settings.cget("fg_color"),
        )
        self.lbl_film_grain.grid(row=cr + 11, column=0, padx=12, pady=(0, 4), sticky="w")
        self.lbl_film_grain.grid_remove()
        self.slider_film_grain = ctk.CTkSlider(
            self._card_settings,
            from_=0.0, to=0.5, number_of_steps=10,
            variable=self.film_grain,
            command=self.on_film_grain_change,
        )
        self.slider_film_grain.grid(row=cr + 12, column=0, padx=12, pady=(0, 10), sticky="ew")
        self.slider_film_grain.grid_remove()

        self.batch_retry_frame = ctk.CTkFrame(self._card_settings, fg_color="transparent")
        self.batch_retry_frame.grid(row=cr + 13, column=0, padx=12, pady=(4, 10), sticky="ew")
        self.batch_retry_frame.grid_columnconfigure(1, weight=1)
        self.lbl_batch_retry = ctk.CTkLabel(
            self.batch_retry_frame, text="Batch Retries"
        )
        self.lbl_batch_retry.grid(row=0, column=0, sticky="w")
        self.entry_batch_retry = ctk.CTkEntry(
            self.batch_retry_frame, width=60, textvariable=self.batch_retry_max
        )
        self.entry_batch_retry.grid(row=0, column=1, sticky="e")

        # === Card: Results (metrics) ===
        self._card_results, cr = self._make_card(self.sidebar, 3, "Results")

        self.metrics_frame = ctk.CTkFrame(
            self._card_results, fg_color=self._card_results.cget("fg_color")
        )
        self.metrics_frame.grid(row=cr, column=0, padx=12, pady=(2, 6), sticky="ew")
        self.metrics_frame.grid_columnconfigure(1, weight=1)

        self.lbl_resolution_in = ctk.CTkLabel(
            self.metrics_frame, text="In: -- x --", font=ctk.CTkFont(size=11),
            fg_color=self.metrics_frame.cget("fg_color"),
        )
        self.lbl_resolution_in.grid(row=0, column=0, sticky="w")
        self.lbl_resolution_out = ctk.CTkLabel(
            self.metrics_frame, text="Out: -- x --", font=ctk.CTkFont(size=11),
            fg_color=self.metrics_frame.cget("fg_color"),
        )
        self.lbl_resolution_out.grid(row=0, column=1, sticky="e")

        self.lbl_metrics_after = ctk.CTkLabel(
            self.metrics_frame, text="Output vs GT",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=self.metrics_frame.cget("fg_color"),
        )
        self.lbl_metrics_after.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.psnr_var = ctk.StringVar(value="PSNR: --")
        self.lbl_psnr_out = ctk.CTkLabel(
            self.metrics_frame, textvariable=self.psnr_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=UI_COLOR_TEXT_PRIMARY,
            fg_color=self.metrics_frame.cget("fg_color"),
        )
        self.lbl_psnr_out.grid(row=2, column=0, sticky="w")
        self.ssim_var = ctk.StringVar(value="SSIM: --")
        self.lbl_ssim_out = ctk.CTkLabel(
            self.metrics_frame, textvariable=self.ssim_var,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=UI_COLOR_TEXT_PRIMARY,
            fg_color=self.metrics_frame.cget("fg_color"),
        )
        self.lbl_ssim_out.grid(row=2, column=1, sticky="e")
        self.lbl_gt_hint = ctk.CTkLabel(
            self.metrics_frame,
            text="Load GT for metrics",
            font=ctk.CTkFont(size=10),
            text_color=UI_COLOR_TEXT_MUTED,
            fg_color=self.metrics_frame.cget("fg_color"),
        )
        self.lbl_gt_hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))

        # Compare switch + slider (inside Results card)
        self._compare_frame = ctk.CTkFrame(
            self._card_results, fg_color=self._card_results.cget("fg_color")
        )
        self._compare_frame.grid(row=cr + 1, column=0, padx=12, pady=(0, 10), sticky="w")
        self.switch_compare = ctk.CTkSwitch(
            self._compare_frame, text="Compare",
            variable=self.compare_mode,
            command=self.on_compare_mode_toggle,
        )
        self.switch_compare.grid(row=0, column=0, padx=(0, 4))
        self.lbl_compare_split = ctk.CTkLabel(
            self._compare_frame, text="50%",
            fg_color=self._compare_frame.cget("fg_color"),
        )
        self.lbl_compare_split.grid(row=0, column=1, padx=(0, 3))
        self.slider_compare = ctk.CTkSlider(
            self._compare_frame,
            from_=0.0, to=1.0, number_of_steps=20,
            variable=self.compare_split,
            command=self.on_compare_split_change,
            width=80,
        )
        self.slider_compare.grid(row=0, column=2)
        self.slider_compare.configure(state="disabled")
        self.lbl_compare_split.grid_remove()
        self.slider_compare.grid_remove()

        # Spacer to push Actions to bottom
        self.sidebar.grid_rowconfigure(4, weight=1)

        # === Card: Actions ===
        self._card_actions, cr = self._make_card(self.sidebar, 5, "Actions")

        self.btn_run = ctk.CTkButton(
            self._card_actions,
            text="Start Restoration",
            command=self.run_processing_thread,
            fg_color=UI_COLOR_PRIMARY,
            hover_color=UI_COLOR_PRIMARY_HOVER,
            height=46,
            font=ctk.CTkFont(size=16, weight="bold"),
        )
        self.btn_run.grid(row=cr, column=0, padx=12, pady=4, sticky="ew")
        self.btn_run_default_fg = self.btn_run.cget("fg_color")
        self.btn_run_default_hover = self.btn_run.cget("hover_color")

        self.btn_batch = ctk.CTkButton(
            self._card_actions,
            text="Run Folder (Batch)",
            command=self.run_batch_folder,
            fg_color=UI_COLOR_SECONDARY_BG,
            hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT,
            border_width=0,
            height=36,
        )
        self.btn_batch.grid(row=cr + 1, column=0, padx=12, pady=4, sticky="ew")

        self.btn_cancel = ctk.CTkButton(
            self._card_actions,
            text="Cancel",
            command=self.request_cancel,
            fg_color=UI_COLOR_SECONDARY_BG,
            hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT,
            border_width=0,
            height=36,
        )
        self.btn_cancel.grid(row=cr + 2, column=0, padx=12, pady=(4, 8), sticky="ew")
        self.btn_cancel.configure(
            state="disabled", text_color_disabled=UI_COLOR_TEXT_MUTED
        )

        # === 2. Main Display Area (Right) ===
        self.display_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=(20, 5))
        self.display_frame.grid_columnconfigure(
            0, weight=1, uniform="images", minsize=320
        )
        self.display_frame.grid_columnconfigure(
            1, weight=1, uniform="images", minsize=320
        )
        self.display_frame.grid_rowconfigure(1, weight=0)
        self.display_frame.grid_rowconfigure(2, weight=1)
        self.display_frame.grid_rowconfigure(3, weight=0)

        # Headers
        ctk.CTkLabel(
            self.display_frame,
            text="Original Input",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=UI_COLOR_TEXT_PRIMARY,
        ).grid(row=0, column=0, pady=5)
        ctk.CTkLabel(
            self.display_frame,
            text="Super-Resolution Output",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=UI_COLOR_TEXT_PRIMARY,
        ).grid(row=0, column=1, pady=5)

        # Image Containers
        self.frame_input = ctk.CTkFrame(
            self.display_frame, fg_color=UI_COLOR_IMAGE_BG,
            corner_radius=8, border_width=1, border_color=UI_COLOR_CARD_BORDER,
        )
        self.frame_input.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_output = ctk.CTkFrame(
            self.display_frame, fg_color=UI_COLOR_IMAGE_BG,
            corner_radius=8, border_width=1, border_color=UI_COLOR_CARD_BORDER,
        )
        self.frame_output.grid(row=2, column=1, sticky="nsew", padx=5, pady=5)

        # Image Labels
        self.lbl_img_in = ctk.CTkLabel(
            self.frame_input,
            text="Click 'Open Image' to load\nor drag an image here",
            corner_radius=6,
            anchor="center",
            text_color=UI_COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.lbl_img_in.pack(expand=True, fill="both", padx=4, pady=4)

        self.lbl_img_out = ctk.CTkLabel(
            self.frame_output,
            text="Output will appear here\nafter processing",
            corner_radius=6,
            anchor="center",
            text_color=UI_COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=13),
        )
        self.lbl_img_out.pack(expand=True, fill="both", padx=4, pady=4)

        # Filename Labels (below images)
        self.lbl_filename_in = ctk.CTkLabel(
            self.display_frame,
            text="No file loaded",
            font=ctk.CTkFont(size=12),
            text_color=UI_COLOR_TEXT_MUTED,
            height=20,
        )
        self.lbl_filename_in.grid(row=1, column=0, pady=(0, 2), sticky="n")

        self.lbl_filename_out = ctk.CTkLabel(
            self.display_frame,
            text="",
            font=ctk.CTkFont(size=12),
            text_color=UI_COLOR_TEXT_MUTED,
            height=20,
        )
        self.lbl_filename_out.grid(row=1, column=1, pady=(0, 2), sticky="n")

        self.output_overlay = ctk.CTkFrame(
            self.frame_output, corner_radius=8, fg_color=UI_COLOR_SECONDARY_BG
        )
        self.output_overlay_label = ctk.CTkLabel(
            self.output_overlay,
            text="Waiting for processing...",
            text_color=UI_COLOR_TEXT_MUTED,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=self.output_overlay.cget("fg_color"),
        )
        self.output_overlay_label.pack(expand=True)
        self.output_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.lbl_resolution_in_display = ctk.CTkLabel(
            self.display_frame, text="Input: -- x --", font=ctk.CTkFont(size=12),
            fg_color=self.display_frame.cget("fg_color"),
        )
        self.lbl_resolution_in_display.grid(row=3, column=0, pady=(0, 5))
        self.lbl_resolution_out_display = ctk.CTkLabel(
            self.display_frame, text="Output: -- x --", font=ctk.CTkFont(size=12),
            fg_color=self.display_frame.cget("fg_color"),
        )
        self.lbl_resolution_out_display.grid(row=3, column=1, pady=(0, 5))

        self.bind_image_interactions()

        # === 2b. Results Toolbar (below images) — buttons only, evenly spaced ===
        self.results_toolbar = ctk.CTkFrame(self, corner_radius=8)
        self.results_toolbar.grid(row=1, column=1, sticky="ew", padx=20, pady=(0, 5))
        for col in range(4):
            self.results_toolbar.grid_columnconfigure(col, weight=1)

        self.btn_compare = ctk.CTkButton(
            self.results_toolbar, text="Comparison",
            command=self.save_comparison,
            fg_color=UI_COLOR_SECONDARY_BG, hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT, border_width=0, height=32,
        )
        self.btn_compare.grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        self.btn_compare.configure(
            state="disabled", text_color_disabled=UI_COLOR_TEXT_MUTED
        )

        self.btn_features = ctk.CTkButton(
            self.results_toolbar, text="Features",
            command=self.export_feature_maps,
            fg_color=UI_COLOR_SECONDARY_BG, hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT, border_width=0, height=32,
        )
        self.btn_features.grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        self.btn_features.configure(
            state="disabled", text_color_disabled=UI_COLOR_TEXT_MUTED
        )

        self.btn_open_run_dir = ctk.CTkButton(
            self.results_toolbar, text="Open Folder",
            command=self.open_last_run_folder,
            fg_color=UI_COLOR_SECONDARY_BG, hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT, border_width=0, height=32,
        )
        self.btn_open_run_dir.grid(row=0, column=2, padx=6, pady=6, sticky="ew")
        self.btn_open_run_dir.configure(
            state="disabled", text_color_disabled=UI_COLOR_TEXT_MUTED
        )

        self.btn_save = ctk.CTkButton(
            self.results_toolbar, text="Save Result",
            command=self.save_result,
            fg_color=UI_COLOR_SECONDARY_BG, hover_color=UI_COLOR_SECONDARY_HOVER,
            text_color=UI_COLOR_SECONDARY_TEXT, state="disabled", height=32,
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.btn_save.grid(row=0, column=3, padx=6, pady=6, sticky="ew")
        self.btn_save.configure(text_color_disabled=UI_COLOR_TEXT_MUTED)

        # === 3. Status Bar (Bottom) ===
        self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0, fg_color=UI_COLOR_BG)
        self.status_frame.grid(row=2, column=1, sticky="ew")

        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", padx=10, width=400, anchor="w",
                                         fg_color=UI_COLOR_BG, text_color=UI_COLOR_TEXT_MUTED)
        self.status_label.pack(side="left")

        # Determinate Progress Bar
        self.progress_bar = ctk.CTkProgressBar(
            self.status_frame, width=300, height=4, mode="determinate",
            progress_color=UI_COLOR_PRIMARY,
            fg_color=UI_COLOR_SECONDARY_BG,
            corner_radius=2,
        )
        self.progress_bar.pack(side="right", padx=20, pady=5)
        self.elapsed_label = ctk.CTkLabel(
            self.status_frame, text="Elapsed: --", padx=10,
            fg_color=UI_COLOR_BG, text_color=UI_COLOR_TEXT_MUTED,
        )
        self.elapsed_label.pack(side="right")
        self.progress_bar.set(0)

        # --- Tooltips ---
        ToolTip(self.btn_load, "Open an image file for super-resolution")
        ToolTip(self.btn_gt, "Load ground truth image for quality metrics")
        ToolTip(self.btn_run, "Start the super-resolution process")
        ToolTip(self.btn_batch, "Process all images in a folder")
        ToolTip(self.btn_cancel, "Cancel the current operation")
        ToolTip(self.btn_compare, "Save side-by-side comparison image")
        ToolTip(self.btn_features, "Export feature map visualizations")
        ToolTip(self.btn_open_run_dir, "Open the output folder in file explorer")
        ToolTip(self.btn_save, "Save the processed result to disk")

    # --- Feature Extraction Hooks ---
    def clear_feature_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def register_feature_hooks(self) -> None:
        """Register forward hooks on Conv2d layers to capture feature maps.

        IMPORTANT: Caller must hold ``_model_lock`` when invoking this method,
        because it iterates over ``self.model.named_modules()``.
        """
        self.clear_feature_hooks()
        with self._state_lock:
            self.feature_maps = []

        def make_hook(name):
            def hook(module, input, output):
                tensor = output
                if isinstance(tensor, (tuple, list)):
                    if not tensor:
                        return
                    tensor = tensor[0]
                if not torch.is_tensor(tensor):
                    return
                if tensor.ndim != 4:
                    return
                _, _, h, w = tensor.shape
                if h < 16 or w < 16 or h > 1024 or w > 1024:
                    return
                detached = (name, tensor.detach().cpu())
                with self._state_lock:
                    if len(self.feature_maps) < self.max_feature_maps:
                        self.feature_maps.append(detached)

            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.hook_handles.append(module.register_forward_hook(make_hook(name)))

    def load_config(self) -> None:
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.default_output_dir = data.get("default_output_dir")
        except Exception as e:
            logger.warning("Failed to load config from %s: %s", self.config_path, e)
            self.default_output_dir = None

    def save_config(self) -> None:
        data = {"default_output_dir": self.default_output_dir}
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save config: %s", e)

    def set_default_output_dir(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.default_output_dir = selected
            self.save_config()
            self.status_label.configure(text="Default output directory updated.")
            if hasattr(self, "entry_output_dir"):
                self.entry_output_dir.configure(state="normal")
                self.entry_output_dir.delete(0, "end")
                self.entry_output_dir.insert(0, selected)
                self.entry_output_dir.configure(state="readonly")

    def truncate_path(self, path: str, max_len: int) -> str:
        if len(path) <= max_len:
            return path
        return "..." + path[-(max_len - 3) :]

    def on_close(self) -> None:
        self._cancel_overlay_animation()
        self._cancel_success_render_jobs()
        if self.resize_job is not None:
            try:
                self.after_cancel(self.resize_job)
            except (TclError, ValueError):
                pass
            self.resize_job = None
        self.clear_feature_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.destroy()

    # --- Logic: Progress Bar Animation ---
    def auto_increment_progress(self) -> None:
        if not self.is_processing:
            return
        current_val = self.progress_bar.get()
        target_val = self.progress_target
        if current_val < target_val:
            gap = target_val - current_val
            # Smooth animation: close 20% of the remaining gap each tick
            increment = max(0.005, gap * 0.2)
            new_val = min(current_val + increment, target_val)
            self.progress_bar.set(new_val)
        elif current_val >= target_val and target_val < 0.95:
            # Progress has caught up to target but processing still running.
            # Creep asymptotically toward 0.98 so the bar never freezes.
            # Each tick closes ~1.2% of remaining gap (with min step), so
            # movement stays visible even during long upscale stages.
            remaining = 0.98 - current_val
            if remaining > 0.002:
                step = max(0.0015, remaining * 0.012)
                self.progress_bar.set(min(0.98, current_val + step))
        self.after(80, self.auto_increment_progress)

    # --- Logic: Model Loading & Switching ---
    def change_model_scale(self, choice: str) -> None:
        """Handle scale change event from combobox"""
        self.scale_factor = 2 if choice == "x2" else 4
        self.status_label.configure(text=f"Switching to {choice} model...")
        self.scale_combo.configure(state="disabled")
        # Reload model in background to avoid freezing UI
        self.progress_bar.set(0.5)
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self) -> None:
        if not self._model_lock.acquire(blocking=False):
            return  # another load is already in progress
        try:
            self.clear_feature_hooks()
            self.model = None
            self.upsampler = None
            self.face_enhancer = None  # invalidate cached face enhancer
            self.face_enhancer_scale = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # Determine which model file to use
            if self.scale_factor == 2:
                model_name = "RealESRGAN_x2plus.pth"
            else:
                model_name = "RealESRGAN_x4plus.pth"

            full_path = os.path.join(self.model_folder, model_name)

            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_name}\nPlease download it to: {self.model_folder}"
                )

            self.model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=self.scale_factor,
            )
            self.register_feature_hooks()

            self.upsampler = RealESRGANer(
                scale=self.scale_factor,
                model_path=full_path,
                model=self.model,
                tile=0,
                tile_pad=10,
                pre_pad=0,
                half=False,
                device=self.device,
            )

            self._after_for_run(
                None,
                0,
                lambda: self.status_label.configure(
                    text=f"Model x{self.scale_factor} loaded | Device: {self.device}"
                ),
            )
            self._after_for_run(None, 0, lambda: self.progress_bar.set(0))
        except Exception as e:
            err_text = str(e)
            self._after_for_run(
                None, 0, lambda: self.status_label.configure(text="Load failed")
            )
            self._after_for_run(
                None, 0, lambda msg=err_text: messagebox.showerror("Model Error", msg)
            )
        finally:
            self._after_for_run(
                None, 0, lambda: self.scale_combo.configure(state="normal")
            )
            self._model_lock.release()

    def load_input_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")]
        )
        if path:
            try:
                img = self.read_image(path)
            except Exception as exc:
                self.status_label.configure(text=f"Load failed: {exc}")
                messagebox.showerror("Error", f"Failed to load image: {exc}")
                return
            self.input_path = path
            with self._state_lock:
                self.img_input = img
            self.reset_view_state()
            status_text = (
                f"Loaded: {os.path.basename(path)} | {self.get_texture_status_text()}"
            )
            self.status_label.configure(text=status_text)
            self.lbl_filename_in.configure(text=f"Input: {os.path.basename(path)}")
            self.lbl_filename_out.configure(text="")
            with self._state_lock:
                self.img_gt = None
                self.img_output = None
                self.feature_maps = []
            self.gt_path = None
            self.render_main_images_stable()
            self.update_compare_controls()
            self.show_output_overlay("Waiting for processing...", animate=False)
            self.btn_save.configure(state="disabled")
            self.btn_compare.configure(state="disabled")
            self.btn_features.configure(state="disabled")
            self.progress_bar.set(0)
            self.progress_target = 0.0
            self.elapsed_label.configure(text="Elapsed: --")
            self.update_resolution_labels()
            self.calculate_metrics()
            self.auto_tune_parameters()

    def load_gt_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")]
        )
        if path:
            try:
                img = self.read_image(path)
            except Exception as exc:
                self.status_label.configure(text=f"Load failed: {exc}")
                messagebox.showerror("Error", f"Failed to load ground truth: {exc}")
                return
            with self._state_lock:
                self.img_gt = img
            self.gt_path = path
            self.calculate_metrics()
            messagebox.showinfo("Info", "Ground Truth loaded")

    def read_image(self, path: str) -> np.ndarray:
        if not path:
            raise ValueError("Empty path")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {os.path.basename(path)}")
        with suppress_stderr():
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                raise ValueError("Empty image file")
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to decode image")
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def prepare_display_image(self, img: np.ndarray) -> np.ndarray:
        """Normalize image to contiguous uint8 BGR for safe GUI rendering."""
        if img is None:
            raise ValueError("Empty image")
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.ndim == 3:
            ch = arr.shape[2]
            if ch == 4:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif ch == 1:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            elif ch != 3:
                arr = arr[:, :, :3]
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.dtype != np.uint8:
            arr = np.nan_to_num(arr)
            max_val = float(np.max(arr)) if arr.size else 0.0
            if max_val <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    def bind_image_interactions(self) -> None:
        widgets = (self.lbl_img_in, self.lbl_img_out)
        for widget in widgets:
            widget.bind("<MouseWheel>", self.on_zoom)
            widget.bind("<ButtonPress-3>", self.on_pan_start)
            widget.bind("<B3-Motion>", self.on_pan_move)
            widget.bind("<ButtonRelease-3>", self.on_pan_end)
            widget.bind("<Configure>", self.on_display_resize)
        self.lbl_img_out.bind("<ButtonPress-1>", self.on_compare_press)
        self.lbl_img_out.bind("<ButtonRelease-1>", self.on_compare_release)
        self.lbl_img_out.bind("<Leave>", self.on_compare_leave)
        self.bind("<ButtonRelease-1>", self.on_compare_release)

    def reset_view_state(self) -> None:
        self.zoom_factor = 1.0
        self.view_center = [0.5, 0.5]
        self.pan_start = None
        self.compare_hold_active = False

    def _cancel_after_job(self, job_id: Optional[str]) -> None:
        if job_id is not None:
            try:
                self.after_cancel(job_id)
            except (TclError, ValueError):
                pass

    def _run_debounced_render(self) -> None:
        self.resize_job = None
        self.render_main_images()

    def _run_resize_render(self, seq: int) -> None:
        if seq != self._resize_seq:
            return
        self._run_debounced_render()

    def render_main_images_stable(self) -> None:
        self._cancel_after_job(self.resize_job)
        self.resize_job = None
        self.resize_job = self.after_idle(self._run_debounced_render)

    def on_display_resize(self, event: Any) -> None:
        # Ignore resize events triggered by image updates
        if self._rendering_in_progress:
            return
        # Only trigger on actual size changes per widget
        widget_id = id(event.widget)
        new_size = (event.width, event.height)
        if self._last_resize_sizes.get(widget_id) == new_size:
            return
        if event.width < 50 or event.height < 50:
            return
        self._last_resize_sizes[widget_id] = new_size
        self._resize_seq += 1
        seq = self._resize_seq
        self._cancel_after_job(self.resize_job)
        self.resize_job = None
        self.resize_job = self.after(170, lambda s=seq: self._run_resize_render(s))

    def on_zoom(self, event: Any) -> None:
        if self.img_input is None or self.is_processing:
            return
        if event.delta > 0:
            zoom = self.zoom_factor * 1.1
        elif event.delta < 0:
            zoom = self.zoom_factor / 1.1
        else:
            return
        self.zoom_factor = float(np.clip(zoom, 1.0, 6.0))
        self.render_main_images()

    def on_pan_start(self, event: Any) -> None:
        if self.img_input is None or self.is_processing:
            return
        self.pan_start = (event.x, event.y)

    def on_pan_move(self, event: Any) -> None:
        if self.img_input is None or self.pan_start is None or self.is_processing:
            return
        dx = event.x - self.pan_start[0]
        dy = event.y - self.pan_start[1]
        self.pan_start = (event.x, event.y)

        widget_w = max(event.widget.winfo_width(), 1)
        widget_h = max(event.widget.winfo_height(), 1)
        view_w, view_h = self.calculate_view_window(self.img_input, widget_w, widget_h)
        if view_w <= 0 or view_h <= 0:
            return
        self.view_center[0] -= dx / widget_w * (view_w / self.img_input.shape[1])
        self.view_center[1] -= dy / widget_h * (view_h / self.img_input.shape[0])
        self.view_center[0] = float(np.clip(self.view_center[0], 0.0, 1.0))
        self.view_center[1] = float(np.clip(self.view_center[1], 0.0, 1.0))
        self.render_main_images()

    def on_pan_end(self, event: Any) -> None:
        self.pan_start = None

    def on_compare_press(self, event: Any) -> None:
        if self.img_input is None or self.img_output is None:
            return
        if self.is_processing:
            return
        if self.compare_mode.get():
            return
        self.compare_hold_active = True
        self.render_main_images()

    def on_compare_release(self, event: Any) -> None:
        if not self.compare_hold_active:
            return
        self.compare_hold_active = False
        self.render_main_images()

    def on_compare_leave(self, event: Any) -> None:
        if not self.compare_hold_active:
            return
        self.compare_hold_active = False
        self.render_main_images()

    def calculate_view_window(self, bgr_img: np.ndarray, widget_w: int, widget_h: int) -> Tuple[int, int]:
        h_img, w_img = bgr_img.shape[:2]
        zoom = max(self.zoom_factor, 1e-3)
        view_w = max(1, int(w_img / zoom))
        view_h = max(1, int(h_img / zoom))
        target_ratio = widget_w / widget_h if widget_h else 1.0
        view_ratio = view_w / view_h if view_h else 1.0
        if view_ratio > target_ratio:
            view_w = int(view_h * target_ratio)
        else:
            view_h = int(view_w / target_ratio)
        view_w = max(1, min(view_w, w_img))
        view_h = max(1, min(view_h, h_img))
        return view_w, view_h

    def _get_image_display_size(self, label_widget: Optional[Any] = None) -> Tuple[int, int]:
        """Return (width, height) available for an image panel."""
        if label_widget is not None and label_widget.winfo_exists():
            parent = label_widget.master
            pw = parent.winfo_width()
            ph = parent.winfo_height()
            if pw < 100 or ph < 100:
                try:
                    parent.update_idletasks()
                except Exception:
                    pass
                pw = max(parent.winfo_width(), parent.winfo_reqwidth())
                ph = max(parent.winfo_height(), parent.winfo_reqheight())
            if pw >= 100 and ph >= 100:
                return max(1, pw - 8), max(1, ph - 8)

        # Fallback before initial layout is complete.
        dw = self.display_frame.winfo_width()
        dh = self.display_frame.winfo_height()
        if dw < 200 or dh < 120:
            try:
                self.update_idletasks()
            except Exception:
                pass
            dw = max(
                self.display_frame.winfo_width(), self.display_frame.winfo_reqwidth()
            )
            dh = max(
                self.display_frame.winfo_height(), self.display_frame.winfo_reqheight()
            )
        if dw < 200 or dh < 120:
            dw = max(self.winfo_width() - UI_SIDEBAR_WIDTH - 40, 360)
            dh = max(self.winfo_height() - 120, 260)
        panel_w = max(1, dw // 2 - 24)
        panel_h = max(1, dh - 90)
        return panel_w, panel_h

    def _is_run_active(self, run_id: Optional[int]) -> bool:
        return run_id is None or run_id == self._current_run_id

    def _run_guarded_ui_callback(
        self, run_id: Optional[int], callback: Callable[[], None]
    ) -> None:
        if not self._is_run_active(run_id):
            logger.debug(
                "Skip stale UI callback run=%s current=%s", run_id, self._current_run_id
            )
            return
        try:
            callback()
        except TclError:
            logger.debug("UI callback skipped due TclError (run=%s)", run_id)
        except Exception as exc:
            logger.warning("UI callback failed run=%s: %s", run_id, exc)

    def _start_ui_dispatch_loop(self) -> None:
        self.after(15, self._drain_ui_queue)

    def _drain_ui_queue(self) -> None:
        processed = 0
        while processed < 64:
            try:
                run_id, delay_ms, callback = self._ui_queue.get_nowait()
            except queue.Empty:
                break
            if delay_ms <= 0:
                self._run_guarded_ui_callback(run_id, callback)
            else:
                self.after(
                    delay_ms,
                    lambda rid=run_id, cb=callback: self._run_guarded_ui_callback(
                        rid, cb
                    ),
                )
            processed += 1
        try:
            self.after(15, self._drain_ui_queue)
        except TclError:
            return

    def _after_for_run(
        self, run_id: Optional[int], delay_ms: int, callback: Callable[[], None]
    ) -> None:
        if threading.current_thread() is threading.main_thread():
            self.after(
                delay_ms,
                lambda rid=run_id, cb=callback: self._run_guarded_ui_callback(rid, cb),
            )
            return
        self._ui_queue.put((run_id, delay_ms, callback))

    def _image_debug_desc(self, img: Optional[np.ndarray]) -> str:
        if img is None:
            return "None"
        arr = np.asarray(img)
        return f"shape={arr.shape} dtype={arr.dtype}"

    def _overlay_mapped_state(self) -> int:
        try:
            return int(self.output_overlay.winfo_ismapped())
        except Exception:
            return -1

    def _log_output_render(
        self,
        run_id: Optional[int],
        phase: str,
        fn_name: str,
        ok: bool,
        img: Optional[np.ndarray],
    ) -> None:
        try:
            panel_w, panel_h = self._get_image_display_size(self.lbl_img_out)
        except Exception:
            panel_w, panel_h = -1, -1
        logger.debug(
            "output_render run=%s phase=%s fn=%s ok=%s img=%s panel=%sx%s overlay=%s processing=%s compare=%s hold=%s",
            run_id,
            phase,
            fn_name,
            int(ok),
            self._image_debug_desc(img),
            panel_w,
            panel_h,
            self._overlay_mapped_state(),
            int(self.is_processing),
            int(self.compare_mode.get()),
            int(self.compare_hold_active),
        )

    def _recreate_label(self, target: str = "output") -> "ctk.CTkLabel":
        """Destroy and recreate an image label to purge stale Tk image handles."""
        is_output = target == "output"
        old_label = self.lbl_img_out if is_output else self.lbl_img_in
        parent = self.frame_output if is_output else self.frame_input
        try:
            old_label.destroy()
        except Exception:
            pass
        if is_output:
            self._output_ctk_image = None
        else:
            self._input_ctk_image = None
        new_label = ctk.CTkLabel(
            parent, text="", corner_radius=6, anchor="center"
        )
        new_label.pack(expand=True, fill="both", padx=4, pady=4)
        if is_output:
            self.lbl_img_out = new_label
        else:
            self.lbl_img_in = new_label
        # Re-bind event handlers lost during widget recreation
        new_label.bind("<MouseWheel>", self.on_zoom)
        new_label.bind("<ButtonPress-3>", self.on_pan_start)
        new_label.bind("<B3-Motion>", self.on_pan_move)
        new_label.bind("<ButtonRelease-3>", self.on_pan_end)
        new_label.bind("<Configure>", self.on_display_resize)
        if is_output:
            new_label.bind("<ButtonPress-1>", self.on_compare_press)
            new_label.bind("<ButtonRelease-1>", self.on_compare_release)
            new_label.bind("<Leave>", self.on_compare_leave)
        return new_label

    def _recreate_output_label(self) -> "ctk.CTkLabel":
        return self._recreate_label("output")

    def _recreate_input_label(self) -> "ctk.CTkLabel":
        return self._recreate_label("input")

    def _get_dpi_scale(self, label_widget: Any) -> float:
        try:
            return label_widget._get_widget_scaling()
        except Exception:
            return 1.0

    def _assign_image_to_label(
        self, pil_img: Image.Image, label_widget: Any, logical_w: int, logical_h: int,
    ) -> bool:
        """Create CTkImage from PIL image and assign to label. Returns success."""
        ctk_img = ctk.CTkImage(
            light_image=pil_img, dark_image=pil_img, size=(logical_w, logical_h)
        )
        is_output_panel = label_widget is self.lbl_img_out
        if not self._set_label_image(label_widget, ctk_img):
            return False
        actual_widget = self.lbl_img_out if is_output_panel else self.lbl_img_in
        actual_widget.lift()
        return True

    def _clear_label_canvas(self, label_widget) -> None:
        """Clear image references to prevent ghosting."""
        try:
            label_widget.configure(image=None, text="")
            label_widget.image = None
        except Exception:
            pass

    def _set_label_image(self, label_widget, ctk_img) -> bool:
        target = "output" if label_widget is self.lbl_img_out else "input"
        for attempt in (1, 2):
            try:
                if attempt == 2:
                    if label_widget is self.lbl_img_out:
                        label_widget = self._recreate_output_label()
                    elif label_widget is self.lbl_img_in:
                        label_widget = self._recreate_input_label()
                # Set new image directly (single configure call to avoid flicker)
                label_widget.configure(image=ctk_img, text="")
                label_widget.image = ctk_img
                if label_widget is self.lbl_img_out:
                    self._output_ctk_image = ctk_img
                elif label_widget is self.lbl_img_in:
                    self._input_ctk_image = ctk_img
                return True
            except TclError as exc:
                logger.warning(
                    "set_label_image failed target=%s attempt=%s err=%s",
                    target,
                    attempt,
                    exc,
                )
                if attempt == 2:
                    return False
            except Exception as exc:
                logger.warning(
                    "set_label_image unexpected target=%s attempt=%s err=%s",
                    target,
                    attempt,
                    exc,
                )
                return False
        return False

    def render_zoomed_image(self, bgr_img: np.ndarray, label_widget: Any) -> bool:
        if not label_widget.winfo_exists():
            return False
        try:
            bgr_img = self.prepare_display_image(bgr_img)
        except Exception:
            return False

        render_w, render_h = self._get_image_display_size(label_widget)
        dpi_scale = self._get_dpi_scale(label_widget)
        logical_w = render_w / dpi_scale
        logical_h = render_h / dpi_scale

        h_img, w_img = bgr_img.shape[:2]

        if self.zoom_factor <= 1.0:
            scale = min(logical_w / w_img, logical_h / h_img)
            disp_w = max(1, int(w_img * scale))
            disp_h = max(1, int(h_img * scale))
            pil_w = max(1, int(disp_w * dpi_scale))
            pil_h = max(1, int(disp_h * dpi_scale))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(bgr_img, (pil_w, pil_h), interpolation=interp)
        else:
            view_w, view_h = self.calculate_view_window(
                bgr_img, int(logical_w), int(logical_h)
            )
            cx = int(self.view_center[0] * w_img)
            cy = int(self.view_center[1] * h_img)
            x1 = max(0, min(cx - view_w // 2, w_img - view_w))
            y1 = max(0, min(cy - view_h // 2, h_img - view_h))
            crop = bgr_img[y1 : y1 + view_h, x1 : x1 + view_w]
            disp_w = int(logical_w)
            disp_h = int(logical_h)
            pil_w = max(1, int(disp_w * dpi_scale))
            pil_h = max(1, int(disp_h * dpi_scale))
            scale_x = pil_w / max(view_w, 1)
            scale_y = pil_h / max(view_h, 1)
            interp = cv2.INTER_AREA if min(scale_x, scale_y) < 1.0 else cv2.INTER_CUBIC
            resized = cv2.resize(crop, (pil_w, pil_h), interpolation=interp)

        try:
            img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_rgb)
            return self._assign_image_to_label(im_pil, label_widget, disp_w, disp_h)
        except Exception as exc:
            logger.warning("render_zoomed_image failed: %s", exc)
            return False

    def render_main_images(self) -> None:
        self._rendering_in_progress = True
        try:
            self._render_main_images_impl()
        finally:
            self._rendering_in_progress = False

    def _render_main_images_impl(self) -> None:
        with self._state_lock:
            img_in = self.img_input
            img_out = self.img_output

        if img_in is None:
            try:
                self.lbl_img_in.configure(image=None, text="Waiting for input...")
                self.lbl_img_in.image = None
                self._input_ctk_image = None
            except Exception:
                return
        else:
            self.render_zoomed_image(img_in, self.lbl_img_in)

        if self.compare_mode.get() and img_in is not None and img_out is not None:
            compare_img = self.build_compare_image(
                img_in, img_out, self.compare_split.get()
            )
            rendered = self.render_zoomed_image(compare_img, self.lbl_img_out)
            self._log_output_render(
                self._current_run_id,
                "render-main-compare",
                "render_zoomed_image",
                rendered,
                compare_img,
            )
            return
        if self.compare_hold_active and img_in is not None:
            rendered = self.render_zoomed_image(img_in, self.lbl_img_out)
            self._log_output_render(
                self._current_run_id,
                "render-main-hold",
                "render_zoomed_image",
                rendered,
                img_in,
            )
            return
        if img_out is None:
            try:
                self.lbl_img_out.configure(image=None, text="")
                self.lbl_img_out.image = None
                self._output_ctk_image = None
            except Exception:
                pass
            if not self.is_processing:
                self.hide_output_overlay()
            self._log_output_render(
                self._current_run_id, "render-main", "clear-output", True, None
            )
            return
        self.hide_output_overlay()
        rendered = self.render_zoomed_image(img_out, self.lbl_img_out)
        self._log_output_render(
            self._current_run_id,
            "render-main",
            "render_zoomed_image",
            rendered,
            img_out,
        )
        if not rendered:
            try:
                self.show_image_ctk(img_out, self.lbl_img_out)
                self._log_output_render(
                    self._current_run_id, "render-main", "show_image_ctk", True, img_out
                )
            except TclError:
                self._log_output_render(
                    self._current_run_id,
                    "render-main",
                    "show_image_ctk",
                    False,
                    img_out,
                )
                return

    def _render_output_frame_once(
        self, run_id: Optional[int] = None, phase: str = "single-pass"
    ) -> bool:
        """Best-effort render for output panel (UI thread)."""
        if not self._is_run_active(run_id):
            self._log_output_render(run_id, phase, "run-guard", False, self.img_output)
            return False
        with self._state_lock:
            img_out = self.img_output
        if img_out is None:
            self._log_output_render(run_id, phase, "missing-output", False, None)
            return False
        if not self.lbl_img_out.winfo_exists():
            self._log_output_render(run_id, phase, "missing-widget", False, img_out)
            return False

        self._rendering_in_progress = True
        try:
            self.hide_output_overlay()
            rendered = self.render_zoomed_image(img_out, self.lbl_img_out)
            self._log_output_render(
                run_id, phase, "render_zoomed_image", rendered, img_out
            )
            if not rendered:
                try:
                    self.show_image_ctk(img_out, self.lbl_img_out)
                    rendered = True
                except Exception:
                    rendered = False
                self._log_output_render(
                    run_id, phase, "show_image_ctk", rendered, img_out
                )

            if rendered:
                try:
                    self.lbl_img_out.lift()
                except TclError:
                    self._log_output_render(run_id, phase, "lift", False, img_out)
                    return False
            return rendered
        finally:
            self._rendering_in_progress = False

    def refresh_output_after_success(self, run_id: Optional[int] = None) -> None:
        """Render final output frame on UI thread after processing success."""
        if not self._is_run_active(run_id):
            return
        if self.img_output is None:
            self._log_output_render(run_id, "success", "missing-output", False, None)
            return

        self._cancel_success_render_jobs()
        first_ok = self._render_output_frame_once(run_id, "success-0")
        retry_plan = [(100, "success-100")]
        if not first_ok:
            retry_plan.append((220, "success-220"))
        for delay_ms, phase in retry_plan:
            job_id = self.after(
                delay_ms,
                lambda rid=run_id, p=phase: self._render_output_frame_once(rid, p),
            )
            self.success_render_jobs.append(job_id)
        self._after_for_run(run_id, 0, self.update_compare_controls)
        self._after_for_run(run_id, 0, self.update_resolution_labels)
        self._after_for_run(run_id, 0, self.calculate_metrics)

    def force_output_refresh(self) -> None:
        self._render_output_frame_once(self._current_run_id, "force-refresh")

    def show_image_file_ctk(self, path: str, label_widget: Any) -> bool:
        """Render an image file directly via PIL into a CTkLabel."""
        if not path or not os.path.exists(path):
            return False
        try:
            with Image.open(path) as pil_img:
                pil_img = pil_img.convert("RGB")
                w_widget, h_widget = self._get_image_display_size(label_widget)
                dpi_scale = self._get_dpi_scale(label_widget)
                logical_w = w_widget / dpi_scale
                logical_h = h_widget / dpi_scale
                ratio = min(logical_w / pil_img.width, logical_h / pil_img.height)
                new_w = max(1, int(pil_img.width * ratio))
                new_h = max(1, int(pil_img.height * ratio))
                pil_w = max(1, int(new_w * dpi_scale))
                pil_h = max(1, int(new_h * dpi_scale))
                resample = getattr(Image, "Resampling", Image).LANCZOS
                resized = pil_img.resize((pil_w, pil_h), resample)
        except Exception as exc:
            logger.warning(
                "show_image_file_ctk failed (%s): %s", os.path.basename(path), exc
            )
            return False
        try:
            return self._assign_image_to_label(resized, label_widget, new_w, new_h)
        except Exception as exc:
            logger.warning(
                "show_image_file_ctk assign failed (%s): %s",
                os.path.basename(path),
                exc,
            )
            return False

    def render_output_from_file(self, path: str, run_id: Optional[int] = None) -> None:
        """Reload output from disk and repaint output panel."""
        if not self._is_run_active(run_id):
            return
        if not path:
            return
        file_rendered = self.show_image_file_ctk(path, self.lbl_img_out)
        self._log_output_render(
            run_id, "from-file", "show_image_file_ctk", file_rendered, self.img_output
        )
        if file_rendered:
            self._cancel_success_render_jobs()
            self.hide_output_overlay()
            self.update_resolution_labels()
            self.calculate_metrics()
            self._log_output_render(
                run_id, "from-file", "commit", True, self.img_output
            )
        else:
            fallback = self._render_output_frame_once(run_id, "from-file-fallback")
            if fallback:
                self.hide_output_overlay()
                self.update_resolution_labels()
                self.calculate_metrics()

    def render_output_after_completion(self, path: str, run_id: Optional[int] = None) -> None:
        """Render final output with a light path first, then file fallback."""
        if not self._is_run_active(run_id):
            return
        self._cancel_success_render_jobs()
        rendered = self._render_output_frame_once(run_id, "complete-memory")
        if rendered:
            self.hide_output_overlay()
            self.update_resolution_labels()
            self.calculate_metrics()
            return
        self.render_output_from_file(path, run_id)

    def build_compare_image(
        self, lr_bgr: np.ndarray, sr_bgr: np.ndarray, split_ratio: float,
    ) -> np.ndarray:
        if lr_bgr is None:
            return sr_bgr
        if sr_bgr is None:
            return lr_bgr
        h, w = sr_bgr.shape[:2]
        lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        split = int(w * float(np.clip(split_ratio, 0.0, 1.0)))
        combined = sr_bgr.copy()
        if split > 0:
            combined[:, :split] = lr_up[:, :split]
        if 0 < split < w:
            feather_px = max(2, min(4, int(round(w * 0.002))))
            x1 = max(0, split - feather_px)
            x2 = min(w, split + feather_px)
            band_w = x2 - x1
            if band_w > 1:
                alpha = np.linspace(0.0, 1.0, band_w, dtype=np.float32)[
                    np.newaxis, :, np.newaxis
                ]
                left = lr_up[:, x1:x2].astype(np.float32)
                right = sr_bgr[:, x1:x2].astype(np.float32)
                band = left * (1.0 - alpha) + right * alpha
                combined[:, x1:x2] = np.clip(band, 0, 255).astype(np.uint8)
        return combined

    def on_compare_mode_toggle(self) -> None:
        self.compare_hold_active = False
        self.update_compare_controls()
        self.render_main_images()

    def on_compare_split_change(self, value: float) -> None:
        percent = int(round(float(value) * 100))
        self.lbl_compare_split.configure(text=f"{percent}%")
        if self.compare_mode.get():
            self.render_main_images()

    def update_compare_controls(self) -> None:
        compare_enabled = self.compare_mode.get()
        if compare_enabled:
            self.lbl_compare_split.grid()
            self.slider_compare.grid()
        else:
            self.lbl_compare_split.grid_remove()
            self.slider_compare.grid_remove()
        if compare_enabled and not self.is_processing:
            self.slider_compare.configure(state="normal")
        else:
            self.slider_compare.configure(state="disabled")

    def _cancel_overlay_animation(self) -> None:
        self._cancel_after_job(self.overlay_animation_job)
        self.overlay_animation_job = None

    def _cancel_success_render_jobs(self) -> None:
        if not self.success_render_jobs:
            return
        for job_id in self.success_render_jobs:
            self._cancel_after_job(job_id)
        self.success_render_jobs = []

    def show_output_overlay(self, text: str, animate: bool = False) -> None:
        self._cancel_overlay_animation()
        self.overlay_base_text = text
        self.overlay_dot_count = 0
        self.output_overlay_label.configure(text=text)
        self.output_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.output_overlay.lift()
        logger.debug(
            "output_overlay show run=%s text=%s mapped=%s",
            self._current_run_id,
            text,
            self._overlay_mapped_state(),
        )
        if animate:
            self.animate_output_overlay()

    def hide_output_overlay(self) -> None:
        self._cancel_overlay_animation()
        try:
            self.output_overlay.place_forget()
            self.output_overlay.lower()
            logger.debug(
                "output_overlay hide run=%s mapped=%s",
                self._current_run_id,
                self._overlay_mapped_state(),
            )
        except TclError:
            return

    def animate_output_overlay(self) -> None:
        if not self.is_processing:
            self.overlay_animation_job = None
            return
        dots = getattr(self, "overlay_dot_count", 0)
        dots = (dots + 1) % 4
        self.overlay_dot_count = dots
        text = f"{self.overlay_base_text}{'.' * dots}"
        self.output_overlay_label.configure(text=text)
        self.overlay_animation_job = self.after(500, self.animate_output_overlay)

    def report_progress(
        self, value: float, status_text: Optional[str] = None,
        overlay_text: Optional[str] = None, run_id: Optional[int] = None,
    ) -> None:
        def update():
            if not self.is_processing:
                return
            self.progress_target = max(self.progress_target, value)
            # Immediately jump the bar to at least the previous target
            # so it never appears stuck behind the actual stage.
            current = self.progress_bar.get()
            if current < value - 0.05:
                self.progress_bar.set(value - 0.05)
            if status_text:
                self.status_label.configure(text=status_text)
            if overlay_text:
                self.overlay_base_text = overlay_text
                self.overlay_dot_count = 0
                self.output_overlay_label.configure(text=overlay_text)

        self._after_for_run(run_id, 0, update)

    def start_elapsed_timer(self) -> None:
        self.processing_start_time = time.perf_counter()
        self.update_elapsed_time()

    def update_elapsed_time(self) -> None:
        if not self.is_processing or self.processing_start_time is None:
            return
        elapsed = time.perf_counter() - self.processing_start_time
        self.elapsed_label.configure(text=f"Elapsed: {elapsed:.1f}s")
        self.after(200, self.update_elapsed_time)

    def set_run_button_processing(self, processing: bool) -> None:
        if processing:
            self.btn_run.configure(
                state="disabled",
                text="Processing...",
                fg_color=UI_COLOR_SECONDARY_BG,
                hover_color=UI_COLOR_SECONDARY_BG,
            )
        else:
            self.btn_run.configure(
                state="normal",
                text="Start Restoration",
                fg_color=self.btn_run_default_fg,
                hover_color=self.btn_run_default_hover,
            )

    def set_run_button_batch(self, processing: bool) -> None:
        if processing:
            self.btn_run.configure(
                state="disabled",
                text="Batch Running...",
                fg_color=UI_COLOR_SECONDARY_BG,
                hover_color=UI_COLOR_SECONDARY_BG,
            )
        else:
            self.set_run_button_processing(False)

    def update_action_buttons(self) -> None:
        if self.is_processing or self.is_batch_processing:
            cancel_state = "normal"
            batch_state = "disabled"
        else:
            cancel_state = "disabled"
            batch_state = "normal"

        target_state = (cancel_state, batch_state)
        if self._action_button_state_cache == target_state:
            return

        if self.btn_cancel.cget("state") != cancel_state:
            self.btn_cancel.configure(state=cancel_state)
            if cancel_state == "normal":
                self.btn_cancel.configure(
                    fg_color=UI_COLOR_DANGER, hover_color=UI_COLOR_DANGER_HOVER,
                    border_width=0, text_color="#FFFFFF",
                )
            else:
                self.btn_cancel.configure(
                    fg_color=UI_COLOR_SECONDARY_BG, border_width=0,
                    text_color=UI_COLOR_SECONDARY_TEXT,
                )
        if self.btn_batch.cget("state") != batch_state:
            self.btn_batch.configure(state=batch_state)
        self._action_button_state_cache = target_state

    def get_batch_retry_limit(self) -> int:
        try:
            value = int(self.batch_retry_max.get())
        except (TypeError, ValueError):
            value = DEFAULT_BATCH_RETRIES
        return max(0, min(value, 5))

    def request_cancel(self) -> None:
        if not self.is_processing and not self.is_batch_processing:
            return
        self.cancel_requested = True
        if self.is_batch_processing:
            self.batch_cancelled = True
        self.status_label.configure(text="Cancelling after current step...")
        self.show_output_overlay("Cancelling", animate=True)

    def show_image_ctk(self, cv_img: np.ndarray, label_widget: Any) -> None:
        try:
            cv_img = self.prepare_display_image(cv_img)
            img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            im_pil = Image.fromarray(img_rgb)
        except Exception as e:
            logger.warning("show_image_ctk preparation failed: %s", e)
            return

        w_widget, h_widget = self._get_image_display_size(label_widget)
        dpi_scale = self._get_dpi_scale(label_widget)
        logical_w = w_widget / dpi_scale
        logical_h = h_widget / dpi_scale

        w_img, h_img = im_pil.size
        ratio = min(logical_w / w_img, logical_h / h_img)

        new_w = max(1, int(w_img * ratio))
        new_h = max(1, int(h_img * ratio))
        pil_w = max(1, int(new_w * dpi_scale))
        pil_h = max(1, int(new_h * dpi_scale))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        im_pil = im_pil.resize((pil_w, pil_h), resample)

        if not self._assign_image_to_label(im_pil, label_widget, new_w, new_h):
            raise TclError("set_label_image failed")

    def show_image_preview(
        self, title: str, bgr_img: np.ndarray, info_text: str,
        save_text: str, on_save: Callable[[Any], None],
    ) -> None:
        preview = ctk.CTkToplevel(self)
        preview.title(title)
        preview.geometry("900x900")
        preview.lift()
        preview.attributes("-topmost", True)
        preview.after(100, lambda: preview.attributes("-topmost", False))
        preview.focus_force()

        frame = ctk.CTkFrame(preview)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        lbl = ctk.CTkLabel(frame, text="")
        lbl.pack(expand=True, fill="both", padx=10, pady=10)

        img_rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        max_w, max_h = 800, 800
        w_img, h_img = im_pil.size
        ratio = min(max_w / w_img, max_h / h_img, 1.0)
        new_w = int(w_img * ratio)
        new_h = int(h_img * ratio)
        ctk_img = ctk.CTkImage(
            light_image=im_pil, dark_image=im_pil, size=(new_w, new_h)
        )
        lbl.configure(image=ctk_img)
        lbl.image = ctk_img

        if info_text:
            info_label = ctk.CTkLabel(preview, text=info_text)
            info_label.pack(pady=(0, 10))

        btn_save = ctk.CTkButton(
            preview, text=save_text, command=lambda: on_save(preview)
        )
        btn_save.pack(pady=(0, 10))

    def update_slider_label(self, label_widget: Any, prefix: str, value: float) -> None:
        label_widget.configure(text=f"{prefix}: {float(value):.2f}")

    def on_face_blend_change(self, value: float) -> None:
        self.update_slider_label(self.lbl_face_blend, "Face Blend", value)

    def on_natural_blend_change(self, value: float) -> None:
        self.update_slider_label(self.lbl_natural_blend, "Natural Blend", value)

    def on_texture_boost_change(self, value: float) -> None:
        self.update_slider_label(self.lbl_texture_boost, "Texture Boost", value)

    def on_film_grain_change(self, value: float) -> None:
        self.update_slider_label(self.lbl_film_grain, "Film Grain", value)

    def get_texture_status_text(self) -> str:
        if TEXTURE_ENABLED and TEXTURE_MODEL_ID:
            return "Texture gen: on"
        return "Texture gen: off (disabled)"

    def detect_faces(self, gray_img: np.ndarray) -> bool:
        """Detect whether faces exist in the grayscale image."""
        if self._face_cascade is None:
            cascade_path = os.path.join(
                cv2.data.haarcascades, "haarcascade_frontalface_default.xml"
            )
            if not os.path.exists(cascade_path):
                return False
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            if self._face_cascade.empty():
                self._face_cascade = None
                return False
        faces = self._face_cascade.detectMultiScale(
            gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)
        )
        return len(faces) > 0

    def auto_tune_parameters(self) -> None:
        if self.img_input is None:
            return
        try:
            metrics = estimate_image_metrics(self.img_input)
            sharpness_norm = clamp_value((metrics["lap_var"] - 20.0) / 380.0, 0.0, 1.0)
            noise_norm = clamp_value((metrics["noise_sigma"] - 2.0) / 18.0, 0.0, 1.0)
            contrast_norm = clamp_value((metrics["contrast"] - 20.0) / 60.0, 0.0, 1.0)
            edge_norm = clamp_value((metrics["edge_density"] - 0.02) / 0.08, 0.0, 1.0)

            face_blend = clamp_value(
                0.6 + sharpness_norm * 0.2 - noise_norm * 0.1, 0.4, 0.9
            )
            natural_blend = clamp_value(
                0.03 + noise_norm * 0.07 + (1.0 - contrast_norm) * 0.05, 0.0, 0.12
            )
            texture_boost = clamp_value(
                0.10
                + (1.0 - sharpness_norm) * 0.18
                + edge_norm * 0.06
                - noise_norm * 0.08,
                0.0,
                0.35,
            )
            if edge_norm > 0.25:
                natural_blend = 0.0
                texture_boost = 0.0
            film_grain = clamp_value(
                0.03 + (1.0 - edge_norm) * 0.12 + (1.0 - contrast_norm) * 0.08, 0.0, 0.5
            )

            self.face_blend.set(face_blend)
            self.natural_blend.set(natural_blend)
            self.texture_boost.set(texture_boost)
            self.film_grain.set(film_grain)
            self.on_face_blend_change(face_blend)
            self.on_natural_blend_change(natural_blend)
            self.on_texture_boost_change(texture_boost)
            self.on_film_grain_change(film_grain)

            self.status_label.configure(text="Auto tuned")
        except Exception as e:
            self.status_label.configure(text=f"Auto tune failed: {e}")

    def get_texture_pipeline(self) -> Optional["StableDiffusionImg2ImgPipeline"]:
        if not TEXTURE_ENABLED or not TEXTURE_MODEL_ID:
            return None
        if StableDiffusionImg2ImgPipeline is None:
            raise RuntimeError(
                "diffusers not installed. Run: pip install diffusers transformers accelerate"
            )
        if self.texture_pipe is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.texture_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                TEXTURE_MODEL_ID, torch_dtype=dtype
            )
            self.texture_pipe.to(self.device)
            if self.device.type == "cuda":
                self.texture_pipe.enable_attention_slicing()
        return self.texture_pipe

    def apply_texture_generation(self, bgr_img: np.ndarray) -> np.ndarray:
        if self.cancel_requested:
            raise UserCancelledError("Cancelled")
        pipe = self.get_texture_pipeline()
        if pipe is None:
            return bgr_img
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        init_image = Image.fromarray(rgb_img)
        result = pipe(
            prompt=TEXTURE_PROMPT,
            image=init_image,
            strength=TEXTURE_STRENGTH,
            guidance_scale=TEXTURE_GUIDANCE,
            num_inference_steps=TEXTURE_STEPS,
        ).images[0]
        if self.cancel_requested:
            raise UserCancelledError("Cancelled")
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    def run_processing_thread(self) -> None:
        if self.is_processing:
            return
        if self.is_batch_processing:
            messagebox.showinfo("Info", "Batch processing is already running.")
            return
        if self.img_input is None:
            self.status_label.configure(text="Load an input image first.")
            messagebox.showinfo("Info", "Please load an input image first.")
            return
        if self.upsampler is None:
            self.status_label.configure(text="Model not ready. Loading...")
            threading.Thread(target=self.load_model, daemon=True).start()
            messagebox.showwarning(
                "Model Loading", "Model is still loading. Please try again shortly."
            )
            return
        # Start each single-image run from full view to avoid stale zoom crops.
        self.reset_view_state()
        self.cancel_requested = False
        self.start_processing(batch_mode=False, on_complete=None)

    def start_processing(self, batch_mode: bool = False, on_complete: Optional[Callable[[bool, bool, Optional[str]], None]] = None) -> None:
        self._current_run_id += 1
        ui_run_id = self._current_run_id
        self._cancel_success_render_jobs()
        run_options = {
            "face_enhance": bool(self.use_face_enhance.get()),
            "face_blend": float(self.face_blend.get()),
            "natural_blend": float(self.natural_blend.get()),
            "texture_boost": float(self.texture_boost.get()),
            "film_grain": float(self.film_grain.get()),
            "scratch_repair": bool(self.use_scratch_repair.get()),
        }
        self.is_processing = True
        self.progress_bar.set(0)
        self.progress_target = 0.05
        self.feature_maps = []
        self.compare_hold_active = False
        self.start_elapsed_timer()
        if batch_mode:
            self.set_run_button_batch(True)
        else:
            self.set_run_button_processing(True)
        self.update_action_buttons()
        self.update_compare_controls()
        self.show_output_overlay("Processing", animate=True)
        if batch_mode:
            status_text = f"Batch {self.batch_index + 1}/{self.batch_total}: {os.path.basename(self.input_path)}"
        else:
            status_text = f"Restoring image (x{self.scale_factor})..."
        self.status_label.configure(text=status_text)
        self.auto_increment_progress()
        logger.info(
            "processing_start ui_run=%s batch=%s input=%s",
            ui_run_id,
            int(batch_mode),
            os.path.basename(self.input_path or ""),
        )

        threading.Thread(
            target=self.process_image,
            args=(batch_mode, on_complete, ui_run_id, run_options),
            daemon=True,
        ).start()

    def _stage_scratch_repair(self, input_img: np.ndarray, run_meta: Dict[str, Any]) -> np.ndarray:
        """Scratch repair pre-processing stage. Returns processed input_img."""
        stage_start = time.perf_counter()
        if self.scratch_model is None:
            self.scratch_model = load_scratch_model(
                SCRATCH_MODEL_PATH, self.device
            )
        if self.scratch_model is not None:
            input_img = apply_scratch_repair(
                input_img,
                self.scratch_model,
                self.device,
                SCRATCH_MASK_THRESHOLD,
                SCRATCH_INPAINT_RADIUS,
            )
        run_meta["timing"]["scratch"] = round(
            time.perf_counter() - stage_start, 3
        )
        run_meta["scratch_repair"] = True
        return input_img

    def _stage_face_enhance(
        self,
        input_img: np.ndarray,
        sr_base: np.ndarray,
        face_blend: float,
        ui_run_id: Optional[int],
        run_meta: Dict[str, Any],
    ) -> Tuple[np.ndarray, bool]:
        """Face enhancement stage. Returns (output, used_face_enhance)."""
        stage_start = time.perf_counter()
        output = sr_base
        used = False
        try:
            if GFPGANer is None:
                raise ImportError("gfpgan not installed")
            _gfpgan_default = os.path.join(
                os.path.expanduser("~"), ".cache", "gfpgan", "GFPGANv1.3.pth"
            )
            gfpgan_path = os.environ.get("GFPGAN_MODEL_PATH", _gfpgan_default)
            if not os.path.isfile(gfpgan_path):
                raise FileNotFoundError(
                    f"GFPGAN model not found: {gfpgan_path}\n"
                    "Please download GFPGANv1.3.pth manually or set GFPGAN_MODEL_PATH."
                )
            if (
                self.face_enhancer is None
                or self.face_enhancer_scale != self.scale_factor
            ):
                self.face_enhancer = GFPGANer(
                    model_path=gfpgan_path,
                    upscale=self.scale_factor,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=self.upsampler,
                )
                self.face_enhancer_scale = self.scale_factor
            _, _, face_output = self.face_enhancer.enhance(
                input_img,
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            if face_output is not None:
                output = blend_images(face_output, sr_base, face_blend)
                used = True
        except Exception as e:
            logger.warning(
                "Face enhance failed (%s), switching to standard mode.", e
            )
            self._after_for_run(
                ui_run_id,
                0,
                lambda: self.status_label.configure(
                    text="Face enhancement unavailable, switching to standard mode..."
                ),
            )
        run_meta["timing"]["face"] = round(time.perf_counter() - stage_start, 3)
        return output, used

    def _stage_blend_and_texture(
        self, output: np.ndarray, input_img: np.ndarray, natural_blend: float,
        texture_boost: float, film_grain: float,
        ui_run_id: Optional[int], run_meta: Dict[str, Any],
        set_stage: Callable[[str, float, str, str], None],
    ) -> np.ndarray:
        """Blend, texture, and finalize stage. Returns final output."""
        stage_start = time.perf_counter()
        if natural_blend <= 0.0 and texture_boost <= 0.0:
            output = suppress_edge_ringing(output, input_img, strength=0.25)
        else:
            dehalo_strength = 0.62 if natural_blend <= 0.02 else 0.50
            output = suppress_edge_ringing(
                output, input_img, strength=dehalo_strength
            )
            output = blend_with_lr(output, input_img, natural_blend)
            output = apply_unsharp_mask(
                output, texture_boost, blend_weight=natural_blend
            )
        run_meta["timing"]["blend"] = round(time.perf_counter() - stage_start, 3)

        texture_ran = False
        if TEXTURE_ENABLED and TEXTURE_MODEL_ID:
            try:
                if self.cancel_requested:
                    raise UserCancelledError("Cancelled")
                set_stage(
                    "texture", 0.88,
                    "Generating texture details...", "Texture refinement",
                )
                stage_start = time.perf_counter()
                output = self.apply_texture_generation(output)
                texture_ran = True
                run_meta["timing"]["texture"] = round(
                    time.perf_counter() - stage_start, 3
                )
            except UserCancelledError:
                raise
            except Exception as e:
                texture_msg = f"Texture generation skipped: {e}"
                self._after_for_run(
                    ui_run_id,
                    0,
                    lambda msg=texture_msg: self.status_label.configure(text=msg),
                )
        else:
            run_meta["timing"]["texture"] = 0.0

        if texture_ran:
            output = suppress_edge_ringing(output, input_img, strength=0.16)
        if natural_blend <= 0.0 and texture_boost <= 0.0:
            film_grain = min(film_grain, 0.02)

        if self.cancel_requested:
            raise UserCancelledError("Cancelled")
        set_stage("finalize", 0.95, "Finalizing output...", "Finalizing")
        stage_start = time.perf_counter()
        output = apply_film_grain(output, film_grain)
        run_meta["timing"]["finalize"] = round(time.perf_counter() - stage_start, 3)
        return output

    def process_image(
        self,
        batch_mode: bool = False,
        on_complete: Optional[Callable[[bool, bool, Optional[str]], None]] = None,
        ui_run_id: Optional[int] = None,
        run_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        success = False
        cancelled = False
        error_message = None
        success_status_text: Optional[str] = None
        if ui_run_id is None:
            ui_run_id = self._current_run_id
        opts = run_options or {}
        face_enhance = bool(opts.get("face_enhance", False))
        face_blend = float(np.clip(opts.get("face_blend", 0.65), 0.4, 0.9))
        natural_blend = float(np.clip(opts.get("natural_blend", 0.0), 0.0, 0.12))
        texture_boost = float(np.clip(opts.get("texture_boost", 0.08), 0.0, 0.35))
        film_grain = float(opts.get("film_grain", 0.0))
        scratch_repair = bool(opts.get("scratch_repair", False))
        run_root = self.batch_output_dir if batch_mode else None
        run_record_id, run_dir = self.start_run_record(
            run_root=run_root, ui_run_id=ui_run_id
        )
        base_name = safe_basename(self.input_path)
        run_output_path = os.path.join(run_dir, f"{base_name}_x{self.scale_factor}.png")
        run_meta = {
            "run_id": run_record_id,
            "timestamp": timestamp_str(),
            "input_path": os.path.basename(self.input_path) if self.input_path else None,
            "run_dir": os.path.basename(run_dir) if run_dir else None,
            "output_path": os.path.basename(run_output_path) if run_output_path else None,
            "scale_factor": self.scale_factor,
            "device": str(self.device),
            "face_enhance": face_enhance,
            "face_blend": face_blend,
            "natural_blend": natural_blend,
            "texture_boost": texture_boost,
            "film_grain": film_grain,
            "texture_enabled": bool(TEXTURE_ENABLED),
            "texture_model": TEXTURE_MODEL_ID,
            "texture_prompt": TEXTURE_PROMPT,
            "texture_strength": TEXTURE_STRENGTH,
            "texture_guidance": TEXTURE_GUIDANCE,
            "texture_steps": TEXTURE_STEPS,
            "scratch_model": os.path.basename(SCRATCH_MODEL_PATH) if SCRATCH_MODEL_PATH else None,
            "gt_path": os.path.basename(self.gt_path) if self.gt_path else None,
            "model_path": (
                "RealESRGAN_x2plus.pth"
                if self.scale_factor == 2
                else "RealESRGAN_x4plus.pth"
            ),
            "gfpgan_model_path": os.path.basename(
                os.environ.get("GFPGAN_MODEL_PATH", "GFPGANv1.3.pth")
            ),
            "env": {
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
            },
            "timing": {},
            "metrics": {},
            "output_files": {
                "input_snapshot": None,
                "output_snapshot": os.path.basename(run_output_path) if run_output_path else None,
                "comparison": None,
                "grid": None,
                "features": [],
            },
        }

        def set_stage(stage, value, status_text, overlay_text):
            run_meta["stage"] = stage
            run_meta["stage_at"] = timestamp_str()
            self.write_run_log(run_dir, run_meta)
            ui_status = status_text
            if batch_mode and status_text:
                ui_status = (
                    f"Batch {self.batch_index + 1}/{self.batch_total}: {os.path.basename(self.input_path)}"
                    f" | {status_text}"
                )
            self.report_progress(value, ui_status, overlay_text, run_id=ui_run_id)

        def check_cancel():
            if self.cancel_requested:
                raise UserCancelledError("Cancelled")

        if not batch_mode:
            self._after_for_run(
                ui_run_id,
                0,
                lambda: self.status_label.configure(
                    text=f"Run {run_record_id} started..."
                ),
            )
        try:
            check_cancel()
            output = None
            used_face_enhance = False

            # Take a snapshot of the input so the worker never mutates self.img_input.
            with self._state_lock:
                input_img = self.img_input.copy()

            # Scratch repair pre-processing (before upscale)
            if scratch_repair:
                check_cancel()
                set_stage("scratch", 0.05, "Repairing scratches...", "Scratch repair")
                input_img = self._stage_scratch_repair(input_img, run_meta)

            set_stage(
                "upscale",
                0.10,
                f"Upscaling image (x{self.scale_factor})...",
                "Upscaling",
            )
            stage_start = time.perf_counter()
            h_in, w_in = input_img.shape[:2]
            tile = auto_tile_size(h_in, w_in, self.scale_factor)
            with self._model_lock:
                self.upsampler.tile = tile
                upsampler_ref = self.upsampler
                model_ref = self.model  # prevent GC during enhance
            run_meta["tile_size"] = tile
            sr_base, _ = upsampler_ref.enhance(input_img, outscale=self.scale_factor)
            del model_ref  # release reference after enhance completes
            run_meta["timing"]["upscale"] = round(time.perf_counter() - stage_start, 3)
            output = sr_base
            check_cancel()
            set_stage(
                "refine", 0.65, "Upscale complete. Refining details...", "Refining"
            )

            if face_enhance:
                check_cancel()
                set_stage(
                    "face", 0.70, "Applying face enhancement...", "Face enhancement"
                )
                output, used_face_enhance = self._stage_face_enhance(
                    input_img, sr_base, face_blend, ui_run_id, run_meta
                )

            check_cancel()
            set_stage("blend", 0.80, "Blending fine details...", "Blending")
            output = self._stage_blend_and_texture(
                output, input_img, natural_blend, texture_boost, film_grain,
                ui_run_id, run_meta, set_stage,
            )

            check_cancel()
            with self._state_lock:
                self.img_output = output
            success = True
            save_image(run_output_path, output)
            try:
                # Reload from disk to ensure a stable uint8 BGR buffer for GUI.
                stable_output = self.read_image(run_output_path)
                with self._state_lock:
                    self.img_output = stable_output
            except Exception as exc:
                logger.warning("Reload output snapshot for GUI failed: %s", exc)

            # Update output filename label
            output_filename = f"{base_name}_x{self.scale_factor}.png"
            self._after_for_run(
                ui_run_id,
                0,
                lambda: self.lbl_filename_out.configure(
                    text=f"Output: {output_filename}"
                ),
            )

            run_input_path = os.path.join(run_dir, f"{base_name}_input.png")
            save_image(run_input_path, input_img)
            run_meta["input_snapshot"] = run_input_path
            run_meta["output_files"]["input_snapshot"] = run_input_path
            run_meta["output_files"]["output_snapshot"] = run_output_path
            with self._state_lock:
                run_meta["output_files"]["features"] = [
                    name for name, _ in self.feature_maps
                ]
            run_meta["stage"] = "complete"
            run_meta["stage_at"] = timestamp_str()

            self.compare_hold_active = False
            if face_enhance and not used_face_enhance:
                success_status_text = f"Done (x{self.scale_factor} Standard Mode)"
            else:
                success_status_text = f"Done (x{self.scale_factor})"

        except Exception as e:
            if isinstance(e, UserCancelledError):
                cancelled = True
                error_message = "Cancelled"
                self._after_for_run(
                    ui_run_id, 0, lambda: self.status_label.configure(text="Cancelled")
                )
                self._after_for_run(
                    ui_run_id,
                    0,
                    lambda: self.show_output_overlay("Cancelled", animate=False),
                )
                run_meta["error"] = "Cancelled"
                run_meta["stage"] = "cancelled"
            else:
                error_message = str(e)
                error_dialog_text = f"Processing failed: {error_message}"
                if not batch_mode:
                    self._after_for_run(
                        ui_run_id,
                        0,
                        lambda msg=error_dialog_text: messagebox.showerror(
                            "Error", msg
                        ),
                    )
                self._after_for_run(
                    ui_run_id,
                    0,
                    lambda: self.show_output_overlay(
                        "Processing failed", animate=False
                    ),
                )
                run_meta["error"] = str(e)
                run_meta["stage"] = "error"
            run_meta["stage_at"] = timestamp_str()
            self.write_run_log(run_dir, run_meta)
        finally:
            self.is_processing = False
            elapsed = None
            if self.processing_start_time is not None:
                elapsed = time.perf_counter() - self.processing_start_time
            if elapsed is not None:
                run_meta["elapsed_sec"] = round(elapsed, 3)
                run_meta["timing"]["total"] = round(elapsed, 3)
                # Record for adaptive progress estimation (keep last 10)
                self.last_processing_durations.append(elapsed)
                if len(self.last_processing_durations) > 10:
                    self.last_processing_durations.pop(0)
            if input_img is not None:
                h, w = input_img.shape[:2]
                run_meta["input_size"] = [int(w), int(h)]
            # Snapshot shared state under lock to avoid races with UI thread
            with self._state_lock:
                img_out_snap = self.img_output
                img_gt_snap = self.img_gt
            if img_out_snap is not None:
                h, w = img_out_snap.shape[:2]
                run_meta["output_size"] = [int(w), int(h)]
            if (
                psnr is not None
                and ssim is not None
                and img_gt_snap is not None
                and img_out_snap is not None
            ):
                h, w = img_out_snap.shape[:2]
                img_gt_out = cv2.resize(img_gt_snap, (w, h))
                run_meta["metrics"]["psnr"] = float(
                    psnr(img_gt_out, img_out_snap, data_range=255)
                )
                run_meta["metrics"]["ssim"] = float(
                    ssim(img_gt_out, img_out_snap, data_range=255, channel_axis=2)
                )
            self.write_run_log(run_dir, run_meta)
            self.progress_target = 1.0

            def apply_success_controls_state() -> None:
                self.reset_view_state()
                if success_status_text:
                    self.status_label.configure(text=success_status_text)
                self.btn_save.configure(state="normal")
                self.btn_compare.configure(state="normal")
                with self._state_lock:
                    has_features = bool(self.feature_maps)
                if has_features:
                    self.btn_features.configure(state="normal")
                self.set_run_button_processing(False)
                self.update_action_buttons()
                self.update_compare_controls()

            def apply_finalize_ui_state() -> None:
                self.progress_bar.set(1.0)
                if batch_mode:
                    if self.is_batch_processing:
                        self.set_run_button_batch(True)
                    else:
                        self.set_run_button_processing(False)
                    self.update_action_buttons()
                    self.update_compare_controls()
                elif not success:
                    self.set_run_button_processing(False)
                    self.update_action_buttons()
                    self.update_compare_controls()
                if elapsed is not None:
                    self.elapsed_label.configure(text=f"Elapsed: {elapsed:.1f}s")
                if success:
                    self.calculate_metrics()

            self._after_for_run(ui_run_id, 0, apply_finalize_ui_state)
            if success:
                self._cancel_success_render_jobs()
                self._after_for_run(
                    ui_run_id,
                    60,
                    lambda p=run_output_path, rid=ui_run_id: (
                        self.render_output_after_completion(p, rid)
                    ),
                )
                if not batch_mode:
                    self._after_for_run(ui_run_id, 190, apply_success_controls_state)
            if batch_mode and on_complete is not None:
                self._after_for_run(
                    ui_run_id,
                    0,
                    lambda: on_complete(success, cancelled, error_message),
                )

            # Release intermediate GPU tensors to prevent VRAM buildup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def update_resolution_labels(self) -> None:
        if self.img_input is None:
            input_text = "In: -- x --"
            input_display = "Input: -- x --"
        else:
            h, w = self.img_input.shape[:2]
            input_text = f"In: {w} x {h}"
            input_display = f"Input: {w} x {h}"
        self.lbl_resolution_in.configure(text=input_text)
        self.lbl_resolution_in_display.configure(text=input_display)
        if self.img_output is None:
            output_text = "Out: -- x --"
            output_display = "Output: -- x --"
        else:
            h, w = self.img_output.shape[:2]
            output_text = f"Out: {w} x {h}"
            output_display = f"Output: {w} x {h}"
        self.lbl_resolution_out.configure(text=output_text)
        self.lbl_resolution_out_display.configure(text=output_display)

    def set_metric_labels(
        self,
        psnr_label: Any,
        ssim_label: Any,
        psnr_value: Optional[float],
        ssim_value: Optional[float],
    ) -> None:
        if psnr_value is None or ssim_value is None:
            neutral = UI_COLOR_TEXT_MUTED
            self.psnr_var.set("PSNR: --")
            self.ssim_var.set("SSIM: --")
            psnr_label.configure(text_color=neutral)
            ssim_label.configure(text_color=neutral)
            return
        self.psnr_var.set(f"PSNR: {psnr_value:.2f} dB")
        self.ssim_var.set(f"SSIM: {ssim_value:.4f}")
        psnr_label.configure(text_color=UI_COLOR_PRIMARY)
        ssim_label.configure(text_color=UI_COLOR_PRIMARY)

    def calculate_metrics(self) -> None:
        with self._state_lock:
            img_out = self.img_output
            img_gt = self.img_gt
        if psnr is None or ssim is None or img_gt is None or img_out is None:
            self.set_metric_labels(self.lbl_psnr_out, self.lbl_ssim_out, None, None)
            self.lbl_gt_hint.configure(text="Load Ground Truth to calculate metrics.")
            return

        self.lbl_gt_hint.configure(text="")

        h, w = img_out.shape[:2]
        img_gt_out = cv2.resize(img_gt, (w, h))
        s_psnr_out = psnr(img_gt_out, img_out, data_range=255)
        s_ssim_out = ssim(img_gt_out, img_out, data_range=255, channel_axis=2)

        self.set_metric_labels(
            self.lbl_psnr_out, self.lbl_ssim_out, s_psnr_out, s_ssim_out
        )

    def start_run_record(self, run_root: Optional[str] = None, ui_run_id: Optional[int] = None) -> Tuple[str, str]:
        run_id = uuid.uuid4().hex[:8]
        base_name = safe_basename(self.input_path)
        run_root = run_root or self.get_output_dir("")
        run_dir = os.path.join(run_root, f"{timestamp_str()}_{base_name}_{run_id}")
        ensure_dir(run_dir)
        self.last_run_dir = run_dir
        self.last_run_id = run_id
        self._after_for_run(
            ui_run_id, 0, lambda: self.btn_open_run_dir.configure(state="normal")
        )
        return run_id, run_dir

    def run_batch_folder(self) -> None:
        if self.is_processing or self.is_batch_processing:
            messagebox.showinfo("Info", "Processing is already running.")
            return
        folder = filedialog.askdirectory()
        if not folder:
            return
        if not os.path.isdir(folder):
            messagebox.showerror("Error", "Selected folder is invalid.")
            return
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(IMAGE_EXTS)
        ]
        files.sort()
        if not files:
            messagebox.showinfo("Info", "No supported images found in the folder.")
            return
        self.batch_queue = files
        self.batch_total = len(files)
        self.batch_index = 0
        self.batch_errors = []
        self.batch_cancelled = False
        self.cancel_requested = False
        self.is_batch_processing = True
        self.batch_folder = folder
        self.batch_retry_counts = {}
        self.batch_retry_limit = self.get_batch_retry_limit()
        self.batch_run_id = uuid.uuid4().hex[:8]
        batch_subdir = os.path.join("batch", timestamp_str())
        self.batch_output_dir = self.get_output_dir(batch_subdir)
        queue_path = os.path.join(self.batch_output_dir, "batch_queue.json")
        write_json_file(
            queue_path,
            {
                "batch_run_id": self.batch_run_id,
                "timestamp": timestamp_str(),
                "folder": folder,
                "total": self.batch_total,
                "files": [os.path.basename(path) for path in files],
            },
        )
        messagebox.showinfo(
            "Batch", f"Found {self.batch_total} images.\nQueue saved to: {queue_path}"
        )
        self.set_run_button_batch(True)
        self.update_action_buttons()
        # Keep output area visible during batch processing.
        self.hide_output_overlay()
        self.start_next_batch_item()

    def start_next_batch_item(self) -> None:
        if self.cancel_requested or self.batch_index >= self.batch_total:
            self.finish_batch()
            return
        path = self.batch_queue[self.batch_index]
        try:
            img = self.read_image(path)
        except Exception as exc:
            retry_count = self.batch_retry_counts.get(path, 0)
            if retry_count < self.batch_retry_limit:
                self.batch_retry_counts[path] = retry_count + 1
                self.status_label.configure(
                    text=f"Batch {self.batch_index + 1}/{self.batch_total}: "
                    f"retry {retry_count + 1} for {os.path.basename(path)}"
                )
                self.after(500, self.start_next_batch_item)  # delay before retry
                return
            self.batch_errors.append(
                {
                    "path": os.path.basename(path),
                    "error": str(exc),
                    "retries": retry_count,
                }
            )
            self.batch_index += 1
            self.after(0, self.start_next_batch_item)
            return
        self.input_path = path
        self.lbl_filename_in.configure(text=f"Input: {os.path.basename(path)}")
        with self._state_lock:
            self.img_input = img
            self.img_output = None
            self.feature_maps = []
        self.lbl_filename_out.configure(text="")
        self.compare_hold_active = False
        self.reset_view_state()
        self.render_main_images()
        self.update_resolution_labels()
        self.calculate_metrics()
        self.start_processing(batch_mode=True, on_complete=self.on_batch_item_complete)

    def on_batch_item_complete(
        self, success: bool, cancelled: bool, error_message: Optional[str],
    ) -> None:
        if cancelled:
            self.batch_cancelled = True
        if not success and error_message:
            self.batch_errors.append(
                {
                    "path": os.path.basename(self.input_path or ""),
                    "error": error_message,
                    "retries": self.batch_retry_counts.get(self.input_path, 0),
                }
            )
        self.batch_index += 1
        if cancelled or self.cancel_requested:
            self.finish_batch()
        else:
            # Small delay so the just-finished output has time to paint
            # before advancing to the next input item.
            self.after(120, self.start_next_batch_item)

    def finish_batch(self) -> None:
        total = self.batch_total
        done = min(self.batch_index, total)
        error_count = len(self.batch_errors)
        self.is_batch_processing = False
        self.cancel_requested = False
        self.set_run_button_processing(False)
        self.update_action_buttons()
        self.hide_output_overlay()
        error_report_path = None
        if error_count:
            error_report_path = os.path.join(
                self.batch_output_dir or "", "batch_errors.json"
            )
            write_json_file(
                error_report_path,
                {
                    "batch_run_id": self.batch_run_id,
                    "timestamp": timestamp_str(),
                    "total": total,
                    "processed": done,
                    "max_retries": self.batch_retry_limit,
                    "errors": self.batch_errors,
                },
            )
        if self.batch_cancelled:
            status = "Batch cancelled"
        else:
            status = "Batch completed"
        self.status_label.configure(text=f"{status}. {done}/{total} processed.")
        if error_count:
            if error_report_path:
                message = (
                    f"{status}. Completed {done}/{total}. Errors: {error_count}.\n"
                    f"Saved: {error_report_path}"
                )
            else:
                message = f"{status}. Completed {done}/{total}. Errors: {error_count}."
        else:
            message = f"{status}. Completed {done}/{total}."
        messagebox.showinfo("Batch", message)

    def write_run_log(self, run_dir: str, payload: Dict[str, Any]) -> str:
        log_path = os.path.join(run_dir, "run_log.json")
        write_json_file(log_path, payload)
        return log_path

    def open_last_run_folder(self) -> None:
        if not self.last_run_dir or not os.path.exists(self.last_run_dir):
            messagebox.showinfo("Info", "No run folder available yet.")
            return
        # Resolve to absolute real path to prevent path traversal
        resolved = os.path.realpath(self.last_run_dir)
        if not os.path.isdir(resolved):
            messagebox.showerror("Error", "Target is not a valid directory.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(resolved)
            elif sys.platform == "darwin":
                subprocess.run(["open", resolved], check=False)
            else:
                subprocess.run(["xdg-open", resolved], check=False)
        except Exception as exc:
            messagebox.showerror("Error", f"Open folder failed: {exc}")

    def get_output_dir(self, subdir: str, prompt: bool = False) -> str:
        selected = filedialog.askdirectory() if prompt else ""
        if selected:
            base_dir = selected
        elif self.default_output_dir:
            base_dir = self.default_output_dir
        else:
            base_dir = os.path.join(self.project_dir, "outputs")
        out_dir = os.path.join(base_dir, subdir)
        ensure_dir(out_dir)
        return out_dir

    def save_comparison(self) -> None:
        with self._state_lock:
            img_in = self.img_input
            img_out = self.img_output
        if img_in is None or img_out is None:
            return
        base_name = safe_basename(self.input_path)
        try:
            lr_h, lr_w = img_in.shape[:2]
            sr_h, sr_w = img_out.shape[:2]
            lr_up = cv2.resize(img_in, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
            preview = np.hstack([lr_up, img_out])
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed: {e}")
            return

        def on_save(preview_window):
            out_dir = self.get_output_dir(
                f"compare/{base_name}_{timestamp_str()}"
            )
            try:
                pair_path, grid_path = make_comparison_images(
                    img_in, img_out, self.scale_factor, base_name, out_dir
                )
                messagebox.showinfo(
                    "Saved", f"Comparison images saved:\n{pair_path}\n{grid_path}"
                )
                preview_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Save comparison failed: {e}")

        info_text = "Preview shows LR (upscaled) | SR"
        self.show_image_preview(
            "Comparison Preview", preview, info_text, "Save Comparison", on_save
        )

    def export_feature_maps(self) -> None:
        with self._state_lock:
            feature_maps_snap = list(self.feature_maps)
        if not feature_maps_snap:
            messagebox.showinfo("Info", "No feature maps captured.")
            return
        base_name = safe_basename(self.input_path)
        grids = []
        for name, tensor in feature_maps_snap:
            grid_img = tensor_to_grid_image(tensor)
            if grid_img is not None:
                grids.append(grid_img)
        if not grids:
            messagebox.showinfo("Info", "No feature grids generated.")
            return

        def on_save(preview_window):
            try:
                out_dir = self.get_output_dir(
                    f"features/{base_name}_{timestamp_str()}"
                )
                saved = save_feature_grids(
                    feature_maps_snap, base_name, self.scale_factor, out_dir
                )
                if saved:
                    messagebox.showinfo("Saved", f"Feature grids saved: {len(saved)}")
                else:
                    messagebox.showinfo(
                        "Info", "No feature grids saved (empty tensors)."
                    )
                preview_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Export features failed: {e}")

        info_text = f"Captured feature maps: {len(grids)}"
        self.show_image_preview(
            "Feature Preview", grids[0], info_text, "Save All", on_save
        )

    def save_result(self) -> None:
        with self._state_lock:
            img_snapshot = self.img_output
        if img_snapshot is None:
            return

        def on_save(preview_window):
            path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[
                    ("PNG", "*.png"),
                    ("JPG", "*.jpg"),
                    ("WebP", "*.webp"),
                    ("TIFF", "*.tiff"),
                    ("BMP", "*.bmp"),
                ],
            )
            if path:
                save_image(path, img_snapshot)
                messagebox.showinfo("Saved", "Image saved successfully")
                preview_window.destroy()

        self.show_image_preview(
            "Result Preview", img_snapshot, None, "Save As", on_save
        )


if __name__ == "__main__":
    app = ModernApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        logger.info("Application closed by user.")
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
