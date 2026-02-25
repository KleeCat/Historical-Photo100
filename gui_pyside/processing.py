"""图像处理逻辑：超分、划痕修复、融合、锐化、纹理。

所有函数均为纯图像处理，不依赖任何 GUI 框架。
"""
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from .utils import ensure_dir, save_image, timestamp_str

logger = logging.getLogger(__name__)

# --- 环境变量配置 ---

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

TEXTURE_MODEL_ID = os.environ.get("TEXTURE_MODEL_ID", "").strip()
TEXTURE_PROMPT = os.environ.get(
    "TEXTURE_PROMPT",
    "restored vintage photo, realistic skin texture, fabric detail, subtle film grain",
)
TEXTURE_STRENGTH = _safe_float("TEXTURE_STRENGTH", 0.35)
TEXTURE_GUIDANCE = _safe_float("TEXTURE_GUIDANCE", 5.0)
TEXTURE_STEPS = _safe_int("TEXTURE_STEPS", 2)
TEXTURE_ENABLED = False
SCRATCH_MODEL_PATH = os.environ.get("SCRATCH_MODEL_PATH", "").strip()
SCRATCH_MASK_THRESHOLD = _safe_float("SCRATCH_MASK_THRESHOLD", 0.5)
SCRATCH_INPAINT_RADIUS = _safe_int("SCRATCH_INPAINT_RADIUS", 3)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
DEFAULT_BATCH_RETRIES = 1


# --- 异常 ---

class UserCancelledError(Exception):
    pass


# --- 划痕修复模型 ---

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


# --- 划痕修复函数 ---

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


# --- 图像增强函数 ---

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
                lr_up, transform, (w, h),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT101,
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


# --- 工具函数 ---

def clamp_value(value: float, min_value: float, max_value: float) -> float:
    """Clamp a float value between min_value and max_value."""
    return max(min_value, min(float(value), max_value))


def auto_tile_size(img_h: int, img_w: int, scale: int) -> int:
    """Return a tile size for RealESRGAN based on image size and available VRAM."""
    pixels = img_h * img_w * scale * scale
    if not torch.cuda.is_available():
        if pixels > 1024 * 1024:
            return 256
        return 0
    try:
        free_vram_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
    except Exception:
        free_vram_mb = 2048
    estimated_mb = pixels * 12 / (1024 * 1024)
    if estimated_mb < free_vram_mb * 0.6:
        return 0
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


# --- 对比和特征图 ---

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
