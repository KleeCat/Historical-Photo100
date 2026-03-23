"""Generate a single-case paper comparison figure for super-resolution methods."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency at runtime
    Image = None
    ImageDraw = None
    ImageFont = None

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - optional dependency at runtime
    torch = None
    nn = None

try:
    from skimage.metrics import structural_similarity as _structural_similarity
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    _structural_similarity = None

try:
    from RRDBNet_arch import RRDBNet as _ESRGANRRDBNet
except Exception:  # pragma: no cover - import depends on runtime path
    _ESRGANRRDBNet = None

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet as _RealESRGANRRDBNet
except Exception:  # pragma: no cover - optional dependency at runtime
    _RealESRGANRRDBNet = None

try:
    from realesrgan import RealESRGANer
except Exception:  # pragma: no cover - optional dependency at runtime
    RealESRGANer = None


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LR_PATH = PROJECT_ROOT / "LR" / "0844x4.png"
DEFAULT_HR_PATH = PROJECT_ROOT / "HR" / "0844.png"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "paper_compare"
MODEL_URLS = {
    "srcnn": "https://www.dropbox.com/s/pd5b2ketm0oamhj/srcnn_x4.pth?dl=1",
    "esrgan": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth"
    ),
    "realesrgan": (
        "https://github.com/xinntao/Real-ESRGAN/releases/download/"
        "v0.1.0/RealESRGAN_x4plus.pth"
    ),
}


def bicubic_upscale_to_size(
    image_bgr: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Resize a BGR image to (height, width) with bicubic interpolation."""
    target_h, target_w = target_size
    return cv2.resize(image_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def center_crop_to_match(
    image_bgr: np.ndarray,
    target_size: tuple[int, int],
) -> np.ndarray:
    """Center-crop a BGR image to (height, width)."""
    target_h, target_w = target_size
    h, w = image_bgr.shape[:2]
    if target_h > h or target_w > w:
        raise ValueError("target_size must not exceed input size for center crop")

    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    return image_bgr[top : top + target_h, left : left + target_w].copy()


def format_metric_lines(psnr: float, ssim: float) -> list[str]:
    """Format PSNR and SSIM values for the figure caption area."""
    return [
        f"PSNR: {psnr:.2f} dB",
        f"SSIM: {ssim:.4f}",
    ]


def format_figure_title(name: str) -> str:
    """Format figure titles for paper-style display."""
    return name.replace(" x4", " ×4")


def validate_inputs(
    lr_path: str | Path,
    hr_path: str | Path,
) -> tuple[Path, Path]:
    """Validate that the LR/HR pair exists and return resolved paths."""
    lr = Path(lr_path).expanduser().resolve()
    hr = Path(hr_path).expanduser().resolve()
    if not lr.is_file():
        raise FileNotFoundError(f"LR image not found: {lr}")
    if not hr.is_file():
        raise FileNotFoundError(f"HR image not found: {hr}")
    return lr, hr


def compute_psnr_ssim(
    image_bgr: np.ndarray,
    reference_bgr: np.ndarray,
) -> dict[str, float]:
    """Compute PSNR/SSIM on two already aligned BGR uint8 images."""
    if image_bgr.shape != reference_bgr.shape:
        raise ValueError("image_bgr and reference_bgr must share the same shape")
    if _structural_similarity is None:
        raise RuntimeError("scikit-image is required to calculate SSIM")

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)

    diff = reference_gray.astype(np.float64) - image_gray.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0.0:
        psnr = float("inf")
    else:
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))

    ssim = float(_structural_similarity(reference_gray, image_gray, data_range=255))
    return {
        "psnr": round(psnr, 2),
        "ssim": round(ssim, 4),
    }


def build_method_result(
    name: str,
    image_bgr: np.ndarray,
    reference_bgr: np.ndarray,
) -> dict[str, Any]:
    """Create a uniform result payload for one super-resolution method."""
    metrics = compute_psnr_ssim(image_bgr, reference_bgr)
    return {
        "name": name,
        "image": image_bgr,
        "psnr": metrics["psnr"],
        "ssim": metrics["ssim"],
    }


def align_to_reference(
    image_bgr: np.ndarray,
    reference_bgr: np.ndarray,
) -> np.ndarray:
    """Align an output image to the reference size with explicit rules."""
    target_size = reference_bgr.shape[:2]
    if image_bgr.shape[:2] == target_size:
        return image_bgr.copy()

    image_h, image_w = image_bgr.shape[:2]
    target_h, target_w = target_size
    if image_h >= target_h and image_w >= target_w:
        return center_crop_to_match(image_bgr, target_size)
    return bicubic_upscale_to_size(image_bgr, target_size)


def build_output_file_map(output_dir: str | Path) -> dict[str, Path]:
    """Build the fixed output artifact path map for the paper figure."""
    base_dir = Path(output_dir)
    return {
        "input": base_dir / "input.png",
        "bicubic": base_dir / "bicubic_x4.png",
        "srcnn": base_dir / "srcnn_x4.png",
        "esrgan": base_dir / "esrgan_x4.png",
        "realesrgan": base_dir / "realesrgan_x4.png",
        "metrics_json": base_dir / "metrics.json",
        "metrics_txt": base_dir / "metrics.txt",
        "comparison": base_dir / "comparison_with_metrics.png",
    }


def build_model_file_map(model_dir: str | Path) -> dict[str, Path]:
    """Build the fixed model path map for the paper figure."""
    base_dir = Path(model_dir)
    return {
        "srcnn": base_dir / "srcnn" / "srcnn_x4.pth",
        "esrgan": base_dir / "esrgan" / "RRDB_ESRGAN_x4.pth",
        "realesrgan": base_dir / "realesrgan" / "RealESRGAN_x4plus.pth",
    }


def make_input_display_image(
    lr_bgr: np.ndarray,
    reference_bgr: np.ndarray,
) -> np.ndarray:
    """Upscale the original LR input only for visual comparison in the paper figure."""
    target_h, target_w = reference_bgr.shape[:2]
    return cv2.resize(
        lr_bgr,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it."""
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def infer_case_id(lr_path: Path, hr_path: Path) -> str:
    """Infer a case id for output folder naming."""
    return hr_path.stem or lr_path.stem.replace("x4", "")


def load_image_bgr(image_path: str | Path) -> np.ndarray:
    """Read a BGR image from disk."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    return image


def save_image_bgr(image_path: str | Path, image_bgr: np.ndarray) -> Path:
    """Write a BGR image to disk."""
    output_path = Path(image_path)
    ensure_dir(output_path.parent)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError(f"Failed to save image: {output_path}")
    return output_path


def download_model_file(model_path: str | Path, model_url: str) -> Path:
    """Download a model to the configured D-drive project directory."""
    target_path = Path(model_path)
    if target_path.is_file():
        return target_path

    ensure_dir(target_path.parent)
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(model_url, tmp_path)
        tmp_path.replace(target_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return target_path


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError("PyTorch is required for SRCNN/ESRGAN/Real-ESRGAN inference")


def _require_pillow() -> None:
    if Image is None or ImageDraw is None or ImageFont is None:
        raise ImportError("Pillow is required to render comparison_with_metrics.png")


def resolve_device(device_name: str):
    """Resolve auto/cpu/cuda to a torch device."""
    _require_torch()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    return torch.device(device_name)


def _torch_load_state_dict(model_path: Path, device) -> dict[str, Any]:
    """Load a torch checkpoint and normalize its top-level structure."""
    _require_torch()
    try:
        state = torch.load(str(model_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(model_path), map_location=device)

    if isinstance(state, dict):
        for key in ("params_ema", "params", "state_dict", "model_state_dict"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break

    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format: {model_path}")

    return {
        str(key).removeprefix("module."): value
        for key, value in state.items()
    }


def normalize_esrgan_state_dict_keys(
    state_dict: dict[str, Any],
) -> dict[str, Any]:
    """Convert old BasicSR ESRGAN key names to the local RRDBNet_arch layout."""
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        mapped_key = key
        if key.startswith("model.0."):
            mapped_key = key.replace("model.0.", "conv_first.", 1)
        elif key.startswith("model.1.sub."):
            mapped_key = key.replace("model.1.sub.", "", 1)
            mapped_key = mapped_key.replace(".conv1.0.", ".conv1.")
            mapped_key = mapped_key.replace(".conv2.0.", ".conv2.")
            mapped_key = mapped_key.replace(".conv3.0.", ".conv3.")
            mapped_key = mapped_key.replace(".conv4.0.", ".conv4.")
            mapped_key = mapped_key.replace(".conv5.0.", ".conv5.")
            if mapped_key.startswith("23."):
                mapped_key = mapped_key.replace("23.", "trunk_conv.", 1)
            else:
                mapped_key = f"RRDB_trunk.{mapped_key}"
        elif key.startswith("model.3."):
            mapped_key = key.replace("model.3.", "upconv1.", 1)
        elif key.startswith("model.6."):
            mapped_key = key.replace("model.6.", "upconv2.", 1)
        elif key.startswith("model.8."):
            mapped_key = key.replace("model.8.", "HRconv.", 1)
        elif key.startswith("model.10."):
            mapped_key = key.replace("model.10.", "conv_last.", 1)
        normalized[mapped_key] = value
    return normalized


def create_srcnn_model():
    """Build the minimal classic SRCNN network."""
    _require_torch()

    class _SRCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
            self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
            self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            return self.conv3(x)

    return _SRCNN()


def load_srcnn_model(model_path: Path, device):
    """Load a pretrained SRCNN x4 model."""
    model = create_srcnn_model().to(device)
    state_dict = _torch_load_state_dict(model_path, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def load_esrgan_model(model_path: Path, device):
    """Load the ESRGAN RRDBNet x4 model."""
    _require_torch()
    if _ESRGANRRDBNet is None:
        raise ImportError("Local RRDBNet_arch.py is required for ESRGAN inference")
    model = _ESRGANRRDBNet(3, 3, 64, 23, gc=32).to(device)
    state_dict = _torch_load_state_dict(model_path, device)
    state_dict = normalize_esrgan_state_dict_keys(state_dict)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def create_realesrgan_upsampler(model_path: Path, device):
    """Create a RealESRGANer x4 upsampler using D-drive weights."""
    _require_torch()
    if _RealESRGANRRDBNet is None or RealESRGANer is None:
        raise ImportError("basicsr and realesrgan are required for Real-ESRGAN inference")
    model = _RealESRGANRRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=4,
    )
    return RealESRGANer(
        scale=4,
        model_path=str(model_path),
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device,
    )


def run_bicubic_method(lr_bgr: np.ndarray, hr_bgr: np.ndarray) -> dict[str, Any]:
    """Generate the bicubic x4 baseline."""
    output_bgr = bicubic_upscale_to_size(lr_bgr, hr_bgr.shape[:2])
    return build_method_result("Bicubic x4", output_bgr, hr_bgr)


def run_srcnn_method(
    lr_bgr: np.ndarray,
    hr_bgr: np.ndarray,
    model_path: Path,
    device,
) -> dict[str, Any]:
    """Run classic SRCNN x4 on bicubic-upscaled input."""
    bicubic_bgr = bicubic_upscale_to_size(lr_bgr, hr_bgr.shape[:2])
    model = load_srcnn_model(model_path, device)
    ycrcb = cv2.cvtColor(bicubic_bgr, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb[:, :, 0].astype(np.float32) / 255.0

    with torch.no_grad():
        input_tensor = (
            torch.from_numpy(y_channel)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
            .float()
        )
        output_tensor = model(input_tensor).clamp(0.0, 1.0)

    output_y = (
        output_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255.0
    ).round().clip(0, 255).astype(np.uint8)
    merged = ycrcb.copy()
    merged[:, :, 0] = output_y
    output_bgr = cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)
    output_bgr = align_to_reference(output_bgr, hr_bgr)
    return build_method_result("SRCNN x4", output_bgr, hr_bgr)


def run_esrgan_method(
    lr_bgr: np.ndarray,
    hr_bgr: np.ndarray,
    model_path: Path,
    device,
) -> dict[str, Any]:
    """Run ESRGAN x4 with the local RRDBNet reference architecture."""
    model = load_esrgan_model(model_path, device)
    rgb = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    input_tensor = (
        torch.from_numpy(np.transpose(rgb, (2, 0, 1)))
        .unsqueeze(0)
        .to(device)
        .float()
    )

    with torch.no_grad():
        output_tensor = model(input_tensor).clamp(0.0, 1.0)

    output_rgb = np.transpose(
        output_tensor.squeeze(0).cpu().numpy(),
        (1, 2, 0),
    )
    output_bgr = cv2.cvtColor(
        (output_rgb * 255.0).round().clip(0, 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    output_bgr = align_to_reference(output_bgr, hr_bgr)
    return build_method_result("ESRGAN x4", output_bgr, hr_bgr)


def run_realesrgan_method(
    lr_bgr: np.ndarray,
    hr_bgr: np.ndarray,
    model_path: Path,
    device,
) -> dict[str, Any]:
    """Run Real-ESRGAN x4 using the project's working dependency chain."""
    upsampler = create_realesrgan_upsampler(model_path, device)
    output_bgr, _ = upsampler.enhance(lr_bgr, outscale=4)
    output_bgr = align_to_reference(output_bgr, hr_bgr)
    return build_method_result("Real-ESRGAN x4", output_bgr, hr_bgr)


def fit_image_into_box(
    image_bgr: np.ndarray,
    box_size: tuple[int, int],
    pad_value: int = 255,
) -> np.ndarray:
    """Resize an image into a fixed (height, width) box with centered padding."""
    box_h, box_w = box_size
    image_h, image_w = image_bgr.shape[:2]
    scale = min(box_w / float(image_w), box_h / float(image_h))
    target_w = max(1, int(round(image_w * scale)))
    target_h = max(1, int(round(image_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image_bgr, (target_w, target_h), interpolation=interpolation)

    canvas = np.full((box_h, box_w, 3), pad_value, dtype=np.uint8)
    top = max((box_h - target_h) // 2, 0)
    left = max((box_w - target_w) // 2, 0)
    canvas[top : top + target_h, left : left + target_w] = resized
    return canvas


def clip_focus_box(
    focus_box: tuple[int, int, int, int],
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int]:
    """Clip a focus box (x, y, w, h) to image bounds."""
    image_h, image_w = image_shape[:2]
    raw_x, raw_y, raw_width, raw_height = [int(round(value)) for value in focus_box]
    x = max(raw_x, 0)
    y = max(raw_y, 0)
    right = min(max(raw_x + raw_width, 0), image_w)
    bottom = min(max(raw_y + raw_height, 0), image_h)
    if right <= x or bottom <= y:
        raise ValueError("focus_box must overlap the image area")
    return (x, y, right - x, bottom - y)


def extract_focus_crop(
    image_bgr: np.ndarray,
    focus_box: tuple[int, int, int, int],
) -> np.ndarray:
    """Extract a cropped focus region from an image."""
    x, y, width, height = clip_focus_box(focus_box, image_bgr.shape)
    return image_bgr[y : y + height, x : x + width].copy()


def _map_focus_box_to_fitted_box(
    image_shape: tuple[int, ...],
    box_size: tuple[int, int],
    focus_box: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """Map a focus box from source image coordinates into a fitted display box."""
    image_h, image_w = image_shape[:2]
    box_h, box_w = box_size
    x, y, width, height = clip_focus_box(focus_box, image_shape)
    scale = min(box_w / float(image_w), box_h / float(image_h))
    target_w = max(1, int(round(image_w * scale)))
    target_h = max(1, int(round(image_h * scale)))
    top = max((box_h - target_h) // 2, 0)
    left = max((box_w - target_w) // 2, 0)
    mapped_x = left + int(round(x * scale))
    mapped_y = top + int(round(y * scale))
    mapped_w = max(1, int(round(width * scale)))
    mapped_h = max(1, int(round(height * scale)))
    return (mapped_x, mapped_y, mapped_w, mapped_h)


def draw_focus_boxes(
    image_bgr: np.ndarray,
    focus_boxes: Sequence[tuple[int, int, int, int]],
    color: tuple[int, int, int] = (0, 0, 255),
    thickness: int = 4,
    labels: Sequence[str] | None = None,
) -> np.ndarray:
    """Draw red focus boxes on an image and return a copy."""
    outlined = image_bgr.copy()
    for index, focus_box in enumerate(focus_boxes):
        x, y, width, height = clip_focus_box(focus_box, outlined.shape)
        cv2.rectangle(outlined, (x, y), (x + width - 1, y + height - 1), color, thickness)
        if labels and index < len(labels):
            label = labels[index]
            label_x = max(x, 0)
            label_y = max(y - 8, 18)
            cv2.putText(
                outlined,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
    return outlined


def build_relative_focus_boxes(
    image_shape: tuple[int, ...],
    relative_boxes: Sequence[tuple[float, float, float, float]],
) -> list[tuple[int, int, int, int]]:
    """Build absolute focus boxes from relative ratios."""
    image_h, image_w = image_shape[:2]
    absolute_boxes: list[tuple[int, int, int, int]] = []
    for rel_x, rel_y, rel_w, rel_h in relative_boxes:
        absolute_boxes.append(
            clip_focus_box(
                (
                    int(round(rel_x * image_w)),
                    int(round(rel_y * image_h)),
                    int(round(rel_w * image_w)),
                    int(round(rel_h * image_h)),
                ),
                image_shape,
            )
        )
    return absolute_boxes


def json_safe_metric_value(value: float) -> float | None:
    """Convert non-finite metrics to JSON-safe values."""
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return None
    return numeric_value


def get_default_focus_boxes(
    case_id: str,
    image_shape: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    """Return paper focus boxes for known representative cases."""
    normalized = case_id.lower()
    if normalized == "0844":
        return build_relative_focus_boxes(
            image_shape,
            [
                (0.22, 0.52, 0.16, 0.12),
                (0.40, 0.68, 0.16, 0.14),
                (0.69, 0.18, 0.16, 0.16),
            ],
        )
    if "lincoln" in normalized or "hesler" in normalized:
        return build_relative_focus_boxes(
            image_shape,
            [
                (0.23, 0.16, 0.22, 0.14),
                (0.46, 0.28, 0.18, 0.12),
                (0.41, 0.47, 0.22, 0.12),
            ],
        )
    return []


def _load_font(size: int):
    _require_pillow()
    font_candidates = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans.ttf",
    ]
    for font_path in font_candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _center_text_x(draw, text: str, font, left: int, width: int) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    return left + max((width - text_width) // 2, 0)


def render_comparison_figure(
    input_display_bgr: np.ndarray,
    method_results: Sequence[dict[str, Any]],
    output_path: str | Path,
    display_max_height: int = 420,
    focus_boxes: Sequence[tuple[int, int, int, int]] | None = None,
) -> Path:
    """Render the final paper figure for overview and zoom-in comparison."""
    _require_pillow()
    display_box_size = (display_max_height, 300)
    zoom_row_height = max(150, int(round(display_max_height * 0.42)))
    focus_boxes = list(focus_boxes or [])
    focus_labels = [str(index + 1) for index in range(len(focus_boxes))]
    columns = [
        {
            "title": "Input",
            "source_image": input_display_bgr,
        }
    ]
    for result in method_results:
        columns.append(
            {
                "title": result["name"],
                "source_image": result["image"],
            }
        )

    cell_height, cell_width = display_box_size
    outer_padding = 24
    label_column_width = 110 if focus_boxes else 0
    column_gap = 18
    title_height = 34
    section_gap = 28
    row_gap = 14
    canvas_width = (
        outer_padding * 2
        + label_column_width
        + len(columns) * cell_width
        + (len(columns) - 1) * column_gap
    )
    canvas_height = outer_padding * 2 + title_height + cell_height
    if focus_boxes:
        canvas_height += section_gap
        canvas_height += len(focus_boxes) * zoom_row_height
        canvas_height += max(len(focus_boxes) - 1, 0) * row_gap

    canvas = Image.new("RGB", (canvas_width, canvas_height), color="white")
    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(18)
    body_font = _load_font(16)

    for index, column in enumerate(columns):
        left = outer_padding + label_column_width + index * (cell_width + column_gap)
        top = outer_padding
        overview_image = fit_image_into_box(column["source_image"], display_box_size)
        if focus_boxes and index == 0:
            mapped_boxes = [
                _map_focus_box_to_fitted_box(column["source_image"].shape, display_box_size, box)
                for box in focus_boxes
            ]
            overview_image = draw_focus_boxes(
                overview_image,
                mapped_boxes,
                labels=focus_labels,
                thickness=3,
            )
        image_bgr = overview_image
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        image_left = left + (cell_width - pil_image.width) // 2
        image_top = top + title_height
        title_text = format_figure_title(column["title"])
        text_x = _center_text_x(draw, title_text, title_font, left, cell_width)
        draw.text((text_x, top), title_text, fill="black", font=title_font)
        canvas.paste(pil_image, (image_left, image_top))

    current_top = outer_padding + title_height + cell_height
    if focus_boxes:
        current_top += section_gap
        for focus_index, focus_box in enumerate(focus_boxes):
            row_label = f"Region {focus_index + 1}"
            label_bbox = draw.textbbox((0, 0), row_label, font=body_font)
            label_height = label_bbox[3] - label_bbox[1]
            label_y = current_top + max((zoom_row_height - label_height) // 2, 0)
            draw.text(
                (outer_padding, label_y),
                row_label,
                fill=(80, 80, 80),
                font=body_font,
            )

            for column_index, column in enumerate(columns):
                left = outer_padding + label_column_width + column_index * (cell_width + column_gap)
                crop_bgr = extract_focus_crop(column["source_image"], focus_box)
                crop_display_bgr = fit_image_into_box(crop_bgr, (zoom_row_height, cell_width))
                crop_rgb = cv2.cvtColor(crop_display_bgr, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(crop_rgb)
                image_left = left + (cell_width - pil_crop.width) // 2
                canvas.paste(pil_crop, (image_left, current_top))
                draw.rectangle(
                    (
                        image_left,
                        current_top,
                        image_left + pil_crop.width - 1,
                        current_top + pil_crop.height - 1,
                    ),
                    outline=(170, 170, 170),
                    width=1,
                )

            current_top += zoom_row_height
            if focus_index < len(focus_boxes) - 1:
                current_top += row_gap

    final_path = Path(output_path)
    ensure_dir(final_path.parent)
    canvas.save(final_path)
    return final_path


def write_metrics_files(
    lr_path: Path,
    hr_path: Path,
    method_results: Sequence[dict[str, Any]],
    output_file_map: dict[str, Path],
) -> None:
    """Write metrics.json and metrics.txt for the single-case experiment."""
    payload = {
        "lr_path": str(lr_path),
        "hr_path": str(hr_path),
        "methods": [
            {
                "name": result["name"],
                "psnr": json_safe_metric_value(result["psnr"]),
                "ssim": json_safe_metric_value(result["ssim"]),
            }
            for result in method_results
        ],
    }
    output_file_map["metrics_json"].write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = [
        f"LR: {lr_path}",
        f"HR: {hr_path}",
        "",
    ]
    for result in method_results:
        lines.append(result["name"])
        lines.extend(format_metric_lines(result["psnr"], result["ssim"]))
        lines.append("")
    output_file_map["metrics_txt"].write_text(
        "\n".join(lines).strip() + "\n",
        encoding="utf-8",
    )


def generate_single_case_comparison(
    lr_path: str | Path,
    hr_path: str | Path,
    output_dir: str | Path,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    device_name: str = "auto",
    display_max_height: int = 420,
) -> dict[str, Any]:
    """Run the full single-case comparison pipeline and write all artifacts."""
    lr_resolved, hr_resolved = validate_inputs(lr_path, hr_path)
    device = resolve_device(device_name)
    output_dir_path = ensure_dir(output_dir)
    output_file_map = build_output_file_map(output_dir_path)
    model_file_map = build_model_file_map(model_dir)
    case_id = infer_case_id(lr_resolved, hr_resolved)

    lr_bgr = load_image_bgr(lr_resolved)
    hr_bgr = load_image_bgr(hr_resolved)

    input_display_bgr = make_input_display_image(lr_bgr, hr_bgr)
    save_image_bgr(output_file_map["input"], input_display_bgr)

    method_results: list[dict[str, Any]] = []

    bicubic_result = run_bicubic_method(lr_bgr, hr_bgr)
    save_image_bgr(output_file_map["bicubic"], bicubic_result["image"])
    method_results.append(bicubic_result)

    srcnn_model_path = download_model_file(model_file_map["srcnn"], MODEL_URLS["srcnn"])
    srcnn_result = run_srcnn_method(lr_bgr, hr_bgr, srcnn_model_path, device)
    save_image_bgr(output_file_map["srcnn"], srcnn_result["image"])
    method_results.append(srcnn_result)

    esrgan_model_path = download_model_file(
        model_file_map["esrgan"],
        MODEL_URLS["esrgan"],
    )
    esrgan_result = run_esrgan_method(lr_bgr, hr_bgr, esrgan_model_path, device)
    save_image_bgr(output_file_map["esrgan"], esrgan_result["image"])
    method_results.append(esrgan_result)

    realesrgan_model_path = download_model_file(
        model_file_map["realesrgan"],
        MODEL_URLS["realesrgan"],
    )
    realesrgan_result = run_realesrgan_method(
        lr_bgr,
        hr_bgr,
        realesrgan_model_path,
        device,
    )
    save_image_bgr(output_file_map["realesrgan"], realesrgan_result["image"])
    method_results.append(realesrgan_result)

    write_metrics_files(lr_resolved, hr_resolved, method_results, output_file_map)
    render_comparison_figure(
        input_display_bgr=input_display_bgr,
        method_results=method_results,
        output_path=output_file_map["comparison"],
        display_max_height=display_max_height,
        focus_boxes=get_default_focus_boxes(case_id, hr_bgr.shape),
    )
    return {
        "output_dir": output_dir_path,
        "output_files": output_file_map,
        "results": method_results,
        "device": str(device),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Generate a single-case 5-column SR paper comparison figure.",
    )
    parser.add_argument("--lr", type=str, default=str(DEFAULT_LR_PATH))
    parser.add_argument("--hr", type=str, default=str(DEFAULT_HR_PATH))
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
    )
    parser.add_argument("--display-max-height", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    lr_path = Path(args.lr).expanduser().resolve()
    hr_path = Path(args.hr).expanduser().resolve()
    case_id = infer_case_id(lr_path, hr_path)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "outputs" / f"paper_compare_{case_id}"
    )
    result = generate_single_case_comparison(
        lr_path=lr_path,
        hr_path=hr_path,
        output_dir=output_dir,
        model_dir=Path(args.model_dir).expanduser().resolve(),
        device_name=args.device,
        display_max_height=args.display_max_height,
    )
    print(f"Output directory: {result['output_dir']}")
    print(f"Comparison figure: {result['output_files']['comparison']}")
    for method in result["results"]:
        print(
            f"{method['name']}: "
            f"PSNR={method['psnr']:.2f} dB, SSIM={method['ssim']:.4f}"
        )


if __name__ == "__main__":
    main()
