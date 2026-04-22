from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .ddcolor_backend import DDColorModelNotFoundError, load_ddcolor_backend, run_ddcolor_inference
from .utils import ensure_dir, safe_basename, save_image, timestamp_str, write_json_file

ColorizationModelNotFoundError = DDColorModelNotFoundError


@dataclass
class ColorizationResult:
    output_image: np.ndarray
    output_path: str
    run_dir: str
    run_meta: dict
    elapsed: float


def prepare_colorization_input(
    input_img: np.ndarray,
    max_side: int = 1024,
) -> tuple[np.ndarray, tuple[int, int]]:
    if input_img is None:
        raise ValueError("input_img is required")

    image = np.asarray(input_img)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("input image must have 1, 3, or 4 channels")

    original_shape = image.shape[:2]
    if max_side and max_side > 0:
        longest_side = max(original_shape)
        if longest_side > max_side:
            scale = max_side / float(longest_side)
            new_w = max(1, int(round(image.shape[1] * scale)))
            new_h = max(1, int(round(image.shape[0] * scale)))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(image), original_shape


def _normalize_color_output(output_img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    image = np.asarray(output_img)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError("colorization backend must return a BGR image")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.shape[:2] != target_shape:
        target_h, target_w = target_shape
        image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

    return np.ascontiguousarray(image)


def _estimate_skin_mask(image: np.ndarray) -> np.ndarray:
    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]
    luma = 0.114 * b + 0.587 * g + 0.299 * r

    return (
        (r > 135.0)
        & (g > 110.0)
        & (b > 80.0)
        & (r > g)
        & (g >= b)
        & ((r - g) > 12.0)
        & ((g - b) > 4.0)
        & (luma > 105.0)
    )


def _fallback_subject_mask(shape: tuple[int, int, int] | tuple[int, int]) -> np.ndarray:
    height, width = shape[:2]
    yy, xx = np.ogrid[:height, :width]
    center_x = (width - 1) / 2.0
    center_y = height * 0.58
    radius_x = max(width * 0.28, 1.0)
    radius_y = max(height * 0.42, 1.0)
    ellipse = (
        ((xx - center_x) ** 2) / (radius_x ** 2)
        + ((yy - center_y) ** 2) / (radius_y ** 2)
    ) <= 1.0
    return ellipse


def _estimate_subject_mask(image: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    fallback_mask = _fallback_subject_mask(image.shape)
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image_u8.shape[:2]

    if min(height, width) < 24:
        return fallback_mask | skin_mask

    margin_x = max(int(round(width * 0.12)), 1)
    margin_top = max(int(round(height * 0.05)), 1)
    rect_w = max(width - 2 * margin_x, max(int(round(width * 0.5)), 2))
    rect_h = max(height - margin_top - 1, max(int(round(height * 0.75)), 2))
    rect = (margin_x, margin_top, rect_w, rect_h)

    mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    x, y, rect_w, rect_h = rect
    mask[y:y + rect_h, x:x + rect_w] = cv2.GC_PR_FGD
    mask[fallback_mask] = cv2.GC_PR_FGD
    if np.any(skin_mask):
        mask[skin_mask] = cv2.GC_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image_u8,
            mask,
            rect,
            bgd_model,
            fgd_model,
            2,
            cv2.GC_INIT_WITH_MASK,
        )
        subject_mask = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    except cv2.error:
        subject_mask = fallback_mask.copy()

    subject_mask |= skin_mask
    subject_ratio = float(subject_mask.mean())
    if subject_ratio < 0.08 or subject_ratio > 0.92:
        subject_mask = fallback_mask | skin_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined = cv2.morphologyEx(subject_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    refined = cv2.dilate(refined, kernel, iterations=1)
    return refined.astype(bool)


def _apply_channel_balance(image: np.ndarray) -> np.ndarray:
    channel_means = image.mean(axis=(0, 1))
    target_mean = float(channel_means.mean())
    if target_mean <= 0:
        return image

    scales = []
    for channel_mean in channel_means:
        delta = float(channel_mean - target_mean)
        if abs(delta) < 8.0:
            scales.append(1.0)
            continue
        scale = 1.0 - 0.28 * (delta / max(float(channel_mean), 1.0))
        scales.append(float(np.clip(scale, 0.88, 1.12)))

    balanced = image * np.asarray(scales, dtype=np.float32).reshape(1, 1, 3)
    return image * 0.82 + balanced * 0.18


def _desaturate_in_lab(image: np.ndarray, chroma_scale: float = 0.9) -> np.ndarray:
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(image_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] = 128.0 + (lab[:, :, 1] - 128.0) * chroma_scale
    lab[:, :, 2] = 128.0 + (lab[:, :, 2] - 128.0) * chroma_scale
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)


def _reduce_cool_magenta_cast(
    image: np.ndarray,
    skin_mask: np.ndarray,
    subject_mask: np.ndarray,
) -> np.ndarray:
    image_u8 = np.clip(image, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(image_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    value = hsv[:, :, 2]
    background_mask = ~subject_mask

    cool_cast_mask = (
        background_mask
        & (sat > 24.0)
        & (value < 245.0)
        & (
            ((hue >= 78.0) & (hue <= 115.0))
            | ((hue >= 128.0) & (hue <= 179.0))
        )
    )

    subject_cool_mask = (
        (~skin_mask)
        & subject_mask
        & (sat > 20.0)
        & (
            ((hue >= 78.0) & (hue <= 115.0))
            | ((hue >= 128.0) & (hue <= 179.0))
        )
    )

    hsv[:, :, 1] *= 0.94
    hsv[:, :, 1][background_mask] *= 0.88
    hsv[:, :, 1][cool_cast_mask] *= 0.5
    hsv[:, :, 1][subject_cool_mask] *= 0.92

    toned = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    gray = cv2.cvtColor(np.clip(toned, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_bgr = np.repeat(gray[:, :, None], 3, axis=2)

    blend_weight = np.zeros(gray.shape, dtype=np.float32)
    blend_weight[background_mask] = 0.14
    blend_weight[cool_cast_mask] = 0.38
    blend_weight = cv2.GaussianBlur(blend_weight, (0, 0), sigmaX=1.1, sigmaY=1.1)

    return toned * (1.0 - blend_weight[:, :, None]) + gray_bgr * blend_weight[:, :, None]


def _warm_skin_regions(image: np.ndarray, skin_mask: np.ndarray) -> np.ndarray:
    warmed = image.copy()
    warmed[:, :, 0] *= 0.86
    warmed[:, :, 1] *= 0.99
    warmed[:, :, 2] *= 1.1

    result = image.copy()
    result[skin_mask] = result[skin_mask] * 0.55 + warmed[skin_mask] * 0.45
    return result


def postprocess_colorized_output(output_img: np.ndarray) -> np.ndarray:
    image = np.asarray(output_img, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("output_img must be a BGR image")

    balanced = _apply_channel_balance(image)
    softened = _desaturate_in_lab(balanced, chroma_scale=0.9)
    skin_mask = _estimate_skin_mask(softened)
    subject_mask = _estimate_subject_mask(softened, skin_mask)
    toned = _reduce_cool_magenta_cast(softened, skin_mask, subject_mask)
    warmed = _warm_skin_regions(toned, skin_mask)
    return np.clip(warmed, 0, 255).astype(np.uint8)


def _raise_cancelled() -> None:
    from .processing import UserCancelledError

    raise UserCancelledError("Cancelled")


def run_colorization_pipeline(
    *,
    input_img: np.ndarray,
    input_path: Optional[str],
    output_base_dir: Optional[str] = None,
    backend: Optional[object] = None,
    model_path: Optional[os.PathLike | str] = None,
    max_side: int = 1024,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ColorizationResult:
    start_time = time.perf_counter()

    if cancel_check and cancel_check():
        _raise_cancelled()

    prepared_img, original_shape = prepare_colorization_input(input_img, max_side=max_side)

    if cancel_check and cancel_check():
        _raise_cancelled()

    backend_obj = backend or load_ddcolor_backend(model_path=model_path, device="cpu")
    colorized = run_ddcolor_inference(
        prepared_img,
        backend=backend_obj,
    )

    if cancel_check and cancel_check():
        _raise_cancelled()

    output_img = _normalize_color_output(colorized, original_shape)
    output_img = postprocess_colorized_output(output_img)

    base_name = safe_basename(input_path)
    run_id = timestamp_str()
    if output_base_dir:
        run_root = output_base_dir
    else:
        input_dir = os.path.dirname(os.path.abspath(input_path or "."))
        run_root = os.path.join(input_dir, "outputs")
    run_dir = ensure_dir(os.path.join(run_root, f"{run_id}_{base_name}"))
    output_path = os.path.join(run_dir, f"{base_name}_colorized.png")
    save_image(output_path, output_img)

    elapsed = time.perf_counter() - start_time
    resolved_model_path = str(model_path or getattr(backend_obj, "model_path", ""))
    run_meta = {
        "run_id": run_id,
        "timestamp": run_id,
        "backend": "ddcolor",
        "input_path": os.path.basename(input_path) if input_path else None,
        "model_path": resolved_model_path,
        "max_side": max_side,
        "output_path": output_path,
        "elapsed_seconds": elapsed,
    }
    write_json_file(os.path.join(run_dir, "colorization_run.json"), run_meta)

    return ColorizationResult(
        output_image=output_img,
        output_path=output_path,
        run_dir=run_dir,
        run_meta=run_meta,
        elapsed=elapsed,
    )
