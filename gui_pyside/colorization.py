from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .utils import ensure_dir, safe_basename, save_image, timestamp_str, write_json_file

DEFAULT_COLORIZATION_MODEL_NAME = "colorization_release_v2.caffemodel"
DEFAULT_COLORIZATION_PROTO_NAME = "colorization_deploy_v2.prototxt"
DEFAULT_COLORIZATION_POINTS_NAME = "pts_in_hull.npy"


class ColorizationModelNotFoundError(FileNotFoundError):
    """Raised when the configured colorization model file is missing."""


@dataclass
class ColorizationResult:
    output_image: np.ndarray
    output_path: str
    run_dir: str
    run_meta: dict
    elapsed: float


def get_colorization_model_path(explicit_path: Optional[os.PathLike | str] = None) -> Path:
    if explicit_path:
        return Path(explicit_path)

    env_path = os.environ.get("COLORIZATION_MODEL_PATH", "").strip()
    if env_path:
        return Path(env_path)

    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "models" / "colorization" / DEFAULT_COLORIZATION_MODEL_NAME


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


def _resolve_support_path(model_path: Path, file_name: str) -> Path:
    return model_path.with_name(file_name)


def default_colorize_backend(
    prepared_img: np.ndarray,
    model_path: os.PathLike | str,
    *,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> np.ndarray:
    model_path = Path(model_path)
    prototxt_path = _resolve_support_path(model_path, DEFAULT_COLORIZATION_PROTO_NAME)
    points_path = _resolve_support_path(model_path, DEFAULT_COLORIZATION_POINTS_NAME)

    missing = [path for path in (model_path, prototxt_path, points_path) if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise ColorizationModelNotFoundError(f"Missing colorization assets: {joined}")

    if cancel_check and cancel_check():
        from .processing import UserCancelledError

        raise UserCancelledError("Cancelled")

    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
    kernel = np.load(points_path)
    class8_ab = net.getLayerId("class8_ab")
    conv8_313_rh = net.getLayerId("conv8_313_rh")
    kernel = kernel.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8_ab).blobs = [kernel.astype(np.float32)]
    net.getLayer(conv8_313_rh).blobs = [np.full((1, 313), 2.606, dtype=np.float32)]

    normalized = prepared_img.astype(np.float32) / 255.0
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_BGR2LAB)
    l_channel = lab_image[:, :, 0]
    resized_l = cv2.resize(l_channel, (224, 224))
    resized_l -= 50
    net.setInput(cv2.dnn.blobFromImage(resized_l))
    ab_channel = net.forward()[0].transpose((1, 2, 0))
    ab_channel = cv2.resize(ab_channel, (prepared_img.shape[1], prepared_img.shape[0]))

    lab_output = np.concatenate((l_channel[:, :, np.newaxis], ab_channel), axis=2)
    bgr_output = cv2.cvtColor(lab_output, cv2.COLOR_LAB2BGR)
    bgr_output = np.clip(bgr_output, 0.0, 1.0)
    return (bgr_output * 255).astype(np.uint8)


def run_colorization_pipeline(
    *,
    input_img: np.ndarray,
    input_path: Optional[str],
    output_base_dir: Optional[str] = None,
    backend: Optional[Callable[..., np.ndarray]] = None,
    model_path: Optional[os.PathLike | str] = None,
    max_side: int = 1024,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> ColorizationResult:
    start_time = time.perf_counter()
    resolved_model_path = get_colorization_model_path(model_path)
    if not resolved_model_path.exists():
        raise ColorizationModelNotFoundError(
            f"Colorization model not found: {resolved_model_path}"
        )

    if cancel_check and cancel_check():
        from .processing import UserCancelledError

        raise UserCancelledError("Cancelled")

    prepared_img, original_shape = prepare_colorization_input(input_img, max_side=max_side)
    if cancel_check and cancel_check():
        from .processing import UserCancelledError

        raise UserCancelledError("Cancelled")

    backend_fn = backend or default_colorize_backend
    colorized = backend_fn(
        prepared_img,
        resolved_model_path,
        cancel_check=cancel_check,
    )
    output_img = _normalize_color_output(colorized, original_shape)

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
    run_meta = {
        "run_id": run_id,
        "timestamp": run_id,
        "input_path": os.path.basename(input_path) if input_path else None,
        "model_path": str(resolved_model_path),
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
