"""后台工作线程：模型加载、图像处理、批处理。

使用 QThread + Signal/Slot 替代 threading.Thread + queue.Queue。
"""
import logging
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from .models import ModelManager
from .processing import (
    UserCancelledError, auto_tile_size, blend_images, blend_with_lr,
    apply_unsharp_mask, apply_film_grain, suppress_edge_ringing,
    clamp_value, estimate_image_metrics,
    TEXTURE_ENABLED, TEXTURE_MODEL_ID, IMAGE_EXTS,
)
from .utils import (
    ensure_dir, save_image, safe_basename, timestamp_str, write_json_file, read_image,
)

logger = logging.getLogger(__name__)


class ModelLoadWorker(QThread):
    """后台加载模型。"""
    progress = Signal(float, str)
    finished = Signal(bool, str)

    def __init__(self, model_manager: ModelManager, scale: int) -> None:
        super().__init__()
        self.model_manager = model_manager
        self.scale = scale

    def run(self) -> None:
        try:
            self.progress.emit(0.3, "Loading model...")
            self.model_manager.load_esrgan(self.scale)
            self.progress.emit(1.0, f"Model x{self.scale} loaded")
            self.finished.emit(True, f"Model x{self.scale} loaded | Device: {self.model_manager.device}")
        except Exception as e:
            logger.error("Model load failed: %s", e)
            self.finished.emit(False, str(e))


class ProcessWorker(QThread):
    """后台单张图像处理。"""
    progress = Signal(float, str)
    stage_changed = Signal(str)
    image_ready = Signal(object)
    metrics_ready = Signal(dict)
    finished = Signal(bool, str)

    def __init__(
        self,
        model_manager: ModelManager,
        img_input: np.ndarray,
        input_path: Optional[str],
        gt_img: Optional[np.ndarray],
        settings: Dict[str, Any],
        output_base_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.model_manager = model_manager
        self.img_input = img_input.copy()
        self.input_path = input_path
        self.gt_img = gt_img.copy() if gt_img is not None else None
        self.settings = settings
        self.output_base_dir = output_base_dir
        self.output_path: Optional[str] = None
        self.run_dir: Optional[str] = None
        self.run_meta: Dict[str, Any] = {}
        self.elapsed: Optional[float] = None

    def _is_cancelled(self) -> bool:
        return self.isInterruptionRequested()

    def _check_cancel(self) -> None:
        if self._is_cancelled():
            raise UserCancelledError("Cancelled")

    def run(self) -> None:
        start_time = time.perf_counter()
        scale = self.settings.get("scale", 4)
        face_enhance = self.settings.get("face_enhance", False)
        face_blend = float(np.clip(self.settings.get("face_blend", 0.65), 0.4, 0.9))
        natural_blend = float(np.clip(self.settings.get("natural_blend", 0.0), 0.0, 0.12))
        texture_boost = float(np.clip(self.settings.get("texture_boost", 0.08), 0.0, 0.35))
        film_grain = float(self.settings.get("film_grain", 0.0))
        scratch_repair = self.settings.get("scratch_repair", False)

        base_name = safe_basename(self.input_path)
        run_id = timestamp_str()
        run_root = self.output_base_dir or os.path.join(
            os.path.dirname(os.path.abspath(self.input_path or ".")),
            "outputs",
        )
        self.run_dir = ensure_dir(os.path.join(run_root, f"{run_id}_{base_name}"))
        self.output_path = os.path.join(self.run_dir, f"{base_name}_x{scale}.png")

        self.run_meta = {
            "run_id": run_id,
            "timestamp": timestamp_str(),
            "input_path": os.path.basename(self.input_path) if self.input_path else None,
            "scale_factor": scale,
            "device": str(self.model_manager.device),
            "face_enhance": face_enhance,
            "natural_blend": natural_blend,
            "texture_boost": texture_boost,
            "film_grain": film_grain,
            "timing": {},
        }

        try:
            input_img = self.img_input
            output = None

            # Scratch repair
            if scratch_repair:
                self._check_cancel()
                self.progress.emit(0.05, "Repairing scratches...")
                self.stage_changed.emit("scratch")
                stage_start = time.perf_counter()
                self.model_manager.load_scratch_model_if_needed()
                from .processing import apply_scratch_repair, SCRATCH_MASK_THRESHOLD, SCRATCH_INPAINT_RADIUS
                input_img = apply_scratch_repair(
                    input_img, self.model_manager.scratch_model,
                    self.model_manager.device,
                    SCRATCH_MASK_THRESHOLD, SCRATCH_INPAINT_RADIUS,
                )
                self.run_meta["timing"]["scratch"] = round(time.perf_counter() - stage_start, 3)

            # Upscale
            self._check_cancel()
            self.progress.emit(0.10, f"Upscaling image (x{scale})...")
            self.stage_changed.emit("upscale")
            stage_start = time.perf_counter()
            h_in, w_in = input_img.shape[:2]
            tile = auto_tile_size(h_in, w_in, scale)
            sr_base, _ = self.model_manager.enhance(input_img, scale, tile)
            self.run_meta["timing"]["upscale"] = round(time.perf_counter() - stage_start, 3)
            output = sr_base

            self._check_cancel()
            self.progress.emit(0.65, "Upscale complete. Refining details...")
            self.stage_changed.emit("refine")

            # Face enhancement
            used_face_enhance = False
            if face_enhance:
                self._check_cancel()
                self.progress.emit(0.70, "Applying face enhancement...")
                self.stage_changed.emit("face")
                stage_start = time.perf_counter()
                try:
                    if (
                        self.model_manager.face_enhancer is None
                        or self.model_manager.face_enhancer_scale != scale
                    ):
                        self.model_manager.load_face_enhancer(scale)
                    _, _, face_output = self.model_manager.face_enhancer.enhance(
                        input_img, has_aligned=False, only_center_face=False, paste_back=True,
                    )
                    if face_output is not None:
                        output = blend_images(face_output, sr_base, face_blend)
                        used_face_enhance = True
                except Exception as e:
                    logger.warning("Face enhance failed: %s", e)
                self.run_meta["timing"]["face"] = round(time.perf_counter() - stage_start, 3)

            # Blend + texture + grain
            self._check_cancel()
            self.progress.emit(0.80, "Blending fine details...")
            self.stage_changed.emit("blend")
            stage_start = time.perf_counter()

            if natural_blend <= 0.0 and texture_boost <= 0.0:
                output = suppress_edge_ringing(output, input_img, strength=0.25)
            else:
                dehalo_strength = 0.62 if natural_blend <= 0.02 else 0.50
                output = suppress_edge_ringing(output, input_img, strength=dehalo_strength)
                output = blend_with_lr(output, input_img, natural_blend)
                output = apply_unsharp_mask(output, texture_boost, blend_weight=natural_blend)
            self.run_meta["timing"]["blend"] = round(time.perf_counter() - stage_start, 3)

            # Texture generation
            if TEXTURE_ENABLED and TEXTURE_MODEL_ID:
                try:
                    self._check_cancel()
                    self.progress.emit(0.88, "Generating texture details...")
                    self.stage_changed.emit("texture")
                    stage_start = time.perf_counter()
                    output = self.model_manager.apply_texture_generation(
                        output, cancel_check=self._is_cancelled
                    )
                    self.run_meta["timing"]["texture"] = round(time.perf_counter() - stage_start, 3)
                except UserCancelledError:
                    raise
                except Exception as e:
                    logger.warning("Texture generation skipped: %s", e)

            # Film grain
            grain_cap = 0.02 if (natural_blend <= 0.0 and texture_boost <= 0.0) else film_grain
            output = apply_film_grain(output, min(film_grain, grain_cap) if natural_blend <= 0.0 and texture_boost <= 0.0 else film_grain)

            self._check_cancel()
            self.progress.emit(0.95, "Finalizing...")
            self.stage_changed.emit("finalize")

            # Save output
            save_image(self.output_path, output)
            # Reload from disk for stable buffer
            stable = read_image(self.output_path)
            if stable is not None:
                output = stable

            self.image_ready.emit(output)

            # Save input snapshot
            input_snap_path = os.path.join(self.run_dir, f"{base_name}_input.png")
            save_image(input_snap_path, input_img)

            self.elapsed = time.perf_counter() - start_time
            self.run_meta["elapsed_sec"] = round(self.elapsed, 3)
            self.run_meta["timing"]["total"] = round(self.elapsed, 3)

            # Write run log
            write_json_file(
                os.path.join(self.run_dir, "run_log.json"), self.run_meta
            )

            status = f"Done (x{scale})"
            if face_enhance and not used_face_enhance:
                status = f"Done (x{scale} Standard Mode)"
            self.progress.emit(1.0, status)
            self.finished.emit(True, status)

        except UserCancelledError:
            self.elapsed = time.perf_counter() - start_time
            self.finished.emit(False, "Cancelled")
        except Exception as e:
            self.elapsed = time.perf_counter() - start_time
            logger.error("Processing failed: %s", e)
            self.finished.emit(False, str(e))


class BatchWorker(QThread):
    """后台批处理。"""
    item_started = Signal(int, int, str)   # (index, total, filename)
    item_done = Signal(int, int)           # (index, total)
    progress = Signal(float, str)
    finished = Signal(bool, str, list)     # (success, message, errors)

    def __init__(
        self,
        model_manager: ModelManager,
        file_list: List[str],
        settings: Dict[str, Any],
        output_dir: str,
        retry_limit: int = 1,
    ) -> None:
        super().__init__()
        self.model_manager = model_manager
        self.file_list = file_list
        self.settings = settings
        self.output_dir = output_dir
        self.retry_limit = retry_limit
        self.errors: List[str] = []

    def run(self) -> None:
        total = len(self.file_list)
        scale = self.settings.get("scale", 4)
        errors = []

        for idx, filepath in enumerate(self.file_list):
            if self.isInterruptionRequested():
                break

            basename = os.path.basename(filepath)
            self.item_started.emit(idx, total, basename)
            self.progress.emit(idx / total, f"Batch {idx + 1}/{total}: {basename}")

            retries = 0
            success = False
            while retries <= self.retry_limit and not success:
                try:
                    img = read_image(filepath)
                    if img is None:
                        raise RuntimeError(f"Failed to read: {basename}")

                    h_in, w_in = img.shape[:2]
                    tile = auto_tile_size(h_in, w_in, scale)
                    sr_output, _ = self.model_manager.enhance(img, scale, tile)

                    # Apply post-processing
                    natural_blend = self.settings.get("natural_blend", 0.0)
                    texture_boost = self.settings.get("texture_boost", 0.08)
                    film_grain = self.settings.get("film_grain", 0.0)

                    if natural_blend <= 0.0 and texture_boost <= 0.0:
                        sr_output = suppress_edge_ringing(sr_output, img, strength=0.25)
                    else:
                        dehalo = 0.62 if natural_blend <= 0.02 else 0.50
                        sr_output = suppress_edge_ringing(sr_output, img, strength=dehalo)
                        sr_output = blend_with_lr(sr_output, img, natural_blend)
                        sr_output = apply_unsharp_mask(sr_output, texture_boost, blend_weight=natural_blend)

                    sr_output = apply_film_grain(sr_output, film_grain)

                    stem = os.path.splitext(basename)[0]
                    out_path = os.path.join(self.output_dir, f"{stem}_x{scale}.png")
                    save_image(out_path, sr_output)
                    success = True

                except Exception as e:
                    retries += 1
                    if retries > self.retry_limit:
                        errors.append(f"{basename}: {e}")
                        logger.error("Batch item failed: %s - %s", basename, e)

            self.item_done.emit(idx, total)

        self.errors = errors
        if self.isInterruptionRequested():
            self.finished.emit(False, "Batch cancelled", errors)
        elif errors:
            self.finished.emit(False, f"Batch done with {len(errors)} errors", errors)
        else:
            self.finished.emit(True, f"Batch complete: {total} images", errors)
