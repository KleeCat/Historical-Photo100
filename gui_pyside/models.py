"""模型加载和管理。

封装 RealESRGAN、GFPGAN、划痕修复、纹理生成等模型的加载与缓存。
"""
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from .processing import (
    ScratchUNet, clean_state_dict, load_scratch_model,
    apply_scratch_repair, suppress_edge_ringing, blend_with_lr,
    apply_unsharp_mask, apply_film_grain, blend_images,
    auto_tile_size, UserCancelledError,
    SCRATCH_MODEL_PATH, SCRATCH_MASK_THRESHOLD, SCRATCH_INPAINT_RADIUS,
    TEXTURE_ENABLED, TEXTURE_MODEL_ID, TEXTURE_PROMPT,
    TEXTURE_STRENGTH, TEXTURE_GUIDANCE, TEXTURE_STEPS,
)

logger = logging.getLogger(__name__)

# Optional imports
try:
    from gfpgan import GFPGANer
except ImportError:
    GFPGANer = None
    logger.warning("gfpgan not installed, face enhancement unavailable.")

try:
    from diffusers import StableDiffusionImg2ImgPipeline
except ImportError:
    StableDiffusionImg2ImgPipeline = None

try:
    from PIL import Image as PILImage
except ImportError:
    PILImage = None


class ModelManager:
    """管理所有 AI 模型的加载和缓存。"""

    def __init__(self, device: torch.device, model_folder: str) -> None:
        self.device = device
        self.model_folder = model_folder
        self.model: Optional[RRDBNet] = None
        self.upsampler: Optional[RealESRGANer] = None
        self.face_enhancer = None
        self.face_enhancer_scale: Optional[int] = None
        self.scratch_model: Optional[nn.Module] = None
        self.texture_pipe = None
        self.hook_handles: list = []
        self.feature_maps: List[Tuple[str, torch.Tensor]] = []
        self.max_feature_maps = 6
        self._face_cascade = None
        self._model_lock = threading.Lock()
        self._state_lock = threading.Lock()

    # --- Feature hooks ---

    def clear_feature_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def register_feature_hooks(self) -> None:
        """Register forward hooks on Conv2d layers to capture feature maps."""
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

        if self.model is not None:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    self.hook_handles.append(
                        module.register_forward_hook(make_hook(name))
                    )

    # --- ESRGAN ---

    def load_esrgan(self, scale: int) -> None:
        """加载 RealESRGAN 模型。"""
        with self._model_lock:
            self.clear_feature_hooks()
            self.model = None
            self.upsampler = None
            self.face_enhancer = None
            self.face_enhancer_scale = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_name = "RealESRGAN_x2plus.pth" if scale == 2 else "RealESRGAN_x4plus.pth"
            full_path = os.path.join(self.model_folder, model_name)
            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"Model file not found: {model_name}\n"
                    f"Please download it to: {self.model_folder}"
                )

            self.model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=scale,
            )
            self.register_feature_hooks()
            self.upsampler = RealESRGANer(
                scale=scale, model_path=full_path, model=self.model,
                tile=0, tile_pad=10, pre_pad=0, half=False, device=self.device,
            )

    # --- Face enhancement ---

    def load_face_enhancer(self, scale: int) -> None:
        """加载 GFPGAN 人脸增强模型。"""
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
        self.face_enhancer = GFPGANer(
            model_path=gfpgan_path, upscale=scale, arch="clean",
            channel_multiplier=2, bg_upsampler=self.upsampler,
        )
        self.face_enhancer_scale = scale

    # --- Scratch repair ---

    def load_scratch_model_if_needed(self) -> None:
        if self.scratch_model is None and SCRATCH_MODEL_PATH:
            self.scratch_model = load_scratch_model(SCRATCH_MODEL_PATH, self.device)

    # --- Texture generation ---

    def get_texture_pipeline(self):
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

    def apply_texture_generation(
        self, bgr_img: np.ndarray, cancel_check=None
    ) -> np.ndarray:
        if cancel_check and cancel_check():
            raise UserCancelledError("Cancelled")
        pipe = self.get_texture_pipeline()
        if pipe is None:
            return bgr_img
        if PILImage is None:
            raise RuntimeError("Pillow not installed")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        init_image = PILImage.fromarray(rgb_img)
        result = pipe(
            prompt=TEXTURE_PROMPT, image=init_image,
            strength=TEXTURE_STRENGTH, guidance_scale=TEXTURE_GUIDANCE,
            num_inference_steps=TEXTURE_STEPS,
        ).images[0]
        if cancel_check and cancel_check():
            raise UserCancelledError("Cancelled")
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    # --- Face detection ---

    def detect_faces(self, gray_img: np.ndarray) -> bool:
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

    # --- Enhance (single image) ---

    def enhance(
        self, img: np.ndarray, scale: int, tile: int = 0
    ) -> Tuple[np.ndarray, Any]:
        """Run RealESRGAN enhancement."""
        with self._model_lock:
            if self.upsampler is None:
                raise RuntimeError("Model not loaded")
            self.upsampler.tile = tile
            upsampler_ref = self.upsampler
        return upsampler_ref.enhance(img, outscale=scale)

    # --- Cleanup ---

    def cleanup(self) -> None:
        self.clear_feature_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
