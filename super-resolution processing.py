import logging
import os
import sys
from typing import Optional, cast

import cv2
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = frozenset({'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'})


def get_image_paths(directory: str) -> list[str]:
    """Return sorted list of image file paths from the given directory."""
    if not os.path.exists(directory):
        logger.error("Directory %s does not exist", directory)
        return []

    try:
        all_files = os.listdir(directory)
    except PermissionError:
        logger.error("Permission error: Cannot access directory %s", directory)
        return []

    image_paths = [
        os.path.join(directory, f)
        for f in all_files
        if os.path.isfile(os.path.join(directory, f))
        and os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
    ]
    return sorted(image_paths)


class ConvBlock(nn.Module):
    """Double convolution block with ReLU activation for UNet encoder/decoder."""

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
    """UNet architecture for scratch detection in historical photographs.

    Input: single-channel grayscale image (1, H, W)
    Output: scratch probability map (1, H, W)
    """

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


def clean_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove 'module.' prefix from state_dict keys."""
    return {key.replace("module.", ""): value for key, value in state_dict.items()}


def load_scratch_model(model_path: str, device: torch.device) -> Optional[ScratchUNet]:
    """Load scratch detection UNet model from checkpoint."""
    if not model_path:
        return None
    if not os.path.exists(model_path):
        logger.warning("Scratch model not found: %s", model_path)
        return None
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as exc:
        logger.error("Failed to load scratch model: %s", exc)
        return None
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model = ScratchUNet()
    missing, unexpected = model.load_state_dict(clean_state_dict(state_dict), strict=False)
    if missing:
        logger.warning("Missing keys in scratch model: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys in scratch model: %s", unexpected)
    model.to(device)
    model.eval()
    return model


def predict_scratch_mask(
    bgr_img: NDArray[np.uint8],
    model: ScratchUNet,
    device: torch.device,
    threshold: float,
) -> Optional[NDArray[np.uint8]]:
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
        mask = cv2.resize(mask, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask = mask.astype(np.uint8)
    return mask


def apply_scratch_repair(
    bgr_img: NDArray[np.uint8],
    model: ScratchUNet,
    device: torch.device,
    threshold: float = 0.5,
    inpaint_radius: int = 3,
) -> NDArray[np.uint8]:
    if model is None:
        return bgr_img
    mask = predict_scratch_mask(bgr_img, model, device, threshold)
    if mask is None or not np.any(mask):
        return bgr_img
    return cv2.inpaint(bgr_img, mask, inpaint_radius, cv2.INPAINT_TELEA)


def esrgan_super_resolution(
    lr_dir: str,
    sr_dir: str,
    model_path: str,
    scale_factor: int = 4,
    use_face_enhance: bool = False,
    tile_size: int = 0,
    use_scratch_repair: bool = False,
    scratch_model_path: str = "",
    scratch_threshold: float = 0.5,
    inpaint_radius: int = 3,
) -> None:
    """
    Perform super-resolution using Real-ESRGAN.

    Parameters:
        lr_dir: Input directory for low-resolution images
        sr_dir: Output directory for super-resolution images
        model_path: Path to RealESRGAN model weights file
        scale_factor: Scaling factor (default 4x)
        use_face_enhance: Whether to use GFPGAN for face enhancement
        tile_size: Tile size for processing, 0 means no tiling
        use_scratch_repair: Whether to apply scratch detection and repair
        scratch_model_path: Path to scratch detection UNet model
        scratch_threshold: Confidence threshold for scratch mask (0.0-1.0)
        inpaint_radius: Radius for inpainting repair (pixels)
    """
    # Create output directory
    os.makedirs(sr_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    # Initialize Real-ESRGAN model (RealESRGAN x4plus standard architecture)
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale_factor
    )

    # Create RealESRGANer upsampler
    upsampler = RealESRGANer(
        scale=scale_factor,
        model_path=model_path,
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=False,  # Use FP32 precision for stability
        device=device
    )
    face_enhancer: Optional[GFPGANer] = None
    if use_face_enhance:
        try:
            gfpgan_model_path = os.environ.get(
                "GFPGAN_MODEL_PATH",
                "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
            )
            face_enhancer = GFPGANer(
                model_path=gfpgan_model_path,
                upscale=scale_factor,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=upsampler,
            )
        except (RuntimeError, OSError, ImportError) as exc:
            logger.warning("Face enhancer init failed (%s); disabling face enhancement.", exc)
            face_enhancer = None

    scratch_model = None
    if use_scratch_repair:
        scratch_model = load_scratch_model(scratch_model_path, device)
        if scratch_model is None:
            logger.warning("Scratch repair disabled (model unavailable)")

    # Get all image files using robust method
    image_paths = get_image_paths(lr_dir)

    if not image_paths:
        logger.warning("No image files found in directory: %s", lr_dir)
        return

    logger.info("Found %d image files to process", len(image_paths))

    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        sr_path = os.path.join(sr_dir, img_name)

        logger.info("Processing image %d/%d: %s", i + 1, len(image_paths), img_name)

        try:
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                logger.warning("Cannot read image: %s", img_name)
                continue
            img = cast(NDArray[np.uint8], img)

            # Handle single-channel images
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # Handle images with alpha channel
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if use_scratch_repair and scratch_model is not None:
                img = apply_scratch_repair(
                    img,
                    scratch_model,
                    device,
                    threshold=scratch_threshold,
                    inpaint_radius=inpaint_radius,
                )

            logger.info("Input image dimensions: %dx%d", img.shape[1], img.shape[0])

            # Perform super-resolution processing
            with torch.no_grad():
                if use_face_enhance and face_enhancer is not None:
                    # Use GFPGAN for face enhancement
                    _, _, output = face_enhancer.enhance(
                        img,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )
                else:
                    # Use Real-ESRGAN for super-resolution
                    output, _ = upsampler.enhance(img, outscale=scale_factor)

            if output is None:
                logger.warning("No output generated for image %s", img_name)
                continue

            # Ensure output is within valid range
            output = output.astype(np.float32)
            output = np.clip(output, 0, 255).astype(np.uint8)

            logger.info("Output image dimensions: %dx%d", output.shape[1], output.shape[0])

            # Save result image
            cv2.imwrite(sr_path, output)
            logger.info("Saved: %s", img_name)

        except (cv2.error, RuntimeError, ValueError, OSError) as e:
            logger.error("Error processing image %s: %s", img_name, e)
            if "CUDA out of memory" in str(e):
                logger.info("Suggestion: Reduce tile_size parameter to decrease GPU memory usage")

    logger.info("Super-resolution processing completed!")


def _safe_float_env(key: str, default: float) -> float:
    """Parse float from environment variable with fallback."""
    try:
        return float(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s, using default %s", key, default)
        return default


def _safe_int_env(key: str, default: int) -> int:
    """Parse int from environment variable with fallback."""
    try:
        return int(os.environ.get(key, str(default)))
    except ValueError:
        logger.warning("Invalid value for %s, using default %s", key, default)
        return default


def main() -> None:
    """Main function to configure and run super-resolution processing."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    base_dir = os.path.dirname(os.path.abspath(__file__))

    esrgan_super_resolution(
        lr_dir=os.environ.get("LR_DIR", os.path.join(base_dir, "LR")),
        sr_dir=os.environ.get("SR_DIR", os.path.join(base_dir, "SR")),
        model_path=os.environ.get(
            "REALESRGAN_MODEL_PATH",
            os.path.join(os.path.expanduser("~"), ".cache", "realesrgan", "RealESRGAN_x4plus.pth"),
        ),
        scale_factor=4,
        use_face_enhance=False,
        tile_size=0,
        use_scratch_repair=False,
        scratch_model_path=os.environ.get("SCRATCH_MODEL_PATH", ""),
        scratch_threshold=_safe_float_env("SCRATCH_MASK_THRESHOLD", 0.5),
        inpaint_radius=_safe_int_env("SCRATCH_INPAINT_RADIUS", 3),
    )


if __name__ == "__main__":
    missing = []
    for pkg in ("basicsr", "realesrgan", "gfpgan", "cv2"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Missing required dependencies: {', '.join(missing)}")
        print("pip install basicsr realesrgan gfpgan opencv-python")
        sys.exit(1)

    main()
