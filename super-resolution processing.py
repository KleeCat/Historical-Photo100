import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer


def get_image_paths(directory):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    image_paths = []

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return image_paths

    # Get all files in directory
    try:
        all_files = os.listdir(directory)
    except PermissionError:
        print(f"Permission error: Cannot access directory {directory}")
        return image_paths

    # Filter image files
    for filename in all_files:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):  # Ensure it's a file not a directory
            _, ext = os.path.splitext(filename)
            if ext.lower() in image_extensions:
                image_paths.append(filepath)

    return image_paths


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ScratchUNet(nn.Module):
    def __init__(self):
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

    def forward(self, x):
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


def clean_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        cleaned[key.replace("module.", "")] = value
    return cleaned


def load_scratch_model(model_path, device):
    if not model_path:
        return None
    if not os.path.exists(model_path):
        print(f"Scratch model not found: {model_path}")
        return None
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as exc:
        print(f"Failed to load scratch model: {exc}")
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


def predict_scratch_mask(bgr_img, model, device, threshold):
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
    return mask


def apply_scratch_repair(bgr_img, model, device, threshold=0.5, inpaint_radius=3):
    if model is None:
        return bgr_img
    mask = predict_scratch_mask(bgr_img, model, device, threshold)
    if mask is None or not np.any(mask):
        return bgr_img
    return cv2.inpaint(bgr_img, mask, inpaint_radius, cv2.INPAINT_TELEA)


def esrgan_super_resolution(
    lr_dir,
    sr_dir,
    model_path,
    scale_factor=4,
    use_face_enhance=False,
    tile_size=0,
    use_scratch_repair=False,
    scratch_model_path="",
    scratch_threshold=0.5,
    inpaint_radius=3,
):
    """
    Perform super-resolution using Real-ESRGAN

    Parameters:
        lr_dir: Input directory for low-resolution images
        sr_dir: Output directory for super-resolution images
        model_path: Path to RealESRGAN model weights file
        scale_factor: Scaling factor (default 4x)
        use_face_enhance: Whether to use GFPGAN for face enhancement
        tile_size: Tile size for processing, 0 means no tiling
    """
    # Create output directory
    os.makedirs(sr_dir, exist_ok=True)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize Real-ESRGAN model
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
    face_enhancer = None

    scratch_model = None
    if use_scratch_repair:
        scratch_model = load_scratch_model(scratch_model_path, device)
        if scratch_model is None:
            print("Scratch repair disabled (model unavailable)")

    # Get all image files using robust method
    image_paths = get_image_paths(lr_dir)

    if not image_paths:
        print(f"No image files found in directory: {lr_dir}")
        return

    print(f"Found {len(image_paths)} image files to process")

    for i, img_path in enumerate(image_paths):
        img_name = os.path.basename(img_path)
        sr_path = os.path.join(sr_dir, img_name)

        print(f"Processing image {i + 1}/{len(image_paths)}: {img_name}")

        try:
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"Cannot read image: {img_name}")
                continue

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

            print(f"Input image dimensions: {img.shape[1]}x{img.shape[0]}")

            # Perform super-resolution processing
            with torch.no_grad():
                if face_enhancer is not None and use_face_enhance:
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

            # Ensure output is within valid range
            output = output.astype(np.float32)
            output = np.clip(output, 0, 255).astype(np.uint8)

            print(f"Output image dimensions: {output.shape[1]}x{output.shape[0]}")

            # Save result image
            cv2.imwrite(sr_path, output)
            print(f"✓ Saved: {img_name}")

        except Exception as e:
            print(f"✗ Error processing image {img_name}: {str(e)}")
            # If it's due to insufficient GPU memory, suggest reducing tile_size
            if "CUDA out of memory" in str(e):
                print("Suggestion: Reduce tile_size parameter to decrease GPU memory usage")

    print("Super-resolution processing completed!")


def main():
    """Main function to configure and run super-resolution processing"""
    # Configuration parameters
    CONFIG = {
        "lr_dir": r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\LR",  # Low-resolution image directory
        "sr_dir": r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\SR",  # Super-resolution output directory
        "model_path": r"C:\Users\ihggk\.cache\realesrgan\RealESRGAN_x4plus.pth",  # Model path (updated as requested)
        "scale_factor": 4,  # Scaling factor
        "use_face_enhance": False,  # Whether to enable face enhancement
        "tile_size": 0,  # Tile size, 0 means no tiling (set to 400-800 if GPU memory is insufficient)
        "use_scratch_repair": False,
        "scratch_model_path": os.environ.get("SCRATCH_MODEL_PATH", ""),
        "scratch_threshold": float(os.environ.get("SCRATCH_MASK_THRESHOLD", "0.5")),
        "inpaint_radius": int(os.environ.get("SCRATCH_INPAINT_RADIUS", "3")),
    }

    # Run super-resolution processing
    esrgan_super_resolution(
        lr_dir=CONFIG["lr_dir"],
        sr_dir=CONFIG["sr_dir"],
        model_path=CONFIG["model_path"],
        scale_factor=CONFIG["scale_factor"],
        use_face_enhance=CONFIG["use_face_enhance"],
        tile_size=CONFIG["tile_size"],
        use_scratch_repair=CONFIG["use_scratch_repair"],
        scratch_model_path=CONFIG["scratch_model_path"],
        scratch_threshold=CONFIG["scratch_threshold"],
        inpaint_radius=CONFIG["inpaint_radius"],
    )


if __name__ == "__main__":
    # Check if required libraries are installed
    try:
        import basicsr
        from realesrgan import RealESRGANer
        from gfpgan import GFPGANer
    except ImportError as e:
        print("Missing required dependencies, please install:")
        print("pip install basicsr")
        print("pip install realesrgan")
        print("pip install gfpgan")  # Optional, for face enhancement
        print("pip install opencv-python")
        exit(1)

    main()
