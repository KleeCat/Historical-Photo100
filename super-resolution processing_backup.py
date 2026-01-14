import os
import cv2
import torch
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


def esrgan_super_resolution(lr_dir, sr_dir, model_path, scale_factor=4, use_face_enhance=False, tile_size=0):
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
        "tile_size": 0  # Tile size, 0 means no tiling (set to 400-800 if GPU memory is insufficient)
    }

    # Run super-resolution processing
    esrgan_super_resolution(
        lr_dir=CONFIG["lr_dir"],
        sr_dir=CONFIG["sr_dir"],
        model_path=CONFIG["model_path"],
        scale_factor=CONFIG["scale_factor"],
        use_face_enhance=CONFIG["use_face_enhance"],
        tile_size=CONFIG["tile_size"]
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