import os
import json
import sys
import contextlib
import warnings
import time
import uuid
import uuid
from datetime import datetime


@contextlib.contextmanager
def suppress_stderr():
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

import cv2
import torch
import torch.nn as nn
import numpy as np
import threading
try:
    from diffusers import StableDiffusionImg2ImgPipeline
except ImportError:
    StableDiffusionImg2ImgPipeline = None
with suppress_stderr():
    import customtkinter as ctk
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
    print("Warning: skimage not installed.")

# --- Global Theme Settings ---
with suppress_stderr():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

TEXTURE_MODEL_ID = os.environ.get("TEXTURE_MODEL_ID", r"D:\Tools\models\stable-diffusion-v1-5").strip()
TEXTURE_PROMPT = os.environ.get(
    "TEXTURE_PROMPT",
    "restored vintage photo, realistic skin texture, fabric detail, subtle film grain"
)
TEXTURE_STRENGTH = float(os.environ.get("TEXTURE_STRENGTH", "0.35"))
TEXTURE_GUIDANCE = float(os.environ.get("TEXTURE_GUIDANCE", "5.0"))
TEXTURE_STEPS = int(os.environ.get("TEXTURE_STEPS", "2"))
TEXTURE_ENABLED = False
SCRATCH_MODEL_PATH = os.environ.get("SCRATCH_MODEL_PATH", "").strip()
SCRATCH_MASK_THRESHOLD = float(os.environ.get("SCRATCH_MASK_THRESHOLD", "0.5"))
SCRATCH_INPAINT_RADIUS = int(os.environ.get("SCRATCH_INPAINT_RADIUS", "3"))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def write_json_file(path, payload):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def timestamp_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_basename(path, fallback="image"):
    if not path:
        return fallback
    return os.path.splitext(os.path.basename(path))[0]


def save_image(path, bgr_img):
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".png", ".jpg", ".jpeg", ".bmp"]:
        ext = ".png"
        path = path + ext
    with suppress_stderr():
        success, buf = cv2.imencode(ext, bgr_img)
    if not success:
        raise RuntimeError("Failed to encode image")
    buf.tofile(path)
    return path


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
        return None
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception:
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


def apply_scratch_repair(bgr_img, model, device, threshold, inpaint_radius):
    if model is None:
        return bgr_img
    mask = predict_scratch_mask(bgr_img, model, device, threshold)
    if mask is None or not np.any(mask):
        return bgr_img
    return cv2.inpaint(bgr_img, mask, inpaint_radius, cv2.INPAINT_TELEA)


def blend_images(img_a, img_b, alpha):
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


def apply_unsharp_mask(bgr_img, strength, radius=1.5):
    weight = float(np.clip(strength, 0.0, 1.0))
    if weight <= 0.0:
        return bgr_img
    blurred = cv2.GaussianBlur(bgr_img, (0, 0), radius)
    sharpened = cv2.addWeighted(bgr_img, 1.0 + weight, blurred, -weight, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_film_grain(bgr_img, strength):
    weight = float(np.clip(strength, 0.0, 1.0))
    if weight <= 0.0:
        return bgr_img
    h, w = bgr_img.shape[:2]
    sigma = 12.0 * weight
    noise = np.random.normal(0.0, sigma, (h, w, 1)).astype(np.float32)
    grain = bgr_img.astype(np.float32) + noise
    return np.clip(grain, 0, 255).astype(np.uint8)


def blend_with_lr(sr_bgr, lr_bgr, strength):
    weight = float(np.clip(strength, 0.0, 1.0))
    if weight <= 0.0:
        return sr_bgr
    h, w = sr_bgr.shape[:2]
    lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
    return blend_images(lr_up, sr_bgr, weight)


def clamp_value(value, min_value, max_value):
    return max(min_value, min(float(value), max_value))


def estimate_image_metrics(bgr_img):
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


def make_comparison_images(lr_bgr, sr_bgr, scale, base_name, out_dir):
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

    lr_crop = lr_up[y1:y1 + crop_size, x1:x1 + crop_size]
    sr_crop = sr_bgr[y1:y1 + crop_size, x1:x1 + crop_size]
    zoom = np.hstack([lr_crop, sr_crop])
    zoom_vis = cv2.resize(zoom, (w, h), interpolation=cv2.INTER_NEAREST)

    grid_top = np.hstack([lr_up, sr_bgr])
    grid_bottom = np.hstack([heat, zoom_vis])
    grid = np.vstack([grid_top, grid_bottom])
    grid_path = os.path.join(out_dir, f"{base_name}_x{scale}_{ts}_grid.png")
    save_image(grid_path, grid)
    return pair_path, grid_path


def tensor_to_grid_image(tensor, grid=4, max_channels=16):
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
        row = np.hstack(imgs[r * grid:(r + 1) * grid])
        rows.append(row)
    return np.vstack(rows)


def save_feature_grids(feature_maps, base_name, scale, out_dir):
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
    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Image Super-Resolution System (ESRGAN)")
        self.geometry("1300x900")  # Increased height for new controls

        # Core variables
        self.model_folder = os.environ.get(
            "REALESRGAN_MODEL_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "realesrgan"),
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.upsampler = None
        self.model = None
        self.scale_factor = 4  # Default
        self.img_input = None
        self.img_output = None
        self.texture_pipe = None
        self.scratch_model = None
        self.img_gt = None
        self.input_path = None
        self.is_processing = False
        self.last_run_dir = None
        self.last_run_id = None
        self.face_blend = ctk.DoubleVar(value=0.7)
        self.natural_blend = ctk.DoubleVar(value=0.15)
        self.texture_boost = ctk.DoubleVar(value=0.2)
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
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(self.project_dir, "output_config.json")
        self.default_output_dir = None
        self.load_config()

        # Layout configuration
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.setup_ui()

        # Start background model loading (Default load x4)
        self.status_label.configure(text=f"Initializing core components ({self.device})...")
        self.progress_bar.set(0.5)
        threading.Thread(target=self.load_model, daemon=True).start()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_ui(self):
        # === 1. Sidebar (Left) ===
        self.sidebar = ctk.CTkFrame(self, width=240, corner_radius=0)
        self.sidebar.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar.grid_rowconfigure(25, weight=1)

        # Logo / Title
        self.logo_label = ctk.CTkLabel(self.sidebar, text="Super Resolution", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Controls
        self.btn_load = ctk.CTkButton(self.sidebar, text="Open Image", command=self.load_input_image, height=40)
        self.btn_load.grid(row=1, column=0, padx=20, pady=10)

        self.btn_gt = ctk.CTkButton(self.sidebar, text="Load Ground Truth", command=self.load_gt_image,
                                    fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), height=40)
        self.btn_gt.grid(row=2, column=0, padx=20, pady=10)

        self.separator = ctk.CTkProgressBar(self.sidebar, height=2, progress_color="gray")
        self.separator.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.separator.set(1)

        # === New Scale Selector ===
        ctk.CTkLabel(self.sidebar, text="Upscale Factor:", font=ctk.CTkFont(size=12, weight="bold")).grid(row=4,
                                                                                                          column=0,
                                                                                                          padx=20,
                                                                                                          pady=(10, 0),
                                                                                                          sticky="w")
        self.scale_var = ctk.StringVar(value="x4")
        self.scale_combo = ctk.CTkComboBox(self.sidebar, values=["x2", "x4"], variable=self.scale_var,
                                           command=self.change_model_scale)
        self.scale_combo.grid(row=5, column=0, padx=20, pady=(5, 10))

        self.btn_output_dir = ctk.CTkButton(self.sidebar, text="Set Default Output Dir",
                                            command=self.set_default_output_dir, height=36)
        self.btn_output_dir.grid(row=6, column=0, padx=20, pady=(10, 4))
        self.lbl_output_dir = ctk.CTkLabel(self.sidebar, text=self.get_output_dir_label_text(),
                                           font=ctk.CTkFont(size=11), wraplength=200, justify="left")
        self.lbl_output_dir.grid(row=7, column=0, padx=20, pady=(0, 10), sticky="w")

        # Face Enhance Switch
        self.use_face_enhance = ctk.BooleanVar(value=False)
        self.switch_face = ctk.CTkSwitch(self.sidebar, text="Face Enhancement", variable=self.use_face_enhance)
        self.switch_face.grid(row=8, column=0, padx=20, pady=10, sticky="w")

        self.lbl_face_blend = ctk.CTkLabel(self.sidebar, text=f"Face Blend: {self.face_blend.get():.2f}")
        self.lbl_face_blend.grid(row=9, column=0, padx=20, pady=(0, 4), sticky="w")
        self.lbl_face_blend.grid_remove()
        self.slider_face_blend = ctk.CTkSlider(self.sidebar, from_=0.0, to=1.0, number_of_steps=20,
                                               variable=self.face_blend, command=self.on_face_blend_change)
        self.slider_face_blend.grid(row=10, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.slider_face_blend.grid_remove()

        self.lbl_natural_blend = ctk.CTkLabel(self.sidebar, text=f"Natural Blend: {self.natural_blend.get():.2f}")
        self.lbl_natural_blend.grid(row=11, column=0, padx=20, pady=(0, 4), sticky="w")
        self.lbl_natural_blend.grid_remove()
        self.slider_natural_blend = ctk.CTkSlider(self.sidebar, from_=0.0, to=0.6, number_of_steps=12,
                                                  variable=self.natural_blend, command=self.on_natural_blend_change)
        self.slider_natural_blend.grid(row=12, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.slider_natural_blend.grid_remove()

        self.lbl_texture_boost = ctk.CTkLabel(self.sidebar, text=f"Texture Boost: {self.texture_boost.get():.2f}")
        self.lbl_texture_boost.grid(row=13, column=0, padx=20, pady=(0, 4), sticky="w")
        self.lbl_texture_boost.grid_remove()
        self.slider_texture_boost = ctk.CTkSlider(self.sidebar, from_=0.0, to=0.6, number_of_steps=12,
                                                  variable=self.texture_boost, command=self.on_texture_boost_change)
        self.slider_texture_boost.grid(row=14, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.slider_texture_boost.grid_remove()

        self.lbl_film_grain = ctk.CTkLabel(self.sidebar, text=f"Film Grain: {self.film_grain.get():.2f}")
        self.lbl_film_grain.grid(row=15, column=0, padx=20, pady=(0, 4), sticky="w")
        self.lbl_film_grain.grid_remove()
        self.slider_film_grain = ctk.CTkSlider(self.sidebar, from_=0.0, to=0.5, number_of_steps=10,
                                               variable=self.film_grain, command=self.on_film_grain_change)
        self.slider_film_grain.grid(row=16, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.slider_film_grain.grid_remove()

        self.btn_run = ctk.CTkButton(self.sidebar, text="Start Restoration", command=self.run_processing_thread,
                                     fg_color="#2CC985", hover_color="#229A66", height=50,
                                     font=ctk.CTkFont(size=16, weight="bold"))
        self.btn_run.grid(row=18, column=0, padx=20, pady=(10, 10))
        self.btn_run_default_fg = self.btn_run.cget("fg_color")
        self.btn_run_default_hover = self.btn_run.cget("hover_color")

        self.btn_compare = ctk.CTkButton(self.sidebar, text="Save Comparison", command=self.save_comparison,
                                         fg_color="#3A7CA5", hover_color="#2D5F7C", height=40,
                                         font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_compare.grid(row=19, column=0, padx=20, pady=10)
        self.btn_compare.configure(state="disabled")

        self.btn_features = ctk.CTkButton(self.sidebar, text="Export Features", command=self.export_feature_maps,
                                          fg_color="#E0A800", hover_color="#B38600", height=40,
                                          font=ctk.CTkFont(size=14, weight="bold"))
        self.btn_features.grid(row=20, column=0, padx=20, pady=10)
        self.btn_features.configure(state="disabled")

        self.btn_save = ctk.CTkButton(self.sidebar, text="Save Result", command=self.save_result, state="disabled",
                                      height=40)
        self.btn_save.grid(row=21, column=0, padx=20, pady=10)

        self.switch_compare = ctk.CTkSwitch(self.sidebar, text="Compare Slider", variable=self.compare_mode,
                                            command=self.on_compare_mode_toggle)
        self.switch_compare.grid(row=22, column=0, padx=20, pady=(10, 4), sticky="w")
        self.lbl_compare_split = ctk.CTkLabel(self.sidebar, text="Split: 50%")
        self.lbl_compare_split.grid(row=23, column=0, padx=20, pady=(0, 4), sticky="w")
        self.slider_compare = ctk.CTkSlider(self.sidebar, from_=0.0, to=1.0, number_of_steps=20,
                                            variable=self.compare_split, command=self.on_compare_split_change)
        self.slider_compare.grid(row=24, column=0, padx=20, pady=(0, 10), sticky="ew")
        self.slider_compare.configure(state="disabled")
        self.lbl_compare_split.grid_remove()
        self.slider_compare.grid_remove()

        # Metrics Display
        self.metrics_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        self.metrics_frame.grid(row=25, column=0, padx=20, pady=(10, 10), sticky="nw")

        self.lbl_resolution_title = ctk.CTkLabel(
            self.metrics_frame,
            text="Resolution",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_resolution_title.pack(anchor="w")
        self.lbl_resolution_in = ctk.CTkLabel(self.metrics_frame, text="Input: -- x --", font=ctk.CTkFont(size=15))
        self.lbl_resolution_in.pack(anchor="w")
        self.lbl_resolution_out = ctk.CTkLabel(self.metrics_frame, text="Output: -- x --", font=ctk.CTkFont(size=15))
        self.lbl_resolution_out.pack(anchor="w", pady=(0, 6))
        self.lbl_resolution_title.pack_forget()
        self.lbl_resolution_in.pack_forget()
        self.lbl_resolution_out.pack_forget()

        self.lbl_metrics_after = ctk.CTkLabel(
            self.metrics_frame,
            text="Output vs GT",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.lbl_metrics_after.pack(anchor="w")
        self.lbl_psnr_out = ctk.CTkLabel(self.metrics_frame, text="PSNR: --", font=ctk.CTkFont(size=15))
        self.lbl_psnr_out.pack(anchor="w")
        self.lbl_ssim_out = ctk.CTkLabel(self.metrics_frame, text="SSIM: --", font=ctk.CTkFont(size=15))
        self.lbl_ssim_out.pack(anchor="w")
        self.lbl_gt_hint = ctk.CTkLabel(
            self.metrics_frame,
            text="Load Ground Truth to calculate metrics.",
            font=ctk.CTkFont(size=11),
            text_color=("gray40", "gray60")
        )
        self.lbl_gt_hint.pack(anchor="w", pady=(2, 0))

        # === 2. Main Display Area (Right) ===
        self.display_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.display_frame.grid_columnconfigure(0, weight=1, uniform="images", minsize=320)
        self.display_frame.grid_columnconfigure(1, weight=1, uniform="images", minsize=320)
        self.display_frame.grid_rowconfigure(1, weight=1)
        self.display_frame.grid_rowconfigure(2, weight=0)

        # Headers
        ctk.CTkLabel(self.display_frame, text="Original Input", font=ctk.CTkFont(size=14, weight="bold")).grid(row=0,
                                                                                                               column=0,
                                                                                                               pady=5)
        ctk.CTkLabel(self.display_frame, text="Super-Resolution Output", font=ctk.CTkFont(size=14, weight="bold")).grid(
            row=0, column=1, pady=5)

        # Image Containers
        self.frame_input = ctk.CTkFrame(self.display_frame, fg_color=("gray85", "#1C1C1C"))
        self.frame_input.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_output = ctk.CTkFrame(self.display_frame, fg_color=("gray85", "#1C1C1C"))
        self.frame_output.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)

        # Image Labels
        self.lbl_img_in = ctk.CTkLabel(self.frame_input, text="Waiting for input...", corner_radius=6)
        self.lbl_img_in.pack(expand=True, fill="both", padx=10, pady=10)

        self.lbl_img_out = ctk.CTkLabel(self.frame_output, text="Waiting for processing...", corner_radius=6)
        self.lbl_img_out.pack(expand=True, fill="both", padx=10, pady=10)

        self.output_overlay = ctk.CTkFrame(self.frame_output, corner_radius=6, fg_color=("gray85", "#222222"))
        self.output_overlay_label = ctk.CTkLabel(
            self.output_overlay,
            text="Waiting for processing...",
            text_color=("gray20", "gray80"),
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.output_overlay_label.pack(expand=True)
        self.output_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        self.lbl_resolution_in_display = ctk.CTkLabel(self.display_frame, text="Input: -- x --",
                                                     font=ctk.CTkFont(size=12))
        self.lbl_resolution_in_display.grid(row=2, column=0, pady=(0, 5))
        self.lbl_resolution_out_display = ctk.CTkLabel(self.display_frame, text="Output: -- x --",
                                                       font=ctk.CTkFont(size=12))
        self.lbl_resolution_out_display.grid(row=2, column=1, pady=(0, 5))

        self.bind_image_interactions()

        # === 3. Status Bar (Bottom) ===
        self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
        self.status_frame.grid(row=2, column=1, sticky="ew")

        self.status_label = ctk.CTkLabel(self.status_frame, text="Ready", padx=10)
        self.status_label.pack(side="left")

        # Determinate Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self.status_frame, width=300, mode="determinate")
        self.progress_bar.pack(side="right", padx=20, pady=5)
        self.elapsed_label = ctk.CTkLabel(self.status_frame, text="Elapsed: --", padx=10)
        self.elapsed_label.pack(side="right")
        self.progress_bar.set(0)

    # --- Feature Extraction Hooks ---
    def clear_feature_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def register_feature_hooks(self):
        self.clear_feature_hooks()
        self.feature_maps = []

        def make_hook(name):
            def hook(module, input, output):
                if len(self.feature_maps) >= self.max_feature_maps:
                    return
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
                self.feature_maps.append((name, tensor.detach().cpu()))
            return hook

        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.hook_handles.append(module.register_forward_hook(make_hook(name)))

    def load_config(self):
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.default_output_dir = data.get("default_output_dir")
        except Exception:
            self.default_output_dir = None

    def save_config(self):
        data = {"default_output_dir": self.default_output_dir}
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def set_default_output_dir(self):
        selected = filedialog.askdirectory()
        if selected:
            self.default_output_dir = selected
            self.save_config()
            self.status_label.configure(text="Default output directory updated.")
            if hasattr(self, "lbl_output_dir"):
                self.lbl_output_dir.configure(text=self.get_output_dir_label_text())

    def get_output_dir_label_text(self):
        if self.default_output_dir:
            return f"Default output: {self.truncate_path(self.default_output_dir, 40)}"
        return "Default output: (project outputs)"

    def truncate_path(self, path, max_len):
        if len(path) <= max_len:
            return path
        return "..." + path[-(max_len - 3):]

    def on_close(self):
        self.clear_feature_hooks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.destroy()

    # --- Logic: Progress Bar Animation ---
    def auto_increment_progress(self):
        if not self.is_processing:
            return
        current_val = self.progress_bar.get()
        target_val = self.progress_target
        if current_val < target_val:
            increment = min(0.03, target_val - current_val)
            self.progress_bar.set(current_val + increment)
        self.after(120, self.auto_increment_progress)

    # --- Logic: Model Loading & Switching ---
    def change_model_scale(self, choice):
        """Handle scale change event from combobox"""
        self.scale_factor = 2 if choice == "x2" else 4
        self.status_label.configure(text=f"Switching to {choice} model...")
        # Reload model in background to avoid freezing UI
        self.progress_bar.set(0.5)
        threading.Thread(target=self.load_model, daemon=True).start()

    def load_model(self):
        try:
            self.clear_feature_hooks()
            self.model = None
            self.upsampler = None
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
                    f"Model file not found: {model_name}\nPlease download it to: {self.model_folder}")

            self.model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32,
                                 scale=self.scale_factor)
            self.register_feature_hooks()

            self.upsampler = RealESRGANer(scale=self.scale_factor, model_path=full_path, model=self.model, tile=0,
                                          tile_pad=10, pre_pad=0, half=False, device=self.device)

            self.status_label.configure(text=f"Model x{self.scale_factor} loaded | Device: {self.device}")
            self.progress_bar.set(0)
        except Exception as e:
            self.status_label.configure(text="Load failed")
            messagebox.showerror("Model Error", str(e))

    def load_input_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.input_path = path
            self.img_input = self.read_image(path)
            self.reset_view_state()
            status_text = f"Loaded: {os.path.basename(path)} | {self.get_texture_status_text()}"
            self.status_label.configure(text=status_text)
            self.img_gt = None
            self.img_output = None
            self.feature_maps = []
            self.render_main_images()
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

    def load_gt_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.png *.jpeg *.bmp")])
        if path:
            self.img_gt = self.read_image(path)
            self.calculate_metrics()
            messagebox.showinfo("Info", "Ground Truth loaded")

    def read_image(self, path):
        with suppress_stderr():
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    def bind_image_interactions(self):
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

    def reset_view_state(self):
        self.zoom_factor = 1.0
        self.view_center = [0.5, 0.5]
        self.pan_start = None
        self.compare_hold_active = False

    def on_display_resize(self, event):
        if self.resize_job is not None:
            self.after_cancel(self.resize_job)
        self.resize_job = self.after(120, self.render_main_images)

    def on_zoom(self, event):
        if self.img_input is None:
            return
        if event.delta > 0:
            zoom = self.zoom_factor * 1.1
        elif event.delta < 0:
            zoom = self.zoom_factor / 1.1
        else:
            return
        self.zoom_factor = float(np.clip(zoom, 1.0, 6.0))
        self.render_main_images()

    def on_pan_start(self, event):
        if self.img_input is None:
            return
        self.pan_start = (event.x, event.y)

    def on_pan_move(self, event):
        if self.img_input is None or self.pan_start is None:
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

    def on_pan_end(self, event):
        self.pan_start = None

    def on_compare_press(self, event):
        if self.img_input is None or self.img_output is None:
            return
        if self.is_processing:
            return
        if self.compare_mode.get():
            return
        self.compare_hold_active = True
        self.render_main_images()

    def on_compare_release(self, event):
        if not self.compare_hold_active:
            return
        self.compare_hold_active = False
        self.render_main_images()

    def on_compare_leave(self, event):
        if not self.compare_hold_active:
            return
        self.compare_hold_active = False
        self.render_main_images()

    def calculate_view_window(self, bgr_img, widget_w, widget_h):
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

    def render_zoomed_image(self, bgr_img, label_widget):
        if not label_widget.winfo_exists():
            return False
        widget_w = label_widget.winfo_width()
        widget_h = label_widget.winfo_height()
        if widget_w < 10 or widget_h < 10:
            widget_w, widget_h = 500, 500

        margin = int(min(widget_w, widget_h) * 0.08)
        render_w = max(1, widget_w - margin * 2)
        render_h = max(1, widget_h - margin * 2)

        view_w, view_h = self.calculate_view_window(bgr_img, render_w, render_h)
        h_img, w_img = bgr_img.shape[:2]
        cx = int(self.view_center[0] * w_img)
        cy = int(self.view_center[1] * h_img)
        x1 = max(0, min(cx - view_w // 2, w_img - view_w))
        y1 = max(0, min(cy - view_h // 2, h_img - view_h))
        crop = bgr_img[y1:y1 + view_h, x1:x1 + view_w]
        resized = cv2.resize(crop, (render_w, render_h), interpolation=cv2.INTER_CUBIC)
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        ctk_img = ctk.CTkImage(light_image=im_pil, dark_image=im_pil, size=(render_w, render_h))
        try:
            label_widget.configure(image=ctk_img, text="")
            label_widget.image = ctk_img
        except TclError:
            return False
        return True

    def render_main_images(self):
        if self.img_input is None:
            self.lbl_img_in.configure(image=None, text="Waiting for input...")
            self.lbl_img_in.image = None
        else:
            self.render_zoomed_image(self.img_input, self.lbl_img_in)

        if self.compare_mode.get() and self.img_input is not None and self.img_output is not None:
            compare_img = self.build_compare_image(self.img_input, self.img_output, self.compare_split.get())
            self.render_zoomed_image(compare_img, self.lbl_img_out)
            return
        if self.compare_hold_active and self.img_input is not None:
            self.render_zoomed_image(self.img_input, self.lbl_img_out)
            return
        if self.img_output is None:
            self.lbl_img_out.configure(image=None, text="")
            self.lbl_img_out.image = None
            return
        if not self.render_zoomed_image(self.img_output, self.lbl_img_out):
            try:
                self.show_image_ctk(self.img_output, self.lbl_img_out)
            except TclError:
                return

    def force_output_refresh(self):
        if self.img_output is None:
            return
        if not self.lbl_img_out.winfo_exists():
            return
        try:
            self.render_zoomed_image(self.img_output, self.lbl_img_out)
            self.show_image_ctk(self.img_output, self.lbl_img_out)
        except TclError:
            return
        try:
            self.lbl_img_out.update_idletasks()
        except TclError:
            return

    def build_compare_image(self, lr_bgr, sr_bgr, split_ratio):
        if lr_bgr is None:
            return sr_bgr
        if sr_bgr is None:
            return lr_bgr
        h, w = sr_bgr.shape[:2]
        lr_up = cv2.resize(lr_bgr, (w, h), interpolation=cv2.INTER_CUBIC)
        split = int(w * float(np.clip(split_ratio, 0.0, 1.0)))
        combined = sr_bgr.copy()
        if split > 0:
            combined[:, :split] = lr_up[:, :split]
        return combined

    def on_compare_mode_toggle(self):
        self.compare_hold_active = False
        self.update_compare_controls()
        self.render_main_images()

    def on_compare_split_change(self, value):
        percent = int(round(float(value) * 100))
        self.lbl_compare_split.configure(text=f"Split: {percent}%")
        if self.compare_mode.get():
            self.render_main_images()

    def update_compare_controls(self):
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

    def show_output_overlay(self, text, animate=False):
        self.overlay_base_text = text
        self.output_overlay_label.configure(text=text)
        self.output_overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.output_overlay.lift()
        if animate:
            self.animate_output_overlay()

    def hide_output_overlay(self):
        self.output_overlay.place_forget()

    def animate_output_overlay(self):
        if not self.is_processing:
            return
        dots = getattr(self, "overlay_dot_count", 0)
        dots = (dots + 1) % 4
        self.overlay_dot_count = dots
        text = f"{self.overlay_base_text}{'.' * dots}"
        self.output_overlay_label.configure(text=text)
        self.after(500, self.animate_output_overlay)

    def report_progress(self, value, status_text=None, overlay_text=None):
        def update():
            self.progress_target = max(self.progress_target, value)
            if status_text:
                self.status_label.configure(text=status_text)
            if overlay_text:
                self.overlay_base_text = overlay_text
                self.overlay_dot_count = 0
                self.output_overlay_label.configure(text=overlay_text)
        self.after(0, update)

    def start_elapsed_timer(self):
        self.processing_start_time = time.perf_counter()
        self.update_elapsed_time()

    def update_elapsed_time(self):
        if not self.is_processing or self.processing_start_time is None:
            return
        elapsed = time.perf_counter() - self.processing_start_time
        self.elapsed_label.configure(text=f"Elapsed: {elapsed:.1f}s")
        self.after(200, self.update_elapsed_time)

    def set_run_button_processing(self, processing):
        if processing:
            self.btn_run.configure(
                state="disabled",
                text="Processing...",
                fg_color=("gray70", "#444444"),
                hover_color=("gray70", "#444444")
            )
        else:
            self.btn_run.configure(
                state="normal",
                text="Start Restoration",
                fg_color=self.btn_run_default_fg,
                hover_color=self.btn_run_default_hover
            )

    def show_image_ctk(self, cv_img, label_widget):
        img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)

        w_widget = label_widget.winfo_width()
        h_widget = label_widget.winfo_height()
        if w_widget < 10: w_widget = 500
        if h_widget < 10: h_widget = 500

        w_img, h_img = im_pil.size
        ratio = min(w_widget / w_img, h_widget / h_img)
        ratio = min(ratio, 1.0) if ratio > 1 else ratio

        new_w = int(w_img * ratio)
        new_h = int(h_img * ratio)

        ctk_img = ctk.CTkImage(light_image=im_pil, dark_image=im_pil, size=(new_w, new_h))

        label_widget.configure(image=ctk_img, text="")
        label_widget.image = ctk_img

    def show_image_preview(self, title, bgr_img, info_text, save_text, on_save):
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
        ctk_img = ctk.CTkImage(light_image=im_pil, dark_image=im_pil, size=(new_w, new_h))
        lbl.configure(image=ctk_img)
        lbl.image = ctk_img

        if info_text:
            info_label = ctk.CTkLabel(preview, text=info_text)
            info_label.pack(pady=(0, 10))

        btn_save = ctk.CTkButton(preview, text=save_text, command=lambda: on_save(preview))
        btn_save.pack(pady=(0, 10))

    def update_slider_label(self, label_widget, prefix, value):
        label_widget.configure(text=f"{prefix}: {float(value):.2f}")

    def on_face_blend_change(self, value):
        self.update_slider_label(self.lbl_face_blend, "Face Blend", value)

    def on_natural_blend_change(self, value):
        self.update_slider_label(self.lbl_natural_blend, "Natural Blend", value)

    def on_texture_boost_change(self, value):
        self.update_slider_label(self.lbl_texture_boost, "Texture Boost", value)

    def on_film_grain_change(self, value):
        self.update_slider_label(self.lbl_film_grain, "Film Grain", value)

    def get_texture_status_text(self):
        if TEXTURE_ENABLED and TEXTURE_MODEL_ID:
            return "Texture gen: on"
        return "Texture gen: off (disabled)"

    def detect_faces(self, gray_img):
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            return False
        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return False
        faces = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
        return len(faces) > 0

    def auto_tune_parameters(self):
        if self.img_input is None:
            return
        try:
            metrics = estimate_image_metrics(self.img_input)
            sharpness_norm = clamp_value((metrics["lap_var"] - 20.0) / 380.0, 0.0, 1.0)
            noise_norm = clamp_value((metrics["noise_sigma"] - 2.0) / 18.0, 0.0, 1.0)
            contrast_norm = clamp_value((metrics["contrast"] - 20.0) / 60.0, 0.0, 1.0)
            edge_norm = clamp_value((metrics["edge_density"] - 0.02) / 0.08, 0.0, 1.0)

            face_blend = clamp_value(0.6 + sharpness_norm * 0.2 - noise_norm * 0.1, 0.4, 0.9)
            natural_blend = clamp_value(0.12 + noise_norm * 0.25 + (1.0 - contrast_norm) * 0.15, 0.0, 0.6)
            texture_boost = clamp_value(0.12 + (1.0 - sharpness_norm) * 0.25 + edge_norm * 0.1 - noise_norm * 0.1,
                                        0.0, 0.6)
            film_grain = clamp_value(0.03 + (1.0 - edge_norm) * 0.12 + (1.0 - contrast_norm) * 0.08, 0.0, 0.5)

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

    def get_texture_pipeline(self):
        if not TEXTURE_ENABLED or not TEXTURE_MODEL_ID:
            return None
        if StableDiffusionImg2ImgPipeline is None:
            raise RuntimeError("diffusers not installed. Run: pip install diffusers transformers accelerate")
        if self.texture_pipe is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            self.texture_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                TEXTURE_MODEL_ID,
                torch_dtype=dtype
            )
            self.texture_pipe.to(self.device)
            if self.device.type == "cuda":
                self.texture_pipe.enable_attention_slicing()
        return self.texture_pipe

    def apply_texture_generation(self, bgr_img):
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
            num_inference_steps=TEXTURE_STEPS
        ).images[0]
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

    def run_processing_thread(self):
        if self.is_processing:
            return
        if self.img_input is None:
            self.status_label.configure(text="Load an input image first.")
            messagebox.showinfo("Info", "Please load an input image first.")
            return
        if self.upsampler is None:
            self.status_label.configure(text="Model not ready. Loading...")
            threading.Thread(target=self.load_model, daemon=True).start()
            messagebox.showwarning("Model Loading", "Model is still loading. Please try again shortly.")
            return
        self.is_processing = True
        self.progress_bar.set(0)
        self.progress_target = 0.05
        self.feature_maps = []
        self.compare_hold_active = False
        self.start_elapsed_timer()
        self.set_run_button_processing(True)
        self.update_compare_controls()
        self.show_output_overlay("Processing", animate=True)
        self.status_label.configure(text=f"Restoring image (x{self.scale_factor})...")
        self.auto_increment_progress()

        threading.Thread(target=self.process_image, daemon=True).start()

    def process_image(self):
        success = False
        run_id, run_dir = self.start_run_record()
        base_name = safe_basename(self.input_path)
        run_output_path = os.path.join(run_dir, f"{base_name}_x{self.scale_factor}.png")
        run_meta = {
            "run_id": run_id,
            "timestamp": timestamp_str(),
            "input_path": self.input_path,
            "run_dir": run_dir,
            "output_path": run_output_path,
            "scale_factor": self.scale_factor,
            "device": str(self.device),
            "face_enhance": bool(self.use_face_enhance.get()),
            "face_blend": float(self.face_blend.get()),
            "natural_blend": float(self.natural_blend.get()),
            "texture_boost": float(self.texture_boost.get()),
            "film_grain": float(self.film_grain.get()),
            "texture_enabled": bool(TEXTURE_ENABLED),
            "texture_model": TEXTURE_MODEL_ID,
            "scratch_model": SCRATCH_MODEL_PATH,
        }
        self.after(0, lambda: self.status_label.configure(text=f"Run {run_id} started..."))
        try:
            output = None
            used_face_enhance = False

            self.report_progress(0.15, f"Upscaling image (x{self.scale_factor})...", "Upscaling")
            sr_base, _ = self.upsampler.enhance(self.img_input, outscale=self.scale_factor)
            output = sr_base
            self.report_progress(0.45, "Upscale complete. Refining details...", "Refining")

            if self.use_face_enhance.get():
                self.report_progress(0.55, "Applying face enhancement...", "Face enhancement")
                try:
                    from gfpgan import GFPGANer
                    face_enhancer = GFPGANer(
                        model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth',
                        upscale=self.scale_factor, arch='clean', channel_multiplier=2, bg_upsampler=self.upsampler)
                    _, _, face_output = face_enhancer.enhance(self.img_input, has_aligned=False, only_center_face=False,
                                                              paste_back=True)
                    if face_output is not None:
                        output = blend_images(face_output, sr_base, self.face_blend.get())
                        used_face_enhance = True
                except Exception as e:
                    print(f"Warning: Face enhance failed ({e}), switching to standard mode.")
                    self.after(0, lambda: self.status_label.configure(
                        text="Face enhancement unavailable, switching to standard mode..."))

            self.report_progress(0.7, "Blending fine details...", "Blending")
            output = blend_with_lr(output, self.img_input, self.natural_blend.get())
            output = apply_unsharp_mask(output, self.texture_boost.get())
            try:
                self.report_progress(0.82, "Generating texture details...", "Texture refinement")
                output = self.apply_texture_generation(output)
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(text=f"Texture generation skipped: {e}"))
            self.report_progress(0.92, "Finalizing output...", "Finalizing")
            output = apply_film_grain(output, self.film_grain.get())

            self.img_output = output
            success = True
            save_image(run_output_path, self.img_output)
            run_input_path = os.path.join(run_dir, f"{base_name}_input.png")
            save_image(run_input_path, self.img_input)
            run_meta["input_snapshot"] = run_input_path

            self.compare_hold_active = False

            self.after(0, self.hide_output_overlay)
            self.after(0, self.render_main_images)
            self.after(0, self.update_idletasks)
            self.after(80, self.render_main_images)
            self.after(140, self.force_output_refresh)
            self.after(0, self.update_compare_controls)
            self.after(0, self.update_resolution_labels)
            self.after(0, self.calculate_metrics)

            if self.use_face_enhance.get() and not used_face_enhance:
                self.after(0, lambda: self.status_label.configure(text=f"Done (x{self.scale_factor} Standard Mode)"))
            else:
                self.after(0, lambda: self.status_label.configure(text=f"Done (x{self.scale_factor})"))

            self.after(0, lambda: self.btn_save.configure(state="normal"))
            self.after(0, lambda: self.btn_compare.configure(state="normal"))
            if self.feature_maps:
                self.after(0, lambda: self.btn_features.configure(state="normal"))

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
            self.after(0, lambda: self.show_output_overlay("Processing failed", animate=False))
            run_meta["error"] = str(e)
        finally:
            self.is_processing = False
            elapsed = None
            if self.processing_start_time is not None:
                elapsed = time.perf_counter() - self.processing_start_time
            if elapsed is not None:
                run_meta["elapsed_sec"] = round(elapsed, 3)
            if self.img_input is not None:
                h, w = self.img_input.shape[:2]
                run_meta["input_size"] = [int(w), int(h)]
            if self.img_output is not None:
                h, w = self.img_output.shape[:2]
                run_meta["output_size"] = [int(w), int(h)]
            self.write_run_log(run_dir, run_meta)
            self.after(0, lambda: self.progress_bar.set(1.0))
            self.after(0, lambda: self.set_run_button_processing(False))
            if elapsed is not None:
                self.after(0, lambda: self.elapsed_label.configure(text=f"Elapsed: {elapsed:.1f}s"))
            if success and elapsed is not None:
                self.after(0, lambda: messagebox.showinfo(
                    "Completed",
                    f"Restoration finished in {elapsed:.1f}s."
                ))

    def update_resolution_labels(self):
        if self.img_input is None:
            input_text = "Input: -- x --"
        else:
            h, w = self.img_input.shape[:2]
            input_text = f"Input: {w} x {h}"
        self.lbl_resolution_in.configure(text=input_text)
        self.lbl_resolution_in_display.configure(text=input_text)
        if self.img_output is None:
            output_text = "Output: -- x --"
        else:
            h, w = self.img_output.shape[:2]
            output_text = f"Output: {w} x {h}"
        self.lbl_resolution_out.configure(text=output_text)
        self.lbl_resolution_out_display.configure(text=output_text)

    def set_metric_labels(self, psnr_label, ssim_label, psnr_value, ssim_value):
        if psnr_value is None or ssim_value is None:
            neutral = ("gray20", "gray70")
            psnr_label.configure(text="PSNR: --", text_color=neutral)
            ssim_label.configure(text="SSIM: --", text_color=neutral)
            return
        psnr_label.configure(text=f"PSNR: {psnr_value:.2f} dB", text_color="#2CC985")
        ssim_label.configure(text=f"SSIM: {ssim_value:.4f}", text_color="#2CC985")

    def calculate_metrics(self):
        if psnr is None or ssim is None:
            self.set_metric_labels(self.lbl_psnr_out, self.lbl_ssim_out, None, None)
            self.lbl_gt_hint.pack(anchor="w", pady=(2, 0))
            return
        if self.img_gt is None:
            self.set_metric_labels(self.lbl_psnr_out, self.lbl_ssim_out, None, None)
            self.lbl_gt_hint.pack(anchor="w", pady=(2, 0))
            return
        if self.img_output is None:
            self.set_metric_labels(self.lbl_psnr_out, self.lbl_ssim_out, None, None)
            self.lbl_gt_hint.pack(anchor="w", pady=(2, 0))
            return

        self.lbl_gt_hint.pack_forget()

        h, w = self.img_output.shape[:2]
        img_gt_out = cv2.resize(self.img_gt, (w, h))
        s_psnr_out = psnr(img_gt_out, self.img_output, data_range=255)
        s_ssim_out = ssim(img_gt_out, self.img_output, data_range=255, channel_axis=2)

        self.set_metric_labels(self.lbl_psnr_out, self.lbl_ssim_out, s_psnr_out, s_ssim_out)

    def start_run_record(self):
        run_id = uuid.uuid4().hex[:8]
        base_name = safe_basename(self.input_path)
        run_root = self.get_output_dir(f"runs({base_name})")
        run_dir = os.path.join(run_root, f"{timestamp_str()}_{base_name}_{run_id}")
        ensure_dir(run_dir)
        self.last_run_dir = run_dir
        self.last_run_id = run_id
        return run_id, run_dir

    def write_run_log(self, run_dir, payload):
        log_path = os.path.join(run_dir, "run_log.json")
        write_json_file(log_path, payload)
        return log_path

    def get_output_dir(self, subdir, prompt=False):
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

    def save_comparison(self):
        if self.img_input is None or self.img_output is None:
            return
        base_name = safe_basename(self.input_path)
        try:
            lr_h, lr_w = self.img_input.shape[:2]
            sr_h, sr_w = self.img_output.shape[:2]
            lr_up = cv2.resize(self.img_input, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
            preview = np.hstack([lr_up, self.img_output])
        except Exception as e:
            messagebox.showerror("Error", f"Preview failed: {e}")
            return

        def on_save(preview_window):
            out_dir = self.get_output_dir("compare", prompt=True)
            try:
                pair_path, grid_path = make_comparison_images(self.img_input, self.img_output, self.scale_factor,
                                                              base_name, out_dir)
                messagebox.showinfo("Saved", f"Comparison images saved:\n{pair_path}\n{grid_path}")
                preview_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Save comparison failed: {e}")

        info_text = "Preview shows LR (upscaled) | SR"
        self.show_image_preview("Comparison Preview", preview, info_text, "Save Comparison", on_save)

    def export_feature_maps(self):
        if not self.feature_maps:
            messagebox.showinfo("Info", "No feature maps captured.")
            return
        base_name = safe_basename(self.input_path)
        grids = []
        for name, tensor in self.feature_maps:
            grid_img = tensor_to_grid_image(tensor)
            if grid_img is not None:
                grids.append(grid_img)
        if not grids:
            messagebox.showinfo("Info", "No feature grids generated.")
            return

        def on_save(preview_window):
            try:
                out_dir = self.get_output_dir("features", prompt=True)
                saved = save_feature_grids(self.feature_maps, base_name, self.scale_factor, out_dir)
                if saved:
                    messagebox.showinfo("Saved", f"Feature grids saved: {len(saved)}")
                else:
                    messagebox.showinfo("Info", "No feature grids saved (empty tensors).")
                preview_window.destroy()
            except Exception as e:
                messagebox.showerror("Error", f"Export features failed: {e}")

        info_text = f"Captured feature maps: {len(grids)}"
        self.show_image_preview("Feature Preview", grids[0], info_text, "Save All", on_save)

    def save_result(self):
        if self.img_output is None:
            return

        def on_save(preview_window):
            path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg")])
            if path:
                save_image(path, self.img_output)
                messagebox.showinfo("Saved", "Image saved successfully")
                preview_window.destroy()

        self.show_image_preview("Result Preview", self.img_output, None, "Save As", on_save)


if __name__ == "__main__":
    app = ModernApp()
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user.")
    except Exception as e:
        print(f"Unexpected error: {e}")
