import os
import cv2
import time
import math
import csv
import queue
import threading
import traceback
import numpy as np
import torch

from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from PIL import Image, ImageTk
    PIL_OK = True
except Exception:
    PIL_OK = False

try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    ESRGAN_OK = True
except Exception:
    ESRGAN_OK = False

try:
    from gfpgan import GFPGANer
    GFPGAN_OK = True
except Exception:
    GFPGAN_OK = False


# =========================
# 默认路径（已替换为你的）
# =========================
DEFAULT_PROJECT_ROOT = r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100"
DEFAULT_LR_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "LR")
DEFAULT_SR_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "SR")
DEFAULT_HR_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "HR")
DEFAULT_EVAL_DIR = os.path.join(DEFAULT_PROJECT_ROOT, "evaluation_results")

# 你原来用的 RealESRGAN_x4plus.pth 路径（如果不一样你再改）
DEFAULT_REALESRGAN_MODEL = r"C:\Users\ihggk\.cache\realesrgan\RealESRGAN_x4plus.pth"


# =========================
# Utils
# =========================
def get_image_paths(directory: str) -> List[str]:
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
    image_paths: List[str] = []
    if not directory or (not os.path.isdir(directory)):
        return image_paths

    try:
        all_files = os.listdir(directory)
    except PermissionError:
        return image_paths

    for filename in all_files:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            _, ext = os.path.splitext(filename)
            if ext.lower() in image_extensions:
                image_paths.append(filepath)

    image_paths.sort()
    return image_paths


def safe_imread_bgr(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def metric_laplacian_var(img_bgr: np.ndarray) -> float:
    g = gray(img_bgr)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def metric_tenengrad(img_bgr: np.ndarray) -> float:
    g = gray(img_bgr).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag2 = gx * gx + gy * gy
    return float(np.mean(mag2))


def metric_entropy(img_bgr: np.ndarray) -> float:
    g = gray(img_bgr)
    hist = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
    s = hist.sum()
    if s <= 0:
        return 0.0
    p = hist / s
    p = p[p > 0]
    ent = -np.sum(p * np.log2(p))
    return float(ent)


def metric_edge_density(img_bgr: np.ndarray) -> float:
    g = gray(img_bgr)
    edges = cv2.Canny(g, 80, 160)
    return float(np.count_nonzero(edges)) / float(edges.size)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10((255.0 * 255.0) / mse)


def ssim_gray(a_gray: np.ndarray, b_gray: np.ndarray) -> float:
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    a = a_gray.astype(np.float32)
    b = b_gray.astype(np.float32)

    mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = numerator / (denominator + 1e-12)
    return float(np.mean(ssim_map))


def ssim_bgr(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    return ssim_gray(gray(a_bgr), gray(b_bgr))


def resize_to(img: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    if img.shape[0] == th and img.shape[1] == tw:
        return img
    return cv2.resize(img, (tw, th), interpolation=cv2.INTER_CUBIC)


def bicubic_upscale(lr_bgr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    oh, ow = out_hw
    return cv2.resize(lr_bgr, (ow, oh), interpolation=cv2.INTER_CUBIC)


def fmt_float(x: Optional[float], nd=4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def export_metrics_csv(path: str, rows: List["ImageMetrics"]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        header = [
            "filename", "time_sec",
            "w_in", "h_in", "w_out", "h_out",
            "sharp_lr_up", "sharp_sr", "delta_sharp",
            "ten_lr_up", "ten_sr", "delta_ten",
            "ent_lr_up", "ent_sr", "delta_ent",
            "edge_lr_up", "edge_sr", "delta_edge",
            "psnr_gt", "ssim_gt"
        ]
        w.writerow(header)
        for m in rows:
            w.writerow([
                m.filename, m.time_sec,
                m.w_in, m.h_in, m.w_out, m.h_out,
                m.sharp_lr_up, m.sharp_sr, m.delta_sharp,
                m.ten_lr_up, m.ten_sr, m.delta_ten,
                m.ent_lr_up, m.ent_sr, m.delta_ent,
                m.edge_lr_up, m.edge_sr, m.delta_edge,
                m.psnr_gt if m.psnr_gt is not None else "",
                m.ssim_gt if m.ssim_gt is not None else "",
            ])


# =========================
# Config & Metrics
# =========================
@dataclass
class SRConfig:
    project_root: str = ""

    lr_dir: str = ""
    sr_dir: str = ""
    gt_dir: str = ""
    eval_dir: str = ""

    realesrgan_model_path: str = ""
    scale_factor: int = 4

    tile_size: int = 0
    tile_pad: int = 10
    pre_pad: int = 0
    half: bool = False

    use_face_enhance: bool = False
    gfpgan_model_path: str = ""

    save_overwrite: bool = True


@dataclass
class ImageMetrics:
    filename: str
    time_sec: float

    w_in: int
    h_in: int
    w_out: int
    h_out: int

    sharp_lr_up: float
    sharp_sr: float
    delta_sharp: float

    ten_lr_up: float
    ten_sr: float
    delta_ten: float

    ent_lr_up: float
    ent_sr: float
    delta_ent: float

    edge_lr_up: float
    edge_sr: float
    delta_edge: float

    psnr_gt: Optional[float] = None
    ssim_gt: Optional[float] = None


# =========================
# Worker Thread
# =========================
class SuperResolutionWorker(threading.Thread):
    def __init__(self, config: SRConfig, msg_q: queue.Queue, stop_event: threading.Event):
        super().__init__(daemon=True)
        self.cfg = config
        self.q = msg_q
        self.stop_event = stop_event
        self.metrics: List[ImageMetrics] = []
        self.processed_paths: List[str] = []
        self.auto_csv_path: Optional[str] = None

    def _log(self, text: str):
        self.q.put(("log", text))

    def _progress(self, cur: int, total: int, filename: str):
        self.q.put(("progress", cur, total, filename))

    def _metrics(self, m: ImageMetrics):
        self.q.put(("metrics", m))

    def _done(self, ok: bool, err: Optional[str] = None):
        self.q.put(("done", ok, err, self.metrics, self.processed_paths, self.auto_csv_path))

    def _init_models(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._log(f"Using device: {device}")

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=self.cfg.scale_factor
        )

        upsampler = RealESRGANer(
            scale=self.cfg.scale_factor,
            model_path=self.cfg.realesrgan_model_path,
            model=model,
            tile=self.cfg.tile_size,
            tile_pad=self.cfg.tile_pad,
            pre_pad=self.cfg.pre_pad,
            half=self.cfg.half,
            device=device
        )

        face_enhancer = None
        if self.cfg.use_face_enhance:
            if not GFPGAN_OK:
                raise RuntimeError("启用了人脸修复，但未安装 gfpgan：pip install gfpgan")
            if not self.cfg.gfpgan_model_path or not os.path.exists(self.cfg.gfpgan_model_path):
                raise RuntimeError("启用了人脸修复，但未提供有效的 GFPGAN 模型权重路径。")

            face_enhancer = GFPGANer(
                model_path=self.cfg.gfpgan_model_path,
                upscale=self.cfg.scale_factor,
                arch='clean',
                channel_multiplier=2,
                bg_upsampler=upsampler,
                device=device
            )
            self._log("GFPGAN face enhancer initialized.")

        return upsampler, face_enhancer

    def run(self):
        try:
            os.makedirs(self.cfg.sr_dir, exist_ok=True)
            if self.cfg.eval_dir:
                os.makedirs(self.cfg.eval_dir, exist_ok=True)

            image_paths = get_image_paths(self.cfg.lr_dir)
            if not image_paths:
                self._done(False, f"输入目录没有找到图片：{self.cfg.lr_dir}")
                return

            self._log(f"Found {len(image_paths)} image(s).")
            upsampler, face_enhancer = self._init_models()

            total = len(image_paths)
            for i, img_path in enumerate(image_paths, start=1):
                if self.stop_event.is_set():
                    self._log("Stop requested. Exiting worker...")
                    break

                img_name = os.path.basename(img_path)
                self._progress(i, total, img_name)

                try:
                    img = safe_imread_bgr(img_path)
                    if img is None:
                        self._log(f"Cannot read image: {img_name}")
                        continue

                    h_in, w_in = img.shape[:2]
                    t0 = time.time()

                    with torch.no_grad():
                        if face_enhancer is not None:
                            _, _, out = face_enhancer.enhance(
                                img,
                                has_aligned=False,
                                only_center_face=False,
                                paste_back=True
                            )
                            output = out
                        else:
                            output, _ = upsampler.enhance(img, outscale=self.cfg.scale_factor)

                    output = ensure_uint8(output)
                    h_out, w_out = output.shape[:2]
                    t1 = time.time()

                    sr_path = os.path.join(self.cfg.sr_dir, img_name)
                    if (not self.cfg.save_overwrite) and os.path.exists(sr_path):
                        base, ext = os.path.splitext(img_name)
                        sr_path = os.path.join(self.cfg.sr_dir, f"{base}_sr{ext}")

                    ok = cv2.imwrite(sr_path, output)
                    if not ok:
                        self._log(f"Failed to save: {sr_path}")
                        continue

                    self.processed_paths.append(sr_path)

                    lr_up = bicubic_upscale(img, (h_out, w_out))

                    sharp_lr = metric_laplacian_var(lr_up)
                    sharp_sr = metric_laplacian_var(output)

                    ten_lr = metric_tenengrad(lr_up)
                    ten_sr = metric_tenengrad(output)

                    ent_lr = metric_entropy(lr_up)
                    ent_sr = metric_entropy(output)

                    edge_lr = metric_edge_density(lr_up)
                    edge_sr = metric_edge_density(output)

                    m = ImageMetrics(
                        filename=img_name,
                        time_sec=float(t1 - t0),
                        w_in=w_in, h_in=h_in, w_out=w_out, h_out=h_out,
                        sharp_lr_up=float(sharp_lr),
                        sharp_sr=float(sharp_sr),
                        delta_sharp=float(sharp_sr - sharp_lr),
                        ten_lr_up=float(ten_lr),
                        ten_sr=float(ten_sr),
                        delta_ten=float(ten_sr - ten_lr),
                        ent_lr_up=float(ent_lr),
                        ent_sr=float(ent_sr),
                        delta_ent=float(ent_sr - ent_lr),
                        edge_lr_up=float(edge_lr),
                        edge_sr=float(edge_sr),
                        delta_edge=float(edge_sr - edge_lr),
                        psnr_gt=None,
                        ssim_gt=None
                    )

                    if self.cfg.gt_dir and os.path.isdir(self.cfg.gt_dir):
                        gt_path = os.path.join(self.cfg.gt_dir, img_name)
                        gt = safe_imread_bgr(gt_path)
                        if gt is not None:
                            gt = resize_to(gt, (h_out, w_out))
                            m.psnr_gt = float(psnr(output, gt))
                            m.ssim_gt = float(ssim_bgr(output, gt))

                    self.metrics.append(m)
                    self._metrics(m)

                    msg = f"✓ {img_name} saved. time={m.time_sec:.3f}s  Δsharp={m.delta_sharp:.2f}  Δten={m.delta_ten:.2f}"
                    if m.psnr_gt is not None and m.ssim_gt is not None:
                        msg += f"  PSNR={m.psnr_gt:.2f} SSIM={m.ssim_gt:.4f}"
                    self._log(msg)

                except RuntimeError as e:
                    s = str(e)
                    self._log(f"✗ Error processing {img_name}: {s}")
                    if "CUDA out of memory" in s:
                        self._log("Suggestion: 设置 tile_size 为 200~800 以降低显存占用。")
                except Exception as e:
                    self._log(f"✗ Error processing {img_name}: {e}")
                    self._log(traceback.format_exc())

            if self.cfg.eval_dir and self.metrics:
                self.auto_csv_path = os.path.join(self.cfg.eval_dir, f"metrics_{now_stamp()}.csv")
                try:
                    export_metrics_csv(self.auto_csv_path, self.metrics)
                    self._log(f"Auto exported CSV: {self.auto_csv_path}")
                except Exception as e:
                    self._log(f"Auto export CSV failed: {e}")

            self._done(True, None)

        except Exception as e:
            self._done(False, f"{e}\n{traceback.format_exc()}")


# =========================
# GUI App
# =========================
class ESRGANApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Real-ESRGAN 批量超分 & 修复评估（已写入默认路径）")
        self.root.geometry("1320x820")

        self.msg_q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[SuperResolutionWorker] = None

        self.image_paths: List[str] = []
        self.metrics_rows: List[ImageMetrics] = []

        self._build_ui()
        self._poll_queue()

        # 启动后自动填充你给的路径
        self._auto_fill_from_default()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        self.var_root = tk.StringVar()
        self.var_lr_dir = tk.StringVar()
        self.var_sr_dir = tk.StringVar()
        self.var_gt_dir = tk.StringVar()
        self.var_eval_dir = tk.StringVar()
        self.var_model_path = tk.StringVar()
        self.var_scale = tk.IntVar(value=4)

        self.var_tile = tk.IntVar(value=0)
        self.var_tile_pad = tk.IntVar(value=10)
        self.var_pre_pad = tk.IntVar(value=0)
        self.var_half = tk.BooleanVar(value=False)

        self.var_face = tk.BooleanVar(value=False)
        self.var_gfpgan_path = tk.StringVar(value="")
        self.var_overwrite = tk.BooleanVar(value=True)

        for c in [1, 4]:
            top.columnconfigure(c, weight=1)

        row = 0
        ttk.Label(top, text="项目根目录:").grid(row=row, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_root).grid(row=row, column=1, sticky="ew", padx=6)
        ttk.Button(top, text="选择", command=self._pick_root_dir).grid(row=row, column=2, padx=6)
        ttk.Button(top, text="一键填充(LR/SR/HR/evaluation_results)", command=self._auto_fill).grid(row=row, column=3, columnspan=3, sticky="w")

        row += 1
        ttk.Label(top, text="LR目录:").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_lr_dir).grid(row=row, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(top, text="选择", command=self._pick_lr_dir).grid(row=row, column=2, padx=6, pady=(8, 0))

        ttk.Label(top, text="SR输出目录:").grid(row=row, column=3, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_sr_dir).grid(row=row, column=4, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(top, text="选择", command=self._pick_sr_dir).grid(row=row, column=5, padx=6, pady=(8, 0))

        row += 1
        ttk.Label(top, text="GT/HR目录(可选):").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_gt_dir).grid(row=row, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(top, text="选择", command=self._pick_gt_dir).grid(row=row, column=2, padx=6, pady=(8, 0))

        ttk.Label(top, text="评估输出目录:").grid(row=row, column=3, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_eval_dir).grid(row=row, column=4, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(top, text="选择", command=self._pick_eval_dir).grid(row=row, column=5, padx=6, pady=(8, 0))

        row += 1
        ttk.Label(top, text="RealESRGAN权重:").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_model_path).grid(row=row, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(top, text="选择", command=self._pick_model_path).grid(row=row, column=2, padx=6, pady=(8, 0))

        ttk.Label(top, text="放大倍数:").grid(row=row, column=3, sticky="w", pady=(8, 0))
        scale_box = ttk.Combobox(top, values=[2, 4], width=5, state="readonly")
        scale_box.grid(row=row, column=4, sticky="w", padx=6, pady=(8, 0))
        scale_box.set(str(self.var_scale.get()))
        scale_box.bind("<<ComboboxSelected>>", lambda e: self.var_scale.set(int(scale_box.get())))

        row += 1
        ttk.Label(top, text="tile_size:").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_tile, width=10).grid(row=row, column=1, sticky="w", padx=6, pady=(8, 0))

        ttk.Label(top, text="tile_pad:").grid(row=row, column=1, sticky="e", padx=(0, 6), pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_tile_pad, width=8).grid(row=row, column=2, sticky="w", pady=(8, 0))

        ttk.Label(top, text="pre_pad:").grid(row=row, column=3, sticky="w", pady=(8, 0))
        ttk.Entry(top, textvariable=self.var_pre_pad, width=8).grid(row=row, column=4, sticky="w", padx=6, pady=(8, 0))

        ttk.Checkbutton(top, text="half(FP16)", variable=self.var_half).grid(row=row, column=5, sticky="w", pady=(8, 0))
        ttk.Checkbutton(top, text="覆盖同名输出", variable=self.var_overwrite).grid(row=row, column=5, sticky="e", pady=(8, 0), padx=(0, 10))

        row += 1
        ttk.Checkbutton(top, text="启用GFPGAN人脸修复", variable=self.var_face, command=self._toggle_gfpgan).grid(
            row=row, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(top, text="GFPGAN权重:").grid(row=row, column=1, sticky="e", pady=(8, 0))
        self.ent_gfpgan = ttk.Entry(top, textvariable=self.var_gfpgan_path)
        self.ent_gfpgan.grid(row=row, column=2, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(top, text="选择", command=self._pick_gfpgan_path).grid(row=row, column=3, padx=6, pady=(8, 0), sticky="w")

        row += 1
        ttk.Label(
            top,
            text="说明：有GT/HR会计算PSNR/SSIM；无GT则显示基于 bicubic 基线的提升(Δsharp/Δten/Δedge等)。"
        ).grid(row=row, column=0, columnspan=6, sticky="w", pady=(8, 0))

        self._toggle_gfpgan()

        btns = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        btns.pack(side=tk.TOP, fill=tk.X)

        self.btn_scan = ttk.Button(btns, text="扫描图片", command=self._scan_images)
        self.btn_start = ttk.Button(btns, text="开始处理", command=self._start)
        self.btn_stop = ttk.Button(btns, text="停止", command=self._stop, state="disabled")
        self.btn_export = ttk.Button(btns, text="导出CSV(手动)", command=self._export_csv, state="disabled")
        self.btn_open_out = ttk.Button(btns, text="打开SR输出目录", command=self._open_output_dir)
        self.btn_open_eval = ttk.Button(btns, text="打开评估目录", command=self._open_eval_dir)

        self.btn_scan.pack(side=tk.LEFT, padx=6)
        self.btn_start.pack(side=tk.LEFT, padx=6)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        self.btn_export.pack(side=tk.LEFT, padx=6)
        self.btn_open_out.pack(side=tk.LEFT, padx=6)
        self.btn_open_eval.pack(side=tk.LEFT, padx=6)

        prog = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        prog.pack(side=tk.TOP, fill=tk.X)

        self.var_prog_text = tk.StringVar(value="Idle.")
        ttk.Label(prog, textvariable=self.var_prog_text).pack(side=tk.LEFT)

        self.progress = ttk.Progressbar(prog, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=10)

        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main, padding=10)
        right = ttk.Frame(main, padding=10)
        main.add(left, weight=1)
        main.add(right, weight=2)

        ttk.Label(left, text="图片列表").pack(anchor="w")
        self.listbox = tk.Listbox(left, height=18)
        self.listbox.pack(fill=tk.X, expand=False, pady=(4, 8))
        self.listbox.bind("<<ListboxSelect>>", self._on_select_image)

        ttk.Label(left, text="预览（LR / SR / GT）").pack(anchor="w")
        prev = ttk.Frame(left)
        prev.pack(fill=tk.BOTH, expand=True)

        self.preview_size = (360, 240)

        self.lbl_lr = ttk.Label(prev, text="LR预览\n(需要 Pillow)", anchor="center")
        self.lbl_sr = ttk.Label(prev, text="SR预览\n(需要 Pillow)", anchor="center")
        self.lbl_gt = ttk.Label(prev, text="GT/HR预览(可选)\n(需要 Pillow)", anchor="center")

        self.lbl_lr.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.lbl_sr.grid(row=0, column=1, padx=6, pady=6, sticky="nsew")
        self.lbl_gt.grid(row=0, column=2, padx=6, pady=6, sticky="nsew")

        prev.columnconfigure(0, weight=1)
        prev.columnconfigure(1, weight=1)
        prev.columnconfigure(2, weight=1)
        prev.rowconfigure(0, weight=1)

        self._imgtk_lr = None
        self._imgtk_sr = None
        self._imgtk_gt = None

        ttk.Label(right, text="评估结果（每张图）").pack(anchor="w")

        cols = [
            "filename", "time_sec", "in_wh", "out_wh",
            "sharp_lr_up", "sharp_sr", "delta_sharp",
            "ten_lr_up", "ten_sr", "delta_ten",
            "ent_lr_up", "ent_sr", "delta_ent",
            "edge_lr_up", "edge_sr", "delta_edge",
            "psnr_gt", "ssim_gt"
        ]

        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=12)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=110, anchor="center")
        self.tree.column("filename", width=220, anchor="w")
        self.tree.pack(fill=tk.X, expand=False, pady=(4, 8))

        self.var_summary = tk.StringVar(value="统计：暂无")
        ttk.Label(right, textvariable=self.var_summary).pack(anchor="w", pady=(0, 8))

        ttk.Label(right, text="日志").pack(anchor="w")
        self.txt_log = tk.Text(right, height=15)
        self.txt_log.pack(fill=tk.BOTH, expand=True)

        if not ESRGAN_OK:
            self._append_log("⚠ 未检测到 realesrgan/basicsr 依赖，无法运行。\n"
                             "请安装：pip install basicsr realesrgan\n")
        if not PIL_OK:
            self._append_log("⚠ 未检测到 Pillow，预览功能不可用。\n"
                             "请安装：pip install pillow\n")

    def _append_log(self, s: str):
        self.txt_log.insert(tk.END, s + "\n")
        self.txt_log.see(tk.END)

    def _toggle_gfpgan(self):
        enabled = bool(self.var_face.get())
        self.ent_gfpgan.configure(state=("normal" if enabled else "disabled"))

    def _auto_fill_from_default(self):
        self.var_root.set(DEFAULT_PROJECT_ROOT)
        self.var_lr_dir.set(DEFAULT_LR_DIR)
        self.var_sr_dir.set(DEFAULT_SR_DIR)
        self.var_gt_dir.set(DEFAULT_HR_DIR)
        self.var_eval_dir.set(DEFAULT_EVAL_DIR)
        self.var_model_path.set(DEFAULT_REALESRGAN_MODEL)
        self._append_log("已自动填充默认路径。")

    def _pick_root_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_root.set(d)

    def _auto_fill(self):
        root = self.var_root.get().strip()
        if not root or not os.path.isdir(root):
            messagebox.showwarning("提示", "请先选择有效的项目根目录。")
            return

        self.var_lr_dir.set(os.path.join(root, "LR"))
        self.var_sr_dir.set(os.path.join(root, "SR"))
        self.var_gt_dir.set(os.path.join(root, "HR"))
        self.var_eval_dir.set(os.path.join(root, "evaluation_results"))
        self._append_log("已根据项目根目录一键填充。")

    def _pick_lr_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_lr_dir.set(d)

    def _pick_sr_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_sr_dir.set(d)

    def _pick_gt_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_gt_dir.set(d)

    def _pick_eval_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.var_eval_dir.set(d)

    def _pick_model_path(self):
        p = filedialog.askopenfilename(title="选择 RealESRGAN 权重", filetypes=[("PyTorch weights", "*.pth"), ("All", "*.*")])
        if p:
            self.var_model_path.set(p)

    def _pick_gfpgan_path(self):
        p = filedialog.askopenfilename(title="选择 GFPGAN 权重", filetypes=[("PyTorch weights", "*.pth"), ("All", "*.*")])
        if p:
            self.var_gfpgan_path.set(p)

    def _open_dir(self, d: str):
        if not d:
            return
        if os.path.isdir(d):
            try:
                os.startfile(d)
            except Exception:
                messagebox.showinfo("提示", f"目录：{d}")
        else:
            messagebox.showwarning("提示", f"目录无效：{d}")

    def _open_output_dir(self):
        self._open_dir(self.var_sr_dir.get().strip())

    def _open_eval_dir(self):
        self._open_dir(self.var_eval_dir.get().strip())

    def _scan_images(self):
        lr = self.var_lr_dir.get().strip()
        if not lr or not os.path.isdir(lr):
            messagebox.showwarning("提示", "请先选择有效的LR目录。")
            return

        self.image_paths = get_image_paths(lr)
        self.listbox.delete(0, tk.END)
        for p in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(p))

        self._append_log(f"Scanned {len(self.image_paths)} image(s).")
        self.var_prog_text.set(f"已扫描：{len(self.image_paths)} 张")

    def _on_select_image(self, event=None):
        if not PIL_OK or not self.image_paths:
            return
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        lr_path = self.image_paths[idx]
        img_name = os.path.basename(lr_path)

        sr_path = os.path.join(self.var_sr_dir.get().strip(), img_name) if self.var_sr_dir.get().strip() else ""
        gt_path = os.path.join(self.var_gt_dir.get().strip(), img_name) if self.var_gt_dir.get().strip() else ""

        self._set_preview(self.lbl_lr, lr_path, which="lr")
        self._set_preview(self.lbl_sr, sr_path if os.path.exists(sr_path) else "", which="sr")
        self._set_preview(self.lbl_gt, gt_path if os.path.exists(gt_path) else "", which="gt")

    def _set_preview(self, label: ttk.Label, path: str, which: str):
        if not PIL_OK:
            return
        if not path or (not os.path.exists(path)):
            label.configure(text=f"{which.upper()}预览\n(无文件)")
            label.configure(image="")
            return

        bgr = safe_imread_bgr(path)
        if bgr is None:
            label.configure(text=f"{which.upper()}预览\n(读取失败)")
            label.configure(image="")
            return

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(rgb)
        im.thumbnail(self.preview_size)
        imgtk = ImageTk.PhotoImage(im)
        label.configure(image=imgtk, text="")
        setattr(self, f"_imgtk_{which}", imgtk)

    def _validate(self) -> Optional[SRConfig]:
        if not ESRGAN_OK:
            messagebox.showerror("错误", "缺少 realesrgan/basicsr 依赖，请先安装。")
            return None

        cfg = SRConfig(
            project_root=self.var_root.get().strip(),
            lr_dir=self.var_lr_dir.get().strip(),
            sr_dir=self.var_sr_dir.get().strip(),
            gt_dir=self.var_gt_dir.get().strip(),
            eval_dir=self.var_eval_dir.get().strip(),
            realesrgan_model_path=self.var_model_path.get().strip(),
            scale_factor=int(self.var_scale.get()),
            tile_size=int(self.var_tile.get()),
            tile_pad=int(self.var_tile_pad.get()),
            pre_pad=int(self.var_pre_pad.get()),
            half=bool(self.var_half.get()),
            use_face_enhance=bool(self.var_face.get()),
            gfpgan_model_path=self.var_gfpgan_path.get().strip(),
            save_overwrite=bool(self.var_overwrite.get()),
        )

        if not cfg.lr_dir or not os.path.isdir(cfg.lr_dir):
            messagebox.showwarning("提示", "LR目录无效。")
            return None
        if not cfg.sr_dir:
            messagebox.showwarning("提示", "SR输出目录无效。")
            return None
        if not cfg.realesrgan_model_path or not os.path.exists(cfg.realesrgan_model_path):
            messagebox.showwarning("提示", "RealESRGAN权重路径无效。")
            return None

        if cfg.use_face_enhance:
            if not GFPGAN_OK:
                messagebox.showwarning("提示", "启用GFPGAN需要安装：pip install gfpgan")
                return None
            if not cfg.gfpgan_model_path or not os.path.exists(cfg.gfpgan_model_path):
                messagebox.showwarning("提示", "GFPGAN权重路径无效。")
                return None

        if not cfg.eval_dir:
            cfg.eval_dir = DEFAULT_EVAL_DIR

        return cfg

    def _start(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("提示", "正在处理中。")
            return

        cfg = self._validate()
        if cfg is None:
            return

        os.makedirs(cfg.sr_dir, exist_ok=True)
        os.makedirs(cfg.eval_dir, exist_ok=True)

        self.stop_event.clear()
        self.progress.configure(value=0)
        self.var_prog_text.set("Starting...")
        self.txt_log.delete("1.0", tk.END)

        for i in self.tree.get_children():
            self.tree.delete(i)
        self.metrics_rows.clear()
        self.var_summary.set("统计：暂无")
        self.btn_export.configure(state="disabled")

        self._append_log("Config:")
        for k, v in asdict(cfg).items():
            self._append_log(f"  {k}: {v}")

        self.worker = SuperResolutionWorker(cfg, self.msg_q, self.stop_event)
        self.worker.start()

        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.btn_scan.configure(state="disabled")

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.stop_event.set()
            self._append_log("Stop signal sent.")
            self.btn_stop.configure(state="disabled")

    def _export_csv(self):
        if not self.metrics_rows:
            return

        default_dir = self.var_eval_dir.get().strip() or DEFAULT_EVAL_DIR
        default_name = f"metrics_{now_stamp()}.csv"
        path = filedialog.asksaveasfilename(
            initialdir=default_dir,
            initialfile=default_name,
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            title="保存评估结果CSV"
        )
        if not path:
            return

        try:
            export_metrics_csv(path, self.metrics_rows)
            self._append_log(f"CSV exported: {path}")
            messagebox.showinfo("完成", "CSV导出完成。")
        except Exception as e:
            messagebox.showerror("错误", f"导出失败：{e}")

    def _update_summary(self):
        if not self.metrics_rows:
            self.var_summary.set("统计：暂无")
            return

        times = [m.time_sec for m in self.metrics_rows]
        dsharp = [m.delta_sharp for m in self.metrics_rows]
        dten = [m.delta_ten for m in self.metrics_rows]
        dent = [m.delta_ent for m in self.metrics_rows]
        dedge = [m.delta_edge for m in self.metrics_rows]

        psnrs = [m.psnr_gt for m in self.metrics_rows if m.psnr_gt is not None]
        ssims = [m.ssim_gt for m in self.metrics_rows if m.ssim_gt is not None]

        s = (
            f"统计：数量={len(self.metrics_rows)}  平均耗时={np.mean(times):.3f}s  "
            f"平均Δsharp={np.mean(dsharp):.2f}  平均Δten={np.mean(dten):.2f}  "
            f"平均Δent={np.mean(dent):.3f}  平均Δedge={np.mean(dedge):.6f}"
        )
        if psnrs and ssims:
            s += f"  |  平均PSNR={np.mean(psnrs):.2f}  平均SSIM={np.mean(ssims):.4f}"

        self.var_summary.set(s)

    def _poll_queue(self):
        try:
            while True:
                msg = self.msg_q.get_nowait()
                typ = msg[0]

                if typ == "log":
                    self._append_log(str(msg[1]))

                elif typ == "progress":
                    cur, total, filename = msg[1], msg[2], msg[3]
                    self.progress.configure(maximum=total, value=cur)
                    self.var_prog_text.set(f"{cur}/{total} 处理中：{filename}")

                elif typ == "metrics":
                    m: ImageMetrics = msg[1]
                    self.metrics_rows.append(m)

                    in_wh = f"{m.w_in}x{m.h_in}"
                    out_wh = f"{m.w_out}x{m.h_out}"

                    vals = [
                        m.filename,
                        fmt_float(m.time_sec, 3),
                        in_wh, out_wh,
                        fmt_float(m.sharp_lr_up, 2), fmt_float(m.sharp_sr, 2), fmt_float(m.delta_sharp, 2),
                        fmt_float(m.ten_lr_up, 2), fmt_float(m.ten_sr, 2), fmt_float(m.delta_ten, 2),
                        fmt_float(m.ent_lr_up, 3), fmt_float(m.ent_sr, 3), fmt_float(m.delta_ent, 3),
                        fmt_float(m.edge_lr_up, 6), fmt_float(m.edge_sr, 6), fmt_float(m.delta_edge, 6),
                        fmt_float(m.psnr_gt, 2) if m.psnr_gt is not None else "",
                        fmt_float(m.ssim_gt, 4) if m.ssim_gt is not None else "",
                    ]
                    self.tree.insert("", tk.END, values=vals)
                    self._update_summary()

                elif typ == "done":
                    ok, err, metrics, processed_paths, auto_csv_path = msg[1], msg[2], msg[3], msg[4], msg[5]

                    if err:
                        self._append_log("ERROR:\n" + str(err))

                    if ok:
                        self._append_log("Done.")
                        self.var_prog_text.set("完成。")
                        if auto_csv_path:
                            self._append_log(f"评估CSV已自动保存到：{auto_csv_path}")
                    else:
                        self.var_prog_text.set("失败。")

                    self.btn_start.configure(state="normal")
                    self.btn_stop.configure(state="disabled")
                    self.btn_scan.configure(state="normal")
                    self.btn_export.configure(state="normal" if self.metrics_rows else "disabled")

        except queue.Empty:
            pass

        self.root.after(100, self._poll_queue)


def main():
    root = tk.Tk()
    ESRGANApp(root)
    root.mainloop()


if __name__ == "__main__":
    if not ESRGAN_OK:
        print("Missing required dependencies. Please install:")
        print("  pip install basicsr realesrgan")
        print("  pip install opencv-python torch numpy pillow")
        print("Optional (face enhance): pip install gfpgan")

    main()
