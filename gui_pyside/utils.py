"""工具函数：图像转换、文件操作。"""
import contextlib
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap


def numpy_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """BGR numpy array → QPixmap。"""
    if img_bgr is None:
        return QPixmap()
    img = img_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    rgb_contiguous = np.ascontiguousarray(rgb)
    qimg = QImage(rgb_contiguous.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    # Must copy because rgb_contiguous data may be garbage collected
    return QPixmap.fromImage(qimg.copy())


def qpixmap_to_numpy(pixmap: QPixmap) -> Optional[np.ndarray]:
    """QPixmap → BGR numpy array。"""
    if pixmap is None or pixmap.isNull():
        return None
    qimg = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 3))
    return cv2.cvtColor(arr.copy(), cv2.COLOR_RGB2BGR)


@contextlib.contextmanager
def suppress_stderr():
    """Temporarily redirect stderr to devnull."""
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


def ensure_dir(path: str) -> str:
    """Create directory if it does not exist and return the path."""
    os.makedirs(path, exist_ok=True)
    return path


def write_json_file(path: str, payload: Dict[str, Any]) -> None:
    """Write a dictionary as a JSON file, creating parent directories as needed."""
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def timestamp_str() -> str:
    """Return a timestamp string suitable for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_basename(path: Optional[str], fallback: str = "image") -> str:
    """Return the stem of a file path, or *fallback* if path is empty."""
    if not path:
        return fallback
    return os.path.splitext(os.path.basename(path))[0]


_SUPPORTED_SAVE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"})


def save_image(path: str, bgr_img: np.ndarray) -> str:
    """Encode and save a BGR image to disk."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _SUPPORTED_SAVE_EXTS:
        ext = ".png"
        path = path + ext
    with suppress_stderr():
        if ext in (".tiff", ".tif"):
            params = []
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, 95]
        elif ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]
        else:
            params = []
        success, buf = cv2.imencode(ext, bgr_img, params)
    if not success:
        raise RuntimeError("Failed to encode image")
    buf.tofile(path)
    return path


def read_image(path: str) -> Optional[np.ndarray]:
    """Read an image from disk, handling Unicode paths."""
    if not path or not os.path.isfile(path):
        return None
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    except Exception:
        return None
