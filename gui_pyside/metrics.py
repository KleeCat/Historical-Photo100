"""指标计算：PSNR / SSIM / LPIPS。"""
import logging
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports
try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr
    from skimage.metrics import structural_similarity as _ssim
except ImportError:
    _psnr = None
    _ssim = None
    logger.warning("skimage not installed, PSNR/SSIM unavailable.")

try:
    import torch
    import lpips as _lpips_mod

    _lpips_fn = None

    def _get_lpips():
        global _lpips_fn
        if _lpips_fn is None:
            _lpips_fn = _lpips_mod.LPIPS(net="alex")
            if torch.cuda.is_available():
                _lpips_fn = _lpips_fn.cuda()
        return _lpips_fn

except ImportError:
    _lpips_mod = None

    def _get_lpips():
        return None


def calculate_metrics(
    sr_img: np.ndarray, gt_img: np.ndarray
) -> Dict[str, Optional[float]]:
    """计算 PSNR、SSIM、可选 LPIPS。

    Both images should be BGR uint8 with the same shape.
    """
    result: Dict[str, Optional[float]] = {
        "psnr": None,
        "ssim": None,
        "lpips": None,
    }
    if sr_img is None or gt_img is None:
        return result

    # Ensure same size
    if sr_img.shape[:2] != gt_img.shape[:2]:
        gt_img = cv2.resize(
            gt_img, (sr_img.shape[1], sr_img.shape[0]), interpolation=cv2.INTER_CUBIC
        )

    # Convert to grayscale for PSNR/SSIM
    sr_gray = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)

    if _psnr is not None:
        try:
            result["psnr"] = round(float(_psnr(gt_gray, sr_gray)), 2)
        except Exception as e:
            logger.debug("PSNR calculation failed: %s", e)

    if _ssim is not None:
        try:
            result["ssim"] = round(
                float(_ssim(gt_gray, sr_gray, data_range=255)), 4
            )
        except Exception as e:
            logger.debug("SSIM calculation failed: %s", e)

    # LPIPS
    lpips_fn = _get_lpips()
    if lpips_fn is not None:
        try:
            import torch

            def _to_tensor(img_bgr):
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                t = t * 2.0 - 1.0
                if torch.cuda.is_available():
                    t = t.cuda()
                return t

            with torch.no_grad():
                val = lpips_fn(_to_tensor(sr_img), _to_tensor(gt_img))
                result["lpips"] = round(float(val.item()), 4)
        except Exception as e:
            logger.debug("LPIPS calculation failed: %s", e)

    return result
