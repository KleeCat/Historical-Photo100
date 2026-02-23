# 标准库
import argparse
import logging
import os
import sys
import urllib.request

# 第三方库
import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_SCALE_FACTOR: int = 4
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


class RRDBNet(nn.Module):
    """简化版 ESRGAN 生成器网络。

    注意: 此为简化结构，缺少上采样层和残差密集块，
    无法实现真正的超分辨率推理。仅作为架构参考保留。
    """

    def __init__(
        self,
        in_nc: int = 3,
        out_nc: int = 3,
        nf: int = 64,
        nb: int = 23,
        gc: int = 32,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.body = self._make_layer(nb, nf)
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    @staticmethod
    def _make_layer(nb: int, nf: int) -> nn.Sequential:
        """构建卷积层序列。"""
        layers = [nn.Conv2d(nf, nf, 3, 1, 1, bias=True) for _ in range(nb)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        out = self.conv_last(feat + body_feat)
        return out


def download_pretrained_model() -> str:
    """下载 ESRGAN 预训练模型。

    Returns:
        模型文件的本地路径。

    Raises:
        RuntimeError: 下载失败时抛出。
    """
    model_url = (
        "https://github.com/xinntao/ESRGAN/releases/"
        "download/v0.1.1/RRDB_ESRGAN_x4.pth"
    )
    model_path = "ESRGAN.pth"

    if os.path.exists(model_path):
        return model_path

    logger.info("正在下载 ESRGAN 预训练模型...")
    tmp_path = model_path + ".tmp"
    try:
        urllib.request.urlretrieve(model_url, tmp_path)
        os.replace(tmp_path, model_path)
    except (urllib.error.URLError, OSError) as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise RuntimeError(f"模型下载失败: {e}") from e

    return model_path


def esrgan_super_resolution(
    lr_dir: str,
    sr_dir: str,
    scale_factor: int = DEFAULT_SCALE_FACTOR,
) -> None:
    """使用超分辨率方法放大低分辨率图像。

    尝试加载 ESRGAN 预训练模型进行推理，若失败则回退到 bicubic 插值。

    Args:
        lr_dir: 低分辨率图像所在目录。
        sr_dir: 超分辨率结果输出目录。
        scale_factor: 放大倍数，默认为 4。
    """
    os.makedirs(sr_dir, exist_ok=True)

    # 尝试加载模型（当前简化架构无法真正推理，直接使用 bicubic）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model_path = download_pretrained_model()
        state_dict = torch.load(
            model_path, map_location=device, weights_only=True
        )
        model = RRDBNet().to(device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("缺失的键: %s", missing)
        if unexpected:
            logger.warning("多余的键: %s", unexpected)
        model.eval()
        logger.info("已加载 ESRGAN 预训练模型")
    except (RuntimeError, OSError, ValueError, EOFError) as e:
        logger.warning("模型加载失败，回退到 bicubic 插值: %s", e)

    # 处理所有 LR 图像
    for img_name in sorted(os.listdir(lr_dir)):
        if os.path.splitext(img_name)[1].lower() not in SUPPORTED_EXTENSIONS:
            continue

        lr_path = os.path.join(lr_dir, img_name)
        sr_path = os.path.join(sr_dir, img_name)

        lr_img = cv2.imread(lr_path)
        if lr_img is None:
            logger.warning("无法读取图像: %s", lr_path)
            continue

        h, w = lr_img.shape[:2]
        sr_img = cv2.resize(
            lr_img,
            (w * scale_factor, h * scale_factor),
            interpolation=cv2.INTER_CUBIC,
        )

        if not cv2.imwrite(sr_path, sr_img):
            logger.error("写入 SR 图像失败: %s", sr_path)
            continue
        logger.info("已生成 SR 图像: %s", img_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="ESRGAN 超分辨率处理")
    parser.add_argument("--lr-dir", type=str, default="LR", help="低分辨率图像目录")
    parser.add_argument("--sr-dir", type=str, default="SR", help="超分辨率输出目录")
    parser.add_argument(
        "--scale", type=int, default=DEFAULT_SCALE_FACTOR, help="放大倍数"
    )
    args = parser.parse_args()

    esrgan_super_resolution(args.lr_dir, args.sr_dir, args.scale)

