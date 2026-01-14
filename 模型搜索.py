import os
import cv2
import torch
import numpy as np
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import urllib.request
from pathlib import Path


def download_pretrained_model(model_url, model_path):
    """下载预训练模型[2](@ref)"""
    if not os.path.exists(model_path):
        print(f"正在下载预训练模型到: {model_path}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        urllib.request.urlretrieve(model_url, model_path)
        print("模型下载完成!")
    return model_path


def initialize_realesrgan(model_path, device='cuda', tile=0):
    """初始化 Real-ESRGAN 引擎[1,4](@ref)"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 创建模型实例
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    # 初始化 RealESRGANer
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=False,
        device=device
    )
    return upsampler


def esrgan_super_resolution(lr_dir, sr_dir, scale_factor=4):
    """使用 Real-ESRGAN 进行超分辨率重建[1,4](@ref)"""
    os.makedirs(sr_dir, exist_ok=True)

    # 配置参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 模型URL和路径配置[2](@ref)
    model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    model_dir = os.path.join(os.path.expanduser("~"), ".cache", "realesrgan")
    model_path = os.path.join(model_dir, "RealESRGAN_x4plus.pth")

    try:
        # 下载或查找模型文件
        if not os.path.exists(model_path):
            # 尝试在常见位置查找现有模型[1](@ref)
            possible_paths = [
                model_path,
                "RealESRGAN_x4plus.pth",
                "models/RealESRGAN_x4plus.pth",
                "/root/autodl-tmp/models/RealESRGAN_x4plus.pth"
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    print(f"找到现有模型: {model_path}")
                    break
            else:
                # 如果没有找到，则下载模型
                model_path = download_pretrained_model(model_url, model_path)
        else:
            print(f"使用现有模型: {model_path}")

        # 初始化超分辨率引擎[4](@ref)
        upsampler = initialize_realesrgan(model_path, device)

        # 处理所有低分辨率图像
        processed_count = 0
        for img_name in os.listdir(lr_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                continue

            lr_path = os.path.join(lr_dir, img_name)
            sr_path = os.path.join(sr_dir, img_name)

            print(f"处理图像: {img_name}")

            try:
                # 读取图像[2](@ref)
                lr_img = cv2.imread(lr_path)
                if lr_img is None:
                    print(f"无法读取图像: {img_name}")
                    continue

                # 使用 Real-ESRGAN 进行超分辨率处理[4](@ref)
                output, _ = upsampler.enhance(lr_img[:, :, ::-1], outscale=scale_factor)  # BGR to RGB

                # 转换回BGR并保存[1](@ref)
                sr_img = output[:, :, ::-1]  # RGB to BGR
                cv2.imwrite(sr_path, sr_img)

                print(f"成功生成超分辨率图像: {img_name} (原始尺寸: {lr_img.shape[:2]} -> 新尺寸: {sr_img.shape[:2]})")
                processed_count += 1

            except Exception as e:
                print(f"处理图像 {img_name} 时出错: {e}")
                continue

        print(f"处理完成! 成功处理 {processed_count} 张图像")

    except Exception as e:
        print(f"初始化或处理过程中出错: {e}")
        print("尝试使用备用的双三次插值方法...")
        # 备用方案：双三次插值
        use_bicubic_fallback(lr_dir, sr_dir, scale_factor)


def use_bicubic_fallback(lr_dir, sr_dir, scale_factor):
    """备用方案：使用双三次插值[3](@ref)"""
    print("使用双三次插值作为备用方案...")

    for img_name in os.listdir(lr_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            continue

        lr_path = os.path.join(lr_dir, img_name)
        sr_path = os.path.join(sr_dir, img_name)

        lr_img = cv2.imread(lr_path)
        if lr_img is None:
            continue

        # 双三次插值[3](@ref)
        h, w = lr_img.shape[:2]
        sr_img = cv2.resize(lr_img, (w * scale_factor, h * scale_factor),
                            interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(sr_path, sr_img)
        print(f"使用双三次插值生成: {img_name}")


if __name__ == "__main__":
    # 配置路径
    lr_dir = r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\LR"
    sr_dir = r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\SR"

    # 检查输入目录是否存在
    if not os.path.exists(lr_dir):
        print(f"错误: 输入目录不存在: {lr_dir}")
    else:
        print("开始超分辨率处理...")
        esrgan_super_resolution(lr_dir, sr_dir)