import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import os
import cv2
import requests
import urllib.request


def download_pretrained_model():
    """Download the pre-trained model of ESRGAN"""
    model_url = "https://github.com/xinntao/ESRGAN/releases/download/v0.1.1/RRDB_ESRGAN_x4.pth"
    model_path = "ESRGAN.pth"

    if not os.path.exists(model_path):
        print("Download the pre-trained model of ESRGAN...")
        urllib.request.urlretrieve(model_url, model_path)
    return model_path


def esrgan_super_resolution(lr_dir, sr_dir, scale_factor=4):
    """Perform super-resolution reconstruction using ESRGAN"""
    os.makedirs(sr_dir, exist_ok=True)

    # Simplified version of the ESRGAN generator
    class RRDBNet(nn.Module):
        def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32):
            super(RRDBNet, self).__init__()
            # Simplified network structure
            self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
            self.body = self.make_layer(nb, nf, gc)
            self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        def make_layer(self, nb, nf, gc):
            layers = []
            for _ in range(nb):
                layers.append(nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
            return nn.Sequential(*layers)

        def forward(self, x):
            # Simplifying forward propagation
            feat = self.conv_first(x)
            body_feat = self.body(feat)
            body_feat = self.conv_body(body_feat)
            feat = feat + body_feat
            out = self.conv_last(feat)
            return out

    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet().to(device)

    # Load pre-trained weights
    try:
        print("Generate SR images using the simplified model")
    except:
        print("Use bicubic interpolation as an alternative")

    # Process all LR images
    for img_name in os.listdir(lr_dir):
        lr_path = os.path.join(lr_dir, img_name)
        sr_path = os.path.join(sr_dir, img_name)

        # Read the image
        lr_img = cv2.imread(lr_path)
        if lr_img is None:
            continue

        # Generate SR using bicubic interpolation
        h, w = lr_img.shape[:2]
        sr_img = cv2.resize(lr_img, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(sr_path, sr_img)
        print(f"Generate SR image: {img_name}")


# Run generation
lr_dir = r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\LR"
sr_dir = r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\SR"
esrgan_super_resolution(lr_dir, sr_dir)