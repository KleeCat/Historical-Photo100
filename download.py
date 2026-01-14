import requests
import os

def download_large_file(url, filename):
    """Download large files in chunks"""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0

        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"Download progress: {downloaded/total_size*100:.2f}%")

download_large_file(
    "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
    "D:/HuaweiMoveData/Users/ihggk/Desktop/Historical-Photo100/DIV2K_valid_LR_bicubic_X4.zip"
)