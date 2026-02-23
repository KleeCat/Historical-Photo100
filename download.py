import argparse
import logging
import os

import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE: int = 8192  # 8 KB
PROGRESS_STEP: int = 10  # 每 10% 输出一次进度


def download_large_file(url: str, filename: str) -> None:
    """Download a large file from a URL in streaming chunks.

    使用临时文件写入，完成后原子重命名，避免中断时留下损坏文件。

    Args:
        url: The URL to download from.
        filename: Local file path to save the downloaded content.

    Raises:
        requests.RequestException: If the download fails.
    """
    tmp_path = filename + ".tmp"
    try:
        with requests.get(url, stream=True, timeout=(10, 30)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            last_reported = -PROGRESS_STEP

            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = int(downloaded / total_size * 100)
                            if percent - last_reported >= PROGRESS_STEP:
                                logger.info("Progress: %d%%", percent)
                                last_reported = percent
                        else:
                            logger.info("Downloaded: %d bytes", downloaded)

        os.replace(tmp_path, filename)
    except (requests.RequestException, KeyboardInterrupt):
        logger.error("Download failed for %s", url)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="下载大文件")
    parser.add_argument(
        "--url",
        default="https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
        help="下载链接",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "DIV2K_valid_LR_bicubic_X4.zip",
        ),
        help="输出文件路径",
    )
    args = parser.parse_args()

    download_large_file(args.url, args.output)
