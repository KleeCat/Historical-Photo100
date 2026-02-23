import logging
import os

import requests

logger = logging.getLogger(__name__)

CHUNK_SIZE = 8192  # 8 KB


def download_large_file(url: str, filename: str) -> None:
    """Download a large file from a URL in streaming chunks.

    Args:
        url: The URL to download from.
        filename: Local file path to save the downloaded content.

    Raises:
        requests.RequestException: If the download fails.
    """
    try:
        with requests.get(url, stream=True, timeout=(10, 30)) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            downloaded = 0
            last_percent = -1

            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = int(downloaded / total_size * 100)
                            if percent != last_percent:
                                logger.info("Progress: %d%%", percent)
                                last_percent = percent
                        else:
                            logger.info("Downloaded: %d bytes", downloaded)
    except requests.RequestException:
        logger.error("Download failed for %s", url)
        if os.path.exists(filename):
            os.remove(filename)
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "DIV2K_valid_LR_bicubic_X4.zip")

    download_large_file(
        "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_LR_bicubic_X4.zip",
        output_path,
    )
