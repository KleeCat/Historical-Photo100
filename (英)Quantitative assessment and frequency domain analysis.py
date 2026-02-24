import os
import logging
import argparse
from typing import Optional
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from pytorch_fid import fid_score
import matplotlib.pyplot as plt
from scipy import fftpack

logger = logging.getLogger(__name__)


class QuantitativeEvaluator:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        try:
            self.lpips_model = lpips.LPIPS(net='vgg').to(device)
            self.lpips_model.eval()
        except (RuntimeError, OSError) as e:
            logger.warning("LPIPS initialization failed: %s", e)
            self.lpips_model = None

    def check_path_exists(self, path: str) -> bool:
        """Check if the path exists"""
        if not os.path.exists(path):
            logger.warning("The path does not exist: %s", path)
            return False
        return True

    def find_matching_filename(self, hr_filename: str, target_dir: str) -> Optional[str]:
        """Find the corresponding LR or SR file name based on the HR file name"""
        name_without_ext = os.path.splitext(hr_filename)[0]
        target_filename = f"{name_without_ext}x4.png"

        possible_names = [
            target_filename,
            f"{name_without_ext}x4.jpg",
            f"{name_without_ext}x4.jpeg",
            f"{name_without_ext}x4.bmp",
            hr_filename
        ]

        for name in possible_names:
            test_path = os.path.join(target_dir, name)
            if os.path.exists(test_path):
                return test_path

        return None

    def calculate_psnr_ssim(self, img1_path: str, img2_path: str) -> tuple[Optional[float], Optional[float]]:
        """Calculate the PSNR and SSIM indicators"""
        if not self.check_path_exists(img1_path) or not self.check_path_exists(img2_path):
            return None, None

        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                logger.warning("Unable to read the image: %s or %s", img1_path, img2_path)
                return None, None

            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            if len(img1.shape) == 3:
                img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            else:
                img1_y, img2_y = img1, img2

            psnr_val = peak_signal_noise_ratio(img1_y, img2_y, data_range=255)
            ssim_val = structural_similarity(img1_y, img2_y, data_range=255)

            return psnr_val, ssim_val
        except (cv2.error, ValueError, OSError) as e:
            logger.error("Error calculating PSNR/SSIM: %s", e)
            return None, None

    def calculate_lpips(self, img1_path: str, img2_path: str) -> Optional[float]:
        """Calculate the LPIPS perceptual similarity"""
        if self.lpips_model is None:
            logger.warning("The LPIPS model is unavailable")
            return None

        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')

            # Align sizes to match img1
            if img1.size != img2.size:
                img2 = img2.resize(img1.size, Image.LANCZOS)

            img1_t = self.transform(img1).unsqueeze(0).to(self.device)
            img2_t = self.transform(img2).unsqueeze(0).to(self.device)

            with torch.no_grad():
                lpips_val = self.lpips_model(img1_t, img2_t)

            return lpips_val.item()
        except (RuntimeError, OSError, ValueError) as e:
            logger.error("Error calculating LPIPS: %s", e)
            return None

    def calculate_fid(self, real_images_dir: str, generated_images_dir: str) -> float:
        """Calculate the FID index"""
        if not self.check_path_exists(real_images_dir) or not self.check_path_exists(generated_images_dir):
            return 999.0

        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [real_images_dir, generated_images_dir],
                batch_size=32,
                device=self.device,
                dims=2048
            )
            return fid_value
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Error calculating FID: %s", e)
            return 999.0

    def frequency_analysis(self, image_path: str, save_spectrum_path: Optional[str] = None) -> dict:
        """Frequency domain energy analysis"""
        if not self.check_path_exists(image_path):
            return {'low_freq_energy': 0, 'mid_freq_energy': 0, 'high_freq_energy': 0, 'spectrum': None}

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return {'low_freq_energy': 0, 'mid_freq_energy': 0, 'high_freq_energy': 0, 'spectrum': None}

            f_transform = fftpack.fft2(img)
            f_shift = fftpack.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)

            if save_spectrum_path:
                plt.figure(figsize=(8, 6))
                plt.imshow(magnitude_spectrum, cmap='hot')
                plt.colorbar()
                plt.title(f'Frequency Spectrum - {os.path.basename(image_path)}')
                plt.savefig(save_spectrum_path, dpi=300, bbox_inches='tight')
                plt.close()

            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            min_dim = min(h, w)

            # Proportional frequency band boundaries
            low_radius = max(int(min_dim * 0.05), 1)
            mid_radius = max(int(min_dim * 0.15), low_radius + 1)

            # Create distance map from center
            y_coords, x_coords = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)

            # Low frequency: within low_radius
            low_mask = dist_from_center <= low_radius
            energy_low = np.mean(magnitude_spectrum[low_mask]) if np.any(low_mask) else 0

            # Mid frequency: between low_radius and mid_radius
            mid_mask = (dist_from_center > low_radius) & (dist_from_center <= mid_radius)
            energy_mid = np.mean(magnitude_spectrum[mid_mask]) if np.any(mid_mask) else 0

            # High frequency: beyond mid_radius
            high_mask = dist_from_center > mid_radius
            energy_high = np.mean(magnitude_spectrum[high_mask]) if np.any(high_mask) else 0

            return {
                'low_freq_energy': energy_low,
                'mid_freq_energy': energy_mid,
                'high_freq_energy': energy_high,
                'spectrum': magnitude_spectrum
            }
        except (ValueError, OSError) as e:
            logger.error("Error during frequency domain analysis: %s", e)
            return {'low_freq_energy': 0, 'mid_freq_energy': 0, 'high_freq_energy': 0, 'spectrum': None}

    def comprehensive_evaluation(self, lr_dir: str, hr_dir: str, sr_dir: str, output_dir: str) -> tuple[list, float]:
        """comprehensive assessment"""
        for dir_path, dir_name in [(lr_dir, "LR"), (hr_dir, "HR"), (sr_dir, "SR")]:
            if not self.check_path_exists(dir_path):
                logger.error("%s directory does not exist: %s", dir_name, dir_path)
                return [], 999.0

        os.makedirs(output_dir, exist_ok=True)

        try:
            image_names = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        except OSError as e:
            logger.error("Failed to read the HR directory: %s", e)
            return [], 999.0

        if not image_names:
            logger.warning("No image files found in the HR directory.")
            return [], 999.0

        logger.info("Found %d images for evaluation", len(image_names))

        results = []
        frequency_results = []
        max_spectrum_samples = 3
        valid_count = 0

        for idx, img_name in enumerate(image_names):
            logger.info("Processing image %d/%d: %s", idx + 1, len(image_names), img_name)

            hr_path = os.path.join(hr_dir, img_name)

            lr_path = self.find_matching_filename(img_name, lr_dir)
            if lr_path is None:
                logger.info("Skip %s - no matching image in LR directory", img_name)
                continue

            sr_path = self.find_matching_filename(img_name, sr_dir)
            if sr_path is None:
                logger.info("Skip %s - no matching image in SR directory", img_name)
                continue

            psnr, ssim = self.calculate_psnr_ssim(hr_path, sr_path)
            if psnr is None:
                logger.info("Skip %s - metric calculation failed", img_name)
                continue

            lpips_val = self.calculate_lpips(hr_path, sr_path)
            if lpips_val is None:
                logger.warning("LPIPS unavailable for %s; continuing without LPIPS", img_name)
                lpips_val = float('nan')

            freq_hr = self.frequency_analysis(hr_path)
            freq_sr = self.frequency_analysis(sr_path)
            freq_lr = self.frequency_analysis(lr_path)

            result = {
                'image_name': img_name,
                'psnr': psnr,
                'ssim': ssim,
                'lpips': lpips_val,
                'freq_analysis': {
                    'hr': freq_hr,
                    'sr': freq_sr,
                    'lr': freq_lr
                }
            }
            results.append(result)

            # Only keep spectrum data for visualization samples
            if len(frequency_results) < max_spectrum_samples:
                frequency_results.append({
                    'name': img_name,
                    'hr_spectrum': freq_hr['spectrum'],
                    'sr_spectrum': freq_sr['spectrum'],
                    'lr_spectrum': freq_lr['spectrum']
                })

            valid_count += 1

            if valid_count % 10 == 0:
                logger.info("%d images processed...", valid_count)

        if valid_count == 0:
            logger.warning("No valid image pairs available for evaluation")
            return [], 999.0

        logger.info("Calculating FID...")
        fid_value = self.calculate_fid(hr_dir, sr_dir)

        self.generate_report(results, fid_value, output_dir, frequency_results, lr_dir)

        logger.info("Evaluation completed! Valid images: %d/%d", valid_count, len(image_names))
        return results, fid_value

    def generate_report(self, results: list, fid_value: float, output_dir: str,
                        frequency_results: list, lr_dir: str) -> None:
        """Generate an evaluation report"""
        if not results:
            logger.warning("No results to generate report.")
            return

        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        lpips_values = [r['lpips'] for r in results if r['lpips'] is not None and not np.isnan(r['lpips'])]
        avg_lpips = np.mean(lpips_values) if lpips_values else float('nan')
        lpips_summary = f"{avg_lpips:.4f} (lower is better)" if lpips_values else "N/A (LPIPS unavailable)"

        report_text = f"""
Assessment Report - Historical-Photo100 DataSet
==================================
Evaluate time: {np.datetime64('now')}
Number of valid images: {len(results)}

Overall indicators:
- Average PSNR: {avg_psnr:.4f} dB
- Average SSIM: {avg_ssim:.4f}
- Average LPIPS: {lpips_summary}
- FID: {fid_value:.4f} (lower is better)

Detailed results:
{'Image Name':<30} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}
{'-' * 60}
"""

        for result in results:
            lpips_display = "N/A"
            if result['lpips'] is not None and not np.isnan(result['lpips']):
                lpips_display = f"{result['lpips']:.4f}"
            report_text += f"{result['image_name']:<30} {result['psnr']:<10.4f} {result['ssim']:<10.4f} {lpips_display:<10}\n"

        report_text += f"\nFrequency domain energy analysis:\n"
        avg_lr_high = np.mean([r['freq_analysis']['lr']['high_freq_energy'] for r in results])
        avg_sr_high = np.mean([r['freq_analysis']['sr']['high_freq_energy'] for r in results])
        avg_hr_high = np.mean([r['freq_analysis']['hr']['high_freq_energy'] for r in results])

        report_text += f"- Average high-frequency energy - LR: {avg_lr_high:.2f}, SR: {avg_sr_high:.2f}, HR: {avg_hr_high:.2f}\n"
        if avg_hr_high - avg_lr_high > 1e-6:
            recovery_rate = (avg_sr_high - avg_lr_high) / (avg_hr_high - avg_lr_high) * 100
            report_text += f"- High-frequency recovery rate: {recovery_rate:.2f}%\n"
        else:
            report_text += "- High-frequency recovery rate: N/A (HR-LR high-frequency energy difference too small)\n"

        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info("Report saved to: %s", report_path)

        self.plot_frequency_comparison(frequency_results, output_dir, lr_dir)
        self.plot_metrics_distribution(results, output_dir)

    def plot_frequency_comparison(self, frequency_results: list, output_dir: str, lr_dir: str) -> None:
        """Draw a frequency domain energy comparison chart"""
        if not frequency_results:
            return

        sample_indices = min(3, len(frequency_results))

        fig, axes = plt.subplots(sample_indices, 4, figsize=(20, 5 * sample_indices))
        if sample_indices == 1:
            axes = axes.reshape(1, -1)

        for i in range(sample_indices):
            data = frequency_results[i]

            try:
                lr_path = self.find_matching_filename(data['name'], lr_dir)
                if lr_path and os.path.exists(lr_path):
                    lr_img = cv2.imread(lr_path)
                    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                    axes[i, 0].imshow(lr_img)
                axes[i, 0].set_title(f'LR - {data["name"]}')
                axes[i, 0].axis('off')
            except (cv2.error, OSError) as e:
                logger.warning("Failed to display LR image: %s", e)
                axes[i, 0].text(0.5, 0.5, 'Image loading failed', ha='center', va='center')
                axes[i, 0].axis('off')

            spectra = [data['lr_spectrum'], data['sr_spectrum'], data.get('hr_spectrum', None)]
            titles = ['LR Spectrum', 'SR Spectrum', 'HR Spectrum']

            for k in range(3):
                if k < len(spectra) and spectra[k] is not None:
                    im = axes[i, k + 1].imshow(spectra[k], cmap='hot')
                    axes[i, k + 1].set_title(titles[k])
                    axes[i, k + 1].axis('off')
                    plt.colorbar(im, ax=axes[i, k + 1])
                else:
                    axes[i, k + 1].text(0.5, 0.5, 'Spectrum data missing', ha='center', va='center')
                    axes[i, k + 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Frequency domain comparison chart generated")

    def plot_metrics_distribution(self, results: list, output_dir: str) -> None:
        """Draw an indicator distribution chart"""
        if not results:
            return

        try:
            metrics = ['psnr', 'ssim', 'lpips']
            metric_names = ['PSNR (dB)', 'SSIM', 'LPIPS']

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i, metric in enumerate(metrics):
                values = [r[metric] for r in results]
                clean_values = [v for v in values if v is not None and not np.isnan(v)]
                if not clean_values:
                    axes[i].text(0.5, 0.5, 'No data', ha='center', va='center')
                    axes[i].set_xlabel(metric_names[i])
                    axes[i].set_ylabel('Frequency')
                    axes[i].set_title(f'{metric_names[i]} Distribution')
                    axes[i].grid(True, alpha=0.3)
                    continue

                axes[i].hist(clean_values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_xlabel(metric_names[i])
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'{metric_names[i]} Distribution')
                axes[i].grid(True, alpha=0.3)

                mean_val = np.mean(clean_values)
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
                axes[i].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            logger.info("Metrics distribution chart generated.")
        except (ValueError, OSError) as e:
            logger.error("Error generating metrics distribution chart: %s", e)


def main():
    parser = argparse.ArgumentParser(description='Quantitative assessment and frequency domain analysis')
    default_base = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--base-dir', default=default_base, help='Base directory (default: script location)')
    parser.add_argument('--lr-dir', default=None, help='LR image directory')
    parser.add_argument('--hr-dir', default=None, help='HR image directory')
    parser.add_argument('--sr-dir', default=None, help='SR image directory')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    base_dir = args.base_dir
    lr_dir = args.lr_dir or os.path.join(base_dir, "LR")
    hr_dir = args.hr_dir or os.path.join(base_dir, "HR")
    sr_dir = args.sr_dir or os.path.join(base_dir, "SR")
    output_dir = args.output_dir or os.path.join(base_dir, "evaluation_results")

    for dir_path, dir_name in [(lr_dir, "LR"), (hr_dir, "HR"), (sr_dir, "SR")]:
        if not os.path.exists(dir_path):
            logger.error("%s directory does not exist: %s", dir_name, dir_path)
            logger.info("Please ensure the following directories exist and contain image files:")
            logger.info("LR: %s", lr_dir)
            logger.info("HR: %s", hr_dir)
            logger.info("SR: %s", sr_dir)
            return

    evaluator = QuantitativeEvaluator()

    logger.info("Starting evaluation...")
    results, fid_value = evaluator.comprehensive_evaluation(lr_dir, hr_dir, sr_dir, output_dir)

    if results:
        logger.info("Evaluation completed! FID: %.4f", fid_value)
        logger.info("Detailed report saved to: %s", output_dir)
    else:
        logger.error("Evaluation failed. Please check paths and image files.")


if __name__ == "__main__":
    main()
