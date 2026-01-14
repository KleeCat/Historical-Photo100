import os
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


class QuantitativeEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Fix the initialization of LPIPS
        try:
            self.lpips_model = lpips.LPIPS(net='vgg').to(device)
            self.lpips_model.eval()
        except Exception as e:
            print(f"LPIPS initialization failed: {e}")
            self.lpips_model = None

    def check_path_exists(self, path):
        """Check if the path exists"""
        if not os.path.exists(path):
            print(f"Warning: The path does not exist: {path}")
            return False
        return True

    def find_matching_filename(self, hr_filename, target_dir):
        """Find the corresponding LR or SR file name based on the HR file name"""
        # HR file name format: 0801.png
        # LR/SR file name format: 0801x4.png
        name_without_ext = os.path.splitext(hr_filename)[0]  # 0801
        target_filename = f"{name_without_ext}x4.png"  # 0801x4.png

        # Check if there are any other possible formats
        possible_names = [
            target_filename,
            f"{name_without_ext}x4.jpg",
            f"{name_without_ext}x4.jpeg",
            f"{name_without_ext}x4.bmp",
            hr_filename  # Also check the original file name
        ]

        for name in possible_names:
            test_path = os.path.join(target_dir, name)
            if os.path.exists(test_path):
                return test_path

        return None

    def calculate_psnr_ssim(self, img1_path, img2_path):
        """Calculate the PSNR and SSIM indicators"""
        if not self.check_path_exists(img1_path) or not self.check_path_exists(img2_path):
            return 0, 0

        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                print(f"Unable to read the image: {img1_path} or {img2_path}")
                return 0, 0

            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # Calculate using the Y channel
            if len(img1.shape) == 3:
                img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            else:
                img1_y, img2_y = img1, img2

            psnr_val = peak_signal_noise_ratio(img1_y, img2_y, data_range=255)
            ssim_val = structural_similarity(img1_y, img2_y, data_range=255)

            return psnr_val, ssim_val
        except Exception as e:
            print(f"There was an error when calculating PSNR/SSIM: {e}")
            return 0, 0

    def calculate_lpips(self, img1_path, img2_path):
        """Calculate the LPIPS perceptual similarity"""
        if self.lpips_model is None:
            print("The LPIPS model is unavailable. Returning the default value of 0")
            return 0.0

        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

            img1 = transform(Image.open(img1_path).convert('RGB')).unsqueeze(0).to(self.device)
            img2 = transform(Image.open(img2_path).convert('RGB')).unsqueeze(0).to(self.device)

            with torch.no_grad():
                lpips_val = self.lpips_model(img1, img2)

            return lpips_val.item()
        except Exception as e:
            print(f"There was an error when calculating the LPIPS: {e}")
            return 0.0

    def calculate_fid(self, real_images_dir, generated_images_dir):
        """Calculate the FID index"""
        if not self.check_path_exists(real_images_dir) or not self.check_path_exists(generated_images_dir):
            return 999.0  # Return a larger value to indicate an error

        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [real_images_dir, generated_images_dir],
                batch_size=32,
                device=self.device,
                dims=2048
            )
            return fid_value
        except Exception as e:
            print(f"There was an error when calculating the FID.: {e}")
            return 999.0

    def frequency_analysis(self, image_path, save_spectrum_path=None):
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

            # Ensure that the index does not exceed the boundary
            low_start_y = max(center_y - 30, 0)
            low_end_y = min(center_y + 30, h)
            low_start_x = max(center_x - 30, 0)
            low_end_x = min(center_x + 30, w)

            low_freq = magnitude_spectrum[low_start_y:low_end_y, low_start_x:low_end_x]
            energy_low = np.mean(low_freq) if low_freq.size > 0 else 0

            # Medium-frequency region
            mid_start_y = max(center_y - 60, 0)
            mid_end_y = min(center_y + 60, h)
            mid_start_x = max(center_x - 60, 0)
            mid_end_x = min(center_x + 60, w)

            mid_freq = magnitude_spectrum[mid_start_y:mid_end_y, mid_start_x:mid_end_x]
            if mid_freq.shape[0] > 60 and mid_freq.shape[1] > 60:
                mid_freq = mid_freq[30:-30, 30:-30]
            energy_mid = np.mean(mid_freq) if mid_freq.size > 0 else 0

            energy_high = np.mean(magnitude_spectrum)

            return {
                'low_freq_energy': energy_low,
                'mid_freq_energy': energy_mid,
                'high_freq_energy': energy_high,
                'spectrum': magnitude_spectrum
            }
        except Exception as e:
            print(f"Error occurred during frequency domain analysis: {e}")
            return {'low_freq_energy': 0, 'mid_freq_energy': 0, 'high_freq_energy': 0, 'spectrum': None}

    def comprehensive_evaluation(self, lr_dir, hr_dir, sr_dir, output_dir):
        """comprehensive assessment"""
        # Check whether all paths exist
        for dir_path, dir_name in [(lr_dir, "LR"), (hr_dir, "HR"), (sr_dir, "SR")]:
            if not self.check_path_exists(dir_path):
                print(f"error: {dir_name}The directory does not exist: {dir_path}")
                return [], 999.0

        os.makedirs(output_dir, exist_ok=True)

        # Obtain the list of images in the HR directory
        try:
            image_names = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        except Exception as e:
            print(f"Failed to read the HR directory: {e}")
            return [], 999.0

        if not image_names:
            print("The image file was not found in the HR directory.")
            return [], 999.0

        print(f"Find {len(image_names)} images for evaluation")

        results = []
        frequency_results = []
        valid_count = 0

        for idx, img_name in enumerate(image_names):
            print(f"process an image {idx + 1}/{len(image_names)}: {img_name}")

            hr_path = os.path.join(hr_dir, img_name)

            # Search for the corresponding LR file
            lr_path = self.find_matching_filename(img_name, lr_dir)
            if lr_path is None:
                print(f"skip {img_name} - The matching image could not be found in the LR directory")
                continue

            # Search for the corresponding SR file
            sr_path = self.find_matching_filename(img_name, sr_dir)
            if sr_path is None:
                print(f"skip {img_name} - No matching image could be found in the SR directory")
                continue

            # parameter
            psnr, ssim = self.calculate_psnr_ssim(hr_path, sr_path)
            lpips_val = self.calculate_lpips(hr_path, sr_path)

            # frequency domain analysis
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
            frequency_results.append({
                'name': img_name,
                'hr_spectrum': freq_hr['spectrum'],
                'sr_spectrum': freq_sr['spectrum'],
                'lr_spectrum': freq_lr['spectrum']
            })
            valid_count += 1

            # Save the progress every time 10 images are processed
            if valid_count % 10 == 0:
                print(f" {valid_count} images have been processed...")

        if valid_count == 0:
            print("There are no valid image pairs available for evaluation")
            return [], 999.0

        # Calculate FID
        print("Calculate the FID index...")
        fid_value = self.calculate_fid(hr_dir, sr_dir)

        # generate a report
        self.generate_report(results, fid_value, output_dir, frequency_results, lr_dir)

        print(f"Evaluation completed! The valid image is: {valid_count}/{len(image_names)}")
        return results, fid_value

    def generate_report(self, results, fid_value, output_dir, frequency_results, lr_dir):
        """Generate an evaluation report"""
        if not results:
            print("No results can be generated for the report.")
            return

        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_lpips = np.mean([r['lpips'] for r in results])

        # Generate report text
        report_text = f"""
assessment report - Historical-Photo100 DataSet
==================================
evaluate time: {np.datetime64('now')}
Number of valid images: {len(results)}

Overall indicators:
- Average PSNR: {avg_psnr:.4f} dB
- Average SSIM: {avg_ssim:.4f}
- Average LPIPS: {avg_lpips:.4f} (As low as possible)
- FID: {fid_value:.4f} (As low as possible)

Detailed results:
{'Image Name':<30} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}
{'-' * 60}
"""

        for result in results:
            report_text += f"{result['image_name']:<30} {result['psnr']:<10.4f} {result['ssim']:<10.4f} {result['lpips']:<10.4f}\n"

        # Frequency domain analysis summary
        report_text += f"\nFrequency domain energy analysis:\n"
        avg_lr_high = np.mean([r['freq_analysis']['lr']['high_freq_energy'] for r in results])
        avg_sr_high = np.mean([r['freq_analysis']['sr']['high_freq_energy'] for r in results])
        avg_hr_high = np.mean([r['freq_analysis']['hr']['high_freq_energy'] for r in results])

        report_text += f"- Average high-frequency energy - LR: {avg_lr_high:.2f}, SR: {avg_sr_high:.2f}, HR: {avg_hr_high:.2f}\n"
        if avg_hr_high - avg_lr_high > 1e-6:
            recovery_rate = (avg_sr_high - avg_lr_high) / (avg_hr_high - avg_lr_high) * 100
            report_text += f"- High-frequency recovery rate: {recovery_rate:.2f}%\n"
        else:
            report_text += "- High-frequency recovery rate: Unable to calculate (The difference in high-frequency energy between HR and LR is too small)\n"

        # Save the report
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"The report has been saved to: {report_path}")

        # Generate visual charts
        self.plot_frequency_comparison(frequency_results, output_dir, lr_dir)
        self.plot_metrics_distribution(results, output_dir)

    def plot_frequency_comparison(self, frequency_results, output_dir, lr_dir):
        """Draw a frequency domain energy comparison chart"""
        if not frequency_results:
            return

        # Select several representative samples for detailed presentation
        sample_indices = min(3, len(frequency_results))

        fig, axes = plt.subplots(sample_indices, 4, figsize=(20, 5 * sample_indices))
        if sample_indices == 1:
            axes = axes.reshape(1, -1)

        for i in range(sample_indices):
            data = frequency_results[i]

            # original image
            try:
                # Use the matching LR image path
                lr_path = self.find_matching_filename(data['name'], lr_dir)
                if lr_path and os.path.exists(lr_path):
                    lr_img = cv2.imread(lr_path)
                    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                    axes[i, 0].imshow(lr_img)
                axes[i, 0].set_title(f'LR - {data["name"]}')
                axes[i, 0].axis('off')
            except Exception as e:
                print(f"Failed to display the LR image: {e}")
                axes[i, 0].text(0.5, 0.5, 'Image loading failed', ha='center', va='center')
                axes[i, 0].axis('off')

            # Frequency domain comparison
            spectra = [data['lr_spectrum'], data['sr_spectrum'], data.get('hr_spectrum', None)]
            titles = ['LR Spectrum', 'SR Spectrum', 'HR Spectrum']

            for k in range(3):
                if k < len(spectra) and spectra[k] is not None:
                    im = axes[i, k + 1].imshow(spectra[k], cmap='hot')
                    axes[i, k + 1].set_title(titles[k])
                    axes[i, k + 1].axis('off')
                    plt.colorbar(im, ax=axes[i, k + 1])
                else:
                    axes[i, k + 1].text(0.5, 0.5, 'Spectrum data is missing.', ha='center', va='center')
                    axes[i, k + 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("The frequency domain comparison chart has been generated")

    def plot_metrics_distribution(self, results, output_dir):
        """Draw an indicator distribution chart"""
        if not results:
            return

        try:
            metrics = ['psnr', 'ssim', 'lpips']
            metric_names = ['PSNR (dB)', 'SSIM', 'LPIPS']

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i, metric in enumerate(metrics):
                values = [r[metric] for r in results]
                axes[i].hist(values, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                axes[i].set_xlabel(metric_names[i])
                axes[i].set_ylabel('frequency')
                axes[i].set_title(f'{metric_names[i]}distribution')
                axes[i].grid(True, alpha=0.3)

                mean_val = np.mean(values)
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.3f}')
                axes[i].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("The indicator distribution chart has been generated.")
        except Exception as e:
            print(f"An error occurred while generating the indicator distribution chart: {e}")


def main():
    # Initialize the evaluator
    evaluator = QuantitativeEvaluator()

    base_dir = r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100"

    lr_dir = os.path.join(base_dir, "LR")  # Low-resolution image directory
    hr_dir = os.path.join(base_dir, "HR")  # High-resolution image directory
    sr_dir = os.path.join(base_dir, "SR")  # Super-resolution result directory
    output_dir = os.path.join(base_dir, "evaluation_results")  # output directory

    # Check if the directory exists
    for dir_path, dir_name in [(lr_dir, "LR"), (hr_dir, "HR"), (sr_dir, "SR")]:
        if not os.path.exists(dir_path):
            print(f"error: {dir_name}The directory does not exist: {dir_path}")
            print("Please ensure that the following directory exists and contains the image files:")
            print(f"LR Directory: {lr_dir}")
            print(f"HR Directory: {hr_dir}")
            print(f"SR Directory: {sr_dir}")
            return

    # Carry out a comprehensive assessment
    print("Start the assessment...")
    results, fid_value = evaluator.comprehensive_evaluation(lr_dir, hr_dir, sr_dir, output_dir)

    if results:
        print(f"Evaluation completed! The FID value is: {fid_value:.4f}")
        print(f"The detailed report is saved in: {output_dir}")
    else:
        print("The assessment failed. Please check the path and the image file")


if __name__ == "__main__":
    main()