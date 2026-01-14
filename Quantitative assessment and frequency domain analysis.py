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
        # 修复LPIPS初始化
        try:
            self.lpips_model = lpips.LPIPS(net='vgg').to(device)
            self.lpips_model.eval()
        except Exception as e:
            print(f"LPIPS初始化失败: {e}")
            self.lpips_model = None

    def check_path_exists(self, path):
        """检查路径是否存在"""
        if not os.path.exists(path):
            print(f"警告: 路径不存在: {path}")
            return False
        return True

    def find_matching_filename(self, hr_filename, target_dir):
        """根据HR文件名找到对应的LR或SR文件名"""
        # HR文件名格式: 0801.png
        # LR/SR文件名格式: 0801x4.png
        name_without_ext = os.path.splitext(hr_filename)[0]  # 0801
        target_filename = f"{name_without_ext}x4.png"  # 0801x4.png

        # 检查是否还有其他可能的格式
        possible_names = [
            target_filename,
            f"{name_without_ext}x4.jpg",
            f"{name_without_ext}x4.jpeg",
            f"{name_without_ext}x4.bmp",
            hr_filename  # 也检查原始文件名
        ]

        for name in possible_names:
            test_path = os.path.join(target_dir, name)
            if os.path.exists(test_path):
                return test_path

        return None

    def calculate_psnr_ssim(self, img1_path, img2_path):
        """计算PSNR和SSIM指标"""
        if not self.check_path_exists(img1_path) or not self.check_path_exists(img2_path):
            return 0, 0

        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            if img1 is None or img2 is None:
                print(f"无法读取图像: {img1_path} 或 {img2_path}")
                return 0, 0

            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

            # 使用Y通道计算
            if len(img1.shape) == 3:
                img1_y = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)[:, :, 0]
                img2_y = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)[:, :, 0]
            else:
                img1_y, img2_y = img1, img2

            psnr_val = peak_signal_noise_ratio(img1_y, img2_y, data_range=255)
            ssim_val = structural_similarity(img1_y, img2_y, data_range=255)

            return psnr_val, ssim_val
        except Exception as e:
            print(f"计算PSNR/SSIM时出错: {e}")
            return 0, 0

    def calculate_lpips(self, img1_path, img2_path):
        """计算LPIPS感知相似度"""
        if self.lpips_model is None:
            print("LPIPS模型不可用，返回默认值0")
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
            print(f"计算LPIPS时出错: {e}")
            return 0.0

    def calculate_fid(self, real_images_dir, generated_images_dir):
        """计算FID指标"""
        if not self.check_path_exists(real_images_dir) or not self.check_path_exists(generated_images_dir):
            return 999.0  # 返回一个较大的值表示错误

        try:
            fid_value = fid_score.calculate_fid_given_paths(
                [real_images_dir, generated_images_dir],
                batch_size=32,
                device=self.device,
                dims=2048
            )
            return fid_value
        except Exception as e:
            print(f"计算FID时出错: {e}")
            return 999.0

    def frequency_analysis(self, image_path, save_spectrum_path=None):
        """频域能量分析"""
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

            # 确保索引不越界
            low_start_y = max(center_y - 30, 0)
            low_end_y = min(center_y + 30, h)
            low_start_x = max(center_x - 30, 0)
            low_end_x = min(center_x + 30, w)

            low_freq = magnitude_spectrum[low_start_y:low_end_y, low_start_x:low_end_x]
            energy_low = np.mean(low_freq) if low_freq.size > 0 else 0

            # 中频区域
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
            print(f"频域分析时出错: {e}")
            return {'low_freq_energy': 0, 'mid_freq_energy': 0, 'high_freq_energy': 0, 'spectrum': None}

    def comprehensive_evaluation(self, lr_dir, hr_dir, sr_dir, output_dir):
        """综合评估"""
        # 检查所有路径是否存在
        for dir_path, dir_name in [(lr_dir, "LR"), (hr_dir, "HR"), (sr_dir, "SR")]:
            if not self.check_path_exists(dir_path):
                print(f"错误: {dir_name}目录不存在: {dir_path}")
                return [], 999.0

        os.makedirs(output_dir, exist_ok=True)

        # 获取HR目录中的图像列表
        try:
            image_names = [f for f in os.listdir(hr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        except Exception as e:
            print(f"读取HR目录失败: {e}")
            return [], 999.0

        if not image_names:
            print("HR目录中没有找到图像文件")
            return [], 999.0

        print(f"找到 {len(image_names)} 张图像进行评估")

        results = []
        frequency_results = []
        valid_count = 0

        for idx, img_name in enumerate(image_names):
            print(f"处理图像 {idx + 1}/{len(image_names)}: {img_name}")

            hr_path = os.path.join(hr_dir, img_name)

            # 查找对应的LR文件
            lr_path = self.find_matching_filename(img_name, lr_dir)
            if lr_path is None:
                print(f"跳过 {img_name} - 在LR目录中找不到匹配的图像")
                continue

            # 查找对应的SR文件
            sr_path = self.find_matching_filename(img_name, sr_dir)
            if sr_path is None:
                print(f"跳过 {img_name} - 在SR目录中找不到匹配的图像")
                continue

            # 计算指标
            psnr, ssim = self.calculate_psnr_ssim(hr_path, sr_path)
            lpips_val = self.calculate_lpips(hr_path, sr_path)

            # 频域分析
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

            # 每处理10张图像保存一次进度
            if valid_count % 10 == 0:
                print(f"已处理 {valid_count} 张图像...")

        if valid_count == 0:
            print("没有有效的图像对可用于评估")
            return [], 999.0

        # 计算FID
        print("计算FID指标...")
        fid_value = self.calculate_fid(hr_dir, sr_dir)

        # 生成报告
        self.generate_report(results, fid_value, output_dir, frequency_results, lr_dir)

        print(f"评估完成！有效图像: {valid_count}/{len(image_names)}")
        return results, fid_value

    def generate_report(self, results, fid_value, output_dir, frequency_results, lr_dir):
        """生成评估报告"""
        if not results:
            print("没有结果可生成报告")
            return

        avg_psnr = np.mean([r['psnr'] for r in results])
        avg_ssim = np.mean([r['ssim'] for r in results])
        avg_lpips = np.mean([r['lpips'] for r in results])

        # 生成报告文本
        report_text = f"""
评估报告 - Historical-Photo100 数据集
==================================
评估时间: {np.datetime64('now')}
有效图像数量: {len(results)}

总体指标:
- 平均 PSNR: {avg_psnr:.4f} dB
- 平均 SSIM: {avg_ssim:.4f}
- 平均 LPIPS: {avg_lpips:.4f} (越低越好)
- FID: {fid_value:.4f} (越低越好)

详细结果:
{'图像名称':<30} {'PSNR':<10} {'SSIM':<10} {'LPIPS':<10}
{'-' * 60}
"""

        for result in results:
            report_text += f"{result['image_name']:<30} {result['psnr']:<10.4f} {result['ssim']:<10.4f} {result['lpips']:<10.4f}\n"

        # 频域分析总结
        report_text += f"\n频域能量分析:\n"
        avg_lr_high = np.mean([r['freq_analysis']['lr']['high_freq_energy'] for r in results])
        avg_sr_high = np.mean([r['freq_analysis']['sr']['high_freq_energy'] for r in results])
        avg_hr_high = np.mean([r['freq_analysis']['hr']['high_freq_energy'] for r in results])

        report_text += f"- 平均高频能量 - LR: {avg_lr_high:.2f}, SR: {avg_sr_high:.2f}, HR: {avg_hr_high:.2f}\n"
        if avg_hr_high - avg_lr_high > 1e-6:
            recovery_rate = (avg_sr_high - avg_lr_high) / (avg_hr_high - avg_lr_high) * 100
            report_text += f"- 高频恢复率: {recovery_rate:.2f}%\n"
        else:
            report_text += "- 高频恢复率: 无法计算 (HR和LR高频能量差异太小)\n"

        # 保存报告
        report_path = os.path.join(output_dir, 'evaluation_report.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(f"报告已保存至: {report_path}")

        # 生成可视化图表
        self.plot_frequency_comparison(frequency_results, output_dir, lr_dir)
        self.plot_metrics_distribution(results, output_dir)

    def plot_frequency_comparison(self, frequency_results, output_dir, lr_dir):
        """绘制频域能量对比图"""
        if not frequency_results:
            return

        # 选择几个典型样本进行详细展示
        sample_indices = min(3, len(frequency_results))

        fig, axes = plt.subplots(sample_indices, 4, figsize=(20, 5 * sample_indices))
        if sample_indices == 1:
            axes = axes.reshape(1, -1)

        for i in range(sample_indices):
            data = frequency_results[i]

            # 原始图像
            try:
                # 使用匹配的LR图像路径
                lr_path = self.find_matching_filename(data['name'], lr_dir)
                if lr_path and os.path.exists(lr_path):
                    lr_img = cv2.imread(lr_path)
                    lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2RGB)
                    axes[i, 0].imshow(lr_img)
                axes[i, 0].set_title(f'LR - {data["name"]}')
                axes[i, 0].axis('off')
            except Exception as e:
                print(f"显示LR图像失败: {e}")
                axes[i, 0].text(0.5, 0.5, '图像加载失败', ha='center', va='center')
                axes[i, 0].axis('off')

            # 频域对比
            spectra = [data['lr_spectrum'], data['sr_spectrum'], data.get('hr_spectrum', None)]
            titles = ['LR Spectrum', 'SR Spectrum', 'HR Spectrum']

            for k in range(3):
                if k < len(spectra) and spectra[k] is not None:
                    im = axes[i, k + 1].imshow(spectra[k], cmap='hot')
                    axes[i, k + 1].set_title(titles[k])
                    axes[i, k + 1].axis('off')
                    plt.colorbar(im, ax=axes[i, k + 1])
                else:
                    axes[i, k + 1].text(0.5, 0.5, '频谱数据缺失', ha='center', va='center')
                    axes[i, k + 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'frequency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print("频域对比图已生成")

    def plot_metrics_distribution(self, results, output_dir):
        """绘制指标分布图"""
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
                axes[i].set_ylabel('频数')
                axes[i].set_title(f'{metric_names[i]}分布')
                axes[i].grid(True, alpha=0.3)

                mean_val = np.mean(values)
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'均值: {mean_val:.3f}')
                axes[i].legend()

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'metrics_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print("指标分布图已生成")
        except Exception as e:
            print(f"生成指标分布图时出错: {e}")


# 使用示例
def main():
    # 初始化评估器
    evaluator = QuantitativeEvaluator()

    # 设置正确的路径 - 修改为您的实际路径
    base_dir = r"D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100"

    lr_dir = os.path.join(base_dir, "LR")  # 低分辨率图像目录
    hr_dir = os.path.join(base_dir, "HR")  # 高分辨率图像目录
    sr_dir = os.path.join(base_dir, "SR")  # 超分辨率结果目录
    output_dir = os.path.join(base_dir, "evaluation_results")  # 输出目录

    # 检查目录是否存在
    for dir_path, dir_name in [(lr_dir, "LR"), (hr_dir, "HR"), (sr_dir, "SR")]:
        if not os.path.exists(dir_path):
            print(f"错误: {dir_name}目录不存在: {dir_path}")
            print("请确保以下目录存在并包含图像文件:")
            print(f"LR目录: {lr_dir}")
            print(f"HR目录: {hr_dir}")
            print(f"SR目录: {sr_dir}")
            return

    # 执行综合评估
    print("开始评估...")
    results, fid_value = evaluator.comprehensive_evaluation(lr_dir, hr_dir, sr_dir, output_dir)

    if results:
        print(f"评估完成！FID值为: {fid_value:.4f}")
        print(f"详细报告保存在: {output_dir}")
    else:
        print("评估失败，请检查路径和图像文件")


if __name__ == "__main__":
    main()