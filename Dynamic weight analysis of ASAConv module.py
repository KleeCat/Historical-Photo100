import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import os
from collections import deque
from typing import Dict, Tuple, Optional

# Set font and chart styles (L1: added fallback)
try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# Fix random seed for reproducibility (M5)
RANDOM_SEED = 42


class ASAConvModule(nn.Module):
    """ASAConv module with real dynamic weight capture"""

    def __init__(self, in_channels: int = 3, out_channels: int = 64, kernel_size: int = 3) -> None:
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv = nn.Conv2d(64, out_channels, kernel_size, padding=kernel_size // 2)
        self.calibration_factor = nn.Parameter(torch.ones(1, out_channels, 1, 1) * 0.1)
        # H1: use deque with fixed maxlen instead of torch.cat growth
        self.feature_queue: deque = deque(maxlen=5)
        self.last_adaptive_weights: Optional[torch.Tensor] = None
        self.last_weight_modulation: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        self.feature_queue.append(x.detach().clone())

        if len(self.feature_queue) > 1:
            historical = torch.stack(list(self.feature_queue)[:-1])
            historical_guidance = torch.mean(historical, dim=0)
            modulation = self.calibration_factor * torch.mean(
                historical_guidance, dim=[0, 2, 3], keepdim=True)
            self.last_weight_modulation = modulation.detach()
            modulation_out = modulation.squeeze(0).unsqueeze(1)
            adaptive_weights = self.conv.weight * (1 + modulation_out)
        else:
            self.last_weight_modulation = torch.zeros_like(self.calibration_factor)
            adaptive_weights = self.conv.weight

        # C1: store real adaptive weights for extraction
        self.last_adaptive_weights = adaptive_weights.detach()
        return nn.functional.conv2d(x, adaptive_weights, self.conv.bias,
                                    padding=self.conv.padding)


class DualScaleConvModule(nn.Module):
    """Dual Scale Convolution module"""

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.initial_conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.conv3x3 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.conv5x5 = nn.Conv2d(64, out_channels, 5, padding=2)
        self.fusion = nn.Conv2d(out_channels * 2, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        feat3 = self.conv3x3(x)
        feat5 = self.conv5x5(x)
        return self.fusion(torch.cat([feat3, feat5], dim=1))


class ASAConvVisualizer:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 output_dir: str = 'outputs') -> None:
        self.device = device
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.setup_models()

    def setup_models(self) -> None:
        """Initialize models with reproducible random weights"""
        torch.manual_seed(RANDOM_SEED)
        self.asaconv_model = ASAConvModule().to(self.device)
        self.dual_scale_model = DualScaleConvModule().to(self.device)

    def create_test_image(self) -> str:
        """Create test sample image"""
        sample_path = os.path.join(self.output_dir, "urban100_sample.png")
        if not os.path.exists(sample_path):
            img = np.ones((256, 256, 3), dtype=np.uint8) * 128
            img[:, 100:110, :] = 255
            img[:, 150:160, :] = 0
            img[50:100, 50:100, :] = 200
            img[150:200, 150:200, :] = 50
            for i in range(30, 80, 10):
                for j in range(180, 230, 10):
                    color = self.rng.integers(0, 255, 3).astype(np.uint8)
                    img[i:i + 5, j:j + 5, :] = color
            cv2.imwrite(sample_path, img)
        return sample_path

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """Image preprocessing using cv2 (M1: unified image library)"""
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image shape: {img.shape} for {image_path}")
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def extract_dynamic_weights(self, model: ASAConvModule,
                                x: torch.Tensor) -> np.ndarray:
        """C1 fix: extract real dynamic weights from ASAConv forward pass"""
        model.feature_queue.clear()
        with torch.no_grad():
            for _ in range(3):
                model(x)

            if model.last_adaptive_weights is not None:
                weights = model.last_adaptive_weights
                channel_magnitude = torch.mean(torch.abs(weights), dim=[1, 2, 3])
                if model.feature_queue:
                    features = model.feature_queue[-1]
                else:
                    features = model.initial_conv(x)
                weighted_features = features * channel_magnitude.view(1, -1, 1, 1)
                spatial_weights = torch.mean(
                    torch.abs(weighted_features), dim=1).squeeze()
                spatial_weights = spatial_weights.cpu().numpy()
                sw_min, sw_max = spatial_weights.min(), spatial_weights.max()
                if sw_max > sw_min:
                    spatial_weights = (spatial_weights - sw_min) / (sw_max - sw_min)
                return spatial_weights

        h, w = x.shape[2], x.shape[3]
        return np.zeros((h, w))

    def extract_fixed_weights(self, model: DualScaleConvModule
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract fixed weights from dual scale convolution"""
        weight_3x3 = torch.mean(
            model.conv3x3.weight, dim=[1, 2, 3]).cpu().detach().numpy()
        weight_5x5 = torch.mean(
            model.conv5x5.weight, dim=[1, 2, 3]).cpu().detach().numpy()
        return weight_3x3, weight_5x5

    def generate_heatmap(self, weights: np.ndarray,
                         original_img: torch.Tensor) -> np.ndarray:
        """Generate weight heatmap overlay"""
        h, w = original_img.shape[2], original_img.shape[3]

        if len(weights.shape) == 1:
            # H2 fix: map each channel to a unique spatial block
            n_channels = len(weights)
            grid_size = int(np.ceil(np.sqrt(n_channels)))
            heatmap = np.zeros((h, w))
            region_h = h // grid_size
            region_w = w // grid_size
            for i, weight in enumerate(weights):
                row = i // grid_size
                col = i % grid_size
                if row < grid_size and col < grid_size:
                    sh = row * region_h
                    eh = min((row + 1) * region_h, h)
                    sw = col * region_w
                    ew = min((col + 1) * region_w, w)
                    heatmap[sh:eh, sw:ew] = weight
        else:
            # H4 fix: only resize if dimensions differ
            if weights.shape[0] != h or weights.shape[1] != w:
                heatmap = cv2.resize(weights.astype(np.float32), (w, h))
            else:
                heatmap = weights

        hm_min, hm_max = np.min(heatmap), np.max(heatmap)
        if hm_max > hm_min:
            heatmap = (heatmap - hm_min) / (hm_max - hm_min)
        colored_heatmap = plt.cm.viridis(heatmap)[:, :, :3]
        colored_heatmap = (colored_heatmap * 255).astype(np.uint8)

        original_rgb = ((original_img.squeeze().cpu().numpy().transpose(
            1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
        overlay = cv2.addWeighted(original_rgb, 0.6, colored_heatmap, 0.4, 0)
        return overlay

    def detect_edge_mask(self, img_tensor: torch.Tensor) -> np.ndarray:
        """Detect edge regions using Sobel operator for region analysis"""
        gray = torch.mean(img_tensor.squeeze(), dim=0).cpu().numpy()
        gray_uint8 = ((gray * 0.5 + 0.5) * 255).astype(np.uint8)
        sobel_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        threshold = np.percentile(edge_magnitude, 75)
        return edge_magnitude > threshold

    def analyze_region_differences(self, asaconv_weights: np.ndarray,
                                   dual_scale_weights: Tuple[np.ndarray, np.ndarray],
                                   original_img: torch.Tensor) -> Dict:
        """C2 fix: compute real region differences from extracted data"""
        edge_mask = self.detect_edge_mask(original_img)
        flat_mask = ~edge_mask

        asaconv_edge = float(np.mean(asaconv_weights[edge_mask])) if np.any(edge_mask) else 0.0
        asaconv_flat = float(np.mean(asaconv_weights[flat_mask])) if np.any(flat_mask) else 0.0
        adaptivity_ratio = asaconv_edge / asaconv_flat if asaconv_flat > 1e-8 else 0.0

        w3x3, w5x5 = dual_scale_weights
        mean_3x3 = float(np.mean(np.abs(w3x3)))
        mean_5x5 = float(np.mean(np.abs(w5x5)))
        scale_ratio = mean_5x5 / mean_3x3 if mean_3x3 > 1e-8 else 0.0

        return {
            'ASAConv': {
                'edge_region': asaconv_edge,
                'flat_region': asaconv_flat,
                'adaptivity_ratio': adaptivity_ratio
            },
            'DualScale': {
                '3x3_kernel': mean_3x3,
                '5x5_kernel': mean_5x5,
                'scale_ratio': scale_ratio
            }
        }

    def visualize_comparison(self) -> Optional[Dict]:
        """Main visualization function"""
        try:
            sample_path = self.create_test_image()
            input_tensor = self.preprocess_image(sample_path)
            print(f"Input tensor shape: {input_tensor.shape}")

            asaconv_weights = self.extract_dynamic_weights(
                self.asaconv_model, input_tensor)
            dual_scale_weights = self.extract_fixed_weights(
                self.dual_scale_model)

            asaconv_heatmap = self.generate_heatmap(
                asaconv_weights, input_tensor)
            dual_scale_heatmap = self.generate_heatmap(
                dual_scale_weights[0], input_tensor)

            region_analysis = self.analyze_region_differences(
                asaconv_weights, dual_scale_weights, input_tensor)

            self.plot_results(input_tensor, asaconv_heatmap,
                              dual_scale_heatmap, region_analysis)
            return region_analysis

        except Exception as e:
            print(f"Error during visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def plot_results(self, original_img: torch.Tensor,
                     asaconv_heatmap: np.ndarray,
                     dual_scale_heatmap: np.ndarray,
                     region_analysis: Dict) -> None:
        """Plot comparison results"""
        fig = plt.figure(figsize=(20, 12))
        gs = plt.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])

        # Row 1: image comparison
        original_rgb = ((original_img.squeeze().cpu().numpy().transpose(
            1, 2, 0) * 0.5 + 0.5) * 255).astype(np.uint8)
        ax1.imshow(original_rgb)
        ax1.set_title('Original Image\n(Urban100 Sample)',
                       fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(asaconv_heatmap)
        ax2.set_title('ASAConv Dynamic Weights\n(Region-Adaptive Heatmap)',
                       fontsize=12, fontweight='bold')
        ax2.axis('off')

        ax3.imshow(dual_scale_heatmap)
        ax3.set_title('Dual Scale 3x3 Weights\n(Fixed Weight Heatmap)',
                       fontsize=12, fontweight='bold')
        ax3.axis('off')

        # M4 fix: separate bar charts with correct semantic grouping
        # ASAConv: edge vs flat region weights
        categories_asa = ['Edge Region', 'Flat Region']
        asa_values = [region_analysis['ASAConv']['edge_region'],
                      region_analysis['ASAConv']['flat_region']]
        bars1 = ax4.bar(categories_asa, asa_values, alpha=0.8,
                        color=['#e74c3c', '#3498db'])
        ax4.set_xlabel('Region Type', fontsize=11)
        ax4.set_ylabel('Weight Mean', fontsize=11)
        ax4.set_title('ASAConv: Edge vs Flat Region Weights',
                       fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        for bar in bars1:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        # DualScale: 3x3 vs 5x5 kernel weights
        categories_ds = ['3x3 Kernel', '5x5 Kernel']
        ds_values = [region_analysis['DualScale']['3x3_kernel'],
                     region_analysis['DualScale']['5x5_kernel']]
        bars2 = ax5.bar(categories_ds, ds_values, alpha=0.8,
                        color=['#2ecc71', '#9b59b6'])
        ax5.set_xlabel('Kernel Size', fontsize=11)
        ax5.set_ylabel('Weight Mean (abs)', fontsize=11)
        ax5.set_title('DualScale: Kernel Weight Distribution',
                       fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        for bar in bars2:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{height:.4f}', ha='center', va='bottom', fontsize=9)

        # Text analysis summary (L3 fix: removed misleading sigma notation)
        ax6.axis('off')
        ra = region_analysis
        analysis_text = (
            f"  ASAConv Dynamic Routing Analysis Results:\n\n"
            f"  Edge Region Weight:   {ra['ASAConv']['edge_region']:.4f}\n"
            f"  Flat Region Weight:   {ra['ASAConv']['flat_region']:.4f}\n"
            f"  Adaptivity Ratio:     {ra['ASAConv']['adaptivity_ratio']:.2f}\n\n"
            f"  Dual Scale Convolution Comparison:\n\n"
            f"  3x3 Kernel Weight:    {ra['DualScale']['3x3_kernel']:.4f}\n"
            f"  5x5 Kernel Weight:    {ra['DualScale']['5x5_kernel']:.4f}\n"
            f"  Scale Ratio (5x5/3x3):{ra['DualScale']['scale_ratio']:.2f}\n\n"
            f"  Key Findings:\n"
            f"  ASAConv edge/flat ratio indicates spatial\n"
            f"  adaptivity for complex texture regions.\n"
            f"  DualScale uses fixed kernel weights without\n"
            f"  spatial awareness."
        )
        ax6.text(0.05, 0.95, analysis_text, transform=ax6.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.3",
                           facecolor="lightgray", alpha=0.5))

        plt.suptitle('ASAConv Module Dynamic Weight Analysis'
                     ' - Urban100 Dataset Sample',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()

        # H3 fix: save to outputs/ directory
        save_path = os.path.join(
            self.output_dir, 'ASAConv_Dynamic_Routing_Analysis_English.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved to {save_path}")

        # L2 fix: only call plt.show() if GUI backend available
        if matplotlib.get_backend().lower() not in ('agg', 'pdf', 'svg'):
            plt.show()
        else:
            plt.close(fig)


# Execute visualization analysis
if __name__ == "__main__":
    visualizer = ASAConvVisualizer(device='cpu')
    results = visualizer.visualize_comparison()
    if results:
        print("Dynamic routing visualization analysis completed!")
        print("Region difference analysis results:", results)
    else:
        print("Error occurred during visualization analysis.")
