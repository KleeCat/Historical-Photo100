"""主窗口：组装 sidebar + display + statusbar，连接所有信号。"""
import json
import logging
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QFileDialog, QMessageBox,
)

from .display import ImageDisplayWidget
from .dialogs import PreviewDialog
from .models import ModelManager
from .metrics import calculate_metrics
from .processing import (
    clamp_value, estimate_image_metrics, make_comparison_images,
    tensor_to_grid_image, save_feature_grids, IMAGE_EXTS,
    UserCancelledError,
)
from .sidebar import SidebarWidget
from .statusbar import StatusBarWidget
from .styles import generate_stylesheet, UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT, c, UI_COLOR_TEXT_PRIMARY, set_dark_mode
from .icon_helper import clear_cache as clear_icon_cache
from .utils import (
    ensure_dir, read_image, save_image, safe_basename, timestamp_str,
    write_json_file,
)
from .workers import ModelLoadWorker, ProcessWorker, BatchWorker

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """PySide6 主窗口。"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Super-Resolution System (ESRGAN)")
        self.resize(UI_WINDOW_WIDTH, UI_WINDOW_HEIGHT)

        # Core state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_folder = os.environ.get(
            "REALESRGAN_MODEL_DIR",
            os.path.join(os.path.expanduser("~"), ".cache", "realesrgan"),
        )
        self.model_manager = ModelManager(self.device, self.model_folder)
        self.scale_factor = 4
        self.img_input: Optional[np.ndarray] = None
        self.img_output: Optional[np.ndarray] = None
        self.img_gt: Optional[np.ndarray] = None
        self.input_path: Optional[str] = None
        self.gt_path: Optional[str] = None
        self.last_run_dir: Optional[str] = None
        self.default_output_dir: Optional[str] = None
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_path = os.path.join(
            os.path.dirname(self.project_dir), "output_config.json"
        )

        # Worker references
        self._model_worker: Optional[ModelLoadWorker] = None
        self._process_worker: Optional[ProcessWorker] = None
        self._batch_worker: Optional[BatchWorker] = None

        # Settings cache
        self._face_enhance = False
        self._scratch_repair = False
        self._face_blend = 0.65
        self._natural_blend = 0.0
        self._texture_boost = 0.08
        self._film_grain = 0.0
        self._dark_mode = False

        # Load config (may set _dark_mode)
        self._load_config()
        set_dark_mode(self._dark_mode)

        # Apply stylesheet
        self.setStyleSheet(generate_stylesheet(dark=self._dark_mode))

        # Build UI
        self._build_ui()
        self._connect_signals()

        # Start model loading
        self.statusbar.set_status(f"Initializing core components ({self.device})...")
        self._start_model_loading()

    def _build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Sidebar
        self.sidebar = SidebarWidget()
        layout.addWidget(self.sidebar)

        # Right side: display + statusbar
        right = QVBoxLayout()
        right.setContentsMargins(16, 12, 16, 8)
        right.setSpacing(8)

        self.display = ImageDisplayWidget()
        right.addWidget(self.display, stretch=1)

        self.statusbar = StatusBarWidget()
        right.addWidget(self.statusbar)

        layout.addLayout(right, stretch=1)

        # Set output dir text if loaded
        if self.default_output_dir:
            self.sidebar.set_output_dir_text(self.default_output_dir)

        # Restore dark mode toggle state
        if self._dark_mode:
            self.sidebar.chk_dark_mode.setChecked(True)

    def _connect_signals(self) -> None:
        sb = self.sidebar
        sb.open_image_clicked.connect(self.open_image)
        sb.load_gt_clicked.connect(self.load_gt)
        sb.scale_changed.connect(self._on_scale_changed)
        sb.output_dir_clicked.connect(self._set_output_dir)
        sb.face_enhance_toggled.connect(self._on_face_enhance_toggled)
        sb.scratch_repair_toggled.connect(lambda v: setattr(self, '_scratch_repair', v))
        sb.face_blend_changed.connect(lambda v: setattr(self, '_face_blend', v))
        sb.natural_blend_changed.connect(lambda v: setattr(self, '_natural_blend', v))
        sb.texture_boost_changed.connect(lambda v: setattr(self, '_texture_boost', v))
        sb.film_grain_changed.connect(lambda v: setattr(self, '_film_grain', v))
        sb.compare_toggled.connect(self.display.set_compare_mode)
        sb.compare_split_changed.connect(self.display.set_compare_split)
        sb.start_clicked.connect(self.start_processing)
        sb.batch_clicked.connect(self.run_batch)
        sb.cancel_clicked.connect(self._cancel_processing)
        sb.dark_mode_toggled.connect(self._toggle_dark_mode)

        dp = self.display
        dp.comparison_clicked.connect(self.save_comparison)
        dp.features_clicked.connect(self.export_features)
        dp.open_folder_clicked.connect(self.open_output_folder)
        dp.save_clicked.connect(self.save_result)

    # --- Config ---

    def _load_config(self) -> None:
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.default_output_dir = data.get("default_output_dir")
            self._dark_mode = data.get("dark_mode", False)
        except Exception as e:
            logger.warning("Failed to load config: %s", e)

    def _save_config(self) -> None:
        try:
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump({
                    "default_output_dir": self.default_output_dir,
                    "dark_mode": self._dark_mode,
                }, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save config: %s", e)

    # --- Model loading ---

    def _start_model_loading(self) -> None:
        self._model_worker = ModelLoadWorker(self.model_manager, self.scale_factor)
        self._model_worker.progress.connect(
            lambda v, t: self.statusbar.set_status(t)
        )
        self._model_worker.finished.connect(self._on_model_loaded)
        self.sidebar.combo_scale.setEnabled(False)
        self._model_worker.start()

    def _on_model_loaded(self, success: bool, msg: str) -> None:
        self.sidebar.combo_scale.setEnabled(True)
        if success:
            self.statusbar.set_status(msg)
        else:
            self.statusbar.set_status("Model load failed")
            QMessageBox.critical(self, "Model Error", msg)

    def _on_scale_changed(self, scale: int) -> None:
        self.scale_factor = scale
        self.statusbar.set_status(f"Switching to x{scale} model...")
        self._start_model_loading()

    # --- File operations ---

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.jpg *.png *.jpeg *.bmp)"
        )
        if not path:
            return
        img = read_image(path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to load image: {path}")
            return
        self.input_path = path
        self.img_input = img
        self.img_output = None
        self.img_gt = None
        self.gt_path = None

        self.display.show_input(img, os.path.basename(path))
        self.display.clear_output()
        self.display.show_overlay("Waiting for processing...")
        self.display.set_toolbar_enabled()
        self.display.reset_view()

        self.statusbar.set_status(f"Loaded: {os.path.basename(path)}")
        self.statusbar.set_progress(0)

        self.sidebar.update_resolution(
            (img.shape[1], img.shape[0]), None
        )
        self.sidebar.update_metrics("PSNR: --", "SSIM: --", "Load GT for metrics")

        self._auto_tune_parameters()

    def load_gt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Ground Truth", "", "Images (*.jpg *.png *.jpeg *.bmp)"
        )
        if not path:
            return
        img = read_image(path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to load GT: {path}")
            return
        self.img_gt = img
        self.gt_path = path
        self.sidebar.lbl_gt_hint.setText(f"GT: {os.path.basename(path)}")
        self._update_metrics()

    def _set_output_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if d:
            self.default_output_dir = d
            self._save_config()
            self.sidebar.set_output_dir_text(d)
            self.statusbar.set_status("Output directory updated.")

    def _on_face_enhance_toggled(self, checked: bool) -> None:
        self._face_enhance = checked

    def _toggle_dark_mode(self, dark: bool) -> None:
        self._dark_mode = dark
        set_dark_mode(dark)
        self.setStyleSheet(generate_stylesheet(dark=dark))
        clear_icon_cache()
        self.display.update_overlay_color(dark)
        self._save_config()

    # --- Processing ---

    def start_processing(self) -> None:
        if self._process_worker and self._process_worker.isRunning():
            return
        if self._batch_worker and self._batch_worker.isRunning():
            QMessageBox.information(self, "Info", "Batch processing is running.")
            return
        if self.img_input is None:
            QMessageBox.information(self, "Info", "Please load an input image first.")
            return
        if self.model_manager.upsampler is None:
            QMessageBox.warning(self, "Model", "Model is still loading. Please wait.")
            return

        self.display.reset_view()
        # Clear stale feature maps from previous run
        with self.model_manager._state_lock:
            self.model_manager.feature_maps = []
        settings = self._gather_settings()
        output_dir = self._get_output_dir()

        self._process_worker = ProcessWorker(
            self.model_manager, self.img_input, self.input_path,
            self.img_gt, settings, output_dir,
        )
        self._process_worker.progress.connect(self._on_process_progress)
        self._process_worker.image_ready.connect(self._on_process_image_ready)
        self._process_worker.finished.connect(self._on_process_finished)

        self.sidebar.set_processing_state(True)
        self.statusbar.start_timer()
        self.display.show_overlay("Processing...")
        self._process_worker.start()

    def _on_process_progress(self, value: float, text: str) -> None:
        self.statusbar.set_progress(value)
        self.statusbar.set_status(text)

    def _on_process_image_ready(self, img: object) -> None:
        if isinstance(img, np.ndarray):
            self.img_output = img
            base = safe_basename(self.input_path)
            self.display.show_output(img, f"{base}_x{self.scale_factor}.png")

    def _on_process_finished(self, success: bool, msg: str) -> None:
        self.statusbar.stop_timer()
        self.sidebar.set_processing_state(False)

        if success:
            self.statusbar.set_status(msg)
            self.last_run_dir = self._process_worker.run_dir if self._process_worker else None
            has_features = bool(self.model_manager.feature_maps)
            self.display.set_toolbar_enabled(
                compare=True, features=has_features,
                folder=self.last_run_dir is not None, save=True,
            )
            self.sidebar.update_resolution(
                (self.img_input.shape[1], self.img_input.shape[0]) if self.img_input is not None else None,
                (self.img_output.shape[1], self.img_output.shape[0]) if self.img_output is not None else None,
            )
            self._update_metrics()
        else:
            if msg == "Cancelled":
                self.statusbar.set_status("Cancelled")
                self.display.show_overlay("Cancelled")
            else:
                self.statusbar.set_status("Processing failed")
                self.display.show_overlay("Processing failed")
                QMessageBox.critical(self, "Error", f"Processing failed: {msg}")

    def _cancel_processing(self) -> None:
        if self._process_worker and self._process_worker.isRunning():
            self._process_worker.requestInterruption()
            self.sidebar.set_cancel_state()
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.requestInterruption()
            self.sidebar.set_cancel_state()

    # --- Batch processing ---

    def run_batch(self) -> None:
        if self._process_worker and self._process_worker.isRunning():
            QMessageBox.information(self, "Info", "Processing is already running.")
            return
        if self._batch_worker and self._batch_worker.isRunning():
            QMessageBox.information(self, "Info", "Batch is already running.")
            return
        if self.model_manager.upsampler is None:
            QMessageBox.warning(self, "Model", "Model is still loading.")
            return

        folder = QFileDialog.getExistingDirectory(self, "Select Batch Folder")
        if not folder or not os.path.isdir(folder):
            return

        files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS)
        ])
        if not files:
            QMessageBox.information(self, "Info", "No supported images found.")
            return

        batch_dir = self._get_output_dir(os.path.join("batch", timestamp_str()))
        queue_path = os.path.join(batch_dir, "batch_queue.json")
        write_json_file(queue_path, {
            "timestamp": timestamp_str(),
            "folder": folder,
            "total": len(files),
            "files": [os.path.basename(p) for p in files],
        })

        QMessageBox.information(
            self, "Batch", f"Found {len(files)} images.\nQueue saved to: {queue_path}"
        )

        settings = self._gather_settings()
        retry_limit = self.sidebar.spin_batch_retry.value()

        self._batch_worker = BatchWorker(
            self.model_manager, files, settings, batch_dir, retry_limit,
        )
        self._batch_worker.item_started.connect(self._on_batch_item_started)
        self._batch_worker.item_done.connect(self._on_batch_item_done)
        self._batch_worker.progress.connect(self._on_process_progress)
        self._batch_worker.finished.connect(self._on_batch_finished)

        self.sidebar.set_batch_state(True)
        self.statusbar.start_timer()
        self.display.hide_overlay()
        self.last_run_dir = batch_dir
        self._batch_worker.start()

    def _on_batch_item_started(self, idx: int, total: int, filename: str) -> None:
        self.statusbar.set_status(f"Batch {idx + 1}/{total}: {filename}")

    def _on_batch_item_done(self, idx: int, total: int) -> None:
        self.statusbar.set_progress((idx + 1) / total)

    def _on_batch_finished(self, success: bool, msg: str, errors: list) -> None:
        self.statusbar.stop_timer()
        self.sidebar.set_processing_state(False)
        self.statusbar.set_status(msg)
        self.display.set_toolbar_enabled(folder=self.last_run_dir is not None)
        if errors:
            error_text = "\n".join(errors[:10])
            QMessageBox.warning(self, "Batch Errors", f"{len(errors)} errors:\n{error_text}")
        else:
            QMessageBox.information(self, "Batch Complete", msg)

    # --- Save / Export ---

    def save_comparison(self) -> None:
        if self.img_input is None or self.img_output is None:
            return
        base_name = safe_basename(self.input_path)
        try:
            sr_h, sr_w = self.img_output.shape[:2]
            lr_up = cv2.resize(self.img_input, (sr_w, sr_h), interpolation=cv2.INTER_CUBIC)
            preview = np.hstack([lr_up, self.img_output])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Preview failed: {e}")
            return

        dlg = PreviewDialog(
            self, "Comparison Preview", preview,
            "Preview: LR (upscaled) | SR", "Save Comparison",
        )

        def on_save(dialog):
            out_dir = self._get_output_dir(f"compare/{base_name}_{timestamp_str()}")
            try:
                pair_path, grid_path = make_comparison_images(
                    self.img_input, self.img_output, self.scale_factor, base_name, out_dir,
                )
                QMessageBox.information(
                    self, "Saved", f"Saved:\n{pair_path}\n{grid_path}"
                )
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Save failed: {e}")

        dlg.set_save_callback(on_save)
        dlg.exec()

    def export_features(self) -> None:
        feature_maps = list(self.model_manager.feature_maps)
        if not feature_maps:
            QMessageBox.information(self, "Info", "No feature maps captured.")
            return
        base_name = safe_basename(self.input_path)
        grids = [tensor_to_grid_image(t) for _, t in feature_maps]
        grids = [g for g in grids if g is not None]
        if not grids:
            QMessageBox.information(self, "Info", "No feature grids generated.")
            return

        dlg = PreviewDialog(
            self, "Feature Preview", grids[0],
            f"Captured feature maps: {len(grids)}", "Save All",
        )

        def on_save(dialog):
            out_dir = self._get_output_dir(f"features/{base_name}_{timestamp_str()}")
            try:
                saved = save_feature_grids(
                    feature_maps, base_name, self.scale_factor, out_dir,
                )
                QMessageBox.information(self, "Saved", f"Feature grids saved: {len(saved)}")
                dialog.accept()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed: {e}")

        dlg.set_save_callback(on_save)
        dlg.exec()

    def save_result(self) -> None:
        if self.img_output is None:
            return

        dlg = PreviewDialog(
            self, "Result Preview", self.img_output, "", "Save As",
        )

        def on_save(dialog):
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Result", "",
                "PNG (*.png);;JPG (*.jpg);;WebP (*.webp);;TIFF (*.tiff);;BMP (*.bmp)",
            )
            if path:
                save_image(path, self.img_output)
                QMessageBox.information(self, "Saved", "Image saved successfully")
                dialog.accept()

        dlg.set_save_callback(on_save)
        dlg.exec()

    def open_output_folder(self) -> None:
        if not self.last_run_dir:
            return
        resolved = os.path.realpath(self.last_run_dir)
        if not os.path.isdir(resolved):
            QMessageBox.critical(self, "Error", "Directory not found.")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(resolved)
            elif sys.platform == "darwin":
                subprocess.run(["open", resolved], check=False)
            else:
                subprocess.run(["xdg-open", resolved], check=False)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Open folder failed: {e}")

    # --- Utility ---

    def _gather_settings(self) -> Dict[str, Any]:
        return {
            "scale": self.scale_factor,
            "face_enhance": self._face_enhance,
            "face_blend": self._face_blend,
            "natural_blend": self._natural_blend,
            "texture_boost": self._texture_boost,
            "film_grain": self._film_grain,
            "scratch_repair": self._scratch_repair,
        }

    def _get_output_dir(self, subdir: str = "") -> str:
        if self.default_output_dir:
            base = self.default_output_dir
        else:
            base = os.path.join(
                os.path.dirname(self.project_dir), "outputs"
            )
        if subdir:
            out = os.path.join(base, subdir)
        else:
            out = base
        ensure_dir(out)
        return out

    def _update_metrics(self) -> None:
        if self.img_output is None or self.img_gt is None:
            return
        try:
            result = calculate_metrics(self.img_output, self.img_gt)
            psnr_text = f"PSNR: {result['psnr']:.2f}" if result["psnr"] is not None else "PSNR: --"
            ssim_text = f"SSIM: {result['ssim']:.4f}" if result["ssim"] is not None else "SSIM: --"
            self.sidebar.update_metrics(psnr_text, ssim_text, f"GT: {os.path.basename(self.gt_path or '')}")
        except Exception as e:
            logger.warning("Metrics calculation failed: %s", e)

    def _auto_tune_parameters(self) -> None:
        if self.img_input is None:
            return
        try:
            metrics = estimate_image_metrics(self.img_input)
            sharpness_norm = clamp_value((metrics["lap_var"] - 20.0) / 380.0, 0.0, 1.0)
            noise_norm = clamp_value((metrics["noise_sigma"] - 2.0) / 18.0, 0.0, 1.0)
            contrast_norm = clamp_value((metrics["contrast"] - 20.0) / 60.0, 0.0, 1.0)
            edge_norm = clamp_value((metrics["edge_density"] - 0.02) / 0.08, 0.0, 1.0)

            fb = clamp_value(0.6 + sharpness_norm * 0.2 - noise_norm * 0.1, 0.4, 0.9)
            nb = clamp_value(0.03 + noise_norm * 0.07 + (1.0 - contrast_norm) * 0.05, 0.0, 0.12)
            tb = clamp_value(
                0.10 + (1.0 - sharpness_norm) * 0.18 + edge_norm * 0.06 - noise_norm * 0.08,
                0.0, 0.35,
            )
            if edge_norm > 0.25:
                nb = 0.0
                tb = 0.0
            fg = clamp_value(
                0.03 + (1.0 - edge_norm) * 0.12 + (1.0 - contrast_norm) * 0.08, 0.0, 0.5
            )

            self._face_blend = fb
            self._natural_blend = nb
            self._texture_boost = tb
            self._film_grain = fg
            self.sidebar.set_slider_values(fb, nb, tb, fg)
            self.statusbar.set_status("Auto tuned")
        except Exception as e:
            logger.warning("Auto tune failed: %s", e)

    def closeEvent(self, event) -> None:
        self.model_manager.cleanup()
        if self._process_worker and self._process_worker.isRunning():
            self._process_worker.requestInterruption()
            self._process_worker.wait(3000)
        if self._batch_worker and self._batch_worker.isRunning():
            self._batch_worker.requestInterruption()
            self._batch_worker.wait(3000)
        event.accept()


def main() -> None:
    """应用入口。"""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
