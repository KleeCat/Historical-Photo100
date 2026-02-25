"""图像显示区：双面板 + 缩放/平移 + 对比模式。

ImagePanel(QGraphicsView) — 单个图像面板，内置缩放/平移。
ImageDisplayWidget(QWidget) — 双面板容器 + 工具栏。
"""
import numpy as np
from PySide6.QtCore import Qt, Signal, QRectF
from PySide6.QtGui import QPixmap, QWheelEvent, QMouseEvent, QPainter
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QFrame, QSizePolicy,
)

from .styles import UI_COLOR_TEXT_MUTED, UI_COLOR_SECONDARY_BG
from .utils import numpy_to_qpixmap


class ImagePanel(QGraphicsView):
    """单个图像面板，支持滚轮缩放和右键平移。"""

    zoom_changed = Signal(float, float, float)  # (zoom_factor, cx, cy)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._zoom = 1.0

        # Rendering
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_image(self, img_bgr: np.ndarray | None) -> None:
        """显示 BGR numpy 图像。"""
        self._scene.clear()
        self._pixmap_item = None
        if img_bgr is None:
            return
        pixmap = numpy_to_qpixmap(img_bgr)
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_in_view()

    def set_pixmap(self, pixmap: QPixmap) -> None:
        """直接设置 QPixmap。"""
        self._scene.clear()
        self._pixmap_item = None
        if pixmap is None or pixmap.isNull():
            return
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self.fit_in_view()

    def fit_in_view(self) -> None:
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
            self._zoom = 1.0

    def clear_image(self) -> None:
        self._scene.clear()
        self._pixmap_item = None

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self._zoom *= factor
        self.scale(factor, factor)
        center = self.mapToScene(self.viewport().rect().center())
        self.zoom_changed.emit(self._zoom, center.x(), center.y())

    def sync_transform(self, zoom: float, cx: float, cy: float) -> None:
        """从另一个面板同步缩放/平移，保留各自的 base fit transform。"""
        if self._pixmap_item is None:
            return
        # First fit to get the base transform, then apply relative zoom
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        if zoom != 1.0:
            self.scale(zoom, zoom)
        self._zoom = zoom
        # Map center from scene coordinates — both panels share scene coords
        self.centerOn(cx, cy)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap_item is not None and self._zoom <= 1.0:
            self.fit_in_view()


class ImageDisplayWidget(QWidget):
    """双面板图像显示区 + 底部工具栏。"""

    # Toolbar signals
    comparison_clicked = Signal()
    features_clicked = Signal()
    open_folder_clicked = Signal()
    save_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        # Headers
        header_row = QHBoxLayout()
        lbl_in = QLabel("Original Input")
        lbl_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_in.setStyleSheet("font-size: 14px; font-weight: bold;")
        lbl_out = QLabel("Super-Resolution Output")
        lbl_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl_out.setStyleSheet("font-size: 14px; font-weight: bold;")
        header_row.addWidget(lbl_in, stretch=1)
        header_row.addWidget(lbl_out, stretch=1)
        layout.addLayout(header_row)

        # Filename labels
        fname_row = QHBoxLayout()
        self.lbl_filename_in = QLabel("No file loaded")
        self.lbl_filename_in.setObjectName("muted")
        self.lbl_filename_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_filename_out = QLabel("")
        self.lbl_filename_out.setObjectName("muted")
        self.lbl_filename_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fname_row.addWidget(self.lbl_filename_in, stretch=1)
        fname_row.addWidget(self.lbl_filename_out, stretch=1)
        layout.addLayout(fname_row)

        # Image panels
        panels_row = QHBoxLayout()
        panels_row.setSpacing(10)

        self.panel_input = ImagePanel()
        self.panel_output = ImagePanel()
        panels_row.addWidget(self.panel_input, stretch=1)
        panels_row.addWidget(self.panel_output, stretch=1)
        layout.addLayout(panels_row, stretch=1)

        # Output overlay
        self._overlay = QLabel("Waiting for processing...", self.panel_output)
        self._overlay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._overlay.setStyleSheet(
            f"background-color: {UI_COLOR_SECONDARY_BG[0]}; "
            f"color: {UI_COLOR_TEXT_MUTED[0]}; "
            f"font-size: 14px; font-weight: bold; border-radius: 8px;"
        )
        self._overlay.setVisible(True)

        # Resolution labels
        res_row = QHBoxLayout()
        self.lbl_res_in = QLabel("Input: -- x --")
        self.lbl_res_in.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_res_out = QLabel("Output: -- x --")
        self.lbl_res_out.setAlignment(Qt.AlignmentFlag.AlignCenter)
        res_row.addWidget(self.lbl_res_in, stretch=1)
        res_row.addWidget(self.lbl_res_out, stretch=1)
        layout.addLayout(res_row)

        # Toolbar
        toolbar = QFrame()
        toolbar.setObjectName("card")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(6, 6, 6, 6)

        self.btn_compare = QPushButton("Comparison")
        self.btn_compare.setToolTip("Save side-by-side comparison image")
        self.btn_compare.setEnabled(False)
        self.btn_compare.clicked.connect(self.comparison_clicked)

        self.btn_features = QPushButton("Features")
        self.btn_features.setToolTip("Export feature map visualizations")
        self.btn_features.setEnabled(False)
        self.btn_features.clicked.connect(self.features_clicked)

        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setToolTip("Open the output folder in file explorer")
        self.btn_open_folder.setEnabled(False)
        self.btn_open_folder.clicked.connect(self.open_folder_clicked)

        self.btn_save = QPushButton("Save Result")
        self.btn_save.setToolTip("Save the processed result to disk")
        self.btn_save.setEnabled(False)
        self.btn_save.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.btn_save.clicked.connect(self.save_clicked)

        for btn in (self.btn_compare, self.btn_features, self.btn_open_folder, self.btn_save):
            tb_layout.addWidget(btn, stretch=1)
        layout.addWidget(toolbar)

        # Zoom sync
        self.panel_input.zoom_changed.connect(self._sync_output_zoom)
        self.panel_output.zoom_changed.connect(self._sync_input_zoom)

        # Compare mode state
        self._compare_mode = False
        self._compare_split = 0.5
        self._compare_input_pixmap: QPixmap | None = None
        self._compare_output_pixmap: QPixmap | None = None

    def _sync_output_zoom(self, zoom: float, cx: float, cy: float) -> None:
        self.panel_output.sync_transform(zoom, cx, cy)

    def _sync_input_zoom(self, zoom: float, cx: float, cy: float) -> None:
        self.panel_input.sync_transform(zoom, cx, cy)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_overlay_geometry()

    def _update_overlay_geometry(self) -> None:
        if self._overlay.isVisible():
            self._overlay.setGeometry(self.panel_output.rect())

    # --- Public API ---

    def show_input(self, img_bgr: np.ndarray, filename: str = "") -> None:
        self.panel_input.set_image(img_bgr)
        if filename:
            self.lbl_filename_in.setText(f"Input: {filename}")
        h, w = img_bgr.shape[:2]
        self.lbl_res_in.setText(f"Input: {w} x {h}")

    def show_output(self, img_bgr: np.ndarray, filename: str = "") -> None:
        self.panel_output.set_image(img_bgr)
        self.hide_overlay()
        if filename:
            self.lbl_filename_out.setText(f"Output: {filename}")
        h, w = img_bgr.shape[:2]
        self.lbl_res_out.setText(f"Output: {w} x {h}")

    def clear_output(self) -> None:
        self.panel_output.clear_image()
        self.lbl_filename_out.setText("")
        self.lbl_res_out.setText("Output: -- x --")

    def show_overlay(self, text: str = "Waiting for processing...") -> None:
        self._overlay.setText(text)
        self._overlay.setVisible(True)
        self._update_overlay_geometry()

    def hide_overlay(self) -> None:
        self._overlay.setVisible(False)

    def set_toolbar_enabled(self, compare: bool = False, features: bool = False,
                            folder: bool = False, save: bool = False) -> None:
        self.btn_compare.setEnabled(compare)
        self.btn_features.setEnabled(features)
        self.btn_open_folder.setEnabled(folder)
        self.btn_save.setEnabled(save)

    def reset_view(self) -> None:
        self.panel_input.fit_in_view()
        self.panel_output.fit_in_view()

    # --- Compare mode ---

    def set_compare_mode(self, enabled: bool) -> None:
        self._compare_mode = enabled
        if not enabled:
            # Restore normal dual-panel view
            self.panel_input.setVisible(True)
            self.panel_output.setVisible(True)
            return
        # Split-view: overlay input clip on output panel
        if not self.panel_output.has_image() or not self.panel_input.has_image():
            return
        self._update_compare_view()

    def set_compare_split(self, value: float) -> None:
        self._compare_split = value
        if self._compare_mode:
            self._update_compare_view()

    def _update_compare_view(self) -> None:
        """Update split-view clipping on the output panel."""
        # For now, compare mode shows both panels side by side
        # with a visual split indicator via the panel visibility
        # Full QGraphicsScene clip implementation is a future enhancement
        pass
