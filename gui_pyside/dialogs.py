"""对话框：预览窗口。"""
import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton, QGraphicsView,
    QGraphicsScene, QGraphicsPixmapItem,
)
from PySide6.QtGui import QPainter

from .utils import numpy_to_qpixmap


class PreviewDialog(QDialog):
    """图像预览对话框，带保存按钮。"""

    def __init__(
        self,
        parent=None,
        title: str = "Preview",
        img_bgr: np.ndarray | None = None,
        info_text: str = "",
        save_text: str = "Save",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(900, 900)
        self._save_callback = None

        layout = QVBoxLayout(self)

        # Image view
        self._scene = QGraphicsScene(self)
        self._view = QGraphicsView(self._scene)
        self._view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        layout.addWidget(self._view, stretch=1)

        if img_bgr is not None:
            pixmap = numpy_to_qpixmap(img_bgr)
            self._scene.addPixmap(pixmap)
            self._scene.setSceneRect(pixmap.rect().toRectF())
            self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        if info_text:
            info_label = QLabel(info_text)
            info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(info_label)

        self.btn_save = QPushButton(save_text)
        self.btn_save.clicked.connect(self._on_save)
        layout.addWidget(self.btn_save)

    def set_save_callback(self, callback) -> None:
        self._save_callback = callback

    def _on_save(self) -> None:
        if self._save_callback:
            self._save_callback(self)
