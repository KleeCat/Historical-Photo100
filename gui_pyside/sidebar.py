"""侧栏：设置卡片和操作按钮。

包含 Input、Settings、Results、Actions 四个卡片区域。
所有用户操作通过 Signal 转发给 MainWindow。
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QCheckBox, QSlider, QFrame,
    QScrollArea, QSpinBox, QSizePolicy,
)

from .styles import (
    UI_SIDEBAR_WIDTH, UI_COLOR_PRIMARY, UI_COLOR_PRIMARY_HOVER,
    UI_COLOR_DANGER, UI_COLOR_SECONDARY_BG,
)


def _make_card(title: str, parent_layout: QVBoxLayout) -> QFrame:
    """创建一个带标题的卡片 QFrame。"""
    card = QFrame()
    card.setObjectName("card")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(0, 8, 0, 8)
    layout.setSpacing(4)
    if title:
        lbl = QLabel(title)
        lbl.setObjectName("section")
        layout.addWidget(lbl)
        layout.itemAt(0).widget().setContentsMargins(12, 0, 0, 0)
    parent_layout.addWidget(card)
    return card


class SidebarWidget(QScrollArea):
    """侧栏控件，包含所有设置和操作按钮。"""

    # --- Signals ---
    open_image_clicked = Signal()
    load_gt_clicked = Signal()
    scale_changed = Signal(int)          # 2 or 4
    output_dir_clicked = Signal()
    face_enhance_toggled = Signal(bool)
    scratch_repair_toggled = Signal(bool)
    face_blend_changed = Signal(float)
    natural_blend_changed = Signal(float)
    texture_boost_changed = Signal(float)
    film_grain_changed = Signal(float)
    compare_toggled = Signal(bool)
    compare_split_changed = Signal(float)
    start_clicked = Signal()
    batch_clicked = Signal()
    cancel_clicked = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(UI_SIDEBAR_WIDTH + 16)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(8)
        self.setWidget(container)

        self._build_title()
        self._build_input_card()
        self._build_settings_card()
        self._build_results_card()
        self._layout.addStretch(1)
        self._build_actions_card()

    # --- Title ---
    def _build_title(self) -> None:
        lbl = QLabel("Super Resolution")
        lbl.setObjectName("title")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(lbl)

    # --- Input Card ---
    def _build_input_card(self) -> None:
        card = _make_card("Input", self._layout)
        lay = card.layout()

        self.btn_open = QPushButton("Open Image")
        self.btn_open.setToolTip("Open an image file for super-resolution")
        self.btn_open.clicked.connect(self.open_image_clicked)
        lay.addWidget(self.btn_open)

        self.btn_gt = QPushButton("Load Ground Truth")
        self.btn_gt.setToolTip("Load ground truth image for quality metrics")
        self.btn_gt.clicked.connect(self.load_gt_clicked)
        lay.addWidget(self.btn_gt)

    # --- Settings Card ---
    def _build_settings_card(self) -> None:
        card = _make_card("Settings", self._layout)
        lay = card.layout()

        # Scale factor
        lbl_scale = QLabel("Upscale Factor:")
        lbl_scale.setStyleSheet("font-weight: bold; font-size: 12px;")
        lay.addWidget(lbl_scale)

        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["x2", "x4"])
        self.combo_scale.setCurrentText("x4")
        self.combo_scale.currentTextChanged.connect(self._on_scale_changed)
        lay.addWidget(self.combo_scale)

        # Output dir row
        dir_row = QHBoxLayout()
        self.entry_output_dir = QLineEdit()
        self.entry_output_dir.setPlaceholderText("Output directory...")
        self.entry_output_dir.setReadOnly(True)
        dir_row.addWidget(self.entry_output_dir, stretch=1)
        self.btn_output_dir = QPushButton("\U0001F4C1")
        self.btn_output_dir.setFixedSize(32, 30)
        self.btn_output_dir.setToolTip("Set output directory")
        self.btn_output_dir.clicked.connect(self.output_dir_clicked)
        dir_row.addWidget(self.btn_output_dir)
        lay.addLayout(dir_row)

        # Face Enhancement toggle
        self.chk_face = QCheckBox("Face Enhancement")
        self.chk_face.toggled.connect(self._on_face_toggled)
        lay.addWidget(self.chk_face)

        # Scratch Repair toggle
        self.chk_scratch = QCheckBox("Scratch Repair")
        self.chk_scratch.toggled.connect(self.scratch_repair_toggled)
        lay.addWidget(self.chk_scratch)

        # Sliders (initially hidden)
        self.lbl_face_blend = QLabel("Face Blend: 0.65")
        self.slider_face_blend = QSlider(Qt.Orientation.Horizontal)
        self.slider_face_blend.setRange(0, 20)
        self.slider_face_blend.setValue(13)  # 0.65 * 20
        self.slider_face_blend.valueChanged.connect(self._on_face_blend)
        lay.addWidget(self.lbl_face_blend)
        lay.addWidget(self.slider_face_blend)
        self.lbl_face_blend.hide()
        self.slider_face_blend.hide()

        self.lbl_natural_blend = QLabel("Natural Blend: 0.00")
        self.slider_natural_blend = QSlider(Qt.Orientation.Horizontal)
        self.slider_natural_blend.setRange(0, 10)
        self.slider_natural_blend.setValue(0)
        self.slider_natural_blend.valueChanged.connect(self._on_natural_blend)
        lay.addWidget(self.lbl_natural_blend)
        lay.addWidget(self.slider_natural_blend)
        self.lbl_natural_blend.hide()
        self.slider_natural_blend.hide()

        self.lbl_texture_boost = QLabel("Texture Boost: 0.08")
        self.slider_texture_boost = QSlider(Qt.Orientation.Horizontal)
        self.slider_texture_boost.setRange(0, 7)
        self.slider_texture_boost.setValue(2)  # ~0.08 / 0.05 per step
        self.slider_texture_boost.valueChanged.connect(self._on_texture_boost)
        lay.addWidget(self.lbl_texture_boost)
        lay.addWidget(self.slider_texture_boost)
        self.lbl_texture_boost.hide()
        self.slider_texture_boost.hide()

        self.lbl_film_grain = QLabel("Film Grain: 0.00")
        self.slider_film_grain = QSlider(Qt.Orientation.Horizontal)
        self.slider_film_grain.setRange(0, 10)
        self.slider_film_grain.setValue(0)
        self.slider_film_grain.valueChanged.connect(self._on_film_grain)
        lay.addWidget(self.lbl_film_grain)
        lay.addWidget(self.slider_film_grain)
        self.lbl_film_grain.hide()
        self.slider_film_grain.hide()

        # Batch retries
        retry_row = QHBoxLayout()
        retry_row.addWidget(QLabel("Batch Retries"))
        self.spin_batch_retry = QSpinBox()
        self.spin_batch_retry.setRange(0, 5)
        self.spin_batch_retry.setValue(1)
        retry_row.addWidget(self.spin_batch_retry)
        lay.addLayout(retry_row)

    # --- Results Card ---
    def _build_results_card(self) -> None:
        card = _make_card("Results", self._layout)
        lay = card.layout()

        # Resolution row
        res_row = QHBoxLayout()
        self.lbl_res_in = QLabel("In: -- x --")
        self.lbl_res_out = QLabel("Out: -- x --")
        res_row.addWidget(self.lbl_res_in)
        res_row.addStretch()
        res_row.addWidget(self.lbl_res_out)
        lay.addLayout(res_row)

        # Metrics header
        lbl_header = QLabel("Output vs GT")
        lbl_header.setStyleSheet("font-weight: bold; font-size: 11px;")
        lay.addWidget(lbl_header)

        # PSNR / SSIM row
        metrics_row = QHBoxLayout()
        self.lbl_psnr = QLabel("PSNR: --")
        self.lbl_psnr.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.lbl_ssim = QLabel("SSIM: --")
        self.lbl_ssim.setStyleSheet("font-weight: bold; font-size: 13px;")
        metrics_row.addWidget(self.lbl_psnr)
        metrics_row.addStretch()
        metrics_row.addWidget(self.lbl_ssim)
        lay.addLayout(metrics_row)

        self.lbl_gt_hint = QLabel("Load GT for metrics")
        self.lbl_gt_hint.setObjectName("muted")
        self.lbl_gt_hint.setStyleSheet("font-size: 10px;")
        lay.addWidget(self.lbl_gt_hint)

        # Compare controls
        compare_row = QHBoxLayout()
        self.chk_compare = QCheckBox("Compare")
        self.chk_compare.toggled.connect(self._on_compare_toggled)
        compare_row.addWidget(self.chk_compare)

        self.lbl_compare_split = QLabel("50%")
        self.lbl_compare_split.hide()
        compare_row.addWidget(self.lbl_compare_split)

        self.slider_compare = QSlider(Qt.Orientation.Horizontal)
        self.slider_compare.setRange(0, 20)
        self.slider_compare.setValue(10)
        self.slider_compare.setFixedWidth(80)
        self.slider_compare.setEnabled(False)
        self.slider_compare.hide()
        self.slider_compare.valueChanged.connect(self._on_compare_split)
        compare_row.addWidget(self.slider_compare)
        lay.addLayout(compare_row)

    # --- Actions Card ---
    def _build_actions_card(self) -> None:
        card = _make_card("Actions", self._layout)
        lay = card.layout()

        self.btn_start = QPushButton("Start Restoration")
        self.btn_start.setObjectName("primary")
        self.btn_start.setFixedHeight(46)
        self.btn_start.setStyleSheet(
            f"font-size: 16px; font-weight: bold; "
            f"background-color: {UI_COLOR_PRIMARY}; color: white; "
            f"border-radius: 6px;"
        )
        self.btn_start.setToolTip("Start the super-resolution process")
        self.btn_start.clicked.connect(self.start_clicked)
        lay.addWidget(self.btn_start)

        self.btn_batch = QPushButton("Run Folder (Batch)")
        self.btn_batch.setToolTip("Process all images in a folder")
        self.btn_batch.clicked.connect(self.batch_clicked)
        lay.addWidget(self.btn_batch)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setToolTip("Cancel the current operation")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_clicked)
        lay.addWidget(self.btn_cancel)

    # --- Slot helpers ---

    def _on_scale_changed(self, text: str) -> None:
        self.scale_changed.emit(2 if text == "x2" else 4)

    def _on_face_toggled(self, checked: bool) -> None:
        self.face_enhance_toggled.emit(checked)
        # Show/hide sliders
        visible = checked
        self.lbl_face_blend.setVisible(visible)
        self.slider_face_blend.setVisible(visible)
        self.lbl_natural_blend.setVisible(True)
        self.slider_natural_blend.setVisible(True)
        self.lbl_texture_boost.setVisible(True)
        self.slider_texture_boost.setVisible(True)
        self.lbl_film_grain.setVisible(True)
        self.slider_film_grain.setVisible(True)

    def _on_face_blend(self, val: int) -> None:
        v = val / 20.0
        self.lbl_face_blend.setText(f"Face Blend: {v:.2f}")
        self.face_blend_changed.emit(v)

    def _on_natural_blend(self, val: int) -> None:
        v = val * 0.02  # 0~10 → 0.0~0.20
        self.lbl_natural_blend.setText(f"Natural Blend: {v:.2f}")
        self.natural_blend_changed.emit(v)

    def _on_texture_boost(self, val: int) -> None:
        v = val * 0.05  # 0~7 → 0.0~0.35
        self.lbl_texture_boost.setText(f"Texture Boost: {v:.2f}")
        self.texture_boost_changed.emit(v)

    def _on_film_grain(self, val: int) -> None:
        v = val * 0.05  # 0~10 → 0.0~0.50
        self.lbl_film_grain.setText(f"Film Grain: {v:.2f}")
        self.film_grain_changed.emit(v)

    def _on_compare_toggled(self, checked: bool) -> None:
        self.compare_toggled.emit(checked)
        self.lbl_compare_split.setVisible(checked)
        self.slider_compare.setVisible(checked)
        self.slider_compare.setEnabled(checked)

    def _on_compare_split(self, val: int) -> None:
        v = val / 20.0
        self.lbl_compare_split.setText(f"{int(v * 100)}%")
        self.compare_split_changed.emit(v)

    # --- Public API ---

    def set_processing_state(self, processing: bool) -> None:
        """切换处理中/空闲状态的按钮可用性。"""
        self.btn_start.setEnabled(not processing)
        self.btn_batch.setEnabled(not processing)
        self.btn_cancel.setEnabled(processing)
        self.combo_scale.setEnabled(not processing)
        if processing:
            self.btn_start.setText("Processing...")
        else:
            self.btn_start.setText("Start Restoration")

    def set_batch_state(self, running: bool) -> None:
        """切换批处理状态。"""
        self.btn_start.setEnabled(not running)
        self.btn_batch.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        if running:
            self.btn_start.setText("Batch Processing...")

    def set_cancel_state(self) -> None:
        """显示取消中状态。"""
        self.btn_cancel.setText("Cancelling...")
        self.btn_cancel.setEnabled(False)

    def update_resolution(self, in_size: tuple | None, out_size: tuple | None) -> None:
        if in_size:
            self.lbl_res_in.setText(f"In: {in_size[0]} x {in_size[1]}")
        else:
            self.lbl_res_in.setText("In: -- x --")
        if out_size:
            self.lbl_res_out.setText(f"Out: {out_size[0]} x {out_size[1]}")
        else:
            self.lbl_res_out.setText("Out: -- x --")

    def update_metrics(self, psnr_val: str, ssim_val: str, gt_hint: str = "") -> None:
        self.lbl_psnr.setText(psnr_val)
        self.lbl_ssim.setText(ssim_val)
        if gt_hint:
            self.lbl_gt_hint.setText(gt_hint)

    def set_slider_values(
        self, face_blend: float, natural_blend: float,
        texture_boost: float, film_grain: float,
    ) -> None:
        """从外部设置滑块值（如 auto_tune）。"""
        self.slider_face_blend.blockSignals(True)
        self.slider_face_blend.setValue(int(face_blend * 20))
        self.lbl_face_blend.setText(f"Face Blend: {face_blend:.2f}")
        self.slider_face_blend.blockSignals(False)

        self.slider_natural_blend.blockSignals(True)
        self.slider_natural_blend.setValue(int(natural_blend / 0.02))
        self.lbl_natural_blend.setText(f"Natural Blend: {natural_blend:.2f}")
        self.slider_natural_blend.blockSignals(False)

        self.slider_texture_boost.blockSignals(True)
        self.slider_texture_boost.setValue(int(texture_boost / 0.05))
        self.lbl_texture_boost.setText(f"Texture Boost: {texture_boost:.2f}")
        self.slider_texture_boost.blockSignals(False)

        self.slider_film_grain.blockSignals(True)
        self.slider_film_grain.setValue(int(film_grain / 0.05))
        self.lbl_film_grain.setText(f"Film Grain: {film_grain:.2f}")
        self.slider_film_grain.blockSignals(False)

    def set_output_dir_text(self, path: str) -> None:
        self.entry_output_dir.setReadOnly(False)
        self.entry_output_dir.setText(path)
        self.entry_output_dir.setReadOnly(True)
