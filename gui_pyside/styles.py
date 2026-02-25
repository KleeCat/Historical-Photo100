"""颜色常量和 QSS 样式表生成。Apple 风格。"""

# --- 颜色常量 ---
UI_COLOR_PRIMARY = "#10B981"
UI_COLOR_PRIMARY_HOVER = "#059669"
UI_COLOR_PRIMARY_PRESSED = "#047857"
UI_COLOR_DANGER = "#EF4444"
UI_COLOR_DANGER_HOVER = "#DC2626"
UI_COLOR_DANGER_MUTED = "#6B4C4A"
UI_COLOR_DANGER_MUTED_HOVER = "#7D5553"
UI_COLOR_SECTION_TEXT = ("#737373", "#A1A1AA")
UI_COLOR_CARD_BG = ("#FFFFFF", "#171717")
UI_COLOR_CARD_BORDER = ("#E8E8ED", "#262626")
UI_COLOR_CARD_SHADOW = ("#D1D1D6", "#0A0A0A")
UI_COLOR_BG = ("#F2F2F7", "#0A0A0A")
UI_COLOR_SIDEBAR_BG = ("#EBEBF0", "#171717")
UI_COLOR_SECONDARY_BG = ("#E5E5EA", "#1E1E1E")
UI_COLOR_SECONDARY_HOVER = ("#D1D1D6", "#2A2A2A")
UI_COLOR_SECONDARY_PRESSED = ("#C7C7CC", "#333333")
UI_COLOR_SECONDARY_TEXT = ("#3A3A3C", "#EDEDED")
UI_COLOR_TEXT_PRIMARY = ("#1C1C1E", "#EDEDED")
UI_COLOR_TEXT_MUTED = ("#8E8E93", "#A1A1AA")
UI_COLOR_IMAGE_BG = ("#E5E5EA", "#0A0A0A")
UI_COLOR_SWITCH_OFF = ("#D1D1D6", "#333333")
UI_COLOR_SWITCH_ON = "#10B981"
UI_COLOR_SEPARATOR = ("#C6C6C8", "#262626")
UI_COLOR_INPUT_FOCUS = "#10B981"
UI_COLOR_BTN_BORDER = ("#E5E5EA", "#262626")
UI_COLOR_SAVE_BG = ("#1C1C1E", "#FFFFFF")
UI_COLOR_SAVE_TEXT = ("#FFFFFF", "#171717")

# --- 尺寸常量 ---
UI_SIDEBAR_WIDTH = 300
UI_WINDOW_WIDTH = 1400
UI_WINDOW_HEIGHT = 920

# --- 全局暗色模式状态 ---
_dark_mode = False


def set_dark_mode(dark: bool) -> None:
    """设置全局暗色模式状态。"""
    global _dark_mode
    _dark_mode = dark


def is_dark_mode() -> bool:
    """获取当前暗色模式状态。"""
    return _dark_mode


def c(color_tuple, dark=None):
    """从 (light, dark) 元组中选择颜色。"""
    if dark is None:
        dark = _dark_mode
    if isinstance(color_tuple, tuple):
        return color_tuple[1] if dark else color_tuple[0]
    return color_tuple


def generate_stylesheet(dark: bool = False) -> str:
    """生成完整 QSS 样式表 — Apple 风格。"""
    bg = c(UI_COLOR_BG, dark)
    sidebar_bg = c(UI_COLOR_SIDEBAR_BG, dark)
    card_bg = c(UI_COLOR_CARD_BG, dark)
    card_border = c(UI_COLOR_CARD_BORDER, dark)
    card_shadow = c(UI_COLOR_CARD_SHADOW, dark)
    text_primary = c(UI_COLOR_TEXT_PRIMARY, dark)
    text_muted = c(UI_COLOR_TEXT_MUTED, dark)
    secondary_bg = c(UI_COLOR_SECONDARY_BG, dark)
    secondary_hover = c(UI_COLOR_SECONDARY_HOVER, dark)
    secondary_pressed = c(UI_COLOR_SECONDARY_PRESSED, dark)
    secondary_text = c(UI_COLOR_SECONDARY_TEXT, dark)
    image_bg = c(UI_COLOR_IMAGE_BG, dark)
    switch_off = c(UI_COLOR_SWITCH_OFF, dark)
    separator = c(UI_COLOR_SEPARATOR, dark)
    btn_border = c(UI_COLOR_BTN_BORDER, dark)
    save_bg = c(UI_COLOR_SAVE_BG, dark)
    save_text = c(UI_COLOR_SAVE_TEXT, dark)

    return f"""
    /* === 全局 === */
    QMainWindow {{
        background-color: {bg};
    }}
    * {{
        color: {text_primary};
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        font-size: 13px;
    }}
    QWidget#centralWidget {{
        background-color: {bg};
    }}

    /* === 侧栏 === */
    QScrollArea#sidebar {{
        background-color: {sidebar_bg};
        border: none;
        border-right: 1px solid {separator};
    }}
    QScrollArea#sidebar > QWidget {{
        background-color: {sidebar_bg};
    }}
    QWidget#sidebarContainer {{
        background-color: {sidebar_bg};
    }}

    /* === 卡片 === */
    QFrame#card {{
        background-color: {card_bg};
        border: 1px solid {card_border};
        border-bottom: 2px solid {card_shadow};
        border-radius: 12px;
        padding: 10px;
    }}

    /* === 标签 === */
    QLabel {{
        color: {text_primary};
        background: transparent;
    }}
    QLabel#muted {{
        color: {text_muted};
        font-size: 12px;
    }}
    QLabel#section {{
        color: {text_muted};
        font-weight: 600;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    QLabel#title {{
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.5px;
    }}
    QLabel#panelPlaceholder {{
        color: {text_muted};
        font-size: 14px;
        background: transparent;
    }}

    /* === 按钮 === */
    QPushButton {{
        background-color: {secondary_bg};
        color: {secondary_text};
        border: 1px solid {btn_border};
        border-radius: 8px;
        padding: 8px 14px;
        font-weight: 600;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {secondary_hover};
    }}
    QPushButton:pressed {{
        background-color: {secondary_pressed};
    }}
    QPushButton#primary {{
        background-color: {UI_COLOR_PRIMARY};
        color: white;
        border: none;
        font-size: 14px;
        font-weight: 700;
        border-radius: 10px;
    }}
    QPushButton#primary:hover {{
        background-color: {UI_COLOR_PRIMARY_HOVER};
    }}
    QPushButton#primary:pressed {{
        background-color: {UI_COLOR_PRIMARY_PRESSED};
    }}
    QPushButton#danger {{
        background-color: {UI_COLOR_DANGER};
        color: white;
        border: none;
    }}
    QPushButton#danger:hover {{
        background-color: {UI_COLOR_DANGER_HOVER};
    }}
    QPushButton#toolbarBtn {{
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#saveBtn {{
        background-color: {save_bg};
        color: {save_text};
        border: none;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        font-weight: 600;
    }}
    QPushButton#saveBtn:hover {{
        background-color: {secondary_text};
    }}

    /* === 滑块 === */
    QSlider::groove:horizontal {{
        height: 6px;
        background: {switch_off};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: white;
        border: 1px solid {card_border};
        width: 18px;
        height: 18px;
        margin: -7px 0;
        border-radius: 9px;
    }}
    QSlider::handle:horizontal:hover {{
        border: 1px solid {UI_COLOR_PRIMARY};
    }}
    QSlider::sub-page:horizontal {{
        background: {UI_COLOR_PRIMARY};
        border-radius: 3px;
    }}

    /* === 进度条 === */
    QProgressBar {{
        background: {switch_off};
        border: none;
        border-radius: 3px;
        max-height: 6px;
        min-height: 6px;
    }}
    QProgressBar::chunk {{
        background: {UI_COLOR_PRIMARY};
        border-radius: 3px;
    }}

    /* === 下拉框 === */
    QComboBox {{
        background-color: {secondary_bg};
        color: {secondary_text};
        border: 1px solid {card_border};
        border-radius: 8px;
        padding: 6px 10px;
        font-weight: 500;
    }}
    QComboBox:hover {{
        border: 1px solid {UI_COLOR_PRIMARY};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {card_bg};
        color: {text_primary};
        selection-background-color: {UI_COLOR_PRIMARY};
        selection-color: white;
        border: 1px solid {card_border};
        border-radius: 8px;
        padding: 4px;
    }}

    /* === 输入框 === */
    QLineEdit {{
        background-color: {secondary_bg};
        color: {secondary_text};
        border: 1px solid {card_border};
        border-radius: 8px;
        padding: 6px 10px;
    }}
    QLineEdit:focus {{
        border: 2px solid {UI_COLOR_INPUT_FOCUS};
    }}

    /* === 复选框 === */
    QCheckBox {{
        color: {text_primary};
        spacing: 8px;
    }}

    /* === 提示框 === */
    QToolTip {{
        background-color: {card_bg};
        color: {text_primary};
        border: 1px solid {card_border};
        padding: 6px 8px;
        border-radius: 6px;
        font-size: 11px;
    }}

    /* === 滚动区域 === */
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
        margin: 4px 0;
    }}
    QScrollBar::handle:vertical {{
        background: {switch_off};
        border-radius: 3px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {text_muted};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* === 图像面板 === */
    QGraphicsView {{
        background-color: {image_bg};
        border: 1px solid {card_border};
        border-radius: 12px;
    }}

    /* === SpinBox === */
    QSpinBox {{
        background-color: {secondary_bg};
        color: {secondary_text};
        border: 1px solid {card_border};
        border-radius: 8px;
        padding: 4px 8px;
        min-height: 28px;
    }}
    QSpinBox:focus {{
        border: 2px solid {UI_COLOR_INPUT_FOCUS};
    }}
    QSpinBox::up-button, QSpinBox::down-button {{
        width: 20px;
        border: none;
    }}
    QSpinBox::up-button {{
        subcontrol-position: top right;
        border-top-right-radius: 8px;
    }}
    QSpinBox::down-button {{
        subcontrol-position: bottom right;
        border-bottom-right-radius: 8px;
    }}

    /* === 分割线 === */
    QFrame#toolbarSeparator {{
        background-color: {btn_border};
        max-height: 1px;
        min-height: 1px;
        border: none;
    }}
    """


# --- ToggleSwitch 自定义控件 ---

from PySide6.QtCore import Qt, Signal, QPropertyAnimation, Property, QEasingCurve, QRectF
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QSizePolicy


class ToggleSwitch(QWidget):
    """iOS 风格的 toggle 开关按钮。"""

    toggled = Signal(bool)

    def __init__(self, text: str = "", checked: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._checked = checked
        self._thumb_pos = 1.0 if checked else 0.0
        self._track_w = 44
        self._track_h = 24

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Track area (painted manually)
        self._track = _ToggleTrack(self)
        self._track.setFixedSize(self._track_w, self._track_h)
        self._track.clicked.connect(self.toggle)
        layout.addWidget(self._track)

        if text:
            self._label = QLabel(text)
            self._label.setCursor(Qt.CursorShape.PointingHandCursor)
            self._label.mousePressEvent = lambda e: self.toggle()
            layout.addWidget(self._label)

        layout.addStretch()
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        # Animation
        self._anim = QPropertyAnimation(self, b"thumb_pos")
        self._anim.setDuration(150)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, val: bool) -> None:
        if val == self._checked:
            return
        self._checked = val
        self._animate(val)
        self.toggled.emit(val)

    def toggle(self) -> None:
        self.setChecked(not self._checked)

    def _get_thumb_pos(self) -> float:
        return self._thumb_pos

    def _set_thumb_pos(self, val: float) -> None:
        self._thumb_pos = val
        self._track.update()

    thumb_pos = Property(float, _get_thumb_pos, _set_thumb_pos)

    def _animate(self, on: bool) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._thumb_pos)
        self._anim.setEndValue(1.0 if on else 0.0)
        self._anim.start()


class _ToggleTrack(QWidget):
    """ToggleSwitch 的轨道绘制区域。"""

    clicked = Signal()

    def __init__(self, switch: ToggleSwitch) -> None:
        super().__init__(switch)
        self._switch = switch
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setStyleSheet("background: transparent;")

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        r = h / 2.0
        thumb_r = h / 2.0 - 3  # thumb slightly smaller than track
        margin = 3
        pos = self._switch._thumb_pos  # 0.0 ~ 1.0

        # Track
        if self._switch._checked or pos > 0.5:
            track_color = QColor(UI_COLOR_PRIMARY)
        else:
            track_color = QColor(c(UI_COLOR_SWITCH_OFF, _dark_mode))

        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(track_color)
        p.drawRoundedRect(QRectF(0, 0, w, h), r, r)

        # Thumb shadow
        thumb_x = margin + pos * (w - 2 * margin - 2 * thumb_r) + thumb_r
        thumb_y = h / 2.0
        shadow_color = QColor(0, 0, 0, 40)
        p.setBrush(shadow_color)
        p.drawEllipse(QRectF(thumb_x - thumb_r, thumb_y - thumb_r + 1, thumb_r * 2, thumb_r * 2))

        # Thumb
        p.setBrush(QColor("white"))
        p.drawEllipse(QRectF(thumb_x - thumb_r, thumb_y - thumb_r, thumb_r * 2, thumb_r * 2))
        p.end()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
