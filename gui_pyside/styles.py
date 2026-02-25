"""颜色常量和 QSS 样式表生成。"""

# --- 颜色常量 ---
UI_COLOR_PRIMARY = "#10B981"
UI_COLOR_PRIMARY_HOVER = "#059669"
UI_COLOR_DANGER = "#EF4444"
UI_COLOR_DANGER_HOVER = "#DC2626"
UI_COLOR_DANGER_MUTED = "#6B4C4A"
UI_COLOR_DANGER_MUTED_HOVER = "#7D5553"
UI_COLOR_SECTION_TEXT = ("#737373", "#737373")
UI_COLOR_CARD_BG = ("#FFFFFF", "#141414")
UI_COLOR_CARD_BORDER = ("#E5E5E5", "#262626")
UI_COLOR_BG = ("#FAFAFA", "#0A0A0A")
UI_COLOR_SECONDARY_BG = ("#F5F5F5", "#1A1A1A")
UI_COLOR_SECONDARY_HOVER = ("#EBEBEB", "#262626")
UI_COLOR_SECONDARY_TEXT = ("#404040", "#D4D4D4")
UI_COLOR_TEXT_PRIMARY = ("#171717", "#EDEDED")
UI_COLOR_TEXT_MUTED = ("#737373", "#6B6B6B")
UI_COLOR_IMAGE_BG = ("#F0F0F0", "#0F0F0F")
UI_COLOR_SWITCH_OFF = ("#D4D4D4", "#404040")
UI_COLOR_SWITCH_ON = "#10B981"

# --- 尺寸常量 ---
UI_SIDEBAR_WIDTH = 240
UI_WINDOW_WIDTH = 1300
UI_WINDOW_HEIGHT = 900


def c(color_tuple, dark=False):
    """从 (light, dark) 元组中选择颜色。"""
    if isinstance(color_tuple, tuple):
        return color_tuple[1] if dark else color_tuple[0]
    return color_tuple


def generate_stylesheet(dark: bool = False) -> str:
    """生成完整 QSS 样式表。"""
    # TODO: Task 13 完善
    bg = c(UI_COLOR_BG, dark)
    card_bg = c(UI_COLOR_CARD_BG, dark)
    card_border = c(UI_COLOR_CARD_BORDER, dark)
    text_primary = c(UI_COLOR_TEXT_PRIMARY, dark)
    text_muted = c(UI_COLOR_TEXT_MUTED, dark)
    secondary_bg = c(UI_COLOR_SECONDARY_BG, dark)
    secondary_hover = c(UI_COLOR_SECONDARY_HOVER, dark)
    secondary_text = c(UI_COLOR_SECONDARY_TEXT, dark)
    image_bg = c(UI_COLOR_IMAGE_BG, dark)
    switch_off = c(UI_COLOR_SWITCH_OFF, dark)

    return f"""
    QMainWindow, QWidget {{
        background-color: {bg};
        color: {text_primary};
        font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
        font-size: 12px;
    }}
    QFrame#card {{
        background-color: {card_bg};
        border: 1px solid {card_border};
        border-radius: 8px;
        padding: 8px;
    }}
    QLabel {{
        color: {text_primary};
        background: transparent;
    }}
    QLabel#muted {{
        color: {text_muted};
    }}
    QLabel#section {{
        color: {text_muted};
        font-weight: bold;
        font-size: 12px;
    }}
    QLabel#title {{
        font-size: 24px;
        font-weight: bold;
    }}
    QPushButton {{
        background-color: {secondary_bg};
        color: {secondary_text};
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: bold;
    }}
    QPushButton:hover {{
        background-color: {secondary_hover};
    }}
    QPushButton#primary {{
        background-color: {UI_COLOR_PRIMARY};
        color: white;
    }}
    QPushButton#primary:hover {{
        background-color: {UI_COLOR_PRIMARY_HOVER};
    }}
    QPushButton#danger {{
        background-color: {UI_COLOR_DANGER};
        color: white;
    }}
    QPushButton#danger:hover {{
        background-color: {UI_COLOR_DANGER_HOVER};
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {switch_off};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: {UI_COLOR_PRIMARY};
        width: 14px;
        height: 14px;
        margin: -5px 0;
        border-radius: 7px;
    }}
    QSlider::sub-page:horizontal {{
        background: {UI_COLOR_PRIMARY};
        border-radius: 2px;
    }}
    QProgressBar {{
        background: {switch_off};
        border: none;
        border-radius: 2px;
        max-height: 4px;
        min-height: 4px;
    }}
    QProgressBar::chunk {{
        background: {UI_COLOR_PRIMARY};
        border-radius: 2px;
    }}
    QComboBox {{
        background-color: {secondary_bg};
        color: {secondary_text};
        border: 1px solid {card_border};
        border-radius: 6px;
        padding: 4px 8px;
    }}
    QComboBox::drop-down {{
        border: none;
        width: 20px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {card_bg};
        color: {text_primary};
        selection-background-color: {UI_COLOR_PRIMARY};
        selection-color: white;
        border: 1px solid {card_border};
    }}
    QLineEdit {{
        background-color: {secondary_bg};
        color: {secondary_text};
        border: 1px solid {card_border};
        border-radius: 6px;
        padding: 4px 8px;
    }}
    QCheckBox {{
        color: {text_primary};
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 36px;
        height: 20px;
        border-radius: 10px;
        background-color: {switch_off};
    }}
    QCheckBox::indicator:checked {{
        background-color: {UI_COLOR_PRIMARY};
    }}
    QToolTip {{
        background-color: #1A1A1A;
        color: #EDEDED;
        border: 1px solid #333333;
        padding: 4px 6px;
        font-size: 10px;
    }}
    QScrollArea {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 6px;
    }}
    QScrollBar::handle:vertical {{
        background: {switch_off};
        border-radius: 3px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QGraphicsView {{
        background-color: {image_bg};
        border: 1px solid {card_border};
        border-radius: 8px;
    }}
    """
