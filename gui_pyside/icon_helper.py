"""SVG 图标加载与着色工具。

从 gui_pyside/icons/ 目录加载 Lucide SVG 文件，
支持自定义颜色和尺寸，适配亮/暗模式。
"""
import re
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer

from .styles import UI_COLOR_TEXT_PRIMARY, is_dark_mode

_ICONS_DIR = Path(__file__).parent / "icons"
_cache: dict[tuple[str, str, int], QIcon] = {}


def load_icon(name: str, color: str | None = None, size: int = 16) -> QIcon:
    """加载并着色一个 SVG 图标。

    Args:
        name: SVG 文件名（不含 .svg 后缀）
        color: 描边颜色（hex），None 则使用亮色模式文字色
        size: 输出像素尺寸
    """
    if color is None:
        color = UI_COLOR_TEXT_PRIMARY[1] if is_dark_mode() else UI_COLOR_TEXT_PRIMARY[0]

    key = (name, color, size)
    if key in _cache:
        return _cache[key]

    svg_path = _ICONS_DIR / f"{name}.svg"
    if not svg_path.exists():
        return QIcon()

    svg_data = svg_path.read_text(encoding="utf-8")
    svg_data = re.sub(r'stroke="[^"]*"', f'stroke="{color}"', svg_data)

    renderer = QSvgRenderer(QByteArray(svg_data.encode("utf-8")))
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    icon = QIcon(pixmap)
    _cache[key] = icon
    return icon


def clear_cache() -> None:
    """清空图标缓存。"""
    _cache.clear()
