# UI Refinement v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将彩色 Emoji 替换为 Lucide 单色 SVG 图标，底部工具栏去容器化，左侧控制栏微调间距和控件比例。

**Architecture:** 新建 `icon_helper.py` 模块统一管理 SVG 图标加载与着色（支持亮/暗模式自动切换）。修改 `styles.py` 调整 Toggle 尺寸、按钮边框 QSS。修改 `sidebar.py` 和 `display.py` 将 Emoji 文本替换为 QIcon。

**Tech Stack:** PySide6 (QSvgRenderer, QIcon, QPixmap, QPainter), Lucide Icons SVG

---

### Task 1: 创建 Lucide SVG 图标文件

**Files:**
- Create: `gui_pyside/icons/` 目录
- Create: 12 个 SVG 文件

**Step 1: 创建 icons 目录**

Run: `mkdir -p gui_pyside/icons`

**Step 2: 下载/创建 Lucide SVG 文件**

从 Lucide Icons 官方获取以下 12 个 SVG（stroke="currentColor", stroke-width="2", fill="none", 24x24 viewBox）：

- `folder-open.svg` — Open Image / Open Folder 按钮
- `ruler.svg` — Load Ground Truth 按钮
- `play.svg` — Start Restoration 按钮
- `folders.svg` — Run Folder (Batch) 按钮
- `x.svg` — Cancel 按钮
- `folder.svg` — Output dir 小按钮
- `columns-2.svg` — Comparison 工具栏按钮
- `brain.svg` — Features 工具栏按钮
- `save.svg` — Save Result 工具栏按钮
- `image.svg` — Input panel placeholder
- `sparkles.svg` — Output panel placeholder

**Step 3: Commit**

```bash
git add gui_pyside/icons/
git commit -m "assets: add Lucide SVG icons for UI refinement"
```

---

### Task 2: 创建 icon_helper.py 图标加载模块

**Files:**
- Create: `gui_pyside/icon_helper.py`

**Step 1: 实现 icon_helper.py**

```python
"""SVG 图标加载与着色工具。

从 gui_pyside/icons/ 目录加载 Lucide SVG 文件，
支持自定义颜色和尺寸，适配亮/暗模式。
"""
import re
from pathlib import Path

from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QIcon, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer

from .styles import UI_COLOR_TEXT_PRIMARY

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
        color = UI_COLOR_TEXT_PRIMARY[0]  # 亮色模式默认

    key = (name, color, size)
    if key in _cache:
        return _cache[key]

    svg_path = _ICONS_DIR / f"{name}.svg"
    if not svg_path.exists():
        return QIcon()

    svg_data = svg_path.read_text(encoding="utf-8")
    # 替换 stroke 颜色
    svg_data = re.sub(
        r'stroke="[^"]*"', f'stroke="{color}"', svg_data
    )
    renderer = QSvgRenderer(QByteArray(svg_data.encode("utf-8")))

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter)
    painter.end()

    icon = QIcon(pixmap)
    _cache[key] = icon
    return icon


def load_icon_dark(name: str, size: int = 16) -> QIcon:
    """加载暗色模式图标（使用暗色文字色）。"""
    return load_icon(name, UI_COLOR_TEXT_PRIMARY[1], size)


def clear_cache() -> None:
    """清空图标缓存。"""
    _cache.clear()
```

**Step 2: Commit**

```bash
git add gui_pyside/icon_helper.py
git commit -m "feat: add icon_helper module for SVG icon loading and coloring"
```

---

### Task 3: 修改 styles.py — Toggle 缩小 + 按钮边框 + 工具栏样式

**Files:**
- Modify: `gui_pyside/styles.py:119-160` (QSS 按钮部分)
- Modify: `gui_pyside/styles.py:325-326` (Toggle 尺寸)

**Step 1: 添加新颜色常量**

在 `styles.py` 第 27 行 `UI_COLOR_INPUT_FOCUS` 之后添加：

```python
UI_COLOR_BTN_BORDER = ("#E5E5EA", "#38383A")
UI_COLOR_SAVE_BG = ("#1C1C1E", "#F2F2F7")
UI_COLOR_SAVE_TEXT = ("#FFFFFF", "#1C1C1E")
```

**Step 2: 修改 QSS — 按钮加边框**

将 `QPushButton` 的 `border: none;` 改为 `border: 1px solid {btn_border};`（新增 `btn_border = c(UI_COLOR_BTN_BORDER, dark)`）。

**Step 3: 添加 QSS — Save 按钮深灰样式**

在 `QPushButton#toolbarBtn` 之后添加：

```css
QPushButton#saveBtn {
    background-color: {save_bg};
    color: {save_text};
    border: none;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
}
QPushButton#saveBtn:hover {
    background-color: {secondary_text};  /* 深灰悬停 */
}
```

**Step 4: 添加 QSS — 工具栏分割线**

```css
QFrame#toolbarSeparator {
    background-color: {btn_border};
    max-height: 1px;
    min-height: 1px;
    border: none;
}
```

**Step 5: 修改 Toggle 尺寸**

`styles.py:325-326`：
```python
self._track_w = 44  # was 52
self._track_h = 24  # was 28
```

**Step 6: 修改卡片内间距 QSS**

将 `QFrame#card` 的 `padding: 8px;` 改为 `padding: 10px;`。

**Step 7: Commit**

```bash
git add gui_pyside/styles.py
git commit -m "style: toggle shrink, button borders, save btn + toolbar separator QSS"
```

---

### Task 4: 修改 sidebar.py — Emoji 替换为 QIcon + 间距调整

**Files:**
- Modify: `gui_pyside/sidebar.py`

**Step 1: 添加 import**

在 `sidebar.py` 第 12 行之后添加：

```python
from .icon_helper import load_icon
```

**Step 2: 替换 Input 卡片按钮 Emoji**

`sidebar.py:88-98`：
```python
self.btn_open = QPushButton("Open Image")
self.btn_open.setIcon(load_icon("folder-open"))
self.btn_open.setFixedHeight(32)
...
self.btn_gt = QPushButton("Load Ground Truth")
self.btn_gt.setIcon(load_icon("ruler"))
self.btn_gt.setFixedHeight(32)
```

**Step 3: 替换 Output dir 按钮 Emoji**

`sidebar.py:122-124`：
```python
self.btn_output_dir = QPushButton()
self.btn_output_dir.setIcon(load_icon("folder", size=14))
self.btn_output_dir.setFixedSize(32, 32)
self.btn_output_dir.setStyleSheet("padding: 0px;")
```

**Step 4: 替换 Actions 卡片按钮 Emoji**

`sidebar.py:252-270`：
```python
self.btn_start = QPushButton("Start Restoration")
self.btn_start.setIcon(load_icon("play", "#FFFFFF"))
...
self.btn_batch = QPushButton("Run Folder (Batch)")
self.btn_batch.setIcon(load_icon("folders"))
...
self.btn_cancel = QPushButton("Cancel")
self.btn_cancel.setIcon(load_icon("x"))
```

**Step 5: 调整卡片内间距**

`sidebar.py:24-25`（`_make_card` 函数）：
```python
layout.setContentsMargins(10, 10, 10, 10)  # was (10, 8, 10, 8)
layout.setSpacing(8)  # was 6
```

**Step 6: 修复 set_processing_state 和 set_batch_state 中的文本**

`sidebar.py:330-332`：
```python
if processing:
    self.btn_start.setText("Processing...")
    self.btn_start.setIcon(QIcon())  # 清除图标
else:
    self.btn_start.setText("Start Restoration")
    self.btn_start.setIcon(load_icon("play", "#FFFFFF"))
```

同理修复 `set_batch_state`。

**Step 7: Commit**

```bash
git add gui_pyside/sidebar.py
git commit -m "ui: replace sidebar emoji with Lucide SVG icons, adjust spacing"
```

---

### Task 5: 修改 display.py — 工具栏去容器化 + Emoji 替换

**Files:**
- Modify: `gui_pyside/display.py`

**Step 1: 添加 import**

```python
from .icon_helper import load_icon
```

**Step 2: 替换 placeholder Emoji**

`display.py:155-156`：
```python
self.panel_input = ImagePanel("Open an image to begin")
self.panel_output = ImagePanel("Output will appear here")
```

placeholder 图标通过 QLabel 的 QPixmap 设置（可选，或保持纯文字）。

**Step 3: 工具栏去容器化**

`display.py:181-213`：

将 toolbar 从 `QFrame#card` 改为普通 `QWidget`（无背景），并在其上方加一条 `QFrame#toolbarSeparator` 分割线：

```python
# Toolbar separator
separator = QFrame()
separator.setObjectName("toolbarSeparator")
separator.setFixedHeight(1)
layout.addWidget(separator)

# Toolbar (no card wrapper)
toolbar = QWidget()
tb_layout = QHBoxLayout(toolbar)
tb_layout.setContentsMargins(0, 6, 0, 2)
```

**Step 4: 替换工具栏按钮 Emoji + Save 按钮样式**

```python
self.btn_compare = QPushButton("Comparison")
self.btn_compare.setIcon(load_icon("columns-2", size=14))
self.btn_compare.setObjectName("toolbarBtn")

self.btn_features = QPushButton("Features")
self.btn_features.setIcon(load_icon("brain", size=14))
self.btn_features.setObjectName("toolbarBtn")

self.btn_open_folder = QPushButton("Open Folder")
self.btn_open_folder.setIcon(load_icon("folder-open", size=14))
self.btn_open_folder.setObjectName("toolbarBtn")

self.btn_save = QPushButton("Save Result")
self.btn_save.setIcon(load_icon("save", "#FFFFFF", size=14))
self.btn_save.setObjectName("saveBtn")  # 深灰样式
```

**Step 5: Commit**

```bash
git add gui_pyside/display.py
git commit -m "ui: toolbar de-containerize, replace emoji with Lucide SVG icons"
```

---

### Task 6: 验证 + 更新 context 文件

**Files:**
- Modify: `.context/CURRENT_TASK.md`
- Modify: `.context/CHANGELOG.md`

**Step 1: 运行 GUI 验证**

Run: `python "gui_pyside/../(gui)super-resolution processing.py"`

验证：
- 所有按钮显示单色线性图标（非彩色 Emoji）
- 底部工具栏无白色容器包裹
- Save Result 按钮为深灰底+白字
- Toggle 开关尺寸缩小
- 侧栏按钮有浅灰边框
- 卡片内间距更宽松

**Step 2: 更新 context 文件**

**Step 3: Final commit**

```bash
git add .context/
git commit -m "docs: update context files for UI refinement v2"
```
