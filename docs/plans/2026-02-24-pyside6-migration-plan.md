# PySide6 迁移实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 GUI 从 CustomTkinter 迁移到 PySide6，模块化拆分，1:1 功能等价。

**Architecture:** 单文件夹 `gui_pyside/` 包含 12 个模块。纯处理逻辑（processing/metrics/models）与 UI 分离。QThread + Signal/Slot 替代 threading + queue。QGraphicsView 替代手动缩放/平移。QSS 统一样式。

**Tech Stack:** Python, PySide6, OpenCV, NumPy, PyTorch, PIL, basicsr, realesrgan, gfpgan

---

### Task 1: 项目初始化

**Files:**
- Create: `gui_pyside/__init__.py`
- Create: `gui_pyside/styles.py`

**Step 1: 安装 PySide6**

Run: `pip install PySide6`

**Step 2: 创建 gui_pyside 目录和 __init__.py**

```python
# gui_pyside/__init__.py
"""PySide6-based GUI for Image Super-Resolution System."""
```

**Step 3: 创建 styles.py — 颜色常量 + QSS 生成**

从 `(gui)super-resolution processing.py:128-147` 迁移所有 `UI_COLOR_*` 常量，并添加 QSS 生成函数。

```python
# gui_pyside/styles.py
"""颜色常量和 QSS 样式表生成。"""

# --- 颜色常量 (light, dark) ---
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
    # 返回包含所有控件样式的 QSS 字符串
    ...
```

**Step 4: 验证语法**

Run: `python -c "from gui_pyside import styles; print('OK')"`

**Step 5: Commit**

```bash
git add gui_pyside/
git commit -m "feat: init gui_pyside with styles module"
```

---

### Task 2: 工具模块 (utils.py)

**Files:**
- Create: `gui_pyside/utils.py`

**Step 1: 创建 numpy↔QPixmap 转换 + 通用工具**

从 `(gui)super-resolution processing.py:187-210` 迁移工具函数，新增图像转换。

```python
# gui_pyside/utils.py
"""工具函数：图像转换、文件操作。"""
import os, json, uuid
from datetime import datetime
import numpy as np
import cv2
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt

def numpy_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    """BGR numpy array → QPixmap。"""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

def ensure_dir(path: str) -> str: ...
def write_json_file(path: str, data: dict) -> None: ...
def timestamp_str() -> str: ...
def safe_basename(path: str) -> str: ...
```

**Step 2: 验证**

Run: `python -c "from gui_pyside.utils import numpy_to_qpixmap; print('OK')"`

**Step 3: Commit**

```bash
git add gui_pyside/utils.py
git commit -m "feat: add utils module with image conversion"
```

---

### Task 3: 图像处理逻辑 (processing.py)

**Files:**
- Create: `gui_pyside/processing.py`
- Source: `(gui)super-resolution processing.py:237-688`

**Step 1: 迁移所有纯处理函数**

直接复制以下函数（无 UI 依赖，无需修改）：

- `ConvBlock`, `ScratchUNet` (PyTorch 模型类)
- `clean_state_dict`, `load_scratch_model`
- `predict_scratch_mask`, `apply_scratch_repair`
- `blend_images`, `apply_unsharp_mask`, `apply_film_grain`
- `blend_with_lr`, `suppress_edge_ringing`
- `clamp_value`, `auto_tile_size`
- `estimate_image_metrics`, `make_comparison_images`
- `tensor_to_grid_image`, `save_feature_grids`

以及相关常量：`SCRATCH_MODEL_PATH`, `SCRATCH_MASK_THRESHOLD`, `SCRATCH_INPAINT_RADIUS`, `IMAGE_EXTS`, `DEFAULT_BATCH_RETRIES`, `TEXTURE_ENABLED`, `TEXTURE_MODEL_ID` 等。

**Step 2: 验证**

Run: `python -c "from gui_pyside.processing import blend_images, apply_film_grain; print('OK')"`

**Step 3: Commit**

```bash
git add gui_pyside/processing.py
git commit -m "feat: add processing module (pure image logic)"
```

---

### Task 4: 模型管理 (models.py)

**Files:**
- Create: `gui_pyside/models.py`
- Source: `(gui)super-resolution processing.py:1497-1562, 2783-2840`

**Step 1: 提取模型加载和管理逻辑**

```python
# gui_pyside/models.py
"""模型加载和管理。"""
import os, logging, torch
from typing import Optional, Tuple
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logger = logging.getLogger(__name__)

class ModelManager:
    """管理所有 AI 模型的加载和缓存。"""
    def __init__(self, device: torch.device, model_folder: str):
        self.device = device
        self.model_folder = model_folder
        self.model: Optional[RRDBNet] = None
        self.upsampler: Optional[RealESRGANer] = None
        self.face_enhancer = None
        self.face_enhancer_scale = None
        self.scratch_model = None
        self.texture_pipe = None
        self.hook_handles = []
        self.feature_maps = []
        self.max_feature_maps = 6

    def load_esrgan(self, scale: int) -> None: ...
    def load_face_enhancer(self, scale: int) -> None: ...
    def load_scratch_model(self) -> None: ...
    def register_feature_hooks(self) -> None: ...
    def clear_feature_hooks(self) -> None: ...
    def enhance(self, img: np.ndarray, **kwargs) -> np.ndarray: ...
```

**Step 2: 验证**

Run: `python -c "from gui_pyside.models import ModelManager; print('OK')"`

**Step 3: Commit**

```bash
git add gui_pyside/models.py
git commit -m "feat: add models module (model loading/management)"
```

---

### Task 5: 指标计算 (metrics.py)

**Files:**
- Create: `gui_pyside/metrics.py`
- Source: `(gui)super-resolution processing.py:3223-3293`

**Step 1: 提取指标计算逻辑**

```python
# gui_pyside/metrics.py
"""图像质量指标计算。"""
import numpy as np, cv2, logging
from typing import Dict, Optional

def calculate_metrics(
    sr_img: np.ndarray, gt_img: np.ndarray
) -> Dict[str, Optional[float]]:
    """计算 PSNR、SSIM、可选 LPIPS。"""
    ...
```

**Step 2: 验证**

Run: `python -c "from gui_pyside.metrics import calculate_metrics; print('OK')"`

**Step 3: Commit**

```bash
git add gui_pyside/metrics.py
git commit -m "feat: add metrics module (PSNR/SSIM/LPIPS)"
```

---

### Task 6: 工作线程 (workers.py)

**Files:**
- Create: `gui_pyside/workers.py`

**Step 1: 创建 QThread 工作线程**

```python
# gui_pyside/workers.py
"""后台工作线程。"""
from PySide6.QtCore import QThread, Signal
import numpy as np

class ModelLoadWorker(QThread):
    """后台加载模型。"""
    progress = Signal(float, str)   # (进度, 状态文字)
    finished = Signal(bool, str)    # (成功?, 消息)

    def __init__(self, model_manager, scale: int):
        super().__init__()
        self.model_manager = model_manager
        self.scale = scale

    def run(self):
        try:
            self.progress.emit(0.3, "Loading model...")
            self.model_manager.load_esrgan(self.scale)
            self.progress.emit(1.0, f"Model x{self.scale} loaded")
            self.finished.emit(True, "OK")
        except Exception as e:
            self.finished.emit(False, str(e))

class ProcessWorker(QThread):
    """后台图像处理。"""
    progress = Signal(float, str)
    stage_changed = Signal(str)
    image_ready = Signal(object)      # numpy array
    metrics_ready = Signal(dict)
    finished = Signal(bool, str)

    def __init__(self, model_manager, img_input, settings: dict):
        super().__init__()
        self.model_manager = model_manager
        self.img_input = img_input
        self.settings = settings

    def run(self):
        # 完整处理流水线：scratch → upscale → face → blend → texture → grain
        ...

class BatchWorker(QThread):
    """后台批处理。"""
    item_started = Signal(int, int, str)  # (索引, 总数, 文件名)
    item_done = Signal(int, int)
    progress = Signal(float, str)
    finished = Signal(bool, str, list)    # (成功?, 消息, 错误列表)

    def run(self):
        ...
```

**Step 2: 验证**

Run: `python -c "from gui_pyside.workers import ModelLoadWorker, ProcessWorker, BatchWorker; print('OK')"`

**Step 3: Commit**

```bash
git add gui_pyside/workers.py
git commit -m "feat: add QThread workers for background tasks"
```

---

### Task 7: 状态栏 (statusbar.py)

**Files:**
- Create: `gui_pyside/statusbar.py`
- Source: `(gui)super-resolution processing.py:1339-1371`

**Step 1: 创建状态栏控件**

```python
# gui_pyside/statusbar.py
"""状态栏：状态文字 + 计时器 + 进度条。"""
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar
from PySide6.QtCore import QTimer
from .styles import *

class StatusBarWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)

        self.status_label = QLabel("Ready")
        self.elapsed_label = QLabel("Elapsed: --")
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setRange(0, 1000)

        layout.addWidget(self.status_label, stretch=1)
        layout.addWidget(self.progress_bar, stretch=0)
        layout.addWidget(self.elapsed_label, stretch=0)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._start_time = None

    def set_status(self, text: str): self.status_label.setText(text)
    def set_progress(self, value: float): self.progress_bar.setValue(int(value * 1000))
    def start_timer(self): ...
    def stop_timer(self): ...
    def _update_elapsed(self): ...
```

**Step 2: 验证**

Run: `python -c "from gui_pyside.statusbar import StatusBarWidget; print('OK')"`

**Step 3: Commit**

```bash
git add gui_pyside/statusbar.py
git commit -m "feat: add status bar widget"
```

---

### Task 8: 侧栏 (sidebar.py)

**Files:**
- Create: `gui_pyside/sidebar.py`
- Source: `(gui)super-resolution processing.py:874-1180`

**Step 1: 创建卡片工厂和侧栏框架**

SidebarWidget(QScrollArea) 包含 4 个 CardFrame：Input、Settings、Results、Actions。
通过 Signal 转发所有用户操作（按钮点击、滑块变化、开关切换）给 MainWindow。

控件映射：
- Open Image / Load GT → QPushButton
- Scale factor → QComboBox (x2/x4)
- Output dir → QLineEdit + QPushButton(folder icon)
- Face Enhancement / Scratch Repair → QCheckBox (QSS toggle 样式)
- Face Blend / Natural Blend / Texture Boost / Film Grain → QSlider(Qt.Horizontal)
- Compare Mode → QCheckBox, Compare Split → QSlider
- Start / Batch / Cancel → QPushButton
- Resolution / PSNR / SSIM 显示 → QLabel

**Step 2: 实现所有卡片内容**

对照原文件 lines 905-1180 逐个创建控件，保持相同的参数和默认值。

**Step 3: 验证**

Run: `python -c "from gui_pyside.sidebar import SidebarWidget; print('OK')"`

**Step 4: Commit**

```bash
git add gui_pyside/sidebar.py
git commit -m "feat: add sidebar widget with all cards"
```

---

### Task 9: 图像显示区 (display.py)

**Files:**
- Create: `gui_pyside/display.py`
- Source: `(gui)super-resolution processing.py:1185-1290, 1667-1798, 2033-2164, 2305-2331`

**Step 1: 创建 ImagePanel(QGraphicsView)**

支持缩放(wheelEvent)、平移(ScrollHandDrag)、硬件加速渲染(SmoothPixmapTransform)。
`set_image(img_bgr)` 接收 numpy 数组，转为 QPixmap 显示。
`zoom_changed` Signal 用于双面板联动。

**Step 2: 创建 ImageDisplayWidget(QWidget)**

双面板容器：标题行 + 左右 ImagePanel + 文件名标签 + 分辨率标签。
输出面板上叠加 overlay QLabel（"Waiting for processing..."）。
两个 panel 的 zoom_changed 信号互相连接实现联动。

**Step 3: 实现对比模式**

用 QGraphicsPixmapItem + setClipRect() 实现分割线。
split 值从 sidebar 的 compare_split_changed Signal 传入。

**Step 4: 实现结果工具栏**

在显示区底部添加 4 个按钮：Comparison、Features、Open Folder、Save Result。
通过 Signal 转发给 MainWindow。

**Step 5: 验证**

Run: `python -c "from gui_pyside.display import ImageDisplayWidget; print('OK')"`

**Step 6: Commit**

```bash
git add gui_pyside/display.py
git commit -m "feat: add image display with zoom/pan/compare"
```

---

### Task 10: 对话框 (dialogs.py)

**Files:**
- Create: `gui_pyside/dialogs.py`
- Source: `(gui)super-resolution processing.py:2539-2577`

**Step 1: 创建预览对话框**

PreviewDialog(QDialog)：显示图像 + Save 按钮。
接收 numpy BGR 数组，用 numpy_to_qpixmap 转换显示。

**Step 2: 验证**

Run: `python -c "from gui_pyside.dialogs import PreviewDialog; print('OK')"`

**Step 3: Commit**

```bash
git add gui_pyside/dialogs.py
git commit -m "feat: add preview dialog"
```

---

### Task 11: 主窗口组装 (main.py)

**Files:**
- Create: `gui_pyside/main.py`
- Source: `(gui)super-resolution processing.py` 全文件的胶水逻辑

**Step 1: 创建 MainWindow(QMainWindow)**

组装 sidebar + display + statusbar。初始化 ModelManager。
`_connect_signals()` 连接所有 sidebar/display Signal 到槽函数。

**Step 2: 实现文件操作槽**

- `open_image()` ← QFileDialog.getOpenFileName + read_image + display
- `load_gt()` ← QFileDialog.getOpenFileName + read_image
- `save_result()` ← QFileDialog.getSaveFileName + cv2.imwrite
- `save_comparison()` ← make_comparison_images + save
- `export_features()` ← save_feature_grids
- `open_output_folder()` ← subprocess / QDesktopServices.openUrl

**Step 3: 实现处理槽**

- `start_processing()` → 创建 ProcessWorker，连接 signals，start()
- `_on_process_progress(float, str)` → 更新 statusbar
- `_on_process_image_ready(ndarray)` → 显示结果
- `_on_process_finished(bool, str)` → 恢复 UI 状态
- `cancel_processing()` → worker.requestInterruption()

**Step 4: 实现批处理槽**

- `run_batch()` → 创建 BatchWorker，连接 signals，start()
- `_on_batch_item_done(int, int)` → 更新进度
- `_on_batch_finished(bool, str, list)` → 显示结果摘要

**Step 5: 实现模型加载**

- `_start_model_loading()` → 创建 ModelLoadWorker，start()
- `change_scale(int)` → 重新加载模型

**Step 6: 创建 main() 入口函数**

```python
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

**Step 7: 验证启动**

Run: `python -c "from gui_pyside.main import MainWindow; print('OK')"`

**Step 8: Commit**

```bash
git add gui_pyside/main.py
git commit -m "feat: add main window with full signal wiring"
```

---

### Task 12: 更新入口文件

**Files:**
- Modify: `(gui)super-resolution processing.py`

**Step 1: 替换入口文件为启动器**

```python
#!/usr/bin/env python3
"""Image Super-Resolution System — PySide6 GUI launcher."""
from gui_pyside.main import main

if __name__ == "__main__":
    main()
```

**Step 2: 验证启动**

Run: `python "(gui)super-resolution processing.py"`
Expected: PySide6 窗口正常打开

**Step 3: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "feat: update entry point to launch PySide6 GUI"
```

---

### Task 13: QSS 样式表完善

**Files:**
- Modify: `gui_pyside/styles.py`

**Step 1: 实现完整的 generate_stylesheet()**

为所有控件编写 QSS 规则，复刻当前极简科技风：
- QMainWindow / QWidget 背景色
- QFrame#card 圆角 8px + 边框 + 背景
- QPushButton 三级样式（primary 绿 / secondary 灰 / ghost 透明）
- QSlider groove + handle 样式
- QCheckBox toggle 开关样式
- QProgressBar 4px 高度 + 绿色填充
- QComboBox / QLineEdit 统一边框和圆角
- QToolTip 深色背景 + 浅色文字
- QScrollArea / QScrollBar 极简滚动条

**Step 2: 验证视觉效果**

Run: `python "(gui)super-resolution processing.py"`
Expected: 视觉效果与当前 CTk 版本一致

**Step 3: Commit**

```bash
git add gui_pyside/styles.py
git commit -m "feat: complete QSS stylesheet for minimal tech style"
```

---

### Task 14: 集成测试 + 最终验证

**Step 1: 功能清单验证**

- [ ] 窗口启动，布局正确
- [ ] 打开图片，显示在左面板
- [ ] 超分处理，结果显示在右面板
- [ ] 缩放/平移正常，两面板联动
- [ ] 对比模式分割线正常
- [ ] 批处理正常
- [ ] 保存结果/对比图/特征图
- [ ] 进度条和状态栏更新
- [ ] 最小化恢复无黑块
- [ ] Tooltip 正常显示

**Step 2: 搜索残留 CTk 引用**

Run: `grep -rn "customtkinter\|ctk\.\|CTk" gui_pyside/`
Expected: 无输出

**Step 3: 更新 .context 文件**

更新 `.context/CURRENT_TASK.md` 和 `.context/CHANGELOG.md`。

**Step 4: 最终提交**

```bash
git add -A
git commit -m "feat: PySide6 migration complete"
```
