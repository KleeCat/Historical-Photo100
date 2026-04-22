# 老照片彩色化 GUI 集成 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 PySide GUI 中新增离线 CPU 老照片彩色化能力，并与超分辨率作为两个独立入口展示，支持单图处理、结果预览、保存、取消和清晰报错。

**Architecture:** 保持现有 `sidebar.py` / `main.py` / `workers.py` 分层，新增 `gui_pyside/colorization.py` 作为彩色化边界模块，负责模型路径检查、预处理、推理和结果保存；`ColorizeWorker` 在线程中调用该模块；GUI 层只处理事件、状态和结果展示。优先使用现有 OpenCV / NumPy / PySide6 依赖，先把本地离线 CPU 路线做稳，再保留后续更换更强彩色化后端的余地。

**Tech Stack:** Python、PySide6、OpenCV、NumPy、unittest、unittest.mock、现有 `gui_pyside` 工具函数、离线本地彩色化模型文件。

---

## Planned File Map

- Create: `gui_pyside/colorization.py`
  - 彩色化模型配置、路径检查、输入预处理、CPU 推理、结果保存。
- Modify: `gui_pyside/workers.py`
  - 新增 `ColorizeWorker`，复用现有 QThread 信号模式。
- Modify: `gui_pyside/sidebar.py`
  - 新增彩色化按钮、信号和处理中状态切换。
- Modify: `gui_pyside/main.py`
  - 新增彩色化入口、worker 生命周期管理、结果展示和错误处理。
- Create: `tests/test_gui_pyside_colorization.py`
  - 纯逻辑测试：模型路径、输入预处理、保存与元数据。
- Create: `tests/test_gui_pyside_colorize_worker.py`
  - 线程测试：成功、失败、取消信号。
- Create: `tests/test_gui_pyside_main_colorization.py`
  - GUI 控制层测试：按钮接线、无图保护、任务互斥、完成后恢复状态。
- Create: `models/colorization/README.md`
  - 说明离线模型文件位置、文件名约定和环境变量覆盖方式。

## Chunk 1: 彩色化核心模块

### Task 1: 定义 `colorization.py` 合同并用测试锁定接口

**Files:**
- Create: `tests/test_gui_pyside_colorization.py`
- Create: `gui_pyside/colorization.py`
- Reference: `gui_pyside/utils.py`
- Reference: `docs/superpowers/specs/2026-04-22-old-photo-colorization-design.md`

- [ ] **Step 1: 写失败测试，先锁定模块接口**

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import numpy as np

from gui_pyside.colorization import (
    ColorizationModelNotFoundError,
    get_colorization_model_path,
    prepare_colorization_input,
    run_colorization_pipeline,
)


class TestColorizationPipeline(unittest.TestCase):
    def test_get_colorization_model_path_prefers_env_override(self):
        ...

    def test_prepare_colorization_input_converts_gray_to_bgr(self):
        gray = np.full((12, 10), 128, dtype=np.uint8)
        prepared, original_shape = prepare_colorization_input(gray, max_side=256)
        self.assertEqual(prepared.shape[2], 3)
        self.assertEqual(original_shape, (12, 10))

    def test_run_colorization_pipeline_raises_when_model_missing(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ColorizationModelNotFoundError):
                run_colorization_pipeline(
                    input_img=np.zeros((8, 8, 3), dtype=np.uint8),
                    input_path="demo.png",
                    output_base_dir=tmp_dir,
                    backend=Mock(),
                    model_path=Path(tmp_dir) / "missing.caffemodel",
                )
```

- [ ] **Step 2: 运行测试，确认它先失败**

Run: `rtk err python -m unittest tests.test_gui_pyside_colorization -v`

Expected: FAIL，错误应指向 `gui_pyside.colorization` 不存在或接口缺失。

- [ ] **Step 3: 先补最小模块骨架，只定义接口和异常**

```python
class ColorizationModelNotFoundError(FileNotFoundError):
    pass


def get_colorization_model_path(...):
    raise NotImplementedError


def prepare_colorization_input(...):
    raise NotImplementedError


def run_colorization_pipeline(...):
    raise NotImplementedError
```

- [ ] **Step 4: 再跑一次测试，确认失败点变成“未实现”而不是导入错误**

Run: `rtk err python -m unittest tests.test_gui_pyside_colorization -v`

Expected: FAIL，报错从 `ModuleNotFoundError` 变为 `NotImplementedError` 或断言失败。

- [ ] **Step 5: 提交当前红测骨架**

Run:

```bash
rtk git add tests/test_gui_pyside_colorization.py gui_pyside/colorization.py
rtk git commit -m "test(gui): 添加彩色化模块接口测试"
```

### Task 2: 实现彩色化管线并让核心测试转绿

**Files:**
- Modify: `gui_pyside/colorization.py`
- Modify: `tests/test_gui_pyside_colorization.py`
- Reference: `gui_pyside/utils.py`
- Create: `models/colorization/README.md`

- [ ] **Step 1: 扩展测试，锁定输出目录、文件命名和元数据保存**

```python
def test_run_colorization_pipeline_saves_image_and_metadata(self):
    fake_backend = Mock(return_value=np.full((16, 16, 3), 180, dtype=np.uint8))
    with TemporaryDirectory() as tmp_dir:
        fake_model = Path(tmp_dir) / "colorization_release_v2.caffemodel"
        fake_model.write_bytes(b"model")
        result = run_colorization_pipeline(
            input_img=np.zeros((16, 16, 3), dtype=np.uint8),
            input_path="portrait.png",
            output_base_dir=tmp_dir,
            backend=fake_backend,
            model_path=fake_model,
        )
        self.assertTrue(Path(result.output_path).is_file())
        self.assertEqual(result.output_image.shape, (16, 16, 3))
        self.assertIn("colorized", Path(result.output_path).name)
        self.assertIn("model_path", result.run_meta)
```

- [ ] **Step 2: 跑测试，确认新断言失败**

Run: `rtk err python -m unittest tests.test_gui_pyside_colorization -v`

Expected: FAIL，说明管线还没有真正保存结果或返回结构化数据。

- [ ] **Step 3: 实现最小可用彩色化模块**

```python
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .utils import ensure_dir, save_image, safe_basename, timestamp_str, write_json_file


@dataclass
class ColorizationResult:
    output_image: np.ndarray
    output_path: str
    run_dir: str
    run_meta: dict
    elapsed: float


def get_colorization_model_path(explicit_path=None):
    # 优先 COLORIZATION_MODEL_PATH，再退回 models/colorization 下的固定文件名
    ...


def prepare_colorization_input(input_img, max_side=1024):
    # 灰度转 BGR；必要时按最长边缩放；返回处理后图像和原始尺寸
    ...


def default_colorize_backend(prepared_img, model_path):
    # 先用 OpenCV DNN 本地模型作为稳定 CPU 基线
    ...


def run_colorization_pipeline(...):
    # 检查模型 -> 预处理 -> 调用 backend -> 恢复尺寸 -> 保存图像与 metadata
    ...
```

实现要求：

- 默认查找 `models/colorization/colorization_release_v2.caffemodel`；
- 同目录读取 `colorization_deploy_v2.prototxt` 和 `pts_in_hull.npy`；
- 允许 `COLORIZATION_MODEL_PATH` 环境变量覆盖主模型路径；
- 输出目录命名风格与现有超分流程一致：`outputs/<timestamp>_<basename>/`；
- 输出文件命名使用 `<basename>_colorized.png`；
- 元数据 JSON 写入同一运行目录。

- [ ] **Step 4: 跑核心测试，确认全部通过**

Run: `rtk summary python -m unittest tests.test_gui_pyside_colorization -v`

Expected: PASS，所有彩色化纯逻辑测试通过。

- [ ] **Step 5: 提交核心模块**

Run:

```bash
rtk git add gui_pyside/colorization.py tests/test_gui_pyside_colorization.py models/colorization/README.md
rtk git commit -m "feat(gui): 新增彩色化核心模块"
```

## Chunk 2: 后台线程与取消语义

### Task 3: 为 `ColorizeWorker` 先写线程测试，再实现线程类

**Files:**
- Create: `tests/test_gui_pyside_colorize_worker.py`
- Modify: `gui_pyside/workers.py`
- Reference: `gui_pyside/colorization.py`

- [ ] **Step 1: 写失败测试，锁定成功、失败、取消信号**

```python
import unittest
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication

from gui_pyside.workers import ColorizeWorker


class TestColorizeWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_worker_emits_success_and_image(self):
        ...

    def test_worker_emits_failure_message(self):
        ...

    def test_worker_honors_interruption(self):
        ...
```

- [ ] **Step 2: 跑测试，确认 `ColorizeWorker` 尚不存在**

Run: `rtk err python -m unittest tests.test_gui_pyside_colorize_worker -v`

Expected: FAIL，报 `ImportError` 或 `AttributeError: module 'gui_pyside.workers' has no attribute 'ColorizeWorker'`。

- [ ] **Step 3: 在 `workers.py` 中实现最小线程类**

```python
class ColorizeWorker(QThread):
    progress = Signal(float, str)
    image_ready = Signal(object)
    finished = Signal(bool, str)

    def __init__(self, img_input, input_path, output_base_dir=None, model_path=None):
        ...

    def run(self) -> None:
        try:
            self.progress.emit(0.1, "Checking colorization model...")
            result = run_colorization_pipeline(
                input_img=self.img_input,
                input_path=self.input_path,
                output_base_dir=self.output_base_dir,
                model_path=self.model_path,
                cancel_check=self.isInterruptionRequested,
            )
            self.run_dir = result.run_dir
            self.output_path = result.output_path
            self.image_ready.emit(result.output_image)
            self.finished.emit(True, f"Colorization complete: {os.path.basename(self.output_path)}")
        except UserCancelledError:
            self.finished.emit(False, "Colorization cancelled")
        except Exception as exc:
            self.finished.emit(False, str(exc))
```

实现要求：

- 复用现有 `UserCancelledError` 取消语义；
- 保存 `output_path` / `run_dir` / `elapsed` 供主窗口读取；
- 不引入 GUI 控件依赖，只发 Qt 信号。

- [ ] **Step 4: 跑线程测试，确认通过**

Run: `rtk summary python -m unittest tests.test_gui_pyside_colorize_worker -v`

Expected: PASS，成功、失败、取消三类行为都可覆盖。

- [ ] **Step 5: 提交线程改动**

Run:

```bash
rtk git add gui_pyside/workers.py tests/test_gui_pyside_colorize_worker.py
rtk git commit -m "feat(gui): 新增彩色化后台线程"
```

## Chunk 3: GUI 接线与交互状态

### Task 4: 先锁定 `SidebarWidget` 的按钮与状态行为

**Files:**
- Create: `tests/test_gui_pyside_main_colorization.py`
- Modify: `gui_pyside/sidebar.py`

- [ ] **Step 1: 在 GUI 测试里先写侧边栏断言**

```python
class TestSidebarColorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_sidebar_exposes_colorize_signal_and_button(self):
        sidebar = SidebarWidget()
        self.assertTrue(hasattr(sidebar, "colorize_clicked"))
        self.assertEqual(sidebar.btn_colorize.text(), "Start Colorization")

    def test_processing_state_disables_colorize_button(self):
        sidebar = SidebarWidget()
        sidebar.set_processing_state(True)
        self.assertFalse(sidebar.btn_colorize.isEnabled())
```

- [ ] **Step 2: 跑测试，确认侧边栏行为尚未实现**

Run: `rtk err python -m unittest tests.test_gui_pyside_main_colorization.TestSidebarColorization -v`

Expected: FAIL，提示 `btn_colorize` 或 `colorize_clicked` 不存在。

- [ ] **Step 3: 修改 `sidebar.py`，加按钮、信号和状态切换**

```python
class SidebarWidget(QScrollArea):
    colorize_clicked = Signal()

    def _build_actions_card(self) -> None:
        ...
        self.btn_colorize = QPushButton("Start Colorization")
        self.btn_colorize.setFixedHeight(32)
        self.btn_colorize.clicked.connect(self.colorize_clicked)
        lay.addWidget(self.btn_colorize)

    def set_processing_state(self, processing: bool) -> None:
        self.btn_start.setEnabled(not processing)
        self.btn_colorize.setEnabled(not processing)
        self.btn_batch.setEnabled(not processing)
        ...
```

实现要求：

- 彩色化按钮放在现有“Start Restoration”附近；
- `set_processing_state` / `set_batch_state` / `set_cancel_state` 都要考虑彩色化按钮；
- 先复用现有图标资源，不额外引入新图标文件。

- [ ] **Step 4: 重新运行侧边栏测试，确认通过**

Run: `rtk summary python -m unittest tests.test_gui_pyside_main_colorization.TestSidebarColorization -v`

Expected: PASS，按钮和状态切换行为正确。

- [ ] **Step 5: 提交侧边栏改动**

Run:

```bash
rtk git add gui_pyside/sidebar.py tests/test_gui_pyside_main_colorization.py
rtk git commit -m "feat(gui): 新增彩色化侧边栏入口"
```

### Task 5: 为 `MainWindow` 写控制层测试，再接入彩色化流程

**Files:**
- Modify: `tests/test_gui_pyside_main_colorization.py`
- Modify: `gui_pyside/main.py`
- Reference: `gui_pyside/workers.py`

- [ ] **Step 1: 先写主窗口控制层失败测试**

```python
class TestMainWindowColorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    @patch("gui_pyside.main.ColorizeWorker")
    def test_start_colorization_requires_input_image(self, worker_cls):
        window = MainWindow()
        window.img_input = None
        window.start_colorization()
        worker_cls.assert_not_called()

    @patch("gui_pyside.main.ColorizeWorker")
    def test_start_colorization_creates_worker_when_image_loaded(self, worker_cls):
        window = MainWindow()
        window.img_input = np.zeros((16, 16, 3), dtype=np.uint8)
        window.input_path = "demo.png"
        window.start_colorization()
        worker_cls.assert_called_once()
```

- [ ] **Step 2: 跑测试，确认 `MainWindow` 还没有彩色化入口**

Run: `rtk err python -m unittest tests.test_gui_pyside_main_colorization.TestMainWindowColorization -v`

Expected: FAIL，提示 `start_colorization`、`_colorize_worker` 或信号连接缺失。

- [ ] **Step 3: 修改 `main.py` 接入彩色化调度**

```python
from .workers import ModelLoadWorker, ProcessWorker, BatchWorker, ColorizeWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        ...
        self._colorize_worker: Optional[ColorizeWorker] = None

    def _connect_signals(self) -> None:
        ...
        self.sidebar.colorize_clicked.connect(self.start_colorization)

    def start_colorization(self) -> None:
        if self.img_input is None:
            QMessageBox.information(self, "Info", "Please open an image first.")
            return
        if self._process_worker and self._process_worker.isRunning():
            ...
        if self._batch_worker and self._batch_worker.isRunning():
            ...
        if self._colorize_worker and self._colorize_worker.isRunning():
            return

        self._colorize_worker = ColorizeWorker(
            img_input=self.img_input,
            input_path=self.input_path,
            output_base_dir=self.default_output_dir,
        )
        self._colorize_worker.progress.connect(self._on_process_progress)
        self._colorize_worker.image_ready.connect(self._on_process_image_ready)
        self._colorize_worker.finished.connect(self._on_colorize_finished)
        self.sidebar.set_processing_state(True)
        self.statusbar.start_timer()
        self._colorize_worker.start()
```

还要补齐：

- `_on_colorize_finished(...)`：恢复按钮状态、停表、更新状态栏、记录 `last_run_dir`；
- `_cancel_processing(...)`：同时处理中彩色化 worker；
- `closeEvent(...)`：退出时等待 `_colorize_worker`；
- 成功后复用已有图像展示逻辑，让右侧显示彩色图。

- [ ] **Step 4: 运行 GUI 测试并补齐必要断言**

Run: `rtk summary python -m unittest tests.test_gui_pyside_main_colorization -v`

Expected: PASS，侧边栏和主窗口的彩色化交互测试全部通过。

- [ ] **Step 5: 提交 GUI 接线**

Run:

```bash
rtk git add gui_pyside/main.py tests/test_gui_pyside_main_colorization.py
rtk git commit -m "feat(gui): 接入彩色化主窗口流程"
```

### Task 6: 做离线模型说明与最终验证

**Files:**
- Modify: `models/colorization/README.md`
- Modify: `docs/superpowers/specs/2026-04-22-old-photo-colorization-design.md`（仅当实现与规格有偏差时）

- [ ] **Step 1: 补充模型说明文档**

README 至少说明：

- 需要的模型文件名：
  - `colorization_release_v2.caffemodel`
  - `colorization_deploy_v2.prototxt`
  - `pts_in_hull.npy`
- 默认目录：`models/colorization/`
- 环境变量覆盖：`COLORIZATION_MODEL_PATH`
- 答辩前必须在离线机器上预放模型文件

- [ ] **Step 2: 跑全部新增自动化测试**

Run:

```bash
rtk summary python -m unittest `
  tests.test_gui_pyside_colorization `
  tests.test_gui_pyside_colorize_worker `
  tests.test_gui_pyside_main_colorization -v
```

Expected: PASS，新增测试全部通过。

- [ ] **Step 3: 做一次静态导入和最小运行检查**

Run:

```bash
rtk summary python -c "from gui_pyside.colorization import run_colorization_pipeline; from gui_pyside.workers import ColorizeWorker; from gui_pyside.main import MainWindow; print('imports ok')"
```

Expected: 输出 `imports ok`，且无导入报错。

- [ ] **Step 4: 做人工答辩彩排**

人工检查清单：

- 启动 GUI；
- 导入一张黑白照片；
- 点击“Start Colorization”；
- 观察界面不冻结；
- 结果显示在右侧；
- 输出目录生成 `*_colorized.png` 和 metadata；
- 再单独演示一次超分辨率按钮；
- 验证两者是两个独立入口。

- [ ] **Step 5: 提交收尾改动**

Run:

```bash
rtk git add models/colorization/README.md tests/test_gui_pyside_colorization.py tests/test_gui_pyside_colorize_worker.py tests/test_gui_pyside_main_colorization.py gui_pyside/colorization.py gui_pyside/workers.py gui_pyside/sidebar.py gui_pyside/main.py
rtk git commit -m "docs(gui): 补充彩色化离线使用说明"
```

## Execution Notes

- 优先遵循 `@test-driven-development`：每一块都先红后绿，再提交。
- 每个任务完成后都执行对应验证命令，不要跳过 `@verification-before-completion`。
- 若 PySide6 事件循环或线程测试不稳定，先在测试里复用单例 `QApplication` / `QCoreApplication`，必要时用 `unittest.mock` 隔离耗时推理。
- 若 OpenCV DNN 后端效果不理想，不要改 UI 边界；只在 `gui_pyside/colorization.py` 内替换默认 backend。
