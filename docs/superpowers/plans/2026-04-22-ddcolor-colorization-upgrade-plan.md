# DDColor 彩色化后端升级 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将当前 GUI 的默认老照片彩色化后端从 OpenCV 旧模型升级为 DDColor，并通过更清晰的模块拆分提升大多数老照片的自然度与答辩可讲解性。

**Architecture:** 保持现有 GUI 入口、线程和主窗口调度不变，把彩色化统一入口保留在 `gui_pyside/colorization.py`，新增 `gui_pyside/ddcolor_backend.py` 作为 DDColor 专用推理层，并把正式运行所需的最小 DDColor 推理代码 vendor 到 `gui_pyside/ddcolor_vendor/`。`colorization.py` 负责统一输入输出、后处理、结果保存与元数据；`ddcolor_backend.py` 只负责 DDColor 模型检查、加载和推理。

**Tech Stack:** Python、PySide6、OpenCV、NumPy、PyTorch（CPU）、DDColor、unittest、unittest.mock、本地离线模型目录 `models/colorization/ddcolor/`。

---

## Planned File Map

- Create: `gui_pyside/ddcolor_backend.py`
  - DDColor 模型路径检查、模型缓存、推理入口、CPU-only 加载。
- Modify: `gui_pyside/colorization.py`
  - 从单一 OpenCV 后端改为统一彩色化服务层，默认调用 DDColor。
- Create: `gui_pyside/ddcolor_vendor/__init__.py`
  - vendor 包入口。
- Create/Modify: `gui_pyside/ddcolor_vendor/model.py`
  - 来自已验证 DDColor 代码的最小模型定义。
- Create/Modify: `gui_pyside/ddcolor_vendor/pipeline.py`
  - 来自已验证 DDColor 代码的最小推理管线。
- Create/Modify: `gui_pyside/ddcolor_vendor/ddcolor_arch_utils/*.py`
  - 仅保留 DDColor 推理必须的架构辅助文件。
- Create: `tests/test_gui_pyside_ddcolor_backend.py`
  - DDColor 后端路径、模型缓存、推理合同测试。
- Modify: `tests/test_gui_pyside_colorization.py`
  - 更新为默认 DDColor 路径与统一后处理测试。
- Modify: `tests/test_gui_pyside_colorize_worker.py`
  - 确认 worker 继续依赖统一入口而非具体 OpenCV 后端。
- Create: `models/colorization/ddcolor/README.md`
  - DDColor 权重、配置、离线放置说明。
- Optional Modify: `gui_pyside/main.py`
  - 若输出标题或状态文本需要更明确区分彩色化结果，可做小幅文案修正。

## Chunk 1: DDColor 资源与后端边界

### Task 1: 先用测试锁定 DDColor 后端合同

**Files:**
- Create: `tests/test_gui_pyside_ddcolor_backend.py`
- Create: `gui_pyside/ddcolor_backend.py`
- Reference: `tmp/DDColor-official/ddcolor/pipeline.py`
- Reference: `tmp/vendor_ddcolor_files.ps1`

- [ ] **Step 1: 写失败测试，锁定模型路径和异常行为**

```python
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gui_pyside.ddcolor_backend import (
    DDColorModelNotFoundError,
    get_ddcolor_model_path,
    get_ddcolor_config_paths,
    load_ddcolor_backend,
)


class TestDDColorBackend(unittest.TestCase):
    def test_get_ddcolor_model_path_prefers_explicit_path(self):
        ...

    def test_get_ddcolor_model_path_defaults_to_models_directory(self):
        ...

    def test_load_ddcolor_backend_raises_when_weight_missing(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(DDColorModelNotFoundError):
                load_ddcolor_backend(model_path=Path(tmp_dir) / "missing.pth")
```

- [ ] **Step 2: 运行测试，确认先失败**

Run: `rtk err python -m unittest tests.test_gui_pyside_ddcolor_backend -v`

Expected: FAIL，提示 `gui_pyside.ddcolor_backend` 不存在或接口缺失。

- [ ] **Step 3: 创建最小骨架，仅定义异常和接口**

```python
class DDColorModelNotFoundError(FileNotFoundError):
    pass


def get_ddcolor_model_path(...):
    raise NotImplementedError


def get_ddcolor_config_paths(...):
    raise NotImplementedError


def load_ddcolor_backend(...):
    raise NotImplementedError
```

- [ ] **Step 4: 再次运行测试，确认失败点转为未实现**

Run: `rtk err python -m unittest tests.test_gui_pyside_ddcolor_backend -v`

Expected: FAIL，从导入错误转为 `NotImplementedError` 或断言失败。

- [ ] **Step 5: 提交红测骨架**

Run:

```bash
rtk git add tests/test_gui_pyside_ddcolor_backend.py gui_pyside/ddcolor_backend.py
rtk git commit -m "test(gui): 添加DDColor后端接口测试"
```

### Task 2: vendor 最小 DDColor 推理代码并让后端测试转绿

**Files:**
- Create/Modify: `gui_pyside/ddcolor_vendor/__init__.py`
- Create/Modify: `gui_pyside/ddcolor_vendor/model.py`
- Create/Modify: `gui_pyside/ddcolor_vendor/pipeline.py`
- Create/Modify: `gui_pyside/ddcolor_vendor/ddcolor_arch_utils/*.py`
- Modify: `gui_pyside/ddcolor_backend.py`
- Modify: `tests/test_gui_pyside_ddcolor_backend.py`
- Create: `models/colorization/ddcolor/README.md`

- [ ] **Step 1: 扩展测试，锁定 vendor 路径与模型缓存语义**

```python
def test_load_ddcolor_backend_caches_model_instances(self):
    fake_backend = object()
    with patch("gui_pyside.ddcolor_backend._build_backend", return_value=fake_backend) as builder:
        model_a = load_ddcolor_backend(model_path="demo.pth", force_reload=True)
        model_b = load_ddcolor_backend(model_path="demo.pth")
    self.assertIs(model_a, model_b)
    builder.assert_called_once()
```

- [ ] **Step 2: 跑测试，确认新断言失败**

Run: `rtk err python -m unittest tests.test_gui_pyside_ddcolor_backend -v`

Expected: FAIL，说明缓存与 vendor 依赖尚未实现。

- [ ] **Step 3: 从已验证资源整理最小 vendor 代码**

来源限定：

- `tmp/DDColor-official/ddcolor/model.py`
- `tmp/DDColor-official/ddcolor/pipeline.py`
- `tmp/DDColor-official/basicsr/archs/ddcolor_arch_utils/*.py`
- `tmp/vendor_ddcolor_files.ps1`

实现要求：

- 不依赖 `tmp/` 作为运行时路径；
- 只保留推理所需最小文件；
- 去掉训练、CLI、demo、Gradio 等无关内容；
- `gui_pyside/ddcolor_vendor/__init__.py` 提供明确导出。

- [ ] **Step 4: 实现 `ddcolor_backend.py` 最小可用版本**

```python
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DDColorBackend:
    model: object
    pipeline: object
    model_path: Path


def get_ddcolor_model_path(explicit_path=None):
    # 默认指向 models/colorization/ddcolor/pytorch_model.pt
    ...


def get_ddcolor_config_paths(model_path):
    # 如需要，返回同目录配置/辅助文件路径
    ...


def load_ddcolor_backend(...):
    # 使用 vendor model/pipeline 构建 CPU backend，并做缓存
    ...


def run_ddcolor_inference(img_bgr, ...):
    # 返回 BGR uint8
    ...
```

实现约束：

- 默认模型目录：`models/colorization/ddcolor/`
- 默认权重文件名：`pytorch_model.pt`
- 优先支持 CPU；
- 如权重缺失，报错信息必须明确指向 `models/colorization/ddcolor/`。

- [ ] **Step 5: 跑后端测试，确认通过**

Run: `rtk summary python -m unittest tests.test_gui_pyside_ddcolor_backend -v`

Expected: PASS，路径、异常、缓存语义均通过。

- [ ] **Step 6: 提交 DDColor 后端基础设施**

Run:

```bash
rtk git add gui_pyside/ddcolor_backend.py gui_pyside/ddcolor_vendor tests/test_gui_pyside_ddcolor_backend.py models/colorization/ddcolor/README.md
rtk git commit -m "feat(gui): 新增DDColor彩色化后端"
```

## Chunk 2: 统一彩色化服务层重构

### Task 3: 先改测试，再把 `colorization.py` 改成统一入口层

**Files:**
- Modify: `tests/test_gui_pyside_colorization.py`
- Modify: `gui_pyside/colorization.py`
- Reference: `gui_pyside/ddcolor_backend.py`

- [ ] **Step 1: 更新失败测试，锁定默认后端切换到 DDColor**

```python
from unittest.mock import Mock, patch


def test_run_colorization_pipeline_uses_ddcolor_backend_by_default(self):
    fake_backend = Mock()
    fake_backend.run.return_value = np.full((16, 16, 3), 150, dtype=np.uint8)
    with patch("gui_pyside.colorization.load_ddcolor_backend", return_value=fake_backend):
        result = run_colorization_pipeline(
            input_img=np.zeros((16, 16, 3), dtype=np.uint8),
            input_path="portrait.png",
            output_base_dir=tmp_dir,
        )
    self.assertTrue(Path(result.output_path).is_file())
```

- [ ] **Step 2: 跑测试，确认当前实现失败**

Run: `rtk err python -m unittest tests.test_gui_pyside_colorization -v`

Expected: FAIL，因为当前默认实现仍是 OpenCV `default_colorize_backend`。

- [ ] **Step 3: 重构 `colorization.py` 为统一服务层**

建议重构后职责：

- 输入校验与预处理；
- 调用 `load_ddcolor_backend` / `run_ddcolor_inference`；
- 统一后处理；
- 统一结果保存；
- 元数据记录当前后端为 `ddcolor`。

建议接口保留：

```python
def prepare_colorization_input(...)
def postprocess_colorized_output(...)
def run_colorization_pipeline(...)
```

实现要求：

- 默认后端直接走 DDColor；
- 删除或降级 `default_colorize_backend` 的主路径地位；
- 元数据中新增 `backend: "ddcolor"`；
- 保持 `ColorizeWorker` 调用接口不变。

- [ ] **Step 4: 增加统一后处理测试并实现**

测试方向：

- 轻度压制过黄；
- 轻度压制异常绿偏色；
- 保持输出为 `uint8 BGR`；
- 不破坏原始尺寸。

Run: `rtk summary python -m unittest tests.test_gui_pyside_colorization -v`

Expected: PASS，默认后端和后处理测试全部通过。

- [ ] **Step 5: 提交统一服务层改动**

Run:

```bash
rtk git add gui_pyside/colorization.py tests/test_gui_pyside_colorization.py
rtk git commit -m "refactor(gui): 重构彩色化统一服务层"
```

## Chunk 3: 与 GUI 集成保持稳定

### Task 4: 验证 worker 与主窗口仍只依赖统一入口

**Files:**
- Modify: `tests/test_gui_pyside_colorize_worker.py`
- Modify: `tests/test_gui_pyside_main_colorization.py`
- Optional Modify: `gui_pyside/workers.py`
- Optional Modify: `gui_pyside/main.py`

- [ ] **Step 1: 更新失败测试，锁定 worker 仍调用统一入口**

```python
with patch.object(workers_module, "run_colorization_pipeline", return_value=fake_result):
    worker = ColorizeWorker(...)
    worker.run()
```

增加断言：

- `run_meta["backend"] == "ddcolor"` 可被透传；
- worker 不直接 import `ddcolor_backend`；
- 主窗口成功消息在彩色化完成后仍可更新输出。

- [ ] **Step 2: 跑相关测试，确认无意中引入 GUI 层耦合**

Run:

```bash
rtk err python -m unittest tests.test_gui_pyside_colorize_worker tests.test_gui_pyside_main_colorization -v
```

Expected: 若失败，应暴露耦合点或状态文案不一致。

- [ ] **Step 3: 仅在必要时小幅修改 GUI 层**

允许的改动范围：

- `workers.py`：若需要补充 `run_meta` 保存；
- `main.py`：若需要根据后端更新状态栏或更正输出标题；
- `sidebar.py`：仅在需要更清晰文案时微调。

禁止：

- 将 DDColor 模型加载逻辑塞入 `worker` 或 `main.py`；
- 改变现有按钮触发关系。

- [ ] **Step 4: 重新运行 GUI 相关测试**

Run:

```bash
rtk summary python -m unittest tests.test_gui_pyside_colorize_worker tests.test_gui_pyside_main_colorization -v
```

Expected: PASS，线程和 GUI 调度层保持稳定。

- [ ] **Step 5: 提交 GUI 兼容性修正**

Run:

```bash
rtk git add gui_pyside/workers.py gui_pyside/main.py gui_pyside/sidebar.py tests/test_gui_pyside_colorize_worker.py tests/test_gui_pyside_main_colorization.py
rtk git commit -m "fix(gui): 兼容DDColor彩色化后端"
```

## Chunk 4: 最终验证与答辩导向优化

### Task 5: 完成离线部署说明与实图验证

**Files:**
- Modify: `models/colorization/ddcolor/README.md`
- Optional Modify: `docs/superpowers/specs/2026-04-22-old-photo-colorization-design.md`

- [ ] **Step 1: 补充 DDColor 离线说明**

README 至少包含：

- 默认权重文件名：`pytorch_model.pt`
- 默认目录：`models/colorization/ddcolor/`
- 是否需要额外配置文件；
- CPU-only 运行说明；
- 不再将 OpenCV `.caffemodel` 视为默认主后端。

- [ ] **Step 2: 跑完整自动化测试集**

Run:

```bash
rtk summary python -m unittest `
  tests.test_gui_pyside_ddcolor_backend `
  tests.test_gui_pyside_colorization `
  tests.test_gui_pyside_colorize_worker `
  tests.test_gui_pyside_main_colorization -v
```

Expected: PASS，新增与既有彩色化测试全部通过。

- [ ] **Step 3: 做一次导入级烟雾验证**

Run:

```bash
rtk summary python -c "from gui_pyside.ddcolor_backend import load_ddcolor_backend; from gui_pyside.colorization import run_colorization_pipeline; print('ddcolor-imports-ok')"
```

Expected: 输出 `ddcolor-imports-ok`。

- [ ] **Step 4: 做一次真实样例图人工彩排**

人工检查：

- 导入黑白老照片；
- 点击 `Start Colorization`；
- 输出明显比当前 OpenCV 方案更自然；
- 人脸、头发、背景颜色分布更合理；
- 不出现严重泛黄、发绿；
- 处理时间仍在可接受范围；
- 结果成功保存。

- [ ] **Step 5: 记录可答辩表述并提交收尾**

建议答辩表述写入备注或文档：

- “彩色化模块默认采用 DDColor 深度学习后端”
- “通过统一服务层和模型推理层分离实现离线部署”
- “支持本地 CPU 运行”

Run:

```bash
rtk git add models/colorization/ddcolor/README.md gui_pyside/ddcolor_backend.py gui_pyside/ddcolor_vendor gui_pyside/colorization.py tests/test_gui_pyside_ddcolor_backend.py tests/test_gui_pyside_colorization.py tests/test_gui_pyside_colorize_worker.py tests/test_gui_pyside_main_colorization.py
rtk git commit -m "feat(gui): 升级DDColor彩色化后端"
```

## Execution Notes

- 优先遵循 `@test-driven-development`：每一块都先红后绿。
- 若 DDColor 权重或依赖加载慢，先把模型缓存设计好，避免每次彩色化重复冷启动。
- 不要把 `tmp/` 里的代码当作运行时依赖；`tmp/` 只能作为 vendor 参考来源。
- 若 CPU-only 下 `large` 模型过慢，优先在 `ddcolor_backend.py` 里明确支持 `tiny` / `modelscope` 权重切换，但不要把复杂选项暴露到 GUI。
- 若真实效果仍有轻微偏色，优先在 `colorization.py` 的统一后处理层做小幅修正，而不是让 GUI 加一堆参数。
