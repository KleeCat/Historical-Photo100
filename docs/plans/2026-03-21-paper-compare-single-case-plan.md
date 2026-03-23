# 单图论文对比图 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `LR\0844x4.png` 与 `HR\0844.png` 生成包含 Input、Bicubic、SRCNN、ESRGAN、Real-ESRGAN 五列结果与 PSNR/SSIM 标注的论文对比图。

**Architecture:** 新增独立脚本 `paper_compare_single_case.py`，不改现有 GUI。脚本负责模型检查/下载、四种方法推理、结果尺寸对齐、PSNR/SSIM 计算，以及最终 5 列拼图输出。模型与输出均固定落到 D 盘项目目录，避免依赖 C 盘缓存。

**Tech Stack:** Python 3.13、PyTorch、OpenCV、Pillow、scikit-image、RealESRGAN、项目内 RRDBNet 参考实现

---

## 文件结构

### 新建文件

- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`
- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\tests\test_paper_compare_single_case.py`

### 只读参考文件

- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\RRDBNet_arch.py`
- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\super-resolution processing.py`
- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(model) super-resolution processing.py`
- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\gui_pyside\metrics.py`

### 运行期生成目录

- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\models\paper_compare\`
- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\outputs\paper_compare_0844\`

---

## Chunk 1: 纯函数骨架、尺寸对齐与指标格式化

### Task 1: 建立测试骨架与最小接口

**Files:**
- Create: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\tests\test_paper_compare_single_case.py`
- Create: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`

- [ ] **Step 1: 写第一个失败测试，约束基础辅助函数接口**

```python
from paper_compare_single_case import (
    bicubic_upscale_to_size,
    center_crop_to_match,
    format_metric_lines,
)


def test_bicubic_upscale_to_size_matches_target_shape():
    ...


def test_center_crop_to_match_returns_hr_shape():
    ...


def test_format_metric_lines_formats_psnr_and_ssim():
    assert format_metric_lines(28.95, 0.8923) == [
        "PSNR: 28.95 dB",
        "SSIM: 0.8923",
    ]
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: FAIL，报 `ImportError` 或缺少函数定义。

- [ ] **Step 3: 在脚本中实现最小辅助函数骨架**

```python
def bicubic_upscale_to_size(...): ...
def center_crop_to_match(...): ...
def format_metric_lines(...): ...
```

- [ ] **Step 4: 再跑测试确认通过**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_paper_compare_single_case.py paper_compare_single_case.py
git commit -m "test(paper): 添加论文单图脚本基础辅助函数测试"
```

### Task 2: 加入输入校验、指标计算与结果元数据结构

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\tests\test_paper_compare_single_case.py`

- [ ] **Step 1: 写失败测试，约束输入路径检查与 metrics 结果结构**

```python
def test_validate_inputs_accepts_existing_lr_hr_pair(tmp_path):
    ...


def test_compute_psnr_ssim_returns_two_numeric_fields():
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: FAIL，提示函数不存在或返回结构不匹配。

- [ ] **Step 3: 实现输入检查与 PSNR/SSIM 纯函数**

要求：

- 只计算 PSNR / SSIM
- 输入图先经过显式尺寸对齐
- 返回结构统一，例如：

```python
{
    "name": "Bicubic",
    "image": np.ndarray,
    "psnr": 24.32,
    "ssim": 0.7215,
}
```

- [ ] **Step 4: 重新运行测试确认通过**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_paper_compare_single_case.py paper_compare_single_case.py
git commit -m "feat(paper): 增加输入校验与指标计算辅助函数"
```

---

## Chunk 2: Bicubic / SRCNN / ESRGAN / Real-ESRGAN 四种方法推理链

### Task 3: 实现 Bicubic 与 SRCNN 最小可运行版本

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\tests\test_paper_compare_single_case.py`

- [ ] **Step 1: 为 Bicubic 与 SRCNN 推理接口写失败测试**

```python
def test_run_bicubic_returns_hr_sized_image(...):
    ...


def test_run_srcnn_returns_hr_sized_image(...):
    ...
```

- [ ] **Step 2: 运行测试确认失败**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现 Bicubic 和 SRCNN**

要求：

- Bicubic：直接 4× 插值到 HR 尺寸
- SRCNN：包含最小网络定义、权重路径、下载检查、推理逻辑
- 下载目录固定到 `models\paper_compare\srcnn\`
- 模型失败时抛出清晰异常，不回退到 Bicubic

- [ ] **Step 4: 重新运行测试确认通过**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_paper_compare_single_case.py paper_compare_single_case.py
git commit -m "feat(paper): 实现 Bicubic 与 SRCNN 单图推理链"
```

### Task 4: 实现 ESRGAN 单图推理链

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\tests\test_paper_compare_single_case.py`

- [ ] **Step 1: 写 ESRGAN 推理接口失败测试**

```python
def test_run_esrgan_returns_hr_sized_image(...):
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: FAIL。

- [ ] **Step 3: 基于 `RRDBNet_arch.py` 接入 ESRGAN 权重推理**

要求：

- 使用标准 RRDBNet
- 权重固定到 `models\paper_compare\esrgan\RRDB_ESRGAN_x4.pth`
- 下载地址复用旧脚本里记录的官方 release 链接
- 输出严格对齐到 HR 尺寸

- [ ] **Step 4: 重新跑测试确认通过**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_paper_compare_single_case.py paper_compare_single_case.py
git commit -m "feat(paper): 实现 ESRGAN 单图推理链"
```

### Task 5: 实现 Real-ESRGAN 单图推理链

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\tests\test_paper_compare_single_case.py`

- [ ] **Step 1: 写 Real-ESRGAN 接口失败测试**

```python
def test_run_realesrgan_returns_hr_sized_image(...):
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: FAIL。

- [ ] **Step 3: 复用项目现有依赖实现 Real-ESRGAN 推理**

要求：

- 权重固定到 `models\paper_compare\realesrgan\RealESRGAN_x4plus.pth`
- 不依赖 C 盘默认缓存
- 若权重不存在则自动下载
- 输出与 HR 尺寸一致

- [ ] **Step 4: 重新跑测试确认通过**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_paper_compare_single_case.py paper_compare_single_case.py
git commit -m "feat(paper): 实现 Real-ESRGAN 单图推理链"
```

---

## Chunk 3: 拼图、落盘与端到端验证

### Task 6: 实现五列拼图渲染与指标标注

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\tests\test_paper_compare_single_case.py`

- [ ] **Step 1: 写失败测试，约束最终画布尺寸与标签渲染接口**

```python
def test_build_comparison_figure_returns_nonempty_canvas(...):
    ...
```

- [ ] **Step 2: 跑测试确认失败**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: FAIL。

- [ ] **Step 3: 实现 5 列拼图构建**

要求：

- 列顺序固定：Input / Bicubic / SRCNN / ESRGAN / Real-ESRGAN
- 图上显示方法名
- 图上显示 `PSNR: xx.xx dB` 与 `SSIM: x.xxxx`
- Input 列只显示 `Input`
- 白底、统一列宽、可直接插论文

- [ ] **Step 4: 重新跑测试确认通过**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m pytest tests/test_paper_compare_single_case.py -q
```

Expected: PASS。

- [ ] **Step 5: Commit**

```bash
git add tests/test_paper_compare_single_case.py paper_compare_single_case.py
git commit -m "feat(paper): 实现论文五列拼图与指标标注"
```

### Task 7: 实现 CLI 主流程与输出落盘

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`

- [ ] **Step 1: 实现 `main()`，串联完整处理流程**

要求：

- 固定默认输入：
  - `LR\0844x4.png`
  - `HR\0844.png`
- 生成输出目录：
  - `outputs\paper_compare_0844\`
- 分别落盘四种方法结果图与最终拼图
- 额外输出 `metrics.json` 与 `metrics.txt`

- [ ] **Step 2: 先做语法验证**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' -m py_compile paper_compare_single_case.py
```

Expected: PASS。

- [ ] **Step 3: Commit**

```bash
git add paper_compare_single_case.py
git commit -m "feat(paper): 串联论文单图对比生成主流程"
```

### Task 8: 做端到端验证

**Files:**
- Verify only

- [ ] **Step 1: 运行完整脚本**

Run:

```powershell
$env:PYTHONPATH='D:\Tools\python-packages313'
& 'C:\Users\ihggk\AppData\Roaming\uv\python\cpython-3.13-windows-x86_64-none\python.exe' paper_compare_single_case.py
```

Expected:

- `outputs\paper_compare_0844\` 目录生成成功
- 单独结果图全部存在
- `comparison_with_metrics.png` 存在
- `metrics.json` / `metrics.txt` 存在

- [ ] **Step 2: 验证输出文件清单**

Run:

```powershell
Get-ChildItem 'D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\outputs\paper_compare_0844' -Name
```

Expected:

- `input.png`
- `bicubic_x4.png`
- `srcnn_x4.png`
- `esrgan_x4.png`
- `realesrgan_x4.png`
- `comparison_with_metrics.png`
- `metrics.json`
- `metrics.txt`

- [ ] **Step 3: 验证指标文件中包含四种方法的 PSNR / SSIM**

Run:

```powershell
Get-Content 'D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\outputs\paper_compare_0844\metrics.txt'
```

Expected: 包含 Bicubic / SRCNN / ESRGAN / Real-ESRGAN 四行结果，且每行含 PSNR 与 SSIM。

- [ ] **Step 4: Commit**

```bash
git add paper_compare_single_case.py outputs/paper_compare_0844
git commit -m "feat(paper): 生成论文单图对比结果"
```
