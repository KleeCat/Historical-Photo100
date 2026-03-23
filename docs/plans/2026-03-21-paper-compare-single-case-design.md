# 单图论文对比图设计文档

日期：2026-03-21

## 背景

老师要求在“不同方法对比实验”部分，除了表格指标外，再补一张对应样例的处理前后对比图。当前仓库已有 Real-ESRGAN 相关运行链路和若干 ESRGAN 参考代码，但没有一条适用于论文单图补图场景的稳定最短链路，也没有现成的 Bicubic、SRCNN、ESRGAN、Real-ESRGAN 五列拼图输出脚本。

本次工作只服务论文出图，不改现有 GUI 主流程，不做多图批处理平台化，不把 Bicubic / SRCNN / ESRGAN 正式接入 `gui_pyside`。

## 已确认输入

- 低分辨率输入图：`D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\LR\0844x4.png`
- 高分辨率真值图：`D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\HR\0844.png`
- 统一输出倍率：4×

## 目标

生成一套论文可直接使用的单图对比素材，包括：

1. Input
2. Bicubic ×4
3. SRCNN ×4
4. ESRGAN ×4
5. Real-ESRGAN ×4

并生成一张横向 5 列的拼图，在图中直接展示各方法对应的 `PSNR (dB)` 与 `SSIM`。

## 非目标

- 不修改 `gui_pyside` 的界面与业务流
- 不实现多图批量论文出图工具
- 不新增完整的多方法实验平台
- 不将失败方法静默回退为 Bicubic 冒充成功

## 现有代码可复用点

### 1. Real-ESRGAN 推理链

- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\super-resolution processing.py`
- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\gui_pyside\models.py`

这两处已经体现了项目内可用的 `RealESRGANer` 调用方式，可作为 Real-ESRGAN 单图推理参考。

### 2. ESRGAN 权重来源与 RRDBNet 结构参考

- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(model) super-resolution processing.py`
- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\RRDBNet_arch.py`

其中旧脚本里已经给出 ESRGAN 官方权重下载地址，`RRDBNet_arch.py` 提供了标准 RRDBNet 参考结构，适合做最小推理实现。

### 3. 指标计算逻辑

- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\gui_pyside\metrics.py`

其中的 PSNR / SSIM 计算逻辑可复用思路，但单图论文脚本会优先保证输出与 HR 对齐后再计算，以避免不透明的自动 resize 影响可解释性。

## 总体方案

新增独立脚本：

- `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\paper_compare_single_case.py`

脚本职责：

1. 检查输入图与 HR 图存在性
2. 确保论文补图所需模型位于 D 盘项目目录
3. 分别运行 Bicubic / SRCNN / ESRGAN / Real-ESRGAN
4. 对各方法结果与 HR 计算 PSNR / SSIM
5. 输出单独结果图、指标文件、最终拼图

## 模型与输出目录

### 模型目录（统一放 D 盘）

`D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\models\paper_compare\`

建议结构：

```text
models/paper_compare/
  srcnn/
    srcnn_x4.pth
  esrgan/
    RRDB_ESRGAN_x4.pth
  realesrgan/
    RealESRGAN_x4plus.pth
```

### 输出目录

`D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\outputs\paper_compare_0844\`

产物：

- `input.png`
- `bicubic_x4.png`
- `srcnn_x4.png`
- `esrgan_x4.png`
- `realesrgan_x4.png`
- `metrics.json`
- `metrics.txt`
- `comparison_with_metrics.png`

## 方法实现策略

### Bicubic

直接使用 OpenCV 或 Pillow 做 4× Bicubic 插值。该方法必须稳定可用，作为最低基线。

### SRCNN

实现最小可运行推理链：

1. 先对 LR 图做 Bicubic ×4
2. 将图像转到亮度通道或 RGB 推理路径
3. 使用轻量 SRCNN 结构做前向推理
4. 恢复与 HR 尺寸一致的结果图

该实现只服务论文补图，不接入 GUI。

### ESRGAN

使用本仓库中的 `RRDBNet_arch.py` 参考结构与 ESRGAN 官方权重 `RRDB_ESRGAN_x4.pth` 实现单图推理。旧文件 `(model) super-resolution processing.py` 里的“失败后回退 bicubic”逻辑不复用；新脚本必须显式区分“真正推理成功”和“缺模型/加载失败”两类状态。

### Real-ESRGAN

复用当前项目环境中已经打通的 `RealESRGANer` 依赖链，但权重路径统一改为 D 盘模型目录，不依赖 C 盘缓存。

## 尺寸对齐与指标策略

PSNR / SSIM 必须以 HR 图为参考。为保证论文中的指标解释明确：

1. 所有方法的最终输出尺寸都应与 `HR\0844.png` 一致
2. 若模型输出存在 1~2 像素边界差异，优先采用显式裁剪对齐
3. 不允许在指标计算时静默把 HR 或结果图随意 resize 成任意尺寸

最终在图中展示：

- `PSNR: xx.xx dB`
- `SSIM: x.xxxx`

Input 列只显示 `Input`，不显示指标。

## 最终拼图样式

### 排版

横向 5 列：

1. Input
2. Bicubic
3. SRCNN
4. ESRGAN
5. Real-ESRGAN

### 每列内容

- 顶部：方法名
- 中部：结果图
- 底部：
  - Input：`Input`
  - 其余方法：
    - `PSNR: xx.xx dB`
    - `SSIM: x.xxxx`

### 视觉要求

- 白底或浅色背景
- 列宽一致
- 标签字号统一
- 输出分辨率适合直接插入论文

## 失败处理

- 模型缺失：自动下载到 `models\paper_compare\...`
- 下载失败：明确报错并终止，不输出伪结果
- 模型加载失败：明确标识具体方法失败，不回退冒充成功
- 指标计算失败：只在确有原因时标记为 `N/A`，同时保留结果图

## 验证标准

脚本完成后至少要验证：

1. 四种方法结果图都能生成
2. `comparison_with_metrics.png` 成功生成
3. `metrics.json` / `metrics.txt` 成功落盘
4. 四种方法的 PSNR / SSIM 均能针对 `HR\0844.png` 正常计算
5. 结果图尺寸与 HR 对齐

## 结论

本方案采用“单独论文出图脚本 + 最小可运行方法实现”的最短路径，既满足老师新增的论文展示要求，也避免在现有 GUI 结构中引入不必要的架构变更。对于当前目标，这是风险最低、产出最快且最容易复核的一条实现路径。
