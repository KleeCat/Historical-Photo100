# 论文局部放大对比图 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将论文单图对比结果改成“上排整图 + 红框标注，下排 3 组局部放大”的更直观可视化形式，并保留现有指标文件输出。

**Architecture:** 在 `paper_compare_single_case.py` 中扩展论文专用渲染层，不改 Bicubic/SRCNN/ESRGAN/Real-ESRGAN 推理与指标计算流程。渲染层新增统一红框配置、局部裁剪与两层排版逻辑，图内移除 PSNR/SSIM，指标继续写入 `metrics.txt/json` 供论文表格使用。

**Tech Stack:** Python 3.13、NumPy、OpenCV、Pillow、unittest

---

## Chunk 1: 先补渲染辅助测试

### Task 1: 为红框与局部裁剪增加失败测试

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\.worktrees\paper-compare-single-case\tests\test_paper_compare_single_case.py`
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\.worktrees\paper-compare-single-case\paper_compare_single_case.py`

- [ ] **Step 1: 写失败测试，约束裁剪框正规化与局部裁剪尺寸**
- [ ] **Step 2: 运行 `unittest` 确认测试失败**
- [ ] **Step 3: 实现最小辅助函数**
- [ ] **Step 4: 重新运行 `unittest` 确认通过**

### Task 2: 为论文专用排版增加失败测试

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\.worktrees\paper-compare-single-case\tests\test_paper_compare_single_case.py`
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\.worktrees\paper-compare-single-case\paper_compare_single_case.py`

- [ ] **Step 1: 写失败测试，约束渲染图能输出、且画布高度明显大于单排旧版**
- [ ] **Step 2: 运行 `unittest` 确认测试失败**
- [ ] **Step 3: 实现新版论文图渲染**
- [ ] **Step 4: 重新运行 `unittest` 确认通过**

---

## Chunk 2: 接入当前林肯样例并验证输出

### Task 3: 为当前样例配置 3 个局部框并重跑

**Files:**
- Modify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\.worktrees\paper-compare-single-case\paper_compare_single_case.py`
- Verify: `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\.worktrees\paper-compare-single-case\tmp\run_compare_lincoln_hesler_case.ps1`

- [ ] **Step 1: 为林肯样例增加默认论文裁剪框配置**
- [ ] **Step 2: 运行对比脚本重新生成 `paper_compare_lincoln_hesler_1857`**
- [ ] **Step 3: 检查 `comparison_with_metrics.png` 是否变成“整图 + 3 组局部放大”**
- [ ] **Step 4: 检查 `metrics.txt/json` 仍然存在且数值未丢失**

