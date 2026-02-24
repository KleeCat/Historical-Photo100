# UI 美化第五轮实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 GUI 从当前配色方案重构为极简科技风（Linear/Vercel 风格），绿色唯一强调色 + 黑白灰。

**Architecture:** 修改单文件 `(gui)super-resolution processing.py` 中的颜色常量、组件样式和布局参数。无新文件创建，无架构变更。

**Tech Stack:** Python, CustomTkinter (ctk), tkinter

---

### Task 1: 更新颜色常量定义

**Files:**
- Modify: `(gui)super-resolution processing.py:128-147`

**Step 1: 替换所有 UI_COLOR_* 常量**

将 lines 128-147 替换为：

```python
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
UI_COLOR_TEXT_MUTED = ("#A3A3A3", "#525252")
UI_COLOR_IMAGE_BG = ("#F0F0F0", "#0F0F0F")
UI_COLOR_SWITCH_OFF = ("#D4D4D4", "#404040")
UI_COLOR_SWITCH_ON = "#10B981"
```

注意：删除 `UI_COLOR_BLUE` 和 `UI_COLOR_BLUE_HOVER`。

**Step 2: 验证语法**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 编译失败（因为还有引用 UI_COLOR_BLUE 的地方）

**Step 3: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: update color constants to minimal tech style"
```

---

### Task 2: 消除所有 UI_COLOR_BLUE 引用

**Files:**
- Modify: `(gui)super-resolution processing.py:909, 1331`

**Step 1: Open Image 按钮（line 909）从蓝色改为次级灰**

将:
```python
fg_color=UI_COLOR_BLUE, hover_color=UI_COLOR_BLUE_HOVER,
```
改为:
```python
fg_color=UI_COLOR_SECONDARY_BG, hover_color=UI_COLOR_SECONDARY_HOVER,
text_color=UI_COLOR_SECONDARY_TEXT,
```

**Step 2: Save Result 按钮（line 1331）从蓝色改为次级灰**

将:
```python
fg_color=UI_COLOR_BLUE, hover_color=UI_COLOR_BLUE_HOVER,
```
改为:
```python
fg_color=UI_COLOR_SECONDARY_BG, hover_color=UI_COLOR_SECONDARY_HOVER,
text_color=UI_COLOR_SECONDARY_TEXT,
```

**Step 3: 搜索确认无残留 BLUE 引用**

Run: `grep -n "UI_COLOR_BLUE" "(gui)super-resolution processing.py"`
Expected: 无输出

**Step 4: 验证语法**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 通过

**Step 5: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: remove blue accent, unify secondary buttons to gray"
```

---

### Task 3: 卡片标题颜色 + 圆角统一

**Files:**
- Modify: `(gui)super-resolution processing.py:859-874` (_make_card 方法)

**Step 1: 确认 _make_card 中 corner_radius 已为 8**

当前 line 862 已是 `corner_radius=8`，无需修改。

**Step 2: 卡片标题颜色已使用 UI_COLOR_SECTION_TEXT**

line 871 已引用 `UI_COLOR_SECTION_TEXT`，Task 1 已将其更新为 `("#737373", "#737373")`，无需额外修改。

**Step 3: 确认 results_toolbar 圆角**

line 1290 `corner_radius=6` 改为 `corner_radius=8` 统一。

**Step 4: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: unify corner radius to 8px"
```

---

### Task 4: 进度条细线化 + 绿色填充

**Files:**
- Modify: `(gui)super-resolution processing.py:1347-1356`

**Step 1: 修改进度条样式**

将:
```python
self.progress_bar = ctk.CTkProgressBar(
    self.status_frame, width=300, mode="determinate"
)
```
改为:
```python
self.progress_bar = ctk.CTkProgressBar(
    self.status_frame, width=300, height=4, mode="determinate",
    progress_color=UI_COLOR_PRIMARY,
    fg_color=UI_COLOR_SECONDARY_BG,
    corner_radius=2,
)
```

**Step 2: 验证语法**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 通过

**Step 3: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: slim progress bar with green fill"
```

---

### Task 5: Switch 开关改为绿色

**Files:**
- Modify: `(gui)super-resolution processing.py:974-984`

**Step 1: 确认 Switch 颜色引用**

当前 `progress_color=UI_COLOR_SWITCH_ON`（Task 1 已改为 `#10B981`），
`button_color=UI_COLOR_SWITCH_OFF`（Task 1 已改为 `("#D4D4D4", "#404040")`）。

无需额外代码修改，Task 1 的常量更新已覆盖。验证即可。

**Step 2: Commit (如有变更)**

与 Task 4 合并提交或跳过。

---

### Task 6: 侧栏紧凑化 + 布局微调

**Files:**
- Modify: `(gui)super-resolution processing.py:866, 902, 1154, 1168, 1180`

**Step 1: 卡片外边距从 pady=(4,2) 改为 pady=(6,3)**

line 866: `pady=(4, 2)` → `pady=(6, 3)` — 保持紧凑但略增呼吸感。

**Step 2: 标题上边距微调**

line 902: `pady=(20, 6)` → `pady=(16, 4)` — 标题区更紧凑。

**Step 3: 按钮间距统一**

- line 1154: `padx=12, pady=4` 保持不变
- line 1168: `padx=12, pady=4` 保持不变
- line 1180: `pady=(4, 10)` → `pady=(4, 8)` — 底部略收紧

**Step 4: 验证语法**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 通过

**Step 5: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: compact sidebar spacing"
```

---

### Task 7: Tooltip 添加

**Files:**
- Modify: `(gui)super-resolution processing.py` (setup_ui 方法末尾，约 line 1357)

**Step 1: 确认 ToolTip 类已存在**

line 150 已有 `class ToolTip`，可直接使用。

**Step 2: 在 setup_ui 末尾添加 Tooltip 绑定**

在 `self.progress_bar.set(0)` (line 1356) 之后添加：

```python
# --- Tooltips ---
ToolTip(self.btn_load, "Open an image file for super-resolution")
ToolTip(self.btn_gt, "Load ground truth image for quality metrics")
ToolTip(self.btn_run, "Start the super-resolution process")
ToolTip(self.btn_batch, "Process all images in a folder")
ToolTip(self.btn_cancel, "Cancel the current operation")
ToolTip(self.btn_compare, "Save side-by-side comparison image")
ToolTip(self.btn_features, "Export feature map visualizations")
ToolTip(self.btn_open_run_dir, "Open the output folder in file explorer")
ToolTip(self.btn_save, "Save the processed result to disk")
```

**Step 3: 验证语法**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 通过

**Step 4: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: add tooltips to all action buttons"
```

---

### Task 8: 状态栏样式统一

**Files:**
- Modify: `(gui)super-resolution processing.py:1339-1355`

**Step 1: 状态栏背景与主背景统一**

将:
```python
self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0)
```
改为:
```python
self.status_frame = ctk.CTkFrame(self, height=30, corner_radius=0, fg_color=UI_COLOR_BG)
```

**Step 2: 状态文字和计时器用次级色**

status_label 和 elapsed_label 的 fg_color 改为引用 `UI_COLOR_BG`，text_color 设为 `UI_COLOR_TEXT_MUTED`：

```python
self.status_label = ctk.CTkLabel(
    self.status_frame, text="Ready", padx=10, width=400, anchor="w",
    fg_color=UI_COLOR_BG, text_color=UI_COLOR_TEXT_MUTED,
)
```

```python
self.elapsed_label = ctk.CTkLabel(
    self.status_frame, text="Elapsed: --", padx=10,
    fg_color=UI_COLOR_BG, text_color=UI_COLOR_TEXT_MUTED,
)
```

**Step 3: 验证语法**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 通过

**Step 4: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: unify status bar style with muted text"
```

---

### Task 9: 图像区域极简边框

**Files:**
- Modify: `(gui)super-resolution processing.py:1214, 1219`

**Step 1: 输入/输出图像 frame 添加 1px 边框**

将:
```python
self.display_frame, fg_color=UI_COLOR_IMAGE_BG
```
改为:
```python
self.display_frame, fg_color=UI_COLOR_IMAGE_BG,
corner_radius=8, border_width=1, border_color=UI_COLOR_CARD_BORDER,
```

对两处（input_frame 和 output_frame）都做同样修改。

**Step 2: 验证语法**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 通过

**Step 3: Commit**

```bash
git add "(gui)super-resolution processing.py"
git commit -m "ui: add subtle border to image panels"
```

---

### Task 10: 最终验证 + 合并提交

**Step 1: 完整语法检查**

Run: `python -m py_compile "(gui)super-resolution processing.py"`
Expected: 通过

**Step 2: 搜索残留的旧颜色引用**

Run: `grep -n "#1677FF\|#0958D9\|#F5F7FA\|#1F2937\|#4B5563\|#9CA3AF\|#6B7280" "(gui)super-resolution processing.py"`
Expected: 仅在注释或非 UI_COLOR 上下文中出现（如有残留则修复）

**Step 3: 更新 .context 文件**

更新 `.context/CURRENT_TASK.md` 和 `.context/CHANGELOG.md`。

**Step 4: 最终提交**

```bash
git add -A
git commit -m "ui: beautify round 5 - minimal tech style complete"
```
