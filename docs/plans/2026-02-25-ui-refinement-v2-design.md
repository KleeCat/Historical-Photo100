# UI Refinement v2 — 精致度提升设计文档

日期: 2026-02-25
基于: PR #4 Apple-style UI redesign 合并后的 main 分支

## 目标

在现有极简科技风骨架上，通过三个维度的微调提升界面精致度和专业感：
1. 彩色 Emoji → Lucide 单色线性 SVG 图标
2. 底部工具栏去容器化 + 视觉层级优化
3. 左侧控制栏间距与控件比例微调

## 一、Lucide SVG 图标系统

### 文件结构

```
gui_pyside/
  icons/              ← Lucide SVG 文件（约 12 个）
  icon_helper.py      ← SVG 加载 + 着色工具
```

### 图标映射

| 位置 | 当前 Emoji | Lucide 图标 |
|------|-----------|------------|
| Open Image | 📂 | folder-open.svg |
| Load Ground Truth | 📏 | ruler.svg |
| Start Restoration | ▶ | play.svg |
| Run Folder (Batch) | 📁 | folders.svg |
| Cancel | ✖ | x.svg |
| Output dir button | 📁 | folder.svg |
| Comparison (toolbar) | ↔ | columns-2.svg |
| Features (toolbar) | 🧠 | brain.svg |
| Open Folder (toolbar) | 📂 | folder-open.svg |
| Save Result (toolbar) | 💾 | save.svg |
| Input placeholder | 🖼 | image.svg |
| Output placeholder | ✨ | sparkles.svg |

### icon_helper.py

- `load_icon(name, color, size)` — 读取 SVG，正则替换 stroke 颜色，QSvgRenderer → QPixmap → QIcon
- 亮色模式默认颜色: `#1C1C1E`（跟随 UI_COLOR_TEXT）
- 暗色模式自动反色: `#F2F2F7`（跟随 UI_COLOR_TEXT dark）
- 图标尺寸: 16px（侧栏按钮）、14px（工具栏按钮）

## 二、底部工具栏去容器化

- 移除工具栏外层 card 样式（白色背景、圆角、阴影边框）
- 按钮直接放置在主背景 `#F2F2F7` 上
- 图像区与按钮区之间加 1px 分割线（亮: `#E5E5EA`，暗: `#38383A`）
- "Save Result" 按钮: 深灰底 `#1C1C1E` + 白字（暗色模式反转）
- 所有工具栏 Emoji 替换为 Lucide SVG 图标

### 视觉层级

```
绿色主按钮(Start) > 深灰按钮(Save) > 浅灰按钮(Comparison/Features/Open Folder)
```

## 三、左侧控制栏微调

### Toggle Switch 缩小

- 轨道: 52×28 → 44×24
- 拇指半径比例不变（h/2 - 3）
- 动画行程相应缩短

### 按钮边框

- 非 primary 按钮加 `border: 1px solid #E5E5EA`（暗: `#38383A`）
- primary 按钮（Start Restoration）保持绿色样式不变

### 间距呼吸感

- 卡片内控件间距: 6px → 8px
- 设置项之间垂直间距适当增加

### 不改动

- 卡片标题保持纯文字，不加图标
- 卡片整体结构和圆角不变
- 按钮高度不变（32px / 38px）

## 涉及文件

| 文件 | 改动 |
|------|------|
| `gui_pyside/icons/*.svg` | 新建，12 个 Lucide SVG |
| `gui_pyside/icon_helper.py` | 新建，图标加载工具 |
| `gui_pyside/styles.py` | Toggle 尺寸、QSS 按钮边框、工具栏样式 |
| `gui_pyside/sidebar.py` | Emoji → QIcon、间距调整 |
| `gui_pyside/display.py` | Emoji → QIcon、工具栏去容器化、Save 按钮样式 |
