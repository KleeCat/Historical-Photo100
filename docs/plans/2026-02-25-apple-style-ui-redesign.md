# Apple 风格 UI 全面优化设计

## 概述
将当前 PySide6 GUI 从扁平极简风格升级为 Apple/macOS 风格，包含柔和阴影、大圆角、层次分明的背景、精致的交互反馈。

## 改动范围

### 1. styles.py — QSS 全面重写
- 卡片：圆角 12px，双层 border 模拟阴影效果，内边距 12px
- 按钮：hover 亮度变化 + pressed 下沉效果，圆角 8px
- 侧栏：独立背景色（比主区域略深/浅），右侧 1px 分割线
- 滑块：groove 6px 高，handle 18x18 带阴影
- 进度条：圆角加粗到 6px
- 输入框：focus 状态加主题色边框
- 全局字体微调：标题用 semibold

### 2. sidebar.py — 卡片布局优化
- 卡片内部 padding 12px，元素间距 8px
- 标题区域加装饰下划线
- 按钮高度统一 36px
- Output Directory 行优化间距

### 3. display.py — 空面板占位 + 工具栏美化
- 空面板显示居中图标 + 提示文字（"Open an image to begin" / "Output will appear here"）
- 工具栏按钮加图标前缀（用 Unicode 符号）
- 面板标题行精简（合并到面板内部）

### 4. statusbar.py — 微调
- 进度条样式跟随新 QSS
- 状态文字字号微调

### 5. main.py — 侧栏分割线
- centralWidget 布局加 QFrame 分割线

## 不改动
- 功能逻辑、信号连接、workers、processing 等不动
- 文件结构不变
