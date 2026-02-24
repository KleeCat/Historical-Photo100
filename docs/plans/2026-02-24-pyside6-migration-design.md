# GUI 框架迁移设计文档：CustomTkinter → PySide6

日期: 2026-02-24
分支: `feat/pyside6-migration`

## 背景

当前 GUI 基于 CustomTkinter + tkinter，存在 Windows 下最小化恢复黑块闪烁问题，
属于 tkinter 渲染机制的已知限制，应用层无法修复。迁移到 PySide6 从根本上解决。

## 设计原则

- 全部功能 1:1 迁移，用户无感知切换
- 保持当前极简科技风（绿色强调色 + 黑白灰）
- 模块化拆分，单文件夹结构
- GUI 与处理逻辑分离

## 模块结构

```
gui_pyside/
  __init__.py
  main.py          # 入口 + MainWindow(QMainWindow)
  sidebar.py       # SidebarWidget(QScrollArea) — 设置卡片和按钮
  display.py       # ImageDisplayWidget + ImagePanel(QGraphicsView)
  statusbar.py     # StatusBarWidget — 状态文字+计时器+进度条
  dialogs.py       # PreviewDialog(QDialog)
  styles.py        # 颜色常量 + generate_stylesheet()
  workers.py       # ModelLoadWorker/ProcessWorker/BatchWorker(QThread)
  utils.py         # numpy↔QPixmap 转换、DPI 工具
  processing.py    # 图像处理逻辑（超分、人脸增强、划痕修复、纹理）
  metrics.py       # PSNR/SSIM/LPIPS 计算
  models.py        # 模型加载/管理（RealESRGAN、GFPGAN、diffusers）
```

原入口文件 `(gui)super-resolution processing.py` 保留为启动器。

## 样式系统

用 QSS（Qt Style Sheet）复刻当前极简科技风配色：

```python
# styles.py — 颜色常量保持不变
UI_COLOR_PRIMARY = "#10B981"
UI_COLOR_PRIMARY_HOVER = "#059669"
UI_COLOR_DANGER = "#EF4444"
UI_COLOR_BG = ("#FAFAFA", "#0A0A0A")          # (light, dark)
UI_COLOR_CARD_BG = ("#FFFFFF", "#141414")
UI_COLOR_CARD_BORDER = ("#E5E5E5", "#262626")
# ... 其余常量同现有定义

def generate_stylesheet(dark: bool = False) -> str:
    """根据明暗模式生成完整 QSS"""
```

一个 `app.setStyleSheet()` 调用切换整个应用主题。

## 线程模型

用 QThread + Signal/Slot 替代 threading.Thread + queue.Queue：

```python
class ModelLoadWorker(QThread):
    progress = Signal(float, str)
    finished = Signal(bool, str)

class ProcessWorker(QThread):
    progress = Signal(float, str)
    image_ready = Signal(object)
    metrics_ready = Signal(dict)
    finished = Signal(bool, str)

class BatchWorker(QThread):
    item_done = Signal(int, int)
    finished = Signal(bool, str)
```

- Signal.emit() 自动跨线程调度到主线程
- 取消操作用 QThread.requestInterruption()
- 不再需要 _ui_queue / _drain_ui_queue / after() 定时器

## 图像显示

用 QGraphicsView + QGraphicsScene 替代 CTkLabel + PIL 手动计算：

- **ImagePanel(QGraphicsView)**：单个图像面板，内置缩放/平移
  - 滚轮缩放：重写 wheelEvent
  - 右键平移：setDragMode(ScrollHandDrag)
  - 硬件加速渲染：setRenderHint(SmoothPixmapTransform)
- **ImageDisplayWidget(QWidget)**：双面板容器，左输入右输出
  - 两个 panel 的 transform 联动同步缩放/平移
- **对比模式**：QGraphicsPixmapItem 叠加 + setClipRect() 实现分割线

去掉 render_zoomed_image()、calculate_view_window() 等 ~150 行手动计算。

## 控件映射

| CTk 控件 | PySide6 替代 | 备注 |
|----------|-------------|------|
| CTkFrame | QFrame / QWidget | QSS 圆角边框 |
| CTkLabel | QLabel | 原生富文本 |
| CTkButton | QPushButton | QSS hover |
| CTkSlider | QSlider | 水平滑块 |
| CTkSwitch | QCheckBox | QSS toggle 样式 |
| CTkComboBox | QComboBox | 下拉选择 |
| CTkEntry | QLineEdit | 文本输入 |
| CTkProgressBar | QProgressBar | QSS 高度/颜色 |
| CTkScrollableFrame | QScrollArea | 原生滚动 |
| CTkToplevel | QDialog | 预览窗口 |
| CTkImage | QPixmap | numpy→QImage→QPixmap |
| ToolTip (自定义) | QToolTip | widget.setToolTip() 一行 |
| filedialog | QFileDialog | 系统原生 |
| messagebox | QMessageBox | 系统原生 |

## 迁移范围

全部功能 1:1 迁移：
- 图像加载/保存/显示
- 超分辨率处理（单张+批量）
- 人脸增强、划痕修复、纹理增强、胶片颗粒
- 缩放/平移/对比模式
- 指标计算（PSNR/SSIM/LPIPS）
- 特征图导出
- 预览窗口
- 暗色/亮色主题切换
