# gui_pyside 模型权重清单

## RealESRGAN 超分辨率模型

| 文件名 | 默认路径 | 用途 |
|--------|----------|------|
| `RealESRGAN_x4plus.pth` | `~/.cache/realesrgan/` | 4x 超分辨率 |
| `RealESRGAN_x2plus.pth` | `~/.cache/realesrgan/` | 2x 超分辨率 |

- 环境变量 `REALESRGAN_MODEL_DIR` 可自定义目录
- 代码位置: `models.py` → `ModelManager.load_esrgan()`

## GFPGAN 人脸增强模型

| 文件名 | 默认路径 | 用途 |
|--------|----------|------|
| `GFPGANv1.3.pth` | `~/.cache/gfpgan/` | 人脸修复增强 |

- 环境变量 `GFPGAN_MODEL_PATH` 可自定义路径
- 代码位置: `models.py` → `ModelManager.load_face_enhancer()`

## GFPGAN 辅助模型（库内部依赖）

| 文件名 | 路径 | 用途 |
|--------|------|------|
| `detection_Resnet50_Final.pth` | `gfpgan/weights/` | 人脸检测 |
| `parsing_parsenet.pth` | `gfpgan/weights/` | 人脸解析 |

- 这两个文件由 GFPGAN 库内部调用，非 gui_pyside 直接引用

## 划痕修复模型

- 由 `processing.py` → `load_scratch_model(model_path, device)` 加载
- 路径由外部传入，无固定默认值
