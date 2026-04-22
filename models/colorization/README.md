# 老照片彩色化离线模型说明

请将以下文件放到当前目录：

- `colorization_release_v2.caffemodel`
- `colorization_deploy_v2.prototxt`
- `pts_in_hull.npy`

默认加载路径：

- `models/colorization/colorization_release_v2.caffemodel`

可通过环境变量覆盖主模型路径：

- `COLORIZATION_MODEL_PATH`

注意：

- 若使用环境变量覆盖主模型路径，`.prototxt` 与 `pts_in_hull.npy` 默认仍从同级目录读取；
- 答辩前请在离线机器上提前准备好全部模型文件；
- 当前实现按 CPU 离线推理设计，优先保证稳定演示。
