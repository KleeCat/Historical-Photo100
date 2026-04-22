# DDColor 模型说明

本目录用于放置答辩演示所需的 DDColor 离线权重。

## 目录结构

- `pytorch_model.pt`：DDColor 推理权重

## 默认加载路径

程序默认从以下路径加载模型：

- `models/colorization/ddcolor/pytorch_model.pt`

也可以通过环境变量覆盖：

- `DDCOLOR_MODEL_PATH`

## 使用建议

- 建议在答辩机器上提前完成权重放置与首次加载验证。
- 当前集成方案默认走 CPU 推理，单张图片耗时会高于旧版 OpenCV 彩色化，但效果通常更自然。
- 如需迁移项目，只要保持本目录结构不变，GUI 默认彩色化功能即可直接加载。
