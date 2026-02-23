# Current Task

Last update: 2026-02-23

## Recently Completed (2026-02-23)
- 对 `super-resolution processing.py` 进行全面代码审查并修复 22 个问题（2 CRITICAL + 7 HIGH + 13 MEDIUM）：
  - [CRITICAL] `torch.load` 添加 `weights_only=True`；`exit(1)` → `sys.exit(1)`。
  - [HIGH] 全部公共函数添加类型提示；异常捕获收窄；docstring 补全；常量提升。
  - [MEDIUM] print→logging；推导式重写；环境变量容错；排序输出；类 docstring；__main__ 重构。

- 修复 `(gui)super-resolution processing.py` 中 6 处类型注解不一致。
- 为 `(gui)super-resolution processing.py` 添加完整的类型提示（100+ 函数和方法）：
  - 为 8 个模块级函数添加类型提示：`clean_state_dict`、`load_scratch_model`、`predict_scratch_mask`、`apply_scratch_repair`、`clamp_value`、`make_comparison_images`、`tensor_to_grid_image`、`save_feature_grids`。
  - 为 ModernApp 类的 100+ 个方法添加类型提示，覆盖所有公共和私有方法。
  - 使用 `typing` 模块注解：`Optional`、`List`、`Tuple`、`Dict`、`Any`、`Callable` 用于复杂类型。
  - 所有参数类型和返回类型现已明确标注，提供更好的 IDE 支持和类型检查。

- Python 代码审查修复（`(gui)super-resolution processing.py`，6 个问题：2 CRITICAL + 4 HIGH）：
  - [CRITICAL] `torch.load` 添加 `weights_only=True` 防止 pickle 任意代码执行。
  - [CRITICAL] `open_last_run_folder` 添加 `realpath` + `isdir` 路径验证。
  - [HIGH] `process_image` finally 块中 `img_output`/`img_gt` 改为 `_state_lock` 下快照读取。
  - [HIGH] `load_gt_image` 中 `self.img_gt` 写入加 `_state_lock` 保护。
  - [HIGH] `upsampler.tile` 修改加 `_model_lock` 保护，取本地引用调用 `enhance()`。
  - [HIGH] `feature_maps` 全链路加锁：hook append、register 重置、export 快照、success 状态检查。

## Recently Completed (2026-02-22)
- 综合代码优化（`(gui)super-resolution processing.py`，净减约 185 行）：
  - 删除 Win32 拖放死代码（~150 行）：`_setup_win32_drop`、`on_drop_files`、`_load_dropped_file`。
  - 提取 `_get_dpi_scale` 和 `_assign_image_to_label` 辅助方法，重构三处图像渲染函数。
  - 合并 `_recreate_output_label` / `_recreate_input_label` 为统一的 `_recreate_label(target)`。
  - 拆分 `process_image` 为 `_stage_scratch_repair`、`_stage_face_enhance`、`_stage_blend_and_texture`。
  - 修复 `_stage_blend_and_texture` 中缺失的 `set_stage` 调用（texture 0.88、finalize 0.95）。
  - 缓存 `CascadeClassifier` 为 `self._face_cascade` 实例变量。
  - 合并 `calculate_metrics` 中 3 个相同的 early-return 分支为单一条件。
  - 移动 `AppearanceModeTracker` 导入到文件顶部。
  - 高频日志降级为 `logger.debug`。
  - 修复 Save Comparison / Export Features 输出目录命名（加入图片名 + 时间戳）。
- 代码审查后优化窗口最小化/恢复机制：
  - 移除冗余 `import time`、减少 `_update_all_scrollable_frames` 调用次数、增强 API 兼容性、`bbox` 空值检查。
- 清理临时文件及 Git 历史。

## Recently Completed (2026-02-21)
- 已完成窗口最小化/恢复机制优化，采用完整的 CTkScrollableFrame 重绘方案。
- 替换三层 `_refresh_idle1/2/sidebar` idle 链为单一 `_force_ctk_redraw()` 方法。
- 新增 `_update_all_scrollable_frames()` 递归更新所有 CTkScrollableFrame 的 Canvas 背景色和 scrollregion。
- 改用 `update_idletasks()` 替代 `update()`，避免 UI 阻塞。
- 添加性能监控，记录执行时间 > 100ms 的警告。
- 根本原因：CTkScrollableFrame 的 `_set_appearance_mode()` 仅更新 Canvas 背景色，不更新 scrollregion；嵌套控件在最小化/恢复时不被重绘。
- 解决方案：直接调用 `AppearanceModeTracker.update_callbacks()` + 递归 Canvas/scrollregion 刷新 + 两层更新策略。

## Recently Completed (2026-02-21)
- 已修复 CTkLabel canvas 渲染残影问题并通过用户可视验收。
- 根本原因：CTkLabel 默认使用透明背景 canvas，文本缩短时旧像素不被清除，产生视觉残留。
- 症状：状态栏 "Done (x4)" 后显示 "output" 残影；overlay "Start restoration" 显示为 "art restorati"（两端截断）。
- 解决方案：给 `status_label` 和 `output_overlay_label` 添加 `fg_color` 匹配父 frame，使 canvas 背景不透明，每次重绘时清除旧像素。

## Recently Completed (2026-02-20)
- 已完成最小化/恢复重影修复并通过用户可视验收。
- 根本原因：CTkButton 内部 CTkCanvas 在 Windows 窗口恢复后不重绘，pure tkinter 无此问题。
- 解决方案：轮询 `self.state()`，检测到 `iconic→normal` 后链式调用三层 `after_idle`，等待 Tk 事件队列完全清空，再用 `wm_attributes('-alpha', 0.99→1.0)` 触发 DWM 重新合成。
- 相关方法：`_poll_window_state`, `_refresh_idle1`, `_refresh_idle2`, `_refresh_sidebar`（约第 800 行）。

## Recently Completed (2026-02-18)
- 已完成侧边栏重影（重绘残影）修复并通过用户可视验收：`Start Restoration` 与 `Compare Slider` 区域不再出现双层错位。
- 已在 `(gui)super-resolution processing.py` 中完成布局稳定化调整：
  - `metrics_frame` 内部统一为 `grid` 布局；
  - `lbl_gt_hint` 改为 `configure(text=...)` 文本切换，避免频繁几何抖动；
  - `metrics_frame` 背景改为与侧栏同色，并为提示行预留 `minsize`。
- 已将左侧容器改为“外层固定 + 内层 `CTkScrollableFrame`”结构，降低高 DPI/溢出场景下的重绘伪影风险。
- 已修复标题显示不完整问题：调整侧栏外层与滚动容器宽度后，`Super Resolution` 文案恢复完整显示。
- 代码提交与远端同步完成：`09fbd40`（`origin/main`）。

## Recently Completed (2026-02-15)
- 已执行历史脱敏清理：使用 `git_filter_repo --replace-text` 重写本地历史，清除已泄露 key 片段；`git log -S <leak-fragment>` 与关键字扫描均无命中。
- 已将重写后的历史推送到远端 `sanitized-main`，并已完成默认分支切换、删除旧 `main`、再创建新 `main`（指向脱敏历史）。
- 远端分支校验：`main` 与 `sanitized-main` 当前均指向脱敏提交 `e9d3531`；旧泄露提交 `a7e38dc` 已不再被远端分支引用。
- 安全收敛：生成 `opencode subagent configuration.txt` 的彻底脱敏版本（不再保留原始会话明文日志），并将 `setup_fucheers_env.bat` 中回显 API Key 改为 `[REDACTED]`。
- 已执行工作区敏感扫描（关键字与泄露片段）确认当前版本无明文 key 命中；下一步为历史重写清理并验证。
- 针对“仍有一闪而过重影”继续做完成态分离优化（同文件）：
  - **完成渲染期防重入**：`_render_output_frame_once` 在渲染期间显式置位 `_rendering_in_progress`，抑制由 `<Configure>` 触发的二次渲染回流。
  - **侧栏更新后置**：单图成功时将侧栏按钮与状态更新延后到完成渲染后（约 190ms），避免与输出渲染同帧抢绘制。
  - **完成链路再轻量化**：最终渲染调度从 70ms 调整为 60ms，且保持 memory-first + file-fallback，减少完成窗口内重绘冲突。
- 验证：`python -m py_compile "(gui)super-resolution processing.py"` 通过。
- 在“重影已明显缩短”基础上继续做无感收敛（同文件）：
  - **按钮更新去重**：`update_action_buttons` 增加状态缓存，目标状态不变时不再重复 `configure`，减少完成瞬间侧栏重绘。
  - **完成渲染轻路径优先**：新增 `render_output_after_completion(...)`，先走内存渲染，失败再文件回退；`render_output_from_file(...)` 不再主线程二次 `read_image`。
  - **完成态单点收敛**：`process_image` 成功分支不再单独提交一组按钮回调，改为在 finalize 回调统一更新（状态文本/按钮/进度），并在调度最终渲染前取消旧成功重绘任务。
- 验证：`python -m py_compile "(gui)super-resolution processing.py"` 通过。
- 用户反馈“处理完成后左侧按钮短暂重影”后的收敛修复（同文件）：
  - **回调合并降频**：`process_image` 成功态与 finally 收尾态的多条 `0ms` `_after_for_run` 改为少量合并回调，避免一帧内对侧栏按钮连续 `configure`。
  - **收尾重绘减载**：成功路径移除 `refresh_output_after_success(...)` 的立即多次重绘，统一改为一次延迟磁盘回读渲染（80ms）。
  - **全局 flush 降低**：`_render_output_frame_once` 去掉 `lbl_img_out.update_idletasks()`，减少收尾阶段对整窗重绘的冲击。
  - **UI 队列节流**：`_drain_ui_queue` 单轮上限从 256 降到 64，避免同一帧执行过多控件 `configure` 造成侧栏拖影。
  - **完成弹窗移除**：单图完成不再弹 `messagebox.showinfo`，改为状态栏完成文本，减少模态框触发的窗口重绘。
- 验证：`python -m py_compile "(gui)super-resolution processing.py"` 通过。
- 继续针对 `(gui)super-resolution processing.py` 的“重影/拖影”做一轮最小风险修复：
  - **图像彩边重影收敛**：`apply_unsharp_mask` 改为仅锐化亮度通道（Y 通道），避免 RGB 同步锐化引入色边；`blend_with_lr` 现在先做相位相关位移检测，并在对齐不可靠时直接跳过融合，降低双影风险。
  - **对比预览伪重影修复**：`build_compare_image` 改为 `INTER_LINEAR` 预览放大，并在分割线添加 2~4px 羽化带，减少硬切缝边误判。
  - **UI 重绘拖影抑制**：`render_main_images_stable`/`on_display_resize` 改为单任务防抖（含 resize 序列号），避免窗口尺寸变化时的重绘风暴。
  - **覆盖层与成功重绘任务收敛**：新增 overlay 动画与成功后重绘任务取消逻辑，`refresh_output_after_success` 从 4 次高频重绘收敛为 1~2 次重试，降低界面拖影概率。
- 验证：`python -m py_compile "(gui)super-resolution processing.py"` 通过。

## Recently Completed (2026-02-14)
- 在“布局保持不变”的前提下做了新一轮最小侵入微调（待用户复测验收）：
  - **重影抑制再收敛**（`suppress_edge_ringing`）：`overshoot` 归一化从 `/22.0` 继续上调到 `/35.0`，`halo_mask` 模糊从 `sigma=0.9` 提升到 `2.0`（更平滑过渡），双边参数从 `d=5,sigmaColor=24,sigmaSpace=20` 进一步收敛到 `d=5,sigmaColor=12,sigmaSpace=20`（降低跨边缘混色）。
  - **无 blend/no-texture 路径去晕降强**（`process_image`）：`natural_blend<=0 && texture_boost<=0` 分支首段去晕强度从 `0.42` 降到 `0.25`，减少去晕本身引入双边的风险。
  - **显示完整性观感修正**（`render_zoomed_image` fit 分支）：移除“把图像合成到整块背景图”的内嵌 letterbox 逻辑，改为仅渲染真实缩放图，由面板自身背景承载留白（不改布局），避免图内黑边导致的“显示不完整”观感。
  - **验证**：`python -m py_compile "(gui)super-resolution processing.py"` 通过；LSP 仍有仓库既有 basedpyright 告警/错误，本轮未新增语法阻断。
- 继续修复 `(gui)super-resolution processing.py` 的双边/重影与显示不完整问题（最小侵入方案）：
  - **重影抑制**：`apply_unsharp_mask` 改为边缘感知细节门控（仅在非强边缘区域增强），避免对高对比边缘继续放大 halo。
  - **融合收紧**：`blend_with_lr` 进一步收紧 `gated_weight` 上限（0.08）并加强边缘区跳过条件，减小低频回注导致的双边感。
  - **纹理阶段显式门控**：`process_image` 仅在 `TEXTURE_ENABLED && TEXTURE_MODEL_ID` 时进入 texture stage，并避免吞掉 `UserCancelledError`。
  - **显示完整性增强**：`render_main_images_stable` 增加更晚时机重绘（350ms）；`_get_image_display_size` 增强初始几何未稳定时的 `update_idletasks` + req-size 兜底；label 重建后增加延迟重绘。
- 语法验证：`python -m py_compile "(gui)super-resolution processing.py"` 通过。
- 新一轮针对“仍未验收通过”的补丁已落地：
  - **新增反光晕阶段**：引入 `suppress_edge_ringing(sr_bgr, lr_bgr, strength)`，在 blend 前对疑似 edge overshoot 区域做局部双边平滑抑制。
  - **交互状态收敛**：处理进行中禁止缩放/平移（`on_zoom`/`on_pan_*` 加 `is_processing` 保护），避免处理中途把视图状态带到完成态。
  - **完成后强制 full-fit**：成功路径在最终刷新前执行 `reset_view_state()`，避免残留 zoom/crop 导致“显示不完整”观感。
  - **输出遮罩策略修正**：`render_main_images` 在存在 `img_out` 时始终执行 `hide_output_overlay()`，降低遮罩残留导致的显示异常。
- 验证结果：`python -m py_compile "(gui)super-resolution processing.py"` 通过；当前 LSP diagnostics 返回 clean。
- 再次加码修复（用户反馈“还是这样”后）：
  - **重影抑制加码**：`suppress_edge_ringing` 参数改强（更高 halo 掩码敏感度 + 更强 bilateral），并在流程中执行双次（blend 前 + finalize 前）。
  - **无附加增强路径下强去晕**：当 `natural_blend<=0` 且 `texture_boost<=0` 时，直接采用更高强度去晕路径，避免“几乎 no-op”导致模型边缘振铃残留。
  - **显示布局收敛**：`display_frame` 的图片行改为 `weight=0`（并保留底部弹性行），新增 `sync_panel_height_for_fit(...)` 依据图像比例同步两侧面板高度，降低大面积灰边造成的“不完整”观感。
  - **渲染入口联动**：`render_main_images()` 在 zoom<=1 时会先执行 `sync_panel_height_for_fit(...)`，再渲染。
- 验证结果（本轮）：`python -m py_compile "(gui)super-resolution processing.py"` 通过；LSP 仍显示大量历史 basedpyright 告警/错误（仓库既有）。
- 布局回滚（用户反馈界面异常后）：
  - 已回退 `display_frame` 的错误网格配置：恢复 `grid_rowconfigure(2, weight=1)`，移除新增空白拉伸行（row 4）。
  - 已移除 `sync_panel_height_for_fit(...)` 及其在 `render_main_images()` 的调用，避免强行改写面板高度导致布局失衡。
  - 验证：`python -m py_compile "(gui)super-resolution processing.py"` 通过。
- 最新迭代（用户反馈“布局正常但两个问题仍在”）：
  - **显示观感修复（不再改布局）**：`render_zoomed_image(...)` 的 fit 分支改为“主图 + 软化背景填充”渲染，避免大面积灰色留白被误判为“显示不完整”；同时根据缩放方向动态选择插值（downscale 用 `INTER_AREA`，upscale 用 `INTER_CUBIC`）。
  - **重影链路收敛**：`suppress_edge_ringing(...)` 改为“overshoot 掩码 + 边缘聚焦 gate”局部抑制，减小全图过宽平滑带来的副作用；双边滤波参数收紧到 `d=7, sigmaColor=40, sigmaSpace=30`。
  - **处理阶段策略调整**：
    - `natural_blend<=0 && texture_boost<=0` 时，首段去晕强度从 `0.92` 降为 `0.78`，避免过度平滑。
    - 常规分支 dehalo 强度从 `0.72/0.60` 降为 `0.62/0.50`。
    - 二次去晕只在 texture 实际执行后才启用，且强度降到 `0.16`；texture 关闭时不再二次去晕。
    - 无融合无锐化时进一步把 `film_grain` 上限压到 `0.02`，减少伪边缘感。
- 验证：`python -m py_compile "(gui)super-resolution processing.py"` 通过，LSP diagnostics（当前检查）clean。

## Recently Completed (2026-02-12)
- 完成两版 GUI 脚本差异分析并输出论文式中文文档：
  - `generated_documents/gui_super_resolution_comparative_study_cn_1770886461751.docx`
  - 对比对象：`(gui)super-resolution processing.py` 与 `(gui)super-resolution processing_server.py`

## In Progress
- Drag-and-drop disabled (windnd/Win32 incompatible with customtkinter); revisit if CTk adds native DnD.

## Known Optimization Opportunities
- **单图完成弹窗**: 每次单图处理完弹 `messagebox.showinfo`，可改为状态栏提示以减少打断。

## Recently Completed (2026-02-10)
- Fixed `actual_widget` identity check bug in `render_zoomed_image`, `show_image_file_ctk`, `show_image_ctk`:
  - Captured `is_output_panel` flag before `_set_label_image` call to prevent wrong panel being lifted after label recreation.
- Fixed batch output panel stale display:
  - `start_next_batch_item` now clears `self.img_output = None` and resets output filename label when advancing to next image.
  - `start_processing` now shows "Processing" overlay uniformly for both single and batch modes.
- Scratch repair thread safety: `process_image` now copies `self.img_input` into local `input_img` under `_state_lock` before processing, never mutates the shared reference.
- Moved `from gfpgan import GFPGANer` to file-level `try/except`; `process_image` checks `GFPGANer is None` instead of re-importing.
- `render_main_images` clear paths now use `_recreate_input_label()` / `_recreate_output_label()` instead of `configure(image="")` to avoid pyimageX errors.
- Unified `_state_lock` usage: added lock snapshots in `render_main_images`, `_render_output_frame_once`, `calculate_metrics`, `save_comparison`, `start_next_batch_item`.

## Recently Completed (2026-02-09)
- Run-scoped callback guard, thread-safe UI queue, persistent CTkImage refs, label recreation for pyimageX fix.
- See CHANGELOG.md for full details.

## Recently Completed (2026-02-08)
- **Thread safety**: `_state_lock` (img_input/img_output), `_model_lock` (model loading).
- **Model load mutex**: ComboBox disabled during load; `_model_lock.acquire(blocking=False)`.
- **Batch retry fix**: 500ms delay + status text before retry; prevents tight recursion.
- **GFPGANer cache**: `face_enhancer` / `face_enhancer_scale` instance vars; only rebuilt on scale change.
- **GPU VRAM cleanup**: `torch.cuda.empty_cache()` in process_image finally block.
- **Auto tile**: `auto_tile_size(h, w, scale)` picks tile based on VRAM; set on `upsampler.tile`.
- **Scratch Repair UI**: switch in sidebar row 9; calls `apply_scratch_repair` before upscale.
- **Output formats**: save_image supports TIFF/WebP with quality params; save dialog updated.
- **Progress bar fix**: creep animation (+0.001/tick) when bar catches target; stage targets rebalanced (upscale 10%→65%).
- **Logging**: `logging` module replaces `print`; `logger = logging.getLogger("super_resolution_gui")`.
- **Type hints**: added to `ensure_dir`, `write_json_file`, `save_image`, `blend_images`, `apply_unsharp_mask`, `apply_film_grain`, `blend_with_lr`, `estimate_image_metrics`, `auto_tile_size`.
- **UI constants**: `UI_WINDOW_SIZE`, `UI_SIDEBAR_WIDTH`, `UI_COLOR_*` centralized.
- **Image display fix**: `render_zoomed_image` / `show_image_ctk` use parent frame size (not label); fit mode at zoom<=1.0.
- **Image panel layout fix**: filenames moved below headers (row 1), image panels row 2, resolution labels row 3 (no overlap).
- **Filename UX**: show `Input: <basename>` on load and `Output: <basename>` after processing.
- **Image sizing refinement**: `_get_image_display_size(label_widget)` now reads panel frame dimensions directly to avoid clipping.
- **Progress creep refinement**: long-stage creep increased to `max(0.0015, remaining * 0.012)` toward 0.98.
- **Output render regression fix**: added `prepare_display_image()` normalization to contiguous `uint8` BGR before GUI render.
- **Output repaint hardening**: added `after(20, show_image_ctk)` after completion to force final output frame refresh.
- **Output render pipeline hardening**: added `refresh_output_after_success()` and consolidated final success repaint on UI thread.
- **Overlay safety**: `render_main_images()` now auto-hides output overlay when processing is finished and output exists.
- **Output stacking fix**: `hide_output_overlay()` now calls `lower()`, and output labels call `lift()` after each render.
- **Stable output buffer**: after saving `run_output_path`, GUI reloads that snapshot for display to avoid transient memory-layout render failures.
- **Agent workflow hardening**: updated `AGENTS.md` with `apply_patch` stale-context guidance for
  `Failed to find expected lines` (re-read region, use minimal hunks, regenerate after partial apply).
- **Batch output visibility fix**: keep previous output visible during batch item transitions; do not force overlay on output panel in batch mode.
- **Batch labeling fix**: update `Input:` filename for each batch item in `start_next_batch_item()`.
- **Batch repaint pacing**: add 120ms delay before launching next batch item to let just-finished output render.
- **Single repaint hardening**: added `_render_output_frame_once()` + multi-pass retries in `refresh_output_after_success()`.
- **Single final fallback**: added delayed `render_output_from_file(run_output_path)` after success for guaranteed output panel refresh.
- **View reset on single runs**: `run_processing_thread()` now calls `reset_view_state()` before processing to avoid stale zoom crop display.
- **Direct file render fallback**: added `show_image_file_ctk(path, label_widget)` and updated `render_output_from_file()` to render from disk first, then sync `img_output` for metrics.
- **GIL crash fixes**: `load_model` Tkinter calls wrapped in `self.after(0, ...)`; windnd removed.
- **Drag-and-drop disabled**: windnd and Win32 SetWindowLongPtr both crash CTk; methods retained for future.

## Handoff Notes (2026-01-24)
- Server uses `/root/rivermind-data/venv-hp` with system-site-packages; do not reinstall torch.
- Required pins: numpy 1.26.4, opencv-python-headless 4.8.1.78, basicsr 1.4.2 (`--no-deps --no-build-isolation`), gfpgan 1.3.8 (`--no-deps`), realesrgan 0.3.0 (`--no-deps`).
- X11 GUI via VcXsrv + `ssh -X`; set `DISPLAY=localhost:0.0` on Windows.
- If diffusers error `torch.distributed.device_mesh` appears, downgrade: diffusers 0.25.1, transformers 4.36.2, accelerate 0.25.0.
- Server should pull latest commit (336a190) or upload files before retest.

## Handoff Notes (2026-01-25)
- Local repo at `D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100`; server repo at `/root/rivermind-data/Historical-Photo100`.
- RDP works via tunnel: `ssh -L 3390:localhost:3389 root@sh01-ssh.gpuhome.cc -p 30011`, then RDP to `localhost:3390` (Xorg). Restart xrdp as root (no sudo).
- Terminal paste/encoding fixed; custom motd error silenced in `~/.bashrc` on server.
- Backup GUI texture updates: filename labels (basename), error handler captures exception string, texture logs (loading/ready/start/done/failed), fp16 variant support, CPU offload + VAE slicing/tiling, `TEXTURE_MAX_DIM` downscale + upsample, and `upsampler` guard in `run_processing_thread`.
- Diffusers pins on server: diffusers 0.25.1, transformers 4.36.2, accelerate 0.25.0, huggingface_hub 0.20.2, safetensors.
- Model path: `/root/rivermind-data/models/stable-diffusion-v1-5` (fp16 only). `TEXTURE_MAX_DIM=1024` tested successfully.
- Local commit `c570464` pushed to `main` (texture refinement stability under VRAM limits).

## Done
- Added gitignore entries for datasets, outputs, and new log folders.
- Show `Batch i/N` prefix in status updates during batch runs.
- Ignore `outputs/batch/` run artifacts in `.gitignore`.
- Added additional cancel checks to shorten batch cancel latency in `(gui)super-resolution processing.py`.
- Compacted sidebar spacing and improved disabled button label contrast in `(gui)super-resolution processing.py`.
- Added batch queue count pop-up and `batch_queue.json` output for batch runs.
- Switched Codespaces devcontainer image to `mcr.microsoft.com/devcontainers/python:3.10` and removed python feature (yarn repo error fix).
- Configured Git LFS tracking and migrated data folders into Git.
- Added `LR/`, `HR/`, `SR/`, `outputs/` directories with sample assets committed.
- Implemented `web_sr_server.py` web UI with desktop-like layout and feature parity (GT metrics, comparison, feature export).
- Added GUI guards for missing input/model before processing to improve stability.
- Added per-run output snapshots and JSON logs under `outputs/<timestamp>_<name>_<id>/` for reproducibility.
- Added per-stage timing and metrics logging (PSNR/SSIM, per-step durations) to `run_log.json`.
- Added "Open Last Run Folder" button for quick access to outputs.
- Added scratch repair and colorization flow to the server GUI pipeline.
- Added non-systemd xrdp startup steps to `.context/remote_access.md`.
- Normalized remaining SSH port references across the repo to 30011.
- Updated SSH port references to 30011 for the new instance.
- Set `TEXTURE_MAX_DIM` default to 1536 for the server GUI.
- Renamed the backup GUI to `(gui)super-resolution processing_server.py` and updated runbook references.
- Updated `AGENTS.md` response suffix from "喵" to "喵~".
- Moved `docs/remote_access.md` to `.context/remote_access.md` and added the response suffix rule to `AGENTS.md`.
- Added `docs/remote_access.md` with RDP tunnel and texture pipeline notes.
- Logged 2026-01-25 RDP and texture pipeline handoff notes.
- Updated `AGENTS.md` with test placement and file IO/optional dependency guidance.
- Clone instance created and data disk verified.
- Real-ESRGAN weights present under `/root/.cache/realesrgan`.
- GUI output refresh fix applied locally.
- Backup GUI has texture generation enabled locally.
- Backup GUI includes torch.xpu compatibility stub for diffusers.
- Stable Diffusion v1.5 diffusers repo trimmed to ~9.4G at `/root/rivermind-data/models/stable-diffusion-v1-5`.
- GT is cleared when loading a new input image in both GUI files.
- Added output refresh fallback to force render via `show_image_ctk` if zoom render fails.
- Updated `AGENTS.md` with build/test commands and code-style guidance.
- Refined `AGENTS.md` with build/lint/test commands and agent style guidance (2026-01-26).

## Command Steps
1) Upload GUI files
```
scp -P 30011 "D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(gui)super-resolution processing.py" root@sh01-ssh.gpuhome.cc:/root/rivermind-data/Historical-Photo100/
scp -P 30011 "D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\(gui)super-resolution processing_server.py" root@sh01-ssh.gpuhome.cc:/root/rivermind-data/Historical-Photo100/
```

2) Connect with X11
```
set DISPLAY=localhost:0.0
ssh -X -o ForwardX11Trusted=yes -o ExitOnForwardFailure=yes root@sh01-ssh.gpuhome.cc -p 30011
```

3) Run texture GUI
```
cd /root/rivermind-data/Historical-Photo100
source /root/rivermind-data/venv-hp/bin/activate
TEXTURE_MODEL_ID="/root/rivermind-data/models/stable-diffusion-v1-5" \
  python "(gui)super-resolution processing_server.py"
```

4) If diffusers is missing
```
python -m pip install diffusers transformers accelerate safetensors huggingface_hub
```

## Notes
- Prefer running GUI via X11 forwarding (VcXsrv + `ssh -X`).
- Keep large models on `/root/rivermind-data` to avoid filling system disk.
