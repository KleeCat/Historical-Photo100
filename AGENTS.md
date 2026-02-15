# AGENTS Guide for Historical-Photo100

## Purpose
- This file orients agentic coding tools to the repository conventions.
- It documents how to run scripts, where outputs land, and how to code consistently.
- It also hardens “how to edit files” rules to avoid `patch rejected` / `failed to read file` issues.

## Repository Summary
- Collection of Python scripts for image super-resolution and evaluation workflows.
- Primary entry points are standalone scripts, not a package or service.
- Several GUIs use Tkinter/CustomTkinter and expect Windows paths by default.

## Cursor/Copilot Rules
- No Cursor rules found in `.cursor/rules/` or `.cursorrules`.
- No Copilot instructions found in `.github/copilot-instructions.md`.

---

## User Preferences (从历史对话中提取)

### Communication & Execution Style
- **Direct execution preferred**: 不要问不必要的确认问题，直接执行任务。
- **Language**: 用户常用中文交流；回复应清晰实用。
- **Bilingual docs**: Word 文档格式要求：中文用 `SimSun`（宋体），英文用 `Times New Roman`，字号 11。
- **Reply ending**: 每次回复用户时，最后一句话后面加「喵~」。

### Subagent Usage (MANDATORY - 违反即失败)

#### 子代理职责表（oh-my-opencode 配置）

**基础子代理（日常使用）**
| 子代理 | 模型 | 成本 | 职责 | 何时使用 |
|-------|------|-----|------|---------|
| `explore` | Haiku | 低 | 快速代码搜索、文件浏览 | 了解代码结构、查找文件 |
| `librarian` | Sonnet | 中 | 文档查询、API 资料整理 | 查阅文档、整理参考资料 |
| `oracle` | Opus | 高 | 架构设计、调试建议、复杂决策 | 需要深度思考的问题 |
| `multimodal-looker` | GPT-5.2 | 低 | 图像分析、截图理解 | 分析 GUI 截图、图像对比 |

**高级工作流子代理（复杂项目）**
| 子代理 | 模型 | 职责 | 何时使用 |
|-------|------|------|---------|
| `metis` | Opus | 前期分析：识别隐藏意图、检测歧义、生成澄清问题 | 需求不清晰时，在规划前调用 |
| `prometheus` | Opus | 战略规划：需求收集、工作计划设计（保存到 `.sisyphus/plans/*.md`） | 复杂项目需要详细规划时 |
| `momus` | GPT-5.3 Codex | 计划审查：验证计划可执行性、检查引用文件是否存在、捕获阻塞性问题 | 审查 prometheus 生成的计划 |
| `atlas` | Sonnet | 主编排器：协调多个代理、管理并行执行、执行 QA 验证 | 需要协调多个子代理完成复杂工作流 |
| `hephaestus` | GPT-5.3 Codex | 主实现代理：代码编写、测试创建、git 操作 | 需要大量代码实现的任务 |

**完整工作流**（复杂项目推荐）：
```
用户请求 → metis(分析) → prometheus(规划) → momus(审查) → atlas(编排) → hephaestus(实现)
```

#### 基本规则
- **任何涉及代码修改的任务**，必须先用 `task(subagent_type="explore", run_in_background=true)` 探索相关代码库结构和上下文，再动手修改。
- **涉及多文件或复杂逻辑时**，必须用 `task(subagent_type="oracle", run_in_background=false)` 咨询架构/调试建议。
- **不要自己一个人闷头干**：即使任务看起来简单，也至少派一个 explore 子代理先摸清代码结构。
- **并行优先**：如果需要同时了解多个文件/模块，启动多个 background task 并行探索。

#### 成本分层策略（借鉴 Claude Code）
- **优先用低成本子代理**：explore (Haiku) 能做的事不要用 oracle (Opus)
- **按需升级**：explore 搞不定再派 librarian，librarian 搞不定再派 oracle
- **并行低成本 > 串行高成本**：宁可派 3 个 explore 并行，也不要派 1 个 oracle 串行做所有事

### 每次用户消息后的强制检查 (CRITICAL)
- **每次收到新的用户消息后**，无论消息内容是什么（包括"继续"、"好"、"改一下"等简短指令），都必须：
  1. 先派至少 1 个 `explore` 子代理确认当前代码状态
  2. 如果涉及复杂逻辑，额外派 `oracle` 子代理评估方案
- **禁止连续操作**：禁止连续执行 3 次以上的 `read`/`apply_patch`/`edit` 而不调用子代理
- **"继续"不等于"跳过探索"**：用户说"继续"只是让你继续任务，不是让你跳过子代理调用
- **自检规则**：每次准备执行 `apply_patch` 或 `edit` 前，问自己：
  - "我这轮对话调用过子代理了吗？"
  - 如果没有，**必须先调用**，否则视为违规

### 网页和文件读取委托规则 (MANDATORY)
- **网页抓取**：任何需要访问网页、API 文档、在线资源的任务，必须派子代理（`explore` 或 `librarian`）去完成，主代理不要直接调用 `web_fetch` / `browser` 等工具
- **大量文件读取**：如果需要读取超过 3 个文件来了解上下文，必须派 `explore` 子代理批量读取并汇总，主代理只接收汇总结果
- **原因**：主代理上下文窗口宝贵，不应被大量原始内容占用；子代理可以处理后返回精炼信息

### 子代理返回格式要求（借鉴 Claude Code）
- **要求子代理返回摘要**：在任务描述中明确要求「只返回关键发现，不要返回原始文件内容」
- **结构化输出**：要求子代理用列表/表格形式返回，便于主代理快速理解
- **包含文件位置**：要求子代理返回 `文件名:行号` 格式，便于后续定位
- **示例 prompt**：
  ```
  探索 GUI 相关代码，找出所有 Tkinter 窗口创建的位置。
  只返回：1) 文件名:行号 2) 简短描述
  不要返回完整代码块。
  ```

### 子代理失败恢复机制
- **超时处理**：如果子代理 30 秒无响应，主代理应取消并重试或换一个子代理类型
- **结果不完整**：如果子代理返回「没找到」但主代理怀疑有遗漏，派另一个子代理用不同关键词再搜
- **禁止无限重试**：同一个子代理任务最多重试 2 次，之后必须换策略或问用户

### Git & Upload Defaults
- **Default commit scope**: 上传/提交时默认包含整个仓库，除了 `.gitignore` 忽略的文件。
- **Do not commit**: `datasets/`, `outputs/`, model weights (`.pth`, `.pt`), local log folders (`136152022024_黄一洲 Log*/`).
- **GitHub visibility reminder**: GitHub 可见性是仓库级别的，不是文件夹级别。公开仓库 = 所有人可见；私有仓库 = 仅授权协作者可见。

### Context Files
- **Keep `.context/` updated**: 每当有重要变更时，同步更新 `.context/CURRENT_TASK.md` 和 `.context/CHANGELOG.md`。
- **Preserve AGENTS.md**: 任何会话都必须保留此文件中的指令。

### GUI Development Notes
- Main GUI file: `(gui)super-resolution processing.py`
- Batch outputs go to timestamped subdirs: `outputs/batch/<timestamp>/`
- Queue snapshot: `batch_queue.json`
- Known fix: guard `render_main_images` to avoid `TclError: image "pyimageX" doesn't exist`

---

## 🚨 OpenCode Desktop: Workspace MUST be started (critical)
**Before ANY file edit tools can work, the workspace must be started/opened.**

### Required checklist (do not skip)
1) In OpenCode Desktop, click the project name menu → **“启用/启动工作区”**.
2) Confirm the right-side file tree is showing the repo files (not empty).
3) Do a quick sanity read using a relative path (example below):
   - `AGENTS.md`
   - `(gui)super-resolution processing.py`

If the workspace is NOT started, tools may produce:
- `apply_patch verification failed: Failed to read file to update: D:\...`
- `The operation was aborted.`

---

## Paths and Tooling Rules (OpenCode)
### File tools vs Shell tools
- **File tools** (`read/edit/write/apply_patch`) MUST use **workspace-relative paths only**:
  - ✅ `AGENTS.md`
  - ✅ `(gui)super-resolution processing.py`
  - ❌ `D:\HuaweiMoveData\...`
  - ❌ `C:\...`

- **Shell / external tools** (Playwright, npm, python, etc.) may use absolute paths if needed by runtime.

### Golden rule
If you see a drive letter (`D:\` / `C:\`) in any `apply_patch` / `edit` / `write` target, **STOP** and convert to workspace-relative path first.

---

## File Change Strategy (IMPORTANT)
Goal: make edits reliable and prevent `patch rejected`.

### Default rule (inside OpenCode)
1) Always `read` the target file first (or at least read the relevant region).
2) Prefer modifications in this order:
   - **`edit`** for small targeted changes (recommended).
   - **`apply_patch`** for structured diffs ONLY when using the correct OpenCode patch format.
3) If tools fail (aborted / cannot read / patch rejected), output a **standard unified diff** for manual application.

### Do NOT mix two patch formats
There are **two completely different patch formats**:

- **A) apply_patch tool format** (OpenCode-specific; requires Begin/End markers)
- **B) unified diff format** (standard `--- a/ +++ b/`; for manual `git apply`)

If you send (B) to `apply_patch`, it will error:
- `Error: Invalid patch format: missing Begin/End markers`

---

## A) apply_patch TOOL format (OpenCode-specific)
### Hard requirements
- MUST be wrapped by:
  - `*** Begin Patch`
  - `*** End Patch`
- MUST contain exactly one of:
  - `*** Update File: <workspace-relative path>`
  - `*** Add File: <workspace-relative path>`
  - `*** Delete File: <workspace-relative path>`
- The file path MUST be workspace-relative (NO `D:\` / `C:\`).
- MUST NOT be empty (must include a real change).
- MUST patch against the latest file content (read first).

### Example (apply_patch tool payload)
```text
*** Begin Patch
*** Update File: (gui)super-resolution processing.py
@@
-import warnings
+import warnings
+import math
*** End Patch
````

### Common apply_patch failure causes (and how to avoid)

* **Failed to read file to update**:

  * Workspace not started OR wrong path OR file not in current workspace.
  * Fix: start workspace + use relative path.
* **Failed to find expected lines**:

  * Root cause: patch context no longer matches the latest file content (function signature/indentation/whitespace changed, or the file was edited earlier in-session).
  * Typical trigger: reusing an old hunk like `def render_zoomed_image(...)` after code has moved or been modified.
  * Fix workflow:
    1. Re-`read` the exact target region immediately before patching (do not reuse stale snippets).
    2. Keep hunks minimal and anchor on unique nearby lines.
    3. Prefer `edit` with exact `oldString` for single-block replacements.
    4. If one patch partially applied, re-read and regenerate remaining hunks from current content.
    5. Watch for CRLF/LF and spacing differences; treat context as exact text.
* **empty patch / patch rejected**:

  * Patch contains no changes OR hunks do not match current file.
  * Fix: read exact lines first; keep hunks minimal; ensure context matches.

---

## B) Unified diff (STANDARD, for manual apply)

Use this ONLY when:

* you are providing a patch for the user to apply manually, OR
* OpenCode tools are failing.

### Hard requirements

* Must start with:

  * `--- a/<relative path>`
  * `+++ b/<relative path>`
* Paths MUST be workspace-relative (no drive letter), and keep filename exact.
* Do NOT use Windows backslashes in diff headers; use `/` (or keep plain name).
* Never output empty diffs.

### Example (unified diff)

```diff
--- a/(gui)super-resolution processing.py
+++ b/(gui)super-resolution processing.py
@@
-import warnings
+import warnings
+import math
```

### Manual apply tips

* In repo root:

  * `git apply -p1 your.patch`
* If it fails:

  * ensure the diff was produced against the latest file content;
  * increase context lines or regenerate the diff.

---

## Docx rule

* Do NOT patch `.docx` via diff.
* Use `python-docx` to edit or regenerate docx while preserving the reference template’s layout.

---

## Environment Setup (Windows)

* Recommend a virtual environment under `tools/`.
* Example setup (adjust CUDA wheel to your GPU):

```bash
python -m venv tools/historical-photo100-venv
tools/historical-photo100-venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install basicsr realesrgan opencv-python numpy pillow
python -m pip install gfpgan customtkinter scikit-image lpips pytorch-fid matplotlib scipy requests
```

* If GPU is not available, install the CPU torch wheel instead.

---

## Run Commands (Scripts)

* GUI (recommended): `python esrgan_gui.py`
* GUI (CustomTk): `python "(gui)super-resolution processing.py"`
* GUI server variant: `python "(gui)super-resolution processing_server.py"`
* CLI batch SR: `python "super-resolution processing.py"`
* CLI model variant: `python "(model) super-resolution processing.py"`
* Evaluation + reports: `python "Quantitative assessment and frequency domain analysis.py"`
* Download sample data: `python download.py`
* Some scripts have hard-coded default paths; adjust configuration blocks when needed.

---

## Build / Lint / Test

* No build step; scripts run directly with Python.
* No lint configuration is present; run linters only if you add them.
* No automated tests are present by default; add tests under `tests/`.
* Pytest is the preferred runner when tests are added.

### Pytest Examples

* Run all tests: `python -m pytest`
* Run a single file: `python -m pytest tests/test_file.py`
* Run a single test: `python -m pytest tests/test_file.py::TestClass::test_name`
* Run by pattern: `python -m pytest -k "pattern"`
* Run with verbose output: `python -m pytest -vv`

---

## Code Style Guidelines

### Imports

* Order: standard library, third-party, then local modules.
* Separate import groups with a blank line.
* One import per line; avoid wildcard imports.
* Remove unused imports promptly.

### Formatting

* Use 4-space indentation and spaces around operators.
* Keep line length around 100 characters when practical.
* Prefer f-strings for string formatting.
* Avoid excessive reformatting of untouched files.

### Naming Conventions

* Functions and variables: `snake_case`.
* Classes and exceptions: `CamelCase`.
* Constants: `UPPER_SNAKE_CASE`.
* Files should keep their existing naming, even if unconventional.

### Types and Docstrings

* Add type hints for new or modified public functions.
* Use `Optional[T]` and `Union`/`|` where appropriate.
* Avoid `Any` unless there is no realistic alternative.
* Short English docstrings for public functions and non-obvious logic.

### Error Handling

* Guard file IO, model loading, and GPU operations with `try/except`.
* Avoid bare `except`; catch specific exceptions.
* Include context in error messages and preserve tracebacks.
* Fail fast with clear messages when input paths are invalid.

### Logging and UX

* Keep CLI output concise and meaningful.
* Avoid per-iteration spam in logs.
* GUI updates should occur on the main thread or via a queue.
* Prefer deterministic progress messages over noisy prints.

### Image Handling

* Normalize channels (BGR/GRAY/BGRA) explicitly before processing.
* Clip/convert to `uint8` before saving images.
* Validate image dimensions and file extensions before processing.
* When comparing images, ensure alignment and matching filenames.

### Configuration

* Keep default paths and constants in one place per script.
* Allow user overrides via variables, CLI flags, or GUI inputs when possible.
* Avoid hard-coding new absolute paths.

---

## Paths, Data, and Outputs

* Common directories: `LR/`, `SR/`, optional `HR/`, and `evaluation_results/`.
* Some scripts also use `outputs/` or `output/`.
* Do not commit datasets, generated outputs, or model weights.
* Respect existing `.gitignore` entries and keep artifacts local.

---

## Proxy Setup (Optional)

* Recommended env vars:

  * `HTTP_PROXY=http://127.0.0.1:7897`
  * `HTTPS_PROXY=http://127.0.0.1:7897`
  * `ALL_PROXY=socks5://127.0.0.1:7897`

* Quick test (expect 401):

  * `curl -I --proxy socks5h://127.0.0.1:7897 https://api.openai.com/v1/models`

---

## Troubleshooting: patch rejected quick diagnosis

If you hit `patch rejected`, do this order:

1. Verify workspace started (“启用/启动工作区”).
2. Verify the file exists in tree and path is relative (no drive letter).
3. If using `apply_patch` tool:

   * ensure Begin/End markers are present,
   * ensure `*** Update File:` path matches exactly,
   * ensure you read the file first and patch hunks match current content.
4. If still failing, output a standard unified diff for manual apply.
