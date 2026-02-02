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