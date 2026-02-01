import argparse
import html
import importlib.util
import mimetypes
import os
import sys
import tempfile
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse


BASE_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = BASE_DIR / "super-resolution processing.py"
JOBS: Dict[str, Dict[str, str]] = {}


def load_cli_module():
    spec = importlib.util.spec_from_file_location("sr_cli", CLI_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load CLI module from {CLI_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sr_cli"] = module
    spec.loader.exec_module(module)
    return module


def default_model_path(scale_factor: int) -> str:
    env_path = os.environ.get("REALESRGAN_MODEL_PATH")
    if env_path:
        return env_path
    model_name = "RealESRGAN_x2plus.pth" if scale_factor == 2 else "RealESRGAN_x4plus.pth"
    return str(Path.home() / ".cache" / "realesrgan" / model_name)


def guess_content_type(path: str) -> str:
    content_type, _ = mimetypes.guess_type(path)
    return content_type or "application/octet-stream"


def render_page(body: str, subtitle: str, status_text: str, badge: str) -> bytes:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Image Super-Resolution System (ESRGAN)</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700&display=swap');
    :root {{
      --bg: #f1f1f1;
      --panel: #f7f7f7;
      --panel-alt: #e6e6e6;
      --stroke: #cfcfcf;
      --text: #1f2937;
      --muted: #6b7280;
      --accent: #3b82f6;
      --accent-strong: #2563eb;
      --success: #22c55e;
      --warning: #facc15;
      --shadow: rgba(0, 0, 0, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Barlow", "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    a {{ color: var(--accent-strong); text-decoration: none; }}
    .app-shell {{
      max-width: 1200px;
      margin: 24px auto;
      background: var(--panel);
      border-radius: 14px;
      box-shadow: 0 10px 30px var(--shadow);
      overflow: hidden;
      border: 1px solid var(--stroke);
    }}
    .app-header {{
      padding: 18px 24px;
      background: #ededed;
      border-bottom: 1px solid var(--stroke);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}
    .app-header h1 {{ margin: 0; font-size: 20px; }}
    .app-header p {{ margin: 6px 0 0; color: var(--muted); font-size: 14px; }}
    .badge {{
      background: #e2f8ec;
      color: #166534;
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      font-weight: 600;
    }}
    .app-body {{ display: grid; grid-template-columns: 260px 1fr; min-height: 560px; }}
    .sidebar {{
      background: var(--panel-alt);
      padding: 20px;
      border-right: 1px solid var(--stroke);
    }}
    .sidebar h2 {{ font-size: 18px; margin: 0 0 16px; }}
    .divider {{ height: 1px; background: var(--stroke); margin: 16px 0; }}
    .field-label {{ display: block; margin-top: 10px; font-weight: 600; font-size: 13px; }}
    .hint {{ font-size: 12px; color: var(--muted); margin-top: 6px; }}
    input[type="file"], select, input[type="number"] {{
      width: 100%;
      margin-top: 6px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--stroke);
      background: #fff;
    }}
    .btn {{
      display: block;
      width: 100%;
      padding: 10px 12px;
      border-radius: 8px;
      border: none;
      margin-top: 10px;
      font-weight: 600;
      cursor: pointer;
      text-align: center;
      text-decoration: none;
    }}
    .btn.primary {{ background: var(--accent); color: #fff; }}
    .btn.primary:hover {{ background: var(--accent-strong); }}
    .btn.secondary {{ background: #dbeafe; color: #1e3a8a; }}
    .btn.success {{ background: var(--success); color: #fff; }}
    .btn.warning {{ background: var(--warning); color: #78350f; }}
    .btn.ghost {{ background: transparent; border: 1px solid var(--stroke); color: var(--text); }}
    .btn.disabled {{ opacity: 0.6; cursor: default; }}
    .toggle {{ display: flex; align-items: center; gap: 10px; margin-top: 10px; font-size: 13px; }}
    .toggle input {{ width: auto; }}
    .workspace {{ padding: 20px; background: #f9f9f9; }}
    .workspace-header {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; font-weight: 600; font-size: 14px; }}
    .panel-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 10px; }}
    .panel {{
      background: #e3e3e3;
      border: 1px solid var(--stroke);
      border-radius: 10px;
      min-height: 360px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #555;
      font-weight: 600;
    }}
    .panel img {{
      max-width: 100%;
      max-height: 100%;
      border-radius: 8px;
    }}
    .panel-footer {{ display: grid; grid-template-columns: 1fr 1fr; margin-top: 12px; font-size: 12px; color: var(--muted); }}
    .metrics {{ margin-top: 16px; }}
    .metrics h3 {{ margin: 0 0 8px; font-size: 13px; }}
    .metrics p {{ margin: 4px 0; font-size: 12px; color: var(--muted); }}
    .status-bar {{
      padding: 10px 20px;
      background: #ededed;
      border-top: 1px solid var(--stroke);
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 12px;
      color: var(--muted);
    }}
    .status-pill {{
      background: #cbd5f5;
      color: #1e3a8a;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 11px;
      margin-left: 8px;
    }}
    @media (max-width: 980px) {{
      .app-body {{ grid-template-columns: 1fr; }}
      .workspace-header, .panel-grid, .panel-footer {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div>
        <h1>Image Super-Resolution System (ESRGAN)</h1>
        <p>{html.escape(subtitle)}</p>
      </div>
      <div class="badge">{html.escape(badge)}</div>
    </header>
    {body}
    <footer class="status-bar">
      <div>{html.escape(status_text)}</div>
      <div>Elapsed: --</div>
    </footer>
  </div>
  <script>
    const fileInput = document.getElementById("image-input");
    const fileName = document.getElementById("file-name");
    if (fileInput && fileName) {{
      fileInput.addEventListener("change", () => {{
        const name = fileInput.files.length ? fileInput.files[0].name : "No file selected";
        fileName.textContent = name;
      }});
    }}
  </script>
</body>
</html>"""
    return html_doc.encode("utf-8")


def render_form(message: str = "") -> bytes:
    status_text = "Model x4 loaded | Device: cpu"
    message_html = f"<div class=\"hint\">{html.escape(message)}</div>" if message else ""
    body = f"""
<form method="POST" enctype="multipart/form-data" action="/process">
  <div class="app-body">
    <aside class="sidebar">
      <h2>Super Resolution</h2>
      <label class="btn primary" for="image-input">Open Image</label>
      <input id="image-input" name="image" type="file" accept=".png,.jpg,.jpeg,.bmp" required />
      <div class="hint" id="file-name">No file selected</div>
      <button type="button" class="btn ghost" disabled>Load Ground Truth</button>
      <div class="divider"></div>
      <label class="field-label" for="scale">Upscale Factor</label>
      <select id="scale" name="scale">
        <option value="4">x4</option>
        <option value="2">x2</option>
      </select>
      <label class="field-label" for="tile">Tile size (0 = no tiling)</label>
      <input id="tile" name="tile" type="number" value="0" min="0" />
      <button type="button" class="btn secondary" disabled>Set Default Output Dir</button>
      <div class="hint">Default output: ./outputs</div>
      <label class="toggle"><input type="checkbox" name="face" />Face Enhancement</label>
      <label class="toggle"><input type="checkbox" name="scratch" />Scratch repair</label>
      <button type="submit" class="btn success">Start Restoration</button>
      <button type="button" class="btn secondary disabled" disabled>Save Comparison</button>
      <button type="button" class="btn warning disabled" disabled>Export Features</button>
      <button type="button" class="btn primary disabled" disabled>Save Result</button>
      <label class="toggle"><input type="checkbox" disabled />Compare Slider</label>
      <div class="metrics">
        <h3>Output vs GT</h3>
        <p>PSNR: --</p>
        <p>SSIM: --</p>
        <p>Load Ground Truth to calculate metrics.</p>
      </div>
      {message_html}
    </aside>
    <section class="workspace">
      <div class="workspace-header">
        <div>Original Input</div>
        <div>Super-Resolution Output</div>
      </div>
      <div class="panel-grid">
        <div class="panel">Waiting for input...</div>
        <div class="panel">Waiting for processing...</div>
      </div>
      <div class="panel-footer">
        <div>Input: -- x --</div>
        <div>Output: -- x --</div>
      </div>
    </section>
  </div>
</form>
"""
    return render_page(body, "Upload one image and run ESRGAN with the desktop pipeline.", status_text, "CPU READY")


def render_result(job: Dict[str, str]) -> bytes:
    status_text = f"Model x{job['scale']} loaded | Device: cpu"
    input_url = f"/asset/{job['id']}/input"
    output_url = f"/asset/{job['id']}/output"
    body = f"""
<div class="app-body">
  <aside class="sidebar">
    <h2>Super Resolution</h2>
    <a class="btn primary" href="/">Process Another Image</a>
    <div class="divider"></div>
    <div class="field-label">Last input</div>
    <div class="hint">{html.escape(job['name'])}</div>
    <div class="field-label" style="margin-top:12px;">Scale</div>
    <div class="hint">x{job['scale']}</div>
    <div class="field-label" style="margin-top:12px;">Processing</div>
    <div class="hint">{html.escape(job['elapsed'])}</div>
    <button type="button" class="btn secondary">Save Comparison</button>
    <button type="button" class="btn warning">Export Features</button>
    <a class="btn primary" href="{output_url}" download>Save Result</a>
    <div class="metrics">
      <h3>Output vs GT</h3>
      <p>PSNR: --</p>
      <p>SSIM: --</p>
      <p>Load Ground Truth to calculate metrics.</p>
    </div>
  </aside>
  <section class="workspace">
    <div class="workspace-header">
      <div>Original Input</div>
      <div>Super-Resolution Output</div>
    </div>
    <div class="panel-grid">
      <div class="panel"><img src="{input_url}" alt="Input" /></div>
      <div class="panel"><img src="{output_url}" alt="Output" /></div>
    </div>
    <div class="panel-footer">
      <div>Input: -- x --</div>
      <div>Output: -- x --</div>
    </div>
  </section>
</div>
"""
    return render_page(body, "Processing complete. Download the output or run another image.", status_text, "DONE")


class WebHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_form())
            return
        if parsed.path.startswith("/result/"):
            job_id = parsed.path.split("/", 2)[2]
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Result not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_result(job))
            return
        if parsed.path.startswith("/asset/"):
            parts = parsed.path.split("/", 3)
            if len(parts) < 4:
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            job_id, kind = parts[2], parts[3]
            job = JOBS.get(job_id)
            if not job:
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            file_path = job.get(kind)
            if not file_path or not os.path.exists(file_path):
                self.send_error(HTTPStatus.NOT_FOUND, "Asset not found")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", guess_content_type(file_path))
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        if self.path != "/process":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        form = self.parse_form()
        if form is None:
            return
        upload = form.get("image")
        if upload is None or not getattr(upload, "filename", None):
            self.send_error(HTTPStatus.BAD_REQUEST, "No file provided")
            return
        scale = int(form.get("scale", "4"))
        tile = int(form.get("tile", "0"))
        use_face = "face" in form
        use_scratch = "scratch" in form
        job_id = uuid.uuid4().hex
        work_dir = Path(tempfile.mkdtemp(prefix="sr_web_"))
        lr_dir = work_dir / "lr"
        sr_dir = work_dir / "sr"
        lr_dir.mkdir(parents=True, exist_ok=True)
        sr_dir.mkdir(parents=True, exist_ok=True)
        input_path = lr_dir / Path(upload.filename).name
        with open(input_path, "wb") as f:
            f.write(upload.file.read())

        module = load_cli_module()
        model_path = default_model_path(scale)
        scratch_model_path = os.environ.get("SCRATCH_MODEL_PATH", "")
        start_time = time.perf_counter()
        try:
            module.esrgan_super_resolution(
                lr_dir=str(lr_dir),
                sr_dir=str(sr_dir),
                model_path=model_path,
                scale_factor=scale,
                use_face_enhance=use_face,
                tile_size=tile,
                use_scratch_repair=use_scratch,
                scratch_model_path=scratch_model_path,
                scratch_threshold=float(os.environ.get("SCRATCH_MASK_THRESHOLD", "0.5")),
                inpaint_radius=int(os.environ.get("SCRATCH_INPAINT_RADIUS", "3")),
            )
        except Exception as exc:
            message = f"Processing failed: {exc}"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_form(message))
            return
        output_path = sr_dir / input_path.name
        if not output_path.exists():
            message = "Processing finished, but no output file was generated."
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(render_form(message))
            return

        elapsed = time.perf_counter() - start_time
        JOBS[job_id] = {
            "id": job_id,
            "input": str(input_path),
            "output": str(output_path),
            "scale": str(scale),
            "name": input_path.name,
            "elapsed": f"{elapsed:.1f}s",
        }

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(render_result(JOBS[job_id]))

    def parse_form(self):
        import cgi

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                },
            )
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, f"Failed to parse form: {exc}")
            return None
        data = {}
        for key in form.keys():
            item = form[key]
            if isinstance(item, list):
                item = item[0]
            if item.filename:
                data[key] = item
            else:
                data[key] = item.value
        return data


def main():
    parser = argparse.ArgumentParser(description="Run ESRGAN web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()

    if not CLI_SCRIPT.exists():
        raise RuntimeError(f"Missing CLI script at {CLI_SCRIPT}")

    server = HTTPServer((args.host, args.port), WebHandler)
    print(f"Web UI running on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
