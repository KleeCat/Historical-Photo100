import argparse
import html
import importlib.util
import os
import sys
import tempfile
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict
from urllib.parse import parse_qs, urlparse


BASE_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = BASE_DIR / "super-resolution processing.py"
JOBS: Dict[str, str] = {}


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


def render_page(body: str) -> bytes:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Super-Resolution Studio</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Barlow:wght@400;600;700&display=swap');
    :root {{
      --bg: #0b1220;
      --panel: #111827;
      --panel-2: #1f2937;
      --card: #0f172a;
      --accent: #4f46e5;
      --accent-2: #22c55e;
      --text: #e5e7eb;
      --muted: #94a3b8;
      --border: #253041;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Barlow", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top, #1f2937 0%, #0b1220 55%);
      color: var(--text);
      margin: 0;
      min-height: 100vh;
    }}
    a {{ color: #93c5fd; text-decoration: none; }}
    .app {{ max-width: 1120px; margin: 32px auto; padding: 0 24px 40px; }}
    .hero {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 24px 28px;
      border-radius: 18px;
      background: linear-gradient(135deg, #0f172a 0%, #111827 55%, #1f2937 100%);
      border: 1px solid var(--border);
      box-shadow: 0 18px 30px rgba(15, 23, 42, 0.35);
    }}
    .hero h1 {{ margin: 0 0 8px; font-size: 28px; letter-spacing: 0.02em; }}
    .hero p {{ margin: 0; color: var(--muted); }}
    .badge {{
      background: rgba(34, 197, 94, 0.15);
      color: #86efac;
      padding: 6px 14px;
      border-radius: 999px;
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .layout {{ display: grid; grid-template-columns: 260px 1fr; gap: 24px; margin-top: 24px; }}
    .panel, .card {{
      background: var(--panel);
      border-radius: 18px;
      padding: 20px;
      border: 1px solid var(--border);
    }}
    .panel h2, .card h2 {{
      margin: 0 0 12px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .list {{ list-style: none; padding: 0; margin: 0; }}
    .list li {{
      margin-bottom: 12px;
      padding: 10px 12px;
      background: var(--panel-2);
      border-radius: 12px;
      color: #d1d5db;
      font-size: 14px;
    }}
    form {{ margin: 0; }}
    label {{ display: block; margin-top: 12px; font-weight: 600; color: #e2e8f0; }}
    input[type="file"], select, input[type="number"] {{
      width: 100%;
      margin-top: 8px;
      padding: 10px 12px;
      background: var(--card);
      border: 1px solid var(--border);
      color: var(--text);
      border-radius: 12px;
    }}
    .row {{ margin-top: 10px; display: grid; gap: 8px; }}
    .toggle {{ display: flex; align-items: center; gap: 10px; color: #d1d5db; }}
    button {{
      margin-top: 16px;
      padding: 12px 16px;
      background: var(--accent);
      color: #fff;
      border: none;
      border-radius: 12px;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 10px 20px rgba(79, 70, 229, 0.35);
    }}
    .note {{ color: var(--muted); margin-top: 12px; }}
    .result {{ margin-top: 20px; }}
    .result img {{
      width: 100%;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #0b1220;
    }}
    @media (max-width: 900px) {{
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    {body}
  </div>
</body>
</html>"""
    return html_doc.encode("utf-8")


def render_form(message: str = "") -> bytes:
    message_html = f"<p class=\"note\">{html.escape(message)}</p>" if message else ""
    body = f"""
<section class="hero">
  <div>
    <h1>Super-Resolution Studio</h1>
    <p>Upload one image and run ESRGAN with the same pipeline as the desktop app.</p>
  </div>
  <div class="badge">CPU Ready</div>
</section>
{message_html}
<div class="layout">
  <section class="panel">
    <h2>Pipeline</h2>
    <ul class="list">
      <li>Real-ESRGAN upscale (x2/x4)</li>
      <li>Optional GFPGAN face restore</li>
      <li>Scratch repair model if provided</li>
      <li>Outputs saved in temp workspace</li>
    </ul>
  </section>
  <section class="card">
    <h2>Input</h2>
    <form method="POST" enctype="multipart/form-data" action="/process">
      <label>Input image</label>
      <input type="file" name="image" accept=".png,.jpg,.jpeg,.bmp" required />
      <label>Scale factor</label>
      <select name="scale">
        <option value="4">x4 (default)</option>
        <option value="2">x2</option>
      </select>
      <label>Tile size (0 = no tiling)</label>
      <input type="number" name="tile" value="0" min="0" />
      <div class="row">
        <label class="toggle"><input type="checkbox" name="face" /> Face enhancement (GFPGAN)</label>
        <label class="toggle"><input type="checkbox" name="scratch" /> Scratch repair</label>
      </div>
      <button type="submit">Run Super-Resolution</button>
    </form>
  </section>
</div>
"""
    return render_page(body)


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
            parts = parsed.path.split("/", 2)
            if len(parts) == 3 and parts[2] in JOBS:
                file_path = JOBS[parts[2]]
                if os.path.exists(file_path):
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "image/png")
                    self.end_headers()
                    with open(file_path, "rb") as f:
                        self.wfile.write(f.read())
                    return
            self.send_error(HTTPStatus.NOT_FOUND, "Result not found")
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        if self.path != "/process":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(HTTPStatus.BAD_REQUEST, "Expected multipart form")
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
        JOBS[job_id] = str(output_path)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        result_url = f"/result/{job_id}"
        body = f"""
<section class="hero">
  <div>
    <h1>Super-Resolution Studio</h1>
    <p>Processing complete. Right click to save the output.</p>
  </div>
  <div class="badge">Complete</div>
</section>
<div class="layout">
  <section class="panel">
    <h2>Next steps</h2>
    <ul class="list">
      <li>Right click the image to download.</li>
      <li><a href="/">Process another image</a></li>
      <li>Keep this tab open while you download.</li>
    </ul>
  </section>
  <section class="card">
    <h2>Result</h2>
    <div class="result">
      <img src="{result_url}" alt="Super-resolution output" />
    </div>
  </section>
</div>
"""
        self.wfile.write(render_page(body))

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
    parser = argparse.ArgumentParser(description="Run super-resolution web UI")
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
