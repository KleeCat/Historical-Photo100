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
  <title>Super-Resolution Web UI</title>
  <style>
    body {{ font-family: Arial, sans-serif; background: #f5f5f5; margin: 0; }}
    header {{ padding: 24px 32px; background: #1f2937; color: #fff; }}
    main {{ padding: 24px 32px; }}
    form {{ background: #fff; padding: 20px; border-radius: 12px; max-width: 640px; }}
    label {{ display: block; margin-top: 12px; font-weight: 600; }}
    input[type="file"], select, input[type="number"] {{ width: 100%; margin-top: 6px; }}
    .row {{ margin-top: 12px; }}
    button {{ margin-top: 16px; padding: 10px 16px; background: #2563eb; color: #fff; border: none; border-radius: 8px; }}
    .note {{ color: #374151; margin-top: 12px; }}
    .result {{ margin-top: 24px; }}
  </style>
</head>
<body>
  <header>
    <h1>Super-Resolution Web UI</h1>
    <p>Upload one image and run ESRGAN in this container.</p>
  </header>
  <main>
    {body}
  </main>
</body>
</html>"""
    return html_doc.encode("utf-8")


def render_form(message: str = "") -> bytes:
    message_html = f"<p class=\"note\">{html.escape(message)}</p>" if message else ""
    body = f"""
{message_html}
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
    <label><input type="checkbox" name="face" /> Face enhancement (GFPGAN)</label>
    <label><input type="checkbox" name="scratch" /> Scratch repair</label>
  </div>
  <button type="submit">Run Super-Resolution</button>
</form>
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
<p class="note">Done! Right click the image to save.</p>
<div class="result">
  <img src="{result_url}" style="max-width: 100%; border-radius: 12px;" />
</div>
<p><a href="/">Process another image</a></p>
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
