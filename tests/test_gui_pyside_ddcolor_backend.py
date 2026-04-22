import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gui_pyside.ddcolor_backend import (
    DDColorModelNotFoundError,
    get_ddcolor_model_path,
    load_ddcolor_backend,
)


class TestDDColorBackend(unittest.TestCase):
    def test_get_ddcolor_model_path_prefers_explicit_path(self):
        explicit = Path("custom") / "demo.pt"

        resolved = get_ddcolor_model_path(explicit)

        self.assertEqual(resolved, explicit)

    def test_get_ddcolor_model_path_defaults_to_models_directory(self):
        expected = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "colorization"
            / "ddcolor"
            / "pytorch_model.pt"
        )

        resolved = get_ddcolor_model_path()

        self.assertEqual(resolved, expected)

    def test_get_ddcolor_model_path_prefers_env_override(self):
        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "env-ddcolor.pt"
            model_path.write_bytes(b"weights")

            with patch.dict(os.environ, {"DDCOLOR_MODEL_PATH": str(model_path)}, clear=False):
                resolved = get_ddcolor_model_path()

        self.assertEqual(resolved, model_path)

    def test_load_ddcolor_backend_raises_when_weight_missing(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(DDColorModelNotFoundError):
                load_ddcolor_backend(model_path=Path(tmp_dir) / "missing.pt")

    def test_load_ddcolor_backend_caches_model_instances(self):
        fake_backend = object()

        with TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "demo.pt"
            model_path.write_bytes(b"weights")

            with patch("gui_pyside.ddcolor_backend._build_backend", return_value=fake_backend) as builder:
                model_a = load_ddcolor_backend(model_path=model_path, force_reload=True)
                model_b = load_ddcolor_backend(model_path=model_path)

        self.assertIs(model_a, model_b)
        builder.assert_called_once()


if __name__ == "__main__":
    unittest.main()
