import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

from gui_pyside.colorization import (
    ColorizationModelNotFoundError,
    get_colorization_model_path,
    prepare_colorization_input,
    run_colorization_pipeline,
)


class TestColorizationPipeline(unittest.TestCase):
    def test_get_colorization_model_path_prefers_env_override(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "env-color-model.caffemodel"
            model_path.write_bytes(b"model")

            with patch.dict(os.environ, {"COLORIZATION_MODEL_PATH": str(model_path)}, clear=False):
                resolved = get_colorization_model_path()

            self.assertEqual(resolved, model_path)

    def test_prepare_colorization_input_converts_gray_to_bgr(self):
        gray = np.full((12, 10), 128, dtype=np.uint8)

        prepared, original_shape = prepare_colorization_input(gray, max_side=256)

        self.assertEqual(prepared.shape, (12, 10, 3))
        self.assertEqual(original_shape, (12, 10))

    def test_prepare_colorization_input_resizes_large_image_by_longest_side(self):
        image = np.zeros((400, 200, 3), dtype=np.uint8)

        prepared, original_shape = prepare_colorization_input(image, max_side=100)

        self.assertEqual(prepared.shape[:2], (100, 50))
        self.assertEqual(original_shape, (400, 200))

    def test_run_colorization_pipeline_raises_when_model_missing(self):
        with TemporaryDirectory() as tmp_dir:
            with self.assertRaises(ColorizationModelNotFoundError):
                run_colorization_pipeline(
                    input_img=np.zeros((8, 8, 3), dtype=np.uint8),
                    input_path="demo.png",
                    output_base_dir=tmp_dir,
                    backend=Mock(),
                    model_path=Path(tmp_dir) / "missing.caffemodel",
                )

    def test_run_colorization_pipeline_saves_image_and_metadata(self):
        fake_backend = Mock(return_value=np.full((16, 16, 3), 180, dtype=np.uint8))

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            fake_model = tmp_path / "colorization_release_v2.caffemodel"
            fake_model.write_bytes(b"model")

            result = run_colorization_pipeline(
                input_img=np.zeros((16, 16, 3), dtype=np.uint8),
                input_path="portrait.png",
                output_base_dir=tmp_dir,
                backend=fake_backend,
                model_path=fake_model,
            )

            self.assertTrue(Path(result.output_path).is_file())
            self.assertEqual(result.output_image.shape, (16, 16, 3))
            self.assertIn("portrait_colorized", Path(result.output_path).name)
            self.assertEqual(Path(result.run_dir).parent, tmp_path)
            self.assertIn("model_path", result.run_meta)
            self.assertTrue((Path(result.run_dir) / "colorization_run.json").is_file())


if __name__ == "__main__":
    unittest.main()
