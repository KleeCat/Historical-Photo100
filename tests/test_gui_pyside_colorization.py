import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import cv2
import numpy as np

from gui_pyside.colorization import (
    ColorizationResult,
    postprocess_colorized_output,
    prepare_colorization_input,
    run_colorization_pipeline,
)
from gui_pyside.ddcolor_backend import DDColorModelNotFoundError


class TestColorizationPipeline(unittest.TestCase):
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

    def test_postprocess_colorized_output_reduces_green_cast(self):
        green_cast = np.full((10, 10, 3), [90, 210, 100], dtype=np.uint8)

        processed = postprocess_colorized_output(green_cast)

        self.assertEqual(processed.shape, green_cast.shape)
        self.assertEqual(processed.dtype, np.uint8)
        self.assertLess(processed[:, :, 1].mean(), green_cast[:, :, 1].mean())

    def test_postprocess_colorized_output_reduces_background_magenta_cast(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[:6, :, :] = [180, 120, 220]
        image[6:, :, :] = [170, 190, 220]

        processed = postprocess_colorized_output(image)

        background_before = int(np.ptp(image[0, 0]))
        background_after = int(np.ptp(processed[0, 0]))

        self.assertLess(background_after, background_before - 8)

    def test_postprocess_colorized_output_keeps_skin_regions_warm(self):
        image = np.zeros((12, 12, 3), dtype=np.uint8)
        image[:6, :, :] = [180, 120, 220]
        image[6:, :, :] = [170, 190, 220]

        processed = postprocess_colorized_output(image)

        skin_before_warmth = int(image[8, 6, 2]) - int(image[8, 6, 0])
        skin_after_warmth = int(processed[8, 6, 2]) - int(processed[8, 6, 0])

        self.assertGreaterEqual(skin_after_warmth, skin_before_warmth)

    def test_postprocess_colorized_output_suppresses_background_more_than_center_subject(self):
        image = np.full((40, 30, 3), [190, 120, 220], dtype=np.uint8)
        image[8:36, 8:24, :] = [210, 185, 230]

        processed = postprocess_colorized_output(image)

        def pixel_saturation(pixel: np.ndarray) -> int:
            hsv = cv2.cvtColor(np.uint8([[pixel]]), cv2.COLOR_BGR2HSV)
            return int(hsv[0, 0, 1])

        background_before = pixel_saturation(image[2, 2])
        background_after = pixel_saturation(processed[2, 2])
        subject_before = pixel_saturation(image[20, 15])
        subject_after = pixel_saturation(processed[20, 15])

        background_ratio = background_after / max(background_before, 1)
        subject_ratio = subject_after / max(subject_before, 1)

        self.assertLess(background_ratio, subject_ratio - 0.12)

    def test_run_colorization_pipeline_raises_when_ddcolor_model_missing(self):
        with patch(
            "gui_pyside.colorization.load_ddcolor_backend",
            side_effect=DDColorModelNotFoundError("missing model"),
        ):
            with self.assertRaises(DDColorModelNotFoundError):
                run_colorization_pipeline(
                    input_img=np.zeros((8, 8, 3), dtype=np.uint8),
                    input_path="demo.png",
                    output_base_dir="outputs",
                )

    def test_run_colorization_pipeline_uses_ddcolor_backend_by_default(self):
        fake_backend = Mock()
        ddcolor_output = np.full((16, 16, 3), [90, 150, 180], dtype=np.uint8)

        with TemporaryDirectory() as tmp_dir:
            with patch("gui_pyside.colorization.load_ddcolor_backend", return_value=fake_backend) as load_backend:
                with patch("gui_pyside.colorization.run_ddcolor_inference", return_value=ddcolor_output) as run_inference:
                    result = run_colorization_pipeline(
                        input_img=np.zeros((16, 16, 3), dtype=np.uint8),
                        input_path="portrait.png",
                        output_base_dir=tmp_dir,
                    )

                    self.assertTrue(Path(result.output_path).is_file())
                    self.assertTrue((Path(result.run_dir) / "colorization_run.json").is_file())

        self.assertIsInstance(result, ColorizationResult)
        load_backend.assert_called_once()
        run_inference.assert_called_once()
        self.assertEqual(result.output_image.shape, (16, 16, 3))
        self.assertEqual(result.run_meta["backend"], "ddcolor")
        self.assertIn("portrait_colorized", Path(result.output_path).name)


if __name__ == "__main__":
    unittest.main()
