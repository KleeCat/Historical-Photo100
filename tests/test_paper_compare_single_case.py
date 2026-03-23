import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from paper_compare_single_case import (
    align_to_reference,
    bicubic_upscale_to_size,
    build_method_result,
    build_output_file_map,
    clip_focus_box,
    compute_psnr_ssim,
    center_crop_to_match,
    draw_focus_boxes,
    extract_focus_crop,
    fit_image_into_box,
    format_figure_title,
    format_metric_lines,
    make_input_display_image,
    normalize_esrgan_state_dict_keys,
    render_comparison_figure,
    validate_inputs,
)


class TestPaperCompareSingleCase(unittest.TestCase):
    def test_bicubic_upscale_to_size_matches_target_shape(self):
        img = np.zeros((4, 5, 3), dtype=np.uint8)
        out = bicubic_upscale_to_size(img, (17, 19))
        self.assertEqual(out.shape, (17, 19, 3))

    def test_center_crop_to_match_returns_hr_shape(self):
        img = np.zeros((12, 14, 3), dtype=np.uint8)
        out = center_crop_to_match(img, (8, 10))
        self.assertEqual(out.shape, (8, 10, 3))

    def test_format_metric_lines_formats_psnr_and_ssim(self):
        self.assertEqual(
            format_metric_lines(28.95, 0.8923),
            [
                "PSNR: 28.95 dB",
                "SSIM: 0.8923",
            ],
        )

    def test_format_figure_title_uses_multiplication_sign(self):
        self.assertEqual(format_figure_title("Bicubic x4"), "Bicubic ×4")
        self.assertEqual(format_figure_title("Real-ESRGAN x4"), "Real-ESRGAN ×4")
        self.assertEqual(format_figure_title("Input"), "Input")

    def test_validate_inputs_accepts_existing_lr_hr_pair(self):
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            lr_path = tmp_path / "sample_lr.png"
            hr_path = tmp_path / "sample_hr.png"
            lr_path.write_bytes(b"lr")
            hr_path.write_bytes(b"hr")

            lr_result, hr_result = validate_inputs(lr_path, hr_path)

            self.assertEqual(lr_result, lr_path.resolve())
            self.assertEqual(hr_result, hr_path.resolve())

    def test_compute_psnr_ssim_returns_two_numeric_fields(self):
        hr = np.full((12, 12, 3), 128, dtype=np.uint8)
        sr = hr.copy()
        sr[0, 0] = 120

        result = compute_psnr_ssim(sr, hr)

        self.assertIn("psnr", result)
        self.assertIn("ssim", result)
        self.assertIsInstance(result["psnr"], float)
        self.assertIsInstance(result["ssim"], float)

    def test_build_method_result_returns_unified_structure(self):
        hr = np.full((10, 10, 3), 128, dtype=np.uint8)
        sr = hr.copy()
        sr[0, 0] = 120

        result = build_method_result("Bicubic", sr, hr)

        self.assertEqual(result["name"], "Bicubic")
        self.assertEqual(result["image"].shape, sr.shape)
        self.assertIn("psnr", result)
        self.assertIn("ssim", result)

    def test_align_to_reference_center_crops_larger_image(self):
        hr = np.zeros((12, 12, 3), dtype=np.uint8)
        sr = np.zeros((14, 14, 3), dtype=np.uint8)

        result = align_to_reference(sr, hr)

        self.assertEqual(result.shape, hr.shape)

    def test_build_output_file_map_returns_expected_filenames(self):
        output_map = build_output_file_map(Path("D:/demo/output"))

        self.assertEqual(output_map["input"].name, "input.png")
        self.assertEqual(output_map["bicubic"].name, "bicubic_x4.png")
        self.assertEqual(output_map["srcnn"].name, "srcnn_x4.png")
        self.assertEqual(output_map["esrgan"].name, "esrgan_x4.png")
        self.assertEqual(output_map["realesrgan"].name, "realesrgan_x4.png")
        self.assertEqual(output_map["metrics_json"].name, "metrics.json")
        self.assertEqual(output_map["metrics_txt"].name, "metrics.txt")
        self.assertEqual(output_map["comparison"].name, "comparison_with_metrics.png")

    def test_make_input_display_image_matches_reference_shape(self):
        lr = np.zeros((4, 5, 3), dtype=np.uint8)
        hr = np.zeros((16, 20, 3), dtype=np.uint8)

        display = make_input_display_image(lr, hr)

        self.assertEqual(display.shape, hr.shape)

    def test_normalize_esrgan_state_dict_keys_maps_old_basicsr_names(self):
        old_state = {
            "model.0.weight": "conv_first",
            "model.1.sub.0.RDB1.conv1.0.weight": "rdb_conv",
            "model.1.sub.23.weight": "trunk_conv",
            "model.3.weight": "upconv1",
            "model.6.weight": "upconv2",
            "model.8.weight": "hrconv",
            "model.10.weight": "conv_last",
        }

        normalized = normalize_esrgan_state_dict_keys(old_state)

        self.assertEqual(normalized["conv_first.weight"], "conv_first")
        self.assertEqual(
            normalized["RRDB_trunk.0.RDB1.conv1.weight"],
            "rdb_conv",
        )
        self.assertEqual(normalized["trunk_conv.weight"], "trunk_conv")
        self.assertEqual(normalized["upconv1.weight"], "upconv1")
        self.assertEqual(normalized["upconv2.weight"], "upconv2")
        self.assertEqual(normalized["HRconv.weight"], "hrconv")
        self.assertEqual(normalized["conv_last.weight"], "conv_last")

    def test_fit_image_into_box_returns_fixed_shape_for_portrait_and_landscape(self):
        portrait = np.zeros((20, 10, 3), dtype=np.uint8)
        landscape = np.zeros((10, 20, 3), dtype=np.uint8)

        portrait_box = fit_image_into_box(portrait, (12, 12))
        landscape_box = fit_image_into_box(landscape, (12, 12))

        self.assertEqual(portrait_box.shape, (12, 12, 3))
        self.assertEqual(landscape_box.shape, (12, 12, 3))
        self.assertTrue(np.all(portrait_box[0, 0] == 255))
        self.assertTrue(np.all(landscape_box[0, 0] == 255))

    def test_clip_focus_box_clamps_to_image_bounds(self):
        clipped = clip_focus_box((8, 7, 10, 10), (12, 14, 3))

        self.assertEqual(clipped, (8, 7, 6, 5))

    def test_extract_focus_crop_returns_expected_region(self):
        image = np.arange(12 * 10 * 3, dtype=np.uint8).reshape(10, 12, 3)

        crop = extract_focus_crop(image, (2, 3, 4, 5))

        self.assertEqual(crop.shape, (5, 4, 3))
        np.testing.assert_array_equal(crop[0, 0], image[3, 2])
        np.testing.assert_array_equal(crop[-1, -1], image[7, 5])

    def test_draw_focus_boxes_paints_red_borders(self):
        image = np.zeros((30, 30, 3), dtype=np.uint8)

        outlined = draw_focus_boxes(image, [(5, 6, 10, 8)], thickness=1)

        np.testing.assert_array_equal(outlined[6, 5], np.array([0, 0, 255], dtype=np.uint8))
        np.testing.assert_array_equal(outlined[13, 14], np.array([0, 0, 255], dtype=np.uint8))

    def test_render_comparison_figure_outputs_multi_row_layout_without_metrics_text(self):
        input_display = np.full((80, 60, 3), 180, dtype=np.uint8)
        method_results = [
            {
                "name": "Bicubic x4",
                "image": np.full((80, 60, 3), 120, dtype=np.uint8),
                "psnr": 30.0,
                "ssim": 0.9,
            },
            {
                "name": "SRCNN x4",
                "image": np.full((80, 60, 3), 100, dtype=np.uint8),
                "psnr": 31.0,
                "ssim": 0.91,
            },
            {
                "name": "ESRGAN x4",
                "image": np.full((80, 60, 3), 90, dtype=np.uint8),
                "psnr": 29.0,
                "ssim": 0.88,
            },
            {
                "name": "Real-ESRGAN x4",
                "image": np.full((80, 60, 3), 80, dtype=np.uint8),
                "psnr": 30.5,
                "ssim": 0.89,
            },
        ]
        focus_boxes = [
            (8, 8, 12, 12),
            (20, 18, 14, 14),
            (16, 42, 18, 16),
        ]

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "comparison.png"

            with patch(
                "paper_compare_single_case.format_metric_lines",
                side_effect=AssertionError("render should not call metric formatter"),
            ):
                render_comparison_figure(
                    input_display_bgr=input_display,
                    method_results=method_results,
                    output_path=output_path,
                    display_max_height=80,
                    focus_boxes=focus_boxes,
                )

            rendered = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(rendered)
            self.assertGreater(rendered.shape[0], 500)
            self.assertGreater(rendered.shape[1], 1200)

    def test_render_comparison_figure_only_draws_focus_boxes_on_input_column(self):
        input_display = np.full((80, 60, 3), 180, dtype=np.uint8)
        method_results = [
            {
                "name": "Bicubic x4",
                "image": np.full((80, 60, 3), 120, dtype=np.uint8),
                "psnr": 30.0,
                "ssim": 0.9,
            },
            {
                "name": "SRCNN x4",
                "image": np.full((80, 60, 3), 100, dtype=np.uint8),
                "psnr": 31.0,
                "ssim": 0.91,
            },
            {
                "name": "ESRGAN x4",
                "image": np.full((80, 60, 3), 90, dtype=np.uint8),
                "psnr": 29.0,
                "ssim": 0.88,
            },
            {
                "name": "Real-ESRGAN x4",
                "image": np.full((80, 60, 3), 80, dtype=np.uint8),
                "psnr": 30.5,
                "ssim": 0.89,
            },
        ]
        focus_boxes = [
            (8, 8, 12, 12),
            (20, 18, 14, 14),
            (16, 42, 18, 16),
        ]

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "comparison.png"

            with patch(
                "paper_compare_single_case.draw_focus_boxes",
                side_effect=lambda image, *args, **kwargs: image,
            ) as draw_boxes:
                render_comparison_figure(
                    input_display_bgr=input_display,
                    method_results=method_results,
                    output_path=output_path,
                    display_max_height=80,
                    focus_boxes=focus_boxes,
                )

            self.assertEqual(draw_boxes.call_count, 1)


if __name__ == "__main__":
    unittest.main()
