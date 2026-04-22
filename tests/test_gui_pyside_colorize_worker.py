import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PySide6.QtCore import QCoreApplication


def _install_worker_dependency_stubs():
    processing_stub = types.ModuleType("gui_pyside.processing")

    class UserCancelledError(Exception):
        pass

    processing_stub.UserCancelledError = UserCancelledError
    processing_stub.auto_tile_size = lambda *args, **kwargs: 0
    processing_stub.blend_images = lambda *args, **kwargs: None
    processing_stub.blend_with_lr = lambda *args, **kwargs: None
    processing_stub.apply_unsharp_mask = lambda *args, **kwargs: None
    processing_stub.apply_film_grain = lambda *args, **kwargs: None
    processing_stub.suppress_edge_ringing = lambda *args, **kwargs: None
    processing_stub.clamp_value = lambda value, *_args, **_kwargs: value
    processing_stub.estimate_image_metrics = lambda *_args, **_kwargs: {}
    processing_stub.TEXTURE_ENABLED = False
    processing_stub.TEXTURE_MODEL_ID = ""
    processing_stub.IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    models_stub = types.ModuleType("gui_pyside.models")

    class ModelManager:  # pragma: no cover - stub for import only
        pass

    models_stub.ModelManager = ModelManager

    sys.modules["gui_pyside.processing"] = processing_stub
    sys.modules["gui_pyside.models"] = models_stub
    sys.modules.pop("gui_pyside.workers", None)
    return importlib.import_module("gui_pyside.workers"), UserCancelledError


workers_module, UserCancelledError = _install_worker_dependency_stubs()
ColorizeWorker = workers_module.ColorizeWorker


class TestColorizeWorker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_worker_emits_success_and_image(self):
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        fake_result = SimpleNamespace(
            output_image=image,
            output_path="outputs/demo_colorized.png",
            run_dir="outputs/demo",
            run_meta={},
            elapsed=0.5,
        )
        progress_events = []
        image_events = []
        finished_events = []

        with patch("gui_pyside.workers.run_colorization_pipeline", return_value=fake_result):
            worker = ColorizeWorker(
                img_input=np.zeros((8, 8, 3), dtype=np.uint8),
                input_path="demo.png",
                output_base_dir="outputs",
            )
            worker.progress.connect(lambda value, text: progress_events.append((value, text)))
            worker.image_ready.connect(lambda output: image_events.append(output))
            worker.finished.connect(lambda ok, message: finished_events.append((ok, message)))

            worker.run()

        self.assertTrue(progress_events)
        self.assertEqual(len(image_events), 1)
        self.assertEqual(image_events[0].shape, (8, 8, 3))
        self.assertTrue(finished_events[0][0])
        self.assertEqual(worker.output_path, "outputs/demo_colorized.png")
        self.assertEqual(worker.run_dir, "outputs/demo")

    def test_worker_emits_failure_message(self):
        finished_events = []

        with patch("gui_pyside.workers.run_colorization_pipeline", side_effect=RuntimeError("boom")):
            worker = ColorizeWorker(
                img_input=np.zeros((8, 8, 3), dtype=np.uint8),
                input_path="demo.png",
            )
            worker.finished.connect(lambda ok, message: finished_events.append((ok, message)))

            worker.run()

        self.assertEqual(finished_events, [(False, "boom")])

    def test_worker_honors_interruption(self):
        finished_events = []

        with patch(
            "gui_pyside.workers.run_colorization_pipeline",
            side_effect=UserCancelledError("Cancelled"),
        ):
            worker = ColorizeWorker(
                img_input=np.zeros((8, 8, 3), dtype=np.uint8),
                input_path="demo.png",
            )
            worker.finished.connect(lambda ok, message: finished_events.append((ok, message)))

            worker.run()

        self.assertEqual(finished_events, [(False, "Colorization cancelled")])


if __name__ == "__main__":
    unittest.main()
