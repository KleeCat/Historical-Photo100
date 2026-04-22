import importlib
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PySide6.QtWidgets import QApplication

from gui_pyside.sidebar import SidebarWidget

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _install_main_dependency_stubs():
    originals = {
        name: sys.modules.get(name)
        for name in (
            "torch",
            "gui_pyside.metrics",
            "gui_pyside.processing",
            "gui_pyside.models",
            "gui_pyside.workers",
            "gui_pyside.main",
        )
    }
    torch_stub = types.ModuleType("torch")
    torch_stub.device = lambda name: name
    torch_stub.cuda = SimpleNamespace(is_available=lambda: False)

    metrics_stub = types.ModuleType("gui_pyside.metrics")
    metrics_stub.calculate_metrics = lambda *_args, **_kwargs: {}

    processing_stub = types.ModuleType("gui_pyside.processing")

    class UserCancelledError(Exception):
        pass

    processing_stub.UserCancelledError = UserCancelledError
    processing_stub.clamp_value = lambda value, *_args, **_kwargs: value
    processing_stub.estimate_image_metrics = lambda *_args, **_kwargs: {}
    processing_stub.make_comparison_images = lambda *_args, **_kwargs: {}
    processing_stub.tensor_to_grid_image = lambda *_args, **_kwargs: None
    processing_stub.save_feature_grids = lambda *_args, **_kwargs: []
    processing_stub.IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp")

    models_stub = types.ModuleType("gui_pyside.models")

    class ModelManager:
        def __init__(self, *_args, **_kwargs):
            self.device = "cpu"
            self.upsampler = object()

        def cleanup(self):
            return None

    models_stub.ModelManager = ModelManager

    workers_stub = types.ModuleType("gui_pyside.workers")

    class _Worker:
        def __init__(self, *_args, **_kwargs):
            self.progress = Mock()
            self.image_ready = Mock()
            self.finished = Mock()

        def start(self):
            return None

        def isRunning(self):
            return False

        def requestInterruption(self):
            return None

        def wait(self, *_args, **_kwargs):
            return None

    workers_stub.ModelLoadWorker = _Worker
    workers_stub.ProcessWorker = _Worker
    workers_stub.BatchWorker = _Worker
    workers_stub.ColorizeWorker = _Worker

    sys.modules["torch"] = torch_stub
    sys.modules["gui_pyside.metrics"] = metrics_stub
    sys.modules["gui_pyside.processing"] = processing_stub
    sys.modules["gui_pyside.models"] = models_stub
    sys.modules["gui_pyside.workers"] = workers_stub
    sys.modules.pop("gui_pyside.main", None)
    main_module = importlib.import_module("gui_pyside.main")

    for name, original in originals.items():
        if name == "gui_pyside.main":
            continue
        if original is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original

    return main_module


main_module = _install_main_dependency_stubs()
MainWindow = main_module.MainWindow


class TestSidebarColorization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_sidebar_exposes_colorize_signal_and_button(self):
        sidebar = SidebarWidget()

        self.assertTrue(hasattr(sidebar, "colorize_clicked"))
        self.assertEqual(sidebar.btn_colorize.text(), "Start Colorization")

    def test_processing_state_disables_colorize_button(self):
        sidebar = SidebarWidget()

        sidebar.set_processing_state(True)

        self.assertFalse(sidebar.btn_colorize.isEnabled())


class TestMainWindowColorization(unittest.TestCase):
    def _make_window_like(self):
        return SimpleNamespace(
            img_input=None,
            input_path=None,
            default_output_dir="outputs",
            _process_worker=None,
            _batch_worker=None,
            _colorize_worker=None,
            sidebar=SimpleNamespace(
                set_processing_state=Mock(),
                set_cancel_state=Mock(),
            ),
            display=SimpleNamespace(show_overlay=Mock()),
            statusbar=SimpleNamespace(
                start_timer=Mock(),
                stop_timer=Mock(),
                set_status=Mock(),
            ),
            _on_process_progress=Mock(),
            _on_process_image_ready=Mock(),
            _on_colorize_finished=Mock(),
        )

    @patch.object(main_module.QMessageBox, "information")
    @patch.object(main_module, "ColorizeWorker")
    def test_start_colorization_requires_input_image(self, worker_cls, info_box):
        window = self._make_window_like()

        MainWindow.start_colorization(window)

        worker_cls.assert_not_called()
        info_box.assert_called_once()

    @patch.object(main_module.QMessageBox, "information")
    @patch.object(main_module, "ColorizeWorker")
    def test_start_colorization_creates_worker_when_image_loaded(self, worker_cls, info_box):
        window = self._make_window_like()
        worker = worker_cls.return_value
        window.img_input = np.zeros((16, 16, 3), dtype=np.uint8)
        window.input_path = "demo.png"

        MainWindow.start_colorization(window)

        worker_cls.assert_called_once()
        window.sidebar.set_processing_state.assert_called_once_with(True)
        window.statusbar.start_timer.assert_called_once()
        worker.start.assert_called_once()
        info_box.assert_not_called()

    def test_cancel_processing_interrupts_colorize_worker(self):
        colorize_worker = SimpleNamespace(
            isRunning=Mock(return_value=True),
            requestInterruption=Mock(),
        )
        window = self._make_window_like()
        window._colorize_worker = colorize_worker

        MainWindow._cancel_processing(window)

        colorize_worker.requestInterruption.assert_called_once()
        window.sidebar.set_cancel_state.assert_called_once()


if __name__ == "__main__":
    unittest.main()
