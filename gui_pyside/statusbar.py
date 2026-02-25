"""状态栏：状态文字 + 计时器 + 进度条。"""
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QProgressBar

from .styles import UI_COLOR_PRIMARY, UI_COLOR_TEXT_MUTED


class StatusBarWidget(QWidget):
    """底部状态栏，包含状态文字、进度条和计时器。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("muted")
        self.status_label.setMinimumWidth(300)

        self.elapsed_label = QLabel("Elapsed: --")
        self.elapsed_label.setObjectName("muted")

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setFixedWidth(300)
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)

        layout.addWidget(self.status_label, stretch=1)
        layout.addWidget(self.elapsed_label, stretch=0)
        layout.addWidget(self.progress_bar, stretch=0)

        # Timer for elapsed display
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(100)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._start_time: float | None = None

        # Timer for smooth progress animation
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(80)
        self._progress_timer.timeout.connect(self._animate_progress)
        self._progress_target = 0.0
        self._is_processing = False

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)

    def set_progress(self, value: float) -> None:
        """Set progress target (0.0 ~ 1.0). Animation will smooth toward it."""
        self._progress_target = max(0.0, min(1.0, value))
        if value >= 1.0:
            self.progress_bar.setValue(1000)

    def start_timer(self) -> None:
        self._start_time = time.perf_counter()
        self._is_processing = True
        self._elapsed_timer.start()
        self._progress_timer.start()

    def stop_timer(self) -> None:
        self._is_processing = False
        self._elapsed_timer.stop()
        self._progress_timer.stop()
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")

    def reset(self) -> None:
        self.set_status("Ready")
        self.progress_bar.setValue(0)
        self._progress_target = 0.0
        self.elapsed_label.setText("Elapsed: --")
        self._start_time = None

    def _update_elapsed(self) -> None:
        if self._start_time is not None:
            elapsed = time.perf_counter() - self._start_time
            self.elapsed_label.setText(f"Elapsed: {elapsed:.1f}s")

    def _animate_progress(self) -> None:
        current = self.progress_bar.value() / 1000.0
        target = self._progress_target
        if current < target:
            gap = target - current
            increment = max(0.005, gap * 0.2)
            new_val = min(current + increment, target)
            self.progress_bar.setValue(int(new_val * 1000))
        elif current >= target and target < 0.95 and self._is_processing:
            remaining = 0.98 - current
            if remaining > 0.002:
                step = max(0.0015, remaining * 0.012)
                self.progress_bar.setValue(int(min(0.98, current + step) * 1000))
