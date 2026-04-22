import os
import unittest

import numpy as np
from PySide6.QtWidgets import QApplication

from gui_pyside.display import ImageDisplayWidget

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class TestImageDisplayOutputMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_display_switches_to_colorization_output_labels(self):
        widget = ImageDisplayWidget()
        image = np.zeros((8, 6, 3), dtype=np.uint8)

        widget.set_output_mode("colorization")
        widget.show_output(image, "portrait_colorized.png")

        self.assertEqual(widget.lbl_output_title.text(), "Colorized Output")
        self.assertEqual(widget.lbl_filename_out.text(), "Colorized: portrait_colorized.png")
        self.assertEqual(widget.lbl_res_out.text(), "Colorized: 6 x 8")


if __name__ == "__main__":
    unittest.main()
