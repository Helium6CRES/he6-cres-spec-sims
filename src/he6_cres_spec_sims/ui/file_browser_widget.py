from PyQt6 import QtWidgets, uic
from pathlib import Path

dr = Path(__file__).parent

class FileBrowserWidget(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        uic.loadUi(dr / "file_browser_widget.ui", self)

        self.browseButton.clicked.connect(self.browse_file)

    def browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select file",
            "",
            "All Files (*.*)"
        )

        if path:
            self.pathLineEdit.setText(path)
