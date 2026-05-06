from PyQt6 import QtWidgets, uic
from pathlib import Path

dr = Path(__file__).parent

class DistributionWidget(QtWidgets.QWidget):
    def __init__(self, parent = None):
        super().__init__(parent)
        uic.loadUi(dr / "distribution_widget.ui", self)

        self.distributionCombo.currentIndexChanged.connect(
            self.stackedWidget.setCurrentIndex
        )
