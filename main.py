import sys
from PySide6 import QtWidgets

from model.oscillo_model import OscilloModel
from view.main_window import OscilloMainWindow
from controller.oscillo_controller import OscilloController

# Determine NI-SCOPE availability once at startup
try:
    import niscope
    NISCOPE_AVAILABLE = True
except Exception:
    NISCOPE_AVAILABLE = False


def main():
    app = QtWidgets.QApplication(sys.argv)
    model = OscilloModel()
    if not NISCOPE_AVAILABLE:
        model.simulate = True

    main_win = OscilloMainWindow(model)
    controller = OscilloController(model, main_win, niscope_available=NISCOPE_AVAILABLE)
    main_win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
