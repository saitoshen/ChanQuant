"""GUI 冒烟测试：离屏渲染主窗口并截图，用于检查布局与图表绘制。"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from gui.app import MainWindow, setup_light_palette, STYLE_SHEET


def main() -> int:
    app = QApplication(sys.argv)
    setup_light_palette(app)
    app.setStyleSheet(STYLE_SHEET)

    win = MainWindow()
    win.resize(1480, 940)
    win.show()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    def shot():
        app.processEvents()
        win.grab().save(os.path.join(out_dir, "gui_kline.png"))
        win.tabs.setCurrentIndex(1)
        app.processEvents()
        QTimer.singleShot(400, shot2)

    def shot2():
        app.processEvents()
        win.grab().save(os.path.join(out_dir, "gui_perf.png"))
        print("symbols:", len(win.store.symbols))
        print("strategy:", win.cb_strategy.currentText())
        print("metrics:", {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in win.res.metrics.items()})
        print("saved screenshots")
        app.quit()

    QTimer.singleShot(1200, shot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
