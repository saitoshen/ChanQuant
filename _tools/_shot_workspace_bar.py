# -*- coding: utf-8 -*-
"""实拍工作区工具条，检查新增的「导入 / 导出」有没有挤坏两行布局。

    python _tools/_shot_workspace_bar.py

禁 offscreen（那套字体没有中文字形），走真实平台 + WA_DontShowOnScreen。
QSS 必须自己挂：它在 gui/app.py::main() 里才挂到全局。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("QT_QPA_PLATFORM", None)

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QPushButton

from gui.app import STYLE_SHEET
from gui.workspace_bar import WorkspaceBar

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE_SHEET)
    app.setFont(QFont("Microsoft YaHei", 9))

    for width in (1280, 1100):
        w = WorkspaceBar()
        w.setAttribute(Qt.WA_DontShowOnScreen, True)
        w.set_workspaces(["默认组合", "短线策略", "截面试跑"], 0)
        w.set_state(1_434_000.0, "equal", "principal", "split")
        w.resize(width, w.sizeHint().height())
        w.show()
        app.processEvents()

        out = os.path.join(HERE, f"_shot_wsbar_{width}.png")
        w.grab().save(out)
        print(f"[{width}px] 实际 {w.width()} x {w.height()}，理想 {w.sizeHint().width()}px → {out}")
        for b in w.findChildren(QPushButton):
            g = b.geometry()
            print(f"  {b.text():<12} x={g.x():>4} w={g.width():>3} "
                  f"min={b.minimumSizeHint().width():>3}")
        tabs = w.tabs
        print(f"  页签 x={tabs.x()} w={tabs.width()} min={tabs.minimumSizeHint().width()}")
        w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
