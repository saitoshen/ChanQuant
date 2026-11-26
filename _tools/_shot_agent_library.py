# -*- coding: utf-8 -*-
"""实拍 ChanAgent 因子库 / 策略库两个页签，检查新加的导入导出按钮是否被挤。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.pop("QT_QPA_PLATFORM", None)

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication, QPushButton

from gui.agent_page import AgentPage
from gui.app import STYLE_SHEET

HERE = os.path.dirname(os.path.abspath(__file__))
WANT = ("导出自定义因子", "导入因子", "导出自定义策略", "导入策略")


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setStyleSheet(STYLE_SHEET)
    app.setFont(QFont("Microsoft YaHei", 9))

    w = AgentPage()
    w.setAttribute(Qt.WA_DontShowOnScreen, True)
    w.resize(1400, 900)
    w.show()
    app.processEvents()

    for text in WANT:
        hits = [b for b in w.findChildren(QPushButton) if b.text() == text]
        if not hits:
            print(f"  [缺失] {text}")
            continue
        b = hits[0]
        g = b.geometry()
        print(f"  {text:<12} x={g.x():>4} w={g.width():>3} "
              f"min={b.minimumSizeHint().width():>3}")

    for i, name in enumerate(("factor", "strategy")):
        w.tabs.setCurrentIndex(i)
        app.processEvents()
        out = os.path.join(HERE, f"_shot_agent_{name}.png")
        w.grab().save(out)
        print(f"  已存 {out}")
    w.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
