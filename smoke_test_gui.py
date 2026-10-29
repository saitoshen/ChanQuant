"""GUI 冒烟测试：离屏渲染主窗口并截图，用于检查布局与图表绘制。"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 研究页不再自动连天勤（连接只在「账户透视和交易」页），所以截图不会去登录账户，
# 不需要额外的开关。
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

    fails: list[str] = []

    def shot2():
        app.processEvents()
        win.grab().save(os.path.join(out_dir, "gui_perf.png"))
        print("symbols:", len(win.store.symbols))
        print("strategy:", win.cb_strategy.currentText())
        print("metrics:", {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in win.res.metrics.items()})

        # 实时刷新每 5 秒重画一次 K 线。重画不能把用户翻到的位置拽回最新 ——
        # 否则想看一眼两天前的信号，图会一直弹回右端
        v = win.view
        total = v._k_total
        v._k_win = 120
        v._k_start = max(0, total - 400)
        before = v._k_start
        v.show_result(v.df, v.raw, v.res, v.bh, v.ctx, keep_window=True)
        if v._k_start != before:
            fails.append(f"实时刷新把窗口挪了：{before} → {v._k_start}")

        # 贴着右端看的时候，新数据来了要跟着最新走
        v._k_win = 120
        v._k_start = max(0, v._k_total - 120)
        v._k_total += 5
        v._draw_kline(keep_window=True)
        if v._k_start != max(0, v._k_total - v._k_win):
            fails.append(f"贴在右端时没跟到最新：{v._k_start} vs "
                         f"{max(0, v._k_total - v._k_win)}")

        # 停在别的页时不该画 K 线：一次重绘是 matplotlib 整图重建（100~230ms），
        # 实时每秒 2.5 帧，主线程被吃掉三成 —— 表现就是「在账户页点下单，
        # 按钮要顿一下才反应」。两道闸各验一次：实时链路整段跳过，以及
        # 就算有人直接叫 `_render_kline` 也只记一笔、切回来再补。
        pair = win._current_pair()
        win._live_key = f"{pair[0]}_{pair[1]}"
        win._live_period = win._period()
        live = v.df.tail(300).copy()
        live["open_oi"] = 0.0
        live["close_oi"] = 0.0
        win._live_last = live.index[-1]          # 当成同一根，只刷最后一根
        win.tabs.setCurrentIndex(0)              # 内层回到「K 线」页签
        win.pages.setCurrentIndex(1)             # 站到「账户透视和交易」页
        app.processEvents()
        v._k_dirty = False
        win._on_live_bars(live)
        if v._k_dirty:
            fails.append("停在其他页时实时推送仍在动 K 线")
        v._render_kline()                        # 直接叫也不许真画
        if not v._k_dirty:
            fails.append("停在其他页时 _render_kline 没有退化成「记一笔」")
        win.pages.setCurrentIndex(0)             # 回到研究页：这一帧要补上
        app.processEvents()
        if v._k_dirty:
            fails.append("回到研究页后没有补画 K 线")
        win._on_live_bars(live)
        if v._k_dirty:
            fails.append("回到研究页后实时推送没有重画 K 线")

        print("saved screenshots")
        for f in fails:
            print("FAIL", f)
        app.quit()

    QTimer.singleShot(1200, shot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
