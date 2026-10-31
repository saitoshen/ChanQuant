"""GUI 冒烟测试：离屏渲染主窗口并截图，用于检查布局与图表绘制。"""

import os
import sys
import time

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
    # 首屏现在是 ChanAgent；这几张截图要看的是研究页，先切过去
    win._switch_page(0)

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
        v._pending_draw.clear()
        win._on_live_bars(live)
        if v._pending_draw:
            fails.append("停在其他页时实时推送仍在动 K 线")
        v._render_kline()                        # 直接叫也不许真画
        if not v._pending_draw:
            fails.append("停在其他页时 _render_kline 没有退化成「记一笔」")
        win.pages.setCurrentIndex(0)             # 回到研究页：这一帧要补上
        app.processEvents()
        if v._pending_draw:
            fails.append("回到研究页后没有补画 K 线")
        win._on_live_bars(live)
        if v._pending_draw:
            fails.append("回到研究页后实时推送没有重画 K 线")

        # 「改参数自动回测」出厂不勾，勾上要先过一道提示框。这里把提示换成
        # 一个固定答复，验的是**接线**：说「还是手动」就该把勾退回去
        if win.chk_auto.isChecked():
            fails.append("「改参数自动回测」出厂是勾上的")
        win._confirm_auto_run = lambda: False
        win.chk_auto.setChecked(True)
        app.processEvents()
        if win.chk_auto.isChecked():
            fails.append("提示框里选了「还是手动」，勾没退回去")
        win._confirm_auto_run = lambda: True
        win.chk_auto.setChecked(True)
        app.processEvents()
        if not win.chk_auto.isChecked():
            fails.append("提示框里选了「开启」，勾没打上")
        win.chk_auto.setChecked(False)
        app.processEvents()

        # 多品种回测整批进后台：验结果真的画出来、按钮真的露出来又收回去、
        # 线程真的退出了（不退出的话关窗会崩在 QThread 上）
        period = win._period()
        keys = [it["key"] for it in win.store.items
                if win.store.get(it["key"], period) is not None
                and not win.store.get(it["key"], period).empty][:2]
        if len(keys) < 2:
            fails.append(f"多品种后台回测：{period} 上凑不出 2 个品种，跳过")
        else:
            win.sym_pick.set_items(win.store.items, keep=keys)
            win._on_pick_changed()
            win._run_multi(keys)
            app.processEvents()
            if win.btn_stop.isHidden():
                fails.append("多品种回测起手后没有露出「停止回测」")
            deadline = time.time() + 120
            while win._m_thread is not None and time.time() < deadline:
                app.processEvents()
                time.sleep(0.02)
            app.processEvents()
            if win._m_thread is not None:
                fails.append("多品种回测 120 秒没跑完")
            elif win.res is None or win.res.label != "多品种等权":
                fails.append(f"多品种回测没画出结果：{getattr(win.res, 'label', None)}")
            if not win.btn_stop.isHidden():
                fails.append("多品种回测跑完后「停止回测」还留着")
            win.sym_pick.set_items(win.store.items, keep=[])
            win._on_pick_changed()
            app.processEvents()

        # 合约层：选具体合约只画 K 线（不跑回测），回主连要把结果重新长出来。
        # 这一层是「品种 / 合约」分开的落点：品种是主连、可回测，合约有到期日、
        # 曲线是断的 —— 拿它跑回测年化会被几个月涨幅放大成天文数字
        pk = win.contract_pick
        if not pk.isVisible():
            fails.append("期货市场下合约下拉是藏着的")
        pk.cb_mode.setCurrentIndex(1)
        app.processEvents()
        if pk.cb_month.count() == 0:
            print("skip 合约层：当前周期没有真实合约数据")
        else:
            if win.res is not None:
                fails.append("选了具体合约还在出回测结果")
            if win.view.df is None or not win.view.bars_only:
                fails.append("选了具体合约没切到「只画 K 线」")
            if win.view._k_total == 0:
                fails.append("具体合约 K 线一根都没画")
            if pk.contract() in ("", None):
                fails.append("具体合约下拉有内容但取不到代码")
            pk.cb_mode.setCurrentIndex(0)
            app.processEvents()
            if win.res is None:
                fails.append("回到主连没有恢复回测结果")
            if win.view.bars_only:
                fails.append("回到主连还停在「只画 K 线」")

        print("saved screenshots")
        for f in fails:
            print("FAIL", f)
        app.quit()

    QTimer.singleShot(1200, shot)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
