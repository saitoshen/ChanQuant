# -*- coding: utf-8 -*-
"""独立的行情源工具条：选源 + 连接 / 断开 + 状态。

## 为什么要独立

以前「行情源」是**跟着账户选择走**的（`gui/trade_page.py` 里 `cb_feed` 按勾选
账户所属柜台生成选项，多选时才显示）。那套有两处讲不通：

1. 行情和账户是两件事，**账户登出不该让行情跟着断**；
2. 策略 7×24 跑，靠的是一条常连的行情 —— 它必须能在没有任何账户登录时也在。

现在这里只做一件事：把 `ttrade.market_feed.MarketFeed` 的选择 / 连接状态摆到
界面上，和账户下拉框**互不影响**。

## 「有无某个源」

选项只列**账户表里有对应账户**的源（天勤要一个天勤模拟账户、CTP 要一个
SimNow 第一套的账户、融航要一个融航账户）。没有账户的源不出现 —— 列出来也
连不上，用户点了只会等一轮超时。
"""

from __future__ import annotations

import threading

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QPushButton,
                               QWidget)

from ttrade.market_feed import MarketFeed, SOURCE_KEYS, SOURCE_LABELS

# 主题色：这几个常量在本项目里是**逐页各自定义**的（`trade_page` / `portfolio_page`
# 各一份），没有公共 theme 模块。这里照抄一份，别去 import 那两个页面 —— 会成环。
C_OK = "#16a34a"
C_WARN = "#b45309"
C_IDLE = "#94a3b8"
C_MUTED = "#64748b"
C_TEXT = "#0f172a"


class FeedBar(QWidget):
    """行情源：一个下拉框 + 一个连接/断开按钮 + 状态。

    `changed` 在**用户主动切源**时发（值是要切到的源 key），由调用方决定要不要
    立刻连。状态刷新走内部定时器，不用调用方管。
    """

    changed = Signal(str)
    #: 后台连接完成（True/False）。**连接不能放在界面线程**：天勤登录 3~4 秒、
    #: CTP 要等交易 / 行情两条通道（各 15 / 10 秒超时），同步做会让整个窗口
    #: 僵住 —— 用户点的「连接」，卡的是整个界面
    _connected = Signal(bool)

    def __init__(self, parent=None, show_label: bool = True) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        if show_label:
            cap = QLabel("行情源")
            cap.setObjectName("Hint")
            lay.addWidget(cap)
        self.cb = QComboBox()
        self.cb.setMinimumWidth(150)
        self.cb.setToolTip(
            "所有账户 / 策略共用的行情连接，和账户登录互不影响。\n"
            "天勤是默认源；CTP（SimNow 第一套）和融航是备用。")
        lay.addWidget(self.cb)
        self.btn = QPushButton("连接")
        self.btn.setCheckable(True)
        self.btn.clicked.connect(self._on_click)
        lay.addWidget(self.btn)
        self.dot = QLabel("●")
        self.dot.setStyleSheet(f"color: {C_IDLE}; font-size: 14px;")
        lay.addWidget(self.dot)
        self.lbl = QLabel("未连接")
        self.lbl.setObjectName("Hint")
        lay.addWidget(self.lbl)
        self._busy = False
        self._err = ""
        self._connected.connect(self._on_connected)
        self._reload()
        self.cb.currentIndexChanged.connect(self._on_pick)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(2000)

    # ---- 源清单 ----

    def _reload(self) -> None:
        """按「账户表里有谁的账户」重建选项。当前源一定保留。"""
        feed = MarketFeed.peek()
        try:
            rows = feed.sources() if feed is not None else self._rows_no_feed()
        except Exception:                        # noqa: BLE001
            rows = self._rows_no_feed()
        keep = self.cb.currentData()
        self.cb.blockSignals(True)
        self.cb.clear()
        for r in rows:
            self.cb.addItem(r["label"], r["key"])
            if r.get("error"):
                self.cb.setItemData(self.cb.count() - 1, r["error"],
                                    Qt.ToolTipRole)
        keys = [r["key"] for r in rows]
        want = keep if keep in keys else self._current_key(keys)
        if want in keys:
            self.cb.setCurrentIndex(keys.index(want))
        self.cb.setEnabled(bool(rows))
        self.cb.blockSignals(False)
        if not rows:
            self.lbl.setText("账户表里没有可用的行情账户")
            self.btn.setEnabled(False)
        else:
            self.btn.setEnabled(True)
            self.chan_changed(keys)
        self.refresh()

    def _rows_no_feed(self) -> list[dict]:
        """`MarketFeed` 还没建过时也得能显示选项 —— 别让控件自己把连接带起来。"""
        from ttrade.market_feed import _accounts_of

        out = []
        for key in SOURCE_KEYS:
            names = [s.name for s in _accounts_of(key)]
            if names:
                out.append({"key": key, "label": SOURCE_LABELS[key],
                            "account": names[0], "accounts": names,
                            "ready": False, "active": False, "error": ""})
        return out

    @staticmethod
    def _current_key(keys: list) -> str:
        from ttrade.market_feed import load_settings

        want = str(load_settings().get("source") or "")
        return want if want in keys else (keys[0] if keys else "")

    def chan_changed(self, keys_hint=None) -> None:
        """给外部一个钩子（子类 / 组合页用它联动账户清单）。"""
        return None

    # ---- 交互 ----

    def _on_pick(self, _idx: int) -> None:
        key = self.cb.currentData() or ""
        if not key:
            return
        self.changed.emit(key)

    def _on_click(self) -> None:
        feed = MarketFeed.peek()
        key = self.cb.currentData() or ""
        if feed is not None and feed.ready and feed.source == key:
            feed.chan(key).close()               # 只断这一条，不影响别人
            self.refresh(force=True)
            return
        self._connect(key)

    def _connect(self, key: str) -> None:
        if self._busy:
            return                       # 已经在连了，别叠第二次登录
        self._busy = True
        self.lbl.setText("连接中…")
        self.dot.setStyleSheet(f"color: {C_WARN}; font-size: 14px;")
        self.btn.setEnabled(False)

        def _work() -> None:
            ok = False
            try:
                feed = MarketFeed.get()              # 首次 get 会建连接
                ok = feed.select(key) if feed.source != key else feed.retry()
            except Exception as exc:                 # noqa: BLE001
                self._err = f"{type(exc).__name__}: {exc}"
            self._connected.emit(bool(ok))

        threading.Thread(target=_work, daemon=True, name="feed-connect").start()

    def _on_connected(self, ok: bool) -> None:
        self._busy = False
        self.btn.setEnabled(True)
        self.refresh(force=True)
        if ok:
            self.changed.emit(self.cb.currentData() or "")

    # ---- 状态 ----

    def refresh(self, force: bool = False) -> None:
        feed = MarketFeed.peek()
        key = self.cb.currentData() or ""
        if feed is None:
            self.dot.setStyleSheet(f"color: {C_IDLE}; font-size: 14px;")
            self.lbl.setText("未连接")
            self.btn.setChecked(False)
            self.btn.setText("连接")
            return
        st = feed.stats() if feed.source == key else {}
        ready = bool(st.get("ready")) and feed.source == key
        err = str(st.get("error") or "")
        self.btn.setChecked(ready)
        self.btn.setText("断开" if ready else "连接")
        if ready:
            n_k, n_q = int(st.get("klines") or 0), int(st.get("quotes") or 0)
            self.dot.setStyleSheet(f"color: {C_OK}; font-size: 14px;")
            self.lbl.setText(f"已连接 · {n_k} 个品种")
            self.lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
            self.lbl.setToolTip(f"K 线 {n_k} 条订阅，盘口 {n_q} 个")
        elif err:
            self.dot.setStyleSheet(f"color: {C_WARN}; font-size: 14px;")
            short = err if len(err) <= 40 else err[:40] + "…"
            self.lbl.setText(short)
            self.lbl.setStyleSheet(f"color: {C_WARN}; font-size: 11px;")
            self.lbl.setToolTip(err)
        else:
            self.dot.setStyleSheet(f"color: {C_IDLE}; font-size: 14px;")
            self.lbl.setText("未连接")
            self.lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
            self.lbl.setToolTip("")

    def current_source(self) -> str:
        return self.cb.currentData() or ""
