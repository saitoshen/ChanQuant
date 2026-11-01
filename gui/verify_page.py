"""策略验证页：逐根重放 + 前视偏差检测。

## 这一页解决什么问题

回测结果好看，不代表策略能用 —— 最常见的死法是**因子偷看了未来**：写代码时
顺手 `df["close"].shift(-1)`、用整段的均值做标准化、或者把当天收盘后才发布的
基本面数据 join 到当天。这类错误在净值曲线上一片繁荣，实盘一跑就废。

所以这一页做两件事：

1. **前视检测**（`ttrade.verify`）：把当前 bar 之后的数据改掉重算，看这根 bar
   的信号变不变。变了就是偷看了未来，并且能定位到具体哪个因子。
2. **逐根重放**：把回测按时间轴放一遍 —— K 线一根根长出来、信号一个个亮起来、
   净值跟着走。看盘时「这条策略当时到底在干什么」比一张最终净值直观得多。

## 为什么播放不需要重算

重放只是**展示**：`Report.df` 里已经有每一根的信号和净值，播放时每帧只画到
第 i 根。真正需要重算的是第 1 件事，那件事已经一次性做完了（8~12 个检查点）。

## 数据从哪来

取「策略研究」页当前那套配置和行情 —— 用户刚在那边调完参数，切过来就能验。
按钮是显式的：这一页不会自己跟着研究页变，否则正在播到一半会被打断。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.ticker import FuncFormatter
from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui.result_view import (
    C_BG,
    C_BH,
    C_DOWN,
    C_GRID,
    C_LONG_SIG,
    C_MUTED,
    C_SHORT_SIG,
    C_STRAT,
    C_TEXT,
    C_UP,
    MplCanvas,
)
from ttrade import verify as V

# 播放窗口里最多画多少根。120 根日线大约半年，5 分钟线大约两天 ——
# 再多就挤成一团，而且每帧重建图元会明显掉帧
WIN = 120

# 速度档：每帧推进几根 + 帧间隔（毫秒）
SPEEDS = (
    ("1×", 1, 260),
    ("2×", 2, 200),
    ("5×", 5, 140),
    ("10×", 10, 110),
)


class _Worker(QObject):
    """验证跑在后台线程：机器学习模式一轮几十秒，主线程里做就是整窗卡死。"""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, cfg, df, market, checks):
        super().__init__()
        self._args = (cfg, df, market, checks)

    def run(self) -> None:
        cfg, df, market, checks = self._args
        try:
            rep = V.run(cfg, df, market, n_checks=checks)
        except Exception as exc:  # noqa: BLE001 - 报错要显示在页面上
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(rep)


class VerifyPage(QWidget):
    """逐根重放 + 前视检测。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.cfg = None            # SStore.StrategyConfig
        self.df = None             # 原始 K 线
        self.market = ""
        self.report = None         # V.Report
        self._i = 0                # 当前播放到第几根
        self._thread: QThread | None = None
        self._worker: _Worker | None = None

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._build()
        self._reset("还没有取策略。到「策略研究」页配好，再点左上角「取当前策略」。")

    # ---------------------------------------------------------- 构建

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_take = QPushButton("取当前策略")
        self.btn_take.setToolTip("把「策略研究」页现在的配置和行情搬过来")
        self.btn_take.clicked.connect(self._take_from_study)
        self.btn_check = QPushButton("开始验证")
        self.btn_check.setToolTip("扰动当前 bar 之后的数据重算，看信号会不会变")
        self.btn_check.clicked.connect(self._start_check)
        self.btn_check.setEnabled(False)

        self.btn_play = QPushButton("▶ 播放")
        self.btn_play.setCheckable(True)
        self.btn_play.setEnabled(False)
        self.btn_play.toggled.connect(self._on_play_toggled)

        self.cb_speed = QComboBox()
        for label, _n, _ms in SPEEDS:
            self.cb_speed.addItem(label)
        self.cb_speed.setCurrentIndex(1)
        self.cb_speed.currentIndexChanged.connect(self._on_speed)

        self.lbl_prog = QLabel("—")
        self.lbl_prog.setObjectName("Hint")

        for w in (self.btn_take, self.btn_check, self.btn_play):
            bar.addWidget(w)
        bar.addWidget(QLabel("速度"))
        bar.addWidget(self.cb_speed)
        bar.addStretch(1)
        bar.addWidget(self.lbl_prog)
        root.addLayout(bar)

        self.lbl_verdict = QLabel("")
        self.lbl_verdict.setWordWrap(True)
        self.lbl_verdict.setFont(QFont("Microsoft YaHei", 10))
        root.addWidget(self.lbl_verdict)

        split = QSplitter(Qt.Horizontal)
        self.canvas = MplCanvas(figsize=(9, 7), dpi=100)
        split.addWidget(self.canvas)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        sl.addWidget(QLabel("前视检测（检查点）"))
        self.tbl_check = self._table(["时间", "信号", "结论"])
        sl.addWidget(self.tbl_check, 1)
        sl.addWidget(QLabel("因子 · 当前 bar"))
        self.tbl_factor = self._table(["因子", "值", "结论"])
        sl.addWidget(self.tbl_factor, 1)
        split.addWidget(side)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([700, 420])
        root.addWidget(split, 1)

    def _table(self, cols: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setAlternatingRowColors(True)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        return t

    # ---------------------------------------------------------- 取数据

    def _take_from_study(self) -> None:
        """从主窗口拿当前配置 + 行情。找不到就提示，不猜。"""
        win = self.window()
        cfg_fn = getattr(win, "_current_config", None)
        df = getattr(win, "df_raw", None)
        if cfg_fn is None or df is None or len(df) < V.WARMUP + 20:
            self._reset("「策略研究」页还没有跑出结果。先在那边点一次「策略回测」。")
            return
        try:
            cfg = cfg_fn()
        except Exception as exc:  # noqa: BLE001
            self._reset(f"取配置失败：{exc}")
            return
        self.cfg = cfg
        self.df = df
        self.market = getattr(win, "market", "") or ""
        self.report = None
        self.btn_check.setEnabled(True)
        self.btn_play.setEnabled(False)
        self.btn_play.setChecked(False)
        self.tbl_check.setRowCount(0)
        self.tbl_factor.setRowCount(0)
        self.lbl_verdict.setText(
            f"已取到：{cfg.title()} · {len(df)} 根。点「开始验证」查前视偏差。")
        self._i = len(df) - 1
        self._draw()

    # ---------------------------------------------------------- 验证

    def _start_check(self) -> None:
        if self.cfg is None or self.df is None:
            return
        if self._thread is not None:
            return
        self.btn_check.setEnabled(False)
        self.lbl_verdict.setText("正在验证：把每个检查点之后的数据改掉重算一遍…")
        n = 4 if (getattr(self.cfg, "factor_mode", "") or "") == "ml" else 8
        self._worker = _Worker(self.cfg, self.df, self.market, n)
        self._thread = QThread(self)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.done.connect(self._on_check_done)
        self._worker.failed.connect(self._on_check_failed)
        self._worker.done.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.start()

    def _on_check_done(self, rep) -> None:
        self._worker = None
        self._thread = None
        self.btn_check.setEnabled(True)
        if getattr(rep, "error", ""):
            self._reset(rep.error)
            return
        self.report = rep
        self._fill_tables(rep)
        self.lbl_verdict.setText(("✅ " if rep.ok else "⚠ ") + rep.summary())
        self.btn_play.setEnabled(True)
        self._i = rep.n_bars - 1
        self._draw()

    def _on_check_failed(self, msg: str) -> None:
        self._worker = None
        self._thread = None
        self.btn_check.setEnabled(True)
        self._reset(f"验证失败：{msg}")

    def _fill_tables(self, rep) -> None:
        self.tbl_check.setRowCount(len(rep.checks))
        for r, c in enumerate(rep.checks):
            self._set(r, 0, c.date, self.tbl_check)
            self._set(r, 1, {1: "多", -1: "空", 0: "空仓"}.get(c.signal, str(c.signal)),
                      self.tbl_check)
            self._set(r, 2, "一致" if c.ok else (c.note or "不一致"), self.tbl_check,
                      C_TEXT if c.ok else C_UP)
        self.tbl_factor.setRowCount(len(rep.factors))
        for r, f in enumerate(rep.factors):
            self._set(r, 0, f"{f.label}", self.tbl_factor)
            self._set(r, 1, "—" if f.value != f.value else f"{f.value:.4f}",
                      self.tbl_factor)
            self._set(r, 2, "未用未来" if f.ok else (f.note or "用了未来"),
                      self.tbl_factor, C_TEXT if f.ok else C_UP)

    @staticmethod
    def _set(r: int, c: int, text: str, table: QTableWidget, color: str = "") -> None:
        item = QTableWidgetItem(str(text))
        if color:
            from PySide6.QtGui import QColor

            item.setForeground(QColor(color))
        table.setItem(r, c, item)

    # ---------------------------------------------------------- 播放

    def _on_play_toggled(self, on: bool) -> None:
        if self.report is None:
            self.btn_play.setChecked(False)
            return
        if on:
            if self._i >= self.report.n_bars - 1:
                self._i = max(0, self.report.n_bars - WIN)   # 播完了就从头来
            self.btn_play.setText("⏸ 暂停")
            self._timer.start(SPEEDS[self.cb_speed.currentIndex()][2])
        else:
            self.btn_play.setText("▶ 播放")
            self._timer.stop()

    def _on_speed(self, _idx: int) -> None:
        if self._timer.isActive():
            self._timer.start(SPEEDS[self.cb_speed.currentIndex()][2])

    def _tick(self) -> None:
        rep = self.report
        if rep is None:
            self._timer.stop()
            return
        _label, step, _ms = SPEEDS[self.cb_speed.currentIndex()]
        self._i = min(rep.n_bars - 1, self._i + step)
        self._draw()
        if self._i >= rep.n_bars - 1:
            self.btn_play.setChecked(False)

    # ---------------------------------------------------------- 画图

    def _reset(self, msg: str) -> None:
        self.lbl_verdict.setText(msg)
        self.btn_check.setEnabled(False)
        self.btn_play.setEnabled(False)
        self._timer.stop()
        self.btn_play.setChecked(False)
        with self.canvas.frozen():
            self.canvas.fresh()
            ax = self.canvas.fig.add_subplot(1, 1, 1)
            ax.set_axis_off()
            ax.text(0.5, 0.5, msg, ha="center", va="center", color=C_MUTED,
                    fontsize=11, wrap=True, transform=ax.transAxes)
        self.lbl_prog.setText("—")

    def _draw(self) -> None:
        rep = self.report
        if rep is None or rep.df is None or not len(rep.df):
            return
        df = rep.df
        i = max(0, min(int(self._i), len(df) - 1))
        lo = max(0, i - WIN + 1)
        view = df.iloc[lo:i + 1]
        x = np.arange(len(view))

        o = view["open"].to_numpy(float)
        h = view["high"].to_numpy(float)
        l = view["low"].to_numpy(float)
        c = view["close"].to_numpy(float)
        up = c >= o

        sig = view["signal"].to_numpy(int)
        # 执行口径：第 i 根的持仓来自第 i-1 根的信号（T 收盘出信号、T+1 开盘成交）
        pos = np.concatenate([[0], sig[:-1]])

        with self.canvas.frozen():
            fig = self.canvas.fig
            fig.clear()
            gs = fig.add_gridspec(
                2, 1, height_ratios=[3.4, 1.0], hspace=0.06,
                left=0.075, right=0.975, top=0.94, bottom=0.09,
            )
            ax = fig.add_subplot(gs[0])
            axe = fig.add_subplot(gs[1], sharex=ax)

            # 持仓底色
            for dir_val, bg in ((1, C_LONG_SIG), (-1, C_SHORT_SIG)):
                mask = (pos == dir_val).astype(int)
                if not mask.any():
                    continue
                edges = np.diff(np.concatenate(([0], mask, [0])))
                for s0, e0 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                    ax.axvspan(s0 - 0.5, e0 - 0.5, color=bg, alpha=0.07, lw=0, zorder=0)

            # K 线：影线 + 实体各一个集合，120 根也只产生 2 个图元
            wick = np.stack([np.column_stack([x, l]), np.column_stack([x, h])], axis=1)
            ax.add_collection(LineCollection(
                wick, colors=np.where(up, C_UP, C_DOWN), linewidths=0.8,
                antialiaseds=False, zorder=3))
            body = np.stack([np.column_stack([x, np.minimum(o, c)]),
                             np.column_stack([x, np.maximum(o, c)])], axis=1)
            ax.add_collection(LineCollection(
                body, colors=np.where(up, C_UP, C_DOWN), linewidths=4.2,
                antialiaseds=False, zorder=3))

            # 信号：只在「这一根和上一根不一样」的地方标
            prev = np.concatenate([[0], sig[:-1]])
            for dir_val, color, mark, dy in ((1, C_LONG_SIG, "^", 1.006),
                                             (-1, C_SHORT_SIG, "v", 0.994)):
                idx = np.where((sig == dir_val) & (prev != dir_val))[0]
                if len(idx):
                    ax.scatter(x[idx], c[idx] * dy, marker=mark, s=46, color=color,
                               zorder=6, linewidths=0)
                idx = np.where((sig != dir_val) & (prev == dir_val))[0]
                if len(idx):
                    ax.scatter(x[idx], c[idx] * (2 - dy), marker="x", s=34,
                               color=color, zorder=6, linewidths=1.2)

            ax.set_xlim(-0.8, max(len(view) - 0.2, 1))
            pad = (h.max() - l.min()) * 0.06 or 1.0
            ax.set_ylim(l.min() - pad, h.max() + pad)
            ax.set_facecolor(C_BG)
            ax.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
            ax.set_xticklabels([])
            ax.set_title(
                f"{rep.label or '策略'} · 逐根重放（第 {i + 1}/{rep.n_bars} 根）",
                color=C_TEXT, fontsize=10, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            # 净值：截至当前 bar 的策略曲线 + 买入持有对照
            eq = rep.res.equity.iloc[lo:i + 1].to_numpy(float)
            if len(eq) and eq[0] == eq[0] and eq[0]:
                eq = eq / eq[0]
            axe.plot(x, eq, color=C_STRAT, lw=1.4, label="策略")
            raw = rep.raw
            if raw is not None and len(raw) >= len(df):
                close = raw["close"].iloc[lo:i + 1].to_numpy(float)
                if len(close) and close[0]:
                    axe.plot(x, close / close[0], color=C_BH, lw=1.0, ls="--",
                             label="买入持有")
            axe.set_facecolor(C_BG)
            axe.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
            axe.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:.2f}"))
            axe.tick_params(axis="x", labelsize=8)
            axe.tick_params(axis="y", labelsize=8)
            axe.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
            for side in ("top", "right"):
                axe.spines[side].set_visible(False)

            # 横轴用日期
            step = max(1, len(view) // 8)
            ticks = list(range(0, len(view), step))
            axe.set_xticks(ticks)
            axe.set_xticklabels([view.index[t].strftime("%m-%d") for t in ticks])

        self.lbl_prog.setText(f"{view.index[-1].strftime('%Y-%m-%d')} · "
                              f"{i + 1}/{rep.n_bars} 根")
