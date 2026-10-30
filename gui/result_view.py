"""回测结果的展示层：K线与信号 / 绩效对比 / 因子分析 / 交易记录 / 阶段分析。

**一处实现，两处挂载**：

- 「策略研究」页 —— 边调参数边看，只有前三页（因子分析是调参的一部分）
- 「策略交易和组合」页 —— 看已保存的策略、看 Agent 跑的回测、看多策略组合报告

## 为什么要抽出来

这三页原来是 `MainWindow` 的方法，靠 `self.df_sig` / `self.canvas_k` 这一堆
属性串起来。要在另一个页面再显示一遍，只有两条路：复制一份（两份实现早晚
对不上，改了一处忘另一处），或者抽成一个独立的视图对象。这里选后者。

视图对象**不持有计算逻辑**：`show_result()` 收到的是已经算好的
`df / raw / res / bh`，它只负责画。谁算的（研究页的线程、组合页的同步计算）
它不关心。

## 为什么交易记录和阶段分析放在这里

它们是「这次回测到底发生了什么」的另一半：绩效页回答「赚了多少」，交易记录
回答「怎么赚的」（哪几笔、持了多久），阶段分析回答「什么时候赚的」（哪年哪月
在亏）。单策略和组合报告都要，所以做成两页，组合报告也直接复用。

## 组合报告为什么没有「基准」

单个策略能和「买入持有」比 —— 同一个标的、同一段时间，多空方向不同而已。
几个策略的组合是跨品种、跨周期的加权，没有天然的对照物，硬拉一个「等权买入
持有」只会误导。所以组合报告只给自己的资金曲线和统计。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.dates import AutoDateLocator, DateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PySide6.QtCore import QEvent, QObject, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QScrollBar,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ttrade import cross as cross_engine
from ttrade import factor_metrics, factors, mlmodels, tqdata
from ttrade.backtest import drawdown_series, month_matrix, stage_table
from ttrade.replay import hold_text

# 配色：与 app.py / portfolio_page.py 同一套。不 import 那边 —— 那两个模块会
# import 本模块，反向引用就成了环。
C_BG = "#ffffff"
C_GRID = "#e5e7eb"
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_UP = "#d92b2b"      # 红涨
C_DOWN = "#0f9d58"    # 绿跌
C_STRAT = "#2563eb"
C_BH = "#94a3b8"
C_HOLD_BG = "#3b82f6"
C_SHORT_BG = "#ef4444"
C_LONG_SIG = "#7c3aed"    # 多头信号：紫
C_SHORT_SIG = "#f59e0b"   # 空头信号：琥珀
C_PROFIT = "#d92b2b"
C_LOSSLINE = "#0f9d58"
C_FACTOR = "#0891b2"
C_FACTOR_FILL = "#22d3ee"
C_IC = "#2563eb"
C_IC_NEG = "#dc2626"

# K 线可视窗口（根）
K_WIN_DEFAULT = 180
K_WIN_MIN = 40
K_REDRAW_GAP = 0.15     # 滚动重绘的最小间隔（秒）

PERIOD_1D = "1d"
PERIOD_LABEL = dict(tqdata.PERIOD_LABEL, **{PERIOD_1D: "日线Tushare"})


def _extract_trades(sig) -> list[tuple[int, int, int]]:
    """把 signal 序列拆成一笔笔持仓：返回 [(开仓下标, 平仓下标, 方向)]。

    方向 1 为多、-1 为空。最后一笔还没平仓时，平仓下标取最后一根——
    这样它也能画出连线，只是线段末端贴着右边缘，一眼能看出「还开着」。
    """
    out: list[tuple[int, int, int]] = []
    cur, entry = 0, 0
    for i, s in enumerate(sig):
        s = int(s)
        if s == cur:
            continue
        if cur != 0:
            out.append((entry, i, cur))
        cur, entry = s, i
    if cur != 0 and len(sig):
        out.append((entry, len(sig) - 1, cur))
    return out


def set_tab_visible(tabs, title: str, on: bool) -> None:
    """按标题显隐页签。Qt 的 `setTabVisible` 要下标，标题是两边唯一对得上的东西。"""
    if tabs is None:
        return
    for i in range(tabs.count()):
        if tabs.tabText(i) == title:
            tabs.setTabVisible(i, bool(on))
            return


def _lots_text(v: float) -> str:
    """手数的可读写法。

    `unit` 口径（股票 / 指数 / 旧结果）是份额，会出现 33.2323 这种小数；
    期货手数是整数。两种都要能读，所以按「是不是整数」分开写。
    """
    if v <= 0:
        return "—"
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-6 else f"{v:,.2f}"


# ------------------------------------------------------------------ 图表基类


class _CanvasFrozen:
    """`MplCanvas.frozen()` 的实现体。支持嵌套：只有最外层退出时才补画。"""

    def __init__(self, canvas: "MplCanvas"):
        self.canvas = canvas

    def __enter__(self):
        self.canvas._frozen_depth += 1
        return self.canvas

    def __exit__(self, exc_type, exc, tb):
        cv = self.canvas
        cv._frozen_depth = max(0, cv._frozen_depth - 1)
        if cv._frozen_depth == 0:
            if exc_type is None:
                cv._flush_draw()
            else:
                # 重建中途出错：别补画半成品，交给下一次完整重建
                cv._dirty = False
        return False


class MplCanvas(FigureCanvasQTAgg):
    """matplotlib 画布。

    重建图元时必须用 `frozen()` 包住。原因是 `fig.clear()` 会把 axes 连同
    它下面的 collection 一起解绑，之后 `collection.get_figure()` 返回 None，
    而 `_CollectionWithSizes.draw()` 里要读 `get_figure(root=True).dpi`。
    只要重建期间插进来一次重绘（窗口缩放、切页签、控件显示/隐藏都会触发
    paintEvent，Qt 完全可以在我们两行代码之间塞一个），就会抛

        AttributeError: 'NoneType' object has no attribute 'dpi'

    所以不能靠「我们只在最后手动 draw() 一次」来保证顺序，得显式把重建期
    设成不可绘制，退出临界区再同步补画一帧。
    """

    def __init__(self, figsize=(10, 6), dpi=100):
        # 这两个标志要在 super().__init__ 之前就位：基类构造里可能触发绘制
        self._frozen_depth = 0
        self._dirty = False
        # 自己记一份事件连接清单。matplotlib 的 canvas.callbacks 并不是画布自己的
        # 注册表，而是 figure._canvas_callbacks 的属性代理：
        #     callbacks = property(lambda self: self.figure._canvas_callbacks)
        # 所以 fresh() 换掉 Figure 就等于换掉注册表，之前 mpl_connect 的连接会
        # 全部静默失效（滚轮缩放就这么消失过一次）。记下来，换完再挂一遍。
        self._conns: list[list] = []
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=C_BG)
        super().__init__(self.fig)
        self.setStyleSheet("background: #ffffff;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()

    def frozen(self) -> _CanvasFrozen:
        return _CanvasFrozen(self)

    def mpl_connect(self, s: str, func):
        cid = super().mpl_connect(s, func)
        self._conns.append([cid, s, func])
        return cid

    def mpl_disconnect(self, cid):
        super().mpl_disconnect(cid)
        self._conns = [c for c in self._conns if c[0] != cid]

    def _replay_conns(self) -> None:
        """把记下来的连接重新挂到当前 Figure 上，并刷新 cid。"""
        conns, self._conns = self._conns, []
        for _cid, s, func in conns:
            self.mpl_connect(s, func)

    def fresh(self) -> Figure:
        """换一张全新的 Figure 挂到本画布上，并返回它。

        重建图表时不要用 `fig.clear()`：清空之后 matplotlib 有概率留下
        「还挂在 fig.axes 上、但已经和 figure 解绑」的 collection
        （`get_figure()` 返回 None）。这种残骸肉眼看不见，要等到下一次绘制
        才在 `_CollectionWithSizes.draw` 里炸出来：

            AttributeError: 'NoneType' object has no attribute 'dpi'

        因为触发条件跟「清空」和「重绘」的先后顺序有关，偶发、难复现。
        整张图换掉最干净：新 Figure 里不可能有旧图元，这个失败模式直接消失。
        代价只是多构造一个 Figure 对象（几十微秒），比 K 线重建本身便宜得多。
        """
        old = self.fig
        fig = Figure(figsize=old.get_size_inches(), dpi=old.dpi, facecolor=C_BG)
        # matplotlib 3.11 的 Figure.set_canvas() 只写 figure.canvas，不会反过来
        # 写 canvas.figure，所以两边都要手动指过去。
        fig._original_dpi = getattr(old, "_original_dpi", old.dpi)
        fig.set_canvas(self)
        self.figure = fig
        self.fig = fig
        # 把旧 Figure 从画布上摘掉：否则它以后任何 stale 都会回调
        # canvas.draw_idle()，白白触发一帧重绘。
        old.canvas = None
        # 事件回调是挂在 Figure 上的（见 __init__ 里的说明），换完必须重挂，
        # 否则滚轮缩放、点击等交互会在第一次重建之后全部失灵。
        self._replay_conns()
        return fig

    def draw(self):
        if self._frozen_depth > 0:
            # 重建中：只记一笔，等 frozen() 退出时统一补画。
            # 顺手清掉 _draw_pending，否则 draw_idle() 会因为「已有待办」
            # 一直提前返回，重建完就再也不会重绘了。
            self._dirty = True
            self._draw_pending = False
            return
        self._dirty = False
        super().draw()

    def _flush_draw(self):
        self._dirty = False
        self._draw_pending = False
        super().draw()


class MetricCard(QFrame):
    def __init__(self, label: str, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(1)
        self.value_lbl = QLabel("—")
        self.value_lbl.setObjectName("MetricValue")
        self.name_lbl = QLabel(label)
        self.name_lbl.setObjectName("MetricLabel")
        lay.addWidget(self.value_lbl)
        lay.addWidget(self.name_lbl)

    def set_value(self, text: str, color: str = C_TEXT) -> None:
        self.value_lbl.setText(text)
        self.value_lbl.setStyleSheet(f"color: {color};")


# ------------------------------------------------------------------ 上下文


@dataclass
class ViewCtx:
    """画一张结果图需要的全部外部信息。

    这些值在「策略研究」页来自左边那排控件，在「组合」页来自保存的
    `StrategyConfig` —— 视图本身不该知道它们从哪来。
    """

    symbol: str = ""
    name: str = ""
    exchange: str = ""
    period: str = PERIOD_1D
    market: str = "cn"
    label: str = ""
    legend: str = ""                   # K 线页顶部的图例说明，随市场不同
    price_note: str = "价格"
    range_years: float | None = None   # None = 全部历史
    rule_key: str = ""                 # 决定画哪几条指标线
    overlay: bool = True               # 画不画策略指标线
    long_only: bool = False
    bars_per_year: float = 252.0
    factor_cfg: object | None = None   # fstrat.FactorConfig
    filter_active: bool = False
    selected_keys: list = field(default_factory=list)
    outcome: object | None = None      # fstrat.FactorOutcome
    # 截面策略的上下文：(frames, CrossConfig, 分数面板)。「截面因子分析」要按
    # 品种池重算一遍因子的横截面 IC，光有结果曲线画不出来
    cross: object | None = None


# 交易所中文名。放在这里而不是从 app 引，避免环依赖
EXCHANGE_CN = {
    "SHFE": "上期所",
    "DCE": "大商所",
    "CZCE": "郑商所",
    "CFFEX": "中金所",
    "INE": "能源中心",
    "GFEX": "广期所",
}


# ------------------------------------------------------------------ 视图


class ResultView(QObject):
    """回测结果视图。宿主只负责「给数据」和「放页面」。

    `kind="single"` 给五页（含因子分析、含基准对比）；`kind="portfolio"` 给
    三页（绩效 / 交易记录 / 阶段分析），没有基准也没有因子 —— 组合是跨品种
    加权的，没有单一对照物。
    """

    status = Signal(str)      # 一行状态，宿主决定放哪（研究页放在输入框上方）

    def __init__(self, kind: str = "single", parent=None) -> None:
        super().__init__(parent)
        self.kind = kind
        self.df = None            # 含 signal 的等长表
        self.raw = None           # 原始行情
        self.res = None
        self.bh = None
        self.ctx = ViewCtx()
        self.trades: list = []    # [(策略名, Trade)]，组合报告里带名字
        self.market = "cn"
        self.period = PERIOD_1D
        self.label = ""
        self.range_years = None
        self.rule_key = ""
        self.overlay = True
        self.long_only = False
        self.factor_cfg = None
        self.filter_active = False
        self.selected_keys: list = []
        self.outcome = None

        self._pages: list[tuple[str, QWidget]] = []
        self._built = False
        self._report: pd.DataFrame | None = None
        self._report_key = ""
        # 截面因子分析的缓存：算一遍要遍历所有因子 × 所有品种，切页签不该重算
        self._cf_rep = None
        self._cf_key = ""
        # K 线窗口状态
        self._k_view = None
        self._k_total = 0
        self._k_win = K_WIN_DEFAULT
        self._k_start = 0
        self._k_last_draw = 0.0
        self._k_pending = False
        # 藏着的时候收到的重画请求：画布 -> (画法, 参数)，露面时补一帧
        self._pending_draw: dict = {}
        # 十字光标：双击开/关，鼠标扫到哪根读哪根
        self._k_cross = False
        self._k_hover = -1
        self._k_axes: list = []
        self._k_cross_v: list = []
        self._k_cross_h = None
        self._k_tip = None
        self._k_bars = None
        self._k_notes: dict = {}
        self._k_redraw_timer = QTimer(self)
        self._k_redraw_timer.setSingleShot(True)
        self._k_redraw_timer.timeout.connect(self._flush_kline_redraw)

    # ---------------------------------------------------------- 页面

    def pages(self) -> list[tuple[str, QWidget]]:
        """(标题, 控件) 列表。第一次调用时构建，之后复用。"""
        if not self._built:
            self._pages = self._build_pages()
            self._built = True
        return self._pages

    def _build_pages(self) -> list[tuple[str, QWidget]]:
        out = []
        if self.kind == "single":
            out.append(("K线与信号", self._build_chart_page()))
        out.append(("绩效对比" if self.kind == "single" else "绩效", self._build_perf_page()))
        if self.kind == "single":
            out.append(("因子分析", self._build_factor_page()))
        else:
            # 组合视图默认给「多品种等权」用，那一页没意义（宿主会把它藏起来）；
            # 截面策略跑完才露出来
            out.append(("截面因子分析", self._build_cross_factor_page()))
        out.append(("交易记录", self._build_trades_page()))
        out.append(("阶段分析", self._build_stages_page()))
        return out

    # ---- K 线与信号

    def add_chart_action(self, w: QWidget) -> None:
        """往「K线与信号」页标题行右侧挂一个控件（研究页的「启动实时」）。

        必须先调过 `pages()`：那一步才把这一页建出来，`chart_actions` 才有地方挂。
        主窗口建完 `ResultView` 就立刻 `pages()` 铺页签，顺序上是安全的。
        """
        self.chart_actions.addWidget(w)

    def _build_chart_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self.lbl_chart_title = QLabel("")
        self.lbl_chart_title.setStyleSheet(
            "font-size: 11px; color: #6b7280; padding-left: 4px;"
        )
        # 标题行右侧留给宿主挂按钮：「策略研究」页的「启动实时」就放这儿。
        # 挂在 K 线与信号页而不是侧边栏，是因为它只影响这张图。
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(6)
        head.addWidget(self.lbl_chart_title, 1)
        self.chart_actions = QHBoxLayout()
        self.chart_actions.setContentsMargins(0, 0, 0, 0)
        self.chart_actions.setSpacing(6)
        head.addLayout(self.chart_actions)
        lay.addLayout(head)

        # 图例文字：三角、底色的含义随市场不同（期货有做空、美股只有做多），
        # 所以由宿主通过 ViewCtx.legend 递进来，这里不写死。
        self.lbl_legend = QLabel("")
        self.lbl_legend.setObjectName("Hint")
        self.lbl_legend.setStyleSheet("padding-left: 4px;")
        self.lbl_legend.setWordWrap(True)
        lay.addWidget(self.lbl_legend)

        # 指标卡片条（横向，放在图表上方）
        cards = QHBoxLayout()
        cards.setSpacing(8)
        self.card_cagr = MetricCard("策略年化")
        self.card_dd = MetricCard("最大回撤")
        self.card_sharpe = MetricCard("夏普比率")
        self.card_excess = MetricCard("vs 持有超额")
        for c in (self.card_cagr, self.card_dd, self.card_sharpe, self.card_excess):
            cards.addWidget(c)
        lay.addLayout(cards)

        self.canvas_k = MplCanvas(figsize=(10, 7), dpi=100)
        lay.addWidget(self.canvas_k, 1)
        # 画布重新露面时补画不可见期间攒下的那一帧，见 `_render_kline`
        self.canvas_k.installEventFilter(self)

        # 横向滚动条：在**不缩放**的前提下平移可视窗口。区间选「全部 20 年」时
        # 5000 根挤在一屏根本没法看，有了它就能保持当前根数左右拖着看历史。
        self.sb_kline = QScrollBar(Qt.Horizontal)
        self.sb_kline.setFixedHeight(15)
        self.sb_kline.setVisible(False)
        self.sb_kline.valueChanged.connect(self._on_kline_scroll)
        lay.addWidget(self.sb_kline)

        # 滚轮缩放：以鼠标位置为锚点增减窗口根数，和滚动条配合使用
        self.canvas_k.mpl_connect("scroll_event", self._on_kline_wheel)
        # 双击开十字光标，移动时跟着走。看盘软件里这是最顺手的查价方式：
        # 不用点住、不用选中，鼠标扫到哪根就读哪根
        self.canvas_k.mpl_connect("button_press_event", self._on_kline_click)
        self.canvas_k.mpl_connect("motion_notify_event", self._on_kline_move)
        self.canvas_k.mpl_connect("axes_leave_event", self._on_kline_leave)
        return w

    # ---- 绩效

    def _build_perf_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.canvas_p = MplCanvas(figsize=(7, 6), dpi=100)
        self.canvas_p.installEventFilter(self)
        splitter.addWidget(self.canvas_p)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.setSpacing(6)
        cap = QLabel("绩效指标对比" if self.kind == "single" else "组合绩效指标")
        cap.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        rl.addWidget(cap)

        self.table = QTableWidget(0, 3 if self.kind == "single" else 2)
        self.table.setHorizontalHeaderLabels(
            ["指标", "趋势策略", "买入持有"] if self.kind == "single" else ["指标", "组合"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, self.table.columnCount()):
            self.table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        rl.addWidget(self.table, 1)
        right.setMinimumWidth(360)
        right.setMaximumWidth(470)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 400])

        lay.addWidget(splitter)
        return w

    # ---- 因子分析

    def _build_factor_page(self) -> QWidget:
        """左边全因子 IC/IR 一览，右边四格图。

        表格里列出**所有**因子（不只是选中的），方便横向比较；选中的几行加粗，
        一眼能看出「我用的这几个因子在池子里算什么水平」。
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(6)
        cap = QLabel("因子 IC / ICIR 一览")
        cap.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        ll.addWidget(cap)
        self.lbl_frep = QLabel("")
        self.lbl_frep.setObjectName("Hint")
        self.lbl_frep.setWordWrap(True)
        ll.addWidget(self.lbl_frep)

        self.tbl_factor = QTableWidget(0, 7)
        self.tbl_factor.setHorizontalHeaderLabels(
            ["因子", "IC", "ICIR", "IC>0", "t(NW)", "单调", "换手"]
        )
        self.tbl_factor.verticalHeader().setVisible(False)
        self.tbl_factor.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_factor.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_factor.setShowGrid(False)
        self.tbl_factor.setSortingEnabled(False)
        hh = self.tbl_factor.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 7):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        ll.addWidget(self.tbl_factor, 1)
        splitter.addWidget(left)

        self.canvas_f = MplCanvas(figsize=(8, 7), dpi=100)
        self.canvas_f.installEventFilter(self)
        splitter.addWidget(self.canvas_f)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 760])

        lay.addWidget(splitter)

        self.lbl_fnote = QLabel("")
        self.lbl_fnote.setObjectName("Hint")
        self.lbl_fnote.setWordWrap(True)
        lay.addWidget(self.lbl_fnote)
        return w

    # ---- 截面因子分析

    def _build_cross_factor_page(self) -> QWidget:
        """截面策略的因子分析：左边横截面 IC 一览，右边四格图。

        和时序那页长得很像，但**口径完全不同** —— 这里问的是「同一根 bar 上，
        这个因子给这批品种排的名次，和它们下一段的相对涨跌对不对得上」。
        所以不能复用 `_build_factor_page`：那边一个品种一条时间轴，这边一张
        横截面面板。
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(6)
        cap = QLabel("横截面 IC 一览")
        cap.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        ll.addWidget(cap)
        self.lbl_cfrep = QLabel("")
        self.lbl_cfrep.setObjectName("Hint")
        self.lbl_cfrep.setWordWrap(True)
        ll.addWidget(self.lbl_cfrep)

        # 末两列是截面特有的：覆盖（这个因子在面板上算出数的比例）和期数
        # （参与统计的 bar 数）。时序那边不用看覆盖 —— 一个品种要么有数据要么没有
        self.tbl_cfactor = QTableWidget(0, 7)
        self.tbl_cfactor.setHorizontalHeaderLabels(
            ["因子", "截面IC", "ICIR", "IC>0", "t(NW)", "覆盖", "期数"]
        )
        self.tbl_cfactor.verticalHeader().setVisible(False)
        self.tbl_cfactor.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_cfactor.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_cfactor.setShowGrid(False)
        self.tbl_cfactor.setSortingEnabled(False)
        hh = self.tbl_cfactor.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, 7):
            hh.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        ll.addWidget(self.tbl_cfactor, 1)
        splitter.addWidget(left)

        self.canvas_cf = MplCanvas(figsize=(8, 7), dpi=100)
        self.canvas_cf.installEventFilter(self)
        splitter.addWidget(self.canvas_cf)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([430, 760])

        lay.addWidget(splitter)

        self.lbl_cfnote = QLabel("")
        self.lbl_cfnote.setObjectName("Hint")
        self.lbl_cfnote.setWordWrap(True)
        lay.addWidget(self.lbl_cfnote)
        return w

    # ---- 交易记录

    def _build_trades_page(self) -> QWidget:
        """逐笔交易。绩效页说「赚了多少」，这里说「怎么赚的」。"""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self.lbl_trades = QLabel("")
        self.lbl_trades.setObjectName("Hint")
        self.lbl_trades.setWordWrap(True)
        lay.addWidget(self.lbl_trades)

        cols = ["#", "策略", "方向", "开仓", "开仓价", "平仓", "平仓价", "手数", "收益", "持仓", "备注"]
        if self.kind == "portfolio":
            cols[1] = "策略"
        else:
            cols.remove("策略")
        self.tbl_trades = QTableWidget(0, len(cols))
        self.tbl_trades.setHorizontalHeaderLabels(cols)
        self.tbl_trades.verticalHeader().setVisible(False)
        self.tbl_trades.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_trades.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl_trades.setShowGrid(False)
        h = self.tbl_trades.horizontalHeader()
        for i in range(len(cols)):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        # 最后一列（备注）多数行是空的。让它撑满剩余宽度的话，表格右边会多出
        # 一条 600px 的空白栏，看起来像「备注」全丢了
        h.setStretchLastSection(False)
        lay.addWidget(self.tbl_trades, 1)
        return w

    # ---- 阶段分析

    def _build_stages_page(self) -> QWidget:
        """年度柱状 + 月度矩阵。

        资金曲线是连续的，看不出「哪一年在亏、哪几个月连亏」。把收益切成自然
        年月之后，这种问题一眼就能看见 —— 也才好判断策略是不是只在某一段行情
        里有效。
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self.canvas_s = MplCanvas(figsize=(9, 3), dpi=100)
        self.canvas_s.installEventFilter(self)
        self.canvas_s.setMinimumHeight(170)
        lay.addWidget(self.canvas_s, 1)

        self.lbl_stage = QLabel("")
        self.lbl_stage.setObjectName("Hint")
        self.lbl_stage.setWordWrap(True)
        lay.addWidget(self.lbl_stage)

        self.tbl_month = QTableWidget(0, 13)
        self.tbl_month.setHorizontalHeaderLabels(["年"] + [f"{m}月" for m in range(1, 13)])
        self.tbl_month.verticalHeader().setVisible(False)
        self.tbl_month.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_month.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_month.setShowGrid(True)
        hh = self.tbl_month.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for i in range(1, 13):
            hh.setSectionResizeMode(i, QHeaderView.Stretch)
        lay.addWidget(self.tbl_month, 2)
        return w

    # ---------------------------------------------------------- 入口

    def show_result(self, df, raw, res, bh, ctx: ViewCtx, trades=None,
                    keep_window: bool = False) -> None:
        """挂一份算好的结果并全部重画。

        `keep_window=True` 用于**同一条策略**的实时刷新：数据多了几根，但用户
        看的还是那一段。不保留的话每 5 秒被拽回最新，想翻历史根本翻不动。
        """
        self.df, self.raw, self.res, self.bh = df, raw, res, bh
        self.ctx = ctx
        self.market = ctx.market
        self.period = ctx.period
        self.label = ctx.label
        self.range_years = ctx.range_years
        self.rule_key = ctx.rule_key
        self.overlay = ctx.overlay
        self.long_only = ctx.long_only
        self.factor_cfg = ctx.factor_cfg
        self.filter_active = ctx.filter_active
        self.selected_keys = list(ctx.selected_keys or [])
        self.outcome = ctx.outcome
        self.trades = list(trades) if trades is not None else [
            (ctx.label, t) for t in (res.trades if res is not None else [])
        ]
        self._report, self._report_key = None, ""
        self._cf_rep, self._cf_key = None, ""

        self.pages()          # 确保控件都建好
        if self.kind == "single":
            self.lbl_legend.setText(ctx.legend)
            self._draw_kline(keep_window)
        self._draw_perf()
        self._update_metrics()
        self._update_table()
        if self.kind == "single":
            self._refresh_factor_tab()
        else:
            self._refresh_cross_factor()
        self._refresh_trades()
        self._refresh_stages()

    def update_kline(self, df) -> None:
        """只换 K 线数据、重画那一块。实时刷新用。

        `show_result()` 会把绩效曲线、阶段表、交易记录**全部**重建一遍 ——
        三万根的净值曲线画一次就是几百毫秒，5 秒来一次，用户只是切到别的
        页签看一眼，整页也照样卡成「未响应」。实时刷新只有 K 线在变，别付
        那份额外的钱。

        代价是绩效卡片跟着停在打开那一刻。要最新的绩效，重开一次这条策略
        就有了 —— 那个动作本来就是一秒级，不该每 5 秒自动跑。
        """
        if df is None or not len(df):
            return
        self.df = df
        if self.kind == "single":
            self._draw_kline(keep_window=True)

    def clear(self, msg: str) -> None:
        """没有结果时把图清空并写一句话。

        不清的话会留着上一个策略的图，配上一句「数据缺失」，看起来像是这个
        策略算出来了但结论为空 —— 比直接空白更误导。
        """
        self.df = self.raw = self.res = self.bh = None
        self.trades = []
        # 画布已经被换成「一句话」了，不能留着旧窗口等 `eventFilter` 补画时
        # 把上一张图又摆回来
        self._k_view = None
        self._k_total = 0
        self._pending_draw.clear()
        self.pages()
        self.status.emit(msg)
        for name in ("card_cagr", "card_dd", "card_sharpe", "card_excess"):
            w = getattr(self, name, None)
            if w is not None:
                w.set_value("—", C_MUTED)
        for name in ("table", "tbl_factor", "tbl_trades", "tbl_month"):
            w = getattr(self, name, None)
            if w is not None:
                w.setRowCount(0)
        canvases = [getattr(self, n, None) for n in ("canvas_k", "canvas_p", "canvas_f", "canvas_s")]
        for cv in canvases:
            if cv is None:
                continue
            try:
                fig = cv.fresh()
                ax = fig.add_subplot(111)
                ax.axis("off")
                ax.text(0.5, 0.5, msg, ha="center", va="center",
                        color=C_MUTED, fontsize=12, wrap=True)
                cv.draw()
            except Exception:  # noqa: BLE001
                pass

    # ---------------------------------------------------------- K 线绘制

    def _item(self) -> dict:
        c = self.ctx
        return {"symbol": c.symbol, "name": c.name, "exchange": c.exchange}

    def _draw_kline(self, keep_window: bool = False) -> None:
        """切好数据、定好可视窗口，再交给 _render_kline 画。

        拆成两步是为了滚动/缩放：滚动时数据不变，只挪窗口，没必要重新切
        DataFrame、重算策略——那部分在 _render_kline 里按窗口切片完成。

        `keep_window` 见 `show_result`。
        """
        df = self.df
        if df is None or not len(df):
            return

        years = self.range_years
        if years and len(df):
            cutoff = df.index[-1] - pd.DateOffset(years=years)
            view = df[df.index >= cutoff]
        else:
            view = df
        if len(view) < 20:
            view = df.tail(120)

        old_total, old_start, old_win = self._k_total, self._k_start, self._k_win
        self._k_view = view
        self._k_total = len(view)
        if keep_window and old_total > 0:
            # 实时刷新不该动用户的位置。原来贴着右端看的，就跟着新右端走；
            # 已经滚到历史里去的，原地不动 —— 否则每 5 秒被拽回最新一次
            self._k_win = max(K_WIN_MIN, min(int(old_win), self._k_total))
            if old_start + old_win >= old_total:
                self._k_start = max(0, self._k_total - self._k_win)
            else:
                self._k_start = max(0, min(self._k_total - self._k_win, old_start))
        else:
            # 换标的 / 换策略 / 换区间时回到最新一段，符合看盘习惯
            self._k_win = min(K_WIN_DEFAULT, self._k_total)
            self._k_start = max(0, self._k_total - self._k_win)
        # 换图是用户主动操作，不受滚动节流限制
        self._k_last_draw = time.monotonic()
        self._k_pending = False
        self._render_kline()

    def _render_kline(self) -> None:
        """按当前窗口 [start, start+win) 重画 K 线，窗口外的数据不进图元。

        整个重建过程在画布的 frozen() 临界区里跑，避免半成品状态被重绘。

        **画布藏起来了就一个图元都不建**。一次重建是 matplotlib 整图重来，8000 根
        要 100~230ms；而这条链路能被实时推送、滚动条回弹、节流定时器补帧好几个
        地方叫到，逐个堵必然漏。堵在总入口，谁叫都白叫，画布重新露面时
        `eventFilter` 补一帧 —— 那会儿数据早就备好了，只是没画。

        还没上过屏（离屏自检、窗口还没 show）时不拦：那种场景没有「看得见看不见」
        之分，拦下来只会让图一直是空的。
        """
        if self._defer_draw(self.canvas_k, self._render_kline):
            return
        with self.canvas_k.frozen():
            self._render_kline_impl()

    # ---------------------------------------------------------- 画布总闸

    def _defer_draw(self, canvas, fn, *args) -> bool:
        """画布藏着就先把这次重画记下来，露面时补 —— 返回 True 表示已推迟。

        matplotlib 整图重建是 100ms 起步的活儿（8000 根 K 线 100~230ms，因子页
        四格图 1 秒以上），画在看不见的页签上纯属白烧主线程：一次「改个参数」
        里有一半时间在画没人看的图，表现就是「点一下要等两秒」。

        窗口本身还没显示（离屏自检、启动第一帧）时不拦 —— 那种场景没有「看得见
        看不见」之分，拦下来只会让图一直是空的。看窗口而不是看画布，是因为因子页
        这种**一开始就不是当前页签**的画布永远等不到第一次 Show。
        """
        if not canvas.window().isVisible() or canvas.isVisible():
            return False
        self._pending_draw[canvas] = (fn, args)
        return True

    def eventFilter(self, obj, ev) -> bool:
        """画布重新露面：把藏起来期间攒下的那一帧补上。

        画布可能是被外层换页藏起来的，也可能是研究页里切到了「绩效」页签 ——
        两种情况都会收到 `QEvent.Show`，所以挂在这里，而不是绑某个具体的切页
        回调（那种地方以后加一个入口就会漏一个）。
        """
        if ev.type() == QEvent.Show:
            pend = self._pending_draw.pop(obj, None)
            if pend is not None:
                fn, args = pend
                fn(*args)
        return super().eventFilter(obj, ev)

    def _render_kline_impl(self) -> None:
        view_all = self._k_view
        if view_all is None or not len(view_all):
            return

        total = self._k_total
        win = max(K_WIN_MIN, min(int(self._k_win), total))
        start = max(0, min(total - win, int(self._k_start)))
        self._k_win, self._k_start = win, start

        view = view_all.iloc[start:start + win]
        spec_label = self.label
        item = self._item()
        sym = item["symbol"]

        x = np.arange(len(view))
        o = view["open"].to_numpy(float)
        h = view["high"].to_numpy(float)
        l = view["low"].to_numpy(float)
        c = view["close"].to_numpy(float)
        vol = view["volume"].to_numpy(float)
        up = c >= o

        # 全量的信号：窗口第一根到底是不是「开仓」，得看它的前一根，
        # 只拿窗口内的数据判断会把每根都误认成开仓点。
        raw_sig = view_all["signal"].to_numpy(int)
        if self.long_only:
            raw_sig = np.maximum(raw_sig, 0)
        # **执行口径**：回测是「T 收盘出信号、T+1 开盘成交」，所以第 i 根上的
        # 持仓和开平动作来自第 i-1 根收盘的信号。图上必须按执行口径画 ——
        # 画在信号那根的收盘价上，看起来像「收盘价成交」，和回测、实盘都对不上
        sig_full = np.concatenate([[0], raw_sig[:-1]])
        open_full = view_all["open"].to_numpy(float)
        sig_view = sig_full[start:start + win]
        prev = np.concatenate([[0], sig_full[:-1]])[start:start + win]

        # 因子模式下多画一行分数曲线：能直接看到「分数在哪里穿过阈值、
        # 穿过去之后策略怎么走」，比只看三角标记有用得多。
        has_score = "factor_score" in view_all.columns
        score_full = (
            view_all["factor_score"].to_numpy(float) if has_score else None
        )

        self._k_notes = self._signal_notes()

        self.canvas_k.fresh()
        if has_score:
            gs = self.canvas_k.fig.add_gridspec(
                3, 1, height_ratios=[3.6, 0.85, 1.0], hspace=0.08,
                left=0.082, right=0.962, top=0.935, bottom=0.085,
            )
            axf = self.canvas_k.fig.add_subplot(gs[2], sharex=None)
        else:
            gs = self.canvas_k.fig.add_gridspec(
                2, 1, height_ratios=[4.0, 1.0], hspace=0.06,
                # left 留够：沪铜、碳酸锂这类五位数报价，0.058 会把最高位切掉
                left=0.082, right=0.962, top=0.935, bottom=0.085,
            )
            axf = None
        ax = self.canvas_k.fig.add_subplot(gs[0])
        axv = self.canvas_k.fig.add_subplot(gs[1], sharex=ax)
        if axf is not None:
            axf.sharex(ax)

        ax.set_facecolor(C_BG)
        axv.set_facecolor(C_BG)
        if axf is not None:
            axf.set_facecolor(C_BG)

        # 持仓区间底色：多头蓝、空头红。合并连续区间用 axvspan，
        # 避免逐点 fill_between 产生上千个多边形。
        # 只做多模式下把空头信号裁掉，保证图形与回测口径一致。
        for dir_val, bg in ((1, C_HOLD_BG), (-1, C_SHORT_BG)):
            mask = (sig_view == dir_val).astype(int)
            if not mask.any():
                continue
            edges = np.diff(np.concatenate(([0], mask, [0])))
            for s0, e0 in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
                ax.axvspan(s0 - 0.5, e0 - 0.5, color=bg, alpha=0.07,
                           lw=0, zorder=0)

        # K 线：用 Collection 一次性提交，5000 根也只产生 2 个图元
        # （此前用 vlines + bar 会产生上万个 Patch，渲染要十几秒，窗口假死）
        span = max(float(h.max() - l.min()), 1e-9)
        width = 0.62 if len(view) < 400 else (0.75 if len(view) < 1200 else 0.9)
        half = width / 2.0
        eps = span * 0.0008

        wick_seg = np.stack(
            [np.column_stack([x, l]), np.column_stack([x, h])], axis=1
        )
        ax.add_collection(
            LineCollection(
                wick_seg,
                colors=np.where(up, C_UP, C_DOWN),
                linewidths=0.75,
                antialiaseds=False,
                zorder=3,
            )
        )

        bottom = np.minimum(o, c)
        top = np.maximum(o, c)
        top = np.where(top - bottom < eps, bottom + eps, top)
        left, right = x - half, x + half
        verts = np.stack(
            [
                np.column_stack([left, bottom]),
                np.column_stack([left, top]),
                np.column_stack([right, top]),
                np.column_stack([right, bottom]),
            ],
            axis=1,
        )
        face = np.where(
            up[:, None], np.array(to_rgba(C_UP)), np.array(to_rgba(C_DOWN))
        )
        ax.add_collection(
            PolyCollection(verts, facecolors=face, edgecolors="none",
                           antialiaseds=False, zorder=3)
        )

        # 策略指标线
        overlay_cfg = {
            "ma_cross": [("fast_ma", "快线 MA", "#f59e0b", 1.3), ("slow_ma", "慢线 MA", "#2563eb", 1.5)],
            "donchian": [("upper", "上轨", "#059669", 1.2), ("lower", "下轨", "#dc2626", 1.2)],
            "atr_channel": [("mid", "均线", "#f59e0b", 1.2), ("upper", "上轨", "#059669", 1.2),
                            ("lower", "下轨", "#dc2626", 1.2)],
            "macd": [],
            "turtle": [("upper", "上轨", "#059669", 1.2), ("lower", "下轨", "#dc2626", 1.2)],
        }
        if self.overlay:
            for col, label, color, lw in overlay_cfg.get(self.rule_key, []):
                if col in view.columns:
                    y = view[col].to_numpy(float)
                    ax.plot(x, y, color=color, lw=lw, label=label, zorder=4, alpha=0.95)

        # 持仓连线：把每笔的开仓点和平仓点用虚线连起来，颜色直接表达这笔赚没赚。
        # 盈利红、亏损绿（A 股习惯）。只在开仓点可见时才画——否则好几笔交易会
        # 全都从窗口左边缘起笔，看着像凭空冒出来的一堆线。平仓在窗口外时线画到
        # 右边缘，正好表示「还持有」。
        for e_i, x_i, d in _extract_trades(sig_full):
            if e_i < start or e_i >= start + win:
                continue
            b = min(x_i, start + win - 1)
            if b <= e_i:
                continue
            # 连线两头都是**成交价**（T+1 开盘），不是收盘价
            gain = (open_full[x_i] - open_full[e_i]) * d
            ax.plot(
                [e_i - start, b - start], [open_full[e_i], open_full[b]],
                color=C_PROFIT if gain >= 0 else C_LOSSLINE,
                ls=(0, (4, 3)), lw=1.15, alpha=0.88, zorder=5,
            )

        # 开平仓信号：颜色=持仓方向（紫多 / 琥珀空），形状=开仓/平仓（▲开 ▼平）
        # 直接反手（多→空 / 空→多）时，平仓和开仓落在同一根上，两个三角会完全重叠，
        # 后画的把先画的盖掉。这种情况只画开仓标记——平仓本来也由连线交代了。
        pos = sig_view
        off = (h.max() - l.min()) * 0.028
        marks = [
            ((pos == 1) & (prev != 1), "^", C_LONG_SIG, -1, "开多"),
            ((pos == 0) & (prev == 1), "v", C_LONG_SIG, +1, "平多"),
            ((pos == -1) & (prev != -1), "v", C_SHORT_SIG, +1, "开空"),
            ((pos == 0) & (prev == -1), "^", C_SHORT_SIG, -1, "平空"),
        ]
        for cond, marker, color, side, label in marks:
            idxs = np.where(cond)[0]
            if not len(idxs):
                continue
            y = l[idxs] - off if side < 0 else h[idxs] + off
            ax.scatter(idxs, y, marker=marker, s=72, c=color,
                       edgecolors="white", linewidths=0.9, zorder=6, label=label)

        # 最新价参考线。只在窗口右端确实是最新一根时才画，而且用中性灰点线：
        # 原来用红/绿虚线画，跟「持仓盈亏连线」是同一个样式，看着像凭空多出一笔
        # 还没平的交易。滚到历史位置时这根线也没意义——那是窗口末价，不是最新价。
        at_latest = start + win >= total
        last_color = C_UP if c[-1] >= o[-1] else C_DOWN
        if at_latest:
            ax.axhline(c[-1], color="#94a3b8", lw=0.8, ls=":", alpha=0.75, zorder=2)
        chg = (c[-1] / c[0] - 1.0) * 100
        tag = "最新" if at_latest else "末价"
        ax.text(
            0.995, 0.975,
            f"{tag} {c[-1]:,.2f}　区间 {chg:+.1f}%",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9.5, color="white", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.34", fc=last_color, ec="none", alpha=0.92),
            zorder=8,
        )

        # 成交量：同样用 Collection 批量提交
        v_bottom = np.zeros_like(vol)
        v_verts = np.stack(
            [
                np.column_stack([left, v_bottom]),
                np.column_stack([left, vol]),
                np.column_stack([right, vol]),
                np.column_stack([right, v_bottom]),
            ],
            axis=1,
        )
        v_face = np.where(
            up[:, None], np.array(to_rgba(C_UP, 0.55)), np.array(to_rgba(C_DOWN, 0.55))
        )
        axv.add_collection(
            PolyCollection(v_verts, facecolors=v_face, edgecolors="none",
                           antialiaseds=False, zorder=2)
        )
        axv.set_ylabel("量", fontsize=8, color=C_MUTED, rotation=0, labelpad=12)
        axv.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K"))

        # ---- 因子分数面板 ----
        if axf is not None and score_full is not None:
            sc = score_full[start:start + win]
            xa = np.arange(len(sc))
            ok = np.isfinite(sc)
            if ok.any():
                axf.plot(xa[ok], sc[ok], color=C_FACTOR, lw=1.1, zorder=4)
                axf.fill_between(xa, 0.0, np.nan_to_num(sc), where=ok,
                                 color=C_FACTOR_FILL, alpha=0.16, lw=0, zorder=1)
            cfg = self.factor_cfg
            if cfg is not None and self.filter_active:
                lv, sv = cfg.long_th, cfg.short_th
                axf.axhline(lv, color=C_LONG_SIG, lw=0.9, ls="--", alpha=0.85, zorder=3)
                # 门槛文字压在分数曲线上，加个半透明白底才读得清
                box = dict(boxstyle="round,pad=0.15", fc="#ffffff", ec="none", alpha=0.72)
                if abs(sv - lv) < 1e-9:
                    # 两个门槛相等（默认都是 0）时只画一条线、一个标签：
                    # 否则两行文字会精确重叠，糊成一团黑字
                    axf.text(0.018, lv, f"门槛 {lv:+.1f}",
                             transform=axf.get_yaxis_transform(),
                             fontsize=7.5, color=C_LONG_SIG, va="bottom", ha="left",
                             zorder=6, bbox=box)
                else:
                    axf.axhline(sv, color=C_SHORT_SIG, lw=0.9, ls="--", alpha=0.85, zorder=3)
                    axf.text(0.018, lv, f"多头门槛 {lv:+.1f}",
                             transform=axf.get_yaxis_transform(),
                             fontsize=7.5, color=C_LONG_SIG, va="bottom", ha="left",
                             zorder=6, bbox=box)
                    axf.text(0.018, sv, f"空头门槛 {sv:+.1f}",
                             transform=axf.get_yaxis_transform(),
                             fontsize=7.5, color=C_SHORT_SIG, va="top", ha="left",
                             zorder=6, bbox=box)
            elif cfg is not None:
                for lv, col in ((cfg.entry_z, C_LONG_SIG), (-cfg.entry_z, C_SHORT_SIG)):
                    axf.axhline(lv, color=col, lw=0.9, ls="--", alpha=0.8, zorder=3)
                for lv in (cfg.exit_z, -cfg.exit_z):
                    axf.axhline(lv, color=C_MUTED, lw=0.7, ls=":", alpha=0.7, zorder=2)
            axf.axhline(0.0, color="#cbd5e1", lw=0.7, zorder=2)
            axf.set_ylabel("因子分", fontsize=8, color=C_MUTED, rotation=0, labelpad=14)
            axf.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
            axf.set_axisbelow(True)
            axf.set_xlim(-1, len(view))

        # 坐标轴
        ax.set_xlim(-1, len(view))
        pad = (h.max() - l.min()) * 0.06
        ax.set_ylim(l.min() - pad, h.max() + pad)
        ax.grid(True, color=C_GRID, lw=0.6, ls="-", alpha=0.9)
        ax.set_axisbelow(True)
        axv.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
        axv.set_axisbelow(True)

        all_axes = [ax, axv] + ([axf] if axf is not None else [])
        for a in all_axes:
            for spine in ("top", "right"):
                a.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                a.spines[spine].set_color("#cbd5e1")
            a.tick_params(colors=C_MUTED, labelsize=9, length=3)
        for a in all_axes[:-1]:
            a.tick_params(labelbottom=False)

        # x 轴日期刻度放在最下面那根轴上。格式跟着可视跨度走：180 根 5 分钟
        # 只有两天半，一律写「年-月」会变成十个一模一样的标签。
        bottom_ax = all_axes[-1]
        n_ticks = min(10, max(4, len(view) // 60))
        tick_pos = np.linspace(0, len(view) - 1, n_ticks).astype(int)
        span_s = (view.index[-1] - view.index[0]).total_seconds()
        if span_s <= 2 * 86400:
            fmt = "%m-%d %H:%M"
        elif span_s <= 150 * 86400:
            fmt = "%m-%d"
        else:
            fmt = "%Y-%m"
        bottom_ax.set_xticks(tick_pos)
        bottom_ax.set_xticklabels(
            [view.index[i].strftime(fmt) for i in tick_pos],
            rotation=0, fontsize=9,
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            # 排成一行：开多/平多/开空/平空 四项加两条均线，两行会挡住左上角的 K 线
            ax.legend(loc="upper left", frameon=True, framealpha=0.92,
                      edgecolor="#e2e8f0", fontsize=8.5, ncol=len(handles),
                      handletextpad=0.4, columnspacing=1.1)
        ax.set_ylabel(self.ctx.price_note, fontsize=9, color=C_MUTED)

        # ---- 十字光标 ----
        # 图元建好就常驻（初始不可见），鼠标移动只改坐标不重建 —— 重建一次
        # 5000 根 K 线的图元要几百毫秒，跟着鼠标走会卡成幻灯片。
        self._k_axes = all_axes
        self._k_bars = view
        self._k_cross_v = [
            a.axvline(0, color="#64748b", lw=0.7, alpha=0.7, zorder=9, visible=False)
            for a in all_axes
        ]
        self._k_cross_h = ax.axhline(
            float(c[0]), color="#64748b", lw=0.7, alpha=0.7, zorder=9, visible=False
        )
        # 压在「最新价」那个色块下面一行：两者都在右上角，贴太近会互相盖住
        self._k_tip = ax.text(
            0.995, 0.875, "", transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, color="#e2e8f0", linespacing=1.5, zorder=10,
            bbox=dict(boxstyle="round,pad=0.42", fc="#0f172a", ec="#475569", alpha=0.78),
        )
        self._k_tip.set_visible(False)
        self._k_hover = -1
        if self._k_cross:
            self._show_cross(len(view) - 1, draw=False)

        # 标题信息
        name = item["name"]
        exch = item["exchange"]
        exch_cn = EXCHANGE_CN.get(exch, exch)
        head = f"{sym}  {name}" + (f"（{exch_cn}）" if exch else "")
        # 「当前状态」永远指最新一根，不是窗口最后一根——否则滚到历史位置时
        # 标题会跟着变，看着像当前真的空仓了。
        state = {1: "多头持仓", -1: "空头持仓"}.get(int(sig_full[-1]), "空仓")
        # 最后一根收盘又出了新信号时，仓位还没变（要等下一根开盘），
        # 但人看着图会问「那现在到底算什么」，所以把待执行的动作写出来
        pend = int(raw_sig[-1]) if len(raw_sig) else 0
        if pend != int(sig_full[-1]):
            state += "（下一根开盘" + {1: "开多", -1: "开空"}.get(pend, "平仓") + "）"
        period = self.period
        plabel = "" if tqdata.is_daily(period) else f" · {PERIOD_LABEL.get(period, period)}"
        ax.set_title(
            f"{head}{plabel}   ·   {spec_label}   ·   "
            f"{view.index[0]:%Y-%m-%d %H:%M} ~ {view.index[-1]:%Y-%m-%d %H:%M}"
            f"   ·   当前状态：{state}",
            fontsize=11.5, color=C_TEXT, loc="left", pad=10,
        )
        if self.market == "us":
            source = "东方财富后复权"
        elif period == PERIOD_1D:
            source = "Tushare 换月前复权"
        else:
            src = tqdata.PERIODS.get(period, (period, 1))
            merged = f"（由 {src[0]} × {src[1]} 合并）" if src[1] > 1 else ""
            source = f"天勤换月前复权{merged}"
        shown = f"{start + 1:,}–{start + win:,}"
        self.lbl_chart_title.setText(
            f"全历史 {len(self.df):,} 根 · 当前显示第 {shown} 根"
            f"（窗口 {win:,} 根）· 滚轮缩放 / 拖下方滚动条看历史 · {source}"
        )

        self._sync_kline_scrollbar()
        # 不用再手动 draw()：退出 frozen() 临界区时会同步补画一帧，
        # 那样能保证「重建完成」和「绘制」之间不会插进别的重绘。

    # ---------------------------------------------------------- 十字光标

    def _signal_notes(self) -> dict:
        """每根 bar 上发生的开平动作，给十字光标的信息框用。

        直接读 `self.trades` 而不是重新从 signal 推：手数、成交价只有成交记录里
        才有，从 signal 只能推出方向，还得再猜一次手数。
        """
        notes: dict = {}
        with_name = self.kind == "portfolio"
        for name, t in self.trades or []:
            who = f"{name} " if with_name else ""
            side = "多" if t.direction > 0 else "空"
            lots = f" {_lots_text(t.lots)} 手" if t.lots > 0 else ""
            notes.setdefault(t.entry_date, []).append(
                f"{who}开{side} @{t.entry_price:,.2f}{lots}"
            )
            if not t.open_position:
                notes.setdefault(t.exit_date, []).append(
                    f"{who}平{side} @{t.exit_price:,.2f}{lots}"
                )
        return notes

    def _on_kline_click(self, ev) -> None:
        """双击开/关十字光标。"""
        if not getattr(ev, "dblclick", False) or ev.inaxes not in self._k_axes:
            return
        self._k_cross = not self._k_cross
        if self._k_cross and ev.xdata is not None:
            self._show_cross(int(round(ev.xdata)))
        else:
            self._hide_cross()

    def _on_kline_move(self, ev) -> None:
        if not self._k_cross or ev.inaxes not in self._k_axes or ev.xdata is None:
            return
        self._show_cross(int(round(ev.xdata)))

    def _on_kline_leave(self, _ev) -> None:
        if self._k_cross:
            self._hide_cross()

    def _show_cross(self, i: int, draw: bool = True) -> None:
        view = self._k_bars
        if view is None or not len(view):
            return
        i = max(0, min(len(view) - 1, int(i)))
        if draw and i == self._k_hover:
            return              # 同一根不重画：鼠标横扫时会连着触发几十次
        self._k_hover = i
        for ln in self._k_cross_v:
            ln.set_xdata([i, i])
            ln.set_visible(True)
        if self._k_cross_h is not None:
            self._k_cross_h.set_ydata([float(view["close"].iloc[i])] * 2)
            self._k_cross_h.set_visible(True)
        if self._k_tip is not None:
            self._k_tip.set_text(self._bar_text(i))
            self._k_tip.set_visible(True)
        if draw:
            self.canvas_k.draw_idle()

    def _hide_cross(self) -> None:
        self._k_hover = -1
        for ln in self._k_cross_v:
            ln.set_visible(False)
        if self._k_cross_h is not None:
            self._k_cross_h.set_visible(False)
        if self._k_tip is not None:
            self._k_tip.set_visible(False)
        self.canvas_k.draw_idle()

    def _bar_text(self, i: int) -> str:
        """一根 K 线的读数：时间、开高低收、涨跌幅、量，外加这一根上的信号。"""
        view = self._k_bars
        row = view.iloc[i]
        ts = view.index[i]
        fmt = "%Y-%m-%d" if tqdata.is_daily(self.period) else "%Y-%m-%d %H:%M"
        o, h, l, c = (float(row[k]) for k in ("open", "high", "low", "close"))
        pct = (c / o - 1.0) * 100.0 if o else 0.0
        lines = [ts.strftime(fmt)]
        lines.append(f"开 {o:,.2f}　高 {h:,.2f}")
        lines.append(f"低 {l:,.2f}　收 {c:,.2f}　{pct:+.2f}%")
        vol = float(row.get("volume", 0) or 0)
        if vol > 0:
            lines.append(f"量 {vol / 1e4:,.1f} 万")
        for note in self._k_notes.get(ts, []) or []:
            lines.append(note)
        return "\n".join(lines)

    # ---------------------------------------------------------- 窗口滚动 / 缩放

    def _sync_kline_scrollbar(self) -> None:
        """让滚动条的范围跟上当前数据量与窗口大小。

        数据比窗口还短时直接隐藏——一根能滚的余地都没有，留着只会让人以为坏了。
        """
        total, win = self._k_total, self._k_win
        sb = getattr(self, "sb_kline", None)
        if sb is None:
            return
        if total <= win:
            sb.setVisible(False)
            return
        sb.setVisible(True)
        sb.blockSignals(True)   # 避免 setValue 反过来触发一次重绘
        sb.setRange(0, total - win)
        sb.setPageStep(win)
        sb.setSingleStep(max(1, win // 10))
        sb.setValue(self._k_start)
        sb.blockSignals(False)

    def _on_kline_scroll(self, value: int) -> None:
        if self._k_view is None or value == self._k_start:
            return
        self._k_start = int(value)
        self._request_kline_redraw()

    def _request_kline_redraw(self) -> None:
        """节流重绘：间隔内只留一个待办，到点补画最终位置。

        拖滚动条时 valueChanged 一秒能来几十次，每次都重建图元会堆死事件队列。
        节流后每秒最多画 K_REDRAW_GAP 帧，而且松手时一定会补最后一帧——
        不能只有防抖，否则快速拖完停住，画面会停在中间某一帧上。
        """
        now = time.monotonic()
        gap = now - self._k_last_draw
        if gap >= K_REDRAW_GAP:
            self._k_pending = False
            self._k_last_draw = now
            self._render_kline()
            return
        self._k_pending = True
        self._k_redraw_timer.start(int((K_REDRAW_GAP - gap) * 1000) + 5)

    def _flush_kline_redraw(self) -> None:
        if not self._k_pending:
            return
        self._k_pending = False
        self._k_last_draw = time.monotonic()
        self._render_kline()

    def _on_kline_wheel(self, event) -> None:
        """滚轮缩放：改变窗口根数，并以鼠标所在位置为锚点保持不漂移。"""
        if self._k_view is None or self._k_total <= 0:
            return
        if event.inaxes is None:
            return
        old_win = self._k_win
        step = 0.8 if event.button == "up" else 1.25
        new_win = int(round(old_win * step))
        new_win = max(K_WIN_MIN, min(self._k_total, new_win))
        if new_win == old_win:
            return

        anchor = event.xdata if event.xdata is not None else (self._k_start + old_win / 2)
        ratio = (anchor - self._k_start) / max(old_win, 1)
        self._k_win = new_win
        self._k_start = int(round(anchor - ratio * new_win))
        self._k_start = max(0, min(self._k_total - new_win, self._k_start))
        # 滚轮频率低，直接画，保证每格都有即时反馈
        self._k_last_draw = time.monotonic()
        self._k_pending = False
        self._render_kline()

    # ---------------------------------------------------------- 绩效图

    def _draw_perf(self) -> None:
        """绩效页重建，同样走 frozen() 临界区。"""
        if self.res is None:
            return
        if self._defer_draw(self.canvas_p, self._draw_perf):
            return
        with self.canvas_p.frozen():
            self._draw_perf_impl()

    def _draw_perf_impl(self) -> None:
        res, bh = self.res, self.bh
        fig = self.canvas_p.fresh()
        gs = fig.add_gridspec(
            2, 1, height_ratios=[3.0, 1.0], hspace=0.12,
            left=0.085, right=0.975, top=0.93, bottom=0.085,
        )
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax)

        eq_s = res.equity / res.equity.iloc[0]
        if bh is not None:
            eq_b = bh.equity / bh.equity.iloc[0]
            ax.plot(eq_b.index, eq_b.values, color=C_BH, lw=1.5, label="买入持有")
        ax.plot(eq_s.index, eq_s.values, color=C_STRAT, lw=1.7, label=res.label)
        ax.fill_between(eq_s.index, 1, eq_s.values,
                        where=eq_s.values >= 1, color=C_STRAT, alpha=0.06)
        ax.fill_between(eq_s.index, 1, eq_s.values,
                        where=eq_s.values < 1, color="#dc2626", alpha=0.06)
        ax.axhline(1.0, color="#cbd5e1", lw=0.8, ls="--")
        # 权益可能被亏穿成负数（回测不再强平，见 `backtest.run_backtest`）。
        # 负数在对数轴上没有位置：matplotlib 会把这一段直接丢掉，曲线看着像
        # 「亏光那天就停了」—— 又变回强平的效果。有非正值就退回线性轴
        positive = bool((eq_s > 0).all()) and (
            bh is None or bool((eq_b > 0).all()))
        if positive:
            ax.set_yscale("log")
        ax.set_ylabel("净值（对数）" if positive else "净值", fontsize=9, color=C_MUTED)
        ax.set_title(
            "策略 vs 买入持有 · 资金曲线" if bh is not None else "组合资金曲线（等权）",
            fontsize=11.5, loc="left", color=C_TEXT, pad=8,
        )
        ax.grid(True, color=C_GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        # 对数轴上如果数据没跨一个数量级，可见刻度全是 minor，而 matplotlib 给
        # minor 的默认格式是「2 × 10⁰」这种科学计数法 —— 净值 1.6 被写成 1.6×10⁰，
        # 一眼看不出是多少。major 和 minor 都换成普通数字。
        plain = FuncFormatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_major_formatter(plain)
        ax.yaxis.set_minor_formatter(plain)
        ax.tick_params(labelbottom=False, colors=C_MUTED, labelsize=9)

        dd_s = drawdown_series(res.equity) * 100
        if bh is not None:
            dd_b = drawdown_series(bh.equity) * 100
            ax2.fill_between(dd_b.index, dd_b.values, 0, color=C_BH, alpha=0.35, label="持有回撤")
        ax2.fill_between(dd_s.index, dd_s.values, 0, color=C_STRAT, alpha=0.45, label="策略回撤")
        ax2.set_ylabel("回撤 %", fontsize=9, color=C_MUTED)
        ax2.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
        ax2.set_axisbelow(True)
        ax2.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax2.tick_params(colors=C_MUTED, labelsize=9)

        ax.xaxis.set_major_locator(AutoDateLocator(maxticks=8))
        ax2.xaxis.set_major_formatter(DateFormatter("%Y"))
        for a in (ax, ax2):
            for spine in ("top", "right"):
                a.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                a.spines[spine].set_color("#cbd5e1")
        for lbl in ax2.get_xticklabels():
            lbl.set_rotation(0)

    # ---------------------------------------------------------- 指标

    def _update_metrics(self) -> None:
        if self.res is None:
            return
        m = self.res.metrics
        b = self.bh.metrics if self.bh is not None else None
        card_cagr = getattr(self, "card_cagr", None)
        if card_cagr is not None:
            card_cagr.set_value(f"{m['cagr']*100:.1f}%", C_UP if m["cagr"] >= 0 else C_DOWN)
            self.card_dd.set_value(f"{m['max_dd']*100:.1f}%", "#ea580c")
            self.card_sharpe.set_value(
                # 权益亏穿过时夏普是 NaN（见 `backtest._metrics`），写「—」
                "—" if m["sharpe"] != m["sharpe"] else f"{m['sharpe']:.2f}",
                "#0f9d58" if m["sharpe"] >= 1
                else ("#1f2937" if m["sharpe"] >= 0.5 else "#ea580c"),
            )
            if b is not None:
                excess = m["cagr"] - b["cagr"]
                self.card_excess.set_value(
                    f"{excess*100:+.1f}%", "#0f9d58" if excess >= 0 else "#d92b2b"
                )
            else:
                self.card_excess.set_value("—", C_MUTED)

        dir_txt = ""
        if not self.long_only:
            dir_txt = (f" · 多头 {m['long_exposure']*100:.0f}% / 空头 {m['short_exposure']*100:.0f}%"
                       f" · 空头 {m['n_short']} 笔")
        self.status.emit(
            f"{m['n_trades']} 笔交易 · 胜率 {m['win_rate']*100:.0f}% · "
            f"仓位时间 {m['exposure']*100:.0f}% · "
            f"平均持仓 {hold_text(m['avg_bars'], self.period, self.ctx.bars_per_year)}{dir_txt}"
        )

    def _update_table(self) -> None:
        if self.res is None:
            return
        m = self.res.metrics
        b = self.bh.metrics if self.bh is not None else None

        def pct(x):
            # 亏穿过之后年化波动是 NaN（见 `backtest._metrics`）：写「—」，
            # 别让表格里出现「nan%」
            return "—" if x != x else f"{x*100:.2f}%"

        def num(x):
            return "—" if x != x else f"{x:.2f}"

        rows = [
            ("年化收益 CAGR", pct(m["cagr"]), pct(b["cagr"]) if b else "—",
             None if b is None else m["cagr"] >= b["cagr"]),
            ("累计收益", pct(m["total_return"]), pct(b["total_return"]) if b else "—",
             None if b is None else m["total_return"] >= b["total_return"]),
            ("最大回撤", pct(m["max_dd"]), pct(b["max_dd"]) if b else "—",
             None if b is None else m["max_dd"] >= b["max_dd"]),
            ("夏普比率", num(m["sharpe"]), num(b["sharpe"]) if b else "—",
             None if b is None or m["sharpe"] != m["sharpe"] else m["sharpe"] >= b["sharpe"]),
            ("卡玛比率", num(m["calmar"]), num(b["calmar"]) if b else "—",
             None if b is None or m["calmar"] != m["calmar"] else m["calmar"] >= b["calmar"]),
            ("年化波动", pct(m["ann_vol"]), pct(b["ann_vol"]) if b else "—",
             None if b is None else m["ann_vol"] <= b["ann_vol"]),
            ("仓位时间占比", pct(m["exposure"]), "100.00%" if b else "—", None),
            ("　其中多头", pct(m["long_exposure"]), "100.00%" if b else "—", None),
            ("　其中空头", pct(m["short_exposure"]), "0.00%" if b else "—", None),
            ("交易次数", f"{m['n_trades']}", "1" if b else "—", None),
            ("　多头 / 空头", f"{m['n_long']} / {m['n_short']}", "1 / 0" if b else "—", None),
            ("胜率", pct(m["win_rate"]), "—", None),
            ("盈亏比", f"{m['profit_factor']:.2f}", "—", None),
            ("平均每笔收益", pct(m["avg_trade"]), "—", None),
            ("平均持仓", hold_text(m["avg_bars"], self.period, self.ctx.bars_per_year), "—", None),
            ("最佳单笔", pct(m["best_trade"]), "—", None),
            ("最差单笔", pct(m["worst_trade"]), "—", None),
            ("期末资金", f"${m['final_equity']:,.0f}",
             f"${b['final_equity']:,.0f}" if b else "—",
             None if b is None else m["final_equity"] >= b["final_equity"]),
        ]

        self.table.setRowCount(len(rows))
        for i, (name, sv, bv, better) in enumerate(rows):
            it0 = QTableWidgetItem(name)
            it0.setForeground(QColor("#475569"))
            it1 = QTableWidgetItem(sv)
            it1.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it1.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            if better is True:
                it1.setForeground(QColor("#0f9d58"))
            elif better is False:
                it1.setForeground(QColor("#d92b2b"))
            self.table.setItem(i, 0, it0)
            self.table.setItem(i, 1, it1)
            if b is not None:
                it2 = QTableWidgetItem(bv)
                it2.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                it2.setForeground(QColor(C_MUTED))
                self.table.setItem(i, 2, it2)
            self.table.setRowHeight(i, 26)

    # ---------------------------------------------------------- 交易记录

    def _refresh_trades(self) -> None:
        """逐笔交易表。**全部列出**，不截断。

        曾经只显示最近 500 笔，理由是「再多人也不会一页页翻」。但组合报告里三条
        策略的起点差七年，890 笔里最早的 390 笔被砍掉之后，表格第一行落在 2023
        年，而资金曲线是从 2016 年开始的 —— 看起来像「早年的交易丢了」。逐笔明细
        本来就是要能对着曲线查的，少一段就等于不能查，宁可让人滚鼠标。
        """
        tbl = getattr(self, "tbl_trades", None)
        if tbl is None:
            return
        # 按开仓时间排。单策略的 `res.trades` 是「平仓顺序」（平一笔追加一笔），
        # 组合的是 `combine_trades` 混排过的 —— 两边的行号对不上同一条时间轴，
        # 对着资金曲线查的时候很容易看串行。这里统一按开仓时间排。
        rows = sorted(self.trades, key=lambda nt: (nt[1].entry_date, nt[1].exit_date))
        with_name = self.kind == "portfolio"
        if with_name:
            # 组合报告的第二列是「哪条策略」，截面策略则是「哪个品种」
            head = "品种" if getattr(self.ctx, "cross", None) else "策略"
            self.tbl_trades.setHorizontalHeaderItem(1, QTableWidgetItem(head))
        n_closed = sum(1 for _n, t in rows if not t.open_position)
        n_open = len(rows) - n_closed
        wins = [t.ret for _n, t in rows if not t.open_position and t.ret > 0]
        losses = [t.ret for _n, t in rows if not t.open_position and t.ret <= 0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0

        if not rows:
            self.lbl_trades.setText("这次回测没有产生交易（信号全程为空仓，或者样本太短）")
            tbl.setRowCount(0)
            return
        self.lbl_trades.setText(
            f"共 {len(rows)} 笔（已平 {n_closed} 笔，持仓中 {n_open} 笔）"
            f" · 盈利 {len(wins)} 笔 / 亏损 {len(losses)} 笔"
            f" · 平均盈利 {avg_win*100:+.2f}% / 平均亏损 {avg_loss*100:+.2f}%"
            + ("　（组合：按策略分别平仓，不合并对冲）"
               if with_name and not getattr(self.ctx, "cross", None) else
               "　（截面：一笔 = 某品种一段连续同向持仓，权重变化不算新的一笔）"
               if with_name else "")
        )

        shown = rows
        tbl.setRowCount(len(shown))
        for i, (name, t) in enumerate(shown):
            d = "多" if t.direction > 0 else "空"
            note = []
            if t.open_position:
                note.append("持仓中")
            vals = [str(i + 1)]
            if with_name:
                vals.append(str(name))
            vals += [
                d,
                f"{t.entry_date:%Y-%m-%d %H:%M}" if not tqdata.is_daily(self.period)
                else f"{t.entry_date:%Y-%m-%d}",
                f"{t.entry_price:,.2f}",
                f"{t.exit_date:%Y-%m-%d %H:%M}" if not tqdata.is_daily(self.period)
                else f"{t.exit_date:%Y-%m-%d}",
                f"{t.exit_price:,.2f}",
                _lots_text(t.lots),
                f"{t.ret*100:+.2f}%",
                f"{t.bars}",
                "、".join(note),
            ]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j >= 3 and j != len(vals) - 1:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == len(vals) - 3:      # 收益列（手数插在它前面，倒数第三个不变）
                    it.setForeground(QColor(C_UP if t.ret >= 0 else C_DOWN))
                if t.open_position:
                    it.setForeground(QColor(C_MUTED))
                tbl.setItem(i, j, it)
            tbl.setRowHeight(i, 22)

    # ---------------------------------------------------------- 阶段分析

    def _refresh_stages(self) -> None:
        if self.res is None or not len(self.res.equity):
            return
        eq = self.res.equity.dropna()
        yrs = stage_table(eq, "YE")
        months = month_matrix(eq)

        self._draw_stage_chart(yrs)
        if yrs.empty:
            self.lbl_stage.setText("样本不足一年，没有阶段数据")
            self.tbl_month.setRowCount(0)
            return

        pos = int((yrs["ret"] > 0).sum())
        best = yrs.loc[yrs["ret"].idxmax()]
        worst = yrs.loc[yrs["ret"].idxmin()]
        self.lbl_stage.setText(
            f"{len(yrs)} 个自然年 · 盈利 {pos} 年 / 亏损 {len(yrs)-pos} 年"
            f" · 最好 {best['period']:%Y} 年 {best['ret']*100:+.1f}%"
            f" · 最差 {worst['period']:%Y} 年 {worst['ret']*100:+.1f}%"
        )

        self.tbl_month.setRowCount(len(months))
        for i, year in enumerate(months.index):
            it = QTableWidgetItem(f"{year}")
            it.setTextAlignment(Qt.AlignCenter)
            it.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            self.tbl_month.setItem(i, 0, it)
            for mth in range(1, 13):
                v = months.loc[year, mth] if mth in months.columns else np.nan
                if pd.isna(v):
                    cell = QTableWidgetItem("")
                else:
                    cell = QTableWidgetItem(f"{v*100:+.1f}")
                    cell.setForeground(QColor(C_UP if v >= 0 else C_DOWN))
                    cell.setBackground(QColor("#fef2f2" if v >= 0 else "#f0fdf4"))
                cell.setTextAlignment(Qt.AlignCenter)
                self.tbl_month.setItem(i, mth, cell)
            self.tbl_month.setRowHeight(i, 24)

    def _draw_stage_chart(self, yrs: pd.DataFrame) -> None:
        canvas = getattr(self, "canvas_s", None)
        if canvas is None:
            return
        if self._defer_draw(canvas, self._draw_stage_chart, yrs):
            return
        with canvas.frozen():
            fig = canvas.fresh()
            ax = fig.add_subplot(111)
            fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16)
            ax.set_facecolor(C_BG)
            if yrs is None or yrs.empty:
                ax.axis("off")
                ax.text(0.5, 0.5, "样本不足一年", ha="center", va="center",
                        color=C_MUTED, fontsize=11)
                return
            xs = np.arange(len(yrs))
            vals = yrs["ret"].to_numpy(float) * 100
            ax.bar(xs, vals, width=0.62, alpha=0.9,
                   color=[C_UP if v >= 0 else C_DOWN for v in vals])
            ax.axhline(0, color="#94a3b8", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{p:%Y}" for p in yrs["period"]], fontsize=8.5)
            ax.set_ylabel("年度收益 %", fontsize=9, color=C_MUTED)
            ax.set_title("年度收益", fontsize=11, loc="left", color=C_TEXT)
            ax.grid(True, axis="y", color=C_GRID, lw=0.5, alpha=0.8)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                ax.spines[spine].set_color("#cbd5e1")
            ax.tick_params(colors=C_MUTED, labelsize=8.5)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
            for x, v in zip(xs, vals):
                ax.text(x, v, f"{v:+.0f}", ha="center",
                        va="bottom" if v >= 0 else "top",
                        fontsize=7.5, color=C_MUTED)

    # ---------------------------------------------------------- 因子分析页

    def _ensure_report(self) -> pd.DataFrame:
        """当前标的的全因子评估表。按「市场+标的」缓存，切参数不用重算。

        47 个因子约 0.5 秒，每次拖滑杆都重算会明显拖慢交互，所以缓存住。
        """
        key = f"{self.market}:{self.ctx.symbol}"
        if self._report is not None and self._report_key == key:
            return self._report
        df = self.raw
        if df is None or df.empty or self.factor_cfg is None:
            return pd.DataFrame()
        cfg = self.factor_cfg
        try:
            rep = factor_metrics.factor_report(
                df, list(factors.FACTORS), horizon=cfg.horizon,
                window=cfg.window, method=cfg.method,
            )
        except Exception:  # noqa: BLE001
            rep = pd.DataFrame()
        self._report = rep
        self._report_key = key
        return rep

    def _refresh_factor_tab(self) -> None:
        """刷新因子分析页：表格 + 四格图。没有因子在用时也显示全因子一览。"""
        df = self.raw
        if df is None or df.empty:
            return
        cfg = self.factor_cfg
        if cfg is None:
            return
        # 整个页签（连算带画）一起推迟：`_ensure_report()` 要给 67 个因子各算一遍
        # IC / ICIR，实测 1.3 秒 —— 没人看这一页的时候算它纯属白等
        if self._defer_draw(self.canvas_f, self._refresh_factor_tab):
            return
        rep = self._ensure_report()
        if rep.empty:
            self.lbl_frep.setText("因子评估失败")
            return

        out = self.outcome
        selected = list(out.pack.fmat.columns) if out is not None else self.selected_keys
        sel_set = set(selected)

        self.tbl_factor.setRowCount(0)
        for _, r in rep.iterrows():
            i = self.tbl_factor.rowCount()
            self.tbl_factor.insertRow(i)
            on = r["key"] in sel_set

            def cell(text, align_right=True, bold=False, color=None):
                it = QTableWidgetItem(str(text))
                if align_right:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if bold:
                    f = it.font()
                    f.setBold(True)
                    it.setFont(f)
                if color:
                    it.setForeground(QColor(color))
                return it

            name = ("● " if on else "　") + str(r["label"])
            self.tbl_factor.setItem(i, 0, cell(name, False, on, "#1d4ed8" if on else None))

            ic = r.get("ic")
            if pd.notna(ic):
                ic_color = "#1d4ed8" if ic > 0 else "#b91c1c"
                self.tbl_factor.setItem(i, 1, cell(f"{ic:+.4f}", True, on, ic_color))
            elif r.get("key") in factors.AUX_KEYS:
                # 展期 / 期权因子在美股、以及没下过衍生数据的期货品种上是空的。
                # 留白会让人以为「算出来是 0」，写清楚是没数据。
                it = cell("无数据", True, False, "#94a3b8")
                it.setToolTip("该标的没有真实合约 / 期权数据，这个因子无法计算")
                self.tbl_factor.setItem(i, 1, it)
            else:
                self.tbl_factor.setItem(i, 1, cell("", True, on))

            for j, col in enumerate(("ir", "ic_pos", "t", "mono", "turnover"), start=2):
                v = r.get(col)
                if pd.isna(v) if isinstance(v, float) else v is None:
                    txt = ""
                elif col == "ic_pos":
                    txt = f"{v:.0%}"
                elif col == "t":
                    txt = f"{v:+.2f}"
                else:
                    txt = f"{v:+.3f}" if col in ("ir", "mono") else f"{v:.3f}"
                self.tbl_factor.setItem(i, j, cell(txt, True, on))
            self.tbl_factor.setRowHeight(i, 23)

        n_ok = int(rep["t"].notna().sum()) if "t" in rep else 0
        n_sig = int((rep["t"].abs() > 2).sum()) if "t" in rep else 0
        self.lbl_frep.setText(
            f"{len(rep)} 个因子 · 前向 {cfg.horizon} 日 · 标准化窗口 {cfg.window} 日\n"
            f"|t|>2 的因子 {n_sig} 个（Newey-West 调整后），有效评估 {n_ok} 个"
        )

        self._plot_factor_charts(df, cfg, rep, out)

    def _plot_factor_charts(self, df, cfg, rep, out) -> None:
        """四格图：IC 时序 / IC 衰减 / 分位收益 / 特征重要度（或分数与收益）。"""
        if self._defer_draw(self.canvas_f, self._plot_factor_charts, df, cfg, rep, out):
            return
        with self.canvas_f.frozen():
            self._plot_factor_charts_impl(df, cfg, rep, out)

    def _plot_factor_charts_impl(self, df, cfg, rep, out) -> None:
        fig = self.canvas_f.fresh()
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=0.075, right=0.975, top=0.93, bottom=0.09,
                            hspace=0.34, wspace=0.22)
        a_ic, a_decay, a_q, a_last = axes.flat

        for a in axes.flat:
            a.set_facecolor(C_BG)
            a.grid(True, color=C_GRID, lw=0.5, alpha=0.75)
            a.set_axisbelow(True)
            for spine in ("top", "right"):
                a.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                a.spines[spine].set_color("#cbd5e1")
            a.tick_params(colors=C_MUTED, labelsize=8.5, length=3)

        keys = list(out.pack.fmat.columns) if out is not None else self.selected_keys
        primary = keys[0] if keys else None

        # ---- 1. IC 时序
        if primary:
            f = factors.standardize(factors.raw(df, primary), window=cfg.window, method=cfg.method)
            fwd = factor_metrics.forward_return(df, cfg.horizon)
            ser = factor_metrics.ic_series(f, fwd, cfg.window).dropna()
            if len(ser) > 10:
                a_ic.plot(ser.index, ser.to_numpy(), color=C_IC, lw=0.9)
                a_ic.axhline(0, color="#94a3b8", lw=0.8)
                mu = float(ser.mean())
                a_ic.axhline(mu, color=C_IC_NEG, lw=0.9, ls="--", alpha=0.85)
                a_ic.set_title(
                    f"滚动 IC · {factors.FACTORS[primary].label}（均值 {mu:+.3f}）",
                    fontsize=9.5, color=C_TEXT, loc="left",
                )
            else:
                a_ic.set_title("滚动 IC（样本不足）", fontsize=9.5, color=C_TEXT, loc="left")
        else:
            a_ic.set_title("滚动 IC（未选因子）", fontsize=9.5, color=C_TEXT, loc="left")
        a_ic.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        # 20 年日频，默认定位器会塞十几个年份标签，在这么小的子图里会连成一片。
        # 限到 5 个刻度并只留年份，够看趋势又不糊。
        a_ic.xaxis.set_major_locator(AutoDateLocator(maxticks=5))
        a_ic.xaxis.set_major_formatter(DateFormatter("%Y"))
        a_ic.tick_params(axis="x", labelsize=8)

        # ---- 2. IC 衰减
        if primary:
            decay = factor_metrics.ic_decay(df, primary, window=cfg.window, method=cfg.method)
            xs = np.arange(len(decay))
            vals = decay.to_numpy(dtype=float)
            a_decay.bar(xs, vals, color=[C_IC if v >= 0 else C_IC_NEG for v in vals],
                        width=0.62, alpha=0.85)
            a_decay.axhline(0, color="#94a3b8", lw=0.8)
            a_decay.set_xticks(xs)
            a_decay.set_xticklabels([str(h) for h in decay.index])
            a_decay.set_xlabel("持有期（交易日）", fontsize=8.5, color=C_MUTED)
            a_decay.set_title("IC 衰减", fontsize=9.5, color=C_TEXT, loc="left")
        a_decay.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

        # ---- 3. 分位收益
        if primary:
            f = factors.standardize(factors.raw(df, primary), window=cfg.window, method=cfg.method)
            q = factor_metrics.quantile_returns(
                f, factor_metrics.forward_return(df, cfg.horizon), 5, cfg.window
            )
            if len(q):
                xs = np.arange(len(q))
                vals = q.to_numpy(dtype=float) * 100
                a_q.bar(xs, vals, color=[C_IC if v >= 0 else C_IC_NEG for v in vals],
                        width=0.6, alpha=0.85)
                a_q.axhline(0, color="#94a3b8", lw=0.8)
                a_q.set_xticks(xs)
                a_q.set_xticklabels([f"Q{i+1}" for i in range(len(q))])
                mono = factor_metrics.monotonicity(q)
                ttl = "分位收益" + ("" if pd.isna(mono) else f"（单调性 {mono:+.2f}）")
                a_q.set_title(ttl, fontsize=9.5, color=C_TEXT, loc="left")
                a_q.set_ylabel("%", fontsize=8.5, color=C_MUTED)
        a_q.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

        # ---- 4. 特征重要度 / 综合分数
        imp = out.pack.importance if out is not None else None
        if imp is not None and len(imp):
            top = imp.head(10)[::-1]
            names = [factors.FACTORS[k].label if k in factors.FACTORS else k for k in top.index]
            a_last.barh(np.arange(len(top)), top.to_numpy(), color=C_FACTOR, alpha=0.85)
            a_last.set_yticks(np.arange(len(top)))
            a_last.set_yticklabels(names, fontsize=8)
            a_last.set_title(
                f"因子重要度 · {mlmodels.MODELS.get(cfg.model, {}).get('label', cfg.model)}",
                fontsize=9.5, color=C_TEXT, loc="left",
            )
        elif out is not None:
            sc = out.pack.score.dropna()
            if len(sc) > 30:
                a_last.plot(sc.index, sc.to_numpy(), color=C_FACTOR, lw=0.8)
                a_last.axhline(0, color="#94a3b8", lw=0.8)
                for lv, col in ((cfg.entry_z, C_LONG_SIG), (-cfg.entry_z, C_SHORT_SIG)):
                    a_last.axhline(lv, color=col, lw=0.8, ls="--", alpha=0.75)
                a_last.set_title("综合因子分数", fontsize=9.5, color=C_TEXT, loc="left")
                a_last.xaxis.set_major_locator(AutoDateLocator(maxticks=5))
                a_last.xaxis.set_major_formatter(DateFormatter("%Y"))
                a_last.tick_params(axis="x", labelsize=8)
        else:
            a_last.set_title("特征重要度（机器学习模式可用）", fontsize=9.5,
                             color=C_MUTED, loc="left")

        for a in axes.flat:
            a.title.set_fontsize(9.5)

        notes = []
        if out is not None:
            notes.extend(out.notes)
        notes.append(
            "口径：IC 为秩相关；ICIR=滚动IC均值/标准差；t 值经 Newey-West 调整"
            "（滚动窗口重叠会严重高估显著性）；分位与方向均按 PIT 滚动计算"
        )
        self.lbl_fnote.setText(" · ".join(n for n in notes if n))

    # ---------------------------------------------------------- 截面因子分析

    def _ensure_cross_report(self, frames, ccfg, score):
        """算一遍横截面 IC，缓存起来。

        这个比时序那页便宜（因子数少），但也要遍历「因子 × 品种」，切页签时
        重算没有意义。缓存键带上分数面板的 id —— 同一条策略重跑会换一个新
        DataFrame，id 变了自然就重算。
        """
        keys = list(getattr(ccfg, "factor_keys", None) or [])
        key = f"{id(score)}|{getattr(ccfg, 'factor_mode', '')}|{','.join(keys)}"
        if self._cf_rep is not None and self._cf_key == key:
            return self._cf_rep
        try:
            rep = cross_engine.factor_report(
                frames, keys, score,
                fcfg=getattr(ccfg, "factor_cfg", None),
                reverse=getattr(ccfg, "factor_reverse", None),
            )
        except Exception as exc:  # noqa: BLE001
            rep = cross_engine.CrossReport(
                pd.DataFrame(), pd.Series(dtype=float),
                pd.Series(dtype=float), pd.Series(dtype=float),
                error=f"截面因子评估失败：{exc}",
            )
        self._cf_rep, self._cf_key = rep, key
        return rep

    def _refresh_cross_factor(self) -> None:
        """刷新截面因子分析页：表格 + 四格图。

        没有截面上下文（「多品种等权」组合）时不能装作有 —— 那个组合是每个
        品种各跑一遍时序策略再平均，根本没有「同一根 bar 上给品种排名」这回事。
        """
        tbl = getattr(self, "tbl_cfactor", None)
        if tbl is None:
            return
        ctx_cross = getattr(self.ctx, "cross", None)
        if not ctx_cross:
            self._cross_factor_placeholder()
            return
        # 算 + 画一起推迟：没人看这一页的时候算它纯属白等
        if self._defer_draw(self.canvas_cf, self._refresh_cross_factor):
            return

        frames, ccfg, score = ctx_cross
        rep = self._ensure_cross_report(frames, ccfg, score)
        if rep is None or not rep.ok:
            msg = (getattr(rep, "error", "") if rep is not None else "") \
                or "截面因子评估失败"
            self.lbl_cfrep.setText(msg)
            tbl.setRowCount(0)
            return

        def cell(text, align_right=True, bold=False, color=None):
            it = QTableWidgetItem(str(text))
            if align_right:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if bold:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
            if color:
                it.setForeground(QColor(color))
            return it

        tbl.setRowCount(0)
        for key, r in rep.table.iterrows():
            i = tbl.rowCount()
            tbl.insertRow(i)
            # 合成分数单独一行：它是真正拿去排序的东西，得一眼能找出来
            main = key == "__score__"
            name = ("★ " if main else "　") + str(r["label"])
            tbl.setItem(i, 0, cell(name, False, main,
                                   C_FACTOR if main else None))

            ic = r.get("ic")
            if pd.notna(ic):
                ic_color = C_IC if ic > 0 else C_IC_NEG
                tbl.setItem(i, 1, cell(f"{ic:+.4f}", True, main, ic_color))
            else:
                tbl.setItem(i, 1, cell("", True, main))

            for j, col in enumerate(("ir", "ic_pos", "t", "cover", "n"), start=2):
                v = r.get(col)
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    txt = ""
                elif col == "ic_pos":
                    txt = f"{v:.0%}"
                elif col == "cover":
                    txt = f"{v:.0%}"
                elif col == "n":
                    txt = f"{int(v)}"
                elif col == "t":
                    txt = f"{v:+.2f}"
                else:
                    txt = f"{v:+.3f}"
                tbl.setItem(i, j, cell(txt, True, main))
            tbl.setRowHeight(i, 23)

        n_sig = int((rep.table["t"].abs() > 2).sum())
        self.lbl_cfrep.setText(
            f"{len(rep.table) - 1} 个因子 · 前向 {rep.horizon} 根 · "
            f"{rep.n_bars} 根 bar · {len(frames)} 个品种\n"
            f"|t|>2 的因子 {n_sig} 个（Newey-West 调整后）"
        )
        self._plot_cross_factor_charts(frames, ccfg, rep)

    def _cross_factor_placeholder(self) -> None:
        """非截面策略：把这一页说明白，而不是画一张空图。"""
        self.lbl_cfrep.setText("这一页只对截面策略有效")
        self.lbl_cfnote.setText(
            "多品种等权组合不是截面策略 —— 它是每个品种各跑一遍时序策略再把"
            "资金曲线平均，不存在「同一根 bar 上给品种排名」这回事，也就没有"
            "横截面 IC 可算。想看横截面 IC，请在「策略研究」里把策略族切到"
            "「截面对冲」或「截面选品种」。"
        )
        self.tbl_cfactor.setRowCount(0)
        with self.canvas_cf.frozen():
            fig = self.canvas_cf.fresh()
            a = fig.add_subplot(111)
            a.set_facecolor(C_BG)
            a.set_xticks([])
            a.set_yticks([])
            for spine in a.spines.values():
                spine.set_visible(False)
            a.text(0.5, 0.52, "非截面策略", ha="center", va="center",
                   fontsize=15, color=C_TEXT, transform=a.transAxes)
            a.text(0.5, 0.44, "没有横截面因子分析", ha="center", va="center",
                   fontsize=10.5, color=C_MUTED, transform=a.transAxes)

    def _plot_cross_factor_charts(self, frames, ccfg, rep) -> None:
        """四格图：截面 IC 时序 / 分位收益 / 多空价差 / 各因子 IC 条形。"""
        if self._defer_draw(self.canvas_cf, self._plot_cross_factor_charts,
                            frames, ccfg, rep):
            return
        with self.canvas_cf.frozen():
            self._plot_cross_factor_charts_impl(frames, ccfg, rep)

    def _plot_cross_factor_charts_impl(self, frames, ccfg, rep) -> None:
        fig = self.canvas_cf.fresh()
        axes = fig.subplots(2, 2)
        fig.subplots_adjust(left=0.085, right=0.975, top=0.93, bottom=0.10,
                            hspace=0.36, wspace=0.26)
        a_ic, a_q, a_sp, a_bar = axes.flat

        for a in axes.flat:
            a.set_facecolor(C_BG)
            a.grid(True, color=C_GRID, lw=0.5, alpha=0.75)
            a.set_axisbelow(True)
            for spine in ("top", "right"):
                a.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                a.spines[spine].set_color("#cbd5e1")
            a.tick_params(colors=C_MUTED, labelsize=8.5, length=3)

        ic = rep.ic.dropna() if rep.ic is not None else pd.Series(dtype=float)
        span_days = 0.0
        if len(ic) > 1:
            try:
                span_days = (ic.index[-1] - ic.index[0]) / pd.Timedelta(days=1)
            except TypeError:
                span_days = float(len(ic))

        def date_axis(a):
            # `maxticks=5` 在 2~3 年的区间上会让 AutoDateLocator 挑不出合适的
            # 刻度间隔（它会打一条 "unable to pick an appropriate interval"
            # 的警告，然后退回一个很怪的默认值）。截面策略的回测窗口常常正好
            # 是这种长度，所以这里给一个 3~7 的区间
            a.xaxis.set_major_locator(AutoDateLocator(minticks=3, maxticks=7))
            # 只写年份时，一年里的几个刻度会全变成同一个「2023」，看着像重影。
            # 跨度越短，标签就写得越细
            if span_days <= 400:
                fmt = "%m-%d"
            elif span_days <= 2000:
                fmt = "%y/%m"
            else:
                fmt = "%Y"
            a.xaxis.set_major_formatter(DateFormatter(fmt))
            a.tick_params(axis="x", labelsize=8)

        # ---- 1. 合成分数的逐期截面 IC
        if len(ic) > 10:
            win = max(5, min(60, len(ic) // 8))
            a_ic.plot(ic.index, ic.to_numpy(), color=C_IC, lw=0.7, alpha=0.45)
            smooth = ic.rolling(win, min_periods=max(2, win // 2)).mean()
            a_ic.plot(smooth.index, smooth.to_numpy(), color=C_IC, lw=1.4)
            a_ic.axhline(0, color="#94a3b8", lw=0.8)
            mu = float(ic.mean())
            a_ic.axhline(mu, color=C_IC_NEG, lw=0.9, ls="--", alpha=0.85)
            a_ic.set_title(
                f"合成分数 · 逐期截面 IC（均值 {mu:+.3f}，{win} 期平滑）",
                fontsize=9.5, color=C_TEXT, loc="left",
            )
            date_axis(a_ic)
        else:
            a_ic.set_title("合成分数 · 逐期截面 IC（样本不足）",
                           fontsize=9.5, color=C_MUTED, loc="left")
        a_ic.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

        # ---- 2. 分位收益：Q1 最弱、Qq 最强，柱子该一级级抬高
        q = rep.quantile.dropna() if rep.quantile is not None else pd.Series(dtype=float)
        if len(q):
            xs = np.arange(len(q))
            vals = q.to_numpy(dtype=float) * 100
            a_q.bar(xs, vals, color=[C_IC if v >= 0 else C_IC_NEG for v in vals],
                    width=0.62, alpha=0.85)
            a_q.axhline(0, color="#94a3b8", lw=0.8)
            a_q.set_xticks(xs)
            a_q.set_xticklabels([f"Q{int(b)}" for b in q.index])
            mono = factor_metrics.monotonicity(q)
            ttl = "分位平均相对收益"
            if not pd.isna(mono):
                ttl += f"（单调性 {mono:+.2f}）"
            a_q.set_title(ttl, fontsize=9.5, color=C_TEXT, loc="left")
            a_q.set_ylabel("%", fontsize=8.5, color=C_MUTED)
        else:
            a_q.set_title("分位平均相对收益（样本不足）",
                          fontsize=9.5, color=C_MUTED, loc="left")
        a_q.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

        # ---- 3. 多空价差累计净值：只做「最强一组减最弱一组」，不含择时
        sp = rep.spread.dropna() if rep.spread is not None else pd.Series(dtype=float)
        if len(sp) > 2:
            nav = sp.to_numpy(dtype=float)
            a_sp.plot(sp.index, nav, color=C_FACTOR, lw=1.2)
            a_sp.fill_between(sp.index, 1.0, nav, color=C_FACTOR, alpha=0.12)
            a_sp.axhline(1.0, color="#94a3b8", lw=0.8, ls="--")
            tot = float(nav[-1] - 1.0)
            a_sp.set_title(
                f"多空价差 · 最强一组减最弱一组（累计 {tot*100:+.1f}%）",
                fontsize=9.5, color=C_TEXT, loc="left",
            )
            date_axis(a_sp)
        else:
            a_sp.set_title("多空价差（样本不足）", fontsize=9.5,
                           color=C_MUTED, loc="left")
        a_sp.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))

        # ---- 4. 各因子横截面 IC：横向条形，一眼看出谁在拖后腿
        tab = rep.table.dropna(subset=["ic"]) if rep.table is not None else pd.DataFrame()
        tab = tab[tab["ic"].notna()]
        if len(tab):
            tab = tab.sort_values("ic")
            names = [("★ " if k == "__score__" else "") + str(r["label"])
                     for k, r in tab.iterrows()]
            vals = tab["ic"].to_numpy(dtype=float)
            colors = [C_FACTOR if k == "__score__"
                      else (C_IC if v >= 0 else C_IC_NEG)
                      for k, v in zip(tab.index, vals)]
            ys = np.arange(len(tab))
            a_bar.barh(ys, vals, color=colors, alpha=0.88)
            a_bar.axvline(0, color="#94a3b8", lw=0.8)
            a_bar.set_yticks(ys)
            a_bar.set_yticklabels(names, fontsize=8)
            a_bar.set_title("各因子截面 IC", fontsize=9.5, color=C_TEXT, loc="left")
            a_bar.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
            if len(tab) > 9:
                a_bar.tick_params(axis="y", labelsize=7)
        else:
            a_bar.set_title("各因子截面 IC（样本不足）", fontsize=9.5,
                            color=C_MUTED, loc="left")

        for a in axes.flat:
            a.title.set_fontsize(9.5)

        notes = [
            "口径：每根 bar 上先给这批品种的因子值和下一段相对收益各排一次名次，"
            "再算两者的秩相关；相对收益 = 该品种前向收益 − 当期截面均值"
        ]
        wname = {"risk": "风险平价", "equal": "等权", "score": "按分数"}.get(
            getattr(ccfg, "weight", "risk"), "")
        if wname:
            notes.append(f"实际下单的资金分配是「{wname}」，这张图只看因子排序能力，不含权重")
        notes.append("多空价差按「最强一组减最弱一组」等权合成，不含调仓成本和滑点")
        self.lbl_cfnote.setText(" · ".join(n for n in notes if n))

    # ---------------------------------------------------------- 实时

    def patch_last_bar(self, merged: pd.DataFrame) -> None:
        """实时推送：把还没走完的那根 bar 的最新价刷到图上，不重算策略。

        只有用户正停在最新一段时才动 —— 滚到历史位置时窗口最后一根是历史 bar，
        改它等于篡改历史。
        """
        view = self._k_view
        if self.df is None or view is None or not len(view):
            return
        ts = merged.index[-1]
        if view.index[-1] != ts:
            return
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in merged.columns]
        for c in cols:
            if c in self.df.columns:
                self.df.loc[ts, c] = merged.loc[ts, c]
            view.loc[ts, c] = merged.loc[ts, c]
        self._render_kline()
