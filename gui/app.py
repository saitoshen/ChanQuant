"""美股趋势跟踪研究台 —— PySide6 桌面程序。

功能：
- 左侧选择标的 / 策略 / 参数 / 显示区间
- 「K线与信号」页：后复权 K 线、策略指标线、持仓区间底色、买卖信号标记、成交量
- 「绩效对比」页：策略 vs 买入持有的资金曲线、回撤曲线、指标对比表
"""

from __future__ import annotations

import json
import os
import sys

# 直接运行 gui/app.py 时，sys.path[0] 是 gui/ 目录，需要把项目根目录补进去，
# 否则 from ttrade.* 会报 ModuleNotFoundError（双击 bat 启动时同样受影响）。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrade.paths import bundle_root, cache_dir, data_dir as resolve_data_dir, log_dir

# 资源根目录：源码运行时是项目目录，PyInstaller 打包后是 sys._MEIPASS。
# 数据目录、matplotlib 缓存都必须走 ttrade.paths，不能再用 ROOT 拼——
# 打包成 macOS .app 后 ROOT 所在的卷是只读的，写入会静默失败。
ASSET_ROOT = bundle_root()

# matplotlib 缓存目录必须在导入 pyplot 之前指定。放在用户缓存目录而不是包内，
# 因为 .app 签名后不可写，且字体缓存重建一次要几秒。
os.environ.setdefault("MPLCONFIGDIR", cache_dir("mpl"))

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ttrade.backtest import BacktestConfig, buy_and_hold, drawdown_series, run_backtest
from ttrade.eastmoney import load_local
from ttrade.strategies import STRATEGIES, run_strategy
from ttrade.universe import BENCHMARK, UNIVERSE, cn_name, sector_of

DATA_DIR = resolve_data_dir("us")

# ------------------------------------------------------------------ 样式常量
C_BG = "#ffffff"
C_PANEL = "#f8fafc"
C_GRID = "#e5e7eb"
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_UP = "#d92b2b"      # 红涨
C_DOWN = "#0f9d58"    # 绿跌
C_STRAT = "#2563eb"
C_BH = "#94a3b8"
C_ENTRY = "#059669"
C_EXIT = "#dc2626"
C_HOLD_BG = "#3b82f6"
C_SHORT_BG = "#ef4444"

from ttrade.plotstyle import setup_matplotlib_fonts

setup_matplotlib_fonts()
# 大样本加速：路径简化 + 分块渲染，5000 根 K 线时可显著降低绘制开销
plt.rcParams["path.simplify"] = True
plt.rcParams["path.simplify_threshold"] = 0.6
plt.rcParams["agg.path.chunksize"] = 20000


def setup_light_palette(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f1f5f9"))
    pal.setColor(QPalette.WindowText, QColor(C_TEXT))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f8fafc"))
    pal.setColor(QPalette.Text, QColor(C_TEXT))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ButtonText, QColor(C_TEXT))
    pal.setColor(QPalette.Highlight, QColor("#2563eb"))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipText, QColor(C_TEXT))
    app.setPalette(pal)


STYLE_SHEET = """
QWidget { font-family: 'Microsoft YaHei', 'Segoe UI'; font-size: 12px; color: #1f2937; }
QMainWindow, QWidget#SidePanel { background: #f1f5f9; }
QScrollArea#SideScroll, QScrollArea#SideScroll > QWidget > QWidget { background: transparent; border: none; }
QWidget#Card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; }
QLabel#SectionTitle { font-size: 11px; font-weight: 600; color: #64748b; letter-spacing: 1px; }
QLabel#AppTitle { font-size: 17px; font-weight: 700; color: #0f172a; }
QLabel#AppSub { font-size: 11px; color: #64748b; }
QLabel#MetricValue { font-size: 16px; font-weight: 700; }
QLabel#MetricLabel { font-size: 10px; color: #64748b; }
QLabel#Hint { font-size: 11px; color: #64748b; }
QComboBox, QSpinBox, QDoubleSpinBox {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;
    padding: 3px 8px; min-height: 22px;
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #94a3b8; }
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #2563eb; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView {
    background: #ffffff; border: 1px solid #cbd5e1; selection-background-color: #dbeafe;
    selection-color: #1e40af; outline: none;
}
QPushButton {
    background: #2563eb; color: #ffffff; border: none; border-radius: 6px;
    padding: 7px 14px; font-weight: 600;
}
QPushButton:hover { background: #1d4ed8; }
    QPushButton:pressed { background: #1e40af; }
QPushButton#Ghost { background: #ffffff; color: #334155; border: 1px solid #cbd5e1; }
QPushButton#Ghost:hover { background: #f8fafc; border-color: #94a3b8; }
QSlider::groove:horizontal { height: 4px; background: #e2e8f0; border-radius: 2px; }
QSlider::sub-page:horizontal { background: #93c5fd; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #2563eb; width: 13px; height: 13px; margin: -5px 0; border-radius: 6px;
}
QSlider::handle:horizontal:hover { background: #1d4ed8; }
QTabWidget::pane { border: 1px solid #e2e8f0; border-radius: 8px; background: #ffffff; top: -1px; }
QTabBar::tab {
    background: transparent; color: #64748b; padding: 8px 18px; border: none;
    border-bottom: 2px solid transparent; font-weight: 600;
}
QTabBar::tab:selected { color: #2563eb; border-bottom: 2px solid #2563eb; }
QTabBar::tab:hover { color: #1f2937; }
QTableWidget { background: #ffffff; border: none; gridline-color: #eef2f7; }
QTableWidget::item { padding: 4px 6px; }
QHeaderView::section {
    background: #f8fafc; color: #475569; border: none; border-bottom: 1px solid #e2e8f0;
    padding: 6px; font-weight: 600; font-size: 11px;
}
QSplitter::handle { background: #e2e8f0; }
QSplitter::handle:horizontal { width: 4px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #cbd5e1; border-radius: 5px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #94a3b8; }
QScrollBar::add-line, QScrollBar::sub-line { height: 0; }
"""


# ------------------------------------------------------------------ 数据仓库
FUT_DIR = resolve_data_dir("cn")

EXCHANGE_CN = {
    "SHFE": "上期所",
    "DCE": "大商所",
    "CZCE": "郑商所",
    "CFFEX": "中金所",
    "INE": "能源中心",
    "GFEX": "广期所",
}

# 市场元信息：目录、副标题、价格说明、回测成本假设
MARKETS = {
    "us": {
        "label": "美股",
        "dir": DATA_DIR,
        "sub": "美股 · 20年日线 · 后复权 · 东方财富",
        "price_note": "价格（后复权）",
        "fee": 0.0005,
        "slippage": 0.0005,
        "allow_short": False,
        "legend": (
            "▲ 绿色向上三角＝开多　▼ 绿色向下三角＝平多　"
            "浅蓝底色＝多头持仓　价格均为后复权（含分红再投资），保证收益率连续"
        ),
    },
    "cn": {
        "label": "国内期货",
        "dir": FUT_DIR,
        "sub": "国内期货 · 20年日线 · 主力连续 · 前复权 · Tushare",
        "price_note": "价格（前复权·主力连续）",
        "fee": 0.0001,
        "slippage": 0.0002,
        "allow_short": True,
        "legend": (
            "▲绿＝开多　▼绿＝平多　▼红＝开空　▲红＝平空　"
            "浅蓝底色＝多头持仓　浅红底色＝空头持仓　价格已按 Tushare 复权因子消除换月跳空"
        ),
    },
}


def load_futures_csv(key: str, data_dir: str) -> pd.DataFrame:
    """读取期货 CSV（key 形如 CU_SHFE）。不依赖 tushare，避免 GUI 启动时加载重依赖。

    剔除价格非正的占位行：部分品种（动力煤、强麦等）在无成交日返回 0 价，
    留在序列里会让 K 线塌到 0、指标失真。

    复权：OHLC 乘以 `adj_factor / adj_factor[-1]`（前复权），最后一根 K 线等于
    真实盘面价，历史价格按换月价差回溯调整，K 线上不会出现换月跳空。
    成交量/成交额/持仓量不复权。早期下载的 CSV 若没有 adj_factor 列，按不复权处理。
    """
    path = os.path.join(data_dir, f"{key}.csv")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    px = df[["open", "high", "low", "close"]]
    df = df[px.gt(0).all(axis=1)].reset_index(drop=True)

    if "adj_factor" in df.columns and len(df):
        f = pd.to_numeric(df["adj_factor"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        base = f[-1] if f[-1] > 0 else 1.0
        f = f / base
        for col in ("open", "high", "low", "close"):
            df[col] = df[col].to_numpy(dtype=float) * f

    return df.set_index("date")


class DataStore:
    """本地 CSV 缓存，按需读取。支持美股 / 国内期货两个市场。"""

    def __init__(self, market: str = "us"):
        self.market = market
        self.data_dir = MARKETS[market]["dir"]
        self._cache: dict[str, pd.DataFrame] = {}
        self.items: list[dict] = []
        self._scan()

    @property
    def symbols(self) -> list[str]:
        return [it["key"] for it in self.items]

    def _scan(self) -> None:
        if not os.path.isdir(self.data_dir):
            return
        if self.market == "us":
            self._scan_us()
        else:
            self._scan_futures()

    def _scan_us(self) -> None:
        found = []
        for fn in os.listdir(self.data_dir):
            if fn.endswith(".csv") and not fn.startswith("_"):
                found.append(fn[:-4].replace("_", "."))
        order = {s: i for i, s in enumerate(UNIVERSE.keys())}
        found.sort(key=lambda s: (order.get(s, 9999), s))
        for sym in found:
            name = cn_name(sym) or ""
            sector = sector_of(sym)
            label = f"{sym}　{name}" + (f"　·　{sector}" if sector else "")
            self.items.append(
                {"key": sym, "label": label, "symbol": sym, "name": name, "exchange": ""}
            )

    def _scan_futures(self) -> None:
        """按清单加载，并与批量回测共用同一套「活跃 + 流动性」筛选规则。"""
        rows: list[dict] = []
        try:
            from ttrade.futures import list_local

            rows = list_local(self.data_dir)
        except Exception:
            rows = []
        if not rows:
            for fn in os.listdir(self.data_dir):
                if fn.endswith(".csv") and not fn.startswith("_"):
                    stem = fn[:-4]
                    if "_" in stem:
                        sym, ex = stem.rsplit("_", 1)
                        rows.append(
                            {"symbol": sym, "name": "", "exchange": ex,
                             "exchange_cn": EXCHANGE_CN.get(ex, ex)}
                        )
        ex_order = {"SHFE": 0, "DCE": 1, "CZCE": 2, "CFFEX": 3, "INE": 4, "GFEX": 5}
        rows.sort(key=lambda r: (ex_order.get(r.get("exchange"), 9), r.get("symbol", "")))
        for r in rows:
            sym, ex = r.get("symbol", ""), r.get("exchange", "")
            key = r.get("key") or f"{sym}_{ex}"
            if not os.path.exists(os.path.join(self.data_dir, f"{key}.csv")):
                continue
            cn = r.get("exchange_cn") or EXCHANGE_CN.get(ex, ex)
            sector = r.get("sector") or ""
            label = f"{sym}　{r.get('name', '')}　·　{sector or cn}"
            self.items.append(
                {"key": key, "label": label, "symbol": sym,
                 "name": r.get("name", ""), "exchange": ex}
            )

    def get(self, key: str) -> pd.DataFrame | None:
        if key in self._cache:
            return self._cache[key]
        try:
            if self.market == "us":
                df = load_local(key, self.data_dir)
            else:
                df = load_futures_csv(key, self.data_dir)
        except Exception:
            return None
        self._cache[key] = df
        return df


# ------------------------------------------------------------------ 图表基类
class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, figsize=(10, 6), dpi=100):
        self.fig = Figure(figsize=figsize, dpi=dpi, facecolor=C_BG)
        super().__init__(self.fig)
        self.setStyleSheet("background: #ffffff;")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


# ------------------------------------------------------------------ 指标卡片
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


# ------------------------------------------------------------------ 主窗口
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("趋势跟踪研究台 · 美股 / 国内期货")
        # 自适应屏幕：小屏（如 1536x864）下不超出可视区域
        screen = QApplication.primaryScreen()
        avail = screen.availableGeometry() if screen else None
        w = min(1480, int(avail.width() * 0.94)) if avail else 1480
        h = min(940, int(avail.height() * 0.92)) if avail else 900
        self.resize(max(w, 1180), max(h, 720))
        self.setMinimumSize(1100, 700)

        self.stores = {k: DataStore(k) for k in MARKETS}
        self.market = "us"
        self.store = self.stores[self.market]
        self.cfg = BacktestConfig(
            fee_rate=MARKETS["us"]["fee"], slippage=MARKETS["us"]["slippage"],
            long_only=not MARKETS["us"]["allow_short"],
        )
        self.df_raw: pd.DataFrame | None = None
        self.df_sig: pd.DataFrame | None = None
        self.res = None
        self.bh = None
        self.param_widgets: dict[str, QWidget] = {}
        self.param_scales: dict[str, int] = {}
        self._pending = False

        self._build_ui()
        self._populate_symbols()

    # ---------------------------------------------------------- UI 构建
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(12)

        root.addWidget(self._build_sidebar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_chart_tab(), "K线与信号")
        self.tabs.addTab(self._build_perf_tab(), "绩效对比")
        root.addWidget(self.tabs, 1)

    def _build_sidebar(self) -> QWidget:
        content = QWidget()
        content.setObjectName("SidePanel")
        lay = QVBoxLayout(content)
        lay.setContentsMargins(0, 0, 6, 0)
        lay.setSpacing(10)
        # 关键：让布局的最小尺寸等于内容真实尺寸，否则 QScrollArea 会把控件压扁
        lay.setSizeConstraint(QLayout.SetMinimumSize)

        # 标题
        head = QWidget()
        hl = QVBoxLayout(head)
        hl.setContentsMargins(4, 0, 4, 4)
        hl.setSpacing(2)
        t = QLabel("趋势跟踪研究台")
        t.setObjectName("AppTitle")
        self.lbl_sub = QLabel(MARKETS[self.market]["sub"])
        self.lbl_sub.setObjectName("AppSub")
        self.lbl_sub.setWordWrap(True)
        hl.addWidget(t)
        hl.addWidget(self.lbl_sub)
        lay.addWidget(head)

        # 选择卡片
        card = QFrame()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 12, 12, 12)
        cl.setSpacing(8)

        cl.addWidget(self._section("市场"))
        self.cb_market = QComboBox()
        for mk, info in MARKETS.items():
            n = len(self.stores[mk].items)
            self.cb_market.addItem(f"{info['label']}（{n} 个标的）", mk)
        cl.addWidget(self.cb_market)

        cl.addWidget(self._section("标的"))
        self.cb_symbol = QComboBox()
        self.cb_symbol.setMaxVisibleItems(18)
        cl.addWidget(self.cb_symbol)

        cl.addWidget(self._section("策略"))
        self.cb_strategy = QComboBox()
        for key, spec in STRATEGIES.items():
            self.cb_strategy.addItem(spec.label, key)
        cl.addWidget(self.cb_strategy)

        self.lbl_desc = QLabel("")
        self.lbl_desc.setObjectName("Hint")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setMinimumHeight(44)
        cl.addWidget(self.lbl_desc)

        cl.addWidget(self._section("参数"))
        self.param_box = QVBoxLayout()
        self.param_box.setContentsMargins(0, 0, 0, 0)
        self.param_box.setSpacing(8)
        cl.addLayout(self.param_box)

        cl.addWidget(self._section("显示区间"))
        self.cb_range = QComboBox()
        for text, years in [("近 1 年", 1), ("近 2 年", 2), ("近 3 年", 3),
                            ("近 5 年", 5), ("近 10 年", 10), ("全部历史", 0)]:
            self.cb_range.addItem(text, years)
        self.cb_range.setCurrentIndex(2)
        cl.addWidget(self.cb_range)

        self.chk_ma = QComboBox()
        cl.addWidget(self._section("K线叠加"))
        self.chk_ma.addItem("显示策略指标线", True)
        self.chk_ma.addItem("仅 K 线", False)
        cl.addWidget(self.chk_ma)

        cl.addWidget(self._section("交易方向"))
        self.chk_short = QCheckBox("允许做空（多空双向）")
        self.chk_short.setChecked(False)
        self.chk_short.setToolTip(
            "勾选后策略会同时做多和做空（signal 取 -1/0/1）；\n"
            "不勾选则把空头信号裁掉，只做多。期货默认勾选。"
        )
        cl.addWidget(self.chk_short)

        lay.addWidget(card)

        # 状态
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setObjectName("Hint")
        self.lbl_status.setWordWrap(True)
        lay.addWidget(self.lbl_status)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_refresh = QPushButton("重新计算")
        self.btn_refresh.setObjectName("Ghost")
        self.btn_refresh.clicked.connect(self.run_analysis)
        btn_row.addWidget(self.btn_refresh)
        lay.addLayout(btn_row)

        # 放进滚动容器：小屏（如 1536x864）下控件不会被挤压重叠
        scroll = QScrollArea()
        scroll.setObjectName("SideScroll")
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(296)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        return scroll

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionTitle")
        return lbl

    def _build_chart_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        self.lbl_chart_title = QLabel("")
        self.lbl_chart_title.setStyleSheet(
            "font-size: 13px; font-weight: 600; color: #0f172a; padding-left: 4px;"
        )
        lay.addWidget(self.lbl_chart_title)

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

        self.lbl_legend = QLabel(MARKETS[self.market]["legend"])
        self.lbl_legend.setObjectName("Hint")
        self.lbl_legend.setStyleSheet("padding-left: 4px;")
        self.lbl_legend.setWordWrap(True)
        lay.addWidget(self.lbl_legend)
        return w

    def _build_perf_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.canvas_p = MplCanvas(figsize=(7, 6), dpi=100)
        splitter.addWidget(self.canvas_p)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.setSpacing(6)
        cap = QLabel("绩效指标对比")
        cap.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        rl.addWidget(cap)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["指标", "趋势策略", "买入持有"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setShowGrid(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        rl.addWidget(self.table, 1)
        right.setMinimumWidth(360)
        right.setMaximumWidth(470)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([700, 400])

        lay.addWidget(splitter)
        return w

    # ---------------------------------------------------------- 数据填充
    def _populate_symbols(self) -> None:
        # 信号只连一次，避免切换市场后重复触发
        if not getattr(self, "_wired", False):
            self.cb_market.currentIndexChanged.connect(self._on_market_changed)
            self.cb_symbol.currentIndexChanged.connect(self._schedule)
            self.cb_strategy.currentIndexChanged.connect(self._on_strategy_changed)
            self.cb_range.currentIndexChanged.connect(self._schedule)
            self.chk_ma.currentIndexChanged.connect(self._schedule)
            self.chk_short.toggled.connect(self._on_short_toggled)
            self._wired = True

        self._fill_symbols()
        self._on_strategy_changed()
        self._schedule(immediate=True)

    def _fill_symbols(self) -> None:
        self.cb_symbol.blockSignals(True)
        self.cb_symbol.clear()
        for it in self.store.items:
            self.cb_symbol.addItem(it["label"], it["key"])
        if self.store.items:
            default = "AAPL" if self.market == "us" else "CU_SHFE"
            idx = self.cb_symbol.findData(default)
            self.cb_symbol.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_symbol.blockSignals(False)

    def _short_enabled(self) -> bool:
        """当前是否允许做空。美股默认不允许，期货默认允许。"""
        if not hasattr(self, "chk_short"):
            return bool(MARKETS.get(self.market, {}).get("allow_short", False))
        return self.chk_short.isChecked()

    def _on_short_toggled(self, _checked: bool) -> None:
        info = MARKETS[self.market]
        self.cfg = BacktestConfig(
            fee_rate=info["fee"], slippage=info["slippage"],
            long_only=not self._short_enabled(),
        )
        self._schedule(immediate=True)

    def _on_market_changed(self) -> None:
        mk = self.cb_market.currentData()
        if mk == self.market:
            return
        self.market = mk
        self.store = self.stores[mk]
        info = MARKETS[mk]
        if hasattr(self, "chk_short"):
            self.chk_short.blockSignals(True)
            self.chk_short.setChecked(bool(info["allow_short"]))
            self.chk_short.blockSignals(False)
        self.cfg = BacktestConfig(
            fee_rate=info["fee"], slippage=info["slippage"],
            long_only=not (info["allow_short"] and self._short_enabled()),
        )
        self.lbl_sub.setText(info["sub"])
        self.lbl_legend.setText(info["legend"])
        self._fill_symbols()
        if not self.store.items:
            cmd = "python download.py" if mk == "us" else "python download_futures.py"
            self.lbl_status.setText(f"{info['label']} 本地无数据，请先运行 {cmd}")
            return
        self._schedule(immediate=True)

    def _on_strategy_changed(self) -> None:
        key = self.cb_strategy.currentData()
        spec = STRATEGIES[key]
        self.lbl_desc.setText(spec.desc)
        self._rebuild_params(spec)
        self._schedule(immediate=True)

    def _rebuild_params(self, spec) -> None:
        while self.param_box.count():
            item = self.param_box.takeAt(0)
            wdg = item.widget()
            if wdg:
                wdg.deleteLater()
        self.param_widgets.clear()
        self.param_scales.clear()

        labels = {
            "fast": "快线周期", "slow": "慢线周期", "signal_n": "信号周期",
            "entry": "入场周期", "exit": "离场周期", "n": "通道周期",
            "k": "ATR 倍数", "atr_n": "ATR 周期", "atr_mult": "止损倍数",
        }
        for pname, default in spec.params.items():
            lo, hi, step = spec.param_meta.get(pname, (1, 100, 1))
            is_float = isinstance(default, float)
            scale = 10 if is_float else 1

            row = QWidget()
            row.setFixedHeight(28)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(6)

            lbl = QLabel(labels.get(pname, pname))
            lbl.setFixedWidth(62)
            lbl.setFixedHeight(28)
            lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")

            sld = QSlider(Qt.Horizontal)
            sld.setRange(int(round(lo * scale)), int(round(hi * scale)))
            sld.setSingleStep(int(round(step * scale)))
            sld.setPageStep(int(round(step * scale)) * 2)
            sld.setValue(int(round(float(default) * scale)))
            sld.setFixedHeight(24)
            sld.setCursor(Qt.PointingHandCursor)

            val = QLabel(f"{default:g}")
            val.setFixedWidth(36)
            val.setFixedHeight(20)
            val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            val.setStyleSheet(
                "color: #1e293b; font-size: 11px; font-weight: 600;"
                "background: #f1f5f9; border-radius: 4px; padding: 1px 4px;"
            )

            def on_change(v, _s=scale, _l=val):
                _l.setText(f"{v / _s:g}")

            sld.valueChanged.connect(on_change)
            sld.valueChanged.connect(self._schedule)

            rl.addWidget(lbl)
            rl.addWidget(sld, 1)
            rl.addWidget(val)

            self.param_box.addWidget(row)
            self.param_widgets[pname] = sld
            self.param_scales[pname] = scale

    def _current_params(self) -> dict:
        out = {}
        for name, wdg in self.param_widgets.items():
            scale = self.param_scales.get(name, 1)
            v = wdg.value() / scale
            out[name] = int(round(v)) if scale == 1 else v
        return out

    # ---------------------------------------------------------- 调度
    def _schedule(self, *_, immediate: bool = False) -> None:
        if not self.store.symbols:
            return
        if immediate:
            self.run_analysis()
            return
        if self._pending:
            return
        self._pending = True
        QTimer.singleShot(220, self._deferred_run)

    def _deferred_run(self) -> None:
        self._pending = False
        self.run_analysis()

    # ---------------------------------------------------------- 核心计算
    def _current_item(self) -> dict:
        key = self.cb_symbol.currentData()
        for it in self.store.items:
            if it["key"] == key:
                return it
        return {"key": key, "symbol": key or "", "name": "", "exchange": ""}

    def run_analysis(self) -> None:
        skey = self.cb_symbol.currentData()
        key = self.cb_strategy.currentData()
        if not skey or not key:
            return
        df = self.store.get(skey)
        if df is None or df.empty:
            self.lbl_status.setText(f"{skey} 数据缺失")
            return

        self.df_raw = df
        params = self._current_params()
        try:
            df_sig = run_strategy(df, key, params)
        except Exception as exc:
            self.lbl_status.setText(f"策略计算失败：{exc}")
            return

        self.df_sig = df_sig
        self.res = run_backtest(df_sig, self.cfg, label=STRATEGIES[key].label)
        self.bh = buy_and_hold(df, self.cfg)

        self._draw_kline()
        self._draw_perf()
        self._update_metrics()
        self._update_table()

    # ---------------------------------------------------------- K 线绘制
    def _draw_kline(self) -> None:
        df = self.df_sig
        item = self._current_item()
        sym = item["symbol"]
        key = self.cb_strategy.currentData()
        spec = STRATEGIES[key]

        years = self.cb_range.currentData()
        if years and len(df):
            cutoff = df.index[-1] - pd.DateOffset(years=years)
            view = df[df.index >= cutoff]
        else:
            view = df
        if len(view) < 20:
            view = df.tail(120)

        # 为了均线连续，计算时用全量，绘图时切片
        start_pos = len(df) - len(view)
        x = np.arange(len(view))
        o = view["open"].to_numpy(float)
        h = view["high"].to_numpy(float)
        l = view["low"].to_numpy(float)
        c = view["close"].to_numpy(float)
        vol = view["volume"].to_numpy(float)
        up = c >= o

        self.canvas_k.fig.clear()
        gs = self.canvas_k.fig.add_gridspec(
            2, 1, height_ratios=[4.0, 1.0], hspace=0.06,
            left=0.058, right=0.962, top=0.935, bottom=0.085,
        )
        ax = self.canvas_k.fig.add_subplot(gs[0])
        axv = self.canvas_k.fig.add_subplot(gs[1], sharex=ax)

        ax.set_facecolor(C_BG)
        axv.set_facecolor(C_BG)

        # 持仓区间底色：多头蓝、空头红。合并连续区间用 axvspan，
        # 避免逐点 fill_between 产生上千个多边形。
        # 只做多模式下把空头信号裁掉，保证图形与回测口径一致。
        sig_view = view["signal"].to_numpy(int)
        if self.cfg.long_only:
            sig_view = np.maximum(sig_view, 0)
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
        show_overlay = self.chk_ma.currentData()
        overlay_cfg = {
            "ma_cross": [("fast_ma", "快线 MA", "#f59e0b", 1.3), ("slow_ma", "慢线 MA", "#2563eb", 1.5)],
            "donchian": [("upper", "上轨", "#059669", 1.2), ("lower", "下轨", "#dc2626", 1.2)],
            "atr_channel": [("mid", "均线", "#f59e0b", 1.2), ("upper", "上轨", "#059669", 1.2),
                            ("lower", "下轨", "#dc2626", 1.2)],
            "macd": [],
            "turtle": [("upper", "上轨", "#059669", 1.2), ("lower", "下轨", "#dc2626", 1.2)],
        }
        if show_overlay:
            for col, label, color, lw in overlay_cfg.get(key, []):
                if col in view.columns:
                    y = view[col].to_numpy(float)
                    ax.plot(x, y, color=color, lw=lw, label=label, zorder=4, alpha=0.95)

        # 开平仓信号：颜色=持仓方向（绿多红空），形状=开仓/平仓（▲开 ▼平）
        pos = sig_view
        prev = np.concatenate([[0], pos[:-1]])
        off = (h.max() - l.min()) * 0.028
        marks = [
            ((pos == 1) & (prev != 1), "^", C_ENTRY, -1, "开多"),
            ((pos != 1) & (prev == 1), "v", C_ENTRY, +1, "平多"),
            ((pos == -1) & (prev != -1), "v", C_EXIT, +1, "开空"),
            ((pos != -1) & (prev == -1), "^", C_EXIT, -1, "平空"),
        ]
        for cond, marker, color, side, label in marks:
            idxs = np.where(cond)[0]
            if not len(idxs):
                continue
            y = l[idxs] - off if side < 0 else h[idxs] + off
            ax.scatter(idxs, y, marker=marker, s=62, c=color,
                       edgecolors="white", linewidths=0.7, zorder=6, label=label)

        # 最新价：右上角浮动标签（避免贴边被裁切）
        last_color = C_UP if c[-1] >= o[-1] else C_DOWN
        ax.axhline(c[-1], color=last_color, lw=0.8, ls=(0, (4, 3)), alpha=0.5, zorder=2)
        chg = (c[-1] / c[0] - 1.0) * 100
        ax.text(
            0.995, 0.975,
            f"最新 {c[-1]:,.2f}　区间 {chg:+.1f}%",
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

        # 坐标轴
        ax.set_xlim(-1, len(view))
        pad = (h.max() - l.min()) * 0.06
        ax.set_ylim(l.min() - pad, h.max() + pad)
        ax.grid(True, color=C_GRID, lw=0.6, ls="-", alpha=0.9)
        ax.set_axisbelow(True)
        axv.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
        axv.set_axisbelow(True)

        for a in (ax, axv):
            for spine in ("top", "right"):
                a.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                a.spines[spine].set_color("#cbd5e1")
            a.tick_params(colors=C_MUTED, labelsize=9, length=3)
        ax.tick_params(labelbottom=False)

        # x 轴日期刻度
        n_ticks = min(10, max(4, len(view) // 60))
        tick_pos = np.linspace(0, len(view) - 1, n_ticks).astype(int)
        axv.set_xticks(tick_pos)
        axv.set_xticklabels(
            [view.index[i].strftime("%Y-%m") for i in tick_pos],
            rotation=0, fontsize=9,
        )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="upper left", frameon=True, framealpha=0.92,
                      edgecolor="#e2e8f0", fontsize=9, ncol=3)
        ax.set_ylabel(MARKETS[self.market]["price_note"], fontsize=9, color=C_MUTED)

        # 标题信息
        name = item["name"]
        exch = item["exchange"]
        exch_cn = EXCHANGE_CN.get(exch, exch)
        head = f"{sym}  {name}" + (f"（{exch_cn}）" if exch else "")
        state = {1: "多头持仓", -1: "空头持仓"}.get(int(pos[-1]), "空仓")
        state_color = C_ENTRY if pos[-1] == 1 else (C_EXIT if pos[-1] == -1 else C_MUTED)
        ax.set_title(
            f"{head}   ·   {spec.label}   ·   "
            f"{view.index[0]:%Y-%m-%d} ~ {view.index[-1]:%Y-%m-%d}   ·   当前状态：{state}",
            fontsize=11.5, color=C_TEXT, loc="left", pad=10,
        )
        source = (
            "数据源：东方财富（后复权，含分红再投资）"
            if self.market == "us"
            else "数据源：Tushare（主力连续合约，已按复权因子消除换月跳空）"
        )
        self.lbl_chart_title.setText(
            f"全历史 {len(df):,} 根日线 · 当前显示 {len(view):,} 根 · {source}"
        )
        self.lbl_chart_title.setStyleSheet(
            f"font-size: 11px; color: {C_MUTED}; padding-left: 4px;"
        )

        self.canvas_k.fig.canvas.draw_idle()

    # ---------------------------------------------------------- 绩效图
    def _draw_perf(self) -> None:
        res, bh = self.res, self.bh
        fig = self.canvas_p.fig
        fig.clear()
        gs = fig.add_gridspec(
            2, 1, height_ratios=[3.0, 1.0], hspace=0.12,
            left=0.085, right=0.975, top=0.93, bottom=0.085,
        )
        ax = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax)

        eq_s = res.equity / res.equity.iloc[0]
        eq_b = bh.equity / bh.equity.iloc[0]

        ax.plot(eq_b.index, eq_b.values, color=C_BH, lw=1.5, label="买入持有")
        ax.plot(eq_s.index, eq_s.values, color=C_STRAT, lw=1.7, label=res.label)
        ax.fill_between(eq_s.index, 1, eq_s.values,
                        where=eq_s.values >= 1, color=C_STRAT, alpha=0.06)
        ax.fill_between(eq_s.index, 1, eq_s.values,
                        where=eq_s.values < 1, color=C_EXIT, alpha=0.06)
        ax.axhline(1.0, color="#cbd5e1", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_ylabel("净值（对数）", fontsize=9, color=C_MUTED)
        ax.set_title("策略 vs 买入持有 · 资金曲线", fontsize=11.5, loc="left", color=C_TEXT, pad=8)
        ax.grid(True, color=C_GRID, lw=0.6, alpha=0.9)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.tick_params(labelbottom=False, colors=C_MUTED, labelsize=9)

        dd_s = drawdown_series(res.equity) * 100
        dd_b = drawdown_series(bh.equity) * 100
        ax2.fill_between(dd_b.index, dd_b.values, 0, color=C_BH, alpha=0.35, label="持有回撤")
        ax2.fill_between(dd_s.index, dd_s.values, 0, color=C_STRAT, alpha=0.45, label="策略回撤")
        ax2.set_ylabel("回撤 %", fontsize=9, color=C_MUTED)
        ax2.grid(True, color=C_GRID, lw=0.5, alpha=0.7)
        ax2.set_axisbelow(True)
        ax2.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax2.tick_params(colors=C_MUTED, labelsize=9)

        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=8))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        for a in (ax, ax2):
            for spine in ("top", "right"):
                a.spines[spine].set_visible(False)
            for spine in ("left", "bottom"):
                a.spines[spine].set_color("#cbd5e1")
        for lbl in ax2.get_xticklabels():
            lbl.set_rotation(0)

        fig.canvas.draw_idle()

    # ---------------------------------------------------------- 指标
    def _update_metrics(self) -> None:
        m, b = self.res.metrics, self.bh.metrics
        self.card_cagr.set_value(f"{m['cagr']*100:.1f}%", C_UP if m["cagr"] >= 0 else C_DOWN)
        self.card_dd.set_value(f"{m['max_dd']*100:.1f}%", "#ea580c")
        self.card_sharpe.set_value(
            f"{m['sharpe']:.2f}",
            "#0f9d58" if m["sharpe"] >= 1 else ("#1f2937" if m["sharpe"] >= 0.5 else "#ea580c"),
        )
        excess = m["cagr"] - b["cagr"]
        self.card_excess.set_value(
            f"{excess*100:+.1f}%", "#0f9d58" if excess >= 0 else "#d92b2b"
        )
        dir_txt = ""
        if not self.cfg.long_only:
            dir_txt = (f" · 多头 {m['long_exposure']*100:.0f}% / 空头 {m['short_exposure']*100:.0f}%"
                       f" · 空头 {m['n_short']} 笔")
            if m.get("n_blowup"):
                dir_txt += f" · 爆仓 {m['n_blowup']} 次"
        self.lbl_status.setText(
            f"{m['n_trades']} 笔交易 · 胜率 {m['win_rate']*100:.0f}% · "
            f"仓位时间 {m['exposure']*100:.0f}% · 平均持仓 {m['avg_bars']:.0f} 天{dir_txt}"
        )

    def _update_table(self) -> None:
        m, b = self.res.metrics, self.bh.metrics

        def pct(x):
            return f"{x*100:.2f}%"

        rows = [
            ("年化收益 CAGR", pct(m["cagr"]), pct(b["cagr"]), m["cagr"] >= b["cagr"]),
            ("累计收益", pct(m["total_return"]), pct(b["total_return"]), m["total_return"] >= b["total_return"]),
            ("最大回撤", pct(m["max_dd"]), pct(b["max_dd"]), m["max_dd"] >= b["max_dd"]),
            ("夏普比率", f"{m['sharpe']:.2f}", f"{b['sharpe']:.2f}", m["sharpe"] >= b["sharpe"]),
            ("卡玛比率", f"{m['calmar']:.2f}", f"{b['calmar']:.2f}", m["calmar"] >= b["calmar"]),
            ("年化波动", pct(m["ann_vol"]), pct(b["ann_vol"]), m["ann_vol"] <= b["ann_vol"]),
            ("仓位时间占比", pct(m["exposure"]), "100.00%", None),
            ("　其中多头", pct(m["long_exposure"]), "100.00%", None),
            ("　其中空头", pct(m["short_exposure"]), "0.00%", None),
            ("交易次数", f"{m['n_trades']}", "1", None),
            ("　多头 / 空头", f"{m['n_long']} / {m['n_short']}", "1 / 0", None),
            ("胜率", pct(m["win_rate"]), "—", None),
            ("盈亏比", f"{m['profit_factor']:.2f}", "—", None),
            ("平均每笔收益", pct(m["avg_trade"]), "—", None),
            ("平均持仓天数", f"{m['avg_bars']:.0f}", "—", None),
            ("最佳单笔", pct(m["best_trade"]), "—", None),
            ("最差单笔", pct(m["worst_trade"]), "—", None),
            ("期末资金", f"${m['final_equity']:,.0f}", f"${b['final_equity']:,.0f}",
             m["final_equity"] >= b["final_equity"]),
        ]

        self.table.setRowCount(len(rows))
        for i, (name, sv, bv, better) in enumerate(rows):
            it0 = QTableWidgetItem(name)
            it0.setForeground(QColor("#475569"))
            it1 = QTableWidgetItem(sv)
            it1.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it1.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            it2 = QTableWidgetItem(bv)
            it2.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            it2.setForeground(QColor(C_MUTED))
            if better is True:
                it1.setForeground(QColor("#0f9d58"))
            elif better is False:
                it1.setForeground(QColor("#d92b2b"))
            self.table.setItem(i, 0, it0)
            self.table.setItem(i, 1, it1)
            self.table.setItem(i, 2, it2)
            self.table.setRowHeight(i, 26)


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("趋势跟踪研究台")
    app.setApplicationDisplayName("趋势跟踪研究台")
    setup_light_palette(app)
    app.setStyleSheet(STYLE_SHEET)

    win = MainWindow()
    win.show()

    if not win.store.symbols:
        if getattr(sys, "frozen", False):
            hint = (
                "安装包内未找到行情数据。\n\n"
                f"期望目录：\n    {DATA_DIR}\n\n"
                "请重新下载完整安装包，或把 data/ 与 data_futures/ 放到\n"
                f"{ASSET_ROOT} 下。"
            )
        else:
            hint = (
                f"未在 {DATA_DIR} 找到行情数据。\n\n"
                "请先在项目目录运行：\n    python download.py\n"
                "    python download_futures.py\n\n下载完成后再启动本程序。"
            )
        QMessageBox.information(win, "缺少数据", hint)
    return app.exec()


def _log_fatal(exc: BaseException) -> None:
    """pythonw / .app 启动时没有控制台，把致命错误写入日志便于排查。

    日志目录走 ttrade.paths.log_dir()，打包后落在用户目录
    （Windows: %LOCALAPPDATA%\\TrendDesk\\logs，macOS: ~/Library/Logs 同级）。
    """
    import datetime
    import traceback

    try:
        log_path = os.path.join(log_dir(), "error.log")
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] 启动失败\n")
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # noqa: BLE001 - 兜底记录后继续抛出
        _log_fatal(exc)
        raise
