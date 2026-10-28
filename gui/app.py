"""美股趋势跟踪研究台 —— PySide6 桌面程序。

功能：
- 左侧选择标的 / 策略族（规则 or 因子）/ 参数 / 显示区间
- 「K线与信号」页：后复权 K 线、策略指标线、持仓区间底色、买卖信号标记、
  成交量，因子策略下额外画一行因子分数
- 「绩效对比」页：策略 vs 买入持有的资金曲线、回撤曲线、指标对比表
- 「因子分析」页：全因子 IC/ICIR/t 值一览 + IC 时序、衰减、分位收益、
  模型重要度
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
import time

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
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
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ttrade import factor_metrics, factors, mlmodels, tqdata, tqlive, updater
from ttrade import factor_strategy as fstrat
from ttrade import strategy_store as SStore
from gui.agent_page import AgentPage
from gui.live import LiveController
from gui.portfolio_page import PortfolioPage
from gui.trade_page import TradePage
from ttrade.backtest import (
    BacktestConfig,
    bars_per_year,
    buy_and_hold,
    drawdown_series,
    run_backtest,
)
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

# 开平仓标记刻意避开 K 线的红绿：原来多头用绿、空头用红，三角混在蜡烛里
# 根本找不出来（尤其连板行情）。改成紫 / 琥珀，和红(#d92b2b)、绿(#0f9d58)、
# 多头背景蓝、空头背景红都拉得开。
C_LONG_SIG = "#7c3aed"    # 多头信号：紫
C_SHORT_SIG = "#f59e0b"   # 空头信号：琥珀

# 持仓连线：盈利红、亏损绿（A 股习惯）。和信号标记是两套色，不会互相干扰。
C_PROFIT = "#d92b2b"
C_LOSSLINE = "#0f9d58"

# K 线可视窗口（根）。窗口小看得清，窗口大看趋势；滚动条在不缩放的前提下平移。
K_WIN_DEFAULT = 180
K_WIN_MIN = 40
K_WIN_MAX = 1500
K_REDRAW_GAP = 0.15     # 滚动重绘的最小间隔（秒）

# 周期。日线走原来的 Tushare CSV（20 年），分钟级走天勤 DuckDB（10 年）。
# 两者主力换月口径不同（天勤按自己的规则，Tushare 按自己的），所以跨周期的
# 绝对价位会有细微差异 —— 这是数据源差异，不是 bug。
PERIOD_1D = "1d"
GUI_PERIODS = [PERIOD_1D] + tqdata.PERIOD_ORDER   # 下拉框顺序：日线在前，越往后越短
PERIOD_LABEL = dict(tqdata.PERIOD_LABEL, **{PERIOD_1D: "日线"})

# 因子分数面板配色：与 K 线的红绿、信号的紫/琥珀都错开，用青蓝系
C_FACTOR = "#0891b2"
C_FACTOR_FILL = "#22d3ee"
C_IC = "#2563eb"
C_IC_NEG = "#dc2626"

# ------------------------------------------------------------------ 策略族
# 两个大类：规则策略（原有 5 个）、因子策略（单/多/机器学习）。
# 「因子过滤」不是独立大类，而是规则策略的一个可选开关 —— 它只做减法，
# 拿掉规则本来要开的仓，不自己产生信号。
FAMILY_RULE = "rule"
FAMILY_FACTOR = "factor"
FAMILIES = [
    (FAMILY_RULE, "规则策略", "经典技术指标，阈值/交叉直接产生信号"),
    (FAMILY_FACTOR, "因子策略", "因子分数过阈值进出场，可单因子/多因子/机器学习"),
]

# 因子模式（因子策略与因子过滤共用）
FACTOR_MODES = [
    ("single", "单因子", "一个因子的标准化分数过阈值进出场"),
    ("multi", "多因子等权", "多个因子按 PIT 方向取号后等权平均"),
    ("ml", "机器学习多因子", "模型学因子权重，walk-forward 滚动训练预测"),
]

# 因子方向口径。auto 是「让数据说」，另两档是固定方向，跨品种可比。
FACTOR_ORIENTS = [
    ("自动（按滚动 IC 定向）", "auto",
     "方向 = sign(扩张窗口 IC)，滞后预测期数，不含未来信息。\n\n"
     "注意：这个符号**每个品种不一样，同一品种还会中途翻**。\n"
     "所以「反转」在自动模式下是相对各品种自己的基线，"
     "同一个勾选框在不同品种上可能落到相反的绝对方向。\n"
     "要做跨品种一致的比较，请选下面两档固定方向。"),
    ("固定：因子高做多", "long_high",
     "不看 IC，一律「因子分数高就做多、低就做空」。\n"
     "反转后 = 因子分数高就做空，跨品种完全一致。"),
    ("固定：因子低做多", "long_low",
     "不看 IC，一律「因子分数低就做多、高就做空」。\n"
     "对 rev_5 这类反转因子，这一档就是教科书的短期反转策略。"),
]

# 因子策略的参数：名字 -> (标签, 最小, 最大, 步长, 默认值)
FACTOR_PARAMS = {
    "window": ("标准化窗口", 120, 500, 20, 252),
    "horizon": ("预测期数", 1, 20, 1, 5),
    "entry_z": ("进场阈值", 0.3, 3.0, 0.1, 1.0),
    "exit_z": ("离场阈值", 0.0, 1.5, 0.1, 0.3),
    "train_window": ("训练窗口", 400, 2500, 100, 1000),
    "step": ("重训间隔", 60, 400, 20, 180),
    "long_th": ("多头门槛", -1.5, 1.5, 0.1, 0.0),
    "short_th": ("空头门槛", -1.5, 1.5, 0.1, 0.0),
}
ML_ONLY_PARAMS = ("train_window", "step")
FILTER_ONLY_PARAMS = ("long_th", "short_th")

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
QFrame#SubCard { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 7px; }
QTableWidget#FactorTable {
    background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px;
    font-size: 11px; gridline-color: transparent;
}
QTableWidget#FactorTable::item { padding: 1px 3px; }
QTableWidget#FactorTable::item:hover { background: #f1f5f9; }
QTableWidget#FactorTable QHeaderView::section {
    background: #f8fafc; color: #64748b; border: none;
    border-bottom: 1px solid #e2e8f0; font-size: 10px; padding: 2px 4px;
}
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

/* 顶层两个大页面的切换按钮。刻意做成「下划线标签」而不是页签方块：
   内层已经有一套 QTabWidget，外层再用方块页签会变成两层页签，分不清层级。 */
QPushButton#PageTab {
    background: transparent; color: #64748b; border: none;
    border-bottom: 2px solid transparent; border-radius: 0;
    padding: 6px 16px; font-size: 13px; font-weight: 600;
}
QPushButton#PageTab:hover { color: #1f2937; background: #f8fafc; }
QPushButton#PageTab:checked { color: #2563eb; border-bottom: 2px solid #2563eb; }
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
QScrollBar:horizontal { background: #f1f5f9; height: 10px; margin: 0; border-radius: 5px; }
QScrollBar::handle:horizontal { background: #cbd5e1; border-radius: 5px; min-width: 48px; }
QScrollBar::handle:horizontal:hover { background: #94a3b8; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""


# ------------------------------------------------------------------ 数据仓库
FUT_DIR = resolve_data_dir("cn")
AUX_DIR = resolve_data_dir("aux")

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
            "▲ 紫色向上三角＝开多　▼ 紫色向下三角＝平多　"
            "浅蓝底色＝多头持仓　红色虚线＝这笔交易盈利　绿色虚线＝亏损　"
            "价格均为后复权（含分红再投资），保证收益率连续"
        ),
    },
    "cn": {
        "label": "国内期货",
        "dir": FUT_DIR,
        "sub": "国内期货 · 日线 20 年（Tushare）/ 5 分钟·1 小时 10 年（天勤）· 主力连续 · 前复权",
        "price_note": "价格（前复权·主力连续）",
        "fee": 0.0001,
        "slippage": 0.0002,
        "allow_short": True,
        "legend": (
            "▲紫＝开多　▼紫＝平多　▼琥珀＝开空　▲琥珀＝平空　"
            "浅蓝底色＝多头持仓　浅红底色＝空头持仓　"
            "红色虚线＝这笔交易盈利　绿色虚线＝亏损　"
            "分钟级按天勤自己的主力切换口径复权，与日线的 Tushare 口径不同"
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

    df = df.set_index("date")
    # 并入期限结构 / 期权波动率列（展期因子、期权因子要用）。缺文件时静默跳过。
    sym, _, ex = key.rpartition("_")
    try:
        from ttrade.auxdata import attach as attach_aux

        df = attach_aux(df, sym, ex, AUX_DIR)
    except Exception:
        pass
    return df


class DataStore:
    """本地缓存，按需读取。支持美股 / 国内期货两个市场、多个周期。

    期货的日线走 Tushare CSV（20 年），分钟级走天勤 DuckDB（10 年）。缓存按
    (标的, 周期) 存，并限制条目数 —— 一个品种的 5 分钟十年是十几万行、约 10 MB，
    全市场都缓存下来会吃掉几个 G。
    """

    CACHE_MAX = 6

    def __init__(self, market: str = "us"):
        self.market = market
        self.data_dir = MARKETS[market]["dir"]
        self._cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._periods: dict[str, set[str]] = {}
        self.last_error = ""
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

    def get(self, key: str, period: str = PERIOD_1D) -> pd.DataFrame | None:
        ck = (key, period)
        if ck in self._cache:
            return self._cache[ck]
        try:
            if self.market == "us" or period in (None, "", PERIOD_1D):
                df = load_local(key, self.data_dir) if self.market == "us" \
                    else load_futures_csv(key, self.data_dir)
            else:
                sym, _, ex = key.rpartition("_")
                # qfq：最后一根等于真实盘面价，和日线的口径一致
                df = tqdata.load_main(sym, ex, period, adjust="qfq")
        except Exception as exc:  # noqa: BLE001
            # 最常见的是下载脚本正持有 DuckDB 的排他锁。把原因记下来，
            # 上层据此提示「正在下载」，而不是含糊的「数据缺失」。
            self.last_error = str(exc)
            return None
        self.last_error = ""
        while len(self._cache) >= self.CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[ck] = df
        return df

    def invalidate(self, key: str, period: str | None = None) -> None:
        """作废缓存。补齐历史后必须调，否则实时段会接在旧尾巴上。"""
        for ck in [c for c in self._cache
                   if c[0] == key and (period is None or c[1] == period)]:
            self._cache.pop(ck, None)
        if period is None:
            self._periods.pop(key, None)

    def has_period(self, key: str, period: str) -> bool:
        """这个标的有没有该周期的数据。按品种缓存，切换品种时只查一次库。"""
        if self.market == "us" or period in (None, "", PERIOD_1D):
            return True
        if key not in self._periods:
            sym, _, ex = key.rpartition("_")
            try:
                self._periods[key] = set(tqdata.available_periods(sym, ex))
            except Exception:  # noqa: BLE001
                return False      # 拿不到就返回 False，但不要把失败结果缓存住
        return period in self._periods[key]


# ------------------------------------------------------------------ 交易切分
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


# ------------------------------------------------------------------ 数据更新线程
class UpdateWorker(QObject):
    """在后台线程里跑数据更新，避免界面假死。

    下载是纯网络 IO，几十个标的顺序请求，放在主线程会让窗口完全卡住
    （连「取消」都点不动）。这里用 QThread + 信号把进度送回主线程。
    """

    progress = Signal(int, int, str)   # done, total, message
    done = Signal(object)              # UpdateResult
    failed = Signal(str)

    def __init__(self, market: str, data_dir: str, range_key: str,
                 source: str = "auto", min_interval: float = 60.0,
                 aux_contracts: bool = False, aux_options: bool = False):
        super().__init__()
        self.market = market
        self.data_dir = data_dir
        self.range_key = range_key
        self.source = source
        self.min_interval = min_interval
        self.aux_contracts = aux_contracts
        self.aux_options = aux_options
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        cb = lambda d, t, m: self.progress.emit(d, t, m)  # noqa: E731
        try:
            if self.market == "us":
                res = updater.update_us(
                    self.data_dir, self.range_key, progress=cb,
                    cancel=lambda: self._cancel,
                    source=self.source, min_interval=self.min_interval,
                )
            else:
                res = updater.update_futures(
                    self.data_dir, self.range_key, progress=cb,
                    cancel=lambda: self._cancel,
                )
                if self.aux_contracts or self.aux_options:
                    # 衍生数据放在主力连续之后跑：即使它失败，主行情也已经落盘了
                    try:
                        res.aux = updater.update_aux(
                            range_key=self.range_key,
                            want_contracts=self.aux_contracts,
                            want_options=self.aux_options,
                            progress=cb,
                            cancel=lambda: self._cancel,
                        )
                    except Exception as exc:
                        res.aux = None
                        res.aux_error = str(exc)[:200]
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.done.emit(res)


# ------------------------------------------------------------------ 因子计算线程
class FactorWorker(QObject):
    """在后台线程里跑因子策略。

    机器学习模式要跑几十折 walk-forward：LightGBM 约 1s，Transformer 约 18s。
    放在主线程里窗口会直接假死（连取消都点不动），所以统一走线程。
    单因子/多因子本身只要几十毫秒，但走同一条路径能少一套分支。
    """

    progress = Signal(int, int, str)   # done, total, message
    done = Signal(object)              # FactorOutcome
    failed = Signal(str)

    def __init__(self, df: pd.DataFrame, mode: str, keys: list[str],
                 cfg: fstrat.FactorConfig, rule_signal=None):
        super().__init__()
        self.df = df
        self.mode = mode
        self.keys = keys
        self.cfg = cfg
        self.rule_signal = rule_signal
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        cb = lambda d, t, m: self.progress.emit(d, t, m)  # noqa: E731
        try:
            if self.rule_signal is None:
                out = fstrat.run_factor_strategy(
                    self.df, self.mode, self.keys, self.cfg,
                    progress=cb, cancel=lambda: self._cancel,
                )
            else:
                out = fstrat.run_rule_with_filter(
                    self.df, self.rule_signal, self.mode, self.keys, self.cfg,
                    progress=cb, cancel=lambda: self._cancel,
                )
        except fstrat.Cancelled:
            return
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(out)


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
        self.fparam_widgets: dict[str, QWidget] = {}
        self.fparam_scales: dict[str, int] = {}
        self._pending = False
        self._upd_thread: QThread | None = None
        self._upd_worker: UpdateWorker | None = None

        # 因子相关状态
        self.family = FAMILY_RULE
        self.factor_mode = "single"
        self.factor_keys: list[str] = list(factors.DEFAULT_MULTI)
        self.factor_outcome = None        # 最近一次 FactorOutcome
        self._strategy_label = ""         # 当前策略在图上的标题文字
        self._f_thread: QThread | None = None
        self._f_worker: FactorWorker | None = None
        self._f_seq = 0                   # 用于丢弃过期的计算结果
        self._f_report: pd.DataFrame | None = None
        self._f_report_key = ""           # 缓存键：标的+市场，避免重复评估
        # 关窗后仍有排队的信号可能被投递到已销毁的控件上，PySide6 会抛
        # "Internal C++ object already deleted"。用一个标志把它们挡在门外。
        self._closing = False
        self._f_applied = -1              # 已落地的结果序号，防止同一结果重复应用

        # K 线可视窗口状态：_k_view 是区间筛选后的全量数据，
        # [_k_start, _k_start+_k_win) 才是当前画出来的那一段。
        self._k_view: pd.DataFrame | None = None
        self._k_total = 0
        self._k_win = K_WIN_DEFAULT
        self._k_start = 0
        self._k_spec_label = ""
        self._k_key = ""
        # 滚动重绘一次约 180ms（重建图元 54ms + Agg 出图 125ms）。
        # 拖动时 valueChanged 每秒能来几十次，不节流会直接把事件队列堆死，
        # 表现就是「拖一下卡三秒」。这里限制最小间隔，并保证松手后一定补最后一帧。
        self._k_last_draw = 0.0
        self._k_pending = False
        self._k_redraw_timer = QTimer(self)
        self._k_redraw_timer.setSingleShot(True)
        self._k_redraw_timer.timeout.connect(self._flush_kline_redraw)

        # 实时与交易
        self._live_base: pd.DataFrame | None = None   # 订阅时的本地历史，实时段接在它后面
        self._live_state = None                        # 最近一次 LiveState
        self._live_key = ""
        self._live_period = ""
        self._live_last = None                         # 上一根 bar 的时间，用来判断「新 bar」
        self._trading = False                          # 绿灯：新信号直接下单
        self._last_target = 0                          # 上次下发的目标仓位，防重复下单
        self._live_seq = 0                             # 每次起停自增，丢掉过期的推送

        self._build_ui()
        self._populate_symbols()
        # 延后一拍：等界面完全建好、第一次分析排上队之后再灌配置，
        # 否则 apply_config 里的 _schedule(immediate=True) 会和首帧抢数据加载。
        QTimer.singleShot(0, self._restore_last_config)

    # ---------------------------------------------------------- UI 构建
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(8)

        outer.addWidget(self._build_page_bar())

        # 四个平行的大页面：策略研究 / 账户透视和交易 / 策略交易和组合 / 交易Agent。
        # 用 QStackedWidget 而不是外层 QTabWidget：页签放在顶部一条自绘的按钮条上，
        # 视觉上和侧边栏的卡片、内层的三个页签区分得开，不会变成「两层页签」。
        self.pages = QStackedWidget()
        self.pages.addWidget(self._build_study_page())
        self.live = LiveController(self)
        self.trade_page = TradePage(self.live)
        self.pages.addWidget(self.trade_page)
        self.portfolio_page = PortfolioPage()
        self.pages.addWidget(self.portfolio_page)
        self.agent_page = AgentPage()
        self.pages.addWidget(self.agent_page)
        outer.addWidget(self.pages, 1)

        self.agent_page.study_requested.connect(self._on_agent_study)
        self.agent_page.trade_requested.connect(self._on_agent_trade)
        self.agent_page.portfolio_requested.connect(self._on_agent_portfolio)

        self.trade_page.connect_requested.connect(self._on_trade_connect)
        self.trade_page.disconnect_requested.connect(self._on_live_stop)
        self._wire_live()

    def _build_page_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("Card")
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(4)

        self.btn_page_study = QPushButton("策略研究")
        self.btn_page_trade = QPushButton("账户透视和交易")
        self.btn_page_port = QPushButton("策略交易和组合")
        self.btn_page_agent = QPushButton("交易Agent")
        self._page_btns = (
            self.btn_page_study, self.btn_page_trade,
            self.btn_page_port, self.btn_page_agent,
        )
        for i, b in enumerate(self._page_btns):
            b.setObjectName("PageTab")
            b.setCheckable(True)
            b.setMinimumWidth(132)
            b.clicked.connect(lambda _c=False, k=i: self._switch_page(k))
            lay.addWidget(b)
        self.btn_page_study.setChecked(True)
        lay.addStretch(1)

        self.lbl_live = QLabel("实时未启动")
        self.lbl_live.setObjectName("Hint")
        lay.addWidget(self.lbl_live)
        return bar

    def _switch_page(self, idx: int) -> None:
        for i, b in enumerate(self._page_btns):
            b.setChecked(i == idx)
        self.pages.setCurrentIndex(idx)
        # 切到组合页时重读磁盘：「策略研究」里刚存的配置要立刻能看到
        if idx == 2:
            self.portfolio_page.reload()
        elif idx == 3:
            self.agent_page.reload()

    # ---------------------------------------------------------- 交易Agent

    def _on_agent_study(self, cfg) -> None:
        """Agent 要「用某策略回测某品种」：切到研究页并套上配置。

        内置模板没有品种（`cfg.symbol` 为空），这时候只换策略和参数，
        品种保持用户当前选的那个 —— 换个品种不该顺带把标的也换掉。
        """
        self._switch_page(0)
        if cfg.symbol:
            self.apply_config(cfg)
            return
        f = self.cb_family.findData("rule")
        if f >= 0:
            self.cb_family.setCurrentIndex(f)
        r = self.cb_strategy.findData(cfg.rule_key)
        if r >= 0:
            self.cb_strategy.setCurrentIndex(r)
        for name, val in (cfg.rule_params or {}).items():
            w = self.param_widgets.get(name)
            if w is None:
                continue
            try:
                w.setValue(int(round(float(val) * self.param_scales.get(name, 1))))
            except (TypeError, ValueError):
                continue
        self._schedule(immediate=True)

    def _on_agent_portfolio(self, cfg) -> None:
        self._switch_page(2)
        self.portfolio_page.accept(cfg)

    def _on_agent_trade(self, action) -> None:
        """Agent 给了交易动作。动作**已经在界面上确认过**才走到这里。"""
        p = action.payload
        if action.kind == "backtest":
            self._on_agent_study(SStore.StrategyConfig(
                symbol=p.get("symbol", ""),
                exchange=p.get("exchange", ""),
                period=p.get("period", "1d"),
                rule_key=p.get("strategy", ""),
                rule_params=dict(p.get("params") or {}),
                volume=int(p.get("volume", 1)),
            ))
            return

        if action.kind != "trade":
            return

        if p.get("mode") == "manual":
            self._agent_manual(p)
            return

        key = p.get("strategy", "")
        if key not in STRATEGIES:
            self.agent_page.notify(f"平台没有这个策略：{key}", bad=True)
            return
        cfg = SStore.StrategyConfig(
            symbol=p.get("symbol", ""),
            exchange=p.get("exchange", ""),
            period=p.get("period", "1d"),
            rule_key=key,
            rule_params=dict(p.get("params") or {}),
            volume=int(p.get("volume", 1)),
            note="short",
        )
        self._switch_page(2)
        self.portfolio_page.accept(cfg)

    def _agent_manual(self, p: dict) -> None:
        """主观买卖：只对**当前实时会话**的品种生效。

        合约代码必须取实时会话的 `underlying`（真实主力），不能拿品种名拼 ——
        主连 `KQ.m@` 只能看行情、不能下单。
        """
        sym = p.get("symbol", "")
        ex = p.get("exchange", "")
        want = f"{sym}_{ex}" if ex else sym
        if not self.live.running or self._live_state is None:
            self.agent_page.notify("还没连上天勤，先到「账户透视和交易」连上账户", bad=True)
            return
        if self._live_key != want:
            self.agent_page.notify(
                f"当前实时会话是 {self._live_key or '空'}，和 {want} 不一致。"
                "先在「账户透视和交易」页把实时切到这个品种", bad=True)
            return
        und = getattr(self._live_state, "underlying", "") or ""
        if not und:
            self.agent_page.notify("还没拿到真实主力合约，等行情推一帧再试", bad=True)
            return
        ok = self.live.place(und, p["side"], p["offset"], int(p.get("volume", 1)), None)
        if ok:
            side = "买入" if p["side"] == "BUY" else "卖出"
            self.agent_page.notify(
                f"已提交：{side} {p['offset']} {und} {p.get('volume', 1)} 手（市价）")

    def _build_study_page(self) -> QWidget:
        page = QWidget()
        root = QHBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        root.addWidget(self._build_sidebar())

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_chart_tab(), "K线与信号")
        self.tabs.addTab(self._build_perf_tab(), "绩效对比")
        self.tabs.addTab(self._build_factor_tab(), "因子分析")
        root.addWidget(self.tabs, 1)
        return page

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

        cl.addWidget(self._section("周期"))
        self.cb_period = QComboBox()
        for p in GUI_PERIODS:
            self.cb_period.addItem(PERIOD_LABEL.get(p, p), p)
        self.cb_period.setToolTip(
            "日线来自 Tushare（20 年）；5 分钟 / 1 小时来自天勤（10 年），\n"
            "15/30 分钟由 5 分钟合并、2/4 小时由 1 小时合并。\n"
            "两个数据源的主力换月口径不同，跨周期的绝对价位会有细微差异。"
        )
        cl.addWidget(self.cb_period)
        self.lbl_period = QLabel("")
        self.lbl_period.setObjectName("Hint")
        self.lbl_period.setWordWrap(True)
        cl.addWidget(self.lbl_period)

        cl.addWidget(self._section("策略类型"))
        self.cb_family = QComboBox()
        for key, label, tip in FAMILIES:
            self.cb_family.addItem(label, key)
            self.cb_family.setItemData(
                self.cb_family.count() - 1, tip, Qt.ToolTipRole
            )
        cl.addWidget(self.cb_family)

        # ---------------------------------------------- 规则策略页
        # 用 show/hide 而不是 QStackedWidget：堆叠容器的高度取所有页的最大值，
        # 规则模式下会被因子页的因子列表撑出一大块空白。
        self.box_rule = QWidget()
        pr = QVBoxLayout(self.box_rule)
        pr.setContentsMargins(0, 0, 0, 0)
        pr.setSpacing(8)
        pr.addWidget(self._section("规则策略"))
        self.cb_strategy = QComboBox()
        for key, spec in STRATEGIES.items():
            self.cb_strategy.addItem(spec.label, key)
        pr.addWidget(self.cb_strategy)

        self.lbl_desc = QLabel("")
        self.lbl_desc.setObjectName("Hint")
        self.lbl_desc.setWordWrap(True)
        self.lbl_desc.setMinimumHeight(44)
        pr.addWidget(self.lbl_desc)

        pr.addWidget(self._section("参数"))
        self.param_box = QVBoxLayout()
        self.param_box.setContentsMargins(0, 0, 0, 0)
        self.param_box.setSpacing(8)
        pr.addLayout(self.param_box)
        cl.addWidget(self.box_rule)

        # ---------------------------------------------- 因子策略页
        self.box_factor = QWidget()
        pf = QVBoxLayout(self.box_factor)
        pf.setContentsMargins(0, 0, 0, 0)
        pf.setSpacing(8)
        pf.addWidget(self._section("因子模式"))
        self.cb_fmode = QComboBox()
        for key, label, tip in FACTOR_MODES:
            self.cb_fmode.addItem(label, key)
            self.cb_fmode.setItemData(self.cb_fmode.count() - 1, tip, Qt.ToolTipRole)
        pf.addWidget(self.cb_fmode)

        self.lbl_fmode = QLabel("")
        self.lbl_fmode.setObjectName("Hint")
        self.lbl_fmode.setWordWrap(True)
        pf.addWidget(self.lbl_fmode)
        cl.addWidget(self.box_factor)

        # ---------------------------------------------- 因子过滤（仅规则模式）
        self.box_filter = QFrame()
        self.box_filter.setObjectName("SubCard")
        fl = QVBoxLayout(self.box_filter)
        fl.setContentsMargins(10, 9, 10, 9)
        fl.setSpacing(7)

        self.chk_filter = QCheckBox("用因子过滤规则信号")
        self.chk_filter.setToolTip(
            "只做减法：规则说要开仓时，因子不同意就不开。\n"
            "因子永远不会让规则策略开出它本来不想开的仓。"
        )
        fl.addWidget(self.chk_filter)

        self.cb_filter_mode = QComboBox()
        for key, label, _tip in FACTOR_MODES:
            self.cb_filter_mode.addItem(label, key)
        fl.addWidget(self.cb_filter_mode)

        self.cb_filter_when = QComboBox()
        self.cb_filter_when.addItem("只在开仓时否决", "entry")
        self.cb_filter_when.addItem("等因子转好再进", "delay")
        self.cb_filter_when.addItem("转差就离场", "hold")
        self.cb_filter_when.setToolTip(
            "只在开仓时否决：规则开仓那一刻因子不同意，整笔跳过，\n"
            "等规则先平掉再重新开仓才给下一次机会。最严格。\n\n"
            "等因子转好再进：因子不同意就先等着，转好了再进。\n"
            "只推迟入场，几乎每笔最终都会进去，对仓位影响很小。\n\n"
            "转差就离场：持仓期间因子一旦转向立即平仓。\n"
            "回撤改善最明显，但换手会大幅上升。"
        )
        fl.addWidget(self.cb_filter_when)
        cl.addWidget(self.box_filter)

        # ---------------------------------------------- 因子选择（两处共用）
        # 因子策略和因子过滤不会同时生效，所以共用一套控件：切换时只换标题，
        # 选中的因子保留下来，从「策略」改成「过滤」不用重新勾一遍。
        self.box_fsel = QWidget()
        sf = QVBoxLayout(self.box_fsel)
        sf.setContentsMargins(0, 0, 0, 0)
        sf.setSpacing(8)

        self.lbl_fsel = self._section("因子选择")
        sf.addWidget(self.lbl_fsel)

        self.cb_fsingle = QComboBox()
        for cat in factors.CATEGORIES:
            keys = [k for k, s in factors.FACTORS.items() if s.category == cat]
            if not keys:
                continue
            for i, k in enumerate(keys):
                spec = factors.FACTORS[k]
                text = f"{cat} · {spec.label}" if i == 0 else f"　　{spec.label}"
                self.cb_fsingle.addItem(text, k)
        idx = self.cb_fsingle.findData(factors.DEFAULT_SINGLE)
        if idx >= 0:
            self.cb_fsingle.setCurrentIndex(idx)

        # 单因子模式：下拉框 + 反转勾选框同一行。
        # 反转状态不在这个勾选框里单独存，而是回写到下面表格的「反」列 ——
        # 这样单因子和多因子共用一份反转状态，切模式时勾过的不会丢，也不会打架。
        self.box_fsingle = QWidget()
        bfl = QHBoxLayout(self.box_fsingle)
        bfl.setContentsMargins(0, 0, 0, 0)
        bfl.setSpacing(6)
        bfl.addWidget(self.cb_fsingle, 1)
        self.chk_frev = QCheckBox("反转")
        self.chk_frev.setToolTip(
            "勾上 = 这个因子反向使用（等效于把因子取负号）。\n"
            "本来分数高做多的，改成分数高做空；分数低做空的，改成分数低做多。\n\n"
            "与多因子表格里的「反」列是同一份状态。"
        )
        bfl.addWidget(self.chk_frev)
        sf.addWidget(self.box_fsingle)

        # 多因子选择：三列表 —— 因子 / 用 / 反。
        # 用 QTableWidget 而不是 QListWidget：每个因子需要两个独立开关
        # （参与合成、反向使用），QListWidget 只有一列勾选框，塞不下第二个。
        self.tbl_fmulti = QTableWidget(0, 3)
        self.tbl_fmulti.setObjectName("FactorTable")
        self.tbl_fmulti.setFixedHeight(176)
        self.tbl_fmulti.setHorizontalHeaderLabels(["因子", "用", "反"])
        self.tbl_fmulti.verticalHeader().setVisible(False)
        # 47 行一屏放不下，行高压紧到 21px，176px 能露 8 行，剩下的滚动
        self.tbl_fmulti.verticalHeader().setDefaultSectionSize(21)
        self.tbl_fmulti.setShowGrid(False)
        self.tbl_fmulti.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_fmulti.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_fmulti.setFocusPolicy(Qt.NoFocus)
        hh = self.tbl_fmulti.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        hh.setSectionResizeMode(1, QHeaderView.Fixed)
        hh.setSectionResizeMode(2, QHeaderView.Fixed)
        hh.setFixedHeight(22)
        self.tbl_fmulti.setColumnWidth(1, 34)
        self.tbl_fmulti.setColumnWidth(2, 34)
        for cat in factors.CATEGORIES:
            for k, spec in factors.FACTORS.items():
                if spec.category != cat:
                    continue
                r = self.tbl_fmulti.rowCount()
                self.tbl_fmulti.insertRow(r)
                name = QTableWidgetItem(f"{cat} · {spec.label}")
                name.setData(Qt.UserRole, k)
                name.setToolTip(spec.desc)
                self.tbl_fmulti.setItem(r, 0, name)
                for col, tip, default in (
                    (1, "勾上表示参与合成", k in factors.DEFAULT_MULTI),
                    (2, "勾上表示这个因子反向使用（取负号）", False),
                ):
                    cell = QTableWidgetItem("")
                    cell.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    cell.setCheckState(Qt.Checked if default else Qt.Unchecked)
                    cell.setTextAlignment(Qt.AlignCenter)
                    cell.setToolTip(tip)
                    self.tbl_fmulti.setItem(r, col, cell)
        sf.addWidget(self.tbl_fmulti)

        sel_row = QWidget()
        srl = QHBoxLayout(sel_row)
        srl.setContentsMargins(0, 0, 0, 0)
        srl.setSpacing(6)
        self.btn_fall = QPushButton("全选")
        self.btn_fall.setObjectName("Ghost")
        self.btn_fall.setFixedHeight(24)
        self.btn_fnone = QPushButton("清空")
        self.btn_fnone.setObjectName("Ghost")
        self.btn_fnone.setFixedHeight(24)
        srl.addWidget(self.btn_fall)
        srl.addWidget(self.btn_fnone)
        srl.addStretch(1)
        self.box_fbtns = sel_row
        sf.addWidget(sel_row)

        self.lbl_fmodel = self._section("模型")
        sf.addWidget(self.lbl_fmodel)
        self.cb_fmodel = QComboBox()
        # 缺依赖的模型不列出来，但要在下拉框上留一句提示 ——
        # 否则「XGBoost 不见了」和「本来就没有 XGBoost」在界面上长得一模一样。
        _why: dict = {}
        for k in mlmodels.available(why=_why):
            self.cb_fmodel.addItem(mlmodels.MODELS[k]["label"], k)
            self.cb_fmodel.setItemData(
                self.cb_fmodel.count() - 1, mlmodels.MODELS[k]["desc"], Qt.ToolTipRole
            )
        if _why:
            self.cb_fmodel.setToolTip(
                "以下模型在本机不可用（依赖缺失）：\n"
                + "\n".join(
                    f"· {mlmodels.MODELS[k]['label']}：{msg}" for k, msg in _why.items()
                )
            )
        sf.addWidget(self.cb_fmodel)

        self.lbl_fparam = self._section("因子参数")
        sf.addWidget(self.lbl_fparam)

        # 方向口径。这一项之前只在 FactorConfig 里存在、界面没暴露，导致「反转」
        # 是相对「自动定向」的 —— 而自动定向取 sign(扩张 IC)，每个品种不一样、
        # 同一品种还会中途翻。于是同一个「反转」勾选框在 ZN 上≈高分做空、
        # 在 C 上≈高分做多，看起来像 bug。选了固定方向，反转就是绝对的、跨品种可比的。
        self.cb_forient = QComboBox()
        for label, key, tip in FACTOR_ORIENTS:
            self.cb_forient.addItem(label, key)
            self.cb_forient.setItemData(self.cb_forient.count() - 1, tip, Qt.ToolTipRole)
        sf.addWidget(self.cb_forient)

        self.fparam_box = QVBoxLayout()
        self.fparam_box.setContentsMargins(0, 0, 0, 0)
        self.fparam_box.setSpacing(8)
        sf.addLayout(self.fparam_box)
        cl.addWidget(self.box_fsel)

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

        # ---------------------------------------------- 策略配置（保存 / 入库）
        cl.addWidget(self._section("策略配置"))
        self.ed_cfg_name = QLineEdit()
        self.ed_cfg_name.setPlaceholderText("给这套配置起个名字（可留空）")
        self.ed_cfg_name.setToolTip(
            "留空时用「品种 · 策略」自动命名，列表里也能改。"
        )
        cl.addWidget(self.ed_cfg_name)

        cfg_row = QWidget()
        crl = QHBoxLayout(cfg_row)
        crl.setContentsMargins(0, 0, 0, 0)
        crl.setSpacing(6)
        self.btn_save_cfg = QPushButton("保存配置")
        self.btn_save_cfg.setObjectName("Ghost")
        self.btn_save_cfg.setToolTip(
            "保存当前这套配置：下次打开程序自动恢复。\n"
            "勾了下面那一项时，同时作为一项加进「策略交易和组合」。"
        )
        self.btn_save_lib = QPushButton("存入策略库")
        self.btn_save_lib.setObjectName("Ghost")
        self.btn_save_lib.setToolTip(
            "作为模板存进策略库，之后在「策略交易和组合」里可以反复取用。"
        )
        crl.addWidget(self.btn_save_cfg)
        crl.addWidget(self.btn_save_lib)
        cl.addWidget(cfg_row)

        self.chk_to_portfolio = QCheckBox("同时加入策略交易列表")
        self.chk_to_portfolio.setToolTip(
            "勾上后，保存配置 / 存入策略库时会同时把这份配置加进\n"
            "「策略交易和组合」的策略列表，可以直接启动自动交易。"
        )
        cl.addWidget(self.chk_to_portfolio)

        lay.addWidget(card)

        lay.addWidget(self._build_update_card())

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

    def _build_update_card(self) -> QWidget:
        """数据更新面板：区间 + 数据源 + 限速 + 更新按钮 + 进度条。"""
        card = QFrame()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(7)

        lay.addWidget(self._section("数据更新"))

        row = QHBoxLayout()
        row.setSpacing(6)
        self.cb_upd_range = QComboBox()
        for key, label, _days in updater.RANGES:
            self.cb_upd_range.addItem(label, key)
        self.cb_upd_range.setToolTip(
            "只补这一段，不重下 20 年。\n"
            "「全部 20 年」会对每个标的整段重下，耗时较长。"
        )
        row.addWidget(self.cb_upd_range, 1)

        self.btn_update = QPushButton("更新")
        self.btn_update.setObjectName("Ghost")
        self.btn_update.setFixedWidth(58)
        self.btn_update.clicked.connect(self._on_update_clicked)
        row.addWidget(self.btn_update)
        lay.addLayout(row)

        self.cb_upd_source = QComboBox()
        for key, label in updater.SOURCE_LABELS.items():
            self.cb_upd_source.addItem(label, key)
        self.cb_upd_source.setToolTip(
            "仅对美股生效。东财给的是后复权价，与本地历史同口径；\n"
            "Tushare us_daily 是未复权且额度极紧，只作兜底。"
        )
        lay.addWidget(self.cb_upd_source)

        # 期货只有一个数据源，用一行静态说明顶替下拉框。
        # 下拉框里那些「东方财富」「自动」选项对期货毫无意义，
        # 摆在那里只会让人以为可以选，所以直接不显示。
        self.lbl_upd_src = QLabel("数据源：Tushare（日线与复权因子同步更新）")
        self.lbl_upd_src.setObjectName("Hint")
        self.lbl_upd_src.setWordWrap(True)
        self.lbl_upd_src.setVisible(False)
        lay.addWidget(self.lbl_upd_src)

        self.cb_upd_rate = QComboBox()
        for sec, label in [(60.0, "Tushare 限速：60 秒/次"),
                           (30.0, "Tushare 限速：30 秒/次"),
                           (10.0, "Tushare 限速：10 秒/次"),
                           (0.0, "Tushare 限速：不限")]:
            self.cb_upd_rate.addItem(label, sec)
        self.cb_upd_rate.setToolTip("只约束 Tushare 美股接口；东财与期货不受影响")
        lay.addWidget(self.cb_upd_rate)

        # 合约 / 期权衍生数据。这两项要按「每个交易日一次全市场横截面」取数，
        # 比主力连续的增量更新慢一个量级，所以做成显式勾选，默认不跟。
        self.lbl_upd_aux = QLabel("衍生数据（展期 / 期权因子用）")
        self.lbl_upd_aux.setObjectName("Hint")
        lay.addWidget(self.lbl_upd_aux)

        self.chk_upd_contracts = QCheckBox("一并更新真实合约（期限结构）")
        self.chk_upd_contracts.setToolTip(
            "下载全部真实合约的日线，生成近/次近/第三近合约的期限结构。\n"
            "展期收益率、展期年化、展期曲率、展期变化依赖它。\n"
            "每个交易日一次全市场请求，比主力连续慢很多（20 年全量约 40 分钟）。"
        )
        lay.addWidget(self.chk_upd_contracts)

        self.chk_upd_options = QCheckBox("一并更新期权（隐含波动率）")
        self.chk_upd_options.setToolTip(
            "下载商品期权日线，反解平值 IV、偏度、期限结构与看跌看涨持仓比。\n"
            "只保留聚合结果（原始期权明细约 1.5GB，默认不落盘）。\n"
            "商品期权 2017 年才上市，更早的日期会自动跳过。"
        )
        lay.addWidget(self.chk_upd_options)

        self.prog_update = QProgressBar()
        self.prog_update.setTextVisible(False)
        self.prog_update.setFixedHeight(6)
        self.prog_update.setVisible(False)
        lay.addWidget(self.prog_update)

        self.lbl_upd = QLabel("")
        self.lbl_upd.setObjectName("Hint")
        self.lbl_upd.setWordWrap(True)
        self.lbl_upd.setVisible(False)
        lay.addWidget(self.lbl_upd)

        return card

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

        # 实时 / 交易开关。放在 K 线页里是因为它们的作用域就是「当前这张图」：
        # 换了品种或周期，实时订阅要跟着换，交易信号也只在当前图上触发。
        bar = QHBoxLayout()
        bar.setSpacing(8)
        self.btn_live = QPushButton("启动实时")
        self.btn_live.setCheckable(True)
        self.btn_live.setObjectName("Ghost")
        self.btn_live.setMinimumWidth(92)
        self.btn_live.toggled.connect(self._on_live_toggled)
        bar.addWidget(self.btn_live)

        self.btn_trade = QPushButton("启动交易")
        self.btn_trade.setCheckable(True)
        self.btn_trade.setObjectName("Ghost")
        self.btn_trade.setMinimumWidth(92)
        self.btn_trade.setToolTip(
            "开启后，当前图上的新信号会直接下到所选天勤账户。\n"
            "需要先「启动实时」并连上账户。"
        )
        self.btn_trade.toggled.connect(self._on_trade_toggled)
        bar.addWidget(self.btn_trade)

        self.lbl_trade_icon = QLabel("☹")
        self.lbl_trade_icon.setStyleSheet(f"font-size: 17px; color: {C_DOWN};")
        bar.addWidget(self.lbl_trade_icon)
        self.lbl_trade_state = QLabel("未交易")
        self.lbl_trade_state.setStyleSheet(f"font-size: 11px; color: {C_MUTED};")
        bar.addWidget(self.lbl_trade_state)

        bar.addStretch(1)
        self.lbl_live_msg = QLabel("")
        self.lbl_live_msg.setObjectName("Hint")
        bar.addWidget(self.lbl_live_msg)
        lay.addLayout(bar)

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

        # 横向滚动条：在**不缩放**的前提下平移可视窗口。区间选「全部 20 年」时
        # 5000 根挤在一屏根本没法看，有了它就能保持当前根数左右拖着看历史。
        self.sb_kline = QScrollBar(Qt.Horizontal)
        self.sb_kline.setFixedHeight(15)
        self.sb_kline.setVisible(False)
        self.sb_kline.valueChanged.connect(self._on_kline_scroll)
        lay.addWidget(self.sb_kline)

        # 滚轮缩放：以鼠标位置为锚点增减窗口根数，和滚动条配合使用
        self.canvas_k.mpl_connect("scroll_event", self._on_kline_wheel)

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

    def _build_factor_tab(self) -> QWidget:
        """因子分析页：左边全因子 IC/IR 一览，右边四格图。

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

    # ---------------------------------------------------------- 数据填充
    def _populate_symbols(self) -> None:
        # 信号只连一次，避免切换市场后重复触发
        if not getattr(self, "_wired", False):
            self.cb_market.currentIndexChanged.connect(self._on_market_changed)
            self.cb_symbol.currentIndexChanged.connect(self._on_symbol_changed)
            self.cb_period.currentIndexChanged.connect(self._on_period_changed)
            self.cb_strategy.currentIndexChanged.connect(self._on_strategy_changed)
            self.cb_range.currentIndexChanged.connect(self._schedule)
            self.chk_ma.currentIndexChanged.connect(self._schedule)
            self.chk_short.toggled.connect(self._on_short_toggled)

            self.cb_family.currentIndexChanged.connect(self._on_family_changed)
            self.cb_fmode.currentIndexChanged.connect(self._on_fmode_changed)
            self.cb_filter_mode.currentIndexChanged.connect(self._on_fmode_changed)
            self.chk_filter.toggled.connect(self._on_filter_toggled)
            self.cb_filter_when.currentIndexChanged.connect(
                lambda *_: self._schedule(immediate=True)
            )
            self.cb_fsingle.currentIndexChanged.connect(self._on_single_factor_changed)
            self.chk_frev.toggled.connect(self._on_frev_toggled)
            self.cb_forient.currentIndexChanged.connect(
                lambda *_: self._schedule(immediate=True)
            )
            self.tbl_fmulti.itemChanged.connect(self._on_factor_selection)
            self.cb_fmodel.currentIndexChanged.connect(self._on_model_changed)
            self.btn_fall.clicked.connect(lambda: self._check_all_factors(True))
            self.btn_fnone.clicked.connect(lambda: self._check_all_factors(False))
            self.btn_save_cfg.clicked.connect(self._on_save_config)
            self.btn_save_lib.clicked.connect(self._on_save_library)
            self._wired = True

        self._fill_symbols()
        self._sync_period_controls()
        self._on_strategy_changed()
        self._on_family_changed()
        self._sync_update_controls()
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
        self._sync_period_hint()

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
        self._sync_update_controls()
        self._fill_symbols()
        self._sync_period_controls()
        if not self.store.items:
            cmd = "python download.py" if mk == "us" else "python download_futures.py"
            self.lbl_status.setText(f"{info['label']} 本地无数据，请先运行 {cmd}")
            return
        self._schedule(immediate=True)

    # ---------------------------------------------------------- 周期
    def _period(self) -> str:
        if not hasattr(self, "cb_period"):
            return PERIOD_1D
        return self.cb_period.currentData() or PERIOD_1D

    def _sync_period_controls(self) -> None:
        """美股没有分钟数据，周期锁在日线；期货按品种实际下载情况标注。"""
        if not hasattr(self, "cb_period"):
            return
        if self.market == "us":
            self.cb_period.blockSignals(True)
            idx = self.cb_period.findData(PERIOD_1D)
            self.cb_period.setCurrentIndex(max(idx, 0))
            self.cb_period.setEnabled(False)
            self.cb_period.blockSignals(False)
            self.lbl_period.setText("美股只有日线")
            return
        self.cb_period.setEnabled(True)
        self._sync_period_hint()

    def _sync_period_hint(self) -> None:
        """在当前下拉项上标出「已下载 / 未下载」，避免选到空周期一脸问号。"""
        if self.market == "us" or not hasattr(self, "cb_period"):
            return
        key = self.cb_symbol.currentData() if hasattr(self, "cb_symbol") else None
        miss = []
        for i in range(self.cb_period.count()):
            p = self.cb_period.itemData(i)
            base = PERIOD_LABEL.get(p, p)
            ok = True if (key is None or p == PERIOD_1D) else self.store.has_period(key, p)
            self.cb_period.setItemText(i, base if ok else f"{base}（未下载）")
            if not ok:
                miss.append(p)
        cur = self._period()
        if cur != PERIOD_1D and not (key and self.store.has_period(key, cur)):
            self.lbl_period.setText(
                f"{PERIOD_LABEL.get(cur, cur)} 数据还没下载：\n"
                f"python download_tq.py --main {cur} --symbols ..."
            )
        elif miss:
            self.lbl_period.setText(f"天勤分钟数据已覆盖 {len(GUI_PERIODS) - 1 - len(miss)} 个周期")
        else:
            self.lbl_period.setText("")

    def _on_period_changed(self) -> None:
        self._sync_period_hint()
        self._schedule(immediate=True)

    def _on_symbol_changed(self) -> None:
        # 每个品种下载到的周期不一样，换品种必须重标一次下拉项
        self._sync_period_hint()
        self._schedule()

    # ---------------------------------------------------------- 数据更新
    def _sync_update_controls(self) -> None:
        """美股才需要选数据源和限速；期货固定 Tushare、不限速，换成一行说明。

        合约 / 期权的「一并更新」只对期货有意义（美股没有这两个数据集），
        所以在美股下整块隐藏，避免摆一个点了没反应的开关。
        """
        if not hasattr(self, "cb_upd_source"):
            return
        is_us = self.market == "us"
        self.cb_upd_source.setVisible(is_us)
        self.cb_upd_rate.setVisible(is_us)
        self.lbl_upd_src.setVisible(not is_us)
        for w in (self.lbl_upd_aux, self.chk_upd_contracts, self.chk_upd_options):
            w.setVisible(not is_us)

    def _on_update_clicked(self) -> None:
        # 正在更新时，这个按钮变成「取消」
        if self._upd_thread is not None:
            if self._upd_worker is not None:
                self._upd_worker.cancel()
            self.btn_update.setEnabled(False)
            self.btn_update.setText("取消中")
            self.lbl_upd.setText("正在取消，等待当前请求结束…")
            return

        mk = self.market
        key = self.cb_upd_range.currentData()
        # 期货没有第二个数据源，直接钉死 Tushare
        source = (
            self.cb_upd_source.currentData() if mk == "us" else updater.SOURCE_TUSHARE
        )
        rate = float(self.cb_upd_rate.currentData() or 0.0)
        aux_c = bool(mk == "cn" and self.chk_upd_contracts.isChecked())
        aux_o = bool(mk == "cn" and self.chk_upd_options.isChecked())

        self._upd_worker = UpdateWorker(
            mk, MARKETS[mk]["dir"], key, source, rate,
            aux_contracts=aux_c, aux_options=aux_o,
        )
        self._upd_thread = QThread(self)
        self._upd_worker.moveToThread(self._upd_thread)
        self._upd_thread.started.connect(self._upd_worker.run)
        self._upd_worker.progress.connect(self._on_update_progress)
        self._upd_worker.done.connect(self._on_update_done)
        self._upd_worker.failed.connect(self._on_update_failed)

        self.prog_update.setRange(0, 0)      # 先转圈，等第一个进度再定总量
        self.prog_update.setVisible(True)
        self.lbl_upd.setVisible(True)
        self.lbl_upd.setText(f"准备更新{MARKETS[mk]['label']}…")
        self.btn_update.setText("取消")
        self.btn_update.setEnabled(True)
        self._upd_thread.start()

    def _on_update_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.prog_update.setRange(0, total)
            self.prog_update.setValue(done)
        self.lbl_upd.setText(msg)

    def _on_update_done(self, res) -> None:
        self._finish_update_thread()
        self.lbl_upd.setVisible(True)
        text = res.summary()
        aux = getattr(res, "aux", None)
        if aux is not None:
            text += f"\n衍生数据：{aux.summary()}"
        err = getattr(res, "aux_error", "")
        if err:
            text += f"\n衍生数据更新失败：{err}"
        self.lbl_upd.setText(text)

        self._reload_store(res.market)

        problems = list(res.errors) + list(res.notes)
        if problems or err:
            more = f"\n… 另有 {len(problems) - 6} 条" if len(problems) > 6 else ""
            QMessageBox.information(
                self, "数据更新",
                text + ("\n\n" + "\n".join(problems[:6]) + more if problems else ""),
            )

    def _on_update_failed(self, message: str) -> None:
        self._finish_update_thread()
        self.lbl_upd.setVisible(True)
        self.lbl_upd.setText("更新失败")
        QMessageBox.warning(self, "数据更新失败", message)

    def _finish_update_thread(self) -> None:
        if self._upd_thread is not None:
            self._upd_thread.quit()
            self._upd_thread.wait(5000)
            self._upd_thread.deleteLater()
            self._upd_thread = None
        if self._upd_worker is not None:
            self._upd_worker.deleteLater()
            self._upd_worker = None
        self.btn_update.setText("更新")
        self.btn_update.setEnabled(True)
        self.prog_update.setVisible(False)

    def _reload_store(self, market: str) -> None:
        """重建该市场的 DataStore（清掉 DataFrame 缓存），保留当前选中的标的。"""
        self.stores[market] = DataStore(market)
        for i in range(self.cb_market.count()):
            mk = self.cb_market.itemData(i)
            if mk == market:
                self.cb_market.setItemText(
                    i, f"{MARKETS[mk]['label']}（{len(self.stores[mk].items)} 个标的）"
                )
        if market != self.market:
            return
        keep = self.cb_symbol.currentData()
        self.store = self.stores[market]
        self._fill_symbols()
        idx = self.cb_symbol.findData(keep)
        if idx >= 0:
            self.cb_symbol.blockSignals(True)
            self.cb_symbol.setCurrentIndex(idx)
            self.cb_symbol.blockSignals(False)
        self._schedule(immediate=True)

    def _on_strategy_changed(self) -> None:
        key = self.cb_strategy.currentData()
        spec = STRATEGIES[key]
        self.lbl_desc.setText(spec.desc)
        self._rebuild_params(spec)
        self._schedule(immediate=True)

    # ---------------------------------------------------------- 策略族 / 因子面板
    def _filter_active(self) -> bool:
        """规则策略是否启用了因子过滤。"""
        return self.family == FAMILY_RULE and self.chk_filter.isChecked()

    def _factor_mode(self) -> str:
        """当前生效的因子模式。过滤时取过滤方式，否则取因子策略模式。"""
        if self._filter_active():
            return self.cb_filter_mode.currentData() or "single"
        return self.cb_fmode.currentData() or "single"

    def _on_family_changed(self) -> None:
        self.family = self.cb_family.currentData() or FAMILY_RULE
        is_rule = self.family == FAMILY_RULE
        self.box_rule.setVisible(is_rule)
        self.box_factor.setVisible(not is_rule)
        self.box_filter.setVisible(is_rule)
        self._sync_factor_panel()
        self._schedule(immediate=True)

    def _on_filter_toggled(self, _checked: bool) -> None:
        self._sync_factor_panel()
        self._schedule(immediate=True)

    def _on_fmode_changed(self) -> None:
        self._sync_factor_panel()
        self._schedule(immediate=True)

    def _on_model_changed(self) -> None:
        self._schedule(immediate=True)

    def _sync_factor_panel(self) -> None:
        """按「在做因子策略」还是「在做因子过滤」切换因子面板的标题与可见性。

        两处共用同一套控件：因子策略和因子过滤不可能同时生效，共用能省一半控件，
        而且从策略改成过滤时勾好的因子还在，不用重勾一遍。
        """
        self.factor_mode = self._factor_mode()
        show = (self.family == FAMILY_FACTOR) or self._filter_active()
        self.box_fsel.setVisible(show)
        if not show:
            return

        filtering = self._filter_active()
        self.lbl_fsel.setText("过滤因子" if filtering else "因子选择")
        single = self.factor_mode == "single"
        self.box_fsingle.setVisible(single)
        self.tbl_fmulti.setVisible(not single)
        self.box_fbtns.setVisible(not single)
        if single:
            self._sync_single_rev()

        is_ml = self.factor_mode == "ml"
        self.lbl_fmodel.setVisible(is_ml)
        self.cb_fmodel.setVisible(is_ml)
        # 方向口径对 ML 没意义：模型自己学方向和权重，orient 根本不参与
        self.cb_forient.setVisible(not is_ml)

        if is_ml and not mlmodels.available():
            self.lbl_fmode.setText("当前环境没有可用的机器学习库，请先安装依赖。")

        self._rebuild_fparams()
        self._refresh_fmode_hint()

    def _refresh_fmode_hint(self) -> None:
        if self.family != FAMILY_FACTOR:
            self.lbl_fmode.setText("")
            return
        for key, _label, tip in FACTOR_MODES:
            if key == self.factor_mode:
                self.lbl_fmode.setText(tip)
                break

    def _check_all_factors(self, on: bool) -> None:
        self.tbl_fmulti.blockSignals(True)
        state = Qt.Checked if on else Qt.Unchecked
        for r in range(self.tbl_fmulti.rowCount()):
            cell = self.tbl_fmulti.item(r, 1)
            if cell is not None:
                cell.setCheckState(state)
        self.tbl_fmulti.blockSignals(False)
        self._on_factor_selection()

    # ---------------------------------------------------------- 因子反转
    # 反转状态只有一份，存在多因子表格的「反」列里；单因子模式那个勾选框
    # 只是它的一个镜像。两份状态各存一份的话，切模式就会出现「这里勾了那里没勾」。
    def _row_of_factor(self, key: str) -> int:
        for r in range(self.tbl_fmulti.rowCount()):
            it = self.tbl_fmulti.item(r, 0)
            if it is not None and it.data(Qt.UserRole) == key:
                return r
        return -1

    def _rev_state(self, key: str) -> bool:
        r = self._row_of_factor(key)
        if r < 0:
            return False
        cell = self.tbl_fmulti.item(r, 2)
        return bool(cell is not None and cell.checkState() == Qt.Checked)

    def _set_rev_state(self, key: str, on: bool) -> None:
        r = self._row_of_factor(key)
        if r < 0:
            return
        cell = self.tbl_fmulti.item(r, 2)
        if cell is None:
            return
        self.tbl_fmulti.blockSignals(True)
        cell.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.tbl_fmulti.blockSignals(False)

    def _sync_single_rev(self) -> None:
        """把当前单因子的反转状态刷到勾选框上（不触发重算）。"""
        k = self.cb_fsingle.currentData()
        self.chk_frev.blockSignals(True)
        self.chk_frev.setChecked(bool(k) and self._rev_state(k))
        self.chk_frev.blockSignals(False)

    def _on_single_factor_changed(self, *_args) -> None:
        self._sync_single_rev()
        self._on_factor_selection()

    def _on_frev_toggled(self, on: bool) -> None:
        k = self.cb_fsingle.currentData()
        if k:
            self._set_rev_state(k, bool(on))
        self._schedule(immediate=True)

    def _reversed_factor_keys(self) -> set[str]:
        """勾了「反」的因子。

        这里**不**看「用」列：单因子模式下因子一定是参与计算的（取下拉框那一个），
        但它在多因子表里的「用」可能是没勾的。若这里也要求「用」勾上，
        单因子就会「勾了反转却不生效」。未参与合成的因子由 build_score 里的
        列交集自然忽略，不需要在这一层挡。
        """
        out = set()
        for r in range(self.tbl_fmulti.rowCount()):
            name = self.tbl_fmulti.item(r, 0)
            rev = self.tbl_fmulti.item(r, 2)
            if name is None or rev is None:
                continue
            if rev.checkState() == Qt.Checked:
                out.add(name.data(Qt.UserRole))
        return out

    def _on_factor_selection(self, *_args) -> None:
        keys = self._selected_factor_keys()
        if self.factor_mode == "ml" and len(keys) < 2:
            self.lbl_status.setText("机器学习多因子至少需要勾选 2 个因子")
        self._schedule()

    def _selected_factor_keys(self) -> list[str]:
        """当前选中的因子。单因子取下拉，多因子/ML 取勾选项（保持库内顺序）。"""
        if self.factor_mode == "single":
            k = self.cb_fsingle.currentData()
            return [k] if k else []
        picked = set()
        for r in range(self.tbl_fmulti.rowCount()):
            name = self.tbl_fmulti.item(r, 0)
            use = self.tbl_fmulti.item(r, 1)
            if name is not None and use is not None and use.checkState() == Qt.Checked:
                picked.add(name.data(Qt.UserRole))
        return [k for k in factors.FACTORS if k in picked]

    # ---------------------------------------------------------- 参数滑块
    def _clear_param_box(self, box, widgets: dict, scales: dict) -> None:
        while box.count():
            item = box.takeAt(0)
            wdg = item.widget()
            if wdg:
                wdg.deleteLater()
        widgets.clear()
        scales.clear()

    def _add_param_slider(self, box, widgets: dict, scales: dict,
                          pname: str, label: str, lo, hi, step, default) -> None:
        """一个参数一行：标签 + 滑杆 + 数值。整数参数 scale=1，小数参数 scale=10。"""
        scale = 10 if isinstance(default, float) else 1

        row = QWidget()
        row.setFixedHeight(28)
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(62)
        lbl.setFixedHeight(28)
        lbl.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")

        sld = QSlider(Qt.Horizontal)
        sld.setRange(int(round(lo * scale)), int(round(hi * scale)))
        sld.setSingleStep(max(1, int(round(step * scale))))
        sld.setPageStep(max(1, int(round(step * scale))) * 2)
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

        box.addWidget(row)
        widgets[pname] = sld
        scales[pname] = scale

    def _rebuild_params(self, spec) -> None:
        self._clear_param_box(self.param_box, self.param_widgets, self.param_scales)
        labels = {
            "fast": "快线周期", "slow": "慢线周期", "signal_n": "信号周期",
            "entry": "入场周期", "exit": "离场周期", "n": "通道周期",
            "k": "ATR 倍数", "atr_n": "ATR 周期", "atr_mult": "止损倍数",
        }
        for pname, default in spec.params.items():
            lo, hi, step = spec.param_meta.get(pname, (1, 100, 1))
            self._add_param_slider(
                self.param_box, self.param_widgets, self.param_scales,
                pname, labels.get(pname, pname), lo, hi, step, default,
            )

    def _rebuild_fparams(self) -> None:
        """因子参数。按模式动态增减：ML 才有训练参数，过滤才有门槛参数。"""
        self._clear_param_box(self.fparam_box, self.fparam_widgets, self.fparam_scales)
        filtering = self._filter_active()
        names = ["window", "horizon"]
        if filtering:
            names += ["long_th", "short_th"]
        else:
            names += ["entry_z", "exit_z"]
            if self.factor_mode == "ml":
                names += ["train_window", "step"]
        for pname in names:
            label, lo, hi, step, default = FACTOR_PARAMS[pname]
            self._add_param_slider(
                self.fparam_box, self.fparam_widgets, self.fparam_scales,
                pname, label, lo, hi, step, default,
            )

    def _current_params(self) -> dict:
        out = {}
        for name, wdg in self.param_widgets.items():
            scale = self.param_scales.get(name, 1)
            v = wdg.value() / scale
            out[name] = int(round(v)) if scale == 1 else v
        return out

    # ---------------------------------------------------------- 策略配置存取
    def _current_config(self) -> SStore.StrategyConfig:
        """把界面上这一套选择收成一份可保存的配置。

        非当前策略族的那一组字段也照收（规则参数在因子模式下照样保留）——
        用户在两边切来切去不该丢参数。
        """
        skey = self.cb_symbol.currentData() or ""
        if self.market == "cn":
            sym, _, ex = skey.rpartition("_")
        else:
            # 美股的 key 就是代码本身（AAPL），没有「交易所」这一维。
            # 不能一律 rpartition("_")：AAPL 会被拆成 ("", "", "AAPL")。
            sym, ex = skey, ""
        is_factor = self.family == FAMILY_FACTOR
        fc = self._factor_cfg() if is_factor else None
        return SStore.StrategyConfig(
            name=self.ed_cfg_name.text().strip(),
            symbol=sym,
            exchange=ex,
            period=self._period(),
            family=self.family,
            rule_key=self.cb_strategy.currentData() or "",
            rule_params=self._current_params(),
            factor_mode=self.factor_mode if is_factor else "",
            factor_keys=self._selected_factor_keys() if is_factor else [],
            factor_reverse={k: True for k in self._reversed_factor_keys()},
            factor_cfg=dict(vars(fc)) if fc is not None else {},
            volume=1,
            note="short" if self.chk_short.isChecked() else "",
        )

    def apply_config(self, cfg: SStore.StrategyConfig) -> None:
        """把一份配置灌回界面。组合页双击一行、启动时恢复上次配置都走这里。

        顺序不能乱：切市场会重建品种列表，所以必须先把市场切过去，再找品种。
        """
        mk = "cn" if cfg.exchange else "us"
        i = self.cb_market.findData(mk)
        if i >= 0 and i != self.cb_market.currentIndex():
            self.cb_market.setCurrentIndex(i)      # 同步触发 _on_market_changed

        skey = f"{cfg.symbol}_{cfg.exchange}" if cfg.exchange else cfg.symbol
        j = self.cb_symbol.findData(skey)
        if j >= 0:
            self.cb_symbol.setCurrentIndex(j)
        k = self.cb_period.findData(cfg.period)
        if k >= 0:
            self.cb_period.setCurrentIndex(k)

        f = self.cb_family.findData(cfg.family)
        if f >= 0:
            self.cb_family.setCurrentIndex(f)
        r = self.cb_strategy.findData(cfg.rule_key)
        if r >= 0:
            self.cb_strategy.setCurrentIndex(r)     # 重建参数控件
        for name, val in (cfg.rule_params or {}).items():
            w = self.param_widgets.get(name)
            if w is not None:
                try:
                    w.setValue(int(round(float(val) * self.param_scales.get(name, 1))))
                except (TypeError, ValueError):
                    continue

        if cfg.family == FAMILY_FACTOR:
            m = self.cb_fmode.findData(cfg.factor_mode)
            if m >= 0:
                self.cb_fmode.setCurrentIndex(m)
            self._apply_factor_keys(cfg.factor_keys, cfg.factor_reverse)
            self._apply_factor_cfg(cfg.factor_cfg)

        self.chk_short.setChecked(cfg.note == "short")
        self.ed_cfg_name.setText(cfg.name)
        self._schedule(immediate=True)

    def _apply_factor_keys(self, keys, reverse) -> None:
        """恢复因子选择：单因子设下拉，多因子勾表格，反向勾「反」列。"""
        picked = set(keys or [])
        if picked:
            i = self.cb_fsingle.findData(sorted(picked)[0])
            if i < 0:  # 保持配置里的顺序，优先第一项
                i = self.cb_fsingle.findData(list(keys)[0])
            if i >= 0:
                self.cb_fsingle.setCurrentIndex(i)
        # 逐行 setCheckState 会逐行触发 itemChanged，每行都排一次重算 ——
        # 47 行就是 47 次。整段屏蔽信号，最后一次统一刷。
        self.tbl_fmulti.blockSignals(True)
        try:
            for r in range(self.tbl_fmulti.rowCount()):
                name = self.tbl_fmulti.item(r, 0)
                if name is None:
                    continue
                k = name.data(Qt.UserRole)
                use = self.tbl_fmulti.item(r, 1)
                rev = self.tbl_fmulti.item(r, 2)
                if use is not None:
                    use.setCheckState(Qt.Checked if k in picked else Qt.Unchecked)
                if rev is not None:
                    on = bool((reverse or {}).get(k))
                    rev.setCheckState(Qt.Checked if on else Qt.Unchecked)
        finally:
            self.tbl_fmulti.blockSignals(False)

    def _apply_factor_cfg(self, cfg: dict) -> None:
        if not cfg:
            return
        for name, wdg in self.fparam_widgets.items():
            if name not in cfg:
                continue
            try:
                wdg.setValue(int(round(float(cfg[name]) * self.fparam_scales.get(name, 1))))
            except (TypeError, ValueError):
                continue
        for key, cb in (("model", self.cb_fmodel), ("orient", self.cb_forient)):
            v = cfg.get(key)
            if v:
                i = cb.findData(v)
                if i >= 0:
                    cb.setCurrentIndex(i)

    def _on_save_config(self) -> None:
        cfg = self._current_config()
        if not cfg.symbol:
            self.lbl_status.setText("还没选标的，无法保存配置")
            return
        SStore.save_last(cfg)
        msg = "已保存配置，下次打开自动恢复"
        if self.chk_to_portfolio.isChecked():
            # 连点两下会加两条一模一样的策略，跑起来就是双倍仓位。同一个策略
            # 的同一套参数已经在组合里就不再追加。
            same = [c for c in SStore.load_portfolio() if c.same_shape(cfg)]
            if same:
                msg += "；组合里已有相同配置，没有重复添加"
            else:
                item = SStore.add_to_portfolio(cfg)
                msg += f"；已加入策略交易列表：{item.title()}"
        self.lbl_status.setText(msg)

    def _on_save_library(self) -> None:
        cfg = self._current_config()
        if not cfg.symbol:
            self.lbl_status.setText("还没选标的，无法存入策略库")
            return
        saved = SStore.add_to_library(cfg)
        msg = f"已存入策略库：{saved.title()}"
        if self.chk_to_portfolio.isChecked():
            item = SStore.add_to_portfolio(cfg)
            msg += f"；已加入策略交易列表：{item.title()}"
        self.lbl_status.setText(msg)

    def _restore_last_config(self) -> None:
        """启动时恢复上次保存的配置。

        失败就静默留在默认状态 —— 一份配置读不出来不该挡住程序启动。
        """
        try:
            cfg = SStore.load_last()
        except Exception:  # noqa: BLE001
            return
        if cfg is None or not cfg.symbol:
            return
        try:
            self.apply_config(cfg)
            self.lbl_status.setText(f"已恢复上次配置：{cfg.title()}")
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"恢复上次配置失败：{exc}")

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
        # 机器学习要跑几十折 walk-forward，拖滑杆时 220ms 就重算一次会把线程
        # 反复打断。给 ML 更长的防抖，让人先拖完再算。
        delay = 600 if self.factor_mode == "ml" and (
            self.family == FAMILY_FACTOR or self._filter_active()
        ) else 220
        QTimer.singleShot(delay, self._deferred_run)

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

    def _factor_cfg(self) -> fstrat.FactorConfig:
        """从滑块读出因子配置。缺失的参数（被隐藏的）走默认值。"""
        def get(name, default):
            sld = self.fparam_widgets.get(name)
            if sld is None:
                return default
            return sld.value() / self.fparam_scales.get(name, 1)

        return fstrat.FactorConfig(
            window=int(get("window", 252)),
            horizon=int(get("horizon", 5)),
            entry_z=float(get("entry_z", 1.0)),
            exit_z=float(get("exit_z", 0.3)),
            orient=self.cb_forient.currentData() or "auto",
            reverse={k: True for k in self._reversed_factor_keys()},
            model=self.cb_fmodel.currentData() or mlmodels.DEFAULT_MODEL,
            train_window=int(get("train_window", 1000)),
            step=int(get("step", 180)),
            filter_mode=self.cb_filter_when.currentData() or "entry",
            long_th=float(get("long_th", 0.0)),
            short_th=float(get("short_th", 0.0)),
        )

    def run_analysis(self, df_override: pd.DataFrame | None = None) -> None:
        skey = self.cb_symbol.currentData()
        if not skey:
            return
        period = self._period()
        # 实时推送时由调用方把「本地历史 + 实时段」拼好的表递进来，
        # 否则从本地仓库读。
        df = df_override if df_override is not None else self.store.get(skey, period)
        if df is None or df.empty:
            why = getattr(self.store, "last_error", "")
            if "另一个程序正在使用" in why or "Cannot open file" in why:
                self._clear_chart(
                    "行情库正被下载脚本占用（DuckDB 单文件排他锁）\n"
                    "等下载跑完，或先停掉 download_tq.py 再看图"
                )
            elif period == PERIOD_1D:
                self._clear_chart(f"{skey} 数据缺失")
            else:
                self._clear_chart(
                    f"{skey} 没有 {PERIOD_LABEL.get(period, period)} 数据\n"
                    f"先跑：python download_tq.py --main {period} --all"
                )
            return

        # 年化口径必须跟着周期走：5 分钟一年约 1.2 万根，按 252 算会把波动
        # 低估一个数量级、夏普虚高一个数量级。
        self.cfg.bars_per_year = bars_per_year(df.index)

        self.df_raw = df
        rule_signal = None
        base_df = df

        if self.family == FAMILY_RULE:
            rkey = self.cb_strategy.currentData()
            if not rkey:
                return
            try:
                base_df = run_strategy(df, rkey, self._current_params())
            except Exception as exc:  # noqa: BLE001
                self.lbl_status.setText(f"策略计算失败：{exc}")
                return
            rule_signal = base_df["signal"].to_numpy(int)

        need_factor = (self.family == FAMILY_FACTOR) or self._filter_active()
        if not need_factor:
            self._finish(rule_signal_df=base_df, label=STRATEGIES[self.cb_strategy.currentData()].label)
            return

        keys = self._selected_factor_keys()
        if not keys:
            self.lbl_status.setText("没有选中因子")
            return
        if self.factor_mode == "ml" and len(keys) < 2:
            self.lbl_status.setText("机器学习多因子至少需要勾选 2 个因子")
            return

        cfg = self._factor_cfg()
        if self.factor_mode == "ml":
            self._start_factor_worker(df, keys, cfg, rule_signal)
            return

        # 单因子 / 多因子只要几十毫秒，同步算完即可，不必起线程
        try:
            if rule_signal is None:
                out = fstrat.run_factor_strategy(df, self.factor_mode, keys, cfg)
            else:
                out = fstrat.run_rule_with_filter(
                    df, rule_signal, self.factor_mode, keys, cfg
                )
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"因子计算失败：{exc}")
            return
        self._apply_outcome(out, rule_signal)

    def _start_factor_worker(self, df, keys, cfg, rule_signal) -> None:
        """ML 模式走后台线程。已经在跑的先取消，避免旧结果盖掉新结果。"""
        self._cancel_factor_worker()
        self._f_seq += 1
        seq = self._f_seq

        worker = FactorWorker(df, self.factor_mode, keys, cfg, rule_signal)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._on_factor_progress)
        worker.done.connect(lambda out, s=seq: self._on_factor_done(out, s))
        worker.failed.connect(lambda msg, s=seq: self._on_factor_failed(msg, s))
        worker.done.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)

        self._f_worker, self._f_thread = worker, thread
        self.lbl_status.setText(
            f"正在训练 {mlmodels.MODELS.get(cfg.model, {}).get('label', cfg.model)}"
            f"（{len(keys)} 因子，{cfg.train_window} 根训练窗口）…"
        )
        thread.start()

    def _cancel_factor_worker(self) -> None:
        if self._f_worker is not None:
            self._f_worker.cancel()
            # 断开信号再退出：否则排队中的 done 会在窗口销毁后才被投递，
            # PySide6 抛 "Internal C++ object already deleted"。
            for sig in (self._f_worker.done, self._f_worker.failed,
                        self._f_worker.progress):
                try:
                    sig.disconnect()
                except (RuntimeError, TypeError):
                    pass
        thread = self._f_thread
        if thread is not None:
            try:
                if thread.isRunning():
                    thread.quit()
                    thread.wait(4000)
            except RuntimeError:
                # C++ 对象已被 deleteLater 销毁，没什么可等的
                pass
        self._f_worker = None
        self._f_thread = None

    def _on_factor_progress(self, done: int, total: int, msg: str) -> None:
        self.lbl_status.setText(f"{msg}（共 {total} 折）")

    def _on_factor_done(self, out, seq: int) -> None:
        self._f_worker = None
        self._f_thread = None
        if self._closing or seq != self._f_seq or seq == self._f_applied:
            return              # 关窗中 / 过期 / 已落地
        self._f_applied = seq
        self._apply_outcome(out, None)

    def _on_factor_failed(self, msg: str, seq: int) -> None:
        self._f_worker = None
        self._f_thread = None
        if self._closing or seq != self._f_seq:
            return
        self.lbl_status.setText(f"机器学习训练失败：{msg}")

    def _clear_chart(self, msg: str) -> None:
        """没有数据时把图清空并写一句话。

        不清的话会留着上一个品种/上一个周期的图，配上一句「数据缺失」，
        看起来像是这个品种算出来了但结论为空 —— 比直接空白更误导。
        """
        self.df_raw = None
        self.df_sig = None
        self.res = None
        self.bh = None
        self.factor_outcome = None
        self.lbl_status.setText(msg)
        for card in ("card_cagr", "card_dd", "card_sharpe", "card_excess"):
            w = getattr(self, card, None)
            if w is not None:
                w.set_value("—", C_MUTED)
        if hasattr(self, "tbl_factor"):
            self.tbl_factor.setRowCount(0)
        try:
            fig = self.canvas_k.fresh()
            ax = fig.add_subplot(111)
            ax.axis("off")
            ax.text(0.5, 0.5, msg, ha="center", va="center",
                    color=C_MUTED, fontsize=12, wrap=True)
            self.canvas_k.draw()
        except Exception:
            pass

    def _finish(self, rule_signal_df: pd.DataFrame, label: str) -> None:
        """规则策略（不含因子）的收尾：回测 + 重绘。"""
        self.factor_outcome = None
        self.df_sig = rule_signal_df
        self._strategy_label = label
        self.res = run_backtest(rule_signal_df, self.cfg, label=label)
        self.bh = buy_and_hold(self.df_raw, self.cfg)
        self._draw_kline()
        self._draw_perf()
        self._update_metrics()
        self._update_table()
        self._refresh_factor_tab()

    def _apply_outcome(self, out: fstrat.FactorOutcome, rule_signal) -> None:
        """因子计算完成后的统一收尾。"""
        if self._closing:
            return
        self.factor_outcome = out
        self.df_sig = out.df
        if self.family == FAMILY_FACTOR:
            label = out.label
        else:
            rlabel = STRATEGIES[self.cb_strategy.currentData()].label
            label = f"{rlabel} + 因子过滤（{out.pack.label}）"
        self._strategy_label = label
        self.res = run_backtest(out.df, self.cfg, label=label)
        self.bh = buy_and_hold(self.df_raw, self.cfg)
        self._draw_kline()
        self._draw_perf()
        self._update_metrics()
        self._update_table()
        self._refresh_factor_tab()

        # 展期/期权因子需要额外的合约与期权数据。美股没有，部分期货品种也可能
        # 没下过衍生数据 —— 这种情况分数全为 NaN，回测结果是「一笔交易都没有」。
        # 不解释一句的话，看起来就像策略本身没信号，实际是数据没到。
        score = getattr(out.pack, "score", None)
        if score is not None and int(score.notna().sum()) == 0:
            why = "需要真实合约 / 期权数据" if self.market == "cn" else "美股没有合约与期权数据"
            self.lbl_status.setText(
                f"所选因子在当前标的没有任何可用值（{why}），因此不会产生信号"
            )

    # ---------------------------------------------------------- 实时行情
    def _wire_live(self) -> None:
        self.live.started.connect(self._on_live_started)
        self.live.stopped.connect(self._on_live_stopped)
        self.live.bars.connect(self._on_live_bars)
        self.live.state.connect(self._on_live_state)
        self.live.failed.connect(self._on_live_failed)
        self.live.log.connect(self._on_live_log)

    def _current_pair(self) -> tuple[str, str] | None:
        item = self._current_item()
        if not item:
            return None
        key = item["key"]
        sym, _, ex = key.rpartition("_")
        if not sym or not ex:
            return None
        return sym, ex

    def _on_live_toggled(self, on: bool) -> None:
        if not on:
            self._on_live_stop()
            return
        if self.market != "cn":
            self._revert_live("实时行情只接了国内期货")
            return
        period = self._period()
        if period == PERIOD_1D:
            self._revert_live("实时行情需要分钟周期，先选 5 分钟或 1 小时")
            return
        pair = self._current_pair()
        if pair is None:
            self._revert_live("先选一个品种")
            return
        spec = self.trade_page._current_spec()
        if spec is None:
            self._revert_live("先在「账户透视和交易」页添加一个天勤账户")
            return
        self.live.start(spec, pair[0], pair[1], period, need_history=True)
        self.btn_live.setText("停止实时")
        self.lbl_live_msg.setText("正在连接…")

    def _revert_live(self, msg: str) -> None:
        self.btn_live.blockSignals(True)
        self.btn_live.setChecked(False)
        self.btn_live.blockSignals(False)
        self.btn_live.setText("启动实时")
        self.lbl_live_msg.setText(msg)

    def _on_trade_connect(self, spec) -> None:
        """从交易页点「连接」：沿用当前选中的品种与周期。

        **不补本地历史**。这个动作的语义是「登录账户」，用户要看的是持仓/资金/
        委托，而 `hub.start()` 之后这些东西就有了。补历史一个品种十年 5m 要几十
        秒到几分钟，撞上天勤运维窗口（19:00-19:30）还会一直等到 19:31 —— 为了
        画 K 线把「登录」卡住几分钟是不划算的。需要长历史的场景是研究页的实时
        K 线，那里走 `_on_live_toggled`，它照样补。
        """
        if self.market != "cn":
            self.trade_page.on_failed("交易只接了国内期货")
            return
        pair = self._current_pair()
        period = self._period() if self._period() != PERIOD_1D else "5m"
        if pair is None:
            self.trade_page.on_failed("先在「策略研究」页选一个品种")
            return
        self.live.start(spec, pair[0], pair[1], period, need_history=False)
        self.lbl_live_msg.setText("正在连接…")
        self.btn_live.blockSignals(True)
        self.btn_live.setChecked(True)
        self.btn_live.blockSignals(False)
        self.btn_live.setText("停止实时")

    def _on_live_stop(self) -> None:
        self.live.stop(wait=False)
        self.btn_live.blockSignals(True)
        self.btn_live.setChecked(False)
        self.btn_live.blockSignals(False)
        self.btn_live.setText("启动实时")
        self.lbl_live_msg.setText("实时已停止")
        if self._trading:
            self.btn_trade.setChecked(False)

    def _on_live_started(self, key: str) -> None:
        sym, _, ex = key.rpartition("_")
        self._live_key = key
        self._live_period = self._period()
        self._live_seq += 1
        self._live_last = None
        self._last_target = 0
        # 工作线程刚把缺口补进库，缓存里的旧数据要作废，否则实时段会接在旧尾巴上
        self.store.invalidate(key, self._live_period)
        self._live_base = self.store.get(key, self._live_period)
        # 从交易页连的账户不补历史（见 `_on_trade_connect`），这时本地一根都没有。
        # 写「0 根历史」看着像出错，直接说清楚是没补。
        n_base = 0 if self._live_base is None else len(self._live_base)
        tail = f"{n_base:,} 根历史" if n_base else "未补本地历史"
        self.lbl_live_msg.setText(
            f"实时中 · {sym} {PERIOD_LABEL.get(self._live_period, '')}（{tail}）"
        )
        self.lbl_live.setText("● 实时")
        self.lbl_live.setStyleSheet(f"color: {C_UP}; font-size: 11px; font-weight: 600;")

    def _on_live_stopped(self) -> None:
        self.lbl_live.setText("实时未启动")
        self.lbl_live.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self.btn_live.blockSignals(True)
        self.btn_live.setChecked(False)
        self.btn_live.blockSignals(False)
        self.btn_live.setText("启动实时")

    def _on_live_log(self, msg: str) -> None:
        self.lbl_live_msg.setText(msg)
        self.trade_page.on_log(msg)

    def _on_live_failed(self, msg: str) -> None:
        self.lbl_live_msg.setText(msg)
        self.trade_page.on_failed(msg)

    def _on_live_state(self, st) -> None:
        self._live_state = st
        q = st.quote or {}
        if q and self._live_base is not None:
            und = q.get("underlying") or ""
            self.lbl_live_msg.setText(
                f"{q.get('name', '')} {und} {q.get('last', 0):.2f}"
                f" · 买 {q.get('bid', 0):.2f} 卖 {q.get('ask', 0):.2f}"
            )

    def _on_live_bars(self, live_df) -> None:
        """实时 K 线到了：新 bar 就整体重算，同一根内只刷价格。"""
        if self._closing or self._live_base is None or live_df is None or live_df.empty:
            return
        pair = self._current_pair()
        if pair is None or self._live_key != f"{pair[0]}_{pair[1]}":
            return          # 推送还是上一个品种的，丢掉
        merged = tqlive.merge_live(self._live_base, live_df)
        if merged.empty:
            return
        ts = merged.index[-1]
        new_bar = self._live_last is not None and ts != self._live_last
        first = self._live_last is None
        self._live_last = ts
        if first or new_bar:
            # 新 bar 走完整重算：信号只认走完的 bar，避免一根没走完的 K 线反复开平
            self.run_analysis(df_override=merged)
            self._maybe_auto_trade()
        else:
            self._patch_live_bar(merged)

    def _patch_live_bar(self, merged: pd.DataFrame) -> None:
        """把还没走完的那根 bar 的最新价刷到图上，不重算策略。

        只有用户正停在最新一段时才动 —— 滚到历史位置时 `_k_view` 的最后一根
        是历史 bar，改它等于篡改历史。
        """
        if self.df_sig is None or self._k_view is None or not len(self._k_view):
            return
        ts = merged.index[-1]
        if self._k_view.index[-1] != ts:
            return
        cols = [c for c in ("open", "high", "low", "close", "volume") if c in merged.columns]
        for c in cols:
            if c in self.df_sig.columns:
                self.df_sig.loc[ts, c] = merged.loc[ts, c]
            self._k_view.loc[ts, c] = merged.loc[ts, c]
        self._render_kline()

    # ---------------------------------------------------------- 交易开关
    def _on_trade_toggled(self, on: bool) -> None:
        if on and not self.live.running:
            self.btn_trade.blockSignals(True)
            self.btn_trade.setChecked(False)
            self.btn_trade.blockSignals(False)
            self.lbl_live_msg.setText("先「启动实时」并连上账户，才能启动交易")
            return
        self._trading = bool(on)
        self._last_target = 0
        if on:
            self.btn_trade.setText("停止交易")
            self.lbl_trade_icon.setText("☺")
            self.lbl_trade_icon.setStyleSheet(f"font-size: 17px; color: {C_DOWN};")
            self.lbl_trade_state.setText("交易中")
            self.lbl_trade_state.setStyleSheet(
                f"font-size: 11px; color: {C_DOWN}; font-weight: 600;"
            )
        else:
            self.btn_trade.setText("启动交易")
            self.lbl_trade_icon.setText("☹")
            self.lbl_trade_icon.setStyleSheet(f"font-size: 17px; color: {C_UP};")
            self.lbl_trade_state.setText("未交易")
            self.lbl_trade_state.setStyleSheet(f"font-size: 11px; color: {C_MUTED};")

    def _maybe_auto_trade(self) -> None:
        """绿灯时按最新信号把仓位调到目标方向（1 手）。

        用 `TargetPosTask` 而不是自己 `insert_order`：反手要「先平后开」，
        上期所还要求区分平今/平仓，这些规则各交易所不一样，手写必然踩坑。
        TargetPosTask 按目标净仓自己拆单、自己追价。
        """
        if not self._trading or self.df_sig is None or not len(self.df_sig):
            return
        st = self._live_state
        if st is None or not st.underlying:
            return
        sig = int(self.df_sig["signal"].iloc[-1])
        target = sig * 1
        if target == self._last_target:
            return
        self._last_target = target
        self.live.set_target(st.underlying, target)
        self.lbl_trade_state.setText(
            f"交易中 · 目标 {target:+d} 手" if target else "交易中 · 空仓"
        )


    # ---------------------------------------------------------- K 线绘制
    def _draw_kline(self) -> None:
        """切好数据、把可视窗口重置到最新一段，再交给 _render_kline 画。

        拆成两步是为了滚动/缩放：滚动时数据不变，只挪窗口，没必要重新切
        DataFrame、重算策略——那部分在 _render_kline 里按窗口切片完成。
        """
        df = self.df_sig
        if df is None or not len(df):
            return

        years = self.cb_range.currentData()
        if years and len(df):
            cutoff = df.index[-1] - pd.DateOffset(years=years)
            view = df[df.index >= cutoff]
        else:
            view = df
        if len(view) < 20:
            view = df.tail(120)

        self._k_view = view
        self._k_total = len(view)
        # 换标的 / 换策略 / 换区间时回到最新一段，符合看盘习惯
        self._k_win = min(K_WIN_DEFAULT, self._k_total)
        self._k_start = max(0, self._k_total - self._k_win)
        self._k_spec_label = getattr(self, "_strategy_label", "")
        self._k_key = self.cb_strategy.currentData() if self.family == FAMILY_RULE else ""
        # 换图是用户主动操作，不受滚动节流限制
        self._k_last_draw = time.monotonic()
        self._k_pending = False
        self._render_kline()

    def _render_kline(self) -> None:
        """按当前窗口 [start, start+win) 重画 K 线，窗口外的数据不进图元。

        整个重建过程在画布的 frozen() 临界区里跑，避免半成品状态被重绘。
        """
        view_all = self._k_view
        if view_all is None or not len(view_all):
            return
        with self.canvas_k.frozen():
            self._render_kline_impl()

    def _render_kline_impl(self) -> None:
        view_all = self._k_view
        if view_all is None or not len(view_all):
            return

        total = self._k_total
        win = max(K_WIN_MIN, min(int(self._k_win), total))
        start = max(0, min(total - win, int(self._k_start)))
        self._k_win, self._k_start = win, start

        view = view_all.iloc[start:start + win]
        spec_label = self._k_spec_label
        item = self._current_item()
        sym = item["symbol"]

        x = np.arange(len(view))
        o = view["open"].to_numpy(float)
        h = view["high"].to_numpy(float)
        l = view["low"].to_numpy(float)
        c = view["close"].to_numpy(float)
        vol = view["volume"].to_numpy(float)
        up = c >= o

        # 全量的信号与收盘价：窗口第一根到底是不是「开仓」，得看它的前一根，
        # 只拿窗口内的数据判断会把每根都误认成开仓点。
        sig_full = view_all["signal"].to_numpy(int)
        if self.cfg.long_only:
            sig_full = np.maximum(sig_full, 0)
        close_full = view_all["close"].to_numpy(float)
        sig_view = sig_full[start:start + win]
        prev = np.concatenate([[0], sig_full[:-1]])[start:start + win]

        # 因子模式下多画一行分数曲线：能直接看到「分数在哪里穿过阈值、
        # 穿过去之后策略怎么走」，比只看三角标记有用得多。
        has_score = "factor_score" in view_all.columns
        score_full = (
            view_all["factor_score"].to_numpy(float) if has_score else None
        )

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
            for col, label, color, lw in overlay_cfg.get(self._k_key, []):
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
            gain = (close_full[x_i] - close_full[e_i]) * d
            ax.plot(
                [e_i - start, b - start], [close_full[e_i], close_full[b]],
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
            cfg = self._factor_cfg()
            filtering = self._filter_active()
            if filtering:
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
            else:
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
        span = (view.index[-1] - view.index[0]).total_seconds()
        if span <= 2 * 86400:
            fmt = "%m-%d %H:%M"
        elif span <= 150 * 86400:
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
        ax.set_ylabel(MARKETS[self.market]["price_note"], fontsize=9, color=C_MUTED)

        # 标题信息
        name = item["name"]
        exch = item["exchange"]
        exch_cn = EXCHANGE_CN.get(exch, exch)
        head = f"{sym}  {name}" + (f"（{exch_cn}）" if exch else "")
        # 「当前状态」永远指最新一根，不是窗口最后一根——否则滚到历史位置时
        # 标题会跟着变，看着像当前真的空仓了。
        state = {1: "多头持仓", -1: "空头持仓"}.get(int(sig_full[-1]), "空仓")
        period = self._period()
        plabel = "" if period == PERIOD_1D else f" · {PERIOD_LABEL.get(period, period)}"
        ax.set_title(
            f"{head}{plabel}   ·   {spec_label}   ·   "
            f"{view.index[0]:%Y-%m-%d %H:%M} ~ {view.index[-1]:%Y-%m-%d %H:%M}"
            f"   ·   当前状态：{state}",
            fontsize=11.5, color=C_TEXT, loc="left", pad=10,
        )
        if self.market == "us":
            source = "东方财富后复权"
        elif period == PERIOD_1D:
            source = "Tushare 换月复权"
        else:
            src = tqdata.PERIODS.get(period, (period, 1))
            merged = f"（由 {src[0]} × {src[1]} 合并）" if src[1] > 1 else ""
            source = f"天勤换月复权{merged}"
        shown = f"{start + 1:,}–{start + win:,}"
        self.lbl_chart_title.setText(
            f"全历史 {len(self.df_sig):,} 根 · 当前显示第 {shown} 根"
            f"（窗口 {win:,} 根）· 滚轮缩放 / 拖下方滚动条看历史 · {source}"
        )
        self.lbl_chart_title.setStyleSheet(
            f"font-size: 11px; color: {C_MUTED}; padding-left: 4px;"
        )

        self._sync_kline_scrollbar()
        # 不用再手动 draw()：退出 frozen() 临界区时会同步补画一帧，
        # 那样能保证「重建完成」和「绘制」之间不会插进别的重绘。

    # ---------------------------------------------------------- 窗口滚动 / 缩放
    def _sync_kline_scrollbar(self) -> None:
        """让滚动条的范围跟上当前数据量与窗口大小。

        数据比窗口还短时直接隐藏——一根能滚的余地都没有，留着只会让人以为坏了。
        """
        total, win = self._k_total, self._k_win
        if total <= win or not hasattr(self, "sb_kline"):
            self.sb_kline.setVisible(False)
            return
        self.sb_kline.setVisible(True)
        self.sb_kline.blockSignals(True)   # 避免 setValue 反过来触发一次重绘
        self.sb_kline.setRange(0, total - win)
        self.sb_kline.setPageStep(win)
        self.sb_kline.setSingleStep(max(1, win // 10))
        self.sb_kline.setValue(self._k_start)
        self.sb_kline.blockSignals(False)

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
            f"仓位时间 {m['exposure']*100:.0f}% · 平均持仓 {self._hold_text(m['avg_bars'])}{dir_txt}"
        )

    def _hold_text(self, bars: float) -> str:
        """平均持仓长度的可读写法。

        `avg_bars` 的单位是「根」，日线一根就是一天，分钟级一根只有几十分钟。
        统一写「天」会把 1 小时的 160 根说成 160 天（实际约 32 个交易日）。
        """
        if self._period() == PERIOD_1D:
            return f"{bars:.0f} 天"
        bpy = float(self.cfg.bars_per_year or 252)
        days = bars / bpy * 365.25 if bpy > 0 else 0.0
        return f"{bars:.0f} 根（约 {days:.0f} 天）"

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
            ("平均持仓", self._hold_text(m["avg_bars"]), "—", None),
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

    # ---------------------------------------------------------- 因子分析页
    def _ensure_report(self) -> pd.DataFrame:
        """当前标的的全因子评估表。按「市场+标的」缓存，切参数不用重算。

        47 个因子约 0.5 秒，每次拖滑杆都重算会明显拖慢交互，所以缓存住。
        """
        key = f"{self.market}:{self.cb_symbol.currentData()}"
        if self._f_report is not None and self._f_report_key == key:
            return self._f_report
        df = self.df_raw
        if df is None or df.empty:
            return pd.DataFrame()
        cfg = self._factor_cfg()
        try:
            rep = factor_metrics.factor_report(
                df, list(factors.FACTORS), horizon=cfg.horizon,
                window=cfg.window, method=cfg.method,
            )
        except Exception:  # noqa: BLE001
            rep = pd.DataFrame()
        self._f_report = rep
        self._f_report_key = key
        return rep

    def _refresh_factor_tab(self) -> None:
        """刷新因子分析页：表格 + 四格图。没有因子在用时也显示全因子一览。"""
        df = self.df_raw
        if df is None or df.empty:
            return
        cfg = self._factor_cfg()
        rep = self._ensure_report()
        if rep.empty:
            self.lbl_frep.setText("因子评估失败")
            return

        out = self.factor_outcome
        selected = list(out.pack.fmat.columns) if out is not None else self._selected_factor_keys()
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
            ic_color = None
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

        keys = list(out.pack.fmat.columns) if out is not None else self._selected_factor_keys()
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
                    f"滚动 IC · {factors.FACTORS[primary].label}"
                    f"（均值 {mu:+.3f}）",
                    fontsize=9.5, color=C_TEXT, loc="left",
                )
            else:
                a_ic.set_title("滚动 IC（样本不足）", fontsize=9.5, color=C_TEXT, loc="left")
        else:
            a_ic.set_title("滚动 IC（未选因子）", fontsize=9.5, color=C_TEXT, loc="left")
        a_ic.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
        # 20 年日频，默认定位器会塞十几个年份标签，在这么小的子图里会连成一片。
        # 限到 5 个刻度并只留年份，够看趋势又不糊。
        a_ic.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
        a_ic.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
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
                a_last.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=5))
                a_last.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
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

    # ---------------------------------------------------------- 生命周期
    def closeEvent(self, event) -> None:
        """关窗前先让更新线程收尾，避免线程还在写 CSV 就被强杀。"""
        self._closing = True
        if self._k_redraw_timer.isActive():
            self._k_redraw_timer.stop()
        if self._upd_worker is not None:
            self._upd_worker.cancel()
        if self._upd_thread is not None:
            self._upd_thread.quit()
            self._upd_thread.wait(3000)
        self._cancel_factor_worker()
        # 组合页每条账户连接一个 QThread，窗口关掉时线程还在跑 wait_update
        # 会让解释器退出时崩在 TqApi 上。这里等它们真的停下来再放行。
        self.portfolio_page.shutdown()
        self.agent_page.shutdown()
        self.live.stop(wait=True)
        super().closeEvent(event)


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
