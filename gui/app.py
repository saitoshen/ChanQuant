"""趋势跟踪研究台 —— PySide6 桌面程序。

功能：
- 左侧选择标的 / 策略族（规则 or 因子）/ 参数 / 显示区间 / 仓位口径
- 「K线与信号」页：**前复权** K 线（最新一根等于真实盘面价，实盘按它算手数）、
  策略指标线、持仓区间底色、买卖信号标记（画在信号后的下一根开盘）、成交量，
  因子策略下额外画一行因子分数；双击出十字光标读单根行情
- 「绩效对比」页：策略 vs 买入持有的资金曲线、回撤曲线、指标对比表
- 「因子分析」页：全因子 IC/ICIR/t 值一览 + IC 时序、衰减、分位收益、
  模型重要度
"""

from __future__ import annotations

import os
import sys
import time

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

import matplotlib.pyplot as plt
import pandas as pd

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ttrade import factors, mlmodels, specs, tqdata, tqlive
from ttrade import factor_strategy as fstrat
from ttrade import replay
from ttrade import strategy_store as SStore
from gui.agent_page import AgentPage
from gui.data_page import DataPage
from gui.live import LiveController
from gui.portfolio_page import LibraryDialog, PortfolioPage
from gui.result_view import ResultView, ViewCtx
from gui.trade_page import TradePage
from ttrade.backtest import BacktestConfig, bars_per_year, buy_and_hold, run_backtest
from ttrade.eastmoney import load_local
from ttrade.strategies import (
    PARAM_LABELS, STRATEGIES, available_text, resolve_key, run_strategy,
)
from ttrade.universe import UNIVERSE, cn_name, sector_of

DATA_DIR = resolve_data_dir("us")

# ------------------------------------------------------------------ 样式常量
# 图表的配色和 K 线窗口常量都在 `gui/result_view.py` 里 —— 这一页只剩侧边栏和
# 状态灯，用到颜色的地方只有下面四个。
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_UP = "#d92b2b"      # 红涨
C_DOWN = "#0f9d58"    # 绿跌

# 周期。日线有**两个源**：Tushare（20 年、复权因子）和天勤（十年出头）。
# 分钟级只有天勤。两个日线的主力换月口径不同（天勤按自己的规则，Tushare 按
# 自己的），所以跨周期的绝对价位会有细微差异 —— 这是数据源差异，不是 bug。
#
# 键名特意分开：老配置里的 `period="1d"` 一直是 Tushare，不能因为加了天勤日线
# 就悄悄换源（历史长度和价格都会变），所以天勤日线用 `1d_tq`。
PERIOD_1D = "1d"            # Tushare 日线，显示为「日线Tushare」
PERIOD_1D_TQ = "1d_tq"      # 天勤日线，显示为「日线」
GUI_PERIODS = (
    [PERIOD_1D_TQ, PERIOD_1D]
    + [p for p in tqdata.PERIOD_ORDER if p != PERIOD_1D_TQ]
)
PERIOD_LABEL = dict(tqdata.PERIOD_LABEL, **{PERIOD_1D: "日线Tushare"})

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

# 仓位口径。研究页、组合页回放、实盘三处必须用同一个口径 —— 同一条策略在
# 三个地方算出三个收益曲线的话，用户只会认为程序算错了。
SIZE_MODES = [
    ("按资金分配", "capital",
     "拿本金的百分之多少去开仓（默认 30%），按每手保证金算能开几手：\n"
     "手数 = 本金 × 比例 ÷ (价格 × 合约乘数 × 保证金比例)。\n"
     "杠杆就是保证金比例给的：商品 / 股指约 10 倍，国债 30~50 倍。"),
    ("按止损金额", "risk",
     "每笔最多亏本金的百分之多少（默认 1%）：\n"
     "手数 = 本金 × 比例 ÷ (止损倍数 × ATR × 合约乘数)。\n"
     "波动大的品种自动少开，和价格高低无关。"),
    ("固定手数", "fixed", "每笔固定下 N 手，不看价格和资金。"),
    ("按本金全额", "unit",
     "名义敞口 = 全部本金（股票口径）。\n"
     "期货上等于不加杠杆，只用来和旧结果对照。"),
]

# 因子策略的参数：名字 -> (标签, 最小, 最大, 步长, 默认值)
FACTOR_PARAMS = {
    "window": ("标准化窗口", 120, 500, 20, 252),
    "horizon": ("预测期数", 1, 20, 1, 5),
    "entry_z": ("进场阈值", 0.3, 3.0, 0.1, 1.0),
    "exit_z": ("离场阈值", 0.0, 1.5, 0.1, 0.3),
    # 单位是**交易日**不是 bar 根数：5m 和日线的「1000」不该是同一段历史。
    # 换算在 factor_strategy._to_bars() 里做，按数据的 bar 密度来
    "train_window": ("训练窗口（交易日）", 400, 2500, 100, 1000),
    "step": ("重训间隔（交易日）", 60, 400, 20, 180),
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
/* 研究页的「启动实时」：默认是普通描边按钮，跑起来变绿 —— 和顶部那个
   「● 实时」同一套颜色，扫一眼就知道现在吃的是实时还是本地历史 */
QPushButton#LiveBtn {
    background: #ffffff; color: #334155; border: 1px solid #cbd5e1;
    padding: 4px 12px; font-weight: 600;
}
QPushButton#LiveBtn:hover { background: #f8fafc; border-color: #94a3b8; }
QPushButton#LiveBtn:checked {
    background: #dcfce7; color: #15803d; border: 1px solid #86efac;
}
QPushButton#LiveBtn:checked:hover { background: #bbf7d0; }
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

# 启动时选中的市场。国内期货是现在主要用的那个，美股留着但不再是默认
DEFAULT_MARKET = "cn"

# 市场元信息：目录、副标题、价格说明、回测成本假设。
# 成本三项来自 `ttrade.replay.MARKET_COST`：组合页的离线回放用同一份数字，
# 否则同一条策略在两个页面会算出不同的收益。
MARKETS = {
    "us": {
        "label": "美股",
        "dir": DATA_DIR,
        "sub": "美股 · 20年日线 · 后复权 · 东方财富",
        "price_note": "价格（后复权）",
        **replay.MARKET_COST["us"],
        "legend": (
            "▲ 紫色向上三角＝开多　▼ 紫色向下三角＝平多　"
            "浅蓝底色＝多头持仓　红色虚线＝这笔交易盈利　绿色虚线＝亏损　"
            "价格均为后复权（含分红再投资），保证收益率连续　"
            "信号在下一根开盘成交　双击 K 线出十字光标"
        ),
    },
    "cn": {
        "label": "国内期货",
        "dir": FUT_DIR,
        "sub": "国内期货 · 日线 20 年（Tushare）/ 5 分钟·1 小时 10 年（天勤）· 主力连续 · 前复权",
        "price_note": "价格（前复权·主力连续）",
        **replay.MARKET_COST["cn"],
        "legend": (
            "▲紫＝开多　▼紫＝平多　▼琥珀＝开空　▲琥珀＝平空　"
            "浅蓝底色＝多头持仓　浅红底色＝空头持仓　"
            "红色虚线＝这笔交易盈利　绿色虚线＝亏损　"
            "分钟级按天勤自己的主力切换口径复权，与日线的 Tushare 口径不同　"
            "前复权：最新一根＝真实盘面价　信号在下一根开盘成交　双击 K 线出十字光标"
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
                # 分钟级同样要挂衍生列（期限结构 / 期权 IV），否则展期、期权那
                # 十几个因子在 1h / 5m 上整列 NaN。日线分支在 load_futures_csv()
                # 里做过，这里一直漏着 —— 「下载了真实合约却算不出展期因子」就是
                # 这个原因。分钟 bar 比衍生表密得多，用 attach_intraday 按上一个
                # 交易日对齐，避免把当天收盘才有的价差灌进上午的 bar。
                #
                # 天勤日线走 attach：那根 K 线和期限结构是同一天同一时刻的，
                # 谁都不领先谁（和 Tushare 日线分支一个口径）。
                try:
                    from ttrade.auxdata import attach as attach_daily
                    from ttrade.auxdata import attach_intraday as attach_intra

                    if tqdata.is_daily(period):
                        df = attach_daily(df, sym, ex, AUX_DIR)
                    else:
                        df = attach_intra(df, sym, ex, AUX_DIR)
                except Exception:
                    pass
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
    # 回测算完了：把结论摊成纯文本推给 Agent 页右下角的「运行输出」。
    # 第二个参数是「这一条允许被下一条覆盖」—— 拖滑杆时 220ms 就重算一次，
    # 不许覆盖的话输出框一秒刷十几条，人根本看不清。
    backtest_done = Signal(str, bool)

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
        # 默认开在国内期货：这是现在主要用的市场，美股只是备着。MARKETS 的
        # 字典顺序是 us 在前，所以这里必须显式指定，不能靠 items() 的第一项
        self.market = DEFAULT_MARKET
        self.store = self.stores[self.market]
        self.cfg = self._new_cfg(MARKETS[self.market])
        self.df_raw: pd.DataFrame | None = None
        self.df_sig: pd.DataFrame | None = None
        self.res = None
        self.bh = None
        self.param_widgets: dict[str, QWidget] = {}
        self.param_scales: dict[str, int] = {}
        self.fparam_widgets: dict[str, QWidget] = {}
        self.fparam_scales: dict[str, int] = {}
        self._pending = False
        # 灌配置时会顺带触发好几次计算（切市场、换品种各算一遍），那些都是中间态。
        # 这段时间的输出先攒着，等最后一次算完再发 —— 否则 Agent 会拿着上一个品种
        # 的数字讨论这一笔，而且看着毫无破绽
        self._pub_guard = 0
        self._pub_queue: tuple[str, bool] | None = None

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

        # 实时与交易
        self._live_base: pd.DataFrame | None = None   # 订阅时的本地历史，实时段接在它后面
        self._live_state = None                        # 最近一次 LiveState
        self._live_key = ""
        self._live_period = ""
        # 当前喂这张图的是哪条会话：`trade`（账户页连的）/ `study`（研究页
        # 「启动实时」）/ 空（只画本地历史）。两条同时跑时只让其中一条画
        self._live_src = ""
        self._live_last = None                         # 上一根 bar 的时间，用来判断「新 bar」
        self._live_seq = 0                             # 每次起停自增，丢掉过期的推送
        self._live_patch_ts = 0.0                      # 上次刷未走完那根 bar 的时刻

        self._build_ui()
        self._populate_symbols()
        # 延后一拍：等界面完全建好、第一次分析排上队之后再灌配置，
        # 否则 apply_config 里的 _schedule(immediate=True) 会和首帧抢数据加载。
        QTimer.singleShot(0, self._restore_last_config)
        # **这里不连天勤**。研究页默认只看本地库里的历史；要实时得在「K线与信号」
        # 页点「启动实时」，换品种 / 换周期会自动断开 —— 回测调参不该顺手登录
        self.lbl_live_msg.setText("本地历史（要实时看 K 线，点上面的「启动实时」）")

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
        self.study_page = self._build_study_page()
        self.pages.addWidget(self.study_page)
        self.live = LiveController(self)
        # 研究页自己的行情会话。**不复用 `self.live`**：那条连接绑着账户，起停会
        # 把「账户透视和交易」页的持仓/委托一起断掉；而研究页要的是纯行情，不带
        # 账户也能连（见 `LiveHub.start` 的空账户分支）。
        self.study_live = LiveController(self)
        self.trade_page = TradePage(self.live)
        self.pages.addWidget(self.trade_page)
        self.portfolio_page = PortfolioPage(loader=self._load_bars)
        self.pages.addWidget(self.portfolio_page)
        self.agent_page = AgentPage()
        self.pages.addWidget(self.agent_page)
        # 市场单选归数据页自己管（和本页的品种类别解耦），这里只把两个市场的
        # 标签和目录递进去，省得数据页反过来 import 主窗口造成循环依赖
        self.data_page = DataPage(MARKETS)
        self.pages.addWidget(self.data_page)
        outer.addWidget(self.pages, 1)

        # 数据补完之后标的池可能变了（新上市品种、刚补的分钟线），主窗口要重读
        self.data_page.data_changed.connect(self._on_data_changed)

        self.agent_page.study_requested.connect(self._on_agent_study)
        self.agent_page.trade_requested.connect(self._on_agent_trade)
        self.agent_page.portfolio_requested.connect(self._on_agent_portfolio)
        # Agent 注册了新因子/新策略之后，对应下拉要重建，否则库里有了却选不到
        self.agent_page.factor_added.connect(self._on_factor_added)
        self.agent_page.strategy_added.connect(self._on_strategy_added)
        # 删掉同理：不重建的话下拉里还留着一条，选中就报「因子不存在」
        self.agent_page.factor_removed.connect(self._on_factor_removed)
        self.agent_page.strategy_removed.connect(self._on_strategy_removed)
        # 回测结论同步到 Agent 页的输出框。这条链是单向的：研究页算完就往那儿推，
        # 输出框不反过来驱动计算
        self.backtest_done.connect(self.agent_page.push_output)

        self.trade_page.connect_requested.connect(self._on_trade_connect)
        self.trade_page.disconnect_requested.connect(self._on_live_stop)
        self._wire_live()
        self._wire_study_live()

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
        self.btn_page_data = QPushButton("数据管理")
        self._page_btns = (
            self.btn_page_study, self.btn_page_trade,
            self.btn_page_port, self.btn_page_agent, self.btn_page_data,
        )
        for i, b in enumerate(self._page_btns):
            b.setObjectName("PageTab")
            b.setCheckable(True)
            b.setMinimumWidth(132)
            b.clicked.connect(lambda _c=False, k=i: self._switch_page(k))
            lay.addWidget(b)
        self.btn_page_study.setChecked(True)
        lay.addStretch(1)

        self.lbl_live = QLabel("本地历史")
        self.lbl_live.setObjectName("Hint")
        lay.addWidget(self.lbl_live)

        # 详细的实时状态（最新价 / 连接进度 / 失败原因）。这一堆 setText 打在
        # 一个从没创建过的属性上，每次连实时都抛 AttributeError —— 槽函数里的
        # 异常只打到 stderr，pythonw 下看不见，表现就是「连上了但什么提示都没有」。
        self.lbl_live_msg = QLabel("")
        self.lbl_live_msg.setObjectName("Hint")
        lay.addWidget(self.lbl_live_msg)
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
        elif idx == 4:
            self._sync_data_page()

    def _sync_data_page(self) -> None:
        """把当前市场告诉数据页：Tushare 按市场走，天勤只有期货。

        `_populate_symbols()` 在数据页创建**之前**就会调到这里（页面栈按顺序建），
        所以必须能容忍 `data_page` 还不存在。
        """
        if not hasattr(self, "data_page"):
            return
        info = MARKETS[self.market]
        self.data_page.set_market(self.market, info["label"], info["dir"])

    def _on_data_changed(self, market: str) -> None:
        """补完数据：标的池、分钟线覆盖都可能变了，重读一遍。"""
        self._reload_store(market)

    # ---------------------------------------------------------- 交易Agent

    def _on_agent_study(self, cfg) -> None:
        """Agent 要「用某策略回测某品种」：切到研究页并套上配置。

        内置模板没有品种（`cfg.symbol` 为空），这时候只换策略和参数，
        品种保持用户当前选的那个 —— 换个品种不该顺带把标的也换掉。

        灌配置的过程里，切市场、换品种、换策略各会触发一次重算，那些都是**中间
        态**（上一个品种、上一个策略算出来的）。这些输出先攒着，只把最后一次发
        出去 —— 不然 Agent 会拿着 AAPL 的双均线数字，一本正经地讨论螺纹。
        """
        self._switch_page(0)
        self._pub_guard += 1
        try:
            if cfg.symbol:
                self.apply_config(cfg)
            else:
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
        finally:
            self._pub_guard -= 1
        self._flush_publish()

    def _on_agent_portfolio(self, cfg) -> None:
        self._switch_page(2)
        self.portfolio_page.accept(cfg)

    def _load_bars(self, market: str, key: str, period: str):
        """给「策略交易和组合」取行情。走主窗口的 DataStore，共用同一份缓存。"""
        store = self.stores.get(market)
        if store is None:
            return None
        return store.get(key, period or PERIOD_1D)

    def _agent_config(self, p: dict, note: str = ""):
        """把 Agent 的动作块翻成一份配置。翻不出来返回 `(None, 原因)`。

        `"strategy"` 这个字段模型有三种可能写法，都得认 —— 它只看得见**名字**，
        截图里没有 key：

        1. `STRATEGIES` 的代号（`ma_cross`）—— 没有歧义，先认这个
        2. **策略库里的模板**（`a62cd55e1389` / `碳酸锂动量60-5m`）—— 用户自己存
           好的整份配置，品种、周期、参数都定了，要整套搬走。用户说「把碳酸锂
           那个策略加进组合」指的就是这个，它**不是**一条规则策略
        3. `STRATEGIES` 的显示名（`双均线交叉`、`双均线交叉 · 内置`）

        以前只拿 `key in STRATEGIES` 硬比，第 2、3 种写法一律回「平台没有这个
        策略」，模型再怎么试都没用 —— 它没有别的信息可用了。
        """
        name = str(p.get("strategy", "") or "").strip()
        if not name:
            return None, "动作块里没有 strategy 字段"
        try:
            vol = int(p.get("volume", 0) or 0)
        except (TypeError, ValueError):
            vol = 0

        key = name if name in STRATEGIES else ""
        lib = None if key else SStore.find_library(name)
        if not key and lib is None:
            key = resolve_key(name)
        if not key and lib is None:
            # 名字去重：同一个模板存过几版（「白银动量60-1h」现在有三条），
            # 报错里连着列三遍只会让人以为坏了
            libs = "、".join(dict.fromkeys(c.title() for c in SStore.load_library()))
            return None, (f"平台没有这个策略：{name}。策略库里现在有：{libs or '（空）'}；"
                          f"规则策略代号有：{available_text()}")

        if lib is not None:
            # 模板是一整份配置：品种、周期、参数全照搬，只让手数听动作块的。
            # 不让 payload 里的 symbol/period 覆盖 —— 那是模型自己从截图猜的，
            # 猜错一个周期就是另一个策略，静默换掉比报错更糟。
            cfg = lib.clone()
            cfg.name = lib.title()
            if vol > 0:
                cfg.sizing, cfg.volume = "fixed", vol
            cfg.note = note
            return cfg, ""

        return SStore.StrategyConfig(
            symbol=p.get("symbol", ""),
            exchange=p.get("exchange", ""),
            period=p.get("period", "1d"),
            rule_key=key,
            rule_params=dict(p.get("params") or {}),
            # Agent 明说了几手就按固定手数，没说就走「分配资金」口径
            sizing="fixed" if vol else "capital",
            volume=vol or 1,
            note=note,
        ), ""

    def _on_agent_trade(self, action) -> None:
        """Agent 给了交易动作。动作**已经在界面上确认过**才走到这里。"""
        p = action.payload
        if action.kind == "backtest":
            cfg, why = self._agent_config(p, note="Agent 回测")
            if cfg is None:
                self.agent_page.notify(why, bad=True)
                return
            # 先告诉 Agent 页「这次结果要喂回模型」。用户自己点回测不该触发一轮
            # 对话，只有模型问出来的结果才值得它接着往下说。放在解析**之后**：
            # 解析失败就直接返回了，标记挂着会让下一次用户自己点的回测莫名
            # 触发一轮对话
            self.agent_page.begin_backtest_feedback()
            # 结果落在「策略交易和组合」，**不**切到「策略研究」：那边是用户手上
            # 正在调的那套参数，模型问一句就把它冲掉说不过去。五个结果页
            # （K线 / 绩效 / 因子 / 交易记录 / 阶段分析）组合页都挂得上。
            self._switch_page(2)
            self._publish(self.portfolio_page.show_config(cfg, note="Agent 回测"), False)
            return

        if action.kind != "trade":
            return

        if p.get("mode") == "manual":
            self._agent_manual(p)
            return

        cfg, why = self._agent_config(p, note="short")
        if cfg is None:
            self.agent_page.notify(why, bad=True)
            return
        self._switch_page(2)
        msg = self.portfolio_page.accept(cfg)
        # 回执里把**实际用上的**品种/周期/手数写全：模型可能报的是「碳酸锂
        # 动量60-5m」而模板里定的是 1h，让用户一眼看出到底跑的是哪一套
        self.agent_page.notify(
            f"{msg}（{cfg.title()}：{cfg.key() or '未选品种'} {cfg.period}"
            f" · {cfg.strategy_label()} · {cfg.size_text()}）")

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

        # 五页结果由 `ResultView` 统一提供。「策略交易和组合」里双击一条策略
        # 打开的是同一个类 —— 一处实现两处挂载，不会出现「这边改好了那边还是
        # 老图」的情况。
        self.view = ResultView("single", self)
        self.tabs = QTabWidget()
        for title, w in self.view.pages():
            self.tabs.addTab(w, title)
        self.view.status.connect(self.lbl_status.setText)
        # 「启动实时」挂在 K 线与信号页的标题行上：它只影响这张图，放侧边栏会
        # 让人以为它跟回测参数有关
        self.view.add_chart_action(self._build_live_btn())
        root.addWidget(self.tabs, 1)
        return page

    def _build_live_btn(self) -> QPushButton:
        """研究页的实时开关。**手动**：换品种 / 换周期就断开，要再看得自己点。"""
        self.btn_live = QPushButton("启动实时")
        self.btn_live.setObjectName("LiveBtn")
        self.btn_live.setCheckable(True)
        self.btn_live.setToolTip(
            "连天勤，按当前品种和周期补齐本地历史，再把实时推送拼到 K 线上。\n"
            "手动开关：换品种或换周期会自动断开 —— 研究页是调参的地方，\n"
            "不该边拖滑杆边反复登录天勤。\n"
            "「日线Tushare」没有实时，要实时日线请把周期切到「日线」（天勤）。"
        )
        self.btn_live.toggled.connect(self._on_study_live_toggled)
        return self.btn_live

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
        # 信号还没连（在 _populate_symbols 里才 connect），这时改选中不会触发重算
        i = self.cb_market.findData(self.market)
        if i >= 0:
            self.cb_market.setCurrentIndex(i)
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
        self._fill_strategy_combo()
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
        self._fill_single_combo()

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
        self._fill_multi_table()
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

        cl.addWidget(self._section("回测区间"))
        self.cb_range = QComboBox()
        for text, years in [("近 1 年", 1), ("近 2 年", 2), ("近 3 年", 3),
                            ("近 5 年", 5), ("近 10 年", 10), ("全部历史", 0)]:
            self.cb_range.addItem(text, years)
        self.cb_range.setCurrentIndex(2)
        self.cb_range.setToolTip(
            "策略和回测都只算这一段（起点前多留 250 根让均线预热，图上不显示）。\n"
            "5 分钟十年是 26 万根，跑因子评估 / 机器学习时选短一点快很多。"
        )
        cl.addWidget(self.cb_range)

        # 仓位口径。和「策略交易和组合」页编辑框里的那一组是同一套字段，
        # 存进同一份 StrategyConfig —— 研究页调出来的手数，保存之后实盘和
        # 组合回放必须照着走，否则「研究页赚 30%、实盘亏 5%」会反复上演。
        cl.addWidget(self._section("仓位"))
        self.cb_size = QComboBox()
        for label, key, tip in SIZE_MODES:
            self.cb_size.addItem(label, key)
            self.cb_size.setItemData(self.cb_size.count() - 1, tip, Qt.ToolTipRole)
        cl.addWidget(self.cb_size)

        # 本金：这条策略的账户规模。资金曲线从它起步，下面两个百分比都按它算。
        # 放在口径选择器**上面**，因为它和选哪种口径无关 —— 换成固定手数时
        # 本金照样决定净值曲线的起点。
        self.sp_principal = QDoubleSpinBox()
        self.sp_principal.setRange(10_000, 1_000_000_000)
        self.sp_principal.setSingleStep(500_000)
        self.sp_principal.setDecimals(0)
        self.sp_principal.setPrefix("本金 ")
        self.sp_principal.setSuffix(" 元")
        self.sp_principal.setValue(1_000_000)
        self.sp_principal.setToolTip(
            "这条策略的账户规模，默认 100 万。\n"
            "资金曲线从它起步；「按资金分配」「按止损金额」都按它的百分比折算，"
            "改了本金手数跟着变。"
        )
        cl.addWidget(self.sp_principal)

        self.sp_capital = QDoubleSpinBox()
        self.sp_capital.setRange(0.1, 500.0)
        self.sp_capital.setSingleStep(5.0)
        self.sp_capital.setDecimals(1)
        self.sp_capital.setPrefix("资金 ")
        self.sp_capital.setSuffix(" %")
        self.sp_capital.setValue(30.0)
        self.sp_capital.setToolTip(
            "拿本金的百分之多少去开仓，默认 30%。\n"
            "手数 = 本金 × 比例 ÷ 每手保证金（价格 × 乘数 × 保证金比例）。"
        )
        cl.addWidget(self.sp_capital)

        self.sp_risk = QDoubleSpinBox()
        self.sp_risk.setRange(0.01, 100.0)
        self.sp_risk.setSingleStep(0.5)
        self.sp_risk.setDecimals(2)
        self.sp_risk.setPrefix("止损 ")
        self.sp_risk.setSuffix(" %")
        self.sp_risk.setValue(1.0)
        self.sp_risk.setToolTip(
            "每笔最多亏本金的百分之多少，默认 1%。\n"
            "手数 = 本金 × 比例 ÷（止损倍数 × ATR × 合约乘数）。"
        )
        cl.addWidget(self.sp_risk)

        self.sp_atr = QSpinBox()
        self.sp_atr.setRange(2, 200)
        self.sp_atr.setPrefix("ATR ")
        self.sp_atr.setSuffix(" 根")
        self.sp_atr.setValue(14)
        cl.addWidget(self.sp_atr)

        self.sp_stop = QDoubleSpinBox()
        self.sp_stop.setRange(0.5, 50.0)
        self.sp_stop.setSingleStep(0.5)
        self.sp_stop.setDecimals(1)
        self.sp_stop.setPrefix("× ")
        self.sp_stop.setSuffix(" 倍")
        self.sp_stop.setValue(5.0)
        self.sp_stop.setToolTip("止损距离 = 倍数 × ATR。手数按「止损金额 ÷ 每手止损金额」反推。")
        cl.addWidget(self.sp_stop)

        self.sp_vol = QSpinBox()
        self.sp_vol.setRange(1, 1000)
        self.sp_vol.setPrefix("固定 ")
        self.sp_vol.setSuffix(" 手")
        self.sp_vol.setValue(1)
        cl.addWidget(self.sp_vol)
        self._sync_size()
        self._apply_sizing(self.cfg)      # 控件建好了，把默认口径灌进配置

        self.chk_ma = QComboBox()
        cl.addWidget(self._section("K线叠加"))
        self.chk_ma.addItem("显示策略指标线", True)
        self.chk_ma.addItem("仅 K 线", False)
        cl.addWidget(self.chk_ma)

        cl.addWidget(self._section("交易方向"))
        self.chk_short = QCheckBox("允许做空（多空双向）")
        self.chk_short.setChecked(bool(MARKETS[self.market]["allow_short"]))
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
        self.btn_load_lib = QPushButton("加载策略")
        self.btn_load_lib.setObjectName("Ghost")
        self.btn_load_lib.setToolTip(
            "从策略库挑一条模板，把它的品种、周期、策略和参数灌回左边。\n"
            "只是灌界面，不保存 —— 想让它下次自动恢复还得点「保存配置」。"
        )
        self.btn_save_lib = QPushButton("存入策略库")
        self.btn_save_lib.setObjectName("Ghost")
        self.btn_save_lib.setToolTip(
            "作为模板存进策略库，之后在「策略交易和组合」里可以反复取用。"
        )
        crl.addWidget(self.btn_load_lib)
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

    # ---------------------------------------------------------- 数据填充
    def _populate_symbols(self) -> None:
        # 信号只连一次，避免切换市场后重复触发
        if not getattr(self, "_wired", False):
            self.cb_market.currentIndexChanged.connect(self._on_market_changed)
            self.cb_symbol.currentIndexChanged.connect(self._on_symbol_changed)
            self.cb_period.currentIndexChanged.connect(self._on_period_changed)
            self.cb_strategy.currentIndexChanged.connect(self._on_strategy_changed)
            self.cb_range.currentIndexChanged.connect(self._schedule)
            self.cb_size.currentIndexChanged.connect(self._on_size_changed)
            for w in (self.sp_principal, self.sp_capital, self.sp_risk, self.sp_atr,
                      self.sp_stop, self.sp_vol):
                w.valueChanged.connect(lambda *_: self._schedule())
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
            self.btn_load_lib.clicked.connect(self._on_load_library)
            self.btn_save_cfg.clicked.connect(self._on_save_config)
            self.btn_save_lib.clicked.connect(self._on_save_library)
            self._wired = True

        self._fill_symbols()
        # 启动时如果记忆的周期没数据（典型：新装包没有天勤分钟库），
        # 自动退到日线再出图，别让首屏空白
        self._sync_period_controls(fallback=True)
        self._on_strategy_changed()
        self._on_family_changed()
        self._sync_data_page()
        self._schedule(immediate=True)

    def _fill_single_combo(self) -> None:
        """重建单因子下拉。注册了新因子之后要重来一遍，否则选不到它。"""
        keep = self.cb_fsingle.currentData()
        self.cb_fsingle.blockSignals(True)
        self.cb_fsingle.clear()
        for cat in factors.CATEGORIES:
            keys = [k for k, s in factors.FACTORS.items() if s.category == cat]
            if not keys:
                continue
            for i, k in enumerate(keys):
                spec = factors.FACTORS[k]
                text = f"{cat} · {spec.label}" if i == 0 else f"　　{spec.label}"
                self.cb_fsingle.addItem(text, k)
        target = keep if self.cb_fsingle.findData(keep) >= 0 else factors.DEFAULT_SINGLE
        idx = self.cb_fsingle.findData(target)
        if idx >= 0:
            self.cb_fsingle.setCurrentIndex(idx)
        self.cb_fsingle.blockSignals(False)

    def _fill_strategy_combo(self) -> None:
        """重建规则策略下拉。注册了新策略之后要重来一遍，否则选不到它。

        内置的排在前面、用户注册的排后面 —— 库里越加越多之后，用户自己的东西
        不该把内置那几套挤到下面去。
        """
        keep = self.cb_strategy.currentData()
        builtin = [k for k in STRATEGIES if not STRATEGIES[k].code]
        mine = [k for k in STRATEGIES if STRATEGIES[k].code]
        self.cb_strategy.blockSignals(True)
        self.cb_strategy.clear()
        for key in builtin + mine:
            self.cb_strategy.addItem(STRATEGIES[key].label, key)
        idx = self.cb_strategy.findData(keep)
        self.cb_strategy.setCurrentIndex(idx if idx >= 0 else 0)
        self.cb_strategy.blockSignals(False)

    def _on_strategy_added(self, key: str) -> None:
        """Agent 注册了新策略：重建下拉并选中它，顺手把参数滑杆也换过去。"""
        self._fill_strategy_combo()
        idx = self.cb_strategy.findData(key)
        if idx >= 0:
            self.cb_strategy.setCurrentIndex(idx)
            f = self.cb_family.findData("rule")
            if f >= 0:
                self.cb_family.setCurrentIndex(f)

    def _on_factor_removed(self, _key: str) -> None:
        """删掉自定义因子：两个因子控件都要重建。

        不选中任何东西 —— 删完再跳到别的因子上，会让人以为「删的是那个」。
        当前选中的正好是被删的那个时，重建后自然落到第一项。
        """
        self._fill_single_combo()
        self._fill_multi_table()

    def _on_strategy_removed(self, _key: str) -> None:
        self._fill_strategy_combo()

    def _multi_state(self) -> dict:
        """当前多因子表格里每个因子的 (参与合成, 反转)。重建时用它保住勾选。"""
        out: dict[str, tuple[bool, bool]] = {}
        for r in range(self.tbl_fmulti.rowCount()):
            item = self.tbl_fmulti.item(r, 0)
            if item is None:
                continue
            k = item.data(Qt.UserRole)
            use = self.tbl_fmulti.item(r, 1)
            rev = self.tbl_fmulti.item(r, 2)
            out[k] = (
                bool(use is not None and use.checkState() == Qt.Checked),
                bool(rev is not None and rev.checkState() == Qt.Checked),
            )
        return out

    def _fill_multi_table(self) -> None:
        """重建多因子表格。保留用户已勾的「用 / 反」—— 加一个因子不该把之前
        的勾选全清掉。"""
        picked = self._multi_state() if self.tbl_fmulti.rowCount() else {}
        self.tbl_fmulti.setRowCount(0)
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
                use_on, rev_on = picked.get(k, (k in factors.DEFAULT_MULTI, False))
                for col, tip, default in (
                    (1, "勾上表示参与合成", use_on),
                    (2, "勾上表示这个因子反向使用（取负号）", rev_on),
                ):
                    cell = QTableWidgetItem("")
                    cell.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                    cell.setCheckState(Qt.Checked if default else Qt.Unchecked)
                    cell.setTextAlignment(Qt.AlignCenter)
                    cell.setToolTip(tip)
                    self.tbl_fmulti.setItem(r, col, cell)

    def _on_factor_added(self, key: str) -> None:
        """Agent 注册了新因子：两个因子控件都得重建，然后选中它。"""
        self._fill_single_combo()
        self._fill_multi_table()
        idx = self.cb_fsingle.findData(key)
        if idx >= 0:
            self.cb_fsingle.setCurrentIndex(idx)
            self.cb_family.setCurrentIndex(self.cb_family.findData("factor"))
            self.cb_fmode.setCurrentIndex(self.cb_fmode.findData("single"))
        # 不再往对话流里补一句「已加入因子库」：`_register_factor` 刚说过一遍，
        # 同一个动作回两行只会让人以为注册了两次。

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

    def _new_cfg(self, info: dict) -> BacktestConfig:
        """按当前界面重建回测配置。

        `self.cfg` 在切市场、切方向时都会整体重建 —— 仓位口径必须一起带过去，
        否则用户调好的「50 万」会在勾一下「允许做空」之后悄悄变回 1 手。
        """
        cfg = BacktestConfig(
            fee_rate=info["fee"], slippage=info["slippage"],
            long_only=not (info["allow_short"] and self._short_enabled()),
        )
        self._apply_sizing(cfg)
        return cfg

    def _apply_sizing(self, cfg: BacktestConfig) -> None:
        """把「仓位」那一组控件灌进配置。控件还没建好时什么都不做。"""
        if not hasattr(self, "cb_size"):
            return
        cfg.sizing = self.cb_size.currentData() or "capital"
        # 界面收的是「本金的百分之多少」，配置里存的是折算后的绝对金额 ——
        # 回测和实盘引擎只认绝对金额（手数 = 金额 ÷ 每手保证金），两边都改会让
        # 「改了本金手数没变」这种 bug 到处冒
        principal = float(self.sp_principal.value())
        cfg.capital = principal * float(self.sp_capital.value()) / 100.0
        cfg.risk_amount = principal * float(self.sp_risk.value()) / 100.0
        cfg.stop_atr_n = int(self.sp_atr.value())
        cfg.stop_mult = float(self.sp_stop.value())
        cfg.volume = int(self.sp_vol.value())
        if principal > 0:
            # 资金曲线从**本金**起步，不是从「分给这条策略的那笔钱」起步。
            # 只改手数不改本金的话，界面上写「100 万本金」，图上的净值还是从
            # 10 万开始，收益百分比看着对、金额全对不上；按止损金额定仓时更糟
            # —— 1 万止损压在 10 万本金上就是每笔赌 10%
            cfg.initial_cash = principal

    def _sync_size(self) -> None:
        """只显示当前口径用得到的输入框。

        本金**一直显示**：它决定净值曲线的起点，四种口径都用得上，藏起来只会
        让人以为改了没用。
        """
        mode = self.cb_size.currentData()
        for w, on in ((self.sp_principal, True),
                      (self.sp_capital, mode == "capital"),
                      (self.sp_risk, mode == "risk"),
                      (self.sp_atr, mode == "risk"),
                      (self.sp_stop, mode == "risk"),
                      (self.sp_vol, mode == "fixed")):
            w.setVisible(on)

    def _on_size_changed(self, *_a) -> None:
        self._sync_size()
        self._apply_sizing(self.cfg)
        self._schedule()

    def _on_short_toggled(self, _checked: bool) -> None:
        self.cfg = self._new_cfg(MARKETS[self.market])
        self._schedule(immediate=True)

    def _on_market_changed(self) -> None:
        mk = self.cb_market.currentData()
        if mk == self.market:
            return
        self.market = mk
        self.store = self.stores[mk]
        info = MARKETS[mk]
        self._stop_study_live()
        if hasattr(self, "chk_short"):
            self.chk_short.blockSignals(True)
            self.chk_short.setChecked(bool(info["allow_short"]))
            self.chk_short.blockSignals(False)
        self.cfg = self._new_cfg(info)
        self.lbl_sub.setText(info["sub"])
        self._sync_data_page()
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

    def _sync_period_controls(self, fallback: bool = False) -> None:
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
        self._sync_period_hint(fallback)

    def _sync_period_hint(self, fallback: bool = False) -> None:
        """在当前下拉项上标出「已下载 / 未下载」，避免选到空周期一脸问号。

        `fallback=True`（启动、换市场、换品种）时，当前周期一根都没有就自动退到
        Tushare 日线。包内自带 94 个品种的日线，新装用户没有天勤分钟库，首屏
        不该是「一张空白图 + 先跑 download_tq.py」。**用户手动选周期时不退**
        （`fallback=False`），否则他刚点开 5m 就被弹回日线，看着像下拉框坏了。
        """
        if self.market == "us" or not hasattr(self, "cb_period"):
            return
        key = self.cb_symbol.currentData() if hasattr(self, "cb_symbol") else None
        cur = self._period()
        switched = ""
        if fallback and key and cur != PERIOD_1D and not self.store.has_period(key, cur):
            # 用 get() 而不是猜文件名：日线走的是另一条读取路径（复权 + 换月拼接），
            # 只有它真的读得出来才值得切
            if self.store.get(key, PERIOD_1D) is not None:
                i = self.cb_period.findData(PERIOD_1D)
                if i >= 0:
                    self.cb_period.blockSignals(True)
                    self.cb_period.setCurrentIndex(i)
                    self.cb_period.blockSignals(False)
                    switched = PERIOD_LABEL.get(cur, cur)
                    cur = PERIOD_1D
        miss = []
        for i in range(self.cb_period.count()):
            p = self.cb_period.itemData(i)
            base = PERIOD_LABEL.get(p, p)
            ok = True if (key is None or p == PERIOD_1D) else self.store.has_period(key, p)
            self.cb_period.setItemText(i, base if ok else f"{base}（未下载）")
            if not ok:
                miss.append(p)
        if switched:
            self.lbl_period.setText(f"{switched} 数据还没下载，已切到日线")
        elif cur != PERIOD_1D and not (key and self.store.has_period(key, cur)):
            src = tqdata.source_period(cur)
            self.lbl_period.setText(
                f"{PERIOD_LABEL.get(cur, cur)} 数据还没下载：\n"
                f"python download_tq.py --main {src} --symbols ..."
            )
        elif miss:
            self.lbl_period.setText(f"天勤数据已覆盖 {len(GUI_PERIODS) - len(miss)} 个周期")
        else:
            self.lbl_period.setText("")

    def _on_period_changed(self) -> None:
        self._stop_study_live()
        self._sync_period_hint()
        self._schedule(immediate=True)

    def _on_symbol_changed(self) -> None:
        self._stop_study_live()
        # 每个品种下载到的周期不一样，换品种必须重标一次下拉项；
        # 新品种没有当前周期就退到日线（同启动，别让用户面对空图）
        self._sync_period_hint(fallback=True)
        self._schedule()

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
        for pname, default in spec.params.items():
            lo, hi, step = spec.param_meta.get(pname, (1, 100, 1))
            self._add_param_slider(
                self.param_box, self.param_widgets, self.param_scales,
                pname, PARAM_LABELS.get(pname, pname), lo, hi, step, default,
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
            sizing=self.cb_size.currentData() or "capital",
            principal=float(self.sp_principal.value()),
            capital_pct=float(self.sp_capital.value()),
            risk_pct=float(self.sp_risk.value()),
            stop_atr_n=int(self.sp_atr.value()),
            stop_mult=float(self.sp_stop.value()),
            volume=int(self.sp_vol.value()),
            note="short" if self.chk_short.isChecked() else "",
            range_years=float(self.cb_range.currentData() or 0),
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
        # 恢复出来的周期可能一根数据都没有：换台机器没下分钟库、或者这个品种
        # 只下了别的周期。这里也要退一次日线 —— `_populate_symbols` 里那次
        # 降级发生在恢复配置**之前**，会被这里 setCurrentIndex 覆盖回去。
        # 自检在没打包 data_tq 的环境里就是卡在这条上（首屏空白 → 首次回测
        # 没结果 → 自检红）。
        self._sync_period_controls(fallback=True)

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
        i = self.cb_size.findData(getattr(cfg, "sizing", "capital") or "capital")
        if i >= 0:
            self.cb_size.setCurrentIndex(i)
        self.sp_principal.setValue(float(getattr(cfg, "principal", 0) or 1_000_000))
        self.sp_capital.setValue(float(getattr(cfg, "capital_pct", 0) or 30.0))
        self.sp_risk.setValue(float(getattr(cfg, "risk_pct", 0) or 1.0))
        self.sp_atr.setValue(int(getattr(cfg, "stop_atr_n", 0) or 14))
        self.sp_stop.setValue(float(getattr(cfg, "stop_mult", 0) or 5.0))
        self.sp_vol.setValue(int(getattr(cfg, "volume", 0) or 1))
        self._sync_size()
        self.ed_cfg_name.setText(cfg.name)
        r = self.cb_range.findData(float(getattr(cfg, "range_years", 0) or 0))
        if r >= 0:
            self.cb_range.setCurrentIndex(r)      # 触发一次重算，窗口跟着换
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

    def _on_load_library(self) -> None:
        """从策略库挑一条，把它的品种 / 周期 / 策略 / 参数灌回界面。

        只灌界面、不写 `last_config`：用户多半是「拿这条当起点改一改」，
        直接落盘的话下次启动会被它顶掉，而他原来那套配置就找不回来了。
        """
        lib = SStore.load_library()
        if not lib:
            self.lbl_status.setText("策略库是空的。先调好一套参数，点「存入策略库」")
            return
        dlg = LibraryDialog(
            lib, self, title="加载策略",
            hint="选一条模板，把它的品种、周期、策略和参数灌回左边。"
                 "加载只是填界面，不会自动保存，也不会改动策略库里的原件。",
        )
        if dlg.exec() != QDialog.Accepted:
            return
        sid = dlg.sid()
        cfg = next((c for c in lib if c.sid == sid), None)
        if cfg is None:
            return
        try:
            self.apply_config(cfg)
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"加载失败：{exc}")
            return
        self.lbl_status.setText(f"已加载「{cfg.title()}」。改完想留住就点「保存配置」")

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
            self._booting = True
            self.apply_config(cfg)
            self.lbl_status.setText(f"已恢复上次配置：{cfg.title()}")
        except Exception as exc:  # noqa: BLE001
            self.lbl_status.setText(f"恢复上次配置失败：{exc}")
        finally:
            self._booting = False

    # ---------------------------------------------------------- 调度
    def _schedule(self, *_, immediate: bool = False) -> None:
        if not self.store.symbols:
            return
        if immediate:
            # 立刻算这一下已经把「等会儿要算」的那次抵掉了。不清掉 `_pending`，
            # 220ms 后那个定时器还会再跑一遍 —— 恢复上次配置时启动就白白多算
            # 一次（一次 1 秒起，图上还要重画一遍），看着像卡住。
            self._pending = False
            if getattr(self, "_booting", False):
                # 启动阶段恢复配置：先让窗口画出来再算。同步算的话，窗口要等
                # 回测跑完才第一次绘制 —— 用户看到的就是「打开就卡住」。
                self._schedule()
                return
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
        if not self._pending:
            return                  # 已经被一次 immediate 顶掉了
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
            elif period == PERIOD_1D_TQ:
                self._clear_chart(
                    f"{skey} 没有天勤日线数据\n"
                    "先跑：python download_tq.py --main 1d --all"
                )
            else:
                self._clear_chart(
                    f"{skey} 没有 {PERIOD_LABEL.get(period, period)} 数据\n"
                    f"先跑：python download_tq.py --main {period} --all"
                )
            return

        # 「回测区间」不只是显示窗口：策略和回测都只算这一段。5 分钟十年
        # 26 万根，因子评估 / 机器学习跑一遍要几十秒，而用户通常只想看最近两年。
        # 前面多留的预热段让均线一进入窗口就是暖的，图上会按区间裁掉。
        df = replay.window(df, self.cb_range.currentData() or 0)

        # 年化口径必须跟着周期走：5 分钟一年约 1.2 万根，按 252 算会把波动
        # 低估一个数量级、夏普虚高一个数量级。
        self.cfg.bars_per_year = bars_per_year(df.index)
        # 仓位口径每次重算都从界面读一遍：控件改了不一定重建 cfg，但这里一定跑
        self._apply_sizing(self.cfg)
        item = self._current_item()
        if self.market == "cn":
            self.cfg.multiplier = specs.multiplier(item.get("symbol", ""),
                                                   item.get("exchange", ""))
            self.cfg.margin_rate = specs.margin_rate(item.get("symbol", ""))
        else:
            self.cfg.multiplier, self.cfg.margin_rate = 1.0, 1.0

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
                self._publish(f"策略计算失败：{exc}", True)
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
            self._publish(f"因子计算失败：{exc}", True)
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
            f"（{len(keys)} 因子，{cfg.train_window} 交易日训练窗口）…"
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
        self._publish(f"机器学习训练失败：{msg}", True)

    def _view_ctx(self, label: str) -> ViewCtx:
        """把左边那排控件当前的选择收成画图需要的上下文。

        `ResultView` 不知道自己被谁挂着，所以要什么就一次给全 —— 它不该反过来
        读 `self.cb_symbol`，那样就没法在组合页复用了。
        """
        item = self._current_item()
        return ViewCtx(
            symbol=item.get("symbol") or "",
            name=item.get("name") or "",
            exchange=item.get("exchange") or "",
            period=self._period(),
            market=self.market,
            label=label,
            legend=MARKETS[self.market]["legend"],
            price_note=MARKETS[self.market]["price_note"],
            range_years=self.cb_range.currentData() or None,
            rule_key=self.cb_strategy.currentData() or "",
            overlay=bool(self.chk_ma.currentData()),
            long_only=self.cfg.long_only,
            bars_per_year=self.cfg.bars_per_year,
            factor_cfg=self._factor_cfg(),
            filter_active=self._filter_active(),
            selected_keys=self._selected_factor_keys(),
            outcome=self.factor_outcome,
        )

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
        self.view.clear(msg)
        # 算不出来也得让 Agent 那边看见原因。「没有结果」和「结果是 0」在界面上
        # 长得一模一样，模型分不清就只能反过来问用户
        self._publish(f"没有结果：{msg}", True)

    def _finish(self, rule_signal_df: pd.DataFrame, label: str) -> None:
        """规则策略（不含因子）的收尾：回测 + 重绘。"""
        self.factor_outcome = None
        self.df_sig = rule_signal_df
        self._strategy_label = label
        self.res = run_backtest(rule_signal_df, self.cfg, label=label)
        self.bh = buy_and_hold(self.df_raw, self.cfg)
        self.view.show_result(self.df_sig, self.df_raw, self.res, self.bh,
                              self._view_ctx(label))
        self._publish_result()

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
        self.view.show_result(self.df_sig, self.df_raw, self.res, self.bh,
                              self._view_ctx(label))

        # 展期/期权因子需要额外的合约与期权数据。美股没有，部分期货品种也可能
        # 没下过衍生数据 —— 这种情况分数全为 NaN，回测结果是「一笔交易都没有」。
        # 不解释一句的话，看起来就像策略本身没信号，实际是数据没到。
        score = getattr(out.pack, "score", None)
        if score is not None and int(score.notna().sum()) == 0:
            why = "需要真实合约 / 期权数据" if self.market == "cn" else "美股没有合约与期权数据"
            self.lbl_status.setText(
                f"所选因子在当前标的没有任何可用值（{why}），因此不会产生信号"
            )
        self._publish_result()

    # ---------------------------------------------------------- 实时行情
    def _wire_live(self) -> None:
        self.live.started.connect(self._on_live_started)
        self.live.stopped.connect(self._on_live_stopped)
        self.live.bars.connect(self._on_live_bars)
        self.live.state.connect(self._on_live_state)
        self.live.failed.connect(self._on_live_failed)
        self.live.log.connect(self._on_live_log)

    def _wire_study_live(self) -> None:
        """研究页那条行情会话。信号各走各的槽，但画的是同一张图。"""
        self.study_live.started.connect(self._on_study_live_started)
        self.study_live.stopped.connect(self._on_study_live_stopped)
        self.study_live.bars.connect(self._on_study_live_bars)
        self.study_live.state.connect(self._on_study_live_state)
        self.study_live.failed.connect(self._on_study_live_failed)
        self.study_live.log.connect(self._on_study_live_log)

    def _current_pair(self) -> tuple[str, str] | None:
        item = self._current_item()
        if not item:
            return None
        key = item["key"]
        sym, _, ex = key.rpartition("_")
        if not sym or not ex:
            return None
        return sym, ex

    def _on_trade_connect(self, specs) -> None:
        """从「账户透视和交易」页点「连接」：这是**唯一**会去连天勤的入口。

        沿用当前选中的品种与周期，只为了有盘口 / 持仓 / 委托可看。
        `specs` 是账户列表里勾选的那些 —— 一个也能连，多个走天勤的多账户模式
        （一条连接、一次登录）。

        **不补本地历史**。这个动作的语义是「登录账户」，用户要看的是持仓/资金/
        委托，而 `hub.start()` 之后这些东西就有了。补历史一个品种十年 5m 要几十
        秒到几分钟，撞上天勤运维窗口（19:00-19:30）还会一直等到 19:31 —— 为了
        画 K 线把「登录」卡住几分钟是不划算的。

        研究页不再自动接实时（早先换品种 / 换周期各接一次，回测的时候动不动就
        登录一遍天勤）。这里连上之后，如果品种和周期跟研究页当前选的一致，那页
        的 K 线照样会跟着实时刷新 —— 走的是同一条 `_on_live_bars`，不用再单开
        一个连接。
        """
        if self.market != "cn":
            self.trade_page.on_failed("交易只接了国内期货")
            return
        if not specs:
            self.trade_page.on_failed("先在账户列表里勾选要连接的账户")
            return
        pair = self._current_pair()
        period = "5m" if tqdata.is_daily(self._period()) else self._period()
        if pair is None:
            self.trade_page.on_failed("先在「策略研究」页选一个品种")
            return
        # 两条会话画的是同一张图，同品种同周期会互相把 `_live_last` 顶掉。账户
        # 这条连上之后本来就会喂研究页的 K 线（见上面的说明），研究页那条先收掉
        self._stop_study_live()
        self.live.start(specs, pair[0], pair[1], period, need_history=False)
        self.lbl_live_msg.setText("正在连接…")

    def _on_live_stop(self) -> None:
        self.live.stop(wait=False)
        self.lbl_live_msg.setText("实时已停止")

    # ---- 研究页的实时（手动） ----

    def _on_study_live_toggled(self, on: bool) -> None:
        if not on:
            self.study_live.stop(wait=False)
            self.lbl_live_msg.setText("实时已停止（只显示本地历史）")
            return
        err = self._start_study_live()
        if err:
            self._sync_live_btn(False)
            self.lbl_live_msg.setText(err)
            return
        self._sync_live_btn(True)

    def _start_study_live(self) -> str:
        """起研究页的行情会话。返回空串表示已经发出去，否则是要显示的原因。"""
        if self.market != "cn":
            return "实时行情只接了国内期货"
        pair = self._current_pair()
        if pair is None:
            return "先选一个品种"
        period = self._period()
        if period not in tqdata.PERIODS:
            # `1d` 是 Tushare 的日线 CSV，没有实时源；天勤日线的键是 `1d_tq`
            return "「日线Tushare」没有实时行情，要实时日线请把周期切到「日线」（天勤）"
        # 不清 `_live_src`：连上之前让原来那条会话继续画（多半就是同一个品种），
        # 连不上也不会把图冻住。接管发生在 `started` 里
        self.lbl_live_msg.setText(
            f"正在连接 {pair[0]} {PERIOD_LABEL.get(period, period)}，先补本地历史…"
        )
        # 补历史放在 worker 线程里（十年 5m 要几十秒），界面不冻；`ready` 之后
        # 才开始收 bar，所以这里不用等
        self.study_live.start(None, pair[0], pair[1], period, need_history=True)
        return ""

    def _sync_live_btn(self, on: bool) -> None:
        """按钮状态跟着会话走。必须屏蔽信号，否则 setChecked 又触发一轮 toggled。"""
        if not hasattr(self, "btn_live"):
            return
        self.btn_live.blockSignals(True)
        self.btn_live.setChecked(on)
        self.btn_live.setText("■ 实时中" if on else "启动实时")
        self.btn_live.blockSignals(False)

    def _stop_study_live(self) -> None:
        """换品种 / 换周期 / 换市场：实时一律断开，按钮回到未点状态。

        研究页是调参的地方。每换一次就重连天勤、重补一遍历史，边拖滑杆边登录
        既慢又占连接 —— 要实时得自己再点一次。
        """
        if hasattr(self, "study_live") and self.study_live.running:
            self.study_live.stop(wait=False)
        if self._live_src == "study":
            self._live_src = ""
            self._live_last = None
            self.lbl_live.setText("本地历史")
            self.lbl_live.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self._sync_live_btn(False)

    def _on_live_started(self, key: str) -> None:
        self._live_started_from(key, "trade")

    def _on_study_live_started(self, key: str) -> None:
        self._live_started_from(key, "study")

    def _live_started_from(self, key: str, src: str) -> None:
        sym, _, ex = key.rpartition("_")
        self._live_key = key
        # 图归最后启动的那条会话。两条同时跑时（研究页点了实时、交易页也连着），
        # 让各自的推送都往上画会互相把 `_live_last` 顶掉，表现是「一根 bar 反复重算」。
        self._live_src = src
        # 用会话自己的周期，不是研究页当前选的那个 —— 交易页可能按 5m 连，
        # 而研究页停在日线上，拿错了会去拼「5 分钟 bar + 日线历史」这种垃圾
        ctl = self.study_live if src == "study" else self.live
        self._live_period = ctl.period or self._period()
        self._live_seq += 1
        self._live_last = None
        # 工作线程刚把缺口补进库，缓存里的旧数据要作废，否则实时段会接在旧尾巴上
        self.store.invalidate(key, self._live_period)
        self._live_base = self.store.get(key, self._live_period)
        # 从交易页连的账户不补历史（见 `_on_trade_connect`），这时本地一根都没有。
        # 写「0 根历史」看着像出错，直接说清楚是没补。
        n_base = 0 if self._live_base is None else len(self._live_base)
        tail = f"{n_base:,} 根历史" if n_base else "未补本地历史"
        extra = ""
        if self._live_period != self._period():
            # 交易页按 5m 连的，这边在看日线 —— 图不会跟着动，别让人以为卡住了
            extra = f"；当前图是 {PERIOD_LABEL.get(self._period(), '')}，不跟随"
        self.lbl_live_msg.setText(
            f"实时中 · {sym} {PERIOD_LABEL.get(self._live_period, '')}（{tail}）{extra}"
        )
        self.lbl_live.setText("● 实时")
        self.lbl_live.setStyleSheet(f"color: {C_UP}; font-size: 11px; font-weight: 600;")

    def _on_live_stopped(self) -> None:
        if self.live.running:
            return      # 换品种时旧线程收尾，新的已经在跑了，别把标签刷回去
        if self._live_src == "study":
            return      # 图还在研究页那条会话上，别把它的状态刷掉
        self._live_src = ""
        self.lbl_live.setText("本地历史")
        self.lbl_live.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self.lbl_live_msg.setText("实时已停止（只显示本地历史；连接在「账户透视和交易」页）")

    def _on_study_live_stopped(self) -> None:
        if self.study_live.running:
            return
        self._sync_live_btn(False)
        if self._live_src != "study":
            return
        self._live_src = ""
        self._live_last = None
        self.lbl_live.setText("本地历史")
        self.lbl_live.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self.lbl_live_msg.setText("实时已停止（只显示本地历史）")

    def _on_live_log(self, msg: str) -> None:
        self.lbl_live_msg.setText(msg)
        self.trade_page.on_log(msg)

    def _on_live_failed(self, msg: str) -> None:
        self.lbl_live_msg.setText(msg)
        self.trade_page.on_failed(msg)

    def _on_live_state(self, st) -> None:
        # 只有带账户的那条连接才更新 `_live_state`：Agent 下单要读它的持仓和
        # 报单额度，行情会话覆盖上去就成了「有行情、没账户」的空状态
        self._live_state = st
        self._show_quote(st)

    def _on_study_live_state(self, st) -> None:
        self._show_quote(st)

    def _show_quote(self, st) -> None:
        q = st.quote or {}
        # 不要求本地历史存在：没补过历史时实时段照样在跑，报价更该看得见
        if q:
            und = q.get("underlying") or ""
            self.lbl_live_msg.setText(
                f"{q.get('name', '')} {und} {q.get('last', 0):.2f}"
                f" · 买 {q.get('bid', 0):.2f} 卖 {q.get('ask', 0):.2f}"
            )

    def _on_study_live_log(self, msg: str) -> None:
        self.lbl_live_msg.setText(msg)

    def _on_study_live_failed(self, msg: str) -> None:
        self.lbl_live_msg.setText(msg)
        self._sync_live_btn(False)

    def _on_live_bars(self, live_df) -> None:
        self._apply_live_bars(live_df, "trade")

    def _on_study_live_bars(self, live_df) -> None:
        self._apply_live_bars(live_df, "study")

    def _apply_live_bars(self, live_df, src: str) -> None:
        """实时 K 线到了：新 bar 就整体重算，同一根内只刷价格。"""
        if src != self._live_src:
            return          # 另一条会话在画这张图，这条的推送丢掉
        if self._closing or live_df is None or live_df.empty:
            return
        pair = self._current_pair()
        if pair is None or self._live_key != f"{pair[0]}_{pair[1]}":
            return          # 推送还是上一个品种的，丢掉
        if self._live_period and self._live_period != self._period():
            return          # 品种对上了但周期不一样（交易页按 5m 连的，研究页在看日线）
        # 人不在研究页就不画。一次重绘是 matplotlib 整图重建，实测 8000 根 1h
        # 要 100~230ms；实时每秒最多推 2.5 帧，主线程三成时间耗在画一张看不见的
        # 图上 —— 表现就是「在账户页点下单，按钮要顿一下才反应」。
        # `_live_last` 故意不更新：切回研究页的下一帧会判成「新 bar」，走一次
        # 完整重算把不在的这一段补上。
        if self.pages.currentWidget() is not self.study_page:
            return
        # 没走完的那根 bar，收盘价每来一个 tick 就变一次。天勤一秒推十几笔，
        # 每笔都 `merge_live` + 重画整张 K 线，主线程直接被占满 —— 界面看着
        # 就是「一打开就卡住」。同一根内限到 2.5 次/秒，肉眼看不出差别。
        ts_new = live_df.index[-1]
        now = time.monotonic()
        if (self._live_last is not None and ts_new == self._live_last
                and now - self._live_patch_ts < 0.4):
            return
        self._live_patch_ts = now
        # 本地一根都没有（这个周期还没补过历史，或者补历史失败了）时也要画：
        # 实时段自己就是全部行情，3000 根 1h 足够算指标。早先这里在
        # `_live_base is None` 上直接 return，表现是「标签写着实时中、图一动不动」
        merged = tqlive.merge_live(self._live_base, live_df)
        if merged.empty:
            return
        # 拿**实时段**的最后一根判断「是不是新 bar」。合并后的最后一根可能是库里
        # 更晚的那根（下载写进了半根活 bar），拿它比会一直判成「同一根」，
        # 实时 bar 永远进不了图
        new_bar = self._live_last is not None and ts_new != self._live_last
        first = self._live_last is None
        self._live_last = ts_new
        if first or new_bar:
            # 新 bar 走完整重算：信号只认走完的 bar，避免一根没走完的 K 线反复开平
            self.run_analysis(df_override=merged)
        else:
            self._patch_live_bar(merged)

    def _patch_live_bar(self, merged: pd.DataFrame) -> None:
        """把还没走完的那根 bar 的最新价刷到图上，不重算策略。

        视图里的 `df` 就是 `self.df_sig` 的同一个对象，所以这里只需要改一次；
        滚到历史位置时窗口最后一根是历史 bar，视图会自己拒掉（改它等于篡改历史）。
        """
        self.view.patch_last_bar(merged)

    # 下单不在这里：「启动交易」那套开关已经下线，实盘下单统一走
    # 「账户透视和交易」页的 `strategy_runner`（按配置、按手数、拆平今平昨）。
    # 研究页只负责看行情和调参数。

    # ---------------------------------------------------------- 结果文本

    def _result_text(self) -> str:
        """把这次回测的结论摊成一段纯文本。

        两个用处：Agent 页右下角的「运行输出」，以及模型自己发起回测时喂回去的
        那段结果。所以写的是**数字**，不是「回测完成」这种话 —— 模型要的就是数字，
        少一行它就要反过来问用户，而用户手上只有这张图。

        正文由 `ttrade.replay.summary` 生成：「策略交易和组合」页双击回放出来的
        结果也要喂回模型，两处必须一字不差，否则模型看到两种格式会当成两回事。
        """
        if self.res is None or self.bh is None or self.df_raw is None:
            return ""
        item = self._current_item()
        sym = item.get("symbol") or item.get("key") or ""
        ex = item.get("exchange") or ""
        head = f"{sym}_{ex}" if ex else sym
        period = self._period()
        rp = replay.Replay(
            df=self.df_sig, raw=self.df_raw, res=self.res, bh=self.bh,
            label=self._strategy_label or "策略", period=period,
            long_only=self.cfg.long_only, bars_per_year=self.cfg.bars_per_year,
        )
        return replay.summary(rp, head, PERIOD_LABEL.get(period, period))

    def _publish_result(self) -> None:
        """算完了就推一份到 Agent 页。算不出来时由 `_clear_chart` 推错误。"""
        text = self._result_text()
        if text:
            self._publish(text, False)

    def _publish(self, text: str, coalesce: bool) -> None:
        """往 Agent 页推一条输出。`_pub_guard` 期间只留最后一条。"""
        if self._pub_guard:
            self._pub_queue = (text, coalesce)
            return
        self.backtest_done.emit(text, coalesce)

    def _flush_publish(self) -> None:
        item, self._pub_queue = self._pub_queue, None
        if item is not None:
            self.backtest_done.emit(*item)

    # ---------------------------------------------------------- 生命周期
    def closeEvent(self, event) -> None:
        """关窗前先让更新线程收尾，避免线程还在写 CSV 就被强杀。"""
        self._closing = True
        self.data_page.shutdown()
        self._cancel_factor_worker()
        # 组合页每条账户连接一个 QThread，窗口关掉时线程还在跑 wait_update
        # 会让解释器退出时崩在 TqApi 上。这里等它们真的停下来再放行。
        self.portfolio_page.shutdown()
        self.agent_page.shutdown()
        self.live.stop(wait=True)
        # 研究页那条行情连接同理：线程还卡在 wait_update 里，解释器退出会崩在 TqApi
        self.study_live.stop(wait=True)
        super().closeEvent(event)


_INSTANCE_NAME = "TrendDesk-single-instance-7f3a"


def _claim_instance(app) -> bool:
    """单实例：已经有窗口在跑就把那个叫到前面，自己退出。

    两个进程连同一个天勤账户时，后连上的那个**拿不到资金和持仓** —— 服务端
    只把账户数据喂给其中一个，界面上就是「已连接，但六个资金卡片全是 —、
    三张表全空」，看着跟卡死一样。更糟的是前一个进程如果还停在组合页的调仓
    循环里，两个进程会互相把仓位改来改去（一个平、一个开，来回刷）。

    用 `QLocalServer` 而不是文件锁：进程被强杀时命名管道由系统回收，不会留下
    一个「上次没删掉的锁文件」把人挡在外面。
    """
    try:
        from PySide6.QtNetwork import QLocalServer, QLocalSocket
    except Exception:      # noqa: BLE001 - 没有 QtNetwork 就别拦，别耽误启动
        return True

    probe = QLocalSocket()
    probe.connectToServer(_INSTANCE_NAME)
    if probe.waitForConnected(300):
        probe.write(b"show")
        probe.flush()
        probe.waitForBytesWritten(300)
        probe.disconnectFromServer()
        return False

    QLocalServer.removeServer(_INSTANCE_NAME)
    server = QLocalServer(app)
    if not server.listen(_INSTANCE_NAME):
        return True        # 服务端起不来也不能拦人，宁可多开也不要打不开

    def _on_conn() -> None:
        conn = server.nextPendingConnection()
        if conn is not None:
            conn.disconnected.connect(conn.deleteLater)
        win = getattr(app, "_main_window", None)
        if win is not None:
            win.showNormal()
            win.raise_()
            win.activateWindow()

    server.newConnection.connect(_on_conn)
    app._instance_server = server      # 挂住，别被 GC 回收
    return True


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("趋势跟踪研究台")
    app.setApplicationDisplayName("趋势跟踪研究台")
    setup_light_palette(app)
    app.setStyleSheet(STYLE_SHEET)

    if not _claim_instance(app):
        return 0

    win = MainWindow()
    app._main_window = win
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
