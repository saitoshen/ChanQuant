"""「策略交易和组合」页。

和「策略研究」「账户透视和交易」平级的第三个大页面。它管的是**跑起来之后的事**：
组合里有哪些策略、每个绑哪个账户、现在跑到哪一步、账户仓位和策略理论仓位对不对得上。

页面结构：

    顶部  默认账户 / 启动全部 / 停止全部 / 状态灯
    左侧  策略列表：增删、单独启停、从策略库添加、绑定交易账户、查看回测
    右侧  监控器：逐合约列出「策略理论 / 账户实际 / 差额」，一键对齐
    底部  交易助手：等几秒撤单重报、追几次、每次追几跳

## 为什么这里也有一套 K 线 / 绩效 / 因子

「策略研究」页里那三页是**调参时**看的：改一个滑杆，图跟着变。但策略存进组合
之后，用户想知道的是「我存下的这一条，历史表现到底怎么样」—— 那不该让他跑回
研究页把参数一格一格调回来。所以双击列表里的一行，就地打开同一套结果视图。

结果视图是 `gui/result_view.ResultView`，和研究页**同一份实现**：一处改动两处
生效，不会出现「这边修好了那边还是老图」。

## 组合报告为什么只有一页绩效

单选一条策略：K 线、绩效对比（对买入持有）、因子分析、交易记录、阶段分析。
多选：只给绩效、交易记录、阶段分析 —— 组合是跨品种、跨周期加权，没有单一
对照物，硬拉一个「等权买入持有」只会误导。

## 为什么按账户分多个 Runner

天勤的 `TqApi` 在构造时绑定资金账号，一个 api 只能对一个账户；同账号再开第二个
连接还会把第一个踢下线。所以「策略 A 跑在账户甲、策略 B 跑在账户乙」只能靠
多个 Runner —— 每个账户一条连接。这一页负责把组合按 `cfg.account` 分组，
一组一个 `RunnerWorker`。

`cfg.account` 为空表示「跟随默认账户」，也就是顶部那个下拉框。

## 为什么启停要经过队列

和 `gui/live.py` 同一个道理：`TqApi` 不是线程安全的，`insert_order` / `cancel_order`
必须在它自己的线程里调。界面线程只往队列里塞闭包，worker 每轮 `wait_update`
之后统一执行。

## 一键对齐为什么连未托管仓位一起平

用户对这个按钮的预期是「账户和策略列表完全一致」。只补托管合约的话，策略停掉后
留在账户里的仓位、手工开的仓会一直挂在那儿 —— 监控器上永远红着，而按钮点下去
毫无反应。所以 `align()` 默认把未托管合约也平到 0。

但平仓是真金白银，而且这些仓位往往是用户自己开的。所以点下去先弹确认框，把
「哪个合约、现在几手、变成几手」一条条列出来，确认了才发单。
`align(close_orphans=False)` 保留「只补托管」的老行为，给脚本用。
"""

from __future__ import annotations

import queue
import time

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.result_view import PERIOD_LABEL, ResultView, ViewCtx
from ttrade import replay
from ttrade import strategy_store as SStore
from ttrade import tqlive
from ttrade.tqlive import merge_live
from ttrade.backtest import combine, combine_trades
from ttrade.strategies import PARAM_LABELS, STRATEGIES
from ttrade.strategy_runner import PortfolioRunner
from ttrade.universe import cn_name

# 与 trade_page.py / app.py 同一套配色。不 import 那两处：app 会 import 本模块。
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_UP = "#d92b2b"
C_DOWN = "#0f9d58"
C_OK = "#16a34a"
C_BAD = "#dc2626"
C_IDLE = "#94a3b8"


def _int(v) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return 0


# ------------------------------------------------------------------ 线程层


class RunnerWorker(QObject):
    """跑在天勤线程里：一条连接 + 一组策略。"""

    snapshot = Signal(str, object)     # (账户名, 快照)
    log = Signal(str)
    failed = Signal(str)

    def __init__(self, key: str, spec, cfgs: list, chase) -> None:
        super().__init__()
        self.key = key
        self.spec = spec
        self.cfgs = cfgs
        self.chase = chase
        self._stop = False
        self._queue: queue.Queue = queue.Queue()

    def submit(self, fn) -> None:
        self._queue.put(fn)

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        runner = PortfolioRunner(self.spec, self.chase)
        for cfg in self.cfgs:
            runner.add(cfg)
        runner.on_log(self.log.emit)
        try:
            runner.start()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{self.key}：{type(exc).__name__}: {exc}")
            runner.close()
            return

        # 连接成功之后再打开之前存下来的「运行中」标记 —— 连不上时打开没有意义，
        # 而且 enable() 里会去撤单，那时候还没有 api 对象。
        for cfg in self.cfgs:
            if cfg.enabled:
                runner.enable(cfg.sid, True)

        last = 0.0
        try:
            while not self._stop:
                try:
                    runner.wait(0.8)
                except Exception as exc:  # noqa: BLE001
                    self.failed.emit(f"{self.key} 行情中断：{exc}")
                    break

                while True:
                    try:
                        fn = self._queue.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        fn(runner)
                    except Exception as exc:  # noqa: BLE001
                        self.failed.emit(f"{self.key} 操作失败：{type(exc).__name__}: {exc}")

                runner.tick()

                now = time.monotonic()
                if now - last > 0.4:
                    last = now
                    self.snapshot.emit(self.key, runner.snapshot())
        finally:
            runner.close()


class PortfolioController(QObject):
    """主线程这一侧：按账户起停线程，转发快照，排队界面操作。"""

    snapshot = Signal(str, object)
    log = Signal(str)
    failed = Signal(str)
    stopped = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: dict[str, RunnerWorker] = {}
        self._threads: dict[str, QThread] = {}

    @property
    def running(self) -> bool:
        return bool(self._workers)

    def start(self, spec, cfgs: list, chase) -> None:
        key = spec.name
        if key in self._workers:
            return
        worker = RunnerWorker(key, spec, cfgs, chase)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.snapshot.connect(self.snapshot)
        worker.log.connect(self.log)
        worker.failed.connect(self.failed)
        # 连接失败时 `run()` 直接 return，不会再进 while 循环。线程的事件循环还
        # 开着，不显式 quit 就永远不发 `finished` —— `_workers` 里会残留一条
        # 「正在跑」的记录，「启动全部」按钮的逻辑就跟着错。
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(lambda k=key: self._on_done(k))
        self._workers[key] = worker
        self._threads[key] = thread
        thread.start()

    def stop(self, wait: bool = False) -> None:
        for key, worker in list(self._workers.items()):
            worker.stop()
            thread = self._threads.get(key)
            if thread is None:
                continue
            thread.quit()
            if wait:
                # worker 可能正卡在 wait_update(0.8)，最多等 5 秒
                thread.wait(5000)

    def _on_done(self, key: str) -> None:
        self._workers.pop(key, None)
        self._threads.pop(key, None)
        if not self._workers:
            self.stopped.emit()

    # ---- 操作：全部排队给对应账户的 worker ----

    def _submit(self, key: str, fn) -> bool:
        worker = self._workers.get(key)
        if worker is None:
            return False
        worker.submit(fn)
        return True

    def enable(self, key: str, sid: str, on: bool) -> bool:
        return self._submit(key, lambda r: r.enable(sid, on))

    def add(self, key: str, cfg) -> bool:
        return self._submit(key, lambda r: r.add(cfg))

    def remove(self, key: str, sid: str) -> bool:
        return self._submit(key, lambda r: r.remove(sid))

    def update_cfg(self, key: str, cfg) -> bool:
        return self._submit(key, lambda r: r.update(cfg))

    def align(self, key: str) -> bool:
        return self._submit(key, lambda r: r.align())

    def req_bars(self, key: str, sids: list) -> bool:
        """让引擎在下一份快照里带上这些策略的实时 K 线。"""
        return self._submit(key, lambda r: r.want_bars(sids))

    def set_chase(self, chase) -> None:
        for key in list(self._workers):
            self._submit(key, lambda r, c=chase: setattr(r, "chase", c))


# ------------------------------------------------------------------ 对话框


class LibraryDialog(QDialog):
    """从策略库挑一条。两处用：加进组合、在研究页加载参数。"""

    def __init__(self, items: list, parent=None, title: str = "从策略库添加",
                 hint: str = "") -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)

        self.lst = QListWidget()
        for cfg in items:
            it = QListWidgetItem(f"{cfg.title()}　{cfg.describe()}")
            it.setData(Qt.UserRole, cfg.sid)
            self.lst.addItem(it)
        if items:
            self.lst.setCurrentRow(0)
        self.lst.itemDoubleClicked.connect(lambda _i: self.accept())
        lay.addWidget(self.lst, 1)

        tip = QLabel(hint or
                     "加入组合会复制一份独立配置，之后在组合里改手数、改账户都不影响库里的模板。")
        tip.setObjectName("Hint")
        tip.setWordWrap(True)
        lay.addWidget(tip)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def sid(self) -> str:
        it = self.lst.currentItem()
        return "" if it is None else str(it.data(Qt.UserRole))


RANGE_CHOICES = [
    ("全部历史", 0.0), ("近 1 年", 1.0), ("近 2 年", 2.0),
    ("近 3 年", 3.0), ("近 5 年", 5.0), ("近 10 年", 10.0),
]


def _hint(text: str) -> QLabel:
    """表单里的一行小字说明。用 QLabel 而不是 groupbox：QSS 里没有它的样式。"""
    lbl = QLabel(text)
    lbl.setObjectName("Hint")
    lbl.setWordWrap(True)
    return lbl


class EditDialog(QDialog):
    """一条组合策略的全部可调项。

    分两段，因为影响的范围不一样：

    - **执行**（手数 / 做空 / 账户）只决定实盘怎么下单；
    - **策略参数**（快线、慢线…）实盘和回测同时生效 —— 改了它，下一根 K 线的
      信号就按新参数算；
    - **回测区间**只决定「双击这条策略看哪一段历史」，实盘信号永远只用最近的
      K 线，和这个值无关。
    """

    def __init__(self, cfg, accounts: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置参数")
        self.setMinimumWidth(430)
        form = QFormLayout(self)

        self.ed_name = QLineEdit(cfg.name)
        self.ed_name.setPlaceholderText(cfg.title())
        form.addRow("名称", self.ed_name)

        # ---- 仓位分配 ----
        # 默认不再是「1 手」：1 手螺纹和 1 手铜差 40 倍名义价值，同一条策略在两个
        # 品种上跑出来的风险完全不是一回事。默认按「分配资金 ÷ 每手保证金」铺满。
        self.cb_size = QComboBox()
        for text, val in (("按资金分配", "capital"),
                          ("按止损金额", "risk"),
                          ("固定手数", "fixed")):
            self.cb_size.addItem(text, val)
        i = self.cb_size.findData(cfg.sizing or "capital")
        self.cb_size.setCurrentIndex(max(i, 0))
        self.cb_size.setToolTip(
            "按资金：手数 = 分配资金 ÷ 每手保证金。\n"
            "杠杆由品种自己的保证金比例决定（商品、股指约 10 倍，国债 30~50 倍）。\n"
            "按止损金额：手数 = 止损金额 ÷（止损倍数 × ATR × 合约乘数）。\n"
            "固定手数：老的「几手」口径。"
        )
        form.addRow("仓位分配", self.cb_size)

        self.sp_capital = QDoubleSpinBox()
        self.sp_capital.setRange(1000, 100_000_000)
        self.sp_capital.setDecimals(0)
        self.sp_capital.setSingleStep(50000)
        self.sp_capital.setSuffix(" 元")
        self.sp_capital.setValue(float(cfg.capital or 500_000))
        self.sp_capital.setToolTip("分给这条策略的钱。手数按每手保证金折算，不按「几手」拍。")
        form.addRow("分配资金", self.sp_capital)

        self.sp_vol = QSpinBox()
        self.sp_vol.setRange(1, 9999)
        self.sp_vol.setValue(max(1, _int(cfg.volume) or 1))
        form.addRow("固定手数", self.sp_vol)

        self.sp_risk = QDoubleSpinBox()
        self.sp_risk.setRange(1000, 100_000_000)
        self.sp_risk.setDecimals(0)
        self.sp_risk.setSingleStep(5000)
        self.sp_risk.setSuffix(" 元")
        self.sp_risk.setValue(float(cfg.risk_amount or 20_000))
        self.sp_risk.setToolTip("打止损时最多亏这么多钱，手数由止损距离反推。")
        form.addRow("止损金额", self.sp_risk)

        self.sp_atr = QSpinBox()
        self.sp_atr.setRange(2, 200)
        self.sp_atr.setValue(max(2, _int(cfg.stop_atr_n) or 14))
        form.addRow("止损 ATR 周期", self.sp_atr)

        self.sp_stop = QDoubleSpinBox()
        self.sp_stop.setRange(0.1, 50.0)
        self.sp_stop.setSingleStep(0.5)
        self.sp_stop.setDecimals(1)
        self.sp_stop.setValue(float(cfg.stop_mult or 5.0))
        form.addRow("止损 ATR 倍数", self.sp_stop)

        self.cb_size.currentIndexChanged.connect(self._sync_size)
        self._sync_size()

        self.chk_short = QCheckBox("允许做空")
        self.chk_short.setChecked(cfg.note == "short")
        form.addRow("", self.chk_short)

        self.cb_acct = QComboBox()
        self.cb_acct.addItem("跟随默认账户", "")
        for a in accounts:
            self.cb_acct.addItem(a, a)
        idx = self.cb_acct.findData(cfg.account or "")
        self.cb_acct.setCurrentIndex(max(idx, 0))
        form.addRow("交易账户", self.cb_acct)

        # ---- 策略参数 ----
        self.sp_params: dict[str, QSpinBox] = {}
        spec = STRATEGIES.get(cfg.rule_key) if cfg.family == "rule" else None
        if spec is not None and spec.params:
            form.addRow(_hint(f"策略参数 · {spec.label}（实盘与回测同时生效）"))
            for name, default in spec.params.items():
                lo, hi, step = spec.param_meta.get(name, (1, 1000, 1))
                sp = QSpinBox()
                sp.setRange(int(lo), int(hi))
                sp.setSingleStep(max(1, int(step)))
                try:
                    sp.setValue(int(cfg.rule_params.get(name, default)))
                except (TypeError, ValueError):
                    sp.setValue(int(default))
                self.sp_params[name] = sp
                form.addRow(PARAM_LABELS.get(name, name), sp)
        elif cfg.family == "factor":
            form.addRow(_hint(
                "因子策略的参数（窗口 / 阈值 / 因子）请在「策略研究」页调好，"
                "再存入策略库、重新加入组合"
            ))

        # ---- 回测区间 ----
        self.cb_range = QComboBox()
        for text, years in RANGE_CHOICES:
            self.cb_range.addItem(text, years)
        i = self.cb_range.findData(float(cfg.range_years or 0))
        self.cb_range.setCurrentIndex(max(i, 0))
        self.cb_range.setToolTip(
            "双击这条策略时回测和看图的范围。\n"
            "实盘信号只用最近的 K 线，不受这里影响。"
        )
        form.addRow("回测区间", self.cb_range)

        lab = QLabel(f"{cfg.key()} {cfg.period}｜{cfg.strategy_label()}")
        lab.setObjectName("Hint")
        form.addRow("策略", lab)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def _sync_size(self) -> None:
        """只显示当前口径用得上的字段，免得一堆数字看不出哪个生效。"""
        mode = self.cb_size.currentData()
        form = self.layout()
        if form is None:
            return
        for w, on in ((self.sp_capital, mode == "capital"),
                      (self.sp_vol, mode == "fixed"),
                      (self.sp_risk, mode == "risk"),
                      (self.sp_atr, mode == "risk"),
                      (self.sp_stop, mode == "risk")):
            try:
                form.setRowVisible(w, on)
            except (TypeError, AttributeError):   # 老 Qt 没有这个重载
                pass

    def apply_to(self, cfg) -> None:
        cfg.name = self.ed_name.text().strip()
        cfg.sizing = self.cb_size.currentData() or "capital"
        cfg.capital = float(self.sp_capital.value())
        cfg.risk_amount = float(self.sp_risk.value())
        cfg.stop_atr_n = int(self.sp_atr.value())
        cfg.stop_mult = float(self.sp_stop.value())
        cfg.volume = self.sp_vol.value()
        cfg.note = "short" if self.chk_short.isChecked() else ""
        cfg.account = self.cb_acct.currentData() or ""
        if self.sp_params:
            cfg.rule_params = {k: sp.value() for k, sp in self.sp_params.items()}
        cfg.range_years = float(self.cb_range.currentData() or 0.0)


# ------------------------------------------------------------------ 页面


class PortfolioPage(QWidget):
    """策略交易和组合。"""

    def __init__(self, loader=None, parent=None) -> None:
        super().__init__(parent)
        # loader(market, key, period) -> DataFrame | None。主窗口把它的 DataStore
        # 递进来 —— 这一页自己不该再扫一遍数据目录（那是几秒钟的事）。
        self._loader = loader
        self._items = SStore.load_portfolio()
        self._accounts = tqlive.load_accounts()
        self._chase = SStore.load_chase()
        self._snaps: dict[str, dict] = {}
        self._slot_view: dict[str, dict] = {}     # sid -> 快照里的 slot
        self._render_ts = 0.0                     # 上次重画两张表的时刻
        self._mon_sig = None                      # 监控表上次画的内容
        # 单策略视图的实时 K 线。`_live_base` 是本地库里的历史，实时段每 5 秒
        # 拼一次 —— 引擎订的是实时行情，界面却只画本地库，打开一看停在几天前，
        # 用户会以为策略没在跟数据。`_live_busy` 防重入：请求发出去了、快照
        # 还没回来时再发一次，只会排一串没用的活。
        self._live_cfg = None
        self._live_base = None
        self._live_sid = ""
        self._live_busy = False
        self._live_req = 0.0
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(5000)
        self._live_timer.timeout.connect(self._poll_live)
        self._views: dict[str, tuple[QTabWidget, ResultView]] = {}
        self.ctl = PortfolioController(self)
        self._build()
        self._load_chase_widgets()
        self._chase_hint(self._chase)
        self.reload()

        self.ctl.snapshot.connect(self._on_snapshot)
        self.ctl.log.connect(self._on_log)
        self.ctl.failed.connect(self._on_failed)
        self.ctl.stopped.connect(self._on_stopped)

    # ---------------------------------------------------------- 构建

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # 两页：管理界面 / 结果视图。用 QStackedWidget 而不是弹窗 —— 弹窗里塞
        # 五个带 matplotlib 的页签，窗口一小图表就糊成一片，还得再调一次尺寸。
        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_manage())
        self.stack.addWidget(self._build_view_page())
        root.addWidget(self.stack, 1)

    def _build_manage(self) -> QWidget:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addWidget(self._build_top())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_strategy_card())
        split.addWidget(self._build_monitor_card())
        split.setSizes([760, 460])
        root.addWidget(split, 1)

        root.addWidget(self._build_chase_card())
        return page

    def _build_view_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        bar = QWidget()
        bar.setObjectName("Card")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 6, 12, 6)
        bl.setSpacing(10)
        self.btn_back = QPushButton("← 返回策略列表")
        self.btn_back.setObjectName("Ghost")
        self.btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        bl.addWidget(self.btn_back)

        self.lbl_view = QLabel("")
        self.lbl_view.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        bl.addWidget(self.lbl_view, 1)
        lay.addWidget(bar)

        # 单策略视图和组合视图各建一次、常驻不销毁：销毁要停 QTimer、要防
        # 已排队的重绘打到已删的画布上，切回来还得重建五个页签。留着更省事，
        # 代价只是多几 MB 的 Figure。
        self._view_stack = QStackedWidget()
        lay.addWidget(self._view_stack, 1)
        return page

    def _ensure_view(self, kind: str) -> tuple[QTabWidget, ResultView]:
        hit = self._views.get(kind)
        if hit is not None:
            return hit
        tabs = QTabWidget()
        view = ResultView(kind, self)
        for title, w in view.pages():
            tabs.addTab(w, title)
        view.status.connect(self.lbl_view.setText)
        self._view_stack.addWidget(tabs)
        self._views[kind] = (tabs, view)
        return tabs, view

    def _show_view(self, kind: str, title: str) -> None:
        tabs, _view = self._ensure_view(kind)
        self._view_stack.setCurrentWidget(tabs)
        self.lbl_view.setText(title)
        self.stack.setCurrentIndex(1)

    def _build_top(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(8)

        lay.addWidget(QLabel("默认账户"))
        self.cb_acct = QComboBox()
        self.cb_acct.setMinimumWidth(200)
        self.cb_acct.setToolTip("组合里没单独指定账户的策略，都用这个账户")
        lay.addWidget(self.cb_acct)

        self.btn_start = QPushButton("启动全部")
        self.btn_start.clicked.connect(self._start_all)
        lay.addWidget(self.btn_start)

        self.btn_stop = QPushButton("停止全部")
        self.btn_stop.setObjectName("Ghost")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_all)
        lay.addWidget(self.btn_stop)

        self.lbl_dot = QLabel("●")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        lay.addWidget(self.lbl_dot)
        self.lbl_conn = QLabel("未启动")
        self.lbl_conn.setObjectName("Hint")
        lay.addWidget(self.lbl_conn)
        # 交易所对日内报单 / 撤单次数有硬上限，超了会被限制交易。这一格让人
        # 随时看得到还剩多少，而不是等停手了才去翻日志
        self.lbl_limit = QLabel("")
        self.lbl_limit.setObjectName("Hint")
        self.lbl_limit.setToolTip(
            "交易所对每个账户的日内报单 / 撤单笔数有硬上限，"
            "本程序在 500 次时自动停手"
        )
        lay.addWidget(self.lbl_limit)

        lay.addStretch(1)
        self.lbl_msg = QLabel("")
        self.lbl_msg.setObjectName("Hint")
        lay.addWidget(self.lbl_msg)
        return card

    def _card(self, title: str) -> tuple[QWidget, QVBoxLayout, QHBoxLayout]:
        card = QWidget()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("SectionTitle")
        head.addWidget(t)
        head.addStretch(1)
        lay.addLayout(head)
        return card, lay, head

    def _table(self, headers: list[str], stretch: int = 0) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setShowGrid(False)
        h = t.horizontalHeader()
        for i in range(len(headers)):
            h.setSectionResizeMode(
                i, QHeaderView.Stretch if i == stretch else QHeaderView.ResizeToContents
            )
        return t

    def _build_strategy_card(self) -> QWidget:
        card, lay, head = self._card("策略列表")

        self.btn_view = QPushButton("查看回测")
        self.btn_view.setToolTip("打开这一条策略的 K 线与信号 / 绩效 / 因子分析 / 交易记录 / 阶段分析")
        self.btn_view.clicked.connect(self._open_single)
        head.addWidget(self.btn_view)

        self.btn_combo = QPushButton("组合报告")
        self.btn_combo.setObjectName("Ghost")
        self.btn_combo.setToolTip("按住 Ctrl 选中多条策略，看它们等权组合起来的资金曲线")
        self.btn_combo.setEnabled(False)
        self.btn_combo.clicked.connect(self._open_combo)
        head.addWidget(self.btn_combo)

        btn_lib = QPushButton("从策略库添加")
        btn_lib.setObjectName("Ghost")
        btn_lib.clicked.connect(self._add_from_library)
        head.addWidget(btn_lib)

        btn_edit = QPushButton("设置参数")
        btn_edit.setObjectName("Ghost")
        btn_edit.setToolTip("改仓位分配（资金 / 止损金额 / 固定手数）、账户、"
                            "策略参数（快线慢线…）和回测区间")
        btn_edit.clicked.connect(self._edit_selected)
        head.addWidget(btn_edit)

        btn_toggle = QPushButton("启动 / 停止")
        btn_toggle.setObjectName("Ghost")
        btn_toggle.clicked.connect(self._toggle_selected)
        head.addWidget(btn_toggle)

        btn_del = QPushButton("删除")
        btn_del.setObjectName("Ghost")
        btn_del.clicked.connect(self._del_selected)
        head.addWidget(btn_del)

        self.tbl_strat = self._table(
            ["名称", "标的", "策略", "仓位", "账户", "理论", "实际", "状态"]
        )
        # 多选是为了组合报告。双击原来是「启动 / 停止」，但那个动作有同名按钮，
        # 而「看一眼这条策略的历史表现」没有别的入口 —— 双击给后者更值。
        self.tbl_strat.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tbl_strat.doubleClicked.connect(lambda _i: self._open_single())
        self.tbl_strat.itemSelectionChanged.connect(self._sync_buttons)
        lay.addWidget(self.tbl_strat, 1)

        hint = QLabel(
            "双击一行 = 打开它的 K 线与绩效。按住 Ctrl 多选后点「组合报告」，"
            "看这几条策略等权组合的资金曲线。启停请用上面的按钮。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        return card

    def _sync_buttons(self) -> None:
        """选中几条就改一次按钮：只有一条时「组合报告」没有意义。"""
        n = len(self._selected_cfgs())
        self.btn_combo.setEnabled(n >= 2)
        self.btn_combo.setText(f"组合报告（{n}）" if n >= 2 else "组合报告")
        self.btn_view.setEnabled(n >= 1)

    def _build_monitor_card(self) -> QWidget:
        card, lay, head = self._card("监控器")

        btn_align = QPushButton("一键对齐")
        btn_align.setToolTip("让账户持仓与策略列表完全一致：托管合约补到理论持仓，\n"
                             "没有策略在管的仓位平掉（会先弹确认框）")
        btn_align.clicked.connect(self._align)
        head.addWidget(btn_align)

        # 多账户时同一个合约会在两张连接里各出现一次，必须带上账户才分得清
        self.tbl_mon = self._table(
            ["账户", "合约", "策略理论", "账户实际", "差额", "说明"], stretch=5
        )
        lay.addWidget(self.tbl_mon, 1)

        self.lbl_mon = QLabel("一致")
        self.lbl_mon.setObjectName("Hint")
        self.lbl_mon.setWordWrap(True)
        lay.addWidget(self.lbl_mon)
        return card

    def _build_chase_card(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)

        t = QLabel("交易助手")
        t.setObjectName("SectionTitle")
        lay.addWidget(t)

        self.chk_chase = QCheckBox("启用")
        self.chk_chase.setToolTip("关掉之后策略发出的委托就一直挂着，不撤不追")
        lay.addWidget(self.chk_chase)

        lay.addWidget(QLabel("等"))
        self.sp_wait = QSpinBox()
        self.sp_wait.setRange(1, 600)
        self.sp_wait.setSuffix(" 秒")
        self.sp_wait.setMinimumWidth(78)
        self.sp_wait.setToolTip("委托挂多久没成交就撤掉重报")
        lay.addWidget(self.sp_wait)

        lay.addWidget(QLabel("未成交则撤单重报，最多追"))
        self.sp_retry = QSpinBox()
        self.sp_retry.setRange(0, 50)
        self.sp_retry.setSuffix(" 次")
        self.sp_retry.setMinimumWidth(78)
        lay.addWidget(self.sp_retry)

        lay.addWidget(QLabel("每次追"))
        self.sp_ticks = QSpinBox()
        self.sp_ticks.setRange(0, 200)
        self.sp_ticks.setSuffix(" 跳")
        self.sp_ticks.setMinimumWidth(78)
        self.sp_ticks.setToolTip("按盘口对价，每次往成交方向多让几跳")
        lay.addWidget(self.sp_ticks)

        self.chk_market = QCheckBox("最后一次用市价")
        lay.addWidget(self.chk_market)

        lay.addStretch(1)
        self.lbl_chase = QLabel("")
        self.lbl_chase.setObjectName("Hint")
        lay.addWidget(self.lbl_chase)

        for w in (self.chk_chase, self.chk_market):
            w.toggled.connect(self._save_chase)
        for w in (self.sp_wait, self.sp_retry, self.sp_ticks):
            w.valueChanged.connect(self._save_chase)
        return card

    # ---------------------------------------------------------- 数据

    def reload(self) -> None:
        """从磁盘重读组合与账户。切回本页时调，能看到「策略研究」里刚加的那条。"""
        self._items = SStore.load_portfolio()
        self._accounts = tqlive.load_accounts()
        self._reload_accounts()
        self._fill_strategies()
        self._render()

    def _reload_accounts(self) -> None:
        cur = self.cb_acct.currentData()
        self.cb_acct.blockSignals(True)
        self.cb_acct.clear()
        for s in self._accounts:
            self.cb_acct.addItem(s.label(), s.name)
        if cur:
            idx = self.cb_acct.findData(cur)
            if idx >= 0:
                self.cb_acct.setCurrentIndex(idx)
        self.cb_acct.blockSignals(False)

    def _spec(self, name: str):
        for s in self._accounts:
            if s.name == name:
                return s
        return None

    def _account_of(self, cfg) -> str:
        """这条策略实际用哪个账户。空 = 跟随默认账户。"""
        return cfg.account or (self.cb_acct.currentData() or "")

    # ---------------------------------------------------------- 渲染

    def _fill_strategies(self) -> None:
        """重建策略表。选中行按 sid 找回，不然每次刷新都会跳回第一行。"""
        keep = set(self._selected_sids())
        t = self.tbl_strat
        t.blockSignals(True)
        try:
            t.setRowCount(len(self._items))
            for i, cfg in enumerate(self._items):
                vals = [
                    cfg.name.strip() or cfg.title(),
                    f"{cfg.key()} {cfg.period}",
                    cfg.strategy_label(),
                    cfg.size_text(),
                    cfg.account or "默认",
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setData(Qt.UserRole, cfg.sid)
                    if j == 3:
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    t.setItem(i, j, item)
                for j in (5, 6, 7):
                    t.setItem(i, j, QTableWidgetItem("—"))
            if keep:
                for i in range(t.rowCount()):
                    it = t.item(i, 0)
                    if it is not None and it.data(Qt.UserRole) in keep:
                        t.selectRow(i)
        finally:
            t.blockSignals(False)
        self._sync_buttons()

    def _selected_sids(self) -> list[str]:
        model = self.tbl_strat.selectionModel()
        if model is None:
            return []
        out = []
        for i in sorted({i.row() for i in model.selectedRows()}):
            it = self.tbl_strat.item(i, 0)
            sid = "" if it is None else str(it.data(Qt.UserRole) or "")
            if sid:
                out.append(sid)
        return out

    def _selected_cfgs(self) -> list:
        """当前选中的所有配置，按表格顺序。组合报告要这个。"""
        sids = self._selected_sids()
        out = []
        for sid in sids:
            for cfg in self._items:
                if cfg.sid == sid:
                    out.append(cfg)
                    break
        return out

    # ---------------------------------------------------------- 结果视图

    def _load(self, cfg):
        """取这个配置的行情。返回 (df, 出错原因)。"""
        if self._loader is None:
            return None, "没有接上行情数据源"
        mk = replay.market_of(cfg)
        key = cfg.key()
        try:
            df = self._loader(mk, key, cfg.period or "1d")
        except Exception as exc:  # noqa: BLE001
            return None, f"{key} 读取失败：{exc}"
        if df is None or df.empty:
            return None, f"{key} 没有 {cfg.period} 的行情数据"
        return df, ""

    def _ctx(self, cfg, rp, period: str = "") -> ViewCtx:
        return ViewCtx(
            symbol=cfg.symbol,
            name=cn_name(cfg.symbol) or "",
            exchange=cfg.exchange,
            period=period or cfg.period,
            market=replay.market_of(cfg),
            label=rp.label,
            legend=("▲紫＝开多　▼紫＝平多" if rp.long_only
                    else "▲紫＝开多　▼紫＝平多　▼琥珀＝开空　▲琥珀＝平空"),
            price_note="价格（后复权）" if not cfg.exchange else "价格（前复权·主力连续）",
            # 配置里存了回测区间就只画这一段。`replay.run` 已经按区间切过数据，
            # 前面多留的 250 根预热段在这里被裁掉 —— 指标是全区间算的，图上只显示
            # 用户选的那几年，两边看到的是同一段行情
            range_years=(float(cfg.range_years) or None),
            rule_key=cfg.rule_key,
            overlay=True,
            long_only=rp.long_only,
            bars_per_year=rp.bars_per_year,
            factor_cfg=rp.factor_cfg,
            filter_active=rp.filter_active,
            selected_keys=rp.keys,
            outcome=rp.outcome,
        )

    def _open_single(self) -> None:
        cfgs = self._selected_cfgs()
        if not cfgs:
            self._on_failed("先选中一条策略")
            return
        if len(cfgs) > 1:
            self._on_log(f"选中了 {len(cfgs)} 条，先显示第一条；要看整体请点「组合报告」")
        self.show_config(cfgs[0])

    def _open_combo(self) -> None:
        cfgs = self._selected_cfgs()
        if len(cfgs) < 2:
            self._on_failed("按住 Ctrl 至少选两条策略，才谈得上组合")
            return
        self.show_combo(cfgs)

    def show_config(self, cfg, note: str = "") -> str:
        """回放一份配置并展示。返回一段纯文本结论（失败时是原因）。

        「交易 Agent」发起的 backtest 走这里：跑完把文本推回 Agent 页的输出框，
        模型才知道算出来是什么。所以失败也要返回原因，不能只弹个红字。
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            df, why = self._load(cfg)
            if df is None:
                return f"回测失败：{why}"
            rp = replay.run(cfg, df)
        finally:
            QApplication.restoreOverrideCursor()
        if not rp.ok:
            return f"回测失败：{rp.error}"

        _tabs, view = self._ensure_view("single")
        view.show_result(rp.df, rp.raw, rp.res, rp.bh, self._ctx(cfg, rp))
        # 记下本地历史这份「底」，实时段每 5 秒往上拼一次。存 df 而不是重读库：
        # 库里那份要解码几万行，5 秒重读一次等于白烧
        self._live_cfg, self._live_base, self._live_sid = cfg, df, cfg.sid
        self._live_busy = False
        self._live_timer.start()
        period = PERIOD_LABEL.get(cfg.period, cfg.period)
        # cfg.title() 已经含「品种 · 策略」，别再拼一遍 key 和策略名
        self._show_view("single", f"{cfg.title()}｜{period}"
                                 + (f"｜{note}" if note else ""))
        return replay.summary(rp, cfg.key(), period)

    def show_combo(self, cfgs: list) -> str:
        """多策略等权组合报告。返回结论文本（失败时是原因）。"""
        reps: list = []
        missing: list[str] = []
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for cfg in cfgs:
                df, why = self._load(cfg)
                if df is None:
                    missing.append(why)
                    continue
                rp = replay.run(cfg, df, with_bh=False)
                if not rp.ok:
                    missing.append(f"{cfg.title()}：{rp.error}")
                    continue
                reps.append((cfg, rp))
        finally:
            QApplication.restoreOverrideCursor()

        if not reps:
            return "组合报告失败：" + "；".join(missing or ["没有可用的策略"])
        if len(reps) < 2:
            # 一条都算不出来时「组合」没有意义，直接说清楚少了谁
            return ("组合报告需要至少 2 条能算出结果的策略。"
                    + ("　" + "；".join(missing) if missing else ""))

        # 周期一致就照抄，混着就留空：留空时交易记录会带上时分（分钟级策略
        # 的交易时间差几十分钟是有意义的），持仓长度也改用「根（约 N 天）」
        periods = {c.period for c, _r in reps}
        period = periods.pop() if len(periods) == 1 else ""
        res = combine([rp.res for _c, rp in reps], label="组合")
        # 这里的 bars_per_year 是给 `hold_text` 换算「平均持仓 N 根」用的，取的是
        # **策略自己的** bar 频率（交易记录里的 bars 就是它）。组合曲线本身落在
        # 日频上，那个频率在 `combine` 内部算指标时用，别混用
        bpy = reps[0][1].bars_per_year
        # 交易记录按开仓时间混排：分组拼接的话，表格看起来像「先跑完 A 再跑 B」
        trades = combine_trades([(c.title(), rp.res) for c, rp in reps])
        ctx = ViewCtx(
            symbol="", name="", exchange="", period=period,
            market=replay.market_of(reps[0][0]), label="组合",
            price_note="净值", long_only=False, bars_per_year=bpy,
            selected_keys=[], overlay=False,
        )
        # 组合是几条策略的净值合成，没有「这一条策略的实时 K 线」可言
        self._stop_live()
        tabs, view = self._ensure_view("portfolio")
        view.show_result(None, None, res, None, ctx, trades=trades)
        names = "、".join(c.title() for c, _r in reps)
        self._show_view("portfolio", f"组合报告｜{len(reps)} 条策略等权｜{names}")

        head = f"组合（{len(reps)} 条策略）"
        rp = replay.Replay(res=res, bh=None, period=period,
                           bars_per_year=bpy)
        text = replay.summary(rp, head, "等权")
        if not period:
            # 混了日线和分钟线时「平均持仓 40 根」没法换算成天数：一根 30 分钟
            # 和一根日线不是同一个单位。与其给个错的天数，不如直说
            text += "\n注意：组合里混了不同周期，平均持仓的根数不可比"
        if missing:
            text += "\n未能加入组合：" + "；".join(missing)
        return text

    # 引擎 0.4 秒推一次快照。每次都把两张表全量重画，合约一多主线程就被占满，
    # 界面上看着就是「一切到这页就卡」。1 秒一次足够看清，也不耽误看数。
    RENDER_GAP = 1.0

    def _on_snapshot(self, key: str, snap: dict) -> None:
        self._snaps[key] = snap
        for slot in snap.get("slots", []):
            self._slot_view[slot["sid"]] = slot
        bars = snap.get("bars") or {}
        if bars:
            self._apply_live(bars)
        if time.monotonic() - self._render_ts < self.RENDER_GAP:
            return
        self._render()

    def _stop_live(self) -> None:
        self._live_timer.stop()
        self._live_cfg = self._live_base = None
        self._live_sid = ""
        self._live_busy = False

    def _poll_live(self) -> None:
        """5 秒一趟：问引擎要一次当前这条策略的实时 K 线。

        引擎 0.4 秒推一次快照，但那里面不带 K 线（3000 行，每份都带太贵），
        这里按需点名要。找不到在跑的账户就安静等着 —— 策略没启动时也有图看，
        只是它停在库里最后一根。
        """
        # 不在结果页就别问了：切回列表还每 5 秒转一遍 3000 行，纯浪费
        if self.stack.currentIndex() != 1:
            return
        if self._live_cfg is None or not self.ctl.running:
            return
        # 请求发出去了却一直没回来（账户线程在重连之类），别把自己锁死
        if self._live_busy and time.monotonic() - self._live_req < 10:
            return
        sid = self._live_sid
        for key, snap in self._snaps.items():
            if any(s.get("sid") == sid for s in (snap.get("slots") or [])):
                self._live_busy = True
                self._live_req = time.monotonic()
                self.ctl.req_bars(key, [sid])
                return

    def _apply_live(self, bars: dict) -> None:
        """把实时段拼到本地历史后面，重跑一遍再画。"""
        self._live_busy = False
        cfg = self._live_cfg
        live = bars.get(self._live_sid)
        if cfg is None or live is None or live.empty:
            return
        try:
            df = merge_live(self._live_base, live)
            if df is None or df.empty:
                return
            rp = replay.run(cfg, df)
        except Exception as exc:  # noqa: BLE001
            self._on_failed(f"实时 K 线刷新失败：{type(exc).__name__}: {exc}")
            return
        if not rp.ok:
            return
        _tabs, view = self._ensure_view("single")
        # keep_window：实时刷新只是多了几根，别把用户翻到的位置拽回最新
        view.show_result(rp.df, rp.raw, rp.res, rp.bh, self._ctx(cfg, rp),
                         keep_window=True)

    def _render(self) -> None:
        self._render_ts = time.monotonic()
        t = self.tbl_strat
        for i, cfg in enumerate(self._items):
            slot = self._slot_view.get(cfg.sid) or {}
            if i >= t.rowCount():
                break
            theo = _int(slot.get("target"))
            real = _int(slot.get("actual"))
            # 仓位列写「口径 → 实际算出来的手数」。手数依赖价格和每手保证金，
            # 存盘时算不出来，只有引擎跑起来才知道 —— 所以两段拼在一起显示。
            lots = _int(slot.get("lots"))
            size = cfg.size_text() + (f" → {lots} 手" if lots > 0 else "")
            self._set_cell(t, i, 3, size, C_TEXT)
            it3 = t.item(i, 3)
            if it3 is not None:
                it3.setToolTip(
                    f"每手保证金按品种自己的比例算（商品/股指约 10 倍，国债 30~50 倍）。"
                    f"\n当前价下满仓 {lots} 手" if lots > 0 else
                    "引擎还没算出实际手数（需要行情）"
                )
            self._set_cell(t, i, 5, str(theo), C_TEXT)
            # 运行中且仓位对不上才标红。停掉的策略留着仓位往往是用户故意的，
            # 那是「未托管」的事，监控器里会单独列出来。
            off = bool(slot.get("running")) and real != theo
            self._set_cell(t, i, 6, str(real), C_BAD if off else C_TEXT)

            # 状态列只放短标签，「买了 2 手 @ 3046」这种细节放 tooltip ——
            # 塞进表格会把列挤成一团，横向滚动条一起来就没人看了。
            if not slot:
                text, color, tip = "未连接", C_IDLE, "还没启动"
            elif slot.get("error"):
                text, color, tip = "错误", C_BAD, slot["error"]
            elif slot.get("running"):
                text, color = "运行中", C_OK
                tip = slot.get("action") or ""
            elif cfg.enabled:
                text, color = "已停止", C_MUTED
                tip = "启动开关是开的，但这一轮没跑起来 —— 看看顶部连接状态和日志"
            else:
                text, color, tip = "未启用", C_IDLE, "点「启动 / 停止」把它打开"
            self._set_cell(t, i, 7, text, color)
            it = t.item(i, 7)
            if it is not None and it.toolTip() != tip:
                it.setToolTip(tip)

        rows = []
        for key, snap in self._snaps.items():
            for r in snap.get("mismatch") or []:
                rows.append((key, r))
        rows.sort(key=lambda kr: (not kr[1].get("managed"), kr[0], kr[1]["symbol"]))
        m = self.tbl_mon
        # 内容没变就不重建。重建一次要给每格新建 QTableWidgetItem，
        # 几十个合约不一致的时候这活儿一秒干一次，主线程就废了
        sig = tuple((k, r["symbol"], r["target"], r["actual"], r["delta"],
                     bool(r.get("managed"))) for k, r in rows)
        if sig != self._mon_sig:
            self._mon_sig = sig
            m.setRowCount(len(rows))
            for i, (key, r) in enumerate(rows):
                managed = bool(r.get("managed"))
                vals = [
                    key, r["symbol"], str(r["target"]), str(r["actual"]),
                    f"{r['delta']:+d}",
                    "策略持仓" if managed else "未托管（策略已停或手工开仓）",
                ]
                for j, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    if j in (2, 3, 4):
                        item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    if j == 4:
                        item.setForeground(QColor(C_BAD if managed else C_MUTED))
                    if j == 5 and not managed:
                        item.setForeground(QColor(C_MUTED))
                    m.setItem(i, j, item)

        limits = [v.get("limits") for v in self._snaps.values() if v.get("limits")]
        if limits:
            self.lbl_limit.setText(" · ".join(limits))
            used = self._snaps and any(
                "500/500" in (v.get("limits") or "") for v in self._snaps.values()
            )
            self.lbl_limit.setStyleSheet(
                f"color: {C_BAD if used else C_MUTED}; font-size: 11px;")
        else:
            self.lbl_limit.setText("")

        aligning = sum(int(v.get("aligning") or 0) for v in self._snaps.values())
        if not self.ctl.running:
            self.lbl_mon.setText("未启动。策略跑起来之后这里才会比对。")
            self.lbl_mon.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        elif aligning:
            self.lbl_mon.setText(
                f"对齐执行中：{aligning} 笔委托在途，等它们成交或撤掉。"
                "这期间策略不调仓，也不用再点。"
            )
            self.lbl_mon.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        elif not rows:
            self.lbl_mon.setText("账户仓位与策略理论持仓一致。")
            self.lbl_mon.setStyleSheet(f"color: {C_OK}; font-size: 11px;")
        else:
            n_managed = sum(1 for _k, r in rows if r.get("managed"))
            n_orphan = len(rows) - n_managed
            bits = []
            if n_managed:
                bits.append(f"{n_managed} 个合约与策略理论持仓不一致")
            if n_orphan:
                bits.append(f"{n_orphan} 个合约没有策略在管")
            self.lbl_mon.setText(
                "，".join(bits) + "。「一键对齐」会把账户摆成和策略列表一致，\n"
                "未托管的仓位也会被平掉（会先列出来让你确认）。"
            )
            self.lbl_mon.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")

    @staticmethod
    def _set_cell(t: QTableWidget, row: int, col: int, text: str, color: str) -> None:
        item = t.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            if col in (5, 6):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            t.setItem(row, col, item)
        if item.text() != text:
            item.setText(text)
        item.setForeground(QColor(color))

    # ---------------------------------------------------------- 起停

    def _start_all(self) -> None:
        if self.ctl.running:
            self._on_failed("已经在运行，先停止再重新启动")
            return
        if not self._items:
            self._on_failed("组合里还没有策略，先「从策略库添加」")
            return
        if not self._accounts:
            self._on_failed("还没有交易账户，请先到「账户透视和交易」页添加")
            return

        groups: dict[str, list] = {}
        for cfg in self._items:
            name = self._account_of(cfg)
            if not name:
                self._on_failed(f"「{cfg.title()}」没有可用账户")
                return
            if self._spec(name) is None:
                self._on_failed(f"账户「{name}」已不存在，请重新绑定")
                return
            groups.setdefault(name, []).append(cfg)

        for name, cfgs in groups.items():
            self.ctl.start(self._spec(name), cfgs, self._chase)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.lbl_dot.setStyleSheet(f"color: {C_OK}; font-size: 15px;")
        self.lbl_conn.setText(f"运行中 · {len(groups)} 个账户 / {len(self._items)} 条策略")
        self.lbl_conn.setStyleSheet(f"color: {C_OK}; font-size: 11px;")
        self._on_log(f"启动 {len(groups)} 个账户的连接")

    def _stop_all(self) -> None:
        if not self.ctl.running:
            return
        self.ctl.stop()
        # 这里不重画：线程还没真退，`ctl.running` 仍是 True，重画会走到
        # 「仓位一致」那个分支，看着像是刚对齐过。等 `_on_stopped` 再刷。
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.lbl_conn.setText("正在停止…")
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        self._on_log("正在停止…")

    def _on_stopped(self) -> None:
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        self.lbl_conn.setText("已停止")
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        self._slot_view.clear()
        self._snaps.clear()
        self._render()

    def _single_or_warn(self) -> object | None:
        """这几个动作一次只能作用在一条策略上。多选时先劝一句，别闷头改第一条。"""
        cfgs = self._selected_cfgs()
        if not cfgs:
            self._on_failed("先选中一条策略")
            return None
        if len(cfgs) > 1:
            self._on_failed(f"选中了 {len(cfgs)} 条。这个动作一次只作用于一条，先只选一条")
            return None
        return cfgs[0]

    def _toggle_selected(self) -> None:
        cfg = self._single_or_warn()
        if cfg is None:
            return
        on = not cfg.enabled
        cfg.enabled = on
        SStore.update_portfolio(cfg)
        key = self._account_of(cfg)
        if self.ctl.running:
            if not self.ctl.enable(key, cfg.sid, on):
                self._on_failed(f"账户「{key}」没在运行，改完要重新启动才生效")
        else:
            self._on_log(f"{cfg.title()}：{'标记为运行' if on else '标记为停止'}，启动后生效")
        self._fill_strategies()
        self._render()

    # ---------------------------------------------------------- 增删改

    def _add_from_library(self) -> None:
        lib = SStore.load_library()
        if not lib:
            self._on_failed("策略库是空的。到「策略研究」页点「存入策略库」")
            return
        dlg = LibraryDialog(lib, self)
        if dlg.exec() != QDialog.Accepted:
            return
        sid = dlg.sid()
        src = next((c for c in lib if c.sid == sid), None)
        if src is None:
            return
        item = SStore.add_to_portfolio(src)
        item.account = self.cb_acct.currentData() or ""
        SStore.update_portfolio(item)
        self.reload()
        self._on_log(f"已加入组合：{item.title()}")

    def _edit_selected(self) -> None:
        cfg = self._single_or_warn()
        if cfg is None:
            return
        dlg = EditDialog(cfg, [s.name for s in self._accounts], self)
        if dlg.exec() != QDialog.Accepted:
            return
        old_acct = self._account_of(cfg)
        dlg.apply_to(cfg)
        SStore.update_portfolio(cfg)
        self._fill_strategies()
        self._render()          # 重建表格会把理论/实际重置成「—」，立刻补回来
        if self.ctl.running and self._account_of(cfg) != old_acct:
            self._on_failed("账户已改。运行中的连接不会自动迁移，请停止后重新启动")
        else:
            if self.ctl.running:
                # 手数改了要让引擎立刻知道，否则得等下一根 bar 才按新手数调仓
                self.ctl.update_cfg(self._account_of(cfg), cfg)
            self._on_log(f"{cfg.title()}：{cfg.size_text()}")

    def accept(self, cfg) -> str:
        """从「交易Agent」页送来的策略：加进组合，正在跑就直接挂上去。

        返回一句给人看的结果，调用方拿去显示。**不自动开新连接** —— 连接是用户
        在「启动全部」时明确点出来的，一个对话不该顺手把账户连上。
        """
        item = SStore.add_to_portfolio(cfg)
        item.account = self.cb_acct.currentData() or ""
        item.enabled = True
        SStore.update_portfolio(item)
        self.reload()
        if not self.ctl.running:
            msg = f"已加入组合：{item.title()}。点「启动全部」开始交易"
        else:
            key = self._account_of(item)
            if self.ctl.add(key, item) and self.ctl.enable(key, item.sid, True):
                msg = f"已加入组合并启动：{item.title()}"
            else:
                msg = f"已加入组合，但账户「{key}」的连接不在运行，启动后才会交易"
        self._on_log(msg)
        return msg

    def _del_selected(self) -> None:
        cfg = self._single_or_warn()
        if cfg is None:
            return
        msg = f"从组合里删除「{cfg.title()}」？\n策略库里的模板不受影响。"
        if QMessageBox.question(self, "删除策略", msg) != QMessageBox.Yes:
            return
        if self.ctl.running:
            self.ctl.remove(self._account_of(cfg), cfg.sid)
        SStore.remove_from_portfolio(cfg.sid)
        self.reload()
        self._on_log(f"已删除：{cfg.title()}")

    # ---------------------------------------------------------- 对齐

    def _align(self) -> None:
        """把账户仓位摆成和策略列表完全一致。**会平掉没有策略在管的仓位**。"""
        if not self.ctl.running:
            self._on_failed("还没启动，无法对齐")
            return
        if any(int(snap.get("aligning") or 0) for snap in self._snaps.values()):
            # 上一轮还在路上。仓位没变不等于没对齐 —— 单子还在排队 / 追价。
            # 这时候再发一轮就是「平了又开、开了又平」。
            self._on_failed("上一轮对齐还在执行（等成交或撤单），稍后再点")
            return
        rows = [r for snap in self._snaps.values() for r in (snap.get("mismatch") or [])]
        if not rows:
            self._on_log("账户仓位与策略理论持仓已经一致，不用对齐")
            return

        managed = [r for r in rows if r.get("managed")]
        orphans = [r for r in rows if not r.get("managed")]

        # 平别人的仓是真金白银，不能点一下就干。把「动哪些、动多少」摆出来，
        # 用户点确认才发单
        lines = []
        if managed:
            lines.append(f"按策略理论持仓调整 {len(managed)} 个合约：")
            for r in managed[:10]:
                lines.append(f"　{r['symbol']}　{r['actual']} → {r['target']}"
                             f"（{r['delta']:+d} 手）")
            if len(managed) > 10:
                lines.append(f"　…… 还有 {len(managed) - 10} 个")
        if orphans:
            lines.append("")
            lines.append(f"平掉 {len(orphans)} 个没有策略在管的合约：")
            for r in orphans[:10]:
                lines.append(f"　{r['symbol']}　持有 {r['actual']:+d} 手 → 0")
            if len(orphans) > 10:
                lines.append(f"　…… 还有 {len(orphans) - 10} 个")
            lines.append("")
            lines.append("这些仓位多半是策略停掉之后留下的，或者手工开的。"
                         "平掉之后无法撤销。")

        ans = QMessageBox.question(
            self, "一键对齐", "\n".join(lines) + "\n\n继续下发委托？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if ans != QMessageBox.Yes:
            return

        n = 0
        for key, snap in self._snaps.items():
            if snap.get("mismatch"):
                self.ctl.align(key)
                n += len(snap["mismatch"])
        self._on_log(f"已下发对齐委托，覆盖 {n} 个合约；剩下交给追单，成交前不用再点")

    # ---------------------------------------------------------- 交易助手

    def _load_chase_widgets(self) -> None:
        c = self._chase
        for w in (self.chk_chase, self.chk_market):
            w.blockSignals(True)
        for w in (self.sp_wait, self.sp_retry, self.sp_ticks):
            w.blockSignals(True)
        self.chk_chase.setChecked(c.enabled)
        self.chk_market.setChecked(c.use_market)
        self.sp_wait.setValue(max(1, _int(c.wait_sec) or 5))
        self.sp_retry.setValue(max(0, _int(c.retries)))
        self.sp_ticks.setValue(max(0, _int(c.ticks)))
        for w in (self.chk_chase, self.chk_market,
                  self.sp_wait, self.sp_retry, self.sp_ticks):
            w.blockSignals(False)

    def _save_chase(self, *_a) -> None:
        cfg = SStore.ChaseConfig(
            enabled=self.chk_chase.isChecked(),
            wait_sec=self.sp_wait.value(),
            retries=self.sp_retry.value(),
            ticks=self.sp_ticks.value(),
            use_market=self.chk_market.isChecked(),
        )
        self._chase = cfg
        SStore.save_chase(cfg)
        self.ctl.set_chase(cfg)
        self._chase_hint(cfg)

    def _chase_hint(self, cfg) -> None:
        """把设置翻成一句人话。数字好填，但「到底会怎么追」不好脑补。"""
        if not cfg.enabled:
            text = "已关闭：委托不撤不追"
        elif cfg.retries <= 0:
            text = f"等 {cfg.wait_sec} 秒不成交就撤单，不重报"
        else:
            tail = "，最后一次用市价" if cfg.use_market else ""
            text = (
                f"等 {cfg.wait_sec} 秒不成交就撤单重报，"
                f"最多 {cfg.retries} 次，每次让 {cfg.ticks} 跳{tail}"
            )
        self.lbl_chase.setText(text)

    # ---------------------------------------------------------- 消息

    def _on_log(self, msg: str) -> None:
        self.lbl_msg.setText(msg)
        self.lbl_msg.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")

    def _on_failed(self, msg: str) -> None:
        self.lbl_msg.setText(msg)
        self.lbl_msg.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")

    # ---------------------------------------------------------- 生命周期

    def shutdown(self) -> None:
        """窗口关闭时调，等线程真的退出再返回，避免解释器退出时崩在 TqApi 上。"""
        self.ctl.stop(wait=True)
