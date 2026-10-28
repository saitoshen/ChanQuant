"""「策略交易和组合」页。

和「策略研究」「账户透视和交易」平级的第三个大页面。它管的是**跑起来之后的事**：
组合里有哪些策略、每个绑哪个账户、现在跑到哪一步、账户仓位和策略理论仓位对不对得上。

页面结构：

    顶部  默认账户 / 启动全部 / 停止全部 / 状态灯
    左侧  策略列表：增删、单独启停、从策略库添加、绑定交易账户
    右侧  监控器：逐合约列出「策略理论 / 账户实际 / 差额」，一键对齐
    底部  交易助手：等几秒撤单重报、追几次、每次追几跳

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

## 一键对齐为什么不碰未托管合约

没有策略在跑的合约，理论持仓恒为 0。如果对齐把它们一起算进去，「一键对齐」就变成了
「把账户清空」—— 策略停掉后留在账户里的仓位、用户手工开的仓，全会被扫掉。
所以 `mismatch()` 给每行打一个 `managed` 标记，`align()` 只补托管合约，
未托管的照常列在监控器里，标灰，让人自己处理。
"""

from __future__ import annotations

import queue
import time

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ttrade import strategy_store as SStore
from ttrade import tqlive
from ttrade.strategy_runner import PortfolioRunner

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

    def set_chase(self, chase) -> None:
        for key in list(self._workers):
            self._submit(key, lambda r, c=chase: setattr(r, "chase", c))


# ------------------------------------------------------------------ 对话框


class LibraryDialog(QDialog):
    """从策略库挑一条加进组合。"""

    def __init__(self, items: list, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("从策略库添加")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)

        self.lst = QListWidget()
        for cfg in items:
            it = QListWidgetItem(f"{cfg.title()}　{cfg.describe()}")
            it.setData(Qt.UserRole, cfg.sid)
            self.lst.addItem(it)
        if items:
            self.lst.setCurrentRow(0)
        lay.addWidget(self.lst, 1)

        hint = QLabel(
            "加入组合会复制一份独立配置，之后在组合里改手数、改账户都不影响库里的模板。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def sid(self) -> str:
        it = self.lst.currentItem()
        return "" if it is None else str(it.data(Qt.UserRole))


class EditDialog(QDialog):
    """改一条组合策略的执行参数。"""

    def __init__(self, cfg, accounts: list[str], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑策略")
        self.setMinimumWidth(380)
        form = QFormLayout(self)

        self.ed_name = QLineEdit(cfg.name)
        self.ed_name.setPlaceholderText(cfg.title())
        form.addRow("名称", self.ed_name)

        self.sp_vol = QSpinBox()
        self.sp_vol.setRange(1, 9999)
        self.sp_vol.setValue(max(1, _int(cfg.volume) or 1))
        form.addRow("手数", self.sp_vol)

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

        lab = QLabel(f"{cfg.key()} {cfg.period}｜{cfg.strategy_label()}")
        lab.setObjectName("Hint")
        form.addRow("策略", lab)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def apply_to(self, cfg) -> None:
        cfg.name = self.ed_name.text().strip()
        cfg.volume = self.sp_vol.value()
        cfg.note = "short" if self.chk_short.isChecked() else ""
        cfg.account = self.cb_acct.currentData() or ""


# ------------------------------------------------------------------ 页面


class PortfolioPage(QWidget):
    """策略交易和组合。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._items = SStore.load_portfolio()
        self._accounts = tqlive.load_accounts()
        self._chase = SStore.load_chase()
        self._snaps: dict[str, dict] = {}
        self._slot_view: dict[str, dict] = {}     # sid -> 快照里的 slot
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
        root.setSpacing(10)
        root.addWidget(self._build_top())

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._build_strategy_card())
        split.addWidget(self._build_monitor_card())
        split.setSizes([760, 460])
        root.addWidget(split, 1)

        root.addWidget(self._build_chase_card())

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

        btn_lib = QPushButton("从策略库添加")
        btn_lib.setObjectName("Ghost")
        btn_lib.clicked.connect(self._add_from_library)
        head.addWidget(btn_lib)

        btn_edit = QPushButton("编辑")
        btn_edit.setObjectName("Ghost")
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
            ["名称", "标的", "策略", "手数", "账户", "理论", "实际", "状态"]
        )
        self.tbl_strat.doubleClicked.connect(lambda _i: self._toggle_selected())
        lay.addWidget(self.tbl_strat, 1)

        hint = QLabel(
            "双击一行 = 启动 / 停止这一条。策略来源是「策略研究」里保存的配置。"
        )
        hint.setObjectName("Hint")
        lay.addWidget(hint)
        return card

    def _build_monitor_card(self) -> QWidget:
        card, lay, head = self._card("监控器")

        btn_align = QPushButton("一键对齐")
        btn_align.setToolTip("把账户仓位补到策略理论持仓")
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
        sel = self._selected_sid()
        t = self.tbl_strat
        t.setRowCount(len(self._items))
        for i, cfg in enumerate(self._items):
            vals = [
                cfg.name.strip() or cfg.title(),
                f"{cfg.key()} {cfg.period}",
                cfg.strategy_label(),
                str(cfg.volume),
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
        if sel:
            for i in range(t.rowCount()):
                it = t.item(i, 0)
                if it is not None and it.data(Qt.UserRole) == sel:
                    t.selectRow(i)
                    break

    def _selected_sid(self) -> str:
        row = self.tbl_strat.currentRow()
        if row < 0:
            return ""
        it = self.tbl_strat.item(row, 0)
        return "" if it is None else str(it.data(Qt.UserRole) or "")

    def _selected_cfg(self):
        sid = self._selected_sid()
        for cfg in self._items:
            if cfg.sid == sid:
                return cfg
        return None

    def _on_snapshot(self, key: str, snap: dict) -> None:
        self._snaps[key] = snap
        for slot in snap.get("slots", []):
            self._slot_view[slot["sid"]] = slot
        self._render()

    def _render(self) -> None:
        t = self.tbl_strat
        for i, cfg in enumerate(self._items):
            slot = self._slot_view.get(cfg.sid) or {}
            if i >= t.rowCount():
                break
            theo = _int(slot.get("target"))
            real = _int(slot.get("actual"))
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
            else:
                text, color, tip = "已停止", C_MUTED, "双击启动"
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

        if not self.ctl.running:
            self.lbl_mon.setText("未启动。策略跑起来之后这里才会比对。")
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
                "，".join(bits) + "。「一键对齐」只补前者。"
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

    def _toggle_selected(self) -> None:
        cfg = self._selected_cfg()
        if cfg is None:
            self._on_failed("先选中一条策略")
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
        cfg = self._selected_cfg()
        if cfg is None:
            self._on_failed("先选中一条策略")
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
            self._on_log(f"{cfg.title()}：{cfg.volume} 手")

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
        cfg = self._selected_cfg()
        if cfg is None:
            self._on_failed("先选中一条策略")
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
        if not self.ctl.running:
            self._on_failed("还没启动，无法对齐")
            return
        rows = [r for snap in self._snaps.values() for r in (snap.get("mismatch") or [])]
        todo = [r for r in rows if r.get("managed")]
        if not todo:
            if rows:
                self._on_log("没有需要对齐的托管合约（列出的都是未托管仓位，请手工处理）")
            else:
                self._on_log("没有需要对齐的合约")
            return
        n = 0
        for key, snap in self._snaps.items():
            if any(r.get("managed") for r in (snap.get("mismatch") or [])):
                self.ctl.align(key)
                n += sum(1 for r in snap["mismatch"] if r.get("managed"))
        self._on_log(f"已下发对齐委托，覆盖 {n} 个合约")

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
