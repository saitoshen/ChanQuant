"""「账户透视和交易」页。

和「策略研究」并列的第二个大页面。数据全部来自 `gui.live.LiveController`
转出来的 `LiveState` 快照 —— 这个页面不碰任何天勤对象，只渲染 dict，
所以天勤断线、重连都不会把界面带崩。

页面结构：

    顶部  账户选择 / 连接 / 断开 / 状态灯
    资金  动态权益 · 可用资金 · 占用保证金 · 持仓盈亏 · 平仓盈亏 · 风险度
    中部  持仓表（左） / 委托表 + 成交表（右）
    底部  标准下单面板：合约 / 买卖 / 开平 / 手数 / 价格 / 下单

「未成交单」不单独开一张表 —— 它就是委托表里 `status == "ALIVE"` 的那些，
单独列一遍只会让人怀疑两张表为什么对不上。委托表默认按「还活着的在前、
时间倒序」排，撤单按钮只对选中的活单生效。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ttrade import tqlive
from ttrade.tqlive import ACCOUNT_KINDS, AccountSpec, LiveState

# 与 app.py 的配色保持一致。这里不 import app 的常量：app 会 import 本模块，
# 顶层互相 import 会成环。
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_UP = "#d92b2b"      # 红涨 / 盈利
C_DOWN = "#0f9d58"    # 绿跌 / 亏损
C_OK = "#16a34a"
C_BAD = "#dc2626"
C_IDLE = "#94a3b8"

OFFSETS = [("开仓", "OPEN"), ("平今", "CLOSETODAY"), ("平仓", "CLOSE")]
DIRECTIONS = [("买入", "BUY"), ("卖出", "SELL")]


def _money(v) -> str:
    try:
        return f"{float(v):,.2f}"
    except (TypeError, ValueError):
        return "—"


def _pnl(v) -> str:
    try:
        return f"{float(v):+,.2f}"
    except (TypeError, ValueError):
        return "—"


def _color(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return C_TEXT
    if f > 0:
        return C_UP
    if f < 0:
        return C_DOWN
    return C_TEXT


class _Stat(QFrame):
    """一个资金指标小卡。"""

    def __init__(self, label: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("SubCard")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(1)
        self.lbl = QLabel(label)
        self.lbl.setObjectName("MetricLabel")
        self.val = QLabel("—")
        self.val.setObjectName("MetricValue")
        lay.addWidget(self.lbl)
        lay.addWidget(self.val)

    def set(self, text: str, color: str = C_TEXT) -> None:
        self.val.setText(text)
        self.val.setStyleSheet(f"font-size: 15px; font-weight: 700; color: {color};")


class AccountDialog(QDialog):
    """新增 / 编辑一个账户。"""

    def __init__(self, spec: AccountSpec | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("交易账户")
        self.setMinimumWidth(420)
        form = QFormLayout(self)

        self.ed_name = QLineEdit(spec.name if spec else "快期模拟")
        self.cb_kind = QComboBox()
        for key, label in ACCOUNT_KINDS:
            self.cb_kind.addItem(label, key)
        if spec:
            idx = self.cb_kind.findData(spec.kind)
            self.cb_kind.setCurrentIndex(max(idx, 0))

        self.ed_user = QLineEdit(spec.user if spec else "")
        self.ed_pwd = QLineEdit(spec.password if spec else "")
        self.ed_pwd.setEchoMode(QLineEdit.Password)
        self.ed_broker = QLineEdit(spec.broker_id if spec else "")
        self.ed_acct = QLineEdit(spec.account_id if spec else "")
        self.ed_acct_pwd = QLineEdit(spec.account_password if spec else "")
        self.ed_acct_pwd.setEchoMode(QLineEdit.Password)

        form.addRow("名称", self.ed_name)
        form.addRow("类型", self.cb_kind)
        form.addRow("天勤账户", self.ed_user)
        form.addRow("天勤密码", self.ed_pwd)
        form.addRow("期货公司代码", self.ed_broker)
        form.addRow("资金账号", self.ed_acct)
        form.addRow("资金密码", self.ed_acct_pwd)

        self.hint = QLabel(
            "快期模拟只需要天勤账户；实盘才需要期货公司代码与资金账号。\n"
            "密码明文存在本机用户目录，不会进仓库。"
        )
        self.hint.setObjectName("Hint")
        self.hint.setWordWrap(True)
        form.addRow(self.hint)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        form.addRow(box)

        self.cb_kind.currentIndexChanged.connect(self._sync)
        self._sync()

    def _sync(self) -> None:
        real = self.cb_kind.currentData() == "account"
        for w in (self.ed_broker, self.ed_acct, self.ed_acct_pwd):
            w.setEnabled(real)

    def spec(self) -> AccountSpec:
        return AccountSpec(
            name=self.ed_name.text().strip() or "未命名",
            kind=self.cb_kind.currentData(),
            user=self.ed_user.text().strip(),
            password=self.ed_pwd.text(),
            broker_id=self.ed_broker.text().strip(),
            account_id=self.ed_acct.text().strip(),
            account_password=self.ed_acct_pwd.text(),
        )


class TradePage(QWidget):
    """账户透视 + 下单。"""

    connect_requested = Signal(object)   # AccountSpec
    disconnect_requested = Signal()

    def __init__(self, live, parent=None) -> None:
        super().__init__(parent)
        self.live = live
        self._state = LiveState()
        self._accounts: list[AccountSpec] = tqlive.load_accounts()
        self._build()
        self._reload_accounts()
        live.state.connect(self.on_state)
        live.failed.connect(self.on_failed)
        live.log.connect(self.on_log)
        live.started.connect(self._on_started)
        live.stopped.connect(self._on_stopped)
        live.order_sent.connect(self._on_order_sent)

    # ---------------------------------------------------------- 构建
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        root.addWidget(self._build_top())

        stats = QHBoxLayout()
        stats.setSpacing(8)
        self.stats = {}
        for key, label in (
            ("balance", "动态权益"),
            ("available", "可用资金"),
            ("margin", "占用保证金"),
            ("float_profit", "持仓盈亏"),
            ("close_profit", "平仓盈亏"),
            ("risk_ratio", "风险度"),
        ):
            w = _Stat(label)
            self.stats[key] = w
            stats.addWidget(w, 1)
        root.addLayout(stats)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(self._wrap("持仓", self._build_positions(), with_btn=None))
        right = QSplitter(Qt.Vertical)
        right.addWidget(self._wrap("委托 / 未成交", self._build_orders(), with_btn="撤单"))
        right.addWidget(self._wrap("成交", self._build_trades(), with_btn=None))
        right.setSizes([300, 220])
        split.addWidget(right)
        split.setSizes([560, 620])
        root.addWidget(split, 1)

        root.addWidget(self._build_order_panel())

    def _build_top(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(8)

        lay.addWidget(QLabel("账户"))
        self.cb_account = QComboBox()
        self.cb_account.setMinimumWidth(200)
        lay.addWidget(self.cb_account)

        btn_add = QPushButton("＋")
        btn_add.setObjectName("Ghost")
        btn_add.setFixedWidth(32)
        btn_add.setToolTip("新增账户")
        btn_add.clicked.connect(self._add_account)
        lay.addWidget(btn_add)

        btn_del = QPushButton("－")
        btn_del.setObjectName("Ghost")
        btn_del.setFixedWidth(32)
        btn_del.setToolTip("删除选中账户")
        btn_del.clicked.connect(self._del_account)
        lay.addWidget(btn_del)

        self.btn_connect = QPushButton("连接")
        self.btn_connect.clicked.connect(self._toggle_connect)
        lay.addWidget(self.btn_connect)

        self.lbl_dot = QLabel("●")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        lay.addWidget(self.lbl_dot)
        self.lbl_conn = QLabel("未连接")
        self.lbl_conn.setObjectName("Hint")
        lay.addWidget(self.lbl_conn)

        lay.addStretch(1)
        self.lbl_quote = QLabel("")
        self.lbl_quote.setStyleSheet(f"color: {C_TEXT}; font-size: 13px; font-weight: 600;")
        lay.addWidget(self.lbl_quote)
        return card

    def _wrap(self, title: str, table: QTableWidget, with_btn: str | None) -> QWidget:
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
        if with_btn:
            btn = QPushButton(with_btn)
            btn.setObjectName("Ghost")
            btn.clicked.connect(self._cancel_selected)
            head.addWidget(btn)
        lay.addLayout(head)
        lay.addWidget(table, 1)
        return card

    def _table(self, headers: list[str]) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setShowGrid(False)
        t.setAlternatingRowColors(False)
        h = t.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(headers)):
            h.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        return t

    def _build_positions(self) -> QTableWidget:
        self.tbl_pos = self._table(
            ["合约", "净", "多", "空", "今多", "今空", "开仓均价", "浮动盈亏", "保证金"]
        )
        return self.tbl_pos

    def _build_orders(self) -> QTableWidget:
        self.tbl_orders = self._table(
            ["合约", "方向", "开平", "委托", "剩余", "价格", "状态", "说明"]
        )
        return self.tbl_orders

    def _build_trades(self) -> QTableWidget:
        self.tbl_trades = self._table(["合约", "方向", "开平", "价格", "手数"])
        return self.tbl_trades

    def _build_order_panel(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)

        lay.addWidget(QLabel("合约"))
        self.cb_contract = QComboBox()
        self.cb_contract.setEditable(True)
        self.cb_contract.setMinimumWidth(150)
        self.cb_contract.setToolTip("可手动输入，如 SHFE.rb2601")
        lay.addWidget(self.cb_contract)

        lay.addWidget(QLabel("方向"))
        self.cb_dir = QComboBox()
        for label, key in DIRECTIONS:
            self.cb_dir.addItem(label, key)
        lay.addWidget(self.cb_dir)

        lay.addWidget(QLabel("开平"))
        self.cb_offset = QComboBox()
        for label, key in OFFSETS:
            self.cb_offset.addItem(label, key)
        lay.addWidget(self.cb_offset)

        lay.addWidget(QLabel("手数"))
        self.sp_vol = QSpinBox()
        self.sp_vol.setRange(1, 9999)
        lay.addWidget(self.sp_vol)

        lay.addWidget(QLabel("价格"))
        self.sp_price = QDoubleSpinBox()
        self.sp_price.setDecimals(4)
        self.sp_price.setRange(0, 9_999_999)
        self.sp_price.setSingleStep(1)
        lay.addWidget(self.sp_price)

        self.chk_market = QCheckBox("市价")
        self.chk_market.setToolTip("勾上则不传限价，交给交易所按市价撮合")
        self.chk_market.toggled.connect(lambda on: self.sp_price.setEnabled(not on))
        lay.addWidget(self.chk_market)

        lay.addStretch(1)
        self.btn_order = QPushButton("下单")
        self.btn_order.setMinimumWidth(96)
        self.btn_order.clicked.connect(self._place)
        lay.addWidget(self.btn_order)

        self.lbl_order_msg = QLabel("")
        self.lbl_order_msg.setObjectName("Hint")
        lay.addWidget(self.lbl_order_msg)
        return card

    # ---------------------------------------------------------- 账户
    def _reload_accounts(self) -> None:
        self.cb_account.blockSignals(True)
        self.cb_account.clear()
        for s in self._accounts:
            self.cb_account.addItem(s.label(), s.name)
        self.cb_account.blockSignals(False)

    def _current_spec(self) -> AccountSpec | None:
        name = self.cb_account.currentData()
        for s in self._accounts:
            if s.name == name:
                return s
        return self._accounts[0] if self._accounts else None

    def _add_account(self) -> None:
        dlg = AccountDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        spec = dlg.spec()
        self._accounts = [s for s in self._accounts if s.name != spec.name] + [spec]
        tqlive.save_accounts(self._accounts)
        self._reload_accounts()
        idx = self.cb_account.findData(spec.name)
        if idx >= 0:
            self.cb_account.setCurrentIndex(idx)

    def _del_account(self) -> None:
        spec = self._current_spec()
        if spec is None:
            return
        if QMessageBox.question(self, "删除账户", f"删除「{spec.name}」？") != QMessageBox.Yes:
            return
        self._accounts = [s for s in self._accounts if s.name != spec.name]
        tqlive.save_accounts(self._accounts)
        self._reload_accounts()

    def _toggle_connect(self) -> None:
        if self.live.running:
            self.disconnect_requested.emit()
            return
        spec = self._current_spec()
        if spec is None:
            self.on_failed("还没有配置交易账户，点「＋」添加一个")
            return
        self.connect_requested.emit(spec)

    def _on_started(self, key: str) -> None:
        self.btn_connect.setText("断开")
        self.lbl_dot.setStyleSheet(f"color: {C_OK}; font-size: 15px;")
        self.lbl_conn.setText(f"已连接 · {self._current_spec().label() if self._current_spec() else ''}")
        self.lbl_conn.setStyleSheet(f"color: {C_OK}; font-size: 11px;")

    def _on_stopped(self) -> None:
        self.btn_connect.setText("连接")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        self.lbl_conn.setText("未连接")
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")

    # ---------------------------------------------------------- 渲染
    def on_state(self, st: LiveState) -> None:
        self._state = st
        a = st.account or {}
        for key, card in self.stats.items():
            v = a.get(key)
            if key == "risk_ratio":
                risk = float(v or 0)
                card.set(f"{risk * 100:.1f}%" if v else "—",
                         C_BAD if risk > 0.8 else C_TEXT)
            elif key in ("float_profit", "close_profit"):
                card.set(_pnl(v), _color(v))
            else:
                card.set(_money(v))

        q = st.quote or {}
        if q:
            last = q.get("last") or 0
            self.lbl_quote.setText(
                f"{q.get('name', '')} {q.get('underlying', '')}  "
                f"{last:.2f}  "
                f"买 {q.get('bid', 0):.2f}/{int(q.get('bid_vol', 0))}  "
                f"卖 {q.get('ask', 0):.2f}/{int(q.get('ask_vol', 0))}"
            )
            # 下单面板的合约默认跟当前主力走，但不要覆盖用户手动输入的内容
            und = q.get("underlying")
            if und and self.cb_contract.currentText() in ("", getattr(self, "_auto_contract", "")):
                self.cb_contract.setCurrentText(und)
                self._auto_contract = und
            if q.get("price_tick"):
                self.sp_price.setSingleStep(float(q["price_tick"]))
                if self.sp_price.value() <= 0:
                    self.sp_price.setValue(float(last or 0))

        self._fill_positions(st.positions)
        self._fill_orders(st.orders)
        self._fill_trades(st.trades)

    def _fill_positions(self, rows: list[dict]) -> None:
        t = self.tbl_pos
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r["symbol"], str(r["pos"]), str(r["pos_long"]), str(r["pos_short"]),
                str(r["pos_long_today"]), str(r["pos_short_today"]),
                f"{r['open_price_long'] or r['open_price_short']:.2f}",
                _pnl(r["float_profit"]),
                _money(r["margin"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j == 7:
                    item.setForeground(QColor(C_UP if r["float_profit"] > 0 else C_DOWN))
                if j > 0:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)

    def _fill_orders(self, rows: list[dict]) -> None:
        t = self.tbl_orders
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r["symbol"], r["direction"], r["offset"],
                str(r["volume_orign"]), str(r["volume_left"]),
                f"{r['limit_price']:.2f}" if r["limit_price"] else "市价",
                "未成交" if r["status"] == "ALIVE" else "已结束",
                r["last_msg"] or "",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setData(Qt.UserRole, r["order_id"])
                if j in (3, 4, 5):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)

    def _fill_trades(self, rows: list[dict]) -> None:
        t = self.tbl_trades
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r["symbol"], r["direction"], r["offset"],
                f"{r['price']:.2f}", str(r["volume"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j in (3, 4):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)

    # ---------------------------------------------------------- 操作
    def _cancel_selected(self) -> None:
        row = self.tbl_orders.currentRow()
        if row < 0:
            self.on_failed("先选中一条委托")
            return
        item = self.tbl_orders.item(row, 0)
        if item is None:
            return
        oid = item.data(Qt.UserRole)
        if not oid:
            return
        self.live.cancel(str(oid))
        self.lbl_order_msg.setText(f"已提交撤单 {oid}")

    def _place(self) -> None:
        contract = self.cb_contract.currentText().strip()
        if not contract:
            self.on_failed("合约不能为空")
            return
        price = None if self.chk_market.isChecked() else self.sp_price.value()
        ok = self.live.place(
            contract,
            self.cb_dir.currentData(),
            self.cb_offset.currentData(),
            self.sp_vol.value(),
            price,
        )
        if ok:
            self.lbl_order_msg.setText(
                f"已提交 {self.cb_dir.currentText()} {self.cb_offset.currentText()} "
                f"{contract} {self.sp_vol.value()} 手"
            )

    def _on_order_sent(self, order_id: str) -> None:
        self.lbl_order_msg.setText(f"下单成功，委托号 {order_id}")

    def on_failed(self, msg: str) -> None:
        self.lbl_order_msg.setText(msg)
        self.lbl_order_msg.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")

    def on_log(self, msg: str) -> None:
        self.lbl_conn.setText(msg)
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
