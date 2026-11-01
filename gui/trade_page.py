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


def _num(v, default=None):
    """把天勤的数值字段转成 float，NaN / None / 非数字都归到 default。

    tqsdk 用 NaN 表示「这一侧没有持仓」—— 比如只有空头时 `open_price_long`
    就是 NaN。NaN 是 truthy，所以 `a or b` 取不到 b，`f"{nan:.2f}"` 会直接把
    "nan" 打到表里（持仓表原来就这么显示的）。
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _money(v) -> str:
    f = _num(v)
    return "—" if f is None else f"{f:,.2f}"


def _pnl(v) -> str:
    f = _num(v)
    return "—" if f is None else f"{f:+,.2f}"


def _avg_price(r: dict) -> str:
    """开仓均价。锁仓时两边都有，按净头寸取；只有一边就取那一边。"""
    pl = _num(r.get("open_price_long"))
    ps = _num(r.get("open_price_short"))
    if r["pos_long"] and r["pos_short"]:
        v = pl if r["pos"] >= 0 else ps
    elif r["pos_long"]:
        v = pl
    else:
        v = ps
    return "—" if v is None else f"{v:.2f}"


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
            "股票模拟（TqSimStock / TqKqStock）用同一个天勤账户，但**数量单位是股**"
            "（100 股起、必须整百），方向只有买入 / 卖出，T+1 当天买的当天不能卖。\n"
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
        self._sig: dict[str, object] = {}     # 每张表上次画的内容，没变就不重画
        self._accounts: list[AccountSpec] = tqlive.load_accounts()
        # 下单面板要不要按「股票口径」显示（股数 / 只有买卖 / 没有平今）。
        # 判据是**勾选的账户**：全是股票账户才算，混着连的时候按期货走 ——
        # 一单同时下到期货和股票账户本来就不成立
        self._stock_mode = False
        self._build()
        # 默认全勾：以前是「下拉框选中谁就连谁」，现在多个账户并存，全勾等于
        # 保持原来的行为；用户想只连一个，取消其余勾选即可
        self._checked = {s.name for s in self._accounts}
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

        # 账户列表摆在资金卡和持仓之间：上面是「一共多少钱」，中间是「钱分在
        # 哪几个户里」，下面是「这些户里有什么仓」。反过来先看持仓再找归属，
        # 多账户时每次都要在心里做一次分组。
        root.addWidget(self._wrap("账户", self._build_accounts(), with_btn="账户", hint=(
            "勾选决定连接哪些账户、上面资金卡和下面三张表统计哪些账户。"
            "双击某一行 = 只看这一个。"
        )), 0)

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

    def _wrap(self, title: str, table: QTableWidget, with_btn: str | None,
              hint: str = "") -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)
        head = QHBoxLayout()
        t = QLabel(title)
        t.setObjectName("SectionTitle")
        head.addWidget(t)
        if hint:
            lbl = QLabel(hint)
            lbl.setObjectName("Hint")
            head.addWidget(lbl)
        head.addStretch(1)
        if with_btn == "撤单":
            btn = QPushButton(with_btn)
            btn.setObjectName("Ghost")
            btn.clicked.connect(self._cancel_selected)
            head.addWidget(btn)
        elif with_btn == "账户":
            for text, tip, slot in (
                ("＋", "新增账户", self._add_account),
                ("－", "删除选中账户", self._del_account),
                ("全选", "全部勾选", lambda: self._check_all(True)),
                ("全不选", "全部取消勾选", lambda: self._check_all(False)),
            ):
                b = QPushButton(text)
                b.setObjectName("Ghost")
                b.setToolTip(tip)
                if text in ("＋", "－"):
                    b.setFixedWidth(32)
                b.clicked.connect(slot)
                head.addWidget(b)
        lay.addLayout(head)
        lay.addWidget(table, 1)
        return card

    def _table(self, headers: list[str], check: bool = False) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setShowGrid(False)
        t.setAlternatingRowColors(False)
        h = t.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch if not check else QHeaderView.Interactive)
        # **不要用 `ResizeToContents`**。它对每一次 `setItem` 都重算全部行的高度
        # 与宽度，而我们是整表重建（一天几百条委托）：同样的 200 行，第一次填
        # 28ms，第二次 12 秒 —— 挂上一笔新委托正好触发第二次，界面直接冻住。
        # 改成固定模式 + 填完调一次 `_fit_columns()`，同一场景 30ms。
        for i in range(1, len(headers)):
            h.setSectionResizeMode(i, QHeaderView.Interactive)
            t.setColumnWidth(i, 90)
        if check:
            t.setColumnWidth(0, 150)
        return t

    def _fit_columns(self, t: QTableWidget, skip: int = 1) -> None:
        """填完之后按内容定一次列宽。

        第 0 列是 Stretch（合约名占满剩余），不参与；其余列各算一次，
        200 行 8 列实测 30ms 上下。带勾选框的账户表第 0 列也不是 Stretch，
        用 `skip=0` 让它一起算。
        """
        for j in range(skip, t.columnCount()):
            t.resizeColumnToContents(j)

    def _build_accounts(self) -> QTableWidget:
        self.tbl_accts = self._table(
            ["账户", "账户平台", "资金账户", "最新权益", "当日盈亏", "净杠杆",
             "总杠杆", "可用资金", "成交金额", "持仓保证金", "手续费"],
            check=True,
        )
        self.tbl_accts.setMaximumHeight(150)
        self.tbl_accts.setToolTip(
            "净杠杆 = |多头市值 − 空头市值| / 权益；总杠杆 =（多头 + 空头市值）/ 权益。\n"
            "市值按「开仓成本 × 合约乘数 + 浮盈」推算，不订阅盘口。"
        )
        # **勾选状态存在 Python 集合里，不存在表格里**：表格每 0.4 秒刷一次数字，
        # 重建 item 会把勾选状态清掉，而「清掉 → itemChanged → 重算显示」正是
        # 上次界面卡死的那个环。这里只在第一次建行时给 item 打勾，之后只改文字。
        self._checked: set[str] = set()
        self._filling = False
        self.tbl_accts.itemChanged.connect(self._on_acct_check)
        self.tbl_accts.itemDoubleClicked.connect(self._on_acct_dbl)
        return self.tbl_accts

    def _build_positions(self) -> QTableWidget:
        self.tbl_pos = self._table(
            ["资金账户", "合约", "净", "多", "空", "今多", "今空",
             "开仓均价", "浮动盈亏", "保证金"]
        )
        return self.tbl_pos

    def _build_orders(self) -> QTableWidget:
        self.tbl_orders = self._table(
            ["资金账户", "时间", "合约", "方向", "开平", "委托", "剩余", "价格",
             "状态", "说明"]
        )
        return self.tbl_orders

    def _build_trades(self) -> QTableWidget:
        self.tbl_trades = self._table(
            ["资金账户", "时间", "合约", "方向", "开平", "价格", "手数"]
        )
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

        self.lbl_vol = QLabel("手数")
        lay.addWidget(self.lbl_vol)
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
        """按配置重建账户表。**只在配置变了时调用**，不是每个行情 tick。

        每次 `on_state` 只改单元格文字（`_fill_accounts`），不重建 item ——
        重建会把勾选状态和当前选中行一起清掉，而清掉之后的 itemChanged 又会
        触发一次显示刷新，很容易转成一个死循环把界面卡住。
        """
        self._filling = True
        t = self.tbl_accts
        t.setRowCount(len(self._accounts))
        for i, s in enumerate(self._accounts):
            it = QTableWidgetItem(s.name)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if s.name in self._checked else Qt.Unchecked)
            it.setData(Qt.UserRole, s.name)
            t.setItem(i, 0, it)
            # 没连上之前先按配置把静态列填好，不至于整行空白
            t.setItem(i, 1, QTableWidgetItem(dict(ACCOUNT_KINDS).get(s.kind, s.kind)))
            t.setItem(i, 2, QTableWidgetItem(s.account_id or s.user or "—"))
            for j in range(3, t.columnCount()):
                t.setItem(i, j, QTableWidgetItem("—"))
        self._filling = False
        self._fit_columns(t, skip=0)
        self._sync_stock_mode()

    def _sync_stock_mode(self) -> None:
        """按勾选的账户决定下单面板是「手」还是「股」。

        股票账户：数量单位是股（100 股起、整百），开平由方向决定（买=开仓、
        卖=平仓），没有平今（T+1）。这些不是界面上的修饰，是交易所规则 ——
        界面上写「手」而实际下的是股，用户会按 100 倍下错单。
        """
        sel = self._checked_specs() or self._accounts
        stock = bool(sel) and all(s.is_stock for s in sel)
        self._stock_mode = stock
        if not hasattr(self, "lbl_vol"):
            return
        self.lbl_vol.setText("股数" if stock else "手数")
        self.sp_vol.blockSignals(True)
        if stock:
            self.sp_vol.setRange(100, 9_999_900)
            self.sp_vol.setSingleStep(100)
            if self.sp_vol.value() < 100:
                self.sp_vol.setValue(100)
            self.cb_offset.setEnabled(False)
            self.cb_offset.setToolTip(
                "股票按方向定开平：买入=开仓、卖出=平仓。T+1，当天买的不能当天卖")
            self.cb_contract.setToolTip("A 股代码，如 SSE.600000 / SZSE.000001")
        else:
            self.sp_vol.setRange(1, 9999)
            self.sp_vol.setSingleStep(1)
            self.cb_offset.setEnabled(True)
            self.cb_offset.setToolTip("")
            self.cb_contract.setToolTip("可手动输入，如 SHFE.rb2601")
        self.sp_vol.blockSignals(False)

    def _checked_specs(self) -> list[AccountSpec]:
        return [s for s in self._accounts if s.name in self._checked]

    def _current_spec(self) -> AccountSpec | None:
        """当前行对应的账户；没有选中行时退回勾选的第一个。"""
        row = self.tbl_accts.currentRow()
        if 0 <= row < len(self._accounts):
            return self._accounts[row]
        sel = self._checked_specs()
        return sel[0] if sel else (self._accounts[0] if self._accounts else None)

    def _on_acct_check(self, item: QTableWidgetItem) -> None:
        """勾选变化：只更新集合，不碰表格（表格由下一次 `on_state` 刷）。"""
        if self._filling or item.column() != 0:
            return
        name = item.data(Qt.UserRole) or item.text()
        if item.checkState() == Qt.Checked:
            self._checked.add(name)
        else:
            self._checked.discard(name)
        # 勾选可能把「期货 + 股票」的混合变成纯股票（或反过来），
        # 下单面板的单位和开平要跟着变
        self._sync_stock_mode()
        self._refresh_view()

    def _on_acct_dbl(self, item: QTableWidgetItem) -> None:
        """双击 = 只看这一个（等价于「单选」）。"""
        name = self.tbl_accts.item(item.row(), 0).data(Qt.UserRole) or item.text()
        self._checked = {name}
        self._filling = True
        for i in range(self.tbl_accts.rowCount()):
            it = self.tbl_accts.item(i, 0)
            if it is not None:
                it.setCheckState(Qt.Checked if i == item.row() else Qt.Unchecked)
        self._filling = False
        self._sync_stock_mode()
        self._refresh_view()

    def _check_all(self, on: bool) -> None:
        self._checked = {s.name for s in self._accounts} if on else set()
        self._filling = True
        for i in range(self.tbl_accts.rowCount()):
            it = self.tbl_accts.item(i, 0)
            if it is not None:
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._filling = False
        self._refresh_view()

    def _refresh_view(self) -> None:
        """勾选变了就地重画一次，不用等下一个 0.4 秒的行情快照。"""
        if self._state is not None:
            self.on_state(self._state, force=True)

    def _add_account(self) -> None:
        dlg = AccountDialog(parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        spec = dlg.spec()
        self._accounts = [s for s in self._accounts if s.name != spec.name] + [spec]
        self._checked.add(spec.name)        # 新加的默认勾上，省一步
        tqlive.save_accounts(self._accounts)
        self._reload_accounts()
        self._select_row(spec.name)

    def _del_account(self) -> None:
        spec = self._current_spec()
        if spec is None:
            return
        if QMessageBox.question(self, "删除账户", f"删除「{spec.name}」？") != QMessageBox.Yes:
            return
        self._accounts = [s for s in self._accounts if s.name != spec.name]
        self._checked.discard(spec.name)
        tqlive.save_accounts(self._accounts)
        self._reload_accounts()

    def _select_row(self, name: str) -> None:
        for i, s in enumerate(self._accounts):
            if s.name == name:
                self.tbl_accts.selectRow(i)
                return

    def _toggle_connect(self) -> None:
        if self.live.running:
            self.disconnect_requested.emit()
            return
        specs = self._checked_specs()
        if not specs:
            self.on_failed("先勾选至少一个账户；没有账户就点「＋」添加")
            return
        self.connect_requested.emit(specs)

    def _on_started(self, key: str) -> None:
        self.btn_connect.setText("断开")
        self.lbl_dot.setStyleSheet(f"color: {C_OK}; font-size: 15px;")
        n = len(self._checked_specs())
        self.lbl_conn.setText(f"已连接 · {n} 个账户" if n > 1 else
                              f"已连接 · {self._current_spec().label() if self._current_spec() else ''}")
        self.lbl_conn.setStyleSheet(f"color: {C_OK}; font-size: 11px;")

    def _on_stopped(self) -> None:
        self.btn_connect.setText("连接")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        self.lbl_conn.setText("未连接")
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        # 断开后把数字清掉：留着上一次的权益和持仓，看着像还连着
        self._sig.clear()
        self._reload_accounts()
        self._state = LiveState()
        self.on_state(self._state, force=True)

    # ---------------------------------------------------------- 渲染
    def on_state(self, st: LiveState, force: bool = False) -> None:
        self._state = st
        self._fill_accounts(st.accounts or [])

        # 只统计勾选的那几个账户。一个都没勾时按「全部」显示 —— 空勾选如果
        # 就什么都不显示，用户会以为连接断了。
        picked = self._checked
        detail = st.by_account or {}
        if picked and detail:
            rows = [detail[n] for n in detail if n in picked]
        else:
            rows = list(detail.values())
        a = self._sum(rows) if rows else (st.account or {})
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

        # 按账户摊平成三张表，顺手补上账户名。快照里本来就带，但测试桩 / 老
        # 版本快照可能没有 —— 缺了会撤单撤到「空账户」上（天勤静默忽略，单子
        # 一直挂着），所以这里补一道。
        pos, ords, trds = [], [], []
        for name, d in detail.items():
            if picked and name not in picked:
                continue
            for bucket, rows_out in (("positions", pos), ("orders", ords), ("trades", trds)):
                for r in d.get(bucket) or []:
                    if not r.get("account"):
                        r["account"] = name
                    rows_out.append(r)
        pos.sort(key=lambda r: (r.get("account", ""), r.get("symbol", "")))
        ords.sort(key=lambda r: (r.get("status") != "ALIVE", -(r.get("insert_date_time") or 0)))
        trds.sort(key=lambda r: -(r.get("trade_date_time") or 0))

        # 表格每 0.4 秒重建一次，一天几百条委托时一次要几百毫秒 —— 切到本页
        # 第一帧就得摆几千个单元格，看着就是「一切过来就卡住」。内容没变就跳过。
        self._fill_if_changed(
            "pos", pos, self._fill_positions,
            tuple((r.get("account"), r.get("symbol"), r.get("pos"),
                   r.get("pos_long_today"), r.get("pos_short_today"),
                   round(r.get("float_profit") or 0, 2), round(r.get("margin") or 0, 2))
                  for r in pos),
            force,
        )
        self._fill_if_changed(
            "ord", ords, self._fill_orders,
            tuple((r.get("account"), r.get("order_id"), r.get("status"),
                   r.get("volume_left"), r.get("last_msg")) for r in ords),
            force,
        )
        self._fill_if_changed(
            "trd", trds, self._fill_trades,
            tuple((r.get("account"), r.get("trade_id")) for r in trds),
            force,
        )

    def _sum(self, rows: list[dict]) -> dict:
        """勾选账户的合计。风险度是比值，不能加，得重算。"""
        out = {}
        for k in ("balance", "available", "margin", "float_profit", "close_profit",
                  "commission", "turnover", "day_profit"):
            out[k] = sum(float((r.get("account") or {}).get(k) or 0.0) for r in rows)
        bal = out["balance"]
        out["risk_ratio"] = (out["margin"] / bal) if bal > 0 else 0.0
        return out

    def _fill_accounts(self, rows: list[dict]) -> None:
        """把实时数字写进账户表。**只改文字，不重建 item**。

        重建 item 会连带清掉勾选状态和当前行，而 `itemChanged` 又接着触发一次
        重画 —— 上次委托框卡死就是这个环。这里行数变了才补行，其余全是 setText。
        """
        live = {r["name"]: r for r in rows}
        if self.tbl_accts.rowCount() != len(self._accounts):
            self._reload_accounts()
        self._filling = True
        for i, s in enumerate(self._accounts):
            r = live.get(s.name)
            if r is None:
                continue
            vals = [
                (1, r.get("platform") or ""),
                (2, r.get("account_id") or ""),
                (3, _money(r.get("balance"))),
                (4, _pnl(r.get("day_profit"))),
                (5, f"{float(r.get('net_leverage') or 0):.2f}"),
                (6, f"{float(r.get('gross_leverage') or 0):.2f}"),
                (7, _money(r.get("available"))),
                (8, _money(r.get("turnover"))),
                (9, _money(r.get("margin"))),
                (10, _money(r.get("commission"))),
            ]
            for j, v in vals:
                it = self.tbl_accts.item(i, j)
                if it is None:
                    it = QTableWidgetItem(v)
                    self.tbl_accts.setItem(i, j, it)
                elif it.text() != v:
                    it.setText(v)
                if j in (3, 4, 5, 6, 7, 8, 9, 10):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 4:
                    it.setForeground(QColor(_color(r.get("day_profit"))))
        self._filling = False

    def _fill_if_changed(self, name: str, rows: list[dict], fill, sig,
                         force: bool = False) -> None:
        if not force and self._sig.get(name) == sig:
            return
        self._sig[name] = sig
        fill(rows)

    def _fill_positions(self, rows: list[dict]) -> None:
        t = self.tbl_pos
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r.get("account", ""), r["symbol"], str(r["pos"]),
                str(r["pos_long"]), str(r["pos_short"]),
                str(r["pos_long_today"]), str(r["pos_short_today"]),
                _avg_price(r),
                _pnl(r["float_profit"]),
                _money(r["margin"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j == 8:
                    item.setForeground(QColor(C_UP if r["float_profit"] > 0 else C_DOWN))
                if j > 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)
        self._fit_columns(t, skip=2)

    def _fill_orders(self, rows: list[dict]) -> None:
        t = self.tbl_orders
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r.get("account", ""), r.get("time", ""),
                r["symbol"], r["direction"], r["offset"],
                str(r["volume_orign"]), str(r["volume_left"]),
                f"{r['limit_price']:.2f}" if r["limit_price"] else "市价",
                "未成交" if r["status"] == "ALIVE" else "已结束",
                r["last_msg"] or "",
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setData(Qt.UserRole, r["order_id"])
                item.setData(Qt.UserRole + 1, r.get("account", ""))
                if j in (5, 6, 7):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)
        self._fit_columns(t, skip=2)

    def _fill_trades(self, rows: list[dict]) -> None:
        t = self.tbl_trades
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            vals = [
                r.get("account", ""), r.get("time", ""),
                r["symbol"], r["direction"], r["offset"],
                f"{r['price']:.2f}", str(r["volume"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j in (5, 6):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)
        self._fit_columns(t, skip=2)

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
        # 多账户下撤单也要指明账户，否则天勤会静默忽略（单子一直挂着）
        self.live.cancel(str(oid), account=str(item.data(Qt.UserRole + 1) or ""))
        self.lbl_order_msg.setText(f"已提交撤单 {oid}")

    def _place(self) -> None:
        contract = self.cb_contract.currentText().strip()
        if not contract:
            self.on_failed("合约不能为空")
            return
        price = None if self.chk_market.isChecked() else self.sp_price.value()
        acc = self._order_account()
        if acc is None:
            self.on_failed("多账户下先在账户列表里点一行，指定下到哪个账户")
            return
        ok = self.live.place(
            contract,
            self.cb_dir.currentData(),
            self.cb_offset.currentData(),
            self.sp_vol.value(),
            price,
            account=acc,
        )
        if ok:
            unit = "股" if self._stock_mode else "手"
            self.lbl_order_msg.setText(
                f"已提交 {self.cb_dir.currentText()} {self.cb_offset.currentText()} "
                f"{contract} {self.sp_vol.value()} {unit}"
                + (f"（{acc}）" if acc else "")
            )

    def _order_account(self) -> str:
        """这一单下到哪个账户。

        只勾了一个就用它；勾了多个则看表格当前选中的那一行 —— 不让人选就
        默认下到「第一个」，是那种事后才发现的错。
        """
        sel = self._checked_specs()
        if len(sel) == 1:
            return sel[0].name
        cur = self._current_spec()
        if cur is not None and (not sel or cur.name in {s.name for s in sel}):
            return cur.name
        return sel[0].name if sel else ""

    def _on_order_sent(self, order_id: str) -> None:
        self.lbl_order_msg.setText(f"下单成功，委托号 {order_id}")

    def on_failed(self, msg: str) -> None:
        self.lbl_order_msg.setText(msg)
        self.lbl_order_msg.setStyleSheet(f"color: {C_BAD}; font-size: 11px;")

    def on_log(self, msg: str) -> None:
        self.lbl_conn.setText(msg)
        self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
