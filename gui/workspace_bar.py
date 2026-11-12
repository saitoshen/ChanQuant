"""组合工作区的工具条与分配预览框。

拆成单独一个模块，不是因为它大，是因为 `portfolio_page.py` 已经两千多行，
再往里塞两个对话框，改一次要在里面翻半天。这里只放**纯界面**：不读盘、不算
分配、不碰引擎 —— 那些都在 `ttrade/workspace_store.py` 和 `portfolio_plan.py`。

## 工具条长什么样

```
 默认组合 │ 新组合 │ ＋    [改名] [删除]
 总资金 [100.0 万]  分配方式 [资金等权 ▾]  校准 [按分配额 ▾]   [量一下占用] [分配预览]
```

分页只是**视图**：切分页不改变任何正在跑的东西，运行按账户合并（见
`workspace_store` 的模块注释）。所以切页签之后不需要重启连接。
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from ttrade import portfolio_plan as PL

C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_WARN = "#b45309"
C_BAD = "#dc2626"


class WorkspaceBar(QWidget):
    """工作区分页 + 资金分配方式的那一条。

    只发信号，不做判断：连「有没有成员」「总资金够不够」都由页面那边看，
    因为它手里才有策略池。这里一旦开始自己算，就会和页面算出两个口径。
    """

    switched = Signal(int)          # 切到第几个工作区
    create = Signal()
    rename = Signal()
    remove = Signal()
    measure = Signal()              # 「量一下占用」
    preview = Signal()              # 「分配预览」
    total_changed = Signal(float)
    mode_changed = Signal(str)
    basis_changed = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        self.tabs = QTabBar()
        self.tabs.setExpanding(False)
        self.tabs.setDrawBase(False)
        self.tabs.setToolTip(
            "每个工作区是一套独立的资金分配方案。同一条策略可以同时出现在\n"
            "两个工作区里，各拿不同的钱。运行的时候按账户合并，不会为了\n"
            "两个工作区去开两条连接抢同一个账户。")
        self.tabs.currentChanged.connect(self.switched.emit)
        lay.addWidget(self.tabs)

        btn_add = QPushButton("＋")
        btn_add.setObjectName("Ghost")
        btn_add.setFixedWidth(30)
        btn_add.setToolTip("新建一个工作区（空的分页，从「从策略库添加」加成员）")
        btn_add.clicked.connect(self.create.emit)
        lay.addWidget(btn_add)

        btn_ren = QPushButton("改名")
        btn_ren.setObjectName("Ghost")
        btn_ren.clicked.connect(self.rename.emit)
        lay.addWidget(btn_ren)

        btn_del = QPushButton("删除工作区")
        btn_del.setObjectName("Ghost")
        btn_del.setToolTip("只删这个工作区，策略本身还在策略池里，别的分页不受影响")
        btn_del.clicked.connect(self.remove.emit)
        lay.addWidget(btn_del)

        lay.addSpacing(10)
        lay.addWidget(QLabel("总资金"))
        self.sp_total = QDoubleSpinBox()
        self.sp_total.setDecimals(1)
        self.sp_total.setRange(0.0, 1e7)          # 单位：万
        self.sp_total.setSingleStep(10.0)
        self.sp_total.setSuffix(" 万")
        self.sp_total.setMinimumWidth(110)
        self.sp_total.setToolTip(
            "这个工作区的总资金。0 = 用成员本金之和（老组合迁移过来就是这个状态）。\n"
            "分配方式按它把钱包分下去。")
        self.sp_total.valueChanged.connect(
            lambda v: self.total_changed.emit(float(v) * 10000.0))
        lay.addWidget(self.sp_total)

        lay.addWidget(QLabel("分配方式"))
        self.cb_mode = QComboBox()
        for k, v in PL.MODES.items():
            self.cb_mode.addItem(v, k)
        self.cb_mode.setToolTip(
            "资金等权：每条策略分到总资金的 1/N。\n"
            "风险平价：按各自净值波动的**相关性**做等风险贡献（ERC），\n"
            "　　　　　波动大、跟别人同涨同跌的分得少。\n"
            "手动设置：按你填的金额比例分。")
        self.cb_mode.currentIndexChanged.connect(
            lambda _i: self.mode_changed.emit(str(self.cb_mode.currentData() or "equal")))
        lay.addWidget(self.cb_mode)

        lay.addWidget(QLabel("校准"))
        self.cb_basis = QComboBox()
        for k, v in PL.BASES.items():
            self.cb_basis.addItem(v, k)
        self.cb_basis.setToolTip(
            "决定「分到多少钱」怎么换算成策略的本金。\n"
            "按分配额：分到 20 万就设成本金 20 万。可预期，本金合计恰好等于总资金。\n"
            "按平均占用：让它的**平均保证金占用**等于分到的那份。\n"
            "按峰值占用：让它的**峰值占用**等于分到的那份（更保守）。\n\n"
            "⚠ 后两种会把本金放大几十到几百倍 —— 日内策略大部分时间空仓，\n"
            "平均占用往往只有本金的 1%。一定要先「量一下占用」再点。")
        self.cb_basis.currentIndexChanged.connect(
            lambda _i: self.basis_changed.emit(str(self.cb_basis.currentData() or "principal")))
        lay.addWidget(self.cb_basis)

        self.btn_measure = QPushButton("量一下占用")
        self.btn_measure.setObjectName("Ghost")
        self.btn_measure.setToolTip(
            "把本工作区每条策略跑一遍历史回测，量出它**实际**压了多少钱\n"
            "（|手数| × 收盘 × 乘数 × 保证金率，逐根取均值）。\n\n"
            "和「分配了多少本金」是两回事：capital_pct=30% 只说开仓那一刻\n"
            "用 30%，日内策略大部分时间空仓，实测平均占用常只有本金的 1~3%。")
        self.btn_measure.clicked.connect(self.measure.emit)
        lay.addWidget(self.btn_measure)

        self.btn_preview = QPushButton("分配预览")
        self.btn_preview.setToolTip("按当前分配方式算出每条策略该拿多少钱，看一眼再决定要不要应用")
        self.btn_preview.clicked.connect(self.preview.emit)
        lay.addWidget(self.btn_preview)

        lay.addStretch(1)
        self.lbl_hint = QLabel("")
        self.lbl_hint.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
        lay.addWidget(self.lbl_hint)

    # ---- 读写 ----
    def set_workspaces(self, names: list[str], index: int) -> None:
        """重建页签。**用 blockSignals 挡住 currentChanged**，否则重建一次
        就会把「当前工作区」改掉，用户切页回来发现停在别处。"""
        self.tabs.blockSignals(True)
        try:
            while self.tabs.count():
                self.tabs.removeTab(0)
            for n in names:
                self.tabs.addTab(n)
            if names:
                self.tabs.setCurrentIndex(max(0, min(int(index), len(names) - 1)))
        finally:
            self.tabs.blockSignals(False)

    def set_state(self, total: float, mode: str, basis: str) -> None:
        for w, v in ((self.sp_total, float(total or 0) / 10000.0),):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        for cb, val in ((self.cb_mode, mode), (self.cb_basis, basis)):
            cb.blockSignals(True)
            i = cb.findData(val)
            cb.setCurrentIndex(i if i >= 0 else 0)
            cb.blockSignals(False)

    def hint(self, text: str, bad: bool = False) -> None:
        self.lbl_hint.setText(text)
        self.lbl_hint.setStyleSheet(
            f"color: {C_BAD if bad else C_MUTED}; font-size: 11px;")


class AllocDialog(QDialog):
    """分配预览：权重 / 目标 / 实测占用 / 缩放 / 本金前后。

    只负责「让人看清楚再点头」。`shares` 是外面算好的 `PL.Share` 列表，
    这里一个数都不改 —— 改数的地方只有一个（`apply_shares`），
    两处都能改的话，看到的和应用的就未必是同一份。
    """

    def __init__(self, shares: list, total: float, basis: str = "principal",
                 measured: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("资金分配预览")
        self.resize(1000, 520)
        self.shares = list(shares)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        head = QLabel(
            f"总资金 {total / 10000:,.1f} 万　·　{PL.BASES.get(basis, basis)}"
            f"　·　{len(self.shares)} 条策略")
        head.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        lay.addWidget(head)

        cols = ["策略", "权重", "目标平均占用", "实测平均占用", "缩放",
                "本金(前)", "本金(后)", "预计平均占用", "说明"]
        t = QTableWidget(len(self.shares), len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setShowGrid(False)
        h = t.horizontalHeader()
        for i in range(len(cols)):
            h.setSectionResizeMode(
                i, QHeaderView.Stretch if i in (0, 8) else QHeaderView.ResizeToContents)
        for i, s in enumerate(self.shares):
            vals = [
                s.name, f"{s.weight * 100:.1f}%",
                f"{s.target:,.0f}", f"{s.avg_margin:,.0f}",
                f"{s.scale:.2f}x", f"{s.old_principal:,.0f}",
                f"{s.new_principal:,.0f}", f"{s.est_avg:,.0f}", s.note,
            ]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j in (1, 2, 3, 4, 5, 6, 7):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 8 and s.note:
                    it.setForeground(Qt.GlobalColor.darkYellow)
                t.setItem(i, j, it)
        lay.addWidget(t, 1)

        # ---- 告警 ----
        # 这几条都是「不报错但会出事」：本金合计远超总资金是最容易踩的，
        # 用户看到 3000 万本金会以为程序算错了，其实是「按占用」口径的正常结果
        for note, bad in self._warns(total, basis, measured):
            lb = QLabel(note)
            lb.setWordWrap(True)
            lb.setStyleSheet(
                f"color: {C_BAD if bad else C_WARN}; font-size: 11px;")
            lay.addWidget(lb)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("应用分配")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def _warns(self, total: float, basis: str, measured: bool):
        out = []
        if not self.shares:
            return [("这个工作区还没有成员，先在策略列表「从策略库添加」", True)]
        after = sum(s.new_principal for s in self.shares)
        if total > 0 and after > total * 1.02:
            out.append((
                f"本金合计 {after / 10000:,.0f} 万 = 总资金的 {after / total:.1f} 倍。"
                "这是「按占用 / 按峰值」校准的正常结果，不是算错 —— 平均占用只有"
                "本金的 1~3%，对齐它就得把本金放大几十倍。确认要这么配再点应用。", False))
        fx = [s.name for s in self.shares if "固定手数" in (s.note or "")]
        if fx:
            out.append((f"有 {len(fx)} 条是固定手数（{'、'.join(fx[:3])}"
                        + ("…" if len(fx) > 3 else "")
                        + "）：改本金不会改变手数，这几条没法按组合调仓。", False))
        if str(basis) != "principal" and not measured:
            out.append(("还没量过占用，校准口径退回了「按分配额」。"
                        "先点「量一下占用」，再来看这一页。", True))
        return out
