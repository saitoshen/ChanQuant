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
    knob_changed = Signal(str)

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
            "风险平价：按**标准化风险**倒数配权，标准差越大分得越少。\n"
            "　　　　　标准化风险 = 净值年化波动 ÷ 资金利用率 × 回测时长系数\n"
            "　　　　　· 除以利用率：同样 20% 波动，用满 90% 本金的比只用 20% 的\n"
            "　　　　　　更「便宜」（每块钱承担的风险低），所以利用率高的反而拿更多\n"
            f"　　　　　· 盈利因子 < {PL.PF_FLOOR:g} ⇒ 落到最低档（不清零，只拿最小那份）\n"
            "　　　　　· 回测 ≥3 年 ×1.00、2~3 年 ×1.12、1~2 年 ×1.25、半年~1 年 ×1.40；\n"
            "　　　　　　不足半年直接最低档\n"
            "　　　　　· 净值亏穿的按「组合典型风险」配，不再额外降档\n"
            "　　　　　　（亏穿多半是杠杆打太高，不是策略本身风险大）\n"
            "　　　　　有净值序列时还会按相关性做等风险贡献（ERC），口径与上面一致。\n"
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
            "按平均占用：让它的**平均开仓占用**等于分到的那份。\n"
            "按峰值占用：让它的**峰值占用**等于分到的那份（更保守）。\n\n"
            "⚠ 后两种会把本金放大 2~5 倍（满仓策略也可能不放大）——\n"
            "「仓位 30%」意味着一次出手只压 30% 的钱，平均下来大约 20~40%。\n"
            "先「量一下占用」再点，预览里会写出倍数。")
        self.cb_basis.currentIndexChanged.connect(
            lambda _i: self.basis_changed.emit(str(self.cb_basis.currentData() or "principal")))
        lay.addWidget(self.cb_basis)

        # 校准倍数记在哪个参数上。和「校准」一格一体两面：
        #   「校准」决定**调到多少**，「调整方式」决定**记在哪个旋钮上**。
        lay.addWidget(QLabel("调整"))
        self.cb_knob = QComboBox()
        for k, v in PL.KNOBS.items():
            self.cb_knob.addItem(v, k)
        self.cb_knob.setToolTip(
            "「校准」算出要放大几倍之后，这个倍数记在哪个参数上。\n\n"
            "占用 ≈ 本金 × 占用百分比 × 保证金率，两个旋钮相乘 —— 同样的占用\n"
            "有无数种拆法，区别只在本金会不会膨胀：\n\n"
            "本金归位＋占用百分比补（推荐）：先把本金设成「分到的那份」，\n"
            "　使本金合计 = 总资金，剩下的差异交给占用百分比。\n"
            "只调本金：百分比不动，本金放大 2~8 倍。本金合计会远超总资金。\n"
            "只调占用百分比：本金不动，百分比放大（上限 500%），顶住之后余下转本金。\n\n"
            "只有「按资金分配」的策略，占用百分比才有意义。按止损金额（用 risk_pct）、\n"
            "固定手数（用 volume）、按本金全额 这几种会自动退回「只调本金」。")
        self.cb_knob.currentIndexChanged.connect(
            lambda _i: self.knob_changed.emit(str(self.cb_knob.currentData() or "split")))
        lay.addWidget(self.cb_knob)

        self.btn_measure = QPushButton("量一下占用")
        self.btn_measure.setObjectName("Ghost")
        self.btn_measure.setToolTip(
            "把本工作区每条策略跑一遍历史回测，量出它**开仓时**实际压了多少钱\n"
            "（|手数| × 开仓价 × 乘数 × 保证金率，逐笔取均值）。\n\n"
            "不是逐根 bar 的均值：那个会把空仓时间算进分母，日内策略摊出来只有\n"
            "本金的 1~2%，量的是「摊到每根 bar 用多少」而不是「出手一次用多少」。")
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

    def set_state(self, total: float, mode: str, basis: str,
                  knob: str = "split") -> None:
        for w, v in ((self.sp_total, float(total or 0) / 10000.0),):
            w.blockSignals(True)
            w.setValue(v)
            w.blockSignals(False)
        for cb, val in ((self.cb_mode, mode), (self.cb_basis, basis),
                        (self.cb_knob, knob)):
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
                 measured: bool = False, parent=None, knob: str = "split") -> None:
        super().__init__(parent)
        self.setWindowTitle("资金分配预览")
        self.resize(1420, 560)
        self.shares = list(shares)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        after = sum(float(s.new_principal or 0) for s in self.shares)
        head_txt = (
            f"总资金 {total / 10000:,.1f} 万　·　{PL.BASES.get(basis, basis)}"
            f"　·　{PL.KNOBS.get(knob, knob)}　·　{len(self.shares)} 条策略")
        if after > total * 1.02:
            head_txt += f"　·　本金合计 {after / 10000:,.1f} 万"
        head = QLabel(head_txt)
        head.setStyleSheet("font-size: 13px; font-weight: 600; color: #0f172a;")
        lay.addWidget(head)

        # 「占用%(前/后)」两列不能省：`knob` 调到 pct / split 时，本金可能
        # 一动没动，全部变化都在百分比上 —— 只列本金会让人以为「没生效」
        #
        # 「标准化风险 / 盈利因子 / 回测年数 / 档位」四列是风险平价的账本：
        # 权重 ∝ 1/标准化风险，而标准化风险 = 波动 ÷ 利用率 × 时长系数。
        # 不列出来的话，用户看到「这条波动只有 3% 却只分到 1.2%」根本无从追问。
        cols = ["策略", "权重", "标准化风险", "盈利因子", "回测年数", "档位",
                "目标占用", "实测平均", "实测峰值", "缩放",
                "本金(前)", "本金(后)", "占用%(前)", "占用%(后)",
                "预计占用", "一手保底", "说明"]
        t = QTableWidget(len(self.shares), len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setShowGrid(False)
        h = t.horizontalHeader()
        for i in range(len(cols)):
            h.setSectionResizeMode(
                i, QHeaderView.Stretch if i in (0, 5, 16)
                else QHeaderView.ResizeToContents)
        for i, s in enumerate(self.shares):
            # 「一手保底」列只写**加了多少钱**，长解释留在说明列 —— 这一列要能
            # 一眼扫出「哪几条被抬了、抬了多少」
            if getattr(s, "lifted", False):
                lift_txt = f"↑ {s.lift_extra:,.0f}"
            elif getattr(s, "lot_need", 0) > 0:
                lift_txt = "缺（未抬）"
            else:
                lift_txt = "—"
            vals = [
                s.name, f"{s.weight * 100:.1f}%",
                f"{s.risk_score:.2f}" if s.risk_score > 0 else "—",
                f"{s.pf:.2f}" if s.pf > 0 else "—",
                f"{s.span:.1f}" if s.span > 0 else "—",
                s.tier or "—",
                f"{s.target:,.0f}", f"{s.avg_margin:,.0f}",
                f"{s.peak_margin:,.0f}", f"{s.scale:.2f}x", f"{s.old_principal:,.0f}",
                f"{s.new_principal:,.0f}",
                f"{s.old_pct:.0f}%" if s.old_pct else "—",
                f"{s.new_pct:.0f}%" if s.new_pct else "—",
                f"{s.est_avg:,.0f}", lift_txt, s.note,
            ]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j in (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 16 and s.note:
                    it.setForeground(Qt.GlobalColor.darkYellow)
                if j == 5 and "最低档" in (s.tier or ""):
                    it.setForeground(Qt.GlobalColor.darkYellow)
                if j == 15 and getattr(s, "lifted", False):
                    it.setForeground(Qt.GlobalColor.darkGreen)
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
        extra = sum(float(getattr(s, "lift_extra", 0) or 0) for s in self.shares)
        # 一手保底抬出来的钱要**先扣掉**再判「本金放大几倍」—— 那是两回事：
        # 一个是「按占用校准」的正常放大，一个是「补到够一手」的额外本金。
        # 混在一起说，用户会以为按占用校准也把钱包吹了
        base_after = after - extra
        if total > 0 and base_after > total * 1.02:
            out.append((
                f"按占用校准后的本金合计 {base_after / 10000:,.0f} 万"
                f"（不含一手保底）= 总资金的 {base_after / total:.2f} 倍。"
                "这是「按平均占用 / 按峰值占用」校准的正常结果，不是算错 —— "
                "「仓位 30%」意味着一次出手只压 30% 的钱，要让平均开仓占用对齐，"
                "本金就得放大 2~5 倍（对齐峰值时更高）。确认要这么配再点应用。",
                False))
        # ---- 一手保底 ----
        lifted = [s for s in self.shares if getattr(s, "lifted", False)]
        if lifted:
            names = "、".join(s.name for s in lifted[:3]) + ("…" if len(lifted) > 3 else "")
            out.append((
                f"有 {len(lifted)} 条分到的钱历史上开不出 1 手（{names}），每手大约要 "
                f"{(sum(float(s.lot_need) for s in lifted) / max(len(lifted), 1)) / 10000:,.1f} 万。"
                f"已按「一手保底」上调 +{extra / 10000:,.1f} 万：这部分**不从其他条目扣**，"
                f"应用后总资金会变成 {after / 10000:,.1f} 万（原 {total / 10000:,.1f} 万）。"
                "上调额取「历史平均一手」到「历史最贵一手」之间一半的位置。",
                False))
        # 抬到上限还开不出一手：必须点名。否则这条会以「已保底」的样子继续空跑，
        # 而它的净值、占用、波动率、盈利因子全是 0 —— 配权规则还会再罚它一次
        short = [s for s in self.shares
                 if "仍不足一手" in (getattr(s, "floor_note", "") or "")
                 or "没有可上调的空间" in (getattr(s, "floor_note", "") or "")]
        if short:
            names = "、".join(s.name for s in short[:3]) + ("…" if len(short) > 3 else "")
            out.append((
                f"有 {len(short)} 条即使上调到上限也开不出 1 手（{names}）："
                "它们历史上大部分时间是空仓，净值 / 波动率 / 盈利因子都不可信。"
                "要么给这几条单独开户，要么把它们从组合里去掉 —— 留在组合里配权"
                "只会让它们的权重看起来「有数」其实是零信息。", True))
        fx = [s.name for s in self.shares if "固定手数" in (s.note or "")]
        if fx:
            out.append((f"有 {len(fx)} 条是固定手数（{'、'.join(fx[:3])}"
                        + ("…" if len(fx) > 3 else "")
                        + "）：改本金不会改变手数，这几条没法按组合调仓。", False))
        # 波动率无效 = 风险平价的权重不是由它自己的风险算出来的。不点名的话，
        # 用户只会看到「权重怎么和别的一样 / 为什么这条这么低」，无处可查
        bad = [s.name for s in self.shares if getattr(s, "vol_bad", False)]
        if bad:
            out.append((
                f"有 {len(bad)} 条没量到波动率（{'、'.join(bad[:3])}"
                + ("…" if len(bad) > 3 else "")
                + "）：按「组合典型风险」配权，这几条的权重不是它自己的风险算出来的。"
                "想让它们参与排序，先点「量一下占用」跑一遍回测。",
                False))
        # 亏穿：**不是**「算不出波动率」。波动率照整条曲线（含翻负段）给了真值，
        # 所以权重小是名副其实的 —— 只是要让人知道根因在杠杆
        blew = [s.name for s in self.shares if getattr(s, "blown", False)]
        if blew:
            out.append((
                f"有 {len(blew)} 条净值曾亏穿（{'、'.join(blew[:3])}"
                + ("…" if len(blew) > 3 else "")
                + "）：波动率按整条资金曲线算（含翻负那段），所以这几条分数偏高、"
                "权重偏小是真实的。亏穿的根因是杠杆压太满 —— 杠杆已经通过"
                "「资金利用率」进了分数，这里不再额外降档。想提高权重就降本金或"
                "降占用百分比，让 util 下来。",
                False))
        # 风险平价的账本：权重不是只由波动率定的，不给一句话说明的话，
        # 用户看到「波动 3% 的只分到 1.2%」完全无从追问
        tiers = [s for s in self.shares if (s.tier or "").strip()]
        if tiers:
            low = [s.name for s in tiers if "最低档" in (s.tier or "")]
            msg = ("风险平价按「标准化风险」倒数分配：年化波动 ÷ 资金利用率 × 回测"
                   f"时长系数。利用率高、回测长、盈利因子 ≥{PL.PF_FLOOR:g} 的多拿。"
                   "「标准化风险」列可以直接横向比，越大分得越少。")
            if low:
                msg += (f"　本次落到最低档 {len(low)} 条：{'、'.join(low[:3])}"
                        + ("…" if len(low) > 3 else "")
                        + f"（盈利因子 <{PL.PF_FLOOR:g} 或回测不足半年；"
                          "指到组合里风险最高那一档，不清零）")
            out.append((msg, False))
        if str(basis) != "principal" and not measured:
            out.append(("还没量过占用，校准口径退回了「按分配额」。"
                        "先点「量一下占用」，再来看这一页。", True))
        return out
