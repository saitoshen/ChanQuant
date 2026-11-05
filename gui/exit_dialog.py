"""因子策略的「出场规则」弹窗。

为什么不铺在侧边栏：这一项本身有四种口径、每种口径各有 2~6 个参数，
平铺出来比整个因子参数区还长。侧边栏已经只剩 300 多像素宽，再塞进去
就只能靠横向滚动 —— 那是上一版刚拆掉的东西。

设计上的三条硬约束：

- **只改用户动过的口径**：切回「分数出场」时把其它参数留在原处，不会因为
  一次开合就把调好的 3 档改回默认
- 参数之间要能互相校验：分档的「回撤幅度」必须随浮盈递减，写反了不会报错，
  只会让规则悄悄变成相反的东西（浮盈越多越松）
- **止损不能塞进单选组**：它是任何口径下都能叠的一条线，做成第五个单选项
  就变成「要么分档回撤、要么止损」的二选一，而这两个恰恰该同时开
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

_HINT = "font-size: 11px; color: #64748b;"

# 三种 ATR 口径的说明。写在单选项下面而不是 tooltip 里：这一屏本来就是
# 「不知道选哪个才点进来」的人在看，悬停提示等于没写
_MODE_TIPS = {
    "score": "分数回落到离场阈值以内就平，不叠加任何 ATR 规则。"
             "上面勾了止损时止损照样生效。",
    "tier": "浮盈按 ATR 折算，分 3 档：浮盈越多，允许的回撤越小。先让利润跑，"
            "跑起来之后收紧 —— 趋势单最怕「赚了 8 个 ATR 又全还回去」。",
    "abs": "从持仓期间的最高浮盈回撤固定 N×ATR 就平，不分档。"
           "比「分档」宽松，相当于一条普通的 ATR 跟踪止损。",
    "box": "持仓期间价格反向穿越「过去 N 根的高低点」就平。"
           "N 决定箱体宽度：N 越大箱体越宽、越不容易被打掉。",
}


class ExitRuleDialog(QDialog):
    """出场规则的设置窗。`values()` 返回可以直接塞进 `FactorConfig` 的 dict。"""

    def __init__(self, cur: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        cur = dict(cur or {})
        self.setWindowTitle("出场规则")
        # 两列：左边是「怎么出场」（止损 + 口径选择），右边是选中口径的参数。
        # 原来是竖着一条，五组说明加起来比屏幕还高，确定按钮被顶到屏幕外 ——
        # 横过来之后每段说明的折行也少了
        self.setMinimumWidth(820)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(10)

        # 内容区可滚动，**按钮放在滚动区外面**：这样即使某天又加了一组参数、
        # 或者笔记本屏幕只有 768 高，确定 / 取消也永远在眼前
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body = QWidget()
        cols = QHBoxLayout(body)
        cols.setContentsMargins(0, 0, 0, 0)
        cols.setSpacing(14)
        scroll.setWidget(body)
        self._scroll = scroll
        self._fitted = False
        lay.addWidget(scroll, 1)

        left = QVBoxLayout()
        left.setSpacing(8)
        right = QVBoxLayout()
        right.setSpacing(8)
        cols.addLayout(left, 5)
        cols.addLayout(right, 4)

        tip = QLabel(
            "出场规则叠在「分数过阈值」的信号之上，只减仓、不反手：平掉之后要等信号"
            "回到 0 / 翻向，或分数重新越过进场阈值才重新开仓。浮盈和回撤都按 ATR "
            "折算（14 根真实波幅均值）。"
        )
        tip.setStyleSheet(_HINT)
        tip.setWordWrap(True)
        left.addWidget(tip)

        # ---- 独立止损
        #
        # 放在口径之前，而且是个独立的勾选框 —— 它不是「第五种口径」，
        # 是任何口径下都能叠的一条线。放进单选组里就变成二选一了，
        # 而「分档回撤 + 止损」恰恰是最常用的一组。
        box_sl = QGroupBox("止损（独立，与上面的口径叠加）")
        sv = QVBoxLayout(box_sl)
        sv.setContentsMargins(10, 8, 10, 8)
        sv.setSpacing(4)
        self.chk_sl = QCheckBox("启用 ATR 止损")
        self.chk_sl.setChecked(bool(cur.get("exit_sl")))
        sv.addWidget(self.chk_sl)
        sl_row = QHBoxLayout()
        sl_row.setContentsMargins(20, 0, 0, 0)
        sl_row.setSpacing(6)
        self.sp_sl = QDoubleSpinBox()
        self.sp_sl.setRange(0.1, 50.0)
        self.sp_sl.setDecimals(2)
        self.sp_sl.setSingleStep(0.25)
        self.sp_sl.setValue(float(cur.get("exit_sl_atr") or 2.0))
        self.sp_sl.setSuffix(" ATR")
        sl_row.addWidget(QLabel("从进场价反向走"))
        sl_row.addWidget(self.sp_sl)
        sl_row.addWidget(QLabel("就平仓"))
        sl_row.addStretch(1)
        sv.addLayout(sl_row)
        sl_hint = QLabel(
            "止损线钉在「进场价」下面（空头在上面）N 个 ATR，和浮盈无关 —— 浮盈到过 "
            "+5ATR，这条线也不会跟着抬。所以它和「绝对回撤」不是一回事：绝对回撤从"
            "最高浮盈往回退（保住利润），止损只管「亏到哪认输」。"
        )
        sl_hint.setStyleSheet(_HINT)
        sl_hint.setWordWrap(True)
        sv.addWidget(sl_hint)
        left.addWidget(box_sl)
        self.chk_sl.toggled.connect(self._sync)

        # ---- 禁止分数出场
        #
        # 和止损一样是独立开关，不是第五种口径：它管的是「要不要让离场阈值
        # 参与」，四种 ATR 口径照旧选。放在口径**之前** —— 先决定分数参不
        # 参与，再决定用哪种 ATR 算法，顺序和思路一致
        box_ns = QGroupBox("分数出场")
        nv = QVBoxLayout(box_ns)
        nv.setContentsMargins(10, 8, 10, 8)
        nv.setSpacing(4)
        self.chk_noscore = QCheckBox("不使用离场阈值出场（分数回落到阈值内不平仓）")
        self.chk_noscore.setChecked(bool(cur.get("exit_no_score")))
        nv.addWidget(self.chk_noscore)
        ns_hint = QLabel(
            "仓位只由下面的 ATR 规则了结。分数反向越过进场阈值时仍然平仓 —— "
            "那是反向信号，不是离场阈值。"
        )
        ns_hint.setStyleSheet(_HINT)
        ns_hint.setWordWrap(True)
        nv.addWidget(ns_hint)
        # 更彻底的一层：反向也不平
        self.chk_noflip = QCheckBox(
            "分数反向越过进场阈值时也不平仓（只按 ATR 出场）")
        self.chk_noflip.setChecked(bool(cur.get("exit_no_flip")))
        nv.addWidget(self.chk_noflip)
        flip_hint = QLabel(
            "给「有浮盈之后尽量不出场」用：胜率降、盈亏比升（截断亏损、让利润奔跑），"
            "也滤掉噪音反向信号。代价是方向只由第一笔决定，之后靠 ATR 退出。"
            "勾上后上面那条自动生效。"
        )
        flip_hint.setStyleSheet(_HINT)
        flip_hint.setWordWrap(True)
        nv.addWidget(flip_hint)
        # 这是「想试更极端口径」才会勾的一条，说明只在勾上时展开 ——
        # 一直挂着三行字，把确定按钮又往下顶一截
        self.lbl_flip_hint = flip_hint
        # 勾了它、口径又是「分数出场」、还没开止损 = 一条出场规则都没有，
        # 仓位会一直拿到回测结束。与其静默忽略这个勾，不如把确定按钮灰掉
        self.lbl_ns_warn = QLabel(
            "这样一条出场规则都没有了（口径是「分数出场」、止损也没开）。"
            "请勾上止损，或者选一个 ATR 口径。"
        )
        self.lbl_ns_warn.setStyleSheet(
            "font-size: 11px; color: #b91c1c; font-weight: 600;")
        self.lbl_ns_warn.setWordWrap(True)
        nv.addWidget(self.lbl_ns_warn)
        left.addWidget(box_ns)
        self.chk_noscore.toggled.connect(self._sync)
        self.chk_noflip.toggled.connect(self._sync)

        mode = str(cur.get("exit_mode") or "score")
        self.rb: dict[str, QRadioButton] = {}
        box_mode = QGroupBox("出场口径")
        mv = QVBoxLayout(box_mode)
        mv.setContentsMargins(10, 8, 10, 8)
        mv.setSpacing(4)
        for key, label in (("score", "分数出场（默认，不叠加 ATR 规则）"),
                           ("tier", "分档浮盈回撤出场"),
                           ("abs", "绝对回撤出场"),
                           ("box", "反向箱体突破出场")):
            rb = QRadioButton(label)
            rb.setChecked(key == mode)
            self.rb[key] = rb
            mv.addWidget(rb)
            h = QLabel(_MODE_TIPS[key])
            h.setStyleSheet(_HINT)
            h.setWordWrap(True)
            h.setContentsMargins(20, 0, 0, 4)
            mv.addWidget(h)
        left.addWidget(box_mode)
        left.addStretch(1)

        # ---- 分档参数（3 档）
        self.box_tier = QGroupBox("分档参数（浮盈达到几 ATR → 从最高点回撤几 ATR 平仓）")
        tv = QFormLayout(self.box_tier)
        tv.setContentsMargins(10, 8, 10, 8)
        tv.setSpacing(6)
        tiers = list(cur.get("exit_tiers") or [[2.0, 2.0], [4.0, 1.2], [6.0, 0.8]])
        while len(tiers) < 3:
            tiers.append([float(len(tiers) + 2) * 2, 1.0])
        self.sp_tier: list[tuple[QDoubleSpinBox, QDoubleSpinBox]] = []
        for i in range(3):
            need, trail = float(tiers[i][0]), float(tiers[i][1])
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            a = QDoubleSpinBox()
            a.setRange(0.0, 50.0)
            a.setDecimals(2)
            a.setSingleStep(0.5)
            a.setValue(need)
            a.setSuffix(" ATR 浮盈")
            b = QDoubleSpinBox()
            b.setRange(0.1, 50.0)
            b.setDecimals(2)
            b.setSingleStep(0.1)
            b.setValue(trail)
            b.setSuffix(" ATR 回撤")
            row.addWidget(a)
            row.addWidget(b)
            self.sp_tier.append((a, b))
            tv.addRow(f"第 {i + 1} 档", row)
        hint = QLabel("浮盈门槛要递增、回撤幅度要递减（浮盈越多、允许回撤越小）。"
                      "点确定时会自动按这两条规则理顺。")
        hint.setStyleSheet(_HINT)
        hint.setWordWrap(True)
        tv.addRow("", hint)
        right.addWidget(self.box_tier)

        # ---- 绝对回撤
        self.box_abs = QGroupBox("绝对回撤")
        av = QFormLayout(self.box_abs)
        av.setContentsMargins(10, 8, 10, 8)
        av.setSpacing(6)
        self.sp_abs = QDoubleSpinBox()
        self.sp_abs.setRange(0.1, 50.0)
        self.sp_abs.setDecimals(2)
        self.sp_abs.setSingleStep(0.25)
        self.sp_abs.setValue(float(cur.get("exit_abs_atr") or 2.0))
        self.sp_abs.setSuffix(" ATR")
        av.addRow("从最高浮盈回撤", self.sp_abs)
        ah = QLabel("从「持仓期间见过的最好浮盈」往回退这么多就平仓。"
                    "浮盈还没到过 0 也一样生效 —— 它同时也是一条初始止损。")
        ah.setStyleSheet(_HINT)
        ah.setWordWrap(True)
        av.addRow("", ah)
        right.addWidget(self.box_abs)

        # ---- 反向箱体
        self.box_box = QGroupBox("反向箱体突破")
        bv = QFormLayout(self.box_box)
        bv.setContentsMargins(10, 8, 10, 8)
        bv.setSpacing(6)
        self.sp_bars = QSpinBox()
        self.sp_bars.setRange(3, 500)
        self.sp_bars.setValue(int(cur.get("exit_box_bars") or 20))
        self.sp_bars.setSuffix(" 根")
        bv.addRow("箱体宽度（回看根数）", self.sp_bars)
        self.sp_buf = QDoubleSpinBox()
        self.sp_buf.setRange(0.0, 10.0)
        self.sp_buf.setDecimals(2)
        self.sp_buf.setSingleStep(0.1)
        self.sp_buf.setValue(float(cur.get("exit_box_buf") or 0.0))
        self.sp_buf.setSuffix(" ATR")
        bv.addRow("边界缓冲", self.sp_buf)
        bh = QLabel("多头跌破「过去 N 根的最低点」、空头涨过「最高点」就平仓。"
                    "缓冲 > 0 表示要穿过边界这么多才算破，用来过滤贴着边界来回抖。")
        bh.setStyleSheet(_HINT)
        bh.setWordWrap(True)
        bv.addRow("", bh)
        right.addWidget(self.box_box)

        # 「分数出场」不需要任何 ATR 参数，右列会整个空掉。留一句说明，
        # 免得看着像「右边本来该有东西但没加载出来」
        self.lbl_none = QLabel("「分数出场」不需要额外参数：分数回落到离场阈值以内就平仓。"
                               "想加一层保护，就勾上左边的「启用 ATR 止损」。")
        self.lbl_none.setStyleSheet(_HINT)
        self.lbl_none.setWordWrap(True)
        self.lbl_none.setContentsMargins(4, 8, 4, 0)
        right.addWidget(self.lbl_none)

        self.sp_atr_n = QSpinBox()
        self.sp_atr_n.setRange(2, 100)
        self.sp_atr_n.setValue(int(cur.get("exit_atr_n") or 14))
        self.sp_atr_n.setSuffix(" 根")
        form_atr = QFormLayout()
        form_atr.setContentsMargins(0, 0, 0, 0)
        form_atr.addRow("ATR 周期（所有口径共用）", self.sp_atr_n)
        right.addLayout(form_atr)
        right.addStretch(1)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel
                                | QDialogButtonBox.Reset)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.button(QDialogButtonBox.Reset).setText("恢复默认")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.Reset).clicked.connect(self._reset)
        lay.addWidget(btns)
        self._btn_ok = btns.button(QDialogButtonBox.Ok)

        for rb in self.rb.values():
            rb.toggled.connect(self._sync)
        self._sync()
        # 高度按**内容**定：滚动区的 sizeHint 是个常数（它不知道里面多高），
        # 照它定会把窗口压成一条。宽度固定，别让它在切换口径时左右跳
        body.adjustSize()
        self.resize(900, min(body.sizeHint().height() + 96, 700))

    def showEvent(self, ev) -> None:      # noqa: N802 - Qt 的命名
        super().showEvent(ev)
        # 真实折行数要等布局跑完才知道（`sizeHint` 是照自然宽度算的，不是照
        # 900px 算的），所以第一次显示后量一下滚动区溢出多少，把窗口补高到
        # 刚好装下 —— 差二三十像素就冒出个滚动条，看着像内容被截了
        if self._fitted:
            return
        self._fitted = True
        screen = self.screen()
        cap = (screen.availableGeometry().height() - 80) if screen else 760
        # 补高会让竖向滚动条消失，视口变宽 → 折行数又变 → 还要再补一点。
        # 迭代三次足够收敛（每轮补的都是个位数像素）
        for _ in range(3):
            extra = self._scroll.verticalScrollBar().maximum()
            if extra <= 0:
                break
            self.resize(self.width(), int(min(self.height() + extra + 4, cap)))
            self.layout().activate()
        self._recenter()

    def _recenter(self) -> None:
        """补高之后重新居中。

        `QDialog.showEvent` 的居中是在 resize **之前**做的，不重算的话窗口会
        从原来的位置朝右下角长出去，看着像跑偏了。
        """
        scr = self.screen()
        par = self.parentWidget()
        if par is not None:
            c = par.window().frameGeometry().center()
            x, y = c.x() - self.width() // 2, c.y() - self.height() // 2
        elif scr is not None:
            c = scr.availableGeometry().center()
            x, y = c.x() - self.width() // 2, c.y() - self.height() // 2
        else:
            return
        if scr is not None:
            a = scr.availableGeometry()
            x = max(a.left(), min(x, a.right() - self.width()))
            y = max(a.top(), min(y, a.bottom() - self.height()))
        self.move(int(x), int(y))

    def _mode(self) -> str:
        for key, rb in self.rb.items():
            if rb.isChecked():
                return key
        return "score"

    def _sync(self) -> None:
        # 「反向也不平」包含「回落不平」：勾了前者，后者自动打勾并灰掉。
        # 反过来不成立 —— 只拦回落、放行反向是常用的中间档
        flip = self.chk_noflip.isChecked()
        self.chk_noscore.blockSignals(True)
        if flip:
            self.chk_noscore.setChecked(True)
        self.chk_noscore.setEnabled(not flip)
        self.chk_noscore.blockSignals(False)
        # 说明只在勾上时展开：三行字常驻会把确定按钮往下顶一截
        self.lbl_flip_hint.setVisible(flip)

        m = self._mode()
        self.box_tier.setVisible(m == "tier")
        self.box_abs.setVisible(m == "abs")
        self.box_box.setVisible(m == "box")
        self.lbl_none.setVisible(m == "score")
        # 勾了「不使用离场阈值出场」之后，「分数出场不需要额外参数」这句
        # 就自相矛盾了 —— 右列此刻确实什么都没有，得说清为什么
        if flip:
            self.lbl_none.setText(
                "已经勾了「分数反向也不平仓」，分数这条规则完全不参与。"
                "请在左边选一个 ATR 口径，或者勾上止损。")
        elif self.chk_noscore.isChecked():
            self.lbl_none.setText(
                "已经勾了「不使用离场阈值出场」，分数这条规则不参与。"
                "请在左边选一个 ATR 口径，或者勾上止损。")
        else:
            self.lbl_none.setText(
                "「分数出场」不需要额外参数：分数回落到离场阈值以内就平仓。"
                "想加一层保护，就勾上左边的「启用 ATR 止损」。")
        # 没勾止损时把倍数灰掉：留着一个能改但不起作用的数字，
        # 最容易出现「我明明设了 1.5 怎么没止损」
        self.sp_sl.setEnabled(self.chk_sl.isChecked())
        # 一条出场规则都没有的组合：直接拦住，别让人点完确定才发现仓位一直
        # 拿到最后（那时候图已经跑完了，看不出是配置问题）
        no_out = (self.chk_noscore.isChecked() and m == "score"
                  and not self.chk_sl.isChecked())
        self.lbl_ns_warn.setVisible(no_out)
        self._btn_ok.setEnabled(not no_out)

    def _reset(self) -> None:
        self.rb["score"].setChecked(True)
        for i, (a, b) in enumerate(self.sp_tier):
            a.setValue(float(i + 1) * 2.0)
            b.setValue([2.0, 1.2, 0.8][i])
        self.sp_abs.setValue(2.0)
        self.sp_bars.setValue(20)
        self.sp_buf.setValue(0.0)
        self.sp_atr_n.setValue(14)
        self.chk_sl.setChecked(False)
        self.sp_sl.setValue(2.0)
        self.chk_noflip.setChecked(False)
        self.chk_noscore.setChecked(False)
        self._sync()

    def values(self) -> dict:
        tiers = [[round(a.value(), 4), round(b.value(), 4)]
                 for a, b in self.sp_tier]
        # 排序 + 递减约束和 `factor_strategy._clean_tiers` 同一套规则：
        # 界面上先理顺，用户点确定就能看到「按什么口径在跑」
        tiers.sort(key=lambda x: x[0])
        for i in range(len(tiers) - 2, -1, -1):
            if tiers[i][1] < tiers[i + 1][1]:
                tiers[i][1] = tiers[i + 1][1]
        return {
            "exit_mode": self._mode(),
            "exit_tiers": tiers,
            "exit_abs_atr": round(float(self.sp_abs.value()), 4),
            "exit_box_bars": int(self.sp_bars.value()),
            "exit_box_buf": round(float(self.sp_buf.value()), 4),
            "exit_atr_n": int(self.sp_atr_n.value()),
            # 止损和口径**分开存**：它不是 exit_mode 的一个取值
            "exit_sl": bool(self.chk_sl.isChecked()),
            "exit_sl_atr": round(float(self.sp_sl.value()), 4),
            # 同理：这是「离场阈值参不参与」，不是第五种口径
            "exit_no_score": bool(self.chk_noscore.isChecked()),
            # 更彻底一层：反向也不平（隐含上面那条）
            "exit_no_flip": bool(self.chk_noflip.isChecked()),
        }


def summarize(cfg: dict | None) -> str:
    """一句话摘要，挂在侧边栏那个按钮下面。"""
    cfg = dict(cfg or {})
    mode = str(cfg.get("exit_mode") or "score")
    # 止损和口径是两条独立的信息，摘要是「A ＋ B」而不是二选一
    sl = (f" ＋ 止损 {float(cfg.get('exit_sl_atr') or 2.0):g}ATR"
          if cfg.get("exit_sl") else "")
    # 「不按分数出场」也得写出来：同一套 ATR 参数、勾没勾这一项，交易笔数差
    # 一大截，摘要不写的话两条配置在列表里长得一模一样
    flip_on = bool(cfg.get("exit_no_flip"))
    ns_on = flip_on or bool(cfg.get("exit_no_score"))
    ns = (" · 反向也不平仓" if flip_on else " · 不按分数出场") if ns_on else ""
    if mode == "score":
        # 只勾止损 + 不按分数出场：这时候分数规则一条都不剩，再写「分数出场」
        # 会让人以为离场阈值还在管
        if ns_on:
            return ("只按 ATR 出场" + ("（反向也不平仓）" if flip_on else "") + sl)
        return ("分数出场（回落到离场阈值就平）" + sl) if sl else \
            "分数出场（回落到离场阈值就平）"
    if mode == "tier":
        tiers = list(cfg.get("exit_tiers") or [])
        bits = "、".join(f"{float(p):g}/{float(t):g}" for p, t in tiers if p is not None)
        return f"分档浮盈回撤（浮盈/回撤 ATR：{bits}）{ns}{sl}"
    if mode == "abs":
        return (f"绝对回撤（最高浮盈回撤 {float(cfg.get('exit_abs_atr') or 2.0):g} ATR）"
                f"{ns}{sl}")
    return (f"反向箱体突破（{int(cfg.get('exit_box_bars') or 20)} 根，"
            f"缓冲 {float(cfg.get('exit_box_buf') or 0.0):g} ATR）{ns}{sl}")
