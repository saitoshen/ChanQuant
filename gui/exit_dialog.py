"""因子策略的「出场规则」弹窗。

为什么不铺在侧边栏：这一项本身有四种口径、每种口径各有 2~6 个参数，
平铺出来比整个因子参数区还长。侧边栏已经只剩 300 多像素宽，再塞进去
就只能靠横向滚动 —— 那是上一版刚拆掉的东西。

设计上的两条硬约束：

- **只改用户动过的口径**：切回「分数出场」时把其它参数留在原处，不会因为
  一次开合就把调好的 3 档改回默认
- 参数之间要能互相校验：分档的「回撤幅度」必须随浮盈递减，写反了不会报错，
  只会让规则悄悄变成相反的东西（浮盈越多越松）
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)

_HINT = "font-size: 11px; color: #64748b;"

# 三种 ATR 口径的说明。写在单选项下面而不是 tooltip 里：这一屏本来就是
# 「不知道选哪个才点进来」的人在看，悬停提示等于没写
_MODE_TIPS = {
    "score": "分数回落到离场阈值以内就平。原来的口径，不叠加任何 ATR 规则。",
    "tier": "浮盈按 ATR 折算，分 3 档：浮盈越多，允许的回撤越小。\n"
            "先让利润跑，跑起来之后收紧 —— 趋势单最怕「赚了 8 个 ATR 又全还回去」。",
    "abs": "从持仓期间的最高浮盈回撤固定 N×ATR 就平，不分档。\n"
           "比「分档」宽松，相当于一条普通的 ATR 跟踪止损。",
    "box": "持仓期间价格反向穿越「过去 N 根的高低点」就平。\n"
           "N 决定箱体宽度：N 越大箱体越宽、越不容易被打掉。",
}


class ExitRuleDialog(QDialog):
    """出场规则的设置窗。`values()` 返回可以直接塞进 `FactorConfig` 的 dict。"""

    def __init__(self, cur: dict | None = None, parent=None) -> None:
        super().__init__(parent)
        cur = dict(cur or {})
        self.setWindowTitle("出场规则")
        self.setMinimumWidth(420)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        tip = QLabel(
            "出场规则叠在「分数过阈值」的信号之上，只减仓、不反手：\n"
            "平掉之后要等信号回到 0 / 翻向，或分数重新越过进场阈值，才重新开仓。\n"
            "浮盈和回撤都按 ATR 折算，ATR 用和回测同一个口径（14 根真实波幅均值）。"
        )
        tip.setStyleSheet(_HINT)
        tip.setWordWrap(True)
        lay.addWidget(tip)

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
        lay.addWidget(box_mode)

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
        lay.addWidget(self.box_tier)

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
        ah = QLabel("从「持仓期间见过的最好浮盈」往回退这么多就平仓。\n"
                    "浮盈还没到过 0 也一样生效 —— 它同时也是一条初始止损。")
        ah.setStyleSheet(_HINT)
        ah.setWordWrap(True)
        av.addRow("", ah)
        lay.addWidget(self.box_abs)

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
        bh = QLabel("多头跌破「过去 N 根的最低点」、空头涨过「最高点」就平仓。\n"
                    "缓冲 > 0 表示要穿过边界这么多才算破，用来过滤贴着边界来回抖。")
        bh.setStyleSheet(_HINT)
        bh.setWordWrap(True)
        bv.addRow("", bh)
        lay.addWidget(self.box_box)

        self.sp_atr_n = QSpinBox()
        self.sp_atr_n.setRange(2, 100)
        self.sp_atr_n.setValue(int(cur.get("exit_atr_n") or 14))
        self.sp_atr_n.setSuffix(" 根")
        form_atr = QFormLayout()
        form_atr.setContentsMargins(0, 0, 0, 0)
        form_atr.addRow("ATR 周期", self.sp_atr_n)
        lay.addLayout(form_atr)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel
                                | QDialogButtonBox.Reset)
        btns.button(QDialogButtonBox.Ok).setText("确定")
        btns.button(QDialogButtonBox.Cancel).setText("取消")
        btns.button(QDialogButtonBox.Reset).setText("恢复默认")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.Reset).clicked.connect(self._reset)
        lay.addWidget(btns)

        for rb in self.rb.values():
            rb.toggled.connect(self._sync)
        self._sync()

    def _mode(self) -> str:
        for key, rb in self.rb.items():
            if rb.isChecked():
                return key
        return "score"

    def _sync(self) -> None:
        m = self._mode()
        self.box_tier.setVisible(m == "tier")
        self.box_abs.setVisible(m == "abs")
        self.box_box.setVisible(m == "box")

    def _reset(self) -> None:
        self.rb["score"].setChecked(True)
        for i, (a, b) in enumerate(self.sp_tier):
            a.setValue(float(i + 1) * 2.0)
            b.setValue([2.0, 1.2, 0.8][i])
        self.sp_abs.setValue(2.0)
        self.sp_bars.setValue(20)
        self.sp_buf.setValue(0.0)
        self.sp_atr_n.setValue(14)
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
        }


def summarize(cfg: dict | None) -> str:
    """一句话摘要，挂在侧边栏那个按钮下面。"""
    cfg = dict(cfg or {})
    mode = str(cfg.get("exit_mode") or "score")
    if mode == "score":
        return "分数出场（回落到离场阈值就平）"
    if mode == "tier":
        tiers = list(cfg.get("exit_tiers") or [])
        bits = "、".join(f"{float(p):g}/{float(t):g}" for p, t in tiers if p is not None)
        return f"分档浮盈回撤（浮盈/回撤 ATR：{bits}）"
    if mode == "abs":
        return f"绝对回撤（最高浮盈回撤 {float(cfg.get('exit_abs_atr') or 2.0):g} ATR）"
    return (f"反向箱体突破（{int(cfg.get('exit_box_bars') or 20)} 根，"
            f"缓冲 {float(cfg.get('exit_box_buf') or 0.0):g} ATR）")
