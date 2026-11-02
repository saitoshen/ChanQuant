"""截面策略的两步配置窗：左边第一步「截面选品种」，右边第二步「策略落地」。

为什么单独开一个窗：截面策略天生是**两步**。第一步决定「拿哪些品种、多哪几个
空哪几个」，第二步决定「选出来的这些品种怎么下单」。两件事竖着排在研究页侧边栏
里，一屏放不下；更麻烦的是第二步的内容会随第一步变（截面对冲根本没有时序腿），
上下并排会让人以为那些参数是一回事，改一个也不知道影响了哪一步。

左右并排之后，「第一步改了什么、第二步跟着变成什么样」是能一眼看见的。格局参照
期权 T 型报价：左边一列、右边一列，中间一道分隔。

**控件不在这里造**。参数控件都在研究页那边（`box_fsel` / `box_rule` 等），切到
截面策略时由宿主「借」进来，切走再还回去 —— 见 `MainWindow._lend`。造第二份的
话，改了一个另一个不会跟着动，两边慢慢走偏，还会让保存配置要认两套名字。
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

_HINT = "font-size: 11px; color: #64748b;"
_BADGE = (
    "background: #2563eb; color: #ffffff; border-radius: 8px;"
    "padding: 1px 7px; font-size: 11px; font-weight: 700;"
)
_TITLE = "font-size: 13px; font-weight: 700; color: #0f172a;"


class CrossSetupDialog(QDialog):
    """两步并排的截面配置窗。宿主负责往里塞面板，这里只管摆放和标题。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("截面策略配置 · 两步")
        self.resize(1080, 780)
        self.setSizeGripEnabled(True)
        # 非模态：改参数要能同时看右边的图和回测结果，锁死窗口就没法对照了
        self.setModal(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(9)

        head = QLabel(
            "截面策略分两步：第一步横向比强弱，决定拿哪些品种、多哪几个空哪几个；"
            "第二步决定这些品种怎么下单。左边改第一步，右边跟着变。"
        )
        head.setObjectName("Hint")
        head.setWordWrap(True)
        lay.addWidget(head)

        cols = QHBoxLayout()
        cols.setSpacing(10)
        self._step1, self.step1_lay, self.lbl_step1_sub = self._pane(
            "第一步", "截面选品种", "按截面因子横向排序，定下多空两端"
        )
        self._step2, self.step2_lay, self.lbl_step2_sub = self._pane(
            "第二步", "策略落地", "选出来的品种怎么下单、腿内怎么分钱"
        )
        cols.addWidget(self._step1, 1)
        cols.addWidget(self._step2, 1)
        lay.addLayout(cols, 1)

        foot = QHBoxLayout()
        self.lbl_foot = QLabel("")
        self.lbl_foot.setObjectName("Hint")
        self.lbl_foot.setWordWrap(True)
        foot.addWidget(self.lbl_foot, 1)
        btn = QPushButton("关闭")
        btn.setObjectName("Ghost")
        btn.clicked.connect(self.close)
        foot.addWidget(btn)
        lay.addLayout(foot)

        self.set_mode("hedge")

    # ------------------------------------------------------------------ 构建
    def _pane(self, step: str, title: str, sub: str):
        """一栏：标题条 + 可滚动的参数区。

        参数区做成滚动容器，是因为第一步里挂着整张因子表（147 行里挑几十个），
        直接铺开会把窗口撑到屏幕外面去。
        """
        frame = QFrame()
        frame.setObjectName("SubCard")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        head = QHBoxLayout()
        head.setSpacing(8)
        badge = QLabel(step)
        badge.setStyleSheet(_BADGE)
        badge.setAlignment(Qt.AlignCenter)
        head.addWidget(badge, 0, Qt.AlignTop)

        col = QVBoxLayout()
        col.setSpacing(1)
        t = QLabel(title)
        t.setStyleSheet(_TITLE)
        s = QLabel(sub)
        s.setStyleSheet(_HINT)
        s.setWordWrap(True)
        col.addWidget(t)
        col.addWidget(s)
        head.addLayout(col, 1)
        outer.addLayout(head)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")
        body = QWidget()
        body.setStyleSheet("background: transparent;")
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 2, 4, 0)
        bl.setSpacing(8)
        # 参数区不满一屏时贴着顶排，不要被拉成均分
        bl.addStretch(1)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        return frame, bl, s

    def attach(self, step1: QWidget, step2: QWidget) -> None:
        """把宿主造好的两块面板塞进左右两栏（末尾那个 stretch 之前）。"""
        for lay, w in ((self.step1_lay, step1), (self.step2_lay, step2)):
            i = lay.indexOf(w)
            if i >= 0:
                lay.takeAt(i)
            lay.insertWidget(max(0, lay.count() - 1), w)

    # ------------------------------------------------------------------ 状态
    def set_mode(self, mode: str, timing: str = "") -> None:
        """按截面模式换第二步的说明。截面对冲没有时序腿，标题要写清楚。"""
        select = mode == "select"
        if select:
            leg = "规则策略" if timing == "rule" else "因子策略"
            self.lbl_step2_sub.setText(
                f"第一步挑出来的品种，每个都由时序{leg}决定是多是空还是空仓"
            )
            self.lbl_foot.setText(
                "选品种模式：截面只负责「买哪几只」，进出场交给时序腿 —— "
                "同一条时序策略在每个入选品种上各跑一遍。"
            )
        else:
            self.lbl_step2_sub.setText(
                "截面对冲没有时序腿：多空两端第一步已经定了，这一步只决定腿内怎么分钱"
            )
            self.lbl_foot.setText(
                "对冲模式：多头腿和空头腿各占 50% 名义额，多空两边各自平分 —— "
                "腿数不一样也不会产生净敞口。"
            )
