"""「按条件选股」对话框（A 股专用）。

截面策略要一批品种，但 A 股有 5577 只 —— 用勾选框列出来既卡又找不到。这个
对话框把「挑池子」拆成两步：先按条件收窄（板块 / 行业 / 成交额 / 市值 /
关键词），再在结果里勾。

结果上限默认 60 只：截面回测是**逐只跑一遍再横截面排序**，几百只跑一次要
几分钟，界面会像死了一样。真要大池子，用户自己把上限调大。
"""

from __future__ import annotations

import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ttrade import stockpick as SP

C_MAX = 400          # 表格最多显示这么多行，再多也看不过来
C_PICK = 60          # 默认最多选这么多只


class StockPickDialog(QDialog):
    """按条件筛 A 股，返回选中的品种键（`SSE.600000` 这种）。"""

    def __init__(self, preselect=None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("按条件选股")
        self.resize(880, 620)
        self._df = pd.DataFrame()
        self._picked: list[str] = list(preselect or [])

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(8)

        # ---- 条件行 1：板块 / 行业 / 关键词
        row1 = QHBoxLayout()
        row1.setSpacing(8)
        self.chk_boards: dict[str, QCheckBox] = {}
        for key, label in SP.BOARDS:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._refresh)
            self.chk_boards[key] = cb
            row1.addWidget(cb)
        row1.addSpacing(10)
        row1.addWidget(QLabel("行业"))
        self.cb_industry = QComboBox()
        self.cb_industry.setMinimumWidth(140)
        self.cb_industry.currentIndexChanged.connect(self._refresh)
        row1.addWidget(self.cb_industry)
        row1.addStretch(1)
        root.addLayout(row1)

        # ---- 条件行 2：成交额 / 市值 / 关键词 / 上限
        row2 = QHBoxLayout()
        row2.setSpacing(8)
        row2.addWidget(QLabel("日均成交额 ≥"))
        self.sp_amount = QDoubleSpinBox()
        self.sp_amount.setRange(0.0, 500.0)
        self.sp_amount.setDecimals(2)
        self.sp_amount.setSingleStep(0.5)
        self.sp_amount.setValue(1.0)
        self.sp_amount.setSuffix(" 亿")
        self.sp_amount.valueChanged.connect(self._refresh)
        row2.addWidget(self.sp_amount)

        row2.addWidget(QLabel("市值"))
        self.sp_mv_lo = QDoubleSpinBox()
        self.sp_mv_lo.setRange(0.0, 50000.0)
        self.sp_mv_lo.setDecimals(0)
        self.sp_mv_lo.setSingleStep(50.0)
        self.sp_mv_lo.setValue(0.0)
        self.sp_mv_lo.setSuffix(" 亿")
        self.sp_mv_lo.valueChanged.connect(self._refresh)
        row2.addWidget(self.sp_mv_lo)
        row2.addWidget(QLabel("~"))
        self.sp_mv_hi = QDoubleSpinBox()
        self.sp_mv_hi.setRange(0.0, 50000.0)
        self.sp_mv_hi.setDecimals(0)
        self.sp_mv_hi.setSingleStep(100.0)
        self.sp_mv_hi.setValue(0.0)
        self.sp_mv_hi.setSuffix(" 亿")
        self.sp_mv_hi.setToolTip("上限填 0 表示不限")
        self.sp_mv_hi.valueChanged.connect(self._refresh)
        row2.addWidget(self.sp_mv_hi)

        self.ed_kw = QLineEdit()
        self.ed_kw.setPlaceholderText("代码 / 名称关键词")
        self.ed_kw.setMaximumWidth(160)
        self.ed_kw.textChanged.connect(self._refresh)
        row2.addWidget(self.ed_kw)
        row2.addStretch(1)
        root.addLayout(row2)

        # ---- 结果表
        self.lbl_info = QLabel("")
        self.lbl_info.setObjectName("Hint")
        self.lbl_info.setWordWrap(True)
        root.addWidget(self.lbl_info)

        self.tbl = QTableWidget(0, 6)
        self.tbl.setHorizontalHeaderLabels(["代码", "名称", "行业", "板块",
                                            "市值(亿)", "日均成交额(亿)"])
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSortingEnabled(True)
        root.addWidget(self.tbl, 1)

        # ---- 底部按钮
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        self.sp_max = QSpinBox()
        self.sp_max.setRange(3, 400)
        self.sp_max.setValue(C_PICK)
        self.sp_max.setPrefix("最多选 ")
        self.sp_max.setSuffix(" 只")
        self.sp_max.valueChanged.connect(self._sync_info)
        bottom.addWidget(self.sp_max)
        btn_all = QLabel("")
        bottom.addWidget(btn_all)
        bottom.addStretch(1)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Ok).setText("加入品种池")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        bottom.addWidget(box)
        root.addLayout(bottom)

        self._load()

    # ---- 数据 ----

    def _load(self) -> None:
        try:
            self._df = SP.snapshot()
        except Exception as exc:  # noqa: BLE001 - 库锁着（正在下载）时要说清楚
            self._df = pd.DataFrame()
            self.lbl_info.setText(
                f"读不到股票库：{type(exc).__name__}: {exc}。"
                "如果正在下载数据，等它跑完再选。")
        self.cb_industry.blockSignals(True)
        self.cb_industry.clear()
        self.cb_industry.addItem("全部行业", "")
        for name in SP.industries(self._df):
            self.cb_industry.addItem(name, name)
        self.cb_industry.blockSignals(False)
        self._refresh()

    def _refresh(self) -> None:
        boards = [k for k, cb in self.chk_boards.items() if cb.isChecked()]
        df = SP.filter_frame(
            self._df,
            boards=boards,
            industry=self.cb_industry.currentData() or "",
            min_amount=float(self.sp_amount.value()),
            min_mv=float(self.sp_mv_lo.value()),
            max_mv=float(self.sp_mv_hi.value()),
            keyword=self.ed_kw.text(),
            limit=C_MAX,
        )
        self._fill_table(df)
        self._sync_info(len(df))

    def _fill_table(self, df: pd.DataFrame) -> None:
        self.tbl.setSortingEnabled(False)
        self.tbl.setRowCount(len(df))
        for i, (_, r) in enumerate(df.iterrows()):
            vals = [
                str(r.get("ts_code") or ""),
                str(r.get("name") or ""),
                str(r.get("industry") or ""),
                str(r.get("board") or ""),
                f"{float(r.get('total_mv') or 0):,.0f}",
                f"{float(r.get('amount') or 0):,.2f}",
            ]
            for j, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if j >= 4:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 0:
                    it.setData(Qt.UserRole, v)
                    # 已经在池子里的行直接勾上，重开对话框时不会「看起来没选」
                    if v in self._picked:
                        it.setCheckState(Qt.Checked)
                    else:
                        it.setCheckState(Qt.Unchecked)
                self.tbl.setItem(i, j, it)
        self.tbl.setSortingEnabled(True)
        self.tbl.resizeColumnsToContents()
        hh = self.tbl.horizontalHeader()
        hh.setSectionResizeMode(1, QHeaderView.Stretch)

    def _sync_info(self, n: int) -> None:
        picked = len(self._selected())
        self.lbl_info.setText(
            f"符合条件的 {n} 只（表里最多列 {C_MAX} 只），已勾 {picked} 只。"
            f" 截面回测逐只跑一遍，池子越大越慢 —— 建议 {self.sp_max.value()} 只以内。"
        )

    # ---- 结果 ----

    def _selected(self) -> list[str]:
        out = []
        for i in range(self.tbl.rowCount()):
            it = self.tbl.item(i, 0)
            if it is not None and it.checkState() == Qt.Checked:
                out.append(it.data(Qt.UserRole) or it.text())
        return out

    def selected(self) -> list[str]:
        """最终池子：本次勾的 + 之前就有、这次没出现在表里的。"""
        picked = self._selected()
        limit = int(self.sp_max.value())
        seen = set(picked)
        # 表外的旧选择保留，否则「筛一次就把之前选的冲掉」很难用
        extra = [k for k in self._picked if k not in seen]
        return (picked + extra)[:limit]

    def accept(self) -> None:
        if len(self._selected()) < 3:
            self.lbl_info.setText("截面策略至少选 3 只（勾第一列的复选框）")
            return
        super().accept()
