"""「数据管理」页里的两个小工具窗：成本参数、品种信息。

为什么不直接铺在数据页上：那页是一列窄表单，塞一张 78 行 × 9 列的表会把
「补数据」那几条挤到看不见。做成弹窗之后，平时不占地方，要看的时候一次看全。

两个窗都遵循同一条原则：**只改用户动过的那几格**。没动过的继续吃默认值
（`symbols_info.defaults`），这样升级带进来的新默认值不会被一份旧快照盖掉。
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ttrade import costs, symbols_info

_HINT = "font-size: 11px; color: #64748b;"


def _num_item(v: float, digits: int = 4) -> QTableWidgetItem:
    """数字格：右对齐，显示时去掉多余的 0（0.0000 → 0）。"""
    txt = f"{v:.{digits}f}".rstrip("0").rstrip(".") or "0"
    it = QTableWidgetItem(txt)
    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
    return it


def _to_float(it: QTableWidgetItem, dflt: float = 0.0) -> float:
    try:
        return float(str(it.text()).strip().replace("，", "").replace(",", ""))
    except (AttributeError, ValueError):
        return dflt


class CostDialog(QDialog):
    """三档成本的参数。改完立刻落盘，研究页 / 组合页下次重算就用新的。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("交易成本参数（国内期货）")
        self.resize(520, 300)
        s = costs.load()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        lay.addWidget(QLabel(
            "单边成本 = 交易所手续费 × 倍数 + 滑点跳数 × 最小变动价位。\n"
            "手续费按品种算（见「品种信息」），这里只调倍数和跳数。"
        ))

        self.tbl = QTableWidget(len(s["profiles"]), 3)
        self.tbl.setHorizontalHeaderLabels(["档位", "手续费倍数", "滑点（跳）"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._keys: list[str] = []
        for r, (key, p) in enumerate(s["profiles"].items()):
            self._keys.append(key)
            name = QTableWidgetItem(p["label"])
            name.setData(Qt.UserRole, key)
            self.tbl.setItem(r, 0, name)
            for col, field, lo, hi, step in (
                (1, "fee_mult", 0.0, 10.0, 0.1),
                (2, "slip_ticks", 0.0, 20.0, 0.5),
            ):
                sp = QDoubleSpinBox()
                sp.setRange(lo, hi)
                sp.setSingleStep(step)
                sp.setDecimals(2)
                sp.setValue(float(p[field]))
                self.tbl.setCellWidget(r, col, sp)
        lay.addWidget(self.tbl, 1)

        self.lbl = QLabel("")
        self.lbl.setStyleSheet(_HINT)
        self.lbl.setWordWrap(True)
        lay.addWidget(self.lbl)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        box.button(QDialogButtonBox.Save).setText("保存")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    def _save(self) -> None:
        s = costs.load()
        for r, key in enumerate(self._keys):
            s["profiles"][key]["fee_mult"] = float(self.tbl.cellWidget(r, 1).value())
            s["profiles"][key]["slip_ticks"] = float(self.tbl.cellWidget(r, 2).value())
        costs.save(s)
        self.accept()


class SymbolInfoDialog(QDialog):
    """品种基本信息：tick、乘数、保证金率、手续费、交易时段。

    `items` 是 `futures.list_local()` 那套（含 symbol / exchange / name）。
    """

    COLS = ["交易所", "品种", "名称", "最小跳动", "乘数", "保证金%",
            "手续费‱", "元/手", "交易时段"]

    def __init__(self, items: list[dict], data_dir: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("品种基本信息（国内期货）")
        self.resize(1080, 620)
        self._items = list(items or [])
        self._data_dir = data_dir

        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)
        tip = QLabel(
            "手续费填两列之一：按成交额比例（‱，万分之几）或按手固定（元/手）。"
            "两列都填时**按手**优先，和回测引擎的判定一致。\n"
            "留空的格子继续吃默认值；「恢复默认」只清当前选中的那几行。"
        )
        tip.setStyleSheet(_HINT)
        tip.setWordWrap(True)
        lay.addWidget(tip)

        self.tbl = QTableWidget(len(self._items), len(self.COLS))
        self.tbl.setHorizontalHeaderLabels(self.COLS)
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tbl.horizontalHeader().setSectionResizeMode(8, QHeaderView.Stretch)
        for c in range(8):
            self.tbl.horizontalHeader().setSectionResizeMode(c, QHeaderView.ResizeToContents)
        lay.addWidget(self.tbl, 1)
        self._fill()

        row = QHBoxLayout()
        btn_def = QPushButton("恢复默认（选中行）")
        btn_def.clicked.connect(self._reset_rows)
        row.addWidget(btn_def)
        row.addStretch(1)
        self.lbl = QLabel("")
        self.lbl.setStyleSheet(_HINT)
        row.addWidget(self.lbl)
        lay.addLayout(row)

        box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)
        box.button(QDialogButtonBox.Save).setText("保存")
        box.button(QDialogButtonBox.Close).setText("关闭")
        box.accepted.connect(self._save)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

    # ---- 建表 ----

    def _fill(self) -> None:
        for r, it in enumerate(self._items):
            sym, ex = it.get("symbol", ""), it.get("exchange", "")
            key = f"{sym}_{ex}"
            info = symbols_info.info(sym, ex)
            tick = float(info.get("tick") or 0.0)
            if tick <= 0:
                tick = self._infer(sym, ex)
            fee_rate = float(info.get("fee_value") or 0.0) * 1e4 \
                if str(info.get("fee_mode")) == "rate" else 0.0
            fee_lot = float(info.get("fee_value") or 0.0) \
                if str(info.get("fee_mode")) == "lot" else 0.0

            fixed = QTableWidgetItem(symbols_info.EXCHANGES.get(ex, ex))
            fixed.setFlags(Qt.ItemIsEnabled)
            fixed.setData(Qt.UserRole, key)
            self.tbl.setItem(r, 0, fixed)
            for col, txt in ((1, sym), (2, it.get("name", ""))):
                cell = QTableWidgetItem(txt)
                cell.setFlags(Qt.ItemIsEnabled)
                self.tbl.setItem(r, col, cell)
            self.tbl.setItem(r, 3, _num_item(tick, 4))
            self.tbl.setItem(r, 4, _num_item(float(info.get("mult") or 0.0), 2))
            self.tbl.setItem(r, 5, _num_item(float(info.get("margin") or 0.0) * 100, 2))
            self.tbl.setItem(r, 6, _num_item(fee_rate, 4))
            self.tbl.setItem(r, 7, _num_item(fee_lot, 3))
            sess = QTableWidgetItem(str(info.get("session") or ""))
            self.tbl.setItem(r, 8, sess)
            for c in (3, 4, 5, 6, 7, 8):
                self.tbl.item(r, c).setToolTip("双击可改")

    def _infer(self, sym: str, ex: str) -> float:
        """表里没有的品种，从本地日线的收盘价反推 tick。"""
        if not self._data_dir:
            return 0.0
        path = os.path.join(self._data_dir, f"{sym}_{ex}.csv")
        try:
            import pandas as pd

            df = pd.read_csv(path, usecols=["close"])
            return symbols_info.infer_tick(df["close"].tail(400))
        except Exception:  # noqa: BLE001
            return 0.0

    # ---- 保存 ----

    def _save(self) -> None:
        n = 0
        for r, it in enumerate(self._items):
            sym, ex = it.get("symbol", ""), it.get("exchange", "")
            key = f"{sym}_{ex}"
            base = symbols_info.defaults(sym, ex)
            tick = _to_float(self.tbl.item(r, 3))
            mult = _to_float(self.tbl.item(r, 4))
            margin = _to_float(self.tbl.item(r, 5)) / 100.0
            fee_rate = _to_float(self.tbl.item(r, 6))
            fee_lot = _to_float(self.tbl.item(r, 7))
            sess = (self.tbl.item(r, 8).text() or "").strip()

            fields: dict = {}
            if tick > 0 and abs(tick - float(base["tick"] or 0)) > 1e-9:
                fields["tick"] = tick
            if mult > 0 and abs(mult - float(base["mult"] or 0)) > 1e-9:
                fields["mult"] = mult
            if abs(margin - float(base["margin"] or 0)) > 1e-9:
                fields["margin"] = margin
            if fee_lot > 0:
                if abs(fee_lot - (float(base["fee_value"]) if base["fee_mode"] == "lot" else 0)) > 1e-9:
                    fields["fee_mode"] = "lot"
                    fields["fee_value"] = fee_lot
            elif fee_rate > 0:
                want = fee_rate / 1e4
                cur = float(base["fee_value"]) if base["fee_mode"] == "rate" else 0.0
                if abs(want - cur) > 1e-9:
                    fields["fee_mode"] = "rate"
                    fields["fee_value"] = want
            if sess and sess != str(base["session"]):
                fields["session"] = sess

            if fields:
                symbols_info.set_override(sym, ex, **fields)
                n += 1
            elif key in symbols_info.load_overrides():
                # 全都改回默认值了：把覆盖删掉，而不是存一份「和默认一样」的快照
                symbols_info.reset_override(sym, ex)
        self.lbl.setText(f"已保存 {n} 个品种的修改" if n else "没有改动")
        if n:
            QMessageBox.information(self, "已保存", f"{n} 个品种已更新。")

    def _reset_rows(self) -> None:
        rows = sorted({i.row() for i in self.tbl.selectedIndexes()})
        if not rows:
            self.lbl.setText("先选几行")
            return
        for r in rows:
            it = self._items[r]
            symbols_info.reset_override(it.get("symbol", ""), it.get("exchange", ""))
        self._fill()
        self.lbl.setText(f"已恢复 {len(rows)} 行")
