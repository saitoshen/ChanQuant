"""「数据管理」页里的两个小工具窗：成本参数、品种信息。

为什么不直接铺在数据页上：那页是一列窄表单，塞一张 78 行 × 9 列的表会把
「补数据」那几条挤到看不见。做成弹窗之后，平时不占地方，要看的时候一次看全。

两个窗都遵循同一条原则：**只改用户动过的那几格**。没动过的继续吃默认值
（`symbols_info.defaults`），这样升级带进来的新默认值不会被一份旧快照盖掉。
"""

from __future__ import annotations

import os

from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
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
    """各档成本的参数。改完立刻落盘，研究页 / 组合页下次重算就用新的。

    四档里有两种滑点口径（跳 / ‱），所以表里两列都摆出来，**用不上的那列置灰**。
    只留一列再按档位改标题的话，用户切到「按 bp」会看见自己刚设的「压力 2 跳」
    变成 0，以为被清掉了。
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("交易成本参数（国内期货）")
        self.resize(620, 320)
        s = costs.load()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        lay.addWidget(QLabel(
            "单边成本 = 交易所手续费 × 倍数 + 滑点。\n"
            "滑点有两种口径：按跳（× 最小变动价位）或按 ‱（× 价格），灰色那列不生效。\n"
            "手续费按品种算（见「品种信息」），这里只调倍数和滑点。\n"
            "跳数在回测里会乘当日复权因子：前复权把历史价格整体缩小过，而一跳是\n"
            "真实世界里的固定金额，不换算等于让越早的成交越便宜。"
        ))

        self.tbl = QTableWidget(len(s["profiles"]), 4)
        self.tbl.setHorizontalHeaderLabels(
            ["档位", "手续费倍数", "滑点（跳）", "滑点（‱）"])
        self.tbl.verticalHeader().setVisible(False)
        self.tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._keys: list[str] = []
        for r, (key, p) in enumerate(s["profiles"].items()):
            self._keys.append(key)
            name = QTableWidgetItem(p["label"])
            name.setData(Qt.UserRole, key)
            self.tbl.setItem(r, 0, name)
            bp_mode = str(p.get("mode") or "tick") == "bp"
            for col, field, lo, hi, step, on in (
                (1, "fee_mult", 0.0, 10.0, 0.1, True),
                (2, "slip_ticks", 0.0, 20.0, 0.5, not bp_mode),
                (3, "slip_bp", 0.0, 200.0, 0.5, bp_mode),
            ):
                sp = QDoubleSpinBox()
                sp.setRange(lo, hi)
                sp.setSingleStep(step)
                sp.setDecimals(2)
                sp.setValue(float(p.get(field) or 0.0))
                sp.setEnabled(on)
                sp.setToolTip("" if on else "这一档用的是另一种口径，改这里不生效")
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
            p = s["profiles"][key]
            p["fee_mult"] = float(self.tbl.cellWidget(r, 1).value())
            p["slip_ticks"] = float(self.tbl.cellWidget(r, 2).value())
            p["slip_bp"] = float(self.tbl.cellWidget(r, 3).value())
        costs.save(s)
        self.accept()


class _SymSyncWorker(QObject):
    """后台连天勤取真实合约参数（乘数 / 保证金率）并写进品种信息表。

    为什么后台：78~87 个品种要先订阅主连、再等第一帧行情，实测几十秒。
    同步跑会把整个窗口卡住，连「取消」都点不动。

    为什么自建天勤会话：数据管理页那条补行情的会话是按需建的，弹这个框的
    时候未必在。`tqfetch.Session` 有连接复用和退出清理，自己建一份比借来借去
    安全 —— 借的那份随时可能被上层 reset。
    """

    progress = Signal(int, int, str)
    done = Signal(object)          # (统计 dict, 取到的行)
    failed = Signal(str)

    def __init__(self, pairs: list) -> None:
        super().__init__()
        self.pairs = list(pairs or [])
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from ttrade import symsync as SYNC

            st = SYNC.refresh(
                self.pairs,
                progress=lambda a, b, m: self.progress.emit(a, b, m),
                cancel=lambda: self._cancel,
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.done.emit(st)


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
            "留空的格子继续吃默认值；「恢复默认」只清当前选中的那几行。\n\n"
            "🔴 乘数与保证金率**实盘和回测共用这一张表** —— 手数是\n"
            "　　本金×占用% ÷ (价格 × 乘数 × 保证金率)，两边算的是同一个式子。\n"
            "以前实盘读天勤合约上的实时值（`quote.margin`），结果主连恒 0、模拟环境\n"
            "不推字段、真合约的 margin 又随结算价天天漂，同一条策略的手数一天里能在\n"
            "1~4 手之间跳，还没有报错。现在只有这一个口径：改表 = 同时改实盘和回测。\n"
            "表里没填（0）才退回按类别默认：商品 10% / 股指 12% / 国债 3%。\n\n"
            "⚠ 保证金率请填**实盘真实冻结**的比例（含经纪商加收），不是交易所裸标准 ——\n"
            "填小了，按它算出来的手数实盘会因为保证金不足被拒单（T 的交易所标准 2%，\n"
            "期货公司实收常在 4%~5%）。\n\n"
            "「从天勤刷新」把**乘数 / 最小跳动 / 保证金率 / 手续费**四个数一次刷成\n"
            "此刻的真实值（保证金率 = quote.margin ÷ (昨结算价 × 乘数)，含加收）；\n"
            "手续费会按天勤给的值自动换算口径：按比例收的（cu、i、AG、股指）存成\n"
            "比例，按手收的（au、SR、m）存成元/手 —— 存错口径当场看不出，价格漂移后\n"
            "成本才会悄悄偏低或偏高。手填过的格子不被覆盖。")
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

        # 「从天勤刷新」：实盘和回测现在共用这张表，所以这张表就是**唯一**的手数
        # 来源。刷新按钮把天勤那边的真值一次写进来，之后两边同时生效。
        self.btn_sync = QPushButton("从天勤刷新（乘数·跳动·保证金·手续费）")
        self.btn_sync.setToolTip(
            "连天勤，逐个品种取**主连 underlying 合约**的真实参数，写进这张表：\n\n"
            "  乘数       ← quote.volume_multiple\n"
            "  保证金率   ← quote.margin ÷ (昨结算价 × 乘数)\n"
            "  最小跳动   ← quote.price_tick\n"
            "  手续费     ← quote.commission（自动判「按手 / 按比例」）\n\n"
            "🔴 这张表**同时决定实盘和回测的手数**，所以刷完两边立刻一致；\n"
            "保证金率用的是昨结算价（天勤算 margin 的基准），不是最新价 —— 用最新价\n"
            "反推会凭空多几个百分点，而且盘中一直在漂。\n\n"
            "只写差异超过 2% 的项（保证金随结算价天天微调，全表重写没意义）。\n"
            "**会覆盖手填值** —— 这是你主动点的，按真实值来。\n\n"
            "⚠ 交易所只报「每手多少元」，历史逐日的保证金率拿不到，所以这是\n"
            "「此刻的值」。回测拿它跑全历史，等于假设保证金率不变。")
        self.btn_sync.clicked.connect(self._sync_from_tq)
        row.addWidget(self.btn_sync)

        self.prog_sync = QProgressBar()
        self.prog_sync.setVisible(False)
        self.prog_sync.setMinimumWidth(160)
        self.prog_sync.setTextVisible(False)
        row.addWidget(self.prog_sync)

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

    # ---- 从天勤刷新 ----

    def _sync_from_tq(self) -> None:
        if not self._items:
            self.lbl.setText("没有品种可刷新")
            return
        if getattr(self, "_sync_thread", None) is not None:
            return
        pairs = [(it.get("symbol", ""), it.get("exchange", "")) for it in self._items
                 if it.get("symbol")]
        self._sync_worker = _SymSyncWorker(pairs)
        self._sync_thread = QThread(self)
        self._sync_worker.moveToThread(self._sync_thread)
        self._sync_thread.started.connect(self._sync_worker.run)
        self._sync_worker.progress.connect(self._on_sync_progress)
        self._sync_worker.done.connect(self._on_sync_done)
        self._sync_worker.failed.connect(self._on_sync_failed)
        self._sync_worker.done.connect(self._sync_thread.quit)
        self._sync_worker.failed.connect(self._sync_thread.quit)
        self._sync_thread.finished.connect(self._sync_cleanup)

        self.btn_sync.setEnabled(False)
        self.btn_sync.setText("正在取…")
        self.prog_sync.setVisible(True)
        self.prog_sync.setRange(0, 0)
        self.lbl.setText("连接天勤，逐个品种取实时乘数与保证金率…")
        self._sync_thread.start()

    def _on_sync_progress(self, done: int, total: int, msg: str) -> None:
        if total > 0:
            self.prog_sync.setRange(0, total)
            self.prog_sync.setValue(done)
        self.lbl.setText(msg)

    def _on_sync_done(self, st) -> None:
        st = st or {}
        self._fill()          # 表里已经写进去了，直接重读
        n = int(st.get("changed") or 0)
        same = int(st.get("same") or 0)
        miss = int(st.get("missing") or 0)
        errs = list(st.get("errors") or [])
        hints = list(st.get("fee_hint") or [])
        text = f"刷新完成：{n} 个品种已更新（乘数 / 最小跳动 / 保证金率 / 手续费）"
        if same:
            text += f"，{same} 个与现值一致（保证金率差异 <2%，跳过）"
        if miss:
            text += f"，{miss} 个没取到"
        if errs:
            # 取不到的必须点名：静默少写几个品种，回测手数就静默不对
            text += "　没取到：" + "、".join(errs[:4]) + ("…" if len(errs) > 4 else "")
        if hints:
            # 手续费「口径」变了要单独说：这不是调了个数，是成本随价格变化的
            # 规律变了，回测曲线形状都会不一样
            text += ("　手续费口径变更："
                     + "；".join(hints[:3])
                     + ("…" if len(hints) > 3 else ""))
        over = list(st.get("overrode") or [])
        if over:
            # 手填值被刷掉要点名：这张表现在同时决定实盘和回测的手数，
            # 用户填的 2% 被刷成 3.5% 之后两边手数会一起变
            text += ("　覆盖了你手填的值：" + "；".join(over[:3])
                     + ("…" if len(over) > 3 else ""))
        self.lbl.setText(text)
        self.lbl.setToolTip(text)

    def _on_sync_failed(self, msg: str) -> None:
        self.lbl.setText(f"刷新失败：{msg}")

    def _sync_cleanup(self) -> None:
        for attr in ("_sync_thread", "_sync_worker"):
            o = getattr(self, attr, None)
            if o is not None:
                o.deleteLater()
                setattr(self, attr, None)
        self.btn_sync.setEnabled(True)
        self.btn_sync.setText("从天勤刷新（乘数·跳动·保证金·手续费）")
        self.prog_sync.setVisible(False)

    def closeEvent(self, ev) -> None:      # noqa: N802 - Qt 命名
        """关窗时让后台线程收手，别让它在窗口析构之后还发信号。"""
        w = getattr(self, "_sync_worker", None)
        if w is not None:
            w.cancel()
        super().closeEvent(ev)


class RuleParamsDialog(QDialog):
    """一条规则策略的参数（稀疏采样里「规则触发」用的那一份）。

    和组合页「设置参数」里那段是同一套做法（整数用 QSpinBox、小数用
    QDoubleSpinBox，上下限来自 `spec.param_meta`），但**值是独立一份**：
    主面板那组滑杆属于「策略 = 这条规则」，这里属于「机器学习 + 规则触发」。
    两条路各调各的 —— 触发用双均线 5/20、主策略用 10/60 是完全正常的用法，
    共用一个 dict 只会互相盖。

    `rows` 是 `(key, 中文名, 下限, 上限, 步长, 默认值, 当前值)`。传默认值是为了
    「恢复默认」这个按钮 —— 没有它，用户把参数调乱了只能靠记忆改回去。
    """

    def __init__(self, label: str, rows, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"规则触发参数 · {label}")
        self.setMinimumWidth(360)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(8)

        tip = QLabel(
            "这条规则只负责「什么时候建样本、往哪边做」，不参与训练特征。\n"
            "参数和主策略那组滑杆是两套：改这里不会动到「策略」下拉里选的规则。"
        )
        tip.setStyleSheet(_HINT)
        tip.setWordWrap(True)
        lay.addWidget(tip)

        form = QFormLayout()
        form.setSpacing(6)
        self.sp: dict[str, QWidget] = {}
        self._defaults: dict[str, float] = {}
        for key, text, lo, hi, step, dflt, cur in rows:
            self._defaults[key] = dflt
            if isinstance(dflt, float):
                w = QDoubleSpinBox()
                w.setDecimals(2)
                w.setRange(float(lo), float(hi))
                w.setSingleStep(float(step))
            else:
                # 整数参数的上下限必须按 int 给：QSpinBox 拿到 (0.1, 1.5) 会夹成
                # (0, 1)，用户什么都没改，门槛就变 0 了（组合页踩过同一个坑）
                w = QSpinBox()
                w.setRange(int(lo), int(hi))
                w.setSingleStep(max(1, int(step)))
            try:
                w.setValue(float(cur) if isinstance(dflt, float) else int(cur))
            except (TypeError, ValueError):
                w.setValue(dflt)
            self.sp[key] = w
            form.addRow(text, w)
        lay.addLayout(form)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel
                               | QDialogButtonBox.Reset)
        box.button(QDialogButtonBox.Ok).setText("确定")
        box.button(QDialogButtonBox.Cancel).setText("取消")
        box.button(QDialogButtonBox.Reset).setText("恢复默认")
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        box.button(QDialogButtonBox.Reset).clicked.connect(self._reset)
        lay.addWidget(box)

    def _reset(self) -> None:
        for key, dflt in self._defaults.items():
            self.sp[key].setValue(dflt)

    def values(self) -> dict:
        out = {}
        for key, w in self.sp.items():
            v = w.value()
            out[key] = float(v) if isinstance(w, QDoubleSpinBox) else int(v)
        return out
