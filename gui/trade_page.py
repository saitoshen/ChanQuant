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

from datetime import datetime
import json
import os

from PySide6.QtCore import Qt, QTime, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QCompleter,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
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
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ttrade import symbols_info, tqlive, tqstore
from ttrade.ctp_client import PRESETS
from ttrade.tqlive import (
    ACCOUNT_KINDS,
    BACKEND_LABELS,
    BACKENDS,
    AccountSpec,
    LiveState,
)
from ttrade.tqsdk_symbols import contract_from_input, to_ts

# 与 app.py 的配色保持一致。这里不 import app 的常量：app 会 import 本模块，
# 顶层互相 import 会成环。
C_TEXT = "#1f2937"
C_MUTED = "#6b7280"
C_TOTAL_BG = "#eef2f7"  # 合计行底色（与组合页策略表最后一行同色）
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


def _fee_text(fee, est=False) -> str:
    """一笔成交的手续费。

    - `None` → 「—」。**不是 0**：拿不到乘数 / 费率没维护时算不出来，写 0.00
      等于告诉用户「这单免佣」，是撒谎。
    - 估算值前面带 `≈`（天勤回报自带 `commission`，CTP / 融航没有逐笔手续费，
      只能按品种费率倒推）。灰字 + `≈` 两个记号，一眼能看出这不是柜台给的数。
    """
    f = _num(fee)
    if f is None:
        return "—"
    # 手续费只会是正数，不画正号；异常负值也照原样显示，好排查
    return f"{'≈' if est else ''}{f:,.2f}"


def _px(v, sym: str = "") -> str:
    """盘口价：0 / NaN 都是「没有这一档」，画「—」而不是 0.00。

    小数位跟着这个合约的**最小变动价位**走（见 `symbols_info.price_text`）：
    铜写 90,040、国债写 118.115、棉花写 5,902 —— 不再一律 `:.2f`。
    """
    f = _num(v)
    return "—" if not f else symbols_info.price_text(f, sym)


def _hms(v) -> str:
    """行情时间 → `HH:MM:SS`。

    两个柜台的格式不一样：天勤给 `2026-11-18 14:35:02.000000`（带微秒），
    CTP 给 `14:35:02`。直接 `[-8:]` 在天勤那串上会截出 `2.000000` —— 所以先取
    空格后面那一段，再取前 8 个字符。
    """
    t = str(v or "").strip()
    if not t:
        return ""
    return t.rsplit(" ", 1)[-1][:8]


def _tick_decimals(tick: float) -> int:
    """最小变动价位要几位小数。实现在 `symbols_info` —— 组合页的价格列用的是
    同一个换算，两边各写一份早晚会走样。这里留个名字，自检按这个名字取。"""
    return symbols_info.tick_decimals(tick)


def _sym_info(sym: str) -> dict:
    """合约 → 品种表信息（`tick` / `mult` / `margin` …），查不到给空表。

    🔴 **裸合约代码（CTP 的 `IM2612`）必须先把交易所补上**（2026-11-23）：
    `symbols_info.info("IM", "")` 的 `tick` 是对的，但 **`mult` 会退回默认 1.0**
    —— 双击报价后那句「乘数 1」就是这么来的（IM 一手其实是 200 倍），用户按它
    算一手多少钱会差两个数量级。交易所从品种前缀表补，和 CTP 那边
    `_exchange_of()` 用的是同一张表（`ctp_live.variety_exchange`）。
    """
    vari, ex = symbols_info.contract_product(str(sym or ""))
    if not vari:
        return {}
    if not ex:
        try:
            from ttrade.ctp_live import variety_exchange

            ex = variety_exchange(vari) or ""
        except Exception:  # noqa: BLE001 - 补不出来就按无交易所查
            ex = ""
    try:
        return symbols_info.info(vari, ex)
    except Exception:  # noqa: BLE001
        return {}


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


# ---------------------------------------------------------------- 账户定时
#
# 「到点自动登录 / 登出」。两行、四个时刻，都是用户 2026-11-17 定的：
#
#     夜盘  登录 20:55   登出 02:35
#     日盘  登录 08:55   登出 15:20
#
# 登录时刻都留了 5 分钟余量（20:55 / 08:55）——柜台开盘前要握手、订阅、拉持仓，
# 卡在整点登录经常出现「第一根 bar 没收到」；登出时刻留 5 分钟（02:35 / 15:20）
# 让收盘那几笔成交和结算回推完再断。
#
# **夜盘登出跨零点**（02:35 在「第二天」）。判据不能写成「今天 02:35 之前」，
# 只能按「当前时刻往回推 grace 秒内有没有一个到点」，见 `acc_due()`。

ACC_TIMER_FILE = "account_timer.json"
# 到点之后多久之内还算「该动作」。程序 21:03 才打开，20:55 的登录仍然要补上；
# 但中午 12:00 打开不该把 08:55 那次再补一遍（那时补登录没意义）。
ACC_GRACE = 600

ACC_DEFAULTS = {
    "enabled": False,
    "night": ["20:55", "02:35"],     # [登录, 登出]
    "day": ["08:55", "15:20"],
    # 2026-11-24 用户要求：定时登出时连策略一起停，登录时再把策略开起来。
    # 默认关，老配置补 False，不改变既有行为。
    "start_all_on_login": False,
    "stop_all_on_logout": False,
}


def acc_timer_path() -> str:
    from ttrade.paths import user_dir

    return os.path.join(user_dir(), ACC_TIMER_FILE)


def load_acc_timer() -> dict:
    """读账户定时配置；坏的/缺的字段按默认值补齐。"""
    cfg = {k: (list(v) if isinstance(v, list) else v)
           for k, v in ACC_DEFAULTS.items()}
    try:
        with open(acc_timer_path(), encoding="utf-8") as f:
            d = json.load(f)
        if isinstance(d, dict):
            cfg["enabled"] = bool(d.get("enabled", cfg["enabled"]))
            for k in ("start_all_on_login", "stop_all_on_logout"):
                cfg[k] = bool(d.get(k, cfg[k]))
            for k in ("night", "day"):
                v = d.get(k)
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    cfg[k] = [str(v[0]), str(v[1])]
    except Exception:  # noqa: BLE001
        pass
    return cfg


def save_acc_timer(cfg: dict) -> None:
    try:
        path = acc_timer_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=1)
        os.replace(tmp, path)
    except Exception:  # noqa: BLE001
        pass


def acc_transitions(cfg: dict) -> list[tuple[str, str, str]]:
    """(kind, "HH:MM", 场次名) 四个时刻。`kind` 是 `login` / `logout`。"""
    out: list[tuple[str, str, str]] = []
    for key, label in (("day", "日盘"), ("night", "夜盘")):
        pair = cfg.get(key) or []
        if len(pair) != 2:
            continue
        out.append(("login", str(pair[0]), label))
        out.append(("logout", str(pair[1]), label))
    return out


def acc_due(cfg: dict, now: datetime, last_token: str = "",
            grace: int = ACC_GRACE):
    """当前时刻该不该动作。返回 `(kind, 标签, token)`，没有就 None。

    「到点」的判据是**往回看 grace 秒**：`now - 时刻` 落在 `[0, grace]` 内。
    这样夜盘 02:35 的登出（跨零点）和 20:55 的登录是同一套逻辑，不用分今天明天。

    `last_token` 是上次已经执行过的那个时刻（`YYYY-MM-DD HH:MM`），同一时刻不会
    执行第二遍 —— 心跳 20 秒一次，否则会连着发十几个登录请求。
    """
    if not cfg.get("enabled"):
        return None
    best = None
    for kind, hhmm, label in acc_transitions(cfg):
        try:
            hh, mm = [int(x) for x in hhmm.split(":")[:2]]
        except (ValueError, AttributeError):
            continue
        ts = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        delta = (now - ts).total_seconds()
        if delta < 0 or delta > grace:
            continue
        if best is None or ts > best[0]:
            best = (ts, kind, label, f"{ts:%Y-%m-%d %H:%M}")
    if best is None or best[3] == last_token:
        return None
    return best[1], best[2], best[3]


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

        # CTP 专用：柜台预设 + 两个前置 + 认证码
        self.cb_preset = QComboBox()
        for key, p in PRESETS.items():
            self.cb_preset.addItem(p.get("label", key), key)
        if spec and getattr(spec, "ctp_preset", ""):
            idx = self.cb_preset.findData(spec.ctp_preset)
            self.cb_preset.setCurrentIndex(max(idx, 0))
        self.ed_md = QLineEdit(getattr(spec, "md_front", "") if spec else "")
        self.ed_td = QLineEdit(getattr(spec, "td_front", "") if spec else "")
        self.ed_appid = QLineEdit(getattr(spec, "app_id", "") if spec else "")
        self.ed_auth = QLineEdit(getattr(spec, "auth_code", "") if spec else "")

        form.addRow("名称", self.ed_name)
        form.addRow("类型", self.cb_kind)
        self.lbl_user = QLabel("天勤账户")
        form.addRow(self.lbl_user, self.ed_user)
        self.lbl_pwd = QLabel("天勤密码")
        form.addRow(self.lbl_pwd, self.ed_pwd)
        # 标签要跟着柜台变：CTP 上它是经纪商代码（SimNow 是 9999），融航上它是
        # 资管后台给的经纪商编号，写成「期货公司代码」用户会不确定该填哪个
        self.lbl_broker = QLabel("期货公司代码")
        form.addRow(self.lbl_broker, self.ed_broker)
        form.addRow("资金账号", self.ed_acct)
        form.addRow("资金密码", self.ed_acct_pwd)
        self.lbl_preset = QLabel("CTP 柜台")
        form.addRow(self.lbl_preset, self.cb_preset)
        self.lbl_md = QLabel("行情前置")
        form.addRow(self.lbl_md, self.ed_md)
        self.lbl_td = QLabel("交易前置")
        form.addRow(self.lbl_td, self.ed_td)
        self.lbl_appid = QLabel("AppID")
        form.addRow(self.lbl_appid, self.ed_appid)
        self.lbl_auth = QLabel("授权码")
        form.addRow(self.lbl_auth, self.ed_auth)

        self.hint = QLabel(
            "快期模拟只需要天勤账户；实盘才需要期货公司代码与资金账号。\n"
            "股票模拟（TqSimStock / TqKqStock）用同一个天勤账户，但**数量单位是股**"
            "（100 股起、必须整百），方向只有买入 / 卖出，T+1 当天买的当天不能卖。\n"
            "CTP 柜台**不走天勤**：填柜台预设 + 投资者账号密码即可，前置地址留空就"
            "按预设补（SimNow 7x24 可以随时连）。限价 / 市价都支持（市价走"
            "AnyPrice + IOC，全部交易所已开放）。\n"
            "融航资管中台**也不走天勤**：用户名 / 密码 / 经纪商代码 / 交易服务器 / "
            "行情服务器 / 产品名称 / 授权编码这七项要填全，缺一项柜台就登不上。\n"
            "**不同柜台的账户可以同时勾选、一起登录**，那时上面会出现「行情源」"
            "下拉框，用来指定 K 线和盘口取哪一家（账户和持仓照旧全部合并显示）。\n"
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
        self.cb_preset.currentIndexChanged.connect(self._fill_preset)
        self._sync()

    def _fill_preset(self) -> None:
        """选预设时把前置地址填进去，手填过的不覆盖（留空才填）。"""
        key = self.cb_preset.currentData()
        p = PRESETS.get(key) or {}
        if not self.ed_md.text().strip():
            self.ed_md.setText(p.get("md_front", ""))
        if not self.ed_td.text().strip():
            self.ed_td.setText(p.get("td_front", ""))
        if not self.ed_appid.text().strip():
            self.ed_appid.setText(p.get("app_id", ""))
        if not self.ed_auth.text().strip():
            self.ed_auth.setText(p.get("auth_code", ""))
        if not self.ed_broker.text().strip():
            self.ed_broker.setText(p.get("broker_id", "9999"))

    def _sync(self) -> None:
        kind = self.cb_kind.currentData()
        real = kind == "account"
        ctp = kind == "ctp"
        rohon = kind == "rohon"
        counter = ctp or rohon          # 柜台：不走天勤，要前置地址那几项
        for w in (self.ed_broker, self.ed_acct, self.ed_acct_pwd):
            w.setEnabled(real)
        # 预设只有 CTP 有（SimNow 那几套）。融航的前置地址由资管后台给，
        # 拿 SimNow 的地址填进去只会连到一个不存在的服务器
        self.cb_preset.setVisible(ctp)
        self.lbl_preset.setVisible(ctp)
        for w in (self.ed_md, self.ed_td, self.ed_appid, self.ed_auth):
            w.setVisible(counter)
        for w in (self.lbl_md, self.lbl_td, self.lbl_appid, self.lbl_auth):
            w.setVisible(counter)
        # 柜台模式下「天勤账户」其实是柜台账号，标签得跟着变，否则用户会去填
        # 一个天勤账号然后一直登不上
        if ctp:
            self.lbl_user.setText("投资者账号")
            self.lbl_pwd.setText("投资者密码")
            self.lbl_broker.setText("经纪商代码")
            self.lbl_appid.setText("AppID")
            self.lbl_auth.setText("授权码")
            self.ed_user.setPlaceholderText("SimNow 账号")
        elif rohon:
            self.lbl_user.setText("融航用户名")
            self.lbl_pwd.setText("融航密码")
            self.lbl_broker.setText("经纪商代码")
            self.lbl_appid.setText("产品名称")
            self.lbl_auth.setText("授权编码")
            self.ed_user.setPlaceholderText("资管中台给的用户名")
        else:
            self.lbl_user.setText("天勤账户")
            self.lbl_pwd.setText("天勤密码")
            self.lbl_broker.setText("期货公司代码")
            self.lbl_appid.setText("AppID")
            self.lbl_auth.setText("授权码")
            self.ed_user.setPlaceholderText("")
        if counter:
            self.ed_broker.setEnabled(True)
        if ctp:
            self._fill_preset()

    def spec(self) -> AccountSpec:
        return AccountSpec(
            name=self.ed_name.text().strip() or "未命名",
            kind=self.cb_kind.currentData(),
            user=self.ed_user.text().strip(),
            password=self.ed_pwd.text(),
            broker_id=self.ed_broker.text().strip(),
            account_id=self.ed_acct.text().strip(),
            account_password=self.ed_acct_pwd.text(),
            ctp_preset=self.cb_preset.currentData() or "simnow_7x24",
            md_front=self.ed_md.text().strip(),
            td_front=self.ed_td.text().strip(),
            app_id=self.ed_appid.text().strip(),
            auth_code=self.ed_auth.text().strip(),
        )


class AccountTimerDialog(QDialog):
    """账户定时：到点自动登录 / 登出。

    两行四个时刻，第一行夜盘、第二行日盘，每行「登录 … 登出 …」。
    夜盘的登出会跨零点（20:55 登 → 次日 02:35 出），所以这里只收时刻，
    不让人填日期 —— 场次是按时刻判的，见 `acc_due()`。
    """

    def __init__(self, cfg: dict, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("账户定时登录 / 登出")
        self.setMinimumWidth(460)
        form = QFormLayout(self)

        self.chk = QCheckBox("启用（到点自动登录 / 登出，程序开着才生效）")
        self.chk.setChecked(bool(cfg.get("enabled")))
        self.chk.setToolTip(
            "登录用的是**账户列表里勾选的那几个账户**，和点「连接」完全一样；\n"
            "登出就是点「断开」。\n\n"
            "到点时如果已经连着（或已经断开），什么也不做，不会重复发请求。\n"
            "程序在那个时刻没开着，这一场就跳过 —— 不会攒着补一次。"
        )
        form.addRow(self.chk)

        # 策略联动（2026-11-24）。放在「启用」正下方：这两条一旦勾上，
        # 到点就不只是连/断账户，还会把组合页那一批策略整批起停。
        self.chk_start = QCheckBox("自动登录后，启动全部策略")
        self.chk_start.setChecked(bool(cfg.get("start_all_on_login")))
        self.chk_start.setToolTip(
            "账户定时登录成功之后，自动点组合页的「启动全部」。\n"
            "等的是**连接成功信号**，不是发出登录请求的那一刻 —— "
            "登录要几秒，早一步启策略会让它对着没连上的柜台下单。")
        form.addRow(self.chk_start)

        self.chk_stop = QCheckBox("自动登出前，停止全部策略")
        self.chk_stop.setChecked(bool(cfg.get("stop_all_on_logout")))
        self.chk_stop.setToolTip(
            "账户定时登出之前，先自动点组合页的「停止全部」。\n"
            "顺序是**先停策略、再断账户**：反过来会让策略对着已经断开的\n"
            "柜台继续跑几秒，最容易留下挂单和错位账本。")
        form.addRow(self.chk_stop)

        self.ed: dict[str, tuple[QTimeEdit, QTimeEdit]] = {}
        for key, label in (("night", "夜盘"), ("day", "日盘")):
            pair = cfg.get(key) or ACC_DEFAULTS[key]
            row = QHBoxLayout()
            row.setSpacing(6)
            row.addWidget(QLabel("登录"))
            a = QTimeEdit()
            a.setDisplayFormat("HH:mm")
            a.setTime(QTime.fromString(str(pair[0]), "HH:mm"))
            row.addWidget(a)
            row.addWidget(QLabel("登出"))
            b = QTimeEdit()
            b.setDisplayFormat("HH:mm")
            b.setTime(QTime.fromString(str(pair[1]), "HH:mm"))
            row.addWidget(b)
            row.addStretch(1)
            box = QWidget()
            box.setLayout(row)
            form.addRow(f"{label}（登录 / 登出）", box)
            self.ed[key] = (a, b)

        hint = QLabel(
            "默认：夜盘 20:55 登 → 02:35 出，日盘 08:55 登 → 15:20 出。\n"
            "登录留 5 分钟余量是给柜台握手、订阅行情、拉持仓；登出留 5 分钟是等\n"
            "收盘那几笔成交和结算回推完。\n\n"
            "只管登录 / 登出；勾了上面的「策略联动」才会顺带整批启停策略。\n"
            "夜盘登出跨零点（02:35），按时刻判，不用管它是哪天。"
        )
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        form.addRow(hint)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self._on_ok)
        box.rejected.connect(self.reject)
        form.addRow(box)

    def _on_ok(self) -> None:
        for key, label in (("night", "夜盘"), ("day", "日盘")):
            a, b = self.ed[key]
            if a.time() == b.time():
                QMessageBox.warning(
                    self, "时刻重复",
                    f"{label}的登录和登出是同一个时刻（{a.time().toString('HH:mm')}），"
                    "这样只会立刻断开。改一下再确定。")
                return
        self.accept()

    def result_cfg(self) -> dict:
        cfg = {"enabled": self.chk.isChecked(),
               "start_all_on_login": self.chk_start.isChecked(),
               "stop_all_on_logout": self.chk_stop.isChecked()}
        for key in ("night", "day"):
            a, b = self.ed[key]
            cfg[key] = [a.time().toString("HH:mm"), b.time().toString("HH:mm")]
        return cfg


class TradePage(QWidget):
    """账户透视 + 下单。"""

    # 参数是 `(specs, feed)`：勾选的账户 + 行情源（`tq` / `ctp` / `rohon`，
    # 单柜台时是空串）。柜台可以混选 —— 天勤和 CTP 同时登录就是靠这个
    connect_requested = Signal(object)
    disconnect_requested = Signal()
    # 只断开一个账户（账户表每行的「登出」）。参数是账户名。
    disconnect_one = Signal(str)
    # 账户定时 → 组合页整批启停（2026-11-24）。主窗口把这两个信号接到
    # `PortfolioPage._start_all` / `_stop_all`，交易页自己不碰策略状态。
    start_all_requested = Signal()
    stop_all_requested = Signal()

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
        self._ctp_mode = False
        # 账户定时（到点自动登录 / 登出）。配置存磁盘，`_acc_last` 只放内存：
        # 同一个时刻跑过就不再跑，重启程序后重跑一次也没关系（已经连着/断着
        # 就什么都不做）。
        self._acc_cfg = load_acc_timer()
        self._acc_last = ""
        # 定时登录是异步的：发出 connect_requested 时账户还没连上。等 `started`
        # 真的到了再启策略，这个标志就是「这一场登录之后要补一次启动全部」。
        self._acc_start_all_pending = False
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
        live.sessions.connect(self._on_sessions)
        live.order_sent.connect(self._on_order_sent)
        live.quote_got.connect(self._on_quote)
        # 20 秒看一眼。定时精度到分钟，没必要每秒唤醒；`acc_due()` 还有
        # 10 分钟宽限窗口，所以 20 秒的粒度不会漏掉任何一场。
        self._acc_timer = QTimer(self)
        self._acc_timer.setInterval(20000)
        self._acc_timer.timeout.connect(self._acc_tick)
        self._acc_timer.start()
        self._sync_acc_btn()

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
        #
        # 「报价」框和账户框并排：左边是「钱分在哪几个户里」，右边是「这些户里
        # 的合约现在什么价」。双击报价行 → 合约 + 最新价直接带到最下面的手动
        # 下单行，省掉「翻代码、抄价格」这两步最容易抄错的动作。
        top = QSplitter(Qt.Horizontal)
        top.setChildrenCollapsible(False)
        top.addWidget(self._wrap("账户", self._build_accounts(), with_btn="账户", hint=(
            "勾选决定连接哪些账户、上面资金卡和下面三张表统计哪些账户。"
            "双击某一行 = 只看这一个。"
        )))
        top.addWidget(self._wrap("报价", self._build_quotes(), with_btn="报价",
                                 hint="双击一行 → 合约和价格带到下面下单行"))
        top.setSizes([720, 400])
        root.addWidget(top, 0)

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

        # 账户定时：到点自动登录 / 登出。和「连接」并排 —— 用户想找的就是
        # 「怎么让它自己连」，放在同一行不用翻设置。
        self.btn_acc_timer = QPushButton("账户定时…")
        self.btn_acc_timer.clicked.connect(self._open_acc_timer)
        lay.addWidget(self.btn_acc_timer)

        self.lbl_dot = QLabel("●")
        self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
        lay.addWidget(self.lbl_dot)
        self.lbl_conn = QLabel("未连接")
        self.lbl_conn.setObjectName("Hint")
        lay.addWidget(self.lbl_conn)

        lay.addStretch(1)
        # 行情源：**独立于账户**的一条连接（2026-11-20 重构）。以前这个框跟着
        # 勾选的账户变 —— 账户一登出，行情就没了，策略跟着断。现在它有自己的
        # 连接 / 断开按钮，所有账户共用这一条。
        from .feed_bar import FeedBar

        self.feed_bar = FeedBar(self)
        self.feed_bar.changed.connect(self._on_feed_changed)
        lay.addWidget(self.feed_bar)
        # 老的下拉框（按勾选账户所属柜台生成）保留成**内部状态**，不再显示：
        # 它决定「登录会话的 K 线取哪家柜台」，跟着全局行情源走。显示两份
        # 「行情源」只会让用户以为是同一件事。
        self.lbl_feed = QLabel("行情源")
        self.lbl_feed.setObjectName("Hint")
        self.cb_feed = QComboBox()
        self.lbl_feed.setVisible(False)
        self.cb_feed.setVisible(False)
        lay.addWidget(self.lbl_feed)
        lay.addWidget(self.cb_feed)

        # 顶部不再单独摆一行报价：下面「报价」表里合约/最新/涨跌/买卖价/时间
        # 都有，同一份数字在屏幕上出现两次只会互相打架（2026-11-20 用户要求去掉）。
        # 这里原来挂 `lbl_quote`，留着空 QLabel 会在账户卡下方压出一条 20px 的空白。
        return card

    def _sync_feed_box(self) -> None:
        """按勾选刷新「行情源」下拉框。

        选项来自**勾选账户所属的柜台**（不是全部账户）：勾了天勤 + CTP 就让用户
        在这两家之间选，没勾的那个不该出现在选项里 —— 选了也连不上，只会白等
        一轮超时。
        """
        backs = []
        for s in self._checked_specs():
            b = getattr(s, "backend", "tq")
            if b not in backs:
                backs.append(b)
        backs = [b for b in BACKENDS if b in backs]
        keep = self.cb_feed.currentData()
        self.cb_feed.blockSignals(True)
        self.cb_feed.clear()
        for b in backs:
            self.cb_feed.addItem(BACKEND_LABELS.get(b, b), b)
        if keep in backs:
            self.cb_feed.setCurrentIndex(backs.index(keep))
        self.cb_feed.blockSignals(False)
        # 选谁：**跟全局行情源走**。全局源不在勾选账户的柜台里（比如只登了一个
        # CTP 账户、而全局源是天勤）时退回第一个可用的 —— 登录会话总得有个
        # 行情才能画 K 线。
        glob = self.feed_bar.current_source() if getattr(self, "feed_bar", None) else ""
        pick = glob if glob in backs else (keep if keep in backs
                                           else (backs[0] if backs else ""))
        if pick in backs:
            self.cb_feed.blockSignals(True)
            self.cb_feed.setCurrentIndex(backs.index(pick))
            self.cb_feed.blockSignals(False)
        # 永远藏着：界面上只有工具栏那一个「行情源」，这里是内部状态
        self.lbl_feed.setVisible(False)
        self.cb_feed.setVisible(False)

    def _on_feed_changed(self, key: str) -> None:
        """用户在工具栏切了行情源：同步给登录会话，别让它还按老柜台画 K 线。"""
        if not key:
            return
        self._sync_feed_box()
        self.on_log(f"行情源：{BACKEND_LABELS.get(key, key)}（账户登录状态不受影响）")

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
                ("新增", "新增账户", self._add_account),
                ("删除", "删除选中账户", self._del_account),
                ("编辑", "编辑选中账户", self._edit_account),
                ("全选", "全部勾选", lambda: self._check_all(True)),
                ("全不选", "全部取消勾选", lambda: self._check_all(False)),
                ("导入", "从 .cqa 文件导入账户配置", self._import_accounts),
                ("导出", "把账户配置导出为 .cqa 文件", self._export_accounts),
            ):
                b = QPushButton(text)
                b.setObjectName("Ghost")
                b.setToolTip(tip)
                b.clicked.connect(slot)
                head.addWidget(b)
        elif with_btn == "报价":
            for text, tip, slot in (
                ("添加", "把下单框里的合约加进报价框", self._watch_add),
                ("删除", "从报价框删掉选中行（持仓里的合约会自己回来）", self._watch_del),
            ):
                b = QPushButton(text)
                b.setObjectName("Ghost")
                b.setToolTip(tip)
                b.clicked.connect(slot)
                head.addWidget(b)
        lay.addLayout(head)
        lay.addWidget(table, 1)
        return card

    def _table(self, headers: list[str], check: bool = False,
               stretch: int | None = 0) -> QTableWidget:
        t = QTableWidget(0, len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.verticalHeader().setVisible(False)
        t.setEditTriggers(QAbstractItemView.NoEditTriggers)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        t.setSelectionMode(QAbstractItemView.SingleSelection)
        t.setShowGrid(False)
        t.setAlternatingRowColors(False)
        h = t.horizontalHeader()
        # **不要用 `ResizeToContents`**。它对每一次 `setItem` 都重算全部行的高度
        # 与宽度，而我们是整表重建（一天几百条委托）：同样的 200 行，第一次填
        # 28ms，第二次 12 秒 —— 挂上一笔新委托正好触发第二次，界面直接冻住。
        # 改成固定模式 + 填完调一次 `_fit_columns()`，同一场景 30ms。
        #
        # `stretch` 是「吃剩余宽度的那一列」。**不能默认第 0 列**：三张表第 0 列
        # 都是「资金账户」（就那么几个字），让它吃剩余的话整列被拉得极宽，真正
        # 该宽的「合约 / 说明」反而只有 90px。
        for i in range(len(headers)):
            h.setSectionResizeMode(i, QHeaderView.Interactive)
            t.setColumnWidth(i, 90)
        if stretch is not None and 0 <= stretch < len(headers):
            h.setSectionResizeMode(stretch, QHeaderView.Stretch)
        if check:
            t.setColumnWidth(0, 150)
        return t

    def _fit_columns(self, t: QTableWidget, skip: int = 0,
                     stretch: int | None = None) -> None:
        """填完之后按内容定一次列宽。

        `stretch` 那一列由 QHeaderView 自己吃剩余宽度，不能参与
        `resizeColumnToContents`（会把它的 Stretch 模式顶掉、钉成定宽）。
        200 行 8 列实测 30ms 上下。
        """
        for j in range(skip, t.columnCount()):
            if stretch is not None and j == stretch:
                continue
            t.resizeColumnToContents(j)

    def _clamp_col(self, t: QTableWidget, col: int, max_w: int) -> None:
        """按内容定宽，但封顶。

        「资金账户」列原先由 Stretch 吃掉全部剩余宽度，一列比整个表还宽；
        纯按内容算又可能被一个超长账户名顶开。所以先按内容、再封顶。
        """
        t.resizeColumnToContents(col)
        if t.columnWidth(col) > max_w:
            t.setColumnWidth(col, max_w)

    def _build_accounts(self) -> QTableWidget:
        # 第 1 列是**每行一个登录 / 登出按钮**。放在「账户」右边而不是最右边：
        # 这张表 11 列数据，横向早就超出卡片宽度了，最右边的按钮基本看不见。
        self.tbl_accts = self._table(
            ["账户", "连接", "账户平台", "资金账户", "最新权益", "当日盈亏",
             "净杠杆", "总杠杆", "可用资金", "成交金额", "持仓保证金", "手续费"],
            check=True,
        )
        self.tbl_accts.setMaximumHeight(150)
        self.tbl_accts.setToolTip(
            "净杠杆 = |多头市值 − 空头市值| / 权益；总杠杆 =（多头 + 空头市值）/ 权益。\n"
            "市值按「开仓成本 × 合约乘数 + 浮盈」推算，不订阅盘口。\n"
            "「连接」列每行一个按钮：单独登录 / 登出这一个账户，不影响其它账户。"
        )
        # **勾选状态存在 Python 集合里，不存在表格里**：表格每 0.4 秒刷一次数字，
        # 重建 item 会把勾选状态清掉，而「清掉 → itemChanged → 重算显示」正是
        # 上次界面卡死的那个环。这里只在第一次建行时给 item 打勾，之后只改文字。
        self._checked: set[str] = set()
        self._filling = False
        # 每行那个按钮：名字 → 按钮，`_sync_conn_buttons()` 统一改文字
        self._acct_btns: dict[str, QPushButton] = {}
        self.tbl_accts.itemChanged.connect(self._on_acct_check)
        self.tbl_accts.itemDoubleClicked.connect(self._on_acct_dbl)
        return self.tbl_accts

    def _build_positions(self) -> QTableWidget:
        # 「合约」列吃剩余宽度（`stretch=1`）：它是这一列里唯一长度不定的
        self.tbl_pos = self._table(
            ["资金账户", "合约", "净", "多", "空", "今多", "今空",
             "开仓均价", "浮动盈亏", "当日盈亏", "保证金"],
            stretch=1,
        )
        self.tbl_pos.setToolTip(
            "浮动盈亏 = 相对**开仓均价**的盈亏（开仓到现在）。\n"
            "当日盈亏 = (现价 − 昨收价) × 净手数 × 乘数，按**昨收**比（2026-11-26 改）。\n"
            "不是柜台逐日盯市那个按昨结算的 PositionProfit；拿不到昨收/现价时才退回柜台值。\n"
            "浮动盈亏仍按开仓均价算，两者在「按昨收开的仓」上才会相同。"
        )
        return self.tbl_pos

    def _build_orders(self) -> QTableWidget:
        # 「说明」列吃剩余宽度：拒单理由是这里唯一说不准长度的一列
        self.tbl_orders = self._table(
            ["资金账户", "时间", "合约", "方向", "开平", "委托", "剩余", "价格",
             "状态", "说明"],
            stretch=9,
        )
        return self.tbl_orders

    def _build_trades(self) -> QWidget:
        """「成交」框：明细表 + 「按品种统计」汇总表，同一个开关切。

        用户 2026-11-20：
        - 「成交框增加一栏『手续费』，显示每笔成交的手续费」
        - 「再在成交框下面增加一个统计选项，选中后可以显示按照品种来统计」

        两张表叠在 `QStackedWidget` 里而不是共用一个表格换表头：列数不同、
        该吃剩余宽度的那一列也不同（明细是「合约」，汇总是「品种」），换表头
        两套列宽会互相串。
        """
        self.tbl_trades = self._table(
            ["资金账户", "时间", "合约", "方向", "开平", "价格", "手数",
             "手续费", "平仓盈亏"],
            stretch=2,
        )
        # 先建好「上一次填进去的成交」再连 toggled：勾选框一旦在构建期被
        # `setChecked()`（将来恢复界面状态会这么做），`_on_trade_sum` 会立刻被
        # 调起来重绘，这时候 `_trade_rows` 还不存在就是 AttributeError。
        self._trade_rows: list[dict] = []
        self.tbl_trades.setToolTip(
            "平仓盈亏是**估算值**：两个柜台的成交回报里都没有盈亏字段，只能按\n"
            "「成交价 − 该方向开仓均价」× 手数 × 乘数倒推。\n"
            "开仓单记 0；开仓均价取不到时显示「—」（不是 0）。\n\n"
            "手续费：天勤回报里自带，直接用；CTP / 融航的回报没有逐笔手续费，\n"
            "按「数据管理 → 品种信息」里的费率估算（前面带 ≈）。只算手续费、\n"
            "不含滑点 —— 滑点是机会成本，不是账单。"
        )

        self.tbl_trade_sum = self._table(
            ["品种", "笔数", "手数", "手续费", "平仓盈亏"], stretch=0)
        self.tbl_trade_sum.setToolTip(
            "按品种把上面那张表汇总一遍。最后一行是合计。\n"
            "手续费前面的 ≈ 表示这一格里混了估算值（柜台没给逐笔手续费）；\n"
            "某一行有算不出来的那几笔时，数字后面带一个 *，表示只加了算得出来的。"
        )

        self.stack_trades = QStackedWidget()
        self.stack_trades.addWidget(self.tbl_trades)
        self.stack_trades.addWidget(self.tbl_trade_sum)

        self.chk_trade_sum = QCheckBox("按品种统计")
        self.chk_trade_sum.setToolTip(
            "勾上：按品种汇总笔数 / 手数 / 手续费 / 平仓盈亏；\n"
            "不勾：看逐笔成交（默认）")
        self.chk_trade_sum.toggled.connect(self._on_trade_sum)

        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        lay.addWidget(self.stack_trades, 1)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addStretch(1)
        row.addWidget(self.chk_trade_sum)
        lay.addLayout(row)

        return box

    def _on_trade_sum(self, on: bool) -> None:
        """切「明细 / 按品种统计」。两张表都已经填好，这里只翻页。"""
        self.stack_trades.setCurrentIndex(1 if on else 0)
        self._fill_trade_sum(self._trade_rows)

    def _build_quotes(self) -> QTableWidget:
        """「报价」框：盯着的合约各一行。

        盯哪些合约 = 持仓里的 + 下单框里填的 + 手动「＋」加的。持仓合约自动
        进来，不用一个个手打 —— 这是这个框最有用的地方。
        """
        # 「合约」列吃剩余宽度，其余固定
        self.tbl_quotes = self._table(
            ["合约", "最新", "涨跌", "买价", "卖价", "时间"], stretch=0)
        self.tbl_quotes.setMaximumHeight(150)
        self.tbl_quotes.setToolTip(
            "自动盯：持仓里的合约、下单框里填的合约。\n"
            "「＋」把下单框当前的合约加进来，「－」删掉选中行。\n"
            "**双击一行** = 合约和最新价直接填到底下的手动下单行。"
        )
        self.tbl_quotes.itemDoubleClicked.connect(self._on_quote_dbl)
        # 数字列钉定宽，**不按内容自适应**：这张卡片只有 300~450 像素，按内容算
        # （表头两个字 + 一列 2 位小数）会要 600 多，直接顶出一条横向滚动条，
        # 「合约」那列还被挤到只剩一半。这里给够显示 `-120.50` 的宽度就够了。
        for j, w in enumerate((70, 70, 62, 62, 70), start=1):
            self.tbl_quotes.setColumnWidth(j, w)
        # 手动加进来的合约（`SHFE.rb2601` 天勤写法）。持仓那部分是每轮从快照
        # 推的，不存在这里 —— 否则平了仓还得手动删，越积越多
        self._watch_extra: list[str] = []
        # 上一次真的发给柜台的那批合约。每轮比对，变了才重新订阅
        self._watch_sent: tuple = ()
        return self.tbl_quotes

    def _build_order_panel(self) -> QWidget:
        card = QWidget()
        card.setObjectName("Card")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(10)

        lay.addWidget(QLabel("合约"))
        # **只填真实合约代码**（`rb2601`），不带交易所前缀 —— 交易所由本地合约表
        # 反查，写错了也不至于下到一个「同名但另一个交易所」的合约上（那种错误
        # 柜台只会回一句「合约不存在」，很难反查）。
        self.cb_contract = QComboBox()
        self.cb_contract.setEditable(True)
        self.cb_contract.setInsertPolicy(QComboBox.NoInsert)
        self.cb_contract.setMinimumWidth(170)
        self.cb_contract.setToolTip(
            "填真实合约代码，如 rb2601（不用填交易所）。\n"
            "输入时会按本地合约表补全；选中 / 回车 / 移开焦点都会把该合约的\n"
            "最新价填到右边「价格」框。")
        self.cb_contract.lineEdit().setPlaceholderText("rb2601")
        lay.addWidget(self.cb_contract)

        # 候选按「用过的近期合约在前」排，用户打头两个字就该看到当月的主力。
        # 放在第一次打字时才读库：这张表要扫一百万行，程序启动时不该为它等。
        self._con_loaded = False
        self._con_map: dict[str, tuple[str, str]] = {}    # rb2601 → (SHFE.rb2601, SHFE)
        self._prod_ex: dict[str, str] = {}                # RB → SHFE
        self._order_symbol = ""                           # 当前解析出来的天勤写法
        self._priced_for = ""                             # 已经填过价的那个合约
        comp = self.cb_contract.completer()
        if comp is not None:
            comp.setCompletionMode(QCompleter.PopupCompletion)
            comp.setFilterMode(Qt.MatchStartsWith)
            comp.setCaseSensitivity(Qt.CaseInsensitive)
        self.cb_contract.lineEdit().textEdited.connect(
            lambda _t: self._ensure_contract_choices())
        # `activated`：点了候选 / 在输入框里按了回车；`editingFinished`：焦点离开。
        # 两个都要 —— 只连一个的话，「打完字直接用鼠标点下单」就永远不填价格。
        self.cb_contract.activated.connect(lambda _i: self._fill_price())
        self.cb_contract.lineEdit().editingFinished.connect(self._fill_price)

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

    # ---------------------------------------------------------- 合约 / 最新价

    def _ensure_contract_choices(self) -> None:
        """把本地合约表灌进下拉框（只灌一次）。

        跨品种一次读全：图上看着 rb、手上要下 m2601 是常事，候选不能只看当前
        品种。`all_contracts()` 自己带缓存，第一次约 0.2 秒、之后是空操作。
        """
        if self._con_loaded:
            return
        self._con_loaded = True
        text = self.cb_contract.currentText()
        bare: list[str] = []
        try:
            df = tqstore.all_contracts()
        except Exception:  # noqa: BLE001 - 没有本地库就退化成「手输」
            df = None
        if df is not None and not df.empty:
            # 最近还有数据的排前面：要下单的多半是当月 / 下月的合约
            df = df.sort_values("last", ascending=False)
            for row in df.itertuples(index=False):
                sym, ex = contract_from_input(str(row.contract))
                if not sym:
                    continue
                code = sym.rsplit(".", 1)[-1]
                self._con_map.setdefault(code, (sym, ex))
                self._prod_ex.setdefault(
                    str(getattr(row, "symbol", "") or "").upper(), ex)
                bare.append(code)
        self.cb_contract.blockSignals(True)
        self.cb_contract.clear()
        self.cb_contract.addItems(bare)
        self.cb_contract.setEditText(text)
        self.cb_contract.blockSignals(False)
        # 候选是「打字那一刻才读库」的：读完要把补全弹出框按当前已输入的内容
        # 重开一次，否则用户打第一个字时看到的候选还是空的
        comp = self.cb_contract.completer()
        if comp is not None and text:
            comp.setCompletionPrefix(text)
            comp.complete()

    def _exchange_of(self, text: str) -> str:
        """合约代码里的品种 → 交易所。`rb2601` → `SHFE`。

        光秃秃的合约代码本身不含交易所，猜错不会报错，只会把单子下到一个不存在
        的合约上。所以只认两种来源：本地合约表里的「品种 → 交易所」，以及
        `data_futures/*.csv` 的文件名（`RB_SHFE.csv`）。查不到就返回空，让界面
        说「不认这个合约」，不猜。
        """
        self._ensure_contract_choices()
        code = str(text or "").strip()
        if "." in code:
            code = code.rsplit(".", 1)[-1]
        letters = "".join(ch for ch in code if ch.isalpha()).upper()
        if not letters:
            return ""
        ex = self._prod_ex.get(letters, "")
        if ex:
            return ex
        try:
            from ttrade.ctp_live import variety_exchange

            return variety_exchange(letters)
        except Exception:  # noqa: BLE001
            return ""

    def _resolve_contract(self, text: str) -> tuple[str, str]:
        """界面上的合约 → `(天勤写法, 交易所)`。A 股原样返回（不变口径）。"""
        raw = str(text or "").strip()
        if not raw:
            return "", ""
        if self._stock_mode:
            return raw, ""
        return contract_from_input(raw, self._exchange_of(raw))

    def _fill_price(self) -> None:
        """合约确定之后，把它的最新价填进「价格」框。

        顺序：连着柜台 → 现取（天勤 `get_quote` / CTP 订阅等 tick）；没连着 →
        退回本地库里这个合约的最后一根收盘价。**两条路都不猜**：都拿不到就把
        价格框留着不动，并说清楚原因 —— 填一个 0 或者昨天的价进去，用户点下单
        就是一张废单。
        """
        text = self.cb_contract.currentText().strip()
        if not text:
            return
        sym, _ex = self._resolve_contract(text)
        if not sym:
            self.lbl_order_msg.setText(
                f"「{text}」不是能认的合约代码，填真实合约代码，如 rb2601")
            return
        self._order_symbol = sym
        # 🔴 最小变动价位**必须在下面所有 early return 之前设**。用户 2026-11-20
        # 报的「铜 8240 按一下箭头只跳 1」就是这么来的：以前这段写在「正在取价 /
        # 本地库没有 / 已经填过价」三个分支的后面，它们一 return，价格框就停在
        # 默认的步长 1 上。交易所一跳 10 元（铜）、1 元（螺纹）、0.5（部分品种），
        # 步长错了按一下要么跳两跳、要么得按十下。
        self._apply_tick(sym)
        # 同一个合约已经填过价就不再去取：`editingFinished` 在焦点离开时也会发，
        # 而取一次价要让 worker 线程等 1~3 秒（等行情推一帧），不能让切窗口
        # 这种动作反复触发
        if sym == self._priced_for and self.sp_price.value() > 0:
            return
        acc = self._order_account()
        if self.live.running and acc:
            if self.live.quote_of(sym, account=acc):
                self.lbl_order_msg.setText(f"正在取 {sym} 的最新价…")
                return
        if self._stock_mode:
            # A 股没有本地「真实合约」表可查，只能靠实时行情取价
            self.lbl_order_msg.setText(
                f"没取到 {text} 的最新价：连上柜台后会自动填，也可以手填价格")
            return
        self._fill_price_local(sym, text)

    def _tick_sym(self, und: str = "") -> str:
        """价格框的步长该按**哪个合约**算：合约框里手填的那个优先。

        🔴 不能拿报价里的合约（2026-11-23 用户报「上下键的跳动不对，很多都跑到
        10 了，连 SA、IM 也是」）：`st.quote` 可能是上一个合约的残留（刚改完合约、
        新报价还没到），按它设步长就停在上一个品种的 tick 上 —— 铜一跳 10，之后
        看什么品种都跳 10。合约框才是用户真正要下的那个合约。
        """
        return self.cb_contract.currentText().strip() or str(und or "")

    def _apply_tick(self, sym: str = "", tick: float = 0.0) -> float:
        """把「价格」框的上下调节幅度设成**这个合约的最小变动价位**。

        为什么必须跟着 tick 走：铜一跳 10 元、螺纹 1 元、还有 0.5 / 0.2 一档的
        品种。固定步长 1 的结果是「按一下跳两跳」（0.5 一档）或者「按十下才动
        一格」（铜）。`QDoubleSpinBox` 的上下按钮、滚轮、上下键全按 `singleStep`
        走，所以调这一个值就够。

        🔴 **给了 `sym` 就一律以品种表为准**（`symbols_info`），`tick` 只在表里
        查不到时兜底。以前是反过来的（调用方传进来的 tick 优先），而传进来的那个
        数可能是**上一个合约**的报价 tick —— 于是「铜之后看 IM，跳一档就是 10」。
        品种表也是保证金 / 手续费 / 追单跳数用的同一张表，口径必须一致。

        顺手把小数位对齐到 tick —— 铜不该显示 `8240.0000`。**但只在不丢信息时
        才减位数**：现值在目标位数下表示不出来（说明 tick 和现价不是一套），
        就保留原来的位数，宁可多显示两位，也不能把 3550.5 抹成 3551。

        返回最终用上的 tick（0 = 查不到，什么都没动）。
        """
        if sym:
            dinfo = _sym_info(sym)
            try:
                tbl = float(dinfo.get("tick") or 0)
            except (TypeError, ValueError):
                tbl = 0.0
            if tbl > 0:
                tick = tbl
        if not (tick > 0) and getattr(self, "_stock_mode", False):
            # A 股不在品种表里（那张表是期货的），但一档就是 0.01 元 ——
            # 留步长 1 会按一下就跳一块钱
            tick = 0.01
        if not (tick > 0 and tick == tick):
            return 0.0
        if abs(float(self.sp_price.singleStep()) - tick) > 1e-12:
            self.sp_price.setSingleStep(tick)
        want = _tick_decimals(tick)
        cur = float(self.sp_price.value())
        if abs(cur - round(cur, want)) > 1e-9:
            # 现价表示不出来 ⇒ 别动位数（动了就会悄悄改掉用户填的价）
            want = max(want, self.sp_price.decimals())
        if want != self.sp_price.decimals():
            self.sp_price.setDecimals(want)
        return tick

    def _fill_price_local(self, sym: str, text: str) -> None:
        """没有实时价时，退回本地库的最后一根收盘价。

        本地库是**日线为主**，所以这个价在盘中几乎肯定不是现价 —— 提示里必须
        写清「本地收盘 + 哪一天」，让人知道要自己改，而不是以为这就是现价。
        """
        try:
            ts_code = to_ts(sym, datetime.now())
            bar = tqstore.contract_last_bar(ts_code)
        except Exception:  # noqa: BLE001
            bar = None
        if not bar:
            self.lbl_order_msg.setText(
                f"没取到 {text} 的行情：连上柜台会自动取最新价，也可以手填价格")
            return
        if not self.chk_market.isChecked():
            self.sp_price.setValue(float(bar["close"]))
        self._priced_for = sym
        self.lbl_order_msg.setText(
            f"{sym} 本地最新收盘 {float(bar['close']):g}"
            f"（{bar['dt']:%m-%d}，未连柜台，盘中请自己核对）")

    def _on_quote(self, info: dict) -> None:
        """取价回执。**要对一下是不是当前那个合约** —— 用户可能已经改了。"""
        if not isinstance(info, dict):
            return
        sym = str(info.get("symbol") or "")
        want = getattr(self, "_order_symbol", "")
        if not sym or (want and sym.rsplit(".", 1)[-1] != want.rsplit(".", 1)[-1]):
            return
        last = float(info.get("last") or 0)
        if last <= 0:
            self._fill_price_local(want, self.cb_contract.currentText().strip())
            return
        tick = float(info.get("tick") or 0)
        if not self.chk_market.isChecked():
            self.sp_price.setValue(last)
        # 先落价再对齐位数：反过来会对着一根「上一个合约的价」判断要不要收小数，
        # 那一根表示不出来就白等一次行情
        self._apply_tick(sym, tick)
        self._priced_for = sym
        stamp = _hms(info.get("datetime"))
        self.lbl_order_msg.setText(
            f"{sym} 最新价 {last:g}" + (f"（{stamp}）" if stamp else ""))

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
        self._acct_btns = {}
        for i, s in enumerate(self._accounts):
            it = QTableWidgetItem(s.name)
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            it.setCheckState(Qt.Checked if s.name in self._checked else Qt.Unchecked)
            it.setData(Qt.UserRole, s.name)
            t.setItem(i, 0, it)
            # 第 1 列：这一行的登录 / 登出按钮。用 cell widget 而不是画在单元格里，
            # 这样点它不会连带改变选中行（改选中行会让「下单到哪个账户」跟着变）
            btn = QPushButton("登录")
            btn.setObjectName("Ghost")
            # 单元格只有 30px 高，全局按钮的内边距（上下各 6px + 22 行高）塞不下，
            # 文字会被压成一条。这里单独把内边距和字号钉小
            btn.setFixedSize(54, 22)
            btn.setStyleSheet("padding: 0px; font-size: 11px;")
            btn.clicked.connect(lambda _=False, n=s.name: self._toggle_account(n))
            t.setCellWidget(i, 1, btn)
            self._acct_btns[s.name] = btn
            # 没连上之前先按配置把静态列填好，不至于整行空白
            t.setItem(i, 2, QTableWidgetItem(dict(ACCOUNT_KINDS).get(s.kind, s.kind)))
            t.setItem(i, 3, QTableWidgetItem(s.account_id or s.user or "—"))
            for j in range(4, t.columnCount()):
                t.setItem(i, j, QTableWidgetItem("—"))
        self._filling = False
        self._fit_columns(t, skip=0)
        t.setColumnWidth(1, 66)      # 按钮列按内容算会太窄，钉一下
        self._sync_conn_buttons()
        self._sync_stock_mode()
        self._sync_feed_box()

    def _sync_stock_mode(self) -> None:
        """按勾选的账户决定下单面板是「手」还是「股」。

        股票账户：数量单位是股（100 股起、整百），开平由方向决定（买=开仓、
        卖=平仓），没有平今（T+1）。这些不是界面上的修饰，是交易所规则 ——
        界面上写「手」而实际下的是股，用户会按 100 倍下错单。
        """
        sel = self._checked_specs() or self._accounts
        stock = bool(sel) and all(s.is_stock for s in sel)
        # 柜台口径（CTP / 融航）统一，判据走 `backend`，别再逐家写 == "ctp"
        ctp = bool(sel) and all(getattr(s, "backend", "tq") != "tq" for s in sel)
        self._stock_mode = stock
        self._ctp_mode = ctp
        if not hasattr(self, "lbl_vol"):
            return
        # 市价单：柜台（CTP / 融航）**现在支持**了（2026-11-23，所有交易所都已
        # 开放，含上期所 / 能源所），所以不再禁用；A 股仍然没有市价单 ——
        # 交易所只有限价，勾了会被柜台拒，直接禁掉并说清原因
        self.chk_market.setEnabled(not stock)
        if stock:
            self.chk_market.setChecked(False)
            self.chk_market.setToolTip("A 股只有限价委托，没有市价单")
        elif ctp:
            self.chk_market.setToolTip(
                "市价单：价格字段留空，按 AnyPrice + IOC 报给柜台"
                "（全部交易所已开放，含上期所 / 能源所）")
        else:
            self.chk_market.setToolTip("")
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
            self.cb_contract.setToolTip(
                "填真实合约代码，如 rb2601（不用填交易所）。\n"
                "输入时会按本地合约表补全；选中 / 回车 / 移开焦点都会把该合约的\n"
                "最新价填到右边「价格」框。")
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
        # 勾选可能把「期货 + 股票」的混合变成纯股票（或反过来），也可能把
        # 「天勤 + CTP」变成只有一家 —— 下单面板的单位和行情源选项都要跟着变
        self._sync_stock_mode()
        self._sync_feed_box()
        self._sync_conn_buttons()      # 顶部「连接/断开」看勾选的这几个连没连
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
        self._sync_feed_box()
        self._sync_conn_buttons()
        self._refresh_view()

    def _check_all(self, on: bool) -> None:
        self._checked = {s.name for s in self._accounts} if on else set()
        self._filling = True
        for i in range(self.tbl_accts.rowCount()):
            it = self.tbl_accts.item(i, 0)
            if it is not None:
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
        self._filling = False
        self._sync_stock_mode()
        self._sync_feed_box()
        self._sync_conn_buttons()
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
        if self.live.is_running(spec.name):
            # 删配置不等于断开。先断开再删，免得留下一条「账户表里没有、但还连着」
            # 的会话 —— 那种会话的持仓会一直算在合计里，而界面上找不到它
            self.disconnect_one.emit(spec.name)
        self._accounts = [s for s in self._accounts if s.name != spec.name]
        self._checked.discard(spec.name)
        tqlive.save_accounts(self._accounts)
        self._reload_accounts()

    def _edit_account(self) -> None:
        """编辑选中账户的配置。名字可以改，但**不能和别的账户重名**。

        改完不会自动重连：正在跑的那条会话用的还是旧配置（旧密码 / 旧前置）。
        这里提示一句，用户点「登出→登录」才生效 —— 静默重连会在他不知道的时候
        换掉连接，正在挂的单也跟着没了。
        """
        spec = self._current_spec()
        if spec is None:
            self.on_failed("先在账户表里选中一行再点「✎」")
            return
        dlg = AccountDialog(spec, parent=self)
        if dlg.exec() != QDialog.Accepted:
            return
        new = dlg.spec()
        if new.name != spec.name and any(s.name == new.name for s in self._accounts):
            QMessageBox.warning(self, "名称重复",
                                f"已经有叫「{new.name}」的账户了，换个名字。")
            return
        idx = self._accounts.index(spec)
        was_checked = spec.name in self._checked
        was_running = self.live.is_running(spec.name)
        self._accounts[idx] = new
        if was_checked:
            self._checked.discard(spec.name)
            self._checked.add(new.name)
        if was_running and new.name != spec.name:
            # 改名等于换了一条会话的键：旧的断开，新的按新名字重登
            self.disconnect_one.emit(spec.name)
        tqlive.save_accounts(self._accounts)
        self._reload_accounts()
        self._select_row(new.name)
        if was_running and new.name == spec.name:
            self.on_log(f"已保存「{new.name}」的配置；当前连接仍在用旧配置，"
                        "点「登出」再「登录」才生效")
        else:
            self.on_log(f"已保存账户「{new.name}」")

    # ---- 账户配置导入 / 导出（.cqa） ----

    def _export_accounts(self) -> None:
        """把整张账户表存成 .cqa。密码明文随包走，先确认再保存。"""
        if not self._accounts:
            QMessageBox.information(self, "导出账户配置", "本地还没有账户，先点「＋」添加。")
            return
        ans = QMessageBox.warning(
            self, "导出账户配置",
            "导出的 .cqa 文件会原样包含资金账号和密码（明文）。\n"
            "请只保存在自己可控的位置，不要随意外发或提交到代码仓库。\n\n"
            "继续导出吗？",
            QMessageBox.Ok | QMessageBox.Cancel, QMessageBox.Cancel)
        if ans != QMessageBox.Ok:
            return
        from ttrade import portable as PORT
        from ttrade.paths import user_dir

        path, _ = QFileDialog.getSaveFileName(
            self, "导出账户配置", os.path.join(user_dir(), "交易账户.cqa"),
            "账户包 (*.cqa)")
        if not path:
            return
        if not path.lower().endswith(".cqa"):
            path += ".cqa"
        try:
            n = PORT.export_accounts(self._accounts, path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        msg = f"已导出 {n} 个账户到 {path}"
        QMessageBox.information(self, "导出账户配置", msg)
        self.on_log(msg)

    def _import_accounts(self) -> None:
        """合并 .cqa：同名覆盖、新名追加；定时配置是本机排班，不跟着导。"""
        from ttrade import portable as PORT
        from ttrade.paths import user_dir

        path, _ = QFileDialog.getOpenFileName(
            self, "导入账户配置", user_dir(), "账户包 (*.cqa)")
        if not path:
            return
        before = {s.name for s in tqlive.load_accounts()}
        try:
            rep = PORT.import_accounts(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        self._accounts = tqlive.load_accounts()
        added = [s.name for s in self._accounts if s.name not in before]
        self._checked.update(added)
        self._reload_accounts()
        msg = "导入账户配置：" + rep.summary()
        if added:
            msg += "\n新增账户：" + "、".join(added)
        if rep.ok:
            msg += "\n\n已连接的账户仍在用旧配置，改过参数的账户请「登出」再「登录」。"
        QMessageBox.information(self, "导入账户配置", msg)
        self.on_log(msg)

    # ---- 逐个账户登录 / 登出 ----

    def _toggle_account(self, name: str) -> None:
        """点击某一行前面的按钮：只登录 / 只登出这一个账户。"""
        spec = next((s for s in self._accounts if s.name == name), None)
        if spec is None:
            return
        if self.live.is_running(name):
            self.disconnect_one.emit(name)
            self.on_log(f"正在断开「{name}」…")
            return
        # 单独登录也要勾上：不勾的话它连着但资金卡和下面三张表都不统计它，
        # 看着像「登录了但什么都没出来」
        if name not in self._checked:
            self._checked.add(name)
            self._tick_row(name, True)
            self._sync_stock_mode()
            self._sync_feed_box()
        self.connect_requested.emit(([spec], self.cb_feed.currentData() or ""))
        self.on_log(f"正在登录「{name}」…")

    def _tick_row(self, name: str, on: bool) -> None:
        for i, s in enumerate(self._accounts):
            if s.name != name:
                continue
            it = self.tbl_accts.item(i, 0)
            if it is not None:
                self._filling = True
                it.setCheckState(Qt.Checked if on else Qt.Unchecked)
                self._filling = False
            return

    def _sync_conn_buttons(self) -> None:
        """按会话现状刷每行的「登录 / 登出」和顶部「连接 / 断开」。

        **不能靠 `on_state` 驱动**：一条会话都没有的时候不会有快照，按钮就会停在
        上一次的文字上（看着像还连着）。所以启动、断开、勾选变化都要调一遍。
        """
        for s in self._accounts:
            btn = self._acct_btns.get(s.name)
            if btn is None:
                continue
            on = self.live.is_running(s.name)
            text = "登出" if on else "登录"
            if btn.text() != text:
                btn.setText(text)
            # 样式每次都写一遍（不是只在文字变了才写）：上面的尺寸口径必须留着，
            # 只改颜色的话内边距会被全局按钮样式顶回去，文字又压成一条
            btn.setStyleSheet(
                "padding: 0px; font-size: 11px;"
                + (f"color: {C_BAD};" if on else ""))
            btn.setToolTip(
                ("断开这一个账户，其它账户照跑" if on else "单独登录这一个账户")
                + f"\n（{BACKEND_LABELS.get(s.backend, s.backend)}）")
        if not hasattr(self, "btn_connect"):
            return
        specs = self._checked_specs()
        all_on = bool(specs) and all(self.live.is_running(s.name) for s in specs)
        self.btn_connect.setText("断开" if all_on else "连接")
        self.btn_connect.setToolTip(
            "断开**勾选的这些**账户" if all_on else "登录勾选的账户（已连着的不会重复登）")

    def _select_row(self, name: str) -> None:
        for i, s in enumerate(self._accounts):
            if s.name == name:
                self.tbl_accts.selectRow(i)
                return

    def _toggle_connect(self) -> None:
        """顶部那个按钮：整批登录 / 整批断开**勾选的账户**。

        已经有「每行一个按钮」之后，这里不能再写成「只要有一条会话在跑就算连着」
        —— 单独登了一个账户，顶部按钮就会变成「断开」，一点就把别的户全关了。
        判据改成「勾选的这些是不是都连着」。
        """
        specs = self._checked_specs()
        if not specs:
            self.on_failed("先勾选至少一个账户；没有账户就点「＋」添加")
            return
        if all(self.live.is_running(s.name) for s in specs):
            if len(self.live.running_names) > len(specs):
                # 有勾选外的账户还连着：说清楚断开的是哪些，别说成「全断开」
                self.on_log(f"正在断开勾选的 {len(specs)} 个账户（其余保持连接）")
            self.disconnect_requested.emit()
            return
        # 行情源只对「多柜台」有意义；单柜台时下拉框是藏着的，`currentData()`
        # 可能就是 None，传空串让 worker 自己按顺序挑
        feed = self.cb_feed.currentData() or ""
        self.connect_requested.emit((specs, feed))

    # ---- 账户定时 ----

    def _open_acc_timer(self) -> None:
        dlg = AccountTimerDialog(self._acc_cfg, self)
        if dlg.exec() != QDialog.Accepted:
            return
        self._acc_cfg = dlg.result_cfg()
        save_acc_timer(self._acc_cfg)
        self._sync_acc_btn()
        self.on_log(
            "账户定时：" + ("已开启" if self._acc_cfg["enabled"] else "已关闭")
            + "　夜盘 {} 登 / {} 出，日盘 {} 登 / {} 出".format(
                *self._acc_cfg["night"], *self._acc_cfg["day"])
            + ("；登录启动全部" if self._acc_cfg["start_all_on_login"] else "")
            + ("；登出停止全部" if self._acc_cfg["stop_all_on_logout"] else ""))

    def _sync_acc_btn(self) -> None:
        """按钮上直接标出开没开，不用点开才知道。"""
        cfg = self._acc_cfg
        on = bool(cfg.get("enabled"))
        self.btn_acc_timer.setText("账户定时（已开）" if on else "账户定时…")
        self.btn_acc_timer.setToolTip(
            "到点自动登录 / 登出（程序开着才生效）。\n"
            f"夜盘：{cfg['night'][0]} 登录 → {cfg['night'][1]} 登出\n"
            f"日盘：{cfg['day'][0]} 登录 → {cfg['day'][1]} 登出\n"
            "登录用的是账户列表里勾选的账户；登出就是断开。\n"
            "已连着 / 已断着时不重复动作。\n"
            + (f"登录 → 启动全部策略：{'开' if cfg.get('start_all_on_login') else '关'}\n"
               f"登出 → 停止全部策略：{'开' if cfg.get('stop_all_on_logout') else '关'}")
            + ("" if on else "\n\n当前：**未启用**（点开勾上「启用」）"))

    def _acc_tick(self) -> None:
        due = acc_due(self._acc_cfg, datetime.now(), self._acc_last)
        if due is None:
            return
        kind, label, token = due
        self._acc_last = token
        hhmm = token[-5:]
        if kind == "login":
            specs = self._checked_specs()
            if not specs:
                self.on_log(f"账户定时：{label} {hhmm} 到点，但没有勾选账户，跳过")
                return
            # 只补**还没连上**的那些：逐账户登录之后，「已经连着」是按账户算的，
            # 不能再用一条总开关判断 —— 那样手动登过一个户，到点就会整批跳过
            todo = [s for s in specs if not self.live.is_running(s.name)]
            if not todo:
                self.on_log(f"账户定时：{label} {hhmm} 到点，勾选的账户都连着，不重复登录")
                return
            self.connect_requested.emit((todo, self.cb_feed.currentData() or ""))
            start_all = bool(self._acc_cfg.get("start_all_on_login"))
            self._acc_start_all_pending = start_all
            self.on_log(f"账户定时：{label} {hhmm} 自动登录（{len(todo)} 个账户）"
                        + ("，连接成功后启动全部策略" if start_all else ""))
        else:
            if not self.live.running:
                return
            # 先停策略、再断账户：反过来那几秒策略仍会对着死会话发单。
            # `_stop_all()` 内部自己判「本来就没在跑」，这里重复发也无害。
            if self._acc_cfg.get("stop_all_on_logout"):
                self._acc_start_all_pending = False
                self.stop_all_requested.emit()
                self.on_log(f"账户定时：{label} {hhmm} 先停止全部策略")
            self.disconnect_requested.emit()
            self.on_log(f"账户定时：{label} {hhmm} 自动断开")

    def _on_started(self, key: str) -> None:
        self._sync_conn_buttons()
        self._sync_conn_label()
        # 定时登录是异步的，只有在 `started` 到了之后才算「真的连上」。
        # 第一个连上的账户到了就启动全部 —— 多账户是一批一起连的，等最后一户
        # 没有可靠的「全部完成」信号，早一点比晚一点安全（策略启动本身会
        # 检查账户会话）。
        if self._acc_start_all_pending:
            self._acc_start_all_pending = False
            self.start_all_requested.emit()
            self.on_log(f"账户定时：{key} 连接成功，启动全部策略")

    def _on_stopped(self) -> None:
        self._sync_conn_buttons()
        self._sync_conn_label()
        # 这次登录没连成就别把「启动全部」留到下一次手动连接时才补 ——
        # 那是几小时后完全另外一个场景，突然把策略开起来最危险
        self._acc_start_all_pending = False
        # 断开后把数字清掉：留着上一次的权益和持仓，看着像还连着
        self._sig.clear()
        self._reload_accounts()
        self._state = LiveState()
        self.on_state(self._state, force=True)

    def _on_sessions(self, names) -> None:
        """连着哪些账户变了（单独登录 / 登出也走这里）。"""
        self._sync_conn_buttons()
        self._sync_conn_label()

    def _sync_conn_label(self) -> None:
        """顶部状态灯和文字。判据是**会话注册表**，不是「有没有快照」。"""
        names = set(self.live.running_names)
        if not names:
            self.lbl_dot.setStyleSheet(f"color: {C_IDLE}; font-size: 15px;")
            self.lbl_conn.setText("未连接")
            self.lbl_conn.setStyleSheet(f"color: {C_MUTED}; font-size: 11px;")
            return
        self.lbl_dot.setStyleSheet(f"color: {C_OK}; font-size: 15px;")
        if len(names) == 1:
            text = f"已连接 · {next(iter(names))}"
        else:
            text = f"已连接 · {len(names)} 个账户"
        # 多柜台时把行情源写在状态里：不然用户看到「K 线不动」会以为断线，
        # 其实是另一家柜台今天没推
        backs = []
        for s in self._checked_specs():
            b = getattr(s, "backend", "tq")
            if b not in backs:
                backs.append(b)
        if len(backs) > 1:
            text += f" · 行情 {BACKEND_LABELS.get(self.cb_feed.currentData(), '')}".rstrip()
        self.lbl_conn.setText(text)
        self.lbl_conn.setStyleSheet(f"color: {C_OK}; font-size: 11px;")

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
            # 🔴 这里**只留下单面板要用到的副作用**（最小变动价位 / 首帧价格），
            # 不再往顶部写一行报价文本 —— 报价框那张表才是唯一出处
            # （2026-11-20 用户要求；两处同时显示时小数位/量口径不一致，看着
            # 像行情在跳）。
            #
            # 🔴 也**不再自动改合约框**（2026-11-23 用户要求「合约框只保留双击
            # 报价后显示合约信息 + 手动敲入补全」）：以前行情推送会把框里的合约
            # 换成报价里的那个（`_auto_contract`），用户正在填的单子被悄悄换掉
            # 是能出人命的 —— 合约那条红线只能由人动手改。
            und = q.get("underlying")
            if q.get("price_tick"):
                if self.sp_price.value() <= 0:
                    self.sp_price.setValue(float(last or 0))
            # 价格框的上下调节幅度 = **合约框里那个合约**的最小变动价位。
            # 不能拿报价里的 tick 直接设：`st.quote` 可能还是上一个合约的
            # （刚改完合约、新报价还没到），步长就会停在上一个品种上 ——
            # 用户看到的「很多都跑到 10 了，连 SA、IM 也是」（10 是之前看过的铜）
            self._apply_tick(self._tick_sym(und))

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
                   round(r.get("float_profit") or 0, 2),
                   round(r.get("position_profit") or 0, 2),
                   round(r.get("margin") or 0, 2))
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
            tuple((r.get("account"), r.get("trade_id"),
                   round(r.get("close_profit") or 0, 2)) for r in trds),
            force,
        )

        # 报价框：数据来自 worker 盯的那批合约，盯哪些由「持仓 + 下单框 +
        # 手动加的」决定。先同步订阅，再画表 —— 反过来的话刚加的合约要等
        # 下一轮（0.4 秒后）才出现，看着像没加上
        self._sync_watch(pos)
        quotes = st.quotes or []
        self._fill_if_changed(
            "quo", quotes, self._fill_quotes,
            tuple((r.get("symbol"), round(r.get("last") or 0, 3),
                   round(r.get("bid") or 0, 3), round(r.get("ask") or 0, 3),
                   r.get("datetime")) for r in quotes),
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
                (2, r.get("platform") or ""),
                (3, r.get("account_id") or ""),
                (4, _money(r.get("balance"))),
                (5, _pnl(r.get("day_profit"))),
                (6, f"{float(r.get('net_leverage') or 0):.2f}"),
                (7, f"{float(r.get('gross_leverage') or 0):.2f}"),
                (8, _money(r.get("available"))),
                (9, _money(r.get("turnover"))),
                (10, _money(r.get("margin"))),
                (11, _money(r.get("commission"))),
            ]
            for j, v in vals:
                it = self.tbl_accts.item(i, j)
                if it is None:
                    it = QTableWidgetItem(v)
                    self.tbl_accts.setItem(i, j, it)
                elif it.text() != v:
                    it.setText(v)
                if j >= 4:
                    it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if j == 5:
                    it.setForeground(QColor(_color(r.get("day_profit"))))
        self._filling = False
        self._sync_conn_buttons()

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
            day = r.get("position_profit")
            vals = [
                r.get("account", ""), r["symbol"], str(r["pos"]),
                str(r["pos_long"]), str(r["pos_short"]),
                str(r["pos_long_today"]), str(r["pos_short_today"]),
                _avg_price(r),
                _pnl(r.get("float_profit")),
                _pnl(day),
                _money(r["margin"]),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                # 8 = 浮动盈亏、9 = 当日盈亏，各自按自己的数着色（两者可能一红一绿）
                if j == 8:
                    item.setForeground(QColor(_color(r.get("float_profit"))))
                elif j == 9:
                    item.setForeground(QColor(_color(day)))
                if j > 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)
        self._fit_columns(t, stretch=1)
        self._clamp_col(t, 0, 110)

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
        self._fit_columns(t, stretch=9)
        self._clamp_col(t, 0, 110)

    def _fill_trades(self, rows: list[dict]) -> None:
        t = self.tbl_trades
        self._trade_rows = list(rows or [])
        t.setRowCount(len(self._trade_rows))
        for i, r in enumerate(self._trade_rows):
            pnl = r.get("close_profit")
            fee = r.get("commission")
            vals = [
                r.get("account", ""), r.get("time", ""),
                r["symbol"], r["direction"], r["offset"],
                symbols_info.price_text(r["price"], r["symbol"]),
                str(r["volume"]),
                _fee_text(fee, r.get("commission_est")),
                _pnl(pnl),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                if j == 8:
                    item.setForeground(QColor(_color(pnl)))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                elif j == 7:
                    # 估算值用灰字：一眼能看出「这不是柜台给的钱」
                    item.setForeground(
                        QColor(C_MUTED if r.get("commission_est") else C_TEXT))
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    item.setToolTip(
                        "这一笔是按品种费率估算的（柜台回报里没有逐笔手续费）"
                        if r.get("commission_est") else "柜台回报里的手续费")
                elif j in (5, 6):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)
        self._fit_columns(t, stretch=2)
        self._clamp_col(t, 0, 110)
        self._fill_trade_sum(self._trade_rows)

    def _fill_trade_sum(self, rows: list[dict]) -> None:
        """按**品种**把成交汇总成一张表：笔数 / 手数 / 手续费 / 平仓盈亏。

        品种取合约代码里的字母部分（`rb2601` / `SHFE.rb2601` 都是 `RB`），认不出
        来的保留原合约名 —— 宁可显示一行奇怪的，也不能把这几笔丢掉。

        🔴 手续费 / 平仓盈亏有「算不出来」（`None`）的那几笔：合计只加算得出来的
        部分，并在数字后面打 `*`，和策略表最后那行「合计」同一套口径。写「—」
        等于整行没有，按 0 算又是撒谎。
        🔴 `fee_est` 的意思统一成「这个数里**有**估算成分」（任一行为估算即真），
        不是「全是估算」：5.00 真单 + 3.50 估算混在一起写个不带 `≈` 的 8.50，
        等于把猜的那半也算成柜台给的。合计行同理，按「有」往下传。
        最后一行是全部品种的合计。
        """
        t = self.tbl_trade_sum
        agg: dict[str, dict] = {}
        for r in rows or ():
            sym = str(r.get("symbol") or "")
            key = symbols_info.contract_product(sym)[0] or sym or "—"
            a = agg.setdefault(key, {"n": 0, "vol": 0, "fee": 0.0, "fee_ok": True,
                                     "fee_est": False, "pnl": 0.0, "pnl_ok": True})
            a["n"] += 1
            a["vol"] += int(r.get("volume") or 0)
            fee = r.get("commission")
            if fee is None:
                a["fee_ok"] = False
            else:
                a["fee"] += float(fee)
                if r.get("commission_est"):
                    a["fee_est"] = True
            pnl = r.get("close_profit")
            if pnl is None:
                a["pnl_ok"] = False
            else:
                a["pnl"] += float(pnl)

        keys = sorted(agg)
        tot = {"n": 0, "vol": 0, "fee": 0.0, "fee_ok": True, "fee_est": False,
               "pnl": 0.0, "pnl_ok": True}
        t.setRowCount(len(keys) + (1 if keys else 0))
        for i, k in enumerate(keys):
            a = agg[k]
            self._sum_row(t, i, k, a)
            for f in ("n", "vol"):
                tot[f] += a[f]
            tot["fee"] += a["fee"]
            tot["pnl"] += a["pnl"]
            tot["fee_ok"] = tot["fee_ok"] and a["fee_ok"]
            tot["pnl_ok"] = tot["pnl_ok"] and a["pnl_ok"]
            tot["fee_est"] = tot["fee_est"] or a["fee_est"]
        if keys:
            self._sum_row(t, len(keys), "合计", tot, bold=True)
        self._fit_columns(t, stretch=0)

    def _sum_row(self, t, i: int, name: str, a: dict, bold: bool = False) -> None:
        """汇总表的一行。`*` = 有算不出来的那几笔，只加了能加的。"""
        fee = "—" if not a["fee_ok"] and not a["fee"] else (
            f"{'≈' if a['fee_est'] else ''}"
            f"{a['fee']:,.2f}{'' if a['fee_ok'] else '*'}")
        pnl = "—" if not a["pnl_ok"] and not a["pnl"] else (
            f"{a['pnl']:+,.2f}{'' if a['pnl_ok'] else '*'}")
        vals = [name, str(a["n"]), str(a["vol"]), fee, pnl]
        for j, v in enumerate(vals):
            item = QTableWidgetItem(v)
            if j:
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if bold:
                f = item.font()
                f.setBold(True)
                item.setFont(f)
                item.setBackground(QColor(C_TOTAL_BG))
            if j == 4:
                item.setForeground(QColor(_color(a["pnl"] if a["pnl_ok"] else None)))
            elif j == 3:
                item.setForeground(QColor(C_MUTED if a["fee_est"] else C_TEXT))
            t.setItem(i, j, item)
        if bold:
            t.item(i, 0).setToolTip("全部品种的合计（只加算得出来的那部分）")

    # ---------------------------------------------------------- 报价框

    def _watch_add(self) -> None:
        """「＋」：把下单框里的合约加进报价框。"""
        sym, _ex = self._resolve_contract(self.cb_contract.currentText())
        if not sym:
            self.lbl_order_msg.setText(
                "合约框里没有能认的合约代码，先填 rb2601 这种再点「＋」")
            return
        if sym not in self._watch_extra:
            self._watch_extra.append(sym)
        # 清掉「上次发过的」，下一轮 on_state 会比出差异重新订阅
        self._watch_sent = ()
        self.lbl_order_msg.setText(f"报价框已加入 {sym}")

    def _watch_del(self) -> None:
        """「－」：从报价框删掉选中行。

        持仓里的合约删不掉（删了下一轮又会被持仓推回来），这种情况直接说明
        原因 —— 否则用户会以为按钮坏了。
        """
        row = self.tbl_quotes.currentRow()
        if row < 0:
            self.lbl_order_msg.setText("先在报价框里选中一行")
            return
        item = self.tbl_quotes.item(row, 0)
        sym = str(item.data(Qt.UserRole) or "") if item is not None else ""
        if not sym:
            return
        if sym in self._watch_extra:
            self._watch_extra.remove(sym)
            self._watch_sent = ()
            self.lbl_order_msg.setText(f"已从报价框删掉 {sym}")
        else:
            self.lbl_order_msg.setText(
                f"{sym} 是持仓 / 下单框里的合约，平仓或改掉合约后会自动消失")

    def _on_quote_dbl(self, item) -> None:
        """双击报价行 → 合约和最新价填进最底下的手动下单行。

        这是**唯一**会自动改「合约」框的入口（用户 2026-11-23 明确要求）：
        合约是能出人命的一条线，只接受**人的动作**（双击报价行 / 手敲合约名）。
        行情推送、切账户、持仓变化都不许替用户换合约。

        填的是**裸合约代码**（`rb2601`），和下单面板的口径一致（交易所在
        `_resolve_contract` 里反查）。勾了「市价」就不动价格框 —— 市价单
        填数字没有意义，改了反而让人以为那个价会生效。
        """
        row = item.row() if item is not None else -1
        if row < 0:
            return
        cell = self.tbl_quotes.item(row, 0)
        if cell is None:
            return
        code = cell.text().strip()
        sym = str(cell.data(Qt.UserRole) or "") or code
        if not code:
            return
        self.cb_contract.setCurrentText(code)
        # 步长跟着这个合约走（`_apply_tick` 内部查品种表，报价的 tick 只兜底）
        tick = self._apply_tick(code)
        self._order_symbol = sym
        # 「显示合约信息」：双击之后不只填价，顺手把**这个合约自己的**一档 / 乘数
        # 说出来 —— 用户接下来要按上下键调价，得知道一格是多少钱。
        # 🔴 查乘数要走 `_sym_info()`：裸代码 `IM2612` 直接查表会给默认的 1.0，
        # 界面上写「乘数 1」而 IM 一手是 200 倍
        dinfo = _sym_info(code) or _sym_info(sym)
        info = f"一跳 {tick:g}" if tick > 0 else "一档未知（品种表里没有）"
        try:
            mult = float(dinfo.get("mult") or 0)
        except (TypeError, ValueError):
            mult = 0.0
        if mult > 0:
            info += f" · 乘数 {mult:g}"
        last = _num(cell.data(Qt.UserRole + 1), 0.0) or 0.0
        if last > 0:
            if not self.chk_market.isChecked():
                self.sp_price.setValue(last)
            self._priced_for = sym
            self.lbl_order_msg.setText(
                f"已带入 {code} @ {last:g}（{info}），确认手数和方向后点「下单」")
            return
        self.lbl_order_msg.setText(f"已带入 {code}（{info}），还没拿到最新价，正在取…")
        self._fill_price()

    def _watch_targets(self, pos_rows: list[dict]) -> list[str]:
        """报价框该盯哪些合约：持仓 → 下单框 → 手动加的，统一成天勤写法。

        CTP 的持仓给的是裸代码 `rb2601`，`get_quote` 认不出交易所，所以这里
        过一遍 `_resolve_contract` 补成 `SHFE.rb2601`；补不出来的直接不要
        （猜交易所会把报价订到一个不存在的合约上，那一行永远是「—」）。
        """
        want: list[str] = []

        def push(raw) -> None:
            s = str(raw or "").strip()
            if not s:
                return
            s, _ex = self._resolve_contract(s)
            if s and s not in want:
                want.append(s)

        for r in pos_rows or ():
            push(r.get("symbol"))
        push(self.cb_contract.currentText())
        for sym in self._watch_extra:
            push(sym)
        return want

    def _sync_watch(self, pos_rows: list[dict]) -> None:
        """合约集合变了才重订行情。

        每轮 `on_state` 都算一遍，但只有真的变了才排队给 worker —— 0.4 秒
        重订一次会把柜台日志刷满，行情也跟着抖。
        """
        if not self.live.running:
            return
        want = self._watch_targets(pos_rows)
        key = tuple(want)
        if key == self._watch_sent:
            return
        self._watch_sent = key
        if not key:
            return
        self.live.watch(list(key))

    def _fill_quotes(self, rows: list[dict]) -> None:
        t = self.tbl_quotes
        t.setRowCount(len(rows))
        for i, r in enumerate(rows):
            sym = str(r.get("symbol") or "")
            code = str(r.get("instrument") or "") or sym.rsplit(".", 1)[-1]
            last = _num(r.get("last"), 0.0) or 0.0
            pre = _num(r.get("pre_close"), 0.0) or 0.0
            # 涨跌用昨结算比。拿不到昨结算就画「—」，别拿 0 当昨结
            chg = (last - pre) if (last > 0 and pre > 0) else None
            vals = [
                code,
                # 小数位跟着这个合约的最小变动价位走：铜 90,040、国债 118.115、
                # 棉花 5,902。一律 `:.2f` 是「假精度」，看着像行情在小数位跳
                symbols_info.price_text(last, sym) if last else "—",
                symbols_info.price_text(chg, sym, signed=True) if chg is not None else "—",
                _px(r.get("bid"), sym),
                _px(r.get("ask"), sym),
                _hms(r.get("datetime")),
            ]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setData(Qt.UserRole, sym)
                if j in (1, 2):
                    item.setForeground(QColor(_color(chg)))
                if j >= 1:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                t.setItem(i, j, item)
            # 双击要带走的「最新价」打在合约格上，省得再去读另一格的文字
            cell = t.item(i, 0)
            if cell is not None:
                cell.setData(Qt.UserRole + 1, last)
        # 这里**不做 `_fit_columns`**：列宽在 `_build_quotes` 里已经钉死，按内容
        # 自适应会把这几个数字列顶宽到出滚动条（见那里的注释）

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
        text = self.cb_contract.currentText().strip()
        if not text:
            self.on_failed("合约不能为空")
            return
        # 界面上填的是裸合约代码（`rb2601`），下单必须给柜台完整写法
        # （天勤要 `SHFE.rb2601`；CTP 那个 hub 自己会把前缀剥掉）
        contract, _ex = self._resolve_contract(text)
        if not contract:
            self.on_failed(
                f"「{text}」认不出是哪个交易所的合约，填真实合约代码，如 rb2601")
            return
        self._order_symbol = contract
        price = None if self.chk_market.isChecked() else self.sp_price.value()
        # 限价必须有个价；市价（`price is None`）柜台和天勤都认，不用拦。
        # 🔴 以前这里挡的是「CTP 只收限价单」，2026-11-23 起柜台支持市价单了
        if price is not None and price <= 0:
            self.on_failed("限价单要填一个大于 0 的价格（要市价就勾上「市价」）")
            return
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
