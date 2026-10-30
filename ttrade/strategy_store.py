"""策略配置、策略库、交易组合的本地持久化。

三个 JSON 都落在 `%LOCALAPPDATA%/TrendDesk/` 下：

- `strategy_lib.json` —— 策略库。用户在「策略研究」里存的配置模板，可复用、可导出。
- `portfolio.json`    —— 交易组合。`策略交易和组合` 页面里那一列正在跑的策略。
- `last_config.json`  —— 策略研究页上次的配置，下次打开自动恢复。

**为什么策略库和组合要分开存**：库是「模板」，组合是「实例」。同一个模板可以
在组合里跑两份（不同品种、不同手数、不同账户），改组合里的参数不该影响库。
两边都存完整配置而不是「库 id + 覆盖项」：模板改了以后，正在跑的策略行为跟着变
是件很吓人的事 —— 交易系统里可预期比省几行 JSON 重要。

所有写入走「临时文件 + 原子替换」：这份文件坏了，用户的策略列表就没了。
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field

from ttrade.paths import user_dir

# 按「止损金额」定仓的策略。它们把止损距离写在策略参数里（stop_mult × ATR），
# 固定手数等于把风险敞口交给运气 —— 老配置没有 sizing 字段时按这条兜底。
RISK_SIZED = {"sbbox_mini"}

LIB_FILE = "strategy_lib.json"
PORT_FILE = "portfolio.json"
LAST_FILE = "last_config.json"


def _clamp_pct(value, dflt: float, hi: float) -> float:
    """百分比参数消毒：非数字 / NaN / 非正数退回默认，超上限就夹到上限。

    手改 JSON 写个 `"capital_pct": "30%"` 或者 `-5` 都不该让整份配置加载失败，
    但也不能让它算出负手数 —— 宁可悄悄退回默认。
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return dflt
    if v != v or v <= 0:
        return dflt
    return min(v, hi)


def _pct_of(amount, principal: float, dflt: float) -> float:
    """老配置的绝对金额 → 占本金的百分比。取不到就退回默认。"""
    try:
        a = float(amount or 0)
    except (TypeError, ValueError):
        return dflt
    if a != a or a <= 0 or principal <= 0:
        return dflt
    return a / principal * 100.0



# ------------------------------------------------------------------ 配置对象


@dataclass
class StrategyConfig:
    """一份可复现的策略配置。

    字段分三组：标的（symbol/exchange/period）、策略（family + 规则参数 或 因子配置）、
    执行（volume/account）。`family` 决定另外两组里哪一组有效，无效的那组保留着
    不删 —— 用户从规则切到因子再切回来，不该丢参数。
    """

    sid: str = ""
    name: str = ""
    symbol: str = ""
    exchange: str = ""
    period: str = "1d"

    # rule / factor
    family: str = "rule"

    # 规则策略
    rule_key: str = ""
    rule_params: dict = field(default_factory=dict)

    # 因子策略
    factor_mode: str = ""
    factor_keys: list = field(default_factory=list)
    factor_reverse: dict = field(default_factory=dict)
    factor_cfg: dict = field(default_factory=dict)

    # ---- 截面策略（family == "cross"）----
    # 一次拿一批品种横向比强弱，所以标的不是一个，是一组。`symbol` / `exchange`
    # 仍然记着「当前在看的那个」，用来在列表里指代这条策略。
    cross_symbols: list = field(default_factory=list)
    cross_mode: str = "hedge"           # hedge = 截面对冲；select = 选品种 + 时序
    cross_rebalance: int = 5            # 调仓间隔（根）
    cross_top_pct: float = 10.0         # 多头：选最强的百分比
    cross_bot_pct: float = 10.0         # 空头：选最弱的百分比（可以和上面不同）
    cross_weight: str = "equal"         # equal / score

    # 「选品种 + 时序交易」里的那条时序腿。规则和因子二选一，另一个保留不删 ——
    # 用户来回切换不该丢参数。
    timing_family: str = "rule"         # rule / factor
    timing_rule_key: str = ""
    timing_rule_params: dict = field(default_factory=dict)
    timing_factor_mode: str = "single"
    timing_factor_keys: list = field(default_factory=list)
    timing_factor_reverse: dict = field(default_factory=dict)
    timing_factor_cfg: dict = field(default_factory=dict)

    # 执行 —— 仓位怎么定
    #
    # 默认**不再固定 1 手**：1 手螺纹和 1 手铜差 40 倍名义价值，同一条策略在两
    # 个品种上跑出来的风险完全不是一回事。默认按「分配资金 ÷ 每手保证金」算手数，
    # 也就是把钱按品种自己的杠杆铺满，而不是按「几手」拍脑袋。
    #
    #   capital：分 `capital` 元给这条策略，手数 = capital / 每手保证金
    #   fixed  ：固定 `volume` 手（老行为，想精确控制敞口时用）
    #   risk   ：按止损金额反推手数 = risk_amount / (stop_mult × ATR × 合约乘数)
    #
    # `capital` / `risk_amount` 都是**本金乘百分比算出来的**（见下面两个 property）。
    # 存百分比而不是绝对值，是为了「把本金从 100 万改成 500 万」时手数跟着走 ——
    # 存「30 万」的话改了本金还得挨个回去重算。
    sizing: str = "capital"

    # 本金：这条策略的账户规模。资金曲线从它起步，资金 / 止损两种口径都按它的
    # 百分比算。默认 100 万，和「策略研究」页、组合页的默认值保持一致。
    principal: float = 1_000_000.0
    capital_pct: float = 30.0       # 「按资金分配」：占本金的百分比
    risk_pct: float = 1.0           # 「按止损金额」：占本金的百分比

    stop_atr_n: int = 14            # 止损用的 ATR 周期
    stop_mult: float = 5.0          # 止损 = stop_mult × ATR
    volume: int = 1                 # 仅 sizing="fixed" 时生效

    account: str = ""          # 账户名（AccountSpec.name），空表示用当前连接
    enabled: bool = False
    note: str = ""
    created: str = ""

    # 回测用的成本档位（`ttrade.costs.PROFILES` 的 key：optimistic / neutral /
    # stress）。空 = 跟全局设置走。存在配置里是为了「同一条策略，乐观和压力两套
    # 成本都跑一遍看结论稳不稳」—— 存成全局的话，改一下就把所有策略的口径都
    # 改了，历史回测结果没法复现。
    cost_profile: str = ""

    # 回测 / 看图的区间（年）。0 = 全部历史。
    # 5 分钟十年是 26 万根，跑一遍因子评估或机器学习要几十秒 —— 这个值让
    # 「看一眼这条策略最近怎么样」不必每次都从头算。**不影响实盘交易**：
    # 实时信号永远只用最近的 K 线。
    range_years: float = 0.0

    # ---- 展示 ----

    # 这两个是**算出来的**，不是存的：本金 × 百分比。上层（回测、实盘引擎、
    # 组合页）读的一直是这两个名字，改成 property 之后那边一行都不用动。
    @property
    def capital(self) -> float:
        """「按资金分配」能动用的钱。"""
        return self.principal * self.capital_pct / 100.0

    @property
    def risk_amount(self) -> float:
        """「按止损金额」每笔最多亏的钱。"""
        return self.principal * self.risk_pct / 100.0

    def key(self) -> str:
        """品种键。美股的 exchange 是空的，不能拼出 `AAPL_` 这种尾巴。"""
        return f"{self.symbol}_{self.exchange}" if self.exchange else self.symbol

    def title(self) -> str:
        """列表里显示的名字。用户没起名就用「品种 · 策略」。"""
        if self.name.strip():
            return self.name.strip()
        if self.family == "cross":
            # 截面策略没有「一个品种」可言：`key()` 只是品种池里的第一个，
            # 拿它拼标题会让人以为这条策略只交易那一个
            return self.strategy_label()
        return f"{self.key()} · {self.strategy_label()}"

    def strategy_label(self) -> str:
        """策略的可读名。规则策略直接查表，因子策略拼模式和因子数。"""
        if self.family == "cross":
            n = len(self.cross_symbols or [])
            mode = {"single": "单因子", "multi": "多因子", "ml": "机器学习"}.get(
                self.factor_mode, self.factor_mode or "多因子")
            if self.cross_mode == "select":
                return f"截面选品种 + 时序（{mode} · {n} 个品种）"
            return f"截面对冲 · {mode} · 多 {self.cross_top_pct:g}% / 空 {self.cross_bot_pct:g}%（{n} 个品种）"
        if self.family == "rule":
            from ttrade.strategies import STRATEGIES

            spec = STRATEGIES.get(self.rule_key)
            return spec.label if spec else (self.rule_key or "未选策略")
        mode = {"single": "单因子", "multi": "多因子", "ml": "机器学习"}.get(
            self.factor_mode, self.factor_mode or "因子"
        )
        n = len(self.factor_keys or [])
        return f"{mode} · {n} 个因子" if n else mode

    def params_text(self) -> str:
        """参数摘要，给表格的「参数」列用。"""
        if self.family == "rule":
            return " ".join(f"{k}={v}" for k, v in (self.rule_params or {}).items())
        if self.family == "cross":
            mode = {"single": "单因子", "multi": "多因子", "ml": "机器学习"}.get(
                self.factor_mode, self.factor_mode or "多因子")
            bits = [f"{mode}·{len(self.factor_keys or [])}因子",
                    f"调仓={int(self.cross_rebalance or 1)}根",
                    f"多{float(self.cross_top_pct or 0):g}%/空{float(self.cross_bot_pct or 0):g}%",
                    f"权重={'等权' if self.cross_weight != 'score' else '按分数'}"]
            if self.cross_mode == "select":
                bits.append(f"时序={'规则' if self.timing_family != 'factor' else '因子'}")
            return " ".join(bits)
        cfg = self.factor_cfg or {}
        bits = [f"窗口={cfg.get('window', 252)}", f"阈值={cfg.get('entry_z', 1.0)}"]
        if self.factor_mode == "ml":
            bits.append(f"模型={cfg.get('model', '')}")
        return " ".join(bits)

    def size_text(self) -> str:
        """仓位口径的可读描述。实际手数要等拿到价格和保证金才知道。"""
        mode = (self.sizing or "capital").lower()
        if mode == "fixed":
            return f"固定 {int(self.volume or 1)} 手"
        if mode == "risk":
            return f"止损 {self.risk_pct:g}% · {self.risk_amount / 10000:g} 万"
        if mode == "unit":
            return "本金全额"
        return f"资金 {self.capital_pct:g}% · {self.capital / 10000:g} 万"

    def describe(self) -> str:
        """一句话说明，给确认框和日志用。"""
        # 截面策略的 `key()` 只是品种池里的第一个，写出来会误导
        head = self.title() if self.family == "cross" else f"{self.title()}｜{self.key()}"
        return (
            f"{head} {self.period}｜{self.strategy_label()}"
            f"｜{self.size_text()}" + (f"｜账户 {self.account}" if self.account else "")
        )

    # ---- 序列化 ----

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factor_keys"] = list(self.factor_keys or [])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "StrategyConfig":
        keep = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**keep)
        if not cfg.sid:
            cfg.sid = uuid.uuid4().hex[:12]
        if not cfg.created:
            cfg.created = time.strftime("%Y-%m-%d %H:%M")
        # 兼容手改过的 JSON：类型不对就退回默认，别让整份配置加载失败
        try:
            cfg.volume = max(1, int(cfg.volume))
        except (TypeError, ValueError):
            cfg.volume = 1
        try:
            cfg.range_years = max(0.0, float(cfg.range_years or 0))
        except (TypeError, ValueError):
            cfg.range_years = 0.0
        if cfg.sizing not in ("capital", "fixed", "risk"):
            cfg.sizing = "capital"
        # 老配置里没有 sizing 字段。按止损金额定仓的策略（止损距离写在策略参数里）
        # 默认就该走 risk —— 用固定手数等于把风险敞口交给运气，和它自己的设计相反
        if "sizing" not in d and cfg.rule_key in RISK_SIZED:
            cfg.sizing = "risk"
        # ---- 本金与仓位百分比 ----
        # v1.6.1 起存的是「本金 + 百分比」，之前存的是绝对金额（capital /
        # risk_amount 两个字段）。老配置必须按**原来的钱**折回去，本金默认 100 万：
        # 50 万折成 50%、2 万折成 2%。不能直接套新默认的 30% / 1% —— 那样用户
        # 一升级，昨天还在跑的仓今天就变成 0.6 倍，谁受得了。
        #
        # 连 capital 字段都没有的老配置（早于「仓位口径」那一版）当年走的是
        # 500000 / 20000 这两个默认值，所以迁移默认也得是 50% / 2%，不是 30% / 1%。
        try:
            cfg.principal = float(cfg.principal or 0)
        except (TypeError, ValueError):
            cfg.principal = 0.0
        if cfg.principal < 10_000 or cfg.principal != cfg.principal:
            cfg.principal = 1_000_000.0
        if "capital_pct" not in d:
            cfg.capital_pct = _pct_of(d.get("capital"), cfg.principal, 50.0)
        if "risk_pct" not in d:
            cfg.risk_pct = _pct_of(d.get("risk_amount"), cfg.principal, 2.0)
        cfg.capital_pct = _clamp_pct(cfg.capital_pct, 30.0, 500.0)
        cfg.risk_pct = _clamp_pct(cfg.risk_pct, 1.0, 100.0)
        try:
            v = float(cfg.stop_mult or 0)
            cfg.stop_mult = v if v >= 0.1 and v == v else 5.0
        except (TypeError, ValueError):
            cfg.stop_mult = 5.0
        try:
            cfg.stop_atr_n = max(1, int(cfg.stop_atr_n or 14))
        except (TypeError, ValueError):
            cfg.stop_atr_n = 14
        if not isinstance(cfg.rule_params, dict):
            cfg.rule_params = {}
        if not isinstance(cfg.factor_cfg, dict):
            cfg.factor_cfg = {}
        if not isinstance(cfg.factor_keys, list):
            cfg.factor_keys = []
        # 截面策略那组字段同样要消毒：手改 JSON 写成字符串 / 0 / 负数时不能让
        # 回测直接除零或者永远不调仓
        for f in ("cross_symbols", "timing_factor_keys"):
            if not isinstance(getattr(cfg, f), list):
                setattr(cfg, f, [])
        for f in ("timing_rule_params", "timing_factor_reverse", "timing_factor_cfg"):
            if not isinstance(getattr(cfg, f), dict):
                setattr(cfg, f, {})
        try:
            cfg.cross_rebalance = max(1, int(cfg.cross_rebalance or 1))
        except (TypeError, ValueError):
            cfg.cross_rebalance = 5
        cfg.cross_top_pct = _clamp_pct(cfg.cross_top_pct, 10.0, 50.0)
        cfg.cross_bot_pct = _clamp_pct(getattr(cfg, "cross_bot_pct", 10.0), 10.0, 50.0)
        if cfg.cross_mode not in ("hedge", "select"):
            cfg.cross_mode = "hedge"
        if cfg.cross_weight not in ("equal", "score"):
            cfg.cross_weight = "equal"
        if cfg.timing_family not in ("rule", "factor"):
            cfg.timing_family = "rule"
        return cfg

    def clone(self, **changes) -> "StrategyConfig":
        """复制一份新配置（换 sid）。组合里同一个模板跑两份时用。"""
        d = self.to_dict()
        d.update(changes)
        d["sid"] = ""
        return StrategyConfig.from_dict(d)

    def same_shape(self, other: "StrategyConfig") -> bool:
        """两条配置是不是「同一个策略的同一套参数」。

        判断重复用。忽略 sid / 名称 / 启用状态这些不改变交易行为的东西，但
        **手数和账户算在内** —— 手数不同是两个不同的仓位；账户不同是两个不同的
        资金账号，同一套策略在两个账户上各跑一份是正常用法，不该算重复。
        """
        a, b = self.to_dict(), other.to_dict()
        # range_years 只影响回测看多少年，cost_profile 只影响回测算出来的收益，
        # 都不改变交易行为，不算「不同配置」
        for k in ("sid", "name", "created", "enabled", "range_years", "cost_profile"):
            a.pop(k, None)
            b.pop(k, None)
        return a == b


# ------------------------------------------------------------------ 读写


def _path(name: str) -> str:
    return os.path.join(user_dir(), name)


def _load(name: str) -> list | dict:
    path = _path(name)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        # 文件坏了就当空 —— 交易系统不能因为一份配置读不出来就起不来
        return []


def _save(name: str, data) -> None:
    """原子写：先写 .tmp 再替换。直接写原文件，中途断电就只剩半个 JSON。"""
    path = _path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=1)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _ensure_sid(cfg: StrategyConfig) -> None:
    """落盘前补一个 sid。

    `StrategyConfig()` 直接构造出来时 sid 是空的，`from_dict()` 会补 —— 但只在
    **读**的时候补。于是「存进去空 sid → 读出来随机 sid → 再存回去」每重启一次
    就换一个身份，`update_portfolio()` / `remove_from_portfolio()` 按 sid 找
    条目全部落空。必须在写的时候就把 sid 定下来。
    """
    if not cfg.sid:
        cfg.sid = uuid.uuid4().hex[:12]


def _load_list(name: str) -> list[StrategyConfig]:
    raw = _load(name)
    if not isinstance(raw, list):
        return []
    out = []
    seen: set[str] = set()
    for it in raw:
        if not isinstance(it, dict):
            continue
        try:
            cfg = StrategyConfig.from_dict(it)
        except Exception:  # noqa: BLE001
            continue
        # 手改 JSON 复制粘贴出来的重复 sid：留第一个，后面的换新号，
        # 否则「删第二条」会把第一条删掉。
        if cfg.sid in seen:
            cfg.sid = uuid.uuid4().hex[:12]
        seen.add(cfg.sid)
        out.append(cfg)
    return out


# ---- 策略库 ----


def load_library() -> list[StrategyConfig]:
    return _load_list(LIB_FILE)


def save_library(items: list[StrategyConfig]) -> None:
    for c in items:
        _ensure_sid(c)
    _save(LIB_FILE, [c.to_dict() for c in items])


def add_to_library(cfg: StrategyConfig) -> StrategyConfig:
    """存进策略库。同 sid 覆盖，否则追加。

    **名字不许重复。** 库里出现三条「白银动量60-1h」之后，用户和 Agent 都没法
    用名字指代任何一条 —— `find_library` 只能猜「最后存的那条」，删的时候还得
    靠列表里的位置。重名直接抛 `ValueError`，让调用方去改名。

    例外是**没起名**的配置：名字是「品种 · 策略」自动拼的，同品种同策略存两份
    参数不同的版本是正常用法，这时自动加序号（`…（2）`），不挡人。
    """
    items = load_library()
    _ensure_sid(cfg)
    for i, old in enumerate(items):
        if old.sid == cfg.sid:
            # 同 sid 是「改这一条」：名字撞到**别的**条目也要拦，否则改个名就
            # 能造出两条重名的
            want = norm_name(cfg.title())
            if any(norm_name(o.title()) == want for o in items if o.sid != cfg.sid):
                raise ValueError(f"策略库里已经有「{cfg.title()}」了，换一个名字")
            items[i] = cfg
            save_library(items)
            return cfg

    if not str(cfg.name or "").strip():
        base = cfg.title()
        taken = {norm_name(c.title()) for c in items}
        if norm_name(base) in taken:
            n = 2
            while norm_name(f"{base}（{n}）") in taken:
                n += 1
            cfg.name = f"{base}（{n}）"
    else:
        want = norm_name(cfg.title())
        dup = next((o for o in items if norm_name(o.title()) == want), None)
        if dup is not None:
            raise ValueError(
                f"策略库里已经有「{dup.title()}」了。换一个名字，"
                "或者先删掉库里那一条再存。"
            )

    if not cfg.created:
        cfg.created = time.strftime("%Y-%m-%d %H:%M")
    items.append(cfg)
    save_library(items)
    return cfg


_MINE_TAIL = re.compile(r"\s*[·•・\-—_]\s*我的\s*$")


def norm_name(s: str) -> str:
    """名字比对用的归一化。

    Agent 是从**截图**里认名字的，看到的写法是「碳酸锂动量60-5m · 我的」——
    那个「· 我的」是左侧列表渲染时加的归属标记，不是名字的一部分。用户嘴上
    说的、库里存的都是「碳酸锂动量60-5m」。不剥掉这截尾巴就永远对不上。
    顺带抹掉全角空格、大小写和所有空白：手打的名字空格位置全靠缘分。
    """
    t = _MINE_TAIL.sub("", str(s or "").strip())
    # NFKC 顺手把全角字母/数字/空格压成半角：截图 OCR、手打输入法都爱出全角
    t = unicodedata.normalize("NFKC", t).replace("（我的）", "").replace("(我的)", "")
    return re.sub(r"\s+", "", t).lower()


def find_library(name: str) -> StrategyConfig | None:
    """按 sid 或名字在策略库里找模板，找不到返回 None。

    同名可能有好几条（用户前后存过两版，比如「白银动量60-1h」现在有三条），
    取**最后存的**那条：他刚存的那份才是他嘴上说的那份。真正要精确的时候
    用 sid，`find_library` 认。
    """
    want = str(name or "").strip()
    if not want:
        return None
    items = load_library()
    for c in items:
        if c.sid and c.sid == want:
            return c
    key = norm_name(want)
    if not key:
        return None
    hits = [c for c in items if norm_name(c.title()) == key]
    return hits[-1] if hits else None


def remove_from_library(sid: str) -> bool:
    items = load_library()
    keep = [c for c in items if c.sid != sid]
    if len(keep) == len(items):
        return False
    save_library(keep)
    return True


# ---- 交易组合 ----


def load_portfolio() -> list[StrategyConfig]:
    return _load_list(PORT_FILE)


def save_portfolio(items: list[StrategyConfig]) -> None:
    for c in items:
        _ensure_sid(c)
    _save(PORT_FILE, [c.to_dict() for c in items])


def add_to_portfolio(cfg: StrategyConfig) -> StrategyConfig:
    """加进组合。**总是复制一份新 sid** —— 组合是实例，和库里的模板解耦。"""
    items = load_portfolio()
    item = cfg.clone()
    item.created = time.strftime("%Y-%m-%d %H:%M")
    items.append(item)
    save_portfolio(items)
    return item


def update_portfolio(cfg: StrategyConfig) -> None:
    items = load_portfolio()
    for i, old in enumerate(items):
        if old.sid == cfg.sid:
            items[i] = cfg
            save_portfolio(items)
            return


def remove_from_portfolio(sid: str) -> bool:
    items = load_portfolio()
    keep = [c for c in items if c.sid != sid]
    if len(keep) == len(items):
        return False
    save_portfolio(keep)
    return True


# ---- 上次配置 ----


def load_last() -> StrategyConfig | None:
    raw = _load(LAST_FILE)
    if isinstance(raw, dict) and raw:
        try:
            return StrategyConfig.from_dict(raw)
        except Exception:  # noqa: BLE001
            return None
    return None


def save_last(cfg: StrategyConfig) -> None:
    _save(LAST_FILE, cfg.to_dict())   # 上次配置不参与按 sid 查找，不必补号


# ------------------------------------------------------------------ 交易助手设置


@dataclass
class ChaseConfig:
    """交易助手：策略发单后没成交怎么追。

    天勤的 `TargetPosTask` 自己会追价，但它按固定节奏重报，遇到涨跌停附近或者
    流动性差的合约经常挂一整天。这里的追单是**上层补刀**：等 N 秒还没成交就撤掉
    重报，价格往盘口方向挪几跳，最多追 M 次。
    """

    enabled: bool = True
    wait_sec: int = 5          # 每次等多久
    retries: int = 3           # 最多追几次
    ticks: int = 2             # 每次追几跳
    use_market: bool = False   # 追到最后一次直接用市价

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChaseConfig":
        keep = {k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__}
        cfg = cls(**keep)
        cfg.wait_sec = max(1, int(cfg.wait_sec))
        cfg.retries = max(0, int(cfg.retries))
        cfg.ticks = max(0, int(cfg.ticks))
        return cfg


CHASE_FILE = "chase.json"


def load_chase() -> ChaseConfig:
    raw = _load(CHASE_FILE)
    if isinstance(raw, dict) and raw:
        try:
            return ChaseConfig.from_dict(raw)
        except Exception:  # noqa: BLE001
            pass
    return ChaseConfig()


def save_chase(cfg: ChaseConfig) -> None:
    _save(CHASE_FILE, cfg.to_dict())
