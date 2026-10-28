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
import time
import uuid
from dataclasses import asdict, dataclass, field

from ttrade.paths import user_dir

# 按「止损金额」定仓的策略。它们把止损距离写在策略参数里（stop_mult × ATR），
# 固定手数等于把风险敞口交给运气 —— 老配置没有 sizing 字段时按这条兜底。
RISK_SIZED = {"sbbox_mini"}

LIB_FILE = "strategy_lib.json"
PORT_FILE = "portfolio.json"
LAST_FILE = "last_config.json"


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

    # 执行 —— 仓位怎么定
    #
    # 默认**不再固定 1 手**：1 手螺纹和 1 手铜差 40 倍名义价值，同一条策略在两
    # 个品种上跑出来的风险完全不是一回事。默认按「分配资金 ÷ 每手保证金」算手数，
    # 也就是把钱按品种自己的杠杆铺满，而不是按「几手」拍脑袋。
    #
    #   capital：分 `capital` 元给这条策略，手数 = capital / 每手保证金
    #   fixed  ：固定 `volume` 手（老行为，想精确控制敞口时用）
    #   risk   ：按止损金额反推手数 = risk_amount / (stop_mult × ATR × 合约乘数)
    sizing: str = "capital"
    capital: float = 500_000.0      # 分配资金（元）
    risk_amount: float = 20_000.0   # 单笔止损金额（元）
    stop_atr_n: int = 14            # 止损用的 ATR 周期
    stop_mult: float = 5.0          # 止损 = stop_mult × ATR
    volume: int = 1                 # 仅 sizing="fixed" 时生效

    account: str = ""          # 账户名（AccountSpec.name），空表示用当前连接
    enabled: bool = False
    note: str = ""
    created: str = ""

    # 回测 / 看图的区间（年）。0 = 全部历史。
    # 5 分钟十年是 26 万根，跑一遍因子评估或机器学习要几十秒 —— 这个值让
    # 「看一眼这条策略最近怎么样」不必每次都从头算。**不影响实盘交易**：
    # 实时信号永远只用最近的 K 线。
    range_years: float = 0.0

    # ---- 展示 ----

    def key(self) -> str:
        """品种键。美股的 exchange 是空的，不能拼出 `AAPL_` 这种尾巴。"""
        return f"{self.symbol}_{self.exchange}" if self.exchange else self.symbol

    def title(self) -> str:
        """列表里显示的名字。用户没起名就用「品种 · 策略」。"""
        if self.name.strip():
            return self.name.strip()
        return f"{self.key()} · {self.strategy_label()}"

    def strategy_label(self) -> str:
        """策略的可读名。规则策略直接查表，因子策略拼模式和因子数。"""
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
            return f"止损 {self.risk_amount / 10000:g} 万"
        if mode == "unit":
            return "本金全额"
        return f"{self.capital / 10000:g} 万"

    def describe(self) -> str:
        """一句话说明，给确认框和日志用。"""
        return (
            f"{self.title()}｜{self.key()} {self.period}｜{self.strategy_label()}"
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
        for attr, lo, dflt in (("capital", 0.0, 500_000.0),
                               ("risk_amount", 0.0, 20_000.0),
                               ("stop_mult", 0.1, 5.0)):
            try:
                v = float(getattr(cfg, attr) or 0)
                setattr(cfg, attr, v if v >= lo and v == v else dflt)
            except (TypeError, ValueError):
                setattr(cfg, attr, dflt)
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
        # range_years 只影响回测看多少年，不改变交易行为，不算「不同配置」
        for k in ("sid", "name", "created", "enabled", "range_years"):
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
    """存进策略库。同 sid 覆盖，否则追加。返回入库后的那份（可能换了 sid）。"""
    items = load_library()
    _ensure_sid(cfg)
    for i, old in enumerate(items):
        if old.sid == cfg.sid:
            items[i] = cfg
            save_library(items)
            return cfg
    if not cfg.created:
        cfg.created = time.strftime("%Y-%m-%d %H:%M")
    items.append(cfg)
    save_library(items)
    return cfg


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
