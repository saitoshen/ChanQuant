"""策略配置、策略库、交易组合的本地持久化。

三个 JSON 都落在 `%LOCALAPPDATA%/ChanQuant/` 下：

- `strategy_lib.json` —— 策略库。用户在「策略研究」里存的配置模板，可复用、可导出。
- `portfolio.json`    —— 交易组合。`策略交易和组合` 页面里那一列正在跑的策略。
- `last_config.json`  —— 策略研究页上次的配置，下次打开自动恢复。

除了这三份，还有一个**不在磁盘上**的策略库成员：`ttrade.strategy_templates`
里的内置模板。它随版本发布，只读，`library_view()` 把两边拼起来给界面用；
`load_library()` 仍然只返回用户自己那份 —— 存盘、删除都只碰用户的东西。

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

from ttrade import strategy_templates as TPL
from ttrade.paths import user_dir

# 按「止损金额」定仓的策略。它们把止损距离写在策略参数里（stop_mult × ATR），
# 固定手数等于把风险敞口交给运气 —— 老配置没有 sizing 字段时按这条兜底。
RISK_SIZED = {"sbbox_mini"}

LIB_FILE = "strategy_lib.json"
PORT_FILE = "portfolio.json"
LAST_FILE = "last_config.json"

# 截面策略「腿内怎么分钱」的中文名。策略库表格里要显示，界面下拉也用它，
# 放一处免得两边慢慢走偏
_WEIGHT_CN = {"risk": "风险平价", "equal": "等权", "score": "按分数", "tier": "分档"}
# 调仓间隔的单位，只用在参数摘要里（「调仓=5天」比「调仓=5」好读）
_REB_UNIT_CN = {"bar": "根", "day": "天", "month": "月"}

# A 股的交易所代号。期货是 SHFE / DCE / CZCE / INE / CFFEX / GFEX，不重叠，
# 所以「交易所名」足以判断市场。和 `replay.STOCK_EXCHANGES` 是同一份约定。
_STOCK_EXCHANGES = ("SSE", "SZSE", "BSE")

# 交易所后缀（Tushare 写法）→ 天勤代号
_TS_SUFFIX = {"SH": "SSE", "SZ": "SZSE", "BJ": "BSE"}


def normalize_symbol(symbol: str, exchange: str = "") -> tuple[str, str]:
    """把各种写法的 A 股代码收敛成「天勤代号 + 交易所」。

    Agent 只看得见截图和对话，A 股代码它有三种写法：`600000`、`600000.SH`、
    `SSE.600000`；交易所也可能是空的。三种都要认，否则模型报一次「用茅台回测
    双均线」就会被回一句「品种不存在」，而它没有别的信息可用了。

    期货（`RB` + `SHFE`）和美股（`AAPL` + 空）原样返回 —— 它们的键是拼出来的，
    不需要在这里动。
    """
    sym = str(symbol or "").strip()
    ex = str(exchange or "").strip().upper()
    if not sym:
        return sym, ex

    # `SSE.600000` / `SZSE.000001`：代号里已经带了交易所
    if "." in sym:
        head, _, tail = sym.partition(".")
        head_u = head.upper()
        if head_u in _STOCK_EXCHANGES or head_u == "BSE":
            return f"{head_u}.{tail}", head_u
        # `600000.SH` 这种反过来写的
        if head.isdigit() and tail.upper() in _TS_SUFFIX:
            return f"{_TS_SUFFIX[tail.upper()]}.{head}", _TS_SUFFIX[tail.upper()]
        return sym, ex

    if ex in _STOCK_EXCHANGES or ex == "BSE":
        return f"{ex}.{sym}", ex

    # 纯 6 位数字：按前缀判交易所。
    # 北交所必须排在沪市前面 —— 北交所 2024 年之后新开的代码是 `920xxx`，
    # 按首位判会被 `9` 抢到上交所，`920185` 就变成 `SSE.920185` 查不到数据。
    # 沪市 `9` 开头只剩 B 股 `900xxx`，深市 B 股是 `200xxx`。
    if sym.isdigit() and len(sym) == 6:
        if sym.startswith(("92", "43", "83", "87", "88")):
            return f"BSE.{sym}", "BSE"
        if sym.startswith(("90", "6", "5")):
            return f"SSE.{sym}", "SSE"
        if sym.startswith(("00", "30", "20", "15", "16", "12", "39")):
            return f"SZSE.{sym}", "SZSE"
        return f"BSE.{sym}", "BSE"
    return sym, ex



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


def _cross_period_names(keys, periods) -> list[str]:
    """策略摘要里那句「跨周期·日线/周线」用的周期名。

    两种写法都要认：新配置把周期编在因子实例 id 里（`mom_60@1d_tq`），老配置
    是另一份 `{因子: 周期}` 映射。只读其中一种的话，换了写法摘要就变空白，
    两条其实不同的策略在列表里长得一模一样。
    """
    from ttrade import factors as F
    from ttrade import tqdata

    names: set[str] = set()
    for k in (keys or []):
        p = F.split_spec(k)[1]
        if p:
            names.add(tqdata.PERIOD_LABEL.get(p, p))
    for v in dict(periods or {}).values():
        if str(v or "").strip():
            names.add(tqdata.PERIOD_LABEL.get(str(v), str(v)))
    return sorted(names)


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

    # ---- 多品种回测的品种清单（规则 / 因子族用）----
    # `symbol` / `exchange` 只记「当前在看的那个」，多品种勾选是另一件事。
    # 以前只存了单品种，于是「存一条跑 78 个品种的策略」再打开就变回 1 个品种，
    # 曲线当然对不上 —— 存盘必须把这批勾选一起带上。
    # 截面族早期用的是 `cross_symbols`（同一份东西的旧名字），保留兼容。
    multi_symbols: list = field(default_factory=list)

    # rule / factor
    family: str = "rule"

    # 规则策略
    rule_key: str = ""
    rule_params: dict = field(default_factory=dict)
    # 反向交易：信号整体取反（多 ↔ 空）。有些规则反过来做才有 alpha —— 比如
    # 趋势策略在震荡品种上，与其重写一条「做空版」，不如把现成的翻过来。
    # 放在配置里而不是界面上乘一下：组合回放、实盘、存盘都必须跟着走，
    # 否则「研究页看到的」和「实盘跑出来的」是两条相反的曲线
    rule_reverse: bool = False

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
    cross_rebalance: int = 5            # 调仓间隔（数量，单位见下一行）
    # 调仓间隔的单位：bar / day / month。选品种模式按天 / 月（换一次等于整篮子
    # 重下一遍，按「根」给的话 5 分钟周期上 5 根就换一次，成本能把收益吃光）；
    # 对冲模式仍然按根。存盘时要一起存 —— 只存数量的话，选品种策略重开一次
    # 就会从「每 5 天」变成「每 5 根」
    cross_rebalance_unit: str = "bar"
    cross_top_pct: float = 10.0         # 多头：选最强的百分比
    cross_bot_pct: float = 10.0         # 空头：选最弱的百分比（可以和上面不同）
    cross_weight: str = "risk"          # risk / equal / score / tier
    # 分档：weight == "tier" 时把选中的几只按分数切成 N 档，各档拿不同份数
    cross_tiers: int = 0                # 0/1 = 不分档
    cross_tier_units: list = field(default_factory=list)   # 空则按 4/2/1 递减
    # 第一步选品种池（第二步才排序）。用「能不能交易」的因子筛池，
    # 和「谁强谁弱」的排序因子分开，流动性就不会把权重带偏
    cross_pool_mode: str = "off"        # off / single / multi / ml
    cross_pool_keys: list = field(default_factory=list)
    cross_pool_reverse: dict = field(default_factory=dict)
    cross_pool_pct: float = 60.0        # 池里保留分数最高的百分之几
    # 第一步（排名 + 筛池）用哪个周期的因子。空 = 跟随策略周期。
    # 5 分钟策略用日线成交额筛池是常见用法，存盘必须带上 —— 只存因子不存周期，
    # 重开一次就悄悄变回「5 分钟成交额」，同一份配置跑出两个结果
    cross_step1_period: str = ""
    # 截面 ML 的标签口径：excess / volret / rank / volret_rank
    cross_label_mode: str = "volret"

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
        """品种键，和 `DataStore` 里的键对齐。

        - 美股：`AAPL`（exchange 是空的，不能拼出 `AAPL_` 这种尾巴）
        - 期货：`RB_SHFE`
        - A 股：`SSE.600000` —— **代码里本来就带交易所前缀**，再拼一次就成
          `SSE.600000_SSE`，`cb_symbol.findData()` 找不到，恢复配置会静默失败
        """
        ex = str(self.exchange or "").upper()
        if ex in _STOCK_EXCHANGES:
            sym = str(self.symbol or "")
            return sym if "." in sym else f"{ex}.{sym}"
        return f"{self.symbol}_{self.exchange}" if self.exchange else self.symbol

    @property
    def market(self) -> str:
        """这条策略属于哪个市场（`us` / `cn` / `cn_stock`）。

        判据只有交易所名，和 `replay.market_of()` 是同一套 —— 但那边是函数、
        这边是属性，上层两边都在用，所以共用一份实现。
        """
        from ttrade import replay as _rp

        return _rp.market_of(self)

    def title(self) -> str:
        """列表里显示的名字。用户没起名就用「品种 · 策略」。"""
        if self.name.strip():
            return self.name.strip()
        if self.family == "cross":
            # 截面策略没有「一个品种」可言：`key()` 只是品种池里的第一个，
            # 拿它拼标题会让人以为这条策略只交易那一个
            return self.strategy_label()
        n = len(self.symbols or [])
        if n >= 2:
            # 多品种：写「78 个品种」而不是其中某一个 —— 只写第一个会让人以为
            # 这条策略只跑那一个，而存盘里明明有 78 个
            return f"{n} 个品种 · {self.strategy_label()}"
        return f"{self.key()} · {self.strategy_label()}"

    @property
    def symbols(self) -> list:
        """这条策略实际要跑的品种清单。

        `multi_symbols` 优先。它之前只有 `cross_symbols` 一个字段 —— 名字叫截面，
        但记的**不是**「截面专用的池子」，而是「勾选面板里当前选中的所有品种」
        （`_current_config` 从 v1.7.0 起就无条件写它，哪个策略族都写）。所以对
        规则 / 因子族的老配置，那份清单同样有效，不能当成残留丢掉 —— 丢掉的话
        库里 19 个品种的策略一加载就变回单品种，再存一遍就把 19 个彻底覆盖掉了。

        两个都是空的才是单品种，用 `key()` 那一个。
        """
        got = [s for s in (self.multi_symbols or []) if s]
        if not got:
            # 老字段只在**真的是一批品种**时才认（≥2）。只勾过一个的老条目不算
            # 多品种 —— 拿那一个替换 `key()` 等于悄悄换了标的，宁可维持原样
            legacy = [s for s in (self.cross_symbols or []) if s]
            got = legacy if len(legacy) >= 2 else []
        if got:
            return got
        k = self.key()
        return [k] if k else []

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
            name = spec.label if spec else (self.rule_key or "未选策略")
            # 反向要写进名字里：列表里两条同名策略一条正着做一条反着做，
            # 不标出来的话只能靠点进去看，而它们的曲线是镜像的
            if self.rule_reverse:
                name = f"{name}（反向）"
            # 「规则 + 因子过滤」也要标出来：库里「唐奇安」和「唐奇安 + 因子过滤」
            # 两行的参数一模一样，交易笔数却能差一半，不标就是两条重复项
            if self.factor_mode and self.factor_keys:
                return f"{name} + 因子过滤"
            return name
        mode = {"single": "单因子", "multi": "多因子", "ml": "机器学习"}.get(
            self.factor_mode, self.factor_mode or "因子"
        )
        n = len(self.factor_keys or [])
        return f"{mode} · {n} 个因子" if n else mode

    def params_text(self) -> str:
        """参数摘要，给表格的「参数」列用。"""
        if self.family == "rule":
            bits = [f"{k}={v}" for k, v in (self.rule_params or {}).items()]
            # 因子过滤的条件不写出来的话，库里两条「参数一模一样」的规则策略
            # 看起来就是重复项，而实际交易笔数能差一半
            if self.factor_mode and self.factor_keys:
                _fc = self.factor_cfg or {}
                mode = {"single": "单因子", "multi": "多因子",
                        "ml": "机器学习"}.get(self.factor_mode, self.factor_mode)
                when = {"entry": "开仓否决", "delay": "转好再进",
                        "hold": "转差离场"}.get(
                    str(_fc.get("filter_mode") or "entry"), "开仓否决")
                bits.append(f"过滤{mode}·{len(self.factor_keys)}因子·{when}")
                _xp = _cross_period_names(list(self.factor_keys or []),
                                          _fc.get("factor_periods"))
                if _xp:
                    bits.append("跨周期·" + "/".join(_xp))
            return " ".join(bits)
        if self.family == "cross":
            mode = {"single": "单因子", "multi": "多因子", "ml": "机器学习"}.get(
                self.factor_mode, self.factor_mode or "多因子")
            bits = [f"{mode}·{len(self.factor_keys or [])}因子",
                    f"调仓={int(self.cross_rebalance or 1)}{_REB_UNIT_CN.get(self.cross_rebalance_unit, '根')}",
                    f"权重={_WEIGHT_CN.get(self.cross_weight, '风险平价')}"]
            if self.cross_mode == "select":
                # 选品种模式没有腿数（池内全进时序腿），写出来是假信息
                bits.append(f"时序={'规则' if self.timing_family != 'factor' else '因子'}")
            else:
                bits.insert(2, f"多{float(self.cross_top_pct or 0):g}%/空"
                               f"{float(self.cross_bot_pct or 0):g}%")
            # 机器学习那一套（稀疏采样 / 合训）也会跟着 `factor_cfg` 走，
            # 截面 ML 一样受影响，摘要里一并写出来
            _mc = self.factor_cfg or {}
            _xp = _cross_period_names(list(self.factor_keys or []) + list(
                self.timing_factor_keys or []), _mc.get("factor_periods"))
            if _xp:
                bits.append("跨周期·" + "/".join(_xp))
            if self.factor_mode == "ml":
                if str(_mc.get("sample_mode") or "all") == "trigger":
                    bits.append("稀疏触发")
                if _mc.get("pooled"):
                    bits.append("合训")
            return " ".join(bits)
        cfg = self.factor_cfg or {}
        bits = [f"窗口={cfg.get('window', 252)}", f"阈值={cfg.get('entry_z', 1.0)}"]
        # 离场阈值现在能取负，对持仓时长的影响不比窗口小：同样的进场阈值，
        # 0.3 和 -1.0 是两条策略（后者要反向坐实才平）。默认值不写，免得刷屏
        _xz = float(cfg.get("exit_z", 0.3) or 0.0)
        if abs(_xz - 0.3) > 1e-9:
            bits.append(f"离场{_xz:g}")
        # 分位口径和 z-score 是同一条策略的两种算法，阈值一样结果差很多，
        # 不写出来列表上两条会像重复项
        if str(cfg.get("method") or "z") != "z":
            bits.append("分位")
        if self.factor_mode == "multi":
            # 权重口径和出场规则对结果的影响不比窗口小：同样是 5 个因子，
            # 等权和滚动 IC 加权是两条策略；出场换成 ATR 分档之后交易笔数
            # 会少一大截。不写出来，列表里两条看起来像重复项
            wm = str(cfg.get("weight_mode") or "equal")
            bits.append({"equal": "等权", "custom": "自定权重",
                         "ic": "IC 加权"}.get(wm, wm))
        if str(cfg.get("exit_mode") or "score") != "score":
            bits.append({"tier": "分档回撤出场", "abs": "ATR 回撤出场",
                         "box": "箱体出场"}.get(str(cfg.get("exit_mode")), "ATR 出场"))
        # 止损独立于口径：只勾止损、口径还是「分数出场」时，上面那条不会写，
        # 这里必须补上 —— 否则列表里「加了止损」和「没加」两条看起来一模一样
        if cfg.get("exit_sl"):
            bits.append(f"止损 {float(cfg.get('exit_sl_atr') or 2.0):g}ATR")
        # 勾了「不按分数出场」也是独立一条：同样的 ATR 参数，勾没勾它交易笔数
        # 差一大截，摘要不写的话两条配置在列表里长得一模一样
        if cfg.get("exit_no_flip"):
            bits.append("反向不平仓")
        elif cfg.get("exit_no_score"):
            bits.append("不按分数出场")
        # 跨周期因子：同一组因子、同一个窗口，把其中一个挪到周线上就是另一条
        # 策略了。列表里不写出来，两条配置看起来一模一样
        _xp = _cross_period_names(list(self.factor_keys or []) + list(
            self.timing_factor_keys or []), cfg.get("factor_periods"))
        if _xp:
            bits.append("跨周期·" + "/".join(_xp))
        if self.factor_mode == "ml":
            bits.append(f"模型={cfg.get('model', '')}")
            # 稀疏采样 / 合并训练都不在「窗口 / 阈值」里体现，而它们对结果的影响
            # 比窗口大得多：同一条 ML 策略开了稀疏是「只在突破那几根学」，不开是
            # 「每根都学」，两条完全不是一回事。列表上不写出来，用户会以为是重复项
            if str(cfg.get("sample_mode") or "all") == "trigger":
                bits.append("稀疏触发")
            if cfg.get("pooled"):
                bits.append("合训")
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
        if isinstance(cfg.rule_reverse, str):
            cfg.rule_reverse = cfg.rule_reverse.strip().lower() in ("1", "true", "yes", "on")
        else:
            cfg.rule_reverse = bool(cfg.rule_reverse)
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
        for f in ("cross_symbols", "multi_symbols", "timing_factor_keys"):
            if not isinstance(getattr(cfg, f), list):
                setattr(cfg, f, [])
        # ---- 老配置的多品种清单搬家 ----
        # v2.5.7 之前「多品种勾选」只有一个落脚点 `cross_symbols`（名字带 cross，
        # 但 `_current_config` 无条件写它，规则 / 因子族也在用）。读的时候搬进
        # `multi_symbols`：库里那些「19 个品种」的老条目一加载就是多品种，再存
        # 一遍写在新字段上，用户不用重新勾一遍 19 个
        if not cfg.multi_symbols and len(cfg.cross_symbols) >= 2:
            cfg.multi_symbols = list(cfg.cross_symbols)
        for f in ("timing_rule_params", "timing_factor_reverse", "timing_factor_cfg"):
            if not isinstance(getattr(cfg, f), dict):
                setattr(cfg, f, {})
        try:
            cfg.cross_rebalance = max(1, int(cfg.cross_rebalance or 1))
        except (TypeError, ValueError):
            cfg.cross_rebalance = 5
        # 单位只认三种；手改 JSON 写错 / 老配置文件里没有这个字段时，按模式
        # 补一个合理的默认（选品种按天，对冲按根）
        if getattr(cfg, "cross_rebalance_unit", None) not in ("bar", "day", "month"):
            cfg.cross_rebalance_unit = "day" if cfg.cross_mode == "select" else "bar"
        cfg.cross_top_pct = _clamp_pct(cfg.cross_top_pct, 10.0, 50.0)
        cfg.cross_bot_pct = _clamp_pct(getattr(cfg, "cross_bot_pct", 10.0), 10.0, 50.0)
        if cfg.cross_mode not in ("hedge", "select"):
            cfg.cross_mode = "hedge"
        if cfg.cross_weight not in ("risk", "equal", "score", "tier"):
            cfg.cross_weight = "risk"
        # 分档：档数夹到 0~6，份数只留正数。手改 JSON 写成字符串时不该让
        # 整个配置加载失败
        try:
            cfg.cross_tiers = max(0, min(6, int(cfg.cross_tiers or 0)))
        except (TypeError, ValueError):
            cfg.cross_tiers = 0
        if not isinstance(cfg.cross_tier_units, list):
            cfg.cross_tier_units = []
        cfg.cross_tier_units = [float(u) for u in cfg.cross_tier_units
                                if isinstance(u, (int, float)) and u > 0]
        if cfg.cross_pool_mode not in ("off", "single", "multi", "ml"):
            cfg.cross_pool_mode = "off"
        if not isinstance(cfg.cross_pool_keys, list):
            cfg.cross_pool_keys = []
        if not isinstance(cfg.cross_pool_reverse, dict):
            cfg.cross_pool_reverse = {}
        cfg.cross_pool_pct = _clamp_pct(cfg.cross_pool_pct, 60.0, 100.0)
        if cfg.cross_label_mode not in ("excess", "volret", "rank", "volret_rank"):
            cfg.cross_label_mode = "volret"
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
    """用户自己的策略库。**不含内置模板** —— 这个函数的返回值会被拿去存盘、
    删除，混进模板就是往用户的 json 里写一堆只读条目。"""
    return _load_list(LIB_FILE)


def is_template(sid: str) -> bool:
    return TPL.is_template(sid)


def library_view() -> list[StrategyConfig]:
    """界面上的「策略库」= 内置模板 + 我的。

    给列表、选择框用；要写盘的场合一律用 `load_library()`。
    模板排前面：新用户库里是空的，先看到的是能学的东西，而不是一片空白。
    """
    return TPL.all_templates() + load_library()


def save_library(items: list[StrategyConfig]) -> None:
    for c in items:
        _ensure_sid(c)
    _save(LIB_FILE, [c.to_dict() for c in items])


class NameConflict(ValueError):
    """名字撞上了。

    两种撞法处理方式不同，所以带上原因：

    - `builtin=True`：撞的是**内置模板**的名字。不能存 —— 左侧策略库把内置和
      「我的」分两组列出来，两组同名之后谁也指代不清。让用户换个名字
    - `builtin=False`：撞的是**自己库里**的一条。可以存，覆盖它就行，
      所以 `item` 带上那一条，界面拿它的名字去问「要覆盖吗」
    """

    def __init__(self, msg: str, *, builtin: bool = False,
                 item: StrategyConfig | None = None) -> None:
        super().__init__(msg)
        self.builtin = builtin
        self.item = item


def builtin_names() -> set[str]:
    """内置模板名字的归一化集合。"""
    return {norm_name(n) for n in TPL.names() if n}


def add_to_library(cfg: StrategyConfig, overwrite: bool = False) -> StrategyConfig:
    """存进策略库。同 sid 覆盖，否则追加。

    **名字不许和内置模板重复。** 左侧策略库把内置和「我的」分两组列出来，两边
    同名之后，用户和 Agent 都没法用名字指代其中任意一条（`find_library` 只能
    挑「我的」那条）。这种直接抛 `NameConflict(builtin=True)`，让调用方去改名。

    **和「我的」里的一条同名时**：`overwrite=False` 抛 `NameConflict(item=…)`，
    界面拿它去问「要覆盖吗」；`overwrite=True` 就**原地替换**那一条（位置、
    创建时间、sid 都沿用原来的）—— 用户改了两个参数再存一次，库里不该出现
    两条「白银动量60-1h」，也不该逼他先删再存。

    例外是**没起名**的配置：名字是「品种 · 策略」自动拼的，同品种同策略存两份
    参数不同的版本是正常用法，这时自动加序号（`…（2）`），不挡人。
    """
    items = load_library()
    # 内置模板不是库里的一条。存它等于「从模板另存一份到我的」：换新号，
    # 名字沿用模板名 —— 但模板名本身被内置占着，所以从（2）开始编号。
    # 用户载入模板、改两个数、点保存，不该先撞上一条「重名，换个名字」，
    # 他根本没起过名；也不该被自动命名成「AG_SHFE · 单因子 · 1 个因子」
    # 这种他自己没见过的名字。
    if TPL.is_template(cfg.sid):
        base = (cfg.name or (TPL.BY_KEY.get(cfg.sid) or {}).get("name", "")).strip()
        cfg.sid = ""
        if base:
            taken = {norm_name(c.title()) for c in items} | builtin_names()
            name, n = base, 2
            while norm_name(name) in taken:
                name = f"{base}（{n}）"
                n += 1
            cfg.name = name
    _ensure_sid(cfg)
    for i, old in enumerate(items):
        if old.sid == cfg.sid:
            # 同 sid 是「改这一条」。改名撞到**别的**条目就拦下来：那不是
            # 「存一份新的」，是「把这条改叫别人的名字」，覆盖掉别人等于
            # 悄悄删了一条策略，代价和意图完全不对等
            want = norm_name(cfg.title())
            dup = next((o for o in items
                        if o.sid != cfg.sid and norm_name(o.title()) == want), None)
            if dup is not None:
                # `item` 故意不给：这一条是「改名撞车」，不是「存一份新的」。
                # 带 item 出去，界面会弹「要覆盖吗」，点了覆盖又回到这条分支，
                # 转一圈还是存不进去 —— 不如直接说清「换个名字」
                raise NameConflict(
                    f"策略库里已经有「{dup.title()}」了。这是改名，不是存新的，"
                    "换一个名字再存。")
            items[i] = cfg
            save_library(items)
            return cfg

    if not str(cfg.name or "").strip():
        base = cfg.title()
        taken = {norm_name(c.title()) for c in items} | builtin_names()
        if norm_name(base) in taken:
            n = 2
            while norm_name(f"{base}（{n}）") in taken:
                n += 1
            cfg.name = f"{base}（{n}）"
    else:
        want = norm_name(cfg.title())
        if want in builtin_names():
            raise NameConflict(
                f"「{cfg.title()}」是内置策略的名字，自定义策略不能和它重名。"
                "换一个名字再存。",
                builtin=True)
        dup = next((o for o in items if norm_name(o.title()) == want), None)
        if dup is not None:
            if not overwrite:
                raise NameConflict(
                    f"策略库里已经有「{dup.title()}」了", item=dup)
            # 原地替换：位置、创建时间、sid 全部沿用。换个新 sid 的话，组合里
            # 引用它的那条、Agent 记住的那条就都指向一个不存在的 id 了
            idx = items.index(dup)
            if not cfg.created:
                cfg.created = dup.created
            cfg.name = dup.name
            cfg.sid = dup.sid
            items[idx] = cfg
            save_library(items)
            return cfg

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

    用户库里没有就**再找一遍内置模板**：新用户的库是空的，他嘴上说的十有八九
    是界面上看到的那条模板名（或者是 Agent 推荐的模板）。顺序不能反 —— 用户
    自己存的那条和模板同名时，他说的是自己那条。
    """
    want = str(name or "").strip()
    if not want:
        return None
    if TPL.is_template(want):
        return TPL.to_config(want)
    items = load_library()
    for c in items:
        if c.sid and c.sid == want:
            return c
    key = norm_name(want)
    if not key:
        return None
    hits = [c for c in items if norm_name(c.title()) == key]
    if hits:
        return hits[-1]
    for k in TPL.keys():
        cfg = TPL.to_config(k)
        if cfg is not None and norm_name(cfg.title()) == key:
            return cfg
    return None


def remove_from_library(sid: str) -> bool:
    """从库里删一条。内置模板删不掉 —— 它不在库里，返回 False 而不是报错。"""
    if TPL.is_template(sid):
        return False
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


# ------------------------------------------------------------------ 追单设置


@dataclass
class ChaseConfig:
    """追单设置：策略发单后没成交怎么追。

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
