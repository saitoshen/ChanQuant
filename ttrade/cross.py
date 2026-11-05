"""截面策略：把「谁强谁弱」当成信号，而不是「这一根该不该买」。

时序策略问的是「这个品种现在该不该持有」，截面策略问的是「这几个品种里谁最强、
谁最弱」。两个问题不一样，回测也不能共用一个引擎 —— 时序引擎一次只持有一个
品种的仓位，截面引擎每根 bar 要同时持有一篮子，多空两侧各一半。

两种玩法：

- **截面对冲**（`mode="hedge"`）：按截面分数排序，多最强的 top X%、空最弱的
  bottom X%，等权，定期调仓。净敞口约等于 0，赚的是「强弱差」。
- **截面选品种 + 时序交易**（`mode="select"`）：**池内所有品种**都交给各自的
  时序策略（规则或因子），由时序信号决定这一腿是多是空还是空仓。截面不负责
  「挑几只」—— 挑几只由第一步的筛池决定，截面分数在这里只决定**腿内谁多分点
  钱**（`weight="score"` / `"tier"`）。时序负责「什么时候」，截面负责「分多少」。

  两种模式对 `top_pct` / `bot_pct` 的用法不一样：对冲模式拿它定腿数，选品种
  模式**根本不看** —— 时序腿是逐品种独立开平的，先按截面砍掉一半品种等于替
  时序策略做了一半决定，而且「多最强的 X%、空最弱的 Y%」在只有一条腿的语境下
  也说不通（谁强谁弱已经由时序信号定了方向）。

分数怎么来：每个因子先算**时序**原值（复用 `factors.raw`），再在每根 bar 上做
**横截面** Z-score —— 这就是和时序因子策略最本质的区别。时序 Z-score 是「跟
自己的历史比」，截面 Z-score 是「跟同期的其他品种比」，后者才消得掉「整个板块
一起涨」这种共同因子。

成本按「换手 × 单边成本」扣：换手是权重变化的绝对值之和，单边成本按品种查
（`symbols_info` + `costs`）。不按笔数扣是因为截面策略天天在调仓，按笔算会把
成本算成 0。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrade import costs, factors, symbols_info
from ttrade import factor_strategy as fstrat
from ttrade.backtest import BacktestConfig, Trade
from ttrade.strategies import STRATEGIES, run_strategy

MIN_CROSS = 3          # 横截面上至少几个品种有值，才算得出一组强弱

# 风险平价用的波动率窗口（根）。60 根日线约三个月，5 分钟 48 根一天的话是
# 一天多一点 —— 都不至于慢到追不上波动率的变化，也不至于快到被一两根异常
# K 线带偏。少于 20 根算不出标准差时退回等权，别让新上市品种拿到 0 权重。
VOL_WINDOW = 60
VOL_MIN_OBS = 20

# 每折最多喂多少行给模型。截面是「品种 × 时间」堆起来的长表，6 个品种 × 十年
# 5 分钟就是 100 万行，一折要十几秒；几十折下来是几十分钟。树模型不缺这几百万
# 行里的重复信息，按等间隔抽到 2 万行，一折 0.5 秒左右，排序结论几乎不变。
ML_MAX_TRAIN_ROWS = 20_000


@dataclass
class CrossConfig:
    """截面策略的配置。`mode` 决定后面哪一组字段有效，无效的保留着不删。"""

    mode: str = "hedge"                 # hedge / select
    # 调仓间隔。`rebalance` 是**数量**，`rebalance_unit` 是单位：
    #   bar   —— 根（对冲模式用；5m 上 48 根约一天）
    #   day   —— 交易日
    #   month —— 自然月（按 21 个交易日折算）
    #
    # 为什么选品种模式按天 / 月给：那条路是「一篮子按权重整体换仓」，换一次等于
    # 把整篮子的仓位按新权重重下一遍，成本按换手吃。按「根」给的话 5 分钟周期上
    # 默认 5 根 = 25 分钟换一次，换手能把收益全吃掉 —— 而选品种本来就不该频繁。
    # 换算成根数要**按数据的实际密度**来（`bars_per_day`），不能写死 48：不同
    # 品种的夜盘时长不同，同一周期一天可能是 48 根也可能是 68 根。
    rebalance: int = 5
    rebalance_unit: str = "bar"         # bar / day / month
    # 腿数**只对冲模式用**。选品种模式下池内品种全进时序腿，这两个值不参与计算
    # （界面上也不显示）—— 见模块说明里那一段
    top_pct: float = 10.0               # 多头：选最强的百分之几
    bot_pct: float = 10.0               # 空头：选最弱的百分之几（可以和多头不同）
    long_only: bool = False
    # risk = 风险平价（按波动率倒数分配，默认）；equal = 等权；score = 按分数加权
    # tier = 按分数分档给份数（见 tiers / tier_units）
    weight: str = "risk"

    # ---- 资金分档（`weight="tier"` 时生效，参考 HL-trend 的 4/2/1 份）
    #
    # 选出来的几只不该一样重：排第一的比排第三的确定性高得多。做法是把选中的
    # 那一批按分数降序切成 `tiers` 等份，第 i 份拿 `tier_units[i]` 份钱。
    tiers: int = 0                      # 0/1 = 不分档
    tier_units: list = field(default_factory=list)   # 空则按 2^(n-1) 递减

    # ---- 第一步：选品种池（可选）
    #
    # 两步走的意义：**过滤**和**排序**用的不是同一套因子。金额 / 持仓这类
    # 「能不能交易」的因子适合筛池，动量 / 基差这类「谁强谁弱」的因子适合排序。
    # 混在一张分数里，金额大的品种会因为「金额因子 z 值高」被排到前面，
    # 但它涨不涨是另一回事 —— 两件事搅在一起，权重就被流动性带偏了。
    pool_mode: str = "off"              # off / single / multi / ml
    pool_keys: list = field(default_factory=list)
    pool_reverse: dict = field(default_factory=dict)
    pool_top_pct: float = 60.0          # 池里保留分数最高的百分之几
    pool_min: int = 0                   # 池子至少留几个（不够就全留）

    # 第一步（截面排名 + 筛池）用**哪个周期**的因子算。空 = 跟随策略周期。
    #
    # 为什么需要：5 分钟策略选品种，靠的往往是「日线成交额」这种慢变量 ——
    # 流动性是几天、几周的量级，用 5 分钟成交额去排名只会被当天的消息面带着
    # 走。反过来，5 分钟因子做排名等于让噪声决定买谁。
    #
    # 对齐规则见 `align_to_clock()`：源周期的分数整体 `shift(1)` 再 as-of 前向
    # 填充，**绝不让当天还没收完的那根参与排名**。
    step1_period: str = ""

    # 策略本身的周期（`frames` 是哪一档）。跨周期因子要靠它算「基准时刻」——
    # 周线 bar 的数据要到那周最后一个交易日收盘才齐，日线 bar 要等到当天 15:00。
    # 空 = 不知道，按「时间戳就是决策时刻」处理（分钟级本来就该这样）
    period: str = ""

    # 截面因子。三种模式和时序那边一样：single / multi / ml
    factor_mode: str = "multi"
    factor_keys: list = field(default_factory=list)
    factor_reverse: dict = field(default_factory=dict)
    factor_cfg: dict = field(default_factory=dict)   # 窗口 / 模型 / 训练窗口 …
    # 机器学习标签：excess = 收益 − 截面均值（老口径）
    #              volret = 收益 ÷ 波动 − 截面均值
    #              rank / volret_rank = 再转成截面分位
    label_mode: str = "volret"

    # 选品种之后怎么交易（只对 select 生效）
    timing_family: str = "rule"         # rule / factor
    timing_rule_key: str = ""
    timing_rule_params: dict = field(default_factory=dict)
    timing_factor_mode: str = ""
    timing_factor_keys: list = field(default_factory=list)
    timing_factor_reverse: dict = field(default_factory=dict)
    timing_factor_cfg: dict = field(default_factory=dict)

    def n_legs(self, n_valid: int) -> int:
        """多头持几个品种（**只对冲模式有意义**）。至少 1 个，最多不超过有效
        品种数的一半。选品种模式不走这里 —— 池内品种全进时序腿。"""
        return _legs(n_valid, self.top_pct)

    def n_short_legs(self, n_valid: int) -> int:
        """空头持几个品种（只对冲模式有意义）。默认和多头一样，但可以单独调。"""
        return _legs(n_valid, self.bot_pct)

    def step(self, bars_per_day: float | None = None) -> int:
        """调仓间隔换算成**根数**。

        `bar` 单位直接返回；`day` / `month` 要乘「一天几根」。密度没给时按日线
        处理（1 根/天）—— 调用方几乎都是先拿到行情面板再算，只有标签、日志这类
        纯文字的地方可能不带密度。
        """
        n = max(1, int(self.rebalance or 1))
        unit = str(self.rebalance_unit or "bar").lower()
        if unit == "bar":
            return n
        days = n * (21 if unit == "month" else 1)
        bpd = float(bars_per_day or 1.0)
        return max(1, int(round(days * bpd)))

    def step_text(self, bars_per_day: float | None = None) -> str:
        """给人看的调仓间隔：`每 5 根` / `每 5 天` / `每 1 个月`。"""
        n = max(1, int(self.rebalance or 1))
        unit = str(self.rebalance_unit or "bar").lower()
        if unit == "month":
            return f"每 {n} 个月"
        if unit == "day":
            return f"每 {n} 天"
        return f"每 {n} 根"


def _legs(n_valid: int, pct: float) -> int:
    k = int(round(n_valid * float(pct or 0) / 100.0))
    return max(1, min(k, max(1, n_valid // 2)))


@dataclass
class CrossResult:
    equity: pd.Series
    weights: pd.DataFrame
    metrics: dict = field(default_factory=dict)
    label: str = ""
    legs: int = 0                       # 最近一次调仓持了几个品种
    n_rebalance: int = 0
    # 这次实际用的调仓间隔（**根数**，已按数据密度换算过）和密度。界面的绩效
    # 表按「调仓区间」切收益，得用同一个数；自己再调一次 `cfg.step()` 会因为
    # 拿不到密度而退回 1 根/天，区间数直接翻几十倍
    step: int = 1
    bars_per_day: float = 1.0
    error: str = ""
    # 逐笔交易（每腿「同向连续持仓」算一笔）。绩效页的「调仓 N 次」说的是调仓
    # 动作，交易记录说的是持仓段 —— 两个数本来就不一样
    trades: list = field(default_factory=list)
    # 截面分数面板。画「截面因子分析」要用，`score_panel` 已经算过一遍，丢掉
    # 就得重算十几秒
    score: pd.DataFrame | None = None
    # 第一步用**别的周期**时，那份还没搬到策略时钟上的分数（原周期）。因子分析
    # 要在它自己的周期上算 —— 把日线分数铺到 5 分钟每根 bar 上再算 IC，等于把
    # 同一个观测重复几十遍，t 值会被吹起来
    score_step1: pd.DataFrame | None = None

    @property
    def ok(self) -> bool:
        return not self.error and self.equity is not None and len(self.equity) > 1


# ------------------------------------------------------------------ 面板


def close_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """收盘价面板（index × 品种）。缺口前向填充：一个品种停牌不该让整行作废。"""
    cols = {k: df["close"] for k, df in frames.items() if df is not None and len(df)}
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index().ffill()


def open_panel(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = {k: df["open"] for k, df in frames.items() if df is not None and len(df)}
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).sort_index().ffill()


def align_to_clock(panel: pd.DataFrame, index) -> pd.DataFrame:
    """把**别的周期**算出来的面板搬到策略的时钟上，只用已经收完的那一根。

    为什么不直接 `reindex(method="ffill")`：日线那根 K 线标的是**当天**，它的
    收盘价要到 15:00 才知道。5 分钟策略在当天 09:35 拿「当天的日线因子」排名，
    等于把收盘结果提前灌进上午的 bar —— IC 会被吹起来，实盘一上就露馅。

    所以先整体 `shift(1)`（丢掉当前这根，它还没收完），再按时间 as-of 前向填充。
    源周期比目标周期细的时候这条规则同样成立：1 小时 bar 标 10:00、11:00 才收完，
    10:35 那根 5 分钟 bar 只能看到 09:00 那根的结果。代价是窗口最前面一段没有
    分数（源周期第一根之前），那几根不参与排名，属于正常损耗。
    """
    if panel is None or panel.empty:
        return pd.DataFrame(index=pd.Index(index))
    p = panel.sort_index()
    idx = pd.DatetimeIndex(index)
    out = p.shift(1).reindex(idx, method="ffill")
    out.index = pd.Index(index)
    return out


def _row_z(row: pd.Series) -> pd.Series:
    """一根 bar 上的横截面 Z-score。样本不足就整行 NaN（不硬算）。"""
    v = row.dropna()
    if len(v) < MIN_CROSS:
        return pd.Series(np.nan, index=row.index)
    sd = float(v.std(ddof=0))
    if not sd or sd != sd:
        return pd.Series(0.0, index=row.index)
    return (row - float(v.mean())) / sd


def _cross_z(panel: pd.DataFrame) -> pd.DataFrame:
    """整块面板一次算完横截面 Z-score。

    和逐行调 `_row_z` 等价，但快两个数量级：`DataFrame.apply(axis=1)` 要走
    Python 层，6 个品种 × 8 个因子 × 3 万根就是 150 万次函数调用，一次回测
    要七十秒 —— 拖一下「调仓间隔」就卡住。pandas 的 `mean/std(axis=1)` 是
    按列向量化的，同样的数学，毫秒级。

    口径必须和 `_row_z` 完全一致：`ddof=0`、缺值跳过、有效样本不足 `MIN_CROSS`
    整行 NaN、横截面全一样（std=0）时记 0（没有强弱可分，不是 NaN）。
    """
    n = panel.notna().sum(axis=1)
    mean = panel.mean(axis=1)
    sd = panel.std(axis=1, ddof=0)
    z = panel.sub(mean, axis=0).div(sd.replace(0.0, np.nan), axis=0)
    z = z.mask(sd.eq(0.0), 0.0)
    return z.mask(n < MIN_CROSS)


def _factor_cfg(cfg: "CrossConfig"):
    """截面因子那一套参数（窗口 / 模型 / 训练窗口 …）转成 `FactorConfig`。"""
    d = dict(getattr(cfg, "factor_cfg", None) or {})
    return fstrat.FactorConfig(
        **{k: v for k, v in d.items()
           if k in fstrat.FactorConfig.__dataclass_fields__}
    )


def score_panel(frames: dict[str, pd.DataFrame], keys: list[str],
                reverse: dict | None = None, mode: str = "multi",
                fcfg=None, progress=None, cancel=None,
                label_mode: str = "volret") -> pd.DataFrame:
    """截面合成分数。三种模式和时序那边对齐：单因子 / 多因子 / 机器学习。

    - `single`：取第一个因子，横截面 Z-score
    - `multi`：每个因子先横截面 Z-score，再按方向取号、等权平均
    - `ml`：把所有品种的样本堆成一张长表，滚动训练一个「相对强弱」模型

    因子在某个品种上没数据（美股没持仓量、新品种没有展期）就留 NaN，合成分数
    会自动跳过 —— 一个品种缺一个因子不该让整组排名作废。

    `label_mode` 只对 ml 生效，口径见 `ml_score_panel`。
    """
    keys = [k for k in (keys or []) if k in factors.FACTORS]
    if not keys:
        raise ValueError("没有选中任何可用因子")
    rev = set(k for k, v in (reverse or {}).items() if v)

    if mode == "ml":
        return ml_score_panel(frames, keys, fcfg, progress=progress, cancel=cancel,
                              label_mode=label_mode)

    parts = []
    for key in keys:
        raw = {}
        for sym, df in frames.items():
            try:
                raw[sym] = factors.raw(df, key)
            except Exception:  # noqa: BLE001
                continue
        if len(raw) < MIN_CROSS:
            continue
        panel = pd.concat(raw, axis=1).sort_index().ffill()
        if not panel.notna().any().any():
            # 这个因子在这批品种上一个值都没有（日线库里没有展期 / 期权列）。
            # 留着它会**毒掉整个平均**：NaN + x 还是 NaN，十几个因子里只要有一个
            # 全空，合成分数就全空，界面上显示「截面分数全空」却查不出是谁
            continue
        z = _cross_z(panel)
        if key in rev:
            z = -z
        parts.append(z)
        if mode == "single":
            # 单因子就取**第一个真能算出数的**因子：勾了十几个因子时排在最前的
            # 往往是日线库里没数据的展期 / 期权因子，盲取 keys[0] 会直接报
            # 「这些因子在当前品种池上都没有足够数据」
            break
    if not parts:
        raise ValueError("这些因子在当前品种池上都没有足够数据")
    # 按时间取各因子的均值，**跳过缺值**：某个品种缺一个因子时不该让整行作废
    stacked = pd.concat(parts, axis=1, keys=list(range(len(parts))))
    out = stacked.T.groupby(level=1).mean().T
    return out.dropna(how="all")


def ml_score_panel(frames: dict[str, pd.DataFrame], keys: list[str], fcfg,
                   progress=None, cancel=None, label_mode: str = "volret") -> pd.DataFrame:
    """横截面机器学习打分：把所有品种的样本堆成一张长表，滚动训练。

    和时序 ML 的区别在**标签**。这里预测的是「同期比别的品种强多少」：每根 bar
    上先把各品种的未来收益减去当期截面均值，模型学的才是排序。用绝对收益当
    标签的话，模型会先学会「这段行情整体在涨」—— 而截面策略做多空对冲，那个
    beta 一分钱也拿不到，学得再准也没用。

    `label_mode` 决定「强多少」怎么度量：

    - `excess`：未来收益 − 截面均值。**老口径**，有量纲问题 —— 铁矿的日常波动
      是玉米的两三倍，同一套权重下去，模型会不自觉地偏爱高波动品种
    - `volret`（默认）：未来收益 ÷ 自身波动率 − 截面均值。把「涨了 3%」换算成
      「涨了 1.2 个标准差」，高波动品种不再天然占优
    - `rank`：未来收益的**截面分位**减 0.5。最抗极端值，代价是丢掉幅度信息
    - `volret_rank`：先除以波动再取分位

    训练样本按时间正序切折，训练集只取到 `b - horizon`：第 i 行的标签要等到
    i+h 才成立，混进训练集就是未来函数。
    """
    from ttrade import factor_strategy as fstrat
    from ttrade import mlmodels as ML
    from ttrade.factor_metrics import forward_return

    keys = [k for k in (keys or []) if k in factors.FACTORS]
    if len(keys) < 2:
        raise ValueError("机器学习至少需要 2 个因子")

    # 特征矩阵必须同构：某个品种缺这一列，整个模型就对不齐。取交集而不是各自
    # 补 0 —— 补 0 在标准化后的因子空间里是「中位数」，会让该品种凭空多出一个
    # 中性观测，排名被稀释
    avail = None
    for df in frames.values():
        got = set(factors.filter_available(df, keys))
        avail = got if avail is None else (avail & got)
    use = [k for k in keys if k in (avail or set())]
    if len(use) < 2:
        raise ValueError("这批品种上没有共同可用的 2 个因子（各自缺的数据不一样）")

    fcfg = fcfg or fstrat.FactorConfig()
    horizon = max(1, int(fcfg.horizon or 1))

    blocks = []
    lm = str(label_mode or "volret").lower()
    for sym, df in frames.items():
        fm = factors.factor_frame(df, use, window=fcfg.window, method=fcfg.method)
        if fm.empty:
            continue
        d = fm.copy()
        d["__fwd"] = forward_return(df, horizon).reindex(fm.index)
        if lm in ("volret", "volret_rank"):
            # 波动率用**同一根 bar 之前**的收益算，`shift` 不需要：rolling 本身就
            # 只回头看。窗口和风险平价那条路一致，两处口径不要分叉
            r = pd.Series(df["close"]).pct_change()
            v = r.rolling(VOL_WINDOW, min_periods=VOL_MIN_OBS).std()
            d["__vol"] = v.reindex(fm.index)
        d["__sym"] = sym
        blocks.append(d)
    if not blocks:
        raise ValueError("这批品种算不出因子矩阵")
    long = pd.concat(blocks).sort_index()
    long.index = pd.MultiIndex.from_arrays([long.index, long["__sym"]])
    long = long.sort_index()
    # 截面去均值：留下的是「相对强弱」，共同的市场 beta 被减掉了
    if lm in ("volret", "volret_rank"):
        vol = long["__vol"].where(long["__vol"] > 1e-12)
        # 波动率为 0 / 缺失的品种，用**同一根 bar 上其它品种的中位数**顶上。
        # 直接留 NaN 会让这个品种的标签整段作废；整行都算不出来（合成数据里
        # 恒定漂移的序列，波动率恰好是 0）时退回 1.0 —— 那等价于老口径，
        # 是「降级」不是「猜」
        vol = vol.fillna(vol.groupby(level=0).transform("median")).fillna(1.0)
        long["__rel"] = long["__fwd"] / vol
    else:
        long["__rel"] = long["__fwd"]
    long["__y"] = long["__rel"] - long.groupby(level=0)["__rel"].transform("mean")
    if lm in ("rank", "volret_rank"):
        # 分位化之后标签落在 [-0.5, 0.5]，和品种数无关。截面只有 5 个品种时
        # 分位是 0.1 一档，信息量比原始幅度少，但不会再被一个涨停带跑
        long["__y"] = long.groupby(level=0)["__y"].rank(pct=True) - 0.5

    X = long[use].to_numpy(dtype=float)
    X = np.where(np.isfinite(X), X, 0.0)
    Y = long["__y"].to_numpy(dtype=float)
    ts = long.index.get_level_values(0).unique()
    pos = np.searchsorted(ts, long.index.get_level_values(0).to_numpy())
    n = len(ts)

    bpd = fstrat.bars_per_day(pd.DatetimeIndex(ts))
    train_bars = fstrat._to_bars(fcfg.train_window, bpd)
    step_bars = fstrat._to_bars(fcfg.step, bpd)
    min_bars = min(int(fcfg.min_train), train_bars)
    if n <= min_bars + horizon + 10:
        raise ValueError("样本太短，跑不了机器学习的滚动训练")

    bounds = []
    t = min_bars
    while t < n:
        bounds.append((max(0, t - train_bars), t, min(n, t + step_bars)))
        t += step_bars

    pred = np.full(len(long), np.nan)
    done = 0
    for a, b, c in bounds:
        if cancel and cancel():
            raise fstrat.Cancelled()
        tr = (pos >= a) & (pos < b - horizon) & np.isfinite(Y)
        rows = np.nonzero(tr)[0]
        if len(rows) < 60:
            continue
        stride = max(1, len(rows) // ML_MAX_TRAIN_ROWS)
        rows = rows[::stride]
        model = ML.build(fcfg.model)
        try:
            model.fit(X[rows], Y[rows])
        except Exception:  # noqa: BLE001
            continue
        want = (pos >= b) & (pos < c)
        try:
            pred[want] = model.predict(X[want])
        except Exception:  # noqa: BLE001
            continue
        done += 1
        if progress:
            progress(done, len(bounds), f"第 {done}/{len(bounds)} 折训练完成")

    if cancel and cancel():
        raise fstrat.Cancelled()

    ser = pd.Series(pred, index=long.index, name="pred").dropna()
    if ser.empty:
        raise ValueError("机器学习一折都没训出来（样本或因子不足）")
    panel = ser.unstack().sort_index()
    # 预测值本身已经同尺度，再做一次横截面标准化是为了和单/多因子那条路的
    # 「分数」口径一致（按分数加权时两者才可比）。排序不变
    return _cross_z(panel).dropna(how="all")


def leverage_vector(frames: dict[str, pd.DataFrame], bt: BacktestConfig) -> pd.Series:
    """每个品种的名义敞口是权益的几倍 —— 和单品种那条路用**同一个仓位口径**。

    截面组合原来恒按 1 倍名义额算（每个品种 1/N），而「多品种回测」按
    「资金分配 ÷ 每手保证金」开仓：资金分配 30%、保证金 10% 就是 3 倍名义额。
    两条路各按各的杠杆跑，结果差一大截，界面上却一个字都没提 —— 用户把截面
    的「选 100% / 100 个月调仓」调成等于没选，还是对不上，就是这个原因。

    `unit`（名义敞口 = 权益）就是 1 倍；`capital` 是「资金分配比例 ÷ 保证金率」；
    `fixed` / `risk` 定不出「占权益多少」（手数取决于 ATR / 固定值），按 1 倍算。
    """
    keys = list(frames)
    mode = str(getattr(bt, "sizing", "unit") or "unit").lower()
    if mode != "capital":
        return pd.Series(1.0, index=keys, dtype=float)
    base = float(getattr(bt, "initial_cash", 0.0) or 0.0)
    budget = float(getattr(bt, "capital", 0.0) or 0.0) or base
    if base <= 0 or budget <= 0:
        return pd.Series(1.0, index=keys, dtype=float)
    pct = budget / base
    out = {}
    for key in keys:
        sym, _, ex = key.rpartition("_")
        margin = float(symbols_info.info(sym, ex).get("margin") or 0.0) or 0.1
        out[key] = pct / margin
    return pd.Series(out, dtype=float)


def cost_vector(frames: dict[str, pd.DataFrame], bt: BacktestConfig,
                profile: str | None = None) -> pd.Series:
    """每个品种「单边成本占名义额的比例」，用来乘换手。

    按手收的手续费要折成比例才能和权重相乘：`元/手 ÷ (价格 × 乘数)`。用中位价
    折，不用最新价 —— 最新价可能是个跳空，会把成本算歪。

    滑点要按**中位复权因子**换一次（`price_scale` 列）：一跳是真实世界里的固定
    金额，前复权价整体被缩放过，不换算的话历史越久远成本越便宜。这里只取一个
    比例给整段用，所以取中位数 —— 权重矩阵是逐根的，成本向量是一个数，用中位数
    和「按价格中位数折手续费」是同一个口径。
    """
    out = {}
    for sym, df in frames.items():
        if df is None or not len(df):
            continue
        sym_code, _, ex = sym.rpartition("_")
        c = BacktestConfig()
        costs.apply(c, sym_code, ex, profile)
        price = float(pd.Series(df["close"]).median() or 0.0)
        scale = 1.0
        if "price_scale" in df.columns:
            s = pd.to_numeric(df["price_scale"], errors="coerce").median()
            scale = float(s) if s == s and s > 0 else 1.0
        # 乘数要查品种信息：`BacktestConfig()` 的默认值是 1，拿它去折算「元/手」
        # 会把螺纹（乘数 10）的手续费算小 10 倍
        mult = float(symbols_info.info(sym_code, ex).get("mult") or 1.0)
        if price <= 0:
            continue
        if c.fee_per_lot > 0:
            fee = c.fee_per_lot / (price * max(mult, 1e-9))
        else:
            fee = c.fee_rate
        slip = c.slip_amount(price, scale) / price
        out[sym] = fee + slip
    return pd.Series(out, dtype=float)


# ------------------------------------------------------------------ 回测


def _metrics(equity: pd.Series, bpy: float) -> dict:
    eq = equity.dropna()
    if len(eq) < 2:
        return {"total_return": 0.0, "cagr": 0.0, "ann_vol": 0.0, "sharpe": 0.0,
                "max_dd": 0.0, "calmar": 0.0}
    ret = eq.pct_change().dropna()
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if eq.iloc[0] > 0 else 0.0
    years = max(len(eq) / max(bpy, 1.0), 1e-9)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    vol = float(ret.std(ddof=0) * np.sqrt(bpy)) if len(ret) else 0.0
    sharpe = float(ret.mean() * bpy / vol) if vol > 0 else 0.0
    peak = eq.cummax()
    dd = float((eq / peak - 1.0).min())
    calmar = float(cagr / abs(dd)) if dd < 0 else 0.0
    return {"total_return": total, "cagr": cagr, "ann_vol": vol,
            "sharpe": sharpe, "max_dd": dd, "calmar": calmar}


def trades_from_weights(weights: pd.DataFrame,
                        close: pd.DataFrame) -> list[Trade]:
    """把权重矩阵翻成逐笔交易：每个品种「同向连续持仓」算一笔。

    截面策略没有「一次开一手、一次平一手」的成交，仓位是按权重挂着的。要给出
    交易记录，只能按**持仓段**切：符号从 0 变成 +1 是开多，从 +1 变成 -1 是
    平多同时开空（两笔），回到 0 是平仓。

    成交时点按 `run()` 里的口径：第 t 根的权重赚的是 t→t+1 的收益，所以第 t 根
    收盘建仓、第 t+1 根收盘平仓。持仓段的最后一根如果一直没平，标成「持仓中」，
    平仓价取最后一根的收盘。

    权重**大小**变了不算新的一笔：风险平价每次调仓都会改大小，按大小切会把一笔
    三年的多头拆成上百笔，那不是交易记录，那是调仓日志。
    """
    out: list[Trade] = []
    if weights is None or weights.empty:
        return out
    idx = weights.index
    n = len(idx)
    for sym in weights.columns:
        if sym not in close:
            continue
        w = weights[sym].to_numpy(dtype=float)
        px = close[sym].to_numpy(dtype=float)
        sign = np.where(np.isfinite(w), np.sign(w), 0.0)
        i = 0
        while i < n:
            s = sign[i]
            if s == 0:
                i += 1
                continue
            j = i
            while j + 1 < n and sign[j + 1] == s:
                j += 1
            # 平仓落在「这一段之后的下一根」：第 j 根的权重赚的是 j→j+1 的收益
            end = j + 1
            opened = end >= n
            k = min(end, n - 1)
            p0, p1 = float(px[i]), float(px[k])
            ret = (p1 / p0 - 1.0) * s if p0 > 0 else 0.0
            # 备注列：进场看权重多大，出场看权重是归零还是反向。截面策略没有
            # ATR 出场规则，这两种就是全部的离场方式
            w0 = float(w[i]) if np.isfinite(w[i]) else 0.0
            if opened:
                why = "样本结束，仍持仓"
            elif end < n and np.isfinite(w[end]) and np.sign(w[end]) == -s:
                why = "权重反向"
            else:
                why = "权重归零"
            out.append(Trade(
                entry_date=idx[i], exit_date=idx[k],
                entry_price=p0, exit_price=p1, ret=float(ret),
                bars=int(k - i), direction=int(s), open_position=bool(opened),
                lots=0.0,          # 截面是权重不是手数，界面上显示「—」
                symbol=str(sym),
                entry_note=f"截面权重 {w0:+.2f} 进场",
                exit_note=why,
            ))
            i = j + 1
    out.sort(key=lambda t: (t.entry_date, t.exit_date))
    return out


def run(frames: dict[str, pd.DataFrame], cfg: CrossConfig, bt: BacktestConfig,
        profile: str | None = None, label: str = "",
        progress=None, cancel=None,
        step1_frames: dict[str, pd.DataFrame] | None = None,
        timing_extra: dict[str, dict] | None = None) -> CrossResult:
    """跑一遍截面策略。出错写进 `CrossResult.error`，不抛异常。

    `progress` / `cancel` 只给机器学习模式用：walk-forward 要跑几十折，界面
    那边靠它显示进度、也靠它中途叫停。

    `step1_frames` 是**第一步（排名 + 筛池）专用的行情**，周期可以和策略不同
    （典型：5 分钟策略用日线成交额筛池）。它的分数会按 `align_to_clock()` 搬到
    策略的时钟上，只用已经收完的那一根。传 None 就是「跟随策略周期」，老行为。

    `timing_extra` 是时序腿的**跨周期因子行情**：`{品种: {周期: 行情}}`，
    由调用方在主线程里读好（见 `factor_strategy.period_frames`）。时序腿的
    因子可以被指到别的周期上，截面那条腿（`factor_keys`）暂时还不支持。
    """
    if not frames or len(frames) < MIN_CROSS:
        return CrossResult(pd.Series(dtype=float), pd.DataFrame(),
                           error=f"截面策略至少要 {MIN_CROSS} 个品种，现在只有 {len(frames)} 个")
    # 第一步用哪份行情：配了周期且真的拿到了另一份才换，否则还是自己的
    src = step1_frames if (cfg.step1_period and step1_frames) else frames
    try:
        score_src = score_panel(src, list(cfg.factor_keys or []),
                                cfg.factor_reverse, mode=cfg.factor_mode or "multi",
                                fcfg=_factor_cfg(cfg), progress=progress, cancel=cancel,
                                label_mode=cfg.label_mode)
    except fstrat.Cancelled:
        raise                     # 用户点的「取消」，不是回测失败
    except Exception as exc:  # noqa: BLE001
        return CrossResult(pd.Series(dtype=float), pd.DataFrame(), error=str(exc))
    if score_src.empty:
        return CrossResult(pd.Series(dtype=float), pd.DataFrame(),
                           error="截面分数全空：选中的因子在这批品种上没有数据")

    # ---- 第一步：选品种池（可选）
    #
    # 放在这里而不是「先跑一遍再过滤结果」：池外品种一旦进了 `_rank`，腿数
    # 就会按全体品种算，多空各 10% 里有一半是流动性差的品种，等权之后它们的
    # 权重被放大，换手和滑点全上去了。
    if str(cfg.pool_mode or "off").lower() != "off" and cfg.pool_keys:
        try:
            pool_src = score_panel(src, list(cfg.pool_keys), cfg.pool_reverse,
                                   mode=cfg.pool_mode, fcfg=_factor_cfg(cfg),
                                   progress=progress, cancel=cancel,
                                   label_mode=cfg.label_mode)
        except fstrat.Cancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            return CrossResult(pd.Series(dtype=float), pd.DataFrame(),
                               error=f"第一步选品种失败：{exc}")
        score_src = _apply_pool(score_src, pool_src, cfg.pool_top_pct, cfg.pool_min)
        if score_src.dropna(how="all").empty:
            return CrossResult(pd.Series(dtype=float), pd.DataFrame(),
                               error="第一步筛完之后池子是空的 —— 把保留比例调大")

    # 分数从第一步的时钟搬到策略的时钟。**同周期不搬** —— `align_to_clock` 会
    # 整体 shift(1)，同一份数据上那样做等于凭空多滞后一根
    close = close_panel(frames)
    score = score_src if src is frames else align_to_clock(score_src, close.index)
    score = score.dropna(how="all")
    if score.empty:
        return CrossResult(pd.Series(dtype=float), pd.DataFrame(),
                           error="第一步的分数搬不到策略的时钟上（两边日期不重叠？）")

    # 截面回测现在整条路都在后台线程里跑（`gui.app.CrossWorker`）。用户拖一下
    # 滑杆就会重排一次，旧的那次得能自己退出，否则几次下来几个线程抢 CPU。
    # 分数面板那一步已经把 `cancel` 传下去了，这里只在几段「一口气算完」的
    # 大块之间补检查点 —— 再细就得进 numpy 循环，不值得
    def _bail() -> None:
        if cancel is not None and cancel():
            raise fstrat.Cancelled

    close = close.reindex(score.index).ffill()
    rets = close.pct_change().fillna(0.0)
    cvec = cost_vector(frames, bt, profile).reindex(score.columns).fillna(0.0)

    # 风险平价要各品种的波动率；等权 / 按分数用不上，不算它（滚动标准差在 5 分钟
    # 全历史上也要几百毫秒）
    vol = vol_panel(frames, score.index) if cfg.weight == "risk" else None
    _bail()
    # 调仓间隔要换算成**根数**：选品种模式按天 / 月给，得靠数据的实际密度落地。
    # 密度按分数面板的索引算（它就是回测的时钟），不能写死 48
    bpd = fstrat.bars_per_day(score.index)
    step = cfg.step(bpd)
    if cfg.mode == "select":
        weights = _select_weights(frames, score, cfg, vol, step=step,
                                  timing_extra=timing_extra)
    else:
        weights = _hedge_weights(score, cfg, vol, step=step)

    _bail()

    # 两个权重构造器都返回**稠密**矩阵（每一根 bar 都有值）：调仓日之间保持
    # 上一组的仓位，调仓日整体换一次。不要在这里再 ffill —— 那会把「主动平掉」
    # 和「还没调仓」混成同一个 NaN。
    weights = weights.reindex(columns=score.columns).fillna(0.0)
    # 杠杆：和单品种那条路同一个仓位口径。截面原来恒按 1 倍名义额算，
    # 而「多品种回测」按「资金分配 ÷ 保证金」开仓（默认 30% / 10% = 3 倍），
    # 两条路因此天然差一大截。放在这里统一乘，权重、换手、成本一起跟着走
    lev = leverage_vector(frames, bt).reindex(weights.columns).fillna(1.0)
    if float(lev.min()) != 1.0 or float(lev.max()) != 1.0:
        weights = weights * lev
    port_ret = (weights.shift(1) * rets).sum(axis=1)
    fee = (weights.diff().abs() * cvec).sum(axis=1).fillna(0.0)
    net = port_ret - fee

    # 单根亏 100% 以上就是爆仓，净值钉在 0 —— 负数净值没有意义，而且会把
    # 「最大回撤 / 夏普」这些比例型指标一起算崩（`combine` 平均时更糟）
    equity = (1.0 + net.clip(lower=-1.0)).cumprod()
    equity.iloc[0] = 1.0
    bpy = bt.bars_per_year or 252.0
    n_reb = len(range(0, len(score), step))
    # 成本占**初始资金**的比例：`fee` 是「当根换手 × 单边成本率」，乘上换手前的
    # 净值就是那一刻的金额。截面净值是归一化曲线，所以这里只能先记比例，
    # 到 `to_result` 里再乘回本金
    m = _metrics(equity, bpy)
    m["cost_pct"] = float((fee * equity.shift(1).fillna(1.0)).sum())
    # 成交额（同样是占初始资金的比例）：成本率 = 成本 ÷ 成交额，绩效页那一行
    # 显示的是它。换手 Σ|Δw| 就是「这一根买卖了净值的多少倍」
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    m["notional_pct"] = float((turnover * equity.shift(1).fillna(1.0)).sum())
    return CrossResult(
        equity=equity, weights=weights, metrics=m,
        label=label, legs=int(weights.abs().gt(0).sum(axis=1).max() or 0),
        n_rebalance=n_reb,
        step=step, bars_per_day=bpd,
        trades=trades_from_weights(weights, close),
        score=score,
        score_step1=(score_src if src is not frames else None),
    )


def to_result(out: "CrossResult", label: str = "", step: int = 0,
              bpy: float = 252.0, initial_cash: float = 100_000.0):
    """把 `CrossResult` 包成 `backtest.Result`，好让现成的绩效页直接吃得下。

    绩效表里的「交易」还是按**调仓区间**算的（区间收益为正算赢）：截面策略的
    仓位是按权重挂着的，按成交笔数算胜率会全是 0。交易记录那一页用的是
    `trades_from_weights` 切出来的持仓段，两者口径不同，各管一处。

    `step` 不给就用 `out.step` —— 那个是引擎按数据密度换算好的**根数**。
    调用方自己拿 `cfg.step()` 传进来的话，选品种模式（按天 / 月）会因为手上
    没有密度而退回 1 根/天，区间数凭空多出几十倍，胜率、盈亏比全跟着错。
    """
    from ttrade.backtest import Result, _metrics

    eq = out.equity.dropna()
    w = out.weights.reindex(eq.index).fillna(0.0)
    has_long = (w > 0).any(axis=1)
    has_short = (w < 0).any(axis=1)
    pos = pd.Series(0.0, index=eq.index)
    pos[has_long.to_numpy()] = 1.0
    pos[(has_short & ~has_long).to_numpy()] = -1.0

    m = dict(_metrics(eq, pos, [], bpy))
    ret = eq.pct_change().fillna(0.0)
    step = max(1, int(step or out.step or 1))
    seg: list[float] = []
    for i in range(0, len(ret), step):
        chunk = ret.iloc[i:i + step]
        if len(chunk):
            seg.append(float((1.0 + chunk).prod() - 1.0))
    n = len(seg)
    wins = [x for x in seg if x > 0]
    losses = [x for x in seg if x <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    m["n_trades"] = n
    m["n_long"] = n if bool(has_long.any()) else 0
    m["n_short"] = n if bool(has_short.any()) else 0
    m["win_rate"] = (len(wins) / n) if n else 0.0
    m["avg_trade"] = (sum(seg) / n) if n else 0.0
    m["profit_factor"] = ((gross_win / gross_loss) if gross_loss > 0
                          else (999.0 if gross_win > 0 else 0.0))
    m["avg_bars"] = float(step)
    m["best_trade"] = max(seg) if seg else 0.0
    m["worst_trade"] = min(seg) if seg else 0.0
    # 绩效表里「期末资金」这一行读的是 `final_equity`，`backtest._metrics` 不算
    # 它（那是 `run_backtest` 事后补的），这里得自己补上
    m["final_equity"] = float(eq.iloc[-1])
    # 成本：截面净值是归一化曲线，金额只能按本金折回去。换手是逐根扣的，
    # 所以「总成本」在这里是「如果本金是 initial_cash，这一路换了多少次手」
    cost_pct = float(out.metrics.get("cost_pct") or 0.0)
    m["total_cost"] = cost_pct * float(initial_cash or 0.0)
    m["avg_cost"] = float(m["total_cost"] / n) if n else 0.0
    # 成本率：两个比例相除，本金直接约掉（乘回金额只是为了让「成交额」这一列
    # 和 `total_cost` 同单位，界面上真要显示也能用）
    notional_pct = float(out.metrics.get("notional_pct") or 0.0)
    m["notional"] = notional_pct * float(initial_cash or 0.0)
    m["cost_rate"] = float(cost_pct / notional_pct) if notional_pct > 0 else 0.0
    return Result(equity=eq, position=pos, trades=list(out.trades),
                  metrics=m, label=label)


@dataclass
class CrossReport:
    """截面因子分析的结论。表格给数字，三条曲线给形状。"""

    table: pd.DataFrame                 # index=因子键（含 "__score__"），列见 `_REP_COLS`
    ic: pd.Series                       # 合成分数的逐期截面 IC
    quantile: pd.Series                 # 分位组的平均前向相对收益（index=1..q）
    spread: pd.Series                   # 多空价差累计净值
    horizon: int = 5
    n_bars: int = 0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error and self.table is not None and not self.table.empty


_REP_COLS = ("label", "ic", "ic_mean", "ir", "ic_pos", "t", "cover")


def _row_ic(panel: pd.DataFrame, fwd: pd.DataFrame, min_n: int = MIN_CROSS) -> pd.Series:
    """逐行（每根 bar）的横截面 Spearman 秩相关。

    秩相关而不是 Pearson：截面只有 6 个品种，Pearson 被一个极端值带偏得太厉害。
    `rank(axis=1)` 一次排完整张表，比逐行调 `spearman()` 快两个数量级。
    """
    rz = panel.rank(axis=1)
    rf = fwd.rank(axis=1)
    both = rz.notna() & rf.notna()
    n = both.sum(axis=1)
    zc = rz.where(both).sub(rz.where(both).mean(axis=1), axis=0)
    fc = rf.where(both).sub(rf.where(both).mean(axis=1), axis=0)
    num = (zc * fc).sum(axis=1)
    den = np.sqrt((zc ** 2).sum(axis=1) * (fc ** 2).sum(axis=1))
    ic = num / den.replace(0.0, np.nan)
    return ic.where(n >= max(2, min_n))


def _rel_fwd(frames: dict[str, pd.DataFrame], index, horizon: int) -> pd.DataFrame:
    """截面相对的前向收益：每根 bar 减去当期的横截面均值。

    和机器学习那边的标签是同一个口径 —— 多空对冲拿不到「整个板块一起涨」的
    beta，因子要解释的是「谁比谁强」，不是「大家一起涨了多少」。
    """
    from ttrade.factor_metrics import forward_return

    cols = {}
    for sym, df in frames.items():
        if df is None or df.empty:
            continue
        cols[sym] = forward_return(df, horizon).reindex(index)
    if not cols:
        return pd.DataFrame(index=index)
    fwd = pd.concat(cols, axis=1).reindex(index)
    return fwd.sub(fwd.mean(axis=1), axis=0)


def factor_report(frames: dict[str, pd.DataFrame], keys: list[str],
                  score: pd.DataFrame, fcfg=None, reverse: dict | None = None,
                  quantiles: int = 5) -> CrossReport:
    """截面因子分析：每个因子在这批品种上「排得准不准」。

    和时序那套因子分析的差别就在**口径**：时序算的是「这个因子跟自己历史比，
    高低能不能预测自己的涨跌」；这里算的是「每根 bar 上，因子给这批品种排的
    名次，和它们下一段相对涨跌的名次对不对得上」。同一个因子两边的 IC 可以
    差很远 —— 动量在时序上常常是负的（追涨杀跌），在截面上经常是正的（强的
    继续强）。
    """
    from ttrade.factor_metrics import newey_west_t

    if score is None or score.empty:
        return CrossReport(pd.DataFrame(), pd.Series(dtype=float),
                           pd.Series(dtype=float), pd.Series(dtype=float),
                           error="没有截面分数")
    # `fcfg` 既可能是 FactorConfig（脚本里直接调），也可能是 CrossConfig 里那个
    # 序列化过的 dict（GUI 走的是这条路）
    if isinstance(fcfg, dict):
        horizon = max(1, int(fcfg.get("horizon", 5) or 5))
    else:
        horizon = max(1, int(getattr(fcfg, "horizon", 5) or 5))
    rev = {k for k, v in (reverse or {}).items() if v}
    index = score.index
    fwd = _rel_fwd(frames, index, horizon)

    rows = []
    panels: dict[str, pd.DataFrame] = {}
    for key in keys or []:
        if key not in factors.FACTORS:
            continue
        raw = {}
        for sym, df in frames.items():
            try:
                raw[sym] = factors.raw(df, key)
            except Exception:  # noqa: BLE001
                continue
        if len(raw) < MIN_CROSS:
            continue
        panel = pd.concat(raw, axis=1).sort_index().ffill().reindex(index)
        if not panel.notna().any().any():
            continue
        z = _cross_z(panel)
        if key in rev:
            z = -z
        panels[key] = z
        rows.append((key, factors.FACTORS[key].label, z))

    # 合成分数也占一行：它是「这一堆因子合起来」的实际排序依据
    rows.append(("__score__", "合成分数（实际用于排序）", score))

    table = []
    for key, label, panel in rows:
        ic = _row_ic(panel, fwd)
        valid = ic.dropna()
        n = int(panel.notna().sum().sum())
        tot = int(panel.shape[0] * max(1, panel.shape[1]))
        rec = {"key": key, "label": label, "n": int(len(valid)),
               "cover": (n / tot) if tot else 0.0,
               "ic": float(valid.mean()) if len(valid) else float("nan"),
               "ic_mean": float(valid.mean()) if len(valid) else float("nan"),
               "ic_std": float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
               "ic_pos": float((valid > 0).mean()) if len(valid) else float("nan"),
               "t": float(newey_west_t(valid.to_numpy())) if len(valid) > 12
                    else float("nan")}
        sd = rec["ic_std"]
        rec["ir"] = (rec["ic_mean"] / sd) if sd and np.isfinite(sd) and sd > 0 \
            else float("nan")
        table.append(rec)

    rep = pd.DataFrame(table).set_index("key") if table else pd.DataFrame()

    # 分位收益 + 多空价差：用合成分数分组，看单调性和「两头差多少」
    qret = pd.Series(dtype=float)
    spread = pd.Series(dtype=float)
    ic_score = pd.Series(dtype=float)
    if "__score__" in rep.index:
        ic_score = _row_ic(score, fwd)
        q = max(2, int(quantiles))
        pct = score.rank(axis=1, pct=True)
        bucket = np.ceil(pct.to_numpy() * q)
        means = {}
        for b in range(1, q + 1):
            mask = np.isfinite(bucket) & (bucket == b)
            means[b] = fwd.where(mask).mean(axis=1)
        qdf = pd.DataFrame(means, index=index)
        qret = qdf.mean().rename_axis("bucket")
        top, bot = qdf[q], qdf[1]
        spread = (1.0 + (top - bot).fillna(0.0)).cumprod()

    return CrossReport(table=rep, ic=ic_score, quantile=qret, spread=spread,
                       horizon=horizon, n_bars=int(len(score)))


def vol_panel(frames: dict[str, pd.DataFrame], index) -> pd.DataFrame:
    """各品种的滚动波动率（收益率的滚动标准差），对齐到截面面板的索引上。

    用**收益率**而不是价格：铜一手 7 万、玉米一手 3 千，价格量级差 20 倍，拿价格
    标准差当风险，玉米会永远拿到最大权重。收益率才可比。

    `shift(1)` 不是为了防未来函数（`run()` 里权重本来就 `shift(1)` 才成交），
    而是让第 t 根的权重只用 t 之前已经收完的收益 —— 否则同一根 bar 的收盘价
    既决定波动率又决定仓位，回测会比实盘乐观。
    """
    close = close_panel(frames).reindex(index).ffill()
    rets = close.pct_change()
    vol = rets.rolling(VOL_WINDOW, min_periods=VOL_MIN_OBS).std()
    return vol.reindex(index).ffill()


def _rp_share(vol_row: np.ndarray, mask: np.ndarray, total: float) -> np.ndarray:
    """风险平价的权重：选中的那几只里，波动率越低分到的钱越多。

    严格的风险平价要解一个优化（让每只的风险贡献相等），这里用**逆波动率**：
    `w_i ∝ 1/σ_i`。它在「各品种相关性相同」时就是精确解，相关系数不一致时是
    业界最常用的近似 —— 比等权好得多，而且一行公式，看得懂也跑得快。

    波动率算不出来（新品种还没攒够 60 根）时退回等权：给 0 权重等于把它踢出
    组合，可它明明是刚被截面选中的强者。
    """
    out = np.zeros(len(vol_row), dtype=float)
    idx = np.nonzero(mask)[0]
    if not len(idx):
        return out
    v = vol_row[idx]
    inv = np.where(np.isfinite(v) & (v > 1e-12), 1.0 / np.maximum(v, 1e-12), np.nan)
    if not np.isfinite(inv).any():
        out[idx] = total / len(idx)
        return out
    # 缺一个就用这一腿的中位数顶上：既不让它变成 0 权重，也不让它因为「分母小」
    # 反而拿到最大权重
    inv = np.where(np.isfinite(inv), inv, np.nanmedian(inv))
    out[idx] = total * inv / inv.sum()
    return out


def _tier_units(cfg: CrossConfig) -> list[float]:
    """每档的份数。没配就按 2^(n-1) 递减：3 档是 4/2/1，4 档是 8/4/2/1。

    为什么默认**指数**递减而不是等差（3/2/1）：排第一的那只和排第三的那只，
    确定性差得不是「一倍」，是一个数量级。HL-trend 用的是 4/2/1，同一个道理。
    """
    units = [float(u) for u in (cfg.tier_units or []) if float(u) > 0]
    if not units:
        n = max(2, int(cfg.tiers or 2))
        units = [2.0 ** (n - 1 - i) for i in range(n)]
    return units


def _tier_share(mat_row: np.ndarray, mask_row: np.ndarray, tiers: int,
                units: list[float], ascending: bool = False) -> np.ndarray:
    """把一行里选中的几只按分数切成 `tiers` 段，各段拿 `units[i]` 份，归一化到 1。

    `ascending=True` 给空头用：空头要的是「最弱的那几个」，按分数升序排，
    否则第一档会分给空头里**最强**的那只 —— 正好反了。
    """
    out = np.zeros(len(mat_row), dtype=float)
    idx = np.nonzero(mask_row)[0]
    if not len(idx):
        return out
    v = np.nan_to_num(mat_row[idx], nan=(np.inf if ascending else -np.inf))
    order = idx[np.argsort(v if ascending else -v, kind="stable")]
    chunks = np.array_split(order, min(max(1, int(tiers)), len(order)))
    for ci, ch in enumerate(chunks):
        if not len(ch):
            continue
        u = units[ci] if ci < len(units) else units[-1]
        out[ch] = float(u)
    total = out.sum()
    return out / total if total > 0 else out


def _apply_pool(score: pd.DataFrame, pool: pd.DataFrame,
                top_pct: float, min_keep: int = 0) -> pd.DataFrame:
    """第一步：把池外的品种打成 NaN，后面那一步就只看池内。

    用**逐 bar 的横截面分位**判池，不是「整段一次算完再定池」：金额因子每天
    都在变，拿全样本的排名定池等于用了未来信息。`rank(pct=True)` 是逐行的，
    天然 PIT。

    池因子那天整行没值（新品种上市前、数据缺口）时**不筛**，原样放行 ——
    宁可在缺数据的那几天少一层过滤，也不要凭空整组空仓。
    """
    p = pool.reindex(index=score.index, columns=score.columns).ffill()
    if p.empty or not p.notna().any().any():
        return score
    # **升序**排名：分数越高（越该留在池里）名次越大，1.0 是最强。
    # 用降序的话 `rank=1.0` 会落在最弱的那只身上，`<= pct` 就变成「留下最弱的
    # 一半」—— 金额因子下这正好是「只留最不流动的品种」，完全反了
    r = p.rank(axis=1, ascending=True, pct=True)
    thr = pd.Series(1.0 - float(top_pct) / 100.0, index=p.index)
    if min_keep:
        n = p.notna().sum(axis=1).replace(0, np.nan)
        thr = np.minimum(thr, (1.0 - float(min_keep) / n).fillna(0.0))
    keep = r.gt(thr, axis=0) & p.notna()
    blank = p.notna().sum(axis=1) < MIN_CROSS
    keep = keep | np.repeat(blank.to_numpy()[:, None], p.shape[1], axis=1)
    return score.where(keep)


def _hedge_weights(score: pd.DataFrame, cfg: CrossConfig,
                   vol: pd.DataFrame | None = None,
                   step: int | None = None) -> pd.DataFrame:
    """多强空弱。多空各占一半名义，净敞口 ≈ 0。

    多空腿数可以不一样（`top_pct` / `bot_pct`）：多 10% 空 10% 是常见的对称
    写法，但「多最强的 10%、空最弱的 20%」也讲得通（下跌时弱者更弱）。两边
    各自按自己的腿数平分 0.5 的名义额，净敞口仍然是 0。

    腿内怎么分（`weight`）：等权 / 按分数 / 风险平价 / 分档。风险平价是**每一腿
    内部**按 1/σ 分 —— 波动大的品种少给钱，多空两侧各自的风险贡献才拉得平；
    跨腿的 0.5 / 0.5 不动，否则净敞口就不是 0 了。
    """
    rows, top, bot, k_long, k_short, ok = _rank(score, cfg, step=step)
    inv_l = 1.0 / np.maximum(k_long, 1)
    inv_s = 1.0 / np.maximum(k_short, 1)
    if cfg.weight == "tier" and int(cfg.tiers or 0) >= 2:
        mat = score.to_numpy(dtype=float)[rows]
        units = _tier_units(cfg)
        w = np.zeros(mat.shape, dtype=float)
        for i in range(len(rows)):
            if not ok[i]:
                continue
            w[i] = 0.5 * _tier_share(mat[i], top[i], int(cfg.tiers), units)
            if not cfg.long_only:
                w[i] -= 0.5 * _tier_share(mat[i], bot[i], int(cfg.tiers), units,
                                          ascending=True)
    elif cfg.weight == "risk" and vol is not None:
        vmat = vol.to_numpy(dtype=float)[rows]
        w = np.zeros(vmat.shape, dtype=float)
        for i in range(len(rows)):
            if not ok[i]:
                continue
            w[i] = _rp_share(vmat[i], top[i], 0.5)
            if not cfg.long_only:
                w[i] -= _rp_share(vmat[i], bot[i], 0.5)
    elif cfg.weight == "score":
        mat = score.to_numpy(dtype=float)[rows]
        pos = np.where(top, np.clip(np.nan_to_num(mat), 0.0, None), 0.0)
        s = pos.sum(axis=1)
        good = s > 0
        share = np.zeros_like(pos)
        share[good] = pos[good] / s[good, None]
        w = np.where(good[:, None], 0.5 * share, top * (0.5 * inv_l)[:, None])
        if not cfg.long_only:
            w = w - bot * (0.5 * inv_s)[:, None]
    else:
        w = top * (0.5 * inv_l)[:, None]
        if not cfg.long_only:
            w = w - bot * (0.5 * inv_s)[:, None]
    return _place(score, rows, w, ok)


def _rank(score: pd.DataFrame, cfg: CrossConfig, all_in: bool = False,
          step: int | None = None):
    """调仓日上「谁在前、谁在后」一次性排完。

    返回 `(行号, 多头掩码, 空头掩码, 多头几个, 空头几个, 有效行掩码)`。
    逐行 `nlargest` 在 5 分钟全历史上是 4 万多次 pandas 调用、七十秒；这里按轴
    排序一次就够 —— 拖一下「调仓间隔」不至于卡住半分钟。

    `all_in=True` 是给「选品种 + 时序」那条路用的：**不按 top_pct 截断**，有分数
    的品种全进（`k_long` 直接取有效数）。选品种模式没有「多几个空几个」这回事，
    腿数由时序信号逐品种决定；这里只负责把「有分数的」挑出来，排序仍然有用
    （`weight="score"` / `"tier"` 要靠它分钱）。

    NaN 必须排到两端之外：多头侧按 `-inf` 参与排序，空头侧按 `+inf`，这样
    「有效品种不足」的品种永远不会被选中，和 `dropna()` 之后再排等价。
    """
    step = max(1, int(step or cfg.step()))
    rows = np.arange(0, len(score), step)
    mat = score.to_numpy(dtype=float)[rows] if len(rows) else np.zeros((0, score.shape[1]))
    n_valid = np.sum(~np.isnan(mat), axis=1)
    ok = n_valid >= MIN_CROSS

    def _n(pct: float) -> np.ndarray:
        k = np.round(n_valid * float(pct or 0) / 100.0).astype(int)
        k = np.maximum(1, np.minimum(k, np.maximum(1, n_valid // 2)))
        return np.where(ok, k, 0)

    if all_in:
        # 全进：k 就是有效数，不受 `n_valid // 2` 那个上限约束（那是给对冲
        # 留的 —— 多空各占一半名义，一边吃掉全部品种就变成单边了）
        k_long = np.where(ok, n_valid, 0)
        k_short = np.zeros(len(rows), dtype=int)
    else:
        k_long = _n(cfg.top_pct)
        k_short = (np.zeros(len(rows), dtype=int) if cfg.long_only
                   else _n(cfg.bot_pct))
    top = np.zeros(mat.shape, dtype=bool)
    bot = np.zeros(mat.shape, dtype=bool)
    width = int(max(k_long.max() if len(k_long) else 0,
                    k_short.max() if len(k_short) else 0))
    if width:
        hi = np.argsort(-np.where(np.isnan(mat), -np.inf, mat), axis=1, kind="stable")
        lo = np.argsort(np.where(np.isnan(mat), np.inf, mat), axis=1, kind="stable")
        for j in range(width):
            sel = np.nonzero(k_long > j)[0]
            if len(sel):
                top[sel, hi[sel, j]] = True
            sel = np.nonzero(k_short > j)[0]
            if len(sel):
                bot[sel, lo[sel, j]] = True
    return rows, top, bot, k_long, k_short, ok


def _place(score: pd.DataFrame, rows, w: np.ndarray, ok) -> pd.DataFrame:
    """把调仓日的权重放回整条时间轴，中间那段沿用上一组。

    有效品种不足的行**留空**而不是清零：`ffill` 之后就是「还没到下一个调仓日，
    先按上一组持仓挂着」，和逐行那版 `continue` 的语义一致。清零等于凭空平仓，
    会在净值和换手上多出一批假动作。
    """
    dense = np.full((len(score), score.shape[1]), np.nan)
    if len(rows):
        dense[rows[ok]] = w[ok]
    return pd.DataFrame(dense, index=score.index,
                        columns=score.columns).ffill().fillna(0.0)


def _select_weights(frames: dict[str, pd.DataFrame], score: pd.DataFrame,
                    cfg: CrossConfig, vol: pd.DataFrame | None = None,
                    step: int | None = None,
                    timing_extra: dict[str, dict] | None = None) -> pd.DataFrame:
    """选品种 + 时序：**池内每个品种**各拿一份钱，由自己的时序信号决定方向。

    时序信号在**各自的行情表**上算（`strategies.run_strategy` / 因子策略），算完
    再对齐到截面面板的索引上。这样 5 分钟周期里各品种交易时段不同也不会串。

    这里**不看 `top_pct` / `bot_pct`**（`all_in=True`）：选品种模式没有「多几个
    空几个」这回事 —— 方向由时序信号逐品种给，先按截面砍掉一半品种等于替时序
    策略做了一半决定。要缩品种池就用第一步的筛池（`pool_top_pct`），那是按
    「能不能交易」筛的，语义清楚得多。

    钱怎么分看 `weight`：等权 / 按分数 / 风险平价（默认）/ 分档。分的是**整个
    池子**的 1.0 名义额，所以没出信号的那几只，那部分钱就空着（现金）。**不把
    空出来的份额转给有信号的品种** —— 那样信号越稀仓位越集中，等于偷偷加杠杆，
    回测和实盘都会对不上。

    分档（`weight="tier"`）在这条路尤其对路：池里第 1 名和第 40 名的确定性差得
    远，等权等于把最没把握的那只和最有把握的那只当成一回事。
    """
    sigs: dict[str, pd.Series] = {}
    for sym, df in frames.items():
        sigs[sym] = _timing_signal(df, cfg, (timing_extra or {}).get(sym))
    sig_panel = pd.concat(sigs, axis=1).sort_index().reindex(score.index).ffill().fillna(0.0)

    rows, top, _bot, k_long, _k_short, ok = _rank(score, cfg, all_in=True,
                                                  step=step)
    if cfg.weight == "tier" and int(cfg.tiers or 0) >= 2:
        mat = score.to_numpy(dtype=float)[rows]
        units = _tier_units(cfg)
        w = np.zeros(mat.shape, dtype=float)
        for i in range(len(rows)):
            if ok[i]:
                w[i] = _tier_share(mat[i], top[i], int(cfg.tiers), units)
    elif cfg.weight == "risk" and vol is not None:
        vmat = vol.to_numpy(dtype=float)[rows]
        w = np.zeros(vmat.shape, dtype=float)
        for i in range(len(rows)):
            if ok[i]:
                w[i] = _rp_share(vmat[i], top[i], 1.0)
    elif cfg.weight == "score":
        # 按分数加权：只取正的分数当权重（截面 z-score 有一半是负数，负分给
        # 负权重等于让「最弱的那只」反向占钱，那是另一个策略了）。整行都是
        # 负分时退回等权 —— 那天没有任何一只「强」，不该因此整组空仓
        mat = score.to_numpy(dtype=float)[rows]
        pos = np.where(top, np.clip(np.nan_to_num(mat), 0.0, None), 0.0)
        s = pos.sum(axis=1)
        good = s > 0
        share = np.zeros_like(pos)
        share[good] = pos[good] / s[good, None]
        inv = 1.0 / np.maximum(k_long, 1)
        w = np.where(good[:, None], share, top * inv[:, None])
    else:
        inv = 1.0 / np.maximum(k_long, 1)
        w = top * inv[:, None]
    held = _place(score, rows, w, ok)

    # 选中的那一段里，按时序信号决定持有方向；没选中就是 0。
    # 不允许做空的市场（美股）要把负信号砍掉 —— 对冲模式早就在 `_hedge_weights`
    # 里处理了，这条路以前漏了，美股截面上照样在做空
    sig = sig_panel.to_numpy(dtype=float)
    if cfg.long_only:
        sig = np.clip(sig, 0.0, None)
    return held * pd.DataFrame(sig, index=held.index, columns=held.columns)


def _timing_signal(df: pd.DataFrame, cfg: CrossConfig,
                   extra: dict | None = None) -> pd.Series:
    """单个品种的时序信号（-1/0/1）。`extra` 是它的跨周期因子行情。"""
    if cfg.timing_family == "rule":
        key = cfg.timing_rule_key
        if key not in STRATEGIES:
            return pd.Series(0.0, index=df.index)
        try:
            out = run_strategy(df, key, dict(cfg.timing_rule_params or {}))
            return pd.Series(out["signal"], index=df.index).astype(float).fillna(0.0)
        except Exception:  # noqa: BLE001
            return pd.Series(0.0, index=df.index)
    keys = [k for k in (cfg.timing_factor_keys or []) if k in factors.FACTORS]
    if not keys:
        return pd.Series(0.0, index=df.index)
    try:
        from ttrade import factor_strategy as fstrat

        fc = fstrat.FactorConfig(**{k: v for k, v in (cfg.timing_factor_cfg or {}).items()
                                    if k in fstrat.FactorConfig.__dataclass_fields__})
        fc.reverse = dict(cfg.timing_factor_reverse or {})
        out = fstrat.run_factor_strategy(
            df, cfg.timing_factor_mode or "single", keys, fc,
            extra=extra, base_period=cfg.period or "")
        return pd.Series(out.df["signal"], index=out.df.index).astype(float).fillna(0.0)
    except Exception:  # noqa: BLE001
        return pd.Series(0.0, index=df.index)
