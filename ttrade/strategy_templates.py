"""内置策略模板 —— 随版本发布的因子组合范例。

## 为什么要内置

策略库里躺着几十条用户自己调出来的配置。它们值钱的地方不是「能跑」，是
**它们演示了因子怎么组合**：哪几个因子放一起、哪个要取反、方向交给模型还是
写死、阈值收紧到多少。可这些配置散在 `%LOCALAPPDATA%/ChanQuant/strategy_lib.json`
里 —— 换台机器就没了，升级版本也不会跟着走，新用户更是一条都看不到。

所以把这批配置收进代码：跟着版本 release，谁装都能看见，谁也改不坏。

## 模板是只读的

模板不写进用户的 `strategy_lib.json`，是每次现算出来的 `StrategyConfig`：

- 改不了 —— 界面上「删除」按钮对模板是灰的，`add_to_library` 也会把模板的
  sid 换掉再存（等于「另存一份到我的」）；
- 删不掉 —— 它根本不在库里，没有可删的东西；
- 要改就用「载入」复制一份出来。和 `StrategyConfig.clone()` 一个道理：模板是
  模板，跑的是实例。

## 每条都带一句「演示什么」

`teach` 字段不是描述这条策略赚不赚钱（历史回测赚钱的模板最没用，它教人追
历史），而是说清**这条配置在示范哪个手法**：逐因子反向、换周期不动参数、
减因子、方向一翻变反转…… 界面上摆的就是这句话。

## 因子集几乎一样，这是故意的

`trend` / `revert` 两组用的基本是同一份因子集，只有方向、模型、个别因子不同。
一条一条看下去会发现：**改一个因子、翻一个方向，结果差得比换品种还远**。
这比给 34 条各不相同的策略有用 —— 变量控制住了，才看得出每个开关在干什么。

## 数值口径

配置里只写**与默认值不同的字段**（`_FCFG_DEFAULTS` / `_SCFG_DEFAULTS`），
其余在 `to_config()` 里补齐。这样每条模板一眼能看完，改也改得动；缺点是
默认值有两份（这里一份、`FactorConfig` / `StrategyConfig` 各一份），所以
冒烟测试里有一条专门比对它们是否还一致。
"""

from __future__ import annotations

# ------------------------------------------------------------------ 分组

TEMPLATE_PREFIX = "tpl_"

# (键, 标题, 这一组在教什么)。顺序就是界面上的顺序：从最简单的骨架往复杂的排。
GROUPS: list[tuple[str, str, str]] = [
    ("single", "单因子 · 动量",
     "一个因子、一条阈值线。这组是骨架：先确认因子本身有没有信息量，再谈组合。"
     "看的时候只盯两件事 —— 换了周期结论还稳不稳，方向交给 IC（auto）和写死"
     "方向差多少。"),
    ("rule", "规则策略（对照组）",
     "不用因子，纯规则。任何因子策略都该跟同品种、同周期的规则策略比一比，"
     "赢不了就别上。顺带演示仓位口径的两种写法：按资金分配、按止损金额反推手数。"),
    ("few", "少量因子 · 展期与期权",
     "两三个因子。这组演示两件事：逐因子反向（同一组里有的要反、有的不用反），"
     "以及因子少的时候线性模型的作用 —— 因子少，模型学不出花样，主要靠人选方向。"),
    ("trend", "多因子 · 趋势向",
     "十几个到二十几个因子 + 机器学习，方向写死 long_high（做多高分）。这是主配方，"
     "也是最该细看的一组：它们的因子集几乎是同一份，改的只有模型、方向和个别因子。"),
    ("revert", "多因子 · 反转向",
     "同一份因子集，方向翻成 long_low，并把一批趋势类因子逐个取反。配方不变、"
     "方向反了，结论完全不同 —— 这是因子组合里最容易被忽略的一层。"),
    ("daily", "日线 · 少因子反转",
     "日线上只用两三个因子。周期越长噪声越少，因子越少越不容易过拟合。"
     "前面几组都在加东西，这组专门演示减法。"),
    ("cross", "截面 · 多空对冲",
     "一次比一批品种的强弱，多强的、空弱的。和前面几组不是一个物种：信号来自"
     "品种之间的横向比较，不是单个品种的时间序列。"),
]

GROUP_LABEL = {k: label for k, label, _ in GROUPS}
GROUP_DESC = {k: desc for k, _, desc in GROUPS}
GROUP_ORDER = [k for k, _, _ in GROUPS]

# ------------------------------------------------------------------ 默认值

# `FactorConfig` 的默认值。**故意在这里抄了一份**：`ttrade.factor_strategy`
# 一 import 就拉 pandas/numpy，而这个模块会被界面的列表填充直接调用，不该为了
# 几个默认数字把 pandas 拖进启动路径。抄一份的代价是要防漂移 —— 冒烟测试里
# 「内置模板默认值」那一条就是干这个的，改了一边没改另一边会直接报错。
_FCFG_DEFAULTS = {
    "window": 252, "method": "z", "horizon": 5,
    "entry_z": 1.0, "exit_z": 0.3, "orient": "auto", "reverse": {},
    "model": "lightgbm", "train_window": 1000, "step": 180, "min_train": 500,
    "filter_mode": "entry", "long_th": 0.0, "short_th": 0.0,
}

# `StrategyConfig` 的默认值，同理。
_SCFG_DEFAULTS = {
    "sizing": "capital", "principal": 1_000_000.0, "capital_pct": 30.0,
    "risk_pct": 1.0, "stop_atr_n": 14, "stop_mult": 5.0, "volume": 1,
    "cost_profile": "", "range_years": 0.0,
    "cross_mode": "hedge", "cross_rebalance": 5, "cross_top_pct": 10.0,
    "cross_bot_pct": 10.0, "cross_weight": "risk",
}

# 下面这份数据里不出现的字段（key / name / group / teach）是模板自己的元信息，
# 不进 `StrategyConfig`。
_META_KEYS = ("key", "name", "group", "teach")

# ------------------------------------------------------------------ 数据


_RAW: list[dict] = [
    dict(
        key='tpl_ag_mom30m', name='白银·单因子动量 30m', group='single',
        teach=(
            "一个因子（60 根动量）+ 一条 z=1.0 阈值线，方向交给滚动 IC 自己判。最小可用骨架。"
        ),
        symbol="AG",
        exchange="SHFE",
        period="30m",
        family="factor",
        factor_mode="single",
        factor_keys=["mom_60"],
        capital_pct=50.0,
        risk_pct=2.0,
    ),
    dict(
        key='tpl_ag_ma30m', name='白银·均线交叉 30m', group='rule',
        teach=(
            "不用因子的对照组：快慢均线金叉死叉。看它在同样的 30m 上比因子版差在哪。"
        ),
        symbol="AG",
        exchange="SHFE",
        period="30m",
        family="rule",
        rule_key="ma_cross",
        rule_params={"fast": 24, "slow": 101},
        capital_pct=50.0,
        risk_pct=2.0,
    ),
    dict(
        key='tpl_ag_mom1h', name='白银·单因子动量 1h', group='single',
        teach=(
            "同上，周期从 30m 换成 1h，持有期 horizon 跟着从 5 根放到 11 根。"
        ),
        symbol="AG",
        exchange="SHFE",
        period="1h",
        family="factor",
        factor_mode="single",
        factor_keys=["mom_60"],
        factor_cfg={"horizon": 11},
        capital_pct=50.0,
        risk_pct=2.0,
    ),
    dict(
        key='tpl_ag_mom5m_long', name='白银·单因子动量 5m（只做多）', group='single',
        teach=(
            "5m 上把方向写死 long_high：只在高分做多、低分平仓，不开空。适合单边做多账户。"
        ),
        symbol="AG",
        exchange="SHFE",
        period="5m",
        family="factor",
        factor_mode="single",
        factor_keys=["mom_60"],
        factor_cfg={"horizon": 11, "orient": "long_high"},
        capital_pct=50.0,
        risk_pct=2.0,
        range_years=1.0,
    ),
    dict(
        key='tpl_lc_mom1h', name='碳酸锂·单因子动量 1h', group='single',
        teach=(
            "同一个因子换到波动更大的品种上，方向写死 long_high，看品种差异。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="1h",
        family="factor",
        factor_mode="single",
        factor_keys=["mom_60"],
        factor_cfg={"horizon": 11, "orient": "long_high"},
        capital_pct=50.0,
        risk_pct=2.0,
        range_years=3.0,
    ),
    dict(
        key='tpl_lc_mom5m', name='碳酸锂·单因子动量 5m', group='single',
        teach=(
            "再缩到 5m。三档周期（30m/1h/5m）并排跑一遍，就知道这个因子在哪个周期上更稳。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="5m",
        family="factor",
        factor_mode="single",
        factor_keys=["mom_60"],
        factor_cfg={"horizon": 11, "orient": "long_high"},
        capital_pct=50.0,
        risk_pct=2.0,
        range_years=3.0,
    ),
    dict(
        key='tpl_cu_roll1h', name='铜·展期三因子 1h', group='few',
        teach=(
            "少量因子的关键手法：3 个展期/期权因子，只把 near_oi_share 取反，其余按原方向。"
        ),
        symbol="CU",
        exchange="SHFE",
        period="1h",
        family="factor",
        factor_mode="multi",
        factor_keys=["near_oi_share", "pcr", "roll_main_oi"],
        factor_cfg={"orient": "long_high", "model": "linear", "reverse": {"near_oi_share": True}},
        sizing="fixed",
        capital_pct=50.0,
    ),
    dict(
        key='tpl_lc_macd1h', name='碳酸锂·MACD 波段 1h', group='rule',
        teach=(
            "纯规则。和上面那条因子版放一起，看「规则」与「因子」在同一个品种周期上的差别。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="1h",
        family="rule",
        rule_key="macd",
        rule_params={"fast": 12, "slow": 26, "signal_n": 9},
        capital_pct=50.0,
    ),
    dict(
        key='tpl_lc_sbbox5m', name='碳酸锂·sbbox 波段 5m', group='rule',
        teach=(
            "规则策略里唯一按「止损金额」定仓的一条：手数由 stop_mult × ATR 反推，"
            "不是拍手数。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="5m",
        family="rule",
        rule_key="sbbox_mini",
        rule_params={"entry": 182, "exit_n": 747, "vol_ratio": 0.4, "range_n": 1, "atr_n": 60, "stop_mult": 5.0},
        sizing="risk",
        capital_pct=10.0,
        stop_atr_n=200,
        volume=10,
    ),
    dict(
        key='tpl_cu_macd1d', name='铜·MACD 日线', group='rule',
        teach=(
            "日线慢周期基准。任何 5m 上的花活都该跟它比一比，跑不过就别上。"
        ),
        symbol="CU",
        exchange="SHFE",
        period="1d",
        family="rule",
        rule_key="macd",
        rule_params={"fast": 12, "slow": 26, "signal_n": 9},
    ),
    dict(
        key='tpl_im_multi5m', name='IM·多因子日内（17 因子）', group='trend',
        teach=(
            "17 个因子 + LightGBM，方向写死 long_high。多因子的标准配方，也是后面几条的起点。"
        ),
        symbol="IM",
        exchange="CFFEX",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"orient": "long_high", "reverse": {"season_rank": True, "season_avg": True, "quiet_window": True, "season_ir": True}},
        range_years=5.0,
    ),
    dict(
        key='tpl_im_multi5m_exit0', name='IM·多因子日内（不止盈）', group='trend',
        teach=(
            "和上一条只差一个 exit_z：0.3 → 0.0，也就是「不设离场阈值、只等信号反向」。"
        ),
        symbol="IM",
        exchange="CFFEX",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "reverse": {"season_rank": True, "season_avg": True, "quiet_window": True, "season_ir": True}},
        range_years=5.0,
    ),
    dict(
        key='tpl_ag_multi_gpr', name='白银·多因子日内（GPR）', group='trend',
        teach=(
            "因子集多加了 iv_term / pcr 两个期权维度，模型换成高斯过程，方向 auto。"
        ),
        symbol="AG",
        exchange="SHFE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "iv_term", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "model": "gpr", "reverse": {"season_rank": True, "season_avg": True, "quiet_window": True, "season_ir": True}},
        range_years=5.0,
    ),
    dict(
        key='tpl_ap_multi', name='苹果·多因子日内（XGBoost）', group='trend',
        teach=(
            "同配方换 XGBoost + 方向 auto，品种换成持仓量小的苹果：小品种上模型更容易过拟合。"
        ),
        symbol="AP",
        exchange="CZCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "iv_term", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"exit_z": 0.0, "model": "xgboost", "reverse": {"season_rank": True, "season_avg": True, "quiet_window": True, "season_ir": True}},
    ),
    dict(
        key='tpl_lh_multi', name='生猪·多因子日内（加 mom_20）', group='trend',
        teach=(
            "在标准 19 因子上再加一个短动量 mom_20，演示「加一个因子」的写法。"
        ),
        symbol="LH",
        exchange="DCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_20", "roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "iv_term", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"exit_z": 0.0, "model": "xgboost", "reverse": {"season_rank": True, "season_avg": True, "quiet_window": True, "season_ir": True}},
    ),
    dict(
        key='tpl_lc_multi', name='碳酸锂·多因子日内', group='trend',
        teach=(
            "标准 19 因子 + XGBoost + long_high。上面几条的「母版」。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "iv_term", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "model": "xgboost", "reverse": {"season_rank": True, "season_avg": True, "quiet_window": True, "season_ir": True}},
    ),
    dict(
        key='tpl_lc_multi2', name='碳酸锂·多因子日内②（精简 15 因子）', group='trend',
        teach=(
            "砍掉 4 个季节性因子，补进 roll_curv_o3。演示「减因子 + 换因子」的取舍。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_curv_o3", "roll_main_oi", "iv_term", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "model": "xgboost", "reverse": {"roll_curv_o3": True, "quiet_window": True}},
    ),
    dict(
        key='tpl_lc_multi3', name='碳酸锂·多因子日内③（再反一个）', group='trend',
        teach=(
            "再把 roll_main_oi 也取反。反向开关是逐因子的，改一个不影响其他。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_curv_o3", "roll_main_oi", "iv_term", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "model": "xgboost", "reverse": {"roll_main_oi": True, "roll_curv_o3": True, "quiet_window": True}},
    ),
    dict(
        key='tpl_lc_multi4', name='碳酸锂·多因子日内④（加 iv_skew）', group='trend',
        teach=(
            "补进 iv_skew（偏度）并取反：波动率偏斜对碳酸锂是反向指标。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_curv_o3", "roll_main_oi", "iv_term", "iv_skew", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "model": "xgboost", "reverse": {"roll_main_oi": True, "roll_curv_o3": True, "quiet_window": True, "iv_skew": True}},
    ),
    dict(
        key='tpl_ec_multi', name='集运欧线·多因子日内', group='revert',
        teach=(
            "方向整体翻成 long_low（做多低分）。同一套因子集，方向一翻就是反转策略。"
        ),
        symbol="EC",
        exchange="INE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_curv_o3", "roll_main_oi", "iv_term", "iv_skew", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour"],
        factor_cfg={"exit_z": 0.0, "orient": "long_low", "model": "xgboost", "reverse": {"roll_main_oi": True, "roll_curv_o3": True, "quiet_window": True, "iv_skew": True}},
    ),
    dict(
        key='tpl_ec_multi2', name='集运欧线·多因子日内②（加季节性）', group='revert',
        teach=(
            "在反转版上补 season_avg / season_ir，季节性在集运这种强周期品种上权重更高。"
        ),
        symbol="EC",
        exchange="INE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_curv_o3", "roll_main_oi", "iv_term", "iv_skew", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.0, "orient": "long_low", "model": "xgboost", "reverse": {"season_avg": True, "roll_curv_o3": True, "season_ir": True, "roll_main_oi": True, "quiet_window": True, "iv_skew": True}},
    ),
    dict(
        key='tpl_cf_multi', name='棉花·多因子日内（反转）', group='revert',
        teach=(
            "exit_z 收到 0.1：离场更灵敏、持仓更短。日内反转常用的一档。"
        ),
        symbol="CF",
        exchange="CZCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_60", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_main_oi", "iv_term", "iv_skew", "pcr", "vwap_gap", "pv_pos", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"intraday_mom": True, "roll_curv_o3": True, "season_avg": True, "season_ir": True, "roll_intraday": True, "roll_main_oi": True, "quiet_window": True, "iv_skew": True}},
    ),
    dict(
        key='tpl_jm_multi', name='焦煤·多因子日内（反转为主）', group='revert',
        teach=(
            "补进 mom_20/mom_60/ma_slope_20 三个趋势因子后整体反向 —— 趋势因子反着用。"
        ),
        symbol="JM",
        exchange="DCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_20", "mom_60", "ma_slope_20", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_main_oi", "iv_term", "iv_skew", "pcr", "vwap_gap", "pv_pos", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"intraday_mom": True, "roll_curv_o3": True, "season_avg": True, "season_ir": True, "roll_intraday": True, "roll_main_oi": True, "quiet_window": True, "iv_skew": True}},
    ),
    dict(
        key='tpl_bu_multi', name='沥青·多因子日内（反转为主）', group='revert',
        teach=(
            "在反转配方上再加 iv_rv / range_break，并把 pv_pos、range_break 一起取反。"
        ),
        symbol="BU",
        exchange="SHFE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_60", "ma_gap_20", "ma_slope_20", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_main_oi", "iv_term", "iv_skew", "iv_rv", "pcr", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"intraday_mom": True, "roll_curv_o3": True, "season_avg": True, "season_ir": True, "roll_intraday": True, "range_break": True, "roll_main_oi": True, "quiet_window": True, "pv_pos": True, "iv_skew": True}},
    ),
    dict(
        key='tpl_pb_rev_d', name='铅·日线反转三因子', group='daily',
        teach=(
            "日线上只用 3 个因子：rev_5（5 日反转）+ roll_intraday（取反）+ pcr。"
            "慢周期少因子。"
        ),
        symbol="PB",
        exchange="SHFE",
        period="1d_tq",
        family="factor",
        factor_mode="multi",
        factor_keys=["rev_5", "roll_intraday", "pcr"],
        factor_cfg={"orient": "long_high", "model": "xgboost", "reverse": {"roll_intraday": True}},
    ),
    dict(
        key='tpl_zn_rev_d', name='锌·日线反转双因子', group='daily',
        teach=(
            "再砍到 2 个因子。日线上因子越少越不容易过拟合，这条是极端简版。"
        ),
        symbol="ZN",
        exchange="SHFE",
        period="1d_tq",
        family="factor",
        factor_mode="multi",
        factor_keys=["rev_5", "roll_intraday"],
        factor_cfg={"orient": "long_high", "model": "xgboost", "reverse": {"roll_intraday": True}},
    ),
    dict(
        key='tpl_sp_multi', name='纸浆·多因子日内（反转为主）', group='revert',
        teach=(
            "同一套反转配方换品种，验证「配方可移植」而不是「只在一个品种上凑出来的」。"
        ),
        symbol="SP",
        exchange="SHFE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_20", "mom_60", "ma_slope_20", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_main_oi", "iv_term", "iv_skew", "pcr", "vwap_gap", "pv_pos", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"quiet_window": True, "roll_main_oi": True, "intraday_mom": True, "roll_curv_o3": True, "season_ir": True, "iv_skew": True, "season_avg": True, "roll_intraday": True}},
    ),
    dict(
        key='tpl_b_multi', name='豆二·多因子日内（反转为主）', group='revert',
        teach=(
            "补进 mom_250 长动量、iv_atm / iv_rv，农产品上长周期动量反转更明显。"
        ),
        symbol="B",
        exchange="DCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_60", "mom_250", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_main_oi", "iv_atm", "iv_term", "iv_skew", "iv_rv", "pcr", "vwap_gap", "pv_pos", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"quiet_window": True, "roll_main_oi": True, "intraday_mom": True, "roll_curv_o3": True, "season_ir": True, "iv_skew": True, "roll_intraday": True}},
    ),
    dict(
        key='tpl_cs_multi', name='淀粉·多因子日内（反转为主）', group='revert',
        teach=(
            "24 个因子，是这批里最宽的一条：展期曲线加了 o6 一档（远月）。"
        ),
        symbol="CS",
        exchange="DCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_60", "mom_250", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_yield_o6", "roll_curv_o6", "roll_change_o6", "roll_main_oi", "iv_atm", "iv_term", "iv_skew", "iv_rv", "pcr", "vwap_gap", "pv_pos", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"quiet_window": True, "roll_main_oi": True, "intraday_mom": True, "roll_curv_o3": True, "season_ir": True, "iv_skew": True, "roll_intraday": True}},
    ),
    dict(
        key='tpl_p_multi', name='棕榈油·多因子日内（反转为主）', group='revert',
        teach=(
            "油脂三兄弟之一。和豆油、豆二并排看，配方相同、结论不同，说明品种比参数重要。"
        ),
        symbol="P",
        exchange="DCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_60", "mom_250", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_main_oi", "iv_atm", "iv_term", "iv_skew", "iv_rv", "pcr", "vwap_gap", "pv_pos", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"quiet_window": True, "roll_main_oi": True, "intraday_mom": True, "roll_curv_o3": True, "season_ir": True, "iv_skew": True, "roll_intraday": True}},
    ),
    dict(
        key='tpl_y_multi', name='豆油·多因子日内（反转为主）', group='revert',
        teach=(
            "同配方换 mom_120 中动量。三个油脂品种的对比组。"
        ),
        symbol="Y",
        exchange="DCE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["mom_60", "mom_120", "near_oi_share", "roll_intraday", "roll_curv_o3", "roll_main_oi", "iv_atm", "iv_term", "iv_skew", "iv_rv", "pcr", "vwap_gap", "pv_pos", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir"],
        factor_cfg={"exit_z": 0.1, "orient": "long_low", "model": "xgboost", "reverse": {"quiet_window": True, "roll_main_oi": True, "intraday_mom": True, "roll_curv_o3": True, "season_ir": True, "iv_skew": True, "roll_intraday": True}},
    ),
    dict(
        key='tpl_cu_multi5m', name='铜·多因子短周期 5m', group='trend',
        teach=(
            "铜的趋势版：17 因子 + LightGBM + long_high，周期 5m。"
        ),
        symbol="CU",
        exchange="SHFE",
        period="5m",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "reverse": {"quiet_window": True, "season_rank": True, "season_avg": True, "season_ir": True}},
        range_years=5.0,
    ),
    dict(
        key='tpl_cu_multi1h', name='铜·多因子中周期 1h', group='trend',
        teach=(
            "同一条策略换 1h。除了周期，参数一个都没改 —— 这是「换周期」的正确做法。"
        ),
        symbol="CU",
        exchange="SHFE",
        period="1h",
        family="factor",
        factor_mode="multi",
        factor_keys=["roll_curv", "near_oi_share", "roll_yield_o3", "roll_yield_o6", "roll_main_oi", "vwap_gap", "pv_pos", "range_break", "intraday_mom", "overnight_gap", "trend_window", "quiet_window", "first_hour", "season_avg", "season_ir", "season_rank", "season_gap"],
        factor_cfg={"exit_z": 0.0, "orient": "long_high", "reverse": {"quiet_window": True, "season_rank": True, "season_avg": True, "season_ir": True}},
    ),
    dict(
        key='tpl_cross_hedge', name='截面·多空对冲（70 品种）', group='cross',
        teach=(
            "完全不同的玩法：一次比 70 个品种的强弱，多最强的 10%、空最弱的 10%。"
        ),
        symbol="LC",
        exchange="GFEX",
        period="1d_tq",
        family="cross",
        factor_mode="ml",
        factor_keys=["mom_20", "mom_60", "mom_120", "vol_20", "vol_surge", "pv_corr", "oi_change", "hl_range", "close_pos", "roll_curv", "roll_change", "near_oi_share", "roll_curv_o3", "roll_yield_o6", "roll_main_oi", "pcr", "season_avg", "season_rank", "season_gap"],
        factor_cfg={"orient": "long_high"},
        cross_symbols=["AD_SHFE", "AG_SHFE", "AL_SHFE", "AO_SHFE", "AU_SHFE", "BR_SHFE", "BU_SHFE", "CU_SHFE", "FU_SHFE", "HC_SHFE", "NI_SHFE", "OP_SHFE", "PB_SHFE", "RB_SHFE", "RU_SHFE", "SN_SHFE", "SP_SHFE", "SS_SHFE", "ZN_SHFE", "A_DCE", "B_DCE", "BZ_DCE", "C_DCE", "CS_DCE", "EB_DCE", "EG_DCE", "I_DCE", "J_DCE", "JD_DCE", "JM_DCE", "L_DCE", "LG_DCE", "LH_DCE", "M_DCE", "P_DCE", "PG_DCE", "PP_DCE", "RR_DCE", "V_DCE", "Y_DCE", "AP_CZCE", "CF_CZCE", "CJ_CZCE", "CY_CZCE", "FG_CZCE", "MA_CZCE", "OI_CZCE", "PF_CZCE", "PK_CZCE", "PL_CZCE", "PR_CZCE", "PX_CZCE", "RM_CZCE", "SA_CZCE", "SF_CZCE", "SH_CZCE", "SM_CZCE", "SR_CZCE", "TA_CZCE", "UR_CZCE", "BC_INE", "EC_INE", "LU_INE", "NR_INE", "SC_INE", "LC_GFEX", "PD_GFEX", "PS_GFEX", "PT_GFEX", "SI_GFEX"],
        principal=10000000.0,
        capital_pct=50.0,
    ),
]

BY_KEY: dict[str, dict] = {t["key"]: t for t in _RAW}

_TEACH = {t["key"]: t["teach"] for t in _RAW}

_ORIENT_CN = {"auto": "方向自适应", "long_high": "做多高分", "long_low": "做多低分"}


# ------------------------------------------------------------------ 查询


def is_template(sid: str) -> bool:
    """这个 id 是不是内置模板。用户配置的 sid 是 12 位 uuid 十六进制，撞不上。"""
    return str(sid or "").startswith(TEMPLATE_PREFIX)


def keys(group: str = "") -> list[str]:
    """模板键，按分组顺序。`group` 给了就只返回那一组。"""
    if group:
        return [t["key"] for t in _RAW if t["group"] == group]
    return [t["key"] for t in _RAW]


def group_of(key: str) -> str:
    t = BY_KEY.get(str(key or ""))
    return t["group"] if t else ""


def teach(key: str) -> str:
    """这条模板在演示什么。取不到返回空串，界面上不显示这一段。"""
    return _TEACH.get(str(key or ""), "")


def _n_factors(t: dict) -> int:
    return len(t.get("factor_keys") or [])


def summary(key: str) -> str:
    """一行配方摘要，给列表 tooltip 和 Agent 用。

    不写收益、不写胜率：模板是用来学组合的，不是收益排行榜。
    """
    t = BY_KEY.get(str(key or ""))
    if t is None:
        return ""
    fam = t.get("family") or "factor"
    if fam == "rule":
        return f"规则 · {t.get('rule_key') or '—'}"
    cfg = t.get("factor_cfg") or {}
    n = _n_factors(t)
    rev = len([k for k, v in (cfg.get("reverse") or {}).items() if v])
    # 「多因子 15 个因子」这种重话别写：multi 就是默认档，省掉前缀更短；
    # 单因子连数量都不用说，因子键在 tooltip 里能看到
    mode = {"single": "单因子", "multi": "", "ml": "机器学习"}.get(
        t.get("factor_mode") or "", "")
    bits = [mode] if mode == "单因子" else [(mode + f" {n} 个因子").strip()]
    if fam == "cross":
        bits.append("截面多空")
    else:
        model = cfg.get("model")
        if model and t.get("factor_mode") == "ml":
            bits.append(str(model))
    bits.append(_ORIENT_CN.get(cfg.get("orient") or "auto", "方向自适应"))
    if rev:
        bits.append(f"{rev} 个取反")
    return " · ".join(bits)


def to_config(key: str, **overrides):
    """把模板还原成一份完整的 `StrategyConfig`。

    **每次调用都新建**，sid 就是模板键本身（`tpl_xxx`）。上层拿它去渲染、
    去载入、去加进组合都没问题 —— `add_to_portfolio` 本来就会 clone 一份新 sid，
    而 `add_to_library` 见到模板 sid 会换号（等于「另存到我的」）。
    """
    t = BY_KEY.get(str(key or ""))
    if t is None:
        return None
    from ttrade.strategy_store import StrategyConfig

    d = {k: v for k, v in t.items() if k not in _META_KEYS}
    d["sid"] = t["key"]
    d["name"] = t["name"]
    # 默认值**必须显式写进去**，不能靠「字典里没有」让 `from_dict` 去补：
    # 那个函数把「缺 capital_pct / risk_pct」当成**老版本配置**，会按
    # `capital` / `risk_amount` 两个绝对金额字段反推，取不到就退回 50% / 2%。
    # 于是一条 capital_pct=30 的模板载入后会变成 50%，仓位凭空多六成。
    for _k, _v in _SCFG_DEFAULTS.items():
        d.setdefault(_k, _v)

    # factor_cfg 补全默认值：`_apply_factor_cfg` 只覆盖「字典里有」的字段，
    # 稀疏字典会让界面留着上一条策略的数值。补全之后载入行为才可预期。
    # 规则策略不用因子，补一份空的进去只会让 `to_dict()` 多出一堆没意义的键。
    if (t.get("family") or "factor") != "rule":
        fcfg = dict(_FCFG_DEFAULTS)
        fcfg.update(t.get("factor_cfg") or {})
        rev = {k: True for k, v in (fcfg.get("reverse") or {}).items() if v}
        fcfg["reverse"] = rev
        d["factor_cfg"] = fcfg
        # 界面上逐因子反向的勾选框读的是 factor_reverse，和 factor_cfg["reverse"]
        # 是同一件事的两个出口。模板里只存一份，这里同步出去。
        if rev and "factor_reverse" not in d:
            d["factor_reverse"] = dict(rev)

    d.update(overrides)
    cfg = StrategyConfig.from_dict(d)
    cfg.created = ""          # 模板没有「创建时间」，别让界面显示成今天
    cfg.note = ""
    cfg.enabled = False
    return cfg


def all_templates(group: str = "") -> list:
    """全部模板（或某一组），顺序即分组顺序。"""
    out = []
    for k in keys(group):
        cfg = to_config(k)
        if cfg is not None:
            out.append(cfg)
    return out


def groups_with_items() -> list[tuple[str, str, list[str]]]:
    """[(组键, 组标题, 模板键列表)]，空组自动跳过 —— 给界面建分组列表用。"""
    out = []
    for k, label, _desc in GROUPS:
        ks = keys(k)
        if ks:
            out.append((k, label, ks))
    return out


def knobs(cfg) -> str:
    """一条配置的关键旋钮，一行。

    和 `StrategyConfig.params_text()` 的区别：那个是给表格列用的短摘要，只说
    窗口和阈值；模板要展示的是**这条跟别人差在哪**，所以入场/离场/持有期/模型/
    方向都要摆出来。方向只在不是 `auto` 时才写 —— 默认那条占地方又没信息。
    """
    fam = getattr(cfg, "family", "") or "factor"
    if fam == "rule":
        return "　".join(f"{k}={v}" for k, v in (cfg.rule_params or {}).items()) or "默认参数"
    if fam == "cross":
        return cfg.params_text()
    f = cfg.factor_cfg or {}
    bits = [
        f"窗口 {f.get('window', 252)}",
        f"入场 z {f.get('entry_z', 1.0)}",
        f"离场 z {f.get('exit_z', 0.3)}",
        f"持有 {f.get('horizon', 5)} 根",
    ]
    model = f.get("model")
    if model and model != _FCFG_DEFAULTS["model"]:
        bits.append(f"模型 {model}")
    orient = f.get("orient") or "auto"
    if orient != "auto":
        bits.append(_ORIENT_CN.get(orient, orient))
    return "　".join(bits)
