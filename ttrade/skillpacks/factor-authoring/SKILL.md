---
name: factor-authoring
description: 把主观想法写成能跑、能回测的因子代码
when_to_use: 用户描述一个平台没有的因子，或者现有因子组合表达不了他的想法
group: factor
risk: write
allowed-tools: doc, factor_list, factor_info, factor_check, factor
version: 1
---

# 写一个新因子

## 先做减法

1. **先在因子库里找**。120 多个内置因子覆盖趋势、反转、波动、量能、形态、日内、
   时段、展期（含 OI3/OI6）、期权、季节性、品种季节、基本面、板块行业、宏观
   十六类。用户说「我要抓放量上涨」，
   可能是 `vol_price_corr` 加上 `mom_20` 的组合，不需要新因子。
2. **能用多因子模式表达的，不要新增因子**。比如「动量除以波动」——
   这是两列相除，属于策略配置层面的事，写成因子只会让库里多一条几乎重复的。
   库里多一个近重复因子，比少一个更难维护。
3. 真要新增，再往下走。

## 必查手册

写代码之前查 `doc`，**不要凭印象猜列名**：

```json
{"action": "doc", "topic": "data"}
```
行情链路、每条链路各自的列名、复权口径、缺数据长什么样。

```json
{"action": "doc", "topic": "primitives"}
```
`_ret` / `_col` / `_oi` / `_safe_div` / `standardize` 的真实源码。

```json
{"action": "doc", "topic": "factor:mom_20"}
```
照抄一个能跑通的写法，比从零写快得多。

## 动作块

```json
{"action": "factor", "key": "vol_price_corr", "label": "量价相关·20 日",
 "category": "量能", "hint_dir": 0,
 "formula": "corr(Δlog(close), Δlog(volume), 20)",
 "desc": "成交量和价格同向变化，说明这段趋势有资金跟随",
 "func": "f_vol_price_corr",
 "code": "def f_vol_price_corr(df):\n    c = _col(df, 'close').pct_change()\n    v = _col(df, 'volume').pct_change()\n    return c.rolling(20).corr(v)"}
```

## 六条硬约束（违反任何一条都注册失败）

1. **签名固定** `def f_xxx(df) -> pd.Series`。`df` 索引是 datetime，一定有
   `open/high/low/close/volume`。
   持仓量**不要直接写列名** —— 日线 CSV 叫 `open_interest`，天勤分钟线叫
   `open_oi`，写死一个在另一种数据上就是全 NaN。用 `_oi(df)`，两个名字都认。
2. **不要写 import**。`np`、`pd` 已注入。`_col(df, 列名)` 取衍生列（没有该列时
   返回全 NaN，不要自己判空）、`_ret(c, n)`、`_safe_div(a, b)` 也可用。
3. **返回等长 Series，索引和 df 完全一致**。不要 `dropna()`、不要
   `reset_index()` —— 预热期的 NaN 要留着，下游统一处理。
4. **必须 PIT（只用过去的信息）**：只用 `rolling` / `ewm` / `shift(正数)`，
   **绝不出现 `shift(-n)`**。用未来数据的因子在回测里好得离谱，实盘一分钱
   赚不到 —— 这是量化里最常见、也最贵的错误。
5. **只算原始值**。标准化（滚动 z-score）平台统一做，不要自己减均值除标准差。
6. `key` 用英文小写加下划线，`label` 用中文；`hint_dir` 是历史倾向
   （+1 高分偏涨、-1 偏跌、0 无明显方向），**只是展示提示**，
   回测里方向仍由滚动 IC 决定。

## 注册失败怎么改

平台会把报错原样发给你，那是最准的线索：

- 说「长度不一致」→ 去 `doc` 的 `contract` 看返回值约定
- 说「没有名为 xxx 的函数」→ `func` 和代码里的函数名对不上
- 说「整列 NaN」→ 列名写错了，去 `data` 看三条链路各自的列名

**不要凭猜测连着改三次同一个地方。** 连着两次失败就先查手册。
