"""平台手册：数据怎么来的、因子怎么写、每个因子的完整实现。

「ChanAgent」要能自己写因子，就必须先知道两件事：**行情长什么样**（哪些列、
怎么复权、缺数据是什么形态）和**平台给了哪些原语**（`_ret` / `_col` 到底是
什么）。原来这两件事只写在提示词的几行摘要里，模型看不到实现，只能猜 ——
写出来的因子常常读一个不存在的列，或者把 `_ret(c, 20)` 当成「20 日收益率」
的黑箱，回头在分钟线上踩 `open_interest` 那个坑。

## 为什么做成模块，而不是塞进提示词

67 个因子加上原语和数据说明接近五万字，每轮对话都发一遍等于把上下文预算烧掉
一半，而且九成的对话根本用不上。所以手册是**按需取**：Agent 用
`{"action": "doc", "topic": "..."}` 要哪段给哪段，界面里也能整份预览/保存。

## 为什么源码要现取

跟 `library_docs` 一个原则：文档和实现各写一份，早晚对不上。这里的代码一律
`inspect.getsource()` 从函数对象上取，改了窗口、换了实现，手册自动跟着变。
公式和「怎么读」仍然手写在 `library_docs.FACTOR_DOCS` 里 —— 数学式子没法从
代码反推。
"""

from __future__ import annotations

import inspect
import textwrap

from ttrade import library_docs as L
from ttrade.factors import FACTORS

# 因子代码里会调用、但模型看不见定义的那些原语。手册里必须把源码摊开，
# 否则 `return _ret(df["close"], 20)` 等于什么都没说。
#
# 这份清单要和 `userfactors._namespace()` 完全一致 —— 手册是模型唯一的教材，
# 写着那里没有的名字，模型照抄就是 NameError。
_FACTOR_PRIMITIVES = (
    "_ret", "_safe_div", "_col", "_oi", "_atr", "_safe_log_ratio", "_nan_series",
    "_roll_yield", "_roll_annual", "_roll_curv", "_roll_change",
)
_STRATEGY_PRIMITIVES = ("sma", "ema", "true_range", "atr", "_position_state")


# ------------------------------------------------------------------ 小工具


def _source(func) -> str:
    """取函数源码。取不到返回空串 —— 打包后可能没有 .py 文件。"""
    try:
        return textwrap.dedent(inspect.getsource(func)).rstrip()
    except (OSError, TypeError):
        return ""


def _prims_used(code: str, names: tuple[str, ...]) -> list[str]:
    """这段代码用到了哪几个原语。按出现顺序，不重复。"""
    out = []
    for n in names:
        # 只认 `_ret(` 这种调用，避免把 `_ret_foo` 也算进来
        if f"{n}(" in code and n not in out:
            out.append(n)
    return out


def _code_block(code: str, lang: str = "python") -> str:
    return f"```{lang}\n{code.strip()}\n```"


# ------------------------------------------------------------------ 数据段


DATA_GUIDE = """## 一、行情数据是怎么来的

平台只有三条数据链路，每条负责一段，**不要以为随便哪个品种都有全部分钟线**。

| 市场 | 周期 | 来源 | 落盘位置 | 复权 |
|---|---|---|---|---|
| 美股 | 日线 | 东方财富 `push2his`（兜底 Tushare `us_daily`） | `data/*.csv` | 东财**后复权**，整段同口径 |
| 国内期货 | 日线 | Tushare `fut_daily` + `fut_daily_adj` | `data_futures/*.csv` | 换月复权，因子由两者反推 |
| 国内期货 | 5m / 15m / 30m / 1h / 2h / 4h | 天勤 TqSdk `klines`（主连） | `data_tq`（DuckDB） | 落盘 hfq，读取默认 qfq |

- 15/30 分钟是 **5 分钟合并**出来的，2/4 小时是 **1 小时合并**出来的，不是另下一份
- 美股没有分钟线；期货没有「不复权」的分钟线可选（`adjust="none"` 是调试用的）
- 期货分钟线只覆盖 78 个活跃品种，十年，最早约 2016 年

### DataFrame 的形状

- `index`：`DatetimeIndex`，升序、无重复、无时区
- 日线的 index 是**当日 00:00**；分钟线的 index 是 **bar 结束时刻**（21:00 那根是
  20:00–21:00 夜盘），和天勤原生口径一致
- 行数：日线一个品种 20 年约 4900 根；1h 十年约 2 万根；5m 十年约 12 万根

### 列名（这里有坑，务必按来源区分）

所有链路都有的：

| 列 | 含义 |
|---|---|
| `open` / `high` / `low` / `close` | 已复权价格 |
| `volume` | 成交量（手） |

期货日线（Tushare CSV）额外有：

| 列 | 含义 |
|---|---|
| `amount` | 成交额（元）。**分钟线没有这一列** |
| `open_interest` | 持仓量。**日线叫这个名字** |
| `adj_factor` | 累计复权因子（绝对口径，锚定上市日） |
| `ts_code` / `name` | Tushare 合约代码、品种名 |

期货分钟线（天勤）额外有：

| 列 | 含义 |
|---|---|
| `open_oi` / `close_oi` | 开盘时 / 收盘时的持仓量。**分钟线叫这个名字** |
| `adj_factor` | 同上，但落盘是 hfq，读取时按 `adjust` 再缩放 |
| `trading_day` | 归属交易日（夜盘算次日），做日频聚合时用它而不是 `index.date` |

> **`open_interest` 与 `open_oi` 是同一个东西的两个名字**，是两条数据源的历史
> 遗留。因子代码里不要直接写 `df["open_interest"]` —— 分钟线上会全 NaN，而且
> 不会报错，表现为「这个因子在 1h 上算不出来」，看起来像天勤不给持仓量。
> 用 `_oi(df)`，它两个名字都认。

衍生列（只有**期货日线**有，由 `data_futures_aux` 并进来，`_col(df, 名字)` 取）：

| 前缀 | 含义 |
|---|---|
| `c1_` / `c2_` / `c3_` | 按到期日取的近月 / 次近 / 第三近合约的 `close` / `dte`（剩余天数）/ `oi` |
| `o3_1_` / `o3_2_` / `o3_3_` | 先从**持仓量前 3** 里挑，再按到期日取三腿 |
| `o6_1_` / `o6_2_` / `o6_3_` | 同上，持仓量前 6 |
| `iv_near` / `iv_next` / `dte_near` / `dte_next` | 近月 / 次月期权平值隐含波动率与剩余天数 |
| `skew` / `curv` / `pcr` / `n_opt` | 微笑偏度、曲率、看跌看涨持仓比、当日有效期权数 |

美股没有衍生列，没下过衍生数据的期货品种也没有 —— 这时候 `_col()` 返回全 NaN，
`filter_available()` 会把这些因子从该标的的列表里剔掉，不会算成 0。

### 复权

- 期货日线：`adj_factor` 是**绝对累计因子**（锚定上市日、与查询窗口无关），
  `close × adj_factor` 得到后复权价。增量补一小段也能直接拼，不会错位
- 期货分钟线：落盘就是 hfq（历史固定不变），读取时 `adjust="qfq"` 会把整段除以
  最后一根因子 —— **最后一根等于真实盘面价**，和日线口径对齐。这也是为什么
  同一个品种的日线和 1h 能放在一张图上看
- 不复权的序列在换月处有跳空，`mom_20` 这类因子会被跳空污染。看到某个品种动量
  因子在换月日附近异常，先怀疑复权，不要怀疑因子

### 缺数据就是 NaN

- 预热期（窗口不够）是 NaN，这是**正常**的，下游统一 `dropna`
- 没有某一列（美股没有持仓量、分钟线没有成交额）→ 对应因子全 NaN
- 全 NaN 的因子会被 `available()` 判为不可用，从下拉框里剔掉

**不要为了让因子「有值」去 `fillna(0)`**：0 在 z-score 里是一个具体的观测，
会把分布拉偏，而且这种污染在回测里看不出来。

### 想改取数逻辑的话，入口在哪

（这一节是给「升级权限」下要动代码的场合用的，只读因子不用管。）

| 要改什么 | 文件 | 函数 |
|---|---|---|
| 美股日线抓取 | `ttrade/eastmoney.py` | `fetch_daily()`（东财 K 线）、`download_universe()` |
| 美股本地读取 | `ttrade/eastmoney.py` | `load_local()`（含质量修复） |
| 期货日线抓取 | `ttrade/futures.py` | `fetch_daily()`（Tushare `fut_daily` + `fut_daily_adj`） |
| 期货日线本地读取 | `ttrade/futures.py` | `load_local()`（读 CSV、乘复权因子） |
| 期货分钟线抓取 | `ttrade/tqfetch.py` | `write_main_many()`（主连）、`write_contracts()`（真实合约） |
| 期货分钟线本地读取 | `ttrade/tqdata.py` | `load_main()`（DuckDB + 复权 + 合并） |
| 主力切换表 / 复权因子 | `ttrade/tqfetch.py` | `ensure_switch()`、`refresh_all()` |
| 衍生列并进 DataFrame | `ttrade/auxdata.py` | `attach()` |

改完行情口径要同步检查两件事：`adj_factor` 是否还对得上，以及 `smoke_test_tq`
里的「1h 因子与切换表逐根一致」那一项（它专门守复权）。
"""


# ------------------------------------------------------------------ 原语段


def primitive_section() -> str:
    """平台注入的原语源码。现取，不手抄。"""
    from ttrade import factors as F
    from ttrade import strategies as S

    blocks = [
        "## 二、平台注入的原语",
        "",
        "写因子/策略时**不要 import**，下面这些已经在命名空间里了。它们是平台和",
        "数据之间的那层适配，直接读源码比猜语义可靠。",
        "",
        "### 因子里能用的",
        "",
    ]
    for name in _FACTOR_PRIMITIVES:
        fn = getattr(F, name, None)
        if fn is None:
            continue
        src = _source(fn)
        if not src:
            continue
        blocks += [_code_block(src), ""]

    blocks += [
        "### 策略里额外能用的",
        "",
    ]
    for name in _STRATEGY_PRIMITIVES:
        fn = getattr(S, name, None)
        if fn is None:
            continue
        src = _source(fn)
        if not src:
            continue
        blocks += [_code_block(src), ""]

    std = _source(F.standardize)
    if std:
        blocks += [
            "### 标准化（平台统一做，因子不要自己减均值除标准差）",
            "",
            _code_block(std),
            "",
        ]
    return "\n".join(blocks).rstrip()


# ------------------------------------------------------------------ 因子段


def factor_section(key: str, with_header: bool = True) -> str:
    """单个因子的完整条目：公式 + 怎么读 + 源码 + 用到的原语。"""
    d = L.factor_doc(key)
    if d is None:
        return f"（没有这个因子：{key}）"
    spec = FACTORS.get(key)
    prims = _prims_used(d.code, _FACTOR_PRIMITIVES)
    lines = []
    if with_header:
        lines.append(f"### {d.key} · {d.label}")
        lines.append("")
    lines += [
        f"- 类别：{d.category}　历史倾向：{d.direction}（只是统计提示，方向由滚动 IC 定）",
        f"- 一句话：{d.desc}",
        f"- 公式：`{d.formula}`",
        f"- 怎么读：{d.explain}",
    ]
    if prims:
        lines.append(f"- 用到的原语：{'、'.join(f'`{p}`' for p in prims)}"
                     "（源码见「平台注入的原语」）")
    if spec is not None and not d.code.strip():
        lines.append("- 实现：（打包后取不到源码）")
    else:
        lines += ["- 实现：", "", _code_block(d.code)]
    return "\n".join(lines).rstrip()


def factor_index() -> str:
    """全部因子的索引，一行一个。先看这个再决定读哪条。"""
    builtin = [k for k, s in FACTORS.items() if not s.code]
    custom = [k for k, s in FACTORS.items() if s.code]
    lines = [
        "## 三、因子索引",
        "",
        f"内置 {len(builtin)} 个、用户自定义 {len(custom)} 个。想看某一条的完整实现，用 "
        "`{\"action\": \"doc\", \"topic\": \"factor:<key>\"}`；想看整类，用 "
        "`\"cat:趋势\"` 这种。",
        "",
        "**动手写新因子之前先扫一遍这张表** —— 库里多一个几乎重复的因子，比少一个",
        "更难维护。很多想法用现有因子组合（多因子模式）就能表达。",
        "",
    ]
    for group, keys in (("内置", builtin), ("用户自定义", custom)):
        if not keys:
            continue
        lines += [f"### {group}", ""]
        if group == "用户自定义":
            lines += [
                "这些是运行时注册进来的（用户自己写的，或 Agent 在对话里写的）。",
                "**里面可能有临时排障用的探针因子**（label 带「临时」的），不要拿它们",
                "当写法范例 —— 要学写法看上面的内置因子。",
                "",
            ]
        cat = ""
        for k in keys:
            d = L.factor_doc(k)
            if d is None:
                continue
            if d.category != cat:
                cat = d.category
                lines += ["", f"**{cat}**", ""]
            lines.append(f"- `{d.key}`｜{d.label}｜{d.formula}｜{d.desc}")
        lines.append("")
    return "\n".join(lines).rstrip()


def factors_section(category: str = "") -> str:
    """按类别给全文。类别为空时给全部。"""
    keys = [k for k, s in FACTORS.items() if not category or s.category == category]
    if not keys:
        cats = sorted({s.category for s in FACTORS.values()})
        return f"（没有类别「{category}」，可选：{'、'.join(cats)}）"
    head = f"## {category}因子（{len(keys)} 个）" if category else f"## 全部因子（{len(keys)} 个）"
    body = "\n\n".join(factor_section(k) for k in keys)
    return f"{head}\n\n{body}"


# ------------------------------------------------------------------ 契约段


CONTRACT = """## 四、写因子的契约

```python
def f_你的名字(df):
    ...
    return series   # 等长 Series，索引与 df 完全一致
```

硬约束，违反任何一条都会注册失败：

1. **签名固定** `def f_xxx(df)`，只接一个参数
2. **不要 import**。`np`、`pd` 和上一节的原语都已注入
3. **返回等长 Series，索引和 df 完全一致**。不要 `dropna()`、不要 `reset_index()`
   —— 预热期的 NaN 要留着，下游统一处理
4. **必须 PIT**：只用 `rolling` / `ewm` / `shift(正数)`，**绝不出现 `shift(-n)`**。
   用未来数据的因子在回测里会好得离谱、实盘一分钱赚不到
5. **只算原始值**，标准化（滚动 z-score）平台统一做
6. `key` 用英文小写加下划线，`label` 用中文

常见的几个错：

| 症状 | 原因 |
|---|---|
| 注册成功但样本上全是 NaN | 读了不存在的列（分钟线写 `open_interest`、美股写 `close_oi`、没下衍生数据就读 `c1_close`） |
| 长度不一致 | 用了 `dropna()` 或 `value_counts()` 这类会改变长度的操作 |
| IC 高得离谱 | 用了 `shift(-n)` 或全样本统计量 |
| 注册失败「条件永远不成立」 | 比较了两个不同索引的 Series，结果全是 False |
"""


# ------------------------------------------------------------------ 组装


def topics() -> list[tuple[str, str]]:
    """可用的主题，给界面和提示词列清单用。"""
    cats = []
    for s in FACTORS.values():
        if s.category not in cats:
            cats.append(s.category)
    out = [
        ("data", "行情数据：三条链路、列名、复权、缺数据"),
        ("primitives", "平台注入的原语源码"),
        ("contract", "写因子/策略的契约与常见错误"),
        ("factors", "全部因子索引（一行一个）"),
    ]
    out += [(f"cat:{c}", f"{c}类因子的完整实现") for c in cats]
    out.append(("factor:<key>", "单个因子的完整实现"))
    return out


def resolve(topic: str) -> tuple[str, str]:
    """把主题解析成 (标题, 正文)。解析不了就返回索引并说明用法。"""
    t = (topic or "").strip()
    low = t.lower()

    if low in ("data", "数据"):
        return "行情数据", DATA_GUIDE
    if low in ("primitives", "原语", "api"):
        return "平台注入的原语", primitive_section()
    if low in ("contract", "契约", "rules"):
        return "因子契约", CONTRACT
    if low in ("factors", "索引", "index", ""):
        return "因子索引", factor_index()
    if low in ("manual", "all", "全部"):
        return "平台手册（全文）", manual()
    if low.startswith("cat:"):
        cat = t.split(":", 1)[1].strip()
        if not any(s.category == cat for s in FACTORS.values()):
            return "没这个类别", factors_section(cat)
        return f"{cat}类因子", factors_section(cat)
    if low.startswith("factor:"):
        key = t.split(":", 1)[1].strip()
        d = L.factor_doc(key)
        if d is None:
            return "没这个因子", (
                f"（没有 `{key}` 这个因子。先看 `factor:factors` 的索引，"
                "或注意 key 是全小写加下划线。）"
            )
        return f"因子 {key}", factor_section(key)
    # 直接给 key 也认，省得模型每次都要写前缀
    if L.factor_doc(t) is not None:
        return f"因子 {t}", factor_section(t)
    return "因子索引", (
        f"（不认识的 topic：`{t}`。\n\n可用：" +
        "、".join(f"`{k}`" for k, _ in topics()) + "）\n\n" + factor_index()
    )


def manual() -> str:
    """整份手册。约四万字，只在用户主动导出/预览时用。

    顺序按「先知道数据长什么样，再看平台给了什么工具，然后查索引，最后才是
    几万字的具体实现」—— 一上来就铺 47 段代码，读的人（和模型）都抓不住重点。
    """
    parts = [
        "# 平台手册 · 数据与因子",
        "",
        "这份手册给「ChanAgent」看：行情怎么来的、平台注入了哪些原语、每个因子的",
        "完整实现。人也能读 —— 界面里「手册」按钮打开的就是它。",
        "",
        DATA_GUIDE,
        "",
        primitive_section(),
        "",
        factor_index(),
        "",
        CONTRACT,
        "",
        "## 五、全部因子的完整实现",
        "",
        factors_section(),
    ]
    return "\n".join(parts).rstrip() + "\n"


def save(path: str) -> tuple[bool, str]:
    """整份手册写盘。给界面上的「保存」按钮用。"""
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(manual())
    except OSError as exc:
        return False, f"写不进去：{exc}"
    return True, f"已写入 {path}（{len(manual())} 字符）"
