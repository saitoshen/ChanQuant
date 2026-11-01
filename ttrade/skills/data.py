"""查数据：行情摘要、因子数值、可用品种。

这是**最常用的一类**动作，以前却完全没有 —— 用户问「螺纹最近 20 天动量多少」，
模型只能答「我没跑过回测」。现在直接算给它。

全部只读、零风险，所以不弹确认框。
"""

from __future__ import annotations

import pandas as pd

from ttrade import blocks as B
from ttrade.skills.base import (
    RISK_READ,
    SkillResult,
    Tool,
    register,
)

# 一次最多返回多少行。再多模型也读不完，还会把上下文挤满
MAX_ROWS = 200


def _store(ctx):
    store = getattr(ctx, "store", None)
    if store is None or not getattr(store, "symbols", None):
        raise RuntimeError("行情库没准备好（界面还没加载完数据）")
    return store


def _looks_like_option(key: str) -> bool:
    """`DCE.m2405-C-3000` / `SHFE.br2407P16600` 这种就是期权合约。

    品种池里只有期货品种（`RB_SHFE` 这种），期权合约永远查不到 —— 但用户
    完全可能直接问「m2405-C-3000 最近怎么样」。认出来就去期权库找。
    """
    from ttrade import optfetch as OF

    return bool(OF.parse_contract(key)) and "." in key


def _bars(ctx, symbol: str, period: str) -> pd.DataFrame:
    store = _store(ctx)
    key = str(symbol or "").strip()
    if not key:
        raise ValueError("要指定品种")
    # A 股代码三种写法（`600000` / `600000.SH` / `SSE.600000`）先收敛到品种池
    # 里的键。模型只看得见对话和截图，不该指望它记住我们的命名规则
    from ttrade.strategy_store import normalize_symbol

    key, _ex = normalize_symbol(key)
    df = store.get(key, str(period or "1d"))
    if df is not None and not df.empty:
        return df

    # 期权合约不在品种池里，退到期权库（`data_tq/option.duckdb`）
    if _looks_like_option(key):
        from ttrade import optfetch as OF
        from ttrade import optstore as OS

        if not OS.exists():
            raise RuntimeError(
                f"{key} 是期权合约，本地还没有期权库 —— 去「数据管理 → 天勤」"
                "勾上「一并更新期权（隐含波动率）」"
            )
        opt = OF.load_contract(key, str(period or "1d"))
        if not opt.empty:
            return opt.set_index("dt")
        raise RuntimeError(
            f"{key} 在期权库里没有 {period} 数据（可能是没下过这个周期，"
            "或者这个合约从来没成交过）"
        )

    have = "、".join(store.symbols[:8])
    raise RuntimeError(f"{key} 在 {period} 周期上没有数据（库里有：{have} …）")


# ------------------------------------------------------------------ query_bars

def _query_bars(ctx, p: dict) -> SkillResult:
    df = _bars(ctx, p["symbol"], p.get("period", "1d"))
    rows = min(int(p.get("rows") or 20), MAX_ROWS)
    tail = df.tail(rows)

    close = tail["close"] if "close" in tail else pd.Series(dtype=float)
    last = float(close.iloc[-1]) if len(close) else float("nan")
    first = float(close.iloc[0]) if len(close) else float("nan")
    chg = (last / first - 1.0) * 100 if first else float("nan")

    out = []
    for ts, row in tail.iterrows():
        out.append([
            ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts),
            _f(row.get("open")), _f(row.get("high")), _f(row.get("low")),
            _f(row.get("close")),
            _f(row.get("volume"), 0),
        ])

    blocks = [
        B.block_metrics(
            f"{p['symbol']} · {p.get('period', '1d')} · 最近 {len(tail)} 根",
            [
                ("最新收盘", _f(last), "flat"),
                (f"{len(tail)} 根累计", f"{chg:+.2f}%", "good" if chg >= 0 else "bad"),
                ("区间最高", _f(float(tail["high"].max())) if "high" in tail else "—", "flat"),
                ("区间最低", _f(float(tail["low"].min())) if "low" in tail else "—", "flat"),
                ("全历史", f"{len(df)} 根", "flat"),
            ],
        ),
        B.block_table(["时间", "开", "高", "低", "收", "量"], out),
    ]
    summary = (f"{p['symbol']} {p.get('period', '1d')} 最近 {len(tail)} 根："
               f"收 {_f(last)}，区间 {chg:+.2f}%")
    return SkillResult(True, summary=summary, blocks=blocks, feed=summary)


def _f(v, nd: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if x != x:                       # NaN
        return "—"
    return f"{x:,.{nd}f}"


# ------------------------------------------------------------------ query_factor

def _query_factor(ctx, p: dict) -> SkillResult:
    from ttrade.factors import FACTORS

    key = str(p["factor"] or "").strip()
    spec = FACTORS.get(key)
    if spec is None:
        near = [k for k in FACTORS if key and key in k][:6]
        raise RuntimeError(
            f"没有名为 {key} 的因子" + (f"（相近的有：{'、'.join(near)}）" if near else "")
        )

    df = _bars(ctx, p["symbol"], p.get("period", "1d"))
    series = spec.func(df)
    if series is None or not len(series.dropna()):
        raise RuntimeError(
            f"因子「{spec.label}」在 {p['symbol']} 的 {p.get('period', '1d')} 上算不出数 —— "
            "多半是这类因子只在分钟线上有效（日内 / 时段类）"
        )

    valid = series.dropna()
    rows = min(int(p.get("rows") or 10), MAX_ROWS)
    tail = valid.tail(rows)
    last = float(valid.iloc[-1])
    # 分位：这个值在它自己的历史里排第几。比绝对值有意义得多
    pct = float((valid <= last).mean())

    table = [[ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts),
              f"{float(v):.4f}"] for ts, v in tail.items()]

    blocks = [
        B.block_metrics(
            f"{spec.label}（{key}）· {p['symbol']} · {p.get('period', '1d')}",
            [
                ("最新值", f"{last:.4f}", "flat"),
                ("历史分位", f"{pct * 100:.0f}%", "good" if pct > 0.8 else
                 ("bad" if pct < 0.2 else "flat")),
                ("有效样本", f"{len(valid)} 根", "flat"),
                ("类别", spec.category, "flat"),
            ],
        ),
        B.block_table(["时间", key], table, note=f"最近 {len(tail)} 个值"),
    ]
    if spec.desc:
        blocks.insert(1, B.block_text(spec.desc))

    summary = (f"{spec.label}（{key}）在 {p['symbol']} 上最新 {last:.4f}，"
               f"历史分位 {pct * 100:.0f}%（{len(valid)} 个有效样本）")
    return SkillResult(True, summary=summary, blocks=blocks, feed=summary)


# ------------------------------------------------------------------ list_symbols

def _list_symbols(ctx, p: dict) -> SkillResult:
    store = _store(ctx)
    items = getattr(store, "items", []) or []
    period = str(p.get("period") or "").strip()

    rows = []
    for it in items[:120]:
        rows.append([it.get("key", ""), it.get("name", ""),
                     it.get("exchange", "") or "—"])

    # 只报「这个品种在库里」，**不要**对每个品种调 `store.get()` 去探周期 ——
    # 那是把全市场的 K 线全读一遍，几分钟下不来。周期有没有数据让
    # `query_bars` 自己报错就行。
    note = f"共 {len(items)} 个品种" + (f"（{period} 周期请用 query_bars 逐个试）"
                                        if period else "")
    return SkillResult(True, summary=note, feed=note, blocks=[
        B.block_table(["代号", "名称", "交易所"], rows,
                      note=note + ("｜只列前 120 个" if len(items) > 120 else "")),
    ])


# ------------------------------------------------------------------ list_contracts

def _split_key(key: str) -> tuple[str, str]:
    """`RB_SHFE` → (`RB`, `SHFE`)；`rb` → (`rb`, ``)。

    交易所后缀一定是大写字母（`SHFE` / `DCE` / `CZCE` / `INE` / `GFEX` / `CFFEX`），
    品种代码本身可能是小写（`rb` / `m`）—— 用大小写判断，比维护一张交易所表稳。
    """
    sym, _, ex = str(key or "").strip().rpartition("_")
    if ex and ex.isalpha() and ex.isupper():
        return sym, ex
    return str(key or "").strip(), ""


def _variety_in(months: pd.DataFrame, symbol: str) -> str | None:
    """期权库里的品种写法**跟着合约代码走**（`IO` 大写、`br` 小写），
    而品种池里是大写（`RB_SHFE`）。按大小写不敏感匹配，返回库里那个写法。
    """
    for v in months["variety"].astype(str).unique():
        if v.lower() == str(symbol).lower():
            return v
    return None


def _list_contracts(ctx, p: dict) -> SkillResult:
    """列合约。期货列真实合约；期权列链（月份），再给 `underlying` 就列行权价。

    用户嘴上说的是「m2405 的认购」，落到库里是三步：品种 → 链 → 合约代码。
    这个工具把前两步摊开，第三步交给 `query_bars`（它已经认期权合约代码了）。
    """
    kind = str(p.get("kind") or "futures").strip().lower()
    sym, ex = _split_key(p.get("symbol", ""))
    if not sym:
        raise ValueError("要指定品种，比如 RB_SHFE / M_DCE")
    period = str(p.get("period") or "1d").strip()
    limit = min(int(p.get("rows") or 60), MAX_ROWS)

    if kind.startswith("opt"):
        from ttrade import optstore as OS

        months = OS.list_months()
        if months is None:
            raise RuntimeError(
                "期权库正在被下载占用（DuckDB 单写单读）—— 等 download_options.py "
                "跑完再查，别当成「没有数据」"
            )
        if months.empty:
            raise RuntimeError(
                "本地还没有期权库 —— 去「数据管理 → 天勤」勾上「一并更新期权"
                "（隐含波动率）」，或者命令行跑 download_options.py"
            )
        v = _variety_in(months, sym)
        if v is None:
            names = "、".join(sorted(months["variety"].unique())[:12])
            raise RuntimeError(f"期权库里没有 {sym}（有的是：{names} …）")

        und = str(p.get("underlying") or "").strip()
        if und:
            mine = months[months["variety"].astype(str) == v]
            mine = mine[mine["underlying"].astype(str) == und]
            if mine.empty:
                have = "、".join(
                    months[months["variety"].astype(str) == v]["underlying"]
                    .astype(str).unique()[:8])
                raise RuntimeError(f"{und} 这条链不存在（{v} 有的是：{have} …）")
            exp = str(p.get("expiry") or "").strip()
            if not exp and mine["expiry"].nunique() > 1:
                # 股指期权所有月份共用一个标的（`SSE.000300`），不指明到期日
                # 就会把 24 个月的行权价混成两千多个 —— 先把月份摊给模型挑
                rows = [[pd.Timestamp(r["expiry"]).strftime("%Y-%m-%d"),
                         int(r["n"])] for _, r in mine.iterrows()]
                note = (f"{und} 有 {len(mine)} 个到期月｜"
                        "带上 `expiry` 再查一次行权价")
                return SkillResult(True, summary=note, feed=note, blocks=[
                    B.block_table(["到期日", "合约数"], rows, note=note),
                ])
            strikes = OS.list_strikes(
                v, und, exp or pd.Timestamp(mine["expiry"].iloc[0]))
            if strikes.empty:
                raise RuntimeError(f"{und} {exp} 这条链上没有合约")
            spot = OS.underlying_close(und, "1d")
            rows = []
            for _, r in strikes.head(limit).iterrows():
                rows.append([_f(r["strike"], 0), str(r["cp"]).upper(),
                             str(r["contract"]), int(r["rows"])])
            note = f"{und} {exp or ''} 共 {len(strikes)} 个合约".replace("  ", " ")
            if spot:
                note += f"，标的 {spot:g}"
            if len(strikes) > limit:
                note += f"｜只列前 {limit} 个"
            return SkillResult(True, summary=note, feed=note, blocks=[
                B.block_table(["行权价", "C/P", "合约代码", "日线根数"], rows, note=note),
            ])

        mine = months[months["variety"].astype(str) == v].copy()
        mine = mine.sort_values(["expiry", "underlying"]).head(limit)
        rows = []
        for _, r in mine.iterrows():
            exp = pd.Timestamp(r["expiry"])
            rows.append([exp.strftime("%Y-%m-%d"), str(r["underlying"]), int(r["n"])])
        note = (f"{v} 有 {len(months[months['variety'].astype(str) == v])} 条期权链"
                "｜挑一条再把它的 `underlying`（必要时加 `expiry`）传回来查行权价")
        return SkillResult(True, summary=note, feed=note, blocks=[
            B.block_table(["到期日", "标的（链）", "合约数"], rows, note=note),
        ])

    from ttrade import tqstore as S

    df = S.list_contracts(sym, ex, period)
    if df.empty:
        raise RuntimeError(
            f"本地没有 {sym}_{ex} 的真实合约 {period} 数据 —— 去「数据管理 → 天勤」"
            "勾上「一并下载真实合约」，或者换个周期（分钟级要下过才有）"
        )
    tail = df.sort_values("contract", ascending=False).head(limit)
    rows = []
    for _, r in tail.iterrows():
        rows.append([str(r["contract"]), f"{r['first']:%Y-%m-%d}",
                     f"{r['last']:%Y-%m-%d}", int(r["rows"])])
    note = (f"{sym}_{ex} 共 {len(df)} 个真实合约（{period}）"
            + (f"｜只列最近 {limit} 个" if len(df) > limit else ""))
    return SkillResult(True, summary=note, feed=note, blocks=[
        B.block_table(["合约", "首根", "末根", "根数"], rows, note=note),
    ])


# ------------------------------------------------------------------ 注册

register(Tool(
    name="query_bars",
    group="data",
    label="查行情",
    description="查某个品种最近 N 根的 OHLCV 和区间涨跌",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["symbol"],
        "properties": {
            "symbol": {"type": "str"},
            "period": {"type": "str", "default": "1d"},
            "rows": {"type": "int", "default": 20, "minimum": 1, "maximum": MAX_ROWS},
        },
    },
    handler=_query_bars,
    prompt=(
        "**查行情**：`query_bars`。用户问「XX 最近怎么样 / 收多少 / 涨了多少」时用它，"
        "别用回测去间接算。`symbol` 填库里的代号（`RB_SHFE` 这种，见 `list_symbols`），"
        "`period` 是 `1d` / `1d_tq` / `1h` / `5m` 之类。"
    ),
    max_turns=3,
))

register(Tool(
    name="query_factor",
    group="data",
    label="查因子值",
    description="算某个因子在某品种上的最新值和历史分位",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["factor", "symbol"],
        "properties": {
            "factor": {"type": "str"},
            "symbol": {"type": "str"},
            "period": {"type": "str", "default": "1d"},
            "rows": {"type": "int", "default": 10, "minimum": 1, "maximum": MAX_ROWS},
        },
    },
    handler=_query_factor,
    prompt=(
        "**查因子值**：`query_factor`。用户问「动量多少 / 这个因子现在几分」时用它。"
        "分位比绝对值重要：0.85 说明处在历史高位。日内和时段类因子（VWAP 偏离、"
        "隔夜跳空等）只有 5m/1h 算得出来，日线上会返回「算不出数」。"
    ),
    max_turns=3,
))

register(Tool(
    name="list_symbols",
    group="data",
    label="列品种",
    description="列出当前市场里可用的品种代号和名称",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "properties": {
            "period": {"type": "str", "default": ""},
        },
    },
    handler=_list_symbols,
    prompt="**列品种**：`list_symbols`。不确定代号怎么写时先查，不要猜。",
))

register(Tool(
    name="list_contracts",
    group="data",
    label="列合约",
    description="列期货真实合约，或某个品种的期权链 / 某条链的行权价",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["symbol"],
        "properties": {
            "symbol": {"type": "str"},
            "kind": {"type": "str", "default": "futures"},
            "underlying": {"type": "str", "default": ""},
            "expiry": {"type": "str", "default": ""},
            "period": {"type": "str", "default": "1d"},
            "rows": {"type": "int", "default": 60, "minimum": 1, "maximum": MAX_ROWS},
        },
    },
    handler=_list_contracts,
    prompt=(
        "**列合约**：`list_contracts`。用户问「RB 有哪些合约」「m 的期权有哪些月份」"
        "「m2405 有哪些行权价」时用它。\n"
        "- 期货：`kind=futures`，`symbol=RB_SHFE`\n"
        "- 期权链：`kind=options`，`symbol=M_DCE`（品种代号，**不是**合约代码）\n"
        "- 行权价：`kind=options`，再带 `underlying=DCE.m2405`；股指期权所有月份"
        "共用一个标的，这时还要带 `expiry=2026-09-18`\n"
        "拿到合约代码之后交给 `query_bars` 查行情。合约有到期日、曲线是断的，"
        "只能看图，不要拿它跑回测。"
    ),
    max_turns=3,
))
