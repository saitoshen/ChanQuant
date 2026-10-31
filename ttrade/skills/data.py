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
