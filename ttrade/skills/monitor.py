"""盯盘：扫一遍全市场，找当下满足某个形态的品种。

## 为什么是「快照」而不是「订阅」

订阅式（挂后台、条件成立就推）要常驻进程 + 托盘 + 通知通道，是 P3 的第二步。
但用户真正想说的第一句话是「**现在**帮我扫一遍哪些品种突破 20 日高点」——
这句话不需要常驻，扫一次、给张表就完了。所以先做快照，订阅放在托盘那一层。

## 口径

- 全部走 `ctx.store.get(key, period)`，和 K 线页看的是同一份数据（含前复权 /
  衍生列）。**不**直接查 DuckDB：那会绕开复权口径，扫出来的信号和图上对不上。
- 「创新高」比的是**前 N 根的最高价**，不含当根 —— 含当根的话任何一根阳线都算
  创新高，信号没有意义。
- 数据陈旧（最后一根比全市场最新那根落后太多）的品种直接跳过，不参与排序。
  停更的品种天天「创新高」是这类扫描最常见的假信号来源。
"""

from __future__ import annotations

import time

import pandas as pd

from ttrade import blocks as B
from ttrade.skills.base import (
    RISK_READ,
    SkillResult,
    Tool,
    register,
)

# 一次最多扫多少品种。分钟级一个品种要读十几万根（约 0.25 秒），全市场 78 个
# 就是 20 秒，会把界面卡住。日线 78 个品种加起来不到 5 秒，全扫没关系。
MAX_SYMBOLS_DAILY = 0          # 0 = 不限制
MAX_SYMBOLS_INTRADAY = 40
# 结果最多列多少行。再多模型读不完，表格也太长
MAX_HITS = 60
# 数据陈旧的判定：落后全市场最新一根超过这么多天就跳过
STALE_DAYS_DAILY = 10
STALE_DAYS_INTRADAY = 4

RULE_LABELS = {
    "breakout_high": "创 N 日新高",
    "breakdown_low": "创 N 日新低",
    "ma_cross": "均线交叉",
    "volume_spike": "放量",
}


# ------------------------------------------------------------------ 单品种判定
def _check(df: pd.DataFrame, p: dict) -> dict | None:
    """返回命中信息（含用于排序的 `rank` 值），没命中返回 None。"""
    rule = p.get("rule") or "breakout_high"
    window = int(p.get("window") or 20)
    close = df["close"]
    need = window + 2
    if rule == "ma_cross":
        need = max(int(p.get("slow") or 20), int(p.get("fast") or 5)) + 2
    if len(df) < need:
        return None

    if rule in ("breakout_high", "breakdown_low"):
        last = float(close.iloc[-1])
        if rule == "breakout_high":
            level = float(df["high"].iloc[-window - 1:-1].max())
            if not (last > level):
                return None
            dist = (last / level - 1.0) * 100 if level else 0.0
            return {"level": level, "dist": dist, "rank": dist, "kind": "新高"}
        level = float(df["low"].iloc[-window - 1:-1].min())
        if not (last < level):
            return None
        dist = (last / level - 1.0) * 100 if level else 0.0
        # 跌破得越深越靠前，所以排序值取负
        return {"level": level, "dist": dist, "rank": -dist, "kind": "新低"}

    if rule == "ma_cross":
        fast = int(p.get("fast") or 5)
        slow = int(p.get("slow") or 20)
        if fast >= slow:
            raise ValueError("快线周期必须小于慢线周期")
        mf = close.rolling(fast).mean()
        ms = close.rolling(slow).mean()
        a0, b0 = mf.iloc[-2], ms.iloc[-2]
        a1, b1 = mf.iloc[-1], ms.iloc[-1]
        if pd.isna(a0) or pd.isna(b0) or pd.isna(a1) or pd.isna(b1):
            return None
        up = a0 <= b0 and a1 > b1
        down = a0 >= b0 and a1 < b1
        if not (up or down):
            return None
        gap = (a1 / b1 - 1.0) * 100 if b1 else 0.0
        return {"level": float(a1), "dist": gap, "rank": abs(gap),
                "kind": f"{fast}/{slow} 金叉" if up else f"{fast}/{slow} 死叉"}

    # volume_spike
    base = df["volume"].iloc[-window - 1:-1].mean()
    if not base or pd.isna(base):
        return None
    vr = float(df["volume"].iloc[-1]) / float(base)
    if vr < float(p.get("ratio") or 2.0):
        return None
    chg = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1.0) * 100 \
        if len(close) > 1 else 0.0
    return {"level": float(base), "dist": chg, "rank": vr, "kind": f"量比 {vr:.1f}"}


# ------------------------------------------------------------------ 扫描
def _scan(ctx, p: dict) -> SkillResult:
    store = getattr(ctx, "store", None)
    if store is None or not getattr(store, "symbols", None):
        raise RuntimeError("行情库没准备好（界面还没加载完数据）")

    rule = str(p.get("rule") or "breakout_high")
    if rule not in RULE_LABELS:
        raise ValueError(f"没有叫 {rule} 的扫描规则（可用：{'、'.join(RULE_LABELS)}）")
    period = str(p.get("period") or "1d")
    window = int(p.get("window") or 20)
    limit = min(int(p.get("limit") or 20), MAX_HITS)

    items = list(store.items)
    daily = _is_daily(period)
    cap = MAX_SYMBOLS_DAILY if daily else MAX_SYMBOLS_INTRADAY
    capped = bool(cap and len(items) > cap)
    if capped:
        items = items[:cap]

    t0 = time.time()
    hits: list[dict] = []
    stale = 0
    empty = 0
    newest = None

    for it in items:
        key = it.get("key") or ""
        try:
            df = store.get(key, period)
        except Exception:            # noqa: BLE001
            df = None
        if df is None or df.empty or "close" not in df:
            empty += 1
            continue
        last = df.index[-1]
        last = pd.Timestamp(last) if not isinstance(last, pd.Timestamp) else last
        if newest is None or last > newest:
            newest = last
        info = _check(df, p)
        if info is None:
            continue
        hits.append({"key": key, "name": it.get("name") or "",
                     "ex": it.get("exchange") or "",
                     "close": float(df["close"].iloc[-1]),
                     "date": last, **info})

    # 陈旧品种要剔掉 —— 但得先知道「最新」是哪一天，所以只能扫完再筛
    tol = STALE_DAYS_DAILY if daily else STALE_DAYS_INTRADAY
    if newest is not None:
        cut = newest - pd.Timedelta(days=tol)
        keep = [h for h in hits if h["date"] >= cut]
        stale = len(hits) - len(keep)
        hits = keep

    hits.sort(key=lambda h: h["rank"], reverse=True)
    shown = hits[:limit]
    elapsed = time.time() - t0

    # ---------------- 渲染
    if rule == "ma_cross":
        head = (f"均线交叉 · {period} · {int(p.get('fast') or 5)}"
                f"/{int(p.get('slow') or 20)}")
    elif rule == "volume_spike":
        head = f"放量 · {period} · 窗口 {window} 根 · 量比≥{float(p.get('ratio') or 2.0):g}"
    else:
        head = f"{RULE_LABELS[rule]} · {period} · 窗口 {window} 根"

    # 只放四列。六列（品种/名称/收盘/关键位/幅度/日期）在窄窗口里会被 Qt 压到
    # 每列三十几像素，「2026-10-30」和「+2.17%」都会断行，看着像坏表。代号和名称
    # 合成一格，日期移到上面的指标卡 —— 一次扫描本来就只有一天。
    if rule == "volume_spike":
        cols = ["品种", "收盘", "量比", "涨跌"]
        rows = [[_sym(h), _f(h["close"]), f"{h['rank']:.1f}", f"{h['dist']:+.2f}%"]
                for h in shown]
    elif rule == "ma_cross":
        cols = ["品种", "收盘", "信号", "均线差"]
        rows = [[_sym(h), _f(h["close"]), h["kind"], f"{h['dist']:+.2f}%"]
                for h in shown]
    else:
        cols = ["品种", "收盘", "关键位", "幅度"]
        rows = [[_sym(h), _f(h["close"]), _f(h["level"]), f"{h['dist']:+.2f}%"]
                for h in shown]

    blocks = [
        B.block_metrics(head, [
            ("命中", f"{len(hits)} 个", "good" if hits else "flat"),
            ("扫描", f"{len(items)} 个品种", "flat"),
            ("数据截止", _dt(newest) if newest is not None else "—", "flat"),
            ("耗时", f"{elapsed:.1f} 秒", "flat"),
        ]),
    ]
    if rows:
        note = f"按{'幅度' if rule in ('breakout_high', 'breakdown_low') else '强度'}排序"
        if len(hits) > len(shown):
            note += f"，共 {len(hits)} 个命中，只列前 {len(shown)} 个"
        # 美股是后复权价（含分红再投资），和盘面价差很远。不写这一句，
        # 用户看到「麦当劳 16 万」会以为数据错了
        if any(not h["ex"] for h in shown):
            note += "｜美股为后复权价"
        blocks.append(B.block_table(cols, rows, note=note))
    else:
        blocks.append(B.block_text(
            f"{period} 上扫了 {len(items)} 个品种，没有满足「{RULE_LABELS[rule]}」的。"
            "窗口调小一点、或换个规则再试。"
        ))
    if stale or empty:
        lines = []
        if stale:
            lines.append(f"{stale} 个命中因为数据陈旧被剔除（落后全市场最新一根超过 {tol} 天）")
        if empty:
            lines.append(f"{empty} 个品种在 {period} 上没有数据，已跳过")
        blocks.append(B.block_warning("扫描说明", lines, level="warn"))
    if capped:
        blocks.append(B.block_warning(
            "扫描范围", [f"{period} 是分钟级，只扫了前 {cap} 个品种（全扫要几十秒，"
                         "会把界面卡住）"], level="warn"))

    summary = (f"{head}：扫 {len(items)} 个品种，命中 {len(hits)} 个"
               + ("（" + "、".join(h["key"] for h in shown[:6])
                  + ("…" if len(shown) > 6 else "") + "）" if shown else ""))
    feed = summary + "\n" + "\n".join(
        f"- {h['key']} {h['name']}：收 {_f(h['close'])}，{h['kind']}，"
        f"{'幅度' if rule in ('breakout_high', 'breakdown_low') else '强度'} "
        f"{h['dist'] if rule != 'volume_spike' else h['rank']:.2f}"
        f"{'%' if rule != 'volume_spike' else ''}，{_dt(h['date'])}"
        for h in shown[:15]
    )
    return SkillResult(True, summary=summary, blocks=blocks, feed=feed)


def _is_daily(period: str) -> bool:
    """日线（含 `1d_tq` 这种变体）才允许全市场扫 —— 分钟级一个品种十几万根。"""
    try:
        from ttrade import tqdata

        return bool(tqdata.is_daily(period))
    except Exception:            # noqa: BLE001
        return str(period).split("_")[0].endswith("d")


def _f(v, nd: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if x != x:
        return "—"
    return f"{x:,.{nd}f}"


def _sym(h: dict) -> str:
    """代号 + 名称合成一格。表格只有四列，分两列会把代号挤成两行。"""
    name = h.get("name") or ""
    return f"{h['key']} {name}".strip()


def _dt(ts) -> str:
    if ts is None:
        return "—"
    if hasattr(ts, "strftime"):
        return ts.strftime("%Y-%m-%d %H:%M") if getattr(ts, "hour", 0) \
            else ts.strftime("%Y-%m-%d")
    return str(ts)


# ------------------------------------------------------------------ 注册
register(Tool(
    name="scan",
    group="monitor",
    label="扫全市场",
    description="按一个形态扫一遍所有品种，返回当下满足条件的清单",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "properties": {
            "rule": {"type": "str", "default": "breakout_high",
                     "enum": list(RULE_LABELS)},
            "period": {"type": "str", "default": "1d"},
            "window": {"type": "int", "default": 20, "minimum": 2, "maximum": 250},
            "limit": {"type": "int", "default": 20, "minimum": 1, "maximum": MAX_HITS},
            "fast": {"type": "int", "default": 5, "minimum": 2, "maximum": 120},
            "slow": {"type": "int", "default": 20, "minimum": 3, "maximum": 250},
            "ratio": {"type": "float", "default": 2.0, "minimum": 1.1, "maximum": 20.0},
        },
    },
    handler=_scan,
    prompt=(
        "**扫全市场**：`scan`。用户问「现在哪些品种创新高 / 跌破 / 金叉 / 放量」"
        "这类**横截面**问题时用它，不要一个品种一个品种地调 `query_bars`。\n"
        "- `rule`：`breakout_high`（创 N 日新高）/ `breakdown_low`（新低）/ "
        "`ma_cross`（均线交叉，配 `fast`/`slow`）/ `volume_spike`（放量，配 `ratio`）\n"
        "- `window` 默认 20；`period` 默认 `1d`，分钟级只扫前 40 个品种\n"
        "- 结果按强度排好序，`feed` 里有前 15 个的明细，直接照着讲，别重算"
    ),
    max_turns=3,
))
