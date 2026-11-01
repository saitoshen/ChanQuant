"""研报：把「写报告要用到的数字」一次取全，再按需出图。

## 为什么要有 `report_pack`

写研报最费时间的不是写字，是**凑数**：季节性要看近十年同月均值、库存要看
EDB、因子要看 IC、策略要跑回测、宏观要看分位……每一样都单独问一遍，模型要
发七八轮工具调用，中间还会忘。

`report_pack` 把这些一次算完，压成一份带表格的事实包。模型拿到之后只需要
判断和表达，不用再翻数据 —— 这样写出来的研报才有数字支撑，而不是通篇
「一般来说」「可能」「有望」。

## 为什么数字都带口径

同一件事有好几种算法（波动率年化 252 还是 244、季节性算不算当年、IC 用
秩相关还是皮尔逊）。报告里必须写清用的是哪一种，否则读者没法复核。所以
事实包里的每个数字都跟着一句口径说明。

## 图表单独一个工具

`chart` 按需出图，返回文件路径。模型把路径写成 markdown 的
`![说明](路径)`，导出 PDF 时会自动内联成图片（`render._render_image` 负责
把本地文件读成 base64 —— 直接写 `file://` 在离屏文档里加载不到，会静默空白）。
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from ttrade import blocks as B
from ttrade.skills.base import (
    RISK_READ,
    SkillResult,
    Tool,
    register,
)

# 策略对比只跑这四条：基准 + 三条最常用的趋势策略。
# 全跑一遍要十几秒，研报里也用不上 9 条。
COMPARE_STRATS = ["buy_hold", "ma_cross", "donchian", "turtle"]

# 因子 IC 取这些类别里的代表，和 `report_charts.IC_KEYS` 同一个思路
PACK_IC_KEYS = [
    "mom_20", "mom_60", "mom_120", "rev_5", "rev_20",
    "vol_20", "atr_pct", "maxdd_60", "efficiency",
    "vol_surge", "oi_change", "turnover_20",
    "roll_yield", "roll_annual", "term_near_far",
    "iv_atm", "iv_skew", "pcr",
    "season_avg", "season_rank", "season_ir",
    "war_chg_5", "spot_basis", "inv_chg_5", "profit_val",
    "sector_mom_10", "commodity_mom_10",
]


def _load(symbol: str) -> pd.DataFrame:
    from ttrade import futures as FU
    from ttrade.paths import data_dir

    from ttrade.report_charts import _split

    sym, ex = _split(symbol)
    if not sym:
        raise RuntimeError(f"品种代码看不懂：{symbol}")
    try:
        df = FU.load_local(sym, ex, data_dir("cn"), adjust="qfq")
    except FileNotFoundError as exc:
        raise RuntimeError(f"{sym} 本地没有日线数据（先去「数据管理」补一次）") from exc
    if df is None or df.empty:
        raise RuntimeError(f"{sym} 本地没有日线数据")
    return df


def _pct(x) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:+.1f}%"


def _num(x, digits: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v:.{digits}f}"


def _same_series(a: pd.Series, b: pd.Series) -> bool:
    """两条序列在重叠日期上是否逐位相同。

    直接 `a.equals(b)` 不够：仓单是周频、期货库存是日频，同一条数据存成了
    两个频率，`equals` 会因为索引不同返回 False。
    """
    idx = a.index.intersection(b.index)
    if len(idx) < 10:
        return False
    va = a.reindex(idx).to_numpy(dtype=float)
    vb = b.reindex(idx).to_numpy(dtype=float)
    return bool(np.allclose(va, vb, equal_nan=True))


# ---------------------------------------------------------------- 事实包
def build_pack(symbol: str) -> dict:
    """把研报要用的事实算出来。返回 `{blocks, feed, charts}`。"""
    from ttrade import factor_metrics as FM
    from ttrade import factors as F
    from ttrade import report_charts as RC
    from ttrade import strategies as S

    df = _load(symbol)
    c = pd.to_numeric(df["close"], errors="coerce")
    last = float(c.iloc[-1])
    blocks: list[dict] = []
    feed: list[str] = []

    # ---- 1. 价格状态
    def ret(n):
        return c.iloc[-1] / c.iloc[-1 - n] - 1.0 if len(c) > n else np.nan

    eff = F.raw(df, "efficiency").iloc[-1] if len(df) > 60 else np.nan
    vol = F.raw(df, "vol_20").iloc[-1] if len(df) > 60 else np.nan
    vol_pct = np.nan
    if len(df) > 300:
        v = F.raw(df, "vol_20").dropna()
        if len(v) > 100:
            vol_pct = float((v <= v.iloc[-1]).mean())

    state = "趋势" if (eff or 0) >= 0.3 else ("震荡" if (eff or 1) < 0.15 else "过渡")
    blocks.append(B.block_metrics(f"{symbol} 价格状态", [
        ("最新价", _num(last), "flat"),
        ("20 日", _pct(ret(20)), "good" if (ret(20) or 0) > 0 else "bad"),
        ("60 日", _pct(ret(60)), "good" if (ret(60) or 0) > 0 else "bad"),
        ("120 日", _pct(ret(120)), "good" if (ret(120) or 0) > 0 else "bad"),
        ("20 日波动", _pct(vol), "flat"),
        ("波动分位", _pct(vol_pct), "flat"),
        ("效率比", _num(eff), "flat"),
        ("属性", state, "flat"),
    ]))
    feed.append(
        f"【价格状态】最新 {last:.2f}；20 日 {_pct(ret(20))}、60 日 {_pct(ret(60))}、"
        f"120 日 {_pct(ret(120))}；20 日年化波动 {_pct(vol)}（历史分位 {_pct(vol_pct)}）；"
        f"60 日效率比 {_num(eff)} → 判定为「{state}」（≥0.30 趋势 / <0.15 震荡）。"
        f"数据区间 {df.index[0].date()} ~ {df.index[-1].date()}，共 {len(df)} 根日线。"
    )

    # ---- 2. 季节性
    dc = pd.Series(c.to_numpy(dtype=float), index=pd.DatetimeIndex(df.index).normalize())
    dc = dc[~dc.index.duplicated(keep="last")].sort_index()
    tab = RC._monthly_table(dc).tail(10)
    mean = tab.mean(axis=0) * 100.0
    win = (tab > 0).mean(axis=0) * 100.0
    rows = [[f"{m}月", f"{mean.get(m, float('nan')):+.2f}%",
             f"{win.get(m, float('nan')):.0f}%"] for m in range(1, 13)]
    blocks.append(B.block_table(["月份", "平均收益", "上涨概率"], rows,
                                note=f"近 {len(tab)} 年同月统计，当年不计入"))
    best = mean.idxmax() if mean.notna().any() else None
    worst = mean.idxmin() if mean.notna().any() else None
    feed.append(
        "【月度季节性】" + "；".join(
            f"{m}月 {mean.get(m, float('nan')):+.2f}%（涨概率 {win.get(m, float('nan')):.0f}%）"
            for m in range(1, 13))
        + f"。历史上最强的是 {best} 月、最弱的是 {worst} 月（近 {len(tab)} 年）。"
    )

    # ---- 3. 基本面（EDB）
    try:
        from ttrade import factors_ext as FE

        var = str(symbol).split("_")[0].lower()
        panel = FE._edb_pick(var)
        if panel:
            items, notes, seen = [], [], []
            for cat, short in (("warehouse", "仓单"), ("inventory", "库存"),
                               ("spot_price", "现货价"), ("profit", "利润/成本"),
                               ("delivery", "交割量")):
                s = panel.get(cat)
                if s is None or len(s) == 0:
                    continue
                s = s.dropna()
                if len(s) == 0:
                    continue
                # 仓单和期货库存在 SHFE 是同一条数据（仓单数量 = 库存小计），
                # 两条都列出来只是把同一个数字写两遍。
                if any(_same_series(s, t) for t in seen):
                    continue
                seen.append(s)
                # 类别名只是槽位，真正是什么要看序列全名：螺纹的 `profit`
                # 里只有「成本(含税)」，写成「加工利润」就是编。
                full = FE._edb_name(var, cat, short)
                cur = float(s.iloc[-1])
                chg = (cur / float(s.iloc[-6]) - 1.0) if len(s) > 6 and float(s.iloc[-6]) else np.nan
                items.append((short, f"{cur:,.2f}", "flat"))
                notes.append(f"{short} {cur:,.2f}（{full}；5 期变化 {_pct(chg)}，"
                             f"{s.index[-1].date()}）")
            if items:
                blocks.append(B.block_metrics(f"{symbol} 基本面（EDB）", items))
                feed.append("【基本面】" + "；".join(notes) + "。")
            basis = F.raw(df, "spot_basis").dropna()
            if len(basis):
                feed.append(f"【现货基差】最新 {_pct(basis.iloc[-1])}，"
                            f"20 日前 {_pct(basis.iloc[-21]) if len(basis) > 21 else '—'}"
                            "（正值＝现货升水）。")
    except Exception:  # noqa: BLE001
        pass

    # ---- 4. 期限结构
    try:
        ry = F.raw(df, "roll_annual").dropna()
        if len(ry):
            feed.append(
                f"【期限结构】年化展期收益率 {_pct(ry.iloc[-1])}，20 日均值 "
                f"{_pct(ry.tail(20).mean())}（正值＝backwardation，近月贵）。")
    except Exception:  # noqa: BLE001
        pass

    # ---- 5. 因子 IC
    keys = F.filter_available(df, PACK_IC_KEYS, min_obs=120)
    if keys and len(df) >= 400:
        rep = FM.factor_report(df, keys, horizon=5, window=252)
        rep = rep[rep["ic"].notna()].copy()
        if not rep.empty:
            rep["abs"] = rep["ic"].abs()
            top = rep.sort_values("abs", ascending=False).head(10)

            def _ir(r) -> str:
                ir = r.get("ir")
                if pd.isna(ir):
                    return "—"
                # 全样本 IC 和滚动 IR 是两种口径，符号相反很常见 —— 标个 `*`
                return f"{ir:+.2f}" + ("*" if r["ic"] * ir < 0 else "")

            rows = [[r["label"], f"{r['ic']:+.3f}", _ir(r), r["category"]]
                    for _, r in top.iterrows()]
            blocks.append(B.block_table(
                ["因子", "IC", "IR", "类别"], rows,
                note="IC = 与未来 5 日收益的全样本秩相关；IR = 滚动 252 日 IC 的均值/标准差。"
                     "带 * 的 IR 与 IC 符号相反 —— 这个因子的方向在变，不能直接当信号"))
            feed.append("【因子 IC 前 10】" + "；".join(
                f"{r['label']} IC {r['ic']:+.3f} IR {_ir(r)}"
                for _, r in top.iterrows())
                + "。（IR 带 * 表示与 IC 符号相反，方向不稳）")

    # ---- 6. 策略对比
    try:
        from ttrade import backtest as BT

        rows, notes = [], []
        for key in COMPARE_STRATS:
            spec = S.STRATEGIES.get(key)
            if spec is None:
                continue
            sig = spec.func(df)
            res = BT.run_backtest(sig, label=spec.label)
            m = res.metrics or {}
            rows.append([spec.label, _pct(m.get("cagr")), _pct(m.get("max_dd")),
                         _num(m.get("sharpe")), _num(m.get("n_trades"), 0)])
            notes.append(f"{spec.label} 年化 {_pct(m.get('cagr'))}、"
                         f"最大回撤 {_pct(m.get('max_dd'))}、夏普 {_num(m.get('sharpe'))}")
        if rows:
            blocks.append(B.block_table(
                ["策略", "年化", "最大回撤", "夏普", "交易数"], rows,
                note="同一品种、同一区间、默认成本；买入持有是基准"))
            feed.append("【策略对比】" + "；".join(notes)
                        + "。（基准是买入持有，跑不赢它说明赚的是 beta 不是择时）")
    except Exception:  # noqa: BLE001
        pass

    # ---- 7. 宏观
    try:
        picks = ["macro_cn_ppi_yoy", "macro_cn_m1_yoy", "macro_cn_pmi_mfg",
                 "macro_us_y10", "macro_dxy", "macro_spx_vol20"]
        got = []
        for k in picks:
            s = F.raw(df, k).dropna()
            if len(s):
                got.append(f"{F.FACTORS[k].label} {float(s.iloc[-1]):.0%}")
        if got:
            feed.append("【宏观分位】" + "；".join(got)
                        + "。（0% = 历史最低，100% = 历史最高）")
    except Exception:  # noqa: BLE001
        pass

    # ---- 8. 板块
    try:
        sec = F.raw(df, "sector_mom_10").dropna()
        comm = F.raw(df, "commodity_mom_10").dropna()
        if len(sec):
            from ttrade import factors_ext as FE

            name = FE._SECTOR_OF.get(str(symbol).split("_")[0].upper(), "未知")
            feed.append(f"【板块】所属「{name}」，板块 10 日动量 {_pct(sec.iloc[-1])}，"
                        f"全市场商品指数 10 日动量 {_pct(comm.iloc[-1]) if len(comm) else '—'}。")
    except Exception:  # noqa: BLE001
        pass

    # ---- 9. 期权因子研判
    try:
        from ttrade.skills.option import build_brief

        brief = build_brief(symbol)
        if brief["has_data"]:
            blocks.extend(brief["blocks"])
            feed.append(brief["feed"])
    except Exception:  # noqa: BLE001
        pass

    charts = [k for k in RC.CHART_KINDS]
    feed.append(f"【可用图表】{'、'.join(charts)}。用 `chart(kind=..., symbol=...)` 出图，"
                "把返回的路径写成 markdown 的 `![说明](路径)` 插进报告。")
    return {"blocks": blocks, "feed": "\n\n".join(feed), "charts": charts}


def _report_pack(ctx, p: dict) -> SkillResult:
    symbol = str(p.get("symbol") or "").strip()
    if not symbol:
        raise RuntimeError("symbol 不能为空，比如 RB_SHFE")
    pack = build_pack(symbol)
    return SkillResult(
        True, summary=f"{symbol} 研报事实包（{len(pack['blocks'])} 组）",
        blocks=pack["blocks"], feed=pack["feed"],
    )


def _chart(ctx, p: dict) -> SkillResult:
    from ttrade import report_charts as RC

    kind = str(p.get("kind") or "").strip()
    symbol = str(p.get("symbol") or "").strip()
    if not symbol:
        raise RuntimeError("symbol 不能为空")
    if not kind:
        raise RuntimeError("kind 不能为空，可选：" + "、".join(RC.CHART_KINDS))
    path = RC.make(kind, symbol)
    label = RC.CHART_KINDS[kind][0]

    # `block_image` 只收 base64；这里给的是路径，渲染层会自己读文件转码
    # （`render._render_image` 支持 `src` 兜底）。
    img = B.block_image("", width=620, caption=f"{symbol} {label}")
    img["src"] = path
    blocks = [
        B.block_metrics(f"{symbol} {label}", [
            ("图表", kind, "flat"),
            ("文件", os.path.basename(path), "flat"),
        ]),
        img,
    ]
    feed = (f"已生成图表「{label}」，文件路径：\n{path}\n"
            f"把它插进报告就在对应位置单独写一行：\n![{symbol} {label}]({path})\n"
            "（图片必须独占一行，混在段落里排版会乱）")
    return SkillResult(True, summary=f"{symbol} {label} 已生成", blocks=blocks, feed=feed)


register(Tool(
    name="report_pack",
    group="doc",
    label="研报事实包",
    description="一次算全写研报要用的数字：季节性、基本面、因子 IC、策略对比、宏观分位",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["symbol"],
        "properties": {"symbol": {"type": "str"}},
    },
    handler=_report_pack,
    prompt=(
        "**研报事实包**：`report_pack(symbol=\"RB_SHFE\")`。\n"
        "写任何品种研报之前**先调它**，一次拿到季节性、库存仓单、基差、期限结构、"
        "因子 IC、策略对比、宏观分位。\n"
        "- 返回的数字都带口径（多少年、什么窗口、秩相关还是皮尔逊），"
        "写进报告时要原样带上口径\n"
        "- 里面没有的东西（产量、进口、政策）就是真的没有，别编，"
        "可以说「未接入」或者用 `web_search` 去找\n"
        "- 一次调用要跑几秒（要跑四条策略回测），同一个品种别重复调"
    ),
    max_turns=3,
))


register(Tool(
    name="chart",
    group="doc",
    label="出图",
    description="生成研报图表（季节性/大波段/因子 IC/期限结构/波动体制）",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "required": ["kind", "symbol"],
        "properties": {
            "kind": {"type": "str"},
            "symbol": {"type": "str"},
        },
    },
    handler=_chart,
    prompt=(
        "**出图**：`chart(kind=\"price_seasonality\", symbol=\"RB_SHFE\")`。\n"
        "可选 kind：`price_seasonality`（价格季节性）、`inventory_seasonality`"
        "（库存季节性）、`big_swings`（历史大波段）、`factor_ic`（因子 IC/IR）、"
        "`term_structure`（期限结构）、`vol_regime`（趋势震荡属性）。\n"
        "- 返回一个文件路径，**把路径写成独占一行的 `![说明](路径)`** 插进报告，"
        "导出 PDF 时会自动变成图片\n"
        "- 图里的数据是真实的，不要在正文里再编一遍不同的数字\n"
        "- 一张图讲一件事，别指望一张图把所有问题说完"
    ),
    max_turns=3,
))
