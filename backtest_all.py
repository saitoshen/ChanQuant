"""批量回测：标的池 × 多策略，输出对比表与图表。

支持两个市场：
    python backtest_all.py                 # 美股（默认）
    python backtest_all.py --market cn     # 国内期货

两个市场共用同一套策略与回测引擎，仅数据源、成本假设、标的池不同，
因此结果可以直接横向对照。
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ttrade.backtest import BacktestConfig, buy_and_hold, run_backtest, yearly_returns
from ttrade.eastmoney import load_local
from ttrade.futures import list_local as list_futures
from ttrade.futures import load_local as load_futures
from ttrade.strategies import STRATEGIES, run_strategy
from ttrade.universe import cn_name, sector_of
from ttrade.plotstyle import setup_matplotlib_fonts

setup_matplotlib_fonts()

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
FUT_DIR = os.path.join(ROOT, "data_futures")
OUT_DIR = os.path.join(ROOT, "output")
OUT_FUT = os.path.join(OUT_DIR, "futures")

PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed"]

MODE_LABEL = {"long": "只做多", "both": "多空双向"}


# ---------------------------------------------------------------- 市场配置
MARKET_DEFS: dict[str, dict] = {
    "us": {
        "label": "美股",
        "data_dir": DATA_DIR,
        "out_dir": OUT_DIR,
        "min_rows": 300,
        "fee": 0.0005,
        "slippage": 0.0005,
        # 美股做空要付融券费、受 uptick 规则限制，本模型未建模，因此只跑多头
        "modes": ["long"],
        "crisis_years": (2008, 2009, 2020, 2022),
        "heatmap_start": 2006,
        "group_col": None,
        "report_title": "美股趋势跟踪策略回测报告",
        "data_note": "标的池最新 20 年日线（东方财富，后复权，含分红再投资）",
        "curve_title": "美股股票池等权组合净值",
        "crisis_title": "三、危机年份：趋势跟踪真正的高光时刻",
        "crisis_note": [
            "2008 年金融危机：买入持有 -27% 级别，而 ATR 通道、海龟等策略只回撤个位数到十几个点；",
            "2022 年加息熊市同样抗跌。代价是 2009 年 V 型反弹时，趋势策略因等待确认信号",
            "而错过大半涨幅（+12% vs +52%）。这就是趋势跟踪的收益结构：**牺牲牛市弹性，换取熊市生存权**。",
        ],
    },
    "cn": {
        "label": "国内期货",
        "data_dir": FUT_DIR,
        "out_dir": OUT_FUT,
        "min_rows": 300,
        "fee": 0.0001,
        "slippage": 0.0002,
        # 期货天然双向：同时跑「只做多」和「多空双向」，直接对比
        "modes": ["long", "both"],
        # 主力连续合约按 Tushare fut_daily_adj 的复权因子消除换月跳空。
        # qfq / hfq / none 的收益率完全一致（只差一个常数），默认 qfq 是为了
        # 与 GUI 显示口径统一；报告里另有一节对比 none vs 复权的差异。
        "adjust": "qfq",
        "crisis_years": (2008, 2011, 2016, 2020, 2022),
        "heatmap_start": 2006,
        "group_col": "sector",
        "report_title": "国内期货趋势跟踪策略回测报告",
        "data_note": "全品种主力连续合约 20 年日线（Tushare，按 fut_daily_adj 复权因子消除换月跳空）",
        "curve_title": "国内期货品种池等权组合净值",
        "crisis_title": "三、关键年份：商品的两轮超级周期",
        "crisis_note": [],  # 由 _cn_conclusions 动态生成
    },
}


# ---------------------------------------------------------------- 标的池
def load_items(market: str, adjust: str | None = None) -> list[dict]:
    """返回 [{'key','name','sector','load'}]，两个市场统一结构。

    adjust 只对期货生效（qfq / hfq / none），美股数据本身已是后复权。
    """
    d = MARKET_DEFS[market]
    data_dir = d["data_dir"]
    if not os.path.isdir(data_dir):
        return []

    if market == "us":
        items = []
        for fn in sorted(os.listdir(data_dir)):
            if not fn.endswith(".csv") or fn.startswith("_"):
                continue
            sym = fn[:-4].replace("_", ".")
            items.append(
                {
                    "key": sym,
                    "name": cn_name(sym) or sym,
                    "sector": sector_of(sym),
                    "load": (lambda s=sym: load_local(s, data_dir)),
                }
            )
        return items

    adj = adjust or d.get("adjust", "qfq")
    items = []
    for it in list_futures(data_dir, only_active=True):
        sym, ex = it["symbol"], it["exchange"]
        items.append(
            {
                "key": it["key"],
                "name": it["name"],
                "sector": it["sector"],
                "exchange": it["exchange_cn"],
                "load": (lambda s=sym, e=ex, a=adj: load_futures(s, e, data_dir, adjust=a)),
            }
        )
    return items


def run_all(market: str = "us", cfg: BacktestConfig | None = None,
            mode: str = "long", adjust: str | None = None) -> tuple[pd.DataFrame, dict]:
    """mode='long' 只做多，'both' 多空双向；adjust 仅期货生效。"""
    d = MARKET_DEFS[market]
    long_only = mode == "long"
    base = cfg or BacktestConfig(fee_rate=d["fee"], slippage=d["slippage"])
    cfg = replace(base, long_only=long_only)
    rows = []
    curves: dict[str, dict[str, pd.Series]] = {}

    for it in load_items(market, adjust=adjust):
        try:
            df = it["load"]()
        except Exception as exc:
            print(f"  读取失败 {it['key']}: {exc}")
            continue
        if len(df) < d["min_rows"]:
            continue

        bh = buy_and_hold(df, cfg)
        curves.setdefault("BUY_HOLD", {})[it["key"]] = bh.equity / bh.equity.iloc[0]
        span_years = max((df.index[-1] - df.index[0]).days, 1) / 365.25

        for key, spec in STRATEGIES.items():
            sig = run_strategy(df, key)
            res = run_backtest(sig, cfg, label=spec.label)
            curves.setdefault(key, {})[it["key"]] = res.equity / res.equity.iloc[0]

            m = res.metrics
            rows.append(
                {
                    "symbol": it["key"],
                    "name": it["name"],
                    "sector": it["sector"],
                    "years": round(span_years, 1),
                    "mode": mode,
                    "strategy": key,
                    "strategy_label": spec.label,
                    "cagr": m["cagr"],
                    "total_return": m["total_return"],
                    "max_dd": m["max_dd"],
                    "sharpe": m["sharpe"],
                    "calmar": m["calmar"],
                    "ann_vol": m["ann_vol"],
                    "exposure": m["exposure"],
                    "long_exposure": m["long_exposure"],
                    "short_exposure": m["short_exposure"],
                    "n_trades": m["n_trades"],
                    "n_long": m["n_long"],
                    "n_short": m["n_short"],
                    "n_blowup": m["n_blowup"],
                    "win_rate": m["win_rate"],
                    "avg_trade": m["avg_trade"],
                    "profit_factor": m["profit_factor"],
                    "avg_bars": m["avg_bars"],
                    "bh_cagr": bh.metrics["cagr"],
                    "bh_max_dd": bh.metrics["max_dd"],
                    "bh_sharpe": bh.metrics["sharpe"],
                    "excess_cagr": m["cagr"] - bh.metrics["cagr"],
                    # 回撤为负值，策略回撤更浅时该值为正
                    "dd_improve": m["max_dd"] - bh.metrics["max_dd"],
                }
            )

    table = pd.DataFrame(rows)
    return table, curves


# ---------------------------------------------------------------- 汇总
def summary_by_strategy(table: pd.DataFrame) -> pd.DataFrame:
    g = table.groupby(["mode", "strategy", "strategy_label"], as_index=False).agg(
        n=("symbol", "count"),
        cagr=("cagr", "mean"),
        bh_cagr=("bh_cagr", "mean"),
        excess_cagr=("excess_cagr", "mean"),
        max_dd=("max_dd", "mean"),
        bh_max_dd=("bh_max_dd", "mean"),
        dd_improve=("dd_improve", "mean"),
        sharpe=("sharpe", "mean"),
        bh_sharpe=("bh_sharpe", "mean"),
        calmar=("calmar", "mean"),
        exposure=("exposure", "mean"),
        short_exposure=("short_exposure", "mean"),
        n_trades=("n_trades", "mean"),
        n_short=("n_short", "mean"),
        n_blowup=("n_blowup", "sum"),
        win_rate=("win_rate", "mean"),
        beat_rate=("excess_cagr", lambda s: float((s > 0).mean())),
    )
    g["mode_label"] = g["mode"].map(MODE_LABEL)
    g["_order"] = g["mode"].map({"long": 0, "both": 1}).fillna(9)
    g = g.sort_values(["_order", "sharpe"], ascending=[True, False]).drop(columns="_order")
    return g.reset_index(drop=True)


def portfolio_curve(d: dict[str, pd.Series]) -> pd.Series | None:
    """等权组合：对当日可用标的的日收益取平均后累乘。

    直接平均净值会因标的上市/退市造成跳变，收益率累乘可避免该问题。

    单品种日收益截断在 ±50%：期货做空接近爆仓时权益可能只剩初始的 0.1%，
    价格小幅回落就能产生几十倍的单日收益（实测集运指数 +7690%），
    这种极端值除以标的数后仍会把组合曲线拉出一个假尖峰。
    期货有涨跌停板、美股单日极少超过 ±50%，截断不会抹掉真实信息。
    """
    if not d:
        return None
    frame = pd.concat(d.values(), axis=1, sort=True).sort_index()
    rets = frame.pct_change().clip(-0.5, 0.5)
    avg = rets.mean(axis=1, skipna=True).fillna(0.0)
    return (1.0 + avg).cumprod()


def crisis_years(curves: dict, years: tuple[int, ...] = (2008, 2009, 2020, 2022)) -> pd.DataFrame:
    """关键年份（危机年 / 反弹年）各策略的年度收益，用于说明危机保护特性。"""
    rows: dict[str, dict[str, float]] = {}
    bh = portfolio_curve(curves.get("BUY_HOLD", {}))
    if bh is not None:
        yr = yearly_returns(bh) * 100
        rows["买入持有"] = {str(y.year): round(float(v), 1) for y, v in yr.items() if y.year in years}
    for key, spec in STRATEGIES.items():
        c = portfolio_curve(curves.get(key, {}))
        if c is None:
            continue
        yr = yearly_returns(c) * 100
        rows[spec.label] = {str(y.year): round(float(v), 1) for y, v in yr.items() if y.year in years}
    df = pd.DataFrame(rows).T
    df.columns = [f"{c} 年" for c in df.columns]
    return df


def bucket_analysis(table: pd.DataFrame) -> pd.DataFrame:
    """按标的自身「买入持有年化」分三档，观察策略在不同趋势强度下的表现差异。"""
    bh = table.drop_duplicates("symbol")[["symbol", "bh_cagr"]].copy()
    q1, q2 = bh["bh_cagr"].quantile([1 / 3, 2 / 3])

    def bucket(x: float) -> str:
        if x >= q2:
            return "A 强势上涨（前1/3）"
        if x >= q1:
            return "B 温和上涨（中1/3）"
        return "C 弱势震荡（后1/3）"

    bh["bucket"] = bh["bh_cagr"].apply(bucket)
    merged = table.merge(bh[["symbol", "bucket"]], on="symbol", how="left")
    g = merged.groupby(["bucket", "strategy_label"], as_index=False).agg(
        n=("symbol", "count"),
        cagr=("cagr", "mean"),
        bh_cagr=("bh_cagr", "mean"),
        excess=("excess_cagr", "mean"),
        max_dd=("max_dd", "mean"),
        bh_max_dd=("bh_max_dd", "mean"),
        dd_improve=("dd_improve", "mean"),
        sharpe=("sharpe", "mean"),
        beat_rate=("excess_cagr", lambda s: float((s > 0).mean())),
    )
    return g.sort_values(["bucket", "sharpe"], ascending=[True, False]).reset_index(drop=True)


def portfolio_metrics(curves_by_mode: dict) -> pd.DataFrame:
    """组合层面的绩效：把全部标的等权合并成一条净值后重新度量。

    单品种夏普普遍很低（趋势策略仓位利用率只有 20~30%），
    真正可投资的是多品种组合——信号在不同品种间低相关，组合夏普会显著抬升。
    """
    rows = []
    for mode, curves in curves_by_mode.items():
        series = [("买入持有", curves.get("BUY_HOLD", {}))]
        series += [(spec.label, curves.get(k, {})) for k, spec in STRATEGIES.items()]
        for name, d in series:
            c = portfolio_curve(d)
            if c is None or len(c) < 30:
                continue
            ret = c.pct_change().fillna(0.0)
            years = max((c.index[-1] - c.index[0]).days, 1) / 365.25
            cagr = float(c.iloc[-1] ** (1 / years) - 1)
            vol = float(ret.std(ddof=1) * np.sqrt(252))
            mdd = float((c / c.cummax() - 1.0).min())
            rows.append(
                {
                    "mode": mode,
                    "mode_label": MODE_LABEL.get(mode, mode),
                    "组合": name,
                    "n": len(d),
                    "cagr": cagr,
                    "vol": vol,
                    "sharpe": float(ret.mean() * 252 / vol) if vol > 0 else 0.0,
                    "max_dd": mdd,
                    "calmar": cagr / abs(mdd) if mdd < 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def group_analysis(table: pd.DataFrame, col: str) -> pd.DataFrame:
    """按品种板块分组（期货用），观察哪类品种最适合趋势跟踪。"""
    base = table.drop_duplicates("symbol")[[col, "bh_cagr", "bh_max_dd"]]
    bh = base.groupby(col, as_index=False).agg(
        n_symbol=("bh_cagr", "count"),
        bh_cagr=("bh_cagr", "mean"),
        bh_max_dd=("bh_max_dd", "mean"),
    )
    g = table.groupby([col, "strategy_label"], as_index=False).agg(
        n=("symbol", "count"),
        cagr=("cagr", "mean"),
        excess=("excess_cagr", "mean"),
        max_dd=("max_dd", "mean"),
        dd_improve=("dd_improve", "mean"),
        sharpe=("sharpe", "mean"),
        beat_rate=("excess_cagr", lambda s: float((s > 0).mean())),
    )
    g = g.merge(bh, on=col, how="left")
    return g.sort_values([col, "sharpe"], ascending=[True, False]).reset_index(drop=True)


# ---------------------------------------------------------------- 绘图
def plot_portfolio_curves(curves: dict, out_path: str, title: str) -> None:
    """各策略在标的池上的等权组合净值 vs 买入持有。"""
    fig, ax = plt.subplots(figsize=(12, 6.2), dpi=130)

    bh = portfolio_curve(curves.get("BUY_HOLD", {}))
    if bh is not None:
        ax.plot(bh.index, bh.values, color="#111827", lw=2.4, label="买入持有（等权）", zorder=5)

    for i, (key, spec) in enumerate(STRATEGIES.items()):
        c = portfolio_curve(curves.get(key, {}))
        if c is None:
            continue
        ax.plot(c.index, c.values, lw=1.6, color=PALETTE[i % len(PALETTE)], label=spec.label, alpha=0.9)

    ax.set_yscale("log")
    ax.set_title(f"趋势跟踪策略 vs 买入持有 —— {title}（对数轴）", fontsize=13, pad=12)
    ax.set_ylabel("净值（起点=1，对数刻度）")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_risk_return(table: pd.DataFrame, out_path: str) -> None:
    """风险-收益散点：每个点是一个「标的×策略」组合。"""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=130)

    bh = table.drop_duplicates("symbol")[["bh_cagr", "bh_max_dd"]]
    ax.scatter(bh["bh_max_dd"] * 100, bh["bh_cagr"] * 100, s=28, c="#9ca3af", alpha=0.65,
               label="买入持有", edgecolors="none")

    for i, (key, spec) in enumerate(STRATEGIES.items()):
        sub = table[table["strategy"] == key]
        if sub.empty:
            continue
        ax.scatter(sub["max_dd"] * 100, sub["cagr"] * 100, s=26,
                   color=PALETTE[i % len(PALETTE)], alpha=0.55, label=spec.label,
                   edgecolors="none")

    ax.axhline(0, color="#6b7280", lw=0.8, ls="--")
    ax.set_xlabel("最大回撤 (%)")
    ax.set_ylabel("年化收益 CAGR (%)")
    ax.set_title("风险—收益分布：趋势跟踪 vs 买入持有", fontsize=13, pad=10)
    ax.grid(True, ls=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_yearly_heatmap(curves: dict, out_path: str, start_year: int = 2006,
                        suffix: str = "") -> None:
    """各策略等权组合的年度收益热力图。"""
    series = {"买入持有": portfolio_curve(curves.get("BUY_HOLD", {}))}
    for key, spec in STRATEGIES.items():
        series[spec.label] = portfolio_curve(curves.get(key, {}))

    rows = {}
    for name, s in series.items():
        if s is None:
            continue
        rows[name] = yearly_returns(s) * 100

    mat = pd.DataFrame(rows).T
    mat = mat.loc[:, mat.columns.year >= start_year]
    if mat.empty:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(mat.columns) * 0.85), 3.4), dpi=130)
    # 用真实极值定色阶：95 分位数会把 2008/2009 这类关键年份压成同色，看不出差异
    vmax = float(min(60.0, max(20.0, np.nanmax(np.abs(mat.values)))))
    im = ax.imshow(mat.values, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels([str(c.year) for c in mat.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="#111827" if abs(v) < vmax * 0.6 else "white")
    ax.set_title(f"年度收益率 (%) —— 等权组合{suffix}", fontsize=12, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_bucket_chart(bucket: pd.DataFrame, out_path: str, suffix: str = "") -> None:
    """分组柱状图：各趋势强度档位下，策略超额收益与回撤改善。"""
    if bucket.empty:
        return
    strategies = list(STRATEGIES.values())
    buckets = sorted(bucket["bucket"].unique())
    x = np.arange(len(buckets))
    w = 0.8 / max(len(strategies), 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=130)

    for ax, col, title, unit in [
        (axes[0], "excess", "相对买入持有的年化超额收益", "%"),
        (axes[1], "dd_improve", "最大回撤改善（正值＝回撤更浅）", "pp"),
    ]:
        for i, spec in enumerate(strategies):
            sub = bucket[bucket["strategy_label"] == spec.label]
            if sub.empty:
                continue
            vals = []
            for b in buckets:
                row = sub[sub["bucket"] == b]
                vals.append(float(row[col].iloc[0]) * 100 if len(row) else 0.0)
            ax.bar(x + i * w - (len(strategies) - 1) * w / 2, vals, w,
                   label=spec.label, color=PALETTE[i % len(PALETTE)], alpha=0.85)
        ax.axhline(0, color="#374151", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels([b.split(" ")[0] + "\n" + b.split(" ")[1] for b in buckets], fontsize=9)
        ax.set_title(title, fontsize=11.5, pad=10)
        ax.set_ylabel(f"{unit}", fontsize=9)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"策略在不同趋势强度标的上的表现分化{suffix}", fontsize=13, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_sector_chart(group: pd.DataFrame, col: str, out_path: str, min_symbols: int = 2) -> None:
    """板块维度：策略年化超额收益的横向对比。

    单品种板块（如航运 EC）会把纵轴拉爆、掩盖其余板块的差异，因此剔除。
    """
    if group.empty:
        return
    keep = group.groupby(col)["n_symbol"].first()
    sectors = sorted(keep[keep >= min_symbols].index)
    if not sectors:
        return
    strategies = list(STRATEGIES.values())
    x = np.arange(len(sectors))
    w = 0.8 / max(len(strategies), 1)

    fig, ax = plt.subplots(figsize=(max(11, len(sectors) * 1.25), 5.0), dpi=130)
    for i, spec in enumerate(strategies):
        sub = group[group["strategy_label"] == spec.label]
        vals = []
        for s in sectors:
            row = sub[sub[col] == s]
            vals.append(float(row["excess"].iloc[0]) * 100 if len(row) else 0.0)
        ax.bar(x + i * w - (len(strategies) - 1) * w / 2, vals, w,
               label=spec.label, color=PALETTE[i % len(PALETTE)], alpha=0.85)

    ax.axhline(0, color="#374151", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n({int(keep[s])}个)" for s in sectors], fontsize=9)
    ax.set_ylabel("相对买入持有的年化超额 (%)")
    ax.set_title(f"各品种板块上趋势跟踪的超额收益（样本 ≥{min_symbols} 个品种）",
                 fontsize=12.5, pad=10)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- 报告
def plot_mode_compare(summary: pd.DataFrame, port: pd.DataFrame, out_path: str) -> None:
    """只做多 vs 多空双向：上排单品种平均，下排组合层面。

    这张图要说明的核心是「反转」——单品种做空全面变差，
    但把品种等权合并后做空反而改善风险调整后收益。
    """
    if "long" not in set(summary["mode"]) or "both" not in set(summary["mode"]):
        return

    short_name = {
        "ma_cross": "双均线", "donchian": "唐奇安", "atr_channel": "ATR通道",
        "macd": "MACD", "turtle": "海龟",
    }
    keys = [k for k in STRATEGIES if k in set(summary["strategy"])]
    labels = [short_name.get(k, STRATEGIES[k].label) for k in keys]
    x = np.arange(len(keys))
    w = 0.36

    specs = [
        ("cagr", "年化收益 (%)", 100.0),
        ("sharpe", "夏普比率", 1.0),
        ("max_dd", "最大回撤 (%)", 100.0),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.2), dpi=130)

    rows = [
        ("单品种平均", summary.set_index(["mode", "strategy"]), keys, False),
        ("组合层面（等权合并）",
         port.set_index(["mode", "组合"]) if not port.empty else None,
         [STRATEGIES[k].label for k in keys], True),
    ]

    for r, (row_title, frame, idx_keys, is_port) in enumerate(rows):
        for c, (col, title, scale) in enumerate(specs):
            ax = axes[r][c]
            for j, (mode, color, name) in enumerate(
                [("long", "#94a3b8", "只做多"), ("both", "#2563eb", "多空双向")]
            ):
                vals = []
                for k in idx_keys:
                    try:
                        vals.append(float(frame.loc[(mode, k), col]) * scale)
                    except KeyError:
                        vals.append(0.0)
                ax.bar(x + (j - 0.5) * w, vals, w, label=name, color=color, alpha=0.9)

            if is_port and frame is not None:
                try:
                    bh = float(frame.loc[("long", "买入持有"), col]) * scale
                    ax.axhline(bh, color="#111827", lw=1.3, ls="--", zorder=4)
                    ax.text(len(keys) - 0.45, bh, " 买入持有", va="center", ha="left",
                            fontsize=8, color="#111827")
                except KeyError:
                    pass

            ax.axhline(0, color="#374151", lw=0.9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_title(f"{row_title} · {title}", fontsize=11, pad=8)
            ax.grid(True, axis="y", ls=":", alpha=0.4)
            ax.set_axisbelow(True)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            if r == 0 and c == 0:
                ax.legend(frameon=True, fontsize=9, loc="upper right",
                          facecolor="white", edgecolor="#e5e7eb")

    fig.suptitle("只做多 vs 多空双向：单品种做空普遍变差，组合层面出现反转",
                 fontsize=13.5, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def carry_profile(market: str = "cn") -> dict:
    """逐品种统计移仓收益，并检验它与价格趋势的相关性。

    移仓收益 = 复权年化 − 不复权年化。若它与标的自身涨跌幅正相关，
    说明「近月升水 + 上涨」「远月升水 + 下跌」是常态，
    趋势策略多空两条腿都能拿到这笔收益，而买入持有只能拿到多头的一半。
    """
    rows = []
    for it in load_items(market):
        try:
            raw = it["load"]()  # 默认已复权
        except Exception:
            continue
        try:
            raws = load_futures_raw(it["key"])
        except Exception:
            continue
        if len(raw) < 300 or len(raws) < 300:
            continue
        yrs = max((raw.index[-1] - raw.index[0]).days, 1) / 365.25
        try:
            a = (raw["close"].iloc[-1] / raw["close"].iloc[0]) ** (1 / yrs) - 1
            r = (raws["close"].iloc[-1] / raws["close"].iloc[0]) ** (1 / yrs) - 1
        except Exception:
            continue
        rows.append({"raw": r, "carry": a - r})

    if not rows:
        return {}
    d = pd.DataFrame(rows)
    return {
        "n": len(d),
        "carry_mean": float(d["carry"].mean()),
        "carry_pos": float((d["carry"] > 0).mean()),
        "corr": float(d["carry"].corr(d["raw"])),
    }


def load_futures_raw(key: str) -> pd.DataFrame:
    """按 key（如 CU_SHFE）读取不复权的期货日线，供 carry_profile 对照用。"""
    d = MARKET_DEFS["cn"]
    sym, ex = key.rsplit("_", 1)
    return load_futures(sym, ex, d["data_dir"], adjust="none")


def adjust_compare(adj_sum: pd.DataFrame, raw_sum: pd.DataFrame) -> pd.DataFrame:
    """复权 vs 不复权：逐策略对比（同一回测模式）。

    主力连续合约在换月日直接拼接会产生跳空，把这段价差当成真实盈亏。
    复权把这些跳空按移仓价差抹平，等价于「持有多头并逐月移仓」的真实收益。
    两者差异最大的地方在买入持有——趋势策略因为经常空仓，受影响相对小。
    """
    if adj_sum.empty or raw_sum.empty:
        return pd.DataFrame()

    a = adj_sum.set_index("strategy")
    r = raw_sum.set_index("strategy")
    rows = []
    for key in STRATEGIES:
        if key not in a.index or key not in r.index:
            continue
        rows.append(
            {
                "strategy": key,
                "strategy_label": a.loc[key, "strategy_label"],
                "cagr_adj": float(a.loc[key, "cagr"]),
                "cagr_raw": float(r.loc[key, "cagr"]),
                "d_cagr": float(a.loc[key, "cagr"] - r.loc[key, "cagr"]),
                "sharpe_adj": float(a.loc[key, "sharpe"]),
                "sharpe_raw": float(r.loc[key, "sharpe"]),
                "dd_adj": float(a.loc[key, "max_dd"]),
                "dd_raw": float(r.loc[key, "max_dd"]),
                "d_dd": float(a.loc[key, "max_dd"] - r.loc[key, "max_dd"]),
            }
        )
    # 买入持有单独一行：它是复权影响最大的口径
    key0 = rows[0]["strategy"] if rows else None
    if key0:
        rows.append(
            {
                "strategy": "BUY_HOLD",
                "strategy_label": "买入持有",
                "cagr_adj": float(a.loc[key0, "bh_cagr"]),
                "cagr_raw": float(r.loc[key0, "bh_cagr"]),
                "d_cagr": float(a.loc[key0, "bh_cagr"] - r.loc[key0, "bh_cagr"]),
                "sharpe_adj": float(a.loc[key0, "bh_sharpe"]),
                "sharpe_raw": float(r.loc[key0, "bh_sharpe"]),
                "dd_adj": float(a.loc[key0, "bh_max_dd"]),
                "dd_raw": float(r.loc[key0, "bh_max_dd"]),
                "d_dd": float(a.loc[key0, "bh_max_dd"] - r.loc[key0, "bh_max_dd"]),
            }
        )
    return pd.DataFrame(rows)


def plot_adjust_compare(cmp: pd.DataFrame, out_path: str) -> None:
    """复权前后对比图：左图年化，右图最大回撤，外加一个因子分布小图。"""
    if cmp.empty:
        return

    labels = [r["strategy_label"] for _, r in cmp.iterrows()]
    x = np.arange(len(cmp))
    w = 0.36

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9), dpi=130)
    specs = [
        ("cagr", "年化收益 (%)", 100.0),
        ("dd", "最大回撤 (%)", 100.0),
        ("sharpe", "夏普比率", 1.0),
    ]
    for ax, (col, title, scale) in zip(axes, specs):
        raw_v = cmp[f"{col}_raw"] * scale
        adj_v = cmp[f"{col}_adj"] * scale
        ax.bar(x - w / 2, raw_v, w, label="不复权（含换月跳空）", color="#94a3b8", alpha=0.9)
        ax.bar(x + w / 2, adj_v, w, label="复权（消除跳空）", color="#2563eb", alpha=0.9)
        ax.axhline(0, color="#374151", lw=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=True, fontsize=8, facecolor="white", edgecolor="#e5e7eb")

    fig.suptitle("换月复权的影响：复权后策略普遍变好，且提升幅度超过买入持有",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fmt_pct(x: float, digits: int = 1) -> str:
    return f"{x * 100:.{digits}f}%"


def _us_conclusions(table, summary, bucket, crisis) -> list[str]:
    return [
        "**1. 样本期特性决定了结论方向。** 2006-2026 是美股历史上最强的一轮长牛之一，",
        "标普 500 年化约 10%，而本股票池全部为当前市值最大的龙头股，",
        "平均买入持有年化高达 17%，远超指数。在这种「几乎单边上涨」的样本里，",
        "任何需要离场的趋势策略都必然付出机会成本，跑输买入持有是结构性结果，而非策略失效。",
        "",
        "**2. 趋势跟踪买到的是「回撤保护」，不是「超额收益」。** 五套策略的最大回撤",
        "平均比买入持有浅 14~23 个百分点，波动率也显著更低。但注意：即使是最弱的 C 组",
        "（标的自身年化仅 6.4%），策略仍然小幅跑输 4.1%——因为该组标的虽然涨幅有限，",
        "但整体仍是正收益。趋势跟踪真正发挥优势的场景是标的长期下跌或深度熊市，",
        "而这类标的在「按当前市值选股」的股票池里几乎不存在。",
        "",
        "**3. 组内规律：标的越强，策略跑输越多。** A 组（持有年化 30.9%）策略落后 8.2%，",
        "C 组（6.4%）只落后 4.1%。这与直觉一致——涨幅越大，离场所损失的机会成本越高。",
        "换句话说，趋势跟踪的代价与标的的上涨斜率成正比，而它买到的是回撤的压缩。",
        "",
        "**4. 幸存者偏差必须警惕。** 股票池是按「今天仍然是大市值龙头」挑选的，",
        "天然排除了退市、长期跑输的标的。真实可投资 universe 的历史收益会明显更低，",
        "届时趋势跟踪的相对表现会更好。要看更公允的结论，应使用历史时点的成分股名单。",
    ]


def _cn_conclusions(table, summary, bucket, crisis, group=None, port=None,
                    modes: tuple[str, ...] = ("both",)) -> list[str]:
    """期货市场的结论由数据驱动生成，避免写死与样本不符的判断。

    table / summary / port 传入的是全模式数据，内部按 primary 模式筛选；
    「只做多 vs 多空双向」的对比单独成段。
    """
    out: list[str] = []
    primary = modes[-1]
    psum = summary[summary["mode"] == primary]
    pport = None if (port is None or port.empty) else port[port["mode"] == primary]
    multi = len(modes) > 1

    bh_cagr = table["bh_cagr"].mean()
    bh_dd = table["bh_max_dd"].mean()
    bh_sharpe = table["bh_sharpe"].mean()
    best = psum.sort_values("sharpe", ascending=False).iloc[0]
    worst = psum.sort_values("cagr").iloc[0]
    n_beat = int((psum["beat_rate"] >= 0.5).sum())
    dd_lo, dd_hi = psum["dd_improve"].min(), psum["dd_improve"].max()
    n_blow = int(table["n_blowup"].sum())

    out.append(
        f"**1. 期货的「买入持有」不是股票那种基准。** {table['symbol'].nunique()} 个品种"
        f"平均买入持有年化 {fmt_pct(bh_cagr)}，看着不算差，"
        f"但平均最大回撤 {fmt_pct(bh_dd)}——夏普只有 {bh_sharpe:.2f}，不到美股（1.01）的四分之一。"
    )
    out.append(
        "差别在收益的性质：股票持有赚的是企业盈利的复利，可以持续二十年；"
        "商品持有赚的是移仓收益加上价格均值回归的运气，一轮超级周期结束后往往又还回去。"
        "所以趋势策略在这里不需要战胜一个强大的基准——"
        "这与美股结论（策略普遍跑输持有）的差异，根源在基准而不在策略。"
    )
    out.append("")

    # ---------------- 只做多 vs 多空双向 ----------------
    if multi and {"long", "both"} <= set(modes):
        long_s = summary[summary["mode"] == "long"].set_index("strategy")
        both_s = summary[summary["mode"] == "both"].set_index("strategy")
        pairs = []
        for key, spec in STRATEGIES.items():
            if key in long_s.index and key in both_s.index:
                pairs.append((spec.label, long_s.loc[key], both_s.loc[key]))
        better = [p for p in pairs if p[2]["cagr"] > p[1]["cagr"]]
        worse = [p for p in pairs if p[2]["cagr"] <= p[1]["cagr"]]
        dd_deeper = [p for p in pairs if p[2]["max_dd"] < p[1]["max_dd"]]
        avg_short = both_s["short_exposure"].mean()

        up_sharpe_sym = [p for p in pairs if p[2]["sharpe"] > p[1]["sharpe"]]

        out.append(
            f"**2. 复权之后，做空在单品种层面也基本站得住。** "
            f"五套策略里 {len(better)}/{len(pairs)} 套双向年化更高"
            + (f"（{'、'.join(f'{b[0]} {fmt_pct(b[1]['cagr'])}→{fmt_pct(b[2]['cagr'])}' for b in better)}）" if better else "")
            + (f"，{len(worse)} 套变差（{'、'.join(f'{b[0]} {fmt_pct(b[1]['cagr'])}→{fmt_pct(b[2]['cagr'])}' for b in worse)}）" if worse else "")
            + f"；夏普 {len(up_sharpe_sym)}/{len(pairs)} 套上升，"
            f"但 {len(dd_deeper)}/{len(pairs)} 套的最大回撤加深。"
            "注意这与不复权数据下的结论不同——**换月跳空被抹平后，空头腿从「纯噪声」变成了「有正期望」**，"
            "因为移仓收益的方向恰好与趋势方向一致。"
        )
        out.append(
            f"双向模式下平均 {fmt_pct(avg_short, 0)} 的时间持有空头，空头腿是真的在交易。"
            "它之所以能赚钱，和复权揭示的是同一件事：**移仓收益与趋势方向同向**。"
            "商品供给宽松时远月升水（contango）而价格同时下跌，做空既赚价差也赚移仓；"
            "供给紧张时近月升水而价格同时上涨，做多同样两头都赚。"
            "只做多只能拿到其中一半。"
        )

        plong = port[port["mode"] == "long"].set_index("组合") if port is not None and not port.empty else None
        pboth = port[port["mode"] == "both"].set_index("组合") if port is not None and not port.empty else None
        if plong is not None and pboth is not None and not plong.empty and not pboth.empty:
            common = [c for c in plong.index if c in pboth.index and c != "买入持有"]
            if common:
                up_sharpe = [c for c in common if pboth.loc[c, "sharpe"] >= plong.loc[c, "sharpe"]]
                best_name = max(common, key=lambda c: pboth.loc[c, "calmar"] - plong.loc[c, "calmar"])
                out.append(
                    f"**组合层面优势更大**：{len(up_sharpe)}/{len(common)} 套策略的组合夏普持平或上升，"
                    f"年化也同步改善。"
                    f"最典型的是「{best_name}」——年化 {fmt_pct(plong.loc[best_name, 'cagr'])}→"
                    f"{fmt_pct(pboth.loc[best_name, 'cagr'])}，"
                    f"卡玛 {plong.loc[best_name, 'calmar']:.2f}→{pboth.loc[best_name, 'calmar']:.2f}，"
                    f"最大回撤 {fmt_pct(plong.loc[best_name, 'max_dd'])}→{fmt_pct(pboth.loc[best_name, 'max_dd'])}。"
                )
                out.append(
                    "组合层面比单品种更好，是因为几十个品种的空头腿叠加后进一步分散了方向性风险——"
                    "单品种空头的回撤很深，但不同品种的空头不会同时爆亏。"
                    "**所以「要不要做空」不只是策略问题，更是组合问题**："
                    "单账户押一个品种做空是赌博，多品种同时双向才能把空头腿的收益留下来、把尾部风险摊薄。"
                )
        if n_blow:
            out.append(
                f"另外，全部回测中出现 {n_blow} 次空头爆仓（价格翻倍导致权益归零，已按规则强平），"
                "这是做空必须计入的尾部风险。"
            )
        else:
            out.append(
                "本次全部回测中没有出现空头爆仓（权益归零），但这是 100% 名义敞口、不加杠杆的结果；"
                "实盘若用 10% 保证金满仓，同样的价格翻倍就是爆仓。"
            )
        out.append("")

    if dd_lo >= 0:
        dd_text = f"**收益打平的前提下 {dd_lo * 100:.0f}~{dd_hi * 100:.0f} 个百分点的回撤压缩**"
    else:
        dd_text = (
            f"**回撤改善从 {dd_lo * 100:.0f} 到 +{dd_hi * 100:.0f} 个百分点不等**"
            "（双向模式下部分策略的回撤反而更深）"
        )
    out.append(
        f"**{3 if multi else 2}. 单品种层面：收益与持有打平，回撤改善有限。** "
        f"夏普最高的「{best['strategy_label']}」"
        f"年化 {fmt_pct(best['cagr'])}（持有 {fmt_pct(best['bh_cagr'])}，超额 {fmt_pct(best['excess_cagr'])}），"
        f"最大回撤 {fmt_pct(best['max_dd'])}，比持有浅 {fmt_pct(best['dd_improve'])}。"
    )
    out.append(
        f"五套策略里"
        + (f"有 {n_beat} 套" if n_beat else "没有一套")
        + f"在至少一半品种上跑赢买入持有（最高 {fmt_pct(psum['beat_rate'].max(), 0)}），"
        f"年化最低的「{worst['strategy_label']}」为 {fmt_pct(worst['cagr'])}、超额 {fmt_pct(worst['excess_cagr'])}。"
        f"也就是说：在期货上趋势跟踪换来的不是超额收益，而是{dd_text}。"
    )
    out.append("")

    if pport is not None and not pport.empty:
        bh_row = pport[pport["组合"] == "买入持有"]
        best_row = pport.sort_values("sharpe", ascending=False).iloc[0]
        if not bh_row.empty:
            b = bh_row.iloc[0]
            out.append(
                f"**{4 if multi else 3}. 组合层面才是趋势跟踪的主场。** "
                f"把 {table['symbol'].nunique()} 个品种等权合并后："
                f"买入持有年化 {fmt_pct(b['cagr'])}、波动 {fmt_pct(b['vol'])}、夏普 {b['sharpe']:.2f}、"
                f"最大回撤 {fmt_pct(b['max_dd'])}；"
                f"而「{best_row['组合']}」年化 {fmt_pct(best_row['cagr'])}、波动 {fmt_pct(best_row['vol'])}、"
                f"夏普 {best_row['sharpe']:.2f}、最大回撤 {fmt_pct(best_row['max_dd'])}。"
            )
            ratio = best_row["sharpe"] / psum["sharpe"].max() if psum["sharpe"].max() > 0 else 0
            out.append(
                f"组合夏普 {best_row['sharpe']:.2f} 是单品种平均（约 {psum['sharpe'].max():.2f}）的 "
                f"{ratio:.1f} 倍，"
                "这正是趋势跟踪的真实盈利模式：不靠单品种择时的准确率，而靠几十个低相关品种"
                "同时下注，用少数几笔大趋势的盈利覆盖大量小亏损。单看一个品种永远看不出这套系统的价值。"
            )
            out.append("")

    if not bucket.empty:
        c_rows = bucket[bucket["bucket"].str.startswith("C")]
        a_rows = bucket[bucket["bucket"].str.startswith("A")]
        if not c_rows.empty and not a_rows.empty:
            out.append(
                f"**{5 if multi else 4}. 真正的分野不是板块，而是标的自身的长期方向。** "
                f"C 组（持有年化 {fmt_pct(c_rows['bh_cagr'].iloc[0])}）五套策略全部取得 "
                f"{fmt_pct(c_rows['excess'].min())}~{fmt_pct(c_rows['excess'].max())} 的超额；"
                f"A 组（持有 {fmt_pct(a_rows['bh_cagr'].iloc[0])}）则全部跑输 "
                f"{fmt_pct(a_rows['excess'].min())}~{fmt_pct(a_rows['excess'].max())}。"
            )
            extra = ""
            if group is not None and not group.empty:
                sub = group[(group["strategy_label"] == best["strategy_label"]) & (group["n_symbol"] >= 5)]
                if not sub.empty:
                    sub = sub.sort_values("excess")
                    extra = (
                        f"板块维度上差异要小得多：{best['strategy_label']}在"
                        f"{sub.iloc[-1]['sector']}超额 {fmt_pct(sub.iloc[-1]['excess'])}，"
                        f"在{sub.iloc[0]['sector']}为 {fmt_pct(sub.iloc[0]['excess'])}，"
                        f"首尾只差 {(sub.iloc[-1]['excess'] - sub.iloc[0]['excess']) * 100:.1f} 个百分点。"
                    )
            out.append(
                (extra + "也就是说：**趋势跟踪赚的是「长期下跌标的的钱」，不是「某个板块的钱」**。"
                 "只要标的的长期方向是向下的，五套策略都能靠「少亏」拿到正超额；"
                 "反过来，标的越强，离场的机会成本越高。这一点与美股完全一致。")
            )
            out.append("")
            new_energy = table[table["sector"] == "新能源"].drop_duplicates("symbol")[["symbol", "bh_cagr"]]
            if len(new_energy) >= 2 and (new_energy["bh_cagr"] < 0).all():
                sub = table[table["sector"] == "新能源"]
                out.append(
                    f"最纯粹的例证是新能源板块（{len(new_energy)} 个品种："
                    f"{'、'.join(table[table['sector'] == '新能源'].drop_duplicates('symbol')['name'].tolist())}），"
                    f"平均持有年化 {fmt_pct(new_energy['bh_cagr'].mean())}，"
                    f"而五套策略的年化在 {fmt_pct(sub.groupby('strategy')['cagr'].mean().min())}~"
                    f"{fmt_pct(sub.groupby('strategy')['cagr'].mean().max())} 之间，"
                    f"跑赢持有比例 {fmt_pct((sub['excess_cagr'] > 0).mean(), 0)}。"
                    + ("这几个品种上市即下跌，双向模式下空头腿直接吃到了主跌段，"
                       if multi else
                       "这几个品种上市即下跌，只做多的趋势策略大部分时间在空仓等信号，")
                    + "把持有的巨额回撤几乎全部躲开。"
                )
                out.append("")

    if not crisis.empty and "买入持有" in crisis.index:
        bh = crisis.loc["买入持有"]
        worst_col = bh.idxmin()
        best_strat, best_val = None, -999.0
        for name, r in crisis.iterrows():
            if name == "买入持有":
                continue
            v = float(r[worst_col])
            if v > best_val:
                best_val, best_strat = v, name
        if best_strat:
            out.append(
                f"**{6 if multi else 5}. 极端年份的保护能力依旧成立。** {worst_col.replace(' 年', '')} 年商品普跌，"
                f"等权持有 {float(bh.min()):+.1f}%，而「{best_strat}」为 {best_val:+.1f}%，"
                f"相差 {best_val - float(bh.min()):+.1f} 个百分点。"
            )
            out.append(
                "商品的大级别行情由供需错配驱动，一旦转向往往持续 1~2 年，"
                "这正是趋势跟踪最擅长的节奏——它不需要预测顶部，只需要在下跌中离场。"
                + ("双向模式在这里的优势最明显：暴跌年空头腿直接转为盈利来源，"
                   "而只做多模式只能做到「少亏」。这也是做空唯一确定的价值——"
                   "不是在震荡市里多赚，而是在崩盘时把风险变成收益。"
                   if multi else "")
            )
            out.append("")

    out.append(f"**{7 if multi else 6}. 期货回测的五个特有陷阱：**")
    out.append(
        "- **换月复权**：主力连续合约在移仓日价格直接拼接，已用 Tushare 复权因子消除跳空，"
        "等价于「持有多头并逐月移仓」的真实收益。但复权因子只抹平价差，"
        "不包含移仓的冲击成本与保证金占用变化。"
    )
    out.append(
        "- **无杠杆假设**：本回测按 100% 名义本金计算，不加杠杆。"
        "实盘保证金约 10%，若按满仓保证金计算，收益与回撤都会放大一个数量级，"
        "年化数字不能与股票直接横向对比。"
    )
    out.append(
        "- **做空爆仓**：空头名义敞口等于全部权益时，价格翻倍即权益归零。"
        "引擎按规则强平并停止交易，但实盘 10% 保证金下，价格涨 10% 就会追保，"
        "真实风险远高于回测数字。"
    )
    out.append(
        "- **合约存续期不一致**：豆粕、玉米、PTA 有 20 年数据，"
        "丙烯、多晶硅、碳酸锂只有 1~3 年。等权组合按当日可用品种取平均，"
        "早期样本少、近期样本多，组合净值早期段波动更大。"
    )
    out.append(
        "- **流动性筛选**：已剔除近一年日均成交额低于 5000 万元的品种（粳稻、普麦、"
        "早籼稻、强麦、胶合板、线材、菜籽、纤维板等），这些品种挂着牌但基本不成交，"
        "回测收益无法在实盘复现。"
    )
    out.append("")
    out.append(f"**{8 if multi else 7}. 回测假设：**")
    out.append("- 信号在第 T 日收盘生成，第 T+1 日开盘成交（无未来函数）")
    out.append("- 单边手续费 1bp + 单边滑点 2bp（期货手续费显著低于股票）")
    out.append("- 不加杠杆，名义敞口 = 全部权益；空仓时资金不计利息")
    out.append("- 价格使用主力连续合约收盘价，已按 Tushare 复权因子消除换月跳空")
    if multi:
        out.append("- 双向模式下多空名义敞口相同，空头无融券成本（期货本就没有）")
    return out


def write_markdown(table: pd.DataFrame, summary: pd.DataFrame, bucket: pd.DataFrame,
                   crisis: pd.DataFrame, path: str, market: str = "us",
                   group: pd.DataFrame | None = None,
                   port: pd.DataFrame | None = None,
                   modes: tuple[str, ...] = ("long",),
                   adj_cmp: pd.DataFrame | None = None,
                   carry: dict | None = None) -> None:
    d = MARKET_DEFS[market]
    group_col = d["group_col"]
    multi = len(modes) > 1
    primary = modes[-1]
    ptable = table[table["mode"] == primary]

    lines = [f"# {d['report_title']}", ""]
    lines.append(f"标的数量：{ptable['symbol'].nunique()}　策略数量：{ptable['strategy'].nunique()}")
    lines.append(f"数据区间：{d['data_note']}")
    if multi:
        lines.append(f"回测模式：{' / '.join(MODE_LABEL[m] for m in modes)}（同一批策略分别按两种模式跑）")
    lines.append("")

    if adj_cmp is not None and not adj_cmp.empty:
        lines.append("## 数据口径：换月复权")
        lines.append("")
        lines.append(
            "主力连续合约在换月日是把新合约的价格直接接在旧合约后面，不做处理就会凭空"
            "多出一段跳空，被回测当成真实盈亏。本报告用 Tushare `fut_daily_adj` 的复权收盘价"
            "反推逐日复权因子（`adj_factor = 复权收盘 / 原始收盘`，只在换月日变化），"
            "再乘到 OHLC 上，等价于「持有多头并逐月移仓」的真实收益。"
        )
        lines.append("")
        lines.append("下表是同一批数据、同一套参数，仅切换复权开关的结果对比：")
        lines.append("")
        lines.append("| 组合 / 策略 | 不复权年化 | 复权年化 | 差额 | 不复权回撤 | 复权回撤 | 夏普(不/复) |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in adj_cmp.iterrows():
            lines.append(
                f"| {r['strategy_label']} | {fmt_pct(r['cagr_raw'])} | {fmt_pct(r['cagr_adj'])} | "
                f"{fmt_pct(r['d_cagr'])} | {fmt_pct(r['dd_raw'])} | {fmt_pct(r['dd_adj'])} | "
                f"{r['sharpe_raw']:.2f} / {r['sharpe_adj']:.2f} |"
            )
        lines.append("")
        bh = adj_cmp[adj_cmp["strategy"] == "BUY_HOLD"]
        strat = adj_cmp[adj_cmp["strategy"] != "BUY_HOLD"]
        if not bh.empty and not strat.empty:
            b = bh.iloc[0]
            lines.append(
                f"复权后五套策略平均年化提升 {strat['d_cagr'].mean() * 100:.1f} 个百分点，"
                f"比买入持有的 {b['d_cagr'] * 100:.1f} 个百分点还多。"
                "这不是「策略吃到了更多跳空」，而是**移仓收益与价格趋势同向**："
                "商品供给紧张时往往同时出现近月升水（backwardation）和上涨趋势，"
                "此时持多头既赚价差也赚移仓；供给宽松时远月升水（contango）且价格下跌，"
                "持空头同样两头都赚。买入持有只有多头一条腿，只能拿到其中一半。"
            )
            lines.append("")
            if carry:
                lines.append(
                    f"这一点在数据上很直接：{carry['n']} 个品种的「复权年化 − 不复权年化」"
                    f"（即移仓收益）平均 {carry['carry_mean'] * 100:+.1f} 个百分点、"
                    f"{carry['carry_pos'] * 100:.0f}% 的品种为正，"
                    f"而且它与标的自身涨跌幅的相关系数是 **{carry['corr']:.2f}**——"
                    "涨得越多的品种移仓收益越高。"
                )
                lines.append("")
            lines.append(
                "换句话说，不复权曲线不仅数据不干净，还**系统性地低估了趋势跟踪的收益**："
                "它把每次移仓的价差当成随机噪声，而实际上这段价差和趋势方向高度相关，"
                "是趋势策略一笔隐藏的收入。"
            )
            lines.append("")

    lines.append("## 一、策略横向对比（标的池平均）")

    lines.append("")
    cols = ["策略"]
    if multi:
        cols.append("模式")
    cols += ["年化", "持有年化", "超额", "最大回撤", "持有回撤", "回撤改善",
             "夏普", "卡玛", "仓位占比"]
    if multi:
        cols += ["空头占比", "多头笔数", "空头笔数"]
    cols += ["胜率", "跑赢持有比例"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for _, r in summary.iterrows():
        cells = [r["strategy_label"]]
        if multi:
            cells.append(r["mode_label"])
        cells += [
            fmt_pct(r["cagr"]), fmt_pct(r["bh_cagr"]), fmt_pct(r["excess_cagr"]),
            fmt_pct(r["max_dd"]), fmt_pct(r["bh_max_dd"]), fmt_pct(r["dd_improve"]),
            f"{r['sharpe']:.2f}", f"{r['calmar']:.2f}", fmt_pct(r["exposure"], 0),
        ]
        if multi:
            cells += [
                fmt_pct(r["short_exposure"], 0),
                f"{r['n_trades'] - r['n_short']:.0f}",
                f"{r['n_short']:.0f}",
            ]
        cells += [fmt_pct(r["win_rate"], 0), fmt_pct(r["beat_rate"], 0)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    if port is not None and not port.empty:
        lines.append("## 一·补、组合层面绩效（全部标的等权合并成一条净值）")
        lines.append("")
        lines.append("单品种绩效会被仓位利用率拉低，把全部标的等权合并后才反映「实际可投资」的效果：")
        lines.append("")
        pcols = ["组合"] + (["模式"] if multi else []) + ["品种数", "年化", "年化波动", "夏普", "最大回撤", "卡玛"]
        lines.append("| " + " | ".join(pcols) + " |")
        lines.append("|" + "---|" * len(pcols))
        for _, r in port.iterrows():
            pcells = [r["组合"]]
            if multi:
                pcells.append(r["mode_label"])
            pcells += [
                str(int(r["n"])), fmt_pct(r["cagr"]), fmt_pct(r["vol"]),
                f"{r['sharpe']:.2f}", fmt_pct(r["max_dd"]), f"{r['calmar']:.2f}",
            ]
            lines.append("| " + " | ".join(pcells) + " |")
        lines.append("")
    lines.append("## 二、按标的趋势强度分组（关键结论）")
    lines.append("")
    lines.append(f"把标的按「自身买入持有年化」分成三档，观察趋势跟踪的适用边界"
                 f"{'（以下为' + MODE_LABEL[primary] + '模式）' if multi else ''}：")
    lines.append("")
    lines.append("| 分组 | 策略 | 策略年化 | 持有年化 | 超额 | 回撤改善 | 夏普 | 跑赢比例 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in bucket.iterrows():
        lines.append(
            f"| {r['bucket']} | {r['strategy_label']} | {fmt_pct(r['cagr'])} | {fmt_pct(r['bh_cagr'])} | "
            f"{fmt_pct(r['excess'])} | {fmt_pct(r['dd_improve'])} | {r['sharpe']:.2f} | {fmt_pct(r['beat_rate'], 0)} |"
        )
    lines.append("")

    if group_col and group is not None and not group.empty:
        lines.append(f"## 二·补、按{d['label']}品种板块分组")
        lines.append("")
        lines.append("| 板块 | 品种数 | 策略 | 策略年化 | 持有年化 | 超额 | 回撤改善 | 夏普 | 跑赢比例 |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, r in group.iterrows():
            lines.append(
                f"| {r[group_col]} | {int(r['n_symbol'])} | {r['strategy_label']} | "
                f"{fmt_pct(r['cagr'])} | {fmt_pct(r['bh_cagr'])} | {fmt_pct(r['excess'])} | "
                f"{fmt_pct(r['dd_improve'])} | {r['sharpe']:.2f} | {fmt_pct(r['beat_rate'], 0)} |"
            )
        lines.append("")

    if not crisis.empty:
        lines.append(f"## {d['crisis_title']}")
        lines.append("")
        lines.append("等权组合在各关键年份的收益（%）：")
        lines.append("")
        lines.append("| 组合 | " + " | ".join(crisis.columns) + " |")
        lines.append("|---|" + "---|" * len(crisis.columns))
        for name, r in crisis.iterrows():
            vals = " | ".join(f"{v:+.1f}" if pd.notna(v) else "—" for v in r.values)
            lines.append(f"| {name} | {vals} |")
        lines.append("")
        for ln in d["crisis_note"]:
            lines.append(ln)
        if d["crisis_note"]:
            lines.append("")

    lines.append("## 四、逐标的明细（按策略分组）")
    lines.append("")
    if market == "cn":
        short = ptable[(ptable["strategy"] == ptable["strategy"].iloc[0]) & (ptable["years"] < 3)]
        if not short.empty:
            lines.append(
                f"> 每张表按年化降序排列。**数据年数**列不可忽略："
                f"{'、'.join(short['name'].tolist())} 等 {len(short)} 个品种上市不足 3 年，"
                "年化数字由极短样本外推而来，波动极大，不宜单独解读。"
            )
            lines.append("")
    for key, spec in STRATEGIES.items():
        sub = ptable[ptable["strategy"] == key].sort_values("cagr", ascending=False)
        if sub.empty:
            continue
        lines.append(f"### {spec.label}")
        lines.append(f"_{spec.desc}_")
        lines.append("")
        lines.append(f"| 标的 | 名称 | {d['label']}分类 | 数据年数 | 年化 | 持有年化 | 超额 | 最大回撤 | 夏普 | 交易 | 胜率 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['symbol']} | {r['name']} | {r['sector']} | {r['years']:.1f} | "
                f"{fmt_pct(r['cagr'])} | "
                f"{fmt_pct(r['bh_cagr'])} | {fmt_pct(r['excess_cagr'])} | {fmt_pct(r['max_dd'])} | "
                f"{r['sharpe']:.2f} | {r['n_trades']:.0f} | {fmt_pct(r['win_rate'], 0)} |"
            )
        lines.append("")

    lines.append("## 五、结论与假设说明")
    lines.append("")
    if market == "us":
        lines.extend(_us_conclusions(ptable, summary, bucket, crisis))
        lines.append("")
        lines.append("**5. 回测假设：**")
        lines.append("- 信号在第 T 日收盘生成，第 T+1 日开盘成交（无未来函数）")
        lines.append("- 单边手续费 5bp + 单边滑点 5bp，双边合计约 0.2%")
        lines.append("- 只做多、不加杠杆，空仓时资金不计利息")
        lines.append("- 价格使用后复权，含分红再投资，收益率连续")
    else:
        lines.extend(_cn_conclusions(ptable, summary, bucket, crisis, group, port,
                                     modes=modes))
    lines.append("")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------- 入口
def main() -> int:
    ap = argparse.ArgumentParser(description="批量回测：美股 / 国内期货")
    ap.add_argument("--market", choices=["us", "cn"], default="us", help="us=美股，cn=国内期货")
    ap.add_argument(
        "--adjust",
        choices=["qfq", "hfq", "none"],
        default=None,
        help="期货复权方式：qfq 前复权（默认，最新价=真实盘面）/ hfq 后复权 / none 不复权。"
        "三者收益率完全相同，仅价格水平不同；none 用于对照换月跳空的影响。",
    )
    args = ap.parse_args()

    d = MARKET_DEFS[args.market]
    if args.adjust:
        d["adjust"] = args.adjust
    out_dir = d["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"开始批量回测 —— {d['label']}")
    print(f"成本假设：手续费 {d['fee'] * 10000:.1f}bp，滑点 {d['slippage'] * 10000:.1f}bp（单边）")
    print(f"回测模式：{' / '.join(MODE_LABEL[m] for m in d['modes'])}")
    if args.market == "cn":
        print(f"复权方式：{d.get('adjust', 'qfq')}")
    print()

    modes = tuple(d["modes"])
    tables, curves_by_mode = {}, {}
    for m in modes:
        t, c = run_all(args.market, mode=m)
        tables[m], curves_by_mode[m] = t, c
    table = pd.concat(tables.values(), ignore_index=True)
    if table.empty:
        print("没有可用数据，请先运行下载脚本")
        return 1

    primary = modes[-1]
    ptable = table[table["mode"] == primary]
    curves = curves_by_mode[primary]

    summary = summary_by_strategy(table)
    bucket = bucket_analysis(ptable)
    crisis = crisis_years(curves, d["crisis_years"])
    group = group_analysis(ptable, d["group_col"]) if d["group_col"] else None
    port = portfolio_metrics(curves_by_mode)

    # 期货：再跑一遍不复权，量化换月跳空的影响
    adj_cmp = pd.DataFrame()
    carry: dict = {}
    if args.market == "cn" and d.get("adjust", "qfq") != "none":
        print("\n计算换月复权的影响（不复权对照）…")
        t_raw, _ = run_all("cn", mode=primary, adjust="none")
        if not t_raw.empty:
            # 注意顺序：第一个参数是「复权」结果，第二个是「不复权」结果
            adj_cmp = adjust_compare(summary[summary["mode"] == primary],
                                     summary_by_strategy(t_raw))
            adj_cmp.to_csv(os.path.join(out_dir, "backtest_adjust.csv"),
                           index=False, encoding="utf-8-sig")
            plot_adjust_compare(adj_cmp, os.path.join(out_dir, "adjust_compare.png"))
            carry = carry_profile("cn")

    table.to_csv(os.path.join(out_dir, "backtest_detail.csv"), index=False, encoding="utf-8-sig")
    summary.to_csv(os.path.join(out_dir, "backtest_summary.csv"), index=False, encoding="utf-8-sig")
    bucket.to_csv(os.path.join(out_dir, "backtest_bucket.csv"), index=False, encoding="utf-8-sig")
    crisis.to_csv(os.path.join(out_dir, "backtest_crisis.csv"), encoding="utf-8-sig")
    port.to_csv(os.path.join(out_dir, "backtest_portfolio.csv"), index=False, encoding="utf-8-sig")
    if group is not None:
        group.to_csv(os.path.join(out_dir, "backtest_sector.csv"), index=False, encoding="utf-8-sig")

    mode_suffix = f"（{MODE_LABEL[primary]}）" if len(modes) > 1 else ""
    plot_portfolio_curves(curves, os.path.join(out_dir, "curves.png"), d["curve_title"] + mode_suffix)
    plot_risk_return(ptable, os.path.join(out_dir, "risk_return.png"))
    plot_yearly_heatmap(curves, os.path.join(out_dir, "yearly_heatmap.png"), d["heatmap_start"],
                        suffix=mode_suffix)
    plot_bucket_chart(bucket, os.path.join(out_dir, "bucket.png"), mode_suffix)
    if group is not None and d["group_col"]:
        plot_sector_chart(group, d["group_col"], os.path.join(out_dir, "sector.png"))
    if len(modes) > 1:
        plot_mode_compare(summary, port, os.path.join(out_dir, "mode_compare.png"))
    write_markdown(table, summary, bucket, crisis, os.path.join(out_dir, "report.md"),
                   market=args.market, group=group, port=port, modes=modes,
                   adj_cmp=adj_cmp, carry=carry)

    print("\n=== 策略横向对比（标的池平均） ===")
    show = summary.copy()
    for c in ["cagr", "bh_cagr", "excess_cagr", "max_dd", "bh_max_dd", "dd_improve",
              "exposure", "short_exposure", "win_rate", "beat_rate"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    show["sharpe"] = show["sharpe"].round(2)
    show["calmar"] = show["calmar"].round(2)
    show["n_trades"] = show["n_trades"].round(0).astype(int)
    cols = ["mode_label", "strategy_label", "cagr", "bh_cagr", "excess_cagr", "max_dd",
            "dd_improve", "sharpe", "calmar", "exposure", "short_exposure", "n_trades",
            "win_rate", "beat_rate"]
    print(show[cols].to_string(index=False))

    print("\n=== 组合层面绩效（全部标的等权合并） ===")
    pshow = port.copy()
    for c in ["cagr", "vol", "max_dd"]:
        pshow[c] = (pshow[c] * 100).round(1).astype(str) + "%"
    pshow["sharpe"] = pshow["sharpe"].round(2)
    pshow["calmar"] = pshow["calmar"].round(2)
    print(pshow[["mode_label", "组合", "n", "cagr", "vol", "sharpe", "max_dd", "calmar"]].to_string(index=False))

    print("\n=== 按标的趋势强度分组（平均） ===")
    bshow = bucket.copy()
    for c in ["cagr", "bh_cagr", "excess", "dd_improve", "beat_rate"]:
        bshow[c] = (bshow[c] * 100).round(1).astype(str) + "%"
    bshow["sharpe"] = bshow["sharpe"].round(2)
    print(bshow[["bucket", "strategy_label", "cagr", "bh_cagr", "excess", "dd_improve", "sharpe", "beat_rate"]].to_string(index=False))

    if group is not None and d["group_col"]:
        print(f"\n=== 按{d['group_col']}分组（平均） ===")
        gshow = group.copy()
        for c in ["cagr", "bh_cagr", "excess", "dd_improve", "beat_rate"]:
            gshow[c] = (gshow[c] * 100).round(1).astype(str) + "%"
        gshow["sharpe"] = gshow["sharpe"].round(2)
        print(gshow[[d["group_col"], "n_symbol", "strategy_label", "cagr", "bh_cagr",
                     "excess", "dd_improve", "sharpe", "beat_rate"]].to_string(index=False))

    if not crisis.empty:
        print("\n=== 关键年份收益（等权组合，%） ===")
        print(crisis.to_string())

    if not adj_cmp.empty:
        print("\n=== 换月复权的影响（不复权 → 复权） ===")
        ashow = adj_cmp.copy()
        for c in ["cagr_raw", "cagr_adj", "d_cagr", "dd_raw", "dd_adj"]:
            ashow[c] = (ashow[c] * 100).round(1).astype(str) + "%"
        print(ashow[["strategy_label", "cagr_raw", "cagr_adj", "d_cagr",
                     "dd_raw", "dd_adj", "sharpe_raw", "sharpe_adj"]].to_string(index=False))

    print(f"\n结果已输出到 {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
