"""研报用的图表：季节性、历史波段、因子 IC、期限结构、波动体制。

## 为什么单独一个模块，不塞进 GUI

这些图有三个用处：研报 PDF、聊天里直接看、Agent 写报告时当依据。三处都要
同一张图，就不能把绘图代码写在任何一个界面里。所以这里是**纯函数**：给品种
代号，返回 PNG 路径；不碰 Qt，不碰界面状态。

## 每张图回答什么问题

| 函数 | 问题 |
|---|---|
| `price_seasonality` | 这个品种一年里哪几个月历史上偏强？ |
| `inventory_seasonality` | 库存/仓单的季节性拐点和价格对不对得上？ |
| `big_swings` | 历史上有哪几次大波段？每次涨跌多少、持续多久？ |
| `factor_ic` | 哪些因子在这个品种上真的有效（IC / IR）？ |
| `term_structure` | 现在的期限结构是陡还是平？和一年前比呢？ |
| `vol_regime` | 现在是趋势还是震荡？波动处在什么分位？ |

## 画图纪律

- **只用截至今天的真实数据**，不画拟合曲线、不补未来。研报里最忌讳「图上
  有个漂亮的预测线」，那是把模型的想法画成了事实。
- 季节性用「往年同月」统计，**当年不算**（和 `factors_seasonal` 一个口径）。
- 中文必须能显示：字体链走 `plotstyle`，缺字会抛异常而不是静默出方框。
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd

from ttrade.paths import user_dir

# 出图分辨率。144 dpi 在 A4 上印刷够用，文件也不至于几 MB
DPI = 144
FIG_W = 9.0
FIG_H = 4.2


def chart_dir() -> str:
    """图表落盘目录。和导出目录并列，方便一起打包/清理。"""
    path = os.path.join(user_dir(), "export", "charts")
    os.makedirs(path, exist_ok=True)
    return path


def _setup():
    from ttrade import plotstyle as PS

    PS.setup_matplotlib_fonts()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, name: str) -> str:
    path = os.path.join(chart_dir(), f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _style(ax, title: str = "", ylabel: str = "") -> None:
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=12, pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def _load(symbol: str, period: str = "1d") -> pd.DataFrame:
    """加载品种日线（带复权与衍生列）。取不到返回空表。"""
    from ttrade import futures as FU
    from ttrade.paths import data_dir

    sym, ex = _split(symbol)
    if not sym:
        return pd.DataFrame()
    try:
        df = FU.load_local(sym, ex, data_dir("cn"), adjust="qfq")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    return df if df is not None else pd.DataFrame()


def _split(symbol: str) -> tuple[str, str]:
    """`RB_SHFE` / `RB` / `rb888` → (`RB`, `SHFE`)。"""
    from ttrade import futures as FU

    raw = str(symbol or "").strip().upper().replace("888", "")
    if "_" in raw:
        sym, ex = raw.split("_", 1)
        return sym, ex
    for c in FU.get_universe(include_cffex=True):
        if c.symbol == raw:
            return raw, c.exchange
    return raw, ""


def _monthly_table(dc: pd.Series) -> pd.DataFrame:
    """年 × 月的收益率表。行是年份，列是月份。"""
    r = dc.pct_change()
    idx = pd.DatetimeIndex(dc.index)
    return r.groupby([idx.year, idx.month]).sum().unstack()


def _daily_close(df: pd.DataFrame) -> pd.Series:
    s = pd.to_numeric(df["close"], errors="coerce")
    s.index = pd.DatetimeIndex(df.index).normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


# ---------------------------------------------------------------- 价格季节性
def price_seasonality(symbol: str, years: int = 10) -> str:
    """月度季节性：过去 N 年同月平均收益（柱）+ 累计曲线（线）。

    只看**往年**，当年不计入 —— 当年那个月还没走完，混进去等于用半截数据
    去算「这个月的平均表现」。
    """
    plt = _setup()
    df = _load(symbol)
    if df.empty or len(df) < 300:
        raise RuntimeError(f"{symbol} 的历史不够（少于 300 根日线），画不了季节性")
    dc = _daily_close(df)
    tab = _monthly_table(dc).tail(years)

    mean = tab.mean(axis=0) * 100.0
    win = (tab > 0).mean(axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    months = list(range(1, 13))
    vals = [float(mean.get(m, np.nan)) for m in months]
    colors = ["#dc2626" if (v or 0) > 0 else "#16a34a" for v in vals]
    ax.bar(months, vals, color=colors, alpha=0.75, width=0.62, label="同月平均收益")
    ax.axhline(0, color="#6b7280", linewidth=0.9)
    ax2 = ax.twinx()
    ax2.plot(months, [float(win.get(m, np.nan)) for m in months],
             color="#1d4ed8", marker="o", markersize=4, linewidth=1.4, label="上涨概率")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("上涨概率（%）", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax.set_xticks(months)
    ax.set_xticklabels([f"{m}月" for m in months], fontsize=8)
    _style(ax, f"{symbol} 月度季节性（近 {len(tab)} 年，% ）", "同月平均收益（%）")
    fig.legend(loc="upper left", fontsize=8, frameon=False, bbox_to_anchor=(0.01, 0.99))
    return _save(fig, f"{symbol}_season_price")


# ---------------------------------------------------------------- 库存季节性
def inventory_seasonality(symbol: str, years: int = 6) -> str:
    """库存 / 仓单的月度季节性。

    用**月末值相对年初的变化率**，不是绝对值 —— 不同年份的库存水平差一个
    数量级（铜 10 万吨和 3 万吨），绝对值叠在一起什么也看不出来。
    """
    plt = _setup()
    from ttrade import factors_ext as FE

    sym = _split(symbol)[0]
    panel = FE._edb_pick(sym.lower()) if sym else {}
    series, label = None, ""
    for cat in ("inventory", "warehouse"):
        s = panel.get(cat)
        if s is not None and len(s) > 60:
            series, label = s, FE._CAT.get(cat, cat)
            break
    if series is None:
        raise RuntimeError(f"{sym} 没有库存/仓单数据（EDB 库里没有这个品种）")

    s = series.copy()
    s.index = pd.DatetimeIndex(s.index)
    monthly = s.resample("ME").last().dropna()
    if len(monthly) < 24:
        raise RuntimeError(f"{sym} 的{label}只有 {len(monthly)} 个月，不够画季节性")

    # 每年内部的月度变化率：以当年 1 月为 100
    rows = {}
    for y, g in monthly.groupby(monthly.index.year):
        if len(g) < 10:
            continue
        rows[y] = (g / g.iloc[0] - 1.0) * 100.0
    if not rows:
        raise RuntimeError(f"{sym} 的{label}按年分组后没有完整年度")
    tab = pd.DataFrame(rows).T
    tab.columns = [c.month for c in tab.columns]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    recent = tab.tail(years)
    for y, row in recent.iterrows():
        ax.plot(row.index, row.to_numpy(dtype=float), color="#94a3b8",
                linewidth=1.0, alpha=0.8)
        ax.annotate(str(y), (row.index[-1], row.iloc[-1]), fontsize=7,
                    color="#94a3b8", xytext=(3, 0), textcoords="offset points")
    avg = recent.mean(axis=0)
    ax.plot(avg.index, avg.to_numpy(dtype=float), color="#b91c1c",
            linewidth=2.2, label=f"{len(recent)} 年平均")
    ax.axhline(0, color="#6b7280", linewidth=0.9)
    ax.set_xticks(list(range(1, 13)))
    ax.set_xticklabels([f"{m}月" for m in range(1, 13)], fontsize=8)
    _style(ax, f"{symbol} {label}季节性（相对当年 1 月，%）", "相对年初变化（%）")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, f"{symbol}_season_inv")


# ---------------------------------------------------------------- 历史大波段
def big_swings(symbol: str, min_pct: float = 20.0, top: int = 8,
               max_days: int = 400) -> str:
    """历史大波段复盘：标出涨跌幅超过 `min_pct` 的连续段。

    做法是「ZigZag」：从低点涨够 `min_pct` 才算一段上涨的结束，反之亦然。
    这不是预测工具，只是把历史上真正的大行情标出来，方便对着基本面事件复盘。

    `max_days` 是个必要的过滤：不复权螺纹 2016 年低点到 2021 年高点是一段
    +482% 的「单边」，ZigZag 会把它算成**一个**波段 —— 数字没错，但对
    「复盘某次行情」毫无帮助。超过 `max_days` 的腿按「多年趋势」处理，不进图。
    """
    plt = _setup()
    df = _load(symbol)
    if df.empty or len(df) < 250:
        raise RuntimeError(f"{symbol} 的历史不够，画不了波段")
    dc = _daily_close(df)

    swings = [s for s in _zigzag(dc, min_pct / 100.0)
              if (s[2] - s[0]).days <= max_days]
    if not swings:
        raise RuntimeError(f"{symbol} 没有 {max_days} 天以内、幅度超过 {min_pct:.0f}% 的波段")
    swings = sorted(swings, key=lambda x: abs(x[3] / x[1] - 1.0), reverse=True)[:top]
    swings.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H + 0.6))
    ax.plot(dc.index, dc.to_numpy(dtype=float), color="#334155", linewidth=1.0)
    for d0, p0, d1, p1 in swings:
        up = p1 > p0
        ax.plot([d0, d1], [p0, p1], color="#dc2626" if up else "#16a34a",
                linewidth=2.4, solid_capstyle="round")
        mid = d0 + (d1 - d0) / 2
        days = (d1 - d0).days
        ax.annotate(f"{p1 / p0 - 1:+.0%} / {days}天", (mid, (p0 + p1) / 2),
                    fontsize=7.5, ha="center",
                    color="#991b1b" if up else "#166534",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.8))
    _style(ax, f"{symbol} 历史大波段（≥{min_pct:.0f}%、≤{max_days}天，前 {len(swings)} 段）",
           "价格")
    return _save(fig, f"{symbol}_swings")


def _zigzag(s: pd.Series, thr: float) -> list[tuple]:
    """极简 ZigZag：反向幅度超过 thr 才确认一个转折。"""
    out = []
    if len(s) < 3:
        return out
    px = s.to_numpy(dtype=float)
    idx = s.index
    lo_i = hi_i = 0
    direction = 0                     # 0 未定 / 1 上涨中 / -1 下跌中
    for i in range(1, len(px)):
        if direction >= 0:
            if px[i] > px[hi_i]:
                hi_i = i
            if px[i] <= px[hi_i] * (1.0 - thr):
                if direction == 1:
                    out.append((idx[lo_i], px[lo_i], idx[hi_i], px[hi_i]))
                direction = -1
                lo_i = i
        if direction <= 0:
            if px[i] < px[lo_i]:
                lo_i = i
            if px[i] >= px[lo_i] * (1.0 + thr):
                if direction == -1:
                    out.append((idx[hi_i], px[hi_i], idx[lo_i], px[lo_i]))
                direction = 1
                hi_i = i
    if direction == 1:
        out.append((idx[lo_i], px[lo_i], idx[hi_i], px[hi_i]))
    elif direction == -1:
        out.append((idx[hi_i], px[hi_i], idx[lo_i], px[lo_i]))
    return out


# ---------------------------------------------------------------- 因子 IC
# 研报里不放 124 条，只放各类目里最有代表性的这些 —— 全放上去一张图会糊成一片，
# 而且月频/季频的基本面因子在短样本上 IC 噪声极大
IC_KEYS = [
    "mom_20", "mom_60", "mom_120", "rev_5", "rev_20",
    "vol_20", "atr_pct", "maxdd_60",
    "vol_surge", "oi_change", "turnover_20", "log_amount_20",
    "roll_yield", "roll_annual", "term_near_far",
    "iv_atm", "iv_skew", "pcr",
    "season_avg", "season_rank",
    "war_chg_5", "spot_basis", "inv_chg_5",
    "sector_mom_10", "commodity_mom_10",
    "macro_cn_ppi_yoy", "macro_dxy", "macro_us_y10",
]


def factor_ic(symbol: str, horizon: int = 5, top: int = 18) -> str:
    """因子 IC / IR 横向条形图。

    IC = 因子值与未来 `horizon` 日收益的秩相关（滚动 252 日的均值）。
    IR = IC 均值 / IC 标准差，衡量「这个 IC 稳不稳」—— IC 0.06 但 IR 只有 0.2
    的因子，实盘基本不能用，因为它的方向随时会翻。
    """
    plt = _setup()
    from ttrade import factor_metrics as FM
    from ttrade import factors as F

    df = _load(symbol)
    if df.empty or len(df) < 400:
        raise RuntimeError(f"{symbol} 的历史不够（少于 400 根），算不出稳定的 IC")
    keys = F.filter_available(df, IC_KEYS, min_obs=120)
    if not keys:
        raise RuntimeError(f"{symbol} 上没有一个可用因子（缺持仓量/展期/期权数据？）")

    rep = FM.factor_report(df, keys, horizon=horizon, window=252)
    rep = rep[rep["ic"].notna()].copy()
    if rep.empty:
        raise RuntimeError("IC 全部为空，样本不足")
    rep["abs_ic"] = rep["ic"].abs()
    rep = rep.sort_values("abs_ic", ascending=False).head(top).sort_values("ic")

    fig, ax = plt.subplots(figsize=(FIG_W, max(3.0, 0.32 * len(rep) + 1.2)))
    colors = ["#dc2626" if v > 0 else "#16a34a" for v in rep["ic"]]
    ax.barh(rep["label"], rep["ic"].to_numpy(dtype=float), color=colors, alpha=0.8)
    for y, (v, ir) in enumerate(zip(rep["ic"], rep.get("ir", pd.Series(dtype=float)))):
        # 全样本 IC 和滚动 IR 是两种口径，符号相反很常见 —— 标个 `*`，
        # 不然读者会以为图算错了。
        flag = "*" if pd.notna(ir) and v * ir < 0 else ""
        ax.annotate(f"{v:+.3f}" + (f"  IR {ir:+.2f}{flag}" if pd.notna(ir) else ""),
                    (v, y), fontsize=7, va="center",
                    ha="left" if v >= 0 else "right",
                    xytext=(4 if v >= 0 else -4, 0), textcoords="offset points")
    ax.axvline(0, color="#6b7280", linewidth=0.9)
    # 给标注留位置。默认 xlim 只按数据范围加 5% 边距，负值那一侧的
    # 「-0.045 IR -0.37」会跑到坐标轴外面，压到左边的因子名上。
    span = float(rep["ic"].abs().max())
    pad = max(span * 0.55, 0.012)
    ax.set_xlim(-span - pad, span + pad)
    ax.set_xlabel(f"IC（未来 {horizon} 日，全样本秩相关）", fontsize=9)
    ax.text(0.0, -0.10,
            "IR = 滚动 252 日 IC 的均值/标准差；带 * 表示 IR 与 IC 符号相反，"
            "说明这个因子的方向在变，不能直接拿来做信号",
            transform=ax.transAxes, fontsize=7, color="#6b7280")
    _style(ax, f"{symbol} 因子 IC / IR（|IC| 前 {len(rep)}）")
    return _save(fig, f"{symbol}_factor_ic")


# ---------------------------------------------------------------- 期限结构
def term_structure(symbol: str) -> str:
    """当前期限结构曲线 vs 一年前。

    横轴是「到期日距今天数」，纵轴是价格相对最近月的比率 —— 不同年份的
    绝对价位差太多，归一化之后才看得出「曲线变陡了还是变平了」。
    """
    plt = _setup()
    df = _load(symbol)
    cols = [c for c in df.columns if c.endswith("_dte")]
    if not df.any().any() or not cols:
        raise RuntimeError(f"{symbol} 没有期限结构数据（aux 文件缺失）")
    pre = "o6_" if "o6_1_dte" in df.columns else ("o3_" if "o3_1_dte" in df.columns else "c")

    def curve_at(ts):
        sub = df.loc[:ts]
        if sub.empty:
            return None
        row = sub.iloc[-1]
        pts = []
        for i in (1, 2, 3):
            dte = row.get(f"{pre}{i}_dte" if pre != "c" else f"c{i}_dte")
            px = row.get(f"{pre}{i}_close" if pre != "c" else f"c{i}_close")
            if pd.notna(dte) and pd.notna(px):
                pts.append((float(dte), float(px)))
        if len(pts) < 2:
            return None
        base = pts[0][1]
        return [p[0] for p in pts], [p[1] / base - 1.0 for p in pts], sub.index[-1]

    now = curve_at(df.index[-1])
    old = curve_at(df.index[-1] - pd.Timedelta(days=365))
    if now is None:
        raise RuntimeError(f"{symbol} 当前没有可用的期限结构（合约太少）")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(now[0], np.array(now[1]) * 100, marker="o", color="#b91c1c",
            linewidth=2.0, label=f"当前（{now[2].date()}）")
    if old is not None:
        ax.plot(old[0], np.array(old[1]) * 100, marker="s", color="#64748b",
                linewidth=1.4, linestyle="--", label=f"一年前（{old[2].date()}）")
    ax.axhline(0, color="#6b7280", linewidth=0.9)
    ax.set_xlabel("距到期天数", fontsize=9)
    _style(ax, f"{symbol} 期限结构（相对最近月，%）", "相对最近月（%）")
    ax.legend(fontsize=8, frameon=False)
    return _save(fig, f"{symbol}_term")


# ---------------------------------------------------------------- 波动体制
def vol_regime(symbol: str) -> str:
    """趋势 / 震荡属性：滚动 60 日的效率比 + 波动率分位。

    上图：效率比（净位移/总路程）。>0.3 是趋势，<0.15 是震荡。
    下图：20 日年化波动率与它的 3 年分位 —— 分位高说明现在的波动处在历史高位。
    """
    plt = _setup()
    df = _load(symbol)
    if df.empty or len(df) < 300:
        raise RuntimeError(f"{symbol} 的历史不够，画不了波动体制")
    c = _daily_close(df)
    n = 60
    net = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n, min_periods=n).sum()
    eff = (net / path.replace(0.0, np.nan)).dropna()
    vol = c.pct_change().rolling(20, min_periods=20).std(ddof=1) * np.sqrt(252.0)
    pct = vol.expanding(min_periods=250).rank(pct=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, FIG_H + 1.6), sharex=True)
    ax1.plot(eff.index, eff.to_numpy(dtype=float), color="#1d4ed8", linewidth=0.9)
    ax1.axhline(0.3, color="#dc2626", linewidth=0.9, linestyle="--")
    ax1.axhline(0.15, color="#16a34a", linewidth=0.9, linestyle="--")
    ax1.fill_between(eff.index, 0.3, eff.to_numpy(dtype=float),
                     where=eff.to_numpy(dtype=float) > 0.3,
                     color="#dc2626", alpha=0.15)
    _style(ax1, f"{symbol} 趋势 / 震荡属性（{n} 日效率比）", "效率比")
    ax1.annotate("0.30 以上＝趋势", (eff.index[-1], 0.3), fontsize=7,
                 color="#dc2626", ha="right", va="bottom")

    ax2.plot(vol.index, (vol * 100).to_numpy(dtype=float), color="#7c3aed",
             linewidth=0.9, label="20 日年化波动（%）")
    _style(ax2, "", "年化波动（%）")
    ax3 = ax2.twinx()
    ax3.plot(pct.index, (pct * 100).to_numpy(dtype=float), color="#f59e0b",
             linewidth=0.9, alpha=0.85, label="历史分位（%）")
    ax3.set_ylim(0, 100)
    ax3.set_ylabel("历史分位（%）", fontsize=9)
    ax3.tick_params(labelsize=8)
    fig.legend(loc="lower left", fontsize=8, frameon=False, ncol=2)
    return _save(fig, f"{symbol}_regime")


# ---------------------------------------------------------------- 汇总
CHART_KINDS = {
    "price_seasonality": ("价格季节性", price_seasonality),
    "inventory_seasonality": ("库存季节性", inventory_seasonality),
    "big_swings": ("历史大波段", big_swings),
    "factor_ic": ("因子 IC/IR", factor_ic),
    "term_structure": ("期限结构", term_structure),
    "vol_regime": ("趋势震荡属性", vol_regime),
}


def make(kind: str, symbol: str) -> str:
    """按名字出一张图，返回 PNG 路径。"""
    item = CHART_KINDS.get(kind)
    if item is None:
        raise KeyError(f"未知图表 {kind}，可选：{'、'.join(CHART_KINDS)}")
    return item[1](symbol)
