"""跑「A股·放量启动」的全市场组合回测。

用法：

    python run_stock_screen.py                      # 默认近 5 年，PE<25 与 PE<30 各跑一遍
    python run_stock_screen.py --start 20211101 --end 20261107 --pe 25
    python run_stock_screen.py --hold 5 --max-pos 10

产出：控制台指标 + `reports/stock_screen_*.md` + 净值图 PNG + 逐笔 CSV。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ttrade import stockrule as SR  # noqa: E402

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")


def pct(x: float) -> str:
    return "—" if not np.isfinite(x if x is not None else np.nan) else f"{x * 100:.2f}%"


def edge(panels: dict, cfg: SR.ScreenConfig, horizons=(1, 3, 5, 10, 20)) -> pd.DataFrame:
    """**纯选股超额**：被选中那批票的远期收益 − 当天全市场等权收益。

    组合净值只有一个数，分不清两件事：
    - 选股本身有没有 edge
    - 成本吃掉了多少（换手 100%/周 × 单边 23bp，一年就是 10%）

    这里一分钱成本都不算，直接量每天「选中的票」比「全市场的票」多涨/少涨
    多少，再对日序列做 t 检验。口径与回测一致：T 收盘筛 ⇒ T+1 开盘买、
    T+1+H 开盘卖，价格后复权。
    """
    oa = panels["open_adj"]
    ok = SR.screen(panels, cfg)
    n_sel = ok.sum(axis=1)
    rows = []
    for h in horizons:
        f = oa.shift(-(1 + h)) / oa.shift(-1) - 1.0
        sel = f.where(ok)
        d_sel = sel.mean(axis=1, skipna=True)
        d_all = f.mean(axis=1, skipna=True)
        diff = (d_sel - d_all).dropna()
        diff = diff[n_sel.reindex(diff.index) > 0]
        if len(diff) < 30:
            rows.append({"H": h, "天数": len(diff)})
            continue
        t = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))
        rows.append({
            "H(根)": h, "天数": len(diff),
            "候选/日": float(n_sel[n_sel > 0].mean()),
            "选中收益": float(d_sel.mean()),
            "全市场": float(d_all.mean()),
            "超额bp": float(diff.mean()) * 1e4,
            "t值": float(t),
            "胜日占比": float((diff > 0).mean()),
        })
    return pd.DataFrame(rows)


def conditions(panels: dict, cfg: SR.ScreenConfig) -> list:
    """把 8 条条件拆成**逐条的布尔宽表**，给分解表用（顺序同策略说明）。"""
    close, vol = panels["close"], panels["volume"]
    amt, pct, pe = SR.amount_wan(panels), panels["pct_chg"], panels["pe_ttm"]
    board = panels["board"].astype(float)
    is_st = panels["is_st"].astype(bool)
    prev_vol, prev_close, prev2 = vol.shift(1), close.shift(1), close.shift(2)
    lim = SR._limit_pct(is_st.shift(1, fill_value=False).to_numpy(), board.to_numpy())
    yday_up = SR._round2(prev2.to_numpy() * (1.0 + lim))
    limup = pd.DataFrame(prev_close.to_numpy() >= yday_up - 1e-6,
                         index=close.index, columns=close.columns)
    hard = pd.DataFrame(
        np.broadcast_to(board.isin([0.0, 1.0, 2.0]).to_numpy()[None, :], close.shape),
        index=close.index, columns=close.columns)
    return [
        ("非ST", ~is_st),
        ("非北交所", hard),
        ("昨日未涨停", ~limup),
        (f"股价<{cfg.max_price:g}", close < cfg.max_price),
        (f"涨幅{cfg.gain_lo:g}~{cfg.gain_hi:g}%",
         (pct >= cfg.gain_lo) & (pct <= cfg.gain_hi)),
        (f"量比≥{cfg.vol_mult:g}", vol >= cfg.vol_mult * prev_vol),
        (f"PE{cfg.pe_min:g}~{cfg.pe_max:g}", (pe > cfg.pe_min) & (pe <= cfg.pe_max)),
        (f"额≥{cfg.amt_min:g}万", amt >= cfg.amt_min),
    ]


def decompose(panels: dict, cfg: SR.ScreenConfig, h: int = 5) -> pd.DataFrame:
    """逐条条件各自的符号：**单独**放开 vs **去掉这一条**。

    全部 8 条一起看只有一个「-8.9bp」的总数，看不出谁在拖方向。这一张表是
    用来回答「哪一条把方向带反了」的 —— 例如「量比 ≥ 3」单独就是 -27.6bp
    (t=-4.2)，说明「放量当天买」本身就是负 alpha，不是别的条件没配好。
    """
    oa = panels["open_adj"]
    close = panels["close"]
    fwd = oa.shift(-(1 + h)) / oa.shift(-1) - 1.0
    d_all = fwd.mean(axis=1, skipna=True)
    alive = close.notna() & fwd.notna()
    conds = conditions(panels, cfg)

    def stat(mask, label):
        m = mask & alive
        sel = fwd.where(m).mean(axis=1, skipna=True)
        diff = (sel - d_all).dropna()
        ns = m.sum(axis=1)
        diff = diff[ns.reindex(diff.index) > 0]
        if len(diff) < 30:
            return {"组合": label, "天数": len(diff)}
        t = diff.mean() / diff.std(ddof=1) * np.sqrt(len(diff))
        return {"组合": label, "候选/日": round(float(ns[ns > 0].mean()), 2),
                "超额bp": round(float(diff.mean()) * 1e4, 1),
                "t值": round(float(t), 2),
                "胜日%": round(float((diff > 0).mean()) * 100, 1)}

    rows = [stat(v, f"单独：{k}") for k, v in conds]
    full = alive.copy()
    for _, v in conds:
        full &= v
    rows.append(stat(full, "★ 全部 8 条"))
    for k, _ in conds:
        m = alive.copy()
        for k2, v2 in conds:
            if k2 != k:
                m &= v2
        rows.append(stat(m, f"去掉：{k}"))
    return pd.DataFrame(rows)


def funnel(panels: dict, cfg: SR.ScreenConfig) -> pd.DataFrame:
    """每条条件的「单独通过率」和「逐条叠加剩余数」。

    只跑最终结果是看不出哪条在起作用的：如果 PE 那条把 95% 的票都筛掉，那
    这份回测其实是在测「低 PE」而不是「放量启动」。

    🔴 条件定义只写一遍（`conditions()`）：漏斗、分解表、正式回测各写一份的话，
    改了一处忘另一处，两张表就会对不上而没人发现。
    """
    close = panels["close"]
    alive = np.isfinite(close.to_numpy())
    total = float(alive.sum())
    rows = []
    for name, m in conditions(panels, cfg):
        m = np.asarray(m, dtype=bool) & alive
        rows.append({"条件": name, "单独通过": int(m.sum()),
                     "单独占比": m.sum() / total})
    # 逐条叠加（按上面顺序）
    stack = alive.copy()
    for (_, m), r in zip(conditions(panels, cfg), rows):
        stack = stack & np.asarray(m, dtype=bool)
        r["叠加剩余"] = int(stack.sum())
    return pd.DataFrame(rows)


def report(cfg: SR.ScreenConfig, res: SR.ScreenResult, panels: dict, tag: str) -> str:
    m = res.metrics
    eq = res.equity / res.equity.iloc[0]
    bench_r = panels["pct_chg"].mean(axis=1, skipna=True).reindex(eq.index).fillna(0) / 100
    bench = (1 + bench_r).cumprod()
    lines = [
        f"## {tag}",
        "",
        f"- 区间：{m['start']} ~ {m['end']}（{m['days']} 个交易日，{m['years']:.2f} 年）",
        f"- 净值：**{m['final_equity']:,.0f}**（本金 {cfg.capital:,.0f}），"
        f"累计 **{pct(m['total_return'])}**，年化 **{pct(m['cagr'])}**",
        f"- 夏普 {m['sharpe']:.2f}　最大回撤 {pct(m['max_dd'])}　"
        f"卡玛 {m['calmar']:.2f}　年化波动 {pct(m['ann_vol'])}",
        f"- 全市场等权基准 {pct(m['bench_total'])}，**超额 {pct(m['excess'])}**",
        f"- 交易 {m.get('n_trades', 0)} 笔　胜率 {pct(m.get('win_rate', float('nan')))}　"
        f"盈亏比 {m.get('profit_factor', float('nan')):.2f}　"
        f"均持 {m.get('avg_days', 0):.1f} 根",
        f"- 平均持仓 {m['avg_positions']:.1f} 只 / 上限 {cfg.max_pos}，"
        f"在场 {m['n_days_in_market']} 天（{m['n_days_in_market'] / m['days']:.0%}）",
        f"- 累计成本 {m.get('total_cost', 0):,.0f} 元"
        f"（占本金 {m.get('total_cost', 0) / cfg.capital:.2%}）",
    ]
    n_open = int(m.get("n_open_at_end", 0))
    if n_open:
        lines.append(
            f"- 期末仍有 **{n_open} 只**未平仓：净值已按最后收盘价计，"
            "但逐笔表里没有它们 —— 胜率/盈亏比/成本合计都少算了这几笔")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20211101")
    ap.add_argument("--end", default="20261107")
    ap.add_argument("--pe", default="25,30", help="PE 上限，逗号分隔，如 25,30")
    ap.add_argument("--hold", type=int, default=5)
    ap.add_argument("--max-pos", type=int, default=10)
    ap.add_argument("--capital", type=float, default=1_000_000.0)
    ap.add_argument("--report", default="stock_screen.md")
    args = ap.parse_args()

    s = f"{args.start[:4]}-{args.start[4:6]}-{args.start[6:]}"
    e = f"{args.end[:4]}-{args.end[4:6]}-{args.end[6:]}"
    print(f"读面板 {s} ~ {e} …", flush=True)
    panels, codes = SR.load_panels(s, e)
    if panels is None:
        print("没有数据：先跑 download_stock_ts.py --meta --st --daily --basic")
        return 1
    print(f"面板 {panels['close'].shape[0]} 个交易日 × {panels['close'].shape[1]} 只股票")

    base = SR.ScreenConfig(start=s, end=e, hold_days=args.hold,
                           max_pos=args.max_pos, capital=args.capital)

    print("\n=== 条件漏斗（全样本）===")
    fn = funnel(panels, base.clone(pe_max=max(float(x) for x in args.pe.split(","))))
    fn["单独占比"] = fn["单独占比"].map(lambda v: f"{v:.2%}")
    print(fn.to_string(index=False))

    print("\n=== 纯选股超额（零成本，看选股本身有没有 edge）===")
    ed = edge(panels, base.clone(pe_max=min(float(x) for x in args.pe.split(","))))
    print(ed.to_string(index=False))

    print("\n=== 条件分解（单独 / 去掉这一条，H=5 根、零成本）===")
    dc = decompose(panels, base.clone(pe_max=min(float(x) for x in args.pe.split(","))))
    print(dc.to_string(index=False))

    md = [f"# A股·放量启动 —— 最近 5 年全市场组合回测", "",
          f"本金 {base.capital:,.0f} 元，最多同时持 {base.max_pos} 只，"
          f"持有 {base.hold_days} 根后开盘卖；成本 手续费 {base.fee_rate:.2%}"
          f"+滑点 {base.slippage:.2%}（单边）、印花税 {base.stamp_duty:.2%}（卖出）。", "",
          "## 条件漏斗", "", fn.to_markdown(index=False), "",
          f"## 纯选股超额（零成本，PE ≤ {min(float(x) for x in args.pe.split(',')):g}）", "",
          "被选中那批票的远期收益 − 当天全市场等权收益。**这一段不含任何成本**，"
          "用来看选股本身的方向；`t值` 对日序列算。", "",
          ed.to_markdown(index=False), "",
          "## 条件分解（H=5 根、零成本）", "",
          "`单独：X` = 全市场里只按这一条过滤；`去掉：X` = 其余七条都成立、单单不要它。"
          "两者相减能看出这一条是在加分还是减分。", "",
          dc.to_markdown(index=False), ""]

    os.makedirs(OUT, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), height_ratios=[3, 1],
                             sharex=True, dpi=120)
    for pe_max in [float(x) for x in args.pe.split(",")]:
        cfg = base.clone(pe_max=pe_max)
        res = SR.run(panels, cfg)
        tag = f"PE < {pe_max:g}"
        print("\n" + "=" * 66)
        print(f"=== {tag} ===")
        for k, v in res.metrics.items():
            if isinstance(v, float):
                print(f"  {k:16s} {v:,.4f}")
            else:
                print(f"  {k:16s} {v}")
        md.append(report(cfg, res, panels, tag))
        eq = res.equity / res.equity.iloc[0]
        axes[0].plot(eq.index, eq.to_numpy(), lw=1.4, label=f"{tag}（{pct(res.metrics['total_return'])}）")
        axes[1].plot(eq.index, (eq / eq.cummax() - 1).to_numpy() * 100, lw=1.0, label=tag)
        # 逐笔
        if len(res.trades):
            res.trades.to_csv(os.path.join(OUT, f"trades_pe{pe_max:g}.csv"),
                              index=False, encoding="utf-8-sig")
            by_year = res.trades.assign(
                year=pd.to_datetime(res.trades["entry_date"]).dt.year
            ).groupby("year").agg(笔数=("code", "size"),
                                  平均收益=("ret", "mean"),
                                  累计净额=("pnl_net", "sum"))
            print("\n  分年：")
            print(by_year.to_string())
            md.append("分年统计：\n\n" + by_year.to_markdown() + "\n")

    bdates = panels["pct_chg"].index
    bench = (1 + panels["pct_chg"].mean(axis=1, skipna=True).fillna(0) / 100).cumprod()
    axes[0].plot(bdates, bench.to_numpy(), lw=1.1, color="#94a3b8",
                 label=f"全市场等权（{pct(bench.iloc[-1] - 1)}）")
    axes[0].set_title("A股·放量启动：账户净值 vs 全市场等权", fontsize=12, loc="left")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylabel("净值")
    axes[1].set_ylabel("回撤 %")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout()
    png = os.path.join(OUT, "equity_stock_screen.png")
    fig.savefig(png)
    print(f"\n净值图：{png}")

    md_path = os.path.join(OUT, args.report)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))
    print(f"报告：{md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
