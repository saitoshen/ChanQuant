"""生成 README 用的示意图（PNG）。

四张图，分别回答四个「光看文字想不出来」的问题：

1. `factor-gallery.png` —— 因子长什么样。同一个品种、同一段行情，8 个因子各自
   跟着价格怎么动。看完就知道「因子是价格的某种变形」，不是玄学指标。
2. `factor-categories.png` —— 67 个因子怎么分类。13 类各多少条、每类举一条。
3. `strategy-signals.png` —— 同一段行情，7 套规则策略给出的买卖点完全不同。
   「策略 = 规则」这句话，一张图比一段文字有用。
4. `template-map.png` —— 34 条内置模板用了哪些因子、哪些取反。热力图一眼看出
   「因子集几乎一样，方向不同」这件事。

**这些图是构建产物，不是手绘的。** 行情一变，重跑一遍就同步了：

    python packaging/make_figures.py

输出到 `docs/img/`，README 直接引用。图里的数字都来自本地库（天勤主连），
不写死任何收益率 —— 示意图只解释「怎么搭」，不做业绩背书。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# 图片底色用纸白，和图标那套「禅圆 + 朱砂」一脉相承：GitHub 的浅色主题下
# 不刺眼，深色主题下也不至于糊成一片黑。
BG = "#FBF9F4"
FG = "#1f2937"
MUTED = "#6b7280"
GRID = "#e5e1d8"
UP = "#d92b2b"        # 红涨（和 GUI 一致）
DOWN = "#0f9d58"      # 绿跌
VERMILION = "#c63c2c"  # 朱砂：取反 / 强调
INK = "#2b3440"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "img")

# 直接 `python packaging/make_figures.py` 时，sys.path[0] 是 packaging/，
# 不是仓库根 —— 拿不到 ttrade。补一下，别要求用户先设 PYTHONPATH。
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _style() -> None:
    from ttrade.plotstyle import setup_matplotlib_fonts

    setup_matplotlib_fonts()
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": FG,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "axes.grid": True,
        "axes.axisbelow": True,
        "figure.dpi": 130,
    })


def _save(fig, name: str) -> str:
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.18)
    import matplotlib.pyplot as plt

    plt.close(fig)
    print("写出", os.path.relpath(path, ROOT), f"{os.path.getsize(path) / 1024:.0f} KB")
    return path


# ------------------------------------------------------------------ 1 因子长相


GALLERY = [
    ("mom_20", "趋势", "20 根涨跌幅：价格往上走，它就在零轴上方"),
    ("rev_5", "反转", "5 根涨跌幅取负：涨得多的反而低分"),
    ("vol_20", "波动", "20 根收益标准差：横盘时贴地，异动时窜起"),
    ("vol_surge", "量能", "当日量 ÷ 20 日均量：放量立刻凸出来"),
    ("breakout_pos", "形态", "价格在 55 根通道里的相对位置，0 底 1 顶"),
    ("vwap_gap", "日内", "偏离当日 VWAP 多少：冲高就正，砸下来就负"),
    ("intraday_mom", "日内", "当日开盘到现在的涨跌幅"),
    ("first_hour", "时段", "只在开盘那一小时有值，其余时段为空 —— 时段类因子都长这样"),
]


def fig_factor_gallery() -> str:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from ttrade import factors as F
    from ttrade import tqdata

    df = tqdata.load_main("RB", "SHFE", "5m", adjust="qfq").tail(460)
    x = np.arange(len(df))
    close = df["close"].to_numpy(float)

    fig = plt.figure(figsize=(13.6, 8.4))
    # 价格只画一次（顶部通栏），下面 4×2 是八条因子曲线 —— 同一根价格线重复八遍
    # 只会让图更挤，读者并不会因此多知道什么
    gs = GridSpec(5, 2, figure=fig, height_ratios=[1.45, 1, 1, 1, 1],
                  hspace=0.34, wspace=0.14)

    ax_p = fig.add_subplot(gs[0, :])
    ax_p.plot(x, close, color=INK, lw=1.0)
    ax_p.set_title("螺纹钢主连 · 5 分钟 · 最近 460 根", loc="left", fontsize=12, pad=8)
    ax_p.set_ylabel("收盘价", fontsize=9)
    ax_p.tick_params(labelbottom=False, labelsize=8)
    ax_p.grid(axis="x", visible=False)

    axes = []
    for i, (key, cat, note) in enumerate(GALLERY):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r + 1, c], sharex=ax_p)
        axes.append(ax)

        raw = F.raw(df, key).astype(float)
        # 展示用 z：滚动 252 根标准化，截到 ±3 —— 不同量纲的因子才画得进同一张图
        z = (raw - raw.rolling(252, min_periods=20).mean()) / \
            raw.rolling(252, min_periods=20).std()
        z = z.clip(-3, 3)
        ax.axhline(0, color=GRID, lw=1)
        ax.plot(x, z.to_numpy(float), color=INK, lw=0.7, alpha=0.5)
        ax.fill_between(x, 0, z.where(z > 0).to_numpy(float), color=UP, alpha=0.55, lw=0)
        ax.fill_between(x, 0, z.where(z < 0).to_numpy(float), color=DOWN, alpha=0.55, lw=0)
        ax.set_ylim(-3.2, 3.2)
        ax.set_yticks([-2, 0, 2])
        ax.tick_params(labelbottom=False, labelsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{F.FACTORS[key].label}　·　{cat}", loc="left", fontsize=10.5, pad=4)
        ax.text(1.0, 1.04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8.2, color=MUTED)

    # 横轴只标最下面一排，日期取真实时间戳，省得读者去猜「460 根」是多长
    ticks = np.linspace(0, len(df) - 1, 6).astype(int)
    labels = [df.index[i].strftime("%m-%d") for i in ticks]
    for ax in axes[-2:]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)
        # 前面每个子图都关掉了 x 标签（sharex 下会重叠），这里要显式打开
        ax.tick_params(labelbottom=True)
    fig.subplots_adjust(bottom=0.09, top=0.94)

    fig.text(0.5, 0.02,
             "上面是价格，下面是八条因子曲线：滚动 252 根标准化，红=高分 绿=低分。"
             "因子不是新指标，是价格的某种变形 —— 所以它们跟着价格动，但各有各的相位。",
             ha="center", fontsize=9.5, color=MUTED)
    return _save(fig, "factor-gallery.png")


# ------------------------------------------------------------------ 2 因子分类


def fig_factor_categories() -> str:
    import matplotlib.pyplot as plt

    from ttrade import factors as F

    cats = list(F.CATEGORIES)
    counts = [sum(1 for s in F.FACTORS.values() if s.category == c) for c in cats]
    examples = []
    for c in cats:
        keys = [s.label for s in F.FACTORS.values() if s.category == c]
        examples.append("、".join(keys[:3]) + ("…" if len(keys) > 3 else ""))

    order = np.argsort(counts)
    cats = [cats[i] for i in order]
    counts = [counts[i] for i in order]
    examples = [examples[i] for i in order]

    fig, ax = plt.subplots(figsize=(11.4, 5.4))
    y = np.arange(len(cats))
    bars = ax.barh(y, counts, color=INK, height=0.62, alpha=0.88)
    # 需要额外数据的类别（真实合约 / 期权 / 分钟级）单独标一下，否则用户会奇怪
    # 为什么自己那儿是空的
    needs = {"展期", "展期 OI3", "展期 OI6", "期权"}
    for b, c in zip(bars, cats):
        if c in needs:
            b.set_color(VERMILION)
            b.set_alpha(0.75)
    ax.set_yticks(y, cats, fontsize=10)
    ax.set_xlabel("因子条数", fontsize=9.5)
    ax.set_xlim(0, max(counts) + 3.2)
    ax.grid(axis="y", visible=False)
    for i, (n, ex) in enumerate(zip(counts, examples)):
        ax.text(n + 0.18, i, f"{n}　{ex}", va="center", fontsize=8.8, color=MUTED)
    ax.set_title(f"{len(F.FACTORS)} 个内置因子 · {len(cats)} 类", loc="left",
                 fontsize=13, pad=12)
    fig.text(0.012, -0.055,
             "朱砂色 = 需要额外数据（真实合约期限结构 / 商品期权 / 分钟行情）；"
             "美股没有这几类数据，界面上会直接剔除，不算成 0。",
             fontsize=9, color=MUTED)
    return _save(fig, "factor-categories.png")


# ------------------------------------------------------------------ 3 规则策略信号


def fig_strategy_signals() -> str:
    import matplotlib.pyplot as plt

    from ttrade import tqdata
    from ttrade.strategies import STRATEGIES, run_strategy

    df = tqdata.load_main("RB", "SHFE", "1d_tq", adjust="qfq").tail(420)
    x = np.arange(len(df))
    close = df["close"].to_numpy(float)

    # 这两套是给 5 分钟设计的（箱体 240 根、日内时段突破），搬到日线上几乎不出
    # 信号 —— 画进来只会让人以为策略坏了。图里注明它们不在这张图上。
    INTRADAY_ONLY = {"sbbox_mini", "im_orb"}
    keys = [k for k in STRATEGIES if k not in INTRADAY_ONLY]
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 8.0), sharex=True)
    axes = axes.ravel()

    for ax, key in zip(axes, keys):
        spec = STRATEGIES[key]
        out = run_strategy(df, key, dict(spec.params))
        sig = out["signal"].to_numpy(int)
        ax.plot(x, close, color=INK, lw=0.85)
        long_in = np.where((sig == 1) & (np.roll(sig, 1) != 1))[0]
        short_in = np.where((sig == -1) & (np.roll(sig, 1) != -1))[0]
        flat = np.where((sig == 0) & (np.roll(sig, 1) != 0))[0]
        ax.scatter(long_in, close[long_in], marker="^", s=26, color=UP,
                   zorder=3, label="开多")
        ax.scatter(short_in, close[short_in], marker="v", s=26, color=DOWN,
                   zorder=3, label="开空")
        ax.scatter(flat, close[flat], marker="o", s=10, facecolor="none",
                   edgecolor=MUTED, lw=0.8, zorder=3, label="离场")
        n = len(long_in) + len(short_in)
        ax.set_title(f"{spec.label}　·　{n} 次入场", loc="left", fontsize=11, pad=5)
        ax.tick_params(labelsize=8)
        ax.grid(axis="x", visible=False)

    # 横轴标日期：索引号对读者没有意义
    ticks = np.linspace(0, len(df) - 1, 6).astype(int)
    labels = [df.index[i].strftime("%y-%m") for i in ticks]
    for ax in axes[:-1]:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=8)

    axes[-1].axis("off")
    axes[-1].text(0.02, 0.92, "同一段行情，五套规则", fontsize=12.5, color=FG, va="top")
    axes[-1].text(
        0.02, 0.80,
        "读的是同一份 K 线，买卖点却完全不同 ——\n"
        "差别只在规则：用几条均线、突破多少根的高低点、\n"
        "止损挂在几倍 ATR。\n\n"
        "▲ 开多　▼ 开空　○ 离场\n"
        "信号在下一根开盘成交，和回测引擎同一套假设。\n\n"
        "另外两套（箱体+波幅突破·迷你、IM 波幅突破·时段版）\n"
        "是给 5 分钟设计的，日线上几乎不出信号，不在这张图里。",
        fontsize=9.5, color=MUTED, va="top", linespacing=1.65,
    )
    fig.suptitle("螺纹钢主连日线 · 最近 420 根（约 1.7 年）", x=0.012, ha="left",
                 fontsize=10.5, color=MUTED, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    return _save(fig, "strategy-signals.png")


# ------------------------------------------------------------------ 4 模板 × 因子


def fig_template_map() -> str:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from collections import Counter

    from ttrade import strategy_templates as TPL

    rows: list[tuple[str, str, str]] = []      # (组, 名字, 键)
    groups: list[tuple[str, int]] = []         # (组名, 该组条数) —— 画分隔线用
    for gkey, glabel, tkeys in TPL.groups_with_items():
        groups.append((glabel, len(tkeys)))
        for k in tkeys:
            rows.append((glabel, TPL.to_config(k).title(), k))

    # 列：模板里出现过至少 5 次的因子，按出现次数排。少于这个数的因子只在一两条
    # 模板里露面，列进来只会把图撑宽，看不出规律。
    cnt: Counter = Counter()
    for _g, _n, k in rows:
        for f in (TPL.BY_KEY[k].get("factor_keys") or []):
            cnt[f] += 1
    cols = [f for f, n in cnt.most_common() if n >= 5]
    if not cols:
        raise SystemExit("模板里没有足够的因子，先检查 strategy_templates")

    grid = np.zeros((len(rows), len(cols)))
    for i, (_g, _n, k) in enumerate(rows):
        t = TPL.BY_KEY[k]
        keys = t.get("factor_keys") or []
        rev = (t.get("factor_cfg") or {}).get("reverse") or {}
        for j, f in enumerate(cols):
            if f in keys:
                grid[i, j] = 2 if rev.get(f) else 1

    fig, ax = plt.subplots(figsize=(12.8, 9.8))
    cmap = ListedColormap(["#ffffff", "#b7b0a2", VERMILION])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, aspect="auto",
              interpolation="nearest")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([cols_short(f) for f in cols], rotation=90, fontsize=8.4)
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels([n for _g, n, _k in rows], fontsize=8.8)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 分组分隔线（只画横线，不画网格）：一眼看出「这是一组」
    y = -0.5
    for glabel, n in groups:
        y += n
        ax.axhline(y, color=INK, lw=0.9)
        # 组名放在矩阵右侧的留白里，别压在格子上 —— 压上去那几行就看不见了
        ax.text(len(cols) - 0.5 + 0.45, y - n / 2, glabel, ha="left", va="center",
                fontsize=9.2, color=MUTED, clip_on=False)

    fig.suptitle("34 条内置模板 × 常用因子", x=0.012, ha="left", y=0.985,
                 fontsize=13.5, color=FG)
    fig.text(0.012, 0.955,
             "灰 = 用到　朱砂 = 取反（反着用）　白 = 没用　｜　"
             "横看一条模板用了哪些因子，竖看一个因子被哪些模板用到",
             fontsize=9.2, color=MUTED)
    fig.subplots_adjust(top=0.93, bottom=0.13, left=0.22, right=0.885)
    return _save(fig, "template-map.png")


def cols_short(key: str) -> str:
    """热力图的列名。**必须能区分同族因子** —— 展期三档口径（原 / OI3 / OI6）
    的标签只差后缀，一律砍到主干的话图上会出现三列一模一样的「展期收益率」，
    读者会以为是重复渲染。"""
    from ttrade.factors import FACTORS

    spec = FACTORS.get(key)
    label = spec.label if spec else key
    if ":" in label:                 # 时段类：后面挂着 09:45-10:15 这种时间窗
        label = label.split(" ")[0]
    label = label.replace(" ", "")
    if "·OI" in label:               # 展期 OI3 / OI6：保住口径后缀
        base, _, n = label.partition("·OI")
        return (base + "OI" + n)[:7]
    if "限幅" in label:              # iv_term 与 iv_term_v2 只差这个
        return label.split("·")[0].split("（")[0][:5] + "限"
    label = label.split("·")[0].split("（")[0]
    return label if len(label) <= 6 else label[:6]


def main() -> int:
    _style()
    paths = [fig_factor_gallery(), fig_factor_categories(),
             fig_strategy_signals(), fig_template_map()]
    print(f"共 {len(paths)} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
