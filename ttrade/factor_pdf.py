"""把因子体检结果渲染成 PDF 手册。

用 `QTextDocument` + `QPdfWriter`（都在 QtGui 里，PySide6 已经打了包）——
不引 reportlab / weasyprint 这类新依赖：为了出一份 PDF 让安装包多几十兆，
或者让用户装不上，都不值得。中文靠系统字体（Windows 微软雅黑 / macOS 苹方），
Qt 会把用到的字形子集嵌进 PDF。

排版上只用 Qt 富文本支持的那一小撮 HTML/CSS：`table` 的 `border` / `cellpadding` /
`bgcolor`，`h1`~`h3`，`p`，`b`，`font`。别写 flex / grid，它不认。
"""

from __future__ import annotations

import os

import numpy as np

from ttrade.factor_profile import ProfileResult

C_HEAD = "#1f2937"
C_MUTED = "#6b7280"
C_GOOD = "#166534"
C_BAD = "#b91c1c"
C_LINE = "#d1d5db"


def _f(v, nd: int = 3, sign: bool = False) -> str:
    """数值 → 字符串，NaN 统一显示成「—」。"""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _pct(v, nd: int = 1) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return f"{x * 100:.{nd}f}%"


def _color_of(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return C_HEAD
    if not np.isfinite(x):
        return C_MUTED
    return C_GOOD if x > 0 else (C_BAD if x < 0 else C_HEAD)


CONTENT_PX = 1000       # A4 宽 210mm − 左右各 16mm ≈ 178mm，144dpi 下约 1010px


def _half_text(f: dict) -> str:
    """半衰期那一格。IC 太小的因子不显示天数 —— 噪声的「半衰期」没有意义。"""
    if f.get("speed") == "信号弱" or f.get("no_daily"):
        return "—"
    h = f.get("half_life")
    try:
        x = float(h)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(x):
        return "—"
    return "&gt; 60 日" if x >= 900 else f"{x:.1f} 日"


def _table(headers: list[str], rows: list[list[str]], widths: list[str] | None = None,
           aligns: list[str] | None = None) -> str:
    """一张表。`widths` 是百分比，`aligns` 每列 "l" / "r"。

    **列宽必须换算成像素再给**：Qt 富文本对 `width="34%"` 这种百分比支持很差，
    实际是按内容自动分配 —— 第一列是「因子名 + 换行的 key」，内容窄，就被压成
    一条，中文全折行。给像素值它才按你说的分。
    """
    widths = widths or []
    aligns = aligns or []

    def _w(i: int) -> str:
        if i < len(widths):
            try:
                return f' width="{int(CONTENT_PX * float(widths[i].rstrip("%")) / 100)}"'
            except ValueError:
                return ""
        return ""

    out = [f'<table border="1" cellspacing="0" cellpadding="4" width="100%" '
           f'style="border-color:{C_LINE};">']
    out.append("<tr>")
    for i, h in enumerate(headers):
        out.append(f'<th{_w(i)} bgcolor="#eef2f7" align="left">'
                   f'<font color="{C_HEAD}">{h}</font></th>')
    out.append("</tr>")
    for r in rows:
        out.append("<tr>")
        for i, cell in enumerate(r):
            al = "right" if (i < len(aligns) and aligns[i] == "r") else "left"
            out.append(f'<td{_w(i)} align="{al}">{cell}</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _h1(t: str) -> str:
    return f'<h1 style="color:{C_HEAD}; font-size:20pt;">{t}</h1>'


def _h2(t: str) -> str:
    return f'<h2 style="color:{C_HEAD}; font-size:14pt;">{t}</h2>'


def _h3(t: str) -> str:
    return f'<h3 style="color:{C_HEAD}; font-size:11pt;">{t}</h3>'


def _p(t: str, color: str = C_HEAD, size: str = "10pt") -> str:
    return f'<p style="color:{color}; font-size:{size};">{t}</p>'


def _break() -> str:
    return '<p style="page-break-before:always;"></p>'


# ------------------------------------------------------------------ 各章节


def _cover(res: ProfileResult) -> str:
    m = res.meta
    span = m.get("span") or ["—", "—"]
    return "".join([
        _h1("TrendDesk 因子库深度分析手册"),
        _p(f"生成时间 {m.get('generated', '—')}　·　分析区间 {span[0]} ~ {span[1]}　·　"
           f"{m.get('n_factors', 0)} 个因子　·　{len(m.get('symbols') or [])} 个品种",
           C_MUTED, "9pt"),
        "<hr>",
        _p("这份手册回答的不是「今天该买什么」，而是「我库里的这批因子各自是什么脾气」："
           "是慢因子还是快因子、在哪些品种上真的有用、适合放到什么周期、和谁搭伙能互补、"
           "以及它的结论有多经得起换参数和换年份。", C_HEAD),
        _p("全部结论都来自本机已经下载好的日线数据，不联网、不预测。换个数据区间重跑，"
           "数字会变 —— 那正是「稳健性」这一栏存在的意义：变得越少越可信。", C_MUTED, "9pt"),
        _h2("先说三个坑"),
        _p("<b>1. IC 小不等于没用。</b>商品横截面 IC 在 0.02~0.05 之间就是能用的信号，"
           "别拿股票多因子的 0.05 当及格线。这里所有 IC 都是<b>秩相关</b>，0.03 意味着"
           "「排名靠前五分之一的品种，未来收益确实略高」，不是「每天多赚 3%」。", C_HEAD),
        _p("<b>2. 显著性看 t，不看 IC。</b>滚动 IC 序列前后高度重叠，直接算 t 会虚高十倍，"
           "所以这里全部用 Newey-West 调整。|t| 小于 2 的结论当参考，不要当依据。", C_HEAD),
        _p("<b>3. 组合要的是低相关，不是高 IC。</b>两个 IC 0.02、相关系数 0.1 的因子合起来，"
           "比一个 IC 0.03、和谁都 0.9 相关的因子有用得多。第六章的组合是按这个逻辑挑的。",
           C_HEAD),
        _h2("目录"),
        _table(["章节", "内容"], [
            ["一、指标口径", "横截面 IC、前向收益、ICIR / t、半衰期、稳健分、PIT"],
            ["二、结论速览", "按 |IC| 排名、快慢分布、最该警惕的重复因子"],
            ["三、快慢因子", "按半衰期分三档，各自的适用周期与成本提醒"],
            ["四、品种适配", "逐品种 IC，哪些因子是「真普适」哪些只靠两三个品种"],
            ["五、组合建议", "三组低相关组合 + 每个因子的互补搭子"],
            ["六、稳健性排名", "按稳健分排序，附窗口敏感性"],
            ["七、逐因子速查", "每个因子一张卡：公式、数字、最灵品种、搭子"],
        ], widths=["22%", "78%"], aligns=["l", "l"]),
        _p(f"共 {res.meta.get('n_factors', 0)} 个因子、{len(res.meta.get('symbols') or [])} 个品种、"
           f"区间 {span[0]} ~ {span[1]}。分析耗时 {res.meta.get('elapsed', '—')} 秒。",
           C_MUTED, "9pt"),
    ])


def _method(res: ProfileResult) -> str:
    return "".join([
        _break(),
        _h1("一、指标口径"),
        _p("这一章值得花两分钟读完 —— 后面所有数字的含义都在这里。"),
        _h2("横截面 IC"),
        _p("每个交易日，把 18 个品种按因子值排个名，再按未来收益排个名，两个排名算秩相关，"
           "得到一个数。十年两千多个交易日就有两千多个数，它们的均值就是 <b>IC 均值</b>。"),
        _p("为什么不用单品种的时间序列 IC：一条序列只有两千个点，噪声大、品种之间也没法比。"
           "横截面一天一个数，样本量高两个数量级，而且直接回答「这个因子擅长挑品种吗」——"
           "这正是商品多因子在做的事。手册里另外给了一列 <b>逐品种 IC</b>，那是单品种口径，"
           "用来回答「它在螺纹上到底灵不灵」。"),
        _h2("前向收益"),
        _p("因子在 T 日收盘后算出来，最早 T+1 开盘才能成交，所以收益区间是 "
           "<b>T+1 开盘 → T+h 收盘</b>。用 <b>T 日收盘 → T+h 收盘</b> 会白送一段吃不到的行情，"
           "反转类因子会被虚抬一大截。"),
        _h2("ICIR / t 值 / 方向胜率"),
        _p("<b>ICIR</b> = IC 均值 ÷ IC 标准差，衡量「这个信号稳不稳」，0.05 以上算不错。"
           "<b>t 值</b>做了 Newey-West 自相关调整，|t| ≥ 2 才谈得上显著。"
           "<b>方向胜率</b>是 IC &gt; 0 的交易日占比，0.5 就是抛硬币。"),
        _h2("半衰期（快慢因子）"),
        _p("把持有期从 1 天拉到 60 天，看 IC 什么时候掉到 1 天时的一半。半衰期 2 天的因子，"
           "两天后信号就废了，只能在 5m / 1h 上用，而且交易成本会吃掉大半；半衰期 40 天的因子"
           "适合日线甚至周线，慢慢拿。"),
        _p(f"IC 绝对值小于 0.01 的因子不给速度结论 —— 噪声的「半衰期」没有意义，"
           f"这类因子在表里标成「信号弱」。", C_MUTED, "9pt"),
        _h2("稳健性得分（0~100）"),
        _p("四项各 25 分：<b>显著性</b>（|t| 到 4 给满）、<b>方向稳定</b>（IC&gt;0 的天数占比）、"
           "<b>分年一致</b>（符号不变的年份比例）、<b>横截面普适</b>（方向一致的品种比例）。"
           "60 分以上可以进组合候选，40 分以下建议只当参考。"),
        _h2("PIT（不使用未来数据）"),
        _p("因子标准化只用滚动窗口（过去 252 天的均值方差），前向收益从 T+1 开盘算，"
           "横截面排名在同一天内做，分年统计只是把同一条 IC 序列按年切开。三处都不看未来。"
           "展期 / 期权这类日频因子在分钟线上用 D-1 快照，夜盘 bar 归下一个交易日 —— "
           "这些口径与回测完全一致。"),
    ])


def _overview(res: ProfileResult) -> str:
    fs = [f for f in res.factors if not f.get("no_daily")]
    nod = [f for f in res.factors if f.get("no_daily")]
    top = sorted(fs, key=lambda f: -abs(f.get("ic_mean") or 0))[:15]
    rows = []
    for f in top:
        rows.append([
            f"<b>{f['label']}</b><br><font color='{C_MUTED}' size='2'>{f['key']} · {f['category']}</font>",
            f"<font color='{_color_of(f['ic_mean'])}'>{_f(f['ic_mean'], 4, True)}</font>",
            _f(f.get("icir"), 3, True),
            _f(f.get("t"), 2, True),
            _pct(f.get("pos"), 0),
            _half_text(f),
            f.get("speed", "—"),
            f"{f.get('robust', 0):.0f}",
        ])
    # 快慢分布
    dist: dict[str, int] = {}
    for f in fs:
        dist[f.get("speed", "—")] = dist.get(f.get("speed", "—"), 0) + 1
    order = ["极快", "快", "中快", "中", "慢", "很慢", "信号弱"]
    dist_row = "　".join(f"{k} {dist[k]} 个" for k in order if k in dist)

    return "".join([
        _break(),
        _h1("二、结论速览"),
        _h2("按 |IC| 排名前 15"),
        _table(["因子", "IC 均值", "ICIR", "t 值", "方向胜率", "半衰期", "速度", "稳健分"],
               rows, widths=["34%", "10%", "9%", "8%", "9%", "11%", "8%", "9%"],
               aligns=["l", "r", "r", "r", "r", "r", "l", "r"]),
        _p("IC 有正有负是正常的：负号说明「因子值越低，未来涨得越多」，回测里方向由滚动 IC "
           "自动决定，不需要手工翻转。", C_MUTED, "9pt"),
        _h2("快慢分布"),
        _p(dist_row or "—"),
        _p("同一个库里的因子如果全是慢因子，组合在日内完全没法用；全是快因子，交易成本会把"
           "收益吃光。快慢搭配是组合能落地的前提。", C_MUTED, "9pt"),
        _h2(f"日线上算不出来的因子（{len(nod)} 个）"),
        _p("这些因子要的是「当天上午还是下午」「开盘首小时」这类日内形态，日线一根 bar 里没有"
           "这个信息，所以在本手册里没有数字。它们只在 5m / 1h 上有效，"
           "选周期时别拿日线的结论去套它们。", C_MUTED, "9pt"),
        _p("、".join(f"{f['label']}（{f['key']}）" for f in nod) or "（无）", C_HEAD, "9pt"),
        _h2("最该警惕的重复"),
        _p("下面这些因子两两相关系数在 0.85 以上，本质上是同一个因子。同时用它们做多因子，"
           "等于把同一注押了好几遍，仓位会不知不觉放大。"),
        _table(["因子 A", "因子 B", "相关系数"],
               [[f"<b>{p['a']}</b>", f"<b>{p['b']}</b>", _f(p["corr"], 3, True)]
                for p in (res.redundant or [])[:8]],
               widths=["40%", "40%", "20%"], aligns=["l", "l", "r"]),
    ])


def _speed_chapter(res: ProfileResult) -> str:
    groups = [
        ("快因子（半衰期 ≤ 3 日）", ("极快", "快"),
         "信号两三天就衰减完。只能放在 5m / 1h 上，且必须把手续费和滑点算进去 —— "
         "快因子的收益很容易被成本吃掉一半。适合做择时的「入场时机」而不是方向。"),
        ("中速因子（3 ~ 20 日）", ("中快", "中"),
         "日线的主战场。持仓几天到几周，成本和信号寿命匹配，是绝大多数 CTA 组合的主力。"),
        ("慢因子（&gt; 20 日）", ("慢", "很慢"),
         "拿到几周甚至几个月。换手低、成本不敏感，但容量有限、回撤期长，"
         "最好和快因子搭配，用快因子管入场、慢因子管方向。"),
    ]
    out = [_break(), _h1("三、快慢因子")]
    for title, kinds, note in groups:
        sel = [f for f in res.factors if f.get("speed") in kinds]
        sel.sort(key=lambda f: -abs(f.get("ic_mean") or 0))
        out.append(_h2(f"{title}　{len(sel)} 个"))
        out.append(_p(note, C_MUTED, "9pt"))
        if not sel:
            out.append(_p("（无）", C_MUTED))
            continue
        out.append(_table(
            ["因子", "类别", "IC", "ICIR", "t", "半衰期", "换手", "稳健"],
            [[f"<b>{f['label']}</b><br><font color='{C_MUTED}' size='2'>{f['key']}</font>",
              f["category"],
              f"<font color='{_color_of(f['ic_mean'])}'>{_f(f['ic_mean'], 4, True)}</font>",
              _f(f.get("icir"), 3, True), _f(f.get("t"), 2, True),
              _half_text(f),
              _f(f.get("turnover"), 3),
              f"{f.get('robust', 0):.0f}"]
             for f in sel],
            widths=["32%", "12%", "11%", "10%", "8%", "11%", "8%", "8%"],
            aligns=["l", "l", "r", "r", "r", "r", "r", "r"]))
    return "".join(out)


def _symbol_chapter(res: ProfileResult) -> str:
    """按板块看：这个板块上哪些因子最灵。"""
    sector_of = {}
    for key in res.meta.get("symbols") or []:
        sym, _, ex = key.rpartition("_")
        sector_of[key] = sym
    # 用每个因子的 per_symbol 反查板块需要板块表；这里退一步按品种列，够用且不猜
    out = [_break(), _h1("四、品种适配")]
    out.append(_p("每个因子的「逐品种 IC」是单品种口径：只拿这一个品种的因子值和它自己的"
                  "未来收益做秩相关。同号的品种越多、越强，说明这个因子不是靠某一个品种撑起来的。"))
    rows = []
    for f in sorted([x for x in res.factors if not x.get("no_daily")],
                    key=lambda x: -(x.get("ts_agree") or 0))[:20]:
        best = f.get("best_symbols") or []
        rows.append([
            f"<b>{f['label']}</b><br><font color='{C_MUTED}' size='2'>{f['key']}</font>",
            _pct(f.get("ts_agree"), 0),
            _f(f.get("ts_ic"), 4, True),
            "、".join(b.split("_")[0] for b in best) or "—",
            f["category"],
        ])
    out.append(_h2("方向最一致的 20 个因子"))
    out.append(_table(["因子", "同号品种占比", "逐品种 IC 均值", "最灵的品种", "类别"],
                      rows, widths=["32%", "14%", "14%", "26%", "14%"],
                      aligns=["l", "r", "r", "l", "l"]))
    out.append(_p("「同号品种占比」低于 50% 的因子，说明它在多数品种上方向是反的 —— "
                  "整体 IC 可能靠两三个品种撑着，换品种就不灵，用之前先在目标品种上单独验一遍。",
                  C_MUTED, "9pt"))
    return "".join(out)


def _combo_chapter(res: ProfileResult) -> str:
    out = [_break(), _h1("五、组合建议")]
    out.append(_p("多因子的收益不来自「每个因子都很强」，而来自「各自押不同的东西」。"
                  "所以下面三组都是先按强度排序，再强制两两低相关（|相关| &lt; 0.55）、"
                  "同类别最多两个 —— 展期那一家子有十几个变体，相关性 0.9 以上，"
                  "放三个进去等于三倍押 carry。"))
    for c in res.combos or []:
        out.append(_h2(f"{c['name']}　{len(c['keys'])} 个因子"))
        out.append(_p(c.get("why", ""), C_MUTED, "9pt"))
        rows = []
        for k in c["keys"]:
            f = next((x for x in res.factors if x["key"] == k), None)
            if f is None:
                continue
            rows.append([
                f"<b>{f['label']}</b><br><font color='{C_MUTED}' size='2'>{f['key']}</font>",
                f["category"], f.get("speed", "—"), f.get("period", "—"),
                _f(f.get("icir"), 3, True), f"{f.get('robust', 0):.0f}",
            ])
        out.append(_table(["因子", "类别", "速度", "建议周期", "ICIR", "稳健"],
                          rows, widths=["30%", "12%", "10%", "30%", "10%", "8%"],
                          aligns=["l", "l", "l", "l", "r", "r"]))
        out.append(_p(f"组合平均 |ICIR| {c.get('icir', 0):.3f}　·　"
                      f"组内最大 |相关| {c.get('max_corr', 0):.2f}"))
    out.append(_h2("怎么用这三组"))
    out.append(_p("「稳健组」适合当默认配置；「显著组」适合样本外验证时对照；"
                  "「分散组」在你想降低单因子失效风险时用。三组都用等权就能跑，"
                  "要加权的话按稳健分加权，不要按 IC 加权 —— IC 越高的因子往往越不稳定，"
                  "按 IC 加权等于把仓位集中到最脆的那个上。"))
    return "".join(out)


def _robust_chapter(res: ProfileResult) -> str:
    out = [_break(), _h1("六、稳健性排名")]
    out.append(_p("按稳健分排序。这一章的作用是「排雷」：IC 高但稳健分低的因子，"
                  "多半是某几年特别灵、其他年份拖后腿，实盘正好赶上后半段就会很难受。"))
    rows = []
    for f in sorted([x for x in res.factors if not x.get("no_daily")],
                    key=lambda x: -(x.get("robust") or 0)):
        win = f.get("win_ic") or {}
        sens = "—"
        if len(win) >= 2:
            vals = [abs(float(v)) for v in win.values() if np.isfinite(float(v))]
            if len(vals) >= 2:
                sens = _f(max(vals) - min(vals), 4)
        rows.append([
            f"<b>{f['label']}</b><br><font color='{C_MUTED}' size='2'>{f['key']}</font>",
            f"{f.get('robust', 0):.0f}",
            _f(f.get("t"), 2, True), _pct(f.get("pos"), 0),
            _pct(f.get("year_ratio"), 0), sens,
        ])
    out.append(_table(["因子", "稳健分", "t 值", "方向胜率", "分年一致", "窗口敏感性"],
                      rows, widths=["34%", "10%", "10%", "12%", "12%", "22%"],
                      aligns=["l", "r", "r", "r", "r", "r"]))
    out.append(_p("「窗口敏感性」是把标准化窗口从 252 天换成 120 / 504 天之后，IC 的变化幅度。"
                  "小于 0.005 说明这个因子不挑参数，放心用；大于 0.02 说明它很吃窗口，"
                  "实盘换个起点结论就可能翻。", C_MUTED, "9pt"))
    return "".join(out)


def _cards(res: ProfileResult) -> str:
    out = [_break(), _h1("七、逐因子速查")]
    out.append(_p("按类别排列，每个因子一张卡：一句话、公式、核心数字、最灵的品种、"
                  "以及三个「低相关且自身够稳」的搭子。"))
    by_cat: dict[str, list[dict]] = {}
    for f in res.factors:
        by_cat.setdefault(f.get("category") or "其他", []).append(f)
    for cat in sorted(by_cat):
        out.append(_h2(f"{cat}（{len(by_cat[cat])} 个）"))
        for f in sorted(by_cat[cat], key=lambda x: -abs(x.get("ic_mean") or 0)):
            mates = "、".join(
                f"{m['key']}（相关 {m['corr']:+.2f}）" for m in (f.get("mates") or [])
            ) or "—"
            out.append(_h3(f"{f['label']}　<font color='{C_MUTED}' size='2'>{f['key']}</font>"))
            out.append(_p(f"{f.get('desc', '')}"))
            if f.get("formula"):
                out.append(_p(f"公式：<code>{f['formula']}</code>", C_MUTED, "9pt"))
            if f.get("no_daily"):
                out.append(_p("日线数据上整列为空 —— 这个因子读的是日内形态（上午 / 下午、"
                              "开盘首小时、夜盘），只能放在 5m / 1h 上评估。", C_MUTED, "9pt"))
                continue
            out.append(_p(
                f"IC {_f(f.get('ic_mean'), 4, True)}　·　ICIR {_f(f.get('icir'), 3, True)}"
                f"　·　t {_f(f.get('t'), 2, True)}　·　方向胜率 {_pct(f.get('pos'), 0)}"
                f"　·　半衰期 {_half_text(f)}（{f.get('speed', '—')}）"
                f"　·　稳健 {f.get('robust', 0):.0f}"))
            out.append(_p(
                f"建议周期：{f.get('period', '—')}　·　换手 {_f(f.get('turnover'), 3)}"
                f"　·　最灵品种：{'、'.join(b.split('_')[0] for b in (f.get('best_symbols') or [])) or '—'}",
                C_MUTED, "9pt"))
            out.append(_p(f"可搭配：{mates}", C_MUTED, "9pt"))
    return "".join(out)


def build_html(res: ProfileResult) -> str:
    body = "".join([
        _cover(res), _method(res), _overview(res), _speed_chapter(res),
        _symbol_chapter(res), _combo_chapter(res), _robust_chapter(res), _cards(res),
    ])
    return (
        "<html><head><meta charset='utf-8'></head>"
        "<body style=\"font-family:'Microsoft YaHei','PingFang SC','Noto Sans CJK SC',sans-serif;"
        "font-size:10pt; color:#1f2937;\">" + body + "</body></html>"
    )


def write_pdf(res: ProfileResult, path: str, title: str = "") -> str:
    """渲染并写盘。返回路径。必须在有 QGuiApplication 的进程里调用。"""
    from PySide6.QtCore import QMarginsF, QSizeF
    from PySide6.QtGui import QPageLayout, QPageSize, QPdfWriter, QTextDocument

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    w = QPdfWriter(path)
    w.setPageSize(QPageSize(QPageSize.A4))
    w.setResolution(144)
    w.setPageMargins(QMarginsF(16, 14, 16, 14), QPageLayout.Millimeter)
    w.setTitle(title or "TrendDesk 因子库深度分析手册")
    w.setCreator("TrendDesk")

    doc = QTextDocument()
    doc.setHtml(build_html(res))
    # 页面尺寸要按**设备像素**给：QPdfWriter 的 width/height 已经算上了 DPI 与页边距
    doc.setPageSize(QSizeF(w.width(), w.height()))
    doc.print_(w)
    return path
