"""板块维度：板块里有哪些品种、同一个因子换个品种还成不成立、品种跟板块怎么联动。

## 为什么单开一个包

`factor` 包回答「这个因子怎么算」，`backtest` 包回答「这一条策略好不好」。
研究里真正卡住人的是**第三件事**：一个因子在某个品种上有效，是它抓到了板块级的
共同驱动，还是这个品种自己的历史巧合？这两件事的后果完全不同 —— 前者可以推到
整个板块，后者只能单品种用，换品种就废（也就是过拟合）。

判这件事只能靠横向表：**同一个因子、同一个方向、同一个周期，在板块内所有品种上
各跑一遍**，然后数「几个品种是正的」。单看一个品种的回测永远看不出来。

## 四个动作

| 动作 | 回答什么 |
|---|---|
| `sector_list` | 这个板块有哪些品种、本地有没有数据（`level=1` 大类 / `level=2` 产业板块） |
| `sector_scan` | 因子 × 品种 × 周期 的横向表，带 IS/OOS 切段 |
| `sector_relation` | 品种和板块指数怎么联动（相关性 / beta / 解释力） |
| `sector_corr` | 板块内部有多像、板块之间有多像（判「算几个独立假设」） |

## 两层板块

- **L1 大类（`futures.SECTOR_MAP`，9 个）**：贵金属/有色/黑色/能源/化工/农产品/金融/新能源/航运。
  宏观驱动维度，做板块指数、品种联动、大类配权用这一层。
- **L2 产业板块（`futures.SECTOR_L2`，16 个）**：化工拆 4（聚酯/烯烃芳烃/煤化工/橡胶纸浆）、
  农产品拆 4（油脂油料/软商品鲜果/谷物养殖/木材）、金融拆 2（股指/国债），其余不变。
  产业链维度，**做因子横截面扫描必须用这一层** —— 化工 21 个品种跨四条链，
  在 L1 上数「几个品种为正」等于把四条链平均掉。

实测（2016-11~2026-11 日线）：板块间平均相关 L1 0.24 → L2 0.19，
有效独立假设数（相关矩阵参与率）5.5 → 8.3。详见 `reports/sector_l2_review.md`。

## 口径（都是踩过坑才定下来的，别改）

- **窗口按「根」不按天**：`factor_cfg.window` 是 bar 根数。换周期时用 `eco`
  口径（252 × 每日根数）把回看长度固定在「约一年」，否则 30m 上 252 根只有
  20 个交易日，等于换了一条策略；
- **方向固定**：`long_high`（高分做多）/ `long_low`（低分做多），**不用 auto**。
  auto 是按滚动 IC 符号定向，方向会随样本长度变，IS / OOS / 全样本根本不是同一
  条策略；而且在短周期上翻得最频繁，等于把仓位来回砍；
- **IS / OOS 用同一条净值的日期切片**，不是「切数据再跑一遍」。后者会让 OOS 段
  重新预热（约 3 个月）后才起效，OOS 被悄悄推迟一个季度；
- **数据必须挂 aux 衍生列**（`auxdata.attach(_intraday)`），否则展期 / 期限结构
  那十几个因子在分钟线上整列 NaN，而且是**静默**的（策略照样跑完，只是换了条）。

方法论全文见技能包 `sector-factor-scan`。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ttrade import blocks as B
from ttrade.skills.base import (
    NO_LIMIT,
    RISK_READ,
    SkillResult,
    Tool,
    register,
)

# 一次扫描的格子上限。因子 × 品种 × 周期 × 方向，超了就慢到没法用 ——
# 让模型先缩小范围，比让它等五分钟好
MAX_CELLS = 400
MAX_SYMBOLS = 12
MAX_PERIODS = 3
MAX_FACTORS = 6

# 每日根数兜底表（实际值优先从数据里数，见 `_bpd`）
BARS_PER_DAY = {"1d": 1.0, "1d_tq": 1.0, "2d": 0.5, "1w": 0.2,
                "4h": 2.0, "2h": 3.9, "1h": 7.0, "30m": 11.2,
                "15m": 22.0, "10m": 33.0, "5m": 60.0, "3m": 100.0, "1m": 300.0}

# 默认因子：上一轮化工扫描里跨品种最能跑出来的几个家族各取一条
DEFAULT_FACTORS = ["roll_main_oi", "roll_curv_o3", "sector_mom_10",
                   "commodity_mom_10", "near_oi_share", "mom_20"]

DIRS = {"high": "long_high", "low": "long_low"}
DIR_CN = {"high": "高分多", "low": "低分多"}


# ------------------------------------------------------------------ 基础设施


def _sector_map() -> dict:
    from ttrade.futures import SECTOR_MAP

    return SECTOR_MAP


def _norm_name(sector: str) -> str:
    """板块名归一：`化工-聚酯` / `化工/聚酯` / `化工·聚酯` 都认。

    用户不会记得分隔符是「·」，模型也常常写成 `-`，所以在查表之前统一一次。
    """
    s = str(sector or "").strip()
    for sep in ("-", "/", "／", "・", "•", "_"):
        s = s.replace(sep, "·")
    return s


def _sector_codes(name: str) -> list[str]:
    """板块名 → 代码列表。两层都认：先查 L1（9 个大类），再查 L2（16 个产业板块）。"""
    from ttrade.futures import SECTOR_L2, SECTOR_MAP

    n = _norm_name(name)
    for table in (SECTOR_MAP, SECTOR_L2):
        if n in table:
            return table[n].split()
    return []


def _sector_keys(sector: str) -> list[str]:
    """板块名 → 品种池里的键（`MA_CZCE`）。

    `SECTOR_MAP` 是「板块名 → 代码串」，要拿**板块名**去查它，不能拿名字当代码比。
    （第一版就是直接把「化工」当代码去和 `symbol` 比较，于是永远 0 个品种。）

    L2 产业板块（`化工·聚酯`）走同一套：代码集更小，只取落在里面的品种。
    """
    from ttrade.futures import list_local
    from ttrade.paths import data_dir as resolve_data_dir

    codes = set(_sector_codes(sector))
    if not codes:                      # 也允许直接传一串代码（"MA TA EG"）
        raw = str(sector or "").upper().replace(",", " ").replace("，", " ")
        codes = {c for c in raw.split() if c}
    if not codes:
        return []
    rows = list_local(resolve_data_dir("cn"))
    return [r["key"] for r in rows if str(r.get("symbol", "")).upper() in codes]


def _store(ctx):
    return getattr(ctx, "store", None)


def _frame(ctx, key: str, period: str) -> pd.DataFrame:
    """取一份带 aux 衍生列的行情。界面在就用界面的缓存，headless 直接读库。"""
    store = _store(ctx)
    if store is not None:
        df = store.get(key, period)
        if df is not None and not df.empty:
            return df
        err = getattr(store, "last_error", "")
        if err:
            raise RuntimeError(f"{key} {period} 读不出来：{err}")
    from ttrade import auxdata, tqdata
    from ttrade.paths import data_dir as resolve_data_dir

    sym, _, ex = str(key).rpartition("_")
    if not sym or not ex:
        raise ValueError(f"品种写法要像 MA_CZCE，收到的是 {key}")
    aux = resolve_data_dir("aux")
    df = tqdata.load_main(sym, ex, period, adjust="qfq")
    if df is None or df.empty:
        raise RuntimeError(f"{key} 在 {period} 上没有数据")
    try:
        if tqdata.is_daily(period):
            df = auxdata.attach(df, sym, ex, aux)
        else:
            df = auxdata.attach_intraday(df, sym, ex, aux)
    except Exception:  # noqa: BLE001 - 没有 aux 文件时照跑，只是展期因子会空
        pass
    return df


def _bpd(period: str, df: pd.DataFrame | None = None) -> float:
    """每个交易日几根 bar。优先从数据里数（夜盘、半天的差异都算进去）。"""
    if df is not None and len(df) > 2:
        try:
            from ttrade.factors import bars_per_day

            v = float(bars_per_day(df.index))
            if v > 0:
                return v
        except Exception:  # noqa: BLE001
            pass
    return float(BARS_PER_DAY.get(period, 1.0))


def eco_cfg(period: str, df: pd.DataFrame | None = None) -> dict:
    """`eco` 口径：回看始终约一年、预测期约 5 天、重算 IC 约 60 天，只改采样频率。"""
    b = max(1e-9, _bpd(period, df))
    return {"window": int(round(252 * b)), "horizon": max(1, int(round(5 * b))),
            "ic_rebal": max(1, int(round(60 * b))), "bpd": b}


def _seg(eq: pd.Series, start=None, end=None) -> dict:
    """从**同一条净值**里切一段算指标 —— 不重新跑，所以不含预热。

    年化用真实跨度（首尾相差多少年），不用根数除 252：分钟线按 252 算会虚高一个
    数量级，这是净值口径的老坑。
    """
    if eq is None or len(eq) < 3:
        return {}
    s = eq.loc[start:end] if (start or end) else eq
    s = s.dropna()
    if len(s) < 3 or float(s.iloc[0]) <= 0:
        return {}
    span = (s.index[-1] - s.index[0]).total_seconds() / 86400.0
    years = max(span / 365.25, 1e-9)
    ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
    cagr = (1.0 + ret) ** (1.0 / years) - 1.0 if (1.0 + ret) > 0 else -1.0
    peak = s.cummax()
    dd = float(((s - peak) / peak.replace(0, np.nan)).min())
    return {"ret": ret, "cagr": cagr, "dd": dd,
            "calmar": (cagr / abs(dd)) if dd < 0 else 0.0, "bars": int(len(s))}


def _run(keys, df, period, orient, start=None, end=None, ec=None) -> dict:
    """跑一条单因子策略，返回全样本 metrics + 净值（用来切段）。"""
    from ttrade import replay
    from ttrade.strategy_store import StrategyConfig

    ec = ec or eco_cfg(period, df)
    cfg = StrategyConfig(
        name="sector_probe", symbol="", exchange="", period=period,
        family="factor", factor_mode="multi", factor_keys=list(keys),
        factor_cfg={"window": ec["window"], "horizon": ec["horizon"],
                    "entry_z": 1.0, "exit_z": 0.3, "weight_mode": "equal",
                    "ic_rebal": ec["ic_rebal"], "method": "z", "orient": orient},
        sizing="budget", principal=1_000_000, capital_pct=50.0, note="short",
    )
    rp = replay.run(cfg, df.loc[start:end] if (start or end) else df,
                    market="cn", with_bh=False)
    if not rp.ok:
        return {"ok": False, "err": (rp.error or "")[:60]}
    m = rp.res.metrics or {}
    return {"ok": True, "calmar": m.get("calmar"), "ret": m.get("total_return"),
            "trades": m.get("n_trades"), "win": m.get("win_rate"),
            "pf": m.get("profit_factor"), "dd": m.get("max_dd"),
            "equity": rp.res.equity, "error": ""}


def _labels(ctx, keys) -> dict:
    """品种键 → 中文名。界面有就在，没有就退回代码。"""
    out = {}
    store = _store(ctx)
    items = getattr(store, "items", None) or []
    for it in items:
        out[str(it.get("key"))] = str(it.get("name") or "")
    return {k: out.get(k, "") for k in keys}


# ------------------------------------------------------------------ sector_list


def _sector_list(ctx, p: dict) -> SkillResult:
    """列板块。`level=2` 时按 16 个产业板块列，并带上它属于哪个大类。

    两个层级回答不同的问题：
    - **L1 大类（9 个）**：宏观驱动，做板块指数、品种联动、大类配权用这一层；
    - **L2 产业板块（16 个）**：产业链，做因子横截面扫描用这一层。
      板块越大越容易把不相关的假设平均成一个结论（化工 21 个跨四条链）。
    """
    from ttrade.futures import L2_PARENT, SECTOR_L2, SECTOR_MAP

    want = _norm_name(p.get("sector") or "")
    level = int(p.get("level") or 1)
    table = SECTOR_L2 if level == 2 else SECTOR_MAP
    names = [want] if want else list(table)
    rows, total_sym = [], 0
    for name in names:
        if name not in table:
            near = [k for k in table if want and want in k]
            raise RuntimeError(
                f"没有「{name}」这个板块（level={level}）" +
                (f"（相近的有：{'、'.join(near)}）" if near else
                 f"（有的是：{'、'.join(table)}）"))
        codes = table[name].split()
        keys = _sector_keys(name)
        total_sym += len(keys)
        missing = [c for c in codes if not any(k.startswith(c + "_") for k in keys)]
        row = [name]
        if level == 2:
            row.append(L2_PARENT.get(name, ""))
        row += [len(codes), len(keys),
                "、".join(k.rpartition("_")[0] for k in keys) or "—",
                "、".join(missing) or "—"]
        rows.append(row)

    cols = ["板块", "代码数", "本地有数据", "品种", "缺数据/未下载"]
    if level == 2:
        cols.insert(1, "大类")
    note = "「本地有数据」= 已通过活跃度与流动性筛选、可以直接回测的品种"
    if level == 2:
        note += ("；产业板块比大类小，做 `sector_scan` 的横截面时用这一层 —— "
                 "成分少于 3 个的（木材/航运）只适合单品种看")
    else:
        note += "；`level=2` 可看 16 个产业板块"
    blocks = [B.block_table(cols, rows, note=note)]

    if want:
        # 单看一个板块时把板块指数的当前状态也报出来（L1/L2 都适用）
        keys = _sector_keys(want)
        if keys:
            per = str(p.get("period") or "1d_tq")
            idx = _sector_index(ctx, keys, per)
            if idx is not None and len(idx) > 2:
                last = float(idx.iloc[-1])
                chg20 = float(idx.iloc[-1] / idx.iloc[-min(21, len(idx))] - 1) * 100
                blocks.insert(0, B.block_metrics(
                    f"{want}板块等权指数 · {per}",
                    [("最新", f"{last:.4f}", "flat"),
                     (f"近 {min(20, len(idx) - 1)} 根", f"{chg20:+.2f}%",
                      "good" if chg20 >= 0 else "bad"),
                     ("成分", f"{len(keys)} 个", "flat")]))
    i_have = cols.index("本地有数据")
    if want:
        summary = f"{want}：{rows[0][i_have]} 个品种可回测"
    else:
        summary = "、".join(f"{r[0]} {r[i_have]} 个" for r in rows)
    return SkillResult(True, summary=summary, blocks=blocks, feed=summary)


def _sector_index(ctx, keys, period, field: str = "close",
                  min_members: int = 2) -> pd.Series | None:
    """板块等权指数 = 各品种 close 归一化后取均值。

    `min_members` 默认 2（「指数」至少要两个成分才有意义），但 `sector_corr`
    要算单品种板块（航运/木材）跟别人的相关，那时传 1 —— 否则它们没有指数，
    板块间相关矩阵会静默缺一整行（实测只剩 1 对有效值，看着像「算不出来」）。
    """
    ser = []
    for k in keys:
        try:
            df = _frame(ctx, k, period)
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 10 or field not in df:
            continue
        s = pd.to_numeric(df[field], errors="coerce").dropna()
        if len(s) < 10:
            continue
        ser.append(s / float(s.iloc[0]))
    if len(ser) < max(1, min_members):
        return None
    return pd.concat(ser, axis=1).sort_index().ffill().mean(axis=1).dropna()


# ------------------------------------------------------------------ sector_scan


def _sector_scan(ctx, p: dict) -> SkillResult:
    from ttrade.factors import FACTORS

    sector = str(p.get("sector") or "").strip()
    if not sector:
        raise RuntimeError("要指定板块，比如 sector=\"化工\"（用 sector_list 看有哪些）")
    keys = _sector_keys(sector)
    if not keys:
        raise RuntimeError(f"板块「{sector}」在本地没有可回测的品种 —— 先下载数据")

    want = [str(x).strip() for x in (p.get("symbols") or []) if str(x).strip()]
    if want:
        norm = {k.rpartition("_")[0].upper(): k for k in keys}
        picked = []
        for w in want:
            u = str(w).upper()
            # 三种写法都认：`MA` / `MA_CZCE` / 中文名在界面上才有，这里只管代码
            k = norm.get(u) or norm.get(u.rpartition("_")[0])
            if k is None:
                raise RuntimeError(f"{w} 不在{sector}板块里（有：{'、'.join(sorted(norm))}）")
            if k not in picked:
                picked.append(k)
        keys = picked
    if len(keys) > MAX_SYMBOLS:
        raise RuntimeError(f"一次最多扫 {MAX_SYMBOLS} 个品种，现在 {len(keys)} 个 —— "
                           "用 symbols 参数先挑一部分")

    periods = [str(x) for x in (p.get("periods") or [p.get("period") or "1d_tq"]) if x]
    periods = [x for x in periods if x]
    if len(periods) > MAX_PERIODS:
        raise RuntimeError(f"一次最多 {MAX_PERIODS} 个周期，先挑重点的")

    fkeys = [str(x).strip() for x in (p.get("factors") or []) if str(x).strip()] \
        or list(DEFAULT_FACTORS)
    bad = [f for f in fkeys if f not in FACTORS]
    if bad:
        raise RuntimeError(f"没有这些因子：{'、'.join(bad)}（用 factor_list 查）")
    if len(fkeys) > MAX_FACTORS:
        raise RuntimeError(f"一次最多 {MAX_FACTORS} 个因子，现在 {len(fkeys)} 个")

    orient = str(p.get("orient") or "both").lower()
    dirs = ["high", "low"] if orient == "both" else [orient if orient in DIRS else "both"]
    dirs = [d for d in dirs if d in DIRS]
    if not dirs:
        raise RuntimeError("orient 只能是 high / low / both")

    cells = len(keys) * len(periods) * len(fkeys) * len(dirs)
    if cells > MAX_CELLS:
        raise RuntimeError(f"这一次要跑 {cells} 格，上限 {MAX_CELLS} —— "
                           "少选几个品种 / 周期 / 因子再来")

    split = str(p.get("split") or "2024-01-01")
    start = str(p.get("start") or "")
    top = max(5, min(int(p.get("top") or 25), 120))
    names = _labels(ctx, keys)

    rows, per_factor = [], {}
    for period in periods:
        for key in keys:
            df = _frame(ctx, key, period)
            ec = eco_cfg(period, df)
            for f in fkeys:
                for d in dirs:
                    r = _run([f], df, period, DIRS[d], start or None, None, ec)
                    if not r.get("ok"):
                        continue
                    eq = r["equity"]
                    is_s = _seg(eq, start or None, split)
                    oos = _seg(eq, split, None)
                    rob = (min(is_s.get("calmar") or 0, oos.get("calmar") or 0)
                           if is_s and oos and (r.get("trades") or 0) >= 15 else 0.0)
                    rows.append({
                        "sym": key.rpartition("_")[0], "cn": names.get(key, ""),
                        "period": period, "factor": f, "dir": d,
                        "calmar": r.get("calmar"), "ret": r.get("ret"),
                        "trades": r.get("trades"), "win": r.get("win"),
                        "is": is_s.get("calmar"), "oos": oos.get("calmar"),
                        "robust": rob, "window": ec["window"]})
                    slot = per_factor.setdefault(f, {"n": 0, "pos": 0, "both": 0, "c": []})
                    slot["n"] += 1
                    slot["c"].append(r.get("calmar") or 0)
                    if (r.get("calmar") or 0) > 0:
                        slot["pos"] += 1
                    if rob > 0:
                        slot["both"] += 1

    if not rows:
        return SkillResult(False, error=f"{sector} 上这些因子一条都没跑出来 —— "
                                        "检查数据是否下载、周期是否存在")

    rows.sort(key=lambda x: -((x["robust"] or 0) * (1 + 0.02 * min(x["trades"] or 0, 50))))
    show = rows[:top]
    table = [[r["sym"] + (f" {r['cn']}" if r["cn"] else ""), r["period"], r["factor"],
              DIR_CN[r["dir"]], _f(r["calmar"]), _pct(r["ret"]), r["trades"] or 0,
              _f(r["is"]), _f(r["oos"]), _f(r["robust"])] for r in show]

    fsum = []
    for f, s in sorted(per_factor.items(), key=lambda kv: -kv[1]["pos"] / max(1, kv[1]["n"])):
        n = max(1, s["n"])
        fsum.append([f, s["n"], s["pos"], f"{100.0 * s['pos'] / n:.0f}%",
                     s["both"], _f(float(np.median(s["c"])))])

    blocks = [
        B.block_metrics(
            f"{sector} · {len(keys)} 品种 × {len(periods)} 周期 × {len(fkeys)} 因子",
            [("格子数", f"{len(rows)}", "flat"),
             ("稳健格数", f"{sum(1 for r in rows if r['robust'] > 0)}", "flat"),
             ("口径", "eco 窗口 / 固定方向 / 净值切段", "flat")],
        ),
        B.block_table(
            ["因子", "格子", "全样本为正", "占比", "三段同号", "卡玛中位"], fsum,
            note="「全样本为正」= 该因子在这个板块里有多少格是赚的；"
                 "占比接近一半说明它在这里没有稳定方向，不要单独用"),
        B.block_table(
            ["品种", "周期", "因子", "方向", "卡玛", "累计", "笔数", "IS", "OOS", "稳健"],
            table,
            note="IS / OOS 是从同一条净值的日期切片算的（不重新预热）；"
                 f"「稳健」= min(IS, OOS) 且三段同号、笔数 ≥ 15；切点 {split}"),
    ]
    good = [r for r in show if r["robust"] > 0]
    summary = (f"{sector}：{len(rows)} 格跑完，{len(good)} 格三段同号；"
               f"最好的 {good[0]['sym']} {good[0]['period']} {good[0]['factor']}"
               f"（{DIR_CN[good[0]['dir']]}，稳健 {good[0]['robust']:.2f}）"
               if good else f"{sector}：{len(rows)} 格跑完，没有三段同号的")
    return SkillResult(True, summary=summary, blocks=blocks, feed=summary)


# ------------------------------------------------------------------ sector_relation


def _sector_relation(ctx, p: dict) -> SkillResult:
    sector = str(p.get("sector") or "").strip()
    if not sector:
        raise RuntimeError("要指定板块，比如 sector=\"化工\"")
    keys = _sector_keys(sector)
    if len(keys) < 3:
        raise RuntimeError(f"板块「{sector}」可回测的品种不足 3 个，算不了关系")
    period = str(p.get("period") or "1d_tq")
    lookback = int(p.get("bars") or 500)

    idx = _sector_index(ctx, keys, period)
    if idx is None or len(idx) < 30:
        raise RuntimeError(f"{period} 上凑不出{sector}板块指数（品种数据太短）")
    idx = idx.iloc[-lookback:]
    iret = idx.pct_change().dropna()

    rows, rets = [], {}
    for k in keys:
        try:
            df = _frame(ctx, k, period)
        except Exception:  # noqa: BLE001
            continue
        if df is None or "close" not in df:
            continue
        s = pd.to_numeric(df["close"], errors="coerce").dropna() / 1.0
        # 归一化成和板块指数同一起点，按日期对齐后再算相关
        s = s.reindex(idx.index).ffill()
        if s.notna().sum() < 30:
            continue
        r = s.pct_change().dropna()
        joined = pd.concat([r.rename("sym"), iret.rename("idx")], axis=1).dropna()
        if len(joined) < 30:
            continue
        c = float(joined["sym"].corr(joined["idx"]))
        var = float(joined["idx"].var())
        beta = float(joined["sym"].cov(joined["idx"]) / var) if var > 0 else float("nan")
        vol = float(joined["sym"].std() * np.sqrt(252 * max(1.0, _bpd(period, df))))
        rets[k.rpartition("_")[0]] = joined["sym"]
        rows.append([k.rpartition("_")[0], _f(c), _f(beta), _f(c * c),
                     f"{100 * vol:.1f}%", len(joined)])

    rows.sort(key=lambda r: -float(r[1]) if r[1] != "—" else 0)
    # 品种之间最高的几对相关 —— 「板块内其实分几组」看这个
    pairs = []
    cols = pd.DataFrame(rets)
    if cols.shape[1] >= 2:
        cm = cols.corr()
        for i, a in enumerate(cm.columns):
            for b in cm.columns[i + 1:]:
                v = cm.loc[a, b]
                if v == v:
                    pairs.append((float(v), a, b))
        pairs.sort(reverse=True)

    blocks = [
        B.block_metrics(
            f"{sector}板块 · {period} · 近 {len(idx)} 根",
            [("成分", f"{len(rows)} 个", "flat"),
             ("板块指数区间涨幅", f"{100 * (idx.iloc[-1] / idx.iloc[0] - 1):+.2f}%",
              "good" if idx.iloc[-1] >= idx.iloc[0] else "bad"),
             ("板块波动（年化）", f"{100 * iret.std() * np.sqrt(252 * max(1.0, _bpd(period))):.1f}%",
              "flat")],
        ),
        B.block_table(
            ["品种", "相关性", "beta", "R²（板块解释力）", "年化波动", "样本"],
            rows[:20],
            note="R² 高 = 这个品种基本跟着板块走，板块级信号可以直接推；"
                 "R² 低 = 它有自己的驱动，必须单独建模"),
    ]
    if pairs:
        blocks.append(B.block_table(
            ["相关性", "品种 A", "品种 B"],
            [[_f(v), a, b] for v, a, b in pairs[:10]],
            note="最高的几对：相关 0.9 以上等于同一个风险，选品种时别当两个用"))
    summary = (f"{sector} {period}：{len(rows)} 个品种，"
               f"与板块指数相关最高 {rows[0][0]} {rows[0][1]}、最低 {rows[-1][0]} {rows[-1][1]}")
    return SkillResult(True, summary=summary, blocks=blocks, feed=summary)


def _f(v, nd: int = 2) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if x != x:
        return "—"
    return f"{x:,.{nd}f}"


def _pct(v) -> str:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return "—"
    if x != x:
        return "—"
    return f"{100 * x:+.1f}%"


# ------------------------------------------------------------------ sector_corr


def _daily_returns(ctx, keys, period: str, bars: int, cache: dict) -> dict[str, pd.Series]:
    """每品种一条日频收益率序列。

    🔴 分钟周期必须**先压日频**再算相关：不同周期最后一个 bar 的时刻不同，
    直接对齐会得到稀疏表，相关性被稀释（组合曲线那边踩过同款坑）。
    """
    out = {}
    for k in keys:
        if k in cache:
            df = cache[k]
        else:
            try:
                df = _frame(ctx, k, period)
            except Exception:  # noqa: BLE001 - 单品种缺数据不该拖垮整张表
                df = None
            cache[k] = df
        if df is None or len(df) < 30 or "close" not in df:
            continue
        s = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(s) < 30:
            continue
        if not _is_daily(period):
            s = s.groupby(s.index.normalize()).last()
        if bars and len(s) > bars:
            s = s.iloc[-bars:]
        out[k] = s.pct_change()
    return out


def _is_daily(period: str) -> bool:
    return str(period).startswith("1d") or str(period) in ("2d", "1w")


def _avg_pair_corr(rets: dict[str, pd.Series], min_common: int = 60) -> dict:
    """一组品种的内部相关：平均 / 最高一对 / 最低一对。"""
    names = list(rets)
    if len(names) < 2:
        return {"avg": float("nan"), "hi": (float("nan"), "", ""),
                "lo": (float("nan"), "", ""), "n": 0, "pairs": 0}
    f = pd.DataFrame(rets).sort_index()
    c = f.corr(min_periods=min_common)
    vals = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            v = c.loc[a, b]
            if v == v:
                vals.append((float(v), a.rpartition("_")[0], b.rpartition("_")[0]))
    if not vals:
        return {"avg": float("nan"), "hi": (float("nan"), "", ""),
                "lo": (float("nan"), "", ""), "n": len(names), "pairs": 0}
    vals.sort()
    return {"avg": float(np.mean([v for v, _, _ in vals])),
            "hi": vals[-1], "lo": vals[0], "n": len(names), "pairs": len(vals)}


def _sector_corr(ctx, p: dict) -> SkillResult:
    """板块内部相关性 + 板块之间的相关矩阵。"""
    from ttrade.futures import L2_PARENT, SECTOR_L2, SECTOR_MAP

    level = int(p.get("level") or 2)
    table = SECTOR_L2 if level == 2 else SECTOR_MAP
    period = str(p.get("period") or "1d_tq")
    bars = int(p.get("bars") or 500)
    min_common = int(p.get("min_common") or 60)
    only = _norm_name(p.get("sectors") or "")

    names = list(table)
    if only:
        want = {w.strip() for w in only.replace(",", " ").replace("，", " ").split() if w.strip()}
        names = [n for n in names if n in want]
        if not names:
            raise RuntimeError(f"没有这些板块：{only}（level={level} 有的是：{'、'.join(table)}）")

    cache: dict = {}
    intra, idx_rets = [], {}
    for n in names:
        keys = _sector_keys(n)
        rets = _daily_returns(ctx, keys, period, bars, cache)
        if len(rets) < 1:
            intra.append([n, L2_PARENT.get(n, ""), 0, "—", "—", "—"])
            continue
        st = _avg_pair_corr(rets, min_common)

        def _pair(t):
            return "%s·%s %.2f" % (t[1], t[2], t[0]) if t[1] else "—"

        row = [n]
        if level == 2:
            row.append(L2_PARENT.get(n, ""))
        row += [len(rets), _f(st["avg"]), _pair(st["hi"]), _pair(st["lo"])]
        intra.append(row)
        # 板块指数 = 成员归一化净值的均值（和 sector_relation 同一套）
        idx = _sector_index(ctx, keys, period, min_members=1)
        if idx is not None and len(idx) > 30:
            if not _is_daily(period):
                idx = idx.groupby(idx.index.normalize()).last()
            if bars and len(idx) > bars:
                idx = idx.iloc[-bars:]
            idx_rets[n] = idx.pct_change()

    cols = ["板块", "成分", "内部平均相关", "最高一对", "最低一对"]
    if level == 2:
        cols.insert(1, "大类")
    blocks = [B.block_table(cols, intra, note=(
        "内部平均相关 = 板块内所有品种两两日收益相关的均值。"
        "**这个数低就说明「板块」本身不是一个假设** —— 该板块的横截面扫描结论要打折。"
        "一般 0.3~0.7 正常；>0.8 等于同一品种重复；<0.2 表示这批品种不该放在一个板块里"))]

    # ---- 板块之间
    if len(idx_rets) >= 2:
        c = pd.DataFrame(idx_rets).sort_index().corr(min_periods=min_common)
        pairs = []
        ns = list(c.columns)
        for i, a in enumerate(ns):
            for b in ns[i + 1:]:
                v = c.loc[a, b]
                if v == v:
                    pairs.append((float(v), a, b))
        pairs.sort(reverse=True)
        blocks.append(B.block_table(
            ["相关性", "板块 A", "板块 B"], [[_f(v), a, b] for v, a, b in pairs[:12]],
            note="相关最高的 12 对：>0.8 的两个板块在组合里等于同一个风险，配权时不能各给一份"))
        if len(pairs) > 8:      # 板块少的时候两张表是同一批，别重复贴
            blocks.append(B.block_table(
                ["相关性", "板块 A", "板块 B"],
                [[_f(v), a, b] for v, a, b in sorted(pairs)[:8]],
                note="相关最低的 8 对：这些才是分散收益的来源 —— 加板块要加这种，"
                     "不是加名字不一样但走势一样的东西"))
        hi, lo = pairs[0], pairs[-1]
        i_avg = cols.index("内部平均相关")
        avgs = [float(r[i_avg]) for r in intra if r[i_avg] not in ("—", "nan")]
        summary = (f"{len(intra)} 个板块（level={level}）：内部平均相关中位 "
                   f"{np.median(avgs) if avgs else 0:.2f}；"
                   f"板块间最高 {hi[0]:.2f}（{hi[1]}×{hi[2]}）、最低 {lo[0]:.2f}（{lo[1]}×{lo[2]}）")
    else:
        summary = f"{len(intra)} 个板块（level={level}）内部相关已算出，板块间样本不足"
    return SkillResult(True, summary=summary, blocks=blocks, feed=summary)


# ------------------------------------------------------------------ 注册


register(Tool(
    name="sector_list",
    group="sector",
    label="查板块",
    description="列出期货板块（L1 大类 9 个 / L2 产业板块 16 个）及其成分品种、本地有没有数据",
    risk=RISK_READ,
    inputSchema={
        "type": "object",
        "properties": {
            "sector": {"type": "str", "default": ""},
            "level": {"type": "int", "default": 1, "enum": [1, 2]},
            "period": {"type": "str", "default": "1d_tq"},
        },
    },
    handler=_sector_list,
    max_turns=3,
    prompt=(
        "**查板块**：`sector_list()` 列 9 个大类，`sector_list(level=2)` 列 16 个产业板块"
        "（`化工·聚酯` 这种），`sector_list(sector=\"化工·聚酯\")` 看单个板块的成分和本地数据。\n"
        "两层回答不同问题：**大类**是宏观驱动（做板块指数、品种联动、大类配权），"
        "**产业板块**是产业链（做因子横截面扫描）。问「这个因子能不能推到整个板块」时"
        "必须用 level=2 —— 化工 21 个品种跨聚酯/烯烃芳烃/煤化工/橡胶纸浆四条链，"
        "在 L1 上数「几个品种为正」等于把四条链平均掉。\n"
        "别凭印象报品种 —— 板块归类在 `futures.SECTOR_MAP` / `futures.SECTOR_L2` 里，"
        "和界面的「板块」列同源。"
    ),
))

register(Tool(
    name="sector_scan",
    group="sector",
    label="板块因子扫描",
    description="同一批因子在板块内所有品种上横向跑一遍，看哪些因子是真的板块级、哪些只是单品种巧合",
    risk=RISK_READ,
    max_turns=3,
    max_result_chars=NO_LIMIT,
    inputSchema={
        "type": "object",
        "required": ["sector"],
        "properties": {
            "sector": {"type": "str"},
            "factors": {"type": "list", "default": []},
            "periods": {"type": "list", "default": []},
            "period": {"type": "str", "default": ""},
            "symbols": {"type": "list", "default": []},
            "orient": {"type": "str", "default": "both",
                       "enum": ["high", "low", "both"]},
            "split": {"type": "str", "default": "2024-01-01"},
            "start": {"type": "str", "default": ""},
            "top": {"type": "int", "default": 25},
        },
    },
    handler=_sector_scan,
    prompt=(
        "**板块因子扫描**：`sector_scan(sector=\"化工\", factors=[\"roll_main_oi\"], "
        "periods=[\"4h\"])`。这是「这个因子能不能推到整个板块」的唯一判据 —— "
        "单看一个品种的回测看不出过拟合，要看它在板块内**几个品种上是正的**。\n"
        "口径固定：窗口按 eco（回看约一年）、方向固定 `long_high`/`long_low`（不给 auto）、"
        "IS/OOS 从同一条净值的日期切片。返回三张表：因子跨品种一致性、逐格明细、口径说明。\n"
        "上限一次 12 品种 × 3 周期 × 6 因子 × 2 方向（超了会报错让你收窄）。"
        "先拿 1~2 个周期试，别一上来跑满 —— 分钟线的格子慢得多。\n"
        "结论要说清「几个品种为正 / 共几个」，只说「表现不错」等于没说。"
    ),
))

register(Tool(
    name="sector_relation",
    group="sector",
    label="板块与品种关系",
    description="算板块等权指数，看每个品种跟板块的相关性/beta/解释力，以及品种之间最高相关的几对",
    risk=RISK_READ,
    max_turns=3,
    max_result_chars=NO_LIMIT,
    inputSchema={
        "type": "object",
        "required": ["sector"],
        "properties": {
            "sector": {"type": "str"},
            "period": {"type": "str", "default": "1d_tq"},
            "bars": {"type": "int", "default": 500},
        },
    },
    handler=_sector_relation,
    prompt=(
        "**板块与品种关系**：`sector_relation(sector=\"化工\")`。返回板块等权指数的涨幅/波动、"
        "每个品种的相关系数 / beta / R²（板块解释力），以及品种之间相关最高的几对。\n"
        "用途：R² 高的品种可以直接吃板块级信号；R² 低的必须单独建模；"
        "相关 0.9 以上的两个品种在组合里等于同一个风险，不能当两个独立品种配权。"
        "问「这几个品种是不是一回事」「板块涨它跟不跟」时用它。"
    ),
))

register(Tool(
    name="sector_corr",
    group="sector",
    label="板块相关矩阵",
    description="16 个产业板块（或 9 个大类）的内部平均相关 + 板块之间的相关，用来判「这些板块算几个独立假设」",
    risk=RISK_READ,
    max_turns=3,
    max_result_chars=NO_LIMIT,
    inputSchema={
        "type": "object",
        "properties": {
            "level": {"type": "int", "default": 2, "enum": [1, 2]},
            "period": {"type": "str", "default": "1d_tq"},
            "bars": {"type": "int", "default": 500},
            "min_common": {"type": "int", "default": 60},
            "sectors": {"type": "str", "default": ""},
        },
    },
    handler=_sector_corr,
    prompt=(
        "**板块相关矩阵**：`sector_corr()` 默认按 16 个产业板块算，`level=1` 按 9 个大类算。"
        "返回两张表：①每个板块**内部**的平均两两相关（低 = 这个板块本身不是一个假设，"
        "横截面结论要打折）；②板块**之间**相关最高/最低的各十几对。\n"
        "口径：分钟周期先压日频再算（不同周期最后 bar 时刻不同，直接对齐会稀释相关性）；"
        "板块指数 = 成员归一化净值的等权均值；`bars` 控制回看根数（日线 500 ≈ 两年）。\n"
        "用途：配权前先看这张表 —— 相关 0.8 以上的两个板块在组合里等同一个风险，"
        "配两份等于变相加杠杆；真正能分散的是相关最低那几对。"
        "问「这几个板块是不是一回事」「加板块有用吗」时用它。"
    ),
))
