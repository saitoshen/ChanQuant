"""真实期货合约日线与期限结构。

## 为什么要真实合约

`data_futures/` 里存的是**主力连续**合约，换月处靠复权因子拼平。它能回答
「这个品种过去涨了多少」，但回答不了「近月和远月谁贵」——而后者才是展期
收益的来源。展期因子必须用**同一天、不同到期日**的多个合约算，所以需要把
全部真实合约拉下来。

## 两种产物

| 目录 | 内容 | 体积 | 用途 |
|---|---|---|---|
| `data_contracts/` | 每品种一个 CSV，含该品种全部合约的全部日线 | 约 130MB | 研究原料、可复算期限结构 |
| `data_futures_aux/` | 每品种每日的近/次近/第三近合约价格与到期天数 | 约 20MB | 展期因子直接读它 |

原始明细体积太大（进 git 会让仓库膨胀，打进安装包也没必要），因此**只把
聚合后的期限结构随包分发**。想复核某一天的曲线，从 `data_contracts/` 重算即可。

## 取数方式：按日横截面，不按合约

Tushare 支持 `fut_daily(trade_date=...)` 一次返回当天全市场。按合约逐个拉要
11000+ 次请求（约 80 分钟），按日只要 4800 次（约 40 分钟），而且横截面正好
是算期限结构需要的东西——同一天的近月和远月天然对齐，不用再做日期 join。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from . import futures as fut
from .paths import data_dir as resolve_data_dir

PAGE = 5000
SLEEP = 0.05            # 每两次请求之间的自然间隔，实测接口约 120 次/分钟
MIN_DTE = 5             # 距最后交易日不足 5 天的合约不参与（临近交割价格失真）
MIN_OI = 1.0            # 持仓量为 0 的合约不参与（挂而不交易的僵尸合约）

# 除了「按到期日取最近三个」，再额外产出两套「先按持仓量筛活跃、再按到期日取最近三个」。
# 起因是主力合约月份是跳跃的：螺纹/热卷/铁矿是 1/5/10，豆粕/棕榈是 1/5/9，所以
# 「最近三个到期」经常全落在冷门月上（螺纹有七成的日子主力根本不在 c1/c2/c3 里，
# c1 的持仓量只占全品种的 0.1%）。筛活跃能把这种情况滤掉。
#
# 为什么是 3 和 6 两档：OI 前 3 太紧（近月合约可能被整体排除，丢掉「近端紧张」
# 这个核心信号），前 8 太松（冷门合约又混回来，退化到不筛）。实测 12 个品种上
# 前 6 的均值 IC 最高。但全 81 品种上前 6 只改善 43% 的品种、且方向一致性下降，
# 所以**不做全局替换**，三套并列，让用户在因子分析页自己比。
N_ACTIVE = (3, 6)

# ts_code 后缀 → 交易所代码
SUFFIX_EX = {"SHF": "SHFE", "DCE": "DCE", "ZCE": "CZCE", "CFX": "CFFEX",
             "INE": "INE", "GFE": "GFEX"}

RAW_COLS = ["date", "ts_code", "open", "high", "low", "close", "settle",
            "volume", "amount", "open_interest"]

# 三套期限结构的列。前缀约定：`c` = 按到期日（原口径），`o3_` / `o6_` = 先筛
# 持仓量前 3 / 前 6 再按到期日。下游因子层统一拼成 `{pre}1_close` 这种形式，
# 所以 `c` + `1_close` = `c1_close`，`o3_` + `1_close` = `o3_1_close`。
CURVE_COLS = ["date",
              "c1_code", "c1_close", "c1_dte", "c1_oi",
              "c2_code", "c2_close", "c2_dte", "c2_oi",
              "c3_code", "c3_close", "c3_dte", "c3_oi"]
for _n in N_ACTIVE:
    for _i in (1, 2, 3):
        CURVE_COLS += [f"o{_n}_{_i}_close", f"o{_n}_{_i}_dte", f"o{_n}_{_i}_oi"]


class Cancelled(Exception):
    """用户取消。与 futures.Cancelled 分开定义，避免循环依赖时名字丢失。"""


# ------------------------------------------------------------------ 基础信息
def fetch_basic(pro=None, cache_path: str | None = None, refresh: bool = False) -> pd.DataFrame:
    """全部**真实**合约的静态信息（`fut_type=1`，不含主力/连续）。

    返回列：ts_code / symbol / name / exchange / fut_code / list_date / delist。
    `fut_code` 是品种代码（CU、M、SR…），也是期限结构的分组键。

    这份表 11000 多行，每次下载都重新拉一遍纯属浪费，默认缓存到
    `data_contracts/_basic.csv`；合约会陆续上市，`refresh=True` 时强制重拉。
    """
    pro = pro or fut.get_pro()
    if cache_path and not refresh and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
            if len(df) > 1000:
                df["delist"] = pd.to_datetime(df["delist"], errors="coerce")
                return df
        except Exception:
            pass

    frames: list[pd.DataFrame] = []
    for ex in ("SHFE", "DCE", "CZCE", "CFFEX", "INE", "GFEX"):
        offset = 0
        while True:
            try:
                d = pro.fut_basic(exchange=ex, fut_type="1", offset=offset, limit=PAGE)
            except Exception:
                break
            if d is None or d.empty:
                break
            frames.append(d)
            if len(d) < PAGE:
                break
            offset += PAGE
            time.sleep(SLEEP)
    if not frames:
        raise RuntimeError("拉取合约清单失败：fut_basic 没有返回任何数据")

    b = pd.concat(frames, ignore_index=True).drop_duplicates("ts_code")
    b["delist"] = pd.to_datetime(b["delist_date"], format="%Y%m%d", errors="coerce")
    keep = ["ts_code", "symbol", "name", "exchange", "fut_code", "list_date", "delist"]
    b = b[[c for c in keep if c in b.columns]].copy()
    # 月均价、TAS 指令等不是独立可交易合约
    b = b[~b["name"].astype(str).str.contains("月均价|TAS", na=False)]
    b = b[b["fut_code"].notna()]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        b.to_csv(cache_path, index=False, encoding="utf-8-sig")
    return b


def trading_days(data_dir: str, start: str = "", end: str = "") -> list[str]:
    """交易日历，直接取本地主力连续数据里出现过的日期。

    不调 `trade_cal` 是为了保证**衍生数据和行情数据严格同日**：如果日历多出
    一天而行情表没有，算出来的因子会多一行对不上 K 线；少一天则出现空洞。
    行情表就是唯一真相。
    """
    dates: set[str] = set()
    if not os.path.isdir(data_dir):
        return []
    for fn in os.listdir(data_dir):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        try:
            col = pd.read_csv(os.path.join(data_dir, fn), usecols=["date"])["date"]
        except Exception:
            continue
        dates.update(pd.to_datetime(col, errors="coerce").dropna().dt.strftime("%Y%m%d"))
    out = sorted(d for d in dates if d)
    if start:
        out = [d for d in out if d >= start]
    if end:
        out = [d for d in out if d <= end]
    return out


# ------------------------------------------------------------------ 单日取数
def fetch_day(pro, day: str, retries: int = 3) -> pd.DataFrame:
    """某一天全市场的真实合约行情。连续合约（`A.DCE` 这种没有数字的）剔掉。"""
    last = None
    for attempt in range(retries):
        try:
            df = pro.fut_daily(trade_date=day)
            break
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    else:
        raise RuntimeError(f"{day} 取数失败：{last}")

    if df is None or df.empty:
        return pd.DataFrame(columns=RAW_COLS)
    # 真实合约的代码里一定带数字（CU2407）；主力连续是 CU.SHF
    df = df[df["ts_code"].astype(str).str.contains(r"\d", regex=True)].copy()
    if df.empty:
        return pd.DataFrame(columns=RAW_COLS)
    out = pd.DataFrame({
        "date": pd.to_datetime(df["trade_date"], format="%Y%m%d"),
        "ts_code": df["ts_code"].astype(str),
        "open": pd.to_numeric(df.get("open"), errors="coerce"),
        "high": pd.to_numeric(df.get("high"), errors="coerce"),
        "low": pd.to_numeric(df.get("low"), errors="coerce"),
        "close": pd.to_numeric(df.get("close"), errors="coerce"),
        "settle": pd.to_numeric(df.get("settle"), errors="coerce"),
        "volume": pd.to_numeric(df.get("vol"), errors="coerce"),
        "amount": pd.to_numeric(df.get("amount"), errors="coerce"),
        "open_interest": pd.to_numeric(df.get("oi"), errors="coerce"),
    })
    return out


def settle_map(df: pd.DataFrame) -> dict[str, float]:
    """ts_code → 结算价（结算价缺失时退回收盘价）。

    反解期权 IV 时**必须**用标的期货的结算价，不能用收盘价：国内商品期权的
    结算价是交易所按「标的期货结算价 + 波动率」统一算出来的，两者是同一次
    结算的产物。实测 2017-12-29 豆粕，用 put-call 平价反解出的隐含远期正好
    等于 M1803 的结算价 2852，而当天收盘价是 2828 —— 差 0.85%，看起来不大，
    但会让同一行权价的 call 和 put 解出相差近 10 个 IV 点，二次拟合于是给出
    一条假的高曲率微笑（曲率 19，正常值在 0.3~1.5）。

    结算价为 0 的情况存在于早期僵尸合约，退回收盘价。
    """
    if df is None or df.empty:
        return {}
    settle = pd.to_numeric(df["settle"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    px = settle.where(settle > 0, close)
    return dict(zip(df["ts_code"], px))


def build_curve(day_df: pd.DataFrame, basic: pd.DataFrame) -> pd.DataFrame:
    """把一天的横截面压成「每品种一条期限结构」，同时产出三套口径。

    选合约的规则：按剩余到期天数排序，取最近的三个**有持仓、未临近交割**的合约。
    不用「成交量最大的三个」——展期收益衡量的是滚动持有成本，对应的就是
    即将到期的那几个合约，用最活跃的反而会跳过真正要滚掉的那一腿。

    另有两套「先按持仓量筛活跃、再按到期日取三腿」的口径（`o3_` / `o6_`），
    动机见 `N_ACTIVE` 上方的注释。

    入参可以是**单日**（在线更新时的正常用法），也可以是**多日**（离线重算）：
    分组键里带了 `date`，两种输入的结果逐位一致。多日一次算完比按天循环快得多 ——
    后者光 pandas 的固定开销就要 8ms/天，4800 个交易日就是 40 秒一个品种。
    """
    if day_df.empty:
        return pd.DataFrame(columns=CURVE_COLS)
    m = day_df.merge(
        basic[["ts_code", "fut_code", "exchange", "delist"]], on="ts_code", how="inner"
    )
    m["close"] = pd.to_numeric(m["close"], errors="coerce")
    m["oi"] = pd.to_numeric(m["open_interest"], errors="coerce").fillna(0.0)
    m["dte"] = (m["delist"] - m["date"]).dt.days
    m = m[(m["close"] > 0) & (m["oi"] >= MIN_OI) & (m["dte"] >= MIN_DTE)]
    if m.empty:
        return pd.DataFrame(columns=CURVE_COLS)

    def _fill(rec: dict, pre: str, sub: pd.DataFrame, with_code: bool) -> None:
        # 用数组遍历而不是 iterrows()：离线重算一个品种要写三万多行，
        # iterrows 每次都要现造一个 Series，光这一项就能占到总耗时的一大半。
        n = len(sub)
        codes = sub["ts_code"].to_numpy() if with_code else None
        close = sub["close"].to_numpy(dtype=float)
        dte = sub["dte"].to_numpy(dtype=float)
        oi = sub["oi"].to_numpy(dtype=float)
        for i in range(n):
            j = i + 1
            if with_code:
                rec[f"{pre}{j}_code"] = codes[i]
            # 价格保留 2 位、持仓量取整：aux 文件要随包分发，
            # 全精度浮点会让 CSV 体积翻一倍而信息量几乎不变
            rec[f"{pre}{j}_close"] = round(float(close[i]), 2)
            rec[f"{pre}{j}_dte"] = int(dte[i])
            rec[f"{pre}{j}_oi"] = int(round(float(oi[i])))

    keys = ["date", "fut_code", "exchange"]

    def _top(cols: list[str], asc: list[bool], n: int) -> pd.DataFrame:
        """每个（日期, 品种）取前 n 行，最后一律按到期日升序排好。

        为什么不逐组 `sort_values`：一个品种 4800 个交易日就是 4800 次组内排序，
        每组 2ms，累计十几秒。改成「先整体稳定排序、再按组取前 n 行」结果完全
        一样（稳定排序保证同键行的相对顺序不变），快两个数量级。
        """
        d = m.sort_values(keys + cols, ascending=[True] * 3 + asc, kind="mergesort")
        d = d.groupby(keys, sort=False).head(n)
        # 筛活跃口径是「先按持仓量挑候选、再回到按到期日排」，所以要再排一次；
        # 原口径本来就按 dte 排好，这一步是空操作。
        d = d.sort_values(keys + ["dte"], kind="mergesort")
        return d.groupby(keys, sort=False).head(3)

    # 只有一个合约的组算不出价差，直接跳过
    counts = m.groupby(keys, sort=False).size()
    keep = set(counts[counts >= 2].index)

    recs: dict[tuple, dict] = {}

    def _put(pre: str, d: pd.DataFrame, with_code: bool) -> None:
        for (day, fc, ex), g in d.groupby(keys, sort=False):
            key = (day, fc, ex)
            if key not in keep:
                continue
            rec = recs.get(key)
            if rec is None:
                rec = recs[key] = {"date": day, "symbol": fc, "exchange": ex}
            _fill(rec, pre, g, with_code)

    # 原口径：纯按到期日。c*_code 留着 —— 排查「选进来的是哪张合约」时要用。
    _put("c", _top(["dte", "close"], [True, True], 3), True)
    # 筛活跃口径：先按持仓量降序取前 N（同量时近月优先），再按到期日排。
    # 这里不写 code 列：两组各三腿，省下 6 列能让随包分发的 aux 小一圈，
    # 真要查是哪张合约，用日期去 data_contracts 明细里对即可。
    for n in N_ACTIVE:
        _put(f"o{n}_", _top(["oi", "dte"], [False, True], n), False)

    rows = list(recs.values())
    if not rows:
        return pd.DataFrame(columns=CURVE_COLS)
    out = pd.DataFrame(rows)
    for c in CURVE_COLS:
        if c not in out.columns:
            out[c] = np.nan
    # symbol/exchange 保留在结果里：下游按品种分组写文件要用。
    # （CURVE_COLS 只是「数据列」，不含这两个分组键）
    return out[["symbol", "exchange"] + CURVE_COLS]


# ------------------------------------------------------------------ 落地
def write_atomic(path: str, df: pd.DataFrame) -> None:
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def merge_write(path: str, new: pd.DataFrame, keys: list[str]) -> int:
    """把新行合并进已有文件（同键以新为准），返回合并后的行数。

    日期列必须统一成 datetime 再拼：从 CSV 读回来的是字符串，内存里的是
    Timestamp，混在一起排序会直接抛 `'<' not supported between Timestamp and str`。
    """
    new = new.copy()
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            old = pd.read_csv(path)
        except Exception:
            old = pd.DataFrame()
        if not old.empty:
            new = pd.concat([old, new], ignore_index=True, sort=False)
    if "date" in new.columns:
        new["date"] = pd.to_datetime(new["date"], errors="coerce")
        new = new.dropna(subset=["date"])
    if keys:
        new = new.drop_duplicates(subset=keys, keep="last")
    new = new.sort_values(keys).reset_index(drop=True)
    write_atomic(path, new)
    return len(new)


def load_manifest(aux_dir: str) -> dict:
    path = os.path.join(aux_dir, "_manifest.json")
    if not os.path.exists(path):
        return {"dates": []}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            return {"dates": []}
        data.setdefault("dates", [])
        return data
    except Exception:
        return {"dates": []}


def save_manifest(aux_dir: str, man: dict) -> None:
    path = os.path.join(aux_dir, "_manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(man, fh, ensure_ascii=False)
    os.replace(tmp, path)


def rebuild_aux(data_root: str | None = None) -> int:
    """从 `data_contracts/` 的原始明细重算期限结构（不需要联网）。

    用于：早期只下了原始数据、或改了 `build_curve` 的选合约规则之后重算。
    返回写出的品种数。
    """
    root = data_root or os.path.dirname(resolve_data_dir("cn"))
    raw_dir = os.path.join(root, "data_contracts")
    aux_dir = os.path.join(root, "data_futures_aux")
    os.makedirs(aux_dir, exist_ok=True)
    basic = fetch_basic(cache_path=os.path.join(raw_dir, "_basic.csv"))
    n = 0
    for fn in sorted(os.listdir(raw_dir)):
        if not fn.endswith(".csv") or fn.startswith("_"):
            continue
        df = pd.read_csv(os.path.join(raw_dir, fn))
        if df.empty:
            continue
        df["date"] = pd.to_datetime(df["date"])
        parts = [build_curve(g, basic) for _, g in df.groupby("date", sort=True)]
        curve = pd.concat([p for p in parts if not p.empty], ignore_index=True)
        if curve.empty:
            continue
        stem = fn[:-4]
        merge_write(os.path.join(aux_dir, f"{stem}.csv"),
                     curve.drop(columns=["symbol", "exchange"]), ["date"])
        n += 1
    return n
