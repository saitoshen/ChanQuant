"""衍生数据的批量构建：真实合约明细 + 期限结构 + 期权波动率。

## 为什么合成一个循环

两类衍生数据都要先拿「当天全市场的期货收盘价」：

- 期限结构：需要各月份合约的价格
- 期权 IV：需要标的期货合约的价格（Black-76 里的 F）

如果分别下载，`fut_daily` 每天会被请求两次。20 年 4800 个交易日就是 4800 次
多余的请求（约 40 分钟）。所以放在同一个日循环里：**每天一次 fut_daily、
一次 opt_daily**，同时喂给两条聚合链路。

## 落盘节奏

按年分批。一年的原始合约约 25 万行，攒在内存里没问题；写盘时和已有文件按
(date, ts_code) 合并去重，所以中途取消、重跑、补历史都不会产生重复行。
期限结构和期权摘要更小，同样按年合并。

## 进度

`data_futures_aux/_manifest.json` 记两个日期列表：`dates`（合约已处理）与
`opt_dates`（期权已处理）。期权 2017 年才有数据，和合约的覆盖区间不同，
所以必须分开记 —— 共用一个列表会导致期权永远「已处理」而实际是空的。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import pandas as pd

from . import contracts as C
from . import futures as fut
from . import options as O
from .paths import data_dir as resolve_data_dir


@dataclass
class BuildResult:
    want_contracts: bool = True
    want_options: bool = True
    days: int = 0
    contract_rows: int = 0
    curve_varieties: int = 0
    opt_varieties: int = 0
    opt_days: int = 0
    failed: int = 0
    cancelled: bool = False
    first: str = ""
    last: str = ""
    elapsed: float = 0.0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        head = "已取消" if self.cancelled else ("失败" if self.failed and not self.days else "完成")
        parts = [head, f"{self.days} 个交易日"]
        if self.want_contracts:
            parts.append(f"合约 {self.contract_rows:,} 行 / {self.curve_varieties} 个品种")
        if self.want_options:
            parts.append(f"期权 {self.opt_varieties} 个品种")
        if self.first:
            parts.append(f"{self.first} ~ {self.last}")
        if self.failed:
            parts.append(f"{self.failed} 天失败")
        parts.append(f"耗时 {self.elapsed:.0f}s")
        return " · ".join(parts)


def _root(data_root: str | None) -> str:
    return data_root or os.path.dirname(resolve_data_dir("cn"))


def build(
    days: list[str],
    data_root: str | None = None,
    want_contracts: bool = True,
    want_options: bool = True,
    keep_raw_contracts: bool = True,
    keep_raw_options: bool = False,
    progress=None,
    cancel=None,
) -> BuildResult:
    """按交易日构建衍生数据。`days` 为 YYYYMMDD 升序列表，调用方负责筛掉已完成的。"""
    t0 = time.time()
    res = BuildResult(want_contracts=want_contracts, want_options=want_options)
    if not days:
        res.elapsed = time.time() - t0
        return res

    root = _root(data_root)
    raw_dir = os.path.join(root, "data_contracts")
    aux_dir = os.path.join(root, "data_futures_aux")
    opt_raw_dir = os.path.join(root, "data_options_raw")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(aux_dir, exist_ok=True)

    pro = fut.get_pro()
    fbasic = C.fetch_basic(pro, cache_path=os.path.join(raw_dir, "_basic.csv")) \
        if want_contracts else None
    obasic = O.fetch_basic(pro, cache_path=os.path.join(raw_dir, "_opt_basic.csv")) \
        if want_options else None

    man = C.load_manifest(aux_dir)
    done = set(man.get("dates") or [])
    done_opt = set(man.get("opt_dates") or [])

    total = len(days)
    by_year: dict[str, list[str]] = {}
    for d in days:
        by_year.setdefault(d[:4], []).append(d)

    pos = {d: i for i, d in enumerate(days)}
    for year in sorted(by_year):
        if cancel and cancel():
            res.cancelled = True
            break

        raw_rows: list[pd.DataFrame] = []
        curves: list[pd.DataFrame] = []
        opts: list[pd.DataFrame] = []

        for day in by_year[year]:
            if cancel and cancel():
                res.cancelled = True
                break
            if progress:
                progress(pos[day], total, f"{year} · {day} · 已完成 {res.days} 天")

            try:
                fut_day = C.fetch_day(pro, day)
            except Exception as exc:
                res.failed += 1
                res.errors.append(f"{day} 期货: {str(exc)[:90]}")
                continue

            if want_contracts:
                if not fut_day.empty:
                    res.contract_rows += len(fut_day)
                    if keep_raw_contracts:
                        raw_rows.append(fut_day)
                    curves.append(C.build_curve(fut_day, fbasic))
                done.add(day)

            if want_options:
                if day < O.OPT_START:
                    # 商品期权 2017 年才上市，更早的日期不必请求 opt_daily
                    done_opt.add(day)
                else:
                    try:
                        opt_day = O.fetch_day(pro, day)
                    except Exception as exc:
                        res.failed += 1
                        res.errors.append(f"{day} 期权: {str(exc)[:90]}")
                        continue
                    if opt_day.empty:
                        # 该日无期权数据（上市初期、节假日），记入已完成，
                        # 否则每次更新都会把这些空白重跑一遍
                        done_opt.add(day)
                    else:
                        # 用结算价而不是收盘价，原因见 contracts.settle_map 的注释：
                        # 期权结算价和期货结算价是同一次结算的产物，用收盘价会
                        # 让 call / put 解出的 IV 系统性地对不上。
                        futmap = C.settle_map(fut_day)
                        met = O.day_metrics(day, opt_day, obasic, futmap)
                        if not met.empty:
                            opts.append(met)
                            res.opt_days += 1
                        if keep_raw_options:
                            O.append_raw(opt_raw_dir, opt_day, obasic)
                        done_opt.add(day)

            res.days += 1
            time.sleep(C.SLEEP)

        # ---- 落盘（每年一次）
        if want_contracts and keep_raw_contracts and raw_rows:
            allraw = pd.concat(raw_rows, ignore_index=True)
            key = pd.DataFrame({
                "symbol": allraw["ts_code"].map(_symbol),
                "exchange": allraw["ts_code"].map(_exchange),
            })
            for (sym, ex), g in allraw.groupby([key["symbol"], key["exchange"]], sort=False):
                C.merge_write(os.path.join(raw_dir, f"{sym}_{ex}.csv"),
                              g[C.RAW_COLS], ["date", "ts_code"])

        if want_contracts:
            res.curve_varieties = max(res.curve_varieties, _flush(curves, aux_dir, ""))
        if want_options:
            res.opt_varieties = max(res.opt_varieties, _flush(opts, aux_dir, "opt_"))

        C.save_manifest(aux_dir, {"dates": sorted(done), "opt_dates": sorted(done_opt)})
        if progress:
            progress(pos[by_year[year][-1]] + 1, total, f"{year} 年已落盘")

    C.save_manifest(aux_dir, {"dates": sorted(done), "opt_dates": sorted(done_opt)})
    span = sorted(done | done_opt)
    if span:
        res.first, res.last = span[0], span[-1]
    res.elapsed = time.time() - t0
    if progress:
        progress(total, total, "完成")
    return res


def _flush(frames: list[pd.DataFrame], aux_dir: str, prefix: str) -> int:
    """把一批聚合结果按品种合并写入 aux 目录，返回品种数。"""
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return 0
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    n = 0
    for (sym, ex), g in df.groupby(["symbol", "exchange"], sort=False):
        g = g.drop(columns=["symbol", "exchange"])
        C.merge_write(os.path.join(aux_dir, f"{prefix}{sym}_{ex}.csv"), g, ["date"])
        n += 1
    return n


def _symbol(ts_code: str) -> str:
    import re

    return re.sub(r"\d+$", "", str(ts_code).split(".")[0])


def _exchange(ts_code: str) -> str:
    suffix = str(ts_code).split(".")[-1]
    return C.SUFFIX_EX.get(suffix, suffix)


def pending_days(
    data_root: str | None = None,
    want_options: bool = False,
    start: str = "",
    end: str = "",
) -> list[str]:
    """本地行情里有、但衍生数据还没处理过的交易日。"""
    root = _root(data_root)
    aux_dir = os.path.join(root, "data_futures_aux")
    all_days = C.trading_days(os.path.join(root, "data_futures"), start, end)
    man = C.load_manifest(aux_dir)
    done = set(man.get("opt_dates" if want_options else "dates") or [])
    return [d for d in all_days if d not in done]
