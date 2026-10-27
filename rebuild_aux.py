"""从本地合约明细离线重算 `data_futures_aux` 的期限结构列，不联网。

用途：`contracts.build_curve()` 的选合约口径一改，已有的 aux 文件就过期了 ——
新口径的列根本不存在，对应的因子会静默变成全 NaN。走正常更新流程
（`download_aux.py`）要按交易日重新拉一遍全市场日线，几天时间；而明细本来就在
本地 `data_contracts/`，直接重算几分钟就完事。

只重写曲线文件（`{SYM}_{EX}.csv`），期权文件（`opt_{SYM}_{EX}.csv`）不碰 ——
两者是分开存的，互不影响。

用法：
    python rebuild_aux.py                 # 全部品种
    python rebuild_aux.py RB_SHFE M_DCE   # 只重算指定品种
    python rebuild_aux.py --dry           # 只报告会改什么，不写盘
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

from ttrade import contracts as C
from ttrade.paths import data_dir


def _varieties(raw_dir: str, only: list[str]) -> list[str]:
    if only:
        names = []
        for a in only:
            a = a.replace("/", "_").replace(".csv", "")
            names.append(a if a.endswith(".csv") else a + ".csv")
        return names
    return sorted(
        fn for fn in os.listdir(raw_dir)
        if fn.endswith(".csv") and not fn.startswith("_")
    )


def rebuild_one(fn: str, raw_dir: str, aux_dir: str, basic: pd.DataFrame) -> tuple[int, int] | None:
    """重算一个品种，返回 (行数, 有 o6_ 数据的行数)；不是真品种时返回 None。"""
    raw = pd.read_csv(os.path.join(raw_dir, fn))
    if raw.empty:
        return 0, 0
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"])

    # 只取这个品种用到的那部分合约清单。全量 basic 有一万多行，
    # merge 一遍太浪费；裁到几百行后整个品种一次算完只要几十毫秒。
    sub = basic[basic["ts_code"].isin(set(raw["ts_code"].astype(str)))]
    if sub.empty:
        # 清单里一条都对不上，说明这个文件根本不是可交易的品种：连续合约（IFL）、
        # TAS 指令合约、月均价合约都在 `fetch_basic` 里被剔掉了，它们本来就没有
        # 期限结构，跳过而不是报错。
        return None

    # 整份明细一次算完：build_curve 的分组键里带了 date，按天循环反而更慢
    curve = C.build_curve(raw, sub)
    if curve.empty:
        return 0, 0
    curve = curve.drop_duplicates("date", keep="last").sort_values("date")
    # 列顺序按 CURVE_COLS 固定，方便 diff 和人工核对
    curve = curve[[c for c in C.CURVE_COLS if c in curve.columns]]
    C.write_atomic(os.path.join(aux_dir, fn), curve)
    return len(curve), int(curve["o6_1_close"].notna().sum()) if "o6_1_close" in curve else 0


def main(argv: list[str]) -> int:
    dry = "--dry" in argv
    only = [a for a in argv if not a.startswith("-")]

    raw_dir = data_dir("contracts")
    aux_dir = data_dir("aux")
    basic_path = os.path.join(raw_dir, "_basic.csv")
    if not os.path.isdir(raw_dir) or not os.path.exists(basic_path):
        print(f"找不到合约明细：{raw_dir}")
        print("离线重算需要两样东西 —— 每品种的合约日线 CSV 和 _basic.csv（合约清单）。")
        print("缺的话只能跑 download_aux.py 从 Tushare 重下一遍。")
        return 1

    basic = pd.read_csv(basic_path)
    basic["delist"] = pd.to_datetime(basic["delist"], errors="coerce")
    print(f"合约清单 {len(basic):,} 条，明细目录 {raw_dir}")
    print(f"输出目录 {aux_dir}{'（dry run，不写盘）' if dry else ''}\n")

    names = _varieties(raw_dir, only)
    t0 = time.time()
    ok = bad = skip = 0
    rows = 0
    for i, fn in enumerate(names, 1):
        try:
            if dry:
                have = os.path.exists(os.path.join(aux_dir, fn))
                print(f"[{i}/{len(names)}] {fn}: {'存在' if have else '缺失'}")
                ok += 1
                continue
            got = rebuild_one(fn, raw_dir, aux_dir, basic)
            if got is None:
                skip += 1
                continue
            n, _ = got
            rows += n
            ok += 1
            if i % 20 == 0 or i == len(names):
                print(f"[{i}/{len(names)}] {fn}: {n} 行 · 累计 {rows:,} 行 · "
                      f"{time.time() - t0:.0f}s")
        except Exception as exc:
            bad += 1
            print(f"[{i}/{len(names)}] {fn}: 失败 — {str(exc)[:120]}")

    tail = f"、跳过 {skip} 个非品种文件（连续 / TAS / 月均价）" if skip else ""
    print(f"\n完成 {ok} 个品种{tail}、失败 {bad} 个、共 {rows:,} 行，"
          f"耗时 {time.time() - t0:.0f}s")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
