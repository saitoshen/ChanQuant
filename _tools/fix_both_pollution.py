# -*- coding: utf-8 -*-
"""全品种清理 5m + 1h 污染期的 bar, 供从天勤重下。

用法（**必须关 GUI**）:
    python _tools/fix_both_pollution.py --check     # 只读统计
    python _tools/fix_both_pollution.py --apply     # 删除 + 备份

删除窗口: 2026-11-19 ~ 2026-11-26 (留出一天缓冲, 覆盖整个污染期 11-20/21/23)
备份: bars_pollution_backup_5m / bars_pollution_backup_1h 两张表, 可回滚。
"""
import argparse
import sys
sys.path.insert(0, r"D:/ChanQuant-Dev")
from ttrade import tqstore
import pandas as pd

WIN = ("2026-11-19 00:00:00", "2026-11-26 23:59:59")
BK = {"5m": "bars_pollution_backup_5m", "1h": "bars_pollution_backup_1h"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    with tqstore.connect(read_only=not a.apply) as con:
        for p in ("5m", "1h"):
            cond = f"period='{p}' AND dt >= '{WIN[0]}' AND dt <= '{WIN[1]}'"
            cnt, syms = con.execute(
                f"SELECT count(*), count(DISTINCT symbol||'_'||exchange) FROM bars WHERE {cond}"
            ).fetchone()
            print(f"[{p}] 窗口内 {cnt} 行 / {syms} 品种")
            if a.apply and cnt:
                con.execute(
                    f"CREATE TABLE IF NOT EXISTS {BK[p]} AS SELECT * FROM bars WHERE 1=0")
                con.execute(f"INSERT INTO {BK[p]} SELECT * FROM bars WHERE {cond}")
                con.execute(f"DELETE FROM bars WHERE {cond}")
                print(f"[{p}] 已删除并备份到 {BK[p]}")
        if a.apply:
            # 打印各周期库尾(应回退到 11-18)
            for p in ("5m", "1h"):
                r = con.execute(
                    "SELECT max(dt) FROM bars WHERE period=?", [p]).fetchone()
                print(f"删除后 {p} 库尾: {r[0]}")
    if a.check:
        print("\n只读检查完成。要执行请加 --apply（会先备份）")


if __name__ == "__main__":
    main()
