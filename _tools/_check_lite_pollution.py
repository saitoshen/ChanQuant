# -*- coding: utf-8 -*-
"""只读检查完整库 / 轻量库的 5m、1h 时间轴污染（不写任何库）。

1h 合法 bar 必须 minute==0；5m 合法 bar 必须 minute%5==0。
窗口 2026-11-19 ~ 11-26 覆盖本轮 5m/1h 互相串库的污染期。
"""
import os
import sys

import duckdb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DBS = (
    ("tq 主库", os.path.join(BASE, "data_tq", "tq.duckdb")),
    ("tq_lite", os.path.join(BASE, "data_tq", "tq_lite.duckdb")),
    ("tq_lite_prev", os.path.join(BASE, "data_tq", "tq_lite_20261106_prev.duckdb")),
    ("dist/ChanQuant 随包副本",
     os.path.join(BASE, "dist", "ChanQuant", "_internal", "data_tq", "tq_lite.duckdb")),
)
WIN = ("2026-11-19 00:00:00", "2026-11-26 23:59:59")


def check(name: str, path: str) -> None:
    if not os.path.isfile(path):
        print(f"\n== {name}: 文件不存在 ==")
        return
    con = duckdb.connect(path, read_only=True)
    print(f"\n== {name}  {os.path.getsize(path) / 1e6:.1f} MB ==")
    for p, n, syms, hi in con.execute(
        "SELECT period, count(*), count(DISTINCT symbol), max(dt)"
        " FROM bars GROUP BY 1 ORDER BY 1"
    ).fetchall():
        print(f"  {p:<4} {n:>12,} 行  {syms:>3} 品种  max={hi}")
    for p, bad in (("1h", "minute(dt) <> 0"), ("5m", "minute(dt) % 5 <> 0")):
        c, s, lo, hi = con.execute(
            f"SELECT count(*), count(DISTINCT symbol), min(dt), max(dt) FROM bars"
            f" WHERE period='{p}' AND dt >= '{WIN[0]}' AND dt <= '{WIN[1]}'"
            f" AND {bad}"
        ).fetchone()
        print(f"  [{p}] 窗口内错位 bar: {c:,} 行 / {s} 品种  {lo} ~ {hi}")
    rows = con.execute(
        "SELECT dt, close FROM bars WHERE symbol='A' AND period='1h'"
        " AND dt <= '2026-11-26 23:59:59' ORDER BY dt DESC LIMIT 5"
    ).fetchall()
    print("  A 1h 末 5 根:", rows)
    con.close()


if __name__ == "__main__":
    for nm, pth in DBS:
        check(nm, pth)
