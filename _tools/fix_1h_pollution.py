# -*- coding: utf-8 -*-
"""修复 1h 库被 5m bar 污染的问题（2026-11-20~11-23）。

用法（**必须先关 GUI**）：
    # 第 1 步：只删非整点脏 bar（推荐先做这个，风险最低）
    C:/ProgramData/anaconda3/python.exe _tools/fix_1h_pollution.py --clean

    # 第 2 步：删掉受影响日期的整点 bar，让 last_dt 回退，随后用 download_tq 重下
    C:/ProgramData/anaconda3/python.exe _tools/fix_1h_pollution.py --rollback-dates

    # 查看当前污染情况（只读）
    C:/ProgramData/anaconda3/python.exe _tools/fix_1h_pollution.py --check

原理：
  1h 表里混进了 minute!=0 的行，它们和 5m 库逐根一致 —— 是 3.9.8 之前
  订阅键没带周期时，5m 序列被误写进 1h 表。整点的 1h bar 本身正确
  （实测 = 12 根 5m 合成值），所以 --clean 只删非整点行即可修复出场箱体。
  --rollback-dates 额外把 11-20~11-25 的整点 bar 也删掉，好让下载器
  从 last_dt(11-19) 起重下这段（INSERT OR REPLACE 覆盖）。

删之前会把被删的行原样备份到 bars_1h_pollution_backup 表，可回滚。
"""
import argparse
import sys
sys.path.insert(0, r"D:/ChanQuant-Dev")

from ttrade import tqstore
import pandas as pd

BACKUP = "bars_1h_pollution_backup"


def check():
    with tqstore.connect(read_only=True) as con:
        n = con.execute(
            "SELECT count(*) FROM bars WHERE period='1h' "
            "AND CAST(strftime('%M', dt) AS INTEGER) <> 0"
        ).fetchone()[0]
        byday = con.execute(
            "SELECT CAST(dt AS DATE) d, count(*) n FROM bars "
            "WHERE period='1h' AND CAST(strftime('%M', dt) AS INTEGER) <> 0 "
            "GROUP BY d ORDER BY d"
        ).fetchall()
        bysym = con.execute(
            "SELECT symbol, exchange, count(*) n FROM bars "
            "WHERE period='1h' AND CAST(strftime('%M', dt) AS INTEGER) <> 0 "
            "GROUP BY symbol, exchange ORDER BY n DESC"
        ).fetchall()
    print(f"1h 库非整点脏 bar 共 {n} 根")
    print("按日期：")
    for d, c in byday:
        print(f"  {d}: {c}")
    print(f"按品种（{len(bysym)} 个）：")
    for s, e, c in bysym[:40]:
        print(f"  {s}_{e}: {c}")
    return n


def ensure_backup(con):
    con.execute(
        f"CREATE TABLE IF NOT EXISTS {BACKUP} AS "
        "SELECT * FROM bars WHERE 1=0"
    )


def clean(rollback_dates: bool):
    with tqstore.connect(read_only=False) as con:
        ensure_backup(con)
        if rollback_dates:
            cond = ("period='1h' AND dt >= '2026-11-20' AND dt <= '2026-11-25 23:59:59'")
            label = "11-20~11-25 整段 1h bar（含脏 bar）"
        else:
            cond = ("period='1h' AND CAST(strftime('%M', dt) AS INTEGER) <> 0")
            label = "1h 库非整点脏 bar"
        cnt = con.execute(f"SELECT count(*) FROM bars WHERE {cond}").fetchone()[0]
        if not cnt:
            print(f"没有需要删除的行（{label}）")
            return
        con.execute(f"INSERT INTO {BACKUP} SELECT * FROM bars WHERE {cond}")
        con.execute(f"DELETE FROM bars WHERE {cond}")
        print(f"已删除 {cnt} 行（{label}），原样备份在 {BACKUP} 表")
        # 打印每个品种删除后的 last_dt
        rows = con.execute(
            "SELECT symbol, exchange, max(dt) FROM bars WHERE period='1h' "
            "GROUP BY symbol, exchange ORDER BY symbol"
        ).fetchall()
        print("删除后各品种 1h 库尾（前 10）：")
        for s, e, last in rows[:10]:
            print(f"  {s}_{e}: {last}")
    print("\n完成。下一步：关 GUI 后从天勤下载补库：")
    print("  python download_tq.py --main 1h --symbols <受影响品种> --days 10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--clean", action="store_true", help="删除非整点脏 bar")
    ap.add_argument("--rollback-dates", action="store_true",
                    help="删除 11-20~11-25 整段 1h bar（供重新下载覆盖）")
    a = ap.parse_args()
    if a.check:
        check()
    elif a.clean:
        clean(False)
    elif a.rollback_dates:
        clean(True)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
