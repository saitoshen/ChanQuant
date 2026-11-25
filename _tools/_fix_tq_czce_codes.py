# -*- coding: utf-8 -*-
"""修复已落库的郑商所合约代码十年错位（按每根 bar 的交易日重新消歧）。

背景：tqfetch.write_contracts 曾用 to_ts(c, date.today()) 把天勤 3 位月份
统一按今天消歧，2016 年的 MA701 被写成 MA2701。这里按交易日的年份重算，
只对「new != old」的行做原子替换，替换前把受影响行导出 gzip 备份。
"""
import os
import sys
import datetime as dt

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ttrade import tqstore  # noqa: E402
from ttrade.tqsdk_symbols import to_ts  # noqa: E402

BAK = os.path.join(
    ROOT, "reports",
    "tq_czce_code_fix_backup_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S") + ".csv.gz",
)


def build_map(con) -> pd.DataFrame:
    rows = con.execute(
        "SELECT contract, year(trading_day) AS y FROM contract_bars"
        " WHERE exchange='CZCE' GROUP BY 1, 2"
    ).df()
    out = []
    for code, year in rows.itertuples(index=False):
        body, suffix = str(code).rsplit(".", 1)
        if not body[-4:].isdigit():
            continue
        prod, short = body[:-4], body[-3:]
        try:
            fixed = to_ts(f"CZCE.{prod}{short}", f"{int(year)}-06-30")
        except Exception:
            continue
        if fixed != code:
            out.append((code, int(year), fixed))
    return pd.DataFrame(out, columns=["old", "y", "new"])


def main() -> None:
    con_cm = tqstore.connect()
    con = con_cm.__enter__()
    try:
        m = build_map(con)
        print(f"错位映射 {len(m)} 组（合约, 年份）", flush=True)
        if m.empty:
            print("无需修复")
            return
        con.register("_m", m)
        n = con.execute(
            "SELECT count(*) FROM contract_bars b JOIN _m m"
            " ON b.contract=m.old AND year(b.trading_day)=m.y"
        ).fetchone()[0]
        print(f"受影响行数 {n}", flush=True)
        if not n:
            return

        affected = con.execute(
            "SELECT b.*, m.new AS fixed_contract FROM contract_bars b"
            " JOIN _m m ON b.contract=m.old AND year(b.trading_day)=m.y"
        ).df()
        os.makedirs(os.path.dirname(BAK), exist_ok=True)
        affected.to_csv(BAK, index=False, compression="gzip")
        print(f"备份已写入 {BAK}（{os.path.getsize(BAK) / 1024 ** 2:.1f} MiB）", flush=True)

        old = affected[["symbol", "exchange", "period", "contract", "dt"]].drop_duplicates()
        new = affected.drop(columns=["fixed_contract"]).copy()
        new["contract"] = affected["fixed_contract"].to_numpy()

        con.register("_fix_old", old)
        con.register("_fix_new", new)
        before = con.execute("SELECT count(*) FROM contract_bars").fetchone()[0]
        con.execute(
            "DELETE FROM contract_bars WHERE (symbol, exchange, period, contract, dt) IN"
            " (SELECT symbol, exchange, period, contract, dt FROM _fix_old)"
        )
        con.execute("INSERT OR REPLACE INTO contract_bars SELECT * FROM _fix_new")
        after = con.execute("SELECT count(*) FROM contract_bars").fetchone()[0]
        print(f"总行数 {before} -> {after}（差 {after - before}）", flush=True)
        if after != before:
            raise RuntimeError("行数变了，修复异常")
        left = con.execute(
            "SELECT count(*) FROM contract_bars b JOIN _m m"
            " ON b.contract=m.old AND year(b.trading_day)=m.y"
        ).fetchone()[0]
        print(f"残留错位行 {left}", flush=True)
        if left:
            raise RuntimeError("仍有错位行")
    finally:
        con_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
