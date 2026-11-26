# -*- coding: utf-8 -*-
"""统计天勤 contract_bars 2016-2021 的真实合约覆盖（导入后复核用）。"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ttrade import tqstore as S  # noqa: E402


def main() -> None:
    cm = S.connect(read_only=True)
    con = cm.__enter__()
    try:
        print("=== 年度总览 ===", flush=True)
        q = """
        SELECT year(trading_day) AS y,
               count(*) AS n,
               count(DISTINCT trading_day) AS days,
               count(DISTINCT symbol) AS syms,
               round(count(*) * 1.0 / count(DISTINCT trading_day), 1) AS per_day
        FROM contract_bars
        WHERE period='1d' AND trading_day >= DATE '2016-01-01'
          AND trading_day < DATE '2022-01-01'
        GROUP BY y ORDER BY y
        """
        print("year\trows\tdays\tsyms\tper_day", flush=True)
        for r in con.execute(q).fetchall():
            print("\t".join(str(x) for x in r), flush=True)

        print("\n=== 每年每品种行数（行数偏少的品种 = 仍可能缺口）===", flush=True)
        q2 = """
        SELECT year(trading_day) AS y, symbol, exchange,
               count(*) AS n,
               count(DISTINCT trading_day) AS days,
               round(count(*) * 1.0 / count(DISTINCT trading_day), 2) AS per_day
        FROM contract_bars
        WHERE period='1d' AND trading_day >= DATE '2016-01-01'
          AND trading_day < DATE '2022-01-01'
        GROUP BY y, symbol, exchange
        HAVING per_day < 3
        ORDER BY y, per_day, symbol
        """
        rows = con.execute(q2).fetchall()
        print(f"每年每品种每日合约数 < 3 的组合共 {len(rows)} 个", flush=True)
        for r in rows[:120]:
            print("\t".join(str(x) for x in r), flush=True)
    finally:
        cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
