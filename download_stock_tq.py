"""从天勤下载 A 股股票行情。

写 `data_tq/stock.duckdb`（和期货 tq.duckdb、期权 option.duckdb 分开）。

用法：

    python download_stock_tq.py --stats
    python download_stock_tq.py --periods 1d                    # 全股票日线（几小时）
    python download_stock_tq.py --periods 1d,1h                 # 日线 + 1h
    python download_stock_tq.py --periods 1d --exchanges SSE    # 只沪市
    python download_stock_tq.py --periods 5m                    # 5m 只取最近一年
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import stockfetch as SF  # noqa: E402
from ttrade import stockstore as SS  # noqa: E402


def _print_stats() -> None:
    st = SS.stats()
    if not st["exists"]:
        print("还没有股票库（data_tq/stock.duckdb）。先跑一次带 --periods 的下载。")
        return
    print(f"股票 {st['stocks']:,} 只，库 {st['size_mb']} MB")
    for p, n in st["bars_tq"].items():
        print(f"  天勤 {p} K 线 {n:,} 行")
    print(f"  Tushare 日线 {st['bars_ts']:,} 行，复权因子 {st['adj']:,} 行")
    print(f"  基本面 {st['basic']:,} 行，资金流 {st['moneyflow']:,} 行，概念 {st['concept']:,} 个")
    if st["first"]:
        print(f"  覆盖 {st['first']} → {st['last']}")
    exs = SS.exchanges()
    print(f"  交易所：{'、'.join(exs)}")
    codes = SS.stock_list()
    print(f"  代码示例：{', '.join(codes[:5])} …" if codes else "")


def main() -> int:
    ap = argparse.ArgumentParser(description="天勤 A 股股票行情")
    ap.add_argument("--periods", default="1d", help="逗号分隔：1d / 1h / 5m")
    ap.add_argument("--exchanges", default="",
                    help="逗号分隔：SSE / SZSE；留空 = 全部")
    ap.add_argument("--start", default=None, help="起始日期，默认 2018-01-01")
    ap.add_argument("--end", default=None, help="结束日期，默认今天")
    ap.add_argument("--stats", action="store_true", help="只看库里有什么")
    args = ap.parse_args()

    if args.stats:
        _print_stats()
        return 0

    periods = tuple(p.strip() for p in args.periods.split(",") if p.strip())
    bad = [p for p in periods if p not in SF.PERIOD_SEC]
    if bad:
        print(f"不认识的周期：{bad}（只支持 {list(SF.PERIOD_SEC)}）")
        return 2

    exchanges = [e.strip() for e in args.exchanges.split(",") if e.strip()] or None
    start = date.fromisoformat(args.start) if args.start else SF.DEFAULT_START
    end = date.fromisoformat(args.end) if args.end else date.today()

    from ttrade import tqfetch as T

    sess = T.Session()
    try:
        api = sess.get()
        can_dd = SF.can_download(api)
        if can_dd:
            print("天勤账户支持 DataDownloader（专业版），走批量路径")
        else:
            print("天勤账户不支持 DataDownloader，走 get_kline_serial 逐只路径\n"
                  "（每只最多 8000 根，日线够全历史，5m 约 3 个月）")

        for period in periods:
            print(f"\n=== 周期 {period} ===", flush=True)
            if period == "5m":
                start_dt = date.today() - __import__("datetime").timedelta(days=SF.RECENT_DAYS_5M)
                print(f"5m 只取最近一年（{start_dt} → {end}）")
            else:
                start_dt = start
                print(f"{period}：{start_dt} → {end}")

            st = SF.run(api, period=period, exchanges=exchanges,
                        start=start_dt, end=end,
                        progress=lambda a, b, m: print(f"  完成 {a}/{b}（共 {m}）", flush=True))
            print(f"{period}: 完成 {st['done']}，跳过 {st['skipped']}"
                  + (f"，错误 {len(st['errors'])} 条" if st["errors"] else ""))
            for err in st["errors"][:5]:
                print(f"  ! {err}")
    finally:
        sess.close()

    _print_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
