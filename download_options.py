"""从天勤下载期货期权行情，并算隐含波动率。

写 `data_tq/option.duckdb`（**和 11GB 的行情库分开**），同时导出
`data_tq/option_iv/tq_iv_<period>.csv`。

    option_meta        合约定义
    option_bars        期权 K 线
    option_underlying  标的 K 线（定价用的 F）
    option_iv          按品种聚合的 IV 摘要

口径说明（**和 Tushare 那条不是一套**）：天勤 K 线只有收盘价，没有结算价，
所以 F 和期权价都用 close；Tushare 那条是结算价 + 日频。两套 IV 不能混用，
文件名带 `tq_` 前缀就是为了不让人看串。

用法：

    python download_options.py --stats
    python download_options.py --varieties IO,MO,HO --periods 1d 5m
    python download_options.py --periods 1d,1h                # 全品种（几小时）
    python download_options.py --periods 1d,1h --varieties rb,cu,m
    python download_options.py --iv-only --periods 1d          # 只重算 IV
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import optfetch as OF  # noqa: E402
from ttrade import optstore as OS  # noqa: E402


def _print_stats() -> None:
    st = OS.stats()
    if not st["exists"]:
        print("还没有期权库（data_tq/option.duckdb）。先跑一次带 --periods 的下载。")
        return
    print(f"合约 {st['contracts']:,} 个，库 {st['size_mb']} MB")
    for p, n in st["bars"].items():
        print(f"  {p} K 线 {n:,} 行，IV {st['iv'].get(p, 0):,} 行")
    print(f"  覆盖 {st['first']} → {st['last']}")
    vs = OS.varieties()
    print(f"  {len(vs)} 个品种：{'、'.join(vs)}")


def main() -> int:
    ap = argparse.ArgumentParser(description="天勤期权行情 + 隐含波动率")
    ap.add_argument("--periods", default="1d", help="逗号分隔：1d / 1h / 5m")
    ap.add_argument("--varieties", default="",
                    help="逗号分隔的品种代码；留空 = 全部（5m 强制只有 IO/MO/HO）")
    ap.add_argument("--start", default=None, help="起始日期，默认 2016-01-01")
    ap.add_argument("--end", default=None, help="结束日期，默认今天")
    ap.add_argument("--full", action="store_true",
                    help="不跳过已下完的合约（默认按合约到期日增量）")
    ap.add_argument("--stats", action="store_true", help="只看库里有什么")
    ap.add_argument("--iv-only", action="store_true", help="只重算 IV 并导出 CSV")
    args = ap.parse_args()

    if args.stats:
        _print_stats()
        return 0

    periods = tuple(p.strip() for p in args.periods.split(",") if p.strip())
    bad = [p for p in periods if p not in OF.PERIOD_SEC]
    if bad:
        print(f"不认识的周期：{bad}（只支持 {list(OF.PERIOD_SEC)}）")
        return 2
    varieties = [v.strip() for v in args.varieties.split(",") if v.strip()] or None

    if args.iv_only:
        for p in periods:
            vs = varieties or OS.varieties()
            for v in vs:
                iv = OF.compute_iv(v, p)
                if iv.empty:
                    continue
                with OS.connect() as con:
                    OS.upsert_iv(con, iv)
            path = OF.export_csv(p)
            print(f"{p}: IV 已重算，导出 {path}")
        _print_stats()
        return 0

    from ttrade import tqfetch as T

    start = date.fromisoformat(args.start) if args.start else OF.DEFAULT_START
    end = date.fromisoformat(args.end) if args.end else date.today()

    sess = T.Session()
    try:
        api = sess.get()
        if not T.can_download(api):
            print("这个天勤账户没有 `tq_dl`（历史数据下载）功能位。\n"
                  "期权没有「最近 8000 根」的替代路径 —— 已摘牌期权的 "
                  "get_kline_serial 返回全 0，只能走 DataDownloader。\n"
                  "升级地址：https://www.shinnytech.com/tqsdk-buy/")
            return 3
        print(f"周期 {periods}，品种 {'全部' if varieties is None else varieties}，"
              f"{start} → {end}", flush=True)
        # 传 Session 而不是 api：跑长任务时要靠它定期重建会话清订阅
        st = OF.update(sess, periods=periods, varieties=varieties,
                       start=start, end=end, incremental=not args.full,
                       progress=lambda a, b, m: print(f"  {m}", flush=True))
    finally:
        sess.close()

    for p, s in st.items():
        print(f"{p}: {s['chains']} 条链，写入 {s['rows']:,} 行，"
              f"跳过 {s['skipped']} 条（已是最新），失败 {s['failed']} 条，"
              f"IV {s['iv']:,} 行")
    _print_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
