"""从天勤（TqSdk）下载 EDB 基本面指标。

与 `download_tq.py`（K 线）、`download.py`（Tushare 日线）并列的第四条数据链路。
写 `data_tq/edb.duckdb`，**和 11GB 的行情库分开** —— 基本面用户可能不想要，
而且它只有 8MB。

指标清单随包（`ttrade/edb_indicators.json`，243 个 ID），不需要浏览器配合。
ID 到中文名、品种、分类的映射是离线的一张表，来源是 `edb.shinnytech.com` 的
目录页；新增指标时才需要重抓，平时的增量更新只走 API。

用法：

    python download_edb.py                     # 增量（默认，回看 30 天重叠）
    python download_edb.py --full              # 从 2010-01-01 整段重下
    python download_edb.py --start 2016-01-01
    python download_edb.py --stats             # 只看库里有什么，不联网
    python download_edb.py --list              # 打印指标清单
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import edb as E  # noqa: E402


def _print_stats() -> None:
    st = E.stats()
    if not st["exists"]:
        print("还没有 EDB 库（data_tq/edb.duckdb）。跑一次不带参数的本脚本即可。")
        return
    print(f"指标 {st['ids']} 个，{st['rows']:,} 行，"
          f"{st['first']} → {st['last']}，{st['size_mb']} MB")
    vs = E.varieties()
    print(f"覆盖 {len(vs)} 个品种：{'、'.join(vs[:30])}"
          + ("…" if len(vs) > 30 else ""))


def _print_list() -> None:
    ind = E.load_indicators()
    print(f"{len(ind)} 个指标：")
    by_cat: dict[str, list[tuple[int, dict]]] = {}
    for eid, m in ind.items():
        by_cat.setdefault(m["category"], []).append((eid, m))
    for cat, rows in by_cat.items():
        print(f"\n[{E.CATEGORY_LABEL.get(cat, cat)}] {len(rows)} 个")
        for eid, m in rows[:8]:
            print(f"  {eid:>6}  {m['variety']:<4} {m['name']}  ({m['unit']}, {m['freq']})")
        if len(rows) > 8:
            print(f"  … 还有 {len(rows) - 8} 个")


def main() -> int:
    ap = argparse.ArgumentParser(description="天勤 EDB 基本面指标下载")
    ap.add_argument("--start", default=None, help="起始日期，默认 2010-01-01")
    ap.add_argument("--end", default=None, help="结束日期，默认今天")
    ap.add_argument("--full", action="store_true",
                    help="从 --start 整段重下（默认只补最近 30 天重叠区）")
    ap.add_argument("--stats", action="store_true", help="只看库里有什么")
    ap.add_argument("--list", action="store_true", help="打印指标清单")
    args = ap.parse_args()

    if args.stats:
        _print_stats()
        return 0
    if args.list:
        _print_list()
        return 0

    from ttrade import tqfetch as T

    start = date.fromisoformat(args.start) if args.start else E.DEFAULT_START
    end = date.fromisoformat(args.end) if args.end else date.today()

    sess = T.Session()
    try:
        api = sess.get()
        print(f"拉取 {len(E.load_indicators())} 个指标：{start} → {end}"
              f"{'（全量）' if args.full else '（增量）'}", flush=True)
        st = E.update(
            api, start=start, end=end, incremental=not args.full,
            progress=lambda a, b, m: print(f"  {m}", flush=True),
        )
    finally:
        sess.close()

    print(f"完成：{st['batches']} 个批次成功，写入 {st['rows']:,} 行"
          + (f"，{st['failed']} 个批次失败（再跑一次会补上）" if st["failed"] else ""))
    _print_stats()
    return 0 if not st["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
