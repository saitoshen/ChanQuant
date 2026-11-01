"""抓宏观数据（23 个序列）进 `data_tq/edb.duckdb` 的 `macro_series` 表。

和 `download_edb.py`（品种级基本面）、`download_tq.py`（K 线）、
`download.py`（Tushare 日线）并列的第五条数据链路。

数据源见 `ttrade/macro.py`：Tushare + 东方财富，美元指数用六个构成货币
按 ICE 公式合成。

用法：

    python download_macro.py                # 增量（回看 400 天重叠，覆盖修订）
    python download_macro.py --full         # 整段重抓
    python download_macro.py --replace      # 先删后写（换数据源时用）
    python download_macro.py --only dxy     # 只抓某几个
    python download_macro.py --stats        # 只看库里有什么，不联网
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import macro as M  # noqa: E402


def _print_stats() -> None:
    st = M.stats()
    if not st["exists"]:
        print("还没有 EDB 库。先跑 download_edb.py，或者直接跑一次本脚本。")
        return
    print(f"宏观序列 {st['codes']}/{len(M.SERIES)} 条，{st['rows']:,} 行，"
          f"{st['first']} → {st['last']}")
    cov = M.coverage()
    for _, r in cov.iterrows():
        flag = "  " if r["n"] else "✗ "
        span = f"{r['first']} → {r['last']}" if r["n"] else "无数据"
        print(f"  {flag}{r['code']:<12} {r['name']:<12} {r['freq']}  "
              f"{r['n']:>5} 行  {span}")


def main() -> int:
    ap = argparse.ArgumentParser(description="宏观数据下载")
    ap.add_argument("--full", action="store_true", help="整段重抓，不做增量")
    ap.add_argument("--replace", action="store_true",
                    help="先删掉这些 code 的旧值再写（换数据源时用）")
    ap.add_argument("--only", default="", help="只抓这些 code，逗号分隔")
    ap.add_argument("--stats", action="store_true", help="只看库里有什么")
    args = ap.parse_args()

    if args.stats:
        _print_stats()
        return 0

    codes = [c.strip() for c in args.only.split(",") if c.strip()] or None
    print(f"抓取 {len(codes) if codes else len(M.SERIES)} 条宏观序列"
          f"{'（全量）' if args.full or args.replace else '（增量）'}", flush=True)
    st = M.update(codes=codes, incremental=not (args.full or args.replace),
                  replace=args.replace,
                  progress=lambda a, b, m: print(f"  {m}", flush=True))
    print(f"完成：{st['ok']} 条成功，{st['empty']} 条无数据，写入 {st['rows']:,} 行")
    for code, msg in st["detail"].items():
        if msg == "无数据":
            print(f"  ✗ {code}（{M.SERIES[code][0]}）：{M.SERIES[code][3]}")
    _print_stats()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
