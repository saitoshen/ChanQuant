"""批量下载美股 20 年日线数据（东方财富，后复权）。

用法：
    python download.py                # 下载全部
    python download.py AAPL NVDA SPY  # 只下载指定标的
    python download.py --years 10     # 指定年数
"""

from __future__ import annotations

import argparse
import os
import sys

from ttrade.eastmoney import download_universe
from ttrade.universe import SYMBOLS

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载美股日线（东方财富）")
    ap.add_argument("symbols", nargs="*", help="股票代码，留空则下载全部股票池")
    ap.add_argument("--years", type=int, default=20, help="回溯年数，默认 20")
    ap.add_argument("--adjust", type=int, default=2, choices=[0, 1, 2], help="0不复权 1前复权 2后复权")
    ap.add_argument("--out", default=DATA_DIR, help="输出目录")
    ap.add_argument("--missing", action="store_true", help="只补下载本地缺失的标的")
    ap.add_argument("--sleep", type=float, default=0.15, help="每只之间的间隔秒数")
    args = ap.parse_args()

    if args.missing:
        have = set()
        if os.path.isdir(args.out):
            for fn in os.listdir(args.out):
                if fn.endswith(".csv") and not fn.startswith("_"):
                    have.add(fn[:-4].replace("_", "."))
        missing = [s for s in SYMBOLS if s not in have]
        if not missing:
            print("本地数据已完整，无需补下载。")
            return 0
        symbols = missing
        print(f"检测到缺失 {len(missing)} 只：{', '.join(missing)}\n")
    else:
        symbols = [s.upper() for s in args.symbols] or SYMBOLS

    print(f"开始下载 {len(symbols)} 只标的，回溯 {args.years} 年，复权方式={args.adjust}")
    print(f"输出目录: {args.out}\n")

    manifest = download_universe(
        symbols, out_dir=args.out, years=args.years, adjust=args.adjust, sleep=args.sleep
    )

    print(f"\n完成：成功 {len(manifest)} / 请求 {len(symbols)}")
    total_rows = sum(m["rows"] for m in manifest)
    print(f"累计 {total_rows:,} 根日线，清单见 {os.path.join(args.out, '_manifest.json')}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
