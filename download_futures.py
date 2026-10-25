"""下载国内期货全品种主力合约 20 年日线（Tushare）。

用法：
    python download_futures.py              # 下载全部
    python download_futures.py --years 10   # 指定年数
    python download_futures.py --missing    # 只补缺失/过期品种
"""

from __future__ import annotations

import argparse
import os
import sys

from ttrade.futures import download_all

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data_futures")


def main() -> int:
    ap = argparse.ArgumentParser(description="下载国内期货日线（Tushare）")
    ap.add_argument("--years", type=int, default=20, help="回溯年数，默认 20")
    ap.add_argument("--out", default=DATA_DIR, help="输出目录")
    ap.add_argument("--missing", action="store_true", help="只补缺失或超过 30 天未更新的品种")
    args = ap.parse_args()

    print(f"开始下载国内期货主力合约，回溯 {args.years} 年")
    print(f"输出目录: {args.out}\n")

    manifest = download_all(args.out, years=args.years, skip_existing=args.missing)

    print(f"\n完成：成功 {len(manifest)} 个品种")
    total = sum(m["rows"] for m in manifest)
    print(f"累计 {total:,} 根日线，清单见 {os.path.join(args.out, '_manifest.json')}")
    return 0 if manifest else 1


if __name__ == "__main__":
    sys.exit(main())
