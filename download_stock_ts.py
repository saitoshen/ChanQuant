"""从 Tushare 下载 A 股股票数据（命令行壳子）。

真正的逻辑在 `ttrade/stockdl.py` —— 界面「数据管理 → A 股」调的是同一份代码。
脚本只负责解析参数、把进度打到终端。

用法：

    python download_stock_ts.py --stats                 # 只看库里有什么
    python download_stock_ts.py --all                   # 全部（列表→日线→基本面→资金流→概念）
    python download_stock_ts.py --daily --start 20200101
    python download_stock_ts.py --basic --moneyflow
    python download_stock_ts.py --concept

接口清单、全历史策略、代码格式转换见 `ttrade/stockdl.py` 的模块说明。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import stockdl as DL  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Tushare A 股股票数据")
    ap.add_argument("--stats", action="store_true", help="只看库里有什么")
    ap.add_argument("--meta", action="store_true", help="下载股票元数据")
    ap.add_argument("--daily", action="store_true", help="下载全历史日线 + 复权因子")
    ap.add_argument("--basic", action="store_true", help="下载基本面")
    ap.add_argument("--concept", action="store_true", help="下载概念板块")
    ap.add_argument("--moneyflow", action="store_true", help="下载资金流")
    ap.add_argument("--all", action="store_true", help="全部")
    ap.add_argument("--start", default=None, help="起始日期 YYYYMMDD")
    ap.add_argument("--end", default=None, help="结束日期 YYYYMMDD")
    args = ap.parse_args()

    if args.stats:
        print(DL.stats_text())
        return 0

    picked = [k for k in ("meta", "daily", "basic", "moneyflow", "concept")
              if getattr(args, k)]
    if args.all:
        picked = [k for k, _ in DL.PARTS]
    if not picked:
        ap.print_help()
        return 1

    label = dict(DL.PARTS)

    def progress(done: int, total: int, msg: str) -> None:
        print(f"[{done}/{total}] {msg}", flush=True)

    res = DL.run(picked, start=args.start, end=args.end, progress=progress)
    print()
    for key in picked:
        st = res.get(key) or {}
        errs = st.get("errors") or []
        tail = f"，错误：{errs[0]}" if errs else ""
        print(f"{label[key]}：{st.get('rows', 0):,} 行{tail}")
    print()
    print(DL.stats_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
