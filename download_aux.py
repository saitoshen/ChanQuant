"""下载真实合约与期权数据，并生成展期/期权因子所需的衍生数据。

用法::

    python download_aux.py                  # 全历史（20 年），合约 + 期权
    python download_aux.py --what contracts # 只下真实合约（快）
    python download_aux.py --what options   # 只下期权
    python download_aux.py --range 1y       # 只补最近一年
    python download_aux.py --keep-raw-options   # 同时保留期权原始日线（1GB+）

产物::

    data_contracts/      每品种一个 CSV，含该品种全部真实合约日线（约 130MB，不进 git）
    data_futures_aux/    期限结构 + 期权波动率摘要（约 50MB，随包分发）
    data_options_raw/    期权原始日线（可选，默认不生成）

为什么要单独跑一次：全历史要请求约 4800 个交易日的横截面，耗时 40~90 分钟，
不适合塞进 GUI 的「更新」按钮里。GUI 里的更新只补最近一段。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import auxbuild as AB          # noqa: E402
from ttrade import contracts as C          # noqa: E402
from ttrade.paths import data_dir          # noqa: E402


def _days_of(range_key: str) -> int:
    return {"1m": 31, "3m": 92, "6m": 183, "1y": 366, "all": 0}[range_key]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="下载真实合约 / 期权并生成衍生数据")
    ap.add_argument("--what", choices=["both", "contracts", "options"], default="both")
    ap.add_argument("--range", dest="range_key",
                    choices=["1m", "3m", "6m", "1y", "all"], default="all")
    ap.add_argument("--root", default=None,
                    help="数据根目录（默认与 data_futures 同级）")
    ap.add_argument("--keep-raw-options", action="store_true",
                    help="同时保留期权原始日线（约 1GB，默认只存聚合结果）")
    ap.add_argument("--refresh-basic", action="store_true",
                    help="强制重拉合约清单（新合约上市后用）")
    args = ap.parse_args(argv)

    root = args.root or os.path.dirname(data_dir("cn"))
    main_dir = os.path.join(root, "data_futures")
    if not os.path.isdir(main_dir):
        print(f"找不到 {main_dir}，请先运行 python download_futures.py")
        return 1

    all_days = C.trading_days(main_dir)
    if not all_days:
        print("本地没有行情数据，无法确定交易日历")
        return 1
    n = _days_of(args.range_key)
    if n:
        cutoff = time.strftime("%Y%m%d", time.localtime(time.time() - n * 86400))
        all_days = [d for d in all_days if d >= cutoff]

    want_c = args.what in ("both", "contracts")
    want_o = args.what in ("both", "options")

    man = C.load_manifest(os.path.join(root, "data_futures_aux"))
    done_c = set(man.get("dates") or [])
    done_o = set(man.get("opt_dates") or [])
    days = [d for d in all_days if (want_c and d not in done_c) or (want_o and d not in done_o)]

    print(f"交易日 {len(all_days)} 个（{all_days[0]} ~ {all_days[-1]}），"
          f"待处理 {len(days)} 个")
    if not days:
        print("已经是最新的，无需下载")
        return 0
    if want_o and args.keep_raw_options:
        print("注意：--keep-raw-options 会额外写 1GB+ 期权明细")

    if args.refresh_basic:
        from ttrade import futures as fut
        pro = fut.get_pro()
        raw_dir = os.path.join(root, "data_contracts")
        os.makedirs(raw_dir, exist_ok=True)
        C.fetch_basic(pro, cache_path=os.path.join(raw_dir, "_basic.csv"), refresh=True)
        print("合约清单已刷新")

    t0 = time.time()
    last = [0.0]

    def on_progress(done: int, total: int, msg: str) -> None:
        now = time.time()
        # 控制台刷新太快会拖慢下载，最多每 0.5 秒打一行
        if now - last[0] < 0.5 and done < total:
            return
        last[0] = now
        pct = done / max(total, 1) * 100
        eta = (now - t0) / max(done, 1) * (total - done)
        print(f"\r  [{done:>5}/{total}] {pct:5.1f}%  剩余约 {eta/60:5.1f} 分钟  {msg:<44}",
              end="", flush=True)

    try:
        res = AB.build(
            days, data_root=root,
            want_contracts=want_c, want_options=want_o,
            keep_raw_options=args.keep_raw_options,
            progress=on_progress,
        )
    except KeyboardInterrupt:
        print("\n已中断（已完成的部分已落盘，重跑会接着来）")
        return 130

    print()
    print(res.summary())
    for e in res.errors[:10]:
        print("  ", e)
    return 0 if not res.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
