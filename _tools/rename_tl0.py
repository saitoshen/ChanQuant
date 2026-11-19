"""把 30 年期国债的品种代码 `TL0` 正名为 `TL`（一次性迁移，可重复跑）。

背景：Tushare `fut_basic` 里 30 年期国债的**品种**代码是 `TL0.CFX` —— 那个 `0`
本来是连续合约标记，被当成了品种名。于是：

- 本地文件叫 `data_futures/TL0_CFFEX.csv`
- 库里 `bars.symbol = 'TL0'`
- 策略配置里品种键写成 `TL0_CFFEX`
- 实盘下单时这条链拼不出交易所，合约名变成 `tlcffex0`，CTP 回
  `[148] 无效的 ExchangeID 字段`（2026-11-19）

`download_futures.py` 已经改成按 `TL` 落盘，这个脚本负责把**存量**改过来：

1. JSON 配置（策略库 / 组合 / 品种参数）：`TL0` → `TL`、`TL0_CFFEX` → `TL_CFFEX`
2. DuckDB 三张表（`bars` / `contract_bars` / `switch`）的 `symbol`
3. 落盘前的 CSV（`data_futures`、`data_tq/curve`）—— 文件改名由 `git mv` 做过，
   这里只兜底：万一还有旧名就改名

用法（**先关掉客户端**，DuckDB 只允许一个进程写）：

    python _tools/rename_tl0.py            # 预览，不动任何东西
    python _tools/rename_tl0.py --apply    # 真改（先备份 JSON）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OLD, NEW = "TL0", "TL"
JSON_FILES = ("strategy_lib.json", "portfolio.json", "symbols.json")

#: 值和键都要看的字段名。整份 JSON 里只要字符串「正好是 TL0 / 以 TL0_ 开头」
#: 就替换，别的（比如因子名里带 tl0 的）不动
def _fix_str(s: str) -> str:
    if s == OLD:
        return NEW
    if s.startswith(OLD + "_"):
        return NEW + s[len(OLD):]
    # 策略名里写死的品种标注：「30年期国债(TL0) 4h 低分多…」
    if f"({OLD})" in s:
        return s.replace(f"({OLD})", f"({NEW})")
    return s


def _fix_obj(obj):
    """递归改 keys + str 值。返回 (新对象, 改了几处)。"""
    changed = 0
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = _fix_str(k) if isinstance(k, str) else k
            if nk != k:
                changed += 1
            nv, n = _fix_obj(v)
            out[nk] = nv
            changed += n
        return out, changed
    if isinstance(obj, list):
        out = []
        for v in obj:
            nv, n = _fix_obj(v)
            out.append(nv)
            changed += n
        return out, changed
    if isinstance(obj, str):
        ns = _fix_str(obj)
        return ns, (1 if ns != obj else 0)
    return obj, 0


def _user_dir() -> str:
    from ttrade.paths import user_dir

    return user_dir()


def do_json(apply: bool) -> None:
    d = _user_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    for name in JSON_FILES:
        p = os.path.join(d, name)
        if not os.path.exists(p):
            print(f"  跳过 {name}（不存在）")
            continue
        with open(p, encoding="utf-8") as fh:
            raw = fh.read()
        try:
            data = json.loads(raw)
        except ValueError as exc:
            print(f"  !! {name} 不是合法 JSON，跳过：{exc}")
            continue
        fixed, n = _fix_obj(data)
        if not n:
            print(f"  {name}：没有需要改的")
            continue
        print(f"  {name}：{n} 处 {OLD} → {NEW}")
        if apply:
            bak = f"{p}.bak_tl0_{stamp}"
            shutil.copy2(p, bak)
            tmp = p + ".tmp"
            with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
                json.dump(fixed, fh, ensure_ascii=False, indent=2)
                fh.write("\n")
            os.replace(tmp, p)
            print(f"     已写入（备份 {os.path.basename(bak)}）")


def do_db(apply: bool) -> None:
    from ttrade import tqstore

    if not tqstore.exists():
        print("  没有 tq.duckdb，跳过")
        return
    try:
        with tqstore.connect(read_only=not apply) as con:
            for t in ("bars", "contract_bars", "switch"):
                n = con.execute(
                    f"SELECT count(*) FROM {t} WHERE symbol = ?", [OLD]).fetchone()[0]
                if not n:
                    print(f"  {t}：没有 {OLD}")
                    continue
                print(f"  {t}：{n} 行 {OLD} → {NEW}")
                if apply:
                    con.execute(f"UPDATE {t} SET symbol = ? WHERE symbol = ?",
                                [NEW, OLD])
            if apply:
                for t in ("bars", "contract_bars", "switch"):
                    chk = con.execute(
                        f"SELECT count(*) FROM {t} WHERE symbol = ?", [OLD]).fetchone()[0]
                    assert chk == 0, f"{t} 还有 {chk} 行没改"
                print("  数据库已更新")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! 数据库没改成：{exc}")
        print("     客户端（ChanQuant）可能还开着 —— DuckDB 只允许一个进程写，"
              "关掉它再跑一次。")


def do_csv() -> None:
    pairs = [("data_futures", f"{OLD}_CFFEX.csv", f"{NEW}_CFFEX.csv"),
             ("data_tq/curve", f"{OLD}_CFFEX.csv", f"{NEW}_CFFEX.csv")]
    for d, old, new in pairs:
        op = os.path.join(ROOT, d, old)
        np_ = os.path.join(ROOT, d, new)
        if os.path.exists(op) and os.path.exists(np_):
            print(f"  {d}：新旧都在，删掉旧的？先不动，请人工确认 {old}")
        elif os.path.exists(op):
            os.replace(op, np_)
            print(f"  {d}：{old} → {new}")
        else:
            print(f"  {d}：已经是 {new}" if os.path.exists(np_) else f"  {d}：两个都没有")


def main() -> int:
    ap = argparse.ArgumentParser(description="TL0 → TL 正名")
    ap.add_argument("--apply", action="store_true", help="真的改；不带就是预览")
    ap.add_argument("--only", choices=("all", "db", "json"), default="all",
                    help="只做某一段。客户端开着的时候先 --only db："
                         "库改完策略就能读到 TL 的行情，配置等关掉客户端再改"
                         "（否则会被它的内存副本回写覆盖）")
    args = ap.parse_args()
    print(f"== TL0 → TL {'（执行）' if args.apply else '（预览，不改任何东西）'}"
          f"  范围={args.only} ==\n")
    if args.only in ("all", "json"):
        print("[1] CSV 文件")
        do_csv()
        print("\n[2] JSON 配置")
        do_json(args.apply)
    if args.only in ("all", "db"):
        print("\n[3] DuckDB")
        do_db(args.apply)
    if not args.apply:
        print("\n预览完毕。确认无误后加 --apply 执行（客户端要先关掉）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
