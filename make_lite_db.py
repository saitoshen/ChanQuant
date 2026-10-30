"""从完整的天勤仓库裁出一个「轻量库」，随安装包分发。

完整库 11GB，其中 8.4GB 是 **5m / 1h 的真实合约明细**（近 5000 个合约）。那是
「从真实合约重算期限结构」才用得上的东西，普通用户打开软件根本走不到。裁掉之后
只剩 270MB 左右，可以随包分发，装上就有 5m/1h/日线的主连数据、换月表、复权因子。

保留：
    bars          主连 5m / 1h / 1d（全部历史）
    contract_bars 真实合约**日线**（期限结构优先用日线，见 tqcurve._PREFER）
    switch        主力切换表
    meta          键值对 + 本次裁剪的记录

裁掉：
    contract_bars 的 5m / 1h（8.4GB，占完整库的 77%）

用法：
    python make_lite_db.py                 # → data_tq/tq_lite.duckdb（287MB）
    python make_lite_db.py --years 5       # 5m 只留近 5 年（169MB）
    python make_lite_db.py --no-contracts  # 不带真实合约日线（259MB）
    python make_lite_db.py --out D:/tq_lite.duckdb

生成完传到「数据包」release：

    gh release create data-2026-10 data_tq/tq_lite.duckdb \
        --title "天勤轻量库 2026-10" --notes "随安装包分发的主连行情"

**不要给它建主键。** 带 PK 的版本 1.1GB（ART 索引吃掉 850MB），查询还更慢 ——
DuckDB 是列存 + zonemap，顺序扫描本来就快，索引反而变成随机读。这里一律 CTAS。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import duckdb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ttrade import tqstore as S  # noqa: E402

PERIODS = ("5m", "1h", "1d")


def human(n: float) -> str:
    # 十进制 MB，和 `tqstore.stats()` 的 size_mb 一致（界面显示的也是这个数）
    return f"{n / 1e6:.1f} MB"


def table_rows(con, schema: str, table: str) -> int:
    try:
        r = con.execute(f"SELECT count(*) FROM {schema}.{table}").fetchone()
    except Exception:  # noqa: BLE001
        return 0
    return int(r[0]) if r else 0


def build(src: str, out: str, years: int, with_contracts: bool) -> None:
    for p in (out, out + ".wal"):
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    con = duckdb.connect(src, read_only=True)
    con.execute(f"ATTACH '{out}' AS lite (READ_WRITE)")

    # 5m 按「源库里 5m 的最新一根」往回推 N 年，不是按今天 —— 库可能停在几天前
    if years > 0:
        bars_sql = (
            "SELECT * FROM bars WHERE period <> '5m' OR dt >= "
            f"(SELECT max(dt) - INTERVAL {years} YEAR FROM bars WHERE period='5m')"
        )
    else:
        bars_sql = "SELECT * FROM bars"

    print(f"源库  {src}")
    print(f"       bars {table_rows(con, 'main', 'bars'):,} 行，"
          f"contract_bars {table_rows(con, 'main', 'contract_bars'):,} 行")
    print(f"输出  {out}")

    t0 = time.time()
    con.execute(f"CREATE TABLE lite.bars AS {bars_sql}")
    if with_contracts:
        con.execute(
            "CREATE TABLE lite.contract_bars AS"
            " SELECT * FROM contract_bars WHERE period = '1d'"
        )
    con.execute("CREATE TABLE lite.switch AS SELECT * FROM switch")
    con.execute("CREATE TABLE lite.meta AS SELECT * FROM meta")
    # CTAS 不带主键，`INSERT OR REPLACE` 就没有冲突键可用，先删再插
    con.execute(
        "DELETE FROM lite.meta WHERE k IN"
        " ('lite_built', 'lite_years', 'lite_contracts', 'schema_version')"
    )
    con.execute(
        "INSERT INTO lite.meta VALUES ('lite_built', ?), ('lite_years', ?),"
        " ('lite_contracts', ?), ('schema_version', ?)",
        [datetime.now().strftime("%Y-%m-%d %H:%M"), str(years or "all"),
         "1d" if with_contracts else "none", S.SCHEMA_VERSION],
    )
    con.execute("CHECKPOINT lite")
    con.execute("DETACH lite")
    con.close()
    print(f"      建库 {time.time() - t0:.1f}s")


def verify(src: str, out: str, years: int, with_contracts: bool) -> list[str]:
    """逐项对一遍。裁剪脚本最怕的是「悄悄少了一半数据」—— 文件小得漂亮，
    用户拿到手才发现某个品种只有三年。所以按周期、按品种都对行数。"""
    bad: list[str] = []
    a = duckdb.connect(src, read_only=True)
    b = duckdb.connect(out, read_only=True)

    # 1. 各周期行数。5m 截断过就不能直接比，比品种数 + 最新一根
    for per in PERIODS:
        n_src = a.execute(
            "SELECT count(*), count(DISTINCT symbol), max(dt) FROM bars WHERE period=?",
            [per],
        ).fetchone()
        n_out = b.execute(
            "SELECT count(*), count(DISTINCT symbol), max(dt) FROM bars WHERE period=?",
            [per],
        ).fetchone()
        if n_src[1] != n_out[1]:
            bad.append(f"{per} 品种数 {n_out[1]} ≠ 源库 {n_src[1]}")
        if n_src[2] != n_out[2]:
            bad.append(f"{per} 最新一根 {n_out[2]} ≠ 源库 {n_src[2]}")
        if years == 0 and n_src[0] != n_out[0]:
            bad.append(f"{per} 行数 {n_out[0]:,} ≠ 源库 {n_src[0]:,}")

    # 2. 逐品种比 5m 的「最后一根」：抽查 8 个，覆盖不同交易所
    picks = [r[0] for r in a.execute(
        "SELECT DISTINCT symbol FROM bars WHERE period='5m' ORDER BY symbol"
    ).fetchall()][::9][:8]
    for sym in picks:
        r_src = a.execute(
            "SELECT max(dt), count(*) FROM bars WHERE period='5m' AND symbol=?", [sym]
        ).fetchone()
        r_out = b.execute(
            "SELECT max(dt), count(*) FROM bars WHERE period='5m' AND symbol=?", [sym]
        ).fetchone()
        if r_src[0] != r_out[0]:
            bad.append(f"{sym} 5m 最新一根 {r_out[0]} ≠ 源库 {r_src[0]}")
        if years == 0 and r_src[1] != r_out[1]:
            bad.append(f"{sym} 5m 行数 {r_out[1]:,} ≠ 源库 {r_src[1]:,}")

    # 3. 真实合约日线
    if with_contracts:
        n_src = a.execute(
            "SELECT count(*), count(DISTINCT contract) FROM contract_bars WHERE period='1d'"
        ).fetchone()
        n_out = b.execute(
            "SELECT count(*), count(DISTINCT contract) FROM contract_bars WHERE period='1d'"
        ).fetchone()
        if n_src != n_out:
            bad.append(f"真实合约日线 {n_out} ≠ 源库 {n_src}")

    # 4. 换月表：少一条就会在换月处留一个假跳空
    n_src = a.execute("SELECT count(*) FROM switch").fetchone()[0]
    n_out = b.execute("SELECT count(*) FROM switch").fetchone()[0]
    if n_src != n_out:
        bad.append(f"换月表 {n_out} ≠ 源库 {n_src}")

    # 5. 真的能查：模拟一次界面切品种
    df = b.execute(
        "SELECT dt, close, adj_factor FROM bars"
        " WHERE symbol='RB' AND exchange='SHFE' AND period='5m' ORDER BY dt DESC LIMIT 5"
    ).fetchall()
    if len(df) != 5:
        bad.append("RB 5m 抽样查询取不到 5 根")

    a.close()
    b.close()
    return bad


def report(out: str) -> None:
    con = duckdb.connect(out, read_only=True)
    print("\n  表            行数")
    for t in ("bars", "contract_bars", "switch", "meta"):
        print(f"  {t:<13} {table_rows(con, 'main', t):>10,}")
    for per, n, syms, lo, hi in con.execute(
        "SELECT period, count(*), count(DISTINCT symbol), min(dt), max(dt)"
        " FROM bars GROUP BY 1 ORDER BY 1"
    ).fetchall():
        print(f"    {per:<4} {n:>10,} 行  {syms} 品种  {lo} ~ {hi}")
    con.close()
    print(f"\n  文件 {human(os.path.getsize(out))}")


def main() -> int:
    ap = argparse.ArgumentParser(description="裁剪天勤轻量库")
    ap.add_argument("--src", default="", help="完整库路径（默认自动找 data_tq/tq.duckdb）")
    ap.add_argument("--out", default="", help="输出路径（默认 data_tq/tq_lite.duckdb）")
    ap.add_argument("--years", type=int, default=0, help="5m 只留近 N 年（0 = 全部）")
    ap.add_argument("--no-contracts", action="store_true", help="不带真实合约日线")
    args = ap.parse_args()

    src = args.src or S.db_path(read_only=False)
    out = args.out or os.path.join(os.path.dirname(src), S.LITE_NAME)
    if not os.path.isfile(src):
        print(f"找不到完整库：{src}\n先用 python download_tq.py 下一份。")
        return 1

    build(src, out, args.years, not args.no_contracts)
    bad = verify(src, out, args.years, not args.no_contracts)
    report(out)
    if bad:
        print("\n校验失败：")
        for b in bad:
            print("  -", b)
        print("\n这个文件不能发。")
        return 1

    print("\n校验通过（品种数 / 最新一根 / 换月表 / 抽样查询）")
    print("上传：gh release create data-2026-10 data_tq/tq_lite.duckdb "
          "--title \"天勤轻量库\" --notes \"随安装包分发的主连行情\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
