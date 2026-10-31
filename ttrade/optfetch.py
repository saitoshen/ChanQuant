"""天勤期权：枚举合约、下载 K 线、算隐含波动率。

三件事：

1. **枚举** —— `query_quotes(ins_class="OPTION")` 一次拿全部合约（含已摘牌），
   按代码前缀分组得到「期权链」（一条链 = 一个标的月份的全部行权价）。
2. **下载** —— 每条链一个 `DataDownloader` 任务，**把标的放在第一个**。
3. **算 IV** —— 每个品种每个时间点取最近两个到期月，反解后拟二次微笑。

三个必须记住的事实（都是实测出来的）：

**① `query_options(标的)` 对已摘牌的标的会卡住**，不能用它枚举。只能
`query_quotes` + 代码分组。

**② `DataDownloader` 的批量上限在 40~50 之间，超过会永久卡在 `wait_update`**。
60 个合约的请求既不下完也不报错，`task.is_finished()` 永远为 False。所以这里
固定每批 30 个，并且**每批都带看门狗**：40 秒没有进度就取消重来。

**③ 批里第一个合约决定时间轴**。`_download_data` 只按 `serials[0]` 走，
第一个合约没数据就整批返回空。所以标的必须排第一 —— 标的永远有行情，顺带
还解决了「算 IV 要用的 F 从哪来」。

**免费账户**：`tq_dl` 是专业版功能位，没有就走不了 `DataDownloader`。期权没有
「最近 8000 根」的替代路径（单合约 `get_kline_serial` 对已摘牌期权返回全 0），
所以免费账户直接拒绝，并把话说清楚。
"""

from __future__ import annotations

import os
import re
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from . import optstore as OS
from . import options as OP

# 六个交易所的期货期权。SSE/SZSE 的 ETF 期权不是「期货的期权」，不收。
EXCHANGES = ["CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX"]

PERIOD_SEC = {"1d": 86400, "1h": 3600, "5m": 300}

# 单批合约数。40 能过、50 会卡死，取 30 留余量。
BATCH = 30

# 一批多久没有进度就判定卡死。正常一批 1~3 秒，40 秒够宽了。
STALL_SEC = 40

# 商品期权的真实起点：豆粕 2017-03-31 第一个。更早的日期取不到数据，
# 但不是错误 —— 接口给空表。
DEFAULT_START = date(2016, 1, 1)

# 5m 只做中金所三个股指期权：MO / IO / HO
CFFEX_5M = ["HO", "IO", "MO"]

_SUFFIX = re.compile(r"^(.*?)([CP])(\d+(?:\.\d+)?)$")


# ------------------------------------------------------------------ 代码解析


def chain_of(contract: str) -> str:
    """`DCE.l2607-P-7900` → `DCE.l2607`。

    期权代码有两种写法，都要认：

        DCE.l2607-P-7900     横线分隔（大商所、广期所）
        SHFE.br2407P16600    紧凑写法（上期所、郑商所、能源中心）
        CFFEX.IO2410-C-2800  横线分隔（中金所）
    """
    ex, _, ins = contract.partition(".")
    if "-" in ins:
        return f"{ex}.{ins.split('-')[0]}"
    m = _SUFFIX.match(ins)
    return f"{ex}.{m.group(1)}" if m else contract


def parse_contract(contract: str) -> tuple[str, float] | None:
    """从代码里拆出 (C/P, 行权价)。拆不出来返回 None。"""
    ins = contract.partition(".")[2]
    if "-" in ins:
        parts = ins.split("-")
        if len(parts) >= 3:
            try:
                return parts[1].upper(), float(parts[2])
            except ValueError:
                return None
        return None
    m = _SUFFIX.match(ins)
    if not m:
        return None
    try:
        return m.group(2).upper(), float(m.group(3))
    except ValueError:
        return None


def variety_of(contract: str) -> str:
    """`SHFE.br2407P16600` → `br`；`CFFEX.IO2410-C-2800` → `IO`。

    取合约名开头的连续字母。六个交易所的期权代码都满足这个规律。
    """
    ins = contract.partition(".")[2]
    m = re.match(r"^[A-Za-z]+", ins)
    return m.group(0) if m else ""


# ------------------------------------------------------------------ 枚举


def list_options(api, exchanges: list[str] | None = None) -> dict[str, list[str]]:
    """`{链代码: [合约...]}`，含已摘牌。

    一次 `query_quotes` 拿全部，再按代码前缀分组 —— 不要对每条链调
    `query_options`，已摘牌标的上那个接口会卡住。
    """
    out: dict[str, list[str]] = {}
    for ex in (exchanges or EXCHANGES):
        syms = api.query_quotes(ins_class="OPTION", exchange_id=[ex])
        for s in syms:
            out.setdefault(chain_of(s), []).append(s)
    for k in out:
        out[k].sort()
    return dict(sorted(out.items()))


def chain_info(api, members: list[str]) -> dict | None:
    """一条链的元数据：只问一个合约，省 70 次请求。

    实测同一条链里所有合约的 `expire_datetime` 完全一致（六个交易所各抽四条
    链验证过），所以取一个代表就够。
    """
    for probe in members:
        try:
            q = api.get_quote(probe)
        except Exception:      # noqa: BLE001
            continue
        exp = getattr(q, "expire_datetime", 0) or 0
        und = getattr(q, "underlying_symbol", "") or ""
        if exp and und:
            return {
                "underlying": und,
                "expiry": pd.Timestamp(float(exp), unit="s",
                                       tz="Asia/Shanghai").date(),
            }
    return None


# ------------------------------------------------------------------ 下载


def _empty_bars() -> pd.DataFrame:
    return pd.DataFrame(columns=OS.BAR_COLS)


def _read_csv(path: str, symbols: list[str], period: str) -> pd.DataFrame:
    """`DataDownloader` 的宽表 CSV → 长表。

    列名是 `<合约>.open` 这种。逐合约 melt，空列（该合约当天没成交）直接丢。
    """
    if not os.path.exists(path) or os.path.getsize(path) < 32:
        return _empty_bars()
    raw = pd.read_csv(path)
    if "datetime" not in raw.columns or len(raw) == 0:
        return _empty_bars()

    dt = pd.to_datetime(raw["datetime"], errors="coerce")
    ok = dt.notna()
    if not ok.any():
        return _empty_bars()
    raw, dt = raw[ok], dt[ok]

    out: list[pd.DataFrame] = []
    for sym in symbols:
        cols = {c: f"{sym}.{c}" for c in
                ("open", "high", "low", "close", "volume", "open_oi", "close_oi")}
        if cols["close"] not in raw.columns:
            continue
        sub = pd.DataFrame({"dt": dt.to_numpy()})
        for k, src in cols.items():
            # 没有这一列就补 NaN，不要 KeyError —— 某个合约只有收盘价（或者
            # 天勤漏了某几列）是很常见的，整条链不该因此丢掉
            sub[k] = (pd.to_numeric(raw[src], errors="coerce").to_numpy()
                      if src in raw.columns else np.nan)
        sub = sub.dropna(subset=["close"])
        if sub.empty:
            continue
        sub["contract"] = sym
        sub["period"] = period
        out.append(sub)
    if not out:
        return _empty_bars()
    return pd.concat(out, ignore_index=True)[OS.BAR_COLS]


def download_chain(api, underlying: str, members: list[str], period: str,
                   start: date, end: date, tmpdir: str,
                   stall: float = STALL_SEC, cancel=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """下一条链，返回 `(期权 bars, 标的 bars)`。

    每批 30 个合约，标的排第一（见模块开头第 ③ 条）。某一批卡住就取消、
    缩小到单个合约重来一次；还不行就跳过这一批 —— 一条链里丢掉 30 个合约
    比整轮停住好。
    """
    from tqsdk.tools import DataDownloader

    dur = PERIOD_SEC[period]
    os.makedirs(tmpdir, exist_ok=True)
    bars: list[pd.DataFrame] = []
    unds: list[pd.DataFrame] = []
    # **一个周期只用一个临时文件**，每批覆盖写。
    #
    # 一开始是「每批一个文件名、读完删掉」，跑几十条链之后删了几百个文件，
    # 被环境的批量删除保护拦下来，整轮直接中断。DataDownloader 本身是 `"w"`
    # 模式打开（会截断），所以复用同一个文件名完全够用，一个删除都不需要。
    path = os.path.join(tmpdir, f"batch_{period}.csv")

    def take(syms: list[str]) -> None:
        df = _read_csv(path, syms, period)
        if df.empty:
            return
        und = df[df["contract"] == underlying]
        if len(und):
            unds.append(und[["contract", "period", "dt", "close"]].copy())
        opt = df[df["contract"] != underlying]
        if len(opt):
            bars.append(opt)

    for i in range(0, len(members), BATCH):
        if cancel is not None and cancel():
            break
        chunk = members[i:i + BATCH]
        syms = [underlying] + chunk
        if _run_batch(api, syms, dur, start, end, path, stall) is not None:
            take(syms)
            continue

        # 整批卡死：退到一个合约一次，能救几个是几个
        for sym in chunk:
            if _run_batch(api, [underlying, sym], dur, start, end, path, stall) is not None:
                take([underlying, sym])

    return (
        pd.concat(bars, ignore_index=True) if bars else _empty_bars(),
        pd.concat(unds, ignore_index=True) if unds else
        pd.DataFrame(columns=["contract", "period", "dt", "close"]),
    )


def _run_batch(api, syms: list[str], dur: int, start: date, end: date,
               path: str, stall: float):
    """跑一批，成功返回 csv 路径，卡住返回 None。

    不删旧文件：`DataDownloader` 是 `"w"` 模式打开的，会自己截断。这里刻意
    一个 `os.remove` 都不写 —— 长任务里删几百个临时文件会被环境的批量删除
    保护拦下来，整轮直接中断。
    """
    from tqsdk.tools import DataDownloader

    try:
        task = DataDownloader(api, symbol_list=list(syms), dur_sec=dur,
                              start_dt=start, end_dt=end, csv_file_name=path)
    except Exception:      # noqa: BLE001
        return None

    last, prog = time.time(), -1.0
    while not task.is_finished():
        try:
            api.wait_update(deadline=time.time() + 3)
        except Exception:      # noqa: BLE001
            break
        cur = task.get_progress()
        if cur != prog:
            prog, last = cur, time.time()
        elif time.time() - last > stall:
            break
    if not task.is_finished():
        try:
            task._task.cancel()
        except Exception:      # noqa: BLE001
            pass
        return None
    return path


# ------------------------------------------------------------------ 隐含波动率


def compute_iv(variety: str, period: str) -> pd.DataFrame:
    """把一个品种的期权横截面压成「每个时间点一行」的 IV 摘要。

    口径和 Tushare 那条（`options.day_metrics`）**故意保持一致**，方便对照：

    - F 用同一时刻的标的收盘价（天勤 K 线只有 close，没有结算价）
    - 只用虚值（K ≥ F 取 call，K < F 取 put）—— 实值几乎没有时间价值，
      一个最小变动价位的误差就能让反解出的 σ 翻几倍
    - 剩余期限 15~180 天，log-moneyness 在 ±20% 内
    - 每个到期月拟二次微笑，取平值 / 斜率 / 曲率
    - 近月取最短，次近取第二短
    - PCR 用**全部**行权价的持仓量（含实值）
    """
    if not OS.exists():
        return pd.DataFrame()
    with OS.connect(read_only=True) as con:
        meta = con.execute(
            "SELECT * FROM option_meta WHERE variety = ?", [variety]
        ).df()
        if meta.empty:
            return pd.DataFrame()
        contracts = meta["contract"].tolist()
        q = ",".join("?" * len(contracts))
        bars = con.execute(
            f"SELECT contract, dt, close, open_oi AS oi FROM option_bars"
            f" WHERE period = ? AND contract IN ({q})",
            [period] + contracts,
        ).df()
        if bars.empty:
            return pd.DataFrame()
        # 标的表里的 contract 是**标的代码**（DCE.m2405），不在期权合约清单里 ——
        # 拿 contracts 去查会一条都查不到，IV 直接全空
        unders = [u for u in meta["underlying"].dropna().unique().tolist() if u]
        if not unders:
            return pd.DataFrame()
        q2 = ",".join("?" * len(unders))
        unds = con.execute(
            f"SELECT contract, dt, close AS fut FROM option_underlying"
            f" WHERE period = ? AND contract IN ({q2})",
            [period] + unders,
        ).df()

    if unds.empty:
        return pd.DataFrame()

    m = bars.merge(meta, on="contract", how="inner")
    m["dt"] = pd.to_datetime(m["dt"])
    unds["dt"] = pd.to_datetime(unds["dt"])
    # 标的表里的 `contract` 就是标的代码（如 DCE.m1709），改名之后按
    # (underlying, dt) 对齐 —— 不这样的话两边的 contract 列会撞在一起
    u = unds.rename(columns={"contract": "underlying", "close": "fut"})
    m = m.merge(u, on=["underlying", "dt"], how="left")
    m["F"] = pd.to_numeric(m["fut"], errors="coerce")
    m["px"] = pd.to_numeric(m["close"], errors="coerce")
    m = m[(m["F"] > 0) & (m["px"] > OP.MIN_PRICE)]
    if m.empty:
        return pd.DataFrame()

    m["dte"] = (pd.to_datetime(m["expiry"]) - m["dt"]).dt.days
    m = m[(m["dte"] >= OP.MIN_DTE) & (m["dte"] <= OP.MAX_DTE)]
    if m.empty:
        return pd.DataFrame()

    m["money"] = np.log(pd.to_numeric(m["strike"], errors="coerce") / m["F"])
    m = m[m["money"].abs() <= OP.MONEY_BAND]
    if m.empty:
        return pd.DataFrame()

    m["is_call"] = m["opt_class"].astype(str).str.upper().str.startswith("C")
    m["iv"] = OP.implied_vol(
        m["px"].to_numpy(), m["F"].to_numpy(),
        pd.to_numeric(m["strike"], errors="coerce").to_numpy(),
        m["dte"].to_numpy() / 365.0, m["is_call"].to_numpy(),
    )
    m = m[(m["iv"] > OP.IV_LO * 1.2) & (m["iv"] < OP.IV_HI * 0.95)]
    if m.empty:
        return pd.DataFrame()

    # PCR 先按全部行权价算（含实值）
    oi = pd.to_numeric(m["oi"], errors="coerce").fillna(0.0)
    pcr = (
        oi.where(~m["is_call"], 0.0).groupby(m["dt"]).sum()
        / oi.where(m["is_call"], 0.0).groupby(m["dt"]).sum().replace(0, np.nan)
    )

    # 拟合只用虚值
    otm = (m["strike"] >= m["F"]) & m["is_call"] | (m["strike"] < m["F"]) & ~m["is_call"]
    o = m[otm].copy()
    if o.empty:
        return pd.DataFrame()
    o["w"] = np.sqrt(pd.to_numeric(o["oi"], errors="coerce").fillna(0.0).clip(lower=1.0))

    rows: list[dict] = []
    for dt, g in o.groupby("dt", sort=True):
        fits: list[tuple[int, float, float, float]] = []
        for exp, gg in g.groupby("expiry", sort=False):
            fit = OP.fit_smile(gg["money"].to_numpy(), gg["iv"].to_numpy(),
                               gg["w"].to_numpy())
            if fit is None:
                continue
            fits.append((int(gg["dte"].iloc[0]), *fit))
        if not fits:
            continue
        fits.sort(key=lambda x: x[0])
        dte, iv, skew, curv = fits[0]
        rec = {"variety": variety, "period": period, "dt": dt,
               "iv_near": round(iv, 4), "dte_near": dte,
               "skew": round(skew, 4), "curv": round(curv, 4),
               "n_opt": int(len(g)),
               "pcr": float(pcr.get(dt, np.nan))}
        if len(fits) > 1:
            rec["dte_next"], rec["iv_next"] = fits[1][0], round(fits[1][1], 4)
        rows.append(rec)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    for c in OS.IV_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[["variety", "period", "dt"] + OS.IV_COLS]


# ------------------------------------------------------------------ 主流程


# 每跑这么多条链强制重建一次会话。
#
# `DataDownloader` 内部会对每个合约 `get_quote_list`，订阅**只增不减**：跑上
# 几十条链之后累计上千个订阅，行情服务端会把连接踢掉，然后就是无限「断开 /
# 重连」，一条都下不动。
#
# 但重建很贵 —— 老连接 `close()` 加新连接握手实测要 80 秒，每 10 条链重建一次
# 等于每条链白加 8 秒。所以这里改成**按症状重建**：连续 3 条链一条数据都没拿到
# 才认为连接废了。60 是兜底上限，防止真出了别的问题时永远不清理。
RECYCLE_EVERY = 60
DEAD_AFTER = 3


# 期权最长挂牌期。商品期权一般提前 1 年挂，股指期权只有 1~3 个月。
# 用它把下载起点抬到「到期前 400 天」：十年区间和一年的下载量差 20 倍，
# 而十年之前那段对这条链来说必然是空表。
LIST_LEAD_DAYS = 400


def _chain_start(start: date, expiry) -> date:
    """这条链的下载起点：全局起点和「到期前 400 天」取大的那个。"""
    try:
        e = pd.Timestamp(expiry).date()
    except (TypeError, ValueError):
        return start
    lo = e - timedelta(days=LIST_LEAD_DAYS)
    return max(start, lo)


def _plan(chains: dict[str, list[str]], period: str,
          varieties: list[str] | None) -> dict[str, list[str]]:
    """挑出这一轮要下的链。

    `5m` 只做中金所三个股指期权（IO/MO/HO）—— 商品期权 5m 十年是几百 GB，
    而用户要的 5m 只用来做股指期权的日内研究。
    """
    want = set(varieties) if varieties else None
    out: dict[str, list[str]] = {}
    for key, members in chains.items():
        v = variety_of(members[0])
        if want is not None:
            if v not in want:
                continue
        elif period == "5m" and v not in CFFEX_5M:
            continue
        out[key] = members
    return out


def update(sess, periods: tuple[str, ...] = ("1d", "1h"),
           varieties: list[str] | None = None,
           start: date = DEFAULT_START, end: date | None = None,
           incremental: bool = True, progress=None, cancel=None,
           tmpdir: str | None = None) -> dict:
    """把期权行情 + IV 摘要拉进库。

    `sess` 要一个 `tqfetch.Session`（有 `get` / `reset`）。传裸 api 也能跑，
    但就没有定期重建会话的保护了 —— 长任务请务必传 Session。

    增量口径是**按合约**：库里最后一根已经到（或超过）到期日前的，说明这个
    合约下完了，跳过。中途断了重跑就是断点续传。

    返回 `{period: {...}}`，每期有 `chains / rows / skipped / failed / iv`。
    """
    end = end or date.today()
    tmp = tmpdir or os.path.join(os.path.dirname(OS.db_path()), "opt_tmp")
    chains = list_options(sess.get() if hasattr(sess, "get") else sess)
    stats: dict[str, dict] = {}

    for period in periods:
        plan = _plan(chains, period, varieties)
        done = OS.done_contracts(period) if incremental else {}
        st = {"chains": len(plan), "rows": 0, "skipped": 0,
              "failed": 0, "iv": 0, "period": period}
        stats[period] = st
        total = len(plan)
        since_recycle = 0
        dead = 0            # 连续几条链一条数据都没拿到

        for n, (underlying, members) in enumerate(plan.items(), 1):
            if cancel is not None and cancel():
                st["cancelled"] = True
                break

            # 兜底重建：正常情况下不该走到这里，走一次要 80 秒
            if since_recycle >= RECYCLE_EVERY and hasattr(sess, "reset"):
                sess.reset()
                since_recycle = 0
            since_recycle += 1

            try:
                api = sess.get() if hasattr(sess, "get") else sess
                info = chain_info(api, members)
            except Exception:      # noqa: BLE001
                if hasattr(sess, "reset"):
                    sess.reset()
                st["failed"] += 1
                continue
            if info is None:
                st["failed"] += 1
                continue
            exp = pd.Timestamp(info["expiry"])
            need = [c for c in members
                    if c not in done or pd.Timestamp(done[c]) < exp - pd.Timedelta(days=1)]
            if not need:
                st["skipped"] += 1
                if progress is not None:
                    progress(n, total, f"{period} {n}/{total}　{underlying} 已是最新")
                continue

            if progress is not None:
                progress(n, total, f"{period} {n}/{total}　{underlying}"
                                   f"（{len(need)} 个合约）")

            rows = []
            for c in need:
                pc = parse_contract(c)
                if pc is None:
                    continue
                rows.append({"contract": c, "variety": variety_of(c),
                             "exchange": c.partition(".")[0],
                             "opt_class": pc[0], "strike": pc[1],
                             "expiry": info["expiry"], "underlying": info["underlying"]})
            meta = pd.DataFrame(rows)
            if meta.empty:
                continue

            try:
                bars, und = download_chain(api, info["underlying"], need, period,
                                           _chain_start(start, exp), end, tmp,
                                           cancel=cancel)
            except Exception:      # noqa: BLE001
                # 连接多半已经废了：重建会话再来一次。还失败就记一笔继续，
                # 一条链失败不该让整轮停住 —— 重跑时它会被重新捡起来
                bars, und = _empty_bars(), pd.DataFrame(
                    columns=["contract", "period", "dt", "close"])
                if hasattr(sess, "reset"):
                    sess.reset()
                    api = sess.get() if hasattr(sess, "get") else sess
                    since_recycle = 0
                    try:
                        bars, und = download_chain(
                            api, info["underlying"], need, period,
                            _chain_start(start, exp), end, tmp, cancel=cancel)
                    except Exception:      # noqa: BLE001
                        pass
            if bars.empty and und.empty:
                st["failed"] += 1
                dead += 1
                if dead >= DEAD_AFTER and hasattr(sess, "reset"):
                    # 连着几条都空，多半不是「这条链本来就没数据」，而是连接
                    # 已经被订阅数压垮了：重建再继续
                    sess.reset()
                    since_recycle = 0
                    dead = 0
                continue
            dead = 0

            with OS.connect() as con:
                OS.upsert_meta(con, meta)
                st["rows"] += OS.upsert_bars(con, bars)
                OS.upsert_underlying(con, und)
                OS.set_meta(con, "last_update", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # IV 在整期下完之后算一次：近月/次近月要跨链
        if progress is not None:
            progress(total, total, f"{period} 计算隐含波动率…")
        vs = sorted({variety_of(c) for m in plan.values() for c in m})
        for v in vs:
            if cancel is not None and cancel():
                break
            iv = compute_iv(v, period)
            if iv.empty:
                continue
            with OS.connect() as con:
                st["iv"] += OS.upsert_iv(con, iv)

        export_csv(period)

    return stats


def load_contract(contract: str, period: str = "1d") -> pd.DataFrame:
    """单个期权合约的 K 线，列和主连同一套，界面 / Agent 可以直接用。

    返回 `dt/open/high/low/close/volume/open_oi/close_oi`，按时间升序。
    库里没有就返回空表（不是报错）—— 期权是可选数据，用户可能没下。
    """
    df = OS.load_bars([contract], period)
    if df.empty:
        return df
    cols = ["dt", "open", "high", "low", "close", "volume", "open_oi", "close_oi"]
    out = df[[c for c in cols if c in df.columns]].copy()
    out["dt"] = pd.to_datetime(out["dt"])
    return out.sort_values("dt").reset_index(drop=True)


def load_chain_bars(underlying: str, period: str = "1d") -> pd.DataFrame:
    """一条链（同一标的月份的全部期权）的全部 K 线，算横截面用。"""
    chains = OS.chains()
    members = chains.get(underlying)
    if not members:
        return pd.DataFrame()
    return OS.load_bars(members, period)


# ------------------------------------------------------------------ 导出 CSV


def export_csv(period: str, out_dir: str | None = None) -> str:
    """把 IV 摘要导成 CSV，和 Tushare 那份**完全分开**。

    天勤口径（收盘价）和 Tushare 口径（结算价）算出来的 IV 差得不是一点点，
    混在一起用会得出互相矛盾的结论。文件名带 `tq_` 前缀就是为了让人一眼看出
    来源。
    """
    from .paths import data_dir

    d = out_dir or os.path.join(
        os.path.dirname(data_dir("cn").rstrip("/\\")), "data_tq", "option_iv"
    )
    os.makedirs(d, exist_ok=True)
    df = OS.load_iv(period=period)
    path = os.path.join(d, f"tq_iv_{period}.csv")
    if df.empty:
        return path
    wide = df.pivot_table(index="dt", columns="variety", values="iv_near",
                          aggfunc="last")
    wide.index.name = "dt"
    wide.to_csv(path, float_format="%.4f")
    return path
