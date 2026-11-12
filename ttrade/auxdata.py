"""衍生数据（期限结构 / 期权波动率）的读取与挂载。

## 为什么单独一层

`data_futures/` 里是主力连续合约，只有一条价格曲线。展期因子需要**不同到期日
的多个合约**，期权因子需要**隐含波动率**——这两类信息不在主力连续里，必须另外
从真实合约和期权明细聚合出来。

聚合结果放在 `data_futures_aux/`：

- `{SYM}_{EX}.csv`：期限结构，每日一行，近/次近/第三近合约的收盘价、到期天数、持仓量
- `opt_{SYM}_{EX}.csv`：期权波动率，每日一行，近/次月平值 IV、偏度、看跌看涨持仓比

## 为什么挂在 df 上，而不是给因子函数加参数

因子函数的签名是 `f(df) -> Series`，`factors.raw()` / `factor_frame()` /
`standardize()` 全部按这个约定工作。如果给展期因子单独加一个 `aux=` 参数，
就要在策略、机器学习、因子分析页三条链路上各加一次分支——分支越多越容易漏。

改成「加载时把衍生列并进 df」，因子函数退化成读列，其它代码一行都不用改。
美股没有这些列，读出来是 NaN，`filter_available()` 会自动把它们从美股因子
列表里剔掉，不需要额外的市场判断。

本模块只依赖 pandas 和标准库，**不导入 tushare**——GUI 启动时就会用到它。
"""

from __future__ import annotations

import os

import pandas as pd

# 期限结构列，三套口径并列：
#   c*  = 纯按到期日（原口径）
#   o3_* / o6_* = 先按持仓量筛出前 3 / 前 6 个活跃合约，再按到期日取最近三腿
# 三套共用同一批列名后缀（`{pre}1_close`），因子层用同一个函数就能算三遍。
CURVE_COLS = [
    "c1_close", "c1_dte", "c1_oi",
    "c2_close", "c2_dte", "c2_oi",
    "c3_close", "c3_dte", "c3_oi",
]
for _n in (3, 6):
    for _i in (1, 2, 3):
        CURVE_COLS += [f"o{_n}_{_i}_close", f"o{_n}_{_i}_dte", f"o{_n}_{_i}_oi"]
# 期权列
OPT_COLS = ["iv_near", "iv_next", "dte_near", "dte_next", "skew", "curv", "pcr", "n_opt"]

_cache: dict[str, tuple[float, pd.DataFrame]] = {}
# IV 档位解析结果（品种 + bar 档 → `1d`/`1h`/`5m`），键里带期权库 mtime
_ivp_cache: dict[str, tuple[float, str]] = {}


def curve_path(aux_dir: str, symbol: str, exchange: str) -> str:
    return os.path.join(aux_dir, f"{symbol}_{exchange}.csv")


def opt_path(aux_dir: str, symbol: str, exchange: str) -> str:
    return os.path.join(aux_dir, f"opt_{symbol}_{exchange}.csv")


def _read(path: str, cols: list[str]) -> pd.DataFrame:
    """带 mtime 校验的读取缓存。更新数据后文件会变，缓存必须失效。"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return pd.DataFrame()
    hit = _cache.get(path)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "date" not in df.columns or df.empty:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("date").set_index("date").sort_index()
    _cache[path] = (mtime, df)
    return df


def load_curve(aux_dir: str, symbol: str, exchange: str) -> pd.DataFrame:
    """期限结构。aux 文件缺失时回退到天勤合约算出来的那份。

    两套文件互不覆盖：`data_futures_aux/` 是 Tushare 的 20 年口径，
    `data_tq/curve/` 是天勤合约（十年出头）。同一个日期混着用两个源会让展期值
    随机跳，所以价格类列只做「有就用、没有才退」，不做合并。

    **唯一例外是 `roll_intra`**：日内价差漂移只有分钟级真实合约算得出来，
    Tushare 那份（日线合约）根本没有这一列。它不是「另一个源的同名数据」，
    而是天勤独有的补充列，所以单独并进来，缺日期就是空。
    """
    df = _read(curve_path(aux_dir, symbol, exchange), CURVE_COLS)
    alt = _read(_tq_curve_path(symbol, exchange), ["roll_intra"])
    if df.empty:
        return alt
    if not alt.empty and "roll_intra" in alt.columns and "roll_intra" not in df.columns:
        df = df.join(alt[["roll_intra"]], how="left")
    return df


def _tq_curve_path(symbol: str, exchange: str) -> str:
    """天勤期限结构的路径。从**数据根目录**推，不从 aux_dir 推 —— aux_dir 可能
    是测试用的临时目录，按它推会指到 Temp 下面去。"""
    from .paths import data_dir as _data_dir

    return os.path.join(
        os.path.dirname(_data_dir("cn").rstrip("/\\")),
        "data_tq", "curve", f"{symbol}_{exchange}.csv",
    )


def load_opt(aux_dir: str, symbol: str, exchange: str,
             period: str = "1d") -> pd.DataFrame:
    """期权 IV 摘要（`period` 档：`1d` / `1h` / `5m`）。

    **为什么会有本地回退**：Tushare 那条链路（`opt_daily` + `settle_map`）要靠
    「标的期货当天结算价」当 Black-76 里的 F，而**股指期权的标的是指数**
    （沪深300 / 上证50 / 中证1000），不是期货合约 —— 于是
    `data_futures_aux/opt_IF_*.csv` 从来就没生成过，IF / IH / IM 的 5 个 IV 因子
    一直是空的。本地天勤库有这三家的 IV 摘要（IO / HO / MO），只是品种名和期货
    不一样（见 `optstore.INDEX_FUT_OPT`），映射过来就能用。

    两套数据口径不同（Tushare 用结算价、天勤用收盘价），所以**不做合并**：
    aux 有就用 aux，没有才回退，同一个品种永远只吃一个源。

    🔴 **Tushare 那份只有日频**（`opt_*.csv` 是日线汇总）。所以 `period`
    不是 `1d` 时直接走本地库 —— 拿日频冒充小时频，挂上去的值会是一天一个台阶，
    `iv_atm` 看着有数其实是常数。
    """
    if period == "1d":
        df = _read(opt_path(aux_dir, symbol, exchange), OPT_COLS)
        if not df.empty:
            return df
    return _opt_local(symbol, period)


def _opt_local(symbol: str, period: str = "1d") -> pd.DataFrame:
    """本地期权库兜底，按期权库文件的 mtime 缓存（下载完要能立刻看到新值）。"""
    from . import optstore as OS

    db = OS.db_path()
    try:
        mtime = os.path.getmtime(db)
    except OSError:
        return pd.DataFrame()
    key = f"local::{symbol}::{period}"
    hit = _cache.get(key)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    df = OS.load_iv_for(symbol, period=period)
    if df.empty:
        return df
    _cache[key] = (mtime, df)
    return df


# 期权 IV 摘要在库里就这几档（`option_iv.period`）。分钟 bar 优先用**同一档**，
# 没有就退一档更粗的（粗档用 asof 对齐同样合法，只是日内变化少一些），
# 全都没有才退回日频那套「D-1 快照铺满当天」。
_IV_PERIODS = ("5m", "15m", "30m", "1h", "2h", "4h")


def bar_period(df: pd.DataFrame) -> str:
    """从 bar 间隔猜周期：`5m` / `15m` / … / `1h` / `2h` / `4h`；日线返回 `1d`。

    用**相邻 bar 的中位间隔**而不是「每天几根」：夜盘、半天交易日（春节前）、
    停牌都会把「根/天」拉偏，而间隔是分钟级的硬事实。
    """
    idx = pd.DatetimeIndex(df.index)
    if len(idx) < 3:
        return "1d"
    gaps = pd.Series(idx).diff().dt.total_seconds().dropna()
    # 跨夜 / 跨周末的间隔是几十个小时，会把中位数污染掉 —— 只留 4 小时以内
    gaps = gaps[(gaps > 0) & (gaps <= 4 * 3600)]
    if gaps.empty:
        return "1d"
    minutes = float(gaps.median()) / 60.0
    if minutes <= 0:
        return "1d"
    if minutes < 60:
        return f"{int(round(minutes))}m"
    return f"{int(round(minutes / 60))}h"


def _pick_iv_tier(cands: list[str], spans: dict, day_span
                  ) -> str:
    """从候选档里挑第一个「够新 + 够深」的；都不合格返回 `1d`。

    抽成纯函数是为了能单测：闸门条件写错不会报错，只会悄悄用一个烂档，
    表现是图上一条水平线或者精度前后跳 —— 靠看数据发现不了。
    """
    for p in cands:
        span = spans.get(p)
        if span is None:
            continue
        # 在更：最晚一条比日频晚不到一个月
        if day_span is not None and span[2] < day_span[2] - pd.Timedelta(days=30):
            continue
        # 深度：天数至少覆盖日频的一半
        if day_span is not None and span[3] < day_span[3] * 0.5:
            continue
        return p
    return "1d"


def _iv_period_for(df: pd.DataFrame, symbol: str) -> str:
    """这份 bar 该用哪一档 IV：先同档，再往粗档退，都没有返回 `1d`。"""
    from . import optstore as OS

    own = bar_period(df)
    if own == "1d":
        return "1d"
    # 解析结果按「品种 + bar 档 + 期权库 mtime」缓存：分钟回测一次要问几十次，
    # 每次都开一个 DuckDB 连接划不来
    try:
        stamp = os.path.getmtime(OS.db_path())
    except OSError:
        return "1d"
    key = f"ivp::{symbol}::{own}"
    hit = _ivp_cache.get(key)
    if hit is not None and hit[0] == stamp:
        return hit[1]
    try:
        slot = _IV_PERIODS.index(own)
    except ValueError:
        slot = 0
    # 同档 → 比它粗的。粗档覆盖更长，退档时不会突然只剩几个月有数
    cands = [own] + list(_IV_PERIODS[slot + 1:])
    day_span = OS.iv_span(symbol, "1d")
    spans = {p: OS.iv_span(symbol, p) for p in cands}
    picked = _pick_iv_tier(cands, spans, day_span)
    _ivp_cache[key] = (stamp, picked)
    return picked



def attach(
    df: pd.DataFrame,
    symbol: str,
    exchange: str,
    aux_dir: str | None = None,
) -> pd.DataFrame:
    """把衍生列按日期并到行情表上。缺文件时原样返回（不报错）。

    用 `join` 而不是 `merge`：df 已经按 date 建了索引，join 能保证不改变行数
    和顺序——K 线渲染、信号对齐都依赖「第 i 行就是第 i 根 K 线」。
    万一衍生文件里有行情表没有的日期，`how="left"` 会直接丢掉，不会多出行。
    """
    if aux_dir is None:
        aux_dir = _default_dir()
    # 目录不存在也往下走：期限结构还有一份天勤合约算出来的备份
    # （`data_tq/curve/`），没有 Tushare 数据的人全靠它
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    parts = [load_curve(aux_dir, symbol, exchange), load_opt(aux_dir, symbol, exchange)]
    out = df
    for part in parts:
        if part.empty:
            continue
        cols = [c for c in part.columns if c not in out.columns]
        if not cols:
            continue
        out = out.join(part[cols], how="left")
    # `join` 会丢掉 attrs（pandas 的 merge 系列不传播 attrs），而 `symbol` 就
    # 挂在 attrs 上 —— 丢了之后 `factors_ext._symbol()` 返回空串，基本面（EDB
    # 仓单 / 库存 / 现货 / 交割 / 利润）、板块这一整片因子全部退化成全 NaN，
    # 因子分析页上写着「无数据」，看起来像数据没下。手动挂回去。
    out.attrs = dict(df.attrs)
    return out


def _trade_day(df: pd.DataFrame) -> pd.Series:
    """每根 bar 属于哪个交易日。

    **不能拿 `index.normalize()` 当交易日**：天勤的夜盘 bar 时间戳是 21:00
    （自然日 D-1），但按交易所口径它属于下一个交易日 D。`tqdata.trading_days()`
    在写入库时就算好了 `trading_day` 列，这里直接用，和 `factors_intraday._day`、
    `factors_seasonal._day` 保持同一个口径。

    按自然日对齐的后果不是前视，是**多用了一天的旧值**：夜盘那段取到的是 D-2
    的快照（因为按 D-1 去 shift(1)），于是同一个交易日里夜盘和日盘读到两个不同
    的展期 / IV 值 —— 在日内因子和 ML 特征里就是一条凭空出现的跳变。
    """
    idx = pd.DatetimeIndex(df.index)
    if "trading_day" in df.columns:
        d = pd.to_datetime(df["trading_day"], errors="coerce")
        if d.notna().any():
            # 实盘那条链路是「本地历史 + 实时 bar」拼出来的，而实时段的
            # `bars_from_serial` 只有 OHLC / 持仓量，`trading_day` 是 NaT。
            # 不补的话这些 bar 会落进 NaT 那一组，拿到一条来路不明的快照。
            # 补法用同一套规则：20:00 之后算次日（落在周末也没关系，下面的
            # ffill 会兜到上一个有快照的交易日）。
            nat = d.isna()
            if nat.any():
                guess = pd.Series(idx.normalize(), index=df.index)
                guess = guess.where(idx.hour < 20, guess + pd.Timedelta(days=1))
                d = d.fillna(guess)
            return d
    return pd.Series(idx.normalize(), index=df.index)


def attach_intraday(
    df: pd.DataFrame,
    symbol: str,
    exchange: str,
    aux_dir: str | None = None,
) -> pd.DataFrame:
    """把衍生列挂到**分钟级**行情上：每根 bar 用「上一个交易日」的快照。

    日线可以直接 `attach()`：那根 K 线的收盘价和期限结构是同一天同一时刻算出来
    的，谁都不领先谁。分钟线不行 —— 同一天 09:05 那根 bar 不可能知道当天 15:00
    收盘之后才有的近远月价差，按日期硬 join 等于把当天收盘的结果提前灌进上午的
    bar，展期因子的 IC 会被这条未来信息吹起来，实盘一上就露馅。

    所以先 `ffill` 把衍生表铺成「每个交易日一行」，再整体 `shift(1)`：第 D 个
    交易日的所有 bar 都用 D-1 收盘时的快照，正好是实盘那一刻能拿到的信息。
    代价是窗口最前面一天没有值，`_col()` 会返回 NaN，因子自然空着。

    为什么不像日线那样 join：分钟表的索引是「时间戳」不是「日期」，直接 join
    只会命中 00:00 那一行（分钟 bar 根本没有这个点），整列还是 NaN —— 这正是
    1h / 5m 的展期因子长期没数值的原因。
    """
    if aux_dir is None:
        aux_dir = _default_dir()
    # 目录不存在也往下走：期限结构还有一份天勤合约算出来的备份
    # （`data_tq/curve/`），没有 Tushare 数据的人全靠它
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    day = _trade_day(df)
    # 补齐的交易日要写回列里。下游 `factors_intraday._day` / `_seasonal._day`
    # 读的是这一列 —— 只在本地算 `day` 不回写的话，实盘拼出来的表（实时段
    # trading_day 是 NaT）到因子那一层又退回自然日，夜盘归错天。
    if "trading_day" in df.columns:
        raw = pd.to_datetime(df["trading_day"], errors="coerce")
        if raw.isna().any():
            df = df.copy()
            df["trading_day"] = day
    parts = [load_curve(aux_dir, symbol, exchange)]
    out = df
    # 期限结构只有日频（Tushare 合约表 / 天勤 curve 都是按天算的）
    for part in parts:
        out = _join_day_snapshot(out, part, day)

    ivp = _use_iv_period(aux_dir, symbol, exchange, df)
    opt = load_opt(aux_dir, symbol, exchange, period=ivp)
    if ivp == "1d" or opt.empty:
        out = _join_day_snapshot(out, opt, day)
    else:
        out = _join_bar_asof(out, opt)
        # 日内档的**覆盖窗口**各家不一样（IO 的 5m 只到 2020-11、MO 根本没有
        # 5m），窗口外不能沿用最后一条快照 —— 那会把几年前的 IV 一路搬到今天，
        # 图上是一条水平线，因子里还当成有效值。缺口用日频补回来。
        miss = [c for c in opt.columns if c in out.columns
                and pd.api.types.is_numeric_dtype(opt[c]) and out[c].isna().any()]
        if miss:
            daily = load_opt(aux_dir, symbol, exchange, period="1d")
            fill = _daily_snapshot(daily, day, miss, out.index)
            for c in fill.columns:
                out[c] = out[c].where(out[c].notna(), fill[c])
    # 同上：join 丢 attrs，`symbol` 一丢基本面 / 板块因子整列 NaN
    out.attrs = dict(df.attrs)
    return out


def _use_iv_period(aux_dir: str, symbol: str, exchange: str,
                   df: pd.DataFrame) -> str:
    """这段 BAR 的期权 IV 该取哪一档。

    **只有走本地库的品种**（股指那三家，没有 Tushare 文件）才用日内档；有 aux
    文件的商品仍然吃 Tushare 日频 —— 同一个品种不能换个周期就换数据源，否则
    1d 和 5m 两张图上的 `iv_atm` 水平不一样，看起来像「换周期策略变强了」。

    另一层闸门是**档位自身要够新**：库里 IO 的 5m 只到 2020-11（日频到 2026），
    拿它当主档的话 2020 那八个月是 5m 精度、剩下六年全靠日频兜底，精度来回
    跳。这种情况退到更粗的 `1h`，全天候一个精度。
    """
    try:
        if os.path.exists(opt_path(aux_dir, symbol, exchange)):
            return "1d"
    except OSError:
        return "1d"
    return _iv_period_for(df, symbol)


def _daily_snapshot(part: pd.DataFrame, day: pd.Series, cols: list[str],
                    index: pd.Index) -> pd.DataFrame:
    """日频衍生列 → `index` 上「上一个交易日」的快照。

    union 会把「行情有、衍生没有」的交易日补进来（比如衍生只更到上周），
    ffill 之后再 shift，取到的就是「这根 bar 之前最近一个有快照的交易日」。
    """
    cols = [c for c in cols if c in part.columns]
    if part.empty or not cols:
        return pd.DataFrame(index=index)
    grid = part.index.union(day.unique()).sort_values()
    snap = part[cols].reindex(grid).ffill().shift(1).reindex(day.unique())
    return pd.DataFrame(
        snap.to_numpy()[pd.Index(day.unique()).get_indexer(day)],
        index=index, columns=cols,
    )


def _join_day_snapshot(out: pd.DataFrame, part: pd.DataFrame,
                       day: pd.Series) -> pd.DataFrame:
    """日频衍生列 → 每根 bar 用「上一个交易日」的快照。"""
    if part.empty:
        return out
    # 分钟表动辄十几万行，合约代码（c1_code 之类）是字符串列，因子用不到，
    # 铺到每根 bar 上只是白占几十 MB
    cols = [c for c in part.columns
            if c not in out.columns and pd.api.types.is_numeric_dtype(part[c])]
    if not cols:
        return out
    return out.join(_daily_snapshot(part, day, cols, out.index))


def _join_bar_asof(out: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    """分钟级衍生列 → 每根 bar 取**严格早于它**的最后一条快照。

    `allow_exact_matches=False` 不是细节：库里 1h 的 IV 快照时间戳和期货 1h
    bar 的时间戳是同几个整点，允许精确匹配就等于「用当根收盘算出来的 IV
    决定当根怎么下单」，回测会好看、实盘拿不到。

    退档（1h IV 挂到 5m bar 上）也走这条：拿到的还是「之前最近一小时」的读数，
    只是日内台阶粗一点，不会引入未来信息。
    """
    if part.empty:
        return out
    cols = [c for c in part.columns
            if c not in out.columns and pd.api.types.is_numeric_dtype(part[c])]
    if not cols:
        return out
    right = part[cols].sort_index()
    right = right[~right.index.duplicated(keep="last")].reset_index()
    right = right.rename(columns={right.columns[0]: "_ts"})
    left = pd.DataFrame({"_ts": pd.DatetimeIndex(out.index)})
    left = left.sort_values("_ts", kind="mergesort")
    merged = pd.merge_asof(left, right, on="_ts", direction="backward",
                           allow_exact_matches=False)
    vals = merged.set_index("_ts")[cols].reindex(pd.DatetimeIndex(out.index))
    # 覆盖窗口之外一律不填：`merge_asof` 的 backward 会**无限**沿用最后一条
    # 快照，而库里的日内档覆盖差别很大（IO 的 5m 只到 2020-11、MO 没有 5m）。
    # 不设这道闸，2020 年那条 IV 会被一直搬到 2026 年。
    lo = right["_ts"].min()
    hi = right["_ts"].max() + pd.Timedelta(days=1)
    idx = pd.DatetimeIndex(out.index)
    keep = (idx >= lo) & (idx <= hi)
    vals.loc[~keep, :] = float("nan")
    return out.join(vals)


def _default_dir() -> str:
    """衍生数据目录。走 `paths.data_dir("aux")` 而不是相对 `__file__` —— 打包后
    代码在 PYZ 里，`../data_futures_aux` 指到的是解包临时目录以外的地方。"""
    from .paths import data_dir

    return data_dir("aux")


def clear_cache() -> None:
    _cache.clear()


def summary(aux_dir: str) -> dict:
    """统计衍生数据覆盖情况，给界面和自检用。

    `stale` 是「列结构落后于当前代码」的曲线文件数。v1.4.0 给期限结构加了两套
    持仓量口径（o3_ / o6_），旧 aux 文件里没有这些列，对应的 8 个因子会静默变成
    全 NaN —— 与其让人对着一堆「算不出来」猜，不如在这里数出来，自检和界面
    可以提示重建（`rebuild_aux.py`）。
    """
    import json

    out = {"curve": 0, "opt": 0, "dates": 0, "first": "", "last": "", "stale": 0}
    if not os.path.isdir(aux_dir):
        return out
    need = [c for c in CURVE_COLS if c != "date"]
    for fn in os.listdir(aux_dir):
        if fn.startswith("opt_") and fn.endswith(".csv"):
            out["opt"] += 1
        elif fn.endswith(".csv") and not fn.startswith("_"):
            out["curve"] += 1
            try:
                # 只读表头一行，不用把整份 CSV 载进来
                with open(os.path.join(aux_dir, fn), encoding="utf-8-sig") as fh:
                    head = fh.readline()
            except Exception:
                out["stale"] += 1
                continue
            if any(c not in head for c in need):
                out["stale"] += 1
    man = os.path.join(aux_dir, "_manifest.json")
    if os.path.exists(man):
        try:
            with open(man, encoding="utf-8") as fh:
                dates = json.load(fh).get("dates") or []
            if dates:
                out["dates"] = len(dates)
                out["first"], out["last"] = dates[0], dates[-1]
        except Exception:
            pass
    return out
