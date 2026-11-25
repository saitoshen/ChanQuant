"""截面策略的**实盘决策**：每一根 bar「这一篮子该按什么权重持」。

## 为什么不能直接调 `cross.run()`

`cross.run()` 是**回测**：净值曲线、逐笔交易、逐品种拆账、绩效指标 —— 实盘一样
都用不上，只要「最后一根该持哪些品种、各占多少钱」。而它一跑就是整段历史
（实测 63 个品种 × 16 个因子），十几秒起步，每根 bar 跑一遍是不现实的。

所以这里只走 `run()` 里**决定权重的那三步**：

    分数面板 `score_panel` → 筛池 `_apply_pool` → 分钱 `_hedge_weights` / `_select_weights`

判据和回测**完全一致**，这是整个文件存在的意义：

  * 调仓日仍是 `np.arange(0, len(score), step)`（`_rank` 的原式子），也就是
    「分数面板最后一行的下标 ≡ 0 (mod step)」；
  * 池子仍只在 `_pool_tick_rows` 的时点上重排，其余行沿用上一次的蒙版
    （`_apply_pool` 里 `where(tick).ffill()` 干的事，状态存在 `pool_keep` 里）；
  * 杠杆仍乘 `leverage_vector()`（Σ|w| = 资金分配比例 ÷ 保证金率）；
  * 有效品种不足 `MIN_CROSS` 的行**沿用上一组**，不是清零 —— 回测里这种行被
    `_place` 跳过（保留 NaN → ffill），实盘清零就是凭空平仓。

任何一处自己换口径，实盘和回测就是两条曲线，而界面上一个字都看不出来。

## 权重 → 手数

`weights` 是「占本金的百分比」（Σ|w| = 杠杆倍数），所以

    名义额 = 本金 × w       手数 = 名义额 ÷ (价格 × 合约乘数)

价格取**能成交的真实价**（`_mark_price`：合约最新价，没有才用收盘），不是前复权
价 —— 前复权只用来算因子和收益，钱是按真实价付的。
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ttrade import cross as CX
from ttrade import factor_strategy as fstrat
from ttrade import replay


class CrossRefused(Exception):
    """这条配置在实盘里跑不出**一比一**的结果（不是算不出来，是口径复现不了）。

    宁可拦住并说清楚，也不能「看着在跑、实际是另一条策略」—— `enable()` 里那些
    拒绝分支都是同一个理由。
    """


def check_support(cfg) -> None:
    """启用前先判一遍。不通过抛 `CrossRefused`，消息直接给用户看。"""
    keys = [k for k in (getattr(cfg, "symbols", None) or []) if k]
    if len(keys) < CX.MIN_CROSS:
        raise CrossRefused(
            f"截面策略至少要 {CX.MIN_CROSS} 个品种才能横向比强弱，"
            f"这条只记了 {len(keys)} 个")
    if not [k for k in (getattr(cfg, "factor_keys", None) or [])]:
        raise CrossRefused("这条截面策略没有选中「排名因子」（第二步靠它决定谁强谁弱）")
    if str(getattr(cfg, "factor_mode", "") or "") == "ml":
        # 2026-11-24 用户要求参考因子 ML 的实盘方式放开：离线产物训一次，
        # 实盘只预测。没有产物 / 配置对不上时明确拦住，不能静默退化成
        # 「把 walk-forward 放到调仓那一刻跑」（几十秒，价格早变了）。
        from ttrade import cross_ml

        why = cross_ml.check_artifact(cfg)
        if why:
            raise CrossRefused(why)
    if str(getattr(cfg, "pool_mode", "off") or "off").lower() == "ml":
        # 排名腿的 ML 产物只装了 factor_keys 的模型；第一步筛池另用 pool_keys，
        # 需要另一份产物。没做之前明说，不要让它每根调仓都跑一遍 walk-forward。
        raise CrossRefused(
            "「第一步筛池」用机器学习暂不支持实盘（产物只覆盖第二步排名因子）；"
            "把筛池口径换成单因子 / 多因子")
    if str(getattr(cfg, "cross_mode", "hedge") or "hedge") == "select":
        fam = str(getattr(cfg, "timing_family", "rule") or "rule")
        if fam == "rule" and not str(getattr(cfg, "timing_rule_key", "") or ""):
            raise CrossRefused(
                "「选品种 + 时序」这条策略没选时序规则 —— 方向由它逐品种给，"
                "缺了这条腿就只剩「池内全做多」")
        if fam == "factor" and not (getattr(cfg, "timing_factor_keys", None) or []):
            raise CrossRefused("「选品种 + 时序」这条策略没选时序因子")


def union_index(frames: dict) -> pd.DatetimeIndex:
    """一批行情的时间并集（**升序去重**）。

    和 `cross.close_panel(frames).index` 是同一个集合，但不建那张宽表 —— 判
    「今天是不是调仓日」只需要下标，63 个品种 × 十几万行拼一遍纯属浪费。
    """
    parts = [f.index for f in (frames or {}).values() if f is not None and len(f)]
    if not parts:
        return pd.DatetimeIndex([])
    if len(parts) == 1:
        return pd.DatetimeIndex(parts[0]).sort_values()
    cat = parts[0]
    for p in parts[1:]:
        cat = cat.union(p)
    return pd.DatetimeIndex(cat).sort_values()


def round_lots(raw: float, prev: int = 0) -> int:
    """名义额折出来的手数 → **四舍五入取整**（用户 2026-11-24 明确口径）。

    🔴 不能用 Python 内建 `round()`：它是银行家舍入（round(1.5)=2、round(2.5)=2），
    和「四舍五入」不是一回事。这里按绝对值 `floor(|x| + 0.5)` 再带回符号：
    1.5→2、2.5→3、−1.5→−2。

    原来这里加过 0.6/0.4 的滞回（无仓时够 0.6 才开、有仓时剩不到 0.4 才平），
    目的是防 0.5 手附近来回打。用户要求**纯四舍五入**，就把滞回去掉 —— 权重
    本身按调仓日才变，一天之内的价格抖动不会每 tick 重算手数，本来也打不起来。

    `prev` 只在 `raw` 是 NaN（某个品种没价 / 没乘数）时用来**保持原仓**：丢一个
    价格不该把那一腿平掉。有有效数字时它不参与取整。
    """
    try:
        raw = float(raw)
    except (TypeError, ValueError):
        return int(prev or 0)
    if not math.isfinite(raw):
        return int(prev or 0)
    if raw == 0:
        return 0
    sign = 1 if raw > 0 else -1
    return sign * int(math.floor(abs(raw) + 0.5))


@dataclass
class CrossLive:
    """一条截面策略的实盘状态机。**每根 bar 调一次 `decide()`**。"""

    #: 父配置（`StrategyConfig`，`family == "cross"`）
    cfg: object
    #: 回测配置（`replay.make_bt` 那一份，杠杆 / 保证金从它读）
    bt: object

    #: 逐品种的行情缓存：`{品种键: (bar_id, df)}`。整篮子要**同一时刻**的行情，
    #: 不能一边算权重一边各腿自己更新 —— 那是「A 已经换到新 bar、B 还停在上一根」
    frames: dict = field(default_factory=dict)
    #: 上一次算权重时各腿的 bar_id 指纹。指纹没变就整段跳过（每 tick 都重算
    #: 一遍分数面板是不可能的）
    sig: object = None
    #: 上一次「重排品种池」采样出来的蒙版：`{品种键: 留不留}`
    pool_keep: dict | None = None
    pool_at: object = None
    #: `len(score) - len(面板索引)`：`dropna(how="all")` 掉的是开头那截预热，
    #: 数量在一段行情里恒定。拿它把「面板行数」换算成「分数行数」，就不必为了
    #: 判「今天是不是调仓日」把整张分数面板重算一遍
    drop: int | None = None

    weights: pd.Series | None = None
    at: object = None            # 上一次真正算出权重的 bar 时间
    step: int = 1
    bpd: float = 1.0
    n_calc: int = 0
    cost: float = 0.0            # 上一次计算花了多少秒
    note: str = ""
    #: 截面 ML 实盘预测器（离线产物，只预测不训练）。由 `PortfolioRunner`
    #: 在启用时挂上；单因子 / 多因子策略这里是 None。
    ml: object | None = None
    ml_note: str = ""            # 模型新鲜度提示（旧了就写进调仓日志）

    # ------------------------------------------------------------ 行情

    def frame_sig(self) -> tuple:
        """各腿 bar_id 的指纹。用来判「有没有新 bar」。"""
        return tuple(sorted((k, int(v[0])) for k, v in self.frames.items()))

    def put(self, key: str, bar_id: int, df: pd.DataFrame) -> None:
        self.frames[key] = (int(bar_id), df)

    def get(self, key: str):
        got = self.frames.get(key)
        return got[1] if got else None

    # ------------------------------------------------------------ 决策

    def due(self, frames: dict) -> tuple[bool, str]:
        """**便宜地**判「这根 bar 值不值得真算一次」。调用方拿它挡掉「去读别的周期
        行情」这类无谓开销，真正算不算还得看 `decide()` 里的精确判据。

        闸门先拿「上次算出来的分数行数 + 之后新增的行数」估一下。估得准时一次
        多余的计算都不做；估不准（行情被补过、多出一截）也只会让计算提前或推迟
        一根，下一次自动校正 —— 这里只是**触发器**，不是最终判据。
        """
        idx = union_index(frames)
        if len(idx) < 2:
            return False, "行情不足 2 根"
        period = str(getattr(self.cfg, "period", "") or "")
        ccfg = replay.cross_config(self.cfg, self.bt, period=period)
        self.bpd = fstrat.bars_per_day(idx)
        self.step = max(1, int(ccfg.step(self.bpd)))
        L = len(idx) + int(self.drop or 0)
        if self.drop is not None and self.at is not None \
                and (L - 1) % self.step \
                and int((idx > self.at).sum()) < 2 * self.step:
            return False, f"未到调仓日（第 {L} 根，每 {self.step} 根一次）"
        return True, ""

    def decide(self, frames: dict, step1_frames: dict | None = None,
               step2_frames: dict | None = None,
               factor_extra: dict | None = None,
               timing_extra: dict | None = None
               ) -> tuple[pd.Series | None, str]:
        """这一根该持的权重（已含杠杆）。返回 `(weights | None, 说明)`。

        `None` 有三种含义，调用方**一律当作「不动仓」**：

          ① 今天不是调仓日（绝大多数情况）；
          ② 有效品种不足 `MIN_CROSS`（回测里这一行被 `_place` 跳过 = 继续持上一组）；
          ③ 算不出来（`说明` 里写着原因，调用方记一次日志）。
        """
        due, note = self.due(frames)
        if not due:
            return None, note
        # `due()` 已经把 bpd / step 算好，这里只补上真算要用的索引和配置。
        # `step1_frames` / `step2_frames` 要在**有帧**时才传 —— `cross_config` 按
        # 「配了周期且真的拿到那份行情」决定 `step1_period` 生不生效
        idx = union_index(frames)
        period = str(getattr(self.cfg, "period", "") or "")
        ccfg = replay.cross_config(self.cfg, self.bt, period=period,
                                   step1_frames=step1_frames,
                                   step2_frames=step2_frames)
        step = self.step

        t0 = time.time()
        try:
            w, note = self._weights_once(frames, ccfg, step, idx,
                                         step1_frames=step1_frames,
                                         step2_frames=step2_frames,
                                         factor_extra=factor_extra,
                                         timing_extra=timing_extra)
        finally:
            self.cost = time.time() - t0
            self.n_calc += 1
        self.note = note
        return w, note

    def _weights_once(self, frames, ccfg, step: int, idx,
                      step1_frames=None, step2_frames=None,
                      factor_extra=None, timing_extra=None):
        """真算一遍。**只在调仓日调用**（一根 bar 几十毫秒到十几秒）。"""
        src = step1_frames if (ccfg.step1_period and step1_frames) else frames
        src2 = step2_frames if (ccfg.step2_period and step2_frames) else frames
        fcfg = CX._factor_cfg(ccfg)
        self.ml_note = ""
        if str(ccfg.factor_mode or "") == "ml":
            # ML 走离线产物：实盘只重算因子 + predict，绝不在调仓那一刻
            # 重跑 walk-forward（几十秒，价格早变了）。产物在 enable 时挂上。
            if self.ml is None:
                return None, "截面 ML 没挂训练产物（在组合页点「训练模型」后重启）"
            try:
                score_src = self.ml.predict(src2)
            except Exception as exc:                  # noqa: BLE001
                return None, f"截面 ML 预测失败：{type(exc).__name__}: {exc}"
            age = float(getattr(self.ml, "age_days", 0.0) or 0.0)
            if age >= 14:
                self.ml_note = f"；模型 {age:.0f} 天前训练，建议重训"
        else:
            try:
                score_src = CX.score_panel(
                    src2, list(ccfg.factor_keys or []), ccfg.factor_reverse,
                    mode=ccfg.factor_mode or "multi", fcfg=fcfg,
                    label_mode=ccfg.label_mode, factor_extra=factor_extra,
                    base_period=(ccfg.step2_period or ccfg.period))
            except Exception as exc:                  # noqa: BLE001
                return None, f"截面分数算不出来：{exc}"
        if score_src is None or score_src.empty:
            return None, "截面分数全空（选中的因子在这批品种上都没有数据）"

        # ---- 第一步：筛池（带状态，见 `_pool_apply`）
        if str(ccfg.pool_mode or "off").lower() != "off" and ccfg.pool_keys:
            try:
                pool_src = CX.score_panel(src, list(ccfg.pool_keys),
                                          ccfg.pool_reverse, mode=ccfg.pool_mode,
                                          fcfg=fcfg, label_mode=ccfg.label_mode)
            except Exception as exc:                  # noqa: BLE001
                return None, f"第一步选品种失败：{exc}"
            if src2 is not src:
                pool_src = CX.align_to_clock(pool_src, score_src.index)
            score_src = self._pool_apply(score_src, pool_src, ccfg)
            if score_src.dropna(how="all").empty:
                return None, "第一步筛完之后池子是空的（把保留比例调大）"

        # 分数从源周期的时钟搬到策略时钟。**同周期不搬** —— `align_to_clock`
        # 会整体 shift(1)，同一份数据上等于凭空多滞后一根
        score = score_src if src2 is frames else CX.align_to_clock(
            score_src, CX.close_panel(frames).index)
        score = score.dropna(how="all")
        if score.empty:
            return None, "分数搬不到策略的时钟上（两边日期不重叠？）"

        # 校准「面板行数 ↔ 分数行数」的差：`dropna(how="all")` 掉的是开头那截
        # 预热，数量在一段行情里恒定
        self.drop = int(len(score) - len(idx))
        L = int(len(score))
        if (L - 1) % step:
            return None, f"未到调仓日（第 {L} 根，每 {step} 根一次）"

        last = score.iloc[[-1]]
        n_ok = int(last.notna().sum(axis=1).iloc[0])
        if n_ok < CX.MIN_CROSS:
            return None, (f"这一根只有 {n_ok} 个品种算得出分数（要 {CX.MIN_CROSS} 个），"
                          "按回测口径保持上一组，不调仓")

        vol = (CX.vol_panel(frames, score.index).iloc[[-1]]
               if str(ccfg.weight or "") == "risk" else None)
        try:
            if str(ccfg.mode or "hedge") == "select":
                got = CX._select_weights(frames, last, ccfg, vol, step=1,
                                         timing_extra=timing_extra)
            else:
                got = CX._hedge_weights(last, ccfg, vol, step=1)
        except Exception as exc:                      # noqa: BLE001
            return None, f"权重算不出来：{exc}"

        w = got.iloc[-1].astype(float)
        # 杠杆：和回测同一处、同一个函数。截面原来恒按 1 倍名义额算，而「资金
        # 分配 ÷ 每手保证金」默认就是 3 倍 —— 不乘这一下，实盘敞口只有回测的 1/3
        w = w * CX.leverage_vector(frames, self.bt).reindex(w.index).fillna(1.0)
        w = w.reindex(list(frames)).fillna(0.0)

        self.weights = w
        self.at = score.index[-1]
        n_leg = int((w.abs() > 0).sum())
        return w, (f"调仓：{n_leg} 条腿，Σ|w|={float(w.abs().sum()):.2f}"
                   f"（杠杆 {float(w.abs().sum()) / max(1e-9, 1.0):.2f}×），"
                   f"第 {L} 根 / 每 {step} 根一次" + self.ml_note)

    # ------------------------------------------------------------ 筛池

    def _pool_apply(self, score: pd.DataFrame, pool: pd.DataFrame,
                    ccfg) -> pd.DataFrame:
        """第一步筛池 —— **带状态**版的 `_apply_pool()`。

        回测那边是把整条时间轴一次算完、再 `where(tick).ffill()`；实盘每一根只看
        得到「当前这一行」，所以自己记住上一次采样出来的蒙版：当前这根是
        `_pool_tick_rows` 认定的重排时点就重新采，否则沿用上一次。

        逐行的数学**照抄** `_apply_pool`：升序 `rank(pct=True)`（分数越高名次越
        大，1.0 最强）、`keep = rank > 1 - top_pct%`。池因子整行没值（新品种上市
        前）时**不筛**，原样放行 —— 宁可在缺数据那几天少一层过滤，也不要凭空
        整组空仓。
        """
        p = pool.reindex(index=score.index, columns=score.columns).ffill()
        if p.empty or not p.notna().any().any():
            return score
        tick = CX._pool_tick_rows(p.index, int(ccfg.pool_reb or 1),
                                  str(ccfg.pool_reb_unit or "day"))
        picks = np.flatnonzero(tick.to_numpy())
        due = bool(len(picks) and int(picks[-1]) == len(p) - 1)
        if self.pool_keep is None or due:
            if not len(picks):
                # 回测里 `staged.fillna(1.0)`：第一个采样点之前**原样放行**
                self.pool_keep = {str(k): True for k in p.columns}
                self.pool_at = None
            else:
                # 🔴 首次进实盘时手上已经是**一整段历史**了。不能拿「今天」当
                # 采样点 —— 回测的蒙版来自上一个采样日（`pool_reb=200` 就是
                # 200 个交易日前）。取最后一个采样行，才能和回测逐位一致。
                row = p.iloc[int(picks[-1])]
                n = int(row.notna().sum())
                if n < CX.MIN_CROSS:
                    keep = {str(k): True for k in row.index}
                else:
                    r = row.rank(ascending=True, pct=True)
                    thr = 1.0 - float(ccfg.pool_top_pct or 0) / 100.0
                    keep = {str(k): bool(v) for k, v in
                            (r.gt(thr) & row.notna()).items()}
                self.pool_keep = keep
                self.pool_at = p.index[int(picks[-1])]
        # 🔴 不能用 `score.where(mask_series)`：pandas 会把那条 Series 按**行索引**
        # 对齐，而它的索引是品种（列），对不上 ⇒ cond 全 False ⇒ 整张分数表被
        # 抹成 NaN，截面直接空仓。显式广播成同形状 bool 矩阵才稳
        row = np.array([[bool(self.pool_keep.get(str(k), True))
                         for k in score.columns]], dtype=bool)
        return score.where(np.broadcast_to(row, score.shape))


# ------------------------------------------------------------------ 权重 → 手数


def targets_from_weights(weights: pd.Series, legs: list, principal: float,
                         price_of, mult_of) -> dict[str, int]:
    """把权重表折成「每个品种几手」。`legs` 是 `[(品种键, 上一步的目标)]`。

    价格和乘数由调用方给（实盘那边要读真实合约的最新价），这里只管算术 ——
    方便自检拿假数据把边界（0.5 手、反向、没价）全打一遍。

    符号必须跟着权重走：截面是多空两边同时持有，折成手数之后仍然是「+3 手多、
    −5 手空」，不能取绝对值再交给上层判方向。
    """
    out: dict[str, int] = {}
    for key, prev in legs:
        try:
            w = float(weights.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            out[key] = int(prev or 0)
            continue
        px = float(price_of(key) or 0.0)
        mult = float(mult_of(key) or 0.0)
        if not (px > 0) or not (mult > 0) or not math.isfinite(w):
            # 没价 / 没乘数：**保持原样**。拿 0 顶上就是把这一腿平掉
            out[key] = int(prev or 0)
            continue
        out[key] = round_lots(principal * w / (px * mult), int(prev or 0))
    return out
