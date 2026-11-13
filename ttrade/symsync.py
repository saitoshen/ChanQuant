"""从天勤取品种的**真实**合约参数（乘数 / 最小跳动 / 保证金率 / 手续费），
刷进「品种信息」表。

## 为什么要有这个模块

实盘下几手是 `分配资金 ÷ 每手保证金`，而每手保证金 = 价格 × 乘数 × 保证金率。
乘数和保证金率**只有交易所说得准**（保证金还随结算价、临近交割调整），所以
实盘一律用天勤合约上的 `quote.volume_multiple` / `quote.margin`，不查任何静态表。

但「数据管理 → 品种信息」那张表是**回测**的输入：研究页要比不同品种、不同年份，
参数必须**可复现** —— 用实时值的话，今天跑和明天跑结果不一样，横向比较就废了。
所以两张口径分开是对的：

| | 乘数 / 保证金率从哪来 | 为什么 |
|---|---|---|
| 实盘（下几手） | 天勤 `quote` 实时值 | 要跟交易所走，否则挂单被拒 / 实际杠杆不符 |
| 组合页「查看回测」/ 量占用 | 品种信息表 | 要和实盘资金分配对得上 |
| 策略研究页回测 | 品种信息表 | 要可复现 |

代价是表会过期 —— 过期之后组合页的手数和实盘就对不上了。这个模块就是那座桥：
一次性把表刷成「此刻的真实值」（乘数、最小跳动、保证金率、手续费）。

## 取哪个合约

取**主连的 underlying**（实盘实际下单的那个合约），不是主连本身 —— 主连是拼出来
的连续序列，没有保证金；只有具体合约有。代码拼装走 `tqsdk_symbols.main_symbol()`，
和实盘订阅用的是同一个函数：自己拼会在中金所翻车（CFFEX 是大写 `KQ.m@CFFEX.T`，
商品是小写 `KQ.m@SHFE.cu`）。
"""

from __future__ import annotations

import math

from . import symbols_info
from .tqsdk_symbols import main_symbol

# 等行情的墙钟上限（秒）。`quote` 是快照字段，订阅后第一帧就带；但主连要等
# 「换月映射」下发、冷门品种要等下一个 tick。等不到就如实报「没取到」，
# 绝不拿 0 或默认值冒充实时值。
WAIT_SECONDS = 25.0


def _px(q) -> float:
    for k in ("last_price", "pre_close", "pre_settlement"):
        try:
            v = float(getattr(q, k, 0) or 0)
        except (TypeError, ValueError):
            continue
        if v == v and v > 0:
            return v
    return 0.0


def _wait(api, ready, cancel=None) -> bool:
    """等到 `ready()` 为真或超时（`WAIT_SECONDS`）。

    🔴 必须有**墙钟**上限，不能按「轮数」等：`api.wait_update()` 在没有新 tick
    时会一直阻塞，循环数根本走不动。第一版按 60 轮写，遇到主连不下发字段就
    整个卡死在那里（要等 4 分钟才被外部杀掉）。
    """
    import time

    end = time.monotonic() + WAIT_SECONDS
    while True:
        if ready():
            return True
        if cancel is not None and cancel():
            return False
        if time.monotonic() >= end:
            return False
        try:
            api.wait_update(deadline=end)
        except TypeError:
            # 老版本 tqsdk 的 `wait_update` 不吃 deadline
            try:
                api.wait_update()
            except Exception:  # noqa: BLE001
                return False
        except Exception:  # noqa: BLE001
            return False


def fetch(pairs: list[tuple[str, str]], api=None, progress=None,
          cancel=None) -> tuple[list[dict], list[str]]:
    """取每个品种「主连的 underlying 合约」的乘数与保证金率。

    返回 `(rows, errors)`。`rows` 每项：
    `symbol / exchange / contract / mult / margin / margin_rate / price`。

    `margin_rate` 是**反推**出来的：`quote.margin ÷ (价格 × 乘数)`。交易所报的是
    「每手多少钱」，而回测表里存的是比例，所以必须除一次。这也是为什么这张表
    必须带价格刷新 —— 保证金率不是常数，价格变了比例就变（交易所按金额收）。
    """
    rows: list[dict] = []
    errors: list[str] = []
    if not pairs:
        return rows, errors

    own = api is None
    sess = None
    if own:
        from . import tqfetch as TQ

        sess = TQ.Session()
        api = sess.get()
    try:
        # ① 先订主连，只为拿 `underlying_symbol`。
        #    🔴 主连本身**没有** `margin`（它是拼接出来的连续序列），直接读它的
        #    quote.margin 永远是 0 —— 这也是第一版为什么卡在「等行情」上：
        #    等待条件永远不满足。必须再订一次真实合约。
        conts = []
        for sym, ex in pairs:
            tq = main_symbol(sym, ex)
            try:
                conts.append((sym, ex, tq, api.get_quote(tq)))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sym}（{tq} 订阅失败：{str(exc)[:40]}）")

        _wait(api, lambda: all(str(getattr(c[3], "underlying_symbol", "") or "")
                              for c in conts), cancel)

        # ② 再订真实合约，参数从这里读
        quotes = []
        for sym, ex, tq, cq in conts:
            und = str(getattr(cq, "underlying_symbol", "") or "")
            if not und:
                errors.append(f"{sym}（主连没下发 underlying）")
                continue
            try:
                quotes.append((sym, ex, und, api.get_quote(und)))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{sym}（{und} 订阅失败：{str(exc)[:40]}）")

        _wait(api, lambda: all(_px(q) > 0
                               and float(getattr(q, "margin", 0) or 0) > 0
                               and float(getattr(q, "volume_multiple", 0) or 0) > 0
                               for _s, _e, _u, q in quotes), cancel)

        for k, (sym, ex, tq, q) in enumerate(quotes, 1):
            if progress is not None:
                progress(k, len(quotes), f"品种参数 {k}/{len(quotes)}　{sym}")
            mult = float(getattr(q, "volume_multiple", 0) or 0)
            margin = float(getattr(q, "margin", 0) or 0)
            px = _px(q)
            und = str(getattr(q, "underlying_symbol", "") or "") or tq
            if mult <= 0 or margin <= 0 or px <= 0:
                errors.append(f"{sym}（没取到乘数/保证金/价格）")
                continue
            rate = margin / (px * mult)
            if not math.isfinite(rate) or rate <= 0 or rate > 1.0:
                # 保证金率超过 100% 说明字段口径不对（比如拿到的不是每手保证金），
                # 宁可报错也不能写进表里 —— 写进去之后回测手数会变成 0
                errors.append(f"{sym}（保证金率反推异常 {rate:.4f}）")
                continue
            rows.append({"symbol": sym, "exchange": ex, "contract": und,
                         "mult": mult, "margin": margin,
                         "margin_rate": rate, "price": px,
                         # 昨结算。🔴 天勤的 `commission` 是**按昨结算价**算的，
                         # 不是按最新价：AG 昨结算 12495 → 9.3712 = 12495×15×5‱
                         # 正好；用最新价 12099 反推会得到 5.2‱，凭空多 4%，
                         # 而且盘中一直漂，每刷一次都在改表。
                         "settle": float(getattr(q, "pre_settlement", 0) or 0),
                         # 最小跳动：交易所真值（界面原来那张 TICKS 表是手维护的，
                         # 新品种/改 tick 之后会悄悄过期）
                         "tick": float(getattr(q, "price_tick", 0) or 0),
                         # 手续费：单边每手（与回测 fee_per_lot 同口径）。
                         # 到底是「按手」还是「按比例折算」，见 `apply()` 的判据
                         "commission": float(getattr(q, "commission", 0) or 0)})
    finally:
        if own and sess is not None:
            sess.close()
    return rows, errors


def apply(rows: list[dict], only_diff: bool = True) -> dict:
    """把取到的参数写进品种信息表（覆盖）。返回统计。

    写三个字段，来源标成 `live`（不是 `user`）：界面上要能分清
    「这个数是我手填的」还是「从天勤刷下来的」。

    ## 四个字段的容差**不一样**，这是有原因的

    | 字段 | 容差 | 为什么 |
    | `mult` | **精确**（相对 1e-6）| 合约乘数是整数常量，不会漂 |
    | `tick` | **精确**（相对 1e-6）| 最小跳动也是交易所写死的 |
    | `margin` | 2%（`only_diff` 时）| 保证金率按结算价天天微调，每次刷新全表重写既慢又会把文件写花 |
    | `fee_value` | 0.1%（`only_diff` 时）| 比例型是反推的，comm 被四舍五入到 4 位，留一点余量免得天天写盘 |

    🔴 **乘数不能套 2% 容差**。`specs.multiplier()` 在没有表值时是**从成交额
    反推**的（成交额 ÷ 价 ÷ 量），反推出来经常差个零头：AU 真实 1000 克/手、
    SC 真实 1000 桶/手，反推成 **999**；T 真实 10000，反推成 **9999**。
    999 vs 1000 只差 0.1%，落在 2% 容差里 ⇒ 被当成「没变化」跳过 ⇒
    **刷完之后还是 999，看起来像没生效，其实是判等判宽了**。
    这类系统性偏差不会自己消失，必须精确比。

    ## 手续费：写，但要先判「按手」还是「按比例」

    天勤 `quote.commission` 是**单边每手**的手续费（与回测的 `fee_per_lot`
    同口径 —— tqsdk 内部 `_get_commission()` 就是「每手手续费」，模拟成交按
    `手数 × commission` 扣一次）。区别在于**交易所的收法有两种**：

    | 收法 | 天勤报什么 | 表里该怎么存 |
    | 按比例（cu 0.5‱、i 1‱、AG 0.5‱、IM 0.23‱）| 价 × 乘数 × 比例，是个带零头的小数 | `fee_mode="rate"` + 反推比例 |
    | 按手（au 20 元、SR 2 元、m 1.5 元）| 一个整齐的元/手数 | `fee_mode="lot"` + 元/手 |

    **存错口径不会当场出错，只会在价格漂移后悄悄失真**：比例型存成 lot，
    价格翻倍时成本还是那个死数（低估）；按手型存成 rate，价格翻倍时成本跟着
    翻倍（高估）。所以必须判断。

    🔴 反推比例的基准是**昨结算价**（`pre_settlement`），不是最新价 —— 天勤
    就是按昨结算算 `commission` 的。用最新价反推，AG 会得到 5.2‱（真实 5‱），
    而且盘中价格一动比例就变，每刷一次都在改表（实测 IF/IH 两次刷新差 0.1%，
    刚好卡在容差边上，白写一遍盘）。

    判据：**按手收的费率永远是 0.1 元的整数倍**（1 / 1.2 / 1.5 / 2 / 3 / 4.3 / 6 / 20），
    而「价 × 乘数 × 比例」几乎不可能正好落在 1 角档上。实测 10 个品种全部
    判对（cu 26.2850、i 6.7300、AG 9.3712、IM 29.9018、RB 0.3031 判成 rate；
    au 20、sc 20、SR 2、m 1.5、T 3 判成 lot）。

    ## 先验：判过一次就锁住，但允许自我纠错

    单靠数值判据会抖 —— 比例型品种某一天的价格恰好让乘积落在 1 角档，就会被
    判成 lot，第二天又变回 rate，表跟着来回改。所以：

    | 表里已有 `fee_mode`（刷新过/手填过） | 判定 |
    | 有 | 现有是 `lot` 且这次也落在 1 角档 → `lot`；其余 → `rate` |
    | 无（第一次刷） | 落在 1 角档 → `lot`，否则 `rate` |

    「有先验」时只保留 lot，不会从 rate 翻回 lot —— **偏保守那一侧是安全的**：
    rate 在价格漂移后偏高（成本高估，策略更难通过），lot 偏低（低估成本 =
    让策略虚假变好）。而 `lot → rate` 是允许的：那说明之前的 lot 判错了
    （真按比例的品种不可能永远落在 1 角档），自动纠正回比例。
    """
    tol = 0.02
    tol_fee = 1e-3
    changed, same, skipped, hints = 0, 0, [], []
    for r in rows:
        sym, ex = str(r.get("symbol") or ""), str(r.get("exchange") or "")
        if not sym:
            continue
        cur = symbols_info.defaults(sym, ex)
        ov = symbols_info.load_overrides().get(f"{sym}_{ex}" if ex else sym) or {}
        # 和「当前生效值」比，不是和默认值比：用户手填过的话，默认值早就不是
        # 生效值了，拿默认值比会得出「没变化」的假结论
        now = {k: float(ov.get(k) if ov.get(k) is not None else cur.get(k) or 0)
               for k in ("mult", "tick", "margin")}
        fields = {}
        for key, val, rel in (("mult", float(r.get("mult") or 0), 1e-6),
                              ("tick", float(r.get("tick") or 0), 1e-6),
                              ("margin", float(r.get("margin_rate") or 0),
                               tol if only_diff else 0.0)):
            if not math.isfinite(val) or val <= 0:
                continue
            old = float(now.get(key) or 0)
            if old > 0 and abs(val - old) <= rel * old:
                continue
            fields[key] = val

        # ---- 手续费：判口径 → 换算 → 与现值比 ----
        comm = float(r.get("commission") or 0)
        px = float(r.get("price") or 0)
        mult = float(fields.get("mult") or now.get("mult") or 0)
        old_mode = str(ov.get("fee_mode") or cur.get("fee_mode") or "rate")
        old_fee = float(ov.get("fee_value") if ov.get("fee_value") is not None
                        else cur.get("fee_value") or 0)
        new_mode = new_fee = None
        if comm > 0 and math.isfinite(comm) and px > 0 and mult > 0:
            # 反推比例的基准必须是**天勤算手续费时用的那个价**（昨结算），
            # 不是最新价 —— 见 `fetch()` 里的说明
            base = float(r.get("settle") or 0) or px
            is_round = abs(comm / 0.1 - round(comm / 0.1)) < 1e-6
            # 表里有没有「这个品种的手续费口径已经定过」的痕迹
            has_prior = "fee_mode" in ov
            if is_round and (old_mode == "lot" or not has_prior):
                new_mode, new_fee = "lot", comm
            else:
                new_mode, new_fee = "rate", comm / (base * mult)
        if new_mode and new_fee and new_fee > 0:
            mode_changed = new_mode != old_mode
            val_changed = (old_fee <= 0
                           or abs(new_fee - old_fee) > tol_fee * old_fee)
            if mode_changed or val_changed:
                fields["fee_mode"] = new_mode
                fields["fee_value"] = new_fee
                # 只有「之前定过口径、这次翻转了」才点名：第一次刷全表都在变，
                # 几十条列出来是噪音；翻转才是要人看一眼的事（成本随价格变化
                # 的规律变了，回测曲线形状都会不一样）
                if mode_changed and has_prior:
                    hints.append(
                        f"{sym} 口径 {old_mode}→{new_mode}"
                        f"（{comm:g} 元/手 @ 价 {px:g}）")

        if not fields:
            same += 1
            continue
        try:
            symbols_info.set_override(sym, ex, manual=False, **fields)
            changed += 1
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{sym}（{type(exc).__name__}: {exc}）")
    return {"rows": len(rows), "changed": changed, "same": same,
            "failed": skipped, "fee_hint": hints}


def refresh(pairs: list[tuple[str, str]] | None = None, api=None,
            progress=None, cancel=None) -> dict:
    """取 + 写一步到位。`pairs` 为空时用本地品种池。"""
    if not pairs:
        from . import tqstore

        pairs = list(tqstore.switch_symbols())
    rows, errors = fetch(pairs, api=api, progress=progress, cancel=cancel)
    st = apply(rows)
    st["errors"] = errors
    st["missing"] = max(0, len(pairs) - len(rows))
    return st


def preview(rows: list[dict]) -> list[dict]:
    """界面用：把取到的值和当前生效值并排摆出来，让人先看一眼差异。"""
    out = []
    for r in rows:
        sym, ex = str(r.get("symbol") or ""), str(r.get("exchange") or "")
        cur = symbols_info.info(sym, ex)
        out.append({
            **r,
            "old_mult": float(cur.get("mult") or 0),
            "old_tick": float(cur.get("tick") or 0),
            "old_margin": float(cur.get("margin") or 0),
            "old_fee_mode": str(cur.get("fee_mode") or "rate"),
            "old_fee": float(cur.get("fee_value") or 0),
            "mult_src": str(cur.get("mult_src") or ""),
            "margin_src": str(cur.get("margin_src") or ""),
        })
    return out
