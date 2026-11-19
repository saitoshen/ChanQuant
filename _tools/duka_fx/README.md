# duka_fx —— Dukascopy 汇率/美元指数 → 5m 因子

补 `macro_series` 里 `dxy` / `usdcnh` 的短板。那两个序列来自 Tushare，是**日频**：

- 铺到 5m 只能按自然日 as-of（`factors_macro._asof(lag_days=1)`），周一的 bar 用的是
  **周日**那一行，周二的 bar 才用上周一；
- 周五那一行几乎永远用不到（被周日行顶掉）；
- 2026-10 起 Tushare 还整段缺周五行。

Dukascopy 给的是分钟级原始成交，聚合成 5m 后滞后只剩一分钟级别，不需要「日频压日频」。

## 用法

```bash
# 下载 + 聚合成 5m（断点续传，已缓存的日期不再请求）
python _tools/duka_fx/duka_fx.py fetch --start 2020-01-01 --end 2026-11-17

# 只用本地缓存重建 5m，一个请求都不发
python _tools/duka_fx/duka_fx.py rebuild

# 生成 5m 因子
python _tools/duka_fx/duka_fx.py factor

# 一条命令走完
python _tools/duka_fx/duka_fx.py all --start 2026-01-01
```

`--symbols usdcnh dxy` 指定品种；`--workers / --retries / --timeout` 调网络行为。

产物（`data_duka/`，已在 `.gitignore` 里）：

| 文件 | 内容 |
|---|---|
| `USDCNH_5m.parquet` | 5m OHLCV，index = 北京时间 |
| `DOLLARIDX_5m.parquet` | 同上 |
| `fx_5m_factors.parquet` | 16 列因子宽表 |
| `_cache/<code>/<Y>/<M>/<D>.json` | 原始响应，按天一份 |

## 时区：固定 +8，与伦敦夏令时无关

Dukascopy 的时间戳是 **UTC 毫秒**，不是伦敦本地时间：

```
北京时间 = UTC + 8      ← 冬夏令时都一样
```

夏令时影响的是「伦敦 09:00 对应北京几点」，不影响时间戳本身。中国不实行夏令时，
`Asia/Shanghai` 恒为 UTC+8。

只有当你拿到的是**用伦敦本地时间标注**的 CSV（别的渠道常见）时，才需要按 3 月最后
一个周日 / 10 月最后一个周日的规则在 +8 和 +7 之间切换 —— 这就是 `--tz london`，
由 `zoneinfo` 的 `Europe/London` 自动处理。

## 接口（2026-11 实测）

```
https://jetta.dukascopy.com/v1/candles/minute/{code}/BID/{Y}/{M}/{D}
```

`{code}` 取自 `https://jetta.dukascopy.com/v1/instruments` 的 `code` 字段（斜杠换成连字符）：

| 品种 | code | priceScale | 最早数据 |
|---|---|---|---|
| 美元兑离岸人民币 | `USD-CNH` | 5 | 2012-01-02 |
| 美元指数 | `DOLLAR.IDX-USD` | 3 | 2017-12-01 |

必须带 `Referer: https://widgets.dukascopy.com/en/historical-data-export`，否则未缓存的
响应会被挡（429）。

响应是**增量编码**，不是直给的 OHLC：

- `times[i]` 是「相对上一根跳过了几根 bar」，不是毫秒偏移；
- `opens[i]` 等是**价格增量**（units），真价 = `units × multiplier`；
- `shift` 是每根 bar 的毫秒数（1m = 60000）。

`decode_candles()` 严格照 `dukascopy-node` 的 `normaliseCandles` 实现，别「优化」。

## 因子（8 × 2 = 16 列）

| 列 | 含义 |
|---|---|
| `fx_<s>_chg_1 / 3 / 6 / 12 / 48` | 5 / 15 / 30 / 60 / 240 分钟变动 |
| `fx_<s>_z_48` | 相对过去 48 根（4 小时）的 z-score |
| `fx_<s>_rv_48` | 48 根已实现波动（年化） |
| `fx_<s>_pct` | 扩张分位（PIT，与 `factors_macro` 口径一致） |

**跨缺口一律置 NaN**：第 i 根与第 i−n 根的时间跨度超过 `1.6 × n × 5min` 时（周末、
长假、数据断档），`chg_n` 不算。否则周一第一根的 `chg_1` 是「周末 60 小时的累计
变动」，看着像 5 分钟信号。

## 自查结果（2026-11-18）

- **截断重算**：砍掉后 40% 重算，重叠部分 8 个因子逐格差异 **0.000e+00**（严格 PIT）；
- **跨缺口**：`usdcnh` 417 处、`dxy` 565 处缺口，`chg_1` 仍有值的 **0** 处；
- **量级**：`usdcnh` 的 `chg_1` 标准差 1.79e-4（≈1.8bp / 5min）、`chg_12` 5.92e-4，
  `rv_48` 年化约 4%，与离岸人民币的实际波动量级吻合；
- **与日频对账**：11-17 那天 Dukascopy 收盘 6.66704，Tushare 同一天 6.667100。

## 已知限制

1. **链路不稳**：本机到 `jetta.dukascopy.com` 时通时断（同一批请求里既有 200 也有
   `502 Bad Gateway`）。脚本靠「缓存 + 断点续传 + 重试」硬扛，失败的日子下次跑
   `fetch` 会补。补不齐时先跑 `rebuild`，缓存里有的都能落地。
2. **美元指数只有 2017-12 起**，2017 年以前的日线还得用 Tushare。
3. **`DOLLAR.IDX/USD` 是 CFD 报价**，不是 ICE 官方结算价：收盘时点、点差、隔夜调整
   都与 ICE DXY 不同。做分位/趋势/相对强弱没问题，复现官方收盘价不行。
4. **美元指数每天只交易 21 小时**（01:00–22:00 UTC ≈ 北京 09:00–06:00），所以它的
   5m 根数天然比 `usdcnh` 少，两条序列 concat 后会有大片 NaN —— 这是市场结构，不是 bug。
