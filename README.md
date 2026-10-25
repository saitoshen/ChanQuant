# 趋势跟踪研究台（美股 + 国内期货）

从**东方财富**拉取美股主要标的近 20 年日线（后复权）、从**Tushare**拉取国内期货全品种主力连续合约日线
（按 `fut_daily_adj` 复权因子消除换月跳空），跑五套经典趋势跟踪策略，与「买入持有」做逐标的、分组、板块、
组合四个层面的对比，并提供一个可视化 GUI 查看 K 线与买卖信号。

**仓库**：<https://github.com/saitoshen/TrendDesk>
**下载成品**：<https://github.com/saitoshen/TrendDesk/releases> — Windows x64 / macOS arm64 / macOS x64

---

## 快速开始

```bash
# 1. 安装依赖（已内置虚拟环境则跳过）
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. 配置 Tushare token（只下美股可跳过）
cp token.example.txt token.txt      # 然后把 token 填进去；token.txt 不入库

# 3. 下载数据
.venv\Scripts\python.exe download.py            # 美股（东财，52 只候选）
.venv\Scripts\python.exe download_futures.py    # 国内期货（Tushare，全品种）

# 4. 批量回测，输出表格 + 图表 + 报告
.venv\Scripts\python.exe backtest_all.py                 # 美股  -> output/
.venv\Scripts\python.exe backtest_all.py --market cn     # 期货  -> output/futures/
.venv\Scripts\python.exe backtest_all.py --market cn --adjust none   # 期货不复权，看换月跳空的影响

# 5. 启动图形界面（左侧「品种类别」可切换美股 / 国内期货）
.venv\Scripts\python.exe gui\app.py
```

Windows 下可直接双击 **`启动GUI.bat`**。

补数据：`download.py --missing`、`download_futures.py --missing` 只补缺失或过期标的。

### Tushare token 配置

代码库里**不存 token**，`ttrade/futures.py::read_token()` 按以下优先级查找：

1. 环境变量 `TUSHARE_TOKEN`（CI 或临时会话）
2. 项目根目录 `token.txt`
3. 用户配置目录 `token.txt`（Windows `%LOCALAPPDATA%\TrendDesk`，macOS `~/Library/Application Support/TrendDesk`）

三处都没有时抛出的错误里会直接写出这三条路径。仓库里只提交 `token.example.txt`。


---

## 项目结构

```
Claw/
├── launcher.py            # 桌面程序入口（源码运行 + 打包入口，--selftest 自检）
├── build.py               # 一键打包（PyInstaller）
├── download.py            # 美股数据下载入口（东财）
├── download_futures.py    # 期货数据下载入口（Tushare）
├── backtest_all.py        # 批量回测 + 报告生成（--market us|cn）
├── gui/app.py             # PySide6 桌面程序（双市场）
├── token.example.txt      # token 模板（真正的 token.txt 不入库）
├── requirements.txt
├── packaging/
│   ├── TrendDesk.spec     # PyInstaller 配置（Windows / macOS 通用）
│   ├── make_icon.py       # 用 Pillow 生成 .ico / .icns / .png
│   └── icon.*             # 生成好的图标
├── ttrade/
│   ├── eastmoney.py       # 东财接口封装（secid 解析、K线抓取、本地读取）
│   ├── quality.py         # 美股行情质量修复（复权跳变 / 脏数据 / 休市占位行）
│   ├── futures.py         # Tushare 期货封装（合约清单、分页下载、复权因子、板块分类、流动性筛选）
│   ├── paths.py           # 路径解析（区分只读资源 / 可写目录，兼容打包）
│   ├── plotstyle.py       # matplotlib 跨平台中文字体
│   ├── selftest.py        # 打包后自检（8 项，含端到端建窗口）
│   ├── strategies.py      # 5 套趋势跟踪策略
│   ├── backtest.py        # 逐日回测引擎 + 绩效指标
│   └── universe.py        # 美股股票池（52 只，按行业分组）
├── .github/workflows/
│   └── build.yml          # 三平台矩阵构建 + 自检 + Release
├── data/                  # 美股 CSV
├── data_futures/          # 期货 CSV + _manifest.json
├── output/                # 回测结果：CSV / PNG / report.md
│   └── futures/           # 期货回测结果
└── dist/                  # 打包产物（build.py 生成，已 gitignore）
```

---

## 五套策略

`signal` 取值 **1=满仓多、0=空仓、-1=满仓空**。策略默认给出对称信号（多/空），
由回测引擎的 `long_only` 开关决定是否裁掉空头。

| 策略 | 做多 | 做空 | 特点 |
|---|---|---|---|
| 双均线交叉 | 快线上穿慢线 | 快线下穿慢线 | 对称反手，始终在市，交易最少 |
| 唐奇安通道突破 | 突破 N 日新高 | 跌破 N 日新低 | 海龟法则核心，反向 M 日通道离场 |
| ATR 通道突破 | 收盘 > 均线 + k×ATR | 收盘 < 均线 − k×ATR | 自适应波动率，回到均线离场 |
| MACD 趋势 | 零轴上金叉 | 零轴下死叉 | 双动量确认，交易最频繁 |
| 海龟（突破+ATR止损） | 唐奇安突破 | 唐奇安跌破 | 2.5×ATR 移动止损，单笔风险可控 |

不加杠杆：名义敞口 = 全部权益（多头买入全额、空头卖出等额名义），空仓时资金闲置。
空头权益归零时按规则强平（爆仓），后续不再交易。

---

## 回测规则

- **无未来函数**：第 T 日收盘生成信号，第 **T+1 日开盘**成交
- **成本**：美股单边手续费 5bp + 滑点 5bp；期货单边手续费 1bp + 滑点 2bp
- **复权**：美股后复权（fqt=2），含分红再投资；期货用 Tushare `fut_daily_adj` 反推的复权因子消除换月跳空
- **做空**：期货默认开启双向，美股默认只做多（未建模融券费与 uptick 规则）
- **绩效**：CAGR / 最大回撤 / 夏普 / 卡玛 / 胜率 / 盈亏比 / 多空仓位占比 / 交易次数

---

## 数据说明与已知问题

### 美股（东财后复权）

存在三类瑕疵，`ttrade/quality.py` 自动处理：

1. **复权因子整体跳变**（如 NKE 2007-03-19 价格凭空翻倍）→ 对后续价格等比修正
2. **单日脏数据**（如 NKE 2012-06-29、AXP 2008-01-02 单日腰斩次日复原）→ 剔除该行
3. **休市日占位行**（如 MSFT 2014-04-18 耶稣受难日，volume=0 且四价相同）→ 剔除

修复后，全池单日最大波动从 86% 降到 18% 以内。

### 国内期货（Tushare 主力连续 + 复权）

#### 换月复权

主力连续合约在移仓日是把新合约价格直接接在旧合约后面，不复权就会凭空多出一段跳空，
被回测当成真实盈亏。做法是：

1. 拉 `fut_daily_adj`（期货复权行情），取复权收盘价；
2. 反推逐日因子 `adj_factor = 复权收盘 / 原始收盘`。该因子在非换月日恒定、
   在换月日跳变，是标准的**后复权累计因子**，且与查询窗口无关（已实测跨窗口零偏差）；
3. 因子乘到 OHLC 上。成交量 / 成交额 / 持仓量不复权。

`load_local(adjust=...)` 支持三种口径，**三者收益率完全相同**，只差一个常数：

| 口径 | 含义 | 最新价 | 用途 |
|---|---|---|---|
| `qfq`（默认） | 前复权，除以末期因子 | 等于真实盘面价 | GUI 显示、回测 |
| `hfq` | 后复权，乘原始因子 | 被放大（铁矿约 8 倍） | 需要历史绝对不变的场景 |
| `none` | 不复权 | 真实价 | 对照实验 |

> 坑：广期所的 ts_code 后缀是 `.GFE` 而不是 `.GFEX`（`LC.GFE` / `SI.GFE` / `PS.GFE`），
> 写错后缀接口会**静默返回空表**，看起来像是"该交易所没有复权数据"。
> `attach_adj_factor` 在拿不到因子时退化为 1.0（等价不复权）并打日志，不会报错。

**复权带来的差异很大**：75 个品种「复权年化 − 不复权年化」平均 **+1.5 个百分点**，
63% 的品种为正，且与标的自身涨跌幅的相关系数达 **0.65**——涨得越多移仓收益越高。
这意味着不复权曲线不只是"脏"，还**系统性低估了趋势策略的收益**。

#### 三层清洗

`ttrade/futures.py` 做了三层清洗，**不做清洗会直接算错**：

1. **0 价占位行**：郑商所僵尸合约（粳稻、普麦、早籼稻、强麦、动力煤、胶合板）在无成交日返回
   `open=0, volume=0`，`close` 固定不变。不剔除会导致回测除零、净值变 `inf`。
   数据层与回测引擎双重过滤。
2. **已摘牌 / 换代码的遗留合约**：ME 甲醇、RO 菜油、WS 强麦、ER 早籼稻、TC 动力煤、WT 硬麦
   已被 MA/OI/WH/RI/ZC 取代，另有 LR 晚籼稻 2022 年停止交易。按「数据末尾距今 < 1 年」剔除。
3. **流动性不足**：按近一年日均成交额 < 5000 万元剔除（粳稻、普麦、早籼稻、强麦、胶合板、
   线材、菜籽、纤维板、动力煤等）。这些品种挂着牌但基本不成交，回测收益无法在实盘复现。

> 注意：流动性统计必须**包含 0 价行**。动力煤 ZC 自 2026-04 起被冻结（open/volume 全 0、
> close 固定 801.4），若先剔除 0 价行再取尾部 250 行，统计到的其实是冻结前一年的成交额，
> 会把僵尸品种误判为活跃品种。

最终标的池：**美股 34 只 / 国内期货 75 个品种**（原始 94 个：剔除 16 个僵尸/流动性不足品种，
再剔除 3 个日线不足 300 根的新上市品种——钯 PD、铂 PT、双胶纸 OP）。

---

## 核心结论

### 美股（2006–2026，34 只）

五套趋势策略**全部跑输买入持有**（年化 2.3%~10.9% vs 17.1%），但**最大回撤平均浅 14~22 个百分点**：

| 组合 | 2008 | 2009 | 2020 | 2022 |
|---|---|---|---|---|
| 买入持有 | -27.3% | +51.8% | +40.1% | -22.7% |
| 双均线交叉 | -15.4% | +28.3% | +21.7% | -11.0% |
| 海龟 | -5.6% | +6.3% | +12.6% | -5.1% |

**趋势跟踪买的是「熊市生存权」，代价是牛市弹性。** 股票池全为当前龙头股（幸存者偏差），
且这 20 年是美股最强长牛之一，任何需要离场的策略都必然付出机会成本。

### 国内期货（2006–2026，75 个品种，复权）

**只做多 vs 多空双向**（组合层面，等权合并）：

| 组合 | 只做多 年化/夏普/回撤 | 多空双向 年化/夏普/回撤 |
|---|---|---|
| 买入持有 | 2.7% / 0.28 / -48.2% | 同（不做空） |
| 双均线交叉 | 2.3% / 0.36 / -22.1% | 1.5% / 0.24 / -20.8% |
| 唐奇安通道突破 | 2.5% / 0.45 / -16.9% | **3.8% / 0.55 / -14.8%** |
| ATR 通道突破 | 3.1% / 0.60 / -11.5% | **5.0% / 0.75 / -11.9%** |
| MACD 趋势 | 3.8% / 0.69 / -12.4% | **4.8% / 0.71 / -12.0%** |
| 海龟（突破+ATR止损） | 1.8% / 0.51 / -9.1% | **3.1% / 0.63 / -10.1%** |

**换月复权前后对比**（单品种平均，多空双向模式）：

| 策略 | 不复权年化 | 复权年化 | 差额 | 不复权回撤 | 复权回撤 |
|---|---|---|---|---|---|
| 双均线交叉 | -3.4% | 0.7% | +4.1% | -57.2% | -51.8% |
| 唐奇安通道突破 | 0.3% | 3.0% | +2.7% | -46.8% | -41.9% |
| ATR 通道突破 | 2.4% | 4.2% | +1.8% | -43.3% | -40.3% |
| MACD 趋势 | 2.5% | 5.1% | +2.6% | -45.8% | -43.5% |
| 海龟（突破+ATR止损） | 0.0% | 2.1% | +2.1% | -39.8% | -35.5% |
| 买入持有 | 2.2% | 3.7% | +1.5% | -51.6% | -52.8% |

四条结论：

1. **期货的「买入持有」不是股票那种基准。** 75 个品种平均持有年化 3.7%、回撤 -52.8%，
   组合层面 2.7% / -48.2%，夏普只有 0.28——不到美股（1.01）的四分之一。
   股票持有赚的是盈利复利，商品持有赚的是移仓收益加均值回归的运气，一轮超级周期结束就还回去。
2. **复权后做空在单品种层面也站得住。** 3/5 套双向年化更高、3/5 套夏普上升，
   但 5/5 套最大回撤加深（空头亏损无上限）。**这与不复权数据的结论不同**——
   换月跳空抹平后，空头腿从「纯噪声」变成「有正期望」。
3. **组合层面优势更大。** 4/5 套组合夏普上升：ATR 通道 3.1%→5.0%（夏普 0.60→0.75）、
   唐奇安 2.5%→3.8%、海龟 1.8%→3.1%。几十个品种的空头腿叠加后进一步分散方向性风险。
   **「要不要做空」不只是策略问题，更是组合问题。**
4. **单品种夏普只有 0.2 左右，组合夏普 0.75。** 趋势跟踪靠多品种分散而非单品种择时。

真正的分野不是板块，而是**标的自身的长期方向**：C 组（持有 -8.5%）五套策略全部取得
+11.1%~13.3% 超额；A 组（持有 +17.7%）全部跑输 -18.6%~-7.1%。最纯粹的例证是新能源
（碳酸锂、多晶硅、工业硅，平均持有 -23.3%），五套策略年化 3.6%~20.1%、跑赢比例 100%。

> 不复权数据下曾出现 1 次空头爆仓（集运指数 EC），复权后该案例消失——
> 因为 EC 的复权因子是 2.57，真实价格路径与不复权曲线差别很大。
> 这提醒一件事：**做空的尾部风险高度依赖数据口径，不能只看回测有没有爆仓。**

---

## GUI 功能

- 左侧：**品种类别（美股 / 国内期货）**、标的、策略、参数（滑块实时调）、显示区间、
  K 线叠加开关、**允许做空开关**（期货默认开、美股默认关）
- **K 线与信号**页：K 线 + 策略指标线 + 持仓区间底色（多头浅蓝 / 空头浅红）+
  四类信号（▲绿开多、▼绿平多、▼红开空、▲红平空）+ 成交量
- **绩效对比**页：策略 vs 买入持有资金曲线（对数轴）+ 回撤曲线 + 17 项指标对照表
- 参数或方向开关改动自动重算（220ms 防抖），底部状态栏显示交易笔数、胜率、多空仓位占比

---

## 数据更新（GUI 内置）

左侧面板的「数据更新」卡片，不用再敲命令行：

| 控件 | 说明 |
|---|---|
| 区间 | 近 1 个月 / 近 3 个月 / 近半年 / 近 1 年 / 全部 20 年 |
| 数据源 | 自动（东财优先，失败回退 Tushare）/ 仅东财 / 仅 Tushare，**仅对美股生效** |
| 限速 | Tushare 美股接口的调用间隔，默认 60 秒/次；**东财与期货不受影响** |
| 更新 / 取消 | 同一个按钮，运行中变成「取消」 |

更新跑在独立 `QThread` 里，界面不会假死；进度条按标的推进，完成后自动重建
`DataStore` 并重绘当前图表，选中的标的保持不变。

命令行等价用法：

```bash
python -m ttrade.updater --market us --range 1m            # 美股，自动选源
python -m ttrade.updater --market us --range all --source eastmoney
python -m ttrade.updater --market cn --range 3m            # 期货，固定 Tushare
```

### 数据源与口径（重要）

| 市场 | 主源 | 备源 | 复权口径 |
|---|---|---|---|
| 美股 | 东方财富 `push2his` | Tushare `us_daily` | 东财=**后复权**；Tushare=**未复权** |
| 国内期货 | Tushare `fut_daily` + `fut_daily_adj` | — | 换月复权（因子由两者反推） |

两个美股源的口径差几十倍，**绝不能混进同一个 CSV**。因此清单里给每个标的记了
`source` 字段：一旦实际使用的源与文件原来源不同，会对该标的**整段重下 20 年**，
而不是把两种价格拼在一起。

Tushare 美股额度实测只有 **5 次/天**（`us_daily`），所以只作兜底；额度耗尽会自动熔断，
后续标的直接走东财，不会逐个空等。

### 增量更新的几个细节

- 起点取「区间起点」与「本地最后一天往前 10 天」的**较早者**：既补缺口，也刷新尾部
- 同一天的数据以新抓到的为准（重叠区被覆盖），日期始终去重且升序
- 写文件先写 `.tmp` 再 `os.replace`，中途崩溃不会留下半个 CSV
- 期货复权因子是绝对累计因子（锚定上市日、与查询窗口无关），增量取一小段也能直接拼接；
  若某品种复权数据为空（早期广期所），新行沿用最后一根已知因子，避免拼接处出现假跳空
- 连续 3 个标的失败且一个都没成功 → 判定网络/接口故障，立即中止而不是空跑几十次

---

## 打包成桌面应用

### 为什么是 PyInstaller 目录模式

**硬约束：PyInstaller 不能交叉编译。** Windows 上打不出 macOS 的 `.app`，反之亦然——
目标平台的 Python 解释器、Qt 库、系统框架必须一起塞进包里。所以「跨平台」只有两条路：
两台机器各打一次，或交给 CI 并行构建。

| 方案 | 产物 | 体积 | 启动 | 评价 |
|---|---|---|---|---|
| **PyInstaller 目录模式** | `.exe` + 依赖文件夹 / `.app` | ~250MB | 快 | **本项目采用** |
| PyInstaller 单文件 | 单个 `.exe` | ~250MB | 慢 | 每次启动解包 32MB 数据，不建议 |
| Nuitka | 原生二进制 | ~150MB | 最快 | 编译十几分钟，PySide6 偶有兼容坑 |
| Briefcase | `.msi` / `.dmg` | ~200MB | 中 | 能出安装包，生态小 |
| Web 化（NiceGUI/Streamlit） | 浏览器 | 小 | — | 形态变了，失去桌面感 |

选目录模式（onedir）的三个理由：启动 1~2 秒而不是 5~10 秒、32MB 数据不用每次解包、
macOS 的 `.app` 本身就是目录结构。

### 本地打包

```bash
.venv\Scripts\python.exe build.py          # Windows -> dist/TrendDesk/TrendDesk.exe
python build.py                            # macOS   -> dist/TrendDesk.app
```

脚本会自动：装 PyInstaller → 生成图标 → 打包 → 报体积 → 打成 zip。
其他参数：`--clean` 先清 `build/` 和 `dist/`，`--no-zip` 不打包，`--install-deps` 顺带装依赖。

### 打包后自检

打出来的包到底能不能用，不用点界面去猜：

```bash
dist\TrendDesk\TrendDesk.exe --selftest              # Windows
dist/TrendDesk.app/Contents/MacOS/TrendDesk --selftest   # macOS
```

自检不开窗口，逐项验证并写日志到 `<用户目录>/logs/selftest.log`，退出码 0 表示全过：

```
[ OK ] 美股数据目录  34 个 CSV
[ OK ] 期货数据目录  94 个 CSV
[ OK ] 读取美股行情  AAPL 5053 行 2006-09-25~2026-10-23
[ OK ] 读取期货行情（含复权）  RB_SHFE 4269 行，最新 3047.00，复权因子列=True
[ OK ] 策略回测  cagr=7.8% sharpe=0.44 trades=66
[ OK ] matplotlib 绘图（含中文字体）  字体 Microsoft YaHei，渲染 11212 B
[ OK ] Qt 平台插件  Qt 6.12.0 离屏渲染正常
[ OK ] 构建主窗口（端到端）  34 个标的，K 线与绩效页已渲染
```

最后一项是真正的端到端验证：把 `MainWindow` 完整搭起来、跑完首次回测，
所以「自检通过」等价于「界面能开、数据能读、图能画」，而不只是「PyInstaller 没报错」。
CI 里也跑这一步。

### 三平台 CI

`.github/workflows/build.yml` 配了三平台矩阵：

| runner | 产物 |
|---|---|
| `windows-latest` | `TrendDesk-win64.zip` |
| `macos-latest`（Apple Silicon） | `TrendDesk-macos-arm64.zip` |
| `macos-15-intel`（Intel） | `TrendDesk-macos-x64.zip` |

触发方式两种：**打 `v*` 标签**（构建 + 发 Release），或到 Actions 页面手动 Run workflow。
推 `main` 不会触发——数据文件大，不必每次提交都重建。

```bash
git tag -a v1.0.1 -m "说明"
git push origin v1.0.1
```

每个平台打完包都会跑一次 `--selftest`，8 项全过才算成功，自检日志直接打到 CI 输出。

> 注意：`data/` 与 `data_futures/`（约 32MB）需要一并提交，否则打出的包没有行情数据。
> `.gitignore` 已排除 `.venv/`、`build/`、`dist/`、`output/`、`token.txt`。

#### CI 上踩过的三个坑（都已修）

| 现象 | 原因 | 修法 |
|---|---|---|
| Windows 卡在「生成图标」 | 英文版 runner 控制台是 cp1252，脚本 `print` 中文抛 `UnicodeEncodeError` | job 级 `PYTHONUTF8=1`，脚本内也 `reconfigure(encoding="utf-8")` |
| macOS 打包成功但自检报 `No module named 'PySide6'` | `requirements.txt` 只写了 `shiboken6`，漏了 `PySide6`。**PyInstaller 不会因此报错**，会安静地打出没有 Qt 的空壳 app | 补全依赖；`build.py` 增加 `check_runtime_deps()` 打包前拦截 |
| Intel Mac 的 job 起不来 | `macos-13` runner 已于 2025-12-04 退役 | 改用 `macos-15-intel` |

### 产物与首次运行

| 平台 | 产物 | 使用方式 |
|---|---|---|
| Windows | `dist/TrendDesk/`（内含 `TrendDesk.exe`） | 整个文件夹拷走，双击 exe |
| macOS | `dist/TrendDesk.app` | 拖进 `/Applications` |

macOS 首次打开若提示「已损坏」或「无法验证开发者」，是因为没有开发者签名（ad-hoc 签名不够）：

```bash
xattr -dr com.apple.quarantine /Applications/TrendDesk.app
```

或者右键点应用 → 打开 → 再点「打开」。要彻底消掉这个提示需要 Apple Developer 账号（$99/年）
做 notarization。

### 路径处理（打包后最容易踩的坑）

`ttrade/paths.py` 把路径分成两类，**不要混用**：

| 类型 | 函数 | 打包后位置 | 可写 |
|---|---|---|---|
| 只读资源 | `bundle_root()` / `data_dir()` | `sys._MEIPASS` | ❌ |
| 可写目录 | `user_dir()` / `cache_dir()` / `log_dir()` | 用户家目录 | ✅ |

macOS 的 `.app` 签名后所在卷是**只读**的。之前 `MPLCONFIGDIR` 和 `error.log` 都指向项目目录，
源码运行没问题，打包后会静默失败。现在分别走 `cache_dir("mpl")` 和 `log_dir()`：

- Windows：`%LOCALAPPDATA%\TrendDesk\{cache,logs}`
- macOS：`~/Library/Caches/TrendDesk` 与 `~/Library/Application Support/TrendDesk/logs`

### 跨平台中文字体

这是打包后第二个容易翻车的地方。原来 `font.sans-serif` 只写了 `Microsoft YaHei` / `SimHei`，
**在 macOS 上这两个字体都不存在**，matplotlib 会静默回退到 DejaVu Sans——中文全变方框，
而且不报错，只有一条 `UserWarning: Glyph xxxxx missing from font(s)` 容易被忽略。

`ttrade/plotstyle.py` 现在按平台列候选字体，只把**实际装了的**写进 rcParams：

| 平台 | 优先级 |
|---|---|
| Windows | Microsoft YaHei → SimHei → Microsoft JhengHei |
| macOS | PingFang SC → Hiragino Sans GB → Heiti SC → STHeiti |
| Linux | Noto Sans CJK SC → Source Han Sans SC → WenQuanYi Micro Hei |

找不到任何中文字体时抛 `RuntimeWarning` 而不是静默出方框；自检里用
`check_cjk_render()` 捕获缺字警告，缺字直接判失败。

### 体积

CI 实测压缩包大小：Windows 104 MB、macOS arm64 82 MB、macOS x64 88 MB；
解压后约 250 MB，主要是 PySide6（~150MB）、pandas、numpy、matplotlib。
spec 里已剔除 `tkinter`、`PyQt5/6`、`PySide2`、`scipy`、`IPython`、`pytest`
以及 QtWebEngine / QtQuick / Qt3D / QtMultimedia 等未使用的 Qt 子模块。
再想瘦身只能换 Nuitka 或砍掉 matplotlib 自绘 K 线。

### 已知限制

- 数据随包分发，**更新行情需要重新打包**（GUI 内没有下载入口，下载走命令行脚本）
- 不含自动更新机制
- macOS 未做 notarization，需要用户手动放行
- **macOS 14+**：PySide6 6.12 的轮子是 `macosx_14_0_universal2`，更老的系统装不上

---

## 免责声明

本项目仅用于策略研究与教学演示，不构成任何投资建议。历史回测结果不代表未来表现。
