# 融航 CTP 兼容运行时（vendored）

这里放的是**融航资管中台**用的 CTP 运行时。不用 `vnpy_rohon` 那个包，
因为：

- `vnpy_rohon` 只出 `cp310` / `cp313` 的 Windows 轮子，本机 Python 是 3.14，
  根本装不上；
- 它的依赖表里写着 `vnpy>=4.0.0`，而 vnpy 4.5 把 `pyside6` 钉在 6.8.2.1，
  一装就把本程序的界面库降级。

融航官方给了一份「替换 dll」：把他们自己的 `thosttraderapi_se.dll` 换到 CTP
客户端里，就能用标准 CTP 接口连融航中台。我们把这套东西隔离在
`vendor/rohon_ctp/`，与系统装的 `openctp-ctp`（6.7.11，给普通 CTP 账户用）
**互不覆盖**，由 `ttrade/ctp_channel.py` 按账户类型选一条。

## 目录里是什么

| 文件 | 来源 |
|---|---|
| `openctp_ctp/` | PyPI `openctp-ctp==6.7.2.8` 的 `cp314-cp314-win_amd64` 轮子 |
| `openctp_ctp.libs/thosttraderapi_se-5906db0c….dll` | **融航的** `thosttraderapi_se.dll`，改名成 pyd 导入表要求的哈希名 |
| `openctp_ctp.libs/thostmduserapi_se-baa372…dll` | openctp-ctp 原装 6.7.2，**融航行情就靠它**（融航没给 md 替换文件） |
| `openctp_ctp.libs/RohonBaseV64.dll`、`WinDataCollect.dll` | 融航替换包自带 |
| `openctp_ctp.libs/msvc{r,p}100.dll`、`mfc100.dll` | 融航 dll 用 VC2010 编译，自带运行时 |
| `_abi.txt` | 这份 pyd 是按哪个 Python ABI 编的（打包时校验用） |

融航替换包的原始位置：
`D:\融航仿真接口替换文件64位（对应ctp-6.7.2版本）更新时间20251217\64`

## 为什么必须锁 6.7.2.8

融航的 dll 是 **CTP 6.7.2** 的接口。用 `openctp-ctp 6.7.11` 的 pyd 去配它会
`DLL load failed — 找不到指定的程序`（ERROR_PROC_NOT_FOUND）：6.7.11 的 pyd
要的导出符号，融航那份 dll 里没有。反过来，6.7.2.8 的 pyd 配 6.7.11 的 dll
同样不行。**版本必须成对**。

## 三条实测结论（2026-11-19）

1. 6.7.2.8 pyd + 融航 dll → `GetApiVersion()` 返回 `6.7.2.3`，登录融航仿真
   账户成功（交易日 20261119、结算确认 ErrorID=0）。
2. 融航 dll 连**标准 CTP 前置**（SimNow `182.254.243.31:30001`）→
   `0xC0000005` 访问违例崩溃。**所以绝不能全局替换** site-packages 里的 dll。
3. 同一进程里同时加载标准 CTP 与融航两套库 → 无论并发还是顺序都 exit 139。

## 另外：融航其实有行情

替换包里只给了 trader 的 dll，但融航的**行情前置就是标准 CTP md 服务**：
库里那份 6.7.2 的 `thostmduserapi_se` 连 `124.160.66.120:41253` 能连上、能登录
（交易日 20261119）、能订阅（2026-11-19 实测）。所以不用向融航另要 md dll，
也不用天勤兜行情 —— 账户里填了 `md_front` 就走融航行情。

之前记的「0xC0000005 崩溃」是探测脚本自己的错：`api.RegisterSpi(Spi(api))`
传的是临时对象，CPython 当场回收，CTP 侧拿着悬空指针就崩。SPI 必须被变量持有
（正式代码里是 `MdClient._spi`）。

## 更新方法

换融航的新版替换文件时：

1. 把新的 `thosttraderapi_se.dll` 复制成 `openctp_ctp.libs/` 里**现有的那个
   哈希名**（pyd 的导入表写死了它，改名字就加载不到）；
2. 同步更新 `RohonBaseV64.dll` 等私有依赖；
3. 如果融航换了 CTP 接口版本（比如升到 6.7.11），要连 `openctp-ctp` 的 pyd
   一起换，并且把 `vendor/rohon_ctp/_abi.txt` 改成本机 Python 的 ABI；
4. 跑一遍 `python output/t_rohon_session.py`（用真实账户登录一次）确认能通。

打包时 `packaging/ChanQuant.spec` 会比对 `_abi.txt` 和构建用的 Python 版本，
对不上就不带这份 vendor（发布包里融航会显示「通道不可用」），以免打出一个
「一点连接就闪退」的包。
