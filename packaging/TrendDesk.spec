# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller 打包配置（Windows / macOS 通用）。

用法（在项目根目录执行）：

    python build.py                       # 推荐，自动装依赖 + 生成图标 + 打包 + 打包成 zip
    pyinstaller packaging/TrendDesk.spec  # 直接调用，跳过前置步骤

为什么用「目录模式」（onedir）而不是单文件（onefile）：

1. 单文件模式每次启动都要把 ~250MB 内容解包到临时目录，冷启动要 5~10 秒；
   目录模式直接运行，1~2 秒进界面。
2. 32MB 的 CSV 数据每次都被重复解包，纯属浪费。
3. macOS 的 `.app` 本身就是目录结构，单文件没有意义。

产物：
    dist/TrendDesk/            Windows：双击 TrendDesk.exe
    dist/TrendDesk.app         macOS：拖进 /Applications
"""

import os
import sys

from PyInstaller.utils.hooks import collect_data_files

# SPECPATH 是 PyInstaller 注入的、本 spec 文件所在目录
HERE = SPECPATH
ROOT = os.path.abspath(os.path.join(HERE, ".."))
PACK = HERE

APP_NAME = "TrendDesk"
BUNDLE_ID = "com.trenddesk.app"
# 版本号必须和要打的 tag 对齐：v1.0.1 的包里这里还写着 1.0.0，
# 装完在「关于」里看到的版本是错的。发版前记得同步改这里。
VERSION = "1.2.0"

ICON = os.path.join(PACK, "icon.ico" if sys.platform == "win32" else "icon.icns")
if not os.path.exists(ICON):
    ICON = os.path.join(PACK, "icon.png")

# 行情数据随包分发，开箱即用（约 32MB：34 只美股 + 94 个期货品种）
DATAS = []
for sub in ("data", "data_futures"):
    src = os.path.join(ROOT, sub)
    if os.path.isdir(src):
        DATAS.append((src, sub))

# xgboost 的原生库是 ctypes 在**运行时**按路径找的（xgboost/libpath.py 里
# 第一个候选就是 <包目录>/lib），PyInstaller 的静态分析看不到，也不会自动收集。
# 少了它打包后 `import xgboost` 直接抛 XGBoostLibraryNotFound，界面上模型下拉
# 会静默少一项 —— 自检里「机器学习模型依赖」那一项就是专门守这个的。
DATAS += collect_data_files("xgboost", includes=["lib/*", "VERSION"])

# 动态导入的模块，静态分析扫不到，必须显式声明
HIDDEN = [
    "matplotlib.backends.backend_qtagg",
    "matplotlib.backends.backend_agg",
    "ttrade.paths",
    "ttrade.selftest",
    # 因子策略的机器学习部分：模型按字符串动态 build()，PyInstaller 扫不到。
    # sklearn 的这几处是 GPR 的实际入口，缺了会在运行时报 ImportError。
    # 注意：别照抄网上抄来的 scipy.special._cdflib —— scipy 1.18 里已经没这个模块，
    # 写进 hiddenimports 只会得到一行 "Hidden import not found" 警告。
    # 真正的验收标准是打包后跑 --selftest，里面有一项专门 import 四个模型。
    "lightgbm",
    "xgboost",
    "sklearn.gaussian_process",
    "sklearn.gaussian_process.kernels",
    "sklearn.metrics.pairwise",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
]

# 用不到的重量级依赖，剔掉能省 100MB 以上
EXCLUDES = [
    # 其他 GUI 框架
    "tkinter", "_tkinter",
    "PyQt5", "PyQt6", "PySide2", "shiboken2",
    # 科学计算里本项目没用到的
    # 注意：scipy **不能**剔 —— sklearn（GPR）和 lightgbm/xgboost 都硬依赖它。
    # 早期版本这里写着 scipy，那时还没有因子功能，剔掉是省了 80MB；
    # 现在剔了会直接 ImportError，打包后自检会红。
    "sympy", "numba", "mplfinance",
    # 开发工具
    "IPython", "jupyter", "jupyter_core", "notebook", "nbconvert", "nbformat",
    "pytest", "sphinx", "docutils", "setuptools", "pip",
    # PySide6 里没碰到的子模块（Qt 绑定层，剔掉不影响 Qt6 底层 DLL）
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtWebChannel", "PySide6.QtWebSockets", "PySide6.QtWebView",
    "PySide6.QtQuick", "PySide6.QtQuick3D", "PySide6.QtQuickWidgets", "PySide6.QtQml",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DInput", "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation", "PySide6.Qt3DExtras",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "PySide6.QtSpatialAudio",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtPdf", "PySide6.QtPdfWidgets",
    "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtUiTools",
    "PySide6.QtSql", "PySide6.QtTest",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtPositioning", "PySide6.QtSensors",
    "PySide6.QtSerialPort", "PySide6.QtSerialBus", "PySide6.QtCanBus",
    "PySide6.QtNetworkAuth", "PySide6.QtRemoteObjects", "PySide6.QtScxml",
    "PySide6.QtStateMachine", "PySide6.QtTextToSpeech", "PySide6.QtHttpServer",
]

a = Analysis(
    [os.path.join(ROOT, "launcher.py")],
    pathex=[ROOT],
    binaries=[],
    datas=DATAS,
    hiddenimports=HIDDEN,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=EXCLUDES,
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe_kwargs = dict(
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,              # 桌面程序，不要黑框
    disable_windowed_traceback=False,
    icon=ICON,
)
if sys.platform == "darwin":
    exe_kwargs.update(
        argv_emulation=False,
        target_arch=None,        # None = 跟随构建机架构；要 universal2 需两套 Python
        codesign_identity=None,  # None = ad-hoc 签名，本机可运行
        entitlements_file=None,
    )

exe = EXE(pyz, a.scripts, [], exclude_binaries=True, **exe_kwargs)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_NAME}.app",
        icon=ICON,
        bundle_identifier=BUNDLE_ID,
        info_plist={
            "CFBundleDisplayName": "趋势跟踪研究台",
            "CFBundleName": APP_NAME,
            "CFBundleShortVersionString": VERSION,
            "CFBundleVersion": VERSION,
            "NSHighResolutionCapable": True,
            "NSRequiresAquaSystemAppearance": False,
            # PySide6 6.12 的 macOS 轮子是 macosx_14_0_universal2，
            # 写 11.0 会让系统误以为能在更老的机器上跑，实际会直接崩
            "LSMinimumSystemVersion": "14.0",
            "LSApplicationCategoryType": "public.app-category.finance",
            "LSBackgroundOnly": False,
        },
    )
