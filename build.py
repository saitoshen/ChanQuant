"""一键打包脚本（Windows / macOS 通用）。

    python build.py                # 打包当前平台
    python build.py --no-zip       # 不生成压缩包
    python build.py --clean        # 先清空 build/ 与 dist/
    python build.py --install-deps # 顺便重装 requirements.txt

做的事：检查依赖 → 生成图标 → 调用 PyInstaller → 报告体积 → 打成 zip/dmg 便于分发。

注意：**PyInstaller 不能交叉编译**。在 Windows 上只能出 .exe，在 macOS 上只能出 .app。
要两个平台的包，要么各跑一次，要么用 .github/workflows/build.yml 让 CI 矩阵并行构建。
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.join(ROOT, "packaging")
SPEC = os.path.join(PACK, "ChanQuant.spec")
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, "build")
APP_NAME = "ChanQuant"

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"

# 英文版 Windows 控制台默认 cp1252，print 中文会抛 UnicodeEncodeError
# （GitHub Actions 的 windows-latest 就是这样，日志里表现为打包脚本莫名失败）。
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def log(msg: str) -> None:
    print(msg, flush=True)


REQUIRED_MODULES = [
    ("PySide6.QtWidgets", "PySide6"),
    ("matplotlib", "matplotlib"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
]


def check_runtime_deps() -> bool:
    """打包前确认运行期依赖都在。

    这一步不能省：PyInstaller 缺了 PySide6 并不会报错，它会安静地打出一个
    没有 Qt 的空壳程序，装到用户机器上双击才崩。宁可在打包阶段就失败。
    """
    missing: list[str] = []
    for mod, pkg in REQUIRED_MODULES:
        try:
            __import__(mod)
        except Exception:
            missing.append(pkg)
    if missing:
        log("缺少运行期依赖：" + "、".join(sorted(set(missing))))
        log("请先执行：pip install -r requirements.txt")
        return False
    return True


def ensure_pyinstaller(auto_install: bool = True) -> bool:
    try:
        import PyInstaller  # noqa: F401

        return True
    except ImportError:
        pass
    if not auto_install:
        log("缺少 PyInstaller，请先执行：pip install pyinstaller")
        return False
    log("未检测到 PyInstaller，正在安装…")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--upgrade", "pyinstaller"],
        cwd=ROOT,
    )
    return r.returncode == 0


def ensure_icon() -> None:
    png = os.path.join(PACK, "icon.png")
    need = not os.path.exists(png) or not os.path.exists(
        os.path.join(PACK, "icon.ico" if IS_WIN else "icon.icns")
    )
    if not need:
        return
    log("生成应用图标…")
    subprocess.run([sys.executable, os.path.join(PACK, "make_icon.py")], cwd=ROOT, check=True)


def clean() -> None:
    for d in (BUILD, DIST):
        if os.path.isdir(d):
            log(f"清理 {d}")
            shutil.rmtree(d, ignore_errors=True)


def run_pyinstaller() -> int:
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", SPEC]
    log("执行：" + " ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


def dir_size(path: str) -> int:
    total = 0
    for base, _dirs, files in os.walk(path):
        for fn in files:
            fp = os.path.join(base, fn)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"


# --------------------------------------------------------------- macOS OpenMP
# PyPI 上 xgboost 的两个 macOS wheel（x86_64 / arm64）都**不带** OpenMP 运行时，
# libxgboost.dylib 里写死的是：
#     LC_LOAD_DYLIB  @rpath/libomp.dylib
#     LC_RPATH       /usr/local/opt/libomp/lib      （x86_64）
#                    /opt/homebrew/opt/libomp/lib   （arm64）
# 也就是说它指望构建机上有 brew 装的 libomp。CI 的 macos-latest 镜像恰好预装了
# /opt/homebrew/opt/libomp，所以 arm64 能过；macos-15-intel 镜像的 /usr/local/opt
# 下没有，x64 就 import 失败。
#
# 更麻烦的是：即便 CI 装了 libomp，打出来的包在**用户机器**上照样跑不起来 ——
# 普通用户不会去 brew install libomp，@rpath 解析不到，XGBoost 直接消失。
# 所以必须把 libomp.dylib 塞进包内，并把 install name 改成 @loader_path 相对引用。


def _find_all(root: str, filename: str) -> list[str]:
    out = []
    for base, _dirs, files in os.walk(root):
        if filename in files:
            out.append(os.path.join(base, filename))
    return out


def _locate_libomp() -> str | None:
    """找构建机上的 libomp.dylib。优先问 brew，再退回两个已知前缀。"""
    cands: list[str] = []
    try:
        r = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            cands.append(os.path.join(r.stdout.strip(), "lib", "libomp.dylib"))
    except OSError:
        pass
    for arch_prefix in ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp"):
        cands.append(os.path.join(arch_prefix, "lib", "libomp.dylib"))
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _xgb_depends_on_rpath_libomp(xgb: str) -> bool:
    r = subprocess.run(["otool", "-L", xgb], capture_output=True, text=True)
    return "@rpath/libomp.dylib" in r.stdout


def fix_macos_openmp(target: str) -> bool:
    """把 libomp.dylib 打进 .app，并改写 libxgboost.dylib 的依赖引用。

    包里出现几份 libxgboost.dylib 就处理几份 —— PyInstaller 在个别情况下会把
    同一个二进制收集两次（datas 一份、binaries 一份），只改第一份会留隐患。
    """
    dylibs = _find_all(target, "libxgboost.dylib")
    if not dylibs:
        log("  ! 包内没找到 libxgboost.dylib，跳过 OpenMP 处理")
        return False

    src = _locate_libomp()
    touched: list[str] = []
    for xgb in dylibs:
        lib_dir = os.path.dirname(xgb)
        dst = os.path.join(lib_dir, "libomp.dylib")
        if not os.path.exists(dst):
            if src is None:
                log("  ! 构建机上没有 libomp.dylib（brew install libomp），")
                log("    打出来的包在用户机器上 XGBoost 会不可用")
                return False
            shutil.copy2(src, dst)
            log(f"  已内置 OpenMP 运行时 → {os.path.relpath(dst, target)}")

        # @loader_path = libxgboost.dylib 自己所在的目录；两个文件同目录，引用天然成立
        r = subprocess.run(
            ["install_name_tool", "-change", "@rpath/libomp.dylib",
             "@loader_path/libomp.dylib", xgb],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            log(f"  ! install_name_tool 失败：{r.stderr.strip()}")
            return False

        # 改过 Mach-O 之后原签名失效；arm64 上未签名/坏签名的 dylib 会被内核直接拒绝加载
        for f in (dst, xgb):
            s = subprocess.run(["codesign", "--force", "--sign", "-", f],
                               capture_output=True, text=True)
            if s.returncode != 0:
                log(f"  ! 重签失败 {os.path.basename(f)}：{s.stderr.strip()}")
                return False
        touched.append(xgb)

    # 外层 .app 的签名封装了 Frameworks 里每个文件，改了内容必须整包重签
    s = subprocess.run(["codesign", "--force", "--deep", "--sign", "-", target],
                       capture_output=True, text=True)
    if s.returncode != 0:
        log(f"  ! 整包重签失败：{s.stderr.strip()}")
        return False

    left = [p for p in touched if _xgb_depends_on_rpath_libomp(p)]
    if left:
        log(f"  ! 还有 {len(left)} 份 libxgboost.dylib 依赖 @rpath/libomp.dylib，改写没生效")
        return False
    log(f"  {len(touched)} 份 libxgboost.dylib 已改为 @loader_path/libomp.dylib")
    return True


def package(target: str, base_name: str) -> str | None:
    """把产物打成压缩包。macOS 用 ditto 以保留 .app 内的符号链接与资源分叉。"""
    out_dir = os.path.join(ROOT, "dist")
    if IS_MAC:
        out = os.path.join(out_dir, f"{base_name}.zip")
        if os.path.exists(out):
            os.remove(out)
        r = subprocess.run(
            ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", target, out]
        )
        return out if r.returncode == 0 else None

    out_base = os.path.join(out_dir, base_name)
    archive = shutil.make_archive(out_base, "zip", root_dir=os.path.dirname(target),
                                  base_dir=os.path.basename(target))
    return archive


def main() -> int:
    ap = argparse.ArgumentParser(description="打包 ChanQuant 禅量")
    ap.add_argument("--no-zip", action="store_true", help="不生成压缩包")
    ap.add_argument("--clean", action="store_true", help="打包前清空 build/ 和 dist/")
    ap.add_argument("--install-deps", action="store_true", help="先安装 requirements.txt")
    args = ap.parse_args()

    log("=" * 62)
    log(f"打包 {APP_NAME}　平台：{platform.system()} {platform.machine()}　Python {platform.python_version()}")
    log("=" * 62)

    if args.install_deps:
        req = os.path.join(ROOT, "requirements.txt")
        if os.path.exists(req):
            log("安装依赖…")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req], cwd=ROOT, check=True)

    if not check_runtime_deps():
        return 1

    if not ensure_pyinstaller():
        return 1

    ensure_icon()

    if args.clean:
        clean()

    rc = run_pyinstaller()
    if rc != 0:
        log(f"打包失败，PyInstaller 退出码 {rc}")
        return rc

    # 定位产物
    if IS_MAC:
        target = os.path.join(DIST, f"{APP_NAME}.app")
        run_hint = f"open {target}"
    else:
        target = os.path.join(DIST, APP_NAME)
        run_hint = os.path.join(target, f"{APP_NAME}.exe")

    if not os.path.exists(target):
        log(f"未找到产物：{target}")
        return 1

    if IS_MAC:
        log("处理 macOS OpenMP 依赖…")
        if not fix_macos_openmp(target):
            # 不算致命错误：包里其它功能都正常，只是机器学习少了 XGBoost。
            # 但要吼一声，否则又会变成「CI 绿了、用户那边模型少一个」。
            log("  ! XGBoost 在 macOS 上可能不可用，详见上面的提示")

    size = dir_size(target)
    log("")
    log("-" * 62)
    log(f"产物目录：{target}")
    log(f"体积：{human(size)}")
    log(f"启动方式：{run_hint}")
    if IS_MAC:
        log(f"自检：    {os.path.join(target, 'Contents', 'MacOS', APP_NAME)} --selftest")
    else:
        log(f"自检：    {run_hint} --selftest")
    log("-" * 62)

    if not args.no_zip:
        plat = "macos-arm64" if IS_MAC and platform.machine() == "arm64" else (
            "macos-x64" if IS_MAC else "win64"
        )
        archive = package(target, f"{APP_NAME}-{plat}")
        if archive:
            log(f"已打包：{archive}（{human(os.path.getsize(archive))}）")
            if IS_MAC:
                log("")
                log("首次打开若提示「已损坏」或「无法验证开发者」，执行：")
                log(f"    xattr -dr com.apple.quarantine {archive}")
                log("或右键点应用 → 打开 → 再点「打开」。")
        else:
            log("压缩失败（不影响 dist/ 里的产物）")

    log("")
    log("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
