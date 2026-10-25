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
SPEC = os.path.join(PACK, "TrendDesk.spec")
DIST = os.path.join(ROOT, "dist")
BUILD = os.path.join(ROOT, "build")
APP_NAME = "TrendDesk"

IS_WIN = sys.platform == "win32"
IS_MAC = sys.platform == "darwin"


def log(msg: str) -> None:
    print(msg, flush=True)


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
    ap = argparse.ArgumentParser(description="打包趋势跟踪研究台")
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
