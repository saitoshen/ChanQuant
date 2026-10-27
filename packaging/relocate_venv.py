"""把 .venv 从别的目录搬过来之后，修掉里面写死的绝对路径。

**什么时候要跑**：项目整体换了位置（C 盘搬到 D 盘、换机器、改目录名），
而 `.venv` 是直接复制过来、没有重建的时候。症状是：

- `pip.exe`、`pyinstaller.exe` 双击或命令行调用**静默失败**（退出码 1、无输出），
  因为 launcher 内嵌的 shebang 还指着旧路径的 python.exe
- `source .venv/Scripts/activate` 之后 `python` 仍然跑到旧环境里去
  （`VIRTUAL_ENV` 是写死的）
- `pip install` 把包装进了旧目录

**为什么 exe 只能重建、不能改**：pip 生成的 console script 是「launcher 可执行
文件 + 一行 shebang + zip 归档」的结构，zip 的中央目录记录着各条目的字节偏移。
shebang 长度一变，偏移全错，zip 直接失效。补 NUL 或补空格凑回原长度也不行 ——
distlib 的 launcher 会把整段读给 CreateProcess，两种填充实测都是退出码 1。
所以走 `distlib.ScriptMaker` 重新生成，结构完整、shebang 是新的。

用法：
    python packaging/relocate_venv.py            # 就地修（以本文件所在项目为根）
    python packaging/relocate_venv.py --check    # 只检查，不写

注意 `pyvenv.cfg` 里的 `home` 指向的是**创建 venv 时那个 Python 安装**，
不在项目里，所以不要动它 —— 动了才真的会坏。
"""

from __future__ import annotations

import argparse
import configparser
import glob
import os
import re
import sys

# activate / activate.bat / activate.fish / Activate.ps1 里的绝对路径。
# 匹配到 `.venv` 结尾即可，前面是什么盘符、什么目录都不重要。
_VENV_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\n\r'\"`]*?[\\/]\.venv")


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fix_text_scripts(scripts: str, new_venv: str, check: bool) -> list[str]:
    """activate 系列：把 VIRTUAL_ENV 换成新路径。"""
    fixed = []
    for name in ("activate", "activate.bat", "activate.fish", "Activate.ps1"):
        p = os.path.join(scripts, name)
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8", errors="surrogateescape") as f:
            s = f.read()
        # 只替换「不是新路径」的那些，避免把已经正确的再套一层
        s2 = _VENV_PATH_RE.sub(lambda m: new_venv if m.group(0) != new_venv else m.group(0), s)
        if s2 != s:
            fixed.append(name)
            if not check:
                with open(p, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
                    f.write(s2)
    return fixed


def rebuild_launchers(venv: str, check: bool) -> list[str]:
    """按各 dist-info 的 entry_points.txt 重新生成 Scripts/*.exe。"""
    scripts = os.path.join(venv, "Scripts")
    site = os.path.join(venv, "Lib", "site-packages")
    specs: list[str] = []
    seen: set[str] = set()
    for ep_file in glob.glob(os.path.join(site, "*.dist-info", "entry_points.txt")):
        cp = configparser.ConfigParser(delimiters=("=",))
        try:
            cp.read(ep_file, encoding="utf-8")
        except Exception:
            continue
        if not cp.has_section("console_scripts"):
            continue
        for name, target in cp.items("console_scripts"):
            name, target = name.strip(), target.strip()
            # `mod:func [extra]` 里的 extras 要剥掉，ScriptMaker 不认
            if "[" in target:
                target = target.split("[", 1)[0].strip()
            if name and target and name not in seen:
                seen.add(name)
                specs.append(f"{name} = {target}")
    if not specs:
        return []

    if check:
        return sorted(seen)

    from pip._vendor.distlib.scripts import ScriptMaker

    maker = ScriptMaker(None, scripts)
    maker.executable = os.path.join(scripts, "python.exe")
    maker.variants = {""}
    maker.clobber = True
    return [os.path.basename(p) for p in maker.make_multiple(specs)]


def fix_pyvenv_cfg(venv: str, new_venv: str, check: bool) -> bool:
    """pyvenv.cfg 只改 `command` 那一行。

    `home` 指向的是创建 venv 时用的 Python 安装，不在项目里 —— 那个绝对不能动，
    动了 venv 就找不到解释器了。`command` 只是「当初怎么建的」这条历史记录，
    不影响运行，但留着旧路径会让 `scan` 一直报残留。
    """
    p = os.path.join(venv, "pyvenv.cfg")
    if not os.path.exists(p):
        return False
    with open(p, "r", encoding="utf-8", errors="surrogateescape") as f:
        lines = f.readlines()
    changed = False
    out = []
    for line in lines:
        if line.lower().startswith("command"):
            new_line = _VENV_PATH_RE.sub(
                lambda m: new_venv if m.group(0) != new_venv else m.group(0), line
            )
            if new_line != line:
                changed = True
                line = new_line
        out.append(line)
    if changed and not check:
        with open(p, "w", encoding="utf-8", errors="surrogateescape", newline="") as f:
            f.writelines(out)
    return changed


def scan_stale(root: str, new_venv: str) -> list[str]:
    """列出仍然含「别的 .venv 绝对路径」的文件（排除字节码缓存）。"""
    venv = os.path.join(root, ".venv")
    stale = []
    for dirpath, dirnames, filenames in os.walk(venv):
        if "__pycache__" in dirnames:
            dirnames.remove("__pycache__")
        for fn in filenames:
            if not fn.endswith((".exe", ".cfg", "activate", ".bat", ".fish", ".ps1")):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "rb") as f:
                    data = f.read()
            except OSError:
                continue
            if b".venv" not in data:
                continue
            text = data.decode("latin-1")
            for m in _VENV_PATH_RE.finditer(text):
                if m.group(0) != new_venv:
                    stale.append(os.path.relpath(p, root))
                    break
    return stale


def main() -> int:
    ap = argparse.ArgumentParser(description="修 .venv 里指向旧目录的绝对路径")
    ap.add_argument("--check", action="store_true", help="只报告，不写文件")
    args = ap.parse_args()

    root = project_root()
    venv = os.path.join(root, ".venv")
    if not os.path.isdir(venv):
        print(f"找不到虚拟环境：{venv}")
        return 2

    print(f"项目根目录：{root}")
    print(f"虚拟环境：  {venv}\n")

    fixed = fix_text_scripts(os.path.join(venv, "Scripts"), venv, args.check)
    print(f"激活脚本{'待修' if args.check else '已修'} {len(fixed)} 个：{fixed or '无'}")

    cfg = fix_pyvenv_cfg(venv, venv, args.check)
    print(f"pyvenv.cfg 的 command 行{'待改' if args.check else '已改'}：{'是' if cfg else '否'}")

    made = rebuild_launchers(venv, args.check)
    # launcher 的内容没法便宜地判断「是否指向本目录」，所以一律重建；
    # check 模式只是把清单列出来，不代表它们一定是坏的。
    verb = "将重建" if args.check else "已重建"
    tail = "：" + "、".join(made[:6]) + " …" if len(made) > 6 else "：" + "、".join(made)
    print(f"launcher {verb} {len(made)} 个{tail if made else ''}")

    stale = scan_stale(root, venv)
    if stale:
        print(f"\n仍有旧路径的文件 {len(stale)} 个：")
        for s in stale[:10]:
            print("  ", s)
    else:
        print("\n没有残留旧路径。")

    # __pycache__ 里的 .pyc 会记录编译时的源路径，只影响 traceback 的显示，
    # 不影响运行，所以不计入失败。
    return 0 if not stale or args.check else 1


if __name__ == "__main__":
    sys.exit(main())
