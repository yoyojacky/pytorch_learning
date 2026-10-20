"""把 examples/ 下的案例脚本转换成 Jupyter 笔记本（notebooks/ 目录）。
转换规则：
- 文件开头的模块 docstring -> 开头的 Markdown 介绍单元格
- 每个 `# ===== N. 小节标题 =====` 标记 -> 一个 Markdown 标题单元格 + 一个代码单元格
- 代码行内的注释保留（即逐行讲解）
"""
import json  # 笔记本文件本质是 JSON
import re  # 用正则识别小节分隔标记
from pathlib import Path  # 处理文件路径

ROOT = Path(__file__).parent  # 项目根目录
EXAMPLES = ROOT / "examples"  # 案例脚本目录
NOTEBOOKS = ROOT / "notebooks"  # 笔记本输出目录
NOTEBOOKS.mkdir(exist_ok=True)  # 不存在则创建

SECTION_RE = re.compile(r"^# =+ (.+?) =+\s*$")  # 匹配 `# ===== 标题 =====` 形式的小节标记


def md_cell(text):
    """构造一个 Markdown 单元格"""
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code_cell(text):
    """构造一个代码单元格"""
    return {"cell_type": "code", "metadata": {}, "source": text,
            "execution_count": None, "outputs": []}


def convert(py_path):
    """把一个 .py 案例脚本转换成 .ipynb 笔记本"""
    lines = py_path.read_text(encoding="utf-8").splitlines(keepends=True)  # 按行读取，保留换行符

    # 1. 提取模块 docstring（文件开头的三引号字符串）作为介绍
    doc = re.match(r'^"""(.*?)"""\s*\n', "".join(lines), re.S).group(1)  # 取出 docstring 文本
    intro = [f"# {doc.strip().splitlines()[0]}\n", "\n"]  # docstring 首行作为一级标题
    intro += [line + "\n" for line in doc.strip().splitlines()[1:]]  # 其余行作为说明文字
    intro += ["\n", "> 逐行讲解见 `docs/` 目录中对应的章节手册。代码中的行内注释即讲解。\n"]

    # 2. 跳过 docstring 部分，从正文开始按小节切分
    end = next(i for i, l in enumerate(lines[1:], 1) if l.strip() == '"""')  # docstring 结束行号
    body = lines[end + 1:]  # 正文从 docstring 结束后开始

    cells = [md_cell(intro)]  # 第一个单元格是介绍
    pending_title, buf = None, []  # pending_title 暂存小节标题；buf 暂存当前小节的代码

    def flush():
        """把暂存的小节（标题 + 代码）写入 cells"""
        if pending_title is not None:  # 有小节标题就先加标题单元格
            cells.append(md_cell([f"## {pending_title}\n"]))
        code = "".join(buf).strip("\n")  # 去掉首尾空行
        if code.strip():  # 只保留非空代码
            cells.append(code_cell([l for l in code.splitlines(keepends=True)]))

    for line in body:
        m = SECTION_RE.match(line.rstrip("\n"))  # 当前行是否为小节标记
        if m:  # 遇到新小节
            flush()  # 先把上一小节写入
            pending_title, buf = m.group(1), []  # 开始新的小节
        else:
            buf.append(line)  # 普通代码行，加入当前小节
    flush()  # 写入最后一个小节

    # 3. 组装笔记本 JSON 结构
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3 (venv)", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,  # 笔记本格式主版本
        "nbformat_minor": 5,  # 次版本
    }
    out = NOTEBOOKS / (py_path.stem + ".ipynb")  # 输出文件名与脚本同名
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")  # 写盘
    print(f"{py_path.name} -> notebooks/{out.name}（{len(cells)} 个单元格）")


for py in sorted(EXAMPLES.glob("*.py")):  # 按文件名顺序转换全部案例
    convert(py)
