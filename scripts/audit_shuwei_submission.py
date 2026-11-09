#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""audit_shuwei_submission.py — 数维杯论文 PDF / 承诺书 / 支撑包检查。

用法:
    python scripts/audit_shuwei_submission.py --paper ASWBMCM2600001.pdf
    python scripts/audit_shuwei_submission.py --paper ASWBMCM2600001.pdf \
        --control ASWBMCM2600001cns.pdf --zip ASWBMCM2600001fj.zip

重点:
  - 文件名 <题号>SWBMCM<队号>.pdf
  - p1 Summary、p2 Contents
  - 总页数 ≤25（含 Summary 和 Contents；附录另计）
  - 页眉 Team ASWBMCM... Page x of y
  - 正文无中文/身份信息
  - 12pt Times New Roman（字体大小粗检）
  - Summary 300–450 词、8–20 数字
  - PDF 元数据、承诺书、支撑 ZIP
退出码: 0 通过（可 WARN），1 有 FAIL。
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
import zipfile
from pathlib import Path

RESULT = []
CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")
ID = re.compile(r"@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|\b(University|College|School|Advisor|Professor)\b", re.I)
PAPER_RE = re.compile(r"^[ABCD]SWBMCM\d{4,}\.pdf$", re.I)
CTRL_RE = re.compile(r"^[ABCD]SWBMCM\d+cns\.(pdf|jpe?g|png)$", re.I)
ZIP_RE = re.compile(r"^[ABCD]SWBMCM\d+fj\.(zip|rar)$", re.I)


def add(level, title, detail=""):
    RESULT.append((level, title, detail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", type=Path, required=True)
    ap.add_argument("--control", type=Path)
    ap.add_argument("--zip", type=Path)
    args = ap.parse_args()

    audit_pdf(args.paper)
    audit_control(args.control)
    audit_zip(args.zip)

    fails = 0
    for level, title, detail in RESULT:
        print(f"[{level}] {title}: {detail}")
        if level == "FAIL":
            fails += 1
    print(f"\nSUMMARY: {len(RESULT)} checks, {fails} FAIL")
    return 1 if fails else 0


def audit_pdf(path: Path):
    if not path.is_file():
        add("FAIL", "论文 PDF 不存在", str(path))
        return
    add("PASS" if PAPER_RE.match(path.name) else "FAIL", "论文文件名",
        path.name + (" 符合 <题号>SWBMCM<队号>.pdf" if PAPER_RE.match(path.name) else " 应为 ASWBMCM2600001.pdf"))
    mb = path.stat().st_size / 1024 / 1024
    add("PASS" if mb <= 20 else "WARN", "PDF 体积", f"{mb:.2f} MB")

    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            add("SKIP", "PDF 深度检查", "未安装 PyMuPDF")
            return

    doc = pymupdf.open(path)
    try:
        pages = [pg.get_text() or "" for pg in doc]
        n = len(pages)
        add("INFO", "PDF 总页数", str(n))

        p1 = pages[0] if pages else ""
        low1 = p1.lower()
        add("PASS" if "summary" in low1 else "FAIL", "第 1 页 Summary", "检出" if "summary" in low1 else "未检出")
        add("PASS" if ("problem chosen" in low1 or "team number" in low1) else "WARN", "Summary 头部",
            "Problem Chosen/Team Number")
        add("PASS" if ("keyword" in low1 or "key words" in low1) else "WARN", "Keywords",
            "检出" if ("keyword" in low1 or "key words" in low1) else "未检出")
        add("PASS" if ("keyword" in low1 or "key words" in low1) else "WARN", "Summary 单页",
            "p1 有 Keywords" if ("keyword" in low1 or "key words" in low1) else "p1 无 Keywords，Summary 可能跨页；人工确认")
        sw = len(re.findall(r"\b[A-Za-z]{2,}\b", p1))
        sn = len(re.findall(r"\d+(?:\.\d+)?%?", p1))
        add("PASS" if 300 <= sw <= 450 else "WARN", "Summary 词数", f"{sw} words（目标 300–450）")
        add("PASS" if 8 <= sn <= 20 else "WARN", "Summary 数字数", f"{sn}（目标 8–20）")

        p2 = pages[1].lower() if n > 1 else ""
        add("PASS" if "contents" in p2 else "FAIL", "第 2 页 Contents",
            "检出" if "contents" in p2 else "未检出")

        # References page, then Appendix heading (avoid body mentions of "Appendix 1")
        ref_page = None
        for i in range(2, len(pages)):
            if re.search(r"(?im)^\s*(?:[IVX]+\.\s*)?(References|Bibliography)\s*$", pages[i]):
                ref_page = i + 1
                break
        appendix = None
        start = ref_page if ref_page else 2
        for i in range(start, len(pages)):
            if re.search(r"(?im)^\s*(?:Appendix|APPENDIX)(?:\s+[A-Z0-9]+)?\s*$", pages[i]):
                appendix = i + 1
                break
        main_pages = (appendix - 1) if appendix else n
        add("PASS" if main_pages <= 25 else "FAIL", "总页数 ≤25（含 Summary/Contents）",
            f"正文+Summary+Contents 约 {main_pages} 页" + (f"（Appendix p{appendix}）" if appendix else ""))

        # Header on body pages
        bad_header = []
        end = appendix - 1 if appendix else n
        for i in range(2, end):
            head = "\n".join(pages[i].splitlines()[:5])
            if not (re.search(r"Team\s+ASWBMCM", head, re.I) and re.search(r"Page\s+\d+\s+of\s+\d+", head, re.I)):
                bad_header.append(i + 1)
        add("PASS" if not bad_header else "WARN", "页眉 Team ASWBMCM / Page x of y",
            "全部通过" if not bad_header else f"{len(bad_header)} 页异常，例 {bad_header[:8]}")

        # Body text; remove repeated Chinese watermark lines
        body_pages = pages[2:end]
        from collections import Counter
        freq = Counter(line.strip() for t in body_pages for line in t.splitlines() if line.strip())
        repeated = {line for line, c in freq.items()
                    if c >= max(3, int(0.5 * len(body_pages))) and CJK.search(line)}
        body = "\n".join("\n".join(line for line in t.splitlines() if line.strip() not in repeated)
                         for t in body_pages)
        cjk = CJK.findall(body)
        add("PASS" if not cjk else "FAIL", "正文中文扫描", f"{len(cjk)} 个 CJK 字符" if cjk else "0 命中")
        ids = sorted(set(m.group(0) for m in ID.finditer(body)))
        add("PASS" if not ids else "FAIL", "正文身份信息", str(ids[:8]) if ids else "0 命中")

        # Font size coarse check
        sizes = []
        for i in range(2, end):
            for block in doc[i].get_text("dict").get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("text", "").strip():
                            sizes.append(round(span.get("size", 0), 1))
        if sizes:
            common = statistics.mode(sizes)
            add("PASS" if 11.0 <= common <= 13.0 else "WARN", "正文字号粗检",
                f"最常见 {common}pt（官方 12pt Times New Roman；以模板/目检为准）")
        else:
            add("SKIP", "正文字号粗检", "未提取到字号")

        md = doc.metadata or {}
        dirty = {k: v for k, v in md.items()
                 if k in ("author", "creator", "producer", "title", "subject", "keywords")
                 and v and not re.fullmatch(r"Microsoft.?\s*Word.*|WPS.*|LibreOffice.*|.*PDF.*|TeX|pdfTeX.*|LaTeX.*|Overleaf.*|xdvipdfmx.*|dvipdfmx.*|LuaTeX.*|XeTeX.*", str(v))}
        add("FAIL" if dirty else "PASS", "PDF 元数据", str(dirty) if dirty else "干净")
    finally:
        doc.close()


def audit_control(path: Path | None):
    if path is None:
        add("WARN", "承诺书", "未提供（提交前必须单独签名上传）")
        return
    if not path.is_file():
        add("FAIL", "承诺书不存在", str(path))
        return
    add("PASS" if CTRL_RE.match(path.name) else "FAIL", "承诺书命名", path.name)
    add("INFO", "承诺书体积", f"{path.stat().st_size/1024:.1f} KB")


def audit_zip(path: Path | None):
    if path is None:
        add("INFO", "支撑材料", "未提供（可选）")
        return
    if not path.is_file():
        add("FAIL", "支撑 ZIP 不存在", str(path))
        return
    mb = path.stat().st_size / 1024 / 1024
    add("PASS" if mb <= 20 else "FAIL", "支撑材料 <20 MB", f"{mb:.2f} MB")
    add("PASS" if ZIP_RE.match(path.name) else "WARN", "支撑材料命名", path.name)
    try:
        with zipfile.ZipFile(path) as z:
            bad = z.testzip()
            add("PASS" if bad is None else "FAIL", "ZIP 完整性", "OK" if bad is None else str(bad))
            names = [x for x in z.namelist() if not x.endswith("/")]
            add("INFO", "ZIP 条目数", str(len(names)))
            bad_names = [x for x in names if "__pycache__" in x or x.endswith((".pyc", ".pyo"))
                         or "/.venv/" in x or "/node_modules/" in x]
            add("PASS" if not bad_names else "FAIL", "ZIP 环境/缓存", str(bad_names[:8]) if bad_names else "无")
            hits = []
            for x in names:
                try:
                    s = z.read(x).decode("utf-8", "ignore")
                except Exception:
                    continue
                if ID.search(s):
                    hits.append(x)
            add("PASS" if not hits else "WARN", "ZIP 身份信息", str(hits[:8]) if hits else "0 命中")
    except zipfile.BadZipFile as e:
        add("FAIL", "ZIP 无法读取", str(e))


if __name__ == "__main__":
    sys.exit(main())
