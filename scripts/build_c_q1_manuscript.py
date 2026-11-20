#!/usr/bin/env python3
"""Build the Q1 review candidate with local TeX Live; no scientific reruns."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PAPER=ROOT/"05_论文/待审/C题_问题一稿_v1"
AUDIT=ROOT/"artifacts/c_q1_manuscript"
STAGE=Path("/mnt/d/shuwei2026_build/q1_review_v1")
ENGINE=Path("/mnt/d/texlive/2026/bin/windows/xelatex.exe")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    AUDIT.mkdir(exist_ok=True)
    STAGE.mkdir(parents=True,exist_ok=True)
    for p in PAPER.rglob("*"):
        if p.is_file() and p.suffix in {".tex",".cls",".cfg",".pdf"} and p.name!="C_Q1_review.pdf":
            target=STAGE/p.relative_to(PAPER)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
    for number in range(1,4):
        log=AUDIT/f"compile_{number}.log"
        with log.open("w") as stream:
            result=subprocess.run([str(ENGINE),"-interaction=nonstopmode","-halt-on-error",
                                   "-file-line-error","main.tex"],cwd=STAGE,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            print(log.read_text()[-2500:])
            raise SystemExit(result.returncode)
    for name in ["main.log","main.aux","main.toc"]:
        shutil.copy2(STAGE/name,AUDIT/name)
    shutil.copy2(STAGE/"main.pdf",PAPER/"C_Q1_review.pdf")
    sources=sorted([*ROOT.glob("src/shuwei_c/*.py"),*ROOT.glob("scripts/*c_q1*.py"),
                    ROOT/"scripts/prepare_c_video.py",ROOT/"scripts/run_c_pipeline.py",
                    ROOT/"scripts/audit_c_figure_revision.py",*ROOT.glob("configs/*.json"),
                    *ROOT.glob("tests/*.py"),ROOT/"requirements-c.txt",ROOT/"requirements-c-vision.txt"])
    # Keep the Q1 snapshot independent of ongoing Q2 development. The historical
    # q2.py prototype remains included; only the new Q2 modules/config/tests are
    # excluded. They belong in the subsequent Q2/full-paper source package.
    sources=[p for p in sources if not (
        (p.parent==ROOT/"src/shuwei_c" and p.name.startswith("q2_")) or
        (p.parent==ROOT/"configs" and p.name=="c_q2.json") or
        (p.parent==ROOT/"tests" and p.name.startswith("test_c_q2_")))]
    sources += [p for p in PAPER.rglob('*') if p.is_file()
                and p.suffix in {'.tex', '.cls', '.cfg', '.json', '.md', '.pdf'}
                and p.name != 'C_Q1_review.pdf']
    entries={str(p.relative_to(ROOT)):sha(p) for p in sources if p.is_file()}
    readme="""# Q1 review source snapshot

This archive contains source, configuration and project tests used or retained in the Q1 workflow.
It is a source snapshot, not a claim of a fresh clean-room reproduction or a final submission package.
Extract at the project root with the official raw-data tree at:
00_赛题与规则/正式发布/C_Problem_Data/Restocking_Problem_and_Data_EN/

Recorded models/results live in artifacts/c_q1_v2 and artifacts/c_q1_v3 in the working project.
Raw competition data, model checkpoints, video caches, external repositories and Python environments
are not redistributed here. Dependencies are specified in requirements-c*.txt.
The candidate README and manuscript map claims to existing artifacts.

Read-only checks with the configured numerical Python:
python scripts/audit_c_q1_v3_fixed.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests

Rendering existing evidence:
python scripts/prepare_c_q1_manuscript_tables.py
python scripts/build_c_q1_manuscript.py

The build script uses the local Windows TeX Live installation from WSL; another machine can instead
run xelatex main.tex three times from the manuscript folder with Times New Roman installed.
Frozen model/test scripts guard their completed stages: do not overwrite recorded evaluations.
"""
    with zipfile.ZipFile(PAPER/"Q1_review_source.zip","w",compression=zipfile.ZIP_DEFLATED) as z:
        for name in entries:z.write(ROOT/name,name)
        z.writestr("README.md",readme)
        z.writestr("SOURCE_SHA256.json",json.dumps(entries,indent=2))
    inputs={str(p.relative_to(PAPER)):sha(p) for p in PAPER.rglob("*") if p.is_file() and p.suffix in {".tex",".cls",".cfg",".pdf"}}
    record=dict(status="compiled_pending_page_review",pdf_sha256=sha(PAPER/"C_Q1_review.pdf"),
                source_archive_sha256=sha(PAPER/"Q1_review_source.zip"),input_sha256=inputs,
                engine=str(ENGINE),compilations=3,source_files=len(entries),
                official_template_sha256={n:sha(ROOT/"00_赛题与规则/官方模板"/n) for n in ["demo.tex","swmcmthesis.cls","swmcmthesis.cfg"]})
    (AUDIT/"build_manifest.json").write_text(json.dumps(record,indent=2,ensure_ascii=False)+"\n")
    print(f"Compiled {PAPER/'C_Q1_review.pdf'}; source files: {len(entries)}")


if __name__=="__main__":
    main()
