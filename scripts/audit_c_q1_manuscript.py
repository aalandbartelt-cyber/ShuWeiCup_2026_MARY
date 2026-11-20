#!/usr/bin/env python3
"""Check the compiled review manuscript against its sources and layout contract."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import pymupdf as fitz

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / '05_论文/待审/C题_问题一稿_v1'
OUT = ROOT / 'artifacts/c_q1_manuscript'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = []
    def check(name, ok, detail=None):
        checks.append(dict(name=name, status='PASS' if ok else 'FAIL', detail=detail))

    pdf = fitz.open(PAPER / 'C_Q1_review.pdf')
    texts = [p.get_text() for p in pdf]
    visible = '\n'.join(texts)
    source = (PAPER / 'main.tex').read_text()
    sections = ['introduction', 'model', 'results', 'later_problems', 'discussion']
    body = '\n'.join((PAPER / 'sections' / (n+'.tex')).read_text() for n in sections)
    refs = (PAPER / 'sections/references.tex').read_text()
    appendix = (PAPER / 'sections/appendix.tex').read_text()
    figure_inventory = json.loads((PAPER / 'figure_sources.json').read_text())
    verification = json.loads((PAPER / 'reference_verification.json').read_text())
    check('no_internal_version_labels', re.search(r'(?i)\bv[23]\b|_v[23]', visible) is None)
    check('draft_provenance_removed', 'Draft provenance' not in visible and 'Codex assisted' not in visible)
    blank = '\n'.join(l for l in (PAPER / 'sections/summary.tex').read_text().splitlines()
                      if l.strip() and not l.lstrip().startswith('%'))
    check('summary_and_keywords_empty', not blank.strip() and r'\keywords{}' in source,
          'User-requested blank sections; not a final submission check.')
    check('full_paper_contents', all(x in texts[1] for x in ['Problem 1:', 'Problem 2:', 'Problem 3:', 'Model Evaluation', 'References']))
    keys = re.findall(r'\\bibitem\{([^}]+)\}', refs)
    cited = list(dict.fromkeys(k for group in re.findall(r'\\cite\{([^}]+)\}', body+appendix)
                              for k in group.split(',')))
    verified_keys = [r['key'] for r in verification['records']]
    check('citations_follow_first_use', keys == cited == verified_keys and 'problem' not in keys,
          dict(entries=keys, first_use=cited, verification=verified_keys))
    check('three_assumptions', len(re.findall(r'\\item ', (PAPER / 'sections/introduction.tex').read_text())) == 3)
    check('two_propositions_and_proofs', body.count(r'\begin{proposition}') == 2 and body.count(r'\begin{proof}') == 2 and visible.count('□') >= 2)
    table_source = '\n'.join(p.read_text() for p in sorted((PAPER/'tables').glob('*.tex')))
    captions = re.findall(r'\\caption\{([^{}]+)\}', body+appendix+table_source)
    included_figures = re.findall(r'\\papergraphic(?:\[[^\]]+\])?\{([^}]+)\}',body+appendix)
    inventoried_figures = [r['local'] for r in figure_inventory]
    check('short_figure_captions', all(len(x.split()) <= 10 for x in captions)
          and sorted(included_figures)==sorted(inventoried_figures),
          dict(captions=captions, included_figures=included_figures, inventory=inventoried_figures))
    check('source_paragraph_settings', r'\setlength{\parindent}{2em}' in source and r'\setlength{\parskip}{0pt}' in source)
    explain = []
    bounds = []
    for i, page in enumerate(pdf):
        check_a4 = abs(page.rect.width-595.276)<1 and abs(page.rect.height-841.89)<1
        bounds.append(check_a4)
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                txt = ''.join(s['text'] for s in line['spans'])
                if re.match(r'^(where |Here |In \(11\),)', txt):
                    explain.append(dict(page=i+1, x=line['bbox'][0], text=txt[:85]))
    check('formula_explanations_flush_left', len(explain)>=6 and all(abs(r['x']-25/25.4*72)<1.5 for r in explain), explain)
    check('a4_pages', all(bounds))
    check('english_visible_text', re.search(r'[\u4e00-\u9fff]', visible) is None)
    check('no_unresolved_references', '??' not in visible)
    check('empty_author_metadata', not pdf.metadata.get('author'))
    log = (OUT / 'main.log').read_text(errors='replace')
    bad = re.findall(r'.*(?:Overfull|undefined references|Citation .* undefined|Font Warning|headheight is too small).*', log)
    check('no_overflow_or_reference_font_warnings', not bad, bad)
    manifest = json.loads((OUT / 'build_manifest.json').read_text())
    check('pdf_matches_build', sha(PAPER / 'C_Q1_review.pdf') == manifest['pdf_sha256'])
    check('official_template_unchanged', all(sha(ROOT/'00_赛题与规则/官方模板'/n)==v for n,v in manifest['official_template_sha256'].items()))
    figure_rows = []
    for row in figure_inventory:
        file = PAPER / row['local']
        fig = fitz.open(file)
        base_sizes = []
        fonts = []
        for block in fig[0].get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    fonts.append(span['font'])
                    # Log-axis exponents are typographic superscripts, not base labels.
                    if span['size'] >= 8:
                        base_sizes.append(span['size'])
        width = .82 if row['number']=='5' else .88
        printed_min = min(base_sizes) * (160/25.4*72*width) / fig[0].rect.width
        ok = (sha(file)==row['sha256'] and sha(ROOT/row['source'])==row['source_sha256']
              and printed_min >= 8-0.02 and all('TimesNewRoman' in font for font in fonts))
        figure_rows.append(dict(number=row['number'], printed_width_mm=160*width,
                               minimum_base_font_pt=printed_min, status='PASS' if ok else 'FAIL'))
    check('figure_sources_fonts_and_print_size', len(figure_rows)==len(included_figures)
          and all(r['status']=='PASS' for r in figure_rows), figure_rows)
    tables_manifest=json.loads((PAPER/'table_sources.json').read_text())
    check('table_source_hashes',bool(tables_manifest) and all(sha(ROOT/p)==h for p,h in tables_manifest.items()))
    check('review_constraints_preserved', r'\renewcommand{\team}{Team CSWBMCM2619167}' in source
          and not re.search(r'\\subsection', (PAPER/'sections/discussion.tex').read_text())
          and 'test results had been observed' in body and 'Validation and training OOF are the main evidence' in body)
    with zipfile.ZipFile(PAPER / 'Q1_review_source.zip') as archive:
        entries = json.loads(archive.read('SOURCE_SHA256.json'))
        valid = archive.testzip() is None and all(sha(ROOT/n)==v for n,v in entries.items())
        check('source_archive_integrity_and_freshness', valid, dict(files=len(entries)))
    appendix_start = next(i+1 for i,t in enumerate(texts) if i>1 and 'Appendix A' in t and 'Supporting Materials' in t)
    check('main_page_limit', appendix_start-1 <= 25, dict(main_including_front_matter=appendix_start-1, appendix=len(pdf)-appendix_start+1, total=len(pdf)))
    result = dict(status='PASS' if all(c['status']=='PASS' for c in checks) else 'FAIL',
                  pdf_sha256=sha(PAPER/'C_Q1_review.pdf'), checks=checks,
                  visual_review='Recorded separately after viewing rendered pages.')
    (OUT / 'layout_verification.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result['status']=='PASS' else 1)


if __name__ == '__main__':
    main()
