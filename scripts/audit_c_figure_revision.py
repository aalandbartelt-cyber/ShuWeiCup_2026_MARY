#!/usr/bin/env python3
"""Audit the user-requested figure details against the actual exported files.

Checks fonts, numerical labels, zoom-copy contracts and preserved v2 evidence.
This cannot replace visual inspection of layout and label legibility.
"""
from pathlib import Path
import json,hashlib,subprocess
from PIL import Image
import numpy as np
import pandas as pd


def main():
    root=Path('artifacts/c_q1_v3');directories=[root/'figures_legend_revision',root/'assessment/figures'];records=[]
    for folder in directories:
        contracts=json.loads((folder/'legend_contracts.json').read_text());local=[]
        for name,contract in contracts.items():
            pdf=folder/(name+'.pdf');font_records=subprocess.check_output(['pdffonts',str(pdf)],text=True).splitlines()[2:]
            assert font_records and all('TimesNewRoman' in line and ' yes ' in line for line in font_records)
            text=subprocess.check_output(['pdftotext',str(pdf),'-'],text=True).replace('\n',' ')
            for label in contract.get('required',[]):assert label in text,(name,label)
            for zoom in contract.get('details',{}).get('zooms',[]):assert zoom['source_values_identical'] and zoom['source_styles_identical']
            for point in contract.get('details',{}).get('bar_labels',[]):assert point['label'] in text,(name,point)
            for point in contract.get('details',{}).get('interval_labels',[]):assert point['label'] in text,(name,point)
            with Image.open(pdf.with_suffix('.png')) as im:assert min(im.info['dpi'])>599;pixels=im.size
            assert 'Times New Roman' in pdf.with_suffix('.svg').read_text()
            local.append(dict(figure=name,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pixels=pixels,font_records=font_records,legend_contract=contract))
        records+=local
        (folder/'verification.json').write_text(json.dumps(dict(status='PASS',figures=local,visual_inspection='Recorded separately; exported text/font presence alone does not establish readable layout.'),indent=2)+'\n')
    assert len(records)==10
    c=json.loads((directories[1]/'legend_contracts.json').read_text())
    expected=[d+': '+method for method in ['v2 adaptive radius','conditional radius'] for d in ['EgoDemo','EgoPro','EgoStandard']]
    assert set(expected).issubset(c['q1v3_radius_development']['legend_labels'])
    assert all('dashed / open' not in label and 'solid / filled' not in label for label in c['q1v3_radius_development']['legend_labels'])
    f=pd.read_csv(root/'numeric_oof/comparison.csv');expected=[]
    for method in ['scalar','multiscale','gate']:
        expected.extend(1000*f[(f.split=='validation')&(f.method==method)].set_index('dataset').loc[['EgoDemo','EgoPro','EgoStandard'],'mean_error'])
    np.testing.assert_allclose([x['value'] for x in c['q1v3_state_development']['details']['bar_labels']],expected)
    old=Path('artifacts/c_q1_v2');freeze=json.loads((old/'freeze.json').read_text())
    for file,digest in freeze['source_hashes'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    review=json.loads((old/'figure_visual_review.json').read_text())
    for file,digest in review['files'].items():assert hashlib.sha256((old/'figures'/file).read_bytes()).hexdigest()==digest,file
    result=dict(status='PASS',figure_count=10,v3_bar_labels_match_unrounded_source=True,complete_six_color_legend=True,v2_frozen_sources_and_original_figures_preserved=True,
        user_requirements=['local zoom with source rectangle and connectors','key values directly labeled','actual-color/style legend entries'],
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path('src/shuwei_c/figure_detail_v3.py'),Path('src/shuwei_c/figure_legends_v3.py'),Path('scripts/report_c_q1_v3_progress.py'),Path('scripts/report_c_q1_v2.py')]})
    (root/'figure_detail_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
