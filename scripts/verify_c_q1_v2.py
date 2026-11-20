#!/usr/bin/env python3
"""Audit frozen sources, comparison identities, probabilities, fonts and exports.

AI-assisted by Codex, 2026-11-20. Visual inspection is recorded separately.
"""
from pathlib import Path
from datetime import datetime
import hashlib,json,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from PIL import Image
from shuwei_c.data import Dataset
from shuwei_c.q1v2_interface import FrozenQ1


def main():
    root=Path('artifacts/c_q1_v2');freeze=json.loads((root/'freeze.json').read_text());test=json.loads((root/'test_run_manifest.json').read_text());checks=[]
    assert datetime.fromisoformat(freeze['timestamp_utc'])<datetime.fromisoformat(test['started_utc'])
    assert hashlib.sha256((root/'freeze.json').read_bytes()).hexdigest()==test['freeze_sha256']
    for file,digest in freeze['source_hashes'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    checks.append(dict(check='freeze_before_test_and_source_hashes',files=len(freeze['source_hashes']),pass_check=True))
    for part in ['state','quality_numerical','quality_video5','quality_video15']:
        manifest=root/'run_manifest_state.json' if part=='state' else root/part/'run_manifest.json'
        m=json.loads(manifest.read_text())
        for file,digest in m['source_hashes'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    checks.append(dict(check='development_run_sources_match',pass_check=True))
    for split in ['validation','test']:
        for h in freeze['state']['horizons']:
            f=pd.read_parquet(root/f'state_{split}_{float(h):.6f}.parquet');keys=['episode','hand','origin','target'];reference=None
            for method,g in f.groupby('method'):
                now=g[keys].sort_values(keys).to_numpy()
                if reference is None:reference=now
                else:np.testing.assert_array_equal(now,reference)
                assert (g.error>=0).all() and np.isfinite(g.error).all()
    checks.append(dict(check='all_state_candidates_share_target_keys',splits=2,horizons=5,pass_check=True))
    for path in list(root.glob('quality_test_*.parquet'))+list(root.glob('quality_*/validation_*.parquet')):
        f=pd.read_parquet(path);p=f[['p0','p1','p2','p3']].to_numpy();assert np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
        np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-12)
    checks.append(dict(check='joint_probability_coherence',pass_check=True))
    fonts=[]
    for path in sorted((root/'figures').glob('*.pdf')):
        result=subprocess.run(['pdffonts',str(path)],check=True,capture_output=True,text=True).stdout
        lines=[line for line in result.splitlines()[2:] if line.strip()]
        assert lines and all('TimesNewRoman' in line for line in lines),(path.name,result)
        assert all(' yes ' in line for line in lines),(path.name,result)
        png=path.with_suffix('.png');svg=path.with_suffix('.svg');assert png.exists() and svg.exists()
        with Image.open(png) as im:assert min(im.info['dpi'])>=599;size=im.size
        assert 'Times New Roman' in svg.read_text()
        fonts.append(dict(figure=path.stem,font_records=lines,pixels=size,pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len(fonts)==7
    checks.append(dict(check='seven_figures_Times_New_Roman_embedded_600dpi_and_svg',pass_check=True))
    model=FrozenQ1();rec=Dataset().records('validation')[0];t,p,b=rec.hands();c=p.mean(axis=2);cut=20
    a=model.forecast(t[:cut+1],c[:cut+1],b[:cut+1],t[cut]+.1,rec.dataset)
    sparse=model.forecast(t[:cut+1],c[:cut+1],b[:cut+1],t[cut]+.1,rec.dataset,response_pattern=(.2,.1))
    np.testing.assert_array_equal(a['position'],sparse['position']);np.testing.assert_allclose(sparse['radius'],a['radius']*model.sparse['multiplier'])
    for h in [.1,.5]:
        risk=model.flag_risk(b[cut],rec.dataset,horizon=h);assert max(risk['left'],risk['right'])<=risk['any']+1e-12<=risk['left']+risk['right']+1e-12
    checks.append(dict(check='frozen_state_sparse_and_quality_interfaces',pass_check=True))
    values=pd.read_csv(root/'state_main_table.csv');assert (values[values.split=='test'].gain_vs_linear_pct<0).any()
    checks.append(dict(check='negative_test_result_retained',pass_check=True))
    (root/'verification.json').write_text(json.dumps(dict(status='PASS',checks=checks,figures=fonts,unit_tests='28 passed; see verification_tests.log',
        visual_review='separate figure_visual_review.json; Codex visual review is not human ground truth'),indent=2)+'\n')
    print(json.dumps(dict(status='PASS',checks=len(checks),figures=len(fonts))))


if __name__=='__main__':main()
