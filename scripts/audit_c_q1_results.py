#!/usr/bin/env python3
"""Recompute selected Q1 results and inspect final embedded figure fonts.

AI-assisted: OpenAI Codex, GPT-6 family, 2026-11-20; human verification pending.
"""
from pathlib import Path
import hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pymupdf
from shuwei_c.data import balanced_weights


def main():
    root=Path('artifacts/c_q1_v1');checks={}
    records=pd.read_csv(root/'record_audit.csv');intervals=pd.read_csv(root/'interval_uses.csv')
    total=int(records.total_hand_frames.sum())
    assert intervals.frames.sum()==total
    assert intervals.loc[intervals.error_reference_allowed,'frames'].sum()==records.qualified_hand_frames.sum()
    checks['interval_hand_frames_partition']=total
    prediction=pd.read_csv(root/'quality_validation_predictions.csv');metrics=pd.read_csv(root/'quality_validation_metrics.csv')
    w=balanced_weights(prediction.environment,prediction.episode)
    for name in metrics.method.unique():
        assert np.isfinite(prediction[name]).all() and prediction[name].between(0,1).all()
        recomputed=float(np.average((np.clip(prediction[name],1e-7,1-1e-7)-prediction.target_bad)**2,weights=w))
        stated=float(metrics[(metrics.method==name)&(metrics.dataset=='ALL')].brier.iloc[0])
        assert abs(recomputed-stated)<1e-12,(name,recomputed,stated)
    checks['brier_methods_recomputed']=int(metrics.method.nunique())
    assert np.all(prediction.target_time_s>=prediction.time_s+.1-1e-9)
    checks['quality_prediction_times']='causal feature rows, future targets separately stored'
    model=pd.read_csv(root/'flag_transition_model.csv')
    expected=prediction.merge(model,on=['dataset','current_bad'],how='left').probability
    np.testing.assert_allclose(expected,prediction.flag_transition,rtol=0,atol=1e-12)
    checks['selected_transition_model_replay']='matches stored predictions'
    font_audit=[]
    for path in sorted((root/'figures').glob('*.pdf')):
        doc=pymupdf.open(path);font_names=[];sizes=[]
        for page in doc:
            for font in page.get_fonts():
                font_names.append(font[3]);embedded=doc.extract_font(font[0])[3]
                assert 'TimesNewRoman' in font[3] and len(embedded)>0,(path,font[3])
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line['spans']:
                        if span['text'].strip():sizes.append(span['size'])
        assert min(sizes)>=7.99,(path,min(sizes))
        font_audit.append(dict(file=path.name,embedded_fonts=sorted(set(font_names)),minimum_font_pt=min(sizes),all_times=True))
    checks['pdf_figures']=len(font_audit)
    (root/'figure_font_audit.json').write_text(json.dumps(font_audit,indent=2)+'\n')
    (root/'result_audit.json').write_text(json.dumps(dict(status='PASS',checks=checks),indent=2)+'\n')
    files=[p for p in root.glob('*.csv')]+list((root/'figures').glob('*'))
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()}
    (root/'result_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps(dict(status='PASS',checks=checks)))


if __name__=='__main__':main()
