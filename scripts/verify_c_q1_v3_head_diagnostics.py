#!/usr/bin/env python3
"""Real validation-cache parity before fixing the test extraction adapter."""
from pathlib import Path
import sys,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import joblib
from shuwei_c.q1v3_head_diagnostics import joint
from shuwei_c.q1v3_shared_risk import features


def main():
    root=Path('artifacts/c_q1_v3');out=root/'head_diagnostic_adapter';out.mkdir(exist_ok=True)
    f=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_validation_0.5.parquet');old=pd.read_parquet('artifacts/c_q1_v2/quality_numerical/features_validation_0.1.parquet')
    ix=pd.MultiIndex.from_frame(old[['episode','origin']]).get_indexer(pd.MultiIndex.from_frame(f[['episode','origin']]))
    expected=np.load(root/'video_quality/visual_validation.npz')['diagnostics'][ix];actual=joint(f,root/'visual_dino15')
    np.testing.assert_array_equal(actual.reshape(len(f),2,4,3,32)[:,:,:2],expected.reshape(len(f),2,4,3,32)[:,:,:2])
    a,_,_=features(f,actual,True);b,_,_=features(f,expected,True);np.testing.assert_array_equal(a,b)
    manifest=json.loads((root/'onset_h0.5/manifest.json').read_text());mask=f.current_state.to_numpy()==0;x=np.c_[f[manifest['numeric_columns']].to_numpy(),a[:,:,9:].reshape(len(f),-1)][mask]
    p=joblib.load(root/'onset_h0.5/full_diagnostics_leaves7.joblib').predict_proba(x)[:,1];reference=pd.read_parquet(root/'onset_h0.5/validation.parquet').diagnostics_leaves7_onset_probability.to_numpy();np.testing.assert_array_equal(p,reference)
    (out/'verification.json').write_text(json.dumps(dict(status='PASS',rows=len(f),feature_max_difference=float(np.max(abs(a-b))),probability_max_difference=float(np.max(abs(p-reference))),source_hash=hashlib.sha256(Path('src/shuwei_c/q1v3_head_diagnostics.py').read_bytes()).hexdigest(),test_used=False),indent=2)+'\n')


if __name__=='__main__':main()
