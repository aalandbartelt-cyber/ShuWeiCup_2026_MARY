"""Q1 information boundary and independent metric checks.

AI-assisted: OpenAI Codex, GPT-6 family, 2026-11-20; human review pending.
"""
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
import pytest
from shuwei_c.data import Recording
from shuwei_c.motion import filter_record
from shuwei_c.q1 import reliability_data,reliability_metrics,load_visual,contiguous_runs,aggregate_state,WeightedScale


def recording():
    t=np.arange(30)/30;p=np.tile(t[:,None,None],(1,21,3));q=np.tile([1.,0.,0.,0.],(30,21,1))
    return Recording('id','EgoDemo','env','operator','train',Path('.'),
        dict(left=dict(time=t,position=p.copy(),quaternion=q.copy()),right=dict(time=t,position=p.copy(),quaternion=q.copy()),
             quality=dict(time=t,bad=np.zeros((30,2),bool))))


def test_reliability_features_do_not_read_future_targets():
    a=recording();b=recording();b.streams['quality']['bad'][20:]=True
    b.streams['left']['position'][20:]=999
    xa,ya,ma=reliability_data([a]);xb,yb,mb=reliability_data([b])
    take=ma.time_s<20/30
    np.testing.assert_array_equal(xa[take],xb[take])
    assert not np.array_equal(ya,yb)
    assert np.all(ma.target_time_s>=ma.time_s+.1-1e-9)


def test_filter_is_causal_and_covariance_positive():
    a=recording();b=recording();b.streams['left']['position'][20:]=999
    noise=dict(r=.01,acceleration=.1)
    sa,ca=filter_record(a,noise);sb,cb=filter_record(b,noise)
    np.testing.assert_array_equal(sa[:20],sb[:20])
    assert np.linalg.eigvalsh(ca).min()>0


def test_visual_only_uses_past_not_nearest_future(tmp_path):
    r=recording();rows=[dict(time_s=.2,detections=[]),dict(time_s=.4,detections=[dict(landmarks=[[.5,.5,0]]*21,handedness_score=.9)])]
    (tmp_path/'id__head_left.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    f,valid=load_visual(r,tmp_path,np.array([.19,.3,.4,.7]))
    assert not valid[0,0] and valid[1,0] and valid[2,0] and not valid[3,0]
    assert f[1,1]==0 and f[2,1]==1


def test_metrics_balance_environments_and_handle_no_positives():
    meta=pd.DataFrame(dict(dataset=['EgoDemo']*4,environment=['a','a','a','b'],episode=['x','x','y','z']))
    y=np.zeros(4);p=np.array([0.,0.,0.,.8])
    rows=reliability_metrics(y,p,meta,'test')
    assert rows[0]['brier']==pytest.approx(.32,abs=1e-7)
    assert rows[0]['auc'] is None and rows[0]['average_precision'] is None


def test_intervals_are_half_open_and_exhaustive():
    assert contiguous_runs([1,1,0,1,0,0,1])==[(0,2),(3,4),(6,7)]


def test_state_summary_hand_and_episode_balancing():
    f=pd.DataFrame([dict(dataset='D',method='M',horizon='H',context='all',environment=e,episode=p,
        mean_euclidean_error=v,n=n) for e,p,v,n in [('a','x',1.,100),('a','x',3.,50),('a','y',4.,10),('b','z',9.,1)]])
    result=aggregate_state(f).iloc[0]
    assert result.mean_euclidean_error==6. # mean([mean([mean(1,3),4]),9])


def test_weighted_scaling_constant_columns_are_finite():
    x=np.array([[1.,1.],[1.,2.],[1.,4.]])
    scaler=WeightedScale().fit(x,sample_weight=np.array([.1,.2,.7]))
    result=scaler.transform(x)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result[:,0],0,atol=1e-12)
