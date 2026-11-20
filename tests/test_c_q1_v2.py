"""Meaningful protocol checks for Q1 v2; AI-assisted by Codex, 2026-11-20."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pytest
from shuwei_c.q1v2_state import matched_targets,history_features,one_euro,weighted_quantile
from shuwei_c.q1v2_quality import status_history,temperature,select_threshold,event_metrics
import pandas as pd
from types import SimpleNamespace


def test_float_timestamp_jitter_does_not_add_an_entire_frame():
    t=np.arange(120)/30
    jitter=t.copy();jitter[3::4]-=8e-6
    i,j=matched_targets(jitter,.1)
    np.testing.assert_array_equal(j-i,np.full(len(i),3))
    assert np.max(abs(jitter[j]-jitter[i]-.1))<2e-5


def test_target_never_uses_terminal_hold_as_future_reference():
    t=np.arange(20)/30
    i,j=matched_targets(t,.2,warmup=.1)
    assert (j>i).all()
    assert (t[i]+.2<=t[-1]+1e-9).all()


def test_history_recovers_constant_velocity_and_ignores_future():
    t=np.arange(90)/30;velocity=np.array([[.1,.2,-.3],[-.1,.5,.2]])
    c=t[:,None,None]*velocity[None,:,:];bad=np.zeros((len(t),2),bool)
    origins=np.array([15,30]);lags=[1/30,.1,.2,.4]
    a=history_features(t,c,bad,origins,lags)
    altered=c.copy();altered[31:]+=700;flags=bad.copy();flags[31:]=True
    b=history_features(t,altered,flags,origins,lags)
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
    np.testing.assert_allclose(a[0],np.broadcast_to(velocity[None,:,:,None],a[0].shape),atol=1e-12)


def test_one_euro_vector_variant_is_rotation_equivariant_and_causal():
    t=np.arange(50)/30;c=np.sin(t[:,None,None]*np.arange(1,7).reshape(1,2,3))
    angle=.7;rotation=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1.]])
    smooth,v=one_euro(t,c,1.,1.)
    rotated,rv=one_euro(t,c@rotation,1.,1.)
    np.testing.assert_allclose(rotated,smooth@rotation,atol=1e-12)
    np.testing.assert_allclose(rv,v@rotation,atol=1e-12)
    changed=c.copy();changed[20:]=100
    future,_=one_euro(t,changed,1.,1.)
    np.testing.assert_array_equal(future[:20],smooth[:20])


def test_weighted_quantile_uses_mass_not_frame_count():
    assert weighted_quantile([1,2,100],[.1,.1,.8],.5)==100


def test_status_age_resets_at_change_and_is_causal():
    t=np.arange(8)*.1;b=np.zeros((8,2),bool);b[3:6,0]=True
    age=status_history(t,b)
    np.testing.assert_allclose(age[:,0],[0,.1,.2,0,.1,.2,0,.1])
    changed=b.copy();changed[5:]=True
    np.testing.assert_array_equal(age[:5],status_history(t,changed)[:5])


def test_joint_calibration_preserves_probability_coherence():
    p=temperature(np.array([[.7,.1,.15,.05],[.01,.19,.4,.4]]),1.8)
    np.testing.assert_allclose(p.sum(axis=1),1)
    left=p[:,1]+p[:,3];right=p[:,2]+p[:,3];any_bad=1-p[:,0]
    assert (any_bad>=np.maximum(left,right)-1e-12).all()
    assert (any_bad<=left+right+1e-12).all()


def test_tied_risk_threshold_does_not_inflate_false_alarm_rate():
    f=pd.DataFrame(dict(environment=['e']*10,episode=['r']*10,current_state=[0]*10,target_state=[0]*10))
    p=np.tile([.9,.04,.04,.02],(10,1));threshold=select_threshold(p,f,.05)
    assert not (1-p[:,0]>threshold).any()


def test_event_is_counted_once_despite_multiple_alarm_frames():
    t=np.arange(30)*.1;b=np.zeros((30,2),bool);b[10:13,0]=True;b[20:23,0]=True
    rec=SimpleNamespace(episode='r',environment='e',dataset='EgoPro',hands=lambda:(t,None,b))
    i=np.arange(4,27);j=i+3
    f=pd.DataFrame(dict(environment=['e']*len(i),episode=['r']*len(i),time_s=t[i],target_time_s=t[j],current_state=b[i].any(axis=1).astype(int),target_state=b[j].any(axis=1).astype(int)))
    p=np.tile([.95,.02,.02,.01],(len(i),1));p[(i>=7)&(i<=9)]=[.2,.4,.3,.1]
    result,events=event_metrics([rec],f,p,.5,.3)
    assert result['events']==2 and result['detected']==1
    assert result['event_recall']==.5
    assert result['median_lead']==pytest.approx(.3)
