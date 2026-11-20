"""Causal/expert geometry and equal-hand weighting checks for Q1 v3."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import pandas as pd
from shuwei_c.q1v3_state import polynomial_expert,row_weights
from shuwei_c.q1v3_calibration import mature_residual_features


def test_polynomial_extrapolates_quadratic_motion_without_future_values():
    t=np.arange(60)/30;v=np.arange(6).reshape(2,3)/20;a=np.array([[.2,-.1,.3],[.4,.1,-.2]])
    c=t[:,None,None]*v+.5*t[:,None,None]**2*a;i=np.array([15,25]);h=np.full(2,.1)
    out=polynomial_expert(t,c,i,h,9)
    expected=.1*(v+t[i,None,None]*a)+.5*.1**2*a
    np.testing.assert_allclose(out,expected,atol=1e-12)
    changed=c.copy();changed[26:]=900
    np.testing.assert_array_equal(out,polynomial_expert(t,changed,i,h,9))


def test_polynomial_is_equivariant_to_rigid_coordinate_changes():
    t=np.arange(60)/30;rng=np.random.default_rng(7);c=rng.normal(size=(60,2,3));rot=np.linalg.qr(rng.normal(size=(3,3)))[0]
    i=np.array([20,35]);h=np.array([.1,.2]);out=polynomial_expert(t,c,i,h,13)
    actual=polynomial_expert(t,c@rot+np.array([9,-3,2]),i,h,13)
    np.testing.assert_allclose(actual,out@rot,atol=1e-12)


def test_weights_balance_hands_before_records_and_environments():
    f=pd.DataFrame(dict(environment=['a']*10+['b']*2,episode=['x']*10+['y']*2,hand=[0]*9+[1]+[0,1]))
    w=row_weights(f);assert abs(w.sum()-1)<1e-12
    assert abs(w[:9].sum()-.25)<1e-12
    assert abs(w[9]-.25)<1e-12
    assert abs(w[10]-.25)<1e-12


def test_residual_feedback_waits_until_its_reference_arrives():
    f=pd.DataFrame(dict(episode=['a']*5,hand=[0]*5,origin_s=[0.,.1,.2,.3,.4],target_s=[.2,.3,.4,.5,.6],gate_normalized_error=[1.,2.,3.,4.,5.]))
    result=mature_residual_features(f)
    assert result.past_residual_known.tolist()==[0.,0.,1.,1.,1.]
    assert np.isclose(result['past_residual_ewm_1.0'].iloc[2],np.log(2))
    changed=f.copy();changed.loc[changed.target_s>.4,'gate_normalized_error']=999
    check=mature_residual_features(changed)
    np.testing.assert_array_equal(result.filter(like='past_residual').to_numpy(),check.filter(like='past_residual').to_numpy())


def test_video_alignment_never_uses_a_future_frame():
    from shuwei_c.q1v3_visual import causal_indices
    times=np.array([.1,.2,.3,.4]);query=np.array([0.,.15,.2,.399])
    index,known,age=causal_indices(times,query)
    np.testing.assert_array_equal(known,[False,True,True,True])
    np.testing.assert_array_equal(index,[0,0,1,2])
    assert np.all(times[index[known]]<=query[known])


def test_shared_risk_is_coherent_and_hand_exchange_equivariant():
    from shuwei_c.q1v3_shared_risk import predict
    rng=np.random.default_rng(12);x=rng.normal(size=(20,2,2));flag=rng.integers(0,2,size=(20,2));prior=rng.dirichlet(np.ones(4),20)
    model=dict(mean=np.zeros(2),std=np.ones(2),coef=rng.normal(size=6))
    p=predict(x,flag,prior,model);swapped=predict(x[:,::-1],flag[:,::-1],prior[:,[0,2,1,3]],model)
    np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-12)
    np.testing.assert_allclose(swapped,p[:,[0,2,1,3]],atol=1e-12)
    left=p[:,1]+p[:,3];right=p[:,2]+p[:,3];any_hand=1-p[:,0]
    assert np.all(any_hand+1e-12>=np.maximum(left,right))
    assert np.all(any_hand<=left+right+1e-12)
